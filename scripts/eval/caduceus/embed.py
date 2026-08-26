#!/usr/bin/env python3
"""Sequence embeddings from a BD3-LM checkpoint, for downstream classification.

Caduceus (Schiff et al. 2024, arXiv:2403.03234) evaluates DNA models by
fine-tuning them on GenomicBenchmarks and the Nucleotide Transformer task
suite. This module supplies the piece those benchmarks need from us: a fixed
vector per DNA sequence.

We take the hidden states just before the output head -- `final_norm(x)` in
`BidirectionalSSM.forward_active`, the same tensor the vocabulary projection
consumes -- and pool over positions. That is the representation the model
actually learned; the output head is a 13-way nucleotide classifier and is not
useful downstream.

**Both caches are left empty on purpose.** Scoring a benchmark sequence is not
block diffusion: there is no prefix to condition on and no clean suffix, so the
honest representation of a standalone sequence is the one the model produces
from the sequence alone. The reverse scan still runs over the sequence itself,
so bidirectional context within the sequence is retained -- which is the whole
point of comparing against a bidirectional baseline.

Sequences are fed CLEAN (no masking). The block-diffusion objective trains the
model to denoise, so a clean input is the zero-noise end of its training
distribution rather than something it has never seen. `--mask-rate` exists to
test sensitivity to that choice.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.score_mavedb import (  # noqa: E402
  load_checkpoint_model, encode_dna)
from scripts.eval.dnahnet.deg import reverse_complement  # noqa: E402

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)

POOLINGS = ("mean", "max", "meanmax", "cls")


def _tapped_hidden_states(model, ids, taps, mask_rate=0.0, generator=None):
  """Yield `(depth, [batch, length, hidden])` for each depth, one forward pass.

  `depth` is 1-indexed; the top depth is the tensor the vocabulary projection
  consumes, i.e. `final_norm(x)`. Intermediate depths are raw layer outputs --
  `final_norm` was fit to the top of the stack, so applying it lower down would
  measure the norm, not the layer.

  Tapping below the top is worth measuring because the backbone TIES its output
  projection to the 13-token input embedding (`configs/model/small_bissm.yaml`
  `tie_word_embeddings: True`). That forces the last hidden state into
  token-embedding space, maximally specialised to "which nucleotide is at
  position i" -- the same reason intermediate layers probe better than the last
  for Nucleotide Transformer and for ESM.
  """
  backbone = model.backbone
  x = ids
  if mask_rate > 0:
    noise = torch.rand(x.shape, generator=generator, device=x.device)
    x = torch.where(noise < mask_rate, model.mask_index, x)

  is_dit = hasattr(backbone, "blocks")
  if not is_dit and not (hasattr(backbone, "layers")
                         and hasattr(backbone, "final_norm")):
    raise TypeError(
      f"{type(backbone).__name__} exposes no layer stack to tap; this script "
      f"supports the SSM and DiT backbones.")

  top = len(backbone.blocks if is_dit else backbone.layers)
  taps = sorted(set(taps))
  if taps[0] < 1 or taps[-1] > top:
    raise ValueError(f"taps must lie in 1..{top}, received {taps}")
  wanted = set(taps)
  batch = x.shape[0]

  if is_dit:
    # The DiT's trailing norm lives inside `output_layer`, which is tied to the
    # 13-token vocabulary and deliberately skipped, so every DiT tap -- top
    # included -- is a raw block output. No block-diffusion mask: a BD
    # checkpoint gets full attention and an AR one stays causal via b.causal,
    # mirroring the SSM branch below.
    h = backbone.vocab_embed(x)
    rotary_cos_sin = backbone.rotary_emb(h)
    # sigma_map exists only when adaLN does (dit.py:679, :687-690), i.e. for BD
    # and not for AR, which was trained with c=None throughout.
    sigma_map = getattr(backbone, "sigma_map", None)
    t_cond = None
    if sigma_map is not None:
      t_cond = torch.nn.functional.silu(sigma_map(
        torch.zeros(batch, device=x.device, dtype=torch.float32)))
    ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if h.is_cuda
           else contextlib.nullcontext())
    with ctx:
      for index in range(taps[-1]):
        h = backbone.blocks[index](h, rotary_cos_sin, c=t_cond,
                                   causal=backbone.causal, sample_mode=False,
                                   mask=None, store_kv=False)
        depth = index + 1
        if depth in wanted:
          yield depth, h.float()
    return

  # SSM backbones: replay the forward up to the output head, stopping at the
  # deepest requested tap so a shallow probe is also a cheaper one.
  #
  # UNI vs BI matters and used to be ignored. This called `scan_active`
  # unconditionally, which is BIDIRECTIONAL; a unidirectional (AR) checkpoint
  # was therefore probed with a reverse scan it was never trained with, and
  # nothing said so. UnidirectionalSSM subclasses BidirectionalSSM and
  # overrides only backbone-level methods, so its layers do expose
  # `scan_active` -- the call succeeded and quietly ran out of distribution.
  uni = type(backbone).__name__ == "UnidirectionalSSM"
  h = backbone.token_embedding(x)
  left = backbone._empty_cache(batch, h.device, h.dtype, "left")
  right = None if uni else backbone._empty_cache(batch, h.device, h.dtype,
                                                 "right")
  with backbone._compute_autocast(h):
    for index in range(taps[-1]):
      if uni:
        h, _ = backbone.layers[index].scan_clean(h, left.states[index])
      else:
        h = backbone.layers[index].scan_active(
          h, left.states[index], right.states[index])
      depth = index + 1
      if depth in wanted:
        yield depth, (backbone.final_norm(h) if depth == top else h).float()


def _hidden_states(model, ids, mask_rate=0.0, generator=None):
  """[batch, length, hidden] just before the vocabulary projection."""
  b = model.backbone
  top = len(b.blocks if hasattr(b, "blocks") else b.layers)
  for _, hidden in _tapped_hidden_states(model, ids, (top,), mask_rate,
                                         generator):
    return hidden
  raise RuntimeError("unreachable")


def pool(hidden, attention_mask, how):
  """Pool [batch, length, hidden] to [batch, dim], ignoring padded positions."""
  m = attention_mask[..., None].to(hidden.dtype)
  if how == "cls":
    return hidden[:, 0]
  summed = (hidden * m).sum(dim=1)
  counts = m.sum(dim=1).clamp(min=1)
  mean = summed / counts
  if how == "mean":
    return mean
  masked = hidden.masked_fill(~attention_mask[..., None], float("-inf"))
  maximum = masked.max(dim=1).values
  if how == "max":
    return maximum
  return torch.cat([mean, maximum], dim=-1)


def _encode_batch(tokenizer, chunk, length, device):
  rows, masks = [], []
  for sequence in chunk:
    ids, token_mask = encode_dna(tokenizer, sequence, length)
    rows.append(ids)
    masks.append(token_mask)
  return (torch.tensor(rows, dtype=torch.long, device=device),
          torch.tensor(masks, dtype=torch.bool, device=device))


def embed_sequences(model, tokenizer, sequences, length, pooling="mean",
                    batch_size=32, mask_rate=0.0, seed=0, device=None,
                    progress_every=0, rc_tta=False):
  """Pooled representations, optionally conjoined over both strands.

  `rc_tta` is Caduceus's post-hoc conjoining (their "-Ph" variant): embed the
  sequence and its reverse complement and average the two pooled vectors. The
  RC is built from the **string**, then encoded -- never by flipping the id
  tensor, which would move the right-hand `N` padding that `encode_dna`
  (`scripts/eval/dnahnet/score_mavedb.py:90-101`) adds to the left and make the
  scan run through it in the wrong place. `scripts/smoke/rc_equivariance.py`
  T8 is the regression test for that.

  Note for interpreting the result on a *baseline* checkpoint: the backbone is
  already exactly equivariant to plain length reversal with both caches empty
  (T1), and mean pooling annihilates a flip, so on those checkpoints this is
  mathematically a *complement*-only ensemble, not "the other reading
  direction" -- we already have that, exactly.
  """
  device = device or next(model.parameters()).device
  generator = torch.Generator(device=device).manual_seed(seed)
  out = []
  with torch.inference_mode():
    for start in range(0, len(sequences), batch_size):
      chunk = sequences[start:start + batch_size]
      ids, keep = _encode_batch(tokenizer, chunk, length, device)
      pooled = pool(_hidden_states(model, ids, mask_rate, generator),
                    keep, pooling)
      if rc_tta:
        rc_ids, rc_keep = _encode_batch(
          tokenizer, [reverse_complement(s) for s in chunk], length, device)
        pooled = (pooled + pool(
          _hidden_states(model, rc_ids, mask_rate, generator),
          rc_keep, pooling)) / 2
      out.append(pooled.cpu().numpy())
      if progress_every and (start // batch_size) % progress_every == 0:
        print(f"    embedded {min(start + batch_size, len(sequences))}"
              f"/{len(sequences)}", flush=True)
  return np.concatenate(out, axis=0)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--sequences", type=Path, required=True,
                      help="newline-delimited DNA, one sequence per line")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--length", type=int, default=None,
                      help="default: the checkpoint's own block_size")
  parser.add_argument("--pooling", choices=POOLINGS, default="mean")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--mask-rate", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--rc-tta", action="store_true",
                      help="post-hoc conjoining: average the pooled vectors "
                           "of the sequence and its reverse complement "
                           "(Caduceus-Ph). Off by default, so existing "
                           "embeddings are unchanged.")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("embedding extraction requires a CUDA GPU")
  device = torch.device("cuda")

  raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  trained = OmegaConf.create(raw.get("hyper_parameters", {}).get("config", {}))
  del raw
  # One block is the unit the denoiser was trained to process with caches
  # empty, so it is the natural window for a standalone sequence.
  length = args.length or int(trained.block_size)
  model, tokenizer, config, step = load_checkpoint_model(
    args.checkpoint, int(trained.model.length), args.batch_size, device)

  sequences = [s.strip().upper() for s in
               args.sequences.read_text().splitlines() if s.strip()]
  print(f"{len(sequences)} sequences | window {length} | pooling {args.pooling}"
        f" | backbone {config.algo.backbone} | step {step}", flush=True)

  vectors = embed_sequences(
    model, tokenizer, sequences, length, args.pooling, args.batch_size,
    args.mask_rate, args.seed, device, progress_every=20,
    rc_tta=args.rc_tta)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.save(args.output, vectors)
  meta = {
    "checkpoint": str(args.checkpoint), "checkpoint_global_step": step,
    "backbone": str(config.algo.backbone), "window": length,
    "pooling": args.pooling, "mask_rate": args.mask_rate,
    "rc_tta": bool(args.rc_tta),
    "num_sequences": len(sequences), "dim": int(vectors.shape[1]),
  }
  args.output.with_suffix(".meta.json").write_text(
    json.dumps(meta, indent=2) + "\n")
  print(f"wrote {args.output}  shape {vectors.shape}")


if __name__ == "__main__":
  main()
