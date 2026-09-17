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


def _tapped_hidden_states(model, ids, taps, mask_rate=0.0, generator=None,
                          sigma=0.0):
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
        torch.full((batch,), float(sigma), device=x.device,
                   dtype=torch.float32)))
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
  # Apply the time embedding as `forward_active` does
  # (bidirectional_ssm.py:990-995). Skipping it silently discarded a trained
  # module on any `time_conditioning: True` checkpoint. No-op where
  # `time_embedding is None`, which is every checkpoint published before
  # 2026-09-13.
  if getattr(backbone, "time_embedding", None) is not None:
    h = h + backbone.time_embedding(
      torch.full((batch,), float(sigma), device=x.device,
                 dtype=torch.float32))[:, None, :]
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


def _hidden_states(model, ids, mask_rate=0.0, generator=None, sigma=0.0):
  """[batch, length, hidden] just before the vocabulary projection."""
  b = model.backbone
  top = len(b.blocks if hasattr(b, "blocks") else b.layers)
  for _, hidden in _tapped_hidden_states(model, ids, (top,), mask_rate,
                                         generator, sigma):
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


RECURRENT_REDUCTIONS = ("headmean", "raw")


def recurrent_summary(model, ids_forward, ids_reverse, layers="last",
                      reduce="headmean"):
  """Readout (A): the fixed-size SSM recurrent state, both directions.

  A generative denoiser has no canonical embedding. The honest sequence-level
  summary in THIS architecture is not a pooled hidden state -- it is the
  recurrent state itself, the `Mamba2State` that a scan would need to continue
  exactly. `prefill_left` over the whole sequence leaves a state that has
  integrated everything left-to-right; `prefill_right` flips its input
  internally, so its state has integrated everything right-to-left. Together
  they summarise the sequence symmetrically.

  Why this goes through `prefill_*` and not the classifier's layer loop: the
  readout path used for pooled features calls `scan_active`, which returns only
  hidden states -- `scan_bidirectional` says outright that "the final boundary
  states are deliberately not computed". `_prefill` is the one entry point that
  accumulates them (`bidirectional_ssm.py:659-661`).

  **Two traps, both handled by the caller contract rather than here.**

  1. PADDING. This reads the state at the end of the scan, so it cannot mask
     padding the way mean pooling does -- right padding is scanned *after* the
     real sequence and would dominate the state we harvest (65% of the window on
     `human_enhancers_ensembl`). Hence two differently-padded inputs:
     `ids_forward` must be LEFT-padded and `ids_reverse` RIGHT-padded, so that in
     each direction the filler is at the START of that direction's scan, where it
     washes out instead of overwriting the answer. `embed_sequences_recurrent`
     enforces this; passing one tensor for both is a silent correctness bug.

  2. NO TIMESTEP. `_prefill` embeds tokens and scans, with no `time_embedding`
     call, so this readout is timestep-free by construction. A sigma sweep
     applies to the pooled readout only -- there is nothing here for it to move.

  `reduce="headmean"` averages the `headdim` axis of `[batch, nheads, headdim,
  d_state]`, giving `nheads * d_state` per layer per direction (1536 for
  `small_bissm`). That is a deliberate 64x narrowing: the raw state is 98,304
  numbers per layer per direction, so `layers="all"` raw is 2.36M features
  against as few as 968 training rows (`dummy_mouse_enhancers_ensembl`), where a
  linear probe has nothing to stand on. `reduce="raw"` keeps the full state for
  the large tasks.
  """
  if reduce not in RECURRENT_REDUCTIONS:
    raise ValueError(f"reduce must be one of {RECURRENT_REDUCTIONS}")
  backbone = model.backbone
  if not hasattr(backbone, "prefill_left"):
    raise TypeError(
      f"{type(backbone).__name__} has no prefill_left; the recurrent readout "
      f"is specific to the SSM backbones")
  uni = type(backbone).__name__ == "UnidirectionalSSM"
  if uni:
    # UnidirectionalSSM overrides only backbone-level methods, so a reverse
    # prefill would either raise or run a direction the checkpoint never
    # trained. Fail loudly rather than return a half-meaningless vector.
    raise TypeError(
      "the recurrent readout needs both scan directions; this checkpoint is "
      "unidirectional. Use --readout pooled for AR arms")

  parts = []
  # Match the dtype regime the states were trained under: diffusion.py:1323-1334
  # wraps every prefill in `_model_autocast_context()`. `getattr` because that
  # method belongs to `Diffusion`, and the unit tests drive the backbone through
  # a bare wrapper; on CPU the real one is a nullcontext anyway.
  autocast = getattr(model, "_model_autocast_context", None)
  with (autocast() if autocast is not None else contextlib.nullcontext()):
    for ids, prefill in ((ids_forward, backbone.prefill_left),
                         (ids_reverse, backbone.prefill_right)):
      cache = prefill(ids, detach=True)
      indices = (range(len(cache.states)) if layers == "all"
                 else [len(cache.states) - 1])
      for index in indices:
        state = cache.states[index].ssm      # [batch, nheads, headdim, d_state]
        if reduce == "headmean":
          state = state.mean(dim=2)          # [batch, nheads, d_state]
        parts.append(state.flatten(start_dim=1).float())
  return torch.cat(parts, dim=-1)


def embed_sequences_recurrent(model, tokenizer, sequences, length,
                              batch_size=32, layers="last", reduce="headmean",
                              device=None, progress_every=0, pad_multiple=8):
  """Readout (A) over a list of sequences, batched by EXACT length.

  Padding is the whole difficulty with a recurrent readout, so this avoids it
  rather than bounding it: sequences are grouped by their own length and each
  group is scanned at that length, so no group is padded up to the task window.
  Measured on the suite, that costs almost nothing -- 4 of the 8 tasks
  (`demo_coding_vs_intergenomic_seqs`, `demo_human_or_worm`,
  `human_enhancers_cohn`, `human_nontata_promoters`) are fixed-length, so they
  form a single group; `human_enhancers_ensembl`, `human_ensembl_regulatory` and
  `human_ocr_ensembl` have 515-572 distinct lengths with groups up to 37,260 rows.
  Only `dummy_mouse_enhancers_ensembl` is awkward -- 538 distinct lengths over
  1,210 rows, so most groups hold one sequence -- and at that size the resulting
  ~1,200 tiny forward passes are irrelevant.

  The alternative, padding everything to the task window, would have scanned
  52-65% filler on four tasks. Because (A) harvests the state at the END of the
  scan it cannot mask that out the way pooling does. Leading padding is merely
  ATTENUATED, not removed (it decays as exp(A*dt*distance) but never vanishes),
  and the attenuation depends on trained per-head timescales that are known to
  run long in places (layer 0 of `hg_bissm_cos` collapses dt toward zero), so
  bounding it was never going to be trustworthy.

  The residual is the round up to `pad_multiple` (8 by default, keeping the scan
  kernel on friendly shapes): at most 7 filler tokens, placed on the LEADING edge
  of each direction's own scan by the two-sided encoding in `recurrent_summary`'s
  contract. `length` is now only a ceiling, asserted, not a target.
  """
  # From the backbone, not the model: `Diffusion` exposes `.parameters()` but the
  # unit tests drive the backbone through a bare wrapper that does not.
  device = device or next(model.backbone.parameters()).device
  groups = {}
  for index, sequence in enumerate(sequences):
    if len(sequence) > length:
      raise ValueError(
        f"sequence of length {len(sequence)} exceeds the window {length}")
    groups.setdefault(len(sequence), []).append(index)

  features, done = None, 0
  with torch.inference_mode():
    for size in sorted(groups):
      indices = groups[size]
      window = max(pad_multiple, -(-size // pad_multiple) * pad_multiple)
      for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        chunk = [sequences[i] for i in rows]
        encoded = _encode_padded(tokenizer, chunk, window, device,
                                 ("left", "right"))
        batch = recurrent_summary(model, encoded["left"][0],
                                  encoded["right"][0], layers,
                                  reduce).cpu().numpy()
        if features is None:
          features = np.empty((len(sequences), batch.shape[1]),
                              dtype=batch.dtype)
        features[rows] = batch
        done += len(rows)
        if progress_every and done % (progress_every * batch_size) < len(rows):
          print(f"    embedded {done}/{len(sequences)}", flush=True)
  if features is None:
    raise ValueError("no sequences to embed")
  return features


_CHAR_TABLE = {}


def _char_table(tokenizer):
  """256-entry uint8 -> token id LUT, cached per tokenizer.

  `encode_dna` tokenises one character at a time through the HF tokenizer, which
  measures 590x slower than this table (3.77s vs 0.01s for 2,000 x 401 nt). On
  `human_ensembl_regulatory` that is 9 minutes of pure CPU per encode pass, and
  the recurrent readout needs two passes -- 18 minutes of tokenising, for a task
  whose GPU work is a fraction of that. `finetune.py` has carried
  `build_char_table` for exactly this reason since the fast encoder landed;
  `embed.py` was still on the slow path. tests/test_caduceus_finetune.py:148
  pins the two to bit-identical output.
  """
  key = id(tokenizer)
  if key not in _CHAR_TABLE:
    unk = tokenizer.convert_tokens_to_ids("[UNK]")
    table = np.full(256, unk, dtype=np.int64)
    for character in "ACGTN":
      table[ord(character)] = tokenizer.convert_tokens_to_ids(character)
    _CHAR_TABLE[key] = (table, int(unk))
  return _CHAR_TABLE[key]


def _encode_padded(tokenizer, chunk, length, device, sides=("right",)):
  """Tokenise ONCE, emit one id/mask tensor pair per requested padding side.

  The recurrent readout needs the same batch padded both ways (left for the
  forward scan, right for the reverse). Encoding twice doubled the tokenisation
  bill for no reason -- the two differ only in where the filler sits, which is a
  roll of an already-computed row.
  """
  table, unk = _char_table(tokenizer)
  pad_id = int(tokenizer.convert_tokens_to_ids("N"))
  out = {}
  for side in sides:
    out[side] = (np.full((len(chunk), length), pad_id, dtype=np.int64),
                 np.zeros((len(chunk), length), dtype=bool))
  for row, sequence in enumerate(chunk):
    sequence = sequence.upper()
    if len(sequence) > length:
      raise ValueError(
        f"Sequence of length {len(sequence)} exceeds model length {length}")
    ids = table[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if (ids == unk).any():
      raise ValueError("Benchmark sequence contains a non-ACGT token")
    for side, (rows, masks) in out.items():
      start = 0 if side == "right" else length - len(sequence)
      rows[row, start:start + len(ids)] = ids
      masks[row, start:start + len(ids)] = True
  return {side: (torch.from_numpy(rows).to(device),
                 torch.from_numpy(masks).to(device))
          for side, (rows, masks) in out.items()}


def _encode_batch(tokenizer, chunk, length, device, pad_side="right"):
  """`pad_side="left"` moves the `N` filler in front of the sequence.

  Irrelevant to pooled readouts, which mask padding out. It matters enormously
  to the recurrent readout, which reads the scan's state at the END of the scan
  and therefore cannot mask anything: see `recurrent_summary`.
  """
  return _encode_padded(tokenizer, chunk, length, device, (pad_side,))[pad_side]


def embed_sequences(model, tokenizer, sequences, length, pooling="mean",
                    batch_size=32, mask_rate=0.0, seed=0, device=None,
                    progress_every=0, rc_tta=False, sigma=0.0):
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
      pooled = pool(_hidden_states(model, ids, mask_rate, generator, sigma),
                    keep, pooling)
      if rc_tta:
        rc_ids, rc_keep = _encode_batch(
          tokenizer, [reverse_complement(s) for s in chunk], length, device)
        pooled = (pooled + pool(
          _hidden_states(model, rc_ids, mask_rate, generator, sigma),
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
