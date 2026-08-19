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

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)

POOLINGS = ("mean", "max", "meanmax", "cls")


def _hidden_states(model, ids, mask_rate=0.0, generator=None):
  """[batch, length, hidden] just before the vocabulary projection."""
  backbone = model.backbone
  x = ids
  if mask_rate > 0:
    noise = torch.rand(x.shape, generator=generator, device=x.device)
    x = torch.where(noise < mask_rate, model.mask_index, x)

  if hasattr(backbone, "layers") and hasattr(backbone, "final_norm"):
    # SSM backbones: replay `forward_active` up to the output head.
    batch = x.shape[0]
    h = backbone.token_embedding(x)
    left = backbone._empty_cache(batch, h.device, h.dtype, "left")
    right = backbone._empty_cache(batch, h.device, h.dtype, "right")
    with backbone._compute_autocast(h):
      for index, layer in enumerate(backbone.layers):
        h = layer.scan_active(h, left.states[index], right.states[index])
      return backbone.final_norm(h).float()

  raise TypeError(
    f"{type(backbone).__name__} exposes no layer stack to tap; this script "
    f"supports the SSM backbones. Add an explicit hook for the DiT if that "
    f"arm is needed.")


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


def embed_sequences(model, tokenizer, sequences, length, pooling="mean",
                    batch_size=32, mask_rate=0.0, seed=0, device=None,
                    progress_every=0):
  device = device or next(model.parameters()).device
  generator = torch.Generator(device=device).manual_seed(seed)
  out = []
  with torch.inference_mode():
    for start in range(0, len(sequences), batch_size):
      chunk = sequences[start:start + batch_size]
      rows, masks = [], []
      for sequence in chunk:
        ids, token_mask = encode_dna(tokenizer, sequence, length)
        rows.append(ids)
        masks.append(token_mask)
      ids = torch.tensor(rows, dtype=torch.long, device=device)
      keep = torch.tensor(masks, dtype=torch.bool, device=device)
      hidden = _hidden_states(model, ids, mask_rate, generator)
      out.append(pool(hidden, keep, pooling).cpu().numpy())
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
    args.mask_rate, args.seed, device, progress_every=20)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.save(args.output, vectors)
  meta = {
    "checkpoint": str(args.checkpoint), "checkpoint_global_step": step,
    "backbone": str(config.algo.backbone), "window": length,
    "pooling": args.pooling, "mask_rate": args.mask_rate,
    "num_sequences": len(sequences), "dim": int(vectors.shape[1]),
  }
  args.output.with_suffix(".meta.json").write_text(
    json.dumps(meta, indent=2) + "\n")
  print(f"wrote {args.output}  shape {vectors.shape}")


if __name__ == "__main__":
  main()
