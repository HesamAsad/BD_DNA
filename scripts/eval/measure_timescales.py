#!/usr/bin/env python3
"""Per-head state half-life of an SSM checkpoint, per layer.

WHAT THIS MEASURES. A selective-SSM state retains a fraction exp(A*dt) of its
value per step, so the influence of a token d positions back decays as
2^(-d/tau) with

    tau = ln 2 / (A * dt),   A = exp(A_log),   dt = softplus(dt_bias)

`dt` is input-dependent at runtime (the projection adds to dt_bias), so this is
the operating-point estimate from the bias alone -- the right quantity for
comparing checkpoints, not an exact receptive field.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT.

This tool measures the BIAS-ONLY operating point, and for a trained checkpoint
that systematically UNDERSTATES the true timescale by roughly 7-11x. Training
drives `dt_bias` up and simultaneously teaches `in_proj`'s dt slice to pull the
sum back down, so the bias alone is not where the model actually operates.
Measured on real validation data (`measure_runtime_timescales.py`, 2026-08-31),
runtime dt came in at 0.1-0.2x the bias on every trained SSM arm:

    arm            tau bias-only          tau at RUNTIME
    hg_ussm_ar     (this tool)            median  14.7 nt, max 2,753 nt
    hg_ussm_bd                            median   8.5 nt, max 1,780 nt
    hg_bissm_bd    median 1.32, max 4.71  median  10.2 nt, max 2,155 nt

The headline this docstring used to carry -- "training destroyed the
timescales, 0/288 heads above 100 nt" -- was an ARTEFACT of the bias-only view.
At runtime hg_bissm_bd has heads out to 2,155 nt and 93/288 above 16 nt. The
qualitative reading survives (these models are still local: a ~10 nt median
against a 32,768 nt context) but the quantitative claim was off by one to two
orders of magnitude and the "zero long heads" part was simply wrong.

That correction matters for what it implies. The full-attention oracle found
ALL context value within +-256 nt, and the runtime tail already reaches ~2 kb --
about 10x the usable range. So the timescales were never the bottleneck, and
the depth-schedule intervention was aimed at a non-problem; its null result is
expected rather than surprising.

USE `measure_runtime_timescales.py` for any claim about what a trained model
actually does. This script remains the right tool for two narrower jobs: the
timescale at INITIALISATION (where dt_proj has not yet learned to compensate),
and checking whether a `freeze_dt` arm's bias stayed pinned.

Usage:
  python scripts/eval/measure_timescales.py outputs/hg38-caduceus/tau_frozen/checkpoints/best.ckpt
  python scripts/eval/measure_timescales.py --glob 'outputs/hg38-caduceus/*/checkpoints/last.ckpt'
"""

from __future__ import annotations

import argparse
import glob as globmod
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Lightning checkpoints pickle references to repo modules (dataloader, etc.),
# so the repo has to be importable before torch.load can unpickle one.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def timescales(path):
  """(per-layer list of tau arrays, flat tau array) for one checkpoint."""
  raw = torch.load(path, map_location="cpu", weights_only=False)
  state = raw.get("state_dict", raw)
  # Sort by NUMERIC layer index. Plain sorted() is lexicographic, which orders
  # layers.10 and layers.11 before layers.2 -- that permutes the per-layer table
  # and makes a clean geometric depth schedule look non-monotonic. The aggregate
  # stats are unaffected (order does not change a median), and A_log/dt_bias
  # stay correctly paired either way since they share a layer index.
  def by_layer(suffix):
    keys = [k for k in state if k.endswith(suffix)]
    def index(k):
      parts = k.split(".")
      for a, b in zip(parts, parts[1:]):
        if a == "layers" and b.isdigit():
          return int(b)
      return -1
    return sorted(keys, key=index)

  a_keys = by_layer("A_log")
  d_keys = by_layer("dt_bias")
  if not a_keys or len(a_keys) != len(d_keys):
    return None, None, raw.get("global_step")
  per_layer = []
  for a, d in zip(a_keys, d_keys):
    A = torch.exp(state[a].float()).flatten()
    dt = F.softplus(state[d].float()).flatten()
    per_layer.append((math.log(2) / (A * dt)).numpy())
  return per_layer, np.concatenate(per_layer), raw.get("global_step")


def report(path, per_layer_only=False):
  layers, flat, step = timescales(path)
  if flat is None:
    print(f"  {path}: no A_log/dt_bias (not an SSM checkpoint)")
    return None
  print(f"\n{path}")
  print(f"  step {step:,}" if isinstance(step, int) else "  step ?")
  print(f"  tau over {flat.size} heads: median {np.median(flat):>10,.1f} nt   "
        f"p90 {np.percentile(flat, 90):>10,.1f}   max {flat.max():>10,.1f}")
  for threshold in (16, 100, 1000, 10000):
    n = int((flat > threshold).sum())
    print(f"    heads with tau > {threshold:>6,} nt: {n:>4} / {flat.size} "
          f"({100 * n / flat.size:5.1f}%)")
  if not per_layer_only:
    print(f"    {'layer':>6}{'median tau':>14}{'max tau':>14}")
    for i, t in enumerate(layers):
      print(f"    {i:>6}{np.median(t):>14,.1f}{t.max():>14,.1f}")
  return flat


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("checkpoints", nargs="*")
  parser.add_argument("--glob", default=None)
  parser.add_argument("--summary", action="store_true",
                      help="omit the per-layer table")
  args = parser.parse_args()

  paths = list(args.checkpoints)
  if args.glob:
    paths += sorted(globmod.glob(args.glob))
  if not paths:
    sys.exit("give a checkpoint path or --glob")

  seen = {}
  for path in paths:
    flat = report(path, per_layer_only=args.summary)
    if flat is not None:
      seen[path] = flat

  if len(seen) > 1:
    print(f"\n{'checkpoint':<52}{'median':>10}{'max':>12}{'>100nt':>9}")
    print("-" * 83)
    for path, flat in seen.items():
      short = path.replace("outputs/hg38-caduceus/", "").replace("/checkpoints", "")
      print(f"{short[-50:]:<52}{np.median(flat):>10,.1f}{flat.max():>12,.1f}"
            f"{int((flat > 100).sum()):>6}/{flat.size}")


if __name__ == "__main__":
  main()
