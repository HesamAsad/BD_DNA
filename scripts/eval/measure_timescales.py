#!/usr/bin/env python3
"""Per-head state half-life of an SSM checkpoint, per layer.

WHAT THIS MEASURES. A selective-SSM state retains a fraction exp(A*dt) of its
value per step, so the influence of a token d positions back decays as
2^(-d/tau) with

    tau = ln 2 / (A * dt),   A = exp(A_log),   dt = softplus(dt_bias)

`dt` is input-dependent at runtime (the projection adds to dt_bias), so this is
the operating-point estimate from the bias alone -- the right quantity for
comparing checkpoints, not an exact receptive field.

WHY IT MATTERS HERE. On the 2026-08 hg38 BiSSM run the timescales were present
at INITIALISATION and training destroyed them:

    at init      tau median 10.8 nt,  max 510 nt,  17/288 heads above 100 nt
    after train  tau median  1.32 nt, max 4.71 nt,  0/288 heads above 100 nt

Training left A roughly alone (median 8.86 -> 0.96) and raised dt about 62x
(0.0088 -> 0.5433), and dt dominates the product. So the narrow band is a
LEARNED outcome, not an initialisation artefact -- which is why a depth
schedule over the init alone is expected to be undone, and why the frozen-dt
arm exists.

Run this on the depth-scheduled runs to answer the actual question: does the
schedule survive training, and if it is forced to survive (frozen dt), does the
likelihood improve?

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
  a_keys = sorted(k for k in state if k.endswith("A_log"))
  d_keys = sorted(k for k in state if k.endswith("dt_bias"))
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
