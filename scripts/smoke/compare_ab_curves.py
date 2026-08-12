#!/usr/bin/env python
"""Diff two CSVLogger metric curves from the boundary_impl A/B.

The two runs share a seed, a data order and an RNG stream, so an equivalent
rewrite differs only by floating-point association. That difference does not
stay bounded: it perturbs the weights, which perturbs later losses, so the
trajectories separate for a while and then reconverge. The **max pointwise
gap is therefore not a test of equivalence** -- a chaotic-but-unbiased
trajectory pair fails it, and a slow systematic drift can pass it.

What actually distinguishes equivalence from a bug:

* the endpoints agree (a real bug keeps the curves apart at the end); and
* the signed gaps are unbiased (a real bug pushes them one way).

So the gate is on the final gap and on the mean signed gap. The max pointwise
gap and the full signed sequence are printed for inspection but not gated.
"""

import argparse
import csv
import glob
import os
import sys


def load(run_dir):
  """Read <run_dir>/csv_logs/version_*/metrics.csv into {metric: {step: value}}."""
  pattern = os.path.join(run_dir, "csv_logs", "version_*", "metrics.csv")
  matches = sorted(glob.glob(pattern))
  if not matches:
    raise SystemExit(f"no metrics.csv under {pattern}")
  series = {}
  for path in matches:
    with open(path, newline="") as handle:
      for row in csv.DictReader(handle):
        try:
          step = int(row["step"])
        except (KeyError, TypeError, ValueError):
          continue
        for key, raw in row.items():
          if key in ("step", "epoch") or raw in (None, ""):
            continue
          try:
            series.setdefault(key, {})[step] = float(raw)
          except ValueError:
            continue
  return series


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--layer-major", required=True)
  parser.add_argument("--block-major", required=True)
  parser.add_argument("--final-tolerance", type=float, default=2e-3,
                      help="max relative gap at the last common step")
  parser.add_argument("--bias-tolerance", type=float, default=5e-3,
                      help="max |mean signed relative gap| over the run")
  args = parser.parse_args()

  new, old = load(args.layer_major), load(args.block_major)
  shared = sorted(set(new) & set(old))
  if not shared:
    raise SystemExit(f"no shared metrics: {sorted(new)} vs {sorted(old)}")

  print(f"{'metric':<26s}{'n':>4s}{'last(new)':>12s}{'last(old)':>12s}"
        f"{'final|rel|':>12s}{'mean rel':>12s}{'max|rel|':>12s}")
  failures = []
  for metric in shared:
    steps = sorted(set(new[metric]) & set(old[metric]))
    if not steps:
      continue
    signed = [(new[metric][s] - old[metric][s]) / max(abs(old[metric][s]), 1e-12)
              for s in steps]
    last = steps[-1]
    final_rel, max_rel = abs(signed[-1]), max(abs(v) for v in signed)
    mean_rel = sum(signed) / len(signed)
    print(f"{metric:<26s}{len(steps):>4d}{new[metric][last]:>12.6f}"
          f"{old[metric][last]:>12.6f}{final_rel:>12.3e}"
          f"{mean_rel:>+12.3e}{max_rel:>12.3e}")
    # Only loss-like metrics must agree. Throughput and memory telemetry are
    # expected to differ -- that difference is the point of the rewrite.
    if not any(k in metric for k in ("loss", "nll", "bpb", "ppl")):
      continue
    if final_rel > args.final_tolerance:
      failures.append(f"{metric}: endpoints differ by {final_rel:.3e} "
                      f"> {args.final_tolerance:.3e}")
    if abs(mean_rel) > args.bias_tolerance:
      failures.append(f"{metric}: signed gap is biased ({mean_rel:+.3e}, "
                      f"|.| > {args.bias_tolerance:.3e})")

  for metric in ("val/nll", "trainer/loss"):
    if metric not in shared:
      continue
    steps = sorted(set(new[metric]) & set(old[metric]))
    signs = "".join("+" if new[metric][s] >= old[metric][s] else "-"
                    for s in steps)
    print(f"\n{metric} signed-gap pattern (chaotic+unbiased looks mixed, "
          f"a bug looks one-sided):\n  {signs}")

  print()
  if failures:
    for line in failures:
      print(f"FAIL: {line}")
    sys.exit(1)
  print(f"PASS: endpoints within {args.final_tolerance:.3e} and unbiased "
        f"within {args.bias_tolerance:.3e}")


if __name__ == "__main__":
  main()
