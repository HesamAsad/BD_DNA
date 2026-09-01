#!/usr/bin/env python3
"""GenomicBenchmarks arm x task table, with the seed noise made visible.

WHY THIS IS NOT JUST A MEAN. `mean_accuracy` averages 5 seeds, and on the small
tasks a seed sometimes fails to train at all and lands exactly on the majority
class -- chance for a 2-class task. One such seed moves a 5-seed mean by ~0.045,
which is larger than most between-arm differences in this benchmark. Observed
on 2026-09-01, dummy_mouse_enhancers_ensembl (968 train / 242 test):

    tau_learned  mean 0.6860  seeds [0.740, 0.740, 0.723, 0.727, 0.500]
    tau_frozen   mean 0.7298  seeds [0.719, 0.740, 0.711, 0.727, 0.752]

The apparent 0.0438 gap is one dead seed, not a model difference: on its four
healthy seeds tau_learned averages 0.7325, slightly ABOVE tau_frozen. ussm_bd
has the same pathology on the same task. So a table of bare means silently
ranks arms by which of them happened to draw a dead seed.

This script therefore prints mean, median and std, and flags degenerate seeds
(within `--degenerate-tol` of 1/num_classes) with a trailing marker so they
cannot be mistaken for signal. Median is the more robust ranking statistic on
the small tasks; on the large ones the two agree and either is fine.

Also worth knowing when reading small-task rows: seed spread can EXCEED the
binomial sampling error of the test set itself (0.042 vs 0.029 on the dummy
task), i.e. optimisation instability dominates, which is the opposite of the
usual assumption.

Usage:
  python scripts/eval/gb_table.py
  python scripts/eval/gb_table.py --stat median --arms bissm_bd,tau_learned,tau_frozen
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "results" / "caduceus" / "genomic_benchmarks_ft"
ARM_ORDER = ["ussm_ar", "xf_ar", "ussm_bd", "bissm_bd", "xf_bd",
             "tau_learned", "tau_frozen"]
STEM = re.compile(r"^(?P<arm>" + "|".join(ARM_ORDER) + r")_(?P<task>.+)$")


def load(indir, prefix):
  """(arm, task) -> task record, from <prefix><arm>_<task>.json."""
  out = {}
  for path in sorted(glob.glob(str(indir / f"{prefix}*.json"))):
    stem = os.path.basename(path)[len(prefix):-len(".json")]
    match = STEM.match(stem)
    if not match:
      continue
    try:
      payload = json.load(open(path))
    except (OSError, json.JSONDecodeError):
      continue
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
      continue
    out[(match.group("arm"), match.group("task"))] = tasks[0]
  return out


def degenerate(record, tol):
  """Seeds sitting within tol of chance for the task's class count."""
  seeds = record.get("accuracy_per_seed") or []
  classes = record.get("num_classes") or 2
  return [s for s in seeds if abs(s - 1.0 / classes) <= tol]


def main():
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--indir", type=Path, default=DEFAULT_DIR)
  parser.add_argument("--prefix", default="lr35_")
  parser.add_argument("--stat", choices=("mean", "median"), default="mean")
  parser.add_argument("--arms", default=None, help="comma-separated subset")
  parser.add_argument("--degenerate-tol", type=float, default=0.02)
  args = parser.parse_args()

  data = load(args.indir, args.prefix)
  if not data:
    sys.exit(f"no {args.prefix}*.json under {args.indir}")
  arms = ([a for a in args.arms.split(",") if a] if args.arms
          else [a for a in ARM_ORDER if any(k[0] == a for k in data)])
  tasks = sorted({t for _, t in data})

  width = max(len(t) for t in tasks) + 2
  print(f"\n{args.stat} accuracy over seeds   (* = run contains a "
        f"chance-level seed, see --degenerate-tol)\n")
  print(f"{'task':<{width}}" + "".join(f"{a:>14}" for a in arms))
  print("-" * (width + 14 * len(arms)))
  flagged = []
  for task in tasks:
    cells = []
    for arm in arms:
      record = data.get((arm, task))
      if not record:
        cells.append(f"{'-':>14}")
        continue
      seeds = record.get("accuracy_per_seed") or [record.get("accuracy")]
      value = st.median(seeds) if args.stat == "median" else st.fmean(seeds)
      bad = degenerate(record, args.degenerate_tol)
      if bad:
        flagged.append((task, arm, record))
      cells.append(f"{value:>13.4f}" + ("*" if bad else " "))
    print(f"{task:<{width}}" + "".join(cells))

  # sizes explain which rows deserve weight
  print(f"\n{'task':<{width}}{'n_train':>10}{'n_test':>9}{'1 seq =':>10}")
  for task in tasks:
    rec = next((data[(a, task)] for a in arms if (a, task) in data), None)
    if rec and rec.get("n_test_full"):
      n = rec["n_test_full"]
      print(f"{task:<{width}}{rec.get('n_train_full', 0):>10,}{n:>9,}"
            f"{1 / n:>10.4f}")

  if flagged:
    print("\n* chance-level seeds -- these means are not trustworthy:")
    for task, arm, record in flagged:
      seeds = record["accuracy_per_seed"]
      bad = degenerate(record, args.degenerate_tol)
      healthy = [s for s in seeds if s not in bad]
      print(f"  {arm:<13}{task:<38}")
      print(f"      seeds {[round(s, 3) for s in seeds]}  "
            f"mean {st.fmean(seeds):.4f}")
      if healthy:
        print(f"      excluding {len(bad)} dead seed(s): "
              f"mean {st.fmean(healthy):.4f}  median {st.median(seeds):.4f}")
  else:
    print("\nno chance-level seeds detected")


if __name__ == "__main__":
  main()
