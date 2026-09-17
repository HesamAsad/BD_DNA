#!/usr/bin/env python3
"""Every MaveDB arm against the baselines that matter, in one table.

The point of this script is that a model number on this benchmark is not
interpretable on its own. Two zero-parameter features beat most of what we have
trained:

    non-synonymous codon count     +0.337
    hgvs_pro protein-event count   +0.30931   (3 distinct values, 0/12 wrong way)
    nucleotide edit count          +0.170

So each arm is reported four ways: raw signed Spearman, partial after removing
the nucleotide counting family, partial after ALSO removing the protein-event
count, and the fraction of its raw signal that survives both. The last column is
the one to read -- it is what the model knows that counting does not.

Signed, never absolute: all 12 assays share one direction, so `abs()` credits an
anti-correlated assay as skill and inflates the weaker arms.
"""

from __future__ import annotations

import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.partial_corr import (  # noqa: E402
  _count_features, _protein_events)

BASELINES = {
  "non-synonymous codon count": 0.337,
  "protein-event count": 0.30931,
  "nucleotide edit count": 0.170,
}


def partial(y, x, controls):
  rank = lambda v: rankdata(v)
  target, predictor = rank(y), rank(x)
  design = np.column_stack([rank(c) for c in controls] + [np.ones(len(y))])
  solve = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
  ry, rx = solve(target), solve(predictor)
  if np.std(ry) < 1e-12 or np.std(rx) < 1e-12:
    return float("nan")
  return spearmanr(ry, rx).statistic


def evaluate(path):
  rows = list(csv.DictReader(open(path)))
  groups = defaultdict(list)
  for row in rows:
    groups[row["score_set_urn"]].append(row)
  raw, p_nt, p_both, dropped, total = [], [], [], 0, 0
  for assay in groups.values():
    total += len(assay)
    keep = [r for r in assay
            if np.isfinite(float(r["predicted_fitness"] or "nan"))]
    dropped += len(assay) - len(keep)
    if len(keep) < 10:
      continue
    y = np.array([float(r["experimental_score"]) for r in keep])
    x = np.array([float(r["predicted_fitness"]) for r in keep])
    feats = [_count_features(r["hgvs_nt"]) for r in keep]
    nt = np.array([f[0] for f in feats], float)
    delta = np.array([f[1] for f in feats], float)
    events = np.array([f[2] for f in feats], float)
    pro = np.array([_protein_events(r.get("hgvs_pro")) for r in keep], float)
    raw.append(spearmanr(x, y).statistic)
    p_nt.append(partial(y, x, [nt, delta, events]))
    p_both.append(partial(y, x, [nt, delta, events, pro]))
  mean = lambda v: float(np.nanmean(v)) if v else float("nan")
  return {
    "assays": len(raw), "scored": total - dropped, "unscored": dropped,
    "raw": mean(raw), "partial_nt": mean(p_nt), "partial_both": mean(p_both),
    "negative_assays": sum(1 for r in raw if r < 0),
  }


def main():
  paths = sys.argv[1:] or sorted(
    glob.glob(str(REPO / "results/dnahnet/mavedb/*/predictions.csv")))
  print(f"{'arm':40s} {'assays':>6s} {'scored':>7s} {'raw':>9s} "
        f"{'part.nt':>9s} {'part.+pro':>10s} {'kept':>6s} {'neg':>4s}")
  print("-" * 100)
  results = []
  for path in paths:
    try:
      row = evaluate(path)
    except Exception as exc:                     # noqa: BLE001
      print(f"{Path(path).parent.name:40s}  FAILED {type(exc).__name__}: {exc}")
      continue
    row["arm"] = Path(path).parent.name
    results.append(row)
  for row in sorted(results, key=lambda r: -(r["partial_both"]
                                             if np.isfinite(r["partial_both"])
                                             else -9)):
    kept = (row["partial_both"] / row["raw"]
            if row["raw"] and np.isfinite(row["partial_both"]) else float("nan"))
    print(f"{row['arm']:40s} {row['assays']:6d} {row['scored']:7d} "
          f"{row['raw']:+9.5f} {row['partial_nt']:+9.5f} "
          f"{row['partial_both']:+10.5f} {kept:5.0%} {row['negative_assays']:4d}")
  print("\nZero-parameter baselines to clear (macro signed Spearman):")
  for name, value in BASELINES.items():
    print(f"  {name:32s} {value:+.5f}")
  print("\n'kept' = partial(+pro) / raw: the share of the arm's signal that is "
        "NOT explained by counting mutations.")


if __name__ == "__main__":
  sys.exit(main())
