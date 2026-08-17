#!/usr/bin/env python3
"""How much MaveDB signal survives after removing the trivial confounds?

The headline macro Spearman on this benchmark is partly explained by features
that need no model at all -- chiefly *how many* nucleotide substitutions a
variant carries, which correlates with the experimental score and also with any
likelihood difference, because more edits mean more chances to lower the
likelihood. Counting protein-level events straight from the `hgvs_pro` column
of any predictions.csv reaches macro |rho| 0.30931 with no model at all, above
every model we have trained, and it points the right way on all 12 assays. That
baseline is computed here, not in `mavedb.py` -- `mavedb.py` only carries
`hgvs_pro` through as a metadata field and summarises model predictions.

This script reports the *partial* Spearman between prediction and experimental
score, controlling for:

  n_nt_diff    number of nucleotide edits parsed from `hgvs_nt`
  len_delta    change in sequence length (indels vs substitutions)
  n_events     number of semicolon-separated events in the `hgvs_nt` bracket

Method: rank-transform every column, regress both the prediction and the target
on the controls by least squares, and correlate the residuals. That is the
standard partial-correlation construction on ranks (Spearman partial), so a
value near zero means "this model adds nothing beyond counting edits".

Reported per assay and as a macro mean over assays. Unlike the main scorer this
does NOT take abs() -- an anti-correlated assay counts against the model, which
is the honest reading given all 12 assays share one direction.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# `c.[135T>C;138A>G]`, `c.201G>C`, `c.=`, and indel forms like `c.12_14del`.
_EVENT = re.compile(r"(\d+)(?:_(\d+))?([A-Za-z>=]+)")


def _count_features(hgvs_nt: str):
  """(n_nt_diff, len_delta, n_events) parsed from an HGVS nucleotide string."""
  if not isinstance(hgvs_nt, str) or hgvs_nt.strip() in {"", "c.=", "n.="}:
    return 0, 0, 0
  body = hgvs_nt.strip()
  body = body[body.find("[") + 1:body.rfind("]")] if "[" in body else body[2:]
  events = [e for e in body.split(";") if e]
  n_nt, delta = 0, 0
  for event in events:
    match = _EVENT.search(event)
    if match is None:
      continue
    start, stop, op = match.group(1), match.group(2), match.group(3).lower()
    span = 1 if stop is None else abs(int(stop) - int(start)) + 1
    if ">" in op:
      n_nt += 1
    elif "delins" in op:
      n_nt += span
    elif "del" in op:
      n_nt += span
      delta -= span
    elif "ins" in op or "dup" in op:
      n_nt += span
      delta += span
    else:
      n_nt += span
  return n_nt, delta, len(events)


def partial_spearman(x, y, controls):
  """Spearman partial correlation of x and y given `controls` (list of arrays).

  Rank-transform everything, then correlate the residuals of x and y after
  least-squares regression on the ranked controls (with an intercept).
  """
  def rank(v):
    return stats.rankdata(np.asarray(v, dtype=float))

  xr, yr = rank(x), rank(y)
  cols = [rank(c) for c in controls]
  # Drop controls that are constant on this assay -- they carry no information
  # and would make the design matrix rank-deficient.
  cols = [c for c in cols if np.ptp(c) > 0]
  if not cols:
    return stats.spearmanr(xr, yr).statistic
  design = np.column_stack([np.ones_like(xr)] + cols)
  def residual(v):
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta
  rx, ry = residual(xr), residual(yr)
  if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
    return float("nan")
  return float(stats.pearsonr(rx, ry).statistic)


def evaluate(path: Path):
  frame = pd.read_csv(path)
  features = frame["hgvs_nt"].apply(_count_features)
  frame["n_nt_diff"] = [f[0] for f in features]
  frame["len_delta"] = [f[1] for f in features]
  frame["n_events"] = [f[2] for f in features]

  rows = []
  for urn, group in frame.groupby("score_set_urn"):
    group = group.dropna(subset=["experimental_score", "predicted_fitness"])
    if len(group) < 10:
      continue
    raw = stats.spearmanr(
      group["predicted_fitness"], group["experimental_score"]).statistic
    partial = partial_spearman(
      group["predicted_fitness"], group["experimental_score"],
      [group["n_nt_diff"], group["len_delta"], group["n_events"]])
    count_only = stats.spearmanr(
      group["n_nt_diff"], group["experimental_score"]).statistic
    rows.append({
      "score_set_urn": urn,
      "target": group["target"].iloc[0],
      "n": int(len(group)),
      "signed_spearman": float(raw),
      "partial_spearman": float(partial),
      "count_baseline_spearman": float(count_only),
    })
  macro = lambda key: float(np.nanmean([r[key] for r in rows]))
  return {
    "predictions": str(path),
    "assays": len(rows),
    "macro_signed_spearman": macro("signed_spearman"),
    "macro_partial_spearman": macro("partial_spearman"),
    "macro_abs_spearman": float(
      np.nanmean([abs(r["signed_spearman"]) for r in rows])),
    "macro_count_baseline_spearman": macro("count_baseline_spearman"),
    "per_assay": rows,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("predictions", type=Path, nargs="+",
                      help="one or more results/.../predictions.csv")
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  results = []
  print(f"{'arm':<34} {'signed':>9} {'partial':>9} {'|rho|':>9} {'count':>9}")
  print("-" * 74)
  for path in args.predictions:
    summary = evaluate(path)
    results.append(summary)
    label = path.parent.name
    print(f"{label:<34} {summary['macro_signed_spearman']:>9.5f} "
          f"{summary['macro_partial_spearman']:>9.5f} "
          f"{summary['macro_abs_spearman']:>9.5f} "
          f"{summary['macro_count_baseline_spearman']:>9.5f}")
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main()
