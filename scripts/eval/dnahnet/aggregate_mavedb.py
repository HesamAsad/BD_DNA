#!/usr/bin/env python3
"""Average independent MaveDB score runs and report their stability."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.dnahnet.mavedb import spearmanr, summarize_predictions


AVERAGED_FIELDS = (
  "predicted_fitness", "predicted_fitness_per_nt",
  "wt_nelbo", "mut_nelbo", "wt_nelbo_per_nt", "mut_nelbo_per_nt",
)


def read_predictions(path: Path) -> list[dict]:
  with path.open(encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def aggregate_runs(runs: list[list[dict]]) -> tuple[list[dict], dict]:
  if len(runs) < 2:
    raise ValueError("At least two independent prediction runs are required")
  reference_accessions = [row["accession"] for row in runs[0]]
  for run in runs[1:]:
    if [row["accession"] for row in run] != reference_accessions:
      raise ValueError("Prediction runs do not contain identical ordered records")

  combined = []
  for row_index, reference in enumerate(runs[0]):
    row = dict(reference)
    for field in AVERAGED_FIELDS:
      row[field] = float(np.mean([
        float(run[row_index][field]) for run in runs]))
    combined.append(row)

  reference_scores = [float(row["predicted_fitness"]) for row in runs[0]]
  agreements = []
  for run_index, run in enumerate(runs[1:], start=1):
    comparison = [float(row["predicted_fitness"]) for row in run]
    agreements.append({
      "left_run": 0,
      "right_run": run_index,
      "prediction_spearman": spearmanr(reference_scores, comparison),
      "mean_absolute_score_delta": float(np.mean(np.abs(
        np.asarray(reference_scores) - np.asarray(comparison)))),
    })
  return combined, {"run_agreements": agreements}


def _atomic_json(path: Path, value):
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def _atomic_csv(path: Path, records):
  fields = list(records[0])
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", newline="", dir=path.parent,
      delete=False) as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    temporary = Path(handle.name)
  os.replace(temporary, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prediction", type=Path, action="append", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  args = parser.parse_args()

  runs = [read_predictions(path) for path in args.prediction]
  combined, stability = aggregate_runs(runs)
  summary = summarize_predictions(combined)
  summary.update(stability)
  summary.update({
    "label": args.label,
    "prediction_runs": [str(path.resolve()) for path in args.prediction],
    "num_independent_runs": len(runs),
    "score_definition": "mean of paired NELBO(WT) - NELBO(mutant) runs",
    "headline_metric": "macro mean absolute per-assay Spearman",
  })

  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_csv(args.output_dir / "predictions.csv", combined)
  _atomic_json(args.output_dir / "summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
