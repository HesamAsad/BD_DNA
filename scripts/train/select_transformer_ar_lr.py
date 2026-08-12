#!/usr/bin/env python3
"""Select the lowest validation-NLL Transformer AR pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def checkpoint_score(path: Path) -> float:
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  scores = []
  for state in checkpoint.get("callbacks", {}).values():
    if isinstance(state, dict) and state.get("best_model_score") is not None:
      scores.append(float(state["best_model_score"]))
  if not scores:
    raise ValueError(f"No validation score found in {path}")
  return min(scores)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pilot-root", type=Path, required=True)
  parser.add_argument("--learning-rates", nargs="+", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  rows = []
  for learning_rate in args.learning_rates:
    checkpoint = args.pilot_root / f"lr{learning_rate}" / "checkpoints/best.ckpt"
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    rows.append({
      "learning_rate": learning_rate,
      "validation_nll": checkpoint_score(checkpoint),
      "checkpoint": str(checkpoint.resolve()),
    })
  rows.sort(key=lambda row: row["validation_nll"])
  result = {"winner": rows[0], "pilots": rows}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
