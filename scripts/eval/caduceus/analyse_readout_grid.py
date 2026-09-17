#!/usr/bin/env python3
"""The 2x2 readout x checkpoint grid, with the pre-registered endpoints.

Reads the four `ro2x2-*` probe result files and reports, in the order the plan
pre-registered them:

  1. (A) vs (C) -- the compression diagnostic. If the pooled hidden states probe
     well but the fixed-size recurrent state does not, the task-relevant signal
     is distributed across positions and was never concentrated into the state.
     That is an independent measurement of the compression limit on third-party
     data, and the most defensible result available here.
  2. rf=0.0 vs rf=0.5 -- the checkpoint verdict. SIX axes differ between these
     checkpoints (corpus, 54500 vs 8000 steps, objective, time conditioning, rf,
     var_min), so this main effect is a whole-checkpoint verdict, never an
     attribution to rf.
  3. THE INTERACTION, which is the actual prediction: (A) should gain more from
     rf=0.5 than (C) does, because only (A) reads the state whose consumer rf>0
     trains. Under rf=0.0 nothing ever consumed the reverse final state, so there
     was no pressure for it to be informative. If this interaction is absent or
     negative, the rf story is wrong and nothing further should be built on it.

Every mean is reported twice: over all 8 tasks, and over the 6 with no measured
train/test leakage (see genomic_benchmarks.LEAKY_TASKS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.eval.caduceus.genomic_benchmarks import (  # noqa: E402
  LEAKY_TASKS, PUBLISHED, REFERENCE, REFERENCE_COLUMNS)

RESULTS = REPO / "results" / "caduceus" / "genomic_benchmarks"
ARMS = [("gb_rf00", "pooled"), ("gb_rf00", "recurrent"),
        ("cos_rf05", "pooled"), ("cos_rf05", "recurrent")]


def load(checkpoint, readout):
  path = RESULTS / f"ro2x2-{checkpoint}-{readout}.json"
  if not path.exists():
    return None
  payload = json.loads(path.read_text())
  rows = payload.get("tasks", payload.get("rows", []))
  return {r["task"]: r for r in rows if "accuracy" in r}


def mean_over(table, tasks):
  values = [table[t]["accuracy"] for t in tasks if t in table]
  return float(np.mean(values)) if values else float("nan")


def main():
  grid = {key: load(*key) for key in ARMS}
  missing = [f"{c}-{r}" for (c, r), v in grid.items() if v is None]
  if missing:
    print(f"NOT YET AVAILABLE: {', '.join(missing)}")
  present = {k: v for k, v in grid.items() if v}
  if not present:
    return 1

  tasks = [t for t, _ in sorted(PUBLISHED.items())]
  clean = [t for t in tasks if t not in LEAKY_TASKS]

  # ---- per-task 2x2 ----
  print(f"\n{'task':34s} " + "".join(f"{c}/{r[:4]:>17s}"
                                     for c, r in ARMS) + "    PS_pub  leak")
  print("-" * 114)
  for task in tasks:
    cells = []
    for key in ARMS:
      table = grid[key]
      cells.append(f"{table[task]['accuracy']:17.4f}"
                   if table and task in table else f"{'-':>17s}")
    ps = REFERENCE.get(task, (None,) * 6)[REFERENCE_COLUMNS.index("ps")]
    flag = "LEAKY" if task in LEAKY_TASKS else ""
    print(f"{task:34s} " + "".join(cells) + f" {ps:9.4f}  {flag}")

  for label, subset in (("MEAN (8 tasks)", tasks),
                        ("MEAN (6 leak-free)", clean)):
    cells = [f"{mean_over(grid[k], subset):17.4f}" if grid[k] else f"{'-':>17s}"
             for k in ARMS]
    print(f"{label:34s} " + "".join(cells))

  # ---- the three endpoints ----
  def cell(checkpoint, readout, subset):
    table = grid[(checkpoint, readout)]
    return mean_over(table, subset) if table else float("nan")

  for label, subset in (("all 8 tasks", tasks), ("6 leak-free tasks", clean)):
    print(f"\n=== endpoints, {label} ===")
    a00, c00 = cell("gb_rf00", "recurrent", subset), cell("gb_rf00", "pooled", subset)
    a05, c05 = cell("cos_rf05", "recurrent", subset), cell("cos_rf05", "pooled", subset)
    print(f"  1. (A) recurrent vs (C) pooled")
    print(f"       on rf=0.0 checkpoint : {a00:.4f} vs {c00:.4f}  -> A-C = {a00 - c00:+.4f}")
    print(f"       on rf=0.5 checkpoint : {a05:.4f} vs {c05:.4f}  -> A-C = {a05 - c05:+.4f}")
    print(f"  2. rf=0.5 minus rf=0.0 (whole-checkpoint, 6 axes differ)")
    print(f"       with (C) pooled      : {c05 - c00:+.4f}")
    print(f"       with (A) recurrent   : {a05 - a00:+.4f}")
    interaction = (a05 - a00) - (c05 - c00)
    print(f"  3. INTERACTION (A's gain minus C's gain) : {interaction:+.4f}")
    if np.isfinite(interaction):
      verdict = ("SUPPORTS the rf story: the recurrent readout benefits more "
                 "from a trained right cache" if interaction > 0 else
                 "DOES NOT support the rf story -- the recurrent readout gains "
                 "no more than the pooled one. Do not build on rf>0.")
      print(f"       -> {verdict}")

  # ---- feature widths, for the p/n caveat ----
  print("\n=== feature width vs training rows (linear-probe sanity) ===")
  for key in ARMS:
    table = grid[key]
    if not table:
      continue
    dim = next(iter(table.values())).get("dim")
    worst = min(tasks, key=lambda t: table[t]["n_train"] if t in table else 1e9)
    n = table[worst]["n_train"]
    print(f"  {key[0]}/{key[1]:10s} dim={dim:>6}   smallest task {worst} "
          f"n={n}  p/n={dim / n:.1f}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
