#!/usr/bin/env python3
"""Assemble the five-arm comparison from all three result sources.

Nothing else joins these up: training writes CSV logs, ppl_ssm_baselines.sh
writes a TSV, and each GenomicBenchmarks job writes its own JSON. Reading them
by hand is where a stale file or a missing arm slips into a table unnoticed --
which is the failure this whole campaign has been cleaning up after. So this
refuses to paper over gaps: an arm with no result is printed as MISSING, never
omitted.

WHAT IT WILL AND WILL NOT LET YOU CONCLUDE.

  * The AR arms report an EXACT token NLL. The BD arms report a NELBO, an UPPER
    bound. AR-vs-BD perplexity gaps are therefore bounds, not measurements, and
    the table marks which is which. AR-vs-AR and BD-vs-BD are measurements.
  * Our val NLL is computed on Caduceus's own validation split, which their
    2^20 interval stretch contaminates: 3.03% of it also appears in train
    (scripts/data/audit_hg38_corpus.py). The bias is COMMON-MODE -- every arm
    sees the same split -- so arm-vs-arm is unaffected, but the absolute number
    is optimistic against a clean split and is not comparable to one.
  * The Caduceus published column is a REFERENCE, not a target. Their model
    pretrained at length 1,024 on the same corpus; ours at whatever
    --length says. A gap that is really a context-length difference should not
    be read as an architecture difference.

Usage:
  python scripts/eval/hg38_report.py
  python scripts/eval/hg38_report.py --length 1024 --json results/hg38_report.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.provenance import write_json  # noqa: E402

# arm key -> (display label, reports a NELBO upper bound rather than exact NLL)
ARMS = [
  ("ussm_ar", "uSSM-AR", False),
  ("ussm_bd", "uSSM-BD", True),
  ("bissm_bd", "BiSSM-BD", True),
  ("xf_ar", "Transformer-AR", False),
  ("xf_bd", "Transformer-BD", True),
]
# ppl_ssm_baselines.sh names its rows differently from the run directories.
PPL_ALIAS = {"ussm_ar": "ussm_ar", "ussm_bd": "ussm_bd", "bissm_bd": "bissm",
             "xf_ar": "xf_ar", "xf_bd": "xf_bd"}


def training_val_nll(run_dir: Path):
  """Best val/nll seen in a run's CSV logs, and the step it happened at.

  Reads every version_* -- a resumed run writes a new one rather than
  appending, and this campaign resumed three arms, so reading only version_0
  would report the best value from before the restart.
  """
  paths = sorted(glob.glob(str(run_dir / "csv_logs" / "version_*" / "metrics.csv")))
  best = None
  last_step = 0
  for path in paths:
    try:
      with open(path) as handle:
        for row in csv.DictReader(handle):
          step = row.get("step") or ""
          if step.isdigit():
            last_step = max(last_step, int(step))
          value = row.get("val/nll")
          if value:
            try:
              value = float(value)
            except ValueError:
              continue
            if best is None or value < best[0]:
              best = (value, int(step) if step.isdigit() else None)
    except OSError:
      continue
  return best, last_step


def latest_ppl(pattern: str, run_root: Path):
  """arm -> row, from the most recent ppl TSV THAT SCORED THESE CHECKPOINTS.

  Matching on "most recent file with rows" is not enough and produced a real
  wrong answer the first time this ran: it picked up
  ppl_ssm_baselines_120135.tsv, a prokaryote-corpus run from before this
  campaign, and printed its NLLs beside the hg38 arms as though they belonged
  to them. Nothing about the numbers looked wrong.

  Every row carries the checkpoint it scored, so require that path to live
  under this campaign's run root. A row that does not is from another corpus,
  another length, or another era, and is dropped.
  """
  root = str(run_root.resolve())
  for path in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
    rows = {}
    with open(path) as handle:
      for row in csv.DictReader(handle, delimiter="\t"):
        arm = (row.get("arm") or "").strip()
        ckpt = (row.get("checkpoint") or "").strip()
        if not arm or not ckpt:
          continue
        if not str(Path(ckpt).resolve()).startswith(root):
          continue
        rows[arm] = row
    if rows:
      return rows, path
  return {}, None


def gb_results(directory: Path, prefix: str):
  """arm -> {task: accuracy} plus the mean, from the per-task JSONs."""
  out = {}
  for path in sorted(glob.glob(str(directory / f"{prefix}*.json"))):
    try:
      payload = json.load(open(path))
    except (OSError, json.JSONDecodeError):
      continue
    label = payload.get("label", "")
    if not label.startswith(prefix):
      continue
    arm = label[len(prefix):].lstrip("_")
    for task in payload.get("tasks", []):
      name = task.get("task")
      if not name:
        continue
      # the label carries the task too; strip it to recover the arm
      key = arm[: -len(name) - 1] if arm.endswith(name) else arm
      out.setdefault(key, {})[name] = {
        "accuracy": task.get("accuracy"),
        "std": task.get("accuracy_std"),
        "ph": task.get("caduceus_ph_published"),
        "ps": task.get("caduceus_ps_published"),
        "n_train_used": task.get("n_train_used"),
        "n_train_full": task.get("n_train_full"),
      }
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--runs", default=None,
                      help="run root; defaults to the launcher's convention")
  parser.add_argument("--ppl-glob", default="logs/eval/ppl_ssm_baselines_*.tsv")
  parser.add_argument("--gb-dir", default="results/caduceus/genomic_benchmarks_ft")
  parser.add_argument("--gb-prefix", default="hg38_")
  parser.add_argument("--json", type=Path, default=None)
  args = parser.parse_args()

  root = Path(args.runs) if args.runs else (
    REPO / ("outputs/hg38-caduceus" if args.length == 8192
            else f"outputs/hg38-caduceus-L{args.length}"))
  ppl_rows, ppl_path = latest_ppl(str(REPO / args.ppl_glob), root)
  gb = gb_results(REPO / args.gb_dir, args.gb_prefix)

  print(f"context length {args.length:,}   runs {root}")
  ppl_note = ppl_path or "none for these checkpoints yet (eval_hg38_arms.sh STAGE=ppl)"
  print(f"perplexity     {ppl_note}")
  print(f"benchmarks     {args.gb_dir}/{args.gb_prefix}*.json "
        f"({len(gb)} arms found)")
  print()

  header = (f"{'arm':<16}{'bound':<7}{'best val/nll':>13}{'@step':>8}"
            f"{'final nll':>11}{'ppl':>9}{'GB mean':>9}{'vs Ph':>8}")
  print(header)
  print("-" * len(header))

  report = {"length": args.length, "runs": str(root),
            "ppl_source": ppl_path, "arms": {}}
  for key, label, bounded in ARMS:
    best, last_step = training_val_nll(root / f"hg_{key}")
    row = ppl_rows.get(PPL_ALIAS.get(key, key), {})
    tasks = gb.get(key, {})
    accs = [t["accuracy"] for t in tasks.values() if t.get("accuracy") is not None]
    phs = [t["ph"] for t in tasks.values() if t.get("ph") is not None]
    gb_mean = sum(accs) / len(accs) if accs else None
    ph_mean = sum(phs) / len(phs) if phs else None

    def fmt(v, spec):
      # Width is the digits BEFORE the '.', not every digit in the spec:
      # ">11.4f" is width 11 precision 4, and joining all digits gives 114.
      width = int("".join(c for c in spec.split(".")[0] if c.isdigit()) or 8)
      if isinstance(v, (int, float)):
        return format(v, spec)
      return "-".rjust(width)

    nll = row.get("val_nll_nats")
    try:
      nll = float(nll)
    except (TypeError, ValueError):
      nll = None
    ppl = row.get("val_ppl")
    try:
      ppl = float(ppl)
    except (TypeError, ValueError):
      ppl = None

    print(f"{label:<16}{'NELBO' if bounded else 'exact':<7}"
          f"{fmt(best[0] if best else None, '>13.4f')}"
          f"{(str(best[1]) if best and best[1] is not None else '-'):>8}"
          f"{fmt(nll, '>11.4f')}{fmt(ppl, '>9.4f')}"
          f"{fmt(gb_mean, '>9.4f')}"
          f"{fmt(gb_mean - ph_mean if gb_mean is not None and ph_mean is not None else None, '>+8.4f')}")

    report["arms"][key] = {
      "label": label, "is_nelbo_upper_bound": bounded,
      "best_val_nll_training": best[0] if best else None,
      "best_val_nll_step": best[1] if best else None,
      "last_step_seen": last_step,
      "final_val_nll": nll, "final_val_ppl": ppl,
      "genomic_benchmarks": tasks,
      "gb_mean_accuracy": gb_mean,
      "caduceus_ph_mean_over_same_tasks": ph_mean,
    }

  missing = [k for k, entry in report["arms"].items()
             if entry["best_val_nll_training"] is None]
  print()
  if missing:
    print(f"MISSING training curves for: {', '.join(missing)}")
  incomplete = [k for k, entry in report["arms"].items()
                if entry["gb_mean_accuracy"] is None]
  if incomplete:
    print(f"MISSING benchmark results for: {', '.join(incomplete)}")
  print()
  print("BD arms report a NELBO (upper bound on NLL); AR arms report exact NLL.")
  print("AR-vs-BD gaps are bounds, not measurements.")
  print("Validation split is 3.03% contaminated by train (Caduceus's own 2^20")
  print("stretch) -- common-mode across arms, so arm-vs-arm is unaffected.")

  if args.json:
    write_json(args.json, report, args)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
  main()
