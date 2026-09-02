#!/usr/bin/env python3
"""Read a copy-gate sweep and report the ladder: which offsets a model can copy.

The gate trains on x[i] = x[i-D] for fixed D. Chance is the uniform floor over
four bases, ln 4 = 1.3863 nats, so the natural score is the fraction of that
floor the model recovered:

    recovered = 1 - val_nll / ln4

    ~0.00   learned nothing; copying at this offset is out of reach
    >0.50   PASS
    ~0.97   what the 512 nt sanity rung reached (val/nll 0.047)

The largest passing D is the model's copy range, and that single number is what
a candidate long-range fix has to move. For every arm measured so far it sits
well under 1024, matching the 1-2 kb effective range found by direct prefix
intervention.

A failure at the SHORT offsets invalidates the whole ladder: it means the
pipeline or the candidate model is broken, not that long-range copying is hard.
The report says so explicitly rather than leaving it to be noticed.

Usage:
  python scripts/eval/copy_gate_report.py --tag baseline
  python scripts/eval/copy_gate_report.py --tag baseline --tag mycandidate
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FLOOR = math.log(4)          # 1.3863 nats, uniform over {A,C,G,T}
PASS = 0.50                  # fraction of the floor that counts as learning


def best_nll(run_dir: Path):
  """Lowest val/nll across every metrics.csv under this run.

  Takes the LARGEST metrics file when several exist: a restarted run leaves a
  short aborted version_0 beside the real version_1, and picking the first by
  name silently reports the stub. That mistake produced a nonsense arm ordering
  earlier in this project.
  """
  best, steps, seen = None, None, 0
  files = sorted(glob.glob(str(run_dir / "**" / "metrics.csv"), recursive=True),
                 key=lambda p: Path(p).stat().st_size, reverse=True)
  for path in files[:1]:
    for row in csv.DictReader(open(path)):
      value = row.get("val/nll")
      if not value:
        continue
      seen += 1
      try:
        v = float(value)
      except ValueError:
        continue
      if best is None or v < best:
        best, steps = v, row.get("step")
  return best, steps, seen


def status(run_dir: Path):
  """Distinguish 'still running' from 'crashed' from 'finished with no metric'."""
  err = run_dir / "train.err"
  out = run_dir / "train.out"
  for path in (out, err):
    if path.exists():
      text = path.read_text(errors="replace")[-8000:]
      if re.search(r"Traceback|CUDA out of memory|RuntimeError|ValueError", text):
        hit = re.search(r"(CUDA out of memory|RuntimeError: .*|ValueError: .*)",
                        text)
        return "CRASHED", (hit.group(1)[:70] if hit else "see train.err")
      if "copygate" in text and "exit=" in text:
        return "finished", ""
  return "running?", ""


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--tag", action="append", required=True, help="repeatable")
  ap.add_argument("--root", type=Path, default=REPO / "results" / "copy_gate")
  args = ap.parse_args()

  print(f"uniform floor = ln4 = {FLOOR:.4f} nats;  "
        f"recovered = 1 - nll/ln4;  PASS at {PASS:.2f}\n")
  ranges = {}
  for tag in args.tag:
    root = args.root / tag
    dirs = sorted(root.glob("D*"), key=lambda p: int(p.name[1:]))
    if not dirs:
      print(f"{tag}: no runs under {root}")
      continue
    print(f"=== {tag}")
    print(f"  {'offset':>8}{'val/nll':>10}{'recovered':>11}{'verdict':>10}  note")
    passed = []
    for d in dirs:
      D = int(d.name[1:])
      nll, step, seen = best_nll(d)
      state, detail = status(d)
      if nll is None:
        print(f"  {D:>8}{'-':>10}{'-':>11}{state:>10}  "
              f"{detail or f'no val/nll yet ({seen} rows)'}")
        continue
      rec = 1.0 - nll / FLOOR
      verdict = "PASS" if rec > PASS else "fail"
      if rec > PASS:
        passed.append(D)
      note = f"step {step}" + (f", {state}" if state != "finished" else "")
      print(f"  {D:>8}{nll:>10.4f}{rec:>11.3f}{verdict:>10}  {note}")
    if passed:
      ranges[tag] = max(passed)
      short = min(int(d.name[1:]) for d in dirs)
      if short not in passed:
        print(f"\n  *** the shortest offset ({short}) FAILED. The ladder is not "
              f"interpretable:\n      fix the pipeline or the model before "
              f"reading anything into the long offsets.")
      else:
        print(f"\n  copy range: {max(passed)} nt "
              f"(largest offset recovered above {PASS:.0%})")
    else:
      print("\n  copy range: NONE -- not even the shortest offset was learned")
    print()

  if len(ranges) > 1:
    print("--- comparison ---")
    base = None
    for tag, r in ranges.items():
      if base is None:
        base = r
        print(f"  {tag:<20}{r:>8} nt   (reference)")
      else:
        print(f"  {tag:<20}{r:>8} nt   {r / base:>5.1f}x the reference")
    print("\n  A candidate that does not extend the range has not fixed "
          "anything,\n  however much it improves likelihood on real DNA.")


if __name__ == "__main__":
  main()
