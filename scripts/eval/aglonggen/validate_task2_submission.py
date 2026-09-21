#!/usr/bin/env python3
"""Check an external AG-LongGen Task 2 submission before it costs AlphaGenome calls.

WHY THIS EXISTS. `task2_score.py` sends 2,400 records to a paid, rate-limited
oracle. Every failure mode below produces a file that scores *successfully* and
gives a number that is not comparable to ours -- a regenerated flank, a
lower-cased base, a resampled locus list. None of them raise. So they are
checked here, before the expensive step, against our released generation file.

Usage:
  python scripts/eval/aglonggen/validate_task2_submission.py \
      --submission evo2_generations.json \
      --reference  results/aglonggen/task2_gen_cosfinal_bissm.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

LENGTH = 16384
VALID = set("ACGT")
# Conditions an autoregressive model can legitimately fill. `ca` and `mismatch`
# require consuming a committed SUFFIX, which an AR model cannot do natively --
# a submission claiming them should say how, so they warn rather than pass.
AR_NATIVE = {"denovo"}
MODEL_FREE = {"real", "dinuc"}


def load(path: Path) -> list[dict]:
  blob = json.loads(Path(path).read_text())
  records = blob.get("records")
  if records is None:
    raise SystemExit(f"{path}: no 'records' key")
  return records


def key(record: dict) -> tuple:
  return (record["chrom"], int(record["start"]), int(record["gap_nt"]))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--submission", type=Path, required=True)
  ap.add_argument("--reference", type=Path, required=True,
                  help="our released generation JSON; supplies the locus list "
                       "and the committed flanks")
  ap.add_argument("--allow-partial", action="store_true",
                  help="accept a submission that does not cover every released "
                       "(locus, gap) cell. Off by default: a partial file scores "
                       "cleanly and yields a mean over a DIFFERENT set of loci "
                       "than ours, which is not comparable.")
  ap.add_argument("--condition", default=None,
                  help="only check this condition (default: all present)")
  args = ap.parse_args()

  sub = load(args.submission)
  ref = load(args.reference)

  # The reference's `real` rows carry the ground-truth 16,384 nt window, which
  # is what the flanks must match byte for byte.
  truth = {key(r): r["sequence"] for r in ref if r["condition"] == "real"}
  ref_cells = {key(r) for r in ref}
  if not truth:
    raise SystemExit("reference has no 'real' rows; wrong file?")

  errors: list[str] = []
  warnings: list[str] = []
  seen: collections.Counter = collections.Counter()

  def err(record, message):
    if len(errors) < 40:
      errors.append(f"  {record.get('chrom','?')}:{record.get('start','?')} "
                    f"gap={record.get('gap_nt','?')} "
                    f"cond={record.get('condition','?')}: {message}")

  for record in sub:
    condition = record.get("condition", "?")
    if args.condition and condition != args.condition:
      continue
    seen[condition] += 1

    for field in ("chrom", "start", "gap_nt", "left_nt", "sequence",
                  "interior", "condition"):
      if field not in record:
        err(record, f"missing field '{field}'")
        break
    else:
      k = key(record)
      sequence, interior = record["sequence"], record["interior"]
      gap_nt, left_nt = int(record["gap_nt"]), int(record["left_nt"])

      if k not in ref_cells:
        err(record, "locus/gap not in the released list -- did you resample?")
        continue
      if len(sequence) != LENGTH:
        err(record, f"len(sequence)={len(sequence)}, expected {LENGTH}")
        continue
      if len(interior) != gap_nt:
        err(record, f"len(interior)={len(interior)}, expected gap_nt={gap_nt}")
      if set(sequence) - VALID:
        bad = "".join(sorted(set(sequence) - VALID))[:8]
        err(record, f"non-ACGT characters present: {bad!r} "
                    "(hg38 is soft-masked -- upper-case it)")
        continue
      if sequence[left_nt:left_nt + gap_nt] != interior:
        err(record, "sequence[left:left+gap] != interior")

      real = truth.get(k)
      if real is not None:
        if sequence[:left_nt] != real[:left_nt]:
          err(record, "LEFT flank was regenerated; it must be committed verbatim")
        if sequence[left_nt + gap_nt:] != real[left_nt + gap_nt:]:
          err(record, "RIGHT flank was regenerated; it must be committed verbatim")
        if condition != "real" and interior == real[left_nt:left_nt + gap_nt]:
          warnings.append(f"  {k} {condition}: interior is byte-identical to the "
                          "real interior -- exact recovery, or a copy bug?")

  # Coverage against the released grid.
  for condition, n in sorted(seen.items()):
    cells = {key(r) for r in sub if r.get("condition") == condition}
    missing = len(ref_cells - cells)
    note = f"  {condition:9s} {n:5d} records, {len(cells):4d}/{len(ref_cells)} cells"
    if missing:
      note += f"  ({missing} MISSING)"
    print(note)
    if missing and not args.allow_partial:
      # Not a warning. A partial file scores without complaint and produces a
      # mean over a different locus set than ours -- silently incomparable.
      errors.append(f"  condition '{condition}' covers {len(cells)} of "
                    f"{len(ref_cells)} released cells ({missing} missing). "
                    "Pass --allow-partial if this is deliberate.")
    if condition not in AR_NATIVE | MODEL_FREE:
      warnings.append(
        f"  condition '{condition}' requires consuming a committed SUFFIX. An "
        "autoregressive model cannot do that natively -- say how it was done.")

  print()
  if warnings:
    print(f"{len(warnings)} warning(s):")
    for w in warnings[:10]:
      print(w)
    print()
  if errors:
    print(f"FAILED: {len(errors)} problem(s) shown"
          f"{' (truncated)' if len(errors) >= 40 else ''}:")
    for e in errors:
      print(e)
    return 1
  print("PASSED -- safe to run task2_score.py on this file.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
