#!/usr/bin/env python3
"""What the hg38 "35 billion token" corpus actually contains.

Run before drawing conclusions from any hg38 result, and before describing a
run's length in epochs.

THREE FACTS THIS ESTABLISHES, none of them obvious from the corpus size:

1. THE BED INTERVALS ARE 2^17 WIDE; CADUCEUS STRETCHES THEM TO 2^20.
   `human-sequences.bed` holds 34,021 train intervals of exactly 131,072 nt.
   HG38Dataset then does `self.df["end"] = self.df["start"] + 2**20`, an 8x
   extension past what the bed declares. We reproduce this deliberately -- the
   point is to see the same corpus they did -- but it is the cause of 2 and 3.

2. "ONE EPOCH" IS ~15 PASSES OVER THE UNIQUE GENOME.
   The intervals overlap heavily. Merged, the 35.67 Gb of windows cover only
   2.34 Gb of distinct sequence, so every base appears about 15.2 times. Our
   68,042-step run is 1.00 epochs over the WINDOW LIST and ~15 epochs over the
   genome. Anything that reasons from "it has only seen the data once" --
   overfitting expectations, scaling-law comparisons, claims about data
   efficiency -- is wrong by that factor.

3. 3.03% OF THE VALIDATION SPLIT IS ALSO IN TRAIN.
   The bed's own split is clean: at the native 131,072 width, train and valid
   overlap by exactly 0 bases. The 2^20 stretch pushes train intervals across
   into valid territory, contaminating 70.4 Mb of the 2.32 Gb validation
   corpus. This is Caduceus's leak, not ours, and we keep it for
   comparability -- but a val NLL on this corpus is optimistic by whatever 3%
   contamination buys, and that should be said rather than discovered.

   It does NOT undermine comparisons among our own arms: every arm sees the
   same corpus and the same contaminated validation split, so the bias is
   common-mode. It does mean our absolute val NLL is not directly comparable to
   a number computed on a clean split.

Usage:
  python scripts/data/audit_hg38_corpus.py
  python scripts/data/audit_hg38_corpus.py --json results/hg38_corpus_audit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MAX_ALLOWED_LENGTH = 2 ** 20          # their MAX_ALLOWED_LENGTH


def merge(intervals):
  """Union length of [start, end) intervals, and their unmerged total."""
  if not len(intervals):
    return 0, 0
  order = intervals[np.argsort(intervals[:, 0])]
  total = int((order[:, 1] - order[:, 0]).sum())
  merged = []
  for start, end in order:
    if merged and start <= merged[-1][1]:
      merged[-1][1] = max(merged[-1][1], end)
    else:
      merged.append([start, end])
  return total, sum(end - start for start, end in merged)


def coverage(frame, end_column):
  total = covered = 0
  for name in sorted(set(frame["chr_name"])):
    sub = frame[frame["chr_name"] == name][["start", end_column]].to_numpy()
    a, b = merge(sub)
    total += a
    covered += b
  return total, covered


def cross_overlap(a_frame, b_frame, end_column):
  """Bases of b that fall inside any interval of a."""
  total = overlap = 0
  for name in sorted(set(b_frame["chr_name"])):
    a = a_frame[a_frame["chr_name"] == name][["start", end_column]].to_numpy()
    b = b_frame[b_frame["chr_name"] == name][["start", end_column]].to_numpy()
    if not len(b):
      continue
    if len(a):
      a = a[np.argsort(a[:, 0])]
      merged = []
      for start, end in a:
        if merged and start <= merged[-1][1]:
          merged[-1][1] = max(merged[-1][1], end)
        else:
          merged.append([start, end])
      merged = np.array(merged)
    else:
      merged = np.zeros((0, 2), dtype=int)
    for start, end in b:
      total += end - start
      if len(merged):
        low = np.maximum(merged[:, 0], start)
        high = np.minimum(merged[:, 1], end)
        overlap += int(np.clip(high - low, 0, None).sum())
  return overlap, total


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--bed", default=str(REPO / "data/hg38/human-sequences.bed"))
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--global-batch", type=int, default=64)
  parser.add_argument("--json", type=Path, default=None)
  args = parser.parse_args()

  frame = pd.read_csv(args.bed, sep="\t",
                      names=["chr_name", "start", "end", "split"])
  native_width = int((frame["end"] - frame["start"]).mode().iloc[0])
  # numpy scalars throughout; cast at the boundary, not everywhere.
  frame = frame.assign(stretched=frame["start"] + MAX_ALLOWED_LENGTH)

  # value_counts() gives numpy int64 keys/values, which json.dumps rejects.
  counts = {str(k): int(v) for k, v in frame["split"].value_counts().items()}
  report = {"bed": args.bed, "native_interval_width": native_width,
            "stretched_to": MAX_ALLOWED_LENGTH,
            "stretch_factor": MAX_ALLOWED_LENGTH / native_width,
            "intervals": counts, "splits": {}}

  print(f"bed intervals are {native_width:,} nt wide; HG38Dataset stretches "
        f"every one to {MAX_ALLOWED_LENGTH:,} "
        f"({MAX_ALLOWED_LENGTH / native_width:.0f}x)")
  print(f"intervals: {report['intervals']}\n")

  print(f"{'split':>7}{'intervals':>11}{'windows Gb':>13}{'unique Gb':>12}"
        f"{'repeats':>10}")
  print("-" * 53)
  for split in ("train", "valid", "test"):
    sub = frame[frame["split"] == split]
    if not len(sub):
      continue
    total, covered = coverage(sub, "stretched")
    repeats = total / covered if covered else 0.0
    report["splits"][split] = {"intervals": int(len(sub)),
                               "window_bases": int(total),
                               "unique_bases": int(covered),
                               "repeats_per_base": float(repeats)}
    print(f"{split:>7}{len(sub):>11,}{total / 1e9:>13.2f}"
          f"{covered / 1e9:>12.2f}{repeats:>9.1f}x")

  train = frame[frame["split"] == "train"]
  valid = frame[frame["split"] == "valid"]
  native_overlap, native_total = cross_overlap(train, valid, "end")
  stretched_overlap, stretched_total = cross_overlap(train, valid, "stretched")
  report["valid_in_train"] = {
    "native_bases": int(native_overlap), "native_total": int(native_total),
    "native_fraction": float(native_overlap / max(native_total, 1)),
    "stretched_bases": int(stretched_overlap),
    "stretched_total": int(stretched_total),
    "stretched_fraction": float(stretched_overlap / max(stretched_total, 1))}

  print(f"\nvalidation bases also present in train:")
  print(f"  at the native {native_width:,} width : {native_overlap:>13,} "
        f"({100 * native_overlap / max(native_total, 1):.2f}%)  <- the bed's own split is clean")
  print(f"  after the 2^20 stretch      : {stretched_overlap:>13,} "
        f"({100 * stretched_overlap / max(stretched_total, 1):.2f}%)  <- introduced by the stretch")

  train_stats = report["splits"]["train"]
  windows = train_stats["window_bases"] // args.length
  steps = windows / args.global_batch
  print(f"\nat length {args.length:,} and global batch {args.global_batch}:")
  print(f"  {windows:,} windows -> {steps:,.0f} steps for 1.00 epochs over the WINDOW LIST")
  print(f"  which is {train_stats['repeats_per_base']:.1f} passes over the "
        f"{train_stats['unique_bases'] / 1e9:.2f} Gb of unique genome")
  report["at_length"] = {"length": int(args.length),
                         "global_batch": int(args.global_batch),
                         "windows": int(windows),
                         "steps_per_window_epoch": float(steps)}

  if args.json:
    from scripts.eval.provenance import write_json
    write_json(args.json, report, args)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
  main()
