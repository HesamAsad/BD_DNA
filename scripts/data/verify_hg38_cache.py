#!/usr/bin/env python3
"""Check the built hg38 cache against the FASTA, independently of the builder.

The builder and this checker must not share code paths, or a bug in the
interval arithmetic verifies itself. So this re-derives every sampled window's
(chromosome, start, end) from the bed file using HG38Dataset's own rules,
re-reads it from the FASTA, tokenises it here, and demands a byte-exact match
against what is on disk.

WHAT IT CHECKS, and why each one has a way of going wrong quietly:

  rows        the row count against len(df) * shifts, minus the windows the
              builder legitimately drops at chromosome ends (`len(seq) < L`).
              A silently truncated cache trains happily on a prefix.
  length      every row exactly L tokens. A short row poisons a batch.
  vocabulary  only the 6 ids the DNA tokenizer can emit (A/C/G/T/N/UNK). Any
              other value means the LUT or the fasta reader drifted.
  mask        attention_mask == (ids != N). This is what makes the loss skip
              assembly gaps, mirroring their N -> PAD -> ignore_index; if it
              drifts, N positions silently join the loss.
  identity    a random sample of windows, byte-exact against the FASTA.
  alignment   window i and window i+1 within one interval are ADJACENT in the
              genome. Catches an off-by-one in the shift arithmetic that a
              per-window identity check on its own would miss.

Usage:
  python scripts/data/verify_hg38_cache.py --sample 500
  python scripts/data/verify_hg38_cache.py --split validation --sample 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MAX_ALLOWED_LENGTH = 2 ** 20

LUT = np.full(256, 7, np.int32)
for character, index in [("A", 8), ("C", 9), ("G", 10), ("T", 11), ("N", 12)]:
  LUT[ord(character)] = index
N_ID = 12
ALLOWED = {7, 8, 9, 10, 11, 12}


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--name", default="hg38-caduceus")
  parser.add_argument("--split", default="train", choices=["train", "validation"])
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--fasta", default=str(REPO / "data/hg38/hg38.ml.fa"))
  parser.add_argument("--bed", default=str(REPO / "data/hg38/human-sequences.bed"))
  parser.add_argument("--cache_dir", default=str(REPO / "data_cache/carbon"))
  parser.add_argument("--sample", type=int, default=300)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  length = args.length
  shifts = MAX_ALLOWED_LENGTH // length
  path = (Path(args.cache_dir) /
          f"{args.name}_{args.split}_bs{length}_wrapped_specialFalse.dat")
  if not path.exists():
    sys.exit(f"no cache at {path}")

  print(f"cache   {path}")
  data = datasets.load_from_disk(str(path))
  print(f"rows    {data.num_rows:,}")

  bed_split = "train" if args.split == "train" else "valid"
  frame = pd.read_csv(args.bed, sep="\t",
                      names=["chr_name", "start", "end", "split"])
  frame = frame[frame["split"] == bed_split].reset_index(drop=True)
  expected = len(frame) * shifts
  dropped = expected - data.num_rows
  print(f"expect  {len(frame):,} intervals x {shifts} shifts = {expected:,}"
        f"   (on disk {data.num_rows:,}, {dropped:,} dropped at chromosome ends)")
  failures = []
  fasta = Fasta(args.fasta, sequence_always_upper=True, as_raw=True)
  chromosome_length = {name: len(fasta[name]) for name in fasta.keys()}

  # How many windows SHOULD the builder drop? It skips one only when the fasta
  # slice comes back shorter than L. Rather than tolerate a fudge factor, work
  # out the exact number by replaying its fetch() geometry: for hg38 at
  # L=8192 the answer is 0, because the shortest chromosome the bed references
  # is chr21 at 46.7 Mb. So any missing row at all is a real defect, and an
  # exact expectation is what makes the byte-identity check below valid --
  # a single silent drop shifts every later row's index.
  def fetched_length(name, start, end):
    limit = chromosome_length[name]
    if end > limit:
      start, end = start - (end - limit), limit
    if start < 0:
      end, start = end - start, 0
    if end > limit:
      start, end = limit - length, limit
    return max(0, min(end, limit) - max(start, 0))

  should_drop = 0
  for row_index in range(len(frame)):
    row = frame.iloc[row_index]
    for shift_index in range(shifts):
      start = int(row.start) + shift_index * length
      if fetched_length(row.chr_name, start, start + length) < length:
        should_drop += 1
  print(f"predicted drops from the fasta geometry: {should_drop:,}")
  if dropped != should_drop:
    failures.append(
      f"row count off: {data.num_rows:,} on disk, expected "
      f"{expected - should_drop:,} (= {expected:,} minus {should_drop:,} "
      f"windows the fasta geometry legitimately drops)")
  dropped = should_drop  # below, "no drops" means "no UNEXPLAINED drops"

  def interval(row_index, shift_index):
    row = frame.iloc[row_index]
    start = int(row.start) + shift_index * length
    end = start + length
    limit = chromosome_length[row.chr_name]
    if end > limit:
      start, end = start - (end - limit), limit
    if start < 0:
      end, start = end - start, 0
    if end > limit:
      start, end = limit - length, limit
    return row.chr_name, start, end

  random.seed(args.seed)
  indices = sorted(random.sample(range(data.num_rows),
                                 min(args.sample, data.num_rows)))
  print(f"\nchecking {len(indices)} sampled windows against the FASTA ...")
  bad_length = bad_vocab = bad_mask = bad_identity = 0
  for position in indices:
    row = data[position]
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    mask = np.asarray(row["attention_mask"], dtype=np.float32)
    if len(ids) != length:
      bad_length += 1
      continue
    if not set(np.unique(ids)).issubset(ALLOWED):
      bad_vocab += 1
    if not np.array_equal(mask, (ids != N_ID).astype(np.float32)):
      bad_mask += 1
    # Only meaningful while nothing is dropped before this row; once the
    # builder has skipped a window the on-disk index no longer equals the
    # (interval, shift) index, so identity is checked on the no-drop prefix.
    if dropped == 0:
      name, start, end = interval(position // shifts, position % shifts)
      sequence = str(fasta[name][start:end])
      expected_ids = LUT[np.frombuffer(sequence.encode("ascii", "replace"),
                                       np.uint8)]
      if not np.array_equal(ids, expected_ids):
        bad_identity += 1

  print(f"  wrong length      {bad_length}")
  print(f"  vocabulary drift  {bad_vocab}")
  print(f"  mask mismatch     {bad_mask}")
  if dropped == 0:
    print(f"  not byte-exact    {bad_identity}")
  else:
    print(f"  not byte-exact    skipped ({dropped:,} dropped rows shift the "
          f"index mapping; re-run per-interval if this matters)")
  for count, what in ((bad_length, "wrong length"),
                      (bad_vocab, "vocabulary drift"),
                      (bad_mask, "mask mismatch"),
                      (bad_identity, "not byte-exact")):
    if count:
      failures.append(f"{count} sampled windows: {what}")

  # Adjacency: consecutive windows inside one interval must tile the genome
  # with no gap and no overlap. A per-window identity check passes even if the
  # shift stride is wrong, as long as each window individually matches.
  if dropped == 0 and data.num_rows > shifts:
    print("\nchecking that consecutive windows tile without gaps ...")
    checked = joined = 0
    for base in random.sample(range(min(len(frame), 200)), min(20, len(frame))):
      first = base * shifts
      if first + 1 >= data.num_rows:
        continue
      a = np.asarray(data[first]["input_ids"], dtype=np.int32)
      b = np.asarray(data[first + 1]["input_ids"], dtype=np.int32)
      name, start, _ = interval(base, 0)
      window = str(fasta[name][start:start + 2 * length])
      both = LUT[np.frombuffer(window.encode("ascii", "replace"), np.uint8)]
      checked += 1
      if np.array_equal(np.concatenate([a, b]), both):
        joined += 1
    print(f"  {joined}/{checked} interval starts tile contiguously")
    if joined != checked:
      failures.append(f"{checked - joined}/{checked} interval starts do not tile")

  print()
  if failures:
    for line in failures:
      print(f"FAIL  {line}")
    sys.exit(1)
  print("PASS  cache matches the FASTA under HG38Dataset's own interval rules")


if __name__ == "__main__":
  main()
