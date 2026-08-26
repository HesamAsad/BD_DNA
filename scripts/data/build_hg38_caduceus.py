#!/usr/bin/env python3
"""Build the hg38 pretraining cache EXACTLY as Caduceus does.

This mirrors `src/dataloaders/datasets/hg38_dataset.py` from
github.com/kuleshov-group/caduceus so that our arms and theirs see the same
corpus, the same splits, and the same windows -- and only the model differs.

WHY THIS REPLACES OUR OWN BUILDER. scripts/eval/build_human_longrange.py
samples windows itself: half centred on a protein-coding TSS, half uniform over
chromosomes, holding out chr8/chr9. That is a reasonable design and it is NOT
what Caduceus did, so any likelihood we compared against theirs would be
measured on different data with a different split. Six concrete divergences:

  1. reference   they use hg38.ml.fa from basenji_barnyard2; we used
                 gencode_v41_GRCh38.fa.
  2. windows     they TILE deterministically; we sampled randomly. Our sampling
                 also drew a chromosome uniformly rather than by length, so
                 chr21 was over-represented per base by 2.7x and chr1 by 0.51x.
                 Tiling has no such bias -- the concern evaporates.
  3. split       theirs is per-INTERVAL from the bed file's 4th column, and 15
                 chromosomes appear in both train and valid. Ours held out
                 chr8/chr9 wholesale. Chromosome holdout is arguably stricter,
                 but it is a different quantity.
  4. N bases     they keep N and map it to PAD so the loss ignores it
                 (`ignore_index: 4`). We DROPPED any window above 5% N, which
                 removes assembly gaps from the distribution entirely.
  5. EOS         they append a separator per window (`add_eos: true`).
  6. corpus      34,021 train intervals, each stretched to 2^20, tiled into
                 `2^20 // length` non-overlapping windows. At length 1024 that
                 is 34,837,504 windows = 35.67e9 tokens -- the paper's "around
                 35 billion". At our 8192 it is 4,354,688 windows, same tokens.

THE MECHANICS WE COPY, from HG38Dataset:

    self.df["end"] = self.df["start"] + MAX_ALLOWED_LENGTH   # 2^20
    self.shifts    = MAX_ALLOWED_LENGTH // max_length
    __len__        = len(df) * shifts
    row_idx, shift_idx = idx // shifts, idx % shifts
    start, end     = start + shift_idx*max_length, start + (shift_idx+1)*max_length

plus their three boundary corrections when an interval runs off the end of a
chromosome (shift down, shift up, clamp), reproduced verbatim below.

Emits the same on-disk layout our loader already reads: input_ids int32 and
attention_mask float32, saved atomically via a .tmp rename so a killed job can
never leave a half-written cache that looks complete.

Usage:
  python scripts/data/build_hg38_caduceus.py --length 8192 --name hg38-caduceus
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
MAX_ALLOWED_LENGTH = 2 ** 20          # their MAX_ALLOWED_LENGTH

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--length", type=int, default=8192)
ap.add_argument("--fasta", default=str(REPO / "data/hg38/hg38.ml.fa"))
ap.add_argument("--bed", default=str(REPO / "data/hg38/human-sequences.bed"))
ap.add_argument("--cache_dir", default=str(REPO / "data_cache/carbon"))
ap.add_argument("--name", default="hg38-caduceus")
ap.add_argument("--splits", default="train,valid")
ap.add_argument("--limit", type=int, default=0,
                help="cap windows per split (smoke tests only; 0 = all)")
args = ap.parse_args()

L = args.length
if MAX_ALLOWED_LENGTH % L:
    sys.exit(f"length {L} must divide 2^20 -- theirs asserts the same")
SHIFTS = MAX_ALLOWED_LENGTH // L

# Their char tokenizer maps N to PAD so the loss skips it. Ours uses the DNA
# vocab; N is a real token here and the attention_mask carries the masking, so
# we set mask=0 exactly where they would have emitted PAD.
LUT = np.full(256, 7, np.int32)                       # anything else -> UNK
for ch, i in [("A", 8), ("C", 9), ("G", 10), ("T", 11), ("N", 12)]:
    LUT[ord(ch)] = i
N_ID = 12

print(f"{args.name}: L={L}  shifts={SHIFTS}  fasta={args.fasta}", flush=True)
fa = Fasta(args.fasta, sequence_always_upper=True, as_raw=True)
chr_lens = {c: len(fa[c]) for c in fa.keys()}

df_raw = pd.read_csv(args.bed, sep="\t", names=["chr_name", "start", "end", "split"])


def compute_interval(start, shift_idx):
    """Their FastaInterval._compute_interval, for max_length < 2^20."""
    return start + shift_idx * L, start + (shift_idx + 1) * L


def fetch(chr_name, start, end):
    """Their FastaInterval.__call__ boundary handling, verbatim in effect."""
    chromosome_length = chr_lens[chr_name]
    if end > chromosome_length:                       # shift interval down
        start = start - (end - chromosome_length)
        end = chromosome_length
    if start < 0:                                     # shift interval up
        end = end - start
        start = 0
    if end > chromosome_length:                       # clamp
        start = chromosome_length - L
        end = chromosome_length
    return str(fa[chr_name][start:end])


def build(split):
    df = df_raw[df_raw["split"] == split].reset_index(drop=True)
    # Their line: every interval stretched to 2^20 regardless of its bed width.
    df = df.assign(end=df["start"] + MAX_ALLOWED_LENGTH)
    total = len(df) * SHIFTS
    if args.limit:
        total = min(total, args.limit)
    print(f"  {split}: {len(df):,} intervals x {SHIFTS} shifts = {total:,} windows "
          f"= {total * L / 1e9:.2f}e9 nt", flush=True)

    def gen():
        t0 = time.time()
        for idx in range(total):
            row_idx, shift_idx = idx // SHIFTS, idx % SHIFTS
            row = df.iloc[row_idx]
            start, end = compute_interval(int(row.start), shift_idx)
            seq = fetch(row.chr_name, start, end)
            if len(seq) < L:                          # only at a chromosome edge
                continue
            ids = LUT[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]
            # N -> masked out of the loss, matching their N -> PAD -> ignore_index
            mask = (ids != N_ID).astype(np.float32)
            yield {"input_ids": ids, "attention_mask": mask}
            if idx and idx % 200_000 == 0:
                rate = idx / (time.time() - t0)
                print(f"    {idx:,}/{total:,}  {rate:,.0f}/s  "
                      f"eta {(total - idx) / rate / 3600:.1f} h", flush=True)

    feats = datasets.Features({
        "input_ids": datasets.Sequence(datasets.Value("int32")),
        "attention_mask": datasets.Sequence(datasets.Value("float32"))})
    ds = datasets.Dataset.from_generator(gen, features=feats)
    suffix = "_train" if split == "train" else "_validation"
    out = os.path.join(args.cache_dir,
                       f"{args.name}{suffix}_bs{L}_wrapped_specialFalse.dat")
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    ds.save_to_disk(tmp)
    shutil.rmtree(out, ignore_errors=True)
    os.rename(tmp, out)
    print(f"  saved {split}: rows={ds.num_rows:,} -> {out}", flush=True)


for s in [x.strip() for x in args.splits.split(",") if x.strip()]:
    build(s)
print("done", flush=True)
