#!/usr/bin/env python3
"""AG-LongGen Task 2, stage 1: generate infilled interiors at 16,384 nt.

THE TASK. Mask an interior span of a real locus, commit both flanks, and have
the model write the interior back. This is the one AG-LongGen task that matches
what our models can actually do -- Task 1 is inverse design from a target track
profile, and nothing in this architecture takes a track profile as input.

WHY 16,384. AlphaGenome accepts exactly {16384, 131072, 524288, 1048576}, not
arbitrary lengths, and 16,384 is the only one within reach of models trained at
8,192. The benchmark note's L in {131k, 524k, 1M} skips it and would have
forced 16x-122x extrapolation for no reason.

THE CONTRAST, and why it is clean. `sample_infill_ca` requires backbone=bissm,
so only the bidirectional arm can consume a right flank at all. That is a
feature here: the comparison is the SAME WEIGHTS with and without the committed
suffix, not one architecture against another, so nothing but the right cache
differs.

    ca        true right flank        the capability under test
    denovo    no right flank          same model, left context only
    mismatch  another locus's flank   a right cache that is populated but wrong

`mismatch` is the control that matters. If `ca` beats `denovo` merely because a
populated cache is better than an empty one, then `mismatch` beats `denovo`
too, and the gain is not information transfer.

CONTAMINATION, stated up front. 3,705 of 3,727 chr8/chr9 intervals (99.4%) are
in our training split, so the spec's mandated held-out chromosomes are NOT held
out for these checkpoints. Scoring there would measure memorisation. Until the
arms are retrained under the chr8/chr9 holdout this samples from the corpus's
own `valid` intervals, which the corpus audit found contiguity-clean with 3.03%
train overlap. `--chroms chr8,chr9` switches to the spec-compliant list and is
the right setting the moment retrained checkpoints exist.

Emits one JSON holding every reconstructed 16,384 nt sequence plus the real and
composition-matched anchors, which `task2_score.py` then sends to AlphaGenome.

Usage:
  python scripts/eval/aglonggen/task2_generate.py \
      --checkpoint outputs/hg38-caduceus/hg_bissm_bd/checkpoints/best.ckpt \
      --fasta data/hg38/hg38.ml.fa --out results/aglonggen/task2_gen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402

LENGTH = 16384
CONDITIONS = ("ca", "denovo", "mismatch")


def dinuc_shuffle(seq: str, rng) -> str:
  """Preserves 1- and 2-mer counts; the composition-matched null."""
  edges = defaultdict(list)
  for a, b in zip(seq, seq[1:]):
    edges[a].append(b)
  for k in edges:
    rng.shuffle(edges[k])
  idx, out = defaultdict(int), [seq[0]]
  for _ in range(len(seq) - 1):
    c = out[-1]
    if idx[c] >= len(edges[c]):
      break
    out.append(edges[c][idx[c]])
    idx[c] += 1
  out = "".join(out)
  return out + seq[len(out):] if len(out) < len(seq) else out


def load_intervals(bed: Path, chroms, split):
  rows = []
  for line in bed.read_text().splitlines():
    f = line.split()
    if len(f) < 4:
      continue
    if split and f[3] != split:
      continue
    if chroms and f[0] not in chroms:
      continue
    rows.append((f[0], int(f[1]), int(f[2])))
  return rows


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--fasta", default=str(REPO / "data/hg38/hg38.ml.fa"))
  ap.add_argument("--bed", default=str(REPO / "data/hg38/human-sequences.bed"))
  ap.add_argument("--split", default="valid",
                  help="bed split to draw loci from; 'valid' is the only "
                       "uncontaminated option for current checkpoints")
  ap.add_argument("--chroms", default=None,
                  help="restrict to these chromosomes, e.g. 'chr8,chr9' for "
                       "the spec-compliant list (needs retrained checkpoints)")
  ap.add_argument("--gap-blocks", type=int, nargs="+", default=[1, 2, 4, 8],
                  help="interior span in blocks of block_size")
  ap.add_argument("--n-loci", type=int, default=24)
  ap.add_argument("--num-steps", type=int, default=64)
  ap.add_argument("--refine-passes", type=int, default=0,
                  help="0 (DEFAULT, and the only setting that should be "
                       "used) = single left-to-right pass. >0 sweeps each "
                       "block against its self-generated neighbours, which "
                       "was MEASURED WORSE on every condition -- gap-2048 "
                       "denovo went from 12%% to 83%% of loci failing. See "
                       "Diffusion.sample_infill_refined.")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", type=Path,
                  default=REPO / "results/aglonggen/task2_gen.json")
  args = ap.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("generation requires a CUDA GPU")
  import pyfaidx

  device = torch.device("cuda")
  model, tokenizer, config, step = load_checkpoint_model(
    args.checkpoint, LENGTH, args.n_loci, device)
  if str(config.algo.backbone) != "bissm":
    raise ValueError("Task 2 infilling requires backbone=bissm; only the "
                     "bidirectional arm can consume a right flank")
  block = int(config.block_size)
  if LENGTH % block:
    raise ValueError(f"{LENGTH} must divide by block_size={block}")

  genome = pyfaidx.Fasta(args.fasta)
  chroms = set(args.chroms.split(",")) if args.chroms else None
  intervals = load_intervals(Path(args.bed), chroms, args.split)
  if not intervals:
    sys.exit(f"no {args.split} intervals for chroms={chroms}")
  rng = np.random.default_rng(args.seed)

  # fixed, released locus list: contiguous ACGT windows, no assembly gaps
  loci = []
  tried = 0
  while len(loci) < args.n_loci and tried < 40 * args.n_loci:
    tried += 1
    c, s, e = intervals[int(rng.integers(len(intervals)))]
    if e - s < LENGTH or c not in genome:
      continue
    start = int(rng.integers(s, max(s + 1, e - LENGTH)))
    seq = str(genome[c][start:start + LENGTH]).upper()
    if len(seq) == LENGTH and seq.count("N") == 0:
      loci.append({"chrom": c, "start": start, "seq": seq})
  if len(loci) < args.n_loci:
    print(f"warning: only {len(loci)} clean loci found")

  ids = torch.tensor(
    [tokenizer.encode(l["seq"], add_special_tokens=False) for l in loci],
    dtype=torch.long, device=device)
  if ids.shape[1] != LENGTH:
    raise ValueError(f"tokenizer produced {ids.shape[1]} ids, expected {LENGTH}")

  def decode(t):
    return "".join(
      x for x in tokenizer.decode(t.tolist()).replace(" ", "") if x in "ACGTN")

  records = []
  total_blocks = LENGTH // block
  for gb in args.gap_blocks:
    if gb >= total_blocks:
      continue
    left_b = (total_blocks - gb) // 2
    gap_nt, left_nt = gb * block, left_b * block
    right_nt = LENGTH - gap_nt - left_nt
    left, right = ids[:, :left_nt], ids[:, left_nt + gap_nt:]
    # mismatch: roll the batch so every row gets a DIFFERENT locus's suffix
    right_mm = torch.roll(right, shifts=1, dims=0)
    print(f"gap {gap_nt} nt  left {left_nt}  right {right_nt}", flush=True)

    for cond in CONDITIONS:
      if cond == "ca":
        r = right
      elif cond == "mismatch":
        r = right_mm
      else:
        r = right[:, :0]                     # empty suffix -> left-only
      torch.manual_seed(args.seed)
      with torch.inference_mode():
        if args.refine_passes > 0:
          full = model.sample_infill_refined(
            left, r, gap_nt, args.num_steps, passes=args.refine_passes)
        else:
          full = model.sample_infill_ca(left, r, gap_nt, args.num_steps)
      # the returned tail is whatever suffix was fed; splice the REAL right
      # flank back so every condition is scored on the same 16,384 window and
      # only the generated interior differs
      recon = torch.cat((full[:, :left_nt + gap_nt], right), dim=1)
      for i, l in enumerate(loci):
        records.append({
          "chrom": l["chrom"], "start": l["start"], "gap_nt": gap_nt,
          "left_nt": left_nt, "condition": cond,
          "interior": decode(recon[i, left_nt:left_nt + gap_nt]),
          "sequence": decode(recon[i]),
        })
      print(f"  {cond}: {len(loci)} sequences", flush=True)

    # anchors, once per gap: the real interior (upper bound) and a
    # composition-matched shuffle of it (lower bound)
    for i, l in enumerate(loci):
      real_int = l["seq"][left_nt:left_nt + gap_nt]
      for name, interior in (("real", real_int),
                             ("dinuc", dinuc_shuffle(real_int, rng))):
        records.append({
          "chrom": l["chrom"], "start": l["start"], "gap_nt": gap_nt,
          "left_nt": left_nt, "condition": name, "interior": interior,
          "sequence": l["seq"][:left_nt] + interior
                      + l["seq"][left_nt + gap_nt:],
        })

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({
    "checkpoint": str(args.checkpoint), "global_step": step,
    "length": LENGTH, "block_size": block, "num_steps": args.num_steps,
    "refine_passes": args.refine_passes, "split": args.split, "chroms": sorted(chroms) if chroms else "all",
    "n_loci": len(loci), "records": records}, indent=2))
  bad = [r for r in records if len(r["sequence"]) != LENGTH]
  print(f"\nwrote {args.out}  ({len(records)} sequences, "
        f"{len(bad)} wrong length)")
  if bad:
    sys.exit(f"{len(bad)} sequences are not {LENGTH} nt")


if __name__ == "__main__":
  main()
