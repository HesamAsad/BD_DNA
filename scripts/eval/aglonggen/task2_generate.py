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
from scripts.eval.provenance import stamp  # noqa: E402

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


# The four classes the AG-LongGen note names. Restricting to them keeps the
# masked interior on the elements the benchmark is about, rather than on the
# CA/TF classes that dominate the registry by count.
CCRE_CLASSES = ("PLS", "pELS", "dELS", "CA-CTCF")


def load_ccres(path: Path, classes=CCRE_CLASSES):
  """chrom -> sorted list of (start, end, class, id). Handles .gz."""
  import gzip
  from collections import defaultdict
  keep, out = set(classes), defaultdict(list)
  opener = gzip.open if str(path).endswith(".gz") else open
  with opener(path, "rt") as handle:
    for line in handle:
      f = line.rstrip("\n").split("\t")
      if len(f) < 10 or f[9] not in keep:
        continue
      out[f[0]].append((int(f[1]), int(f[2]), f[9], f[3]))
  for c in out:
    out[c].sort()
  return out


def ccres_in(ccres, chrom, start, end):
  """Elements overlapping [start, end) on chrom."""
  import bisect
  rows = ccres.get(chrom) or []
  i = bisect.bisect_left(rows, (start - 5000, 0, "", ""))
  hit = []
  for s, e, cls, eid in rows[i:]:
    if s >= end:
      break
    if e > start:
      hit.append({"start": s, "end": e, "class": cls, "id": eid})
  return hit


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
  ap.add_argument("--allow-contaminated-split", action="store_true",
                  help="opt out of the train/valid chromosome-overlap refusal")
  ap.add_argument("--split", default="valid",
                  help="bed split to draw loci from; 'valid' is the only "
                       "uncontaminated option for current checkpoints")
  ap.add_argument("--chroms", default=None,
                  help="restrict to these chromosomes, e.g. 'chr8,chr9' for "
                       "the spec-compliant list (needs retrained checkpoints)")
  ap.add_argument("--gap-blocks", type=int, nargs="+", default=[1, 2, 4, 8],
                  help="interior span in blocks of block_size")
  ap.add_argument("--n-loci", type=int, default=24)
  # n_loci was ALSO the model batch size (it is passed straight to
  # load_checkpoint_model as eval_batch_size), so --n-loci 1000 tried to push
  # 1000 x 16,384 tokens through the backbone at once and died with
  # "CUDA error: an illegal memory access was encountered". Loci count and
  # batch size are now separate; generation chunks over loci.
  ap.add_argument("--batch-size", type=int, default=0,
                  help="model batch; 0 = min(n_loci, 32)")
  ap.add_argument("--ccre-bed", default=str(REPO / "data/encode/GRCh38-cCREs.ENCFF420VPZ.bed.gz"),
                  help="ENCODE SCREEN registry; the benchmark requires the "
                       "masked interior to contain real regulatory elements")
  ap.add_argument("--require-ccre", type=int, default=1,
                  help="minimum cCREs the masked interior must contain. 0 "
                       "reproduces the earlier arbitrary-locus sampling, which "
                       "masked regions with no functional content at all.")
  ap.add_argument("--num-steps", type=int, default=64)
  ap.add_argument("--refine-passes", type=int, default=0,
                  help="0 (DEFAULT, and the only setting that should be "
                       "used) = single left-to-right pass. >0 sweeps each "
                       "block against its self-generated neighbours, which "
                       "was MEASURED WORSE on every condition -- gap-2048 "
                       "denovo went from 12%% to 83%% of loci failing. See "
                       "Diffusion.sample_infill_refined.")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--reverse-off", action="store_true",
                  help="rebuild a bissm checkpoint as a UnidirectionalSSM "
                       "(weights are shared, so this is a WEIGHT-MATCHED causal "
                       "arm, not a separately-trained one). Runs denovo only.")
  ap.add_argument("--out", type=Path,
                  default=REPO / "results/aglonggen/task2_gen.json")
  args = ap.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("generation requires a CUDA GPU")
  import pyfaidx

  device = torch.device("cuda")
  batch_size = args.batch_size if args.batch_size > 0 else min(args.n_loci, 32)
  model, tokenizer, config, step = load_checkpoint_model(
    args.checkpoint, LENGTH, batch_size, device, reverse_off=args.reverse_off)
  print(f"loci={args.n_loci} model batch={batch_size}", flush=True)
  # bissm runs the full ca/denovo/mismatch contrast. A CAUSAL arm can only run
  # `denovo`, and that is exactly the spec's decisive contrast -- the same loci
  # under a forward-only and a bidirectional model. --reverse-off rebuilds a
  # bissm checkpoint AS a UnidirectionalSSM, so the causal arm is weight-matched
  # rather than a separately-trained run.
  backbone = str(config.algo.backbone)
  if backbone not in ("bissm", "ussm"):
    raise ValueError(f"Task 2 infilling needs backbone bissm or ussm, got {backbone}")
  causal = backbone == "ussm"
  conditions = ("denovo",) if causal else CONDITIONS
  if causal:
    print("CAUSAL arm: only the denovo condition is defined (no right flank)",
          flush=True)
  block = int(config.block_size)
  if LENGTH % block:
    raise ValueError(f"{LENGTH} must divide by block_size={block}")

  genome = pyfaidx.Fasta(args.fasta)
  chroms = set(args.chroms.split(",")) if args.chroms else None
  # CONTAMINATION GUARD. --bed defaults to data/hg38/human-sequences.bed, whose
  # "valid" split shares 15 chromosomes with its own train split (1,405 valid
  # intervals sit on chromosomes the model trained on). Drawing Task 2 loci from
  # there measures memorisation, not infilling. The clean bed
  # (data/hg38/clean/human-sequences-no89.bed) holds chr8+chr9 out wholesale.
  # Refuse rather than warn: a warning in a long log is exactly how the previous
  # contamination survived several rounds of review.
  _tr, _va = set(), set()
  for _ln in open(args.bed):
    _f = _ln.split()
    if len(_f) >= 4:
      (_tr if _f[3] == "train" else _va if _f[3] == "valid" else set()).add(_f[0])
  _shared = sorted(_tr & _va)
  if args.split == "valid" and _shared and not args.allow_contaminated_split:
    sys.exit(
      f"REFUSING: {args.bed} puts {len(_shared)} chromosome(s) in BOTH train and "
      f"valid ({','.join(_shared[:6])}{'...' if len(_shared) > 6 else ''}).\n"
      f"  Task 2 loci drawn from there overlap training data.\n"
      f"  Use --bed data/hg38/clean/human-sequences-no89.bed (valid = chr8+chr9 "
      f"only), or pass --allow-contaminated-split if you truly intend this.")

  intervals = load_intervals(Path(args.bed), chroms, args.split)
  if not intervals:
    sys.exit(f"no {args.split} intervals for chroms={chroms}")
  rng = np.random.default_rng(args.seed)

  ccres = {}
  if args.require_ccre > 0:
    ccres = load_ccres(Path(args.ccre_bed))
    print(f"loaded cCREs on {len(ccres)} chromosomes "
          f"({sum(len(v) for v in ccres.values()):,} elements, "
          f"classes {'/'.join(CCRE_CLASSES)})", flush=True)

  # Fixed, released locus list: contiguous ACGT windows whose CENTRE -- the span
  # that will be masked at the widest gap -- carries real regulatory elements.
  # Sampling for clean ACGT alone masks arbitrary sequence, which is not the
  # benchmark's task and cannot test regulatory reconstruction.
  widest = max(args.gap_blocks) * 256
  loci = []
  tried = 0
  while len(loci) < args.n_loci and tried < 400 * args.n_loci:
    tried += 1
    c, s, e = intervals[int(rng.integers(len(intervals)))]
    if e - s < LENGTH or c not in genome:
      continue
    start = int(rng.integers(s, max(s + 1, e - LENGTH)))
    seq = str(genome[c][start:start + LENGTH]).upper()
    if len(seq) != LENGTH or seq.count("N"):
      continue
    mid = start + LENGTH // 2
    hits = ccres_in(ccres, c, mid - widest // 2, mid + widest // 2) if ccres else []
    if args.require_ccre > 0 and len(hits) < args.require_ccre:
      continue
    loci.append({"chrom": c, "start": start, "seq": seq, "ccres": hits})
  if len(loci) < args.n_loci:
    print(f"warning: only {len(loci)} loci met the cCRE requirement "
          f"in {tried} tries")
  if ccres:
    n = [len(l["ccres"]) for l in loci]
    print(f"cCREs in the widest masked span: min {min(n)} median "
          f"{sorted(n)[len(n)//2]} max {max(n)}", flush=True)
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

    for cond in conditions:
      if cond == "ca":
        r_all = right
      elif cond == "mismatch":
        r_all = right_mm
      else:
        r_all = right[:, :0]                 # empty suffix -> left-only
      # CHUNKED over loci. right_mm is rolled over the FULL set before slicing,
      # so a row still gets another locus's suffix regardless of chunk boundaries
      # (rolling inside a chunk would hand a 1-row chunk its own suffix back).
      # The seed is reset per chunk with a chunk-dependent but CONDITION-INDEPENDENT
      # value, so ca/denovo/mismatch still see identical noise for the same loci --
      # that pairing is what makes the contrast valid.
      parts = []
      for lo in range(0, len(loci), batch_size):
        hi = min(lo + batch_size, len(loci))
        torch.manual_seed(args.seed * 100003 + lo)
        r_chunk = r_all[lo:hi] if r_all.shape[-1] else r_all[:hi - lo, :0]
        with torch.inference_mode():
          if args.refine_passes > 0:
            out = model.sample_infill_refined(
              left[lo:hi], r_chunk, gap_nt, args.num_steps,
              passes=args.refine_passes)
          else:
            out = model.sample_infill_ca(
              left[lo:hi], r_chunk, gap_nt, args.num_steps)
        parts.append(out[:, :left_nt + gap_nt])
        if len(loci) > 200 and (hi % (batch_size * 5) == 0 or hi == len(loci)):
          print(f"    {cond}: {hi}/{len(loci)}", flush=True)
      full = torch.cat(parts, dim=0)
      # the returned tail is whatever suffix was fed; splice the REAL right
      # flank back so every condition is scored on the same 16,384 window and
      # only the generated interior differs
      recon = torch.cat((full, right), dim=1)
      for i, l in enumerate(loci):
        gs, ge = l["start"] + left_nt, l["start"] + left_nt + gap_nt
        records.append({
          "chrom": l["chrom"], "start": l["start"], "gap_nt": gap_nt,
          "left_nt": left_nt, "condition": cond,
          "interior": decode(recon[i, left_nt:left_nt + gap_nt]),
          "sequence": decode(recon[i]),
          "ccres_in_gap": ccres_in(ccres, l["chrom"], gs, ge) if ccres else [],
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
  # PROVENANCE. right_flank_probability is the field whose absence let an
  # rf=0.0 checkpoint be interpreted as "the suffix carries no information"
  # through an entire Task 2 campaign. Record it, and the whole model/algo
  # block, so a result can never again be read without knowing how it trained.
  # PROVENANCE MUST COME FROM THE CHECKPOINT, NOT THE RUNTIME CONFIG.
  # load_checkpoint_model (score_mavedb.py:60) deliberately sets
  # `config.model.right_flank_probability = 0.0` for inference -- correct, since
  # sample_infill_ca supplies the right cache explicitly rather than letting a
  # Bernoulli gate drop it -- and it sets config.model.length to the GENERATION
  # length. Reading either back as provenance reports the inference setting as
  # though it were the training setting. Measured 2026-09-08: a checkpoint
  # trained at rf=0.5, length 8192 was recorded as rf=0.0, trained_length=16384.
  # rf=0.0 is precisely the value that signalled the original untrained-flank
  # catastrophe, so this field was not merely wrong, it was maximally misleading.
  _hp = torch.load(args.checkpoint, map_location="cpu",
                   weights_only=False).get("hyper_parameters", {})
  _tc = _hp.get("config", {})
  _tm, _ta = _tc.get("model", {}), _tc.get("algo", {})
  def _g(d, k, default=None):
    return d.get(k, default) if hasattr(d, "get") else default
  prov = {
    # what the model was TRAINED with (from the checkpoint's own hyper_parameters)
    "right_flank_probability": _g(_tm, "right_flank_probability"),
    "time_conditioning": _g(_ta, "time_conditioning"),
    "var_min": _g(_ta, "var_min"),
    "trained_length": _g(_tm, "length"),
    "trained_block_size": _g(_tc, "block_size"),
    "train_data": _g(_tc.get("data", {}) if hasattr(_tc, "get") else {}, "train"),
    # what THIS generation run used
    "backbone": str(config.algo.backbone),
    "generation_length": LENGTH,
    "inference_right_flank_probability": float(
      config.model.get("right_flank_probability", 0.0)),
  }
  if not prov["right_flank_probability"]:
    print("WARNING: checkpoint trained with right_flank_probability="
          f"{prov['right_flank_probability']!r} -- the suffix pathway was never "
          "trained, so any ca-vs-mismatch contrast measures an untrained input "
          "slot. See the 2026-09-07 root cause.", flush=True)
  print("provenance:", json.dumps(prov), flush=True)
  report = {
    "checkpoint": str(args.checkpoint), "global_step": step,
    "provenance": prov,
    "length": LENGTH, "block_size": block, "num_steps": args.num_steps,
    "refine_passes": args.refine_passes, "split": args.split,
    "chroms": sorted(chroms) if chroms else "all",
    "n_loci": len(loci), "n_loci_requested": args.n_loci, "records": records}
  # argv/git/time, so the sample size behind a result is recoverable from it
  stamp(report, args)
  args.out.write_text(json.dumps(report, indent=2))
  bad = [r for r in records if len(r["sequence"]) != LENGTH]
  print(f"\nwrote {args.out}  ({len(records)} sequences, "
        f"{len(bad)} wrong length)")
  if bad:
    sys.exit(f"{len(bad)} sequences are not {LENGTH} nt")


if __name__ == "__main__":
  main()
