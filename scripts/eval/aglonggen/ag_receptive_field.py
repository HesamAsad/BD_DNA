#!/usr/bin/env python3
"""PRE-FLIGHT 1 for AG-LongGen: how far does AlphaGenome actually see?

WHY THIS GATES THE WHOLE BENCHMARK. AG-LongGen scores generated DNA by
comparing AlphaGenome tracks. That only measures long-range coherence if
AlphaGenome's own predictions depend on long-range context. If the oracle is
effectively local, then every task in the spec reduces to a local-composition
score dressed up as a 1 Mb evaluation, and no amount of care on the generator
side fixes it. The spec's stated control (GC/composition-matched backgrounds)
guards against a DIFFERENT confound -- local base composition -- and cannot
detect this one.

THE MEASUREMENT. Partition a real locus into blocks of size b and permute the
block order. A position that sits further than AlphaGenome's receptive field R
from any block boundary still sees exactly its original local context, so its
prediction must be unchanged. A position within R of a boundary sees foreign
sequence and its prediction moves. So:

    |AG(permuted) - AG(real)| as a function of distance-to-nearest-boundary

decays to zero at exactly d = R. That curve IS the oracle's effective range,
and it is the ceiling on any long-range claim the benchmark can make. We
un-permute the predicted tracks back into original coordinates first, so the
comparison is like-for-like per position.

Anchors, both required to read the curve:
  real vs real (repeat call)  -- the noise floor; AG may not be deterministic
  dinucleotide shuffle        -- destroys everything but composition; the
                                 saturation level, i.e. "maximally different"

INTERPRETATION. If the curve reaches the noise floor within a few kb, the
oracle is local: Tasks 1-3 at 131 kb-1 Mb would be scoring local content only,
and Task 4's effective-range curve is bounded by R, not by the generator. If it
stays elevated out to tens of kb, the benchmark's premise holds and R tells us
the largest span worth generating.

COST. n_loci x (2 + len(block_sizes) + 1) calls. The default is 32 loci at
16,384 bp, so 192 calls -- comfortably inside the free tier.

Usage:
  export ALPHAGENOME_API_KEY=...
  python scripts/eval/aglonggen/ag_receptive_field.py \
      --fasta /path/to/hg38.fa --out results/aglonggen/ag_receptive_field.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# AlphaGenome accepts only these sequence lengths -- NOT arbitrary lengths up to
# 1 Mb, which is what the benchmark note assumes. 16,384 is the smallest, and is
# the only one within reach of models trained at 8,192.
SUPPORTED = (16384, 131072, 524288, 1048576)
# 1D positional outputs. CONTACT_MAPS is 2D and is handled separately, not here.
OUTPUTS = ("RNA_SEQ", "ATAC", "DNASE", "CAGE", "CHIP_HISTONE", "SPLICE_SITES")


def block_permute(seq: str, block: int, rng) -> tuple[str, np.ndarray]:
  """Permute whole blocks. Returns the new sequence and the permutation.

  order[i] = index of the ORIGINAL block now sitting at position i, which is
  what lets us put the predicted tracks back into original coordinates.
  """
  n = len(seq) // block
  order = rng.permutation(n)
  out = "".join(seq[o * block:(o + 1) * block] for o in order)
  return out + seq[n * block:], order


def unpermute(values: np.ndarray, order: np.ndarray, block: int,
              stride: int) -> np.ndarray:
  """Send predicted tracks back to original coordinates.

  `stride` is nucleotides per predicted bin, so a block spans block//stride
  bins. Without this the comparison comes out meaningless for any output that
  is not at single-base resolution.
  """
  per = max(block // stride, 1)
  out = np.empty_like(values)
  for new_i, orig_i in enumerate(order):
    out[orig_i * per:(orig_i + 1) * per] = values[new_i * per:(new_i + 1) * per]
  return out


def dinuc_shuffle(seq: str, rng) -> str:
  """Altschul-Erikson dinucleotide shuffle: preserves 1- and 2-mer counts.

  The composition-matched floor. Anything the oracle still predicts correctly
  here it is reading off composition alone.
  """
  from collections import defaultdict
  edges = defaultdict(list)
  for a, b in zip(seq, seq[1:]):
    edges[a].append(b)
  for k in edges:
    rng.shuffle(edges[k])
  idx = defaultdict(int)
  out = [seq[0]]
  for _ in range(len(seq) - 1):
    c = out[-1]
    if idx[c] >= len(edges[c]):
      break
    out.append(edges[c][idx[c]])
    idx[c] += 1
  out = "".join(out)
  return out + seq[len(out):] if len(out) < len(seq) else out


def stack(output, outputs=OUTPUTS):
  """Output object -> {name: (positions, tracks) array}, per-track z-scored.

  Standardising per track BEFORE any distance is taken is not cosmetic. Raw
  tracks differ in dynamic range by orders of magnitude, so an unnormalised
  average over tracks -- which is what the benchmark note's track-MSE
  specifies -- is dominated by whichever track happens to have the largest
  scale. The spec lists calibration as a separate "credibility control"; it
  belongs inside the metric.
  """
  got = {}
  for name in outputs:
    td = getattr(output, name.lower(), None)
    if td is None:
      continue
    v = np.asarray(td.values, dtype=np.float32)
    if v.ndim == 1:
      v = v[:, None]
    mu = v.mean(axis=0, keepdims=True)
    sd = v.std(axis=0, keepdims=True)
    got[name] = (v - mu) / np.maximum(sd, 1e-6)
  return got


def curve(pred, ref, order, block, length):
  """Mean |Δ| binned by distance to the nearest block boundary."""
  rows = {}
  for name, ref_v in ref.items():
    if name not in pred:
      continue
    p, r = pred[name], ref_v
    if p.shape != r.shape:
      continue
    stride = max(length // p.shape[0], 1)
    p = unpermute(p, order, block, stride)
    delta = np.abs(p - r).mean(axis=1)          # per position, over tracks
    pos = np.arange(p.shape[0]) * stride
    per = block
    d_bound = np.minimum(pos % per, per - (pos % per))   # nt to nearest seam
    rows[name] = (d_bound, delta)
  return rows


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--fasta", required=True)
  ap.add_argument("--chroms", default="chr8,chr9",
                  help="held-out chromosomes, per the AG-LongGen split")
  ap.add_argument("--length", type=int, default=16384, choices=SUPPORTED)
  ap.add_argument("--block-sizes", type=int, nargs="+", default=[512, 2048, 8192])
  ap.add_argument("--n-loci", type=int, default=32)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--ontology", default="UBERON:0002048",
                  help="one ontology term keeps the track count small")
  ap.add_argument("--out", type=Path,
                  default=Path("results/aglonggen/ag_receptive_field.json"))
  args = ap.parse_args()

  key = os.environ.get("ALPHAGENOME_API_KEY")
  if not key:
    sys.exit("set ALPHAGENOME_API_KEY (free non-commercial key from "
             "https://github.com/google-deepmind/alphagenome)")

  import pyfaidx
  from alphagenome.models import dna_client

  genome = pyfaidx.Fasta(args.fasta)
  client = dna_client.create(key)
  rng = np.random.default_rng(args.seed)
  outs = [getattr(dna_client.OutputType, o) for o in OUTPUTS]

  def predict(seq):
    return stack(client.predict_sequence(
      sequence=seq, requested_outputs=outs, ontology_terms=[args.ontology]))

  # fixed, released locus list: contiguous ACGT windows on the held-out chroms
  names = {c: (c if c in genome else c.replace("chr", "")) for c in
           args.chroms.split(",")}
  loci = []
  while len(loci) < args.n_loci:
    c = list(names)[len(loci) % len(names)]
    key_c = names[c]
    span = len(genome[key_c])
    s = int(rng.integers(0, span - args.length))
    seq = str(genome[key_c][s:s + args.length]).upper()
    if seq.count("N") == 0:
      loci.append((c, s, seq))

  agg = {b: {} for b in args.block_sizes}
  floor, ceiling = [], []
  for i, (c, s, seq) in enumerate(loci):
    ref = predict(seq)
    # noise floor: the same sequence twice
    if i < 4:
      rep = predict(seq)
      floor += [float(np.abs(rep[k] - ref[k]).mean()) for k in ref if k in rep]
    # saturation: composition preserved, all order destroyed
    if i < 8:
      dn = predict(dinuc_shuffle(seq, rng))
      ceiling += [float(np.abs(dn[k] - ref[k]).mean()) for k in ref if k in dn]
    for b in args.block_sizes:
      perm, order = block_permute(seq, b, rng)
      rows = curve(predict(perm), ref, order, b, args.length)
      for name, (d, delta) in rows.items():
        agg[b].setdefault(name, []).append((d, delta))
    print(f"  locus {i+1}/{len(loci)}  {c}:{s:,}", flush=True)

  report = {"length": args.length, "n_loci": len(loci),
            "noise_floor": float(np.mean(floor)) if floor else None,
            "dinuc_ceiling": float(np.mean(ceiling)) if ceiling else None,
            "curves": {}}
  print(f"\nnoise floor (same seq twice) : {report['noise_floor']}")
  print(f"dinuc ceiling (all order gone): {report['dinuc_ceiling']}")
  for b in args.block_sizes:
    binned = {}
    for name, pairs in agg[b].items():
      d = np.concatenate([p[0] for p in pairs])
      v = np.concatenate([p[1] for p in pairs])
      edges = np.array([0, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
      edges = edges[edges <= b // 2]
      idx = np.digitize(d, edges) - 1
      binned[name] = {int(edges[k]): float(v[idx == k].mean())
                      for k in range(len(edges)) if (idx == k).any()}
    report["curves"][str(b)] = binned
    print(f"\nblock {b}: mean |delta| by distance from a permutation seam")
    for name, row in binned.items():
      cells = "  ".join(f"{k}nt:{val:.4f}" for k, val in sorted(row.items()))
      print(f"  {name:<14}{cells}")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2))
  print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()
