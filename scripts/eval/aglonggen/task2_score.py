#!/usr/bin/env python3
"""AG-LongGen Task 2, stage 2: score infilled loci with AlphaGenome.

Reads the sequences `task2_generate.py` produced and asks the frozen oracle how
close each reconstruction is, functionally, to the real locus.

TWO METRIC DECISIONS THAT DIFFER FROM THE BENCHMARK NOTE, both forced by the
receptive-field pre-flight (`ag_receptive_field.py`, 2026-09-02):

1. PER TRACK FAMILY, NOT A FLAT MEAN. The note defines
   track-MSE = 1/T sum_tracks ||AG(x) - y*||^2. Measured, the families do not
   behave alike: block-permuting a locus moves RNA_SEQ by 0.39 and
   CHIP_HISTONE by 0.22 even far from any seam (a 0.60 saturation ceiling),
   while ATAC, DNASE and CAGE decay to ~0.04 within about 1 kb and
   SPLICE_SITES is local and tiny. RNA_SEQ is where AlphaGenome's long-range
   sensitivity actually lives. Averaging it against four mostly-local families
   dilutes exactly the signal the benchmark exists to measure, so every family
   is reported separately and RNA_SEQ is the headline.

2. SHARED NORMALISATION. Tracks differ in dynamic range by orders of
   magnitude, so an unnormalised average is dominated by whichever track has
   the largest scale. But normalising each prediction by its OWN statistics is
   also wrong: it would absorb a genuine global shift into the z-scoring and
   manufacture agreement. Both sides are standardised with the REAL locus's
   per-track mean and standard deviation.

Sequence-level scores (k-mer divergence, GC, exact recovery) need no API and
are computed for every record regardless.

AlphaGenome is deterministic -- the same sequence twice gave a noise floor of
exactly 0.0 -- so no repeat-call averaging is needed and any difference below
is real.

Usage:
  export ALPHAGENOME_API_KEY=...
  python scripts/eval/aglonggen/task2_score.py \
      --gen results/aglonggen/task2_gen.json \
      --out results/aglonggen/task2_scores.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUTPUTS = ("RNA_SEQ", "ATAC", "DNASE", "CAGE", "CHIP_HISTONE", "SPLICE_SITES")
ORDER = ("real", "ca", "mismatch", "denovo", "dinuc")


def kmer_js(a: str, b: str, k: int = 6) -> float:
  """Jensen-Shannon divergence between k-mer distributions, in bits."""
  def counts(s):
    c = collections.Counter(s[i:i + k] for i in range(len(s) - k + 1))
    n = max(sum(c.values()), 1)
    return c, n
  ca, na = counts(a)
  cb, nb = counts(b)
  total = 0.0
  for key in set(ca) | set(cb):
    p, q = ca.get(key, 0) / na, cb.get(key, 0) / nb
    m = 0.5 * (p + q)
    if p:
      total += 0.5 * p * np.log2(p / m)
    if q:
      total += 0.5 * q * np.log2(q / m)
  return float(total)


def gc(s: str) -> float:
  return (s.count("G") + s.count("C")) / max(len(s), 1)


def identity(a: str, b: str) -> float:
  n = min(len(a), len(b))
  return sum(x == y for x, y in zip(a[:n], b[:n])) / max(n, 1)


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--gen", type=Path,
                  default=REPO / "results/aglonggen/task2_gen.json")
  ap.add_argument("--out", type=Path,
                  default=REPO / "results/aglonggen/task2_scores.json")
  ap.add_argument("--ontology", default="UBERON:0002048")
  ap.add_argument("--skip-oracle", action="store_true",
                  help="sequence-level scores only; no API calls")
  args = ap.parse_args()

  payload = json.loads(args.gen.read_text())
  records = payload["records"]
  print(f"{len(records)} records from {args.gen}")

  # the real interior per locus, to score sequence recovery against
  real_interior = {(r["chrom"], r["start"], r["gap_nt"]): r["interior"]
                   for r in records if r["condition"] == "real"}
  # AG(real full locus) is the reference. Splicing the real interior back
  # reproduces the original window, so the same sequence recurs at every gap --
  # dedupe by hash or we would pay for it four times over.
  def h(s):
    return hashlib.sha1(s.encode()).hexdigest()

  cache = {}
  if not args.skip_oracle:
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
      sys.exit("set ALPHAGENOME_API_KEY, or pass --skip-oracle")
    from alphagenome.models import dna_client
    client = dna_client.create(key)
    outs = [getattr(dna_client.OutputType, o) for o in OUTPUTS]

    def raw(seq):
      d = h(seq)
      if d in cache:
        return cache[d]
      o = client.predict_sequence(
        sequence=seq, requested_outputs=outs, ontology_terms=[args.ontology])
      got = {}
      for name in OUTPUTS:
        td = getattr(o, name.lower(), None)
        if td is None:
          continue
        v = np.asarray(td.values, dtype=np.float32)
        got[name] = v[:, None] if v.ndim == 1 else v
      cache[d] = got
      return got

    unique = {h(r["sequence"]) for r in records}
    print(f"{len(unique)} unique sequences to score "
          f"({len(records) - len(unique)} duplicates avoided)")

  rows = []
  for i, r in enumerate(records):
    key = (r["chrom"], r["start"], r["gap_nt"])
    truth = real_interior.get(key, "")
    row = {k: r[k] for k in ("chrom", "start", "gap_nt", "condition")}
    row.update({"kmer_js_bits": kmer_js(r["interior"], truth) if truth else None,
                "gc": gc(r["interior"]),
                "gc_abs_err": abs(gc(r["interior"]) - gc(truth)) if truth else None,
                "identity": identity(r["interior"], truth) if truth else None})
    if not args.skip_oracle:
      ref_rec = next(x for x in records
                     if (x["chrom"], x["start"], x["gap_nt"]) == key
                     and x["condition"] == "real")
      ref, pred = raw(ref_rec["sequence"]), raw(r["sequence"])
      for name in OUTPUTS:
        if name not in ref or name not in pred or ref[name].shape != pred[name].shape:
          continue
        # standardise BOTH sides with the real locus's statistics
        mu = ref[name].mean(axis=0, keepdims=True)
        sd = np.maximum(ref[name].std(axis=0, keepdims=True), 1e-6)
        a, b = (pred[name] - mu) / sd, (ref[name] - mu) / sd
        row[f"mse_{name}"] = float(((a - b) ** 2).mean())
        av, bv = a.ravel(), b.ravel()
        if av.std() > 1e-8 and bv.std() > 1e-8:
          row[f"r_{name}"] = float(np.corrcoef(av, bv)[0, 1])
      if (i + 1) % 25 == 0:
        print(f"  scored {i+1}/{len(records)}", flush=True)
    rows.append(row)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(
    {"gen": str(args.gen), "n_records": len(rows), "rows": rows}, indent=2))
  print(f"\nwrote {args.out}")

  gaps = sorted({r["gap_nt"] for r in rows})
  metrics = [("mse_RNA_SEQ", "RNA-seq MSE", True), ("r_RNA_SEQ", "RNA-seq r", False),
             ("mse_CHIP_HISTONE", "histone MSE", True),
             ("kmer_js_bits", "6-mer JS bits", True),
             ("identity", "identity", False)]
  for field, title, lower_better in metrics:
    if not any(field in r and r[field] is not None for r in rows):
      continue
    print(f"\n{title}  ({'lower' if lower_better else 'higher'} is better)")
    print(f"{'condition':<12}" + "".join(f"{g:>12,}" for g in gaps))
    for cond in ORDER:
      cells = ""
      for g in gaps:
        vals = [r[field] for r in rows
                if r["condition"] == cond and r["gap_nt"] == g
                and r.get(field) is not None]
        cells += f"{np.mean(vals):>12.4f}" if vals else f"{'-':>12}"
      if cells.strip():
        print(f"{cond:<12}{cells}")


if __name__ == "__main__":
  main()
