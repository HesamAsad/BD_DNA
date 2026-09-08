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
# PROCAP joins CAGE as a transcription-initiation readout: there is NO TSS
# OutputType, so these two are the TSS proxies Score 2 has to work from.
OUTPUTS = ("RNA_SEQ", "ATAC", "DNASE", "CAGE", "CHIP_HISTONE",
           "SPLICE_SITES", "PROCAP")
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


# ---------------------------------------------------------------- Score 2
def peak_positions(values, lo, hi, z=3.0, min_sep=50):
  """Positions of prominent local maxima inside [lo, hi), in bin units.

  A peak is a bin exceeding mean + z*std of the window, kept only if it is the
  largest within +-min_sep bins. Deliberately simple: the comparison below is
  real-vs-reconstructed under the SAME detector, so detector bias cancels.
  """
  v = values[lo:hi].mean(axis=1) if values.ndim > 1 else values[lo:hi]
  if v.size == 0:
    return np.array([], dtype=int)
  thresh = v.mean() + z * v.std()
  cand = np.flatnonzero(v > thresh)
  keep = []
  for c in cand[np.argsort(-v[cand])]:
    if all(abs(c - k) >= min_sep for k in keep):
      keep.append(int(c))
  return np.array(sorted(keep), dtype=int)


def placement_score(pred, ref, left_nt, gap_nt, length, ccres, start):
  """Score 2: are regulatory elements placed at the right distances?

  Two complementary readouts, both restricted to the masked interior:

  (a) PEAK GEOMETRY -- a symmetric Chamfer distance between the transcription
      -initiation peak sets of AG(reconstructed) and AG(real). This asks
      whether the model put initiation signal WHERE the real locus has it, not
      merely whether the average level matches, which is what track-MSE
      measures and why the spec lists placement separately.

  (b) cCRE-ANCHORED OFFSET -- for each ENCODE cCRE that falls in the interior,
      the distance from that element to the nearest initiation peak, in the
      reconstruction versus in the real locus. `ccres_in_gap` is emitted by
      task2_generate and was previously never read.
  """
  out = {}
  for name in ("CAGE", "PROCAP"):
    if name not in ref or name not in pred or ref[name].shape != pred[name].shape:
      continue
    stride = max(length // ref[name].shape[0], 1)
    lo, hi = left_nt // stride, (left_nt + gap_nt) // stride
    if hi <= lo:
      continue
    pr = peak_positions(pred[name], lo, hi)
    rf = peak_positions(ref[name], lo, hi)
    out[f"npeak_{name}_pred"] = int(pr.size)
    out[f"npeak_{name}_real"] = int(rf.size)
    if pr.size and rf.size:
      d1 = np.abs(pr[:, None] - rf[None, :]).min(axis=1).mean()
      d2 = np.abs(rf[:, None] - pr[None, :]).min(axis=1).mean()
      out[f"chamfer_{name}_nt"] = float(0.5 * (d1 + d2) * stride)
    elif pr.size or rf.size:
      out[f"chamfer_{name}_nt"] = float(gap_nt)   # one side empty: worst case
    if ccres:
      errs = []
      for c in ccres:
        mid = ((c["start"] + c["end"]) // 2 - start) // stride - lo
        if not (0 <= mid < hi - lo):
          continue
        dp = float(np.abs(pr - mid).min()) * stride if pr.size else gap_nt
        dr = float(np.abs(rf - mid).min()) * stride if rf.size else gap_nt
        errs.append(abs(dp - dr))
      if errs:
        out[f"ccre_offset_err_{name}_nt"] = float(np.mean(errs))
        out["n_ccre_scored"] = len(errs)
  return out


# ---------------------------------------------------------------- Score 3b
_ONEHOT = {c: i for i, c in enumerate("ACGT")}


def _encode(seq):
  a = np.zeros((len(seq), 4), dtype=np.float32)
  for i, ch in enumerate(seq):
    j = _ONEHOT.get(ch)
    if j is not None:
      a[i, j] = 1.0
  return a


def load_motifs(path, top=None):
  """JASPAR/MEME PFMs -> list of (name, log-odds matrix [w,4])."""
  from Bio import motifs as biomotifs
  with open(path) as handle:
    parsed = biomotifs.parse(handle, "minimal")
  out = []
  for m in parsed:
    pssm = m.pssm
    mat = np.array([[pssm[b][i] for b in "ACGT"] for i in range(m.length)],
                   dtype=np.float32)
    mat[~np.isfinite(mat)] = -10.0
    out.append((m.name or m.matrix_id, mat))
  return out[:top] if top else out


def motif_hits(seq, motifs_, z=4.0):
  """Hit COUNT per motif via a vectorised sliding-window log-odds scan.

  Biopython's own search loops in Python and is far too slow at this scale
  (879 motifs x ~600 sequences), so the window sum is done with
  sliding_window_view instead.
  """
  if len(seq) < 30:
    return {}
  x = _encode(seq)
  counts = {}
  for name, mat in motifs_:
    w = mat.shape[0]
    if w > x.shape[0]:
      continue
    win = np.lib.stride_tricks.sliding_window_view(x, (w, 4)).squeeze(1)
    scores = np.einsum("nwc,wc->n", win, mat)
    thr = scores.mean() + z * (scores.std() or 1.0)
    counts[name] = int((scores > thr).sum())
  return counts


def motif_recovery(gen, real, motifs_):
  """Cosine similarity and L1 distance between motif-hit profiles."""
  if not motifs_ or not real:
    return {}
  g, r = motif_hits(gen, motifs_), motif_hits(real, motifs_)
  keys = sorted(set(g) | set(r))
  if not keys:
    return {}
  a = np.array([g.get(k, 0) for k in keys], dtype=np.float64)
  b = np.array([r.get(k, 0) for k in keys], dtype=np.float64)
  na, nb = np.linalg.norm(a), np.linalg.norm(b)
  return {
    "motif_cosine": float(a @ b / (na * nb)) if na and nb else 0.0,
    "motif_l1_per_motif": float(np.abs(a - b).sum() / len(keys)),
    "motif_hits_gen": int(a.sum()), "motif_hits_real": int(b.sum()),
  }


# Distance bins, in nt from the RIGHT edge of the masked interior. These span
# the measured range regime (effective range 1-2 kb) with headroom.
DIST_EDGES = (0, 64, 128, 256, 512, 1024, 2048, 4096)


def positional_mse(pred, ref, left_nt, gap_nt, length):
  """Per-position squared error inside the interior, binned by distance to the
  RIGHT flank.

  WHY THIS EXISTS. task2_generate always centres the gap so the interior abuts
  the right flank, and only the gap WIDTH varies -- so a width ladder confounds
  "how wide is the hole" with "how far is this nucleotide from the committed
  suffix". Binning positions inside ONE gap by their distance to the right edge
  separates them: width is held fixed and distance varies within a single
  sequence. `ca - denovo` per bin is the bidirectional range curve.
  """
  out = {}
  for name, r in ref.items():
    if name not in pred or pred[name].shape != r.shape:
      continue
    a = pred[name]
    stride = max(length // r.shape[0], 1)
    lo, hi = left_nt // stride, (left_nt + gap_nt) // stride
    if hi <= lo:
      continue
    # standardise with the REAL locus's per-track statistics, as elsewhere
    mu = r.mean(axis=0, keepdims=True)
    sd = np.maximum(r.std(axis=0, keepdims=True), 1e-6)
    err = (((a[lo:hi] - mu) / sd - (r[lo:hi] - mu) / sd) ** 2).mean(axis=1)
    # distance in nt from each position to the right edge of the interior
    dist = (np.arange(hi - lo)[::-1] + 1) * stride
    for i, edge in enumerate(DIST_EDGES):
      top = DIST_EDGES[i + 1] if i + 1 < len(DIST_EDGES) else None
      sel = (dist > edge) & ((dist <= top) if top else True)
      if sel.any():
        out[f"posmse_{name}_{edge}"] = float(err[sel].mean())
  return out


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--gen", type=Path,
                  default=REPO / "results/aglonggen/task2_gen.json")
  ap.add_argument("--out", type=Path,
                  default=REPO / "results/aglonggen/task2_scores.json")
  ap.add_argument("--ontology", default="UBERON:0002048")
  ap.add_argument("--label", default=None,
                  help="arm identifier carried into the scores payload; without "
                       "it a two-arm comparison cannot say which row is which")
  ap.add_argument("--jaspar", default=str(REPO / "data/jaspar/JASPAR2024_CORE_vertebrates_nr_pfms_meme.txt"),
                  help="motif PFMs for Score 3b; pass '' to skip motif scoring")
  ap.add_argument("--motif-top", type=int, default=0,
                  help="use only the first N motifs (0 = all 879)")
  ap.add_argument("--skip-oracle", action="store_true",
                  help="sequence-level scores only; no API calls")
  args = ap.parse_args()

  payload = json.loads(args.gen.read_text())
  records = payload["records"]
  motifs_ = []
  if args.jaspar and Path(args.jaspar).exists():
    motifs_ = load_motifs(args.jaspar, args.motif_top or None)
    print(f"loaded {len(motifs_)} motifs for Score 3b from {args.jaspar}")
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
    if motifs_ and truth:
      row.update(motif_recovery(r["interior"], truth, motifs_))
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
      row.update(positional_mse(pred, ref, r["left_nt"], r["gap_nt"],
                                payload.get("length", 16384)))
      row.update(placement_score(pred, ref, r["left_nt"], r["gap_nt"],
                                 payload.get("length", 16384),
                                 r.get("ccres_in_gap") or [], r["start"]))
      if (i + 1) % 25 == 0:
        print(f"  scored {i+1}/{len(records)}", flush=True)
    rows.append(row)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(
    {"gen": str(args.gen), "label": args.label,
     "checkpoint": payload.get("checkpoint"),
     "provenance": payload.get("provenance"),
     "n_records": len(rows), "rows": rows}, indent=2))
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
