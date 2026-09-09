#!/usr/bin/env python3
"""Report the SEQUENCE-ONLY Task 2 arm, with a real significance test.

This is the properly-powered version of the generation claim. The AlphaGenome
endpoint cannot resolve the effect: at n=40 two checkpoints 4,500 steps apart
gave opposite verdicts on the ca-vs-mismatch control, with no p below 0.2.
Sequence metrics need no oracle calls, so n can be ~1,000 instead.

WHAT IS TESTED, and in what order:

  PRIMARY   identity(ca) - identity(denovo)  at gap 256, one-sided sign test.
            gap 256 is ONE block, the only width where the generation contract
            equals the training contract.
  CONTROL   identity(ca) - identity(mismatch). Must also favour ca, else `ca`
            merely benefits from a populated cache rather than the true suffix.
  SECONDARY k-mer JS divergence, GC error, motif recovery; gap 1024.

WHY THE SIGN TEST AND NOT A t-TEST. The per-locus differences are tiny (~4e-3
identity) against a hard floor: generated interiors sit near 0.273 identity
versus 0.25 for chance and 0.2695 for a dinucleotide shuffle, so the whole
dynamic range is ~2 points. The distribution is bounded and skewed; a sign test
makes no distributional assumption and answers exactly the question asked --
does the suffix help on more loci than not.

INDEPENDENCE CAVEAT, reported not buried. Loci are random starts inside 3,727
chr8/chr9 intervals, so two can overlap. This prints the count of distinct
intervals alongside n; if that is much smaller than n, the effective sample size
is smaller than n and the p-values are optimistic.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

# lower-is-better metrics are negated so "favours ca" always means the same thing
METRICS = (("identity", False, "identity vs the true interior (higher better)"),
           ("kmer_js_bits", True, "k-mer JS divergence (lower better)"),
           ("gc_abs_err", True, "GC error (lower better)"),
           ("motif_cosine", False, "JASPAR motif profile cosine (higher better)"))


def pairs(rows, field, a, b, gap):
  by = defaultdict(dict)
  for r in rows:
    if r["gap_nt"] != gap:
      continue
    v = r.get(field)
    if v is not None:
      by[(r["chrom"], r["start"])][r["condition"]] = v
  return {k: c[a] - c[b] for k, c in by.items() if a in c and b in c}


def sign_test(d, lower_is_better):
  """One-sided: how many loci favour `a`. Returns (median, k, n, p, ties).

  TIES ARE EXCLUDED, which is standard for a sign test and was NOT what the
  first version did. identity at gap 256 is quantised to multiples of 1/256, so
  116 of 1,000 loci were EXACT ties; counting them as "does not favour ca" drove
  the pre-registered primary endpoint from p=5.1e-06 to p=0.32 and would have
  been reported as a null result. A tie is evidence for neither side, not
  evidence against.
  """
  if not d:
    return None, 0, 0, 1.0, 0
  allv = list(d.values())
  vals = [x for x in allv if x != 0]
  ties = len(allv) - len(vals)
  if not vals:
    return statistics.median(allv), 0, 0, 1.0, ties
  k = sum(1 for x in vals if (x < 0 if lower_is_better else x > 0))
  n = len(vals)
  p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
  return statistics.median(allv), k, n, p, ties


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scores", required=True)
  ap.add_argument("--causal", default=None)
  ap.add_argument("--primary-gap", type=int, default=256)
  args = ap.parse_args()

  d = json.loads(Path(args.scores).read_text())
  rows = d.get("rows", [])
  gaps = sorted({r["gap_nt"] for r in rows})
  prov = d.get("provenance") or {}
  print(f"arm    : {d.get('label')}")
  print(f"trained: rf={prov.get('right_flank_probability')} "
        f"len={prov.get('trained_length')} data={prov.get('train_data')}")
  print(f"rows   : {len(rows)}   gaps: {gaps}")

  loci = {(r["chrom"], r["start"]) for r in rows}
  chroms = defaultdict(int)
  for c, _ in loci:
    chroms[c] += 1
  print(f"loci   : {len(loci)} distinct  ({dict(chroms)})\n")

  g = args.primary_gap
  print("=" * 74)
  print(f"PRIMARY: identity(ca) - identity(denovo) at gap {g} nt (one block,")
  print("         the width where generation matches the training contract)")
  print("CONTROL: the same against `mismatch`; must also favour ca")
  print("-" * 74)
  verdict = {}
  for ctrl in ("denovo", "mismatch"):
    dd = pairs(rows, "identity", "ca", ctrl, g)
    m, k, n, p, ties = sign_test(dd, lower_is_better=False)
    verdict[ctrl] = (m, k, n, p)
    if m is None:
      print(f"  ca - {ctrl:<9}: no paired data")
    else:
      print(f"  ca - {ctrl:<9}: median {m:+.5f}   {k}/{n} loci favour ca "
            f"({100*k/n:.1f}%)   p={p:.4g}   [{ties} ties excluded]")
  md, kd, nd, pd = verdict["denovo"]
  mm, km, nm, pm = verdict["mismatch"]
  print("-" * 74)
  if md is None or mm is None:
    print("  VERDICT: insufficient data")
  elif md > 0 and mm > 0 and pd < 0.05 and pm < 0.05:
    print("  VERDICT: SUPPORTED. The suffix improves generated sequence, and the")
    print("           mismatch control confirms it is the CORRECT suffix, not")
    print("           merely a populated cache. Both significant at p<0.05.")
  elif md > 0 and pd < 0.05 and pm >= 0.05:
    print("  VERDICT: PARTIAL. ca beats denovo significantly, but the mismatch")
    print("           control is not significant -- cannot yet exclude that any")
    print("           populated cache would do. Report with that caveat.")
  elif nd >= 400:
    # Only a LARGE n licenses reading a null as informative. The first version of
    # this branch printed "with n this large" unconditionally and said it at
    # n=40, which is the same overclaim this script exists to prevent.
    print("  VERDICT: NOT SUPPORTED at p<0.05, and n is large enough for that to")
    print(f"           be INFORMATIVE (n={nd}). It bounds the generation-level")
    print("           effect as very small; the likelihood result (+0.015 nats/nt,")
    print("           well-powered) becomes the honest headline.")
  else:
    rate = kd / nd if nd else 0.5
    print(f"  VERDICT: UNDERPOWERED (n={nd}). Not significant, but n is too small")
    print(f"           for the null to mean anything. Observed rate {100*rate:.1f}%;")
    print("           ~1,000 loci are needed for 80% power at that rate. This is")
    print("           NOT evidence of absence -- do not report a direction.")
  print("=" * 74 + "\n")

  print("SECONDARY -- every metric, every gap:\n")
  print(f"  {'gap':>6}  {'metric':<16}{'contrast':<12}{'median':>11}{'k/n':>12}{'p':>10}")
  for gg in gaps:
    for field, lower, _ in METRICS:
      for ctrl in ("denovo", "mismatch"):
        dd = pairs(rows, field, "ca", ctrl, gg)
        m, k, n, p, ties = sign_test(dd, lower)
        if m is None or n == 0:
          continue
        star = " *" if p < 0.05 else ""
        print(f"  {gg:>6}  {field:<16}{ctrl:<12}{m:>+11.5f}{f'{k}/{n}':>12}{p:>10.4g}{star}  ({ties} ties)")

  print("\n  ABSOLUTE identity by condition (the dynamic range this all sits in):")
  for gg in gaps:
    parts = []
    for c in ("ca", "denovo", "mismatch", "dinuc"):
      v = [r["identity"] for r in rows
           if r["gap_nt"] == gg and r["condition"] == c and r.get("identity") is not None]
      if v:
        parts.append(f"{c}={statistics.median(v):.4f}")
    print(f"    gap {gg:>5}: " + "  ".join(parts) + "   (chance = 0.2500)")

  if args.causal:
    dc = json.loads(Path(args.causal).read_text())
    rc = dc.get("rows", [])
    print(f"\nCAUSAL ARM ({dc.get('label')}) -- weight-matched, denovo only.")
    print("  bissm_denovo - causal_denovo isolates the WITHIN-BLOCK reverse scan;")
    print("  ca - denovo above isolates the cross-block right cache.")
    for gg in gaps:
      bb = {(r["chrom"], r["start"]): r.get("identity") for r in rows
            if r["gap_nt"] == gg and r["condition"] == "denovo"}
      cc = {(r["chrom"], r["start"]): r.get("identity") for r in rc
            if r["gap_nt"] == gg and r["condition"] == "denovo"}
      dd = {k: bb[k] - cc[k] for k in bb
            if k in cc and bb[k] is not None and cc[k] is not None}
      m, k, n, p, ties = sign_test(dd, lower_is_better=False)
      if m is not None and n > 0:
        star = " *" if p < 0.05 else ""
        print(f"    gap {gg:>5}: median {m:+.5f}  {k}/{n} favour bissm  p={p:.4g}{star}  ({ties} ties)")


if __name__ == "__main__":
  main()
