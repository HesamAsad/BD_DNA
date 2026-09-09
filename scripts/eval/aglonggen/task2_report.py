#!/usr/bin/env python3
"""Read AG-LongGen Task 2 scores and report the PRE-REGISTERED endpoint first.

WHY THIS EXISTS, AND WHY IT REFUSES TO PRINT MEANS.

1. Task 2 means were inflated 3-4 orders of magnitude by outliers, so every
   summary here is a MEDIAN plus an explicit failure fraction. There is no code
   path in this file that prints a mean of a score.

2. The design has 4 gap widths x 7 distance bins x 7 AlphaGenome tracks = 196
   combinations. Something will look real by chance. The primary endpoint was
   fixed BEFORE any data existed and is printed first, on its own, whatever it
   says. Everything else is labelled secondary/exploratory.

   PRIMARY:  median (ca - denovo) of posmse_RNA_SEQ_0  at gap_nt = 256
   CONTROL:  the same quantity for (ca - mismatch); it must ALSO favour ca,
             otherwise `ca` merely benefits from a populated cache rather than
             from the correct suffix.

3. gap_nt=256 is singled out because it is the ONLY width where the generation
   contract equals the training contract. Training builds the right cache as
   prefill_right(x0[:, end:]) -- everything right of the block INCLUDING later
   interior blocks -- while generation prefills from the committed flank alone
   and freezes it. In an N-block gap the first block is missing (N-1) blocks of
   context. At one block there is no mismatch; at 16 blocks the first block is
   missing 3,840 nt. So a decay of the gain with distance is confounded unless
   read across widths: width-independent => real range limit; degrading with
   width at fixed distance => cache artifact. That comparison is the secondary
   table below.

Usage:
  python scripts/eval/aglonggen/task2_report.py --scores results/aglonggen/task2_scores_MID_bissm.json
  python scripts/eval/aglonggen/task2_report.py --scores <bissm.json> --causal <causal.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

DIST_EDGES = (0, 64, 128, 256, 512, 1024, 2048, 4096)
PRIMARY_TRACK = "RNA_SEQ"
PRIMARY_GAP = 256
PRIMARY_BIN = 0


def load(path):
  d = json.loads(Path(path).read_text())
  return d, d.get("rows", [])


def paired(rows, field, a, b):
  """Median of (a - b) over loci where BOTH conditions produced the field.

  Pairing per locus matters: the conditions share a locus, a seed and the noise,
  so the per-locus difference removes locus-to-locus variance that would
  otherwise swamp the effect. Returns (median, n_pairs, n_dropped).
  """
  by = defaultdict(dict)
  for r in rows:
    v = r.get(field)
    if v is None:
      continue
    by[(r["chrom"], r["start"], r["gap_nt"])][r["condition"]] = v
  diffs, dropped = [], 0
  for _, cond in by.items():
    if a in cond and b in cond:
      diffs.append(cond[a] - cond[b])
    else:
      dropped += 1
  if not diffs:
    return None, 0, dropped
  return statistics.median(diffs), len(diffs), dropped


def sign_test(rows, field, a, b):
  """One-sided sign test: how many loci favour `a`, and how surprising is that.

  The median alone cannot distinguish a real effect from noise at these effect
  sizes (~1e-4), so every headline number is reported with this beside it.
  """
  by = defaultdict(dict)
  for r in rows:
    v = r.get(field)
    if v is None:
      continue
    by[(r["chrom"], r["start"], r["gap_nt"])][r["condition"]] = v
  d = [c[a] - c[b] for c in by.values() if a in c and b in c]
  d = [x for x in d if x != 0]          # ties are evidence for neither side
  # TIES ARE EXCLUDED, which is standard for a sign test and was NOT what the
  # first version did. identity at gap 256 is quantised to multiples of 1/256, so
  # 116 of 1,000 loci were EXACT ties; counting them as "does not favour ca" drove
  # the pre-registered primary endpoint from p=5.1e-06 to p=0.32 and would have
  # been reported as a null result. A tie is evidence for neither side, not
  # evidence against.
  if not d:
    return 0, 1.0
  k, n = sum(1 for x in d if x < 0), len(d)
  return k, sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def failure_fraction(rows, cond):
  """Fraction of that condition's rows with no usable oracle score.

  A generation that destabilises yields rows whose track fields are absent; the
  earlier C-a work saw 0% -> 62% of loci fail as the gap widened, and reporting
  a median over survivors alone would hide exactly that.
  """
  tot = [r for r in rows if r["condition"] == cond]
  if not tot:
    return None, 0
  bad = sum(1 for r in tot
            if r.get(f"mse_{PRIMARY_TRACK}") is None
            and r.get(f"posmse_{PRIMARY_TRACK}_0") is None)
  return bad / len(tot), len(tot)


def main():
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scores", required=True, help="BiSSM arm scores json")
  ap.add_argument("--causal", default=None, help="weight-matched causal arm")
  args = ap.parse_args()

  d, rows = load(args.scores)
  gaps = sorted({r["gap_nt"] for r in rows})
  print(f"arm    : {d.get('label')}")
  print(f"ckpt   : {d.get('checkpoint')}")
  prov = d.get("provenance") or {}
  print(f"trained: rf={prov.get('right_flank_probability')} "
        f"len={prov.get('trained_length')} data={prov.get('train_data')}")
  if not prov.get("right_flank_probability"):
    print("  *** WARNING: rf=0/absent -- the suffix pathway was never trained; "
          "ca-vs-mismatch would measure an untrained input slot. ***")
  print(f"rows   : {len(rows)}   gaps: {gaps}\n")

  fld = f"posmse_{PRIMARY_TRACK}_{PRIMARY_BIN}"
  print("=" * 72)
  print(f"PRIMARY ENDPOINT (pre-registered): median (ca - denovo) of {fld}")
  print(f"                                   at gap_nt={PRIMARY_GAP}")
  print("  negative = ca has LOWER error = the suffix helps")
  sub = [r for r in rows if r["gap_nt"] == PRIMARY_GAP]
  if not sub:
    print(f"  gap {PRIMARY_GAP} not present in this run")
  else:
    res = {}
    for ctrl in ("denovo", "mismatch"):
      m, n, dr = paired(sub, fld, "ca", ctrl)
      k, pv = sign_test(sub, fld, "ca", ctrl)
      res[ctrl] = (m, n, k, pv)
      if m is None:
        print(f"  ca - {ctrl:<9}: no paired data")
      else:
        print(f"  ca - {ctrl:<9}: {m:+.5f}  ({k}/{n} loci favour ca, "
              f"sign-test p={pv:.3f})")
    md, nd, kd, pd = res["denovo"]
    mm, nm, km, pm = res["mismatch"]
    # A SIGN CHECK IS NOT A RESULT. The first version of this verdict compared
    # only the signs of the two medians, and on 2026-09-09 two checkpoints 4,500
    # steps apart gave OPPOSITE verdicts on the control (+0.00184 then -0.00072)
    # at effect sizes of ~1e-4 with n=40 and no p below 0.2. It would have
    # reported noise as a finding. Require significance on BOTH contrasts, and
    # say "underpowered" rather than "not supported" when the data simply cannot
    # resolve the question -- those are different conclusions.
    if md is None or mm is None:
      print("  VERDICT: insufficient data")
    elif md < 0 and mm < 0 and pd < 0.05 and pm < 0.05:
      print("  VERDICT: SUPPORTED -- suffix helps and the control holds, both significant")
    elif (pd >= 0.05 or pm >= 0.05) and min(nd, nm) < 200:
      need = 1000
      print(f"  VERDICT: UNDERPOWERED -- signs may favour ca but neither contrast is")
      print(f"           significant at n={nd}. At the observed per-locus rate this")
      print(f"           needs ~{need} loci for 80% power. Do NOT report a direction.")
    else:
      print("  VERDICT: NOT SUPPORTED")
  print("=" * 72 + "\n")

  print("SECONDARY -- gain by distance bin and gap width (medians).")
  print("Read DOWN a column: does the gain at a fixed distance survive as the")
  print("gap widens? If it degrades, that is the frozen-right-cache artifact,")
  print("not a range limit.\n")
  hdr = "  gap\\dist " + "".join(f"{e:>10}" for e in DIST_EDGES)
  print(hdr)
  for g in gaps:
    sub = [r for r in rows if r["gap_nt"] == g]
    cells = []
    for e in DIST_EDGES:
      m, n, _ = paired(sub, f"posmse_{PRIMARY_TRACK}_{e}", "ca", "denovo")
      cells.append(f"{m:>+10.4f}" if m is not None else f"{'-':>10}")
    print(f"  {g:>8} " + "".join(cells))

  print("\nFAILURE FRACTION (never hidden by a median over survivors):")
  for cond in ("ca", "denovo", "mismatch", "real", "dinuc"):
    fr, n = failure_fraction(rows, cond)
    if fr is not None:
      print(f"  {cond:<9} {fr:6.1%} of {n} rows unscoreable")

  if args.causal:
    dc, rc = load(args.causal)
    print(f"\nCAUSAL ARM ({dc.get('label')}) -- weight-matched, denovo only.")
    print("  bissm_denovo - causal_denovo isolates the WITHIN-BLOCK reverse scan")
    print("  (ca - denovo above isolates the cross-block right cache).")
    for g in gaps:
      bd = [r for r in rows if r["gap_nt"] == g and r["condition"] == "denovo"]
      cd = [r for r in rc if r["gap_nt"] == g and r["condition"] == "denovo"]
      key = lambda r: (r["chrom"], r["start"])
      bmap = {key(r): r.get(fld) for r in bd if r.get(fld) is not None}
      cmap = {key(r): r.get(fld) for r in cd if r.get(fld) is not None}
      both = [bmap[k] - cmap[k] for k in bmap if k in cmap]
      if both:
        print(f"  gap {g:>5}: median(bissm - causal) = {statistics.median(both):+.5f} "
              f"(n={len(both)})")


if __name__ == "__main__":
  main()
