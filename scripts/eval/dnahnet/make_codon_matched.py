#!/usr/bin/env python
"""Build a CODON-MATCHED reference for each single-amino-acid substitution.

THE PROBLEM. MaveDB's DNA is synthetic and each variant was independently
codon-randomised, so a one-amino-acid change carries a median of ELEVEN
nucleotide changes. Scoring `mutant - wild-type` therefore contrasts two
sequences that differ in both the protein AND ~10 unrelated synonymous codons.
`partial_corr.py` regresses out a ranked edit COUNT, which removes a rank-linear
association but not codon identity, GC pattern, position effects or non-linear
interactions.

THE CONTROL. Keep the mutant's own synonymous choices everywhere, and revert
ONLY the codon that changes the amino acid, using the wild-type's codon at that
position. The matched reference then encodes the wild-type protein while sharing
every unrelated synonymous change with the mutant, so

    s_matched = l(mutant) - l(matched reference)

differs from the ordinary score by exactly the thing we want to control.

Emitted as a drop-in .jsonl.gz with `wt_sequence` replaced by the matched
reference, so any existing --score-mode scores it unchanged.
"""
import argparse, gzip, json
from pathlib import Path

CODONS = {}
for _b1 in "TCAG":
  for _b2 in "TCAG":
    for _b3 in "TCAG":
      CODONS[_b1 + _b2 + _b3] = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"[
        (("TCAG".index(_b1) * 16) + ("TCAG".index(_b2) * 4) + "TCAG".index(_b3))]


def translate(seq):
  return "".join(CODONS.get(seq[i:i + 3], "X") for i in range(0, len(seq) - 2, 3))


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--data", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  kept, skipped = [], {"length": 0, "not_one_codon": 0, "protein_mismatch": 0}
  with gzip.open(args.data, "rt") as handle:
    for line in handle:
      record = json.loads(line)
      wt, mut = record["wt_sequence"], record["mut_sequence"]
      if len(wt) != len(mut) or len(wt) % 3:
        skipped["length"] += 1
        continue
      wt_codons = [wt[i:i + 3] for i in range(0, len(wt), 3)]
      mut_codons = [mut[i:i + 3] for i in range(0, len(mut), 3)]
      differing = [k for k, (a, b) in enumerate(zip(wt_codons, mut_codons))
                   if CODONS.get(a, "X") != CODONS.get(b, "X")]
      if len(differing) != 1:
        skipped["not_one_codon"] += 1
        continue
      k = differing[0]
      matched = list(mut_codons)
      matched[k] = wt_codons[k]            # revert ONLY the coding change
      matched = "".join(matched)
      # the matched reference must encode the wild-type protein exactly
      if translate(matched) != translate(wt):
        skipped["protein_mismatch"] += 1
        continue
      out = dict(record)
      out["wt_sequence"] = matched
      out["codon_matched"] = True
      out["matched_codon_index"] = k
      out["nt_diff_original"] = sum(1 for a, b in zip(wt, mut) if a != b)
      out["nt_diff_matched"] = sum(1 for a, b in zip(matched, mut) if a != b)
      kept.append(out)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  with gzip.open(args.out, "wt") as handle:
    for record in kept:
      handle.write(json.dumps(record) + "\n")

  orig = sum(r["nt_diff_original"] for r in kept) / max(len(kept), 1)
  new = sum(r["nt_diff_matched"] for r in kept) / max(len(kept), 1)
  print(f"wrote {args.out}  n={len(kept)}  skipped={skipped}")
  print(f"  mean nt difference to the reference: {orig:.2f} -> {new:.2f}")


if __name__ == "__main__":
  main()
