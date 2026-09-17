#!/usr/bin/env python3
"""Mark which GenomicBenchmarks TEST sequences come from chr8 or chr9.

WHY. `hg_bissm_cos` was pretrained on `hg38-cad-no89`, a bed-filtered hg38 with
chr8 and chr9 removed. 7.67% of the benchmark's test sequences (14,564 of
189,859) live on those two chromosomes, so for that checkpoint they are a genuine
held-out slice: sequence the model has never seen even through self-supervised
pretraining. Accuracy on that slice versus the rest is a direct test of whether
hg38 pretraining contamination drives the benchmark -- a control that neither the
current GB checkpoint (`human-lr8192v2`, full hg38) nor Caduceus can run.

HOW. The HuggingFace mirror carries only `seq` and `label`, no coordinates. The
upstream repo carries coordinates but not sequence. They cannot be joined by row
index: the mirror is shuffled within each class block (verified -- class counts
match, indexed sequences do not). So the join is by sequence content, and only
the chr8/chr9 coordinates need extracting from the reference, which is what makes
this cheap.

AMBIGUITY IS REPORTED, NOT GUESSED. A short sequence can occur at many loci --
`human_enhancers_ensembl` contains sequences as short as 2 nt -- so a sequence
found on chr8/9 may also occur elsewhere. Any sequence whose chr8/9 membership is
not unambiguous is marked `-1` and excluded from both sides of the comparison
rather than assigned to one.

THE CONTROL APPLIES TO 5 OF THE 8 TASKS, and the other three fail for reasons
that must not be silently folded into "ambiguous" (the first version of this
script did exactly that, and reported `dummy_mouse_enhancers_ensembl` as 100%
ambiguous when the truth is that the question is meaningless there):

  dummy_mouse_enhancers_ensembl   MOUSE coordinates. Its regions are named
    chr1..chr19 -- mouse has 19 autosomes -- so they LOOK like human
    chromosomes, extract successfully from hg38, and yield the wrong sequence.
    Its "chr8"/"chr9" are mouse chr8/chr9, unrelated to the human chr8/chr9 that
    `hg38-cad-no89` excludes. Not placeable; excluded.
  demo_human_or_worm              Half C. elegans (regions I, II, III, IV, V, X,
    MtDNA). The worm class is trivially absent from human pretraining, so the
    comparison is confounded with organism. Excluded.
  demo_coding_vs_intergenomic_seqs  The coding class is addressed by TRANSCRIPT
    ID (ENST...), not by genomic coordinate, so it cannot be placed on a
    chromosome without a GTF. Excluded.

Detection is automatic rather than hardcoded: a task whose extracted coordinates
mostly fail to match any mirror sequence is reporting that the reference genome is
wrong for it, and is marked not-applicable with that evidence attached.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

CHR89 = {"chr8", "chr9", "8", "9"}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence):
  return sequence.translate(COMPLEMENT)[::-1]


def main():
  if len(sys.argv) < 3:
    print(__doc__)
    print("usage: build_chr89_index.py <csv_dir> <out.json> [genome.fa]")
    return 2
  csv_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
  genome = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data/hg38/hg38.ml.fa"

  from datasets import load_dataset
  from pyfaidx import Fasta
  fasta = Fasta(str(genome), as_raw=True, sequence_always_upper=True)

  tasks = sorted({p.name.split("__")[0] for p in csv_dir.glob("*__test__*.csv.gz")})
  summary = {}
  for task in tasks:
    # Every test coordinate, so that "also occurs off chr8/9" is detectable.
    on89, elsewhere = set(), set()
    for path in sorted(csv_dir.glob(f"{task}__test__*.csv.gz")):
      with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
          chrom = row["region"]
          if chrom not in fasta:
            continue
          start, end = int(row["start"]), int(row["end"])
          sequence = str(fasta[chrom][start:end]).upper()
          if row["strand"] == "-":
            sequence = reverse_complement(sequence)
          (on89 if chrom in CHR89 else elsewhere).add(sequence)

    data = load_dataset("katarinagresova/Genomic_Benchmarks_" + task)["test"]
    column = "seq" if "seq" in data.column_names else "sequence"
    flags = []
    for sequence in data[column]:
      sequence = sequence.upper()
      here, there = sequence in on89, sequence in elsewhere
      # 1 = chr8/9 only; 0 = elsewhere only; -1 = both (genuinely ambiguous);
      # -2 = neither, i.e. this coordinate set does not describe this reference.
      flags.append(1 if (here and not there)
                   else 0 if (there and not here)
                   else -1 if (here and there) else -2)
    flags = np.asarray(flags)
    placed = int((flags >= 0).sum())
    applicable = placed >= 0.5 * len(flags)
    summary[task] = {
      "n": int(len(flags)),
      "chr89": int((flags == 1).sum()),
      "other": int((flags == 0).sum()),
      "ambiguous": int((flags == -1).sum()),
      "unplaceable": int((flags == -2).sum()),
      "applicable": bool(applicable),
      "reason": ("" if applicable else
                 f"only {placed}/{len(flags)} test sequences could be placed on "
                 f"this reference, so its coordinates do not describe hg38 "
                 f"(wrong organism, or transcript-relative addressing)"),
      "flags": flags.tolist(),
    }
    s = summary[task]
    mark = "" if applicable else "   NOT APPLICABLE"
    print(f"{task:36s} n={s['n']:6d}  chr8/9={s['chr89']:6d} "
          f"({100 * s['chr89'] / s['n']:5.2f}%)  other={s['other']:6d}  "
          f"ambig={s['ambiguous']:5d}  unplaceable={s['unplaceable']:6d}{mark}",
          flush=True)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(summary) + "\n")
  usable = {k: v for k, v in summary.items() if v["applicable"]}
  total = sum(v["n"] for v in usable.values())
  hit = sum(v["chr89"] for v in usable.values())
  amb = sum(v["ambiguous"] for v in usable.values())
  print(f"\nAPPLICABLE to {len(usable)}/{len(summary)} tasks: "
        f"n={total} chr8/9={hit} ({100 * hit / total:.2f}%) "
        f"ambiguous={amb} ({100 * amb / total:.3f}%)")
  for name, value in sorted(summary.items()):
    if not value["applicable"]:
      print(f"  EXCLUDED {name}: {value['reason']}")
  print(f"wrote {out_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
