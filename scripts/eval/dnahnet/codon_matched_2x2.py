#!/usr/bin/env python3
"""The {AR, BD} x {original, codon-matched} grid at matched geometry.

WHY. MaveDB's DNA sequences are codon-randomised per variant, so a single amino
acid substitution arrives as a median of 11 nucleotide changes, 10 of which are
synonymous and invisible to the assay. `make_codon_matched.py` rebuilds each
reference so only the minimal edits remain. The question this script answers is
not "is codon matching good" -- it plainly is, for everything -- but whether it
is DIFFERENTIAL: does it recover more for block diffusion than for
autoregression, and therefore explain the AR-to-BD drop?

Read it this way:
  * the codon-matching effect is WITHIN an arm, so geometry is held fixed and
    the comparison is clean;
  * the AR-vs-BD contrast is only legible WITHIN a geometry column -- the AR
    arms run --genomic-prefix at L=512, and a BD arm without the prefix at
    L=256 is not comparable to them.

Every arm is restricted to the same accessions before anything is computed.
Join on `accession`: the codon-matched build rewrites `hgvs_nt`, so joining on
that silently matches nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path("results/dnahnet/mavedb")

# label -> (original run, codon-matched run). Both halves of a row share a
# geometry; rows do not necessarily share one with each other.
GRID: dict[str, tuple[str, str]] = {
    "uSSM-AR (gp, L=512)": ("ussm-ar-genomicprefix-154314", "cm-ussmAR-gp-158393"),
    "Transformer-AR (gp, L=512)": (
        "transformer-ar-genomicprefix-154315",
        "cm-xfAR-gp-158394",
    ),
    "BiSSM-BD b8 block_marginal (L=256)": (
        "b8-blockmarginal-157720",
        "cm-b8-blockmarg-158250",
    ),
    "BiSSM-BD b8 seq_unmask (L=256)": ("b8-sequnmask-157712", "cm-b8-sequnmask-158251"),
    "BiSSM-BD b8 block_marginal (gp, L=512)": (
        "gp-b8-blockmarg-158402",
        "cmgp-b8-blockmarg-158401",
    ),
    "BiSSM-BD b8 seq_unmask (gp, L=512)": (
        "gp-b8-sequnmask-158404",
        "cmgp-b8-sequnmask-158403",
    ),
}
AR_ROWS = [k for k in GRID if "-AR" in k]
BD_ROWS = [k for k in GRID if "BiSSM" in k]


def per_assay(frame: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for urn, group in frame.groupby("score_set_urn"):
        group = group.dropna(subset=["experimental_score", "predicted_fitness"])
        if len(group) < 10 or group.predicted_fitness.nunique() < 2:
            continue
        out[urn] = spearmanr(group.predicted_fitness, group.experimental_score).statistic
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()

    def load(run: str) -> pd.DataFrame:
        return pd.read_csv(args.root / run / "predictions.csv")

    missing = [r for pair in GRID.values() for r in pair
               if not (args.root / r / "predictions.csv").exists()]
    if missing:
        print("missing runs, cannot assemble the grid:")
        for m in missing:
            print("   ", m)
        return 1

    # the codon-matched build defines the row set; every arm is cut down to it
    shared = set(load(GRID["uSSM-AR (gp, L=512)"][1]).accession)
    for _, cm in GRID.values():
        shared &= set(load(cm).accession)
    print(f"shared accessions across all {len(GRID)} arms: {len(shared)}\n")

    gains: dict[str, np.ndarray] = {}
    print(f"{'arm':40s} {'original':>9} {'codon-m':>9} {'gain':>8}  paired over assays")
    print("-" * 96)
    for label, (orig, cm) in GRID.items():
        mo = per_assay(load(orig)[lambda d: d.accession.isin(shared)])
        mc = per_assay(load(cm)[lambda d: d.accession.isin(shared)])
        keys = sorted(set(mo) & set(mc))
        delta = np.array([mc[k] - mo[k] for k in keys])
        gains[label] = delta
        p = wilcoxon(delta).pvalue
        print(
            f"{label:40s} {np.mean([mo[k] for k in keys]):9.4f} "
            f"{np.mean([mc[k] for k in keys]):9.4f} {delta.mean():+8.4f}  "
            f"{(delta > 0).sum():2d}/{len(keys)} positive, p={p:.4f}"
        )

    ar = np.mean([gains[k] for k in AR_ROWS], axis=0)
    bd = np.mean([gains[k] for k in BD_ROWS], axis=0)
    diff = ar - bd
    p = wilcoxon(diff).pvalue
    print()
    print(
        f"AR mean gain {ar.mean():+.4f}   BD mean gain {bd.mean():+.4f}   "
        f"difference {diff.mean():+.4f}, {(diff > 0).sum()}/{len(diff)} positive, p={p:.3f}"
    )
    verdict = "DIFFERENTIAL" if p < 0.05 else "NOT differential"
    print(
        f"\ncodon matching is {verdict}: it "
        + (
            "does explain part of the AR-to-BD drop."
            if p < 0.05
            else "lifts both objectives alike and does NOT explain the AR-to-BD drop."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
