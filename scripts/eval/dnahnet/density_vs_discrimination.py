#!/usr/bin/env python3
"""Is the AR-to-BD drop a worse density model, or a worse discriminator?

The gap survives codon matching, sharpness, and the MC-sample count. What is
left is the pretraining objective -- but "objective" could mean two different
things, and they call for different fixes:

  DENSITY    the BD model simply assigns worse likelihood to these sequences,
             and the ranking follows. Fix: train it better / longer.
  DISCRIM.   the BD model's likelihood on wild-type sequences is competitive,
             but the *difference* it assigns between wild type and mutant does
             not track the assay. Fix: the scoring functional, not the model.

This separates them. `seq_unmask` at block b is the autoregressive chain rule,
  l_seq(x) = sum_b sum_j log q(x_{b,j} | x_{<b}, x_{b,<j}),
so a BD arm scored that way and an AR arm scored with its own exact NLL are
summing log-probabilities over the same tokens under the same factorisation.
`wt_loss / wt_scored_tokens` is therefore directly comparable between them --
which is NOT true of any NELBO-based BD arm, and is the reason this script
refuses anything but `seq_unmask` on the BD side.

Both arms must also share a genomic prefix setting: 256 nt of extra left
context is worth real nats and would otherwise be read as a density gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("results/dnahnet/mavedb")
AR_MODES = {"nelbo"}          # exact NLL for a block_size=1 arm
BD_MODES = {"seq_unmask"}     # the AR chain rule under a diffusion model


def _summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ar", required=True, help="run dir of the AR arm")
    ap.add_argument("--bd", required=True, help="run dir of the BD arm (seq_unmask)")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()

    ar_dir, bd_dir = args.root / args.ar, args.root / args.bd
    ar_s, bd_s = _summary(ar_dir), _summary(bd_dir)

    if bd_s.get("score_mode") not in BD_MODES:
        raise SystemExit(
            f"BD arm scored with {bd_s.get('score_mode')!r}; only {sorted(BD_MODES)} "
            "share the AR factorisation. A NELBO is a bound, not a likelihood, and "
            "comparing it to an exact NLL measures the bound's slack, not the model."
        )
    if ar_s.get("score_mode") not in AR_MODES or int(ar_s.get("block_size", 0)) != 1:
        raise SystemExit(
            f"AR arm has score_mode={ar_s.get('score_mode')!r} "
            f"block_size={ar_s.get('block_size')}; expected an exact-NLL block-1 arm."
        )
    if bool(ar_s.get("genomic_prefix")) != bool(bd_s.get("genomic_prefix")):
        raise SystemExit(
            f"genomic_prefix differs: AR={ar_s.get('genomic_prefix')} "
            f"BD={bd_s.get('genomic_prefix')}. 256 nt of extra left context is worth "
            "real nats and would be misread as a density gap."
        )

    ar = pd.read_csv(ar_dir / "predictions.csv")
    bd = pd.read_csv(bd_dir / "predictions.csv")
    shared = sorted(set(ar.accession) & set(bd.accession))
    ar = ar.set_index("accession").loc[shared]
    bd = bd.set_index("accession").loc[shared]
    print(f"AR  {args.ar}\nBD  {args.bd}\nshared accessions: {len(shared)}\n")

    # --- density: NLL per nucleotide on the WILD TYPE, same tokens, same rule ---
    a_nll = ar.wt_loss / ar.wt_scored_tokens
    b_nll = bd.wt_loss / bd.wt_scored_tokens
    print("DENSITY   wild-type NLL per nucleotide (lower is better)")
    print(f"  AR  {a_nll.mean():.4f}")
    print(f"  BD  {b_nll.mean():.4f}")
    print(f"  gap {b_nll.mean() - a_nll.mean():+.4f} nats/nt "
          f"({(b_nll.mean() - a_nll.mean()) / np.log(2):+.4f} bits/nt)\n")

    # --- discrimination: per-assay rank correlation of the wt-minus-mut score ---
    def macro(frame: pd.DataFrame) -> float:
        vals = []
        for _, g in frame.reset_index().groupby("score_set_urn"):
            g = g.dropna(subset=["experimental_score", "predicted_fitness"])
            if len(g) < 10 or g.predicted_fitness.nunique() < 2:
                continue
            vals.append(spearmanr(g.predicted_fitness, g.experimental_score).statistic)
        return float(np.mean(vals))

    ma, mb = macro(ar), macro(bd)
    print("DISCRIMINATION   macro signed Spearman (higher is better)")
    print(f"  AR  {ma:.4f}")
    print(f"  BD  {mb:.4f}")
    print(f"  ratio {ma / mb:.2f}x\n")

    # A uniform model over {A,C,G,T} costs ln 4. Express each arm's density as the
    # fraction of that budget it recovers, so the two axes are on comparable scales.
    floor = np.log(4)
    ar_frac = (floor - a_nll.mean()) / floor
    bd_frac = (floor - b_nll.mean()) / floor
    print(f"fraction of the ln4 uniform budget recovered:  AR {ar_frac:.1%}   BD {bd_frac:.1%}")
    print(f"relative density shortfall of BD: {1 - bd_frac / ar_frac:.1%}")
    print(f"relative ranking  shortfall of BD: {1 - mb / ma:.1%}")
    print()
    if (1 - bd_frac / ar_frac) < 0.5 * (1 - mb / ma):
        print("VERDICT: the ranking shortfall is much larger than the density "
              "shortfall -> DISCRIMINATION, not density. The BD model models these "
              "sequences nearly as well and still ranks their variants worse, so "
              "the lever is the scoring functional, not more pretraining.")
    else:
        print("VERDICT: the density shortfall is comparable to the ranking "
              "shortfall -> the BD model is simply worse on this data here, and "
              "more/better pretraining is the lever to pull.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
