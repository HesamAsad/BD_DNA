# Minimal dnaHNet benchmark alignment

This repository compares the de-novo BiSSM and Transformer BD3-LM checkpoints
against the evaluation protocols in Shah et al., *dnaHNet: A Scalable and
Hierarchical Foundation Model for Genomic Sequence Learning* (arXiv:2602.10603).
The models and training objectives are intentionally different, so results are
protocol-aligned rather than data- or compute-matched unless explicitly stated.

## Headline deliverables

1. MaveDB E. coli K-12 variant-effect prediction: **signed** macro per-assay
   Spearman from WT-versus-mutant scores. Absolute Spearman is still reported
   for comparability with runs predating 2026-09-13, but it is not the headline:
   it credits an anti-correlated assay as skill, and all 12 assays share one
   direction, so taking `abs()` inflates the block-diffusion arms about 1.5x
   against 1.13x for the autoregressive ones -- it NARROWS a real gap.
2. DEG gene essentiality: AUROC from WT-versus-15-bp-stop likelihood
   differences in 8,192-nt gene-centred windows. Protocol, assumptions,
   baselines and cost are in `docs/deg_benchmark_plan.md`; the harness is
   `scripts/eval/dnahnet/{prepare_deg,score_deg,deg}.py` + `deg_score.sh`.
3. Single-GPU BF16 throughput, peak memory and latency from 2^10 through 2^19
   nucleotides.

The direct comparison is always de novo. C-a scores condition on an observed
right flank and therefore belong in a separate supplemental evaluation.

## Wall-clock forward scaling

dnaHNet Appendix A.5 measures batch-one BF16 forward passes on a single H100
from 2^10 through 2^19 nucleotides. Run the analogous checkpoint-backed
diffusion likelihood forward on H200 for each arm:

```bash
bsub -env "all,CKPT=/absolute/bissm.ckpt,LABEL=bissm" \
  < scripts/eval/dnahnet/forward_profile.sh
bsub -env "all,CKPT=/absolute/transformer.ckpt,LABEL=transformer" \
  < scripts/eval/dnahnet/forward_profile.sh
```

Each point reconstructs the checkpoint at that context length, scores one
fixed `t=0.5` corruption at batch size one in BF16, excludes one warm-up, and
reports the median of three forwards plus peak allocated GPU memory. Results
are written after every length so an OOM still leaves a usable prefix of the
curve. Plot the same three panels used by dnaHNet:

```bash
python scripts/eval/dnahnet/plot_forward_profile.py \
  --result BiSSM=results/dnahnet/forward/bissm.json \
  --result Transformer=results/dnahnet/forward/transformer.json \
  --output results/dnahnet/forward/forward_scaling.png
```

This matches the paper's metric family and length sweep, but not its hardware
or objective: the paper uses autoregressive H100 forwards, whereas these are
diffusion likelihood forwards on H200. Do not present the curves as direct
dnaHNet speed ratios without an author checkpoint and identical hardware.

## MaveDB snapshot

The paper reports twelve nucleotide-level E. coli K-12 datasets containing
21,250 variants but does not provide their accessions. The current MaveDB API
contains twelve historical `combined scores` records that sum to exactly
21,250. They are pinned in
`scripts/eval/dnahnet/mavedb_manifest.json`. A thirteenth matching combined set
is now returned by a broad live search, so rediscovering records at runtime
would silently change the benchmark.

Prepare the CC0 snapshot:

```bash
/software/cellgen/team361/ha11/envs/nichejepa/bin/python \
  scripts/eval/dnahnet/prepare_mavedb.py
```

The preparation step verifies every title, variant count, target DNA hash,
finite experimental score and HGVS mutation. The pinned records contain the
identity sequence, substitutions, deletions, insertions and delins operations.

Score a checkpoint on one GPU:

```bash
bsub -env "all,CKPT=/absolute/checkpoint.ckpt,LABEL=bissm" \
  < scripts/eval/dnahnet/mavedb_score.sh
```

The scorer pads each short coding target to one 256-nt diffusion block with
`N`, excludes padding and the repository's ignored first position from the
score, and evaluates `NELBO(WT) - NELBO(mutant)`. WT and mutant share the same
time and corruption mask for each Monte Carlo sample. It reports per-assay
Spearman, macro mean **signed** Spearman (headline), macro mean absolute
Spearman (legacy), the count of anti-correlated assays, and pooled Spearman, so
the paper's otherwise unspecified aggregation choice remains auditable.

**Scoring modes.** `--score-mode` selects the estimator, and the choice matters
more than any model difference measured here:

- `nelbo` (default): the training objective's paired Monte Carlo bound. The
  harness default is now **128** samples, not 8. The measured curve on BiSSM-BD
  is 0.0966 (8) -> 0.1242 (32) -> 0.1318 (64) -> 0.1360 (128), so the old
  default understated every block-diffusion arm by about 0.04, roughly a third
  of its signal. Costs ~3h21m per run at L=512.
- `pll`: deterministic pseudo-log-likelihood, one masked position at a time.
- `infill_nt` / `infill_codon`: **Score I**, the masked infilling preference.
  Mask what the variant changed, one forward pass, read which spelling the model
  prefers. Deterministic, no Monte Carlo, and no length confound because only
  the changed positions contribute. `infill_codon` marginalises to amino acids,
  so a synonymous change contributes exactly zero.
- `state_displacement`: Score II's cheap premise test -- how far the variant
  moves the SSM recurrent summary. Uses `prefill_right`, deliberately: the
  mutations sit a median 171 nt from the 3' end, which a left prefill would
  attenuate by ~2^-37 and report as a null.

**What the numbers must be read against.** A zero-parameter baseline that counts
amino-acid events in `hgvs_pro` reaches macro signed rho **+0.30931** and points
the right way on 12 of 12 assays -- above every model in this repo and above
dnaHNet's published 0.3266. Counting non-synonymous codon changes reaches
**+0.337**. Measured on the 19,349 length-preserving pairs, the median variant
changes 13 nucleotides across 11 codons but only **1** amino acid, leaving a
median of 10 synonymous codon changes, so nucleotide-level scoring is diluted
roughly 10:1 by changes the assay cannot see. Every run now emits the protein
baseline into its own `summary.json`, and `partial_corr.py` controls for it.
uSSM-AR keeps +0.20548 of its +0.21644 partial rho once both counting families
are removed, which is the evidence that its signal is not merely a count.

Run at least two independent seeds, then average their per-variant likelihood
differences and record the between-seed agreement:

```bash
python scripts/eval/dnahnet/aggregate_mavedb.py \
  --prediction results/dnahnet/mavedb/bissm-seed1/predictions.csv \
  --prediction results/dnahnet/mavedb/bissm-seed2/predictions.csv \
  --output-dir results/dnahnet/mavedb/bissm-ensemble \
  --label BiSSM
```

Plot one or more completed results with the paper's Table 5 reference values:

```bash
python scripts/eval/dnahnet/plot_mavedb.py \
  --result BiSSM=results/dnahnet/mavedb/bissm/summary.json \
  --result "BD3-LM Transformer=results/dnahnet/mavedb/transformer/summary.json" \
  --output results/dnahnet/mavedb/mavedb_comparison.png
```

The first completed comparison uses two seeds with eight paired Monte Carlo
samples per seed. Its seed-to-seed score correlation is only about 0.48 for
both backbones, so it is a preliminary ranking estimate. Increase the Monte
Carlo count before using small differences as a publication claim.

## Interpretation constraints

- dnaHNet uses exact autoregressive likelihood; BD3-LM uses a Monte Carlo
  diffusion NELBO. Keep those labels visible.
- The completed BiSSM run saw 4.19B training nucleotides; dnaHNet reports a
  144B-nucleotide corpus and compute sweeps. Always report data and compute.
- These coding fragments fit inside one diffusion block, so MaveDB tests local
  biological syntax, not the recurrent prefix cache. DEG and the synthetic
  retrieval suite test longer-context behavior.
- The paper does not publish the twelve MaveDB accessions or an evaluation
  repository. If author-provided artifacts appear, compare their manifest with
  the pinned snapshot before claiming an exact reproduction.
