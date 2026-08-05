# Minimal dnaHNet benchmark alignment

This repository compares the de-novo BiSSM and Transformer BD3-LM checkpoints
against the evaluation protocols in Shah et al., *dnaHNet: A Scalable and
Hierarchical Foundation Model for Genomic Sequence Learning* (arXiv:2602.10603).
The models and training objectives are intentionally different, so results are
protocol-aligned rather than data- or compute-matched unless explicitly stated.

## Headline deliverables

1. MaveDB E. coli K-12 variant-effect prediction: absolute Spearman correlation
   from WT-versus-mutant likelihood differences.
2. DEG gene essentiality: AUROC from WT-versus-15-bp-stop likelihood
   differences in 8,192-nt gene-centred windows.
3. Single-GPU BF16 throughput, peak memory and latency from 2^10 through 2^19
   nucleotides.

The direct comparison is always de novo. C-a scores condition on an observed
right flank and therefore belong in a separate supplemental evaluation.

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
Spearman, macro mean absolute Spearman (headline), and pooled Spearman so the
paper's otherwise unspecified aggregation choice remains auditable.

Plot one or more completed results with the paper's Table 5 reference values:

```bash
python scripts/eval/dnahnet/plot_mavedb.py \
  --result BiSSM=results/dnahnet/mavedb/bissm/summary.json \
  --result Transformer=results/dnahnet/mavedb/transformer/summary.json \
  --output results/dnahnet/mavedb/mavedb_comparison.png
```

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
