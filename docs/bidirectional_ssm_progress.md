# Bidirectional SSM implementation progress

This file is the durable implementation ledger for the leakage-safe
bidirectional SSM work. Each completed milestone is committed separately on
`codex/bidirectional-ssm`.

## Goal

Build a block-diffusion DNA backbone with:

1. a timestep-independent forward cache over committed clean prefix tokens;
2. a reverse scan restricted to the active noisy block for de-novo generation;
3. an optional timestep-independent reverse cache over an observed clean right
   flank for C-a infilling; and
4. an exact commit API whose memory does not grow with generated sequence
   length.

The target computation at layer `l` is:

```text
left_state[l]  = scan(clean_prefix)
right_state[l] = scan(reverse(clean_suffix))        # C-a only
active_fwd     = scan(noisy_block, left_state[l])
active_rev     = reverse(scan(reverse(noisy_block), right_state[l]))
active         = active + active_fwd + active_rev
```

Clean prefix/suffix paths never receive the diffusion timestep and never see
the clean target block.

## Reused upstream work

- `kuleshov-group/bd3lms`: existing diffusion objective, tokenizer, data
  pipeline, likelihood evaluation, and semi-autoregressive block sampler.
- `state-spaces/mamba`: Mamba-2 projection/initialization conventions and the
  `mamba_chunk_scan_combined` production kernel. The local reference scan is a
  readable PyTorch implementation of the same diagonal selective recurrence,
  used for CPU tests and as an installation fallback.
- Chaturvedi et al., *Training Hybrid Block Diffusion Language Models with
  Partial Bidirectionality*: block size 256, Mamba state 64, convolution width
  4, expansion 2, and scan chunk 128 for the first model configuration.

The paper's named GitHub repository was checked on 2026-08-03 but is not
publicly accessible, so no unreleased source is assumed.

## Milestones

- [x] Preserve the previous long-range experiment stack in commit `d424822`.
- [x] Create the dedicated branch `codex/bidirectional-ssm`.
- [x] Add segment-continuable Mamba-2 mixer with immutable cache states.
- [x] Add prefix/active/suffix bidirectional backbone.
- [x] Add single-active-block unbiased training integration.
- [x] Add native block commit sampling integration.
- [x] Add C-a right-flank preparation API.
- [x] Add multi-block C-a infilling sampler with a fixed right cache.
- [x] Add cache equivalence, leakage, and gradient tests.
- [x] Add production and smoke-test configurations/scripts.
- [x] Pass fused-vs-reference and end-to-end backward smoke on an H200.
- [x] Supervise every block per training step (folded boundary caches).
- [x] Prokaryote perplexity comparison against the Transformer BD3-LM.
- [x] Add and smoke-test the protocol-aligned dnaHNet MaveDB harness.

### GPU acceptance attempts

- LSF `96521`: model was not reached. The optional `causal-conv1d` source
  package could not build because batch nodes expose the CUDA runtime without
  `nvcc`. Resolution: use Mamba-SSM 2.2.4's pure-Python/Triton installation
  (`MAMBA_SKIP_CUDA_BUILD=TRUE`) and keep the segment convolution in PyTorch.
- LSF `96524`: model was not reached. Mamba-SSM 2.2.4 hard-imported its
  optional Mamba-1 `selective_scan_cuda` extension, and the smoke script did
  not put the repository root on `PYTHONPATH`. Resolution: pin the current
  official pure-Python/Triton package by Git commit, install it with
  `--no-deps`, and export the repository path explicitly.
- LSF `96525`: **passed** on one H200. BF16 fused-vs-reference output maximum
  absolute difference was `0.0078125`; split-vs-one-shot fused continuation had
  the same maximum difference; convolution states matched exactly; the
  two-layer prefix/active/suffix backward pass completed; both input caches
  remained bitwise unchanged. The smoke ended with `BISSM_GPU_SMOKE_OK`.

## Initial production geometry

- 100.69M trainable parameters at width 768 / 12 layers.
- Block size 256, Mamba state 64, convolution width 4, expansion 2, chunk 128.
- One-direction batch-1 recurrent cache: approximately 2.40 MiB in BF16;
  de-novo uses one direction and C-a uses two. This size is independent of the
  number of committed tokens.

### Full training-stack smoke

- LSF `96529`: the actual 100.69M model completed 10 optimizer steps on Carbon
  DNA at length 1024 / block 256. Validation NLL moved from 2.6343 at global
  step 2 to 2.2724 at step 10, and a checkpoint was saved. After
  `Trainer.fit` reported `max_steps=10`, the job remained alive in teardown
  with the auto-selected 128 persistent data workers, so it was terminated to
  release the H200. The launcher now exposes and defaults to a bounded worker
  count; a separate short run verifies clean process teardown.
- LSF `96534`: teardown check with `NUM_WORKERS=4`. Two optimizer steps, a
  saved checkpoint, and the process exited on its own 77 s after start
  (`Successfully completed`), confirming the worker bound fixed the hang.

## All-block training objective

The first version sampled one active block per step and rescaled by the block
count. That is unbiased for the loss *value*, but gradient only ever reached
`1/num_blocks` of the tokens, so at the production geometry (L=8192, block
256) it was a 32:1 learning-signal handicap against the Transformer's
all-block `[x_t; x_0]` objective — a perplexity comparison would have measured
the estimator, not the backbone.

`model.active_blocks=all` (the default) now supervises every block per step:

1. `prefill_left_boundaries` scans the clean sequence once, keeping the state
   entering each block; `prefill_right_boundaries` does the same from the far
   end for C-a, and block `i`'s suffix state never includes block `i`.
2. `stack_boundary_caches` folds those per-block states into one cache of
   batch `batch * num_blocks`, matching a `[batch, L] -> [batch*num_blocks,
   block]` reshape of the noisy sequence.
3. One batched `forward_active` call denoises every block.

Cost is one clean scan plus one batched active scan instead of a half-length
prefix scan plus one block, for `num_blocks` times the supervised tokens.
`model.active_blocks=one` keeps the original estimator.

Tests cover boundary-state equivalence against per-block prefills (both
directions), folded-vs-per-block logit equality, agreement between the two
estimators on the sampled block, and a per-position leakage sweep showing that
block `i`'s de-novo logits move only for clean tokens strictly before it. The
H200 acceptance smoke asserts the same folded-vs-per-block equality under the
fused kernel in BF16.

### Prokaryote perplexity comparison

Protocol — everything except the backbone is held equal: `carbon-prokaryote`
shard 1 capped at 400k rows (8.09B train / 76.9M validation nucleotides, built
once by `scripts/eval/pregen_prok_caches.sh` and read by both arms), L=8192,
block 256, global batch 64 (524k nt per optimizer step), 8000 steps (4.19B nt,
about half an epoch), dropout 0, lr 3e-4 constant-warmup, seed 1, and the same
1024-sequence validation subset every 500 steps.

Sizing sweep on one H200 (`scripts/smoke/smoke_prok_arms.sh`, LSF `96596`),
peak memory and steady-state step time at L=8192:

| arm | micro batch | peak | step | throughput |
|---|---|---|---|---|
| BiSSM (all blocks) | 2 | 39.8 GiB | ~2.0 s | ~8.2k nt/s |
| BiSSM (all blocks) | 4 | 75.1 GiB | ~2.2 s | ~15k nt/s |
| BiSSM (all blocks) | 8 | OOM | — | — |
| Transformer | 4 | 41.6 GiB | ~0.6 s | ~55k nt/s |
| Transformer | 8 | 79.1 GiB | ~0.8 s | ~82k nt/s |

BiSSM step time is flat from micro batch 2 to 4, i.e. the all-block path is
launch-bound on the `num_blocks x n_layers` sequence of per-block scans, not
compute-bound. Collapsing that into a per-layer chunked state-passing scan is
the obvious next optimization; at present the SSM arm costs about 3.7x the
Transformer's wall-clock per nucleotide.

Runs: LSF `96602` (BiSSM, micro batch 4) and `96604` (Transformer control,
micro batch 8), both 4xH200, both completed 8000 steps and exited cleanly.

#### Result

`scripts/eval/ppl_prok_compare.sh` (LSF `98003`) scored both final
checkpoints on the same 512 validation batches -- 16.8M held-out nucleotides,
identical data for both arms:

| arm | val NLL (nats/nt) | perplexity | bits/nt | wall clock |
|---|---|---|---|---|
| BiSSM | 1.25232 | 3.4985 | 1.8067 | 20 h 15 m |
| Transformer BD3-LM | 1.24577 | 3.4756 | 1.7973 | 3 h 59 m |

At matched tokens the Transformer is ahead by 0.0066 nats/nt (0.0094 bits/nt,
0.66% perplexity). Both are far below the ln(4) = 1.3863 uniform-DNA baseline.
Training-time validation agreed: the arms bottomed out at 1.25490 and 1.24636
respectively at step 7500.

Read this as a likelihood result only. The SSM is close but not ahead, and it
paid 5.1x the wall clock for that, so nothing here yet justifies the recurrent
backbone on perplexity alone -- the case for it has to come from the O(1)
commit cache and the C-a infilling mode, which the Transformer cannot do at
bounded memory.

Checkpoint bookkeeping: both jobs were submitted in the same second, so Hydra
resolved both to `outputs/carbon-prokaryote/2026.08.03/162751` and interleaved
their checkpoints. Plain names are the Transformer, `-v1` names are the BiSSM;
ownership is verifiable from the state dict (`blocks.0.attn` vs
`mixer.in_proj`). The launchers now stamp the LSF job id into `hydra.run.dir`,
so this cannot recur.

#### Known cost: the all-block path is launch-bound

Steady state was 0.44 it/s from the first hour to the last, i.e. no
degradation over 20 h; the SSM arm is simply 3.7x the Transformer's cost per
nucleotide. `_boundary_caches` walks the clean sequence block by block and
runs all 12 layers at each step, so a step issues `12 x 32 = 384` sequential
scan calls, each over only 4 x 256 tokens. Two independent signs of
launch-bound behaviour: step time rose only 10% (2.0 s to 2.2 s) when the
micro batch doubled from 2 to 4, and a FLOP count puts roughly 0.12 s of
arithmetic inside a 2.27 s step (about 5% arithmetic efficiency, consistent
with the ~20% kernel-busy figure in wandb).

The fix is chunked state passing: compute every block's local scan in one
folded batched call, carry boundary states across blocks with a cheap
elementwise recurrence, and apply the standard output correction. That is
about 2 kernel calls per layer instead of 32 -- roughly 24 launches per step
rather than 384. This is the same restructuring Mamba-2's chunk scan performs
internally, applied one level up.

### dnaHNet benchmark alignment

The first downstream benchmark is pinned to twelve historical MaveDB E. coli
K-12 combined-score sets. Their counts sum exactly to the 21,250 variants
reported by dnaHNet; a thirteenth matching set is now present in the live API,
so the committed manifest prevents dataset drift. All 21,250 CC0 rows, target
hashes, finite scores and coding-HGVS substitutions/indels validate. H200 jobs
`98848` (BiSSM) and `98849` (Transformer) passed an eight-variant end-to-end
smoke: both loaded the step-8000 EMA weights at a rebuilt 256-nt model length,
produced finite paired WT/mutant NELBO differences, returned exactly zero for
the identity variant, wrote predictions, and aggregated per-assay Spearman.

The full benchmark uses common diffusion times and corruption masks within
each WT/mutant pair and reports the macro mean absolute Spearman across the
twelve assays. It is protocol-aligned, not data- or compute-matched: dnaHNet
uses exact autoregressive likelihood and substantially more pretraining data.
The preparation, scorer, plotter and interpretation contract are documented in
`docs/dnahnet_benchmark_plan.md`.

#### First MaveDB result

Full H200 evaluations completed for BiSSM (`98850`, `98856`) and the matched
BD3-LM Transformer (`98851`, `98857`). Each run scores all 21,250 variants with
eight paired NELBO samples; the reported ensemble averages each variant's two
independent-seed scores before computing Spearman:

| arm | seed 1 macro | seed 2 macro | ensemble macro | ensemble pooled | seed score agreement |
|---|---:|---:|---:|---:|---:|
| BiSSM | 0.0851 | 0.0947 | **0.1035** | 0.1739 | 0.4794 |
| Transformer BD3-LM | 0.1030 | 0.1161 | **0.1239** | 0.0455 | 0.4784 |

The Transformer leads the BiSSM by 0.0204 macro absolute Spearman on this
evaluation. Both independent seeds give the same ordering. The raw per-variant
seed agreement is nevertheless modest, so these MC-8 numbers are preliminary;
raise the Monte Carlo count before interpreting smaller changes. Macro is the
headline because pooled correlation is strongly affected by between-assay
score offsets and composition.

For scale, dnaHNet Table 5 reports 0.3266 for dnaHNet, 0.3110 for
StripedHyena2, and 0.1555 for its Transformer at the largest listed compute.
Those are reference bars, not matched controls: our checkpoints saw 4.19B
nucleotides rather than dnaHNet's reported 144B, and paired diffusion NELBO is
not exact autoregressive likelihood. These MaveDB targets are at most 216 nt
and fit in one 256-nt block, so the result exercises within-block modelling but
not the recurrent prefix cache or C-a right-flank path.

Machine-readable outputs and the inspected figure are under the ignored
`results/dnahnet/mavedb/` tree; the figure is
`mavedb_comparison.png`.

#### dnaHNet-style forward scaling

The paper's Appendix A.5 protocol was adapted directly: batch-one BF16 forward
passes on one H200, powers-of-two context lengths from 1K through 512K, one
warm-up and the median of three measured calls. Here a forward is a fixed
`t=0.5` diffusion likelihood evaluation, so the metric family and length sweep
match dnaHNet while the objective and H200 hardware do not.

LSF `98871` completed every BiSSM length. Transformer `98876` completed through
256K; at 512K its internal `[x_t; x_0]` working sequence reaches 1,048,576 and
TorchInductor flex attention requires a stride above 32-bit range. Triton 2.7.1
does not implement 64-bit indexing for that template, so this is a cleanly
recorded backend limit rather than an H200 OOM. LSF `98872` first exposed the
exception and `98875` verified its classification before the final full rerun.

| context | BiSSM nt/s | Transformer nt/s | BiSSM peak | Transformer peak |
|---:|---:|---:|---:|---:|
| 1K | 15.0k | 97.1k | 0.66 GiB | 0.61 GiB |
| 8K | 18.4k | 291.9k | 1.18 GiB | 1.05 GiB |
| 64K | 18.2k | 118.9k | 5.35 GiB | 4.60 GiB |
| 128K | 17.9k | 68.1k | 10.11 GiB | 8.68 GiB |
| 256K | 18.1k | 35.4k | 19.64 GiB | 16.93 GiB |
| 512K | 18.0k | backend limit | 38.70 GiB | — |

BiSSM throughput is essentially length-independent, but this unoptimized
all-block likelihood path is still 2.0x slower at 256K and much slower at short
contexts. Both arms use O(length) memory for a full forward. BiSSM must retain
every per-block boundary state to score all blocks simultaneously; its O(1)
state claim applies instead to incremental block generation, where only the
latest cache is kept. The next systems experiment should therefore measure
`denoise_block + commit` separately rather than presenting this full-forward
curve as a constant-memory result.

Machine-readable profiles and the inspected triptych are under the ignored
`results/dnahnet/forward/` tree; the figure is `forward_scaling.png`.

## Deliberate first-version constraints
- The first backbone is all-Mamba. Sparse local-attention layers will be added
  only after cache/leakage tests pass, because exact attention continuation also
  requires a bounded boundary cache.
- C-a is an infilling mode. A clean right-flank cache is never used for de-novo
  perplexity or generation.
