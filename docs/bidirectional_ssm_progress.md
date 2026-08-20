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

#### Fixed: the boundary prefill is now layer-major

Implemented 2026-08-09. `_boundary_caches` no longer loops over blocks. Each
layer consumes the whole clean prefix in two well-shaped calls
(`SegmentMamba2.scan_with_block_boundaries`, `models/mamba2_segment.py`):

1. one full-length causal convolution from a zero state, after which each
   block's convolution boundary state is a strided `unfold` of the retained raw
   input history rather than a separate per-block convolution;
2. one full-length scan for the true layer outputs;
3. one *folded* `[batch * nblocks, block]` scan giving every block's local final
   state, combined across blocks by `_block_state_passing` -- the standard SSD
   inter-chunk recurrence `S[i] = decay * S[i-1] + local[i-1]`, unrolled into a
   single masked matmul so the cross-block carry costs one kernel rather than a
   per-block Python loop. The decay exponent is a difference of prefix sums of
   `dt` with `A < 0`, so every term the mask keeps is non-positive; the state
   passing runs under `autocast(enabled=False)` because bf16 there would
   quantise the caches every block's denoiser is conditioned on.

Layer *l*'s boundary states depend only on layer *l-1*'s output over the prefix
and on layer *l*'s own recurrence, so nothing couples the layers within a block
and the block loop was never necessary. Call counts per micro-batch at
L=8192/block 256/12 layers: **372 -> 24 scans and 372 -> 12 convolutions.**

The block-major implementation is retained as `_boundary_caches_sequential` and
is the oracle the unit tests compare against; `model.boundary_impl`
(`layer_major` default, `block_major`) selects between them at runtime as a
rollback switch.

Equivalence, on the kernel training actually uses (fused Mamba-2, BF16
autocast, 100.69M params, one H200, LSF `102784`):

| micro batch | loss rel_diff | cache.conv | cache.ssm | worst grad rel |
|---|---:|---:|---:|---:|
| 4 | 6.2e-5 | 0 (exact) | 3.7e-7 | 2.5e-2 |
| 8 | 1.4e-5 | 0 (exact) | 3.7e-7 | 2.9e-2 |

The gradient figure is bf16 reassociation noise on depthwise-convolution
weights, not drift: the same comparison in fp32 on CPU agrees to 3e-4, and the
unit-test oracle matches every parameter at 2e-5. The pre-existing GPU
acceptance smoke still reports max_abs_diff 0 for the folded-vs-per-block
logits and both caches, so the leakage guarantee is unchanged.

Speed and memory, same job:

| micro batch | path | fwd+bwd | peak | speedup |
|---|---|---:|---:|---:|
| 4 | block-major | 1.9531 s | 66.93 GiB | 1.00x |
| 4 | layer-major | 0.5145 s | 69.07 GiB | **3.80x** |
| 8 | block-major | 2.2539 s | 133.17 GiB | 1.00x |
| 8 | layer-major | 0.9812 s | 137.45 GiB | **2.30x** |

The batch-4 vs batch-8 split confirms the original diagnosis. Per sequence the
old path goes 0.488 -> 0.282 s as the batch doubles (sublinear: bigger tiles
fill the GPU it was leaving idle), while the new path goes 0.129 -> 0.123 s
(flat: compute-bound). The win is therefore largest exactly at the production
micro batch of 4.

Two things this did **not** deliver. Peak memory rose slightly rather than
falling -- the fp32 `[batch, nblocks, heads, headdim, dstate]` state tensors and
the full-length activations cost about what the 372 saved scan graphs did -- so
micro batch 8 remains off the table for real AdamW training (137.45 of 143.77
GiB with no optimizer state). And the folded local-state scan duplicates the
scan arithmetic, so the prefill now does about 2x the scan FLOPs to remove
30/31 of the launches; that trade is overwhelmingly favourable at the ~2%
arithmetic efficiency measured above, but it shrinks as the block count falls
and would be roughly neutral at `block_size` 1024.

#### Measured in situ: 3.74x end-to-end

LSF `102790` trained the same model twice for 200 optimizer steps, identical
seed and data order, differing only in `model.boundary_impl`. Steady-state
median throughput (steps >= 50, `trainer/tokens_per_s`, 4xH200):

| path | tokens/s | s per micro-batch | 8000 steps, train only |
|---|---:|---:|---:|
| block-major | 56,010 | 2.340 | 20.80 h |
| layer-major | **209,260** | **0.626** | **5.57 h** |

**3.74x**, against a pre-run extrapolation of 2.0-2.3x. That extrapolation was
wrong because it charged 0.611 s per micro-batch to non-model overhead, derived
by subtracting the standalone benchmark's 1.953 s from the observed 2.564 s
step. The real in-training model cost is 2.34 s, so the true non-model residual
is only ~0.22 s: **the pipeline is model-bound, and the end-to-end speedup
tracks the model-compute speedup almost exactly** (3.74x vs 3.80x). Adding
validation at the recorded protocol's cadence, the 8000-step prokaryote run
should take **~6 h instead of 22h35m**, putting the BiSSM arm within ~1.5x of
the Transformer BD3-LM control's 3h59m rather than 5.8x.

Loss equivalence in situ, same job (`scripts/smoke/compare_ab_curves.py`):

| metric | last (layer) | last (block) | final rel | mean signed rel | max rel |
|---|---:|---:|---:|---:|---:|
| `val/nll` | 1.335450 | 1.335254 | 1.5e-4 | +8.1e-4 | 5.4e-3 |
| `trainer/loss` | 1.308125 | 1.308084 | 3.2e-5 | +2.0e-3 | 2.5e-2 |

Note the max pointwise gap (5.4e-3 on `val/nll`) is much larger than the final
gap. That is expected and is **not** evidence of a defect: identical seeds plus
a floating-point association difference perturb the weights, which perturbs
later losses, so the trajectories separate and then reconverge. The signed-gap
pattern is mixed (`-++---++` for `val/nll`), i.e. unbiased, and the endpoints
agree to 1.5e-4. A genuine equivalence break would show a one-sided pattern and
endpoints that stay apart. The comparison script therefore gates on the final
gap and the mean signed gap, not on the max pointwise gap, which is the wrong
statistic for a chaotic trajectory pair.

Harnesses: `scripts/smoke/bench_boundary_caches.{py,sh}` (equivalence + speed on
GPU) and `scripts/smoke/ab_boundary_impl.sh` with
`scripts/smoke/compare_ab_curves.py` (two short training runs differing only in
`model.boundary_impl`, curves diffed from the CSVLogger output).

### Matched prokaryote comparison (supersedes the table above)

The comparison recorded earlier is withdrawn. The Transformer arm differed from
the SSM arms in three ways at once, each larger than the 0.0015 nats/nt it was
being compared on:

| confound | size | evidence |
|---|---|---|
| backbone forward precision | **0.116** nats | same checkpoint, FP32 1.24577 vs bf16 1.36194 (LSF 103283 / 103280) |
| EMA | 0.9999 vs none | `96604` carries EMA state, `100570` does not |
| optimizer recipe | 0.005 nats | the recipe swap alone moved BiSSM 1.25232 -> 1.24725 |

A fourth applied only to the AR arms: `Diffusion.forward`'s FP32 autocast is not
symmetric, because `models/dit.py` re-opens bf16 around its own blocks while the
SSM stack had no such re-entry. `algo=ar backbone=ussm` therefore ran entirely in
FP32, at the H200's ~67 TFLOP/s ceiling. Fixed by mirroring that re-entry
(`BidirectionalSSM._compute_autocast`); a no-op for the block-diffusion arms,
which already enter through a bf16 context.

Every arm below is now scored on raw (non-EMA) weights from the fixed-budget
`0-8000` checkpoint, each backbone at its own tuned recipe, 512 x 4 x 8192 nt.

| arm | val NLL | PPL | MaveDB signed rho | MaveDB abs rho | neg assays | wall clock (fit / train-only) | LSF |
|---|---:|---:|---:|---:|---:|---:|---|
| uSSM-AR | **1.19305** | 3.2971 | **+0.26410** | 0.29781 | 2/12 | 2h33m / 2h17m | 105320 |
| Transformer-AR | 1.19864 | 3.3156 | +0.18803 | 0.22017 | 2/12 | 2h11m / -- | 20260808-v1 |
| Transformer-BD | **1.24653** | 3.4782 | **+0.09056** | 0.14123 | 6/12 | 3h37m / 3h26m | 103661 |
| BiSSM-BD | 1.24749 | 3.4816 | +0.08519 | 0.13330 | 5/12 | 5h41m / 4h53m | 103297 |
| uSSM-BD | 1.28691 | 3.6216 | +0.07089 | 0.10891 | 3/12 | 4h19m / 3h44m | 103298 |
| BiSSM, reverse OFF | 1.32824 | -- | +0.04760 | 0.09183 | 4/12 | -- | ablation |
| *trivial baseline*: variant-event count | -- | -- | *+0.30931* | *0.30931* | 0/12 | -- | zero parameters |
| dnaHNet (published) | -- | -- | -- | *0.3266* | -- | -- | 6.4e19 FLOPs |

**Signed rho is the honest metric.** `mavedb.py:278` takes `abs()` per assay before
averaging, which credits anti-correlation as skill. All 12 assays provably share one
direction (BLOSUM62 exchangeability positive in all 12, z = +3.9 to +10.4), so the sign
should not be discarded. Absolute inflates the BD arms ~1.5x and the AR arms only 1.13x,
so the AR-over-BD gap is WIDER than the abs column suggests.

**A zero-parameter feature beats every model we have.** Counting variant events parsed
from `hgvs_pro` scores 0.30931 -- above uSSM-AR's 0.29781, below dnaHNet's 0.3266. The
benchmark is substantially measuring how many mutations a variant carries. Any MaveDB
number must be reported next to this baseline. (An earlier internal figure of 0.3649 was
a mis-parse: the regex `[A-Z][a-z]{2}\d+` produces only three distinct values across all
21,250 variants -- a variant-class code, not a count. Retracted.)

**Compute differs 3.1x across the arms; report it per arm, not as one number.** An
earlier version of this doc claimed "our compute is 2.53e18 FLOPs (6ND), 0.32x
dnaHNet's smallest budget". That is the 6ND *reference* value and it is true only of
uSSM-AR, which actually sits at 2.652e18 (0.33x of dnaHNet's 8e18). The real values run
from 2.652e18 (uSSM-AR) to 8.194e18 (Transformer-BD), and Transformer-BD therefore
*exceeds* dnaHNet's smallest budget of 8e18 while scoring 0.14123 against their 0.2601.
The compute-efficiency claim belongs to uSSM-AR alone: 0.29781 at a third of that
budget. uSSM-AR is also the only arm using dnaHNet's exact estimator, so it is the only
directly comparable row.

**Pseudo-likelihood scoring does not rescue the BD arms.** Deterministic, exact per term,
no Monte Carlo (`score_mavedb.py --score-mode pll`): uSSM-BD 0.12801 (+0.019 vs NELBO),
Transformer-BD 0.13880 (-0.002), BiSSM-BD 0.12915 (-0.007). Two independent estimators
agree the BD arms sit at 0.13-0.14 while AR sits at 0.30, so the gap is the training
objective and not the measurement. BiSSM losing under PLL refutes the prediction that
two-sided conditioning would favour it; the likely cause is distribution shift, since a
single masked token in an otherwise-clean block is far into the low-noise tail.

**Wall clocks are not validation-matched.** Counted from the logs: uSSM-AR,
Transformer-AR, BiSSM-BD and uSSM-BD each ran 320 validation passes, BiSSM-Ca ran 160,
and Transformer-BD ran only 80. The odd one out is Transformer-BD, not the SSM family
(launcher defaults `VAL_EVERY` 100 vs 200, doubled again because Lightning counts
micro-batches). Use the train-only column for architecture comparison:
validation-excluded, uSSM-BD costs 1.09x the Transformer rather than 1.19x.

**`trainer/total_pflop` is architecture-blind and wrong, in a different direction per
arm.** The formula reads only n_params, L, n_layers, d, block_size and cross_attn, so it
charges a quadratic attention term to backbones that have no attention, and never sees
the extra traversals block diffusion runs on an SSM. It logged a bit-identical 4433.47
for all four SSM arms. True values, recomputed from the forward paths by
`scripts/eval/training_flops.py` (see `results/training_flops.json`):

| arm | true PFLOP | logged | error |
|---|---:|---:|---:|
| uSSM-AR | 2652 | 4433.47 | +67% |
| Transformer-AR | 4567 | 4569.40 | +0.06% |
| uSSM-BD | 5343 | 4433.47 | -17% |
| BiSSM-BD | 6571 | 4433.47 | -33% |
| BiSSM-Ca | 7916 | 4433.47 | -44% |
| Transformer-BD | 8194 | 8687.00 | +6% |

The earlier four-arm list in this doc (2652 / 5339 / 6566 / 8193) was slightly stale and
omitted Transformer-AR and BiSSM-Ca entirely. `tokens_per_s` and `total_gtokens` are
unaffected.

MaveDB is MC-32, two seeds ensembled per-variant before Spearman, matching the
recorded protocol. Every retrained checkpoint reproduces its published MaveDB
number to within 0.003 (BiSSM 0.13330 vs 0.13174, Transformer 0.14123 vs
0.13907, uSSM-BD 0.10891 vs 0.11145, uSSM-AR 0.29781 vs 0.29870), so none of
the four confounds touched variant effect -- they were a likelihood-measurement
problem. Note MC count matters more than the fixes did: at MC-8 uSSM-BD scored
0.069 against 0.095 at MC-32.

Two results follow. Within block diffusion the Transformer leads BiSSM by
**0.00096** nats/nt, not the 0.0066 previously reported -- seven times smaller,
and inside plausible seed variance. Within AR, uSSM-AR leads the Transformer by
**0.00560** nats/nt; that is a conservative floor, because Transformer-AR's
number came from `best.ckpt`, an early-stopped selection, while every other row
uses the fixed-budget endpoint. Re-scoring from `0-8000.ckpt`
returned 1.1986446380615234, identical to 16 digits: for that run best
validation fell on the final step, so the two checkpoints are the same weights
(`global_step` 8000, matching tensor hash) and no correction applies.

Precision sensitivity turns out to be a property of *which sub-computation* runs
reduced, not of the model: the SSM tolerates bf16 in its layer stack for 0.00104
nats (1.19201 -> 1.19305), while the DiT fails to train at all without FP32 in
its embedding and logit tail.

### Block size and the reverse scan

Two ablations on the same BiSSM checkpoint, MC-32 -> MC-128 scoring, two seeds
each. Block sizes were retrained from scratch (LSF `107083` block 128, `107084`
block 512; block 256 is `103297`); the reverse-scan arm reuses the block-256
weights via `score_mavedb.py --reverse-off`, which rebuilds the backbone as
`ussm` so the in-block reverse scan never runs. `UnidirectionalSSM` subclasses
`BidirectionalSSM` and the two directions share one `SegmentMamba2`, so every
parameter name matches and the weights load unchanged.

MaveDB is scored at matched `model_length 512` across block sizes, because
`score_mavedb.py` requires `model_length % block_size == 0` and block 512 cannot
be scored at 256. Targets are 132-216 nt, so the remainder is N padding excluded
from the loss.

| variant | val NLL | MaveDB s1 | s2 | mean | vs block 256 |
|---|---:|---:|---:|---:|---:|
| block 128 | **1.24637** | 0.13479 | 0.12702 | 0.13090 | -0.003 |
| block 256 (baseline) | 1.24749 | 0.13765 | 0.13044 | 0.13404 | -- |
| block 512 | 1.24881 | 0.13972 | 0.13132 | 0.13552 | +0.002 |
| reverse scan OFF | 1.32824 | 0.09183 | 0.09483 | **0.09333** | **-0.043** |

**Granularity does nothing; existence does a lot.** Across a 4x range of block
sizes MaveDB moves 0.0046, smaller than the 0.007-0.008 seed spread within each
arm -- the sweep cannot distinguish them, so block size is not a lever for this
benchmark. Disabling the reverse scan on the same weights costs 0.0427, a 31%
relative drop and roughly 10x both the block-size effect and the seed noise.

The two are consistent: the reverse scan runs inside every block whatever its
size, so a 216-nt variant always sits within bidirectionally-modelled windows
and moving the boundaries only relocates them. Block size changes how that
context is partitioned; reverse-off removes it.

Under signed rho the ablation is larger still: +0.08519 -> +0.04760, a **44%** drop
rather than 31%.

Caveat on magnitude: reverse-off (0.09333) is *worse* than uSSM-BD (0.10891),
which was trained unidirectional from the start -- the BiSSM weights expect a
pathway that is then missing. "Removing the reverse scan at inference costs 31%"
is the safe claim; against a fair unidirectional baseline the architectural gain
is +25% (0.13600 vs 0.10891).

The NLL ordering across block sizes (128 < 256 < 512) is the block-diffusion
bound tightening as blocks shrink toward exact AR likelihood, not the model
improving, and it is bought with proportionally less generation parallelism.

MC count was also swept on the block-256 checkpoint: 0.0966 (MC-8), 0.1242 (32),
0.1318 (64), 0.1360 (128), i.e. +0.028 / +0.008 / +0.004 per doubling. MC-128 is
about the point of diminishing returns. The effect is not SSM-specific --
Transformer-BD moves 0.1341 -> 0.1417 over the same range -- so it is a property
of the diffusion NELBO estimator and changes no ordering.

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

## Long context: the crossover, measured then trained (2026-08-18)

The SSM arms do 65% of Transformer-BD's FLOPs and take 109% of its wall clock,
and they train at micro batch 4 where the Transformer runs 8. The natural
reading is that we underfill the GPU. That reading is wrong, and the real
answer is length, not batch.

### Batch size is not the lever, for either architecture

`scripts/smoke/sizing_sweep.py`, one H200, real fwd+bwd+AdamW steps, L=8192:

| arm | micro batch | clean prefill | peak GiB | nt/s |
|---|---:|---|---:|---:|
| BiSSM | 4 | stored | 70.14 | **68,459** |
| BiSSM | 4 | recomputed | 45.07 | 61,346 |
| BiSSM | 8 | stored | OOM | -- |
| BiSSM | 8 | recomputed | 88.67 | 64,350 |
| Transformer | 4 | n/a | 37.47 | **86,870** |
| Transformer | 8 | n/a | 73.52 | 78,299 |

Both are SLOWER per nucleotide at batch 8, and the Transformer loses 10% while
using half the card, so this is not a memory ceiling. A kernel that responds
this weakly to doubled parallelism is limited by arithmetic intensity, not
occupancy -- which is what the SSD scan's [128,64]x[64,64] tiles predict.

**Actionable:** Transformer-BD trains at micro batch 8. Micro batch 4 with
accum 4 is identical math and about 11% faster. The sweep runs an optimizer
step per micro batch, which charges batch 4 *more* optimizer cost per token,
so the real gain is at least that.

`model.checkpoint_boundary_prefill` recomputes the clean prefill in backward:
-36% peak memory (70.14 -> 45.07 GiB), ~10% throughput cost, loss and every
gradient unchanged (tests in `tests/test_bissm_diffusion_integration.py`). It
does not pay for itself at 8192 and is off by default. It is REQUIRED past
~16384: at 32768 micro batch 2 the stored path needs ~138 of 139.72 GiB.

### The crossover is at L ~= 14,900 nt (superseded value: 18,600)

Micro batch 2, one H200:

| L | BiSSM nt/s | Transformer nt/s | ratio |
|---:|---:|---:|---|
| 8,192 | 52,859 | 82,510 | Transformer 1.56x |
| 16,384 | 60,423 | 66,245 | Transformer 1.10x |
| 32,768 | **62,917** | 45,293 | **BiSSM 1.39x** |

The Transformer's advantage shrinks 0.68x per doubling. BiSSM gets *faster*
per nucleotide as context grows; the Transformer gets slower. Memory does NOT
cross over (88.72 vs 73.98 GiB at 32768, both ~linear, since flex attention is
already memory-linear) -- the SSM memory win is only for the generation cache.

### Trained at 32,768: speed confirmed, quality not

Same source shard and row cap as the 8192 caches, rechunked, so both lengths
see identical nucleotides. Global batch 64 (each arm keeps its tuned LR),
2000 steps = 4.1943e9 nt, matching the 8192 budget. 4xH200 exclusive.

| arm | val NLL | wall clock | last improvement | LSF |
|---|---:|---:|---:|---|
| Transformer-BD | **1.25180** | 6.94 h | step 1950 | 112645 |
| BiSSM-BD | 1.25672 | **5.03 h** | step 1750 | 112870 |

**Speed: predicted 1.39x, measured 1.38x.** A single-GPU microbenchmark held to
within 1% on a seven-hour 4-GPU DDP job. The crossover survives DDP.

**Quality: no.** BiSSM is 0.00492 nats behind, and the gap GREW with context --
it was 0.00096 at 8192, so ~5x worse at 32768. Longer context does not make
recurrence the better DNA model on this data.

A mid-run report of a "steady 0.007 lead" for BiSSM was a log-reading error
(misaligned step numbers). Corrected: BiSSM converges faster over the first
~200 steps, then matches, then slowly falls behind.

**Do not compare these NLLs to the 8192 rows** -- 2000 steps against 8000.
Both arms took that hit equally, so arm-vs-arm is sound; level-vs-level is not.

**Net:** past ~14,900 nt the SSM trains faster, for 0.005 nats. That is a speed
claim with a measured price, not a modelling claim.

**The crossover moved twice as the implementation improved, so quote the last
value.** 18,557 was measured before any of the memory work; 13,137 came from
three points on a partially-fixed tree; **14,896** is the current figure, from
five points spanning 2k-32k on the tree at commit 5dad03c
(scripts/eval/scaling_curves.py, results/figures/). Measured ratios,
Transformer-BD over BiSSM at micro batch 2:

| L | ratio | ahead |
|---:|---:|---|
| 2,048 | 5.27x | Transformer |
| 4,096 | 2.76x | Transformer |
| 8,192 | 1.34x | Transformer |
| 16,384 | 0.90x | **BiSSM** |
| 32,768 | 0.60x | **BiSSM** |

The 1.38x figure from the 32k training run stands as a *training* measurement at
that one length; the scaling sweep above is the microbenchmark it was predicted
from, re-run on the current code.

**The three scalings tell different stories and must not be conflated.** FLOPs:
attention 3.5x per doubling, the scan 2.0x -- a 10x gap by 131k. Throughput: the
crossover sits at ~14,900, far from where FLOPs alone predict, because a Mamba-2
scan feeds tensor cores far worse than a flash-attention matmul; at 8,192 the SSM
does 21% FEWER FLOPs yet runs 1.34x slower. Memory: BOTH families are linear
(~2x per doubling), because flash attention never materialises the L x L score
matrix -- the SSM sits above the Transformer at every length and no length
reverses it.
