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

## Deliberate first-version constraints

- The first correct training objective samples one active block per sequence
  batch and multiplies its loss by the number of blocks. This is an unbiased
  estimator of the all-block objective and avoids the current Transformer's
  `[x_t; x_0]` masking trick, which cannot represent exact recurrent boundary
  states without a custom batched scan.
- The first backbone is all-Mamba. Sparse local-attention layers will be added
  only after cache/leakage tests pass, because exact attention continuation also
  requires a bounded boundary cache.
- C-a is an infilling mode. A clean right-flank cache is never used for de-novo
  perplexity or generation.
