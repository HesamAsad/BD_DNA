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
- [ ] Add prefix/active/suffix bidirectional backbone.
- [ ] Add single-active-block unbiased training integration.
- [ ] Add native block commit sampling integration.
- [ ] Add C-a right-flank preparation API.
- [ ] Add cache equivalence, leakage, and gradient tests.
- [ ] Add production and smoke-test configurations/scripts.

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
