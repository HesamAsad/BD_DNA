# Unidirectional SSM baseline: evidence and run plan

## Decision

Pause further bidirectional-SSM changes until two missing controls exist:

1. **uSSM-AR:** a standard causal Mamba-2 DNA language model trained with
   exact next-nucleotide negative log-likelihood.
2. **uSSM-BD:** the same causal Mamba-2 parameters trained with the existing
   BD3-LM block-diffusion loss, with no reverse active-block scan.

The first asks whether the recurrent backbone is a good DNA language model.
The second changes only the directionality of the current BiSSM and therefore
asks whether its reverse scan is useful. Comparing BiSSM-BD directly with
uSSM-AR would confound directionality and objective.

The new `UnidirectionalSSM` deliberately retains the BiSSM module and parameter
names. The current BiSSM shares one set of Mamba weights between its forward and
reverse scans, so uSSM-BD and BiSSM-BD have exactly the same parameter count.
A BiSSM checkpoint can also be passed through uSSM once as a diagnostic
"reverse-off" ablation, although that is not a substitute for training uSSM.

## What the papers change for this project

### DiffusionGemma

Reusable now:

- Treat the committed prefix as a causal, clean, timestep-independent cache and
  restrict extra denoising computation to the active canvas/block.
- Retain a causal language-modeling signal rather than assuming diffusion alone
  will preserve the starting model's capabilities. DiffusionGemma keeps a
  causal encoder objective during downstream adaptation.
- Measure total forwards per emitted token and low-batch latency, not just
  parallel tokens predicted per denoising step.

Useful later, after the baseline is sound:

- Self-conditioning, with explicit conditioning dropout so the model remains
  usable at the first denoising step.
- Entropy-bounded token commitment and adaptive stopping.
- Sampler-specific distillation for very small denoising-step budgets.

Do not import these yet. The report itself shows that few-step sampler tuning
can trade away high-step quality and can terminate incorrectly on repetitive
low-entropy outputs. Adding it now would hide whether the backbone and objective
work.

### Partial bidirectionality paper

Reusable now:

- Its leakage contract is correct: the reusable prefix state must depend only
  on the clean prefix and must not depend on diffusion time. Any reverse scan is
  local to the active block.
- Supervise all block positions, so downstream block losses train the prefix
  recurrence. The repository already implements this all-block objective.
- Tune SSM learning rate instead of inheriting a single Transformer setting.
  The paper sweeps `5e-4` through `8e-3`; the existing project used only
  `3e-4` for BiSSM.
- Report the recurrent cache cost and end-to-end generation path separately
  from whole-sequence forward throughput.

Critical missing control:

- The paper compares partially bidirectional Mamba with attention and fully
  bidirectional diffusion models, but does not report a forward-only Mamba under
  the same block-diffusion objective. It therefore does not establish that the
  reverse scan is necessary.

### Related work

- Original Mamba and Mamba-2 establish the standard starting point: a causal
  recurrent LM with fixed-size state and an efficient sequence-mode training
  implementation.
- MambaByte and dnaHNet provide direct precedent for causal autoregressive
  modeling of raw bytes/nucleotides. This supports starting with exact
  next-base likelihood before adding bidirectionality.
- DiffuMamba's diffusion variants are bidirectional. Its causal Mamba appears
  in systems-throughput comparisons, not as the matched diffusion-quality
  control needed here.
- Caduceus shows why bidirectionality and reverse-complement structure can be
  valuable for DNA representation learning. That is a later biological
  inductive-bias comparison, not a replacement for a causal generative
  baseline.

Primary references:

- [Mamba](https://arxiv.org/abs/2312.00752)
- [Mamba-2](https://arxiv.org/abs/2405.21060)
- [Official Mamba implementation](https://github.com/state-spaces/mamba)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
- [DiffuMamba](https://openreview.net/pdf/aa81eef8944d43b8edf40eef885b40c95b8903b9.pdf)
- [MambaByte](https://arxiv.org/abs/2401.13660)
- [Caduceus](https://arxiv.org/abs/2403.03234)
- [dnaHNet](https://arxiv.org/abs/2602.10603)

## Baseline matrix

| Arm | Direction | Objective | Purpose |
|---|---|---|---|
| uSSM-AR | forward only | exact next-base NLL | Standard SSM baseline; tests the backbone |
| uSSM-BD | forward only | BD3-LM NELBO, block 256 | Isolates the diffusion-objective cost |
| BiSSM-BD | forward prefix + reverse active block | BD3-LM NELBO, block 256 | Isolates the benefit of the reverse scan |
| Transformer-BD | block attention | BD3-LM NELBO, block 256 | Existing architecture reference |

The exact-NLL and diffusion-NELBO values must be labelled as such. They can be
compared as predictive coding bounds in nats/base or bits/base, but the
diffusion value is a bound/Monte Carlo estimate, not an identical estimator.

## Run order

### 0. Immediate diagnostic: no training

Evaluate the existing BiSSM checkpoint twice: normally and through uSSM with
the reverse path disabled. This reveals how dependent the trained checkpoint is
on its reverse path, but it is an out-of-distribution ablation and must not be
reported as a trained unidirectional result.

```bash
CKPT_BISSM=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/outputs/carbon-prokaryote/2026.08.03/162751/checkpoints/0-8000-v1.ckpt \
  bsub < scripts/eval/ppl_ssm_baselines.sh
```

### 1. Stability and learning-rate pilots

Run 500-step pilots for uSSM-AR and uSSM-BD at `3e-4` first. If both are finite
and learning, complete the LR grid `3e-4, 1e-3, 3e-3, 8e-3`. Keep data order,
global batch, token budget, validation batches, optimizer, and schedule fixed.
The high endpoint is intentionally only a pilot because it may diverge on this
DNA setup.

The launcher defaults to the paper's Mamba-style optimizer recipe: AdamW
`beta2=0.95`, weight decay `0.1`, gradient clipping `1.0`, cosine decay with a
10% warmup, and no EMA. These are explicit environment variables in the
launcher rather than hidden defaults. Use the identical settings for the
paired uSSM-BD/BiSSM-BD comparison; the historical BiSSM run used the
repository's older Transformer-oriented defaults.

```bash
OBJECTIVE=ar DIRECTION=uni LR=3e-4 MAX_STEPS=500 \
  bsub < scripts/train/train_dna_ssm_baseline.sh

OBJECTIVE=bd3lm DIRECTION=uni LR=3e-4 MAX_STEPS=500 \
  bsub < scripts/train/train_dna_ssm_baseline.sh
```

Rank pilot runs by validation nats/base after the same number of processed
nucleotides, not only by training loss or wall time. If rankings are close,
extend the top two settings to 2,000 steps rather than selecting from noise.

### 2. Full causal baselines

Train the best uSSM-AR and uSSM-BD settings for the existing 8,000-step,
4.19-billion-nucleotide budget. Evaluate at least 512 held-out batches for the
final table.

```bash
OBJECTIVE=ar DIRECTION=uni LR=<winner> MAX_STEPS=8000 \
  bsub < scripts/train/train_dna_ssm_baseline.sh

OBJECTIVE=bd3lm DIRECTION=uni LR=<winner> MAX_STEPS=8000 \
  bsub < scripts/train/train_dna_ssm_baseline.sh
```

### 3. Matched directionality comparison

Before concluding that unidirectional or bidirectional is better, rerun
BiSSM-BD with the same winning/tied optimizer recipe used for uSSM-BD. The old
BiSSM checkpoint used a single `3e-4` setting and a different schedule, so it is
historical context, not the final controlled comparison.

```bash
OBJECTIVE=bd3lm DIRECTION=bi LR=<matched-lr> MAX_STEPS=8000 \
  bsub < scripts/train/train_dna_ssm_baseline.sh
```

### 4. Downstream and systems gates

Only after the held-out likelihood table is stable:

1. Re-run MaveDB with at least 32 Monte Carlo masks. The current 8-mask,
   two-seed result has low seed agreement and is not precise enough for a close
   model decision.
2. Add a genuinely long-context intervention: true prefix state versus zeroed
   and batch-shuffled prefix state. MaveDB sequences shorter than one 256-base
   block cannot test the recurrent cache.
3. Benchmark `denoise active block -> commit recurrent state -> next block`,
   including cache memory and tokens/second. Whole-sequence forward timing does
   not measure the claimed streaming advantage.
4. Consider self-conditioning or an entropy-adaptive sampler only if uSSM-BD
   has acceptable likelihood but generation needs fewer denoising steps.

Do not run a block-size-1 all-block sweep with the current implementation. It
would create thousands of boundary-cache launches per sequence. Optimize
chunked boundary-state extraction first if that endpoint becomes necessary.

## Interpretation rules

- **uSSM-AR good, uSSM-BD poor:** the diffusion objective/parameterization is
  the bottleneck, not the causal SSM.
- **uSSM-BD approximately equals BiSSM-BD:** the reverse scan is unnecessary;
  keep the simpler causal model.
- **BiSSM-BD clearly beats uSSM-BD under a matched recipe:** retain reverse
  active-block context, with the measured quality/latency tradeoff.
- **Both SSM arms trail the Transformer:** investigate optimization and the
  recurrent state bottleneck before adding more directionality.
- **Likelihood improves but long-prefix interventions do nothing:** the model
  is relying on local sequence statistics; do not claim long-range memory.

## Current evidence to retain as context

At the existing matched 4.19B-nucleotide run, Transformer-BD achieved validation
NLL 1.2458 versus BiSSM-BD 1.2523, while BiSSM took about 5.1 times as long in
wall-clock time. This is a small quality gap but a large implementation-cost
gap. Profiling attributes much of that cost to sequential boundary-cache kernel
launches, so scientific directionality should be settled before optimizing that
path.

## Submitted first-stage jobs

Submitted on 2026-08-06 at 02:08 BST. All jobs initially entered `PEND` because
H200 resource reservations were unavailable; this is ordinary scheduler
capacity, not a configuration error.

| Job | Arm | LR | Steps / purpose |
|---:|---|---:|---|
| 99652 | Existing BiSSM reverse-on/off evaluation | — | 512 validation batches |
| 99653 | uSSM-AR | 3e-4 | 500-step pilot |
| 99654 | uSSM-BD | 3e-4 | 500-step pilot |
| 99655 | BiSSM-BD | 3e-4 | 500-step pilot |
| 99656 | uSSM-AR | 1e-3 | 500-step pilot |
| 99657 | uSSM-BD | 1e-3 | 500-step pilot |
| 99658 | BiSSM-BD | 1e-3 | 500-step pilot |
| 99659 | uSSM-AR | 3e-3 | 500-step pilot |
| 99660 | uSSM-BD | 3e-3 | 500-step pilot |
| 99661 | BiSSM-BD | 3e-3 | 500-step pilot |
| 99662 | uSSM-AR | 8e-3 | 500-step pilot |
| 99663 | uSSM-BD | 8e-3 | 500-step pilot |
| 99664 | BiSSM-BD | 8e-3 | 500-step pilot |

Do not submit the 8,000-step jobs until these pilots identify a stable winning
learning rate for each objective. If two settings are within validation noise,
extend both to 2,000 steps before choosing.
