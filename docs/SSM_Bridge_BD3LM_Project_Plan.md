# SSM-Bridge BD3-LM for Long-Context DNA

## Staged experimental roadmap — v2 (post-review)

Project plan based on the current dual-stream BD3-LM codebase and results | Revised 20 July 2026

> **Executive decision (v2)**
> Do **not** implement the SSM-Bridge next. Two rounds of adversarial review converged on the same conclusion: the v1 roadmap (a) promoted SSM-Bridge from a hypothesis to the presumed solution, (b) confounded architecture, task, and objective changes in a single leap, and (c) blurred long-range *de novo generation* against two-flank *infilling*. SSM-Bridge is therefore **demoted from the default plan to a contingent solution**. First run a small, staged set of experiments that separately answer whether real DNA carries usable distal signal, whether the current architecture can exploit it, whether the current objective rewards using it, and whether recurrence is actually required. Build new architecture only where a gate demands it, and never change more than one variable at a time.

> **What changed from v1.** The "Final architecture choice" is gone. The **five hypotheses** in §1.1 are now the spine of the plan; the architecture is an *output* of the decision rule in §10, not an input. The project is explicitly split into two tracks (§1.2) that must not be mixed. The headline route-use metric is no longer aggregate ΔBPB but distance-resolved interior effects (§6.2). SSM-Bridge remains fully specified (§3) so the endpoint is concrete — but it is gated, not scheduled.

## 1. Overall conclusion

The engineering arc is successful: the implementation trains at very long sequence lengths, avoids target leakage, uses compiled sparse attention correctly, and no longer collapses because of dead attention projections. The remaining problem is scientific rather than infrastructural: the model has *access* to a long-range route, but nothing yet shows that the route carries value, that it is incentivised, or that it — rather than the objective or the data — is the bottleneck.

The gate trajectories on prokaryotic **and** human DNA (the cross route stays flat or decays while local self-attention and MLP routes strengthen) are consistent with **at least three distinct causes**, and the current evidence cannot separate them:

1. real distal signal is weak for the current generative objective;
2. useful signal exists, but the objective does not reward using it;
3. useful signal exists, but the coarse representation or cross-attention route cannot transmit it effectively.

Gate magnitude is telemetry, not proof. The correct next move is to **disentangle** these causes with cheap, targeted experiments — not to assume the most fixable one (cause 2) and build an elaborate architecture for it.

### 1.1 The five hypotheses to disentangle

These are currently entangled; every stage below maps to exactly one.

| Hypothesis | Question | Answered by |
| --- | --- | --- |
| **H_signal** | Does real DNA contain usable distal information *for this objective*? | Stage 2 (real-data oracles) |
| **H_capacity** | Can the current architecture exploit distal information when it is present by construction? | Stage 1 (synthetic echo) |
| **H_incentive** | Does the current loss encourage the long-range route to be used? | Stage 3 (incentive test on `dit_dual`) |
| **H_architecture** | Is the current k-mer/cross-attention route itself the bottleneck? | Stage 4 (stronger stateless encoder) |
| **H_systems** | Is an SSM required for constant-memory long-*output* generation? | Stage 5 (contingent SSM) |

No architecture is built before the hypothesis it addresses has been tested. This ordering is the entire point of v2.

### 1.2 Two tracks — do not mix them

The single most consequential clarification from review: long-range **de novo generation** and two-flank **infilling** are different problems, with different mechanisms, benchmarks, and defensible claims. The v1 plan used infilling's clean causal test (the observed right flank) to imply a de-novo-flavoured long-range result. That is invalid, because the reverse-flank state that makes infilling powerful **does not exist during de novo generation**.

| | **Track D — de novo generation** | **Track I — infilling** |
| --- | --- | --- |
| Target | `p(x) = ∏ₘ p(xₘ ∣ x₍<m₎)` | `p(G ∣ L, R)`, flanks observed |
| Downstream context | none — the future does not exist | real, leak-free right flank `R` |
| Central mechanisms | forward recurrent belief; persistent compressed prefix; rolling multi-block noisy canvas; active-block partial bidirectionality | right-flank encoder; reverse state; exact near-boundary window; downstream interventions; alternating refinement |
| Minimal test | **B0** — forward-state-only, K=1 | **C-a** — two-flank conditioning, K=1 |
| Is C-a central? | **No** (reverse flank inert de novo) | **Yes** |

**Choose a primary track in Stage 0.** Infilling may remain a diagnostic secondary task if de novo is primary. For de novo, **B0 (forward-state-only) is the more relevant minimal test than C-a**, and recent partial-bidirectional BDLM work — which restricts the reverse scan to the active block so committed-prefix states stay cacheable — is closer to the de novo problem than fixed-right-flank C-a. [9]

### 1.3 What Mechanism C-a does and does not solve

| Question | Conclusion |
| --- | --- |
| Is C-a probabilistically valid? | Yes. The gap is generated as `p(G ∣ left flank, committed gap prefix, right flank)`. The right flank is observed conditioning, so a reverse state over it is leak-free. |
| Does C-a give true downstream context? | Yes for the fixed external right flank — exact evidence about the destination boundary and distal suffix. |
| Does C-a give full within-gap bidirectionality? | No. Unknown future gap chunks are absent. C-a is two-boundary conditioning, not full within-gap bidirectionality. |
| **Does C-a transfer to de novo generation?** | **No.** The reverse right-flank state is absent de novo; replacing it with a learned null vector does not carry the mechanism over. C-a is a Track-I mechanism and a diagnostic, not a de novo solution. |
| Is one fixed downstream vector sufficient? | Probably not — a single vector is a severe retrieval bottleneck and is identical for all generated positions unless distance/position are supplied. |
| Best improvement to C-a? | Add an exact right-boundary window, multiple recurrent memory slots, distance-to-boundary conditioning, and later a bounded noisy canvas whose reverse scan is initialised from the right-flank state. |

## 2. Design principles and non-negotiable constraints

| Principle | Requirement |
| --- | --- |
| Single-nucleotide fidelity | The generative target remains A/C/G/T at single-base resolution. Chunking is a computational schedule, not a lossy tokenization scheme. |
| No clean-target leakage | A state used to predict chunk `m` may depend only on committed clean context, observed flanks, and current noisy/provisional chunks. **Precise definition:** a reverse scan over a noisy active canvas is *not* inherently leaky — it is legal if it consumes only current noisy/partially-denoised `x_t`, committed clean context, and known external flanks. Leakage occurs iff clean unknown `x₀` gap tokens enter the scan during training. |
| Train–sample state match | The recurrent states, active-buffer layout, and slot-wise noise patterns seen during training must match the sampler. This is the dominant correctness risk for multi-block training (see §8), above the leak risk. |
| Linear or bounded computation | No component may attend densely over the full sequence. Any attention path operates over a fixed local window or a fixed number of memory slots. |
| Constant sampling memory in total length | Carried recurrent states and the active canvas are fixed-size; generated sequence may be streamed to host storage. Note this delivers *long-output* generation, not automatically *long-context-retaining* generation (§10). |
| Functional pathway evidence | Every claimed long-range route must pass true/zero/shuffled state ablations and distance-resolved context interventions. Gate norms alone are telemetry, not evidence. |
| Honest oracle framing | A full-attention / Caduceus / Evo "oracle" gives a **model-dependent lower bound** on usable distal information, not a model-independent proof. A negative oracle is an early gate, not a universal proof that no long-range signal exists (§6.4). |
| Reverse-complement consistency | The architecture and evaluation should either be RC-equivariant or explicitly trained and tested for RC consistency. |
| Backward compatibility | The current `dit_dual`/`dit` paths, configs, checkpoints, and eval modes remain runnable as baselines. |

## 3. The candidate architecture (contingent): SSM-Bridge Multi-Block DNA Diffusion

> **Read this section as the *endpoint*, not the *next step*.** SSM-Bridge is documented in full so the target is concrete and the staged prompts have something to build toward. It is constructed only if the §10 decision rule reaches Stage 5. Nothing here is scheduled ahead of the gating experiments.

### 3.1 State and factorization

For a generated or infilled region divided into chunks `G_1..G_M`, maintain a forward state `H_L,m` summarising the observed left flank and committed chunks `G_<m`. For infilling, precompute a reverse state `H_R` from the fixed right flank. A bounded active canvas contains `K` noisy chunks starting at `m`; the leftmost is the next commit candidate.

> **Operational factorization.** The model approximates `p(G_{m:m+K-1} ∣ observed left, committed prefix, observed right)` with a denoiser over a fixed noisy canvas. The forward scan is initialised by `H_L,m-1`; the reverse scan by `H_R` for infilling and by a learned null state for de novo. Only the leftmost completed chunk commits; then `H_L` updates and the canvas shifts.

### 3.2 Components

1. **Chunk embedder and summary** — embed nucleotides at full resolution; pool each committed chunk to a summary vector via a learned attention pool or short local convolution (not an exponential 4^k vocabulary). *Note:* unlike the current k-mer coarse encoder (an exact bijection, leak-free by construction), a learned pool is lossy and is itself a place long-range information can bottleneck (§8).
2. **Forward committed-prefix recurrence** — Mamba-2 layers update fixed-size states once per committed chunk: the exact streaming memory of the generated past.
3. **Reverse observed-suffix recurrence** — for infilling, scan the known right flank right-to-left once; retain a global suffix state plus a small bank of projected slots.
4. **Active noisy canvas** — `K` chunks, each with its own diffusion time/noise level; fixed-size even at million-base final length.
5. **Partial-bidirectional fine mixer** — a forward scan through the canvas initialised from `H_L` and a reverse scan initialised from `H_R`, the reverse scan **restricted to the active canvas** so committed-prefix states remain exactly reusable. [9]
6. **Optional exact local path** — sparse/local attention or depthwise convolution in a small nucleotide window, for sharp motifs/copying while the recurrent path handles long-range state.
7. **Conditioning interface** — expose forward/reverse state slots through fixed-size cross-attention and/or AdaLN; include remaining-gap distance and absolute/relative chunk position.
8. **Commit and shift rule** — after `T` steps or an adaptive confidence criterion, commit the leftmost chunk, update `H_L`, shift the canvas, append a fully masked chunk.

### 3.3 Why a bounded canvas is the key addition — and its two failure modes

Mechanism C-a alone can use the real right flank but cannot see the unknown sequence between the current chunk and that flank. A `K`-chunk noisy canvas supplies provisional future variables that are legal generative state (not leaked clean targets); the reverse scan carries the right boundary through those provisional chunks, giving each active position a position-aligned downstream representation. This is the most direct way to recover bounded forward context at constant memory.

**Two failure modes must be designed against from the start:**

- **Train–sample noise mismatch (dominant).** At sampling, active blocks carry heterogeneous, history-dependent noise states. Standard one-noisy-block teacher forcing does not reproduce this. Multi-Block Diffusion introduces noise-group training with heterogeneous slot-wise schedules; Diffusion Forcing independently motivates per-element noise levels. Adapt these at chunk level. **Do not build the rolling bridge until there is a concrete matched multi-block training plan.** [2, 8]
- **Leakage (narrower than v1 implied).** The reverse scan over the canvas is leak-free provided it never consumes clean unknown `x₀` gap tokens. This is a unit-test obligation, not an architectural blocker (§2, §8).

### 3.4 Training losses

| Loss / intervention | Purpose | When to enable |
| --- | --- | --- |
| `L_diff` | Masked block-diffusion NELBO / weighted CE over noisy active chunks. | From the first model. |
| `L_use` | Margin/ranking loss requiring the true state to beat a matched shuffled or zero state on **distance-gated distal targets** (never all tokens — see §6.2). | Stage 3 onward. |
| Route dropout | Randomly drop local context, suffix state, or recurrent slots so no route monopolises optimisation. | Stage 3; tune conservatively. |
| Conditional guidance dropout | Train conditioned and null-suffix branches → explicit right-context guidance at inference. | Track I, Stage 3+. |
| `L_psr` | Contrastive multiscale prediction of future representations from the causal state. | Only after the generator demonstrably uses the state. |
| RC consistency | Match outputs under reverse-complement or use tied RC-equivariant blocks. | After the architecture is stable. |

### 3.5 Recommended initial scales (for the contingent SSM)

| Hyperparameter | Initial sweep | Rationale |
| --- | --- | --- |
| Commit chunk C | 1,024 / 2,048 / 4,096 nt | Avoid thousands of sequential commits; affordable with Mamba. |
| Active chunks K | 1 / 2 / 4 / 8 | K=1 is the B0/C-a baseline; K>1 tests provisional future context. |
| Active canvas K·C | 4–32 kb | Meaningful bounded bidirectionality without full-sequence recompute. |
| Denoising steps T | 8 / 16 | Measure quality–throughput; do not assume text schedules transfer. |
| State slots | 8 / 16 / 32 per direction | Avoid a one-vector bottleneck at constant memory. |
| Exact boundary window | 512 / 2,048 / 4,096 nt | Preserve precise junction info a compressed state may lose. |
| Attention density | none / local every layer / 1 attn per 4–6 Mamba layers | Sparse attention can complement Mamba mixing. [5] |

## 4. Decision on the original mechanisms

| Mechanism | Decision | Reason |
| --- | --- | --- |
| A: predictive / PSR belief | Keep, but late. | Anticipatory causal-state learning; does not reveal the realised future or guarantee use. Multiscale contrastive targets, not short-horizon cosine. |
| B: generated coarse plan | Defer. | Adds a latent-variable model, plan collapse, cross-scale consistency, a second sampler. The fine model could ignore it exactly as it ignored the coarse stream. |
| B-lite: sparse anchors | Optional later. | If global organisation stays weak, generate interpretable anchor patches, then infill. Easier to verify than latent tokens. |
| **C-a: fixed backward flank belief** | **Track I only; stateless prototype first.** | Exact, leak-free downstream conditioning for infilling and a clean intervention experiment — but **not** central to Track D. Test it first with a *stateless* reverse-attention encoder (Stage 4), not a recurrent SSM. |
| **B0: forward committed-prefix state** | **Elevated — Track D minimal test.** | The de-novo-relevant analogue of C-a: does a fixed recurrent summary of the committed past improve generation? This, not C-a, is the minimal de novo experiment. |
| C-b: iterative refinement | After the streaming bridge. | A draft makes within-gap future available provisionally; alternating remasking recovers acausal consistency in linear passes. |
| C-c: B plus C | Not initially. | Too many simultaneous mechanisms to diagnose. |

## 5. Staged roadmap with acceptance gates (v2)

Each stage tests one hypothesis and must pass its gate before the next. Stages 4–5 are contingent.

### Stage 0 — Freeze the scientific claims

Choose the **primary deliverable** before any modelling:

- **Option D (de novo):** demonstrate and exploit long-range causal context in arbitrary-length DNA generation.
- **Option I (infilling):** generate large missing genomic regions that respond correctly to distant left *and* right flanks.

Infilling may remain a diagnostic secondary task if de novo is primary. Store all outputs in a machine-readable schema (checkpoint, config hash, dataset hash, target/control accuracy, BPB, memory, throughput, intervention deltas). **Gate:** the benchmark and eval are deterministic across two reruns; the primary track is written down.

#### Stage 0a — MANDATORY data-integrity audit (added 2026-07-20 after a real failure)

No training run counts as evidence until its cache passes `scripts/eval/audit_human_caches.py`.
This is not boilerplate — the first audit found a **20.7% train/val leak**:

| Check | Requirement | Why |
| --- | --- | --- |
| **Contiguity** | every cached window byte-**exactly** matches the reference at its manifest `(chrom, start, +L)` | proves one contiguous interval from one chromosome; the only test that detects silent `_group_texts` concatenate-then-rechunk stitching |
| **Purity** | zero special tokens (BOS/EOS/MASK) mid-sequence; all rows exactly length `L` | concatenation tripwire |
| **Split disjointness** | 0% exact-duplicate windows between train and each val split | **found 53/256 = 20.7% (L=98304) and 50/256 = 19.5% (L=32768)** leaked via deterministic TSS-anchored windows drawn from one shared pool |
| **Train dedup** | no duplicate windows inside train | was 5.7% |
| **Loader provenance** | the run log's `Loading data from:` path is the intended pre-built cache | a filename mismatch silently triggers a rebuild down a different code path |

**Rule: partition the genome by chromosome before sampling** (val only from held-out chroms,
e.g. chr8/chr9). Disjoint TSS sets alone are insufficient — they leave partial-overlap and
paralog leakage that exact-duplicate detection cannot see. Note leakage inflates val
performance, so it can never *explain away* a negative long-range result.

### Stage 1 — Capacity control (H_capacity)

Run the planted-echo benchmark with the **existing `dit_dual`**. Measure target accuracy vs gap, true-vs-shuffled coarse context, zeroed route, gate trajectories, and prediction sensitivity **only at target positions** with matched non-target controls.

**Gate:** the synthetic control is at chance; within-block echoes are separated from cross-block; changing the distal source changes the target prediction. If the architecture cannot solve even the artificial task, fix capacity/optimisation before anything else.

#### Stage 1 HARD PRECONDITION — signal density and the within-block sanity check

Learned the expensive way (2026-07-20): the first planted-echo run used `n_echoes=6,
motif_len=32` in a 24,576-nt random background = **0.78% of positions predictable**. A perfect
copier would have cut NLL by ~0.016 bits, so there was effectively no gradient. The model sat
at the uniform floor (`val/nll` 1.384 vs `ln(4)`=1.3863) and scored **chance on every bucket,
including within-block (0.2505)**. Every downstream number that run produced was meaningless.

Two rules, checked **before** interpreting anything:

1. **Density.** Predictable positions must be a real fraction of the sequence — target
   ~15-30%, not <1%. (Fixed config: L=12288, 80 echoes × 24 nt → 15.6% of positions;
   40,960 val echo pairs vs 3,072.) The random background is irreducible noise that otherwise
   drowns the learning signal.
2. **Within-block sanity is load-bearing, not a formality.** If within-block copy accuracy is
   not clearly above 0.25, the task is unlearnable or the setup is broken — **stop and fix the
   benchmark; do not read the cross-block number, and do not read the gates.**

**Corollary for `L_use` (Stage 3):** on the dead task, `L_use` still drove `gate_cross` up
**8.8×** (0.3494 vs 0.0399 at step 4000, reaching parity with self-attention) while copy
accuracy stayed at chance. That is the margin being satisfied by *degrading the shuffled
branch* — "forced to react, not taught to use" — and it is precisely why **gate telemetry can
never be the acceptance criterion**. Always pair it with a behavioural retrieval metric.

### Stage 2 — Real-data signal existence (H_signal)

Two controlled oracle probes (§6.4), each a **model-dependent lower bound**, not a proof of absence:

- **2A. Causal-prefix oracle (Track D relevance):** does progressively longer *clean past* context reduce target-block BPB, holding the local window fixed?
- **2B. Two-flank oracle (Track I relevance):** does true distal *right* context reduce **deep-interior** gap BPB, holding the near boundary fixed?

Evaluate at `d ∈ {1, 4, 8, 16, 32, 64} kb` where feasible. A 20–40 kb full-attention oracle is a reasonable start but cannot rule out signal that only appears at ≥100 kb.

**Gate:** do **not** proceed toward any new long-range architecture if neither probe shows a statistically meaningful distal effect on real DNA. (Reframe the project to a local/efficiency story instead.)

### Stage 3 — Incentive test on the existing architecture (H_incentive)

Retrain/fine-tune **`dit_dual`** with `L = L_diff + λ_use · L_use` plus randomised route dropout, using hard matched-negative coarse contexts. **Test the causal original task first**; test infilling separately only if Track I is pursued (infilling changes the conditional distribution, so it is a *task* change, not an *objective* change — keep them factored, §6.3).

**Gate (both required):** (1) held-out BPB improves or does not regress; (2) the **specific, distance-resolved** causal intervention effect at distal targets/interior positions increases. A model that merely becomes sensitive to shuffled context while its likelihood worsens has been *forced to react*, not *taught to use* information — that fails the gate.

> **If Stage 3 passes, that may be the paper:** "long-range pathway collapse in block diffusion is primarily an objective/incentive problem; matched counterfactual training restores causal use without changing the backbone." Cleaner and arguably more interesting than another Mamba architecture.

### Stage 4 — Architecture test *without* recurrent sampling (H_architecture)

Only if Stage 2 is positive **and** Stage 3 fails. Replace the coarse representation with a stronger but **stateless** encoder, keeping the same `L_use`/dropout:

- Track I: a reverse-attention or bidirectional right-flank encoder feeding the existing cross-attention (the right flank is observed, so bidirectional encoding is leak-free).
- Track D: a larger/multiscale causal prefix encoder.

**Gate:** determines whether *representation capacity* (not recurrence) was the bottleneck. If the stateless stronger encoder recovers causal use, recurrence is unnecessary for the science — it becomes a pure systems choice.

### Stage 5 — SSM implementation (H_systems, contingent)

Build the SSM only if one of: the stateless encoder works but its **sampling memory** is unacceptable; the compressed route provably cannot transmit the real signal; or **persistent context-retaining long-output generation** is now the explicit systems goal. First SSM model is minimal — forward committed-prefix state (de novo), optional reverse fixed-flank state (infilling), **K=1, no rolling canvas, no PSR, no coarse plan**. Add the multi-block noisy canvas only after exact state-update and no-leak tests pass **and** a matched multi-block noise-training plan exists (§3.3).

**Gate:** memory flat in total generated length; runtime ~linear in canvas length; a larger commit chunk/canvas than the attention baseline at a useful quality–throughput point.

### Later, still contingent

- **Partial-bidirectional Mamba fine mixer** — enlarge the bidirectional active region without quadratic attention. [9]
- **RC symmetry** — augmentation first, tied MambaDNA/Caduceus-style equivariance only if it measurably helps. [3]
- **Alternating refinement** — one forward draft + reverse/alternating sweeps over uncertain chunks, linear passes only.
- **Multiscale PSR** — enrich the causal state's horizon *after* state use is established.
- **Sparse-anchor planning** — only if whole-region organisation remains deficient; prefer sparse anchors / Set-Diffusion-style positions to a free latent plan. [10]

## 6. Experimental program

### 6.1 Benchmark ladder

| Dataset / task | Purpose | Required variants |
| --- | --- | --- |
| Synthetic planted echo | Unambiguous long-range MI + exact copy target (H_capacity). | Left-source, right-source, cross-chunk, unseen gaps, motif lengths, matched random controls. |
| Synthetic bridge constraints | Require both flanks jointly, not one-sided copy (Track I). | Frame/parity checksum, paired motif, start/end-compatible code, two-source XOR-like target. |
| Human hg38 next-region / infilling | Real long-range composition, repeats, gene architecture (H_signal). | Matched spans by chromosome, repeat class, GC, distance, annotation. |
| Carbon prokaryote | Systems baseline and local-DNA control. | **Not** a valid sole test of long-range value (~0 MI beyond ~1 kb). |
| Regulatory element generation | Compare with bidirectional DNA diffusion quality metrics. [6] | SFID/FRED, motif content, activity predictors where valid. |
| Long-form de novo rollout | Drift, memory, stability at 98k–1M+. | K=1 vs K>1, null vs guided state, refinement on/off. |

### 6.2 Headline evidence that a context route is used

**Aggregate ΔBPB over all gap tokens is banned as a headline metric.** It is won by boundary continuity: the right flank pins the composition of the last few hundred bp of a gap — a *local* effect — so aggregate ΔBPB reads positive even with zero interior use. This is the exact dilution error that made the whole-block-shuffle test uninformative. The headline metrics are distance-resolved:

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **Interior BPB improvement** `ΔBPB_int(r)` | `BPB(shuffled distal) − BPB(true distal)`, computed only for positions with `dist(i,L) > r` and `dist(i,R) > r`, for `r ∈ {0.5,1,2,4,8,16} kb`. | A genuine long-range mechanism keeps a positive effect as `r` grows. |
| **Influence-vs-distance curve** `I(d)` | Mean `log p(x_i ∣ R_true) − log p(x_i ∣ R_counterfactual)` for positions with `dist(i,R) ∈ [d, d+Δ]`. | The *shape* is more informative than any single aggregate. |
| **Targeted effect ratio (TER)** | intervention effect on intended targets ÷ effect on matched non-target positions. | Distinguishes structured use from indiscriminate distribution shift. |
| **Matched flank swaps** | Swap `R` for a hard negative matched on near-boundary sequence, GC, repeat class, chromosome/compartment, annotation, and gap length. | Prevents winning by composition or chromosome identity. |
| **Exact local-boundary control** | Fix the first `b` bases of `R` (`R = R_near ∥ R_distal`), shuffle only `R_distal`. | Any interior effect cannot be explained by junction continuity. |
| Memory/throughput scaling | Peak memory and bases/s vs generated length and canvas size. | Verifies the systems claim independently of quality. |
| Route gradient sensitivity | Norm/Jacobian from target logits to state slots. | Diagnostic only; never a headline. |

### 6.3 Factorial ablation (architecture × task × objective)

The v1 matrix confounded all three axes in one leap. Decompose them; the boxed row is the most important and cheapest experiment in the plan:

| Architecture | Task | Objective | Tests |
| --- | --- | --- | --- |
| `dit` single-stream | causal block diffusion | original | A0 local baseline |
| `dit_dual` | causal block diffusion | original | A1 validated baseline / negative reference |
| **`dit_dual`** | **causal block diffusion** | **`+ L_use`** | incentive (partial) |
| **`dit_dual`** | **causal block diffusion** | **`+ route dropout`** | incentive (partial) |
| **`dit_dual`** | **causal block diffusion** | **`+ L_use + route dropout`** | **A1+ — incentive on current arch (Stage 3)** |
| stateless right-flank encoder | infilling | original | Track I representation baseline |
| stateless right-flank encoder | infilling | `+ L_use + dropout` | Track I representation + incentive (Stage 4) |
| SSM (later, contingent) | matched task | matched objective | systems (Stage 5) |

Keep the **task** axis (causal vs infilling) separate from the **objective** axis (`L_diff` vs `+L_use`): infilling changes `p(xₘ∣x₍<m₎) → p(G∣L,R)`, so bundling it into "A1+" would re-entangle architecture, incentive, and task. A1+ therefore tests incentive on the **causal** problem; the stateless C-a prototype tests downstream conditioning on the **infilling** problem.

### 6.4 Real-data signal oracles (Stage 2 detail)

Use **one** model with **randomised context availability during training** (context dropout), so the same parameters support the with- and without-distal-context conditions and the comparison is not confounded by two separately-trained models.

- **Oracle A — causal distal-prefix value (de novo).** `Δ_past(d) = BPB(xₘ ∣ near past) − BPB(xₘ ∣ near past, distal past extending d bases)`. Evaluate with true / shuffled-matched / zeroed distal prefix.
- **Oracle B — distal-right-context value (infilling).** `Δ_right(d,r) = BPB(G_interior ∣ L, R_near) − BPB(G_interior ∣ L, R_near, R_distal at distance d)`. Keep `R_near` identical across conditions; score only tokens `≥ r` from *both* gap boundaries.

**Honesty caveats (why this is a lower bound, not a proof):**

- A failure can mean the oracle was undertrained or mismatched to the task, not that no signal exists.
- **Caduceus** is an MLM (pseudo-likelihood) model, not an exact normalised conditional-likelihood model; its scores are not directly comparable to BD3-LM BPB, and its long-range *variant-effect* performance is supervised biological signal, not evidence of long-range *nucleotide-generative* signal. It was trained to ~131 kb. [3]
- **Evo**-style models are causal AR: they probe distal *past* value (Track D), not true downstream context (Track I).
- Therefore pretrained models are useful *screens*; the decisive probe is a controlled oracle trained for *your* objective, at several distances.

## 7. Codebase implementation map

| File / module | Planned change |
| --- | --- |
| `models/dit_dual.py`, `models/dit.py` | **Freeze** except eval hooks and the Stage 3 `L_use`/route-dropout plumbing. Do not morph into the new architecture. |
| `scripts/eval/` | Synthetic cache builder (exists); **real-data oracle harness** (Oracle A/B with context dropout, distance buckets); matched counterfactual generator; distance-resolved metric + report aggregation. |
| `diffusion.py` | Add `L_use` (distance-gated), route dropout, conditional guidance dropout, and eval-time state interventions — **without** changing the default objective or checkpoint keys. |
| `models/local_mixer.py` / stateless encoder (new) | Stage 4 stateless reverse-attention / bidirectional right-flank encoder and multiscale causal prefix encoder, feeding existing cross-attention. |
| `models/ssm_bridge.py`, `models/mamba_dna.py` (new) | **Stage 5, contingent.** Minimal forward/reverse Mamba states, state-slot projection, output contract compatible with `Diffusion`; RC wrapper later. |
| `dataloader.py` | Deterministic span-infilling examples + metadata (left/right flank, gap mask, distances); keep standard DNA loaders untouched. |
| `main.py` | Eval modes for oracle probes, state interventions, flank swaps, buffer rollout, RC consistency, consolidated result export. |
| `configs/model/*.yaml` | Separate configs per mechanism; never hide major mechanisms behind one overloaded preset. |
| `tests/` | Mask/leak tests, train–sample state equivalence, recurrence step-vs-scan equality, RC consistency, static-shape buffer, checkpoint compatibility. |

## 8. Engineering and scientific risk register

| Risk | Failure signature | Mitigation / decision |
| --- | --- | --- |
| Route atrophies again | True/zero/shuffled states give indistinguishable logits. | Test incentive on the current arch first (Stage 3); multi-slot read, `L_use`, route dropout, targeted tasks before adding recurrence. |
| **Oracle is a lower bound only** | Oracle shows ~0 distal effect. | Treat as an early gate, not proof; try longer distances and a stronger/longer oracle before concluding "no signal." |
| **Train–sample noise mismatch (dominant for K>1)** | Validation forward looks good; rollout degrades/oscillates. | Heterogeneous noise-group training + static block-buffer matching inference [2,8]; do not build the rolling bridge without this plan. |
| Clean-`x₀` leakage into reverse scan | Interior BPB improves impossibly / train-eval gap. | No-leak unit tests: mutate hidden `x₀` gap tokens, assert every conditioning tensor is unchanged. |
| One-state retrieval bottleneck | Global BPB improves but exact echo/copy fails. | Keep exact boundary windows; multiple state slots; retain sparse local attention. |
| Confounded attribution | A change "works" but architecture+task+objective all moved. | Enforce the §6.3 factorial; change one axis at a time. |
| Counterfactual shortcut | State ranking solved by GC/species/chromosome mismatch. | Hard negatives matched on composition, chromosome, repeat class, local annotation; log/reject unmatched negatives. |
| Metric won by boundary continuity | Aggregate ΔBPB positive, interior flat. | Distance-resolved interior metrics only (§6.2). |
| Novelty overlap | Architecture resembles R2LM / partial-BDLM Mamba / Set Diffusion. | Position novelty in DNA-specific two-boundary infilling, RC symmetry, million-scale state-use evidence, and biological interventions — not the generic architecture (§9). |
| Tiny commit chunks | Constant memory but enormous call count. | Sweep 1k–4k+ chunks; exploit linear Mamba to enlarge the commit region. |
| Pure Mamba misses sharp interactions | Long-range improves but motifs/copy degrade. | Sparse hybrid / bounded local attention; do not insist on attention-free purity. [4,5] |

## 9. Literature alignment and novelty boundary

Review verified the nearest work; it substantially narrows the *architectural* novelty and moves the defensible contribution to methodology and DNA specificity.

| Work | Status | Implication |
| --- | --- | --- |
| Block Diffusion [1] | — | Flexible-length block diffusion + cacheable L→R block generation; this project is a DNA-scale extension. |
| Diffusion Forcing [2] | — | Heterogeneous per-element noise; needed for the noisy active canvas. |
| Caduceus [3] | verified | Bidirectional Mamba + RC equivariance are strong genomic biases; but MLM pseudo-likelihood ≠ generative BPB (§6.4). |
| HybriDNA [4] | — | Transformer-Mamba2 hybrids for long single-nucleotide DNA; supports retaining a small exact attention path. |
| DiffuMamba [5] | — | Bidirectional Mamba as a masked-diffusion denoiser; sparse hybrid attention effective. |
| D3LM [6] | — | Full-bidirectional DNA diffusion quality reference; does not target arbitrary-length constant-memory generation. |
| **Bifocal / R2LM [7]** | **verified** | Precise causal-attention left route (+KV cache) plus a reverse-direction Mamba residual sidecar for compressed right context — **very close** to "forward precise + compressed reverse belief." C-a's generic idea is **not new**. |
| Multi-Block DLM [8] | verified | A running set of noisy blocks needs inference-matched multi-block teacher forcing + static block buffer. |
| **Partial-bidirectional BDLM-Mamba [9]** | **verified** | Restricts the reverse Mamba scan to the active denoising block for exact prefix cache reuse — **directly overlaps** the de novo active-block SSM. |
| **Set Diffusion [10]** | **verified** | Flexible-position/length token sets, any-order decoding, KV-cache updates, stronger infilling than block diffusion — a serious alternative factorization to cite. |

**What remains defensibly novel** (methodology + DNA, not architecture):

1. A rigorous demonstration that long-range pathways in genomic diffusion models collapse or are ignored despite architectural availability.
2. A signal-value protocol separating synthetic capacity, real causal-prefix value, and downstream-flank value (with honest model-dependence).
3. Matched counterfactual interventions showing whether context causally affects predictions **deep inside** a genomic gap.
4. DNA-specific, RC-consistent recurrent memory.
5. Single-nucleotide, tens-to-hundreds-of-kb infilling / streaming-generation evidence.
6. A clean finding that objective interventions are **sufficient — or insufficient —** to recruit the route.

> **Defensible novelty target.** A DNA-specific, RC-aware block-diffusion system that performs long-range **infilling and/or context-retaining generation** at single-nucleotide resolution, and that demonstrates long-range use through matched source/suffix interventions deep inside the region — rather than through perplexity or gate telemetry. The architecture is a means; the causal evidence is the contribution.

## 10. Decision rule

### 10.1 The gate

| Condition | Decision |
| --- | --- |
| Synthetic (Stage 1) fails | Debug architecture/capacity first; nothing else is interpretable. |
| Synthetic succeeds, **real oracle (Stage 2) fails** | **Do not build a generic long-range architecture.** Reframe to a local/efficiency story, or escalate oracle length/strength before concluding. |
| Real oracle succeeds, **current route + incentives (Stage 3) succeeds** | **Retain the architecture.** Contribution is training/evaluation methodology; SSM only later for scaling. |
| Real oracle succeeds, incentives fail, **stronger stateless route (Stage 4) succeeds** | Representation bottleneck. Consider SSM for **scaling/sampling memory**, not for the science. |
| Real oracle succeeds, **all routes fail** | Revisit objective, data scale, and model capacity **before** any sampler engineering. |

SSM-Bridge is built only in the fourth row (for systems reasons) or when context-retaining streaming generation becomes the explicit goal. Review did not invalidate SSM-Bridge; it correctly demoted it from default to contingent.

### 10.2 "1M generation" means two different things

Keep these distinct in every claim:

- **1M-output generation** — the sampler emits one million bases. The current code supports this comparatively well.
- **1M-context-retaining generation** — predictions near the end use information from the distant generated prefix. The current sampler does **not** support this: the semi-autoregressive sampler defaults to `context_size = 1024` and slices each forward pass to that recent window, the dual backbone truncates long-range coarse context beyond the window, and there is no KV-cache sampling.

Accurate statement: *the system has demonstrated million-base output generation, but not million-base context retention during generation.* The stateful long-context sampler is **greenfield engineering**, not an extension of a working one — cost it accordingly.

## Appendix A — staged coding-agent prompts

> **Gating.** Prompts P0–P3 are the near-term program (Stages 0–3) and are safe to start. **P4 onward are contingent** on the §10 gate — do not begin the stateless-encoder or SSM prompts until Stage 2 is positive and (for SSM) Stage 3 has failed. Each prompt must produce a reviewable patch with tests, preserve all existing backbones, inspect the repo before editing, and report exact commands and measured outputs.

### P0 — Baseline audit and synthetic long-range harness (Stage 1)

```text
Work in an existing BD3-LM repo for single-nucleotide DNA diffusion. First read and map:
models/dit_dual.py, models/dit.py, diffusion.py, dataloader.py, main.py, the dual-stream
configs, and scripts/eval/gen_synthetic_longrange.py + main.py mode=synth_copy_eval.

Task: baseline audit + reproducibility harness, no change to model behaviour.
1. Document the exact train/sample information flow for the single-stream and dit_dual backbones.
2. Build the missing synthetic planted-echo caches for a smoke config; add a deterministic full-cache command.
3. Add eval-time interventions for the current coarse route: normal / zeroed / GC-matched-shuffled
   coarse memory, and source-motif intervention for planted echoes.
4. Export one JSONL row per checkpoint/config/gap-bucket: BPB, target copy accuracy, control accuracy,
   logit KL, peak memory, bases/s. Bucket by distance and within- vs cross-block.
5. Do not modify training semantics or checkpoint keys. Add unit tests for deterministic caches and
   intervention shape/device correctness.
Acceptance: existing commands still run; synthetic control near chance; interventions produce
comparable outputs; results reproducible under a fixed seed.
```

### P1 — Real-data signal oracles (Stage 2) — NEW

```text
Implement the two real-data signal probes from the plan §6.4, using ONE model with randomised
context availability (context dropout) so the same params serve both conditions.

Oracle A (causal distal-prefix value, Track D):
- Train a model to predict a target block from a fixed local past window, with a randomly-present
  distal past extending d bases. During eval compute BPB with true / GC-matched-shuffled / zeroed
  distal prefix. Report Delta_past(d) for d in {1,4,8,16,32,64} kb where the length budget allows.

Oracle B (distal-right-context value, Track I):
- Train an infilling model with clean left flank, noised gap, clean right flank, and randomly-present
  distal right context beyond a fixed near-boundary window R_near. Eval Delta_right(d,r) = BPB drop
  from adding true distal right context, scoring ONLY gap tokens >= r from BOTH boundaries.

Requirements:
- Keep the near-boundary window identical across conditions; change only context farther than d.
- Full-attention backbone at 20-40 kb is acceptable as a start; log that longer signals are not ruled out.
- Emit distance-bucketed JSONL + an influence-vs-distance plot per oracle.
- Hard-negative matching (GC, chromosome, repeat class, gap length); reject unmatched negatives.
Acceptance: with a planted synthetic distal dependency the oracle recovers a positive distance-resolved
effect (sanity); on human hg38 report Delta(d) with confidence intervals; the pipeline states clearly
that a null result is a model-dependent lower bound, not proof of absence.
```

### P2 — Incentive test on the existing architecture (Stage 3) — NEW

```text
Add incentive training + causal-use evaluation to the EXISTING dit_dual, changing only the objective
(keep the causal block-diffusion task; do NOT switch to infilling here).

1. Implement matched counterfactual coarse states: true / zeroed / hard-shuffled (GC/chrom/repeat/length
   matched). 
2. Add a distance-gated margin loss L_use = softplus(NLL_true - NLL_shuffled + margin), applied ONLY to
   distal-dependent targets (synthetic) or interior/distal masked tokens (real) -- never all gap tokens.
3. Add independent route dropout for local window / coarse route / cross-attention slots.
4. Metrics: distance-resolved interior ΔBPB(r) for r in {0.5,1,2,4,8,16} kb, influence-vs-distance I(d),
   targeted effect ratio, logit KL true-vs-counterfactual, gradient norm to coarse route.
5. Keep every feature separately configurable; disabling all new losses recovers the A1 baseline exactly.
Acceptance (BOTH): held-out BPB does not regress; the SPECIFIC distance-resolved intervention effect at
distal/interior positions increases. A model that reacts to shuffled context but has worse likelihood
FAILS -- report it as "forced to react", not "taught to use".
```

### P3 — Infilling dataset and no-leak tests (Track I prep)

```text
Implement a deterministic span-infilling data path: clean left flank, noised middle gap, clean right
flank; target is only the gap (Mechanism C-a). The right flank is observed conditioning.
1. Sample spans from Carbon and hg38 caches; return left_flank_ids, gap_x0_ids, right_flank_ids,
   gap_attention_mask, absolute positions where available, gap length + distances to both boundaries.
2. Support fixed and variable gap lengths with deterministic validation spans.
3. No clean gap tokens in any conditioning field. Add synthetic variants where a right-flank motif
   determines a gap target motif + matched random controls.
4. Keep the standard next-region loader untouched. Document cache naming so smoke caches cannot shadow
   full caches.
5. Tests: mutate hidden gap_x0 while keeping observed fields fixed; assert all conditioning tensors
   are byte-identical.
```

### P4 — Stateless right-flank encoder prototype (Stage 4, contingent) — NEW

```text
CONTINGENT: only after Stage 2 positive and Stage 3 failed.
Add a STATELESS (no recurrent sampler) reverse/bidirectional right-flank encoder to the existing
attention backbone for infilling, to test whether representation capacity -- not recurrence -- is the
bottleneck.
1. Encode the clean right flank with a reverse-causal or bidirectional attention encoder (leak-free:
   the flank is observed). Expose its outputs through the existing cross-attention + AdaLN.
2. Include distance-to-right-boundary embedding and a configurable exact right-boundary nucleotide window.
3. Reuse the Stage 3 L_use + route dropout unchanged. Do NOT add Mamba, recurrent state, or a new sampler.
4. Evaluate with the distance-resolved metrics (interior ΔBPB(r), I(d), TER, matched swaps, boundary control).
Acceptance: if the stateless stronger encoder recovers distance-resolved downstream use, conclude the
current coarse route was a representation bottleneck and recurrence is NOT required for the science.
```

### P5 — Minimal SSM-Bridge backbone (Stage 5, contingent)

```text
CONTINGENT on the §10 gate (systems need or context-retaining generation goal). Preserve all existing
backbones. Files: models/ssm_bridge.py, models/mamba_dna.py, diffusion.py dispatch only, config
model=ssm_bridge_ca.
Minimal K=1: forward Mamba-2 recurrence over committed prefix / left flank; optional reverse Mamba-2 over
the clean right flank (infilling); bidirectional denoise within the single active chunk; condition each
active nucleotide on timestep, forward/reverse state slots (8 or 16), distance-to-boundary, and an exact
right-boundary window. AdaLN and/or fixed-slot cross-attention; no dense attention over full flanks;
NO rolling multi-block canvas, NO PSR, NO coarse plan. Normal-init residual projections; avoid the prior
dead-gate/double-zero failure.
Tests: scan == recurrent step within tolerance; hidden gap-x0 mutations cannot alter encoded flank states;
right-flank change changes reverse state; bf16 CUDA + fp32 CPU smoke; existing dit/dit_dual checkpoints
still load. Report gate/state-slot telemetry AND the distance-resolved use metrics.
```

### P6+ — Further contingent prompts

The following remain valid but are gated behind P5 and are not started until a minimal SSM is causally used: **counterfactual objective + guidance for the SSM**; **multi-block noisy canvas + block-buffer sampler** (requires the matched multi-block noise-training plan of §3.3); **partial-bidirectional Mamba fine mixer** [9]; **RC consistency and tied equivariance** [3]; **linear-time alternating refinement**; **multiscale predictive-state auxiliary**; and **experiment orchestration + automatic report** (versioned manifest over A0/A1/A1+/Stage-4/Stage-5 ablations, config-compatibility validation, JSONL aggregation, provenance hashes, smoke + full modes). Each keeps the acceptance criteria from v1 but inherits the distance-resolved metrics of §6.2.

## Appendix B — Reference pseudocode

### B.1 Infilling with a boundary-initialized noisy bridge (contingent SSM)

```text
Inputs: left flank L, right flank R, gap length M chunks, active width K
H_L <- ForwardState(L)
H_R <- ReverseState(R)
canvas <- K fully masked chunks
output <- []

for m = 1 ... M:
    for denoising step s = 1 ... T:
        t_1:K <- slot-wise noise schedule           # heterogeneous, inference-matched
        F_1:K <- ForwardMamba(canvas, init_state=H_L)
        B_K:1 <- ReverseMamba(canvas, init_state=H_R)  # reads noisy x_t only, never clean x0
        logits <- Denoiser(canvas, F, B, exact_left_window,
                           exact_right_boundary_window,
                           distance_to_right, t_1:K)
        canvas <- reverse_step(canvas, logits, t_1:K)

    commit <- canvas[1]
    append commit to output
    H_L <- ForwardMambaStep(H_L, summarize(commit))
    canvas <- shift_left(canvas)
    canvas[K] <- fully masked chunk

return output
```

### B.2 State-use ranking loss

For the same noisy target, evaluate with the true state `h`, a null state `h0`, and a matched shuffled state `h-`:

> **Counterfactual route-use loss.** `L_use = softplus(NLL(x ∣ h_true) − NLL(x ∣ h_negative) + delta)`, applied to **distance-gated distal/interior targets only** (never all tokens). On real DNA use it cautiously and report whether it changes calibration or local BPB.

## Appendix C — References

[1] Marianne Arriola et al. "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models." ICLR 2025 Oral. arXiv:2503.09573. [Source](https://arxiv.org/abs/2503.09573)

[2] Boyuan Chen et al. "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion." arXiv:2407.01392, 2024. [Source](https://arxiv.org/abs/2407.01392)

[3] Yair Schiff et al. "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling." ICML 2024. arXiv:2403.03234. [Source](https://arxiv.org/abs/2403.03234)

[4] Mingqian Ma et al. "HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model." arXiv:2502.10807, 2025. [Source](https://arxiv.org/abs/2502.10807)

[5] Vaibhav Singh et al. "DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone." arXiv:2511.15927v3, 2026. [Source](https://arxiv.org/abs/2511.15927)

[6] Zhao Yang et al. "D3LM: A Discrete DNA Diffusion Language Model for Bidirectional DNA Understanding and Generation." MLGenX 2026 workshop. arXiv:2603.01780. [Source](https://arxiv.org/abs/2603.01780)

[7] Yuhang Chen et al. "Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation." arXiv:2606.27732, 2026. [Source](https://arxiv.org/abs/2606.27732)

[8] Yijie Jin et al. "Multi-Block Diffusion Language Models." arXiv:2606.29215, 2026. [Source](https://arxiv.org/abs/2606.29215)

[9] Pranshu Chaturvedi et al. "Training Hybrid Block Diffusion Language Models with Partial Bidirectionality." arXiv:2607.02805, 2026. [Source](https://arxiv.org/abs/2607.02805)

[10] Marianne Arriola and Volodymyr Kuleshov. "Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding." ICML 2026. arXiv:2607.01775. [Source](https://arxiv.org/abs/2607.01775)

## Final project statement

> **Recommended continuation.** Treat the current dual-stream result as a valuable negative finding: long-range *access* is not long-range *use*. Do not build the SSM-Bridge next. Instead, disentangle the five hypotheses in order — capacity (synthetic), signal (real-data oracles), incentive (`dit_dual + L_use`), architecture (stateless stronger encoder) — and let the §10 decision rule choose the architecture. This sequence yields a publishable scientific story on every branch: it separates systems scalability, information availability, optimisation incentives, and genuine biological long-range dependence, and it never confounds them. The SSM-Bridge is the endpoint if — and only if — the gates demand it.
