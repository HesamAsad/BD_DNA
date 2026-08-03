# Synthetic long-range-dependency benchmark for DNA BD3-LM

> ## ⚠️ SUPERSEDED — the echo task below is ILL-POSED. Use fixed-offset duplication.
>
> **What went wrong (measured 2026-07-20, two failed runs).** The "random motif at `s`,
> copied at `s+g`" design cannot be solved by *any* model, so it can never test an
> architecture. To reconstruct a masked target span the model must first work out WHERE to
> copy from, but:
> 1. the eval masks the target span **entirely** → no content cue to match on;
> 2. the sequence around the target is unrelated random background → no contextual link
>    to the source;
> 3. `g` is drawn from a **set** of gaps and `s` is random per sequence → no positional rule.
>
> No information in the input identifies the source. **Even a perfect model scores chance.**
>
> **Evidence.** `synthLR24k` (6 echoes × 32 nt = 0.78% predictable): within-block 0.2505,
> cross-block 0.2504, control 0.2516, `val/nll` 1.384 = uniform floor `ln 4` = 1.3863.
> Rebuilt 20× denser as `synthLR12k` (80 × 24 nt = 15.6%, 40,960 val echo pairs):
> **identical failure** — within-block 0.2522, cross-block 0.2485, `val/nll` flat 1.383-1.389
> for both `l_use=0` and `l_use=1.0`. Density was a red herring; the task is ill-posed.
>
> **Replacement — `scripts/eval/gen_synthetic_duplication.py`:** `x[i] = x[i-D]` for a FIXED
> offset `D`. Every position past `D` is predictable by copying from exactly `D` back, so the
> model learns ONE relative offset (trivial for RoPE-relative attention), density is maximal,
> and full-span masking is legitimate because no cue is needed. It emits an echo-manifest, so
> `main.py mode=synth_copy_eval` works unchanged. Run it as a **ladder**:
>
> | dataset | `D` | reachable by fine attn (`block_size × window_blocks`)? | ideal `val/nll` | role |
> |---|---|---|---|---|
> | `synthDUPshort` | 512 | yes | 0.058 | **sanity** — must pass before anything else is read |
> | `synthDUPlong` | 6144 (4 blocks) | no | 0.693 | **the test** — solvable only via the coarse route |
>
> **The sanity rung PASSES** (job 81513: `val/nll` 1.3863 → **0.0466**), which proves the
> model, training loop, data path and eval are all functional — the echo failures were the
> benchmark, not the pipeline.
>
> **Two rules this cost us:** (1) a benchmark must have a *solvable* information path, not
> just a planted correlation; (2) never interpret a long-range number whose short-range
> control is at chance. See plan §Stage 1 hard precondition.
>
> The original design is retained below for the record only.

**Purpose.** Decide whether a large bidirectional block (or the coarse cross-stream)
can *learn to use* >block_size context — separated from whether the data *has* any
long-range structure. On the real prokaryote corpus this is unanswerable: mutual
information decays to ~0 beyond ~1 kb (0.018 bits at d=1 → ~0 at d≥10 kb), and we
*observed the consequence* — in run 56995 (block=24,576) `gate_cross` **decays**
(0.0095→0.006 while `gate1`/`gate2` grow ~3×; ‖cross_out‖ 16.0→14.8). No long-range
signal ⇒ no gradient to keep the long-range pathway alive ⇒ it atrophies. This
benchmark plants long-range signal **by construction** so the architecture question
can actually be tested.

## 1. Task — "echo / copy across blocks"

Each sequence is random ACGT (ids A=8,C=9,G=10,T=11) with **K planted echo pairs**.
An echo pair writes a **random** m-nt motif at a source position `s` and copies it
verbatim at `s+g` (gap `g`). Key properties:
- **Random motif per sequence** ⇒ the model must *retrieve* the source, not memorise
  a fixed pattern.
- **Gap sweep** `g ∈ {within-block, > block_size}` ⇒ predicting a target at
  `g > block_size` requires reaching across a block boundary. The windowed fine
  self-attention (±window_blocks) structurally **cannot**; only the **coarse
  cross-attention** (block-causal, sees all earlier blocks at 1/k resolution) can.
  The coarse k-mer (k=6, vocab 4⁶) preserves exact 6-mer identity, so a target span
  is exactly reconstructible from the source's coarse tokens.

So **copy accuracy at the target span, as a function of gap, is a direct readout of
long-range usage**: a local model → chance (0.25) once `g > block_size`; a model that
engages the coarse pathway → high accuracy at any gap whose source is in an earlier
block.

## 2. Data generation

`scripts/eval/gen_synthetic_longrange.py` — writes caches the existing pipeline loads
via `data.train=/data.valid=<name>` (with `data=carbon-prokaryote` for the loader
plumbing) plus an echo manifest for the metric:
```
python scripts/eval/gen_synthetic_longrange.py \
  --length 98304 --n_train 8192 --n_val 512 \
  --motif_len 32 --n_echoes 6 --block_size 24576 \
  --gaps 2048,30000,50000,75000 --name synthLR
# -> data_cache/carbon/synthLR_{train,validation}_bs98304_wrapped_specialFalse.dat
#    data_cache/carbon/synthLR_echo_manifest.json
```
Validated: target spans are exact copies of their sources; cross-block pairs present;
background is pure ACGT. (Build the large train set on a CPU node — `from_generator`
keeps it memory-light; the validation set + manifest are small.)

Gaps to include: at least one **within-block** (e.g. 2,048 — control, must be easy)
and several **> block_size** (30k/50k/75k for block=24,576). For an L=98,304 / 4-block
model, 75k spans nearly the whole sequence.

## 3. Training protocol

Train the **big-block** model and the **block=18 baseline** on `synthLR` (identical
data), monitoring `gate_cross` with `scripts/diag_gate_trajectory.py`:
```
# big block (24,576) — the candidate
BLOCK_SIZE=24576 LENGTH=98304 GLOBAL_BATCH=16 \
  bsub < scripts/train/train_dna_bigblock_dual.sh   # + data.train=synthLR data.valid=synthLR (see launcher note)
# small block (18) — baseline that CANNOT solve cross-block copies
model=small_dual block_size=18 ...                  # same data
```
(Add `data.train=synthLR data.valid=synthLR` to the launcher's hydra args, or pass
via the `EXTRA` env hook.) Optionally mix synthLR with real DNA (e.g. 50/50) so the
model stays a DNA model while still seeing planted signal.

## 4. Metric — targeted copy accuracy by gap (`mode=synth_copy_eval`)

`main.py:_synth_copy_eval` (added): loads the val cache + manifest, and for each
sequence builds `x_t` = `x_0` with **only the target spans masked** (sources left
clean), runs the EMA forward, and measures **exact per-nucleotide copy accuracy at
the target spans, bucketed by gap** — plus a **control**: an equal number of random
(non-echo) masked spans (expected ~0.25). Decisive because it conditions on the
source being visible and asks only "did the model copy it across the gap".
```
IO masks only target spans -> accuracy(gap):
  gap < block_size : high for ANY working model (local suffices)   [sanity]
  gap > block_size : HIGH only if the long-range pathway is used   [the result]
  control (random spans) : ~0.25                                   [calibration]
```

## 5. Expected outcomes & how to read them

| observable | "architecture CAN use long-range" | "cannot / doesn't" |
|---|---|---|
| copy acc @ gap > block (big-block) | high (→1.0) | ~0.25 |
| copy acc @ gap < block | high | high |
| `gate_cross` trajectory on synthLR | **grows** | flat/decays (as on prokaryote) |
| block=18 baseline @ gap > block | ~0.25 (structurally cannot) | ~0.25 |

The **money comparison**: `gate_cross` **grows on synthLR but decayed on prokaryote**,
and **cross-block copy accuracy is high for the big-block but chance for block=18**.
That would prove (a) the long-range pathway works and is learnable when signal exists,
and (b) the prokaryote null was a property of the *data*, not the architecture — which
is the question the block-shuffle test could not answer.

## 6. Caveats / extensions
- **Solvability via coarse:** a target at gap `g` is reconstructible only if its source
  lies in a *strictly earlier* block (coarse is block-causal) — ensure the generator's
  cross-block gaps put source and target in different blocks (they do for g>block).
- **Difficulty knobs:** motif_len (longer = easier to localise), n_echoes (density),
  number of distractor motifs, multi-hop (A→B→C), or KV-retrieval (key early, value
  late) for a harder induction-style probe.
- **Mix with real DNA** to avoid the model degenerating into a pure copy machine.
- **Use the right metric** — targeted copy accuracy / context-swap KL, never aggregate
  block-shuffle NLL (shown underpowered at large block_size).
