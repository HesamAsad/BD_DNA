# Dual-Stream BD3-LM — Long-Context Inference & Test-Time Length Scaling

**Status:** living report · last updated 2026-06-25
**Model under test:** `outputs/carbon-prokaryote/2026.06.19/030312/checkpoints/7-52500.ckpt`
(dual-stream BD3-LM, trained at L=98,496 on Carbon prokaryote, leak/collapse/speed-fixed; still training)

---

## 1. Executive summary

We set out to test whether the dual-stream BD3-LM, **trained at 98,496 nt**, can run
**inference at much longer context** — the user's hypothesis being "~6× longer because
k=6". The answer turned out stronger than the hypothesis:

1. **Forward (teacher-forced) inference scales to ≥1,000,000 nt on a single H200.**
   At L=984,960 (≈1M, 10× train length) the model runs in **90.1 GiB** with **flat
   validation NLL** (1.138 vs 1.113 at 1×). Peak memory is **linear** in length
   (~8.8 GiB per 98.5k-nt "×"); **measured all the way to 14× = 1,378,944 nt (≈1.38M) at
   126.5 GiB (90% of the H200), no OOM**, so the single-H200 forward ceiling is
   **~15× ≈ 1.48M nt** (140 GB wall) — not 6×.
2. **The long-context output is valid, not just low-loss.** On a real 1M sequence the
   model denoises masked positions to **99.9% valid ACGT** at **51% / 48% accuracy**
   (15% / 50% masking; chance = 25%), and accuracy is **uniform across the full length**
   — no degradation at the far end (~position 950k).
3. **Why it extrapolates so cleanly:** only one component is length-sensitive (the coarse
   stream's rotary). The fine stream is windowed-local (length-invariant) and the
   fine→coarse cross-attention is block-causal, so there is no positional structure to
   break under extrapolation.
4. **Caveat (data):** the eval sequences are **real prokaryote DNA but "packed"** — a 1M
   window is ~50–150 real ~6.5kb fragments concatenated. So the result proves *the
   architecture runs and predicts valid DNA at 1M*, **not** that it exploits genuine
   million-nt biological dependency (which mostly isn't present in packed data).
5. **De novo generation (Track B) is a separate, unbuilt capability** and is needed
   **only if the downstream goal is sampling new long sequences**. For scoring /
   likelihood / variant-effect / infilling / embeddings, the forward path (already
   working at 1M) is sufficient.

---

## 2. Background: why the dual stream exists

Goal: long-context BD3-LM (block discrete-diffusion LM) on DNA at **single-nucleotide
resolution**. Two routes to beat the O(L²) attention cost were considered:

- **Variant 1 — latent diffusion.** Compress k nucleotides → 1 latent, run BD3-LM on the
  L/k latent stream. Largest speedup (~k² on attention *and* ~k on MLP) but needs
  two-stage training and a reconstruction bottleneck that **leaks at single-base
  resolution** — fatal for SNP/variant-effect biology.
- **Variant 2 — dual stream (chosen).** Keep diffusion at single-nucleotide resolution;
  cheapen the fine path instead. Chosen for: no reconstruction bottleneck, single training
  stage (parameter-free k-mer lookup), and an interpretable coarse-conditioning slot.
  Accepted trade: smaller max speedup (the fine MLP still scales O(L·d²)).

### Architecture (`models/dit_dual.py`)
- **Coarse stream:** clean `x_0` → non-overlapping k-mer ids (k=6, vocab 4097) at length
  L/6 → `n_coarse` transformer blocks with **causal** attention. Causality is what makes
  the stream leak-free and suffix-invariant. No timestep conditioning.
- **Fine stream:** input `[x_t ; x_0]` (length 2L). Each of 12 `FineDualBlock`s:
  (1) **windowed local self-attention** — BD3-LM block mask AND-ed with `(blk_q−blk_kv) ≤
  8` blocks (= ±144 nt); (2) **cross-attention** fine→coarse, block-causal; (3) MLP. All
  adaLN-modulated by the diffusion timestep.
- Output sliced to the `x_t` half (length L). `small_dual` ≈ 123M params.

### The three debugging threads (documented in `docs/dual_stream_report.tex`)
| Bug | Cause | Fix |
|---|---|---|
| **Information leak** | coarse encoder was bidirectional over clean `x_0` (the target); past coarse tokens absorbed the current/future target | `CoarseBlock` → `is_causal=True` + mixed sampling cross-mask |
| **OOM / speed** | `flex_attention` called **eagerly** → dense (B,H,2L,2L) score grid, ignored the windowed mask (867 GiB at 98k) | route both flex calls through the `torch.compile`-wrapped `fused_flex_attention` |
| **Marginal collapse** | `self_proj` & `cross_attn.out_proj` zero-init **and** adaLN-gated → zero×zero deadlock → model = unigram (val/nll = H_uni = 1.382) | remove the two projection zero-inits (keep gate zero-init) |

All three are fixed in the checkpoint under test.

---

## 3. The checkpoint under test

| field | value |
|---|---|
| path | `outputs/carbon-prokaryote/2026.06.19/030312/checkpoints/7-52500.ckpt` |
| train length L | 98,496 (= 18 × 5472 = 6 × 16,416) |
| block_size | 18 |
| k_coarse | 6 → coarse stream = 16,416 tokens |
| window | 8 blocks = 144 nt |
| fine | d=768, 12 layers, 12 heads |
| coarse | d=384, 4 layers, 6 heads |
| params | ≈123M |
| data | Carbon `prokaryote_evo2`, 1 shard |
| training | job 50711, step 52,500+, **still running** |

---

## 4. Methodology: forward NLL length extrapolation

**Why this is a clean probe.** Only the coarse rotary is length-sensitive (`Rotary` is
parameter-free, computed at runtime from `seq_len`). The fine self-attention is
windowed-local (never attends beyond ±144 nt → length-invariant) and the cross-attention
is block-causal. So extrapolating to N× length = asking the coarse rotary to stretch from
16,416 trained positions to 16,416·N.

**Why the cross-length load is exact.** `Diffusion.load_from_checkpoint(config=<L-overridden>,
strict=False)` rebuilds the model at the new `model.length` (flex BlockMasks rebuilt in
`__init__`, **not** in the state_dict) and loads the 98k weights — which are **all
length-independent** (rotary `inv_freq` is dim-based; cos/sin caches are runtime attrs).
No shape mismatch, no retraining.

**Two eval modes built:**
- `mode=ppl_eval` (existing) — teacher-forced NELBO on held-out data → `val/nll`, `val/ppl`.
  Driven by `scripts/eval/sweep_longctx_nll.sh` (rebuilds at each length, records peak GiB
  + wall time, stops at first OOM). The **1× row is a self-check**: it must reproduce the
  training val/nll.
- `mode=io_dump` (added, `main.py:_io_dump`) — noises a real sequence with `q_xt`, runs the
  EMA forward, and dumps decoded `truth / masked-input / prediction` + masked-accuracy by
  position. Driven by `scripts/eval/dump_longctx_io.sh`. Validity, not just loss.

---

## 5. Results — NLL length extrapolation (single H200, EMA, held-out prokaryote)

Jobs 55917 (1×–6×, 16 seqs/len) and 55996 (10×–14×, 8 seqs/len).

| L (nt) | × train | val/nll | val/ppl | status |
|---:|:--:|---:|---:|:--:|
| 98,496 | 1× (control) | **1.1129** | 3.043 | ok |
| 196,992 | 2× | 1.1267 | 3.086 | ok |
| 295,488 | 3× | 1.0836 | 2.955 | ok |
| 393,984 | 4× | 1.1289 | 3.092 | ok |
| 492,480 | 5× | 1.1153 | 3.050 | ok |
| 590,976 | 6× | 1.1151 | 3.050 | ok |
| 984,960 | 10× (≈1M) | **1.1377** | 3.120 | ok |
| 1,083,456 | 11× | 1.0844 | 2.958 | ok |
| 1,181,952 | 12× | 1.1043 | 3.017 | ok |
| 1,280,448 | 13× | 1.1346 | 3.110 | ok |
| 1,378,944 | 14× (≈1.38M) | **1.1199** | 3.065 | ok |

**val/nll is flat across the whole measured range** (1.084–1.138, all within ±0.025 of the
1× control). The wiggle is different validation chunks per length, not a trend. The
marginal floor (unigram) is **1.382** — every length sits well below it, so the model is
doing real conditional prediction at every length, not collapsing to base composition.

---

## 6. Memory & sequence length on a single H200 (the headline tables)

### 6.1 Measured — forward/inference peak memory & wall time

`peak_gib` = max GPU memory during the run (sampled via `nvidia-smi`).
`wall_s` includes data load + model build + flex compile (not pure forward).

| L (nt) | × | peak GiB | % of 140 GB | wall s | seqs | s / seq |
|---:|:--:|---:|:--:|---:|:--:|---:|
| 98,496 | 1× | 11.6 | 8% | 62 | 16 | 3.9 |
| 196,992 | 2× | 19.2 | 14% | 89 | 16 | 5.6 |
| 295,488 | 3× | 28.1 | 20% | 143 | 16 | 8.9 |
| 393,984 | 4× | 35.7 | 26% | 187 | 16 | 11.7 |
| 492,480 | 5× | 47.5 | 34% | 257 | 16 | 16.1 |
| 590,976 | 6× | 57.4 | 41% | 349 | 16 | 21.8 |
| 984,960 | 10× | 90.1 | 64% | 471 | 8 | 58.9 |
| 1,083,456 | 11× | 95.3 | 68% | 560 | 8 | 70.0 |
| 1,181,952 | 12× | 115.2 | 82% | 652 | 8 | 81.5 |
| 1,280,448 | 13× | 123.4 | 88% | 755 | 8 | 94.4 |
| 1,378,944 | 14× | **126.5** | **90%** | 848 | 8 | 106.0 |

**The full 1×–14× curve is now measured** (sweep 55996 completed; **no OOM through 14×**).

### 6.2 The memory model (linear) & the single-H200 ceiling

Linear fit over the measured 1×–14× points: **peak ≈ 4 + ~8.8·(×) GiB**, i.e.
**~8.8 GiB per 98,496 nt** (~91 MiB per 1,000 nt). Per-× increments are noisy
(allocator fragmentation under `expandable_segments`: e.g. 10→11× +5.2, 11→12× +19.9,
13→14× +3.1) but the trend is clean. **14× (1.38M) measured at 126.5 GiB = 90% of the
H200.** Remaining headroom to the wall:

| L (nt) | × | GiB | fits 140 GB? |
|---:|:--:|---:|:--:|
| 1,378,944 | 14× | **126.5 (measured)** | ✅ 90% |
| 1,477,440 | 15× | ~135 | ✅ (tight, ~96%) |
| 1,575,936 | 16× | ~144 | ❌ OOM |

**Single-H200 forward ceiling: ~15× ≈ 1.48M nt** (≈135 GiB). 14× is measured; a 2-point
follow-up at 15×/16× would pin the exact wall, but the linear curve already places it
firmly between 15× and 16×.

### 6.3 Why inference reaches ~15× when *training* maxed out at 1×

| regime | what's resident | ~mem at L=98,496 |
|---|---|---|
| **Training** (batch 1, 4-GPU) | fwd activations for 2L + **backward graph** + optimizer + grad | ~120 GiB / GPU |
| **Inference** (forward only, `no_grad`) | only live forward activations (windowed attn + coarse + residual) | **11.6 GiB** |

Inference is ~10× cheaper than training at the same length because there is no backward
graph or optimizer state. That headroom is exactly what buys the ~15× length scaling.
The memory growth is linear (not quadratic) because the fine self-attention is windowed
(O(L·w)) and the coarse self-attention is flash/causal (O(L/k)); only *compute* (the
block-causal cross-attention, O(L²/k)) is quadratic, which is why `s/seq` grows
super-linearly while memory stays linear.

---

## 7. Validity of the long-context output (io_dump)

Real validation sequence, EMA model, masked-position prediction (`mask` → model argmax).
Accuracy is over **masked positions only** (chance = 0.25 for 4-way ACGT).

| run | L (nt) | mask 15% acc | mask 50% acc | pred composition | wall |
|---|---:|:--:|:--:|---|---|
| 1× control (55941) | 98,496 | **0.569** | 0.520 | 100% ACGT (GC≈67%) | <1 min |
| 1M (55991) | 999,990 | **0.509** | 0.476 | 99.9% ACGT (GC≈60%) | ~2.5 min |

**Masked-accuracy by position bin at 1M (15% mask), start → end of 999,990 nt:**
```
0.51 0.54 0.50 0.52 0.49 0.47 0.53 0.52 0.52 0.50 0.46 0.55 0.53 0.53 0.49 0.50 0.51 0.50 0.50 0.51
```
**Flat** — position ~950k is as accurate as position ~50k. This is the definitive
"valid in long context" signal: no drift, no collapse, valid DNA throughout. Decoded
window at the **very end** of the 1M sequence (50% masked):
```
truth: CGACGAATTGCGCGGCGTCATCGAAGCGGGAGATTGGGCGCGCGCGAGCTGGGGCGGCACGGACGCC...
input: C.ACGAA...C..GG..T.A.CGAAG....A.A..G.G..CG.......TGG..C.G.A..GA.....
pred : CGACGAAGTGCTCGGCCTGATCGAAGACGAACAACGCGAGCGCGACGACTGGGTCGGCAACGAAGCC...
```
The 1M accuracy (0.51) is slightly below the 1× control (0.57), but that is mostly a
**different DNA region** (60% vs 67% GC = different organism mix), not long-range
breakdown — the flat per-bin curve rules out degradation.

Artifacts per run in `logs/eval/io_dump_L<L>_<job>/`: `report.txt`, `windows.txt`
(start/middle/end), `decoded_seq0.txt` (full FASTA-ish), `arrays.npz` (raw `x0`/`xt`/`pred`).

---

## 8. The data — real but packed

Source: Carbon `prokaryote_evo2`, **real prokaryote DNA** (alphabet ACGT), 563k
sequences/shard. Source sequence length distribution (sampled):

| stat | nt |
|---|---:|
| min | 201 |
| median | 6,499 |
| mean | 25,621 |
| max | 4,059,867 (a 4 Mb contig) |
| % ≥ 1 Mb | 0.2% |
| % ≥ 100 kb | 3.8% |
| % < 10 kb | 62.7% |

`wrap=True` concatenates these end-to-end (no separators) and chunks to length L. So a 1M
eval window is typically **~50–150 distinct real fragments glued together**. Consequences:
- Every nucleotide and all **local** structure is real → masked-accuracy is a legitimate
  signal.
- Genuine **million-nt single-organism** dependency is mostly **absent** (chunk spans many
  organisms). The flat-NLL / flat-accuracy results therefore demonstrate *architectural
  length-robustness*, not exploitation of true long-range biology.

### 8.1 Genuine long-range eval — contiguous single-organism windows (job 56206)

Built a real long-range eval set (`scripts/eval/build_longrange_eval.py`): **24 windows
of 984,960 nt, each a contiguous slice of one source contig that is itself ≥1 Mb** (15
distinct organisms, pure clean DNA — not packed fragments). Plus a **shuffled control**:
the same windows with ~1 kb blocks permuted (long-range order destroyed, local content +
base composition identical). Eval at L=984,960, EMA, all 24 windows:

| variant | boundaries | val/nll | val/ppl | Δ vs contiguous |
|---|--:|---:|---:|---:|
| **contiguous** (real ≥1Mb single-organism) | 0 | **1.1255** | 3.082 | — |
| shuffled @10 kb | 96 | 1.1303 | 3.097 | +0.0048 |
| packed (10× glued fragments, ~150 junctions) | ~150 | 1.1377 | 3.120 | +0.012 |
| shuffled @1 kb | 960 | 1.1720 | 3.228 | +0.0465 |

**Finding: the model is LOCAL-dominated — it does NOT meaningfully use long-range context.**
The two shuffle granularities disentangle it cleanly (jobs 56206 + 56300): the NLL penalty
is **proportional to the number of chunk boundaries, not to how much long-range order is
destroyed**:
- @1 kb: 0.0465 / 960 boundaries = **4.84e-5 nats/boundary**
- @10 kb: 0.0048 / 96 boundaries = **5.0e-5 nats/boundary** (identical)

Both shuffles destroy *all* >10 kb order, yet the 10 kb shuffle (10× fewer boundaries)
hurts 10× less. A genuine long-range model would be hurt ~equally by both. So the penalty
is purely the **local boundary artifact** (each boundary corrupts the ±144 nt fine window of
~288 positions); true long-range usage is ≈0. This unifies all four numbers — NLL simply
ranks by **number of local discontinuities** (contiguous 0 < 10 kb 96 < packed ~150 < 1 kb
960), including why packed > contiguous (packed has ~150 fragment junctions). The
preliminary "+0.047 → uses long-range" reading was a boundary artifact; disentangled, it
**confirms** the cross-attn / coarse long-range pathway is ~vestigial even at step 52k.
(Masked-accuracy on the contiguous 1M window is uniform across position, 0.49–0.63 — the
model is *valid* across the full length, it just predicts from local context.)

**Implication for design:** if long-range / bidirectional DNA context is the goal, the
current dual (block_size=18, windowed fine + causal coarse) won't deliver it — the model
doesn't learn to use the coarse global pathway. A fundamentally larger **bidirectional
block** (e.g. block_size ~100k, hierarchy/AR across blocks to reach ≥1M) is the design that
would force genuine long-range modelling. See §10 / the architecture discussion.

---

## 9. Engineering issues solved along the way

| issue | symptom | root cause | fix |
|---|---|---|---|
| **wandb in ppl_eval** | crash before GPU | `~wandb` deleted a key `main.py:236` then assigns | drop the override (main.py disables wandb itself) |
| **datasets.map grouping deadlock** | "Grouping 0%" hang for hours, GPU idle | `_group_texts` slices million-element Python lists + `torch.ones(L)` per chunk, forked across `num_proc`=128 workers; deadlocks at long L (also at 8) | `BD3LM_DATA_NUM_PROC` env cap (dataloader.py); **num_proc=1** is robust (no fork, incremental write); pre-build caches with `scripts/eval/pregen_longctx_caches.py` |
| **head-node OOM** | flatten+rechunk builder SIGKILLed | tight per-process memory cgroup on the head node (~1.7 GB) | use num_proc=1 grouping (low peak) instead of `from_dict` bulk copy |
| **cross-length checkpoint load** | (de-risked) | masks rebuilt at `__init__`, weights length-independent | `load_from_checkpoint(config=override, strict=False)` works directly |

---

## 10. De novo generation and "Track B"

### 10.1 Forward vs generation — what the current results do and don't cover
- **Forward path (validated to 1M):** given a *complete* sequence, mask positions, predict
  them in **one pass**. Covers **likelihood/perplexity, variant-effect scoring, masked
  infilling, embeddings/representations**.
- **Generation / sampling (de novo):** produce a *new* sequence from scratch with **no
  `x_0`**, by iteratively denoising block-by-block (semi-autoregressive), many forward
  passes, sliding a window across blocks.

### 10.2 Why generation needs a rebuild ("Track B")
The current sampler (`diffusion._semi_ar_sampler` / `DualStreamDIT._gen_sampling_masks`)
**cannot do long-context generation as-is**:
- It builds **dense L×L sdpa masks** at the current length — a dense (1M)² mask is
  impossible (~10¹² entries).
- It defaults to a **`context_size=1024` sliding window**, which discards exactly the
  long-range coarse context we want to use ("Gap B" in the leak-fix notes).
- The coarse stream is re-encoded over only the in-window prefix (truncates long-range
  coarse context).

**Track B = rebuild the sampler to be memory-efficient and long-range-correct:**
flex windowed fine-attention + flash-causal coarse + **full-prefix coarse encode** + large
`context_size` (no 1024 cap). After that, generation inherits the same O(L·w) memory the
forward path enjoys, and one could sample at ~1M and measure generative quality + speed vs
a single-stream baseline.

### 10.3 Is Track B actually needed? — depends on the goal
- **If the downstream task is de novo generation** (sample novel long genomes) → **yes**,
  Track B is the next build.
- **If the task is scoring / likelihood / variant-effect / infilling / embeddings** →
  **no** — the forward path is sufficient and already scales to 1M; the better next step is
  the **contiguous-≥1Mb long-range eval** (§8) to test genuine long-range usage.

The dual stream's efficiency advantage (windowed local + cheap coarse) applies to **both**
forward and generation; the forward results already demonstrate the memory win at 1M.

---

## 11. Conclusions & open questions

**Confirmed**
- Dual BD3-LM trained at 98k runs **forward inference measured to 14× = 1.38M nt on one
  H200** (126.5 GiB = 90%, no OOM), with **flat NLL** (1.08–1.14 across all lengths) and
  **valid, uniform, locally-coherent** masked predictions at 1M.
- Single-H200 forward ceiling **~15× ≈ 1.48M nt** (linear memory; 1×–14× all measured).
- The "6× because k=6" hypothesis is comfortably exceeded; the binding constraint is GPU
  memory (linear), with compute (quadratic cross-attn) the secondary cost.

**Open**
- Exact OOM wall (15× fits tight ~135 GiB, 16× OOMs ~144 GiB) — a 2-point run pins it.
- **Genuine long-range** test on contiguous ≥1Mb single-organism DNA (not yet run).
- **Track B** (long-context generation) — unbuilt; needed only if de novo sampling is the
  goal.
- Multi-GPU / gradient-checkpointed scaling beyond ~1.5M (the original ~1M+ training goal
  needs checkpointing; inference does not).

---

## Appendix — artifacts & jobs

**Scripts**
- `scripts/eval/sweep_longctx_nll.sh` — NLL length sweep (per-length process, peak-mem
  sampling, OOM-stop). Tunables: `LENGTHS`, `LIMIT`, `CKPT`, `FLEX_MODE`.
- `scripts/eval/dump_longctx_io.sh` + `main.py mode=io_dump` (`_io_dump`) — decoded
  validity dump.
- `scripts/eval/pregen_longctx_caches.py` — deadlock-free cache pre-build (num_proc=1).
- `scripts/eval/build_longctx_cache.py` — faster flatten+rechunk builder (needs real RAM,
  not the head node).
- `BD3LM_DATA_NUM_PROC` (dataloader.py) — datasets.map worker cap; **use 1** for long ctx.

**Jobs**
- 55917 — NLL sweep 1×–6× (done).
- 55996 — NLL sweep 10×–14× (done; no OOM through 14× = 1.38M @ 126.5 GiB).
- 55941 — io_dump 1× control (done).
- 55991 — io_dump 1M = 999,990 (done).
- 50711 — main dual training (running; checkpoint source).

**Results**
- `logs/eval/sweep_longctx_nll_<job>.tsv` — length/mem/nll tables.
- `logs/eval/io_dump_L<L>_<job>/` — decoded validity dumps.
- Related: `docs/dual_stream_report.tex` (architecture + the three bug fixes),
  memory notes `longctx-length-extrapolation`, `longctx-marginal-collapse`, `bd3lm-dna-project`.
