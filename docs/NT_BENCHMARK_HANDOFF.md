# Running the Nucleotide Transformer benchmark (Caduceus Table 2)

Everything needed to produce numbers comparable to the Caduceus paper's Table 2,
using our checkpoints and our fine-tuning harness. Written 2026-09-21.

Sources, all verified against primary material rather than recalled:
`kuleshov-group/caduceus` at `main` (`configs/dataset/nucleotide_transformer.yaml`,
`configs/experiment/hg38/nucleotide_transformer.yaml`,
`configs/pipeline/nucleotide_transformer.yaml`,
`slurm_scripts/run_nucleotide_transformer.sh`), arXiv:2403.03234v2 Table 2 and
Appendix D.2, and the HuggingFace dataset card.

---

## 0. Where everything lives

All paths in this document are relative to the repository root:

```
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
```

So `scripts/eval/...` means `$REPO/scripts/eval/...`. The absolute paths for the
things you need first:

| what | absolute path |
|---|---|
| repository root | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms` |
| python interpreter | `/software/cellgen/team361/ha11/envs/nichejepa/bin/python` |
| **the checkpoint** | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/outputs/hg38-caduceus/hg_bissm_cos/checkpoints/best.ckpt` |
| the harness to extend | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/scripts/eval/caduceus/finetune.py` |
| the loader to mirror | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/scripts/eval/caduceus/genomic_benchmarks.py` |

**Prerequisites, worth checking before you start:**

- Both `/lustre/scratch126` and `/software` are **shared cluster filesystems**,
  so this only works from a node on the same cluster (`tiger22`). There is no
  copy of this outside it.
- You need to be in the **`team361` unix group**. The repository itself is
  world-readable, but **the checkpoint is `-rw-rw---- ha11:team361`, group-only**, so without that group you will get a permission error on the one file you cannot do without. Check with `id -nG | tr ' ' '\n' | grep team361`.
- Run from `$REPO`, not from a copy. Each script derives the repo root from
  its own location (`Path(__file__).resolve().parents[N]`, with N depending
  on how deep the script sits) and resolves sibling data paths from it, so
  moving a script out of its directory breaks those lookups silently.

## 1. How this differs from GenomicBenchmarks

Same shape, three substantive differences. If you have run our GB harness, do
not assume the rest carries over.

| | GenomicBenchmarks (Table 1) | Nucleotide Transformer (Table 2) |
|---|---|---|
| tasks | 8 | **18** |
| metric | accuracy everywhere | **MCC (12), F1-binary (5), accuracy (1)** |
| seeds | 5 | **10** |
| error bar | standard deviation | **max &minus; min over the 10 seeds** |
| epochs | 10 (our recipe) | **20** |
| sequence length | 256-4864 | 200-600 |
| data | `katarinagresova/genomic_benchmarks_*` | `InstaDeepAI/nucleotide_transformer_downstream_tasks` |

**The error bar convention is the single easiest thing to get wrong.** The paper
states: *"Error bars indicate the difference between the maximum and minimum
values across 10 random seeds used for CV."* It is a **range, not a standard
deviation**, so it is roughly 3x larger than an sd would be for the same spread
and is not comparable to one. Report `max - min`, and say so explicitly.

## 2. The 18 tasks

Exact values from `configs/dataset/nucleotide_transformer.yaml`. `max_length`
and `d_output` are **per task** and must be set per task.

| task | max_length | classes | metric | train rows |
|---|---|---|---|---|
| `enhancers` | 200 | 2 | mcc | 14,968 |
| `enhancers_types` | 200 | **3** | mcc | 14,968 |
| `H3` | 500 | 2 | mcc | 13,468 |
| `H3K4me1` | 500 | 2 | mcc | 28,509 |
| `H3K4me2` | 500 | 2 | mcc | 27,614 |
| `H3K4me3` | 500 | 2 | mcc | 33,119 |
| `H3K9ac` | 500 | 2 | mcc | 25,003 |
| `H3K14ac` | 500 | 2 | mcc | 29,743 |
| `H3K36me3` | 500 | 2 | mcc | 31,392 |
| `H3K79me3` | 500 | 2 | mcc | 25,953 |
| `H4` | 500 | 2 | mcc | 13,140 |
| `H4ac` | 500 | 2 | mcc | 30,685 |
| `promoter_all` | 300 | 2 | **f1_binary** | 53,276 |
| `promoter_no_tata` | 300 | 2 | **f1_binary** | 47,767 |
| `promoter_tata` | 300 | 2 | **f1_binary** | 5,517 |
| `splice_sites_acceptors` | 600 | 2 | **f1_binary** | 19,961 |
| `splice_sites_all` | 400 | **3** | **accuracy** | 27,000 |
| `splice_sites_donors` | 600 | 2 | **f1_binary** | 19,775 |

Two tasks are 3-class (`enhancers_types`, `splice_sites_all`); the rest are
binary. `splice_sites_all` is the only task scored by accuracy.

**Name mismatch to resolve at load time.** Caduceus's config uses
`splice_sites_acceptors` / `splice_sites_donors` (plural); the HuggingFace
dataset card lists `splice_sites_acceptor` / `splice_sites_donor` (singular).
Try both and use whichever `load_dataset` accepts. Do not silently skip a task
that fails to load -- 16 of 18 is not Table 2.

## 3. The protocol

From `slurm_scripts/run_nucleotide_transformer.sh` and Appendix D.2:

- **10 folds = 10 seeds**, `for seed in $(seq 1 10)`, passed as
  `dataset.train_val_split_seed`. It is **not** k-fold partitioning: each seed
  redraws a **90/10 train/validation split from the train split**.
- **The test split is the official HF `test` split**, fixed across all seeds.
  There is no official validation split; that is why validation is carved from
  train.
- **Early stopping on the validation metric** -- `monitor: val/${dataset.metric}`,
  `mode: max`. Restore best-validation weights, then evaluate test **once**.
- **20 epochs** (`trainer.max_epochs=20`).
- **lr in {1e-3, 2e-3}, batch in {128, 512}**, chosen per task; Appendix D.2
  Table 6 gives the per-task pick for Caduceus-Ph and Caduceus-PS separately.
  Their sweep selects on validation, so if you sweep, sweep on validation only.
- Reported number is **test** performance, mean over the 10 seeds.

This is structurally identical to our GB protocol, which matters: our
`--val-fraction` already defaults to **0.1**, and `split_train_val(labels,
fraction, seed, stratified)` already carves validation out of train per seed
with test held fixed. **Do not rebuild the split logic.**

## 4. Which checkpoint to use

**Primary: `outputs/hg38-caduceus/hg_bissm_cos/checkpoints/best.ckpt`**

| | |
|---|---|
| architecture | BiSSM (bidirectional Mamba-2), block diffusion, ~100.7M params |
| pretraining corpus | `hg38-cad-no89` -- bed-filtered hg38 with **chr8/chr9 excluded** |
| length / block | 8,192 / 256 |
| `right_flank_probability` | 0.5 |
| `time_conditioning` | **true** |
| steps | 54,500 |
| val/nll | 1.0795 on its own validation split |
| GB precedent | 0.8736 8-task mean, so it is known to load and fine-tune cleanly |

Chosen over the alternatives for three reasons: it is **finished and stable**
(not a moving target), its **chr8/chr9 exclusion** is what makes the
contamination control in &sect;8 possible, and it already has a working GB
result through this exact harness.

**Second arm, optional, available from ~18:00 on 2026-09-21:**
`outputs/human-rf05-20260920/b256/checkpoints/best.ckpt` (the `fd_human` run --
60,000 steps, full corpus, `rf=0.5`). Useful as a "does more pretraining help
NT" contrast, but note it differs on **two** axes at once, not one: a different
corpus (`human-lr8192v2`, chr8/9 NOT excluded) and `time_conditioning: false`.

Three things to get right when loading:

- **The backbone kind is inferred from the checkpoint** (`finetune.py:353`).
  You do not pass an architecture flag.
- **`time_conditioning: true` is handled**, but only since 2026-09-13.
  `Classifier.forward` applies `b.time_embedding(sigma)` at
  `finetune.py:497-499`; before that it silently dropped a trained module on
  any such checkpoint. `hg_bissm_cos` carries 4 trained `time_embedding`
  tensors, so if you port this harness anywhere, carry that code with it.
- **Never point at `best.ckpt` of a RUNNING job.** It is rewritten every time
  validation improves; use a frozen `<epoch>-<step>.ckpt`. `hg_bissm_cos` is
  finished so its `best.ckpt` is stable; `fd_human`'s is not until it lands.

**Do not compare val/nll between the two.** They validate on different splits
(`hg38-cad-no89` vs `human-lr8192v2-gene`), so 1.0795 and `fd_human`'s figure
are not the same measurement.

## 5. What to add to our harness

Our `scripts/eval/caduceus/finetune.py` runs the GB suite and already handles
multi-seed, per-seed validation carving, a single guarded test evaluation,
multi-class heads, and per-task windows. It computes **accuracy only**. Five
concrete gaps:

1. **Metrics.** Add MCC and binary F1 alongside accuracy, selected per task.
   `evaluate()` currently returns top-1 accuracy (`finetune.py:639`). The
   cleanest change is to have it return predictions and labels, and score
   outside, so the metric is data rather than control flow. Both are available
   in `sklearn.metrics` (`matthews_corrcoef`, `f1_score(average="binary")`).
   Note `f1_binary` needs the positive class defined -- follow the label
   convention in the HF data, do not assume 1.
2. **A loader** mirroring `scripts/eval/caduceus/genomic_benchmarks.py`:
   `_open_task` / `task_stats` / `load_task` over
   `InstaDeepAI/nucleotide_transformer_downstream_tasks`. Keep the same
   signatures and the harness picks it up unchanged. Columns are `sequence`,
   `name`, `label`, `task`.
3. **A reference table** of the Table 2 numbers (&sect;5) so each run prints its
   delta, exactly as `REFERENCE` does in `genomic_benchmarks.py:126`.
4. **Reporting.** Emit `max - min` over seeds as the error bar **and** label the
   field so nobody reads it as an sd. Keep the sd too if you like, under a
   different key. Our GB result JSON writes `accuracy_std`; do not reuse that
   name for this.
5. **Run with `--seeds 1,2,3,4,5,6,7,8,9,10 --epochs 20`.** Both are flags we
   already have; no code change.

**Do not try to run our checkpoint inside the Caduceus repo.** Their
`dna_embedding_caduceus` path loads a Caduceus model config; our BD3-LM BiSSM
checkpoint is a different architecture and will not load. Extending our harness
is the supported route.

## 6. The numbers to beat (Caduceus, Table 2)

Mean over 10 seeds, error bar = max &minus; min. Higher is better throughout.

| task | metric | Caduceus-Ph | Caduceus-PS |
|---|---|---|---|
| H3 | mcc | 0.815 &plusmn; 0.048 | 0.799 &plusmn; 0.029 |
| H3K14ac | mcc | 0.631 &plusmn; 0.026 | 0.541 &plusmn; 0.212 |
| H3K36me3 | mcc | 0.601 &plusmn; 0.129 | 0.609 &plusmn; 0.109 |
| H3K4me1 | mcc | 0.523 &plusmn; 0.039 | 0.488 &plusmn; 0.102 |
| H3K4me2 | mcc | 0.487 &plusmn; 0.170 | 0.388 &plusmn; 0.101 |
| H3K4me3 | mcc | 0.544 &plusmn; 0.045 | 0.440 &plusmn; 0.202 |
| H3K79me3 | mcc | 0.697 &plusmn; 0.077 | 0.676 &plusmn; 0.026 |
| H3K9ac | mcc | 0.622 &plusmn; 0.030 | 0.604 &plusmn; 0.048 |
| H4 | mcc | 0.811 &plusmn; 0.022 | 0.789 &plusmn; 0.020 |
| H4ac | mcc | 0.621 &plusmn; 0.054 | 0.525 &plusmn; 0.240 |
| Enhancer | mcc | 0.546 &plusmn; 0.073 | 0.491 &plusmn; 0.066 |
| Enhancer types | mcc | 0.439 &plusmn; 0.054 | 0.416 &plusmn; 0.095 |
| Promoter: All | f1 | 0.970 &plusmn; 0.004 | 0.967 &plusmn; 0.004 |
| Promoter NonTATA | f1 | 0.969 &plusmn; 0.011 | 0.968 &plusmn; 0.006 |
| Promoter TATA | f1 | 0.953 &plusmn; 0.016 | 0.957 &plusmn; 0.015 |
| Splice All | acc | 0.940 &plusmn; 0.027 | 0.927 &plusmn; 0.021 |
| Splice Acceptor | f1 | 0.937 &plusmn; 0.033 | 0.936 &plusmn; 0.077 |
| Splice Donor | f1 | 0.948 &plusmn; 0.025 | 0.874 &plusmn; 0.289 |

**Read the error bars before reading the means.** Several are enormous:
Caduceus-PS on Splice Donor is 0.874 &plusmn; 0.289 and on H4ac 0.525 &plusmn;
0.240. A range that wide over 10 seeds means some seeds collapsed. A single-seed
run on those tasks tells you essentially nothing, and a mean-only comparison
against them is not meaningful. The histone tasks (mcc 0.4-0.8) are where the
headroom is; the promoter and splice tasks are near ceiling for everyone.

## 7. Gotchas

1. **The error bar is a range, not an sd.** Stated twice on purpose.
2. **Per-task `max_length`.** 200 to 600 across tasks. Sizing one window for the
   whole suite will pad the 200 nt tasks by 3x and change their numbers. Our
   harness sizes the window per task from **train only** (fixed 2026-09-21); keep
   that.
3. **Two tasks are 3-class.** `enhancers_types` and `splice_sites_all`. Our
   `num_classes` comes from train labels, which is correct, but check the head
   width actually lands at 3 for those.
4. **`promoter_tata` has 5,517 train rows** -- by far the smallest, and the one
   most likely to produce a wide seed range. Expect it to behave like
   `dummy_mouse` does on GB.
5. **`f1_binary` needs a positive class.** Check the label convention in the HF
   data rather than assuming label 1 is positive; an inverted F1 looks
   plausible and is wrong.
6. **10 seeds x 18 tasks = 180 fine-tuning runs.** At GB's scale that is a large
   job; budget it and consider running the 12 mcc tasks first, since that is
   where the signal is.
7. **Sweeping lr/batch is part of their protocol, not a shortcut.** Table 6
   picks per task. If you fix a single setting across all 18, say so -- it is a
   defensible simplification but it is not what the published row did.

## 8. Contamination, stated up front

These tasks are human genomic sequence. Our checkpoints are pretrained on hg38,
as are Caduceus, HyenaDNA and the Nucleotide Transformer itself, so this is
common-mode and standard practice for the benchmark. It is not a reason to
discount the comparison, but any absolute claim of generalisation needs the
caveat. We have one checkpoint (`hg38-cad-no89`) trained with chr8/chr9
excluded; if the NT task intervals can be mapped to chromosomes, scoring that
slice separately is a control no published row in Table 2 can offer -- and it is
why the primary checkpoint above is the chr8/9-excluded one.

## 9. Contact points

- our GB harness, to extend: `scripts/eval/caduceus/finetune.py`
- the loader to mirror: `scripts/eval/caduceus/genomic_benchmarks.py`
- our GB handoff, same structure: `docs/EVO2_BASELINE_HANDOFF.md` is the
  infilling one; GB numbers live in `results/caduceus/genomic_benchmarks_ft/`
- Caduceus reference: <https://github.com/kuleshov-group/caduceus>,
  arXiv:2403.03234
