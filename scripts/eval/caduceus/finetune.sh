#!/usr/bin/env bash
#BSUB -J gb_ft
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 72:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
# GPU MODEL: unconstrained, deliberately, since 2026-09-13. This line used to
# read `...:gmodel=NVIDIAH200`, which made every GenomicBenchmarks evaluation
# queue behind the `iclr_2026` advance reservation on the farm-gpu050x hosts
# (window 8/22-9/30) while nine non-H200 hosts sat completely idle. Nothing on
# this path needs an H200: measured host peak across the historical
# gb_ft_*/gb_probe_*.out logs is 2.4-2.7 GB for the fine-tune and 0.3-6.5 GB for
# the probe, against 80 GB of device. Put the constraint back only for a job
# whose memory you have actually measured as needing it.
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_ft_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_ft_%J.err
#
# GenomicBenchmarks fine-tuning. Drives scripts/eval/caduceus/finetune.py.
#
# WHAT CHANGED FROM THE VERSION THAT PRODUCED results/caduceus/
# genomic_benchmarks_ft/{ft-human,ft-human32k,ft-prok-denovo}.json:
#
#  * GB_MAX_TRAIN / GB_MAX_TEST are NO LONGER honoured silently. The previous
#    script forwarded whatever `bsub -env all` happened to carry in, and 20000
#    / 8000 were in the environment. That trained the three weakest tasks on
#    8.6-16% of their data and the caps appeared in no log and no JSON -- they
#    had to be recovered by factoring the accuracy denominators. Capping now
#    requires GB_ALLOW_CAPS=1 as well, and every value lands in the summary
#    JSON's "args" block either way.
#  * -W 12:00 -> 24:00, mem 64000 -> 128000, -n 8 -> 16. Uncapped training on
#    human_ensembl_regulatory alone is 231k sequences; the harness also holds a
#    pristine backbone snapshot plus one best-validation state on the host.
#  * PRESET defaults to v2 (Caduceus's 10 epochs / batch 128 / cosine+warmup,
#    plus length-bucketed padding, per-group grad clipping, a head LayerNorm,
#    stratified validation and step-level early stopping). PRESET=legacy
#    reproduces the old harness exactly.
#
# SEPARATORS. `bsub -env "all, VAR=val"` splits its own argument on COMMAS, so a
# value containing one is mangled there. SEEDS and SWEEP both want commas, so
# this script accepts `+` for `,` and `^` for `;` in TASKS, SEEDS and SWEEP --
# the same substitution TASKS already used. Either spelling works when the
# variables are exported into bsub's environment instead (`VAR=x bsub -env all`,
# the shorthand in docs/lsf_conventions.md section 3), which is the safer habit.
#
# Two shapes, both driven from here:
#   sweep  SWEEP='backbone_lr=1e-5+3e-5+1e-4+3e-4^head_lr=1e-3+3e-3' \
#          TASKS=human_ocr_ensembl+human_enhancers_ensembl SEEDS=0
#   final  SEEDS=0+1+2   (no SWEEP; scores every task on all 8)
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface/datasets
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" logs

CKPT=${CKPT:?set CKPT}
LABEL=${LABEL:?set LABEL}
TASKS=${TASKS:-all}; TASKS=${TASKS//+/,}
PRESET=${PRESET:-v2}
SEEDS=${SEEDS:-0}; SEEDS=${SEEDS//+/,}
SWEEP=${SWEEP:-}; SWEEP=${SWEEP//+/,}; SWEEP=${SWEEP//^/;}
EXTRA=(--preset "$PRESET" --seeds "$SEEDS")

# Optimisation / readout knobs. Anything left unset takes the preset's value,
# which finetune.py records in the JSON, so an omitted variable here is still
# recoverable from results/ later.
[ -n "${EPOCHS:-}" ]            && EXTRA+=(--epochs "$EPOCHS")
[ -n "${BATCH_SIZE:-}" ]        && EXTRA+=(--batch-size "$BATCH_SIZE")
[ -n "${EVAL_BATCH_SIZE:-}" ]   && EXTRA+=(--eval-batch-size "$EVAL_BATCH_SIZE")
[ -n "${BACKBONE_LR:-}" ]       && EXTRA+=(--backbone-lr "$BACKBONE_LR")
# PATIENCE=0 disables early stopping (finetune.py:615 treats 0 as falsy). The
# v2 preset sets 8, which on human_ocr_ensembl halted every run at step 4,900
# of 9,830 -- half the epoch budget unused, on the task where the backbone most
# needs the adaptation.
[ -n "${PATIENCE:-}" ]          && EXTRA+=(--patience "$PATIENCE")
[ -n "${HEAD_LR:-}" ]           && EXTRA+=(--head-lr "$HEAD_LR")
[ -n "${WEIGHT_DECAY:-}" ]      && EXTRA+=(--weight-decay "$WEIGHT_DECAY")
[ -n "${DROPOUT:-}" ]           && EXTRA+=(--dropout "$DROPOUT")
# Validation hygiene. cohn carves a 10% val slice = 2,084 rows, SE ~0.0095,
# and best-of-4-epoch selection on that is a coin flip -- measured as the cause
# of the persistent cohn val->test drop (there is NO train/test distribution
# shift: a 1..4-mer domain classifier scores AUC 0.494).
[ -n "${STRATIFIED_VAL:-}" ]   && EXTRA+=(--stratified-val)
[ -n "${VAL_FRACTION:-}" ]     && EXTRA+=(--val-fraction "$VAL_FRACTION")
[ -n "${EVALS_PER_EPOCH:-}" ]  && EXTRA+=(--evals-per-epoch "$EVALS_PER_EPOCH")
# RC_TTA averages the two strands at SCORING time. It measured -0.0027 alone,
# but that was WITHOUT rc_aug: evaluate() notes the RC view is uncalibrated
# because the model never saw RC. With RC_AUG=0.5 the fine-tune HAS seen both
# strands, so train and eval finally match -- worth retesting together.
[ -n "${RC_AVERAGE:-}" ]       && EXTRA+=(--rc-average "$RC_AVERAGE")
[ "${PAD_INVARIANT:-0}" = "1" ] && EXTRA+=(--pad-invariant)
[ -n "${SCAN_PATH:-}" ]        && EXTRA+=(--scan-path "$SCAN_PATH")
[ -n "${POOLING:-}" ]           && EXTRA+=(--pooling "$POOLING")
[ -n "${LAYER:-}" ]             && EXTRA+=(--layer "$LAYER")
[ -n "${SIGMA:-}" ]             && EXTRA+=(--sigma "$SIGMA")
# These two had NO hook, so `bsub -env "...,SCHEDULER=cosine"` was accepted and
# silently ignored -- the same shape as the DATA_TRAIN and GB_MAX_TRAIN bugs
# this harness has already been bitten by. It matters here because
# build_schedule returns a constant 1.0 when scheduler='none', making
# warmup_frac inert: every legacy (batch 16) run records warmup_frac 0.05 and
# received no warmup at all, so the backbone_lr ceiling was never established
# with warmup.
[ -n "${SCHEDULER:-}" ]        && EXTRA+=(--scheduler "$SCHEDULER")
[ -n "${WARMUP_FRAC:-}" ]      && EXTRA+=(--warmup-frac "$WARMUP_FRAC")
# EPOCHS and EPOCHS_OVERRIDE both emit --epochs. Setting both silently emitted
# "--epochs A --epochs B" and argparse kept B -- a confusion hazard with no
# warning, so refuse it outright rather than pick one.
if [ -n "${EPOCHS:-}" ] && [ -n "${EPOCHS_OVERRIDE:-}" ]; then
  echo "ERROR: set EPOCHS or EPOCHS_OVERRIDE, not both (got $EPOCHS / $EPOCHS_OVERRIDE)" >&2
  exit 2
fi
[ -n "${EPOCHS_OVERRIDE:-}" ]  && EXTRA+=(--epochs "$EPOCHS_OVERRIDE")
[ -n "${PAD_TO:-}" ]            && EXTRA+=(--pad-to "$PAD_TO")
# The checkpoint never saw [PAD] in DNA pretraining, so its embedding is
# still at init: a true PAD may be WORSE than the N nucleotide, not better.
# Empirical question, hence a flag rather than a default.
[ -n "${PAD_TOKEN:-}" ]         && EXTRA+=(--pad-token "$PAD_TOKEN")
[ -n "${PAD_SIDE:-}" ]          && EXTRA+=(--pad-side "$PAD_SIDE")
[ -n "${GB_WINDOW_FROM:-}" ]    && EXTRA+=(--window-from "$GB_WINDOW_FROM")
[ "${LOG_LENGTH:-0}" = "1" ]    && EXTRA+=(--log-length)
[ -n "${LENGTH_BINS:-}" ]       && EXTRA+=(--length-bins "$LENGTH_BINS")
[ "${RC_TTA:-0}" = "1" ]        && EXTRA+=(--rc-tta)
[ -n "${RC_AUG:-}" ]            && EXTRA+=(--rc-aug "$RC_AUG")
[ -n "$SWEEP" ]                 && EXTRA+=(--sweep "$SWEEP")
[ -n "${SWEEP_SEEDS:-}" ]       && EXTRA+=(--sweep-seeds "$SWEEP_SEEDS")
# NB: do not call this WINDOW -- GNU screen exports WINDOW=<n> and
# `bsub -env all` carries it in, which silently passed --window 0.
[ -n "${GB_WINDOW:-}" ]         && EXTRA+=(--window "$GB_WINDOW")

# Subsampling is opt-in twice over. GB_MAX_TRAIN/GB_MAX_TEST arriving through
# `bsub -env all` from an interactive shell is exactly how the first campaign
# came to train on 8.6% of human_ensembl_regulatory without anyone noticing.
if [ -n "${GB_MAX_TRAIN:-}${GB_MAX_TEST:-}" ]; then
  if [ "${GB_ALLOW_CAPS:-0}" = "1" ]; then
    echo "WARNING: subsampling ON -- max_train=${GB_MAX_TRAIN:-none} max_test=${GB_MAX_TEST:-none}"
    echo "WARNING: these numbers are NOT comparable to the published Caduceus row (full splits)."
    [ -n "${GB_MAX_TRAIN:-}" ] && EXTRA+=(--max-train "$GB_MAX_TRAIN")
    [ -n "${GB_MAX_TEST:-}" ]  && EXTRA+=(--max-test "$GB_MAX_TEST")
  else
    echo "IGNORING inherited GB_MAX_TRAIN=${GB_MAX_TRAIN:-} GB_MAX_TEST=${GB_MAX_TEST:-} (set GB_ALLOW_CAPS=1 to mean it)"
  fi
fi

echo "[$(date)] GenomicBenchmarks FINE-TUNE | label=$LABEL | ckpt=$CKPT"
echo "  tasks=$TASKS preset=$PRESET seeds=$SEEDS"
echo "  argv: ${EXTRA[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u scripts/eval/caduceus/finetune.py \
  --checkpoint "$CKPT" --label "$LABEL" --tasks "$TASKS" \
  ${EXTRA[@]+"${EXTRA[@]}"}
