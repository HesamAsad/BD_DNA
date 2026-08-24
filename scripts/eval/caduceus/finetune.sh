#!/usr/bin/env bash
#BSUB -J gb_ft
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
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
[ -n "${HEAD_LR:-}" ]           && EXTRA+=(--head-lr "$HEAD_LR")
[ -n "${WEIGHT_DECAY:-}" ]      && EXTRA+=(--weight-decay "$WEIGHT_DECAY")
[ -n "${DROPOUT:-}" ]           && EXTRA+=(--dropout "$DROPOUT")
[ -n "${POOLING:-}" ]           && EXTRA+=(--pooling "$POOLING")
[ -n "${LAYER:-}" ]             && EXTRA+=(--layer "$LAYER")
[ -n "${PAD_TO:-}" ]            && EXTRA+=(--pad-to "$PAD_TO")
# The checkpoint never saw [PAD] in DNA pretraining, so its embedding is
# still at init: a true PAD may be WORSE than the N nucleotide, not better.
# Empirical question, hence a flag rather than a default.
[ -n "${PAD_TOKEN:-}" ]         && EXTRA+=(--pad-token "$PAD_TOKEN")
[ -n "${PAD_SIDE:-}" ]          && EXTRA+=(--pad-side "$PAD_SIDE")
[ -n "${GB_WINDOW_FROM:-}" ]    && EXTRA+=(--window-from "$GB_WINDOW_FROM")
[ "${LOG_LENGTH:-0}" = "1" ]    && EXTRA+=(--log-length)
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
