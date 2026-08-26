#!/usr/bin/env bash
# Evaluate all five hg38 arms: perplexity on the hg38 validation split, then
# GenomicBenchmarks by fine-tuning. Companion to
# scripts/train/launch_hg38_arms.sh, and pinned the same way -- every value in
# this file, every submission recorded to docs/runs/.
#
# THE THING THIS FILE EXISTS TO PREVENT. Both evaluators default to
# `data=carbon-prokaryote` with dna_num_files=1 / dna_max_rows=400000. Scoring
# an hg38 checkpoint without DATA_TRAIN reports its perplexity on 2% of a
# PROKARYOTE validation set, and the number looks entirely plausible sitting
# next to the others. There is no error, no warning, and nothing downstream
# that would catch it.
#
# Usage:
#   scripts/eval/eval_hg38_arms.sh              # ppl + benchmarks
#   STAGE=ppl scripts/eval/eval_hg38_arms.sh    # perplexity only
#   STAGE=gb  scripts/eval/eval_hg38_arms.sh    # benchmarks only
#   DRY=1     scripts/eval/eval_hg38_arms.sh    # print, submit nothing
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

RUNS=${RUNS:-outputs/hg38-caduceus}   # matches launch_hg38_arms.sh's pinned dirs
WHICH=${WHICH:-best}                  # best.ckpt (val/nll-selected) or last.ckpt
STAGE=${STAGE:-all}
DATA_TRAIN=hg38-caduceus
LENGTH=8192
BLOCK_SIZE=256
# Full validation split, and full benchmark data. Both defaults in this repo
# used to be silent subsamples; do not reintroduce them here.
LIMIT=1.0
PRESET=${PRESET:-v2}
# SEEDS is a LIST, not a count: finetune.sh does `SEEDS=${SEEDS//+/,}`, so
# SEEDS=5 would run ONE seed numbered 5, not five seeds. Caduceus reports a
# 5-seed mean, so 0+1+2+3+4. The `+` separator exists because `bsub -env` uses
# commas, which would otherwise split this into five environment variables.
SEEDS=${SEEDS:-0+1+2+3+4}
# Caduceus fine-tunes for 10 epochs; the old harness used 4 and hit
# best_epoch == last on 3 of 8 tasks, i.e. it was still improving when it
# stopped.
EPOCHS=${EPOCHS:-10}

# arm | run dir leaf | the CKPT_* variable ppl_ssm_baselines.sh reads
ARMS=(
  "ussm_ar|hg_ussm_ar|CKPT_USSM_AR"
  "ussm_bd|hg_ussm_bd|CKPT_USSM_BD"
  "bissm_bd|hg_bissm_bd|CKPT_BISSM"
  "xf_ar|hg_xf_ar|CKPT_XF_AR"
  "xf_bd|hg_xf_bd|CKPT_XF"
)

missing=0
PPL_ENV=""
for entry in "${ARMS[@]}"; do
  IFS='|' read -r arm leaf var <<< "$entry"
  ckpt="$RUNS/$leaf/checkpoints/$WHICH.ckpt"
  if [ ! -f "$ckpt" ]; then
    echo "  MISSING $ckpt"
    missing=$((missing + 1))
    continue
  fi
  PPL_ENV="$PPL_ENV,$var=$ckpt"
done
if [ "$missing" -gt 0 ]; then
  echo
  echo "FATAL: $missing of ${#ARMS[@]} checkpoints are missing. Evaluating a"
  echo "       partial set silently produces a table with holes that reads as"
  echo "       'these are the results'. Wait for training, or set WHICH=last."
  exit 2
fi

mkdir -p logs/eval docs/runs
RECORD="docs/runs/hg38_eval_$(date +%Y%m%d-%H%M%S).txt"
echo "# submitted $(date) from $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY)')" > "$RECORD"

submit () { # jobname script env
  if [ "${DRY:-0}" = "1" ]; then
    echo "bsub -J $1 -G s10396 -env \"all,$3\" < $2"
    return
  fi
  out=$(bsub -J "$1" -G s10396 -env "all,$3" < "$2" 2>&1)
  echo "$out"
  jobid=$(echo "$out" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+" | head -1)
  echo "${jobid:-SUBMIT_FAILED} $1 $2 $3" >> "$RECORD"
}

# ---- perplexity: one job, all five arms, on the hg38 validation split -------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "ppl" ]; then
  # XF_MODEL/XF_AR_MODEL must match the geometry these arms TRAINED with
  # (small_xf_matched, 832/13), not the script's `small` default -- loading one
  # under the other is a shape mismatch.
  # PPL_EMA is deliberately unset: ppl_ssm_baselines.sh reads training.ema from
  # each checkpoint, which is what the arm was actually trained with.
  ppl_env="DATA_TRAIN=$DATA_TRAIN,DATA_VALID=$DATA_TRAIN,L=$LENGTH"
  ppl_env="$ppl_env,BLOCK_SIZE=$BLOCK_SIZE,LIMIT=$LIMIT"
  ppl_env="$ppl_env,XF_MODEL=small_xf_matched,XF_AR_MODEL=small_xf_matched"
  ppl_env="$ppl_env${PPL_ENV}"
  submit hg_ppl_all scripts/eval/ppl_ssm_baselines.sh "$ppl_env"
fi

# ---- GenomicBenchmarks: one job per arm, fine-tuning ------------------------
# No geometry knob needed: finetune.py rebuilds the backbone from the config
# stored in the checkpoint, so each arm loads as it was trained.
# No GB_MAX_TRAIN/GB_MAX_TEST: the caps are opt-in twice over now, and every
# published number before 2026-08-25 was silently capped by exactly that path.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "gb" ]; then
  for entry in "${ARMS[@]}"; do
    IFS='|' read -r arm leaf _ <<< "$entry"
    ckpt="$RUNS/$leaf/checkpoints/$WHICH.ckpt"
    submit "hg_gb_$arm" scripts/eval/caduceus/finetune.sh \
      "CKPT=$ckpt,LABEL=hg38_$arm,PRESET=$PRESET,SEEDS=$SEEDS,EPOCHS=$EPOCHS"
  done
fi

echo
echo "recorded to $RECORD"
[ "${DRY:-0}" = "1" ] || cat "$RECORD"
