#!/usr/bin/env bash
# Submit all five hg38 arms, with every variable pinned IN THIS FILE.
#
# WHY THIS FILE EXISTS. The first attempt (LSF 120960-120964) passed each arm's
# settings inline through `bsub -env`. LSF does not give that environment back:
# `bjobs -l` shows only the script body's DEFAULTS, so there was no way to
# confirm after the fact that an arm had actually received DATA_TRAIN,
# MAX_STEPS or LR. Those five were killed unverified. A launcher in git is
# auditable, diffable, and re-runnable; an inline -env is none of the three.
#
# THE BUG THAT MADE IT MATTER. Two of the three launchers had no DATA_TRAIN
# knob at all -- train_dna_bd3lm_prok_tuned.sh and train_dna_ar_transformer.sh
# hardcoded `data=carbon-prokaryote` with dna_num_files=1 / dna_max_rows=400000
# (2.09% of the prokaryote corpus). A DATA_TRAIN in the environment was
# accepted by the shell and read by nothing, so both Transformer arms would
# have trained on 2% of the WRONG corpus while carrying an hg38 job name and
# being plotted against three arms that did read it. Both now implement the
# knob and both now refuse variables they cannot honour.
#
# PARAMETER MATCHING. All five arms are within 1% of 100M parameters. That
# needed a fix: the BD Transformer defaulted to model=small (hidden 768), which
# is 85.0M -- 15.6% below the SSM arms -- because a Mamba-2 layer carries more
# parameters than an attention layer of the same width. Both Transformer arms
# now use configs/model/small_xf_matched.yaml (832/13, 99.8M, head_dim 64).
#
# WHAT IS DELIBERATELY NOT MATCHED ACROSS ARMS. Each arm uses the optimiser
# recipe that its own tuning found, not one shared recipe: the SSM arms and
# Transformer-AR take lr 1e-3 / beta2 0.95 / wd 0.1, and Transformer-BD takes
# 3e-4 / 0.999 / 0 because that is what its own pilot preferred. Corpus,
# sequence length, global batch, step count, and the validation protocol ARE
# matched, because those are what make the comparison a comparison.
#
# Usage:
#   scripts/train/launch_hg38_arms.sh            # submit, waiting on the build
#   AFTER=0 scripts/train/launch_hg38_arms.sh    # submit with no dependency
#   DRY=1  scripts/train/launch_hg38_arms.sh     # print the bsub lines only
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

# ---- shared across every arm; changing one of these changes the comparison --
DATA_TRAIN=hg38-caduceus
DATA_VALID=hg38-caduceus
LENGTH=8192
BLOCK_SIZE=256
GLOBAL_BATCH=64
# 4,354,688 windows / 64 = 68,042.0 optimizer steps = EXACTLY 1.00 epochs.
# Not a round number by accident -- it is the corpus, and it divides exactly.
MAX_STEPS=68042
# val_check_interval counts MICRO-batches, NOT optimizer steps -- so a fixed
# VAL_EVERY means different things to arms with different micro batches, and
# our arms do differ (the Transformer-BD launcher defaults to 8, the rest to
# 4). Passing one VAL_EVERY to all five would have validated Transformer-BD
# every 250 optimizer steps and everyone else every 125, putting its curve on a
# different x grid than the four it is being plotted against. So the INTENT is
# pinned in optimizer steps and VAL_EVERY is derived per arm below.
VAL_EVERY_OPT_STEPS=125          # ~544 validation points over 68,042 steps
VAL_BATCHES=128
# Cost of that cadence: one validation is 128x4 = 512 forward sequences against
# 125x64x3 = 24,000 sequence-units of training in the same interval, i.e. ~2%.
GPUS=4
# The build job. Every arm waits on it; an arm that starts early would find no
# cache and rebuild 35.7e9 nt inside a training job.
BUILD_JOB=${BUILD_JOB:-120910}
AFTER=${AFTER:-1}

DEP=""
if [ "$AFTER" = "1" ]; then
  if bjobs "$BUILD_JOB" >/dev/null 2>&1; then
    DEP="-w done($BUILD_JOB)"
  else
    echo "build job $BUILD_JOB is no longer in the queue; checking the cache"
    for split in train validation; do
      path="data_cache/carbon/${DATA_TRAIN}_${split}_bs${LENGTH}_wrapped_specialFalse.dat"
      if [ ! -d "$path" ]; then
        echo "FATAL: no dependency to wait on and $path does not exist."
        echo "       Rerun the build, or pass AFTER=0 if you know it is ready."
        exit 2
      fi
    done
    echo "  both splits present; submitting with no dependency"
  fi
fi

COMMON="DATA_TRAIN=$DATA_TRAIN,DATA_VALID=$DATA_VALID,LENGTH=$LENGTH"
COMMON="$COMMON,GLOBAL_BATCH=$GLOBAL_BATCH,MAX_STEPS=$MAX_STEPS"
COMMON="$COMMON,VAL_BATCHES=$VAL_BATCHES"

# name | script | micro batch | arm-specific env
# MICRO_BATCH is stated per arm rather than left to each script's default,
# because it sets accumulate_grad_batches and therefore the validation grid.
ARMS=(
  "hg_ussm_ar|scripts/train/train_dna_ssm_baseline.sh|4|OBJECTIVE=ar,DIRECTION=uni,LR=1e-3,BETA2=0.95,WEIGHT_DECAY=0.1"
  "hg_ussm_bd|scripts/train/train_dna_ssm_baseline.sh|4|OBJECTIVE=bd3lm,DIRECTION=uni,LR=1e-3,BETA2=0.95,WEIGHT_DECAY=0.1,BLOCK_SIZE=$BLOCK_SIZE"
  "hg_bissm_bd|scripts/train/train_dna_ssm_baseline.sh|4|OBJECTIVE=bd3lm,DIRECTION=bi,LR=1e-3,BETA2=0.95,WEIGHT_DECAY=0.1,BLOCK_SIZE=$BLOCK_SIZE"
  "hg_xf_ar|scripts/train/train_dna_ar_transformer.sh|4|MODEL=small_xf_matched,LR=1e-3,BETA2=0.95,WEIGHT_DECAY=0.1"
  "hg_xf_bd|scripts/train/train_dna_bd3lm_prok_tuned.sh|8|MODEL=small_xf_matched,LR=3e-4,BETA2=0.999,WEIGHT_DECAY=0,BLOCK_SIZE=$BLOCK_SIZE"
)

mkdir -p logs results/runs
RECORD="results/runs/hg38_arms_$(date +%Y%m%d-%H%M%S).txt"
echo "# submitted $(date) from $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY)')" > "$RECORD"
echo "# shared: $COMMON" >> "$RECORD"

for entry in "${ARMS[@]}"; do
  IFS='|' read -r name script micro env <<< "$entry"
  if (( GLOBAL_BATCH % (micro * GPUS) != 0 )); then
    echo "FATAL: $name global batch $GLOBAL_BATCH is not divisible by "
    echo "       micro $micro x $GPUS gpus; dataloader.py:764 rejects this."
    exit 2
  fi
  accum=$(( GLOBAL_BATCH / (micro * GPUS) ))
  val_every=$(( VAL_EVERY_OPT_STEPS * accum ))
  full="$COMMON,MICRO_BATCH=$micro,VAL_EVERY=$val_every,$env"
  echo "  $name: micro=$micro accum=$accum -> val every $val_every micro-batches"
  echo "         = every $VAL_EVERY_OPT_STEPS optimizer steps (same grid for all arms)"
  if [ "${DRY:-0}" = "1" ]; then
    echo "bsub -J $name -G s10396 $DEP -env \"all,$full\" < $script"
    continue
  fi
  # -env "all,..." keeps the submitting environment (module paths, HOME) and
  # layers the arm's settings on top. Without `all` the job starts with an
  # almost empty environment and the python on PATH is the system one.
  out=$(bsub -J "$name" -G s10396 $DEP -env "all,$full" < "$script" 2>&1)
  echo "$out"
  jobid=$(echo "$out" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+" | head -1)
  echo "${jobid:-SUBMIT_FAILED} $name $script $full" >> "$RECORD"
done

echo
echo "recorded to $RECORD"
[ "${DRY:-0}" = "1" ] || cat "$RECORD"
