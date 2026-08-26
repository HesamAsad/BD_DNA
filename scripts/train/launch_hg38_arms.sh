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

# -U iclr_2026: the group's advance reservation. WITHOUT IT jobs sit in PEND
# with "Not enough job slot(s) while advance reservation is active: 9 hosts"
# -- the reservation holds those hosts out of the general pool, so a job that
# does not ask for it is blocked BY the reservation instead of being served by
# it. All seven jobs queued on 2026-08-26 went PEND -> RUN the instant
# `bmod -U iclr_2026` was applied, with 1100 of 1280 reserved CPUs idle.
# Override with RSV= to use a different reservation, or RSV="" for none.
# Wall clock, overriding each script's `#BSUB -W`. Needed because the esub
# REFUSES bmod on a running job ("Request aborted by esub"), with or without
# -G and -gpu -- so a limit that turns out to be too short cannot be raised in
# place, only by killing and resuming. Set it correctly at submission.
# BiSSM-BD is the arm that needs this: its bidirectional scan runs at ~0.56
# it/s against uSSM-BD's 2.39, so a full epoch is ~135 h, not the ~26 h that a
# resume-inflated early average suggested.
WALL=${WALL:-}
WALL_ARG=""
[ -n "$WALL" ] && WALL_ARG="-W $WALL"
RSV=${RSV-iclr_2026}
RSV_ARG=""
[ -n "$RSV" ] && RSV_ARG="-U $RSV"

# ---- shared across every arm; changing one of these changes the comparison --
DATA_TRAIN=hg38-caduceus
DATA_VALID=hg38-caduceus
BLOCK_SIZE=256
# Context length is a knob so the 1024 and 32768 campaigns reuse this launcher.
# TOKENS_PER_STEP is held FIXED across lengths, which is what makes the three
# campaigns comparable: the corpus is a fixed 35.67e9 tokens, so a constant
# 524,288 tokens per step gives exactly 68,042 steps at every length.
#
#     L=1,024   34,837,504 windows / batch 512 = 68,042 steps
#     L=8,192    4,354,688 windows / batch  64 = 68,042 steps
#     L=32,768   1,088,672 windows / batch  16 = 68,042 steps
#
# Same steps, same tokens, same epochs -- context length is then the only
# variable. MICRO_BATCH follows as GLOBAL_BATCH/16, which holds accumulation at
# 4 everywhere, so VAL_EVERY=500 stays 125 optimizer steps at every length too.
LENGTH=${LENGTH:-8192}
TOKENS_PER_STEP=${TOKENS_PER_STEP:-524288}
GLOBAL_BATCH=$(( TOKENS_PER_STEP / LENGTH ))
GPUS=4
if (( GLOBAL_BATCH < 16 || GLOBAL_BATCH % 16 != 0 )); then
  echo "FATAL: L=$LENGTH gives global batch $GLOBAL_BATCH, which is not a"
  echo "       multiple of 16 (= $GPUS gpus x accumulation 4). Pick a length"
  echo "       that divides $TOKENS_PER_STEP into a multiple of 16."
  exit 2
fi
# 4,354,688 windows / 64 = 68,042.0 optimizer steps = exactly 1.00 epochs over
# the WINDOW LIST. Not one pass over the genome: scripts/data/audit_hg38_corpus.py
# shows the 2^20 stretch makes those 35.67 Gb of windows cover only 2.34 Gb of
# distinct sequence, so this is ~15.2 passes over the genome. Do not reason
# about overfitting or data efficiency from the word "epoch" here.
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
  # micro 4, not the launcher's default 8. Two reasons, both from widening this
  # arm to small_xf_matched: 832/13 is ~8% more activation and attention memory
  # than the 768/12 the micro-8 setting was proven at, and no peak-memory
  # telemetry survives from those runs to check it against the H200's 139.7 GiB
  # -- an OOM would cost a multi-hour queue wait to discover. Gradient
  # accumulation is exact at a fixed global batch, so this changes nothing but
  # speed, and this arm has ~4x wall-time headroom. It also makes accum=4
  # uniform across all five arms.
  "hg_xf_bd|scripts/train/train_dna_bd3lm_prok_tuned.sh|4|MODEL=small_xf_matched,LR=3e-4,BETA2=0.999,WEIGHT_DECAY=0,BLOCK_SIZE=$BLOCK_SIZE"
)

# docs/, not results/ -- results/ is gitignored (.gitignore:6), so a record
# written there would never reach the repository, which is the one thing this
# record exists to do.

# ---- collidable-environment guard -------------------------------------------
# `bsub -env "all,..."` snapshots the SUBMITTING shell, so any bare variable the
# target script reads and we do not explicitly set is inherited silently. This
# is not hypothetical: GNU screen exports WINDOW=<n>, it rode in through
# `-env all`, and it passed `--window 0` to the benchmark harness for an entire
# campaign. WINDOW is still set to 0 in a screen session today.
#
# Fail before submitting rather than discover it in a result. Names we set
# ourselves are fine -- ours win, because they come after `all` in the -env
# string. Everything else that the target reads and the environment happens to
# hold is a collision.
COLLIDABLE="L LIMIT WINDOW MODEL EPOCHS SEEDS SEED LENGTH BATCH BATCH_SIZE
            STEPS MAX_STEPS LR PRESET TASKS LABEL CKPT DROPOUT LAYER POOLING
            EMA ATTN SWEEP SCHEDULER"
guard_env () { # $1 = comma-separated VAR=VAL string we are about to pass
  local passing=",$(echo "$1" | tr ',' '\n' | cut -d= -f1 | tr '\n' ',')"
  local bad=""
  for v in $COLLIDABLE; do
    if printenv "$v" >/dev/null 2>&1; then
      case "$passing" in *",$v,"*) : ;; *) bad="$bad $v=$(printenv "$v")";; esac
    fi
  done
  if [ -n "$bad" ]; then
    echo "FATAL: these are set in this shell, are READ by the target script,"
    echo "       and are not among the values being passed -- they would be"
    echo "       inherited silently through 'bsub -env all':"
    for kv in $bad; do echo "         $kv"; done
    echo "       unset them and resubmit."
    exit 2
  fi
}

[ "${DRY:-0}" = "1" ] || mkdir -p logs docs/runs
RECORD="docs/runs/hg38_arms_$(date +%Y%m%d-%H%M%S).txt"
REC () { [ "${DRY:-0}" = "1" ] || echo "$@" >> "$RECORD"; }
[ "${DRY:-0}" = "1" ] || echo "# submitted $(date) from $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY)')" > "$RECORD"
REC "# shared: $COMMON"

# ONLY=hg_xf_bd+hg_ussm_bd restricts the submission to named arms, for
# restarting a subset without disturbing arms that are running correctly.
ONLY=${ONLY:-}; ONLY=",${ONLY//+/,},"
for entry in "${ARMS[@]}"; do
  IFS='|' read -r name script _micro env <<< "$entry"
  # MICRO_BATCH is derived from the length, not taken from the table: the
  # table's value is only right for L=8192. GLOBAL_BATCH/16 holds accum at 4.
  micro=$(( GLOBAL_BATCH / 16 ))
  if [ "$ONLY" != ",," ]; then
    case "$ONLY" in *",$name,"*) : ;; *) continue;; esac
  fi
  if (( GLOBAL_BATCH % (micro * GPUS) != 0 )); then
    echo "FATAL: $name global batch $GLOBAL_BATCH is not divisible by "
    echo "       micro $micro x $GPUS gpus; dataloader.py:764 rejects this."
    exit 2
  fi
  accum=$(( GLOBAL_BATCH / (micro * GPUS) ))
  val_every=$(( VAL_EVERY_OPT_STEPS * accum ))
  # RUN_DIR is PINNED and carries no job id. Every launcher defaults it to a
  # path containing ${LSB_JOBID}, so a resubmitted job lands in a fresh
  # directory, finds no last.ckpt, and silently restarts from step 0 --
  # discarding up to 43 hours. checkpointing.resume_ckpt_path is
  # ${save_dir}/checkpoints/last.ckpt (config.yaml:103) and save_dir is the
  # hydra run dir, so a stable path is the whole resume mechanism. These runs
  # are long enough, and NCCL timeouts frequent enough, that this matters.
  # It also puts the five arms under one parent, which is what
  # `training_curves.py --glob "outputs/hg38-caduceus/*"` expects, and the
  # leaf names match its INFER table.
  # L=8192 keeps the original path so the arms already running there still
  # resume from their last.ckpt; other lengths get their own root.
  if [ "$LENGTH" = "8192" ]; then
    run_dir="outputs/hg38-caduceus/$name"
  else
    run_dir="outputs/hg38-caduceus-L${LENGTH}/$name"
  fi
  full="$COMMON,MICRO_BATCH=$micro,VAL_EVERY=$val_every,RUN_DIR=$run_dir,$env"
  # Past ~L=16384 the stored boundary prefill does not fit: the SSM launcher's
  # own note measures 138 of the H200's 139.72 GiB at L=32768 micro 2, which
  # will not survive DDP buffers. Recomputing costs ~10% throughput and saves
  # ~36% of peak. Only the SSM arms read this flag.
  if (( LENGTH >= 16384 )) && [[ "$script" == *ssm_baseline* ]]; then
    full="$full,CHECKPOINT_PREFILL=true"
  fi
  echo "  $name: micro=$micro accum=$accum -> val every $val_every micro-batches"
  echo "         = every $VAL_EVERY_OPT_STEPS optimizer steps (same grid for all arms)"
  echo "         run_dir=$run_dir$([ -f "$run_dir/checkpoints/last.ckpt" ] && echo '  (will RESUME from last.ckpt)')"
  if [ "${DRY:-0}" = "1" ]; then
    echo "bsub -J $name -G s10396 $RSV_ARG $WALL_ARG $DEP -env \"all,$full\" < $script"
    continue
  fi
  # -env "all,..." keeps the submitting environment (module paths, HOME) and
  # layers the arm's settings on top. Without `all` the job starts with an
  # almost empty environment and the python on PATH is the system one.
  guard_env "$full"
  out=$(bsub -J "$name" -G s10396 $RSV_ARG $WALL_ARG $DEP -env "all,$full" < "$script" 2>&1)
  echo "$out"
  jobid=$(echo "$out" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+" | head -1)
  REC "${jobid:-SUBMIT_FAILED} $name $script $full"
done

echo
[ "${DRY:-0}" = "1" ] || echo "recorded to $RECORD"
[ "${DRY:-0}" = "1" ] || cat "$RECORD"
