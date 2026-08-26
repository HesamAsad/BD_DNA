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

# -U iclr_2026: the group's advance reservation. WITHOUT IT jobs sit in PEND
# with "Not enough job slot(s) while advance reservation is active: 9 hosts"
# -- the reservation holds those hosts out of the general pool, so a job that
# does not ask for it is blocked BY the reservation instead of being served by
# it. All seven jobs queued on 2026-08-26 went PEND -> RUN the instant
# `bmod -U iclr_2026` was applied, with 1100 of 1280 reserved CPUs idle.
# Override with RSV= to use a different reservation, or RSV="" for none.
RSV=${RSV-iclr_2026}
RSV_ARG=""
[ -n "$RSV" ] && RSV_ARG="-U $RSV"

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

# The 8 GenomicBenchmarks tasks, fanned out one job each per arm rather than
# one job per arm covering all 8.
#
# WHY. LSF 120181 spent 56,806 s (15.8 h) on ONE task
# (human_ensembl_regulatory) at preset=legacy, ONE seed, across an 8-point
# sweep -- about 2 h per config at legacy's 4 epochs. At the 10 epochs Caduceus
# uses that is ~5 h per seed, so ~25 h for five seeds on that task ALONE. A
# single job covering all 8 tasks could not finish inside any sane limit, and
# under the old 24 h it would have been killed partway through task 1 of 8,
# leaving a results file that looks like a legitimate partial run.
#
# 5 arms x 8 tasks = 40 one-GPU jobs, each 5 seeds, each independent, each free
# to schedule whenever a GPU frees. The small tasks finish in minutes.
# task | short code. The code only names the LSF job: truncating the task name
# instead collides (`human_enhancers_cohn` and `human_enhancers_ensembl` share
# their first 15 characters), and two dozen identically-named jobs is
# unmonitorable. The full task name still goes in LABEL, which is what names
# the result JSON.
GB_TASKS=(
  "dummy_mouse_enhancers_ensembl|dummy"
  "demo_coding_vs_intergenomic_seqs|coding"
  "demo_human_or_worm|worm"
  "human_enhancers_cohn|cohn"
  "human_enhancers_ensembl|enhens"
  "human_ensembl_regulatory|reg"
  "human_nontata_promoters|prom"
  "human_ocr_ensembl|ocr"
)

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

mkdir -p logs/eval docs/runs
RECORD="docs/runs/hg38_eval_$(date +%Y%m%d-%H%M%S).txt"
echo "# submitted $(date) from $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY)')" > "$RECORD"

submit () { # jobname script env
  if [ "${DRY:-0}" = "1" ]; then
    echo "bsub -J $1 -G s10396 $RSV_ARG -env \"all,$3\" < $2"
    return
  fi
  guard_env "$3"
  out=$(bsub -J "$1" -G s10396 $RSV_ARG -env "all,$3" < "$2" 2>&1)
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
  n=0
  for entry in "${ARMS[@]}"; do
    IFS='|' read -r arm leaf _ <<< "$entry"
    ckpt="$RUNS/$leaf/checkpoints/$WHICH.ckpt"
    for spec in "${GB_TASKS[@]}"; do
      task="${spec%%|*}"; code="${spec##*|}"
      # LABEL carries the full task name so the 40 result JSONs cannot
      # overwrite each other -- finetune.py names its output from the label.
      submit "hg_gb_${arm}_${code}" scripts/eval/caduceus/finetune.sh \
        "CKPT=$ckpt,LABEL=hg38_${arm}_${task},TASKS=$task,PRESET=$PRESET,SEEDS=$SEEDS,EPOCHS=$EPOCHS"
      n=$((n + 1))
    done
  done
  echo "  submitted $n GenomicBenchmarks jobs (${#ARMS[@]} arms x ${#GB_TASKS[@]} tasks, $SEEDS seeds each)"
fi

echo
echo "recorded to $RECORD"
[ "${DRY:-0}" = "1" ] || cat "$RECORD"
