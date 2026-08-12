#!/usr/bin/env bash
#BSUB -J select_xf_ar
#BSUB -G s10396
#BSUB -q training-normal
#BSUB -n 2
#BSUB -W 1:00
#BSUB -R "select[mem>16000]"
#BSUB -R "rusage[mem=16000]"
#BSUB -M 16000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/select_xf_ar_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/select_xf_ar_%J.err
# Run only after all four pilots finish successfully. It selects the lowest
# val/NLL, submits one 8k run, then queues exact PPL and MaveDB behind it.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
PILOT_ROOT=${PILOT_ROOT:?set PILOT_ROOT}
SWEEP_TAG=${SWEEP_TAG:?set SWEEP_TAG}
RESULT_JSON="$REPO/results/transformer_ar/${SWEEP_TAG}/lr_selection.json"
FULL_DIR="$REPO/outputs/carbon-prokaryote/transformer-ar-${SWEEP_TAG}/full"

"$PYTHON" scripts/train/select_transformer_ar_lr.py \
  --pilot-root "$PILOT_ROOT" --learning-rates 1e-4 3e-4 1e-3 3e-3 \
  --output "$RESULT_JSON"
WINNER_LR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["learning_rate"])' "$RESULT_JSON")

FULL_SUBMISSION=$(bsub \
  -J "xf-ar-full-${SWEEP_TAG}" \
  -env "all,LR=$WINNER_LR,MAX_STEPS=8000,RUN_DIR=$FULL_DIR" \
  < scripts/train/train_dna_ar_transformer.sh)
FULL_JOB=$(printf '%s\n' "$FULL_SUBMISSION" | sed -n 's/Job <\([0-9][0-9]*\)>.*/\1/p')
if [[ -z "$FULL_JOB" ]]; then
  echo "FATAL: could not parse full-run job ID: $FULL_SUBMISSION"
  exit 2
fi
echo "winner_lr=$WINNER_LR full_job=$FULL_JOB full_dir=$FULL_DIR"

FINAL_CKPT="$FULL_DIR/checkpoints/last.ckpt"
bsub -J "xf-ar-ppl-${SWEEP_TAG}" -w "done($FULL_JOB)" \
  -env "all,CKPT_XF_AR=$FINAL_CKPT" < scripts/eval/ppl_ssm_baselines.sh
bsub -J "xf-ar-mavedb-${SWEEP_TAG}" -w "done($FULL_JOB)" \
  -env "all,CKPT=$FINAL_CKPT,LABEL=transformer-ar-exact,BATCH_SIZE=64,MC_SAMPLES=1,SEED=1" \
  < scripts/eval/dnahnet/mavedb_score.sh
bsub -J "xf-ar-decode-${SWEEP_TAG}" -w "done($FULL_JOB)" \
  -env "all,CKPT=$FINAL_CKPT,LABEL=transformer-ar,GENERATION_LENGTH=512" \
  < scripts/eval/ar_decode_benchmark.sh
