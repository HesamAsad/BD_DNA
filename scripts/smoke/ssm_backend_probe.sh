#!/bin/bash
#BSUB -J ssm_backend_probe
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_backend_probe_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_backend_probe_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# `bsub -env` splits its own argument on commas, so list-valued variables are
# passed with '+' as the separator and translated back here. Both spellings
# work when the script is run directly.
ARMS=${ARMS:-bissm,dit}
BATCH_SIZES=${BATCH_SIZES:-4,8}
LENGTHS=${LENGTHS:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
CHECKPOINT_MODES=${CHECKPOINT_MODES:-off,on}
ARMS=${ARMS//+/,}
BATCH_SIZES=${BATCH_SIZES//+/,}
LENGTHS=${LENGTHS//+/,}
CHECKPOINT_MODES=${CHECKPOINT_MODES//+/,}
WARMUP=${WARMUP:-2}
ITERS=${ITERS:-5}
LABEL=${LABEL:-sweep}
OUTPUT=${OUTPUT:-$REPO/results/sizing/$LABEL.json}

mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] sizing sweep | arms=$ARMS batch=$BATCH_SIZES lengths=$LENGTHS block=$BLOCK_SIZE ckpt=$CHECKPOINT_MODES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/smoke/ssm_backend_probe.py \
  --length "${PROBE_LENGTH:-8192}" --batch "${PROBE_BATCH:-4}" \
  --output /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/results/sizing/backend-probe.json
