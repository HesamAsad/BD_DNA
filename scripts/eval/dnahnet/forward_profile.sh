#!/usr/bin/env bash
#BSUB -J dnahnet_forward_profile
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/dnahnet_forward_profile_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/dnahnet_forward_profile_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.ckpt}
LABEL=${LABEL:?set LABEL=bissm or LABEL=transformer}
LENGTHS=${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072,262144,524288}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-3}
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT=${OUTPUT:-$REPO/results/dnahnet/forward/${LABEL}-${RUN_TAG}.json}

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-default}
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] forward profile | label=$LABEL | checkpoint=$CKPT | lengths=$LENGTHS | output=$OUTPUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u scripts/eval/dnahnet/profile_forward.py \
  --checkpoint "$CKPT" \
  --label "$LABEL" \
  --lengths "$LENGTHS" \
  --warmups "$WARMUPS" \
  --repeats "$REPEATS" \
  --output "$OUTPUT"

echo "[$(date)] forward profile exited"
