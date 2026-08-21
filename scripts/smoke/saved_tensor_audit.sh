#!/bin/bash
#BSUB -J saved_tensor_audit
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/saved_tensor_audit_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/saved_tensor_audit_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton

ARM=${ARM:-ussm-ar}
LENGTH=${LENGTH:-8192}
BATCH=${BATCH:-4}
VARIANTS=${VARIANTS:-none}
VARIANTS=${VARIANTS//;/,}
MODE=${MODE:---audit}
LABEL=${LABEL:-audit}
OUTPUT=${OUTPUT:-$REPO/results/sizing/$LABEL.json}

mkdir -p "$(dirname "$OUTPUT")" logs
echo "[$(date)] saved tensor audit | arm=$ARM L=$LENGTH batch=$BATCH variants=$VARIANTS mode=$MODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/smoke/saved_tensor_audit.py \
  --arm "$ARM" --length "$LENGTH" --batch-size "$BATCH" \
  --variants "$VARIANTS" $MODE --output "$OUTPUT"
