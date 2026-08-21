#!/bin/bash
#BSUB -J conv_variants
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 3:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/conv_variants_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/conv_variants_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

ARMS=${ARMS:-base,ckpt,pad,pad+ckpt,shift,attn}
CENSUS=${CENSUS:-base,pad,shift,ckpt,pad+ckpt}
REAL_MODEL=${REAL_MODEL:-base,pad}
REAL_ARMS=${REAL_ARMS:-ussm-ar,bissm}
LABEL=${LABEL:-conv_variants}
ARMS=${ARMS//+++/+}
CENSUS=${CENSUS//+++/+}

nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u scripts/smoke/conv_variants.py \
  --batch "${BATCH:-4}" \
  --length "${LENGTH:-8192}" \
  --warmup "${WARMUP:-2}" \
  --iters "${ITERS:-5}" \
  --arms "$ARMS" \
  --census "$CENSUS" \
  --real-model "$REAL_MODEL" \
  --real-arms "$REAL_ARMS" \
  --output "$REPO/results/sizing/$LABEL.json"
