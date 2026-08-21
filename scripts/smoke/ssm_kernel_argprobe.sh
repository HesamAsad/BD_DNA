#!/bin/bash
#BSUB -J ssm_argprobe
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_argprobe_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_argprobe_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

ARMS=${ARMS:-base,gate,conv,cs256,nostate,gate+conv+nostate,attn}
CENSUS=${CENSUS:-base,gate+conv,attn}
ARMS=${ARMS//+++/+}
LABEL=${LABEL:-argprobe}

nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u scripts/smoke/ssm_kernel_argprobe.py \
  --batch "${BATCH:-4}" \
  --length "${LENGTH:-8192}" \
  --arms "$ARMS" \
  --census "$CENSUS" \
  --output "$REPO/results/sizing/$LABEL.json"
