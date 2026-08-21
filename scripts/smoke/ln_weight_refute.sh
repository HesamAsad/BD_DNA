#!/bin/bash
#BSUB -J ln_weight_refute
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ln_weight_refute_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ln_weight_refute_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$REPO/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO/cache/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" logs results/sizing

nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/smoke/ln_weight_refute.py \
  --arms "${ARMS:-dit-ar,dit}" \
  --length "${LENGTH:-8192}" \
  --batch-size "${BATCH:-4}" \
  --output "$REPO/results/sizing/ln_weight_refute.json"
