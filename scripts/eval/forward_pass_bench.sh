#!/bin/bash
#BSUB -J fwd_bench
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 6:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/fwd_bench_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/fwd_bench_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" logs results/sizing

ARMS=${ARMS:-bissm,ussm,ussm-ar,dit,dit-ar}; ARMS=${ARMS//+/,}
LENGTHS=${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072,262144,524288}
LENGTHS=${LENGTHS//+/,}
BATCH=${BATCH:-1}
LABEL=${LABEL:-forward_pass}

echo "[$(date)] forward-pass bench | arms=$ARMS batch=$BATCH"
echo "  lengths=$LENGTHS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/eval/forward_pass_bench.py \
  --arms "$ARMS" --lengths "$LENGTHS" --batch "$BATCH" \
  --output "$REPO/results/sizing/$LABEL.json"
