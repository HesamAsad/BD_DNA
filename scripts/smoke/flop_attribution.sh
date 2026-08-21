#!/bin/bash
#BSUB -J flop_attribution
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/flop_attribution_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/flop_attribution_%J.err
#
# Breaks the SSM FLOP residual down by aten operator and by owning module, for
# both SSM arms at three lengths, so the answer can be read as "constant FLOP
# per token per layer" (or not) directly off the summary table at the end.
#
# One forward+backward per (arm, length) under FlopCounterMode, batch 1. No
# timing loop, so the wall time is dominated by model construction and by
# Triton's first-call compilation.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  logs results/sizing

# `bsub -env` splits its own argument on commas, so list-valued variables are
# passed with '+' as the separator and translated back here. Both spellings
# work when the script is run directly.
ARMS=${ARMS:-bissm,ussm-ar}
LENGTHS=${LENGTHS:-2048,8192,16384}
ARMS=${ARMS//+/,}
LENGTHS=${LENGTHS//+/,}
BATCH=${BATCH:-1}
BLOCK_SIZE=${BLOCK_SIZE:-256}
MAX_TRACES=${MAX_TRACES:-3}
LABEL=${LABEL:-flop_attribution}
OUTPUT=${OUTPUT:-$REPO/results/sizing/$LABEL.json}

echo "[$(date)] flop attribution | arms=$ARMS lengths=$LENGTHS batch=$BATCH block=$BLOCK_SIZE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/smoke/flop_attribution.py \
  --arms "$ARMS" \
  --lengths "$LENGTHS" \
  --batch "$BATCH" \
  --block-size "$BLOCK_SIZE" \
  --max-traces "$MAX_TRACES" \
  --output "$OUTPUT"

echo "[$(date)] flop_attribution done -> $OUTPUT"
