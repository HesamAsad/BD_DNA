#!/usr/bin/env bash
#BSUB -J profile_ctxlen_dna
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/profile_ctxlen_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/profile_ctxlen_%J.err
#
# Sweep context length (batch=1) and log GPU memory per transformer block to
# find the longest context that fits one GPU.  Submit: bsub < scripts/profile/profile_context_length.sh
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ATTN=${ATTN:-flex}                       # flex: block-sparse mask (needed for long ctx; sdpa's dense 2L*2L mask OOMs early)
export BD3LM_LOG_BLOCK_MEM=1                     # per-block memory logging in dit.py
# flex torch.compile mode: 'default' avoids the max-autotune compile-time memory
# spike that OOMs long context (steady-state activation memory is unchanged).
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}
# Build the block mask block-wise (compiled) instead of a dense (2L x 2L) grid,
# so long context isn't OOM-limited by mask construction.
export BD3LM_COMPILE_MASK=${BD3LM_COMPILE_MASK:-1}
# Override the sweep points if you like, e.g. PROFILE_LENGTHS=65536,131072,262144
export PROFILE_LENGTHS=${PROFILE_LENGTHS:-4096,16384,65536,131072,262144,524288,1048576}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" logs

echo "[`date`] context-length memory sweep | host=$(hostname) | LSF=${LSB_JOBID:-local} | ATTN=$ATTN | lengths=$PROFILE_LENGTHS"
"$PYTHON" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
  || { echo 'FATAL: torch sees no GPU on this node.'; exit 3; }

"$PYTHON" -u scripts/profile/profile_context_length.py

echo "[`date`] sweep finished"
