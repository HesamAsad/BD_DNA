#!/usr/bin/env bash
#BSUB -J pregen_prok_caches
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/pregen_prok_caches_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/pregen_prok_caches_%J.err
#
# CPU-only cache build (the -gpu request only exists to satisfy the esub).
# Submit: bsub -env "all, LENGTHS=1024 8192, DNA_MAX_ROWS=120000" \
#           < scripts/eval/pregen_prok_caches.sh
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTHS=${LENGTHS:-"1024 8192"}
export DNA_NUM_FILES=${DNA_NUM_FILES:-1}
export DNA_MAX_ROWS=${DNA_MAX_ROWS:-120000}
# Grouping deadlocks with a large fork count on long block sizes; 16 is safe at
# these lengths (rows are 1k-8k elements, not multi-million).
export BD3LM_DATA_NUM_PROC=${BD3LM_DATA_NUM_PROC:-16}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" logs

echo "[$(date)] pregen prokaryote caches | host=$(hostname) | LSF=${LSB_JOBID:-local} | lengths=$LENGTHS | num_files=$DNA_NUM_FILES | max_rows=$DNA_MAX_ROWS | num_proc=$BD3LM_DATA_NUM_PROC"
"$PYTHON" -u scripts/eval/pregen_prok_caches.py $LENGTHS
echo "[$(date)] pregen exited"
