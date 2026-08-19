#!/usr/bin/env bash
#BSUB -J build_human
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/build_human_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/build_human_%J.err
#
# CPU-only cache build; the -gpu request only exists to satisfy the esub.
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" logs

LENGTH=${LENGTH:-8192}
N_TRAIN=${N_TRAIN:-262144}
N_VAL=${N_VAL:-1024}
NAME=${NAME:-human-lr${LENGTH}v2}
VAL_CHROMS=${VAL_CHROMS:-chr8,chr9}; VAL_CHROMS=${VAL_CHROMS//+/,}
SEED=${SEED:-0}

echo "[$(date)] build human cache | name=$NAME L=$LENGTH n_train=$N_TRAIN n_val=$N_VAL val_chroms=$VAL_CHROMS"
"$PYTHON" -u scripts/eval/build_human_longrange.py \
  --length "$LENGTH" --n_train "$N_TRAIN" --n_val "$N_VAL" \
  --name "$NAME" --val_chroms "$VAL_CHROMS" --seed "$SEED"
echo "[$(date)] build exited"
