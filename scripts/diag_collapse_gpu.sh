#!/usr/bin/env bash
#BSUB -J diag_collapse
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000]"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/diag_collapse_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/diag_collapse_%J.err
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms; cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=default
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=${CKPT:-outputs/carbon-prokaryote/2026.06.17/044459/checkpoints/last.ckpt}
VAL=${VAL:-data_cache/carbon/carbon-prokaryote_validation_bs98496_wrapped_specialFalse_nf1.dat}
echo "[`date`] diag_collapse | host=$(hostname) | ckpt=$CKPT"
"$PYTHON" -u scripts/diag_collapse_gpu.py "$CKPT" "$VAL" 4
echo "[`date`] done"
