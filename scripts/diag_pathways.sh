#!/usr/bin/env bash
#BSUB -J diag_pathways
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/diag_pathways_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/diag_pathways_%J.err
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
# Post-fix PRE-COLLAPSE checkpoint (val/nll ~1.24, healthy attention).
CKPT=${CKPT:-outputs/carbon-prokaryote/2026.06.18/111035/checkpoints/last.ckpt}
VAL=${VAL:-data_cache/carbon/carbon-prokaryote_validation_bs98496_wrapped_specialFalse_nf1.dat}
echo "[`date`] diag_pathways | host=$(hostname) | ckpt=$CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "########## (1) conditionality / context-swap / noise stratification ##########"
"$PYTHON" -u scripts/diag_collapse_gpu.py "$CKPT" "$VAL" 4
echo "########## (2) pathway ablation: self vs cross attribution + coverage ##########"
"$PYTHON" -u scripts/diag_ablate_pathways.py "$CKPT" "$VAL" 6
echo "[`date`] done"
