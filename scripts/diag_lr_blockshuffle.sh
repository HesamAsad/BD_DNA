#!/usr/bin/env bash
#BSUB -J diag_lrbs
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/diag_lrbs_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/diag_lrbs_%J.err
set -uo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms; cd "$REPO"
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
CKPT=$REPO/outputs/carbon-prokaryote/2026.06.26/204845/checkpoints/1-8000.ckpt
CACHE=$REPO/data_cache/carbon/carbon-prok-lr983_validation_bs983040_wrapped_specialFalse_nf1.dat
export BD3LM_COMPILE_MASK=1 BD3LM_FLEX_COMPILE_MODE=default BD3LM_DATA_NUM_PROC=1
export HF_HOME=$REPO/../cache/huggingface TORCH_HOME=$REPO/../cache/torch XDG_CACHE_HOME=$REPO/../cache/xdg
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
LOG="logs/diag_lrbs_${LSB_JOBID:-local}.log"
echo "[`date`] diag_longrange_blockshuffle starting on $(hostname)" | tee "$LOG"
"$PYTHON" -u scripts/diag_longrange_blockshuffle.py "$CKPT" "$CACHE" 10 6 >> "$LOG" 2>&1
echo "[`date`] DONE rc=$? -> $LOG" | tee -a "$LOG"
