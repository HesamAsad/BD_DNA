#!/usr/bin/env bash
#BSUB -J ab_throughput
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 1:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>32000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=32000]"
#BSUB -M 32000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ab_throughput_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ab_throughput_%J.err
#
# Single-vs-dual-stream throughput A/B at a mid context length. 50 steps,
# 1 H200, batch=1. Switch backbone via BACKBONE env: single | dual.
# The new tokens_per_s / pflop_per_s / gpu_mem_gb metrics land in wandb at
# log_every_n_steps=10.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
BACKBONE=${BACKBONE:-dual}            # single | dual
LENGTH=${LENGTH:-4608}                # = 18*256 = 6*768 (clean alignment for dual)
BLOCK_SIZE=${BLOCK_SIZE:-18}
MAX_STEPS=${MAX_STEPS:-50}
BATCH=${BATCH:-1}                     # per-GPU batch (= global batch on 1 GPU)

export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-default}
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token || true
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

if [ "$BACKBONE" = "dual" ]; then
  MODEL_CFG="model=small_dual"
  BACKBONE_OVERRIDE="algo.backbone=dit_dual"
  RUN_NAME="ab_dual_L${LENGTH}_b${BATCH}_${BD3LM_FLEX_COMPILE_MODE}"
else
  MODEL_CFG="model=small"
  BACKBONE_OVERRIDE="algo.backbone=dit"
  RUN_NAME="ab_single_L${LENGTH}_b${BATCH}_${BD3LM_FLEX_COMPILE_MODE}"
fi

echo "[`date`] A/B throughput | host=$(hostname) | LSF=${LSB_JOBID:-local} | BACKBONE=$BACKBONE | LENGTH=$LENGTH | run=$RUN_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
  || { echo 'FATAL: torch sees no GPU on this node.'; exit 3; }

"$PYTHON" -u main.py \
    $MODEL_CFG \
    algo=bd3lm \
    $BACKBONE_OVERRIDE \
    model.attn_backend=flex \
    data=carbon-prokaryote \
    data.dna_num_files=1 \
    data.dna_max_rows=3000 \
    model.length=$LENGTH \
    block_size=$BLOCK_SIZE \
    loader.global_batch_size=$BATCH \
    loader.eval_global_batch_size=$BATCH \
    loader.batch_size=$BATCH \
    loader.eval_batch_size=$BATCH \
    trainer.max_steps=$MAX_STEPS \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval=$MAX_STEPS \
    trainer.limit_val_batches=2 \
    training.from_pretrained=null \
    wandb.name=$RUN_NAME \
    mode=train

echo "[`date`] A/B run finished"
