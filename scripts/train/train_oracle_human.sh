#!/usr/bin/env bash
#BSUB -J oracle_human
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 72:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/oracle_human_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/oracle_human_%J.err
#
# P1 (plan Stage 2 / H_signal): the full-attention ORACLE on human DNA. A standard
# MDLM (single-stream dit, full bidirectional attention) trained on 32 kb human
# windows. Full attention CAN use distal context if it exists, so its
# distance-resolved BPB drop (measured later by the oracle eval mode) is a
# model-DEPENDENT lower bound on real long-range signal. Not the windowed dual
# model under study -- deliberately full attention, as an upper bound on usable
# distal information.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-32768}
# v2 = chromosome-holdout caches (val only from chr8/chr9). The original human-lr32768
# had 19.5% train/val exact-duplicate leakage -- audited and rebuilt 2026-07-20.
DATA_TRAIN=${DATA_TRAIN:-human-lr32768v2}
DATA_VALID=${DATA_VALID:-human-lr32768v2-gene}
MODEL=${MODEL:-small}
ALGO=${ALGO:-mdlm}
BATCH=${BATCH:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-$BATCH}
MAX_STEPS=${MAX_STEPS:-40000}
# MUST be <= batches-per-epoch (n_train/BATCH), else Lightning raises. At 8192 train
# rows / BATCH=8 that is 1024 -- val_check_interval=2000 killed job 81050.
VAL_EVERY=${VAL_EVERY:-500}
WANDB=${WANDB:-}

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

RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="bd3lm-oracle-${ALGO}-${DATA_TRAIN}-L${LENGTH}-${RUN_TAG}"
WANDB_ARG="wandb.name=$WANDB_NAME"
[ "$WANDB" = "null" ] && WANDB_ARG="wandb=null"

echo "[`date`] ORACLE $ALGO | host=$(hostname) | LSF=${LSB_JOBID:-local} | train=$DATA_TRAIN valid=$DATA_VALID | L=$LENGTH | model=$MODEL | batch=$BATCH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'cuda',ok); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }

"$PYTHON" -u main.py \
    model=$MODEL \
    algo=$ALGO \
    data=carbon-prokaryote \
    data.train=$DATA_TRAIN \
    data.valid=$DATA_VALID \
    data.dna_num_files=null \
    model.length=$LENGTH \
    loader.global_batch_size=$GLOBAL_BATCH \
    loader.eval_global_batch_size=$GLOBAL_BATCH \
    loader.batch_size=$BATCH \
    loader.eval_batch_size=$BATCH \
    trainer.max_steps=$MAX_STEPS \
    trainer.log_every_n_steps=50 \
    trainer.val_check_interval=$VAL_EVERY \
    trainer.limit_val_batches=50 \
    training.from_pretrained=null \
    $WANDB_ARG \
    mode=train

echo "[`date`] oracle $ALGO exited"
