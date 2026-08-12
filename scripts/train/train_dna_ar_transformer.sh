#!/usr/bin/env bash
#BSUB -J dna_ar_xf
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_ar_transformer_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_ar_transformer_%J.err
# Exact next-nucleotide Transformer control, matched to the recurrent baselines.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-8192}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-4}
EVAL_MICRO_BATCH=${EVAL_MICRO_BATCH:-4}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}
MAX_STEPS=${MAX_STEPS:-500}
LR=${LR:-3e-4}
BETA2=${BETA2:-0.95}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
EMA=${EMA:-0}
VAL_EVERY=${VAL_EVERY:-100}
VAL_BATCHES=${VAL_BATCHES:-64}
NUM_WORKERS=${NUM_WORKERS:-16}
WANDB_MODE=${WANDB_MODE:-online}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

EXTRA_ARGS=()
[[ "$WANDB_MODE" == "off" ]] && EXTRA_ARGS+=(wandb=null)
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
RUN_NAME="dna-ar-transformer-lr${LR}-b2${BETA2}-wd${WEIGHT_DECAY}-L${LENGTH}-${RUN_TAG}"
RUN_DIR=${RUN_DIR:-outputs/carbon-prokaryote/$(date +%Y.%m.%d)/${RUN_NAME}}

echo "[$(date)] $RUN_NAME | host=$(hostname) | steps=$MAX_STEPS | global_batch=$GLOBAL_BATCH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u main.py \
  model=small_ar_transformer algo=ar data=carbon-prokaryote block_size=1 \
  data.dna_num_files="$DNA_NUM_FILES" \
  data.dna_max_rows="$DNA_MAX_ROWS" \
  model.length="$LENGTH" \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$EVAL_MICRO_BATCH" \
  loader.num_workers="$NUM_WORKERS" \
  sampling.kv_cache=true \
  optim.lr="$LR" optim.beta2="$BETA2" optim.weight_decay="$WEIGHT_DECAY" \
  training.ema="$EMA" lr_scheduler=cosine_decay_warmup \
  trainer.max_steps="$MAX_STEPS" trainer.log_every_n_steps=10 \
  trainer.val_check_interval="$VAL_EVERY" trainer.limit_val_batches="$VAL_BATCHES" \
  trainer.num_sanity_val_steps=0 training.from_pretrained=null \
  wandb.name="$RUN_NAME" hydra.run.dir="$RUN_DIR" mode=train \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date)] $RUN_NAME exited"
