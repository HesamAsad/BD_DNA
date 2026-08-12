#!/usr/bin/env bash
#BSUB -J dna_ssm_baseline
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_ssm_baseline_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_ssm_baseline_%J.err
#
# Controlled SSM baselines for the DNA project.
#
# OBJECTIVE=bd3lm DIRECTION=uni: causal Mamba-2 block denoiser (primary missing
# baseline). OBJECTIVE=bd3lm DIRECTION=bi reruns the partial-bidirectional arm
# under the identical sweep recipe. OBJECTIVE=ar DIRECTION=uni trains the
# standard next-nucleotide causal Mamba-2 baseline.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
OBJECTIVE=${OBJECTIVE:-bd3lm}       # bd3lm | ar
DIRECTION=${DIRECTION:-uni}         # uni | bi (bi is valid only for bd3lm)
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-4}
EVAL_MICRO_BATCH=${EVAL_MICRO_BATCH:-4}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}
MAX_STEPS=${MAX_STEPS:-500}         # LR-ranking pilot; use 8000 for the winner
LR=${LR:-3e-4}
BETA2=${BETA2:-0.95}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
EMA=${EMA:-0}
VAL_EVERY=${VAL_EVERY:-100}
VAL_BATCHES=${VAL_BATCHES:-64}
NUM_WORKERS=${NUM_WORKERS:-16}
WANDB_MODE=${WANDB_MODE:-online}

if [[ "$OBJECTIVE" != "bd3lm" && "$OBJECTIVE" != "ar" ]]; then
  echo "FATAL: OBJECTIVE must be bd3lm or ar"
  exit 2
fi
if [[ "$DIRECTION" != "uni" && "$DIRECTION" != "bi" ]]; then
  echo "FATAL: DIRECTION must be uni or bi"
  exit 2
fi
if [[ "$OBJECTIVE" == "ar" && "$DIRECTION" != "uni" ]]; then
  echo "FATAL: the standard AR baseline is unidirectional"
  exit 2
fi
if (( LENGTH % BLOCK_SIZE != 0 )); then
  echo "FATAL: LENGTH must be divisible by BLOCK_SIZE (L=$LENGTH B=$BLOCK_SIZE)"
  exit 2
fi

if [[ "$OBJECTIVE" == "ar" ]]; then
  MODEL_ALGO_ARGS=(model=small_ussm algo=ar algo.backbone=ussm block_size=1)
elif [[ "$DIRECTION" == "uni" ]]; then
  MODEL_ALGO_ARGS=(model=small_ussm algo=bd3lm_ussm "block_size=$BLOCK_SIZE")
else
  MODEL_ALGO_ARGS=(model=small_bissm algo=bd3lm_bissm "block_size=$BLOCK_SIZE")
fi

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

EXTRA_ARGS=()
[[ "$WANDB_MODE" == "off" ]] && EXTRA_ARGS+=(wandb=null)

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
RUN_NAME="dna-${OBJECTIVE}-${DIRECTION}-mamba2-lr${LR}-b2${BETA2}-wd${WEIGHT_DECAY}-L${LENGTH}-${RUN_TAG}"
RUN_DIR=${RUN_DIR:-outputs/carbon-prokaryote/$(date +%Y.%m.%d)/${RUN_NAME}}

echo "[$(date)] $RUN_NAME | host=$(hostname) | steps=$MAX_STEPS | global_batch=$GLOBAL_BATCH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import mamba_ssm,torch; assert torch.cuda.is_available(); print('torch',torch.__version__,'mamba',mamba_ssm.__version__,'gpus',torch.cuda.device_count())"

"$PYTHON" -u main.py \
  "${MODEL_ALGO_ARGS[@]}" \
  data=carbon-prokaryote \
  data.dna_num_files="$DNA_NUM_FILES" \
  data.dna_max_rows="$DNA_MAX_ROWS" \
  model.length="$LENGTH" \
  model.active_blocks=all \
  model.right_flank_probability=0.0 \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$EVAL_MICRO_BATCH" \
  loader.num_workers="$NUM_WORKERS" \
  sampling.kv_cache=true \
  optim.lr="$LR" \
  optim.beta2="$BETA2" \
  optim.weight_decay="$WEIGHT_DECAY" \
  training.ema="$EMA" \
  lr_scheduler=cosine_decay_warmup \
  trainer.max_steps="$MAX_STEPS" \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval="$VAL_EVERY" \
  trainer.limit_val_batches="$VAL_BATCHES" \
  trainer.num_sanity_val_steps=0 \
  training.from_pretrained=null \
  wandb.name="$RUN_NAME" \
  hydra.run.dir="$RUN_DIR" \
  mode=train \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date)] $RUN_NAME exited"
