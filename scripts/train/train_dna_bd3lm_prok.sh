#!/usr/bin/env bash
#BSUB -J train_dna_bd3lm_prok
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 168:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_prok_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_prok_%J.err
#
# Transformer BD3-LM control arm for the prokaryote perplexity comparison.
# Deliberately mirrors scripts/train/train_dna_bissm.sh knob for knob (data,
# length, block size, batch, steps, validation cadence, dropout) so the only
# difference between the two runs is the backbone.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}
MICRO_BATCH=${MICRO_BATCH:-8}
# Kept separate from MICRO_BATCH so both arms of a comparison validate on
# byte-identical batches even when their training micro batch differs.
EVAL_MICRO_BATCH=${EVAL_MICRO_BATCH:-$MICRO_BATCH}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
MAX_STEPS=${MAX_STEPS:-1000000}
VAL_EVERY=${VAL_EVERY:-2000}
VAL_BATCHES=${VAL_BATCHES:-50}
NUM_WORKERS=${NUM_WORKERS:-16}
DROPOUT=${DROPOUT:-0.0}          # matched to small_bissm (0.0), not small.yaml's 0.1
ATTN=${ATTN:-flex}
WANDB_MODE=${WANDB_MODE:-online}

if (( LENGTH % BLOCK_SIZE != 0 )); then
  echo "FATAL: LENGTH must be divisible by BLOCK_SIZE (L=$LENGTH B=$BLOCK_SIZE)"
  exit 2
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
[ -n "${DNA_MAX_ROWS:-}" ] && EXTRA_ARGS+=( "data.dna_max_rows=$DNA_MAX_ROWS" )
[ "$WANDB_MODE" = "off" ] && EXTRA_ARGS+=( "wandb=null" )

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
WANDB_NAME="bd3lm-dna-prok-xf-L${LENGTH}-B${BLOCK_SIZE}-${RUN_TAG}"
# Hydra's default run directory is timestamped to the second, so two jobs
# launched together land in the same directory and interleave their
# checkpoints. Stamp the job id instead; it also makes a resume path obvious.
RUN_DIR=${RUN_DIR:-outputs/carbon-prokaryote/$(date +%Y.%m.%d)/xf-${RUN_TAG}}

echo "[$(date)] Transformer BD3-LM prokaryote control | host=$(hostname) | LSF=${LSB_JOBID:-local} | length=$LENGTH | block=$BLOCK_SIZE | attn=$ATTN | wandb=$WANDB_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }

"$PYTHON" -u main.py \
  model=small \
  algo=bd3lm \
  data=carbon-prokaryote \
  data.dna_num_files="$DNA_NUM_FILES" \
  model.length="$LENGTH" \
  model.dropout="$DROPOUT" \
  model.attn_backend="$ATTN" \
  block_size="$BLOCK_SIZE" \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$EVAL_MICRO_BATCH" \
  loader.num_workers="$NUM_WORKERS" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval="$VAL_EVERY" \
  trainer.limit_val_batches="$VAL_BATCHES" \
  training.from_pretrained=null \
  wandb.name="$WANDB_NAME" \
  hydra.run.dir="$RUN_DIR" \
  mode=train \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date)] Transformer control exited"
