#!/usr/bin/env bash
#BSUB -J train_dna_bissm
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bissm_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bissm_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}
MICRO_BATCH=${MICRO_BATCH:-8}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
MAX_STEPS=${MAX_STEPS:-1000000}
RIGHT_FLANK_PROBABILITY=${RIGHT_FLANK_PROBABILITY:-0.0}

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
if [ "$RIGHT_FLANK_PROBABILITY" = "0.0" ]; then
  MODE_TAG=denovo
else
  MODE_TAG=ca
fi
WANDB_NAME="bd3lm-dna-bissm-${MODE_TAG}-L${LENGTH}-B${BLOCK_SIZE}-${RUN_TAG}"

echo "[$(date)] BiSSM DNA | host=$(hostname) | LSF=${LSB_JOBID:-local} | length=$LENGTH | block=$BLOCK_SIZE | right_prob=$RIGHT_FLANK_PROBABILITY | wandb=$WANDB_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import mamba_ssm,torch; assert torch.cuda.is_available(); print('torch',torch.__version__,'mamba',mamba_ssm.__version__,'gpus',torch.cuda.device_count())" || {
  echo "FATAL: install requirements with: $PYTHON -m pip install --no-build-isolation -r requirements.txt"
  exit 3
}

"$PYTHON" -u main.py \
  model=small_bissm \
  algo=bd3lm_bissm \
  data=carbon-prokaryote \
  data.dna_num_files="$DNA_NUM_FILES" \
  model.length="$LENGTH" \
  model.right_flank_probability="$RIGHT_FLANK_PROBABILITY" \
  block_size="$BLOCK_SIZE" \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$MICRO_BATCH" \
  sampling.kv_cache=true \
  trainer.max_steps="$MAX_STEPS" \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval=2000 \
  trainer.limit_val_batches=50 \
  training.from_pretrained=null \
  wandb.name="$WANDB_NAME" \
  mode=train

echo "[$(date)] BiSSM training exited"
