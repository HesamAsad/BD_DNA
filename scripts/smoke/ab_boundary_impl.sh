#!/usr/bin/env bash
#BSUB -J ab_boundary_impl
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ab_boundary_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ab_boundary_%J.err
#
# In-situ A/B for the layer-major boundary-cache rewrite.
#
# Trains the SAME model twice for MAX_STEPS optimizer steps, identical seed and
# data order, differing only in model.boundary_impl. Dropout is 0 and all noise
# sampling happens before the backbone is called, so the two runs consume the
# same RNG stream: their loss curves should track to bf16 reassociation noise.
# Divergence beyond that means the rewrite is not loss-equivalent in situ, which
# is the one thing the unit tests and the single-step GPU benchmark cannot see.
#
# wandb is off so Lightning falls back to CSVLogger (main.py:468) and both
# curves land in <run_dir>/csv_logs/version_0/metrics.csv for direct diffing.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"

MAX_STEPS=${MAX_STEPS:-200}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-4}
LR=${LR:-1e-3}
SEED=${SEED:-1}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}
VAL_EVERY=${VAL_EVERY:-100}
VAL_BATCHES=${VAL_BATCHES:-32}
NUM_WORKERS=${NUM_WORKERS:-16}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
BASE_DIR=outputs/ab-boundary/${RUN_TAG}
mkdir -p "$BASE_DIR"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import mamba_ssm,torch; assert torch.cuda.is_available(); print('torch',torch.__version__,'mamba',mamba_ssm.__version__,'gpus',torch.cuda.device_count())"

for IMPL in layer_major block_major; do
  RUN_DIR="${BASE_DIR}/${IMPL}"
  echo
  echo "[$(date)] === boundary_impl=${IMPL} | steps=${MAX_STEPS} | seed=${SEED} ==="
  "$PYTHON" -u main.py \
    model=small_bissm \
    algo=bd3lm_bissm \
    "block_size=$BLOCK_SIZE" \
    data=carbon-prokaryote \
    data.dna_num_files="$DNA_NUM_FILES" \
    data.dna_max_rows="$DNA_MAX_ROWS" \
    model.length="$LENGTH" \
    model.active_blocks=all \
    model.right_flank_probability=0.0 \
    "model.boundary_impl=$IMPL" \
    loader.global_batch_size="$GLOBAL_BATCH" \
    loader.eval_global_batch_size="$GLOBAL_BATCH" \
    loader.batch_size="$MICRO_BATCH" \
    loader.eval_batch_size="$MICRO_BATCH" \
    loader.num_workers="$NUM_WORKERS" \
    optim.lr="$LR" \
    optim.beta2=0.95 \
    optim.weight_decay=0.1 \
    training.ema=0 \
    seed="$SEED" \
    lr_scheduler=cosine_decay_warmup \
    trainer.max_steps="$MAX_STEPS" \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval="$VAL_EVERY" \
    trainer.limit_val_batches="$VAL_BATCHES" \
    trainer.num_sanity_val_steps=0 \
    training.from_pretrained=null \
    wandb=null \
    hydra.run.dir="$RUN_DIR" \
    mode=train
  echo "[$(date)] ${IMPL} finished"
done

echo
echo "=== A/B comparison ==="
"$PYTHON" -u scripts/smoke/compare_ab_curves.py \
  --layer-major "${BASE_DIR}/layer_major" \
  --block-major "${BASE_DIR}/block_major"
