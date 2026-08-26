#!/usr/bin/env bash
#BSUB -J dna_ar_xf
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 72:00
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
# Geometry is a knob so the AR and BD Transformer arms can share one
# parameter-matched config; small_ar_transformer and small_xf_matched are
# the same 832/13 geometry under different names.
MODEL=${MODEL:-small_ar_transformer}
# Point at a pre-built cache instead of the carbon corpus (e.g. hg38-caduceus).
# Until 2026-08-26 THIS SCRIPT HAD NO SUCH KNOB: it hardcoded
# data=carbon-prokaryote with dna_num_files=1 / dna_max_rows=400000, which is
# 2.09% of the prokaryote corpus. A DATA_TRAIN passed through `bsub -env` was
# accepted by the shell and silently ignored, so an arm submitted as an hg38 run
# would have trained on 2% of the WRONG corpus under an hg38 job name. The SSM
# launcher already had it; these two did not, and nothing compared them.
# dna_num_files/dna_max_rows are meaningless for a pre-built cache and are
# nulled so the cache filename resolves (dataloader.py:492 appends _nf/_mr tags
# only for carbon-* names, but leaving them set is still misleading).
DATA_TRAIN=${DATA_TRAIN:-}
DATA_VALID=${DATA_VALID:-}

# Fail loudly on a variable this script cannot honour. The whole class of bug
# above is "an env var was set, the shell accepted it, nothing read it".
for _unsupported in DIRECTION OBJECTIVE SSM_A_INIT_MAX SSM_DT_MAX SSM_STATE_SIZE \
                    RIGHT_FLANK_PROBABILITY CHECKPOINT_PREFILL; do
  if [ -n "${!_unsupported:-}" ]; then
    echo "FATAL: $_unsupported is set but this launcher does not implement it."
    echo "       It would be silently ignored. Use the SSM launcher instead."
    exit 2
  fi
done

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

EXTRA_ARGS=()
[[ "$WANDB_MODE" == "off" ]] && EXTRA_ARGS+=(wandb=null)
if [ -n "$DATA_TRAIN" ]; then
  EXTRA_ARGS+=(data.train="$DATA_TRAIN")
  EXTRA_ARGS+=(data.valid="${DATA_VALID:-$DATA_TRAIN}")
  # Overwrite rather than append a second, contradictory override. Passing
  # both `data.dna_max_rows=400000` and `data.dna_max_rows=null` works (Hydra
  # takes the last) but the resolved command then shows two values for one key,
  # which is unreadable in a log and one reordering away from a silent 2% cap.
  DNA_NUM_FILES=null
  DNA_MAX_ROWS=null
fi
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
RUN_NAME="dna-ar-transformer-lr${LR}-b2${BETA2}-wd${WEIGHT_DECAY}-L${LENGTH}-${RUN_TAG}"
RUN_DIR=${RUN_DIR:-outputs/carbon-prokaryote/$(date +%Y.%m.%d)/${RUN_NAME}}

echo "[$(date)] $RUN_NAME | host=$(hostname) | steps=$MAX_STEPS | global_batch=$GLOBAL_BATCH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u main.py \
  model="$MODEL" algo=ar data=carbon-prokaryote block_size=1 \
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
