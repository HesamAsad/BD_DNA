#!/usr/bin/env bash
#BSUB -J train_bigblock_dual
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_bigblock_dual_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_bigblock_dual_%J.err
#
# BIG-BIDIRECTIONAL-BLOCK dual-stream BD3-LM training on Carbon prokaryote.
# ---------------------------------------------------------------------------
# Tests whether a LARGE bidirectional block (block_size 10k-100k, vs the 18 nt
# of the current model) makes the model actually USE long-range DNA context.
# The current block=18 dual was measured local-dominated (1kb/10kb shuffle
# disentanglement, jobs 56206/56300: NLL penalty scaled with #boundaries, not
# destroyed long-range -> true long-range usage ~0).
#
# KEY: at a FIXED model.length (98304) the memory is ~unchanged vs the current
# run (job 50711) regardless of block_size — only the within-block dense
# attention grows (flex keeps it flash-memory). So this DEFAULT preset trains on
# the SAME 4xH200 as the current model and is directly comparable (same length,
# data, GPUs; only block_size differs). Re-run the shuffle eval afterwards to see
# if long-range usage appears.
#
# Presets (override via env BLOCK_SIZE / LENGTH; bump GPUs via `bsub -gpu num=N`):
#   default (controlled)  : BLOCK_SIZE=24576  LENGTH=98304  -> 4 blocks, 4xH200
#   bigger bidir window   : BLOCK_SIZE=49152  LENGTH=98304  -> 2 blocks, 4xH200
#   single 98k bidir block: BLOCK_SIZE=98304  LENGTH=98304  -> 1 block (=MDLM/full
#                           bidirectional over 98k; coarse inert), 4xH200
#   smoke (cheap)         : BLOCK_SIZE=1536   LENGTH=6144   -> 4 blocks; submit with
#                           `bsub -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"`
#   1M hierarchy (LATER)  : BLOCK_SIZE=98304  LENGTH=983040 -> 10 blocks; NEEDS
#                           gradient checkpointing (NOT yet implemented) + 8-16 GPU.
#
# Divisibility (asserted by the model): LENGTH % BLOCK_SIZE == 0,
# BLOCK_SIZE % 6 == 0, LENGTH % 6 == 0.  98304 = 2^15*3 keeps all the above clean.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
BLOCK_SIZE=${BLOCK_SIZE:-24576}     # BIG bidirectional block (vs 18). multiple of k_coarse=6
LENGTH=${LENGTH:-98304}             # same as current run -> same memory/GPUs, comparable
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
MAX_STEPS=${MAX_STEPS:-1000000}

# Sanity: divisibility (fail fast before the GPU job spins up).
if (( LENGTH % BLOCK_SIZE != 0 || BLOCK_SIZE % 6 != 0 || LENGTH % 6 != 0 )); then
  echo "FATAL: need LENGTH%BLOCK_SIZE==0, BLOCK_SIZE%6==0, LENGTH%6==0 (got L=$LENGTH B=$BLOCK_SIZE)"; exit 2
fi
N_BLOCKS=$(( LENGTH / BLOCK_SIZE ))

export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# datasets.map GROUPING only deadlocks at HUGE chunk size (model.length >~ 1M).
# At normal lengths (<=~600k) leave num_proc at the default (all cores) so the
# 557k-seq TRAIN set tokenizes fast (num_proc=1 => ~3h, too slow even for a
# smoke). Only cap to 1 for the 1M-hierarchy preset. Override via env.
if [ -z "${BD3LM_DATA_NUM_PROC:-}" ] && (( LENGTH >= 600000 )); then
  export BD3LM_DATA_NUM_PROC=1
fi
# Optional: cap train rows for a quick smoke (e.g. DNA_MAX_ROWS=20000) so it
# doesn't tokenize all 557k sequences.
EXTRA=()
[ -n "${DNA_MAX_ROWS:-}" ] && EXTRA+=( "data.dna_max_rows=$DNA_MAX_ROWS" )
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token || true
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="bd3lm-dna-prok-bigblock-B${BLOCK_SIZE}-L${LENGTH}-${RUN_TAG}"

echo "[`date`] BIG-BLOCK DUAL | host=$(hostname) | LSF=${LSB_JOBID:-local} | block=$BLOCK_SIZE | length=$LENGTH | N_blocks=$N_BLOCKS | global_batch=$GLOBAL_BATCH | wandb=$WANDB_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }

"$PYTHON" -u main.py \
    model=small_dual_bigblock \
    algo=bd3lm \
    algo.backbone=dit_dual \
    data=carbon-prokaryote \
    data.dna_num_files=$DNA_NUM_FILES \
    model.length=$LENGTH \
    block_size=$BLOCK_SIZE \
    model.attn_backend=flex \
    loader.global_batch_size=$GLOBAL_BATCH \
    loader.eval_global_batch_size=$GLOBAL_BATCH \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    trainer.max_steps=$MAX_STEPS \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval=2000 \
    trainer.limit_val_batches=50 \
    training.from_pretrained=null \
    wandb.name=$WANDB_NAME \
    mode=train \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "[`date`] big-block dual training exited"
