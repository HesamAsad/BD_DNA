#!/usr/bin/env bash
#BSUB -J train_dna_longctx
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_longctx_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_longctx_%J.err
#
# Long-context BD3-LM training on the Carbon prokaryote subset (single-nucleotide).
# Requires H200 (gmodel pinned) and the block-wise compiled mask so context isn't
# limited by mask construction. Measured H200 batch=1 ceiling ~122,880 (120k);
# default LENGTH=98,304 (96k) leaves headroom for optimizer/EMA/loss memory.
# Submit:        bsub < scripts/train/train_dna_longctx.sh
# Bigger ctx:    LENGTH=114688 bsub -env "all, LENGTH=114688, PYTHON=..." < ...
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-98304}            # context length; multiple of BLOCK_SIZE; H200 batch=1 max ~122880
BLOCK_SIZE=${BLOCK_SIZE:-16}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}   # effective batch (sequences); accumulate = GLOBAL_BATCH/num_gpus (1/gpu)
DNA_NUM_FILES=${DNA_NUM_FILES:-1}  # prokaryote shards to tokenize (1 ~= 563k seqs; raise for more data)
MAX_STEPS=${MAX_STEPS:-1000000}

export BD3LM_COMPILE_MASK=1                                          # block-wise compiled mask (REQUIRED for long ctx)
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-default}   # avoid max-autotune compile-memory spike
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

echo "[`date`] long-ctx DNA BD3-LM | host=$(hostname) | LSF=${LSB_JOBID:-local} | length=$LENGTH | block_size=$BLOCK_SIZE | global_batch=$GLOBAL_BATCH | shards=$DNA_NUM_FILES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU (need a +cu124/H200 node).'; exit 3; }

# batch_size=1 per GPU; Lightning gradient-accumulates to reach GLOBAL_BATCH.
"$PYTHON" -u main.py \
    model=small \
    algo=bd3lm \
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
    wandb.name=bd3lm-dna-prok-len${LENGTH}-bs${GLOBAL_BATCH} \
    mode=train

echo "[`date`] long-ctx training exited"
