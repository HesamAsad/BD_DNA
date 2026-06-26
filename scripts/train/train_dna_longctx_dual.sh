#!/usr/bin/env bash
#BSUB -J train_dna_longctx_dual
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_longctx_dual_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_longctx_dual_%J.err
#
# Long-context Variant-2 (dual-stream) BD3-LM training on Carbon prokaryote.
# Default L=98496 = 18 * 5472 = 6 * 16416 (divisible by both block_size and
# k_coarse=6 for exact cross-attn alignment). With block_size=18, attention
# per layer is local (window=8 blocks) + cross to a 16,416-token coarse stream.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-98496}             # divisible by 18 AND 6
BLOCK_SIZE=${BLOCK_SIZE:-18}        # multiple of k_coarse=6 -> exact alignment
GLOBAL_BATCH=${GLOBAL_BATCH:-16}    # = num_gpus * batch * accumulate
DNA_NUM_FILES=${DNA_NUM_FILES:-1}   # bump for more data (one shard ~ 563k seqs)
MAX_STEPS=${MAX_STEPS:-1000000}

export BD3LM_COMPILE_MASK=1                                          # required: long-ctx compiled flex masks
# max-autotune: +~44% throughput on the dual's windowed sparse flex kernel vs
# 'default' (measured A/B @L=98496, 1xH200: 0.303->0.435 it/s; single-stream
# unaffected). One-time ~21min autotune compile, amortized over a real run; it
# fit at the 98k single-GPU memory ceiling. Override to 'default' for quick jobs.
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-max-autotune-no-cudagraphs}
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

# Unique wandb run per launch. The config derives `wandb.id = ${name}_${seed}`,
# so a constant name reused the SAME wandb id every run -> runs overwrote each
# other in the UI. Stamp the name with the LSF job id (or a timestamp locally)
# so both the name AND the derived id are unique. Computed once here in bash, so
# it stays identical across the DDP ranks Lightning launches.
RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="bd3lm-dna-prok-dual-len${LENGTH}-bs${GLOBAL_BATCH}-${RUN_TAG}"

echo "[`date`] long-ctx DUAL DNA BD3-LM | host=$(hostname) | LSF=${LSB_JOBID:-local} | length=$LENGTH | block_size=$BLOCK_SIZE | global_batch=$GLOBAL_BATCH | shards=$DNA_NUM_FILES | wandb=$WANDB_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }

"$PYTHON" -u main.py \
    model=small_dual \
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
    mode=train

echo "[`date`] long-ctx dual training exited"
