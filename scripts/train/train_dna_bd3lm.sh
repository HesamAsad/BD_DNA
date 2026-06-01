#!/usr/bin/env bash
#BSUB -J train_dna_bd3lm
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 168:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_%J.err
#
# Train BD3-LM from scratch on the Carbon 10B-token eukaryote subset, at
# single-nucleotide resolution.  Submit: bsub < scripts/train/train_dna_bd3lm.sh
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

# Env with torch>=2.5 for FlexAttention (set ATTN=sdpa to run on torch 2.4.1).
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
ATTN=${ATTN:-flex}
BLOCK_SIZE=${BLOCK_SIZE:-16}     # diffusion block size; must divide model.length
BATCH=${BATCH:-64}               # per-GPU batch; 64*4gpus*2accum=512 (batch=128 OOMs 141GB H200)

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0                  # keep transformers/tensorboard from importing tensorflow
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation / OOM risk
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token || true
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

echo "[`date`] DNA BD3-LM training | host=$(hostname) | LSF=${LSB_JOBID:-local} | block_size=$BLOCK_SIZE | ATTN=$ATTN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# Fail fast if torch can't see the GPU (use a +cu124 build matching the driver).
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU. Use a +cu124 build matching the node driver (e.g. torch 2.6.0+cu124); a +cu126 build reports cuda=False on these nodes.'; exit 3; }

# global_batch_size = batch_size * num_gpus * accumulate_grad_batches.
# BATCH=64 on 4 GPUs -> accumulate=2, effective batch 512 (the proven recipe),
# ~70GB/GPU. (BATCH=128 OOMs the 141GB H200.) To push memory use further, try
# BATCH=96 with loader.global_batch_size=384 (accumulate=1, ~100GB, but a
# smaller effective batch).
"$PYTHON" -u main.py \
    model=small \
    algo=bd3lm \
    data=carbon-eukaryote10b \
    model.length=1024 \
    block_size=$BLOCK_SIZE \
    model.attn_backend=$ATTN \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=$BATCH \
    loader.eval_batch_size=$BATCH \
    trainer.max_steps=1000000 \
    trainer.log_every_n_steps=25 \
    trainer.val_check_interval=2000 \
    trainer.limit_val_batches=200 \
    training.from_pretrained=null \
    wandb.name=bd3lm-dna-euk10b-b${BATCH}-block_size${BLOCK_SIZE} \
    mode=train

echo "[`date`] training job exited"
