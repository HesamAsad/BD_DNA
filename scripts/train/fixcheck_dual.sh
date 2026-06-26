#!/usr/bin/env bash
#BSUB -J fixcheck_dual
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>48000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=48000]"
#BSUB -M 48000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/fixcheck_dual_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/fixcheck_dual_%J.err
#
# Confirm the FineDualBlock zero-init deadlock fix: dual-stream BD3-LM should now
# learn (val/nll drop BELOW the unigram floor 1.382) and the self/cross output
# projections should leave zero. L=4608 reuses the cached mr3000 prokaryote shard
# (no re-tokenization), short warmup so the signal shows in ~1.5k steps.
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms; cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=default
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs
echo "[`date`] fixcheck_dual | host=$(hostname) | LSF=${LSB_JOBID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u main.py \
    model=small_dual algo=bd3lm algo.backbone=dit_dual data=carbon-prokaryote \
    data.dna_num_files=1 data.dna_max_rows=3000 \
    model.length=4608 block_size=18 model.attn_backend=flex \
    loader.global_batch_size=32 loader.eval_global_batch_size=32 \
    loader.batch_size=8 loader.eval_batch_size=8 \
    lr_scheduler.num_warmup_steps=100 \
    trainer.max_steps=1500 trainer.log_every_n_steps=25 \
    trainer.val_check_interval=250 trainer.limit_val_batches=10 \
    training.from_pretrained=null wandb=null mode=train
echo "[`date`] fixcheck training exited"
