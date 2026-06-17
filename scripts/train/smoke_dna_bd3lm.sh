#!/usr/bin/env bash
#BSUB -J smoke_dna_bd3lm
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 1:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000]"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_dna_bd3lm_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_dna_bd3lm_%J.err
#
# Quick end-to-end sanity check of the DNA loader + BD3-LM on 1 GPU:
# small model, 1 corpus shard capped to 20k sequences, ~50 train steps.
# Submit:           bsub < scripts/train/smoke_dna_bd3lm.sh
# Or interactively: bsub -Is -q training-parallel -G s10396 -n 16 \
#                     -gpu "num=1:mode=exclusive_process" -R "span[hosts=1]" \
#                     -M 64000 -R "rusage[mem=64000]" bash scripts/train/smoke_dna_bd3lm.sh
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

# Point this at the env that has torch>=2.5 once you've upgraded; nichejepa
# (torch 2.4.1) works too if you set ATTN=sdpa (FlexAttention needs torch>=2.5).
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
ATTN=${ATTN:-flex}      # flex (torch>=2.5) | sdpa (works on torch 2.4.1)

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0                  # keep transformers/tensorboard from importing tensorflow
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

echo "[`date`] DNA BD3-LM smoke test | host=$(hostname) | LSF=${LSB_JOBID:-local} | ATTN=$ATTN"
# Fail fast with a clear message if torch can't see the GPU (otherwise the
# config's accumulate_grad_batches interpolation dies with a cryptic ZeroDivision).
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU. Use a +cu124 build matching the node driver (e.g. torch 2.6.0+cu124); a +cu126 build reports cuda=False on these nodes.'; exit 3; }

"$PYTHON" -u main.py \
    model=small \
    algo=bd3lm \
    data=carbon-eukaryote10b \
    data.dna_num_files=1 \
    data.dna_max_rows=20000 \
    model.length=1024 \
    block_size=16 \
    model.attn_backend=$ATTN \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    trainer.max_steps=50 \
    trainer.val_check_interval=25 \
    trainer.limit_val_batches=4 \
    trainer.log_every_n_steps=5 \
    training.from_pretrained=null \
    wandb=null \
    mode=train

echo "[`date`] smoke test finished OK"
