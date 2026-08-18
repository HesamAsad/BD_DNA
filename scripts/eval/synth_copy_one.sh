#!/usr/bin/env bash
#BSUB -J synth_copy_eval
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 4
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/eval/synthcopy_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/eval/synthcopy_%J.err
#
# P2 decisive metric: targeted copy accuracy at planted echo TARGET spans, bucketed
# by (within-block | cross-block, gap). Cross-block echoes can ONLY be solved via the
# coarse cross-attention route (fine self-attn is windowed +/-1 block), so:
#   cross-block acc ~0.25 (chance)  -> coarse route unused
#   cross-block acc >> 0.25         -> coarse route carries long-range information
# Random non-echo control spans calibrate chance. Set CKPT=<path> TAG=<name>.
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.ckpt}
TAG=${TAG:-run}
LENGTH=${LENGTH:-24576}
BLOCK_SIZE=${BLOCK_SIZE:-1536}
DATA=${DATA:-synthLR24k}
MODEL=${MODEL:-small_dual_bigblock}
export SYNTH_NSEQ=${SYNTH_NSEQ:-128}

export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-default}
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs logs/eval

echo "[`date`] SYNTH COPY EVAL | tag=$TAG | ckpt=$CKPT | block=$BLOCK_SIZE L=$LENGTH nseq=$SYNTH_NSEQ"
"$PYTHON" -u main.py mode=synth_copy_eval \
    model=$MODEL algo=bd3lm algo.backbone=dit_dual \
    data=carbon-prokaryote data.valid=$DATA data.dna_num_files=null \
    model.length=$LENGTH block_size=$BLOCK_SIZE model.attn_backend=flex \
    loader.eval_global_batch_size=1 loader.eval_batch_size=1 \
    eval.checkpoint_path=$CKPT \
    wandb=null
echo "[`date`] synth copy eval ($TAG) exited"
