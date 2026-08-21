#!/bin/bash
#BSUB -J conv_variants2
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 3:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/conv_variants2_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/conv_variants2_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "########## pass 1: micro + census + 12-layer stack ##########"
"$PYTHON" -u scripts/smoke/conv_variants.py \
  --batch 4 --length 8192 --warmup 2 --iters 5 --micro \
  --arms base,pad,pad2,convckpt,ckpt,attn \
  --census pad2,convckpt \
  --real-model "" \
  --output "$REPO/results/sizing/conv_variants2_stack.json"

# One process per variant: max_memory_allocated is absolute, so a shared
# process would carry the previous arm's live tensors into the next row.
for VARIANT in base pad pad2; do
  echo "########## pass 2: production Diffusion step, variant=$VARIANT ##########"
  "$PYTHON" -u scripts/smoke/conv_variants.py \
    --batch 4 --length 8192 --warmup 2 --iters 5 \
    --equivalence 0 --arms "" --census "" \
    --real-model "$VARIANT" --real-arms ussm-ar,bissm \
    --output "$REPO/results/sizing/conv_variants2_real_$VARIANT.json"
done
