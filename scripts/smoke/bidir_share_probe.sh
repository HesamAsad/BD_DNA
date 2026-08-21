#!/bin/bash
#BSUB -J bidir_share
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/bidir_share_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/bidir_share_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# /nfs/team361 is full; a Triton cache under $HOME dies with Errno 28.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

LABEL=${LABEL:-bidir_share}
BATCH=${BATCH:-4}
LENGTH=${LENGTH:-8192}
BLOCK=${BLOCK:-256}

nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo
echo "=============================================================="
echo "1) isolated BiMambaLayer: equivalence, saved tensors, layer time"
echo "=============================================================="
"$PYTHON" -u scripts/smoke/bidir_share_probe.py \
  --batch-size "$BATCH" --length "$LENGTH" --block-size "$BLOCK" \
  --output "$REPO/results/sizing/${LABEL}_layer.json"

echo
echo "=============================================================="
echo "2) end-to-end training step: peak memory and throughput"
echo "=============================================================="
for IMPL in split fused; do
  echo "--- bidirectional_impl=$IMPL ---"
  "$PYTHON" -u scripts/smoke/sizing_sweep.py \
    --arms bissm --batch-sizes "$BATCH" --lengths "$LENGTH" \
    --block-size "$BLOCK" --checkpoint-modes off \
    --warmup 2 --iters 5 \
    --overrides "model.bidirectional_impl=$IMPL" \
    --output "$REPO/results/sizing/${LABEL}_step_${IMPL}.json"
done

echo
echo "=============================================================="
echo "3) whole-step saved-tensor audit, both spellings"
echo "=============================================================="
for IMPL in split fused; do
  echo "--- bidirectional_impl=$IMPL ---"
  "$PYTHON" -u scripts/smoke/saved_tensor_audit.py \
    --arm bissm --length "$LENGTH" --batch-size "$BATCH" \
    --block-size "$BLOCK" --variants none --audit --top 24 \
    --overrides "model.bidirectional_impl=$IMPL" \
    --output "$REPO/results/sizing/${LABEL}_audit_${IMPL}.json"
done

echo
echo "=============================================================="
echo "4) does the saved memory buy a larger micro batch?"
echo "=============================================================="
"$PYTHON" -u scripts/smoke/sizing_sweep.py \
  --arms bissm --batch-sizes 6,8 --lengths "$LENGTH" \
  --block-size "$BLOCK" --checkpoint-modes off \
  --warmup 2 --iters 3 \
  --overrides "model.bidirectional_impl=fused" \
  --output "$REPO/results/sizing/${LABEL}_batch_fused.json"

echo "[$(date)] done"
