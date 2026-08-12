#!/usr/bin/env bash
#BSUB -J bench_boundary_caches
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 1:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/bench_boundary_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/bench_boundary_%J.err
#
# Verifies the layer-major boundary-cache rewrite on the kernel training uses:
# fused Mamba-2, BF16 autocast, production geometry. Also reruns the existing
# BiSSM GPU acceptance smoke, which asserts folded-vs-per-block equality.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"
mkdir -p logs

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== BiSSM GPU acceptance smoke (folded vs per-block, unchanged test) ==="
"$PYTHON" -u scripts/smoke/smoke_bissm_gpu.py

echo
echo "=== Boundary-cache rewrite: equivalence + speed at L=8192 block=256 ==="
"$PYTHON" -u scripts/smoke/bench_boundary_caches.py \
  --length "${LENGTH:-8192}" \
  --block-size "${BLOCK_SIZE:-256}" \
  --batch-size "${BATCH_SIZE:-4}"

echo
echo "=== Same, at micro batch 8 (old path OOM'd here; new path should fit) ==="
"$PYTHON" -u scripts/smoke/bench_boundary_caches.py \
  --length "${LENGTH:-8192}" \
  --block-size "${BLOCK_SIZE:-256}" \
  --batch-size 8 || echo "micro batch 8 did not complete (see error above)"

echo "[$(date)] bench_boundary_caches done"
