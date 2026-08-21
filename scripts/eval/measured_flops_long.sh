#!/bin/bash
#BSUB -J flops_long
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/flops_long_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/flops_long_%J.err
set -euo pipefail

# Extends the measured-FLOPs sweep past 32768 for the two SSM arms only.
#
# The main sweep (results/sizing/measured_flops.json) ran at micro batch 2 and
# stopped at 32768, where the `dit` arm OOMed. That truncates the two measured
# curves in the scaling figure at 2^15 -- exactly where the Transformer curves
# begin to bend away -- so the panel cuts off the divergence it exists to show.
#
# Batch 1 halves the activation footprint. Extrapolating the measured training
# peak (bissm, batch 2, L=32768: 46.25 GiB with checkpoint_boundary_prefill on)
# linearly gives ~92 GiB per sequence at 131072, which fits an H200's 140 GB.
#
# Writes to its OWN output file. It does NOT touch measured_flops.json, whose
# 2048..32768 rows the sweep script would otherwise overwrite wholesale.
# One length per job: at 131072 a second model build in the same process would
# have to allocate against a fragmented pool.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# Unconditional, not ${VAR:-default}: ~/.bashrc exports a value that `bsub -env
# all` inherits, so a default-only assignment silently loses to it and the cache
# lands on a full NFS volume.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
# Fragmentation guard for the 131072 build; does not affect what is dispatched.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

LENGTH=${LENGTH:-65536}
LABEL=${LABEL:-measured_flops_L$LENGTH}

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "== measured FLOPs sweep: arms=bissm,ussm-ar length=$LENGTH batch=1 =="

"$PYTHON" -u scripts/eval/measured_flops_sweep.py \
  --arms bissm,ussm-ar \
  --lengths "$LENGTH" \
  --batch 1 \
  --output "$REPO/results/sizing/$LABEL.json"

echo "== done: results/sizing/$LABEL.json =="
