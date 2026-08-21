#!/bin/bash
#BSUB -J split_scan_probe
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/split_scan_probe_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/split_scan_probe_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
CC1D=/lustre/scratch126/cellgen/lotfollahi/ha11/pkg/cc1d
cd "$REPO"
# causal_conv1d is deliberately NOT installed into the shared env: it is on a
# private prefix so this probe cannot change any production run's numerics.
export PYTHONPATH="$REPO:$CC1D"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# /nfs/team361 is full; a Triton cache under $HOME dies with Errno 28.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

ARM=${ARM:-ussm-ar}
VARIANTS=${VARIANTS:-none,arsplit}
VARIANTS=${VARIANTS//+/,}
BATCH=${BATCH:-4}
LENGTH=${LENGTH:-8192}
BLOCK=${BLOCK:-256}
LABEL=${LABEL:-splitscan_$ARM}
EXTRA=${EXTRA:---check --peak --audit}

nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvcc --version 2>/dev/null || echo "nvcc: NOT PRESENT (prebuilt wheel used, none needed)"

"$PYTHON" -u scripts/smoke/split_scan_probe.py \
  --arm "$ARM" \
  --variants "$VARIANTS" \
  --batch-size "$BATCH" \
  --length "$LENGTH" \
  --block-size "$BLOCK" \
  $EXTRA \
  --output "$REPO/results/sizing/$LABEL.json"
