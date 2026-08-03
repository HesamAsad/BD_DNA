#!/usr/bin/env bash
#BSUB -J smoke_bissm_gpu
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 1:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_bissm_gpu_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_bissm_gpu_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"
mkdir -p logs

export MAX_JOBS=${MAX_JOBS:-8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON" -c "import mamba_ssm"; then
  MAMBA_KEEP_CUDA_BUILD=FALSE "$PYTHON" -m pip install \
    --user --no-deps --no-build-isolation -r requirements-mamba.txt
fi

"$PYTHON" -u scripts/smoke/smoke_bissm_gpu.py
