#!/bin/bash
#BSUB -J refute_splitproj
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/refute_splitproj_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/refute_splitproj_%J.err
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"; export PYTHONPATH="$REPO" HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "########## LIVE TREE (padded _causal_conv landed): uSSM-AR ##########"
"$PYTHON" -u scripts/smoke/refute_rms_audit.py --arm ussm-ar \
  --variants none,splitproj,xbc,splitproj+xbc --audit --peak --top 14 \
  --output results/sizing/refute_splitproj_ussmar.json
echo "[$(date)] done"
