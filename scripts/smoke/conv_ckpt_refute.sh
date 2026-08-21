#!/bin/bash
#BSUB -J conv_ckpt_refute
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/conv_ckpt_refute_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/conv_ckpt_refute_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/scripts/smoke"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
mkdir -p "$TRITON_CACHE_DIR" logs results/sizing

nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "########## equivalence (fp32) ##########"
"$PYTHON" -u scripts/smoke/conv_ckpt_refute.py --mode equiv \
  --output "$REPO/results/sizing/ckptrefute_equiv.json"

echo "########## pass 1: 12-layer AR stack ##########"
"$PYTHON" -u scripts/smoke/conv_ckpt_refute.py --mode stack \
  --variant "cat,cat+ckpt,,ckpt,cat,ckpt" \
  --output "$REPO/results/sizing/ckptrefute_stack.json"

echo "########## pass 2: production Diffusion step, one process per row ##########"
for SPEC in "ussm-ar:" "ussm-ar:cat" "ussm-ar:cat+ckpt" "ussm-ar:ckpt" \
            "bissm:" "bissm:ckpt" "bissm:ckpt+rckpt"; do
  ARM="${SPEC%%:*}"; VAR="${SPEC#*:}"
  TAG="${VAR:-landed}"; TAG="${TAG//+/_}"
  echo "---- real $ARM variant=${VAR:-landed}"
  "$PYTHON" -u scripts/smoke/conv_ckpt_refute.py --mode real \
    --arm "$ARM" --variant "$VAR" \
    --output "$REPO/results/sizing/ckptrefute_real_${ARM}_${TAG}.json"
done
echo "ALL DONE"
