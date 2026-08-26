#!/bin/bash
#BSUB -J build_hg38_cad
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/build_hg38_cad_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/build_hg38_cad_%J.err
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" logs data_cache/carbon
LENGTH=${LENGTH:-8192}
NAME=${NAME:-hg38-caduceus}
echo "[$(date)] building the Caduceus hg38 corpus | L=$LENGTH name=$NAME"
"$PYTHON" -u scripts/data/build_hg38_caduceus.py --length "$LENGTH" --name "$NAME" --splits train,valid
echo "[$(date)] done"; du -sh data_cache/carbon/${NAME}_* 2>/dev/null
