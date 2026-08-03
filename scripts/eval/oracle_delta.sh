#!/usr/bin/env bash
#BSUB -J oracle_delta
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 4
#BSUB -W 3:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/eval/oracledelta_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/eval/oracledelta_%J.err
#
# H_signal (plan Stage 2): does REAL human DNA reward distal context, and out to what range?
# Delta(d) = NLL(target | context beyond radius d shuffled) - NLL(target | full true context).
# Mask is IDENTICAL across conditions (only the target span is masked), so sigma cannot
# confound; only the informativeness of the distal region changes. Shuffle is a permutation
# of the distal tokens => exact composition match, structure destroyed.
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT=/path/to/oracle.ckpt}
LENGTH=${LENGTH:-32768}
DATA_VALID=${DATA_VALID:-human-lr32768v2-gene}
MODEL=${MODEL:-small}
ALGO=${ALGO:-mdlm}
export ORACLE_NSEQ=${ORACLE_NSEQ:-32}
export ORACLE_NTGT=${ORACLE_NTGT:-4}
export ORACLE_TGTLEN=${ORACLE_TGTLEN:-256}
export ORACLE_RADII=${ORACLE_RADII:-0,256,1024,4096,16384}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs logs/eval

echo "[`date`] ORACLE DELTA | ckpt=$CKPT | valid=$DATA_VALID L=$LENGTH radii=$ORACLE_RADII"
"$PYTHON" -u main.py mode=oracle_delta \
    model=$MODEL algo=$ALGO \
    data=carbon-prokaryote data.valid=$DATA_VALID data.dna_num_files=null \
    model.length=$LENGTH \
    loader.eval_global_batch_size=1 loader.eval_batch_size=1 \
    eval.checkpoint_path=$CKPT \
    wandb=null
echo "[`date`] oracle delta exited"
