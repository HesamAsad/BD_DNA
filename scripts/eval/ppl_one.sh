#!/usr/bin/env bash
#BSUB -J ppl_one
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_one_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_one_%J.err
#
# One ppl_eval on a pre-built cache. Env: DATA_VALID (required), L, LIMIT, CKPT.
set -uo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms; cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:-$REPO/outputs/carbon-prokaryote/2026.06.19/030312/checkpoints/7-52500.ckpt}
DATA_VALID=${DATA_VALID:?set DATA_VALID}
L=${L:-984960}; BLOCK_SIZE=${BLOCK_SIZE:-18}; LIMIT=${LIMIT:-24}
export BD3LM_COMPILE_MASK=1 BD3LM_FLEX_COMPILE_MODE=${FLEX_MODE:-default} BD3LM_DATA_NUM_PROC=1
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs logs/eval
RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
LOG="logs/eval/ppl_one_${RUN_TAG}_${DATA_VALID}.log"
echo "[`date`] ppl_one | data.valid=$DATA_VALID L=$L LIMIT=$LIMIT"
"$PYTHON" -u main.py mode=ppl_eval \
    model=small_dual algo=bd3lm algo.backbone=dit_dual \
    data=carbon-prokaryote data.valid=$DATA_VALID data.dna_num_files=1 \
    model.length=$L block_size=$BLOCK_SIZE model.attn_backend=flex \
    loader.eval_global_batch_size=1 loader.eval_batch_size=1 \
    trainer.limit_val_batches=$LIMIT \
    eval.checkpoint_path="$CKPT" eval.disable_ema=False > "$LOG" 2>&1
NLL=$(grep -E 'val/nll' "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
PPL=$(grep -E 'val/ppl' "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
echo "[`date`] RESULT data.valid=$DATA_VALID  val/nll=${NLL:-NA}  val/ppl=${PPL:-NA}"
