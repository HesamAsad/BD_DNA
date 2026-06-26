#!/usr/bin/env bash
#BSUB -J longrange_eval
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/longrange_eval_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/longrange_eval_%J.err
#
# GENUINE long-range eval at ~1M on CONTIGUOUS single-organism windows (>=1Mb
# contigs) vs the same windows BLOCK-SHUFFLED (long-range order destroyed).
#   - ppl_eval on carbon-prok-lr      -> NLL with real long-range structure
#   - ppl_eval on carbon-prok-lrshuf  -> NLL with long-range destroyed (control)
#   - io_dump  on carbon-prok-lr      -> decoded masked predictions on real DNA
# If NLL(contig) < NLL(shuf) the model USES >1kb context; if ~equal it's local-
# dominated (consistent with the ~vestigial cross-attn). Compare both to the
# PACKED 10x point (val/nll 1.1377).  Caches are pre-built (build_longrange_eval.py).
set -uo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:-$REPO/outputs/carbon-prokaryote/2026.06.19/030312/checkpoints/7-52500.ckpt}
L=${L:-984960}
BLOCK_SIZE=${BLOCK_SIZE:-18}
LIMIT=${LIMIT:-24}

export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${FLEX_MODE:-default}
export BD3LM_DATA_NUM_PROC=1
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs logs/eval
RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
SUMMARY="logs/eval/longrange_eval_${RUN_TAG}.tsv"
echo -e "variant\tdata_valid\tval_nll\tval_ppl" > "$SUMMARY"

echo "[`date`] long-range eval | host=$(hostname) | L=$L | ckpt=$CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[ -f "$CKPT" ] || { echo "FATAL: ckpt missing: $CKPT"; exit 4; }

run_ppl () {  # $1=variant label  $2=data.valid name
  local LOG="logs/eval/longrange_eval_${RUN_TAG}_$1.log"
  echo "---- ppl_eval [$1] data.valid=$2 ----"
  "$PYTHON" -u main.py mode=ppl_eval \
      model=small_dual algo=bd3lm algo.backbone=dit_dual \
      data=carbon-prokaryote data.valid=$2 data.dna_num_files=1 \
      model.length=$L block_size=$BLOCK_SIZE model.attn_backend=flex \
      loader.eval_global_batch_size=1 loader.eval_batch_size=1 \
      trainer.limit_val_batches=$LIMIT \
      eval.checkpoint_path="$CKPT" eval.disable_ema=False > "$LOG" 2>&1
  local NLL=$(grep -E 'val/nll' "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  local PPL=$(grep -E 'val/ppl' "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo -e "$1\t$2\t${NLL:-NA}\t${PPL:-NA}" | tee -a "$SUMMARY"
}

run_ppl "contiguous" "carbon-prok-lr"
run_ppl "shuffled"   "carbon-prok-lrshuf"

echo "---- io_dump [contiguous] ----"
IO_DUMP_DIR="$REPO/logs/eval/io_dump_longrange_L${L}_${RUN_TAG}" \
IO_DUMP_FRACS=0.15,0.5 IO_DUMP_NSEQ=1 IO_DUMP_BINS=20 \
  "$PYTHON" -u main.py mode=io_dump \
    model=small_dual algo=bd3lm algo.backbone=dit_dual \
    data=carbon-prokaryote data.valid=carbon-prok-lr data.dna_num_files=1 \
    model.length=$L block_size=$BLOCK_SIZE model.attn_backend=flex \
    loader.eval_global_batch_size=1 loader.eval_batch_size=1 \
    eval.checkpoint_path="$CKPT" eval.disable_ema=False \
    > "logs/eval/longrange_eval_${RUN_TAG}_iodump.log" 2>&1 \
  && sed -n '1,30p' "$REPO/logs/eval/io_dump_longrange_L${L}_${RUN_TAG}/report.txt"

echo "======================================================================"
echo "[`date`] long-range eval done. PACKED 10x reference: val/nll=1.1377"
column -t -s "$(printf '\t')" "$SUMMARY"
