#!/usr/bin/env bash
#BSUB -J ppl_ssm_baselines
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_ssm_baselines_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_ssm_baselines_%J.err
#
# Supply any subset of CKPT_USSM_BD, CKPT_USSM_AR, CKPT_BISSM, CKPT_XF, and
# CKPT_XF_AR.
# Setting CKPT_BISSM also evaluates that checkpoint through the unidirectional
# architecture as an immediate reverse-path ablation (not a trained baseline).
set -uo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
L=${L:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
EVAL_BATCH=${EVAL_BATCH:-4}
LIMIT=${LIMIT:-512}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}
USSM_EMA=${USSM_EMA:-0}
HISTORICAL_EMA=${HISTORICAL_EMA:-0.9999}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p logs/eval

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
TSV="logs/eval/ppl_ssm_baselines_${RUN_TAG}.tsv"
printf 'arm\tval_nll_nats\tval_ppl\tbits_per_nt\tcheckpoint\n' > "$TSV"

ARMS=()
[[ -n "${CKPT_USSM_BD:-}" ]] && ARMS+=("ussm_bd|$CKPT_USSM_BD")
[[ -n "${CKPT_USSM_AR:-}" ]] && ARMS+=("ussm_ar|$CKPT_USSM_AR")
if [[ -n "${CKPT_BISSM:-}" ]]; then
  ARMS+=("bissm|$CKPT_BISSM" "bissm_reverse_off|$CKPT_BISSM")
fi
[[ -n "${CKPT_XF:-}" ]] && ARMS+=("xf_bd|$CKPT_XF")
[[ -n "${CKPT_XF_AR:-}" ]] && ARMS+=("xf_ar|$CKPT_XF_AR")
if (( ${#ARMS[@]} == 0 )); then
  echo "FATAL: set at least one checkpoint variable"
  exit 2
fi

for SPEC in "${ARMS[@]}"; do
  ARM=${SPEC%%|*}
  CKPT=${SPEC#*|}
  case "$ARM" in
    ussm_bd|bissm_reverse_off)
      MODEL_ARGS=(model=small_ussm algo=bd3lm_ussm "block_size=$BLOCK_SIZE")
      SSM_ARGS=(model.active_blocks=all model.right_flank_probability=0.0) ;;
    ussm_ar)
      MODEL_ARGS=(model=small_ussm algo=ar algo.backbone=ussm block_size=1)
      SSM_ARGS=(model.active_blocks=all model.right_flank_probability=0.0) ;;
    bissm)
      MODEL_ARGS=(model=small_bissm algo=bd3lm_bissm "block_size=$BLOCK_SIZE")
      SSM_ARGS=(model.active_blocks=all model.right_flank_probability=0.0) ;;
    xf_bd)
      MODEL_ARGS=(model=small algo=bd3lm model.dropout=0.0 model.attn_backend=flex "block_size=$BLOCK_SIZE")
      SSM_ARGS=() ;;
    xf_ar)
      MODEL_ARGS=(model=small_ar_transformer algo=ar block_size=1)
      SSM_ARGS=() ;;
  esac
  if [[ "$ARM" == "ussm_bd" || "$ARM" == "ussm_ar" || "$ARM" == "xf_ar" ]]; then
    # New baseline launcher defaults to no EMA, matching the Mamba-style
    # optimizer recipe. The historical BiSSM/Transformer checkpoints have EMA.
    EMA_ARGS=("training.ema=$USSM_EMA" eval.disable_ema=False)
  else
    EMA_ARGS=("training.ema=$HISTORICAL_EMA" eval.disable_ema=False)
  fi
  LOG="logs/eval/ppl_ssm_${ARM}_${RUN_TAG}.log"
  "$PYTHON" -u main.py mode=ppl_eval \
    "${MODEL_ARGS[@]}" \
    data=carbon-prokaryote \
    data.dna_num_files="$DNA_NUM_FILES" \
    data.dna_max_rows="$DNA_MAX_ROWS" \
    model.length="$L" \
    "${SSM_ARGS[@]}" \
    loader.eval_global_batch_size="$EVAL_BATCH" \
    loader.eval_batch_size="$EVAL_BATCH" \
    loader.num_workers=4 \
    trainer.limit_val_batches="$LIMIT" \
    eval.checkpoint_path="$CKPT" \
    "${EMA_ARGS[@]}" > "$LOG" 2>&1
  RC=$?

  NLL=$(tr '\r' '\n' < "$LOG" | grep -E 'val/nll' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  if [[ -n "$NLL" ]]; then
    read PPL BITS <<< "$("$PYTHON" -c "import math; n=float('$NLL'); print(f'{math.exp(n):.4f} {n/math.log(2):.4f}')")"
  else
    PPL=NA
    BITS=NA
    NLL="fail(rc=$RC)"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$ARM" "$NLL" "$PPL" "$BITS" "$CKPT" >> "$TSV"
  echo "[$(date)] $ARM -> val/nll=$NLL ppl=$PPL bits/nt=$BITS"
done

echo "=== summary ($TSV) ==="
cat "$TSV"
