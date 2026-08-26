#!/usr/bin/env bash
#BSUB -J dna_bd3lm_xf_tuned
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_prok_tuned_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_prok_tuned_%J.err
#
# Transformer BD3-LM control under the SAME Mamba-style optimizer recipe as the
# SSM arms (scripts/train/train_dna_ssm_baseline.sh): AdamW beta2, weight decay,
# cosine-decay warmup, no EMA, and an LR chosen by the same pilot procedure.
#
# scripts/train/train_dna_bd3lm_prok.sh cannot express this: it passes no
# optim.*, training.ema or lr_scheduler override, so it is pinned to the repo
# defaults (lr 3e-4, beta2 0.999, wd 0, ema 0.9999, constant_warmup/2500) that
# produced LSF 96604. This launcher is that script's knobs plus the recipe.
#
# Model geometry is deliberately left at model=small (hidden 768 / 12 heads,
# 92.4M params) so the arm is (a) byte-comparable to 96604 and (b) loadable by
# the xf_bd arm of scripts/eval/ppl_ssm_baselines.sh:73 without modification.
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-8}
EVAL_MICRO_BATCH=${EVAL_MICRO_BATCH:-4}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}
MAX_STEPS=${MAX_STEPS:-8000}
LR=${LR:-1e-3}
BETA2=${BETA2:-0.95}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
EMA=${EMA:-0}
VAL_EVERY=${VAL_EVERY:-200}      # MICRO-batches, not optimizer steps
VAL_BATCHES=${VAL_BATCHES:-64}
NUM_WORKERS=${NUM_WORKERS:-16}
DROPOUT=${DROPOUT:-0.0}
ATTN=${ATTN:-flex}
MODEL=${MODEL:-small}
# Scheduler is a knob because the fair-comparison arm needs 96604's own recipe
# (constant_warmup) with EMA off, not the SSM arms' cosine decay.
SCHEDULER=${SCHEDULER:-cosine_decay_warmup}
WANDB_MODE=${WANDB_MODE:-online}

if (( LENGTH % BLOCK_SIZE != 0 )); then
  echo "FATAL: LENGTH must be divisible by BLOCK_SIZE (L=$LENGTH B=$BLOCK_SIZE)"
  exit 2
fi

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

EXTRA_ARGS=()
[[ "$WANDB_MODE" == "off" ]] && EXTRA_ARGS+=(wandb=null)
# BD3-LM's central recipe contribution: clipped noise schedules. The paper
# (arXiv:2503.09573 sec 4.3, 5.2) shows the AR-vs-diffusion perplexity gap is
# driven by GRADIENT VARIANCE, not anything fundamental -- only masked tokens
# contribute to the loss and E[mask rate]=0.5 under a linear schedule, so
# gradients are estimated from 2x fewer tokens. Table 8, block size 4:
# linear U[0,1] gives PPL 30.18 at NELBO variance 23.45; clipped U[0.45,0.95]
# gives 29.21 at variance 6.24.
#
# `clip_search_widths` is a list of WINDOW WIDTHS; metrics.py:init_valid_vars
# scans each width across offsets at clip_search_delta, and the search picks
# whichever window minimises measured NELBO variance. Empty list = only
# (eps, 1) = the plain linear schedule, i.e. the feature switched off. It was
# empty in every DNA run we have done.
#
# The upstream scripts (scripts/train/train_{lm1b,owt}_bd3lm.sh:26) pass
# [0.5,0.6,0.7,0.8,0.9]; our DNA launchers never carried it across. We widen
# the low end because the optimum is block-size dependent and moves toward
# LIGHTER masking as blocks grow -- paper Table 2: L'=4 prefers U[0.5,1],
# L'=16 prefers U[0.3,0.8], L'=128 prefers U[0,0.5]. Our block is 256, larger
# than anything they tested, so 0.3/0.4 are added to let the variance search
# reach tighter, lighter windows rather than have us guess one.
CLIP_SEARCH_WIDTHS=${CLIP_SEARCH_WIDTHS:-[0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
EXTRA_ARGS+=(algo.clip_search_widths="$CLIP_SEARCH_WIDTHS")


RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
RUN_NAME="dna-bd3lm-xf-lr${LR}-b2${BETA2}-wd${WEIGHT_DECAY}-ema${EMA}-${SCHEDULER}-L${LENGTH}-${RUN_TAG}"
# Pass RUN_DIR explicitly if you want a resubmit to resume: the default embeds
# the job id, and a new job id means a new dir means no last.ckpt to resume.
RUN_DIR=${RUN_DIR:-outputs/carbon-prokaryote/$(date +%Y.%m.%d)/${RUN_NAME}}

echo "[$(date)] $RUN_NAME | host=$(hostname) | model=$MODEL | steps=$MAX_STEPS | lr=$LR | global_batch=$GLOBAL_BATCH | run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }

"$PYTHON" -u main.py \
  model="$MODEL" \
  algo=bd3lm \
  data=carbon-prokaryote \
  data.dna_num_files="$DNA_NUM_FILES" \
  data.dna_max_rows="$DNA_MAX_ROWS" \
  model.length="$LENGTH" \
  model.dropout="$DROPOUT" \
  model.attn_backend="$ATTN" \
  block_size="$BLOCK_SIZE" \
  loader.global_batch_size="$GLOBAL_BATCH" \
  loader.eval_global_batch_size="$GLOBAL_BATCH" \
  loader.batch_size="$MICRO_BATCH" \
  loader.eval_batch_size="$EVAL_MICRO_BATCH" \
  loader.num_workers="$NUM_WORKERS" \
  optim.lr="$LR" \
  optim.beta2="$BETA2" \
  optim.weight_decay="$WEIGHT_DECAY" \
  training.ema="$EMA" \
  lr_scheduler="$SCHEDULER" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval="$VAL_EVERY" \
  trainer.limit_val_batches="$VAL_BATCHES" \
  trainer.num_sanity_val_steps=0 \
  training.from_pretrained=null \
  wandb.name="$RUN_NAME" \
  hydra.run.dir="$RUN_DIR" \
  mode=train \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date)] $RUN_NAME exited"
