#!/usr/bin/env bash
#BSUB -J synthlr_incentive
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/synthlr_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/synthlr_%J.err
#
# P2 (plan Stage 1 + Stage 3): does the coarse (long-range) route get USED on data
# with signal guaranteed by construction, and does the L_use incentive recruit it?
# Model = small_dual_bigblock with window_blocks=1 at block_size=1536 -> the fine
# self-attn is local (+/-1 block); ANY cross-block echo (gap > ~2 blocks) can ONLY
# be solved via the coarse cross-attention. Data = synthLR24k (planted copy pairs).
#   L_USE_WEIGHT=0    -> baseline (H_capacity: does it use the route spontaneously?)
#   L_USE_WEIGHT=1.0  -> incentive (H_incentive: does L_use recruit the route?)
# Eval afterwards with scripts/eval/synth_copy_eval (copy accuracy by within/cross
# block gap) + scripts/diag_gate_trajectory.py (gate_cross trajectory).
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-24576}
BLOCK_SIZE=${BLOCK_SIZE:-1536}          # 16 blocks; multiple of k_coarse=6
DATA=${DATA:-synthLR24k}
L_USE_WEIGHT=${L_USE_WEIGHT:-0}
L_USE_MARGIN=${L_USE_MARGIN:-0.2}
# B3 "aligned pointer": put fine queries and coarse keys on a shared nucleotide
# coordinate system in the cross-attention. Default false = original behaviour.
CROSS_ALIGN=${CROSS_ALIGN:-false}
# B0/B1 probes: cross_mode=gather_fine|gather_coarse hard-wires the pointer to
# i-GATHER_OFFSET; FORCE_GATE overrides the zero-init adaLN cross gate so a null
# is interpretable (without it the gate can hold the route off regardless).
CROSS_MODE=${CROSS_MODE:-attn}
GATHER_OFFSET=${GATHER_OFFSET:-0}
FORCE_GATE=${FORCE_GATE:-null}
# A2: learned relative-position bias on fine->coarse cross-attention (dense sdpa).
CROSS_REL_BIAS=${CROSS_REL_BIAS:-false}
BATCH=${BATCH:-4}
GLOBAL_BATCH=${GLOBAL_BATCH:-$BATCH}    # 1 GPU, accumulate=1
MAX_STEPS=${MAX_STEPS:-8000}
WANDB=${WANDB:-}                        # set WANDB=null for smoke (-> CSVLogger)

export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-default}
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token || true
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

TAG=$([ "$(echo "$L_USE_WEIGHT" | awk '{print ($1>0)}')" = 1 ] && echo "incentive-w${L_USE_WEIGHT}" || echo "baseline")
# In gather (B0/B1) mode the cross_attn module is bypassed, so its params never
# reach the loss and DDP aborts with "parameters that were not used in producing
# the loss". Tell DDP that is intentional. Only for the probes -- the default
# path keeps find_unused_parameters off (it costs throughput).
EXTRA_ARGS=()
[ "$CROSS_MODE" != "attn" ] && EXTRA_ARGS+=( "++strategy.find_unused_parameters=true" )

RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="bd3lm-synthlr-${TAG}-B${BLOCK_SIZE}-L${LENGTH}-${RUN_TAG}"
WANDB_ARG="wandb.name=$WANDB_NAME"
[ "$WANDB" = "null" ] && WANDB_ARG="wandb=null"

echo "[`date`] SYNTHLR INCENTIVE | host=$(hostname) | LSF=${LSB_JOBID:-local} | data=$DATA | block=$BLOCK_SIZE | L=$LENGTH | l_use=$L_USE_WEIGHT | batch=$BATCH | $TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'cuda',ok); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }

"$PYTHON" -u main.py \
    model=small_dual_bigblock \
    algo=bd3lm \
    algo.backbone=dit_dual \
    +algo.l_use_weight=$L_USE_WEIGHT \
    +algo.l_use_margin=$L_USE_MARGIN \
    +model.cross_align=$CROSS_ALIGN \
    +model.cross_mode=$CROSS_MODE \
    +model.gather_offset=$GATHER_OFFSET \
    +model.force_gate_cross=$FORCE_GATE \
    +model.cross_rel_bias=$CROSS_REL_BIAS \
    data=carbon-prokaryote \
    data.train=$DATA \
    data.valid=$DATA \
    data.dna_num_files=null \
    model.length=$LENGTH \
    block_size=$BLOCK_SIZE \
    model.attn_backend=flex \
    loader.global_batch_size=$GLOBAL_BATCH \
    loader.eval_global_batch_size=$GLOBAL_BATCH \
    loader.batch_size=$BATCH \
    loader.eval_batch_size=$BATCH \
    trainer.max_steps=$MAX_STEPS \
    trainer.log_every_n_steps=25 \
    trainer.val_check_interval=1000 \
    trainer.limit_val_batches=20 \
    training.from_pretrained=null \
    $WANDB_ARG \
    mode=train \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[`date`] synthlr incentive ($TAG) exited"
