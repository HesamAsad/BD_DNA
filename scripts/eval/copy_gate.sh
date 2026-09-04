#!/usr/bin/env bash
# THE COPY GATE -- the cheap unit test every candidate long-range fix must pass
# before it earns a full training run.
#
# WHY THIS EXISTS. Four independent measurements now put these models' effective
# range at 1-2 kb: the full-attention oracle finds all context value within
# +-256 nt, the coarse route never learns a fixed-offset copy four blocks away,
# runtime tau tops out near 2 kb, and the direct prefix-intervention curve dies
# by radius 2048 (BiSSM, the bidirectional arm, dies soonest at 1024). Any
# proposed fix -- new objective, new architecture, auxiliary supervision --
# has to move THAT number, and a full hg38 arm costs ~100 GPU-hours to find out.
# This gate costs a fraction of one and answers the same question.
#
# THE TASK. x[i] = x[i - D] for every i >= D, with D FIXED. Every position past
# D is predictable by copying from exactly D back, so there is no retrieval
# ambiguity -- which is precisely what sank the earlier "echo" benchmark, where
# the masked target gave the model no cue about where to copy FROM and a
# perfect model would still have scored chance. Here the offset is a constant
# the model can simply learn.
#
# READING THE LADDER. Uniform floor is ln 4 = 1.3863 nats (chance on 4 bases).
# Report `recovered = 1 - nll/ln4`:
#   recovered ~ 0     the model learned nothing; copying at this D is beyond it
#   recovered > 0.5   PASS
#   recovered ~ 0.97  what the 512 nt sanity rung achieved (val/nll 0.047)
# The largest D that passes is the model's copy range. Today that is well under
# 1024 for every arm we have. A fix that does not raise it is not a fix.
#
# The SHORT offsets are not optional. If D=256 fails, the pipeline or the
# candidate model is broken and the long offsets carry no information at all.
#
# Usage:
#   bash scripts/eval/copy_gate.sh                       # default SSM arm
#   MODEL=small ALGO=bd3lm BACKBONE=bissm TAG=mycand \
#     OFFSETS="256 1024 4096" bash scripts/eval/copy_gate.sh
set -uo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}

# --- what to test -----------------------------------------------------------
MODEL=${MODEL:-small}
# Use the dedicated configs (bd3lm_bissm / bd3lm_ussm): they set
# cross_attn=False, which the recurrent backbones REQUIRE. Passing
# `algo=bd3lm algo.backbone=bissm` leaves cross_attn=True and every run dies
# with "requires algo.cross_attn=False" before it reaches a training step.
ALGO=${ALGO:-bd3lm_bissm}
BACKBONE=${BACKBONE:-}   # empty = whatever the algo config specifies
TAG=${TAG:-baseline}
# Offsets span the measured effective range (1-2 kb) on both sides, so a fix
# shows up as the ladder extending rather than as a single point moving.
OFFSETS=${OFFSETS:-"256 512 1024 2048"}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
MAX_STEPS=${MAX_STEPS:-3000}
# 8192 x 16384 is far more than a fixed-offset copy needs, and is what blew
# up the head-node build. 2048 sequences is ample for an easy target.
N_TRAIN=${N_TRAIN:-2048}
N_VAL=${N_VAL:-256}
NUM_WORKERS=${NUM_WORKERS:-2}
# The DiT needs an attention backend and rejects the config default
# (flash_attn) on the block-diffusion path; every working arm in this repo
# uses flex. The recurrent backbones ignore this, so it is safe to always
# pass it and it keeps SSM and Transformer ladders on one code path.
ATTN=${ATTN:-flex}
GLOBAL_BATCH=${GLOBAL_BATCH:-32}
MICRO_BATCH=${MICRO_BATCH:-4}
WALL=${WALL:-12:00}
RSV=${RSV-iclr_2026}
CACHE=$REPO/data_cache/carbon
OUT=$REPO/results/copy_gate/$TAG
mkdir -p "$OUT" logs

# Refuse variables that look like they should do something and do not, which is
# the failure mode that has silently wasted the most compute on this project.
for bad in EXTRA_ARGS MODEL_ARGS HYDRA_ARGS EXTRA_MODEL_ARGS; do
  if [ -n "${!bad:-}" ]; then
    echo "ERROR: \$$bad is set but this script ignores it. Add it explicitly." >&2
    exit 2
  fi
done

echo "copy gate | tag=$TAG model=$MODEL algo=$ALGO backbone=${BACKBONE:-<from algo config>}"
echo "  offsets: $OFFSETS   L=$LENGTH block=$BLOCK_SIZE steps=$MAX_STEPS"
echo "  results -> $OUT"

for D in $OFFSETS; do
  if [ "$D" -ge $((LENGTH / 2)) ]; then
    echo "  skip D=$D: needs LENGTH > 2*D for a duplicated tail"; continue
  fi
  NAME="copyD${D}L${LENGTH}"
  # The dataset is built INSIDE the job, not here. Building on the head node
  # is OOM-killed at the default 8,192 sequences of 16,384 nt, and the failure
  # is easy to miss: it lands in a `|| continue` and the loop marches on having
  # submitted nothing. Compute nodes have the memory this needs.

  RUN_DIR=$OUT/D${D}
  mkdir -p "$RUN_DIR"
  cat > "$RUN_DIR/job.sh" <<EOF
#!/usr/bin/env bash
#BSUB -J copygate_${TAG}_D${D}
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W $WALL
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000]" -R "rusage[mem=128000]" -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd $REPO
#BSUB -o $RUN_DIR/train.out
#BSUB -e $RUN_DIR/train.err
set -uo pipefail
cd $REPO
export PYTHONPATH=$REPO
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -d "$CACHE/${NAME}_train_bs${LENGTH}_wrapped_specialFalse.dat" ]; then
  echo "building $NAME"
  $PYTHON -u scripts/eval/gen_synthetic_duplication.py \\
    --offset $D --length $LENGTH --name $NAME --cache_dir $CACHE \\
    --n_train $N_TRAIN --n_val $N_VAL || { echo "BUILD FAILED D=$D"; exit 1; }
fi
$PYTHON -u main.py mode=train \\
  model=$MODEL algo=$ALGO ${BACKBONE:+algo.backbone=$BACKBONE} \\
  data=carbon-prokaryote data.train=$NAME data.valid=$NAME \\
  data.dna_num_files=null \\
  model.length=$LENGTH block_size=$BLOCK_SIZE \\
  model.attn_backend=$ATTN \\
  loader.global_batch_size=$GLOBAL_BATCH \\
  loader.eval_global_batch_size=$GLOBAL_BATCH \\
  loader.batch_size=$MICRO_BATCH loader.eval_batch_size=$MICRO_BATCH \\
  loader.num_workers=$NUM_WORKERS \\
  trainer.max_steps=$MAX_STEPS trainer.log_every_n_steps=25 \\
  trainer.val_check_interval=250 trainer.limit_val_batches=16 \\
  training.from_pretrained=null \\
  hydra.run.dir=$RUN_DIR/hydra \\
  wandb=null
echo "copygate D=$D exit=\$?"
EOF
  jid=$(bsub ${RSV:+-U $RSV} < "$RUN_DIR/job.sh" 2>&1 | grep -oE "[0-9]{5,}" | head -1)
  echo "  D=$D -> job $jid"
done

echo
echo "when they finish:  python scripts/eval/copy_gate_report.py --tag $TAG"
