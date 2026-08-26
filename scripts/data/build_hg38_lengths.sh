#!/usr/bin/env bash
#BSUB -J build_hg38_len
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/build_hg38_len_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/build_hg38_len_%J.err
set -uo pipefail

# Build the hg38 cache at ADDITIONAL context lengths, for the 1024 and 32768
# campaigns that follow the 8192 one.
#
# WHY THE LENGTHS. 8192 optimises neither thing we want to show:
#   * GenomicBenchmarks sequences are 200-802 nt for 7 of 8 tasks (max 4,776),
#     and Caduceus pretrained at 1024 -- so 1024 is both better matched to the
#     benchmark and the length that makes their published row a real reference.
#   * Our own scaling work puts the Transformer/SSM crossover at ~11,608
#     tokens, so at 8192 the Transformer is still the cheaper arm. The SSM
#     efficiency claim needs 32768.
#
# WHY THESE THREE ARE EXACTLY COMPARABLE. The corpus is a fixed 35.67e9 tokens,
# so holding TOKENS PER STEP at 524,288 gives the same 68,042 steps at every
# length:
#     L=1,024   34,837,504 windows / batch 512 = 68,042 steps
#     L=8,192    4,354,688 windows / batch  64 = 68,042 steps
#     L=32,768   1,088,672 windows / batch  16 = 68,042 steps
# Same steps, same tokens, same number of epochs. Context length is then the
# ONLY variable between the three campaigns.
#
# ONE JOB, NOT TWO. The build imports no torch and never touches the GPU, but
# the training-parallel esub refuses a job without a `-gpu` request, so every
# build holds an H200 idle for hours. Running both lengths sequentially in one
# job halves that waste while our five 8192 arms are competing for the same
# GPUs. The cache name is shared -- the filename carries bs{L}, so the three
# lengths coexist without colliding.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs data_cache/carbon

NAME=${NAME:-hg38-caduceus}
LENGTHS=${LENGTHS:-1024 32768}

echo "[$(date)] building $NAME at lengths: $LENGTHS"
df -h /lustre/scratch126/cellgen/lotfollahi | tail -1

rc_total=0
for L in $LENGTHS; do
  train="data_cache/carbon/${NAME}_train_bs${L}_wrapped_specialFalse.dat"
  valid="data_cache/carbon/${NAME}_validation_bs${L}_wrapped_specialFalse.dat"
  if [ -d "$train" ] && [ -d "$valid" ]; then
    echo "[$(date)] L=$L already built, skipping"
    continue
  fi
  echo
  echo "=============================================================="
  echo "[$(date)] L=$L"
  "$PYTHON" -u scripts/data/build_hg38_caduceus.py --length "$L" --name "$NAME"
  rc=$?
  echo "[$(date)] L=$L exit=$rc"
  rc_total=$((rc_total + rc))
  if [ "$rc" = "0" ]; then
    # Verify immediately, while the fasta page cache is still warm. A cache
    # that is wrong is worth knowing about now, not when a training arm has
    # already consumed it for twelve hours.
    "$PYTHON" -u scripts/data/verify_hg38_cache.py --length "$L" --name "$NAME" \
      --split train --sample 300
    rc_total=$((rc_total + $?))
    "$PYTHON" -u scripts/data/verify_hg38_cache.py --length "$L" --name "$NAME" \
      --split validation --sample 300
    rc_total=$((rc_total + $?))
  fi
done

echo
echo "[$(date)] summed exit codes: $rc_total  (0 = every length built and verified)"
du -sh data_cache/carbon/${NAME}_* 2>/dev/null
exit $rc_total
