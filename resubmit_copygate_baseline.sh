#!/usr/bin/env bash
# Resubmit the copy_gate baseline sweep (D = 1024, 2048, 4096).
# Reconstructed verbatim from the LSF records of jobs 129220/129221/129222.
set -euo pipefail
ROOT=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PY=/software/cellgen/team361/ha11/envs/nichejepa/bin/python

for D in ${DVALS:-1024 2048 4096}; do
  OUT="$ROOT/results/copy_gate/baseline/D${D}"
  mkdir -p "$OUT"
  bsub <<EOF
#!/usr/bin/env bash
#BSUB -J copygate_baseline_D${D}
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000]" -R "rusage[mem=128000]" -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd $ROOT
#BSUB -o $OUT/train.out
#BSUB -e $OUT/train.err
set -uo pipefail
cd $ROOT
export PYTHONPATH=$ROOT
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -d "$ROOT/data_cache/carbon/copyD${D}L16384_train_bs16384_wrapped_specialFalse.dat" ]; then
  echo "building copyD${D}L16384"
  $PY -u scripts/eval/gen_synthetic_duplication.py \\
    --offset ${D} --length 16384 --name copyD${D}L16384 \\
    --cache_dir $ROOT/data_cache/carbon \\
    --n_train 2048 --n_val 256 || { echo "BUILD FAILED D=${D}"; exit 1; }
fi
$PY -u main.py mode=train \\
  model=small algo=bd3lm_bissm \\
  data=carbon-prokaryote data.train=copyD${D}L16384 data.valid=copyD${D}L16384 \\
  data.dna_num_files=null \\
  model.length=16384 block_size=256 \\
  loader.global_batch_size=32 \\
  loader.eval_global_batch_size=32 \\
  loader.batch_size=4 loader.eval_batch_size=4 \\
  loader.num_workers=4 \\
  trainer.max_steps=4000 trainer.log_every_n_steps=25 \\
  trainer.val_check_interval=250 trainer.limit_val_batches=16 \\
  training.from_pretrained=null \\
  hydra.run.dir=$OUT/hydra \\
  wandb=null
echo "copy gate D=${D} exit=\$?"
EOF
done
