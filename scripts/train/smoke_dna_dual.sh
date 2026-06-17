#!/usr/bin/env bash
#BSUB -J smoke_dual_dna
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 1:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>32000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=32000]"
#BSUB -M 32000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_dual_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_dual_%J.err
#
# End-to-end smoke for the DualStreamDIT backbone (Variant 2). Tiny: L=576,
# block_size=18, k_coarse=6 (exact alignment), batch=1, 10 steps. Verifies:
#   - dual-stream model builds (compiled fine_local + fine_to_coarse masks)
#   - forward + backward run without OOM/error
#   - both streams + cross-attention exercise their code paths
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
export BD3LM_COMPILE_MASK=1                                          # block-wise compiled masks
export BD3LM_FLEX_COMPILE_MODE=${BD3LM_FLEX_COMPILE_MODE:-default}   # cheaper compile for the smoke
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

echo "[`date`] dual-stream smoke | host=$(hostname) | LSF=${LSB_JOBID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
  || { echo 'FATAL: torch sees no GPU on this node.'; exit 3; }

"$PYTHON" -u main.py \
    model=small_dual \
    algo=bd3lm \
    algo.backbone=dit_dual \
    data=carbon-prokaryote \
    data.dna_num_files=1 \
    data.dna_max_rows=2000 \
    model.length=576 \
    block_size=18 \
    loader.global_batch_size=1 \
    loader.eval_global_batch_size=1 \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    trainer.max_steps=10 \
    trainer.log_every_n_steps=2 \
    trainer.val_check_interval=10 \
    trainer.limit_val_batches=2 \
    training.from_pretrained=null \
    wandb=null \
    mode=train

echo "[`date`] smoke OK"
