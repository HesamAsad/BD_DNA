#!/bin/bash
#BSUB -J prep_full_prok
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>256000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=256000]"
#BSUB -M 256000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/prep_full_prok_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/prep_full_prok_%J.err
set -euo pipefail

# Tokenise the FULL prokaryote corpus once, so all five arms share one cache.
#
# Every run so far used dna_num_files=1, dna_max_rows=400000 -- 400,000 of
# ~19,142,000 rows, i.e. 2.09% of the corpus. That subset holds 8.09e9 nt and
# our 8000-step budget is 4.19e9, so runs to date saw 0.52 EPOCHS: no example
# twice, but also only 1.08% of what exists.
#
# Full corpus is ~0.39e12 nt. Cached as input_ids int32 + attention_mask
# float32 that is ~3.12 TB; lustre has 3.0 PB free, so storage is not the
# constraint. Tokenising is: this is the long pole, hence its own job.
#
# The cache key is (dataset, split, model length, wrap, special tokens,
# num_files, max_rows) -- NOT block_size -- so one build serves the AR arms
# (block_size 1) and the BD arms (block_size 256) alike.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface/datasets
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
# datasets.map deadlocks at high num_proc on this filesystem; 16 is what the
# long-context prep settled on. Raising it is the first thing to try if this is
# too slow, but do it deliberately.
export BD3LM_DATA_NUM_PROC=${BD3LM_DATA_NUM_PROC:-16}
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" logs data_cache/carbon

LENGTH=${LENGTH:-8192}
echo "[$(date)] tokenising the FULL prokaryote corpus at length $LENGTH"
echo "  num_proc=$BD3LM_DATA_NUM_PROC   expect ~3.12 TB of cache"
df -h /lustre/scratch126/cellgen/lotfollahi | tail -1

"$PYTHON" -u - <<PYEOF
import sys, time
sys.path.insert(0, "$REPO")
import main  # registers the OmegaConf resolvers
import hydra, dataloader
from dataloader import DNATokenizer

with hydra.initialize_config_dir(version_base=None, config_dir="$REPO/configs"):
  config = hydra.compose(config_name="config", overrides=[
    "model=small_ussm", "algo=ar", "data=carbon-prokaryote",
    "model.length=$LENGTH", "block_size=1",
    "loader.batch_size=1", "loader.eval_batch_size=1",
    "loader.global_batch_size=1", "trainer.devices=1", "trainer.num_nodes=1",
    # THE POINT OF THIS JOB: no cap on either axis.
    "data.dna_num_files=null", "data.dna_max_rows=null",
  ])
start = time.time()
train, valid = dataloader.get_dataloaders(config, DNATokenizer(), skip_train=False)
print(f"train batches: {len(train):,}")
print(f"valid batches: {len(valid):,}")
print(f"tokenised in {(time.time()-start)/3600:.2f} h")
PYEOF

echo "[$(date)] done"
du -sh data_cache/carbon | awk '{print "  cache now: "$1}'
