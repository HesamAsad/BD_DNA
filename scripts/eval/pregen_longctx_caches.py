"""CPU-only pre-generation of wrapped validation caches for the long-context
NLL sweep. Builds one `carbon-prokaryote_validation_bs{L}_wrapped...` cache per
requested length so the GPU sweep job only ever loads ready caches (never runs
the datasets.map grouping that deadlocked at num_proc=128).

Usage:  BD3LM_DATA_NUM_PROC=8 python scripts/eval/pregen_longctx_caches.py L1 L2 ...
"""
import os
import sys

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import omegaconf  # noqa: E402
import dataloader  # noqa: E402

cfg = omegaconf.OmegaConf.load('configs/data/carbon-prokaryote.yaml')
tok = dataloader.DNATokenizer()
lengths = [int(x) for x in sys.argv[1:]]
DNA_NUM_FILES = int(os.environ.get('DNA_NUM_FILES', '1'))
print(f'num_proc={os.environ.get("BD3LM_DATA_NUM_PROC", "default")} '
      f'dna_num_files={DNA_NUM_FILES} lengths={lengths}', flush=True)

for L in lengths:
  print(f'=== generating validation cache bs{L} ===', flush=True)
  ds = dataloader.get_dataset(
    cfg.valid, tok,
    wrap=cfg.wrap,
    mode='validation',
    cache_dir=cfg.cache_dir,
    block_size=L,
    insert_eos=cfg.insert_valid_eos,
    insert_special_tokens=cfg.insert_valid_special,
    streaming=cfg.streaming,
    dna_corpus_dir=cfg.dna_corpus_dir,
    dna_subset=cfg.dna_subset,
    dna_seq_column=cfg.dna_seq_column,
    dna_valid_frac=cfg.dna_valid_frac,
    dna_num_files=DNA_NUM_FILES,
    dna_max_rows=cfg.dna_max_rows)
  print(f'==> bs{L}: num_rows={ds.num_rows}', flush=True)
print('ALL DONE', flush=True)
