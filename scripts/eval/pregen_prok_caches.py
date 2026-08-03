"""Pre-generate wrapped train+validation caches for the prokaryote perplexity
comparison, so the GPU jobs only ever load ready caches.

Both arms of the comparison (BiSSM and the Transformer BD3-LM baseline) read
exactly the same cache files, which is what makes the perplexity numbers
comparable: identical shard, identical row cap, identical 1% validation split
(seed 42), identical wrapping.

Usage:
  BD3LM_DATA_NUM_PROC=16 DNA_NUM_FILES=1 DNA_MAX_ROWS=120000 \
    python scripts/eval/pregen_prok_caches.py 1024 8192
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
lengths = [int(x) for x in sys.argv[1:]] or [1024]
DNA_NUM_FILES = int(os.environ.get('DNA_NUM_FILES', '1'))
_max_rows = os.environ.get('DNA_MAX_ROWS', '')
DNA_MAX_ROWS = int(_max_rows) if _max_rows else cfg.dna_max_rows

print(f'num_proc={os.environ.get("BD3LM_DATA_NUM_PROC", "default")} '
      f'dna_num_files={DNA_NUM_FILES} dna_max_rows={DNA_MAX_ROWS} '
      f'lengths={lengths}', flush=True)

for L in lengths:
  for mode in ('train', 'validation'):
    print(f'=== generating {mode} cache bs{L} ===', flush=True)
    ds = dataloader.get_dataset(
      cfg.train if mode == 'train' else cfg.valid, tok,
      wrap=cfg.wrap,
      mode=mode,
      cache_dir=cfg.cache_dir,
      block_size=L,
      insert_eos=(cfg.insert_train_eos if mode == 'train'
                  else cfg.insert_valid_eos),
      insert_special_tokens=(cfg.insert_train_special if mode == 'train'
                             else cfg.insert_valid_special),
      streaming=cfg.streaming,
      dna_corpus_dir=cfg.dna_corpus_dir,
      dna_subset=cfg.dna_subset,
      dna_seq_column=cfg.dna_seq_column,
      dna_valid_frac=cfg.dna_valid_frac,
      dna_num_files=DNA_NUM_FILES,
      dna_max_rows=DNA_MAX_ROWS)
    print(f'==> {mode} bs{L}: num_rows={ds.num_rows} '
          f'tokens={ds.num_rows * L:,}', flush=True)
print('ALL DONE', flush=True)
