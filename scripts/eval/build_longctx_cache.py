"""Fast, deadlock-free builder for long-context wrapped validation caches.

The stock `datasets.map(_group_texts, ...)` grouping deadlocks at ~1M block_size
(it slices million-element Python lists + allocates torch.ones(L) per chunk
across forked workers). This bypasses it entirely: take an EXISTING good wrapped
cache (same tokenization the validated pipeline produced), flatten its token
stream, and re-chunk to the requested length(s) with Arrow's C writer. Byte-
compatible schema (input_ids int32 Sequence, attention_mask float32 Sequence).

Usage:
  python scripts/eval/build_longctx_cache.py --source <bsX.dat> L1 L2 ...
"""
import argparse
import os

import numpy as np
import datasets

CACHE = '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon'
SRC_DEFAULT = os.path.join(
    CACHE, 'carbon-prokaryote_validation_bs98496_wrapped_specialFalse_nf1.dat')

ap = argparse.ArgumentParser()
ap.add_argument('--source', default=SRC_DEFAULT)
ap.add_argument('--overwrite', action='store_true')
ap.add_argument('lengths', nargs='+', type=int)
args = ap.parse_args()

print(f'source: {args.source}', flush=True)
src = datasets.load_from_disk(args.source)
# Flatten the whole validation token stream (identical tokens to the pipeline).
all_ids = np.concatenate(
    [np.asarray(x, dtype=np.int32) for x in src['input_ids']])
print(f'total validation tokens: {all_ids.shape[0]:,}', flush=True)

feats = datasets.Features({
    'input_ids': datasets.Sequence(datasets.Value('int32')),
    'attention_mask': datasets.Sequence(datasets.Value('float32')),
})

for L in args.lengths:
  path = os.path.join(
      CACHE, f'carbon-prokaryote_validation_bs{L}_wrapped_specialFalse_nf1.dat')
  if os.path.exists(path) and not args.overwrite:
    print(f'bs{L}: exists, skip ({path})', flush=True)
    continue
  n = all_ids.shape[0] // L
  if n < 1:
    print(f'bs{L}: SKIP — only {all_ids.shape[0]:,} tokens < L', flush=True)
    continue
  chunks = all_ids[:n * L].reshape(n, L)
  mask = np.ones((n, L), dtype=np.float32)
  ds = datasets.Dataset.from_dict(
      {'input_ids': list(chunks), 'attention_mask': list(mask)},
      features=feats)
  # save_to_disk writes a directory; build in a tmp dir then rename into place.
  tmp = path + '.tmp'
  if os.path.exists(tmp):
    import shutil
    shutil.rmtree(tmp)
  ds.save_to_disk(tmp)
  if os.path.exists(path):
    import shutil
    shutil.rmtree(path)
  os.rename(tmp, path)
  print(f'bs{L}: num_rows={n} saved -> {path}', flush=True)

print('ALL DONE', flush=True)
