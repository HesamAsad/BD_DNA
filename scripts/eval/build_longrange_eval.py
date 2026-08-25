"""Build a GENUINE long-range eval set: contiguous single-organism windows from
source contigs that are themselves >= 1 Mb (vs the standard `wrap=True` cache,
where a 1M window is ~50-150 unrelated fragments glued together).

Produces two caches (loadable via `data.valid=<name>` override, no new config):
  <name>              : contiguous L-nt windows from >=1Mb contigs (real long-range)
  <name>shuf          : SAME windows with ~1kb blocks permuted (long-range order
                        destroyed, local content preserved) -- the control that
                        isolates whether the model actually USES >1kb context.

Usage: python scripts/eval/build_longrange_eval.py [--length L --n_windows N ...]
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import datasets
import pyarrow.parquet as pq

CACHE = '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon'
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Split these apart so get_carbon_dna_dataset can be called with EXACTLY the
# arguments configs/data/carbon-prokaryote.yaml:16-21 gives it in training.
CORPUS_ROOT = ('/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/carbon/'
               'carbon-pretraining-corpus')
CORPUS_SUBSET = 'prokaryote_evo2'
SEQ_COLUMN = 'text'
CACHE_DIR = os.environ.get(
  'HF_DATASETS_CACHE',
  '/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface/datasets')
CORPUS = f'{CORPUS_ROOT}/{CORPUS_SUBSET}'   # kept for the manifest/output paths

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=984960)   # ~1M, == the packed 10x point
ap.add_argument('--n_windows', type=int, default=24)
ap.add_argument('--per_contig', type=int, default=2)    # cap windows/contig for organism diversity
ap.add_argument('--min_contig', type=int, default=1_000_000)
ap.add_argument('--shuf_chunk', type=int, default=1026)  # 1026 = 18*57 = 6*171, divides 984960
ap.add_argument('--name', default='carbon-prok-lr')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max_scan', type=int, default=60000)
# The split that defines "held out". These must match what the evaluated
# checkpoint TRAINED with, or the windows are not held out for that model.
ap.add_argument('--valid_frac', type=float, default=0.01)
ap.add_argument('--split_seed', type=int, default=42)
ap.add_argument('--num_files', type=int, default=None)
ap.add_argument('--max_rows', type=int, default=None)
ap.add_argument('--assert_disjoint', action='store_true', default=True)
args = ap.parse_args()
L = args.length
assert L % args.shuf_chunk == 0, f'{L} not divisible by shuf_chunk {args.shuf_chunk}'
# The cache is just tokens chunked to length L; block_size divisibility is the
# model's concern at eval time (block=18 needs L%18, block=24576 needs L%24576).
# Here only k_coarse(=6) matters (coarse encode) plus the shuffle chunk above.
assert L % 6 == 0, f'{L} not divisible by k_coarse=6'

# DNATokenizer mapping: A=8 C=9 G=10 T=11 N=12, anything else -> [UNK]=7.
lut = np.full(256, 7, dtype=np.int32)
for ch, i in [('A', 8), ('C', 9), ('G', 10), ('T', 11), ('N', 12)]:
  lut[ord(ch)] = i
def tok(s):
  return lut[np.frombuffer(s.encode('ascii', 'replace'), dtype=np.uint8)]

# HELD OUT FOR REAL. This used to glob the corpus and walk pyarrow batches
# directly, which never applied dataloader.py:466's
# `train_test_split(test_size=valid_frac, seed=seed)`. The eval windows were
# therefore drawn from the ~99% of documents the model TRAINED on, and every
# absolute nats-per-boundary number built from them was measured on training
# data. Reuse the loader's own split so this cannot drift again.
sys.path.insert(0, str(REPO))
from dataloader import get_carbon_dna_dataset  # noqa: E402

splits = get_carbon_dna_dataset(
  corpus_dir=CORPUS_ROOT, subset=CORPUS_SUBSET, seq_column=SEQ_COLUMN,
  cache_dir=CACHE_DIR, valid_frac=args.valid_frac,
  num_files=args.num_files, max_rows=args.max_rows, seed=args.split_seed)
held_out = splits['validation']
train_ids = set()
if args.assert_disjoint:
  # Belt and braces: the split is by row, so a contig cannot be in both. Prove
  # it rather than trusting it, because that is exactly what went wrong.
  train_ids = {str(r) for r in splits['train']['id']} if 'id' in \
      splits['train'].column_names else set()
print(f'held-out split: {held_out.num_rows:,} of '
      f'{splits["train"].num_rows + held_out.num_rows:,} rows '
      f'(valid_frac={args.valid_frac}, seed={args.split_seed})', flush=True)
print(f'scanning the VALIDATION split for >= {args.min_contig:,} nt contigs ...',
      flush=True)
windows, manifest = [], []
scanned = 0
for _batch_start in range(0, held_out.num_rows, 64):
  b = held_out[_batch_start:_batch_start + 64]
  texts = b['text']
  ids = b['id'] if 'id' in held_out.column_names else \
      [f'row{_batch_start + k}' for k in range(len(texts))]
  for j in range(len(texts)):
    scanned += 1
    s = texts[j]
    if len(s) < max(args.min_contig, L):
      continue
    cid = str(ids[j])
    if args.assert_disjoint and cid in train_ids:
      raise SystemExit(f'contig {cid} is in the TRAIN split -- refusing to '
                       f'build a "held-out" eval from training data')
    for w in range(min(len(s) // L, args.per_contig)):
      seg = s[w * L:(w + 1) * L]
      windows.append(tok(seg))
      manifest.append({'contig_id': cid, 'win': w, 'offset': w * L,
                       'contig_len': len(s)})
      if len(windows) >= args.n_windows:
        break
    if len(windows) >= args.n_windows:
      break
  if len(windows) >= args.n_windows or scanned >= args.max_scan:
    break

X = np.stack(windows).astype(np.int32)  # (N, L)
ncontigs = len(set(m['contig_id'] for m in manifest))
print(f'scanned {scanned:,} seqs -> {X.shape[0]} windows from {ncontigs} contigs', flush=True)

feats = datasets.Features({
  'input_ids': datasets.Sequence(datasets.Value('int32')),
  'attention_mask': datasets.Sequence(datasets.Value('float32'))})

def save(arr, name):
  am = np.ones_like(arr, dtype=np.float32)
  ds = datasets.Dataset.from_dict(
    {'input_ids': list(arr), 'attention_mask': list(am)}, features=feats)
  p = os.path.join(CACHE, f'{name}_validation_bs{L}_wrapped_specialFalse_nf1.dat')
  tmp = p + '.tmp'
  if os.path.exists(tmp):
    shutil.rmtree(tmp)
  ds.save_to_disk(tmp)
  if os.path.exists(p):
    shutil.rmtree(p)
  os.rename(tmp, p)
  print(f'saved {name}: rows={arr.shape[0]} -> {p}', flush=True)

# (1) contiguous
save(X, args.name)
# (2) shuffled control: permute shuf_chunk-sized blocks (same perm for all windows)
rng = np.random.RandomState(args.seed)
nchunks = L // args.shuf_chunk
perm = rng.permutation(nchunks)
Xs = X.reshape(X.shape[0], nchunks, args.shuf_chunk)[:, perm, :].reshape(X.shape[0], L)
save(Xs, args.name + 'shuf')

json.dump({'args': vars(args), 'n_contigs': ncontigs, 'manifest': manifest},
          open(os.path.join(CACHE, f'{args.name}_manifest.json'), 'w'), indent=1)
print('DONE', flush=True)
