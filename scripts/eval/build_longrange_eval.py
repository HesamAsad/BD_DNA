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
CORPUS = ('/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/carbon/'
          'carbon-pretraining-corpus/prokaryote_evo2')

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=984960)   # ~1M, == the packed 10x point
ap.add_argument('--n_windows', type=int, default=24)
ap.add_argument('--per_contig', type=int, default=2)    # cap windows/contig for organism diversity
ap.add_argument('--min_contig', type=int, default=1_000_000)
ap.add_argument('--shuf_chunk', type=int, default=1026)  # 1026 = 18*57 = 6*171, divides 984960
ap.add_argument('--name', default='carbon-prok-lr')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max_scan', type=int, default=60000)
args = ap.parse_args()
L = args.length
assert L % args.shuf_chunk == 0, f'{L} not divisible by shuf_chunk {args.shuf_chunk}'
assert L % 18 == 0 and L % 6 == 0

# DNATokenizer mapping: A=8 C=9 G=10 T=11 N=12, anything else -> [UNK]=7.
lut = np.full(256, 7, dtype=np.int32)
for ch, i in [('A', 8), ('C', 9), ('G', 10), ('T', 11), ('N', 12)]:
  lut[ord(ch)] = i
def tok(s):
  return lut[np.frombuffer(s.encode('ascii', 'replace'), dtype=np.uint8)]

f = sorted(glob.glob(f'{CORPUS}/*.parquet'))[0]
print(f'scanning {f.split("/")[-1]} for >= {args.min_contig:,} nt contigs ...', flush=True)
pf = pq.ParquetFile(f)
windows, manifest = [], []
scanned = 0
for b in pf.iter_batches(batch_size=64, columns=['text', 'id']):
  texts, ids = b.column('text'), b.column('id')
  for j in range(len(texts)):
    scanned += 1
    s = texts[j].as_py()
    if len(s) < max(args.min_contig, L):
      continue
    cid = str(ids[j].as_py())
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
