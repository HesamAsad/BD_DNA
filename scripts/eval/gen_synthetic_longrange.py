"""Synthetic long-range-dependency DNA benchmark ("echo / copy across blocks").

WHY: run 56995 showed gate_cross DECAYING on prokaryote because that corpus has ~0
mutual information beyond ~1 kb -- there is no long-range signal to learn, so the
block-shuffle/long-range question is unanswerable on it. This builds DNA-token
sequences whose long-range MI is non-zero BY CONSTRUCTION, isolating "can the
architecture USE >block_size context" from "is there long-range structure to use".

TASK: each sequence is random ACGT background with K planted ECHO PAIRS. An echo
pair writes a random m-nt motif at a source position s and copies it verbatim at
s+g. Predicting the TARGET copy requires retrieving the SOURCE from g nt away --
solvable only via long-range context when g spans a block boundary (within-block
self-attn cannot see across blocks; the coarse cross-attn can). Random motif content
per sequence => the model must RETRIEVE, not memorise.

OUTPUTS (loadable via data.train=/data.valid=<name>, data=carbon-prokaryote):
  <name>_train_bs{L}_wrapped_specialFalse.dat
  <name>_validation_bs{L}_wrapped_specialFalse.dat
  <name>_echo_manifest.json   -> per val row: [{source,target,gap,motif_len}, ...]
                                 (used by the targeted copy-accuracy metric).

See docs/synthetic_longrange_benchmark.md for the full protocol + metric.
"""
import argparse
import json
import os
import shutil

import numpy as np
import datasets

NUC = np.array([8, 9, 10, 11], dtype=np.int32)  # A C G T (DNATokenizer ids)

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=98304)
ap.add_argument('--n_train', type=int, default=8192)
ap.add_argument('--n_val', type=int, default=512)
ap.add_argument('--motif_len', type=int, default=32)
ap.add_argument('--n_echoes', type=int, default=6)          # echo pairs / sequence
ap.add_argument('--block_size', type=int, default=24576)    # informs cross-block gaps
ap.add_argument('--gaps', default='2048,30000,50000,75000')  # mix <block and >block
ap.add_argument('--name', default='synthLR')
ap.add_argument('--cache_dir',
                default='/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
L, M = args.length, args.motif_len
GAPS = [int(x) for x in args.gaps.split(',')]
assert all(g + M < L for g in GAPS), 'gap+motif must fit in length'

def make_seq(rng):
  """Random ACGT with up to n_echoes non-overlapping copy pairs."""
  x = NUC[rng.integers(0, 4, size=L)]
  used = np.zeros(L, bool)
  echoes = []
  tries = 0
  while len(echoes) < args.n_echoes and tries < args.n_echoes * 25:
    tries += 1
    g = GAPS[rng.integers(0, len(GAPS))]
    s = int(rng.integers(0, L - g - M))
    t = s + g
    if used[max(0, s - 1):s + M + 1].any() or used[max(0, t - 1):t + M + 1].any():
      continue
    motif = NUC[rng.integers(0, 4, size=M)]
    x[s:s + M] = motif
    x[t:t + M] = motif
    used[s:s + M] = True
    used[t:t + M] = True
    echoes.append({'source': s, 'target': t, 'gap': g, 'motif_len': M})
  return x, echoes

FEATS = datasets.Features({
  'input_ids': datasets.Sequence(datasets.Value('int32')),
  'attention_mask': datasets.Sequence(datasets.Value('float32'))})

def save(ds, mode):
  p = os.path.join(args.cache_dir, f'{args.name}_{mode}_bs{L}_wrapped_specialFalse.dat')
  tmp = p + '.tmp'
  shutil.rmtree(tmp, ignore_errors=True)
  ds.save_to_disk(tmp)
  shutil.rmtree(p, ignore_errors=True)
  os.rename(tmp, p)
  print(f'saved {mode}: rows={ds.num_rows} -> {p}', flush=True)

# --- train: stream via from_generator (memory-light; never holds all rows) ---
def train_gen():
  rng = np.random.default_rng(args.seed)
  for _ in range(args.n_train):
    x, _ = make_seq(rng)
    yield {'input_ids': x, 'attention_mask': np.ones(L, dtype=np.float32)}

print(f'building {args.name}: L={L} n_train={args.n_train} n_val={args.n_val} '
      f'motif={M} echoes={args.n_echoes} gaps={GAPS} block={args.block_size}', flush=True)
save(datasets.Dataset.from_generator(train_gen, features=FEATS), 'train')

# --- val: in memory (small) + echo manifest for the targeted metric ---
rng = np.random.default_rng(args.seed + 1)
val_rows, manifest = [], []
for _ in range(args.n_val):
  x, e = make_seq(rng)
  val_rows.append(x)
  manifest.append(e)
val = datasets.Dataset.from_dict(
  {'input_ids': [r for r in val_rows],
   'attention_mask': [np.ones(L, dtype=np.float32) for _ in val_rows]}, features=FEATS)
save(val, 'validation')
json.dump({'args': vars(args), 'echoes': manifest},
          open(os.path.join(args.cache_dir, f'{args.name}_echo_manifest.json'), 'w'))
n_pairs = sum(len(e) for e in manifest)
by_gap = {g: sum(1 for e in manifest for p in e if p['gap'] == g) for g in GAPS}
print(f'val echo pairs: {n_pairs} total | by gap: {by_gap} | '
      f'cross-block (>{args.block_size}): {sum(v for g,v in by_gap.items() if g>args.block_size)}',
      flush=True)
print('DONE', flush=True)
