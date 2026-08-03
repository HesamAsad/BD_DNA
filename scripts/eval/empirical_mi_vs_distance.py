"""Empirical pairwise-nucleotide mutual information vs distance, on tokenized DNA caches.

Cheap CPU screen for H_signal (does real DNA carry distal single-nucleotide signal?).
MI(d) = sum_{a,b in ACGT} p(a,b) log2[ p(a,b) / (p(a) p(b)) ], where p(a,b) is the joint
of nucleotides at positions (i, i+d) pooled over all i and all sequences. This is a
NECESSARY-but-not-sufficient probe: it sees only pairwise single-nt dependence, so a null
result does not rule out higher-order / motif-level long-range structure (that is what the
model oracle in P1 is for). But a POSITIVE result at large d is hard evidence of real signal.

Reproduces the prokaryote reference (~0.018 bits at d=1 -> ~0 by 10 kb) as a control.

Usage:
  python scripts/eval/empirical_mi_vs_distance.py            # runs the built-in cache set
  python scripts/eval/empirical_mi_vs_distance.py <cache.dat> [n_seqs]
"""
import os
import sys

import numpy as np
import datasets

# DNATokenizer ids -> 0..3 for ACGT; everything else (N=12, specials) -> -1 (skipped).
LUT = np.full(256, -1, np.int64)
for tok, base in [(8, 0), (9, 1), (10, 2), (11, 3)]:
  LUT[tok] = base

DISTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
         5000, 10000, 20000, 50000]

CACHE_DIR = '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon'
DEFAULT_SETS = [
  ('human-gene',   f'{CACHE_DIR}/human-lr98304-gene_validation_bs98304_wrapped_specialFalse.dat'),
  ('human-unif',   f'{CACHE_DIR}/human-lr98304-uniform_validation_bs98304_wrapped_specialFalse.dat'),
  ('prok-control', f'{CACHE_DIR}/carbon-prokaryote_validation_bs98304_wrapped_specialFalse_nf1.dat'),
]


def mi_from_joint(joint):
  """joint: 4x4 counts -> MI in bits."""
  tot = joint.sum()
  if tot == 0:
    return float('nan'), 0
  p = joint / tot
  pa = p.sum(1, keepdims=True)
  pb = p.sum(0, keepdims=True)
  denom = pa * pb
  with np.errstate(divide='ignore', invalid='ignore'):
    terms = p * (np.log2(p) - np.log2(denom))
  return float(np.nansum(terms)), int(tot)


def mi_vs_distance(path, n_seqs):
  ds = datasets.load_from_disk(path).with_format('numpy')
  n = min(n_seqs, ds.num_rows)
  seqs = [LUT[np.asarray(ds[i]['input_ids'], dtype=np.int64) & 0xFF] for i in range(n)]
  out = {}
  for d in DISTS:
    joint = np.zeros((4, 4), np.int64)
    for s in seqs:
      if s.shape[0] <= d:
        continue
      a, b = s[:-d], s[d:]
      m = (a >= 0) & (b >= 0)
      if not m.any():
        continue
      idx = a[m] * 4 + b[m]
      joint += np.bincount(idx, minlength=16).reshape(4, 4)
    mi, tot = mi_from_joint(joint)
    out[d] = (mi, tot)
  return out, n


def main():
  if len(sys.argv) > 1:
    sets = [('custom', sys.argv[1])]
    n_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 128
  else:
    sets = DEFAULT_SETS
    n_seqs = int(os.environ.get('MI_NSEQ', '128'))

  print(f'MI(d) in bits | n_seqs<= {n_seqs} | ACGT only (N/specials skipped)\n')
  header = 'dist      ' + '  '.join(f'{name:>12s}' for name, _ in sets)
  print(header)
  print('-' * len(header))
  results = {}
  for name, path in sets:
    if not os.path.exists(path):
      print(f'# MISSING: {name} -> {path}', flush=True)
      results[name] = None
      continue
    results[name], used = mi_vs_distance(path, n_seqs)
    print(f'# loaded {name}: {used} seqs', flush=True)
  print()
  for d in DISTS:
    row = f'{d:>7d}   '
    for name, _ in sets:
      r = results.get(name)
      cell = f'{r[d][0]:.5f}' if r else 'NA'
      row += f'  {cell:>12s}'
    print(row, flush=True)
  print('\nDONE', flush=True)


if __name__ == '__main__':
  main()
