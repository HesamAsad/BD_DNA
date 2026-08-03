"""Build HUMAN (hg38) long-range DNA caches for the dual BD3-LM.

Real long-range structure (gene architecture over 10-200 kb, isochores over
>100 kb, dispersed/satellite repeats) -- unlike prokaryote (~0 MI beyond ~1 kb).
Each row is ONE contiguous genomic window of length L (NO cross-window packing, so
the long-range structure inside a row is real single-locus DNA).

Strategies:
  gene    : GENCODE protein-coding genes -> window anchored on the TSS
            (flank5 upstream, gene body downstream), enriches gene-scale long range.
  uniform : random genomic windows (skip assembly-gap N runs), broad coverage
            (repeats + isochores).
  mix     : ~half gene, half uniform (used for TRAIN to honour "both" strategies).

Outputs (loadable via data.train=/data.valid=<name>, data=carbon-prokaryote):
  <name>_train_bs{L}_wrapped_specialFalse.dat
  <name>-gene_validation_bs{L}_wrapped_specialFalse.dat     (+ _echo_manifest-style json)
  <name>-uniform_validation_bs{L}_wrapped_specialFalse.dat

hg38 FASTA + GENCODE v41 are already staged on the cluster (defaults below).
"""
import argparse
import gzip
import json
import os
import shutil

import numpy as np
import datasets
from pyfaidx import Fasta

FA = '/lustre/scratch126/cellgen/lotfollahi/kz1/scprinter_cache/gencode_v41_GRCh38.fa.gz.decomp'
GFF = '/lustre/scratch126/cellgen/lotfollahi/kz1/scprinter_cache/gencode_v41_GRCh38.gff3.gz'
CACHE = '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon'
MAIN_CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=98304)
ap.add_argument('--n_train', type=int, default=8192)
ap.add_argument('--n_val', type=int, default=256)        # per strategy (gene, uniform)
ap.add_argument('--flank5', type=int, default=0)         # upstream of TSS; default L//4
ap.add_argument('--max_n_frac', type=float, default=0.05)
ap.add_argument('--name', default=None)                  # default human-lr{L}
ap.add_argument('--fasta', default=FA)
ap.add_argument('--gff', default=GFF)
ap.add_argument('--cache_dir', default=CACHE)
ap.add_argument('--seed', type=int, default=0)
# LEAKAGE FIX: validation windows are drawn ONLY from held-out chromosomes and train
# windows ONLY from the rest. The previous version drew both splits from the same
# genome-wide TSS pool, and gene_window() is deterministic given a TSS, so ~20% of
# val-gene windows were byte-identical to training windows (audited 53/256 at L=98304).
# Chromosome holdout also removes partial-overlap and same-locus/paralog leakage.
ap.add_argument('--val_chroms', default='chr8,chr9',
                help='held-out chromosomes used ONLY for validation')
args = ap.parse_args()
L = args.length
NAME = args.name or f'human-lr{L}'
FLANK5 = args.flank5 or (L // 4)

# A=8 C=9 G=10 T=11 N=12, anything else (incl. soft-mask handled by .upper()) -> UNK=7
LUT = np.full(256, 7, np.int32)
for ch, i in [('A', 8), ('C', 9), ('G', 10), ('T', 11), ('N', 12)]:
  LUT[ord(ch)] = i

fa = Fasta(args.fasta, sequence_always_upper=True, as_raw=True)
CHROM_LEN = {c: len(fa[c]) for c in MAIN_CHROMS if c in fa}
VAL_CHROMS = [c for c in args.val_chroms.split(',') if c in CHROM_LEN]
TRAIN_CHROMS = [c for c in CHROM_LEN if c not in set(VAL_CHROMS)]
assert VAL_CHROMS and TRAIN_CHROMS, f'bad chrom split: val={VAL_CHROMS}'

def tok_window(chrom, start):
  """Tokenise the L-window [start, start+L) on chrom; return None if too many N."""
  s = fa[chrom][start:start + L]
  if len(s) < L:
    return None
  ids = LUT[np.frombuffer(s.encode('ascii', 'replace'), np.uint8)]
  if (ids == 12).mean() > args.max_n_frac:   # too much assembly gap
    return None
  return ids

# ---- TSS list from GENCODE (protein-coding genes) ----
def load_tss():
  tss = []
  op = gzip.open(args.gff, 'rt')
  for line in op:
    if line[0] == '#':
      continue
    f = line.rstrip('\n').split('\t')
    if len(f) < 9 or f[2] != 'gene':
      continue
    if 'gene_type=protein_coding' not in f[8]:
      continue
    chrom, st, en, strand = f[0], int(f[3]) - 1, int(f[4]), f[6]
    if chrom not in CHROM_LEN:
      continue
    t = st if strand == '+' else en - 1
    tss.append((chrom, t, strand))
  op.close()
  return tss

def gene_window(chrom, t, strand, rng):
  start = (t - FLANK5) if strand == '+' else (t - (L - FLANK5))
  start = int(min(max(start, 0), CHROM_LEN[chrom] - L))
  return start

def uniform_start(rng, chroms):
  c = chroms[rng.integers(0, len(chroms))]
  if CHROM_LEN.get(c, 0) <= L:
    return None
  return c, int(rng.integers(0, CHROM_LEN[c] - L))

FEATS = datasets.Features({
  'input_ids': datasets.Sequence(datasets.Value('int32')),
  'attention_mask': datasets.Sequence(datasets.Value('float32'))})

def save(ds, suffix):
  p = os.path.join(args.cache_dir, f'{NAME}{suffix}_bs{L}_wrapped_specialFalse.dat')
  tmp = p + '.tmp'
  shutil.rmtree(tmp, ignore_errors=True)
  ds.save_to_disk(tmp)
  shutil.rmtree(p, ignore_errors=True)
  os.rename(tmp, p)
  print(f'saved {suffix or "_train"}: rows={ds.num_rows} -> {p}', flush=True)

print(f'{NAME}: L={L} flank5={FLANK5} | loading TSS...', flush=True)
TSS = load_tss()
print(f'protein-coding TSS on main chroms: {len(TSS)}', flush=True)

# ---- TRAIN: mix of gene + uniform, streamed (memory-light) ----
def train_gen():
  """Train windows: TRAIN chromosomes only, de-duplicated by (chrom, start)."""
  rng = np.random.default_rng(args.seed)
  tss_tr = [x for x in TSS if x[0] in set(TRAIN_CHROMS)]
  seen = set()
  made = 0
  while made < args.n_train:
    if rng.random() < 0.5 and tss_tr:            # gene
      chrom, t, strand = tss_tr[rng.integers(0, len(tss_tr))]
      start = gene_window(chrom, t, strand, rng)
    else:                                        # uniform
      u = uniform_start(rng, TRAIN_CHROMS)
      if not u:
        continue
      chrom, start = u
    if (chrom, start) in seen:                   # no duplicate windows
      continue
    ids = tok_window(chrom, start)
    if ids is None:
      continue
    seen.add((chrom, start))
    made += 1
    yield {'input_ids': ids, 'attention_mask': np.ones(L, np.float32)}

# Cap the Arrow writer buffer so the streamed build fits the head-node memory
# cgroup (default 1000 rows x L int32 OOMs at L>=~196k). ~250 MB per flush.
_wb = max(64, 250_000_000 // (L * 4))
save(datasets.Dataset.from_generator(
  train_gen, features=FEATS, writer_batch_size=_wb), '_train')

# ---- VAL: separate gene + uniform sets (in memory) + manifests ----
def build_val(kind):
  """Val windows: HELD-OUT chromosomes only, de-duplicated -> zero train overlap."""
  rng = np.random.default_rng(args.seed + (1 if kind == 'gene' else 2))
  tss_val = [x for x in TSS if x[0] in set(VAL_CHROMS)]
  seen = set()
  rows, mani = [], []
  guard = 0
  while len(rows) < args.n_val and guard < args.n_val * 200:
    guard += 1
    if kind == 'gene':
      if not tss_val:
        break
      chrom, t, strand = tss_val[rng.integers(0, len(tss_val))]
      start = gene_window(chrom, t, strand, rng)
      meta = {'chrom': chrom, 'win_start': int(start), 'tss': int(t),
              'tss_in_window': int(t - start), 'strand': strand}
    else:
      u = uniform_start(rng, VAL_CHROMS)
      if not u:
        continue
      chrom, start = u
      meta = {'chrom': chrom, 'win_start': int(start)}
    if (chrom, start) in seen:
      continue
    ids = tok_window(chrom, start)
    if ids is None:
      continue
    seen.add((chrom, start))
    rows.append(ids)
    mani.append(meta)
  ds = datasets.Dataset.from_dict(
    {'input_ids': [r for r in rows],
     'attention_mask': [np.ones(L, np.float32) for _ in rows]}, features=FEATS)
  save(ds, f'-{kind}_validation')
  json.dump({'args': vars(args), 'windows': mani},
            open(os.path.join(args.cache_dir, f'{NAME}-{kind}_manifest.json'), 'w'))

# train cache gets the _train suffix already; give val its own dataset_name base:
# the loader builds <data.valid>_validation_..., so name val datasets <NAME>-gene / <NAME>-uniform.
for kind in ('gene', 'uniform'):
  build_val(kind)
print('DONE', flush=True)
