"""Audit the cached DNA windows for data-integrity problems that would invalidate
the long-range experiments. Checks, per cache:

  1. STRUCTURE   - every row is exactly length L; dtype/shape sane.
  2. PURITY      - tokens are ACGTN only (ids 8-12); ZERO special tokens
                   (BOS=2 / EOS=3 / MASK=4 / PAD=5) mid-sequence. A special token
                   inside a row is a concatenation artifact.
  3. CONTIGUITY  - (val caches, using the build manifest's chrom/win_start) the
                   cached tokens EXACTLY match the reference genome re-extracted at
                   [win_start, win_start+L) on that chromosome. Exact match == the
                   row is ONE contiguous interval from ONE chromosome (hg38 is a
                   single haploid reference, so single-haplotype is inherent).
  4. LEAKAGE     - exact-duplicate windows shared between the TRAIN cache and each
                   VAL cache (hash of the first 4096 tokens; identical windows
                   collide, distinct genomic loci never do). Catches the
                   deterministic-TSS train/val overlap.

Usage: python scripts/eval/audit_human_caches.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import datasets

CACHE = '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon'
FA = '/lustre/scratch126/cellgen/lotfollahi/kz1/scprinter_cache/gencode_v41_GRCh38.fa.gz.decomp'

# Same LUT the builder used: A=8 C=9 G=10 T=11 N=12, everything else -> UNK=7.
LUT = np.full(256, 7, np.int32)
for ch, i in [('A', 8), ('C', 9), ('G', 10), ('T', 11), ('N', 12)]:
  LUT[ord(ch)] = i
SPECIAL = {0: 'CLS', 1: 'SEP', 2: 'BOS', 3: 'EOS', 4: 'MASK', 5: 'PAD', 6: 'RSV', 7: 'UNK'}

# (train_name, L, [(val_name, manifest_name, kind), ...])
GROUPS = [
  ('human-lr98304_train', 98304, [
    ('human-lr98304-gene_validation',    'human-lr98304-gene_manifest.json',    'gene'),
    ('human-lr98304-uniform_validation', 'human-lr98304-uniform_manifest.json', 'uniform')]),
  ('human-lr32768_train', 32768, [
    ('human-lr32768-gene_validation',    'human-lr32768-gene_manifest.json',    'gene'),
    ('human-lr32768-uniform_validation', 'human-lr32768-uniform_manifest.json', 'uniform')]),
  # v2 = chromosome-holdout rebuild; expect 0% leakage (val drawn ONLY from chr8/chr9)
  ('human-lr32768v2_train', 32768, [
    ('human-lr32768v2-gene_validation',    'human-lr32768v2-gene_manifest.json',    'gene'),
    ('human-lr32768v2-uniform_validation', 'human-lr32768v2-uniform_manifest.json', 'uniform')]),
]


def load(name, L):
  p = os.path.join(CACHE, f'{name}_bs{L}_wrapped_specialFalse.dat')
  if not os.path.exists(p):
    return None, p
  return datasets.load_from_disk(p).with_format('numpy'), p


def check_purity_structure(ds, L, tag, scan=None):
  n = ds.num_rows if scan is None else min(scan, ds.num_rows)
  bad_len = 0
  special_rows = 0
  special_counts = {}
  n_frac_max = 0.0
  for i in range(n):
    ids = np.asarray(ds[i]['input_ids'])
    if ids.shape[0] != L:
      bad_len += 1
    sp = ids[ids < 8]
    if sp.size:
      special_rows += 1
      for v in np.unique(sp):
        special_counts[int(v)] = special_counts.get(int(v), 0) + int((sp == v).sum())
    n_frac_max = max(n_frac_max, float((ids == 12).mean()))
  sc = {SPECIAL.get(k, k): v for k, v in sorted(special_counts.items())}
  ok = (bad_len == 0 and special_rows == 0)
  print(f"  [{tag}] rows={ds.num_rows} scanned={n} | wrong_len={bad_len} | "
        f"rows_with_special={special_rows} {sc if sc else ''} | maxN_frac={n_frac_max:.3f} "
        f"-> {'PASS' if ok else 'FAIL'}")
  return ok


def check_contiguity(ds, L, manifest_path, fa, n_check=20):
  mp = os.path.join(CACHE, manifest_path)
  if not os.path.exists(mp):
    print(f"  [contiguity] manifest missing: {manifest_path} -> SKIP")
    return None
  windows = json.load(open(mp))['windows']
  n = min(n_check, ds.num_rows, len(windows))
  exact, mism = 0, []
  for i in range(n):
    w = windows[i]
    chrom, start = w['chrom'], int(w['win_start'])
    if chrom not in fa:
      continue
    ref = fa[chrom][start:start + L]
    if len(ref) < L:
      continue
    ref_ids = LUT[np.frombuffer(ref.encode('ascii', 'replace'), np.uint8)]
    cached = np.asarray(ds[i]['input_ids'])
    if np.array_equal(ref_ids, cached):
      exact += 1
    else:
      d = int((ref_ids != cached).sum())
      mism.append((i, chrom, start, d))
  ok = (exact == n and n > 0)
  print(f"  [contiguity] {exact}/{n} rows EXACTLY match reference genome at "
        f"(chrom, win_start, +{L}) -> {'PASS' if ok else 'FAIL'}")
  for (i, c, s, d) in mism[:5]:
    print(f"      MISMATCH row {i}: {c}:{s} differs at {d}/{L} positions")
  return ok


def fp(ids, k=4096):
  return hashlib.blake2b(np.asarray(ids[:k], np.int32).tobytes(), digest_size=16).digest()


def check_leakage(train_ds, val_specs, L):
  print("  [leakage] hashing train windows (first 4096 tok)...", flush=True)
  train_hashes = set()
  for i in range(train_ds.num_rows):
    train_hashes.add(fp(train_ds[i]['input_ids']))
  print(f"      train rows={train_ds.num_rows}, unique fingerprints={len(train_hashes)}"
        f" ({train_ds.num_rows - len(train_hashes)} internal dups)")
  allok = True
  for (val_name, _mani, kind) in val_specs:
    vds, _ = load(val_name, L)
    if vds is None:
      continue
    leaked = sum(1 for i in range(vds.num_rows) if fp(vds[i]['input_ids']) in train_hashes)
    vuniq = len({fp(vds[i]['input_ids']) for i in range(vds.num_rows)})
    ok = (leaked == 0)
    allok &= ok
    print(f"      val[{kind}] rows={vds.num_rows} unique={vuniq} | "
          f"EXACT-DUP with train = {leaked} ({100*leaked/max(vds.num_rows,1):.1f}%) "
          f"-> {'PASS' if ok else 'LEAK'}")
  return allok


def main():
  print(f"FASTA: {FA}\n{'exists' if os.path.exists(FA) else 'MISSING'}\n")
  fa = None
  if os.path.exists(FA):
    from pyfaidx import Fasta
    fa = Fasta(FA, sequence_always_upper=True, as_raw=True)

  for train_name, L, val_specs in GROUPS:
    print(f"\n{'='*70}\nGROUP {train_name}  (L={L})\n{'='*70}")
    tds, tp = load(train_name, L)
    if tds is None:
      print(f"  TRAIN cache MISSING: {tp}")
      continue
    print("STRUCTURE + PURITY:")
    check_purity_structure(tds, L, 'train', scan=500)
    for (val_name, mani, kind) in val_specs:
      vds, _ = load(val_name, L)
      if vds is None:
        print(f"  [{kind}] val cache missing"); continue
      check_purity_structure(vds, L, f'val-{kind}')
    if fa is not None:
      print("CONTIGUITY vs reference:")
      for (val_name, mani, kind) in val_specs:
        vds, _ = load(val_name, L)
        if vds is not None:
          check_contiguity(vds, L, mani, fa)
    else:
      print("CONTIGUITY: SKIPPED (FASTA unavailable)")
    print("TRAIN/VAL LEAKAGE:")
    check_leakage(tds, val_specs, L)

  # ---- synthLR sanity (P2 data; built directly, should be pure ACGT + exact echoes) ----
  print(f"\n{'='*70}\nsynthLR24k SANITY (P2 data)\n{'='*70}")
  sds, sp = load('synthLR24k_validation', 24576)
  if sds is not None:
    check_purity_structure(sds, 24576, 'synthLR-val', scan=200)
    man = os.path.join(CACHE, 'synthLR24k_echo_manifest.json')
    if os.path.exists(man):
      echoes = json.load(open(man))['echoes']
      okc = 0; tot = 0
      for i in range(min(50, sds.num_rows)):
        row = np.asarray(sds[i]['input_ids'])
        for e in echoes[i]:
          s, t, m = e['source'], e['target'], e['motif_len']
          tot += 1
          if np.array_equal(row[s:s+m], row[t:t+m]):
            okc += 1
      print(f"  [echo integrity] {okc}/{tot} echo pairs are exact source==target copies "
            f"-> {'PASS' if okc == tot and tot > 0 else 'FAIL'}")
  print("\nAUDIT DONE", flush=True)


if __name__ == '__main__':
  main()
