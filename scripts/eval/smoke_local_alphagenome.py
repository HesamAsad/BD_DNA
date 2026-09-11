#!/usr/bin/env python3
"""Smoke test: does the local (JAX, open-weights) AlphaGenome model actually
run on our GPU nodes, and does it work at our model's native 8192nt length?

Deliberately bypasses create_from_huggingface() (which would re-download the
1.4GB checkpoint fresh, since it calls snapshot_download with no cache_dir
and the existing download lives outside the standard HF cache layout) and
instead loads directly from the already-cached local checkpoint via create().

Uses get_embeddings_synthetic since our actual use case (Task 2 generated
sequences) has no real genomic position -- exercises the same code path we'd
actually use, not just the genomic-interval path from the sample script.
"""
import sys
import time

import jax
print(f"jax backend: {jax.default_backend()}", flush=True)
print(f"jax devices: {jax.devices()}", flush=True)

sys.path.insert(0, "/lustre/scratch126/cellgen/lotfollahi/ha11/alphagenome_research/src")

CKPT = "/lustre/scratch126/cellgen/lotfollahi/ha11/alphagenome_research/alphagenome-all-folds"

t0 = time.time()
from alphagenome_research.model import dna_model
model = dna_model.create(CKPT)
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

from alphagenome_research.model.embedding_utils import get_embeddings_synthetic

import random
random.seed(0)
seq_8192 = "".join(random.choice("ACGT") for _ in range(8192))
seq_131072 = "".join(random.choice("ACGT") for _ in range(131072))

for length_label, seq in [("131072 (recommended min)", seq_131072), ("8192 (native)", seq_8192)]:
  t0 = time.time()
  preds, emb = get_embeddings_synthetic(
      model, seq,
      requested_outputs=[dna_model.OutputType.RNA_SEQ],
      ontology_terms=["UBERON:0002048"],
  )
  dt = time.time() - t0
  print(f"\n=== length {length_label} ===", flush=True)
  print(f"predict+embed time: {dt:.1f}s", flush=True)
  print(f"rna_seq shape: {preds.rna_seq.values.shape}", flush=True)
  import numpy as np
  v = np.asarray(preds.rna_seq.values)
  print(f"rna_seq stats: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f} "
        f"nan_frac={np.isnan(v).mean():.4f}", flush=True)
  print(f"embeddings_1bp shape: {None if emb.embeddings_1bp is None else emb.embeddings_1bp.shape}", flush=True)
  print(f"embeddings_128bp shape: {None if emb.embeddings_128bp is None else emb.embeddings_128bp.shape}", flush=True)
  print(f"embeddings_pair shape: {None if emb.embeddings_pair is None else emb.embeddings_pair.shape}", flush=True)

print("\nsmoke_local_alphagenome exit=0", flush=True)
