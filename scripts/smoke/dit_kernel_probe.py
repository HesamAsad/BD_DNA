#!/usr/bin/env python
"""Kernel-level microbenchmarks behind the DiT audit.

1. GEMM efficiency at the Transformer's hidden size (832 = 6.5 x 128) against
   the SSM's (768 = 6 x 128) and the next clean size (896 = 7 x 128), at the
   exact shapes the block uses.
2. The LayerNorm variants (current two-kernel fp32, fused-affine fp32, native
   bf16), fwd+bwd, at [4, 8191, 832].
3. bias_dropout_add_scale: torch.jit.script wrapper vs plain Python, at
   dropout=0.0.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms')
from models import dit  # noqa: E402


def bench(fn, iters=30, warmup=10):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) / iters


def gemm_probe():
  print('\n=== bf16 GEMM efficiency vs hidden size ===')
  print(f"{'d':>5} {'M':>7} {'N':>6} {'K':>6} {'ms':>8} {'TFLOP/s':>9}")
  M = 4 * 8192
  for d in (768, 832, 896, 1024):
    for name, n, k in (('qkv', 3 * d, d), ('out', d, d),
                       ('fc1', 4 * d, d), ('fc2', d, 4 * d)):
      a = torch.randn(M, k, device='cuda', dtype=torch.bfloat16)
      b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16)
      t = bench(lambda: torch.mm(a, b))
      tflops = 2 * M * n * k / t / 1e12
      print(f'{d:>5} {M:>7} {n:>6} {k:>6} {t*1e3:>8.3f} {tflops:>9.1f}  {name}')
      del a, b
      torch.cuda.empty_cache()


def norm_probe():
  print('\n=== LayerNorm variants, fwd+bwd, [4, 8191, 832] ===')
  B, L, D = 4, 8191, 832
  w = torch.ones(D, device='cuda', requires_grad=True)

  def make_x(dtype):
    return torch.randn(B, L, D, device='cuda', dtype=dtype, requires_grad=True)

  def current(x):
    with torch.amp.autocast('cuda', enabled=False):
      y = F.layer_norm(x.float(), [D])
    return y * w[None, None, :]

  def fused(x):
    with torch.amp.autocast('cuda', enabled=False):
      return F.layer_norm(x.float(), [D], weight=w)

  def native(x):
    with torch.amp.autocast('cuda', enabled=False):
      return F.layer_norm(x, [D], weight=w.to(x.dtype))

  for label, fn, dtype in (('current (fp32 in)', current, torch.float32),
                           ('fused  (fp32 in)', fused, torch.float32),
                           ('current (bf16 in)', current, torch.bfloat16),
                           ('fused  (bf16 in)', fused, torch.bfloat16),
                           ('native (bf16 in)', native, torch.bfloat16)):
    x = make_x(dtype)
    g = None

    def step():
      nonlocal g
      y = fn(x)
      if g is None or g.shape != y.shape or g.dtype != y.dtype:
        g = torch.randn_like(y)
      y.backward(g)

    t = bench(step, iters=20, warmup=5)
    print(f'  {label:<20} {t*1e3:7.3f} ms   out dtype={fn(x).dtype}')
    del x
    torch.cuda.empty_cache()


def dropout_probe():
  print('\n=== bias_dropout_add_scale at dropout=0.0, [4, 8191, 832] bf16 ===')
  B, L, D = 4, 8191, 832
  x = torch.randn(B, L, D, device='cuda', dtype=torch.bfloat16,
                  requires_grad=True)
  res = torch.randn(B, L, D, device='cuda', dtype=torch.bfloat16)
  scale = torch.ones(1, device='cuda', dtype=torch.bfloat16)
  g = torch.randn_like(x)

  def jitted():
    y = dit.bias_dropout_add_scale_fused_train(x, None, scale, res, 0.0)
    y.backward(g)

  def plain():
    y = dit.bias_dropout_add_scale(x, None, scale, res, 0.0, True)
    y.backward(g)

  def minimal():
    y = res + F.dropout(x, 0.0, True)
    y.backward(g)

  for label, fn in (('torch.jit.script', jitted), ('plain python', plain),
                    ('res + dropout', minimal)):
    t = bench(fn, iters=30, warmup=10)
    before = torch.cuda.memory_allocated()
    y = fn.__self__ if False else None
    print(f'  {label:<18} {t*1e3:7.3f} ms')


if __name__ == '__main__':
  print(torch.cuda.get_device_name(0), torch.__version__)
  gemm_probe()
  norm_probe()
  dropout_probe()
