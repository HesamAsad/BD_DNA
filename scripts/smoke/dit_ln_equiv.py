#!/usr/bin/env python
"""Numerical equivalence of the proposed LayerNorm rewrites, on CUDA.

Compares, against the current dit.py:234-237 implementation:
  fused  : F.layer_norm(x.float(), [D], weight=w)          (fp32, safe claim)
  native : F.layer_norm(x, [D], weight=w.to(x.dtype))      (bf16 out, risky)
in output and in both gradients, at the real shapes.
"""
import torch
import torch.nn.functional as F

B, L = 4, 8191


def run(D, dtype):
  torch.manual_seed(0)
  x0 = torch.randn(B, L, D, device='cuda', dtype=dtype)
  w0 = torch.randn(D, device='cuda') * 0.1 + 1.0
  g = torch.randn(B, L, D, device='cuda')

  def go(kind):
    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    with torch.amp.autocast('cuda', enabled=False):
      if kind == 'current':
        y = F.layer_norm(x.float(), [D])
        y = y * w[None, None, :]
      elif kind == 'fused':
        y = F.layer_norm(x.float(), [D], weight=w)
      else:
        y = F.layer_norm(x, [D], weight=w.to(x.dtype))
    y.backward(g.to(y.dtype))
    return y.float(), x.grad.float(), w.grad.float()

  ref = go('current')
  for kind in ('fused', 'native'):
    got = go(kind)
    rel = []
    for a, b in zip(ref, got):
      d = (a - b).abs().max().item()
      s = a.abs().max().item()
      rel.append(f'{d:.3e} (rel {d / max(s, 1e-30):.2e})')
    print(f'  D={D} in={str(dtype):>14} {kind:<7} '
          f'|dy|={rel[0]}  |dgx|={rel[1]}  |dgw|={rel[2]}')


if __name__ == '__main__':
  print(torch.cuda.get_device_name(0))
  print('max abs difference vs the current implementation:')
  for D in (832, 768):
    for dt in (torch.float32, torch.bfloat16):
      run(D, dt)
