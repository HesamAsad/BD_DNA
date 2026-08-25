#!/usr/bin/env python3
"""Exact aten-op and memory-traffic audit of one DiT block, BD vs AR. CPU only.

The analytic FLOP formula (scripts/eval/training_flops.py:45-52) explicitly
ignores "elementwise ops, norms and softmax ... each below 2% of total".  That
is true of FLOPs and irrelevant to TIME: those ops are bandwidth- and
launch-bound, so they can cost a large fraction of the step while contributing
~0 to the denominator.  This script counts them exactly.

It runs a real forward+backward of

  DDiTBlock        (Transformer-BD:  d=768, adaLN on, dropout 0.1, split qkv)
  DDiTBlockCausal  (Transformer-AR:  d=832, adaLN off, dropout 0.0)

under a TorchDispatchMode that records every aten call and the bytes of its
tensor inputs and outputs, then splits the tally into GEMM-shaped work (which
the FLOP formula charges) and everything else (which it does not).

Attention itself is excluded from the comparison -- the roofline fit in
scripts/smoke/h4_transformer_bd_decompose.py handles that term separately.

Usage:  python scripts/smoke/dit_block_traffic_audit.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.dit import (  # noqa: E402
    DDiTBlock, DDiTBlockCausal, Rotary, block_diff_mask,
    apply_rotary_pos_emb_torchscript)

GEMM_OPS = {
  "mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "einsum",
  "_scaled_dot_product_attention_math", "_scaled_dot_product_flash_attention",
  "_scaled_dot_product_efficient_attention", "scaled_dot_product_attention",
  "_scaled_dot_product_flash_attention_backward",
  "_scaled_dot_product_efficient_attention_backward",
  "convolution", "convolution_backward",
}
# ops that are pure views / metadata: no traffic
VIEW_OPS = {
  "view", "_unsafe_view", "reshape", "expand", "permute", "transpose", "t",
  "slice", "select", "squeeze", "unsqueeze", "detach", "alias", "as_strided",
  "narrow", "split", "chunk", "split_with_sizes", "contiguous", "_to_copy_",
}


def _bytes(x):
  if isinstance(x, torch.Tensor):
    return x.numel() * x.element_size()
  if isinstance(x, (list, tuple)):
    return sum(_bytes(v) for v in x)
  return 0


class TrafficMode(TorchDispatchMode):
  def __init__(self):
    self.calls = defaultdict(int)
    self.traffic = defaultdict(int)

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}
    name = func.overloadpacket.__name__
    out = func(*args, **kwargs)
    if name not in VIEW_OPS:
      self.calls[name] += 1
      self.traffic[name] += _bytes(args) + _bytes(list(kwargs.values())) + _bytes(out)
    return out

  def split(self):
    gemm = sum(v for k, v in self.traffic.items() if k in GEMM_OPS)
    other = sum(v for k, v in self.traffic.items() if k not in GEMM_OPS)
    ngemm = sum(v for k, v in self.calls.items() if k in GEMM_OPS)
    nother = sum(v for k, v in self.calls.items() if k not in GEMM_OPS)
    return gemm, other, ngemm, nother


def run_bd(L, block_size, batch=1):
  torch.manual_seed(0)
  d, heads, cond = 768, 12, 128
  blk = DDiTBlock(n=L, dim=d, n_heads=heads, adaLN=True, cond_dim=cond,
                  dropout=0.1, block_size=block_size, attn_backend="sdpa").train()
  rot = Rotary(d // heads)
  x = torch.randn(batch, 2 * L, d, requires_grad=True)
  c = torch.randn(batch, cond)
  cos, sin = rot(x[:, :L])
  mask = block_diff_mask(
    b=None, h=None, q_idx=torch.arange(L * 2)[:, None],
    kv_idx=torch.arange(L * 2)[None, :], block_size=block_size, n=L)
  with TrafficMode() as m:
    y = blk(x, (cos, sin), c=c, causal=False, mask=mask)
    y.sum().backward()
  return m, 2 * L * batch          # tokens through the backbone


def run_ar(L, batch=1):
  torch.manual_seed(0)
  d, heads, cond = 832, 13, 128
  blk = DDiTBlockCausal(n=L, dim=d, n_heads=heads, adaLN=False, cond_dim=cond,
                        dropout=0.0, attn_backend="sdpa").train()
  rot = Rotary(d // heads)
  x = torch.randn(batch, L, d, requires_grad=True)
  cos, sin = rot(x)
  with TrafficMode() as m:
    y = blk(x, (cos, sin), c=None, causal=True)
    y.sum().backward()
  return m, L * batch


def rotary_only(L, d, heads, batch=1):
  """Isolate `apply_rotary_pos_emb_torchscript` on a packed qkv."""
  torch.manual_seed(0)
  rot = Rotary(d // heads)
  probe = torch.zeros(batch, L, d)
  cos, sin = rot(probe)
  qkv = torch.randn(batch, L, 3, heads, d // heads, requires_grad=True)
  with TrafficMode() as m:
    y = apply_rotary_pos_emb_torchscript(qkv, cos, sin)
    y.sum().backward()
  return m


def report(tag, m, tokens, exclude_attn=True):
  gemm, other, ngemm, nother = m.split()
  if exclude_attn:
    attn_traffic = sum(v for k, v in m.traffic.items()
                       if "scaled_dot_product" in k)
    gemm -= attn_traffic
  print(f"\n{tag}  ({tokens} tokens)")
  print(f"  GEMM-shaped ops : {ngemm:>4} calls   {gemm/tokens:>10,.0f} B/token")
  print(f"  everything else : {nother:>4} calls   {other/tokens:>10,.0f} B/token")
  print(f"  ratio other/GEMM traffic: {other/max(gemm,1):.2f}")
  print(f"  top non-GEMM ops by traffic (B/token):")
  rows = sorted(((v, k) for k, v in m.traffic.items()
                 if k not in GEMM_OPS), reverse=True)
  for v, k in rows[:14]:
    print(f"      {k:<38} {m.calls[k]:>3}x  {v/tokens:>10,.0f}")
  return other / tokens, nother


def main():
  L, block = 512, 256
  print("=" * 78)
  print("ONE DiT BLOCK, forward + backward, exact aten traffic (attention excluded)")
  print("=" * 78)
  m_bd, tok_bd = run_bd(L, block)
  m_ar, tok_ar = run_ar(L)
  bd_bt, bd_n = report("Transformer-BD  DDiTBlock   (d=768, adaLN, dropout 0.1)",
                       m_bd, tok_bd)
  ar_bt, ar_n = report("Transformer-AR  DDiTBlockCausal (d=832, no adaLN, dropout 0)",
                       m_ar, tok_ar)

  print("\n" + "=" * 78)
  print("PER-TOKEN NON-GEMM TRAFFIC, BD vs AR")
  print("=" * 78)
  print(f"  BD {bd_bt:>10,.0f} B/token/layer over {bd_n} ops")
  print(f"  AR {ar_bt:>10,.0f} B/token/layer over {ar_n} ops"
        f"   (torchscript rotary; the real AR arm uses flash's fused kernel)")
  print(f"  BD/AR = {bd_bt/ar_bt:.2f}x traffic, {bd_n/ar_n:.2f}x ops")

  # The AR arm does NOT run the torchscript rotary: dit.py:363-365 routes
  # attn_backend == 'flash_attn' to flash_attn's fused in-place kernel.
  m_rot = rotary_only(L, 832, 13)
  rot_ar_traffic = sum(m_rot.traffic.values()) / L
  m_rot_bd = rotary_only(L, 768, 12)
  rot_bd_traffic = sum(m_rot_bd.traffic.values()) / L
  # flash apply_rotary_emb_qkv_ is in-place over q,k only: read+write 2/3 of the
  # packed tensor, fwd and bwd.
  flash_equiv = 2 * (2 * (2 * 832) * 2)      # fwd(r+w) + bwd(r+w), bf16=2B
  print("\n" + "=" * 78)
  print("THE ROTARY, ISOLATED  (`apply_rotary_pos_emb_torchscript`, dit.py:201)")
  print("=" * 78)
  print(f"  BD path, d=768 : {m_rot_bd.calls and sum(m_rot_bd.calls.values()):>3} aten ops, "
        f"{rot_bd_traffic:>10,.0f} B/token/layer (fp32 probe; halve for bf16)")
  print(f"  same code d=832: {sum(m_rot.calls.values()):>3} aten ops, "
        f"{rot_ar_traffic:>10,.0f} B/token/layer")
  print(f"  flash fused in-place equivalent (analytic, bf16): "
        f"{flash_equiv:>10,.0f} B/token/layer")
  print(f"  -> the AR arm replaces {sum(m_rot.calls.values())} kernels with 1, and "
        f"{rot_bd_traffic/2:,.0f} -> {flash_equiv:,.0f} B/token/layer in bf16 "
        f"= {(rot_bd_traffic/2)/flash_equiv:.1f}x less traffic")
  print(f"  ops: {dict(m_rot_bd.calls)}")

  print("\n" + "=" * 78)
  print("IS THE TORCHSCRIPT FUSER EVEN ON?  dit.py:26-29")
  print("=" * 78)
  print(f"  _jit_set_profiling_executor(False) at dit.py:27 disables the "
        f"profiling executor,\n  which is what the TorchScript fuser runs "
        f"under.  `modulate_fused` (dit.py:151) and\n  "
        f"`bias_dropout_add_scale_fused_train` (dit.py:129) therefore execute "
        f"as separate\n  eager kernels despite the name.  The op counts above "
        f"are the evidence.")


if __name__ == "__main__":
  main()
