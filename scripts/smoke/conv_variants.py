#!/usr/bin/env python
"""Lens A: stop `_causal_conv` retaining 208 MiB/layer.

The census in `ssm_kernel_argprobe.py` showed one Mamba-2 layer retains two
full-width conv tensors at batch 4 / L=8192:

  [4, 1664, 8196] bf16  104.1 MiB   ConvolutionBackward0._saved_input  (`history`)
  [4, 8192, 1664] bf16  104.0 MiB   SiluBackward0._saved_self          (`convolved`)

`history = cat(initial_state, raw)` is a *fresh contiguous copy* of the whole
projected xBC stream, so it is a second copy of memory the graph already holds:
`z` is a view of the same `in_proj` output, and the fused gated RMSNorm keeps
`z`, which pins the entire 201.5 MiB `zxbcdt` storage regardless.  If the conv
reads a *view* of `zxbcdt` instead of a copy, its saved input is free.

Variants compared, all numerically equal to `base`:

  base      the repo as-is
  ckpt      torch.utils.checkpoint around _causal_conv (round-1 lead A(i))
  pad       no `cat`: F.conv1d(raw_view, padding=d_conv-1) for the body plus a
            width-7 conv for the d_conv-1 head tokens that need the boundary
            state.  Kills the `history` save only.
  pad+ckpt  the same restructuring inside a checkpoint: the recompute now reads
            a view rather than rebuilding a 104 MiB cat.
  shift     no conv at all: the depthwise kernel written as 4 broadcast
            multiply-adds over slices of xBC in its native [b, l, c] layout, so
            nothing is transposed and every operand is a view.
  attn      causal flash-attention block, d=832, as calibration.

Three measurements: numerical equivalence, a saved-tensor census of one mixer,
and peak memory + step time for a 12-layer stack (fwd+bwd+AdamW).  With
--real-model it also runs the production `Diffusion._loss` step for the
`ussm-ar` and `bissm` arms through `sizing_sweep.run_case`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import models.bidirectional_ssm as bs  # noqa: E402
from models import mamba2_segment as m2  # noqa: E402


# ---------------------------------------------------------------- variants


def causal_conv_pad(self, xBC, initial_state):
  """Depthwise causal conv continued from a boundary state, without the cat."""
  raw = xBC.transpose(1, 2)
  seqlen = raw.shape[-1]
  pad = self.d_conv - 1
  weight, bias = self.conv1d.weight, self.conv1d.bias
  state = initial_state.to(dtype=raw.dtype)
  if seqlen <= pad:
    history = torch.cat((state, raw), dim=-1)
    convolved = F.conv1d(history, weight, bias, groups=self.conv_dim)
    convolved = convolved.narrow(-1, 1, seqlen).transpose(1, 2)
    return F.silu(convolved), history.narrow(
      -1, history.shape[-1] - self.d_conv, self.d_conv).contiguous()
  # Body: zero-padded conv over a *view* of the in_proj output.  Output j uses
  # raw[j - pad .. j], i.e. the causal window, with zeros where the boundary
  # state should be; only the first `pad` outputs are therefore wrong.
  body = F.conv1d(raw, weight, bias, groups=self.conv_dim, padding=pad)
  # Head: the same conv over [state | raw[:pad]] -- 7 columns wide.
  head = F.conv1d(
    torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
    weight, bias, groups=self.conv_dim).narrow(-1, 1, pad)
  convolved = torch.cat((head, body.narrow(-1, pad, seqlen - pad)), dim=-1)
  return (F.silu(convolved.transpose(1, 2)),
          raw.narrow(-1, seqlen - self.d_conv, self.d_conv).contiguous())


def causal_conv_pad2(self, xBC, initial_state):
  """`pad`, but the head is written into the body instead of re-concatenated.

  `pad` pays one extra full-width copy: `cat(head, body[pad:])` rewrites all
  8192 columns to fix 3 of them. Writing the 3 columns in place keeps the
  body's own storage, so the only full-width traffic is the conv itself.
  """
  raw = xBC.transpose(1, 2)
  seqlen = raw.shape[-1]
  pad = self.d_conv - 1
  weight, bias = self.conv1d.weight, self.conv1d.bias
  state = initial_state.to(dtype=raw.dtype)
  if seqlen <= pad:
    return m2.SegmentMamba2._causal_conv_base(self, xBC, initial_state)
  full = F.conv1d(raw, weight, bias, groups=self.conv_dim, padding=pad)
  head = F.conv1d(
    torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
    weight, bias, groups=self.conv_dim).narrow(-1, 1, pad)
  # Autograd allows this: a convolution's backward never reads its own output,
  # so overwriting the d_conv-1 columns that the zero padding got wrong only
  # inserts a CopySlices node.
  full.narrow(-1, 0, pad).copy_(head)
  return (F.silu(full.narrow(-1, 0, seqlen).transpose(1, 2)),
          raw.narrow(-1, seqlen - self.d_conv, self.d_conv).contiguous())


def causal_conv_convckpt(self, xBC, initial_state):
  """Keep the fast contiguous conv; recompute it in backward instead of
  storing `history`.  Forward is byte-for-byte the repo's; the backward pays
  one extra cat + conv, and `convolved` is still stored for the SiLU."""
  raw = xBC.transpose(1, 2)
  seqlen = raw.shape[-1]
  convolved = torch.utils.checkpoint.checkpoint(
    _conv_body, self, raw, initial_state, use_reentrant=False)
  if seqlen < self.d_conv:
    final = torch.cat(
      (initial_state.to(dtype=raw.dtype), raw), dim=-1).narrow(
        -1, seqlen, self.d_conv).contiguous()
  else:
    final = raw.narrow(-1, seqlen - self.d_conv, self.d_conv).contiguous()
  return F.silu(convolved), final


def _conv_body(self, raw, initial_state):
  history = torch.cat((initial_state.to(dtype=raw.dtype), raw), dim=-1)
  convolved = F.conv1d(history, self.conv1d.weight, self.conv1d.bias,
                       groups=self.conv_dim)
  return convolved.narrow(-1, 1, raw.shape[-1]).transpose(1, 2)


def causal_conv_shift(self, xBC, initial_state):
  """The depthwise kernel as 4 broadcast multiply-adds in [b, l, c] layout."""
  seqlen = xBC.shape[1]
  pad = self.d_conv - 1
  weight = self.conv1d.weight.squeeze(1).to(dtype=xBC.dtype)  # [c, k]
  bias = self.conv1d.bias.to(dtype=xBC.dtype)
  state = initial_state.to(dtype=xBC.dtype).transpose(1, 2)  # [b, d_conv, c]
  if seqlen <= pad:
    return m2.SegmentMamba2._causal_conv_base(self, xBC, initial_state)

  def taps(source, out_len, offset):
    acc = source.narrow(1, offset, out_len) * weight[:, 0]
    for k in range(1, self.d_conv):
      acc = torch.addcmul(
        acc, source.narrow(1, offset + k, out_len), weight[:, k])
    return acc + bias

  head_source = torch.cat((state, xBC.narrow(1, 0, pad)), dim=1)
  convolved = torch.cat(
    (taps(head_source, pad, 1), taps(xBC, seqlen - pad, 0)), dim=1)
  return (F.silu(convolved),
          xBC.narrow(1, seqlen - self.d_conv, self.d_conv)
             .transpose(1, 2).contiguous())


def causal_conv_checkpointed(self, xBC, initial_state):
  return torch.utils.checkpoint.checkpoint(
    self._causal_conv_inner, xBC, initial_state, use_reentrant=False)


def swbb_pad(self, u, block_size):
  """`scan_with_block_boundaries` without the zero-state cat.

  The initial state is zero here, so the zero-padded conv is already exact --
  no head correction at all.  The per-block conv boundary windows come from
  strided views of `raw` plus an explicit zero window for block 0.
  """
  if u.ndim != 3 or u.shape[-1] != self.d_model:
    raise ValueError(
      f"Expected [batch, length, {self.d_model}], received {tuple(u.shape)}")
  batch_size, seqlen, _ = u.shape
  if block_size <= 0 or seqlen % block_size:
    raise ValueError("bad block size")
  num_seg = seqlen // block_size
  if block_size < self.d_conv:
    return m2.SegmentMamba2._swbb_base(self, u, block_size)

  zxbcdt = self.in_proj(u)
  z, xBC, dt = torch.split(
    zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
  raw = xBC.transpose(1, 2)
  pad = self.d_conv - 1
  convolved = F.conv1d(
    raw, self.conv1d.weight, self.conv1d.bias,
    groups=self.conv_dim, padding=pad).narrow(-1, 0, seqlen)
  xBC = F.silu(convolved.transpose(1, 2))
  # Window i holds the d_conv raw inputs preceding token i * block_size.
  windows = raw.narrow(
    -1, block_size - self.d_conv, seqlen - block_size + self.d_conv
  ).unfold(-1, self.d_conv, block_size).permute(0, 2, 1, 3)
  conv_states = torch.cat(
    (raw.new_zeros(batch_size, 1, self.conv_dim, self.d_conv), windows), dim=1)

  x, B, C = torch.split(
    xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
  x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
  B = rearrange(B, "b l n -> b l 1 n")
  C = rearrange(C, "b l n -> b l 1 n")

  backend = self._select_backend(u)
  y, _ = self._scan(x, dt, B, C, None, backend)

  def fold(tensor):
    return tensor.reshape(
      batch_size * num_seg, block_size, *tensor.shape[2:])

  _, local_ssm = self._scan(fold(x), fold(dt), fold(B), fold(C), None, backend)
  local_ssm = local_ssm.reshape(
    batch_size, num_seg, self.nheads, self.headdim, self.d_state)
  ssm_states = self._block_state_passing(dt, local_ssm, block_size)
  return self._gated_output(y, z), conv_states, ssm_states


CONV_IMPLS = {"pad": causal_conv_pad, "shift": causal_conv_shift,
              "pad2": causal_conv_pad2, "convckpt": causal_conv_convckpt}


def swbb_cmaj(self, u, block_size):
  """`scan_with_block_boundaries` with the channel-major xBC projection."""
  if u.ndim != 3 or u.shape[-1] != self.d_model:
    raise ValueError("bad input")
  batch_size, seqlen, _ = u.shape
  if block_size <= 0 or seqlen % block_size or block_size < self.d_conv:
    return m2.SegmentMamba2._swbb_base(self, u, block_size)
  num_seg = seqlen // block_size
  weight = self.in_proj.weight
  z = F.linear(u, weight[:self.d_inner])
  dt = F.linear(u, weight[self.d_inner + self.conv_dim:])
  raw = torch.matmul(
    weight[self.d_inner:self.d_inner + self.conv_dim], u.transpose(1, 2))
  pad = self.d_conv - 1
  convolved = F.conv1d(
    raw, self.conv1d.weight, self.conv1d.bias,
    groups=self.conv_dim, padding=pad).narrow(-1, 0, seqlen)
  xBC = F.silu(convolved.transpose(1, 2))
  windows = raw.narrow(
    -1, block_size - self.d_conv, seqlen - block_size + self.d_conv
  ).unfold(-1, self.d_conv, block_size).permute(0, 2, 1, 3)
  conv_states = torch.cat(
    (raw.new_zeros(batch_size, 1, self.conv_dim, self.d_conv), windows), dim=1)

  x, B, C = torch.split(
    xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
  x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
  B = rearrange(B, "b l n -> b l 1 n")
  C = rearrange(C, "b l n -> b l 1 n")
  backend = self._select_backend(u)
  y, _ = self._scan(x, dt, B, C, None, backend)

  def fold(tensor):
    return tensor.reshape(
      batch_size * num_seg, block_size, *tensor.shape[2:])

  _, local_ssm = self._scan(fold(x), fold(dt), fold(B), fold(C), None, backend)
  local_ssm = local_ssm.reshape(
    batch_size, num_seg, self.nheads, self.headdim, self.d_state)
  ssm_states = self._block_state_passing(dt, local_ssm, block_size)
  return self._gated_output(y, z), conv_states, ssm_states


def scan_segment_cmaj(self, u, initial_state=None):
  """`scan_segment` with the xBC projection produced channel-major.

  `pad`/`pad2` remove the `cat` but leave the conv reading a *strided* view,
  and the measured price is in the conv's backward.  The strided view exists
  only because `in_proj` emits [b, l, 3224] while the conv wants [b, c, l].
  Splitting that one GEMM into three slices of the same weight -- z, dt as
  usual, xBC as `W_xBC @ u^T` -- makes the conv's input contiguous *and* the
  projection output itself, so the conv retains a tensor the graph was already
  paying for.  No new parameters: the slices are views of `in_proj.weight`, so
  checkpoints load unchanged.
  """
  if u.ndim != 3 or u.shape[-1] != self.d_model:
    raise ValueError(
      f"Expected [batch, length, {self.d_model}], received {tuple(u.shape)}")
  if self.in_proj.bias is not None:
    raise NotImplementedError("cmaj assumes in_proj has no bias")
  batch_size, seqlen, _ = u.shape
  if initial_state is None:
    initial_state = self.zero_state(batch_size, device=u.device, dtype=u.dtype)
  weight = self.in_proj.weight
  w_z = weight[:self.d_inner]
  w_xbc = weight[self.d_inner:self.d_inner + self.conv_dim]
  w_dt = weight[self.d_inner + self.conv_dim:]

  z = F.linear(u, w_z)
  dt = F.linear(u, w_dt)
  # [c, d] @ [b, d, l] -> [b, c, l], contiguous, no transpose of the result.
  raw = torch.matmul(w_xbc, u.transpose(1, 2))

  pad = self.d_conv - 1
  cw, cb = self.conv1d.weight, self.conv1d.bias
  state = initial_state.conv.to(dtype=raw.dtype)
  full = F.conv1d(raw, cw, cb, groups=self.conv_dim, padding=pad)
  head = F.conv1d(torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
                  cw, cb, groups=self.conv_dim).narrow(-1, 1, pad)
  full.narrow(-1, 0, pad).copy_(head)
  xBC = F.silu(full.narrow(-1, 0, seqlen).transpose(1, 2))
  final_conv = raw.narrow(-1, seqlen - self.d_conv, self.d_conv).contiguous()

  x, B, C = torch.split(
    xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
  x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
  B = rearrange(B, "b l n -> b l 1 n")
  C = rearrange(C, "b l n -> b l 1 n")
  y, final_ssm = self._scan(
    x, dt, B, C, initial_state.ssm, self._select_backend(u))
  return (self._gated_output(y, z),
          m2.Mamba2State(final_conv, final_ssm))


def patch_class(variant):
  cls = m2.SegmentMamba2
  cls._causal_conv_base = cls._causal_conv
  cls._swbb_base = cls.scan_with_block_boundaries
  parts = set(variant.split("+"))
  inner = cls._causal_conv
  for name, fn in CONV_IMPLS.items():
    if name in parts:
      inner = fn
  cls._causal_conv_inner = inner
  cls._causal_conv = (causal_conv_checkpointed if "ckpt" in parts else inner)
  if parts & {"pad", "pad2"}:
    cls.scan_with_block_boundaries = swbb_pad
  if "cmaj" in parts:
    cls._scan_segment_base = cls.scan_segment
    cls.scan_segment = scan_segment_cmaj
    cls.scan_with_block_boundaries = swbb_cmaj
  return parts


def restore_class():
  cls = m2.SegmentMamba2
  if hasattr(cls, "_scan_segment_base"):
    cls.scan_segment = cls._scan_segment_base
    del cls._scan_segment_base
  if hasattr(cls, "_causal_conv_base"):
    cls._causal_conv = cls._causal_conv_base
    cls.scan_with_block_boundaries = cls._swbb_base
    del cls._causal_conv_base, cls._swbb_base
  if hasattr(cls, "_causal_conv_inner"):
    del cls._causal_conv_inner


# ---------------------------------------------------------------- models


class SSMStack(nn.Module):
  def __init__(self, n_layers=12, dim=768, vocab=13):
    super().__init__()
    self.embed = nn.Embedding(vocab, dim)
    self.layers = nn.ModuleList([
      bs.BiMambaLayer(dim=dim, d_state=64, d_conv=4, expand=2, headdim=64,
                      chunk_size=128, mlp_ratio=4.0, dropout=0.0,
                      backend="auto")
      for _ in range(n_layers)])
    self.norm = bs.RMSNorm(dim)
    self.head = nn.Linear(dim, vocab, bias=False)

  def forward(self, idx):
    x = self.embed(idx)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      for layer in self.layers:
        x, _ = layer.scan_clean(x, None)
      x = self.norm(x)
    return self.head(x.float())


class AttnBlock(nn.Module):
  def __init__(self, dim, n_heads):
    super().__init__()
    self.n_heads = n_heads
    self.norm1 = bs.RMSNorm(dim)
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.proj = nn.Linear(dim, dim, bias=False)
    self.norm2 = bs.RMSNorm(dim)
    self.mlp = bs.FeedForward(dim, 4.0, 0.0)

  def forward(self, x):
    import flash_attn
    qkv = self.qkv(self.norm1(x))
    qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3,
                    h=self.n_heads)
    y = flash_attn.flash_attn_qkvpacked_func(qkv, causal=True)
    x = x + self.proj(rearrange(y, "b s h d -> b s (h d)"))
    return x + self.mlp(self.norm2(x))


class AttnStack(nn.Module):
  def __init__(self, n_layers=12, dim=832, n_heads=13, vocab=13):
    super().__init__()
    self.embed = nn.Embedding(vocab, dim)
    self.layers = nn.ModuleList(
      [AttnBlock(dim, n_heads) for _ in range(n_layers)])
    self.norm = bs.RMSNorm(dim)
    self.head = nn.Linear(dim, vocab, bias=False)

  def forward(self, idx):
    x = self.embed(idx)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      for layer in self.layers:
        x = layer(x)
      x = self.norm(x)
    return self.head(x.float())


# ---------------------------------------------------------------- checks


def equivalence(dtype=torch.float32, batch=2, seqlen=1024, block_size=256):
  """Each variant must reproduce base outputs, states and gradients."""
  torch.manual_seed(0)
  mixer = m2.SegmentMamba2(d_model=768, d_state=64, d_conv=4, expand=2,
                           headdim=64, chunk_size=128).cuda().to(dtype)
  xBC = torch.randn(batch, seqlen, mixer.conv_dim, device="cuda", dtype=dtype)
  state = torch.randn(batch, mixer.conv_dim, mixer.d_conv, device="cuda",
                      dtype=dtype)
  u = torch.randn(batch, seqlen, 768, device="cuda", dtype=dtype)

  def conv_run(fn):
    x = xBC.detach().clone().requires_grad_(True)
    s = state.detach().clone().requires_grad_(True)
    mixer.zero_grad(set_to_none=True)
    out, final = fn(mixer, x, s)
    (out.square().mean() + final.square().mean()).backward()
    return (out.detach(), final.detach(), x.grad.clone(), s.grad.clone(),
            mixer.conv1d.weight.grad.clone(), mixer.conv1d.bias.grad.clone())

  def swbb_run(fn):
    v = u.detach().clone().requires_grad_(True)
    mixer.zero_grad(set_to_none=True)
    out, conv_states, ssm_states = fn(mixer, v, block_size)
    (out.square().mean() + conv_states.square().mean()
     + ssm_states.square().mean()).backward()
    return (out.detach(), conv_states.detach(), ssm_states.detach(),
            v.grad.clone())

  report = {}
  base = conv_run(m2.SegmentMamba2._causal_conv)
  names = ("out", "final_state", "grad_xBC", "grad_state", "grad_w", "grad_b")
  for name, fn in CONV_IMPLS.items():
    got = conv_run(fn)
    report[f"_causal_conv/{name}"] = {
      k: float((a - b).abs().max()) for k, a, b in zip(names, base, got)}
  def segment_run(fn):
    v = u.detach().clone().requires_grad_(True)
    s = m2.Mamba2State(
      state.detach().clone().requires_grad_(True),
      torch.zeros(batch, mixer.nheads, mixer.headdim, mixer.d_state,
                  device="cuda", dtype=dtype))
    mixer.zero_grad(set_to_none=True)
    out, final = fn(mixer, v, s)
    (out.square().mean() + final.conv.square().mean()
     + final.ssm.square().mean()).backward()
    return (out.detach(), final.conv.detach(), final.ssm.detach(),
            v.grad.clone(), mixer.in_proj.weight.grad.clone())

  base_seg = segment_run(m2.SegmentMamba2.scan_segment)
  got = segment_run(scan_segment_cmaj)
  report["scan_segment/cmaj"] = {
    k: float((a - b).abs().max()) for k, a, b in
    zip(("out", "final_conv", "final_ssm", "grad_u", "grad_in_proj"),
        base_seg, got)}

  base_swbb = swbb_run(m2.SegmentMamba2.scan_with_block_boundaries)
  for name, fn in (("pad", swbb_pad), ("cmaj", swbb_cmaj)):
    got = swbb_run(fn)
    report[f"scan_with_block_boundaries/{name}"] = {
      k: float((a - b).abs().max()) for k, a, b in
      zip(("out", "conv_states", "ssm_states", "grad_u"), base_swbb, got)}
  report["scale"] = {"out_absmax": float(base[0].abs().max()),
                     "swbb_out_absmax": float(base_swbb[0].abs().max())}
  del mixer
  torch.cuda.empty_cache()
  return report


def census(batch, length, variant):
  torch.manual_seed(0)
  module = m2.SegmentMamba2(d_model=768, d_state=64, d_conv=4, expand=2,
                            headdim=64, chunk_size=128).cuda()
  params = {p.untyped_storage().data_ptr() for p in module.parameters()}
  u = torch.randn(batch, length, 768, device="cuda", dtype=torch.bfloat16,
                  requires_grad=True)
  params.add(u.untyped_storage().data_ptr())
  seen, order = {}, []

  def pack(t):
    if t.is_cuda and t.numel() > 4096:
      key = t.untyped_storage().data_ptr()
      if key not in seen and key not in params:
        seen[key] = t.untyped_storage().nbytes()
        order.append((tuple(t.shape), str(t.dtype).replace("torch.", ""),
                      seen[key] / 2**20))
    return t

  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  base = torch.cuda.memory_allocated()
  with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      out, _ = module.scan_segment(u)
  torch.cuda.synchronize()
  result = {
    "unique_saved_MiB": round(sum(seen.values()) / 2**20, 1),
    "live_after_forward_MiB": round(
      (torch.cuda.memory_allocated() - base) / 2**20, 1),
    "forward_peak_MiB": round(
      (torch.cuda.max_memory_allocated() - base) / 2**20, 1),
    "tensors": [{"shape": list(s), "dtype": d, "MiB": round(m, 1)}
                for s, d, m in order],
  }
  del out, u, module
  torch.cuda.empty_cache()
  return result


def microbench(batch, length, iters=20):
  """Isolate the conv itself: contiguous input vs a strided view."""
  conv_dim, d_conv = 1664, 4
  weight = torch.randn(conv_dim, 1, d_conv, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
  bias = torch.randn(conv_dim, device="cuda", dtype=torch.bfloat16,
                     requires_grad=True)
  # `zxbcdt` as the model sees it; xBC is the middle slice.
  zxbcdt = torch.randn(batch, length, 3224, device="cuda",
                       dtype=torch.bfloat16, requires_grad=True)
  state = torch.zeros(batch, conv_dim, d_conv, device="cuda",
                      dtype=torch.bfloat16)
  pad = d_conv - 1

  def cat_conv():
    raw = zxbcdt.narrow(-1, 1536, conv_dim).transpose(1, 2)
    history = torch.cat((state, raw), dim=-1)
    out = F.conv1d(history, weight, bias, groups=conv_dim)
    return out.narrow(-1, 1, length).transpose(1, 2)

  def view_conv():
    raw = zxbcdt.narrow(-1, 1536, conv_dim).transpose(1, 2)
    return F.conv1d(raw, weight, bias, groups=conv_dim,
                    padding=pad).narrow(-1, 0, length).transpose(1, 2)

  def view_conv_head_cat():
    raw = zxbcdt.narrow(-1, 1536, conv_dim).transpose(1, 2)
    body = F.conv1d(raw, weight, bias, groups=conv_dim, padding=pad)
    head = F.conv1d(torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
                    weight, bias, groups=conv_dim).narrow(-1, 1, pad)
    return torch.cat(
      (head, body.narrow(-1, pad, length - pad)), dim=-1).transpose(1, 2)

  def view_conv_head_inplace():
    raw = zxbcdt.narrow(-1, 1536, conv_dim).transpose(1, 2)
    full = F.conv1d(raw, weight, bias, groups=conv_dim, padding=pad)
    head = F.conv1d(torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
                    weight, bias, groups=conv_dim).narrow(-1, 1, pad)
    full.narrow(-1, 0, pad).copy_(head)
    return full.narrow(-1, 0, length).transpose(1, 2)

  def contiguous_copy():
    return zxbcdt.narrow(-1, 1536, conv_dim).transpose(1, 2).contiguous()

  out = {}
  for name, fn in (("cat+conv", cat_conv), ("view_conv", view_conv),
                   ("view_conv+head+cat", view_conv_head_cat),
                   ("view_conv+head+inplace", view_conv_head_inplace),
                   ("transpose.contiguous", contiguous_copy)):
    for mode in ("fwd", "fwd+bwd"):
      for _ in range(3):
        y = fn()
        if mode != "fwd":
          y.square().mean().backward()
      torch.cuda.synchronize()
      t0 = time.time()
      for _ in range(iters):
        y = fn()
        if mode != "fwd":
          y.square().mean().backward()
      torch.cuda.synchronize()
      out[f"{name}/{mode}"] = round((time.time() - t0) / iters * 1e3, 4)
  del zxbcdt
  torch.cuda.empty_cache()
  return out


def run_stack(arm, batch, length, warmup, iters):
  torch.manual_seed(0)
  model = AttnStack().cuda() if arm == "attn" else SSMStack().cuda()
  opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
  idx = torch.randint(0, 13, (batch, length), device="cuda")
  target = torch.randint(0, 13, (batch, length), device="cuda")
  times = []
  torch.cuda.empty_cache()
  for i in range(warmup + iters):
    if i == warmup:
      torch.cuda.synchronize()
      torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, 13), target.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if i >= warmup:
      times.append(time.time() - t0)
  peak = torch.cuda.max_memory_allocated() / 2**30
  final_loss = float(loss.detach())
  del model, opt, logits, loss
  torch.cuda.empty_cache()
  times.sort()
  median = times[len(times) // 2]
  return {"arm": arm, "peak_GiB": round(peak, 3), "step_s": round(median, 4),
          "tok_per_s": round(batch * length / median), "loss": final_loss}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--batch", type=int, default=4)
  ap.add_argument("--length", type=int, default=8192)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--arms", default="base,ckpt,pad,pad+ckpt,shift,attn")
  ap.add_argument("--census", default="base,pad,shift,ckpt,pad+ckpt")
  ap.add_argument("--micro", action="store_true")
  ap.add_argument("--equivalence", default="1")
  ap.add_argument("--real-model", default="pad")
  ap.add_argument("--real-arms", default="ussm-ar,bissm")
  ap.add_argument("--output",
                  default=str(REPO / "results/sizing/conv_variants.json"))
  args = ap.parse_args()

  print(torch.cuda.get_device_name(0), flush=True)
  print("fused available:", m2.fused_mamba2_available(), flush=True)
  out = {"config": vars(args)}

  if args.micro:
    out["micro_ms"] = microbench(args.batch, args.length)
    print("\n=== micro (ms per call, batch/length as configured) ===")
    print(json.dumps(out["micro_ms"], indent=1), flush=True)

  if args.equivalence == "1":
    restore_class()
    patch_class("")  # install _causal_conv_base/_swbb_base for the checks
    out["equivalence"] = equivalence()
    restore_class()
    print("\n=== equivalence (fp32, max abs diff vs base) ===")
    print(json.dumps(out["equivalence"], indent=1), flush=True)

  out["census"] = {}
  for variant in [v for v in args.census.split(",") if v]:
    restore_class()
    patch_class(variant)
    out["census"][variant] = census(args.batch, args.length, variant)
    restore_class()
    print(f"\n=== census {variant} ===")
    print(json.dumps(out["census"][variant], indent=1), flush=True)

  out["stack"] = []
  for arm in [a for a in args.arms.split(",") if a]:
    restore_class()
    if arm != "attn":
      patch_class(arm)
    try:
      row = run_stack(arm, args.batch, args.length, args.warmup, args.iters)
    except torch.cuda.OutOfMemoryError:
      row = {"arm": arm, "oom": True}
    restore_class()
    torch.cuda.empty_cache()
    out["stack"].append(row)
    print(json.dumps(row), flush=True)

  if args.real_model:
    import sizing_sweep as ss
    out["real_model"] = []
    device = torch.device("cuda")
    # An absolute peak is only comparable to another process's if this one
    # starts empty: max_memory_allocated is absolute, so anything still live
    # from the census or stack arms shifts every later row by that amount.
    out["real_model_baseline_MiB"] = round(
      torch.cuda.memory_allocated(device) / 2**20, 1)
    print("live before real-model rows (MiB):",
          out["real_model_baseline_MiB"], flush=True)
    for variant in [v for v in args.real_model.split(",") if v]:
      restore_class()
      if variant != "base":
        patch_class(variant)
      for arm in [a for a in args.real_arms.split(",") if a]:
        row = ss.run_case(arm, args.length, 256, args.batch, False,
                          args.warmup, args.iters, device)
        row["variant"] = variant
        out["real_model"].append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
      restore_class()

  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  Path(args.output).write_text(json.dumps(out, indent=2))
  print("wrote", args.output)


if __name__ == "__main__":
  main()
