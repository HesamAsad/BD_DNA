#!/usr/bin/env python
"""Lens E: saved-tensor census + candidate fixes for the Transformer (DiT) arm.

Reuses the machinery in ``saved_tensor_audit.py`` (identity saved-tensor hooks,
storage dedup, call-site attribution, real fwd+bwd+AdamW peak) and registers
DiT-specific patches on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import saved_tensor_audit as sta  # noqa: E402


# ---------------------------------------------------------------------------
# Candidate fixes for models/dit.py
# ---------------------------------------------------------------------------

def patch_ln_fused():
  """Fold the affine weight into F.layer_norm.

  dit.py:234-237 computes ``F.layer_norm(x.float(), [dim])`` and then a
  separate ``x * self.weight``.  The separate multiply retains the fp32
  layer-norm output ([B, L, D] fp32) as MulBackward0._saved_self.  Passing the
  weight to layer_norm keeps the arithmetic in fp32 and identical, but
  native_layer_norm_backward needs only (input, mean, rstd, weight), so the
  normalized output is never retained.
  """
  from models import dit

  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      return F.layer_norm(x.float(), [self.dim], weight=self.weight)

  dit.LayerNorm.forward = forward


def patch_ln_native():
  """Let layer_norm consume the activation dtype directly.

  Drops the explicit ``.float()``.  ATen's layer_norm already accumulates in
  fp32 for bf16 inputs, and the fp32 result is immediately downcast by the
  autocast Linear that consumes it, so the only real numeric change is that the
  affine weight is rounded to bf16 before the multiply.  Combined with a bf16
  residual stream this makes the layer-norm input alias the residual (free).
  """
  from models import dit

  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      return F.layer_norm(x, [self.dim], weight=self.weight.to(x.dtype))

  dit.LayerNorm.forward = forward


def patch_bf16_residual():
  """Keep the DiT residual stream in bf16.

  ``vocab_embed`` runs outside dit.py's inner bf16 autocast (dit.py:772,816) so
  it emits fp32, and every residual add (``residual + out`` in
  bias_dropout_add_scale) promotes bf16 back to fp32.  The residual stream is
  therefore fp32 [B, L, D] for all 12 blocks.
  """
  from models import dit

  original = dit.EmbeddingLayer.forward

  def forward(self, x):
    return original(self, x).to(torch.bfloat16)

  dit.EmbeddingLayer.forward = forward


def patch_no_scale_mul():
  """Skip the identity ``scale * x`` when there is no adaLN gate.

  dit.py:443-447 / 537-541 build ``scale = torch.ones(1)`` and route through
  bias_dropout_add_scale, i.e. two extra full-tensor elementwise kernels per
  block on the non-adaLN (AR) path.
  """
  from models import dit

  def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(dit.modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      x = x_skip + F.dropout(self.attn_out(x), self.dropout, self.training)
      x = x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)
    return x

  def forward_causal(self, x, rotary_cos_sin, c=None, causal=True, mask=None,
                     store_kv=False, **kwargs):
    del kwargs
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa = scale_msa = gate_msa = None
    shift_mlp = scale_mlp = gate_mlp = None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = dit.rearrange(
         self.adaLN_modulation(c), '(b h) d -> b h d',
         b=batch_size).chunk(6, dim=-1)
    x_skip = x
    if c is not None:
      x = dit.modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flash_attn':
      qkv = dit.einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
      x = dit.flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0.0, causal=True)
      x = dit.einops.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      x = self.cross_attn(qkv, c)
    return attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp,
                    x_skip)

  dit.DDiTBlock.attn_mlp = attn_mlp
  dit.DDiTBlockCausal.forward = forward_causal


def patch_nojit():
  """Drop the torch.jit.script wrappers on bias_dropout_add_scale/modulate.

  dit.py:26-29 disables the profiling executor, so TorchScript never forms a
  fusion group (verified: the optimized graph contains no FusionGroup /
  TensorExprGroup).  What it does do is route ``aten::dropout`` through
  symbolic_script's gradient formula, which calls ``aten::native_dropout``
  unconditionally -- bypassing ATen's ``p == 0`` short circuit and retaining a
  full-shape bool mask at every call site even when model.dropout == 0.0.
  """
  from models import dit

  def train_fn(x, bias, scale, residual, prob):
    return dit.bias_dropout_add_scale(x, bias, scale, residual, prob, True)

  def eval_fn(x, bias, scale, residual, prob):
    return dit.bias_dropout_add_scale(x, bias, scale, residual, prob, False)

  dit.bias_dropout_add_scale_fused_train = train_fn
  dit.bias_dropout_add_scale_fused_inference = eval_fn
  dit.modulate_fused = dit.modulate


def patch_qkv_nocat():
  """Avoid the extra full-width qkv copy on the block-diffusion path.

  dit.py:594-596 projects the xt and x0 halves separately and then
  ``torch.cat``s them, allocating a second [B, 2L, 3D] tensor.  Writing the two
  projections into one preallocated buffer removes that copy.
  """
  from models import dit

  def forward(self, x, rotary_cos_sin, c, causal=False, mask=None,
              sample_mode=False, store_kv=False):
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa = scale_msa = gate_msa = None
    shift_mlp = scale_mlp = gate_mlp = None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = dit.rearrange(
         self.adaLN_modulation(c), '(b h) d -> b h d',
         b=batch_size).chunk(6, dim=-1)
    x_skip = x
    if c is not None:
      x = dit.modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    if mask is not None and not sample_mode:
      # one projection over the whole 2L input; rotary is then applied to each
      # half in place, which is exactly what the two-call version did.
      qkv = self.attn_qkv(x)
      qkv = dit.einops.rearrange(
        qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
      with torch.amp.autocast('cuda', enabled=False):
        cos, sin = rotary_cos_sin
        cos, sin = cos.to(qkv.dtype), sin.to(qkv.dtype)
        dit.apply_rotary_pos_emb(qkv[:, :self.n], cos, sin)
        dit.apply_rotary_pos_emb(qkv[:, self.n:], cos, sin)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flash_attn' and mask is None:
      qkv = dit.einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
      x = dit.flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=causal)
      x = dit.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    elif self.attn_backend == 'flex' and dit.FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend == 'sdpa':
      x = self.cross_attn(qkv, mask=mask)
    else:
      raise ValueError('Unknown attention backend')
    if self.kv_cache is not None:
      x = x[:, -self.block_size:]
    return self.attn_mlp(
      x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)

  dit.DDiTBlock.forward = forward


def patch_rotary_flash():
  """Use flash_attn's in-place rotary on the non-flash_attn backends too.

  get_qkv (dit.py:513-520) only reaches apply_rotary_pos_emb (the fused,
  in-place flash kernel) when attn_backend == 'flash_attn'.  The flex and sdpa
  backends fall back to apply_rotary_pos_emb_torchscript (dit.py:202-203),
  which is eager ``qkv*cos + rotate_half(qkv)*sin``: it allocates a second
  full-width qkv, and retains the bf16 cos/sin casts.  Isolated from `nocat`.
  """
  from models import dit

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    qkv = self.attn_qkv(x)
    qkv = dit.einops.rearrange(
      qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = dit.apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv

  dit.DDiTBlock.get_qkv = get_qkv


def patch_qkv_nocat_ts():
  """`nocat` with the ORIGINAL torchscript rotary, to separate the two effects."""
  from models import dit

  def forward(self, x, rotary_cos_sin, c, causal=False, mask=None,
              sample_mode=False, store_kv=False):
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa = scale_msa = gate_msa = None
    shift_mlp = scale_mlp = gate_mlp = None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = dit.rearrange(
         self.adaLN_modulation(c), '(b h) d -> b h d',
         b=batch_size).chunk(6, dim=-1)
    x_skip = x
    if c is not None:
      x = dit.modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    if mask is not None and not sample_mode:
      qkv = self.attn_qkv(x)
      qkv = dit.einops.rearrange(
        qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
      with torch.amp.autocast('cuda', enabled=False):
        cos, sin = rotary_cos_sin
        cos, sin = cos.to(qkv.dtype), sin.to(qkv.dtype)
        qkv = torch.cat(
          (dit.apply_rotary_pos_emb_torchscript(qkv[:, :self.n], cos, sin),
           dit.apply_rotary_pos_emb_torchscript(qkv[:, self.n:], cos, sin)),
          dim=1)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flex' and dit.FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    else:
      x = self.cross_attn(qkv, mask=mask)
    return self.attn_mlp(
      x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)

  dit.DDiTBlock.forward = forward


def patch_rotary_slim():
  """Cache only the half-width rotary tables actually consumed.

  Rotary.forward (dit.py:173-177) builds [1, L, 3, 1, D] fp32 cos/sin via
  ``.repeat(1, 1, 3, 1, 1)`` and every block then slices ``cos[0,:,0,0,:D//2]``
  and casts it to bf16 -- 12 redundant casts of a 3x-oversized table per step.
  """
  from models import dit

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum('i,j->ij', t, self.inv_freq.clone())
      self._cos_half = freqs.cos()
      self._sin_half = freqs.sin()
      self.cos_cached = self._cos_half
      self.sin_cached = self._sin_half
    return self._cos_half, self._sin_half

  def apply_rotary_pos_emb(qkv, cos, sin):
    return dit.flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    if self.kv_cache is not None:  # sampling path unchanged
      raise RuntimeError('rotary_slim patch is training-only')
    qkv = self.attn_qkv(x)
    qkv = dit.einops.rearrange(
      qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv

  dit.Rotary.forward = forward
  dit.DDiTBlockCausal.get_qkv = get_qkv
  dit.DDiTBlock.get_qkv = get_qkv


sta.PATCHES.update({
  'lnfused': patch_ln_fused,
  'lnnative': patch_ln_native,
  'bf16resid': patch_bf16_residual,
  'nojit': patch_nojit,
  'noscale': patch_no_scale_mul,
  'nocat': patch_qkv_nocat,
  'nocatts': patch_qkv_nocat_ts,
  'rotflash': patch_rotary_flash,
  'rotslim': patch_rotary_slim,
})
sta.ORDER = ['rsqrt', 'rmsnorm', 'gated', 'conv', 'xbc', 'splitproj',
             'lnfused', 'lnnative', 'bf16resid', 'nojit', 'noscale', 'rotflash', 'nocat',
             'nocatts', 'rotslim']

if __name__ == '__main__':
  sta.main_cli()
