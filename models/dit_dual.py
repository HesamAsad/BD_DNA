"""Dual-stream BD3-LM backbone (Variant 2): local fine + global coarse.

The fine stream runs BD3-LM block-diffusion at *single-nucleotide* resolution
but with attention RESTRICTED to a local window (cheap), and adds cross-
attention to a CHEAP coarse stream that encodes the clean context x_0 at length
L/k via non-overlapping k-mer tokens. Diffusion target stays fine.

Streams
-------
  Coarse: input = x_0 (length L) -> k-mer ids (length L/k, vocab = 4^k + 1) ->
          n_c-layer transformer with CAUSAL attention at coarse-token grain
          (c_j sees c_0..c_j). Causality is what makes the stream leak-free: a
          coarse token read by fine block i depends only on x_0 of strictly-past
          blocks, never the clean current/future target. Cheap because L/k is
          short. No timestep conditioning (coarse encodes the clean past signal).
  Fine:   input = [x_t; x_0] (length 2L). Each layer:
          (1) local self-attention with the BD3-LM block-diffusion mask
              restricted to +/- window_blocks blocks.
          (2) cross-attention to the coarse memory (block-causal alignment).
          (3) MLP. All adaLN-modulated by the diffusion timestep.

Masks
-----
  fine_local_block_mask : the standard `block_diff_mask` from `dit.py` ANDed
                          with a locality predicate `(block_q - block_kv) <=
                          window_blocks` (all BD3-LM rules have block_q >=
                          block_kv, so this restricts band width on the lower-
                          triangular side).
  fine_to_coarse_mask  : strict block-causal cross-attention. Fine block i sees
                          coarse tokens whose entire k-mer window lies inside
                          the "completed" x_0 portion (M_OBC for xt queries,
                          M_BC for x_0 queries). NOTE: this position-level mask
                          is leak-free ONLY because the coarse self-attention is
                          causal (see CoarseBlock). With a bidirectional coarse
                          stream the visible past coarse tokens would already
                          have absorbed the clean current/future target -> leak.

For best alignment use `block_size` divisible by `k_coarse` (k=6 with
block_size 12/18/24 is exact); otherwise the coarse memory is slightly
under-covered at block boundaries (no leak, just less context — the fine
stream still has direct access to x_0 via its self-attention).

Diffusion-side integration
--------------------------
DualStreamDIT.forward(indices, sigma, sample_mode=False) returns logits of
shape (B, L, vocab) — same contract as DIT. Set `algo.backbone=dit_dual` in
hydra config to use it; no other Diffusion changes needed. Sampling
(sample_mode=True) re-encodes the coarse stream from the partial generation
each stride; because the coarse stream is causal, the past coarse tokens the
fine blocks read are identical to training. CAVEAT: for sliding-window
generation (generated length > sampling context_size) the coarse stream is
currently re-encoded over only the in-window prefix, so long-range coarse
context beyond the window is truncated (see _gen_sampling_masks /
_semi_ar_sampler). Encode the coarse stream over the full prefix to make
>context_size generation exact.
"""

import math
import os
import typing
from functools import partial

import einops
from einops import rearrange
import flash_attn          # for apply_rotary_emb_torch (aligned cross-attention)
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except ImportError:
  FLEX_ATTN_AVAILABLE = False

# Reuse all primitives from the original DiT.
from .dit import (
  LayerNorm,
  EmbeddingLayer,
  Rotary,
  TimestepEmbedder,
  DDiTFinalLayer,
  split_and_apply_rotary_pos_emb,
  # The @torch.compile-wrapped flex helper. CRITICAL: calling
  # `flex_attention` eagerly (outside torch.compile) falls back to the dense
  # math reference path that materialises the full (B, H, Q, K) score grid and
  # ignores BlockMask sparsity -> O((2L)^2) memory. Routing every flex call
  # through this wrapper lets inductor lower it to the fused block-sparse Triton
  # kernel (the single-stream DIT does the same). Honors BD3LM_FLEX_COMPILE_MODE.
  fused_flex_attention,
)


# ----------------------------------------------------------------------------
# Non-overlapping k-mer encoding (used by the coarse stream)
# ----------------------------------------------------------------------------

# DNATokenizer ids: [CLS]=0 [SEP]=1 [BOS]=2 [EOS]=3 [MASK]=4 [PAD]=5
# [RESERVED]=6 [UNK]=7 A=8 C=9 G=10 T=11 N=12
_NUC_OFFSET = 8


def coarse_vocab_size(k: int) -> int:
  """4^k clean k-mers + 1 UNK = 4^k + 1.
  k=6 -> 4097, k=4 -> 257, k=8 -> 65,537. k=16 = 4^16 = ~4.3B, infeasible."""
  return 4 ** k + 1


def encode_kmer(nucleotide_ids: torch.Tensor, k: int) -> torch.Tensor:
  """DNATokenizer ids -> non-overlapping k-mer ids in [0, 4^k].

  Vocab: ids 0..4^k - 1 are clean ACGT k-mers (base-4 encoding of A=0/C=1/
  G=2/T=3); id 4^k is UNK (used when any of the k positions is N or a special
  token). Last axis must be divisible by k.
  """
  L = nucleotide_ids.shape[-1]
  assert L % k == 0, f"sequence length {L} must be divisible by k={k}"
  Lc = L // k
  leading = nucleotide_ids.shape[:-1]
  flat = nucleotide_ids.reshape(-1, Lc, k)

  is_acgt = (flat >= _NUC_OFFSET) & (flat < _NUC_OFFSET + 4)
  # mod 4 keeps A=0..T=3 for ACGT, junk elsewhere (gets replaced by UNK).
  nuc_vals = (flat - _NUC_OFFSET).clamp(min=0, max=3)
  powers = (4 ** torch.arange(k - 1, -1, -1,
                              device=flat.device, dtype=flat.dtype))
  kmer_ids = (nuc_vals * powers).sum(dim=-1)
  all_acgt = is_acgt.all(dim=-1)
  unk_id = 4 ** k
  kmer_ids = torch.where(all_acgt, kmer_ids,
                         torch.full_like(kmer_ids, unk_id))
  return kmer_ids.reshape(*leading, Lc)


# ----------------------------------------------------------------------------
# Masks (kept inductor-friendly: only bool ops, integer compares; no Eq(bool,int))
# ----------------------------------------------------------------------------

def fine_local_block_mask(b, h, q_idx, k_idx, block_size, n, window_blocks):
  """BD3-LM block-diffusion mask restricted to a local band of +/-
  `window_blocks` blocks (within each half of the [xt; x0] concat)."""
  x0_flag_q = q_idx >= n
  x0_flag_kv = k_idx >= n
  p_q = torch.where(x0_flag_q, q_idx - n, q_idx)
  p_kv = torch.where(x0_flag_kv, k_idx - n, k_idx)
  block_q = p_q // block_size
  block_kv = p_kv // block_size
  same_flag = ~(x0_flag_q ^ x0_flag_kv)
  block_diagonal = (block_q == block_kv) & same_flag
  offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
  block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
  # All three rules satisfy block_q >= block_kv, so the band is single-sided.
  near = (block_q - block_kv) <= window_blocks
  return (block_diagonal | offset_block_causal | block_causal) & near


def fine_to_coarse_mask(b, h, q_idx, k_idx, block_size, n_fine, k_coarse):
  """Strict block-causal cross-attention from fine queries (length 2L) to
  coarse keys (length L/k). Fine block i in xt sees coarse tokens whose full
  k-mer window lies in x_0 blocks 0..i-1; in x_0, blocks 0..i."""
  x0_flag = q_idx >= n_fine
  p = torch.where(x0_flag, q_idx - n_fine, q_idx)
  block_q = p // block_size
  n_blocks_visible = torch.where(x0_flag, block_q + 1, block_q)
  visible_nucs = n_blocks_visible * block_size  # = upper bound (exclusive)
  # coarse token j visible iff (j+1)*k_coarse <= visible_nucs
  #  -> j <= visible_nucs // k_coarse - 1 (handles xt block 0: visible=0 -> -1).
  max_coarse = visible_nucs // k_coarse - 1
  return k_idx <= max_coarse


# ----------------------------------------------------------------------------
# Cross-attention (fine queries, coarse keys/values) with a flex BlockMask
# ----------------------------------------------------------------------------

class CrossAttention(nn.Module):
  """Multi-head cross-attention with a flex block-mask. Q from fine stream,
  K/V from coarse memory."""

  def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float = 0.0):
    super().__init__()
    assert d_q % n_heads == 0, f"d_q={d_q} not divisible by n_heads={n_heads}"
    self.n_heads = n_heads
    self.head_dim = d_q // n_heads
    self.norm_q = LayerNorm(d_q)
    self.norm_kv = LayerNorm(d_kv)
    self.q_proj = nn.Linear(d_q, d_q, bias=False)
    self.kv_proj = nn.Linear(d_kv, 2 * d_q, bias=False)
    self.out_proj = nn.Linear(d_q, d_q)
    # IMPORTANT: do NOT zero-init out_proj. The cross-attn residual in
    # FineDualBlock is `gate_cross * out_proj(...)`, where `gate_cross` comes
    # from a zero-init adaLN modulation. Zero-init'ing the projection TOO makes
    # the product a dead zero: grad w.r.t. the gate is proportional to the
    # projection output (=0) and grad w.r.t. the projection is proportional to
    # the gate (=0), so neither ever leaves zero and cross-attention is
    # permanently disabled. Standard adaLN-zero zeroes ONLY the gate; the output
    # projection uses normal init (as the MLP here does), which lets the gate
    # receive gradient and open. (See longctx marginal-collapse diagnosis.)

  def forward(self, q, kv, mask, use_sdpa: bool = False, rotary_qk=None,
              add_mask=None):
    """`mask` is a flex BlockMask when use_sdpa=False, else a dense bool tensor
    of shape (Lq, Lk) (broadcast over batch & heads).

    `rotary_qk` (B3 "aligned pointer" ablation, model.cross_align=true): a tuple
    (cos_q, sin_q, cos_k, sin_k) placing fine queries and coarse keys on ONE
    SHARED nucleotide coordinate system, so the relative offset (p - k*j) between
    a fine position p and the coarse token j covering nucleotides [k*j, k*j+k) is
    directly computable by the attention. WITHOUT this the module has NO
    positional signal at all -- only the block-causal mask, which says "you may
    look at blocks < i" but nothing about WHERE within them -- so a fixed-offset
    copy has to be implemented as a content-reconstructed pointer. That is the
    prime suspect for the rung-B failure (val/nll stuck at ln4 for 6000 steps)."""
    B, Lq, _ = q.shape
    Lk = kv.shape[1]
    Q = self.q_proj(self.norm_q(q)).view(B, Lq, self.n_heads, self.head_dim)
    KV = self.kv_proj(self.norm_kv(kv)).view(
      B, Lk, 2, self.n_heads, self.head_dim)
    K = KV[:, :, 0]
    V = KV[:, :, 1]
    if rotary_qk is not None:
      cos_q, sin_q, cos_k, sin_k = rotary_qk
      with torch.amp.autocast('cuda', enabled=False):
        Q = flash_attn.layers.rotary.apply_rotary_emb_torch(
          Q.float(), cos_q.to(Q.dtype).float(), sin_q.to(Q.dtype).float()).to(V.dtype)
        K = flash_attn.layers.rotary.apply_rotary_emb_torch(
          K.float(), cos_k.to(K.dtype).float(), sin_k.to(K.dtype).float()).to(V.dtype)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    if add_mask is not None:
      # A2 relative-position bias path: `add_mask` (1,H,Lq,Lk) is a FLOAT additive
      # bias that already folds in the block-causal mask (disallowed keys = -inf)
      # plus the learned bucketed fine->coarse offset bias. Dense sdpa so the bias
      # applies per (query, key) pair.
      out = F.scaled_dot_product_attention(Q, K, V, attn_mask=add_mask)
      finite = torch.isfinite(add_mask).any(dim=-1, keepdim=True)   # zero dead rows
      out = torch.where(finite, out, out.new_zeros(()))
    elif use_sdpa:
      out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
      # A fully-masked query row (e.g. the first block, which has no past coarse
      # tokens under the M_OBC alignment) makes sdpa's softmax return NaN, while
      # the flex training path returns 0 for such rows. Zero them so the sampling
      # cross-attention matches training (and never emits NaNs).
      if mask is not None and mask.dtype == torch.bool:
        row_valid = mask.any(dim=-1).view(1, 1, -1, 1)
        out = torch.where(row_valid, out, out.new_zeros(()))
    else:
      # Compiled flex (see fused_flex_attention import note) — never the eager
      # dense-score path.
      out = fused_flex_attention(Q, K, V, mask=mask)
    out = out.transpose(1, 2).reshape(B, Lq, -1)
    return self.out_proj(out)


# ----------------------------------------------------------------------------
# Coarse encoder block (CAUSAL attention, no timestep conditioning)
# ----------------------------------------------------------------------------

class CoarseBlock(nn.Module):
  """Transformer encoder block with CAUSAL self-attention (c_j sees c_0..c_j).
  Causality keeps the stream leak-free (a past coarse token never absorbs the
  clean current/future target) and suffix-invariant (train == sample). The
  coarse stream encodes the clean past signal x_0, so no timestep conditioning
  is needed here."""

  def __init__(self, dim: int, n_heads: int,
               mlp_ratio: int = 4, dropout: float = 0.0):
    super().__init__()
    assert dim % n_heads == 0
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.norm1 = LayerNorm(dim)
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_proj = nn.Linear(dim, dim)
    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim),
      nn.GELU(),
      nn.Linear(mlp_ratio * dim, dim),
    )

  def forward(self, x, rotary_cos_sin):
    B, L, _ = x.shape
    qkv = self.qkv(self.norm1(x)).view(
      B, L, 3, self.n_heads, self.head_dim)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
    Q, K, V = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    # CAUSAL at coarse-token granularity: c_j attends to c_0..c_j only. With
    # block_size % k_coarse == 0 this aligns to block boundaries, so a coarse
    # token read by fine block i (the cross-attn is strict block-causal) depends
    # only on x_0 of blocks < i -> no clean current/future target leaks into the
    # fine prediction. Causality also makes c_j suffix-invariant, so the coarse
    # tokens match between training (full L/k seq) and sampling (truncated prefix).
    attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    x = x + self.attn_proj(attn_out.transpose(1, 2).reshape(B, L, -1))
    x = x + self.mlp(self.norm2(x))
    return x


# ----------------------------------------------------------------------------
# Fine block: local self-attn + cross-attn to coarse + MLP (adaLN-modulated)
# ----------------------------------------------------------------------------

class FineDualBlock(nn.Module):
  """Fine-stream block at the [x_t; x_0] level (length 2L). Three sub-layers:
     1) local self-attention with the windowed BD3-LM block-diffusion mask,
     2) cross-attention to the coarse memory,
     3) MLP. All adaLN-modulated by the diffusion timestep."""

  def __init__(self, n: int, dim: int, n_heads: int,
               d_coarse: int, cond_dim: int, gather_dim: int = 0,
               mlp_ratio: int = 4, dropout: float = 0.0):
    super().__init__()
    assert dim % n_heads == 0
    self.n = n  # length of each half (xt or x0)
    self.n_heads = n_heads
    self.head_dim = dim // n_heads

    self.norm1 = LayerNorm(dim)
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.self_proj = nn.Linear(dim, dim)
    # IMPORTANT: do NOT zero-init self_proj (same deadlock as
    # CrossAttention.out_proj). The self-attn residual is `gate1 * self_proj(...)`
    # with `gate1` from a zero-init adaLN modulation; zeroing both factors pins
    # self-attention at zero for the whole run, degenerating the fine stream into
    # position-wise MLPs that collapse to the unigram marginal. Keep normal init;
    # the zero-init gate opens via gradient (exactly as gate2/MLP already do).

    self.cross_attn = CrossAttention(dim, d_coarse, n_heads, dropout)
    # B0/B1 diagnostic probes: replace cross-attention with a HARD-WIRED gather of
    # the known-correct source, removing the selection/search problem entirely so
    # that representation+decoding can be tested on its own.
    self.gather_proj = nn.Linear(gather_dim, dim) if gather_dim else None

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim),
      nn.GELU(),
      nn.Linear(mlp_ratio * dim, dim),
    )

    # 7 adaLN modulators: self {shift, scale, gate}, cross {gate}, mlp {shift, scale, gate}
    self.adaLN_modulation = nn.Linear(cond_dim, 7 * dim)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def forward(self, x, c_mem, rotary_cos_sin, t_cond, self_mask, cross_mask,
              sample_mode: bool = False, cross_rotary=None,
              gathered=None, force_gate_cross=None, cross_add_mask=None):
    """If sample_mode=True, `x` has shape (B, L_curr, d) (no xt/x0 split),
    masks are dense bool tensors, and attention uses sdpa. Otherwise (training),
    `x` is (B, 2L, d), masks are flex BlockMasks, and attention uses flex."""
    B, L_x, _ = x.shape
    if t_cond is None:
      shift1 = scale1 = shift2 = scale2 = 0.0
      gate1 = gate_cross = gate2 = 1.0
    else:
      mod = self.adaLN_modulation(t_cond).unsqueeze(1).chunk(7, dim=-1)
      shift1, scale1, gate1, gate_cross, shift2, scale2, gate2 = mod

    # 1) Self-attention
    x_norm = self.norm1(x) * (1 + scale1) + shift1
    qkv = self.qkv(x_norm).view(B, L_x, 3, self.n_heads, self.head_dim)

    if sample_mode:
      # Single-stream sequence (no xt/x0 split). Apply rotary once on the
      # whole length, attend via sdpa with the dense block-causal local mask.
      q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
      Q, K, V = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
      attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=self_mask)
    else:
      # Training: rotary on the xt and x0 halves separately (positions 0..n-1
      # each), concatenate, then flex attention with the BlockMask.
      qkv_xt = qkv[:, :self.n]
      qkv_x0 = qkv[:, self.n:]
      q_xt, k_xt, v_xt = split_and_apply_rotary_pos_emb(qkv_xt, rotary_cos_sin)
      q_x0, k_x0, v_x0 = split_and_apply_rotary_pos_emb(qkv_x0, rotary_cos_sin)
      q = torch.cat([q_xt, q_x0], dim=1)
      k = torch.cat([k_xt, k_x0], dim=1)
      v = torch.cat([v_xt, v_x0], dim=1)
      Q, K, V = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
      # Compiled flex (see fused_flex_attention import note) — the eager call
      # here was the O((2L)^2) score-grid materialisation that caused the OOMs.
      attn_out = fused_flex_attention(Q, K, V, mask=self_mask)

    attn_out = attn_out.transpose(1, 2).reshape(B, L_x, -1)
    x = x + gate1 * self.self_proj(attn_out)

    # 2) Cross-attention to coarse memory
    # Long-range route. `force_gate_cross` bypasses the zero-init adaLN gate: without
    # it, ANY probe here is uninterpretable when it nulls, because the gate can hold
    # the route off (measured: gate_cross stayed ~0.044 in the B3 aligned-pointer run,
    # so improved addressing had no path to express itself in the loss).
    if force_gate_cross is not None:
      gate_cross = force_gate_cross
    if gathered is not None:
      x = x + gate_cross * self.gather_proj(gathered)
    else:
      x = x + gate_cross * self.cross_attn(
        x, c_mem, mask=cross_mask, use_sdpa=sample_mode, rotary_qk=cross_rotary,
        add_mask=cross_add_mask)

    # 3) MLP
    x_norm = self.norm2(x) * (1 + scale2) + shift2
    x = x + gate2 * self.mlp(x_norm)
    return x


# ----------------------------------------------------------------------------
# DualStreamDIT — the model
# ----------------------------------------------------------------------------

class DualStreamDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  """BD3-LM with a dual-stream backbone (fine local + coarse global).

  Reads from config.model (in addition to the standard small/medium fields):
    - k_coarse:        k-mer size for the coarse stream (default 6 -> vocab 4097)
    - window_blocks:   fine local-attention window in BLOCKS (default 8)
    - d_coarse:        coarse hidden dim (default hidden_size // 2)
    - n_coarse_layers: coarse encoder depth (default max(2, n_blocks // 3))
    - n_heads_coarse:  coarse heads (default max(4, n_heads // 2))

  Forward signature matches `models.dit.DIT` so it slots into
  `diffusion.Diffusion.forward` without changing the call site.
  """

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config

    self.n = config.model.length
    self.block_size = config.block_size
    self.k_coarse = getattr(config.model, 'k_coarse', 6)
    # B3 "aligned pointer" ablation. When true, fine queries and coarse keys are
    # placed on a SHARED nucleotide coordinate system in the cross-attention (see
    # CrossAttention.forward). Default false = the original position-blind
    # cross-attention, so this is a controlled A/B, not a silent change.
    self.cross_align = bool(getattr(config.model, 'cross_align', False))
    # A2 addressing repair: learned ADDITIVE relative-position bias on the
    # fine->coarse cross-attention logits, over signed log-bucketed nucleotide
    # offset delta = p_fine - k*j_coarse. Directly targets the confirmed SELECTION
    # failure (the route can store+decode but cannot LOCATE the right coarse token).
    # Default off = original position-blind cross-attention (controlled A/B).
    self.cross_rel_bias = bool(getattr(config.model, 'cross_rel_bias', False))
    self.cross_rel_buckets = int(getattr(config.model, 'cross_rel_buckets', 64))
    if self.cross_rel_bias:
      self.cross_bias_table = nn.Parameter(
        torch.zeros(self.cross_rel_buckets, int(config.model.n_heads)))
    # --- B0/B1 diagnostic probes (decompose encode -> select -> decode) ---
    #  'attn'          : normal learned cross-attention (default, ships).
    #  'gather_fine'   : B0 ORACLE. Hand each position the CLEAN fine embedding at
    #                    i-gather_offset. For the duplication task that IS the answer,
    #                    so it must solve instantly; if not, the benchmark / masking /
    #                    output head is broken.
    #  'gather_coarse' : B1. Hand each position ONLY the correct coarse token
    #                    j*=(i-gather_offset)//k. Selection is removed, so this asks
    #                    exactly: do coarse k-mer states preserve the base, and can the
    #                    decoder extract it? Isolates representation+decoding.
    # `force_gate_cross` overrides the zero-init adaLN gate so a null is interpretable.
    self.cross_mode = str(getattr(config.model, 'cross_mode', 'attn'))
    self.gather_offset = int(getattr(config.model, 'gather_offset', 0))
    _fg = getattr(config.model, 'force_gate_cross', None)
    self.force_gate_cross = None if _fg in (None, '', 'null', 'None') else float(_fg)
    self.window_blocks = getattr(config.model, 'window_blocks', 8)

    d_fine = config.model.hidden_size
    cond_dim = config.model.cond_dim
    n_heads_fine = config.model.n_heads
    n_fine_layers = config.model.n_blocks

    d_coarse = getattr(config.model, 'd_coarse', max(64, d_fine // 2))
    n_coarse_layers = getattr(config.model, 'n_coarse_layers',
                              max(2, n_fine_layers // 3))
    n_heads_coarse = getattr(config.model, 'n_heads_coarse',
                              max(4, n_heads_fine // 2))

    assert self.n % self.k_coarse == 0, \
      f"model.length ({self.n}) must be divisible by k_coarse ({self.k_coarse})"
    assert self.n % self.block_size == 0, \
      f"model.length ({self.n}) must be divisible by block_size ({self.block_size})"
    self.n_coarse_tokens = self.n // self.k_coarse

    self.vocab_size = vocab_size
    self.vocab_size_coarse = coarse_vocab_size(self.k_coarse)

    # Embeddings
    self.vocab_embed = EmbeddingLayer(d_fine, vocab_size)
    self.coarse_embed = EmbeddingLayer(d_coarse, self.vocab_size_coarse)

    # Timestep conditioning (used by fine stream's adaLN; coarse ignores it)
    self.sigma_map = TimestepEmbedder(cond_dim)

    # Rotary (fine rotary uses fine head_dim; coarse uses coarse head_dim)
    self.rotary_emb = Rotary(d_fine // n_heads_fine)
    self.rotary_emb_coarse = Rotary(d_coarse // n_heads_coarse)

    # Streams
    self.coarse_blocks = nn.ModuleList([
      CoarseBlock(d_coarse, n_heads_coarse)
      for _ in range(n_coarse_layers)
    ])
    # gather width depends on which source the B0/B1 probe hands the block
    # (defined here, where d_fine / d_coarse are in scope)
    self.gather_dim = {'gather_fine': d_fine,
                       'gather_coarse': d_coarse}.get(self.cross_mode, 0)
    self.fine_blocks = nn.ModuleList([
      FineDualBlock(n=self.n, dim=d_fine, n_heads=n_heads_fine,
                    d_coarse=d_coarse, cond_dim=cond_dim,
                    gather_dim=self.gather_dim,
                    dropout=getattr(config.model, 'dropout', 0.0))
      for _ in range(n_fine_layers)
    ])

    # Output (operates on fine stream; we slice the xt half before/after)
    self.output_layer = DDiTFinalLayer(
      hidden_size=d_fine,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=True,
      tie_word_embeddings=config.model.tie_word_embeddings,
    )

    # Build masks. Compile-the-mask (block-wise build) is the only way to fit
    # long context — same env switch as DIT.
    compile_mask = os.environ.get('BD3LM_COMPILE_MASK', '0') == '1'
    self.self_block_mask = create_block_mask(
      partial(fine_local_block_mask,
              block_size=self.block_size, n=self.n,
              window_blocks=self.window_blocks),
      B=None, H=None,
      Q_LEN=2 * self.n, KV_LEN=2 * self.n,
      _compile=compile_mask)
    self.cross_block_mask = create_block_mask(
      partial(fine_to_coarse_mask,
              block_size=self.block_size, n_fine=self.n,
              k_coarse=self.k_coarse),
      B=None, H=None,
      Q_LEN=2 * self.n, KV_LEN=self.n_coarse_tokens,
      _compile=compile_mask)

    # Diffusion.to() looks for `.block_diff_mask` to move it to device;
    # alias the fine mask so the existing code path works.
    self.block_diff_mask = self.self_block_mask

  # Same shim as DIT.gen_mask so Diffusion._validate_configuration doesn't trip.
  def gen_mask(self, seqlen, block_size, attn_backend='flex'):
    return

  @torch.no_grad()
  def gate_stats(self):
    """Mean adaLN gate magnitudes + attn-proj norms across the fine blocks, at
    the constant t_cond (sigma=0; gates are weight-constants under
    time_conditioning=false). Cheap (~12 small matmuls) -> safe to log every
    train step. `gate_cross` / `gate_cross_over_self` track whether the COARSE
    (long-range) cross-attention pathway is engaging or atrophying."""
    dev = next(self.parameters()).device
    t_cond = F.silu(self.sigma_map(torch.zeros(1, device=dev)))   # (1, cond_dim)
    g1 = gc = g2 = 0.0
    n = len(self.fine_blocks)
    for blk in self.fine_blocks:
      mod = blk.adaLN_modulation(t_cond).chunk(7, dim=-1)
      # order: shift1, scale1, gate1, gate_cross, shift2, scale2, gate2
      g1 = g1 + mod[2].abs().mean()
      gc = gc + mod[3].abs().mean()
      g2 = g2 + mod[6].abs().mean()
    g1, gc, g2 = g1 / n, gc / n, g2 / n
    return {
      'gate1': g1, 'gate_cross': gc, 'gate2': g2,
      'gate_cross_over_self': gc / (g1 + 1e-9),
      'cross_out_norm': torch.stack(
        [b.cross_attn.out_proj.weight.norm() for b in self.fine_blocks]).mean(),
      'self_proj_norm': torch.stack(
        [b.self_proj.weight.norm() for b in self.fine_blocks]).mean(),
    }

  def _gen_sampling_masks(self, L_curr: int):
    """Dense bool masks for sdpa sampling at the current generation length.

    Semantically the input during sampling is the partial x_0 sequence, so the
    self-attention is just the block-causal+local portion of the BD3-LM mask
    (the M_BC rule restricted to a +/- window_blocks band). The cross-attn is
    MIXED to match training exactly: the in-progress (last) block is the x_t
    query and uses M_OBC (coarse blocks 0..i-1), while the finished prefix
    blocks are x_0-style context and use M_BC (coarse blocks 0..j). See the
    inline comment below.

    NOTE: only supports block_size % k_coarse == 0. `encode_kmer` (called on the
    partial buffer in forward's sample path) asserts L_curr % k_coarse == 0, and
    the block-snapped window only guarantees L_curr % block_size == 0 — so a
    block_size not divisible by k_coarse (e.g. 16 with k=6) crashes at stride 0
    (L_curr=block_size). To support it, pad/trim the coarse encode to a multiple
    of k_coarse.
    """
    device = next(self.parameters()).device
    q = torch.arange(L_curr, device=device).view(-1, 1)
    k = torch.arange(L_curr, device=device).view(1, -1)
    block_q = q // self.block_size
    block_kv = k // self.block_size
    # M_BC + locality (BC -> block_q >= block_kv; locality -> band <= window).
    self_mask = (block_q >= block_kv) & ((block_q - block_kv) <= self.window_blocks)

    Lc_curr = max(L_curr // self.k_coarse, 1)
    coarse_idx = torch.arange(Lc_curr, device=device).view(1, -1)
    # Cross-attn alignment must match TRAINING exactly:
    #   * the in-progress (last) block is the x_t query -> M_OBC: it reads coarse
    #     tokens of blocks 0..i-1 only, NOT its own block (whose k-mers are UNK
    #     here and which the x_t query never sees at train time). The first block
    #     (last_block == 0) reads no coarse tokens (CrossAttention zero-fills it).
    #   * the already-finished prefix blocks are x_0-style context -> M_BC: block
    #     j reads coarse tokens of blocks 0..j (own clean block included), exactly
    #     how the x_0 half is encoded during training.
    last_block = (L_curr // self.block_size) - 1
    is_inprogress = block_q == last_block
    n_blocks_visible = torch.where(is_inprogress, block_q, block_q + 1)
    visible_nucs = n_blocks_visible * self.block_size
    max_coarse = visible_nucs // self.k_coarse - 1
    cross_mask = coarse_idx <= max_coarse
    return self_mask, cross_mask

  @staticmethod
  def _apply_coarse_ablate(c, spec):
    """Counterfactual on the coarse memory (B, L/k, d_coarse), for the L_use
    incentive probe. 'zero' -> no coarse info; 'shuffle' -> permute coarse tokens
    along the sequence axis (dim=1) so retrieval-by-position is destroyed while the
    tensor stays a valid, same-statistics feature matrix. Leak-safe: c is derived
    from x0 only, so ablating it can only REMOVE information, never add any."""
    if spec is None:
      return c
    if spec == 'zero':
      return torch.zeros_like(c)
    if spec == 'shuffle':
      perm = torch.randperm(c.shape[1], device=c.device)
      return c[:, perm, :]
    raise ValueError(f'unknown coarse_ablate spec: {spec}')

  @staticmethod
  def _rel_bucket(delta, num_buckets, max_dist):
    """Signed, symmetric, half-exact-half-log bucketing of an integer offset
    (T5-style). delta in (-max_dist, max_dist) -> bucket id in [0, num_buckets)."""
    n = num_buckets // 2
    ret = (delta > 0).to(torch.long) * n
    d = delta.abs()
    max_exact = n // 2
    small = d < max_exact
    large = max_exact + (
      torch.log(d.float().clamp(min=1.0) / max_exact)
      / math.log(max(max_dist, max_exact + 1) / max_exact) * (n - max_exact)
    ).to(torch.long)
    large = large.clamp(max=n - 1)
    return ret + torch.where(small, d, large)

  def _build_cross_add_mask(self, device):
    """A2: (1, H, 2n, n_coarse) FLOAT additive bias for the cross-attention =
    learned bucketed relative-offset bias + block-causal mask (-inf on disallowed
    keys). Computed once per forward, shared across all fine blocks."""
    if not self.cross_rel_bias:
      return None
    n, k, ncoarse, bs = self.n, self.k_coarse, self.n_coarse_tokens, self.block_size
    q = torch.arange(2 * n, device=device)
    x0_flag = q >= n
    p = torch.where(x0_flag, q - n, q)                       # fine nucleotide coord
    kj = torch.arange(ncoarse, device=device) * k            # coarse start nucleotide
    delta = p[:, None] - kj[None, :]                         # (2n, ncoarse)
    bucket = self._rel_bucket(delta, self.cross_rel_buckets, n)
    bias = self.cross_bias_table[bucket]                     # (2n, ncoarse, H)
    bias = bias.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)   # (1, H, 2n, ncoarse)
    # block-causal allow-mask (same rule as fine_to_coarse_mask)
    block_q = p // bs
    nvis = torch.where(x0_flag, block_q + 1, block_q) * bs
    max_coarse = nvis // k - 1                               # (2n,)
    allow = torch.arange(ncoarse, device=device)[None, :] <= max_coarse[:, None]
    bias = bias.masked_fill(~allow[None, None], float('-inf'))
    return bias

  def _build_gathered(self, x, c):
    """B0/B1 probes: hard-wire the pointer to the source at i - gather_offset, so the
    SELECTION problem is removed and only representation+decoding is under test.
    Returns (B, 2n, d_src) aligned with the [xt; x0] query stream, zeroed where the
    source position does not exist (i < offset)."""
    if self.cross_mode == 'attn':
      return None
    D, n = self.gather_offset, self.n
    idx = torch.arange(n, device=x.device) - D
    valid = (idx >= 0).view(1, -1, 1)
    idx = idx.clamp(min=0)
    if self.cross_mode == 'gather_fine':          # B0 oracle: the clean source itself
      g = x[:, n:, :].index_select(1, idx)
    elif self.cross_mode == 'gather_coarse':      # B1: only the correct coarse token
      j = (idx // self.k_coarse).clamp(max=self.n_coarse_tokens - 1)
      g = c.index_select(1, j)
    else:
      raise ValueError(f'unknown cross_mode {self.cross_mode}')
    g = g * valid.to(g.dtype)
    return torch.cat([g, g], dim=1)

  def _build_cross_rotary(self, rotary_f):
    """Shared-coordinate rotary for fine->coarse cross-attention (B3 ablation).

    `rotary_f` is the FINE rotary table for positions 0..n-1 (fine head_dim).
      Fine queries : the training stream is [xt; x0], each half indexed 0..n-1
                     (matching how self-attention applies rotary per half), so
                     cos_q tiles the base table twice -> (2n, D/2).
      Coarse keys  : coarse token j spans nucleotides [k*j, k*j+k), so its
                     coordinate is k*j. Subsampling the SAME fine table every k
                     yields exactly positions 0, k, 2k, ... -> (n_coarse, D/2).
    Using ONE table for both guarantees a shared coordinate frame, so the offset
    (p - k*j) that a fixed-distance copy needs is directly representable by the
    attention instead of having to be reconstructed from content."""
    if not self.cross_align:
      return None
    cos, sin = rotary_f
    half = cos.shape[-1] // 2
    base_c = cos[0, :, 0, 0, :half]
    base_s = sin[0, :, 0, 0, :half]
    return (torch.cat([base_c, base_c], dim=0),
            torch.cat([base_s, base_s], dim=0),
            base_c[::self.k_coarse],
            base_s[::self.k_coarse])

  def forward(self, indices, sigma, sample_mode=False, store_kv=False,
              coarse_ablate=None):
    """
    Training: indices is (B, 2L) = [xt; x0]. Returns (B, L, vocab) on xt half.

    Sampling (sample_mode=True, store_kv=False): indices is (B, L_curr), the
    partial x_0 generation state. Returns (B, L_curr, vocab). kv_cache
    sampling (store_kv / sampling.kv_cache=True) is not implemented.
    """
    if store_kv or (sample_mode and getattr(
        self.config.sampling, 'kv_cache', False)):
      raise NotImplementedError(
        "DualStreamDIT kv_cache sampling not implemented; set "
        "sampling.kv_cache=False to use the dense-mask sampling path.")

    B, L_in = indices.shape

    if sample_mode:
      # Sampling: input IS the x_0 stream (partially generated). Encode coarse
      # from it directly and use sdpa with dense masks built at L_in.
      coarse_ids = encode_kmer(indices, k=self.k_coarse).long()
      x = self.vocab_embed(indices)
      c = self.coarse_embed(coarse_ids)
      if sigma is None:
        t_cond = None
      else:
        t_cond = F.silu(self.sigma_map(sigma))
      rotary_c = self.rotary_emb_coarse(c)
      for block in self.coarse_blocks:
        c = block(c, rotary_c)
      self_mask, cross_mask = self._gen_sampling_masks(L_in)
      rotary_f = self.rotary_emb(x)  # rotary at the full L_in
      with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for block in self.fine_blocks:
          x = block(x, c, rotary_f, t_cond,
                    self_mask=self_mask, cross_mask=cross_mask,
                    sample_mode=True)
        x = self.output_layer(x, t_cond)
      return x  # (B, L_in, vocab); the diffusion caller slices the last block.

    # ------------------------ Training path ------------------------
    assert L_in == 2 * self.n, \
      f"Expected training indices of length 2*L={2 * self.n}, got {L_in}"
    x0 = indices[:, self.n:]
    coarse_ids = encode_kmer(x0, k=self.k_coarse).long()
    x = self.vocab_embed(indices)
    c = self.coarse_embed(coarse_ids)
    if sigma is None:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))
    rotary_c = self.rotary_emb_coarse(c)
    for block in self.coarse_blocks:
      c = block(c, rotary_c)
    c = self._apply_coarse_ablate(c, coarse_ablate)
    rotary_f = self.rotary_emb(x[:, :self.n])
    cross_rot = self._build_cross_rotary(rotary_f)
    gathered = self._build_gathered(x, c)
    cross_add = self._build_cross_add_mask(x.device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for block in self.fine_blocks:
        x = block(x, c, rotary_f, t_cond,
                  self_mask=self.self_block_mask,
                  cross_mask=self.cross_block_mask,
                  cross_rotary=cross_rot,
                  gathered=gathered,
                  force_gate_cross=self.force_gate_cross,
                  cross_add_mask=cross_add)
      x = self.output_layer(x, t_cond)
    return x[:, :self.n]
