"""Reverse-complement equivariance for the bidirectional SSM backbone.

DNA is double stranded: a locus and its reverse complement are the same
physical object read from the other strand.  The symmetry operation is

    rc(x)[t] = complement(x[L - 1 - t])

with ``A<->T`` and ``C<->G``.  Caduceus (arXiv:2403.03234) builds a model that
respects it; this module adds the same property to ``BidirectionalSSM``.

Design (Z2-symmetric weights)
-----------------------------
Write ``Flip`` for reversal along length and ``Comp`` for complementation, so
``RC = Flip . Comp``.  Two facts make the construction cheap here:

* ``BiMambaLayer`` shares every direction parameter between the forward and
  reverse scans and *sums* them, so the backbone is already **exactly**
  ``Flip``-equivariant (with the left/right caches swapped).  This is verified
  to ~1e-15 by ``scripts/smoke/rc_equivariance.py`` test T1.
* therefore full ``RC`` equivariance reduces to ``Comp`` equivariance alone.

``Comp`` acts on hidden states as ``rho``, the involution that swaps the two
halves of the channel axis.  A layer commutes with ``rho`` iff every linear map
inside it has the block form ``[[A, B], [B, A]]`` (note ``B`` need not be zero,
so the two halves *do* mix -- strictly more expressive than Caduceus-PS, which
forces ``B = 0``), every elementwise nonlinearity is channelwise (SiLU, GELU,
softplus all are), and every per-channel vector is ``rho``-invariant.

The constraint is imposed by *weight materialisation*: the free half of each
tensor is stored, and the full tensor is rebuilt on access through
``torch.nn.utils.parametrize``.  ``SegmentMamba2``'s scans, caches, boundary
prefill and fused kernels are therefore **completely untouched** -- which is
the point, because that is the most delicate code in the repo.

Costs at ``hidden_size=768``: parameters roughly halve (the free dimension is
~50.7M instead of 101.4M), FLOPs and activation memory are unchanged.

Known limitation (documented, not hidden): because the backbone is *also*
exactly ``Flip``-equivariant, an RC-equivariant model built this way remains
invariant -- under mean pooling -- to reading a sequence backwards *without*
complementing.  Plain reversal is not a symmetry of DNA.  See the module note
at the bottom of ``scripts/smoke/rc_equivariance.py`` and the report.

Token space
-----------
``DNATokenizer`` (``dataloader.py:228-240``) numbers the vocabulary

    [CLS]=0 [SEP]=1 [BOS]=2 [EOS]=3 [MASK]=4 [PAD]=5 [RESERVED]=6 [UNK]=7
    A=8 C=9 G=10 T=11 N=12

so the complement involution is ``pi = [0..7, 11, 10, 9, 8, 12]``: ``N`` and
all eight specials are fixed points.  ``[MASK]`` being a fixed point is what
makes ``pi`` commute with the absorbing-state corruption in ``q_xt``.
``[BOS]``/``[EOS]`` being fixed points is *semantically* wrong (``rc`` turns a
``[BOS] ... [EOS]`` sequence into ``[EOS] ... [BOS]``), which is why
``assert_rc_safe_special_tokens`` refuses configs that insert them.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

# A/C/G/T/N ids under `DNATokenizer`, followed by the eight fixed-point
# specials.  Kept as a literal so a vocabulary change fails loudly in
# `complement_permutation` instead of silently corrupting the augmentation.
DNA_COMPLEMENT_IDS: tuple[int, ...] = (
  0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12)
DNA_VOCAB_SIZE = len(DNA_COMPLEMENT_IDS)


def complement_permutation(
    vocab_size: int,
    tokenizer=None,
    device=None) -> torch.Tensor:
  """The complement involution ``pi`` as an index tensor of length ``vocab_size``.

  ``vocab_size`` may exceed the tokenizer's vocabulary by one when
  ``Diffusion`` appends its own mask id (``diffusion.py:60-65``); the extra
  entry is a fixed point.
  """
  if vocab_size < DNA_VOCAB_SIZE:
    raise ValueError(
      f"RC equivariance needs the DNA vocabulary (>= {DNA_VOCAB_SIZE} ids), "
      f"got vocab_size={vocab_size}")
  perm = list(DNA_COMPLEMENT_IDS) + list(range(DNA_VOCAB_SIZE, vocab_size))
  if tokenizer is not None:
    expected = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    for base, mate in expected.items():
      base_id = tokenizer.convert_tokens_to_ids(base)
      mate_id = tokenizer.convert_tokens_to_ids(mate)
      if base_id >= vocab_size or perm[base_id] != mate_id:
        raise ValueError(
          "Tokenizer vocabulary does not match DNA_COMPLEMENT_IDS: "
          f"complement({base})={mate} maps {base_id} -> {perm[base_id]}, "
          f"expected {mate_id}")
    mask_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_id is not None and mask_id < vocab_size and perm[mask_id] != mask_id:
      raise ValueError("[MASK] must be a fixed point of the complement map")
  return torch.tensor(perm, dtype=torch.long, device=device)


def rc_token_ids(ids: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
  """Reverse-complement a ``[batch, length]`` id tensor.

  Always apply this to a *full, contiguous* window.  If a sequence is right
  padded (the eval harness pads with the ``N`` id, see
  ``scripts/eval/dnahnet/score_mavedb.py:90-101``) this puts the padding on the
  left, which is **not** the encoding of the reverse-complemented string and
  gives different hidden states because the scan runs through the padding.
  Reverse-complement the *string* before encoding instead.
  """
  if ids.ndim != 2:
    raise ValueError("Expected [batch, length] token ids")
  return perm.to(ids.device)[torch.flip(ids, dims=(1,))]


def assert_rc_safe_special_tokens(config):
  """Refuse configs whose token stream can contain [BOS]/[EOS] at a boundary.

  ``[BOS]``/``[EOS]`` are fixed points of ``pi`` but swap roles under ``rc``,
  and ``Diffusion._preserve_observed_bos`` (``diffusion.py:383-387``) pins
  position 0.  The DNA corpora set ``insert_train_special: False``
  (``configs/data/carbon-prokaryote.yaml:12``) so no BOS/EOS block delimiter is
  ever emitted; assert it rather than assume it.
  """
  data = getattr(config, 'data', None)
  if data is None:
    return
  for key in ('insert_train_special', 'insert_valid_special'):
    if bool(data.get(key, True)):
      raise ValueError(
        f"data.{key}=True inserts [BOS]/[EOS] block delimiters, which "
        "reverse-complement to [EOS] ... [BOS]. Set it to False before "
        "enabling reverse-complement augmentation or equivariance.")


# ---------------------------------------------------------------------------
# Parametrizations.  Each stores the free half of a tensor and rebuilds the
# full tensor in `forward`; `right_inverse` is the orthogonal projection onto
# the constrained subspace, so registering one on an already-initialised weight
# keeps the closest equivariant weight rather than throwing it away.
# ---------------------------------------------------------------------------


class _RCParametrization(nn.Module):
  """Base class carrying the shape of the free parameter."""

  free_shape: tuple

  @property
  def free_numel(self) -> int:
    return int(math.prod(self.free_shape))


class SymSym(_RCParametrization):
  """``W == W[rho_out][:, rho_in]`` for two half-swap involutions.

  Materialises ``[[A, B], [B, A]]``.  ``B`` is free, so the two channel halves
  mix; Caduceus-PS is the ``B = 0`` special case.
  """

  def __init__(self, out_features: int, in_features: int):
    super().__init__()
    if out_features % 2 or in_features % 2:
      raise ValueError(
        f"SymSym needs even dimensions, got ({out_features}, {in_features})")
    self.out_half = out_features // 2
    self.in_half = in_features // 2
    self.free_shape = (2, self.out_half, self.in_half)

  def forward(self, p: torch.Tensor) -> torch.Tensor:
    a, b = p[0], p[1]
    return torch.cat(
      (torch.cat((a, b), dim=1), torch.cat((b, a), dim=1)), dim=0)

  def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
    o, i = self.out_half, self.in_half
    a = (w[:o, :i] + w[o:, i:]) / 2
    b = (w[:o, i:] + w[o:, :i]) / 2
    return torch.stack((a, b), dim=0)


class ColSym(_RCParametrization):
  """``W == W[:, rho_in]``: the output index is ``rho``-invariant.

  Used for the ``B``/``C`` rows of ``in_proj``, which are broadcast across all
  heads (``mamba2_segment.py:562-563``) and so must not move under ``rho``.
  """

  def __init__(self, out_features: int, in_features: int):
    super().__init__()
    if in_features % 2:
      raise ValueError(f"ColSym needs an even input width, got {in_features}")
    self.in_half = in_features // 2
    self.free_shape = (out_features, self.in_half)

  def forward(self, p: torch.Tensor) -> torch.Tensor:
    return torch.cat((p, p), dim=1)

  def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
    return (w[:, :self.in_half] + w[:, self.in_half:]) / 2


class RowSym(_RCParametrization):
  """``t[i] == t[rho(i)]`` along dim 0 for a half-swap ``rho`` on a prefix.

  ``tied`` leading entries are tied pairwise; any remaining entries are free.
  Covers ``conv1d.weight``/``bias`` (whose ``B``/``C`` channels are free),
  every RMSNorm weight, ``A_log``/``dt_bias``/``D``, and the output rows of the
  timestep embedder.
  """

  def __init__(
      self,
      total: int,
      trailing_shape: Sequence[int] = (),
      tied: Optional[int] = None):
    super().__init__()
    tied = total if tied is None else tied
    if tied % 2 or tied > total:
      raise ValueError(f"RowSym needs an even tied prefix <= {total}, got {tied}")
    self.total = total
    self.tied = tied
    self.tied_half = tied // 2
    self.free_shape = (self.tied_half + (total - tied), *trailing_shape)

  def forward(self, p: torch.Tensor) -> torch.Tensor:
    head = p[:self.tied_half]
    tail = p[self.tied_half:]
    return torch.cat((head, head, tail), dim=0)

  def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
    head = (w[:self.tied_half] + w[self.tied_half:self.tied]) / 2
    return torch.cat((head, w[self.tied:]), dim=0)


class VocabColSym(_RCParametrization):
  """``W == W[pi][:, rho]``: ``E[v] = concat(F[v], F[pi(v)])``.

  This is the embedding constraint ``Emb(pi(v)) = rho . Emb(v)`` and, applied
  to ``[vocab, hidden]``, it is *also* the output-head constraint
  ``logits(rho h) = P logits(h)``.  The two coincide, so ``tie_word_embeddings``
  stays valid: one free table satisfies both.

  Note this is where Caduceus's ``flip_chan`` shortcut does **not** transfer.
  Their alphabet is exactly ``{A,C,G,T}``, so reversing four logits *is*
  complementation.  Our 13-token vocabulary has eight specials before ``A`` and
  ``N`` after ``T``; a naive channel flip would map ``[CLS] <-> N``.
  """

  def __init__(self, perm: torch.Tensor, hidden_size: int):
    super().__init__()
    if hidden_size % 2:
      raise ValueError(f"VocabColSym needs an even width, got {hidden_size}")
    self.hidden_half = hidden_size // 2
    self.register_buffer('perm', perm.clone(), persistent=False)

  @property
  def free_shape(self) -> tuple:
    return (int(self.perm.numel()), self.hidden_half)

  def forward(self, f: torch.Tensor) -> torch.Tensor:
    return torch.cat((f, f[self.perm]), dim=1)

  def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
    h = self.hidden_half
    return (w[:, :h] + w[self.perm][:, h:]) / 2


class RowBlocks(_RCParametrization):
  """Stacks several row-blocks with different constraints into one weight.

  ``in_proj`` emits ``[z, x, B, C, dt]`` (``mamba2_segment.py:102-104``), whose
  output indices carry three different ``rho`` actions: ``rho_inner`` on ``z``
  and ``x``, the identity on ``B`` and ``C``, and the head half-swap on ``dt``.
  The free parameters are kept in one flat tensor so the parametrization has a
  single ``original``.
  """

  def __init__(self, blocks: Sequence[_RCParametrization], rows: Sequence[int]):
    super().__init__()
    if len(blocks) != len(rows):
      raise ValueError("RowBlocks needs one row count per block")
    self.blocks = nn.ModuleList(blocks)
    self.rows = tuple(int(r) for r in rows)
    self.sizes = tuple(b.free_numel for b in blocks)
    self.free_shape = (sum(self.sizes),)

  def forward(self, flat: torch.Tensor) -> torch.Tensor:
    parts, offset = [], 0
    for block, size in zip(self.blocks, self.sizes):
      parts.append(block(flat[offset:offset + size].view(block.free_shape)))
      offset += size
    return torch.cat(parts, dim=0)

  def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
    parts, offset = [], 0
    for block, rows in zip(self.blocks, self.rows):
      parts.append(block.right_inverse(w[offset:offset + rows]).reshape(-1))
      offset += rows
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register(module: nn.Module, name: str, parametrization: _RCParametrization):
  if getattr(module, name, None) is None:
    raise ValueError(f"{type(module).__name__} has no tensor {name!r} to tie")
  parametrize.register_parametrization(module, name, parametrization)


def _xavier_bound(weight: torch.Tensor) -> float:
  fan_out, fan_in = weight.shape[0], weight.shape[1]
  return math.sqrt(6.0 / (fan_in + fan_out))


def _fill_uniform(module: nn.Module, name: str, bound: float):
  with torch.no_grad():
    module.parametrizations[name].original.uniform_(-bound, bound)


def apply_rc_equivariance(
    model,
    perm: torch.Tensor,
    *,
    a_init_max: float = 16.0,
    dt_min: float = 1e-3,
    dt_max: float = 1e-1,
    dt_init_floor: float = 1e-4,
    embedding_std: float = 0.02,
    reinitialize: bool = True):
  """Constrain a ``BidirectionalSSM`` to be exactly reverse-complement equivariant.

  Call *after* the model's own ``_initialize_weights``.  Every constrained
  tensor is projected onto its equivariant subspace by ``right_inverse``, then
  (with ``reinitialize=True``) resampled so the materialised weight keeps the
  distribution the unconstrained initialiser intended -- projection alone
  shrinks the entrywise variance by 2.

  Returns the number of free parameters after the constraint.
  """
  hidden = model.hidden_size
  if hidden % 2:
    raise ValueError(f"model.hidden_size must be even, got {hidden}")

  tied_head = model.output.weight is model.token_embedding.weight
  if tied_head:
    # `torch.nn.utils.parametrize` resizes the incoming Parameter *in place*
    # (`_maybe_set` -> `Tensor.set_`), so the alias `output.weight is
    # token_embedding.weight` has to be broken before registering or the second
    # registration would see an already-halved tensor. The two are re-tied
    # below through their shared free table.
    model.output.weight = nn.Parameter(
      model.token_embedding.weight.detach().clone())

  # --- embedding / head -----------------------------------------------------
  _register(model.token_embedding, 'weight', VocabColSym(perm, hidden))
  if reinitialize:
    with torch.no_grad():
      model.token_embedding.parametrizations.weight.original.normal_(
        mean=0.0, std=embedding_std)
  _register(model.output, 'weight', VocabColSym(perm, hidden))
  if tied_head:
    # Re-tie through the free table: both modules compute the same weight from
    # the same `original`, so `parameters()` still deduplicates them.
    model.output.parametrizations.weight.original = (
      model.token_embedding.parametrizations.weight.original)
  elif reinitialize:
    _fill_uniform(model.output, 'weight', 1.0 / math.sqrt(hidden))

  # --- timestep embedder ----------------------------------------------------
  # Only the final output must be rho-invariant (it is broadcast over every
  # position and added to the residual stream); the frequency-space layers are
  # unconstrained.
  if model.time_embedding is not None:
    last = model.time_embedding.mlp[-1]
    bound = _xavier_bound(last.weight)
    _register(last, 'weight', RowSym(hidden, trailing_shape=(last.in_features,)))
    if reinitialize:
      _fill_uniform(last, 'weight', bound)
    if last.bias is not None:
      _register(last, 'bias', RowSym(hidden))

  # --- layers ---------------------------------------------------------------
  for layer in model.layers:
    _register(layer.mixer_norm, 'weight', RowSym(hidden))
    _register(layer.mlp_norm, 'weight', RowSym(hidden))

    mixer = layer.mixer
    d_inner, d_state, nheads = mixer.d_inner, mixer.d_state, mixer.nheads
    if d_inner % 2 or (d_inner // 2) % mixer.headdim:
      raise ValueError(
        f"rho must be head aligned: d_inner={d_inner} headdim={mixer.headdim}")
    if nheads % 2:
      raise ValueError(f"rho needs an even head count, got {nheads}")
    if mixer.in_proj.bias is not None or mixer.out_proj.bias is not None:
      raise ValueError(
        "RC equivariance assumes the mixer projections are bias free")
    if mixer.conv1d.groups != mixer.conv_dim:
      raise ValueError("RC equivariance assumes a depthwise mixer convolution")

    in_bound = _xavier_bound(mixer.in_proj.weight)
    _register(mixer.in_proj, 'weight', RowBlocks(
      blocks=(SymSym(d_inner, hidden),      # z
              SymSym(d_inner, hidden),      # x
              ColSym(d_state, hidden),      # B, broadcast across heads
              ColSym(d_state, hidden),      # C, broadcast across heads
              SymSym(nheads, hidden)),      # dt, one per head
      rows=(d_inner, d_inner, d_state, d_state, nheads)))
    if reinitialize:
      _fill_uniform(mixer.in_proj, 'weight', in_bound)

    out_bound = _xavier_bound(mixer.out_proj.weight)
    _register(mixer.out_proj, 'weight', SymSym(hidden, d_inner))
    if reinitialize:
      _fill_uniform(mixer.out_proj, 'weight', out_bound)

    # Depthwise: the x channels follow rho_inner, the B/C channels are fixed.
    conv_bound = 1.0 / math.sqrt(mixer.d_conv)
    _register(mixer.conv1d, 'weight', RowSym(
      mixer.conv_dim, trailing_shape=(1, mixer.d_conv), tied=d_inner))
    if reinitialize:
      _fill_uniform(mixer.conv1d, 'weight', conv_bound)
    if mixer.conv1d.bias is not None:
      _register(mixer.conv1d, 'bias', RowSym(mixer.conv_dim, tied=d_inner))
      if reinitialize:
        _fill_uniform(mixer.conv1d, 'bias', conv_bound)

    _register(mixer, 'norm_weight', RowSym(d_inner))
    for name in ('A_log', 'dt_bias', 'D'):
      _register(mixer, name, RowSym(nheads))
      # `mamba2_segment.py:119-128` marks these three no-weight-decay; the flag
      # has to move to the tensor the optimizer actually sees.
      mixer.parametrizations[name].original._no_weight_decay = True
    if reinitialize:
      with torch.no_grad():
        half = nheads // 2
        a = torch.empty(half).uniform_(1.0, float(a_init_max))
        mixer.parametrizations.A_log.original.copy_(torch.log(a))
        dt = torch.exp(
          torch.rand(half) * (math.log(dt_max) - math.log(dt_min))
          + math.log(dt_min)).clamp(min=dt_init_floor)
        mixer.parametrizations.dt_bias.original.copy_(
          dt + torch.log(-torch.expm1(-dt)))
        mixer.parametrizations.D.original.fill_(1.0)

    mlp_in, mlp_out = layer.mlp.net[0], layer.mlp.net[3]
    mlp_hidden = mlp_in.out_features
    if mlp_hidden % 2:
      raise ValueError(f"rho needs an even MLP width, got {mlp_hidden}")
    in_bound = _xavier_bound(mlp_in.weight)
    out_bound = _xavier_bound(mlp_out.weight)
    _register(mlp_in, 'weight', SymSym(mlp_hidden, hidden))
    _register(mlp_out, 'weight', SymSym(hidden, mlp_hidden))
    if reinitialize:
      _fill_uniform(mlp_in, 'weight', in_bound)
      _fill_uniform(mlp_out, 'weight', out_bound)
    if mlp_in.bias is not None:
      _register(mlp_in, 'bias', RowSym(mlp_hidden))
    if mlp_out.bias is not None:
      _register(mlp_out, 'bias', RowSym(hidden))

  _register(model.final_norm, 'weight', RowSym(hidden))
  return rc_free_parameter_count(model)


def rc_free_parameter_count(model) -> int:
  """Trainable parameters after the constraint (``original`` tensors only)."""
  return sum(p.numel() for p in model.parameters())


def swap_halves(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """``rho``: swap the two halves of ``x`` along ``dim``."""
  first, second = x.chunk(2, dim=dim)
  return torch.cat((second, first), dim=dim)
