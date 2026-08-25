"""Leakage-safe bidirectional Mamba backbone for block diffusion.

There are two valid directions of context:

* de novo: a clean, timestep-free left cache plus a reverse scan confined to
  the current noisy block;
* C-a infilling: the same left cache plus a clean, timestep-free right cache
  that initializes the active block's reverse scan.

The clean target block is never used to construct either cache.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .mamba2_segment import Mamba2State, SegmentMamba2, rmsnorm_fn


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """RMS normalisation, fused on CUDA.

    The eager body below retains `normalized` for the weight multiply that
    follows it -- a full-width activation per call, and there are 25 of these
    per stack (two per layer plus `final_norm`). Measured at batch 4 / L=8192
    that is 25 x 96 MiB = 2.34 GiB the fused kernel does not need, because it
    saves its input and recomputes the rest in backward.

    Same kernel already accepted in production by `SegmentMamba2._gated_output`,
    with z=None here since there is no gate. Guarded on CUDA: the fused path is
    unavailable on CPU, and every equivalence test in the suite is CPU-only, so
    the strict fp64 comparisons keep exercising the eager body below.
    """
    if rmsnorm_fn is not None and x.is_cuda:
      return rmsnorm_fn(x, self.weight, None, eps=self.eps)
    # Promote rather than hard-cast: `.float()` upcasts bf16/fp16 as intended
    # but silently *downcasts* fp64, which caps the channel-sum accuracy at
    # ~1e-7. The channel sum is the one reduction whose association order
    # changes under the RC channel involution, so an fp64 equivariance check
    # cannot get below fp32 noise without this. Identical for fp32/bf16/fp16.
    accum_dtype = torch.promote_types(x.dtype, torch.float32)
    variance = x.to(accum_dtype).pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
    return normalized * self.weight.to(dtype=x.dtype)


class TimestepEmbedder(nn.Module):
  """DiT-style sinusoidal scalar embedding used on active tokens only."""

  def __init__(self, hidden_size: int, frequency_size: int = 128):
    super().__init__()
    self.frequency_size = frequency_size
    self.mlp = nn.Sequential(
      nn.Linear(frequency_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size))

  def forward(self, sigma: torch.Tensor) -> torch.Tensor:
    half = self.frequency_size // 2
    frequencies = torch.exp(
      -math.log(10_000) * torch.arange(
        half, device=sigma.device, dtype=torch.float32) / half)
    angles = sigma.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    # The sinusoid is always built in fp32; hand it to the MLP in the MLP's own
    # dtype. No-op for an fp32 model (and autocast still owns the cast under
    # bf16), but it is what lets a `.double()` model run at all -- otherwise
    # `mat1 and mat2 must have the same dtype`.
    return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
  def __init__(self, dim: int, ratio: float, dropout: float):
    super().__init__()
    hidden = int(dim * ratio)
    self.net = nn.Sequential(
      nn.Linear(dim, hidden),
      nn.GELU(approximate="tanh"),
      nn.Dropout(dropout),
      nn.Linear(hidden, dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


@dataclass(frozen=True)
class DirectionalCache:
  """Per-layer recurrent states for one clean direction."""

  states: tuple[Mamba2State, ...]
  length: int
  direction: str

  def detach(self) -> "DirectionalCache":
    return DirectionalCache(
      tuple(state.detach() for state in self.states),
      self.length,
      self.direction)

  def clone(self) -> "DirectionalCache":
    return DirectionalCache(
      tuple(state.clone() for state in self.states),
      self.length,
      self.direction)

  @property
  def nbytes(self) -> int:
    return sum(
      state.conv.numel() * state.conv.element_size()
      + state.ssm.numel() * state.ssm.element_size()
      for state in self.states)


def stack_boundary_caches(
    caches: "tuple[DirectionalCache, ...]") -> DirectionalCache:
  """Folds one cache per block into a single batch-major cache.

  Row ``b * num_blocks + i`` of the result holds sequence ``b``'s boundary
  state for block ``i``, which is the layout produced by reshaping a
  ``[batch, num_blocks * block_size]`` sequence to ``[batch * num_blocks,
  block_size]``. This is what lets every block be denoised in one batched
  call instead of one call per block.
  """
  if not caches:
    raise ValueError("stack_boundary_caches requires at least one cache")
  direction = caches[0].direction
  num_layers = len(caches[0].states)
  if any(cache.direction != direction for cache in caches):
    raise ValueError("Cannot stack caches from different directions")
  if any(len(cache.states) != num_layers for cache in caches):
    raise ValueError("Cannot stack caches with different layer counts")

  states = []
  for layer_index in range(num_layers):
    conv = torch.stack(
      [cache.states[layer_index].conv for cache in caches], dim=1)
    ssm = torch.stack(
      [cache.states[layer_index].ssm for cache in caches], dim=1)
    states.append(
      Mamba2State(conv.flatten(0, 1), ssm.flatten(0, 1)))
  # The folded rows cover different context lengths, so a single scalar length
  # is meaningless here; -1 marks it as heterogeneous.
  return DirectionalCache(tuple(states), length=-1, direction=direction)


class BiMambaLayer(nn.Module):
  def __init__(
      self,
      dim: int,
      d_state: int,
      d_conv: int,
      expand: int,
      headdim: int,
      chunk_size: int,
      mlp_ratio: float,
      dropout: float,
      backend: str,
      a_init_max: float = 16.0,
      dt_max: float = 1e-1,
      bidirectional_impl: str = "fused",
  ):
    super().__init__()
    if bidirectional_impl not in {"fused", "split"}:
      raise ValueError(
        "model.bidirectional_impl must be 'fused' or 'split', got "
        f"{bidirectional_impl!r}")
    self.bidirectional_impl = bidirectional_impl
    self.mixer_norm = RMSNorm(dim)
    # Direction weights are deliberately shared, matching common BiMamba
    # practice and avoiding a needless 2x parameter increase.
    self.mixer = SegmentMamba2(
      d_model=dim,
      d_state=d_state,
      d_conv=d_conv,
      expand=expand,
      headdim=headdim,
      chunk_size=chunk_size,
      backend=backend,
      # Per-step state retention is exp(A*dt) with A = -exp(A_log) and
      # dt = softplus(.+dt_bias), so these two ceilings set how long suffix
      # information survives. At the defaults the measured per-head half-life
      # is 4.66 nucleotides and no head exceeds 256, while the C-a range
      # measurement shows the DATA carries right-context value out to ~1 kb.
      # Lowering either ceiling initialises slower-decaying heads.
      A_init_range=(1.0, float(a_init_max)),
      dt_max=float(dt_max))
    self.mlp_norm = RMSNorm(dim)
    self.mlp = FeedForward(dim, mlp_ratio, dropout)
    self.dropout = nn.Dropout(dropout)

  def scan_clean(
      self,
      x: torch.Tensor,
      initial_state: Optional[Mamba2State],
  ) -> tuple[torch.Tensor, Mamba2State]:
    mixed, final_state = self.mixer.scan_segment(
      self.mixer_norm(x), initial_state)
    x = x + self.dropout(mixed)
    x = x + self.dropout(self.mlp(self.mlp_norm(x)))
    return x, final_state

  def scan_clean_with_boundaries(
      self,
      x: torch.Tensor,
      block_size: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``scan_clean`` over a whole prefix, keeping every block-boundary state.

    Returns the layer output alongside stacked ``[batch, num_blocks + 1, ...]``
    convolution and SSM states, where index ``i`` is the state entering block
    ``i`` of this layer's input.
    """
    mixed, conv_states, ssm_states = self.mixer.scan_with_block_boundaries(
      self.mixer_norm(x), block_size)
    x = x + self.dropout(mixed)
    x = x + self.dropout(self.mlp(self.mlp_norm(x)))
    return x, conv_states, ssm_states

  def scan_active(
      self,
      x: torch.Tensor,
      left_state: Mamba2State,
      right_state: Mamba2State,
  ) -> torch.Tensor:
    normalized = self.mixer_norm(x)
    # The paper uses a direct sum: no zero-initialized route gate that can
    # silently disable either context direction.
    if self.bidirectional_impl == "fused":
      # Same sum, but the input projection and the output projection are run
      # once instead of once per direction; see `scan_bidirectional`.
      mixed = self.mixer.scan_bidirectional(normalized, left_state, right_state)
    else:
      forward, _ = self.mixer.scan_segment(normalized, left_state)
      reverse, _ = self.mixer.scan_segment(
        torch.flip(normalized, dims=(1,)), right_state)
      mixed = forward + torch.flip(reverse, dims=(1,))
    x = x + self.dropout(mixed)
    x = x + self.dropout(self.mlp(self.mlp_norm(x)))
    return x


class BidirectionalSSM(nn.Module):
  """Partial-bidirectional block denoiser with optional C-a suffix cache."""

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    model = config.model
    self.vocab_size = vocab_size
    self.block_size = int(config.block_size)
    self.hidden_size = int(model.hidden_size)
    self.time_conditioning = bool(config.algo.time_conditioning)
    # 'layer_major' scans the whole clean prefix once per layer and harvests
    # every block's boundary state from it; 'block_major' is the original
    # one-short-scan-per-block loop, kept as an escape hatch and as the
    # equivalence oracle the tests compare against. They compute the same
    # states and the same gradient; only the floating-point association order
    # and the kernel-launch count differ.
    self.boundary_impl = str(model.get("boundary_impl", "layer_major"))
    if self.boundary_impl not in {"layer_major", "block_major"}:
      raise ValueError(
        "model.boundary_impl must be 'layer_major' or 'block_major', got "
        f"{self.boundary_impl!r}")

    # The clean boundary prefill runs with gradients on (diffusion.py calls
    # `prefill_*_boundaries_stacked` outside any `no_grad`, and the cache is
    # not detached), so it stores activations for `(num_blocks - 1) *
    # block_size` tokens across every layer -- roughly half of all the token
    # positions a training step touches. Recomputing that in backward instead
    # is mathematically identical and costs about one extra forward over the
    # prefix. It exists because peak memory, not arithmetic, is what pins the
    # SSM arms to micro batch 4 while the Transformer runs 8.
    # `true`/`false` pin the choice; `auto` decides once, here, from the run's
    # geometry. Both branches are BITWISE identical (tests/
    # test_bissm_diffusion_integration.py:180), so `auto` can only ever change
    # time and memory -- never a number, never a checkpoint's output. The
    # trade is one extra forward over the prefix (~1.16x of the arm's FLOPs,
    # and ~2573 of BiSSM's 8152 operator dispatches per step) bought against
    # `_prefill_activation_bytes` of peak memory, so it is only worth taking
    # when that memory is actually scarce.
    self.checkpoint_boundary_prefill = self._resolve_prefill_checkpoint(
      config, model)

    self.token_embedding = nn.Embedding(vocab_size, self.hidden_size)
    self.time_embedding = (
      TimestepEmbedder(self.hidden_size, int(model.get("cond_dim", 128)))
      if self.time_conditioning else None)
    self.layers = nn.ModuleList([
      BiMambaLayer(
        dim=self.hidden_size,
        d_state=int(model.get("ssm_state_size", 64)),
        d_conv=int(model.get("ssm_conv_size", 4)),
        expand=int(model.get("ssm_expand", 2)),
        headdim=int(model.get("ssm_head_dim", 64)),
        chunk_size=int(model.get("ssm_chunk_size", 128)),
        mlp_ratio=float(model.get("mlp_ratio", 4.0)),
        dropout=float(model.dropout),
        backend=str(model.get("ssm_backend", "auto")),
        a_init_max=float(model.get("ssm_a_init_max", 16.0)),
        dt_max=float(model.get("ssm_dt_max", 1e-1)),
        bidirectional_impl=str(
          model.get("bidirectional_impl", "fused")))
      for _ in range(int(model.n_blocks))
    ])
    self.final_norm = RMSNorm(self.hidden_size)
    self.output = nn.Linear(self.hidden_size, vocab_size, bias=False)
    if bool(model.get("tie_word_embeddings", True)):
      self.output.weight = self.token_embedding.weight

    self._sampling_left_cache: Optional[DirectionalCache] = None
    self._sampling_right_cache: Optional[DirectionalCache] = None
    self._initialize_weights()

    # Reverse-complement equivariance (models/rc_equivariance.py). Off unless
    # `model.rc_equivariant` is set, and the whole block is skipped when it is:
    # no parametrization is registered, no state-dict key changes, and every
    # existing checkpoint keeps loading and producing identical numbers.
    self.rc_equivariant = bool(model.get("rc_equivariant", False))
    if self.rc_equivariant:
      from . import rc_equivariance
      self.register_buffer(
        "complement_ids",
        rc_equivariance.complement_permutation(vocab_size),
        persistent=False)
      self.rc_free_parameters = rc_equivariance.apply_rc_equivariance(
        self,
        self.complement_ids,
        a_init_max=float(model.get("ssm_a_init_max", 16.0)),
        dt_max=float(model.get("ssm_dt_max", 1e-1)))

  def _initialize_weights(self):
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    for module in self.modules():
      if isinstance(module, nn.Linear) and module is not self.output:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def _compute_autocast(self, x: torch.Tensor):
    """bf16 for the layer stack, mirroring ``models/dit.py``'s inner autocast.

    ``Diffusion.forward`` wraps every backbone call in an explicit FP32
    autocast, because the Transformer needs its embedding, rotary and logit
    tail in FP32 or it fails to train. The Transformer's *blocks* escape that
    by re-opening a bf16 autocast of their own; this backbone had no such
    re-entry, so ``algo=ar`` with an SSM -- the one path that reaches the FP32
    wrapper without going through ``_forward_pass_bissm`` -- executed entirely
    in FP32, at the H200's ~67 TFLOP/s FP32 ceiling instead of on bf16 tensor
    cores. That, not the architecture, was the bulk of uSSM-AR's 3.09x
    training-time gap against the Transformer AR arm.

    Under the block-diffusion path this is a no-op: ``_forward_pass_bissm``
    already supplies a bf16 autocast, and nesting the same dtype changes
    nothing.
    """
    if x.device.type != "cuda":
      return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)

  def _empty_cache(
      self,
      batch_size: int,
      device: torch.device,
      dtype: torch.dtype,
      direction: str,
  ) -> DirectionalCache:
    return DirectionalCache(
      tuple(layer.mixer.zero_state(
        batch_size, device=device, dtype=dtype) for layer in self.layers),
      length=0,
      direction=direction)

  def _validate_cache(
      self,
      cache: DirectionalCache,
      batch_size: int,
      direction: str,
  ):
    if cache.direction != direction:
      raise ValueError(
        f"Expected a {direction} cache, received {cache.direction}")
    if len(cache.states) != len(self.layers):
      raise ValueError("Cache layer count does not match the model")
    if cache.states and cache.states[0].conv.shape[0] != batch_size:
      raise ValueError("Cache batch size does not match the active block")

  def _prefill(
      self,
      token_ids: torch.Tensor,
      cache: Optional[DirectionalCache],
      direction: str,
  ) -> DirectionalCache:
    if token_ids.ndim != 2:
      raise ValueError("Clean context must have shape [batch, length]")
    batch_size = token_ids.shape[0]
    x = self.token_embedding(token_ids)
    if cache is None:
      cache = self._empty_cache(batch_size, x.device, x.dtype, direction)
    self._validate_cache(cache, batch_size, direction)

    final_states = []
    for layer_index, layer in enumerate(self.layers):
      x, state = layer.scan_clean(x, cache.states[layer_index])
      final_states.append(state)
    return DirectionalCache(
      tuple(final_states), cache.length + token_ids.shape[1], direction)

  def prefill_left(
      self,
      clean_prefix_ids: torch.Tensor,
      cache: Optional[DirectionalCache] = None,
      *,
      detach: bool = False,
  ) -> DirectionalCache:
    result = self._prefill(clean_prefix_ids, cache, "left")
    return result.detach() if detach else result

  def _validate_boundary_request(
      self,
      token_ids: torch.Tensor,
      block_size: int,
  ) -> int:
    if token_ids.ndim != 2:
      raise ValueError("Clean context must have shape [batch, length]")
    length = token_ids.shape[1]
    if block_size <= 0 or length % block_size:
      raise ValueError(
        f"Length ({length}) must be a positive multiple of the block size "
        f"({block_size})")
    return length // block_size

  def _boundary_caches_sequential(
      self,
      token_ids: torch.Tensor,
      block_size: int,
      direction: str,
  ) -> tuple[DirectionalCache, ...]:
    """Reference block-major implementation, kept as the equivalence oracle.

    Entry ``i`` is the state produced by ``token_ids[:, :i * block_size]``, so
    entry 0 is the empty state. This walks the blocks in order, running every
    layer at each one; ``_boundary_caches`` computes the same states with two
    full-length scans per layer instead, and the tests assert they agree.
    """
    num_blocks = self._validate_boundary_request(token_ids, block_size)
    batch_size = token_ids.shape[0]

    # Match the dtype `_prefill` would pick for an empty cache (the embedding
    # output dtype, which autocast may differ from the weight dtype).
    cache = self._empty_cache(
      batch_size,
      token_ids.device,
      self.token_embedding(token_ids[:, :1]).dtype,
      direction)
    boundaries = [cache]
    for index in range(num_blocks - 1):
      start = index * block_size
      cache = self._prefill(
        token_ids[:, start:start + block_size], cache, direction)
      boundaries.append(cache)
    return tuple(boundaries)

  # Bytes of stored prefill activation per (prefix token x layer x hidden
  # unit), fitted to the single measured on/off pair in the tree:
  # logs/sizing_sweep_112593.out:8-9, BiSSM L=8192 batch 4 hidden 768
  # n_blocks 12, peak 70.14 GiB (off) - 45.07 GiB (on) = 25.07 GiB over
  # 4 * 7936 * 12 * 768 = 292,552,704 (token x layer x unit).
  # Cross-checked at a second point: the same coefficient predicts batch 8 to
  # need 50.1 GiB more than its 88.67 GiB checkpointed peak, i.e. 138.8 of
  # 139.72 GiB -- and that configuration did OOM (logs/...:10).
  _PREFILL_ACT_BYTES_PER_TOKEN_LAYER_UNIT = 92.0

  @classmethod
  def _prefill_activation_bytes(cls, batch, prefix_tokens, n_layers, hidden):
    return (cls._PREFILL_ACT_BYTES_PER_TOKEN_LAYER_UNIT
            * batch * prefix_tokens * n_layers * hidden)

  @classmethod
  def _resolve_prefill_checkpoint(cls, config, model):
    """Decide `checkpoint_boundary_prefill` once, at construction.

    Accepts `true`, `false`, or `auto`. `auto` turns recompute ON only when the
    activations it would save are a large enough share of the device that
    storing them plausibly threatens the peak; below that the recompute is a
    pure loss (it costs ~14% more arithmetic and a third of the step's operator
    dispatches, and buys memory that was never scarce).

    Checked against every on/off datapoint in the tree; the rule is correct or
    conservative at all four:

      L=2048  b2  ->   2.0% -> off  (measured peak 4.25 GiB of 139.72)
      L=8192  b4  ->  17.9% -> off  (measured: off fits at 70.14 GiB, and is
                                     12% faster than on)
      L=8192  b8  ->  35.9% -> on   (measured: off OOMs)
      L=32768 b2  ->  36.7% -> on   (conservative: off would have fit at an
                                     estimated ~97 GiB, so this leaves ~12%
                                     on the table at the longest geometry)
    """
    setting = model.get("checkpoint_boundary_prefill", "auto")
    if isinstance(setting, str):
      normalized = setting.strip().lower()
      if normalized not in {"auto", "true", "false"}:
        raise ValueError(
          "model.checkpoint_boundary_prefill must be true, false or 'auto', "
          f"got {setting!r}")
      if normalized != "auto":
        return normalized == "true"
    else:
      return bool(setting)

    total_bytes = getattr(cls, "_device_total_bytes", None)
    if total_bytes is None:
      if not torch.cuda.is_available():
        # No device to run out of: the CPU paths in the test suite and the
        # equivalence probes want the cheap, non-recomputing branch.
        return False
      total_bytes = torch.cuda.get_device_properties(
        torch.cuda.current_device()).total_memory

    block_size = int(config.block_size)
    length = int(config.model.length)
    num_blocks = max(length // block_size, 1)
    prefix_tokens = max((num_blocks - 1) * block_size, 0)
    batch = int(config.loader.batch_size)
    n_layers = int(config.model.n_blocks)
    hidden = int(config.model.hidden_size)
    share = cls._prefill_activation_bytes(
      batch, prefix_tokens, n_layers, hidden) / float(total_bytes)
    budget = float(model.get("checkpoint_prefill_budget_frac", 0.20))
    decision = share > budget
    print(f"[bissm] checkpoint_boundary_prefill=auto -> {decision} "
          f"(prefill activations ~{share:.1%} of device, budget {budget:.0%}; "
          f"L={length} batch={batch} layers={n_layers} hidden={hidden})",
          flush=True)
    return decision

  def _stacked_boundary_states(
      self,
      token_ids: torch.Tensor,
      block_size: int,
      num_blocks: int,
  ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-layer block-boundary states, one pass per layer.

    Returns ``(conv, ssm)`` lists holding one ``[batch, num_blocks, ...]``
    tensor per layer, where index ``i`` is the state entering block ``i``.

    Only tokens strictly before the last block can reach any of these states,
    so the last block is never scanned -- which is also what keeps a block's
    own clean target out of its denoiser's cache.
    """
    prefix = token_ids[:, :(num_blocks - 1) * block_size]
    x = self.token_embedding(prefix)
    conv_per_layer, ssm_per_layer = [], []
    # Only worth checkpointing when a backward pass will actually consume the
    # stored activations; under `no_grad`/`inference_mode` there is nothing to
    # trade away and `checkpoint` would just add a second forward.
    recompute = (self.checkpoint_boundary_prefill
                 and torch.is_grad_enabled()
                 and x.requires_grad)
    for layer in self.layers:
      if recompute:
        # `use_reentrant=False` is required here: the reentrant implementation
        # cannot return the two state tensors alongside `x`, and it mishandles
        # a layer whose output is consumed by more than one downstream branch,
        # which is exactly the shape of this loop.
        x, conv_states, ssm_states = torch.utils.checkpoint.checkpoint(
          layer.scan_clean_with_boundaries, x, block_size,
          use_reentrant=False)
      else:
        x, conv_states, ssm_states = layer.scan_clean_with_boundaries(
          x, block_size)
      conv_per_layer.append(conv_states)
      ssm_per_layer.append(ssm_states)
    return conv_per_layer, ssm_per_layer

  def _boundary_caches(
      self,
      token_ids: torch.Tensor,
      block_size: int,
      direction: str,
  ) -> tuple[DirectionalCache, ...]:
    """Scans a clean sequence once, keeping the state entering each block.

    Entry ``i`` is the state produced by ``token_ids[:, :i * block_size]``, so
    entry 0 is the empty state.

    The scan is layer-major: each layer consumes the whole prefix in two
    well-shaped calls rather than one short call per block. Layer ``l``'s
    boundary states depend only on layer ``l - 1``'s output over the prefix
    and on layer ``l``'s own recurrence, so nothing couples the layers within
    a block and the block loop is unnecessary.
    """
    if self.boundary_impl == "block_major":
      return self._boundary_caches_sequential(
        token_ids, block_size, direction)
    num_blocks = self._validate_boundary_request(token_ids, block_size)
    if num_blocks == 1:
      return (self._empty_cache(
        token_ids.shape[0],
        token_ids.device,
        self.token_embedding(token_ids[:, :1]).dtype,
        direction),)
    conv_per_layer, ssm_per_layer = self._stacked_boundary_states(
      token_ids, block_size, num_blocks)
    return tuple(
      DirectionalCache(
        tuple(
          Mamba2State(conv[:, index], ssm[:, index])
          for conv, ssm in zip(conv_per_layer, ssm_per_layer)),
        index * block_size,
        direction)
      for index in range(num_blocks))

  def _boundary_caches_stacked(
      self,
      token_ids: torch.Tensor,
      block_size: int,
      direction: str,
      reverse_blocks: bool = False,
  ) -> DirectionalCache:
    """``_boundary_caches`` folded straight into the batch dimension.

    Equivalent to ``stack_boundary_caches(self._boundary_caches(...))`` but
    without materialising ``num_blocks * n_layers`` intermediate slices; row
    ``b * num_blocks + i`` holds sequence ``b``'s state entering block ``i``.
    """
    if self.boundary_impl == "block_major":
      caches = self._boundary_caches_sequential(
        token_ids, block_size, direction)
      return stack_boundary_caches(
        tuple(reversed(caches)) if reverse_blocks else caches)
    num_blocks = self._validate_boundary_request(token_ids, block_size)
    if num_blocks == 1:
      return self._empty_cache(
        token_ids.shape[0],
        token_ids.device,
        self.token_embedding(token_ids[:, :1]).dtype,
        direction)
    conv_per_layer, ssm_per_layer = self._stacked_boundary_states(
      token_ids, block_size, num_blocks)
    states = []
    for conv, ssm in zip(conv_per_layer, ssm_per_layer):
      if reverse_blocks:
        conv, ssm = conv.flip(1), ssm.flip(1)
      states.append(
        Mamba2State(conv.flatten(0, 1), ssm.flatten(0, 1)))
    # The folded rows cover different context lengths, so a single scalar
    # length is meaningless here; -1 marks it as heterogeneous.
    return DirectionalCache(tuple(states), length=-1, direction=direction)

  def prefill_left_boundaries(
      self,
      clean_ids: torch.Tensor,
      block_size: int,
  ) -> tuple[DirectionalCache, ...]:
    """Left state entering every block: entry ``i`` scans ``[:, :i*block]``."""
    return self._boundary_caches(clean_ids, block_size, "left")

  def prefill_right_boundaries(
      self,
      clean_ids: torch.Tensor,
      block_size: int,
  ) -> tuple[DirectionalCache, ...]:
    """Right state entering every block from the far end.

    Entry ``i`` is the reverse scan of the clean suffix strictly after block
    ``i``; block ``num_blocks - 1`` therefore receives the empty state. The
    target block itself is never part of its own cache.
    """
    reversed_boundaries = self._boundary_caches(
      torch.flip(clean_ids, dims=(1,)), block_size, "right")
    # Reversed block ``j`` is original block ``num_blocks - 1 - j``, so the
    # state entering reversed block ``j`` is the original block's suffix state.
    return tuple(reversed(reversed_boundaries))

  def prefill_left_boundaries_stacked(
      self,
      clean_ids: torch.Tensor,
      block_size: int,
  ) -> DirectionalCache:
    """``prefill_left_boundaries`` already folded into the batch dimension."""
    return self._boundary_caches_stacked(clean_ids, block_size, "left")

  def prefill_right_boundaries_stacked(
      self,
      clean_ids: torch.Tensor,
      block_size: int,
  ) -> DirectionalCache:
    """``prefill_right_boundaries`` already folded into the batch dimension."""
    return self._boundary_caches_stacked(
      torch.flip(clean_ids, dims=(1,)),
      block_size,
      "right",
      reverse_blocks=True)

  def prefill_right(
      self,
      clean_suffix_ids: torch.Tensor,
      *,
      detach: bool = False,
  ) -> DirectionalCache:
    # Scan from the far-right token toward the gap boundary.
    reversed_suffix = torch.flip(clean_suffix_ids, dims=(1,))
    result = self._prefill(reversed_suffix, None, "right")
    return result.detach() if detach else result

  def forward_active(
      self,
      noisy_block_ids: torch.Tensor,
      sigma: Optional[torch.Tensor],
      left_cache: Optional[DirectionalCache] = None,
      right_cache: Optional[DirectionalCache] = None,
  ) -> torch.Tensor:
    if noisy_block_ids.ndim != 2:
      raise ValueError("Active block must have shape [batch, length]")
    batch_size = noisy_block_ids.shape[0]
    x = self.token_embedding(noisy_block_ids)
    if self.time_embedding is not None:
      if sigma is None:
        raise ValueError("sigma is required when time conditioning is enabled")
      if sigma.ndim != 1 or sigma.shape[0] != batch_size:
        raise ValueError("sigma must have shape [batch]")
      x = x + self.time_embedding(sigma)[:, None, :]

    if left_cache is None:
      left_cache = self._empty_cache(batch_size, x.device, x.dtype, "left")
    if right_cache is None:
      right_cache = self._empty_cache(batch_size, x.device, x.dtype, "right")
    self._validate_cache(left_cache, batch_size, "left")
    self._validate_cache(right_cache, batch_size, "right")

    with self._compute_autocast(x):
      for layer_index, layer in enumerate(self.layers):
        x = layer.scan_active(
          x,
          left_cache.states[layer_index],
          right_cache.states[layer_index])
      return self.output(self.final_norm(x))

  def prepare_right_cache(self, clean_suffix_ids: torch.Tensor):
    """Prepare the fixed C-a cache used by subsequent active-block calls."""
    self._sampling_right_cache = self.prefill_right(
      clean_suffix_ids, detach=True)

  def reset_kv_cache(self, eval_batch_size=None):
    """Compatibility hook used by the existing block sampler."""
    del eval_batch_size
    self._sampling_left_cache = None
    self._sampling_right_cache = None

  @property
  def sampling_cache_nbytes(self) -> int:
    return sum(
      cache.nbytes for cache in (
        self._sampling_left_cache, self._sampling_right_cache)
      if cache is not None)

  def forward(
      self,
      indices: torch.Tensor,
      sigma: Optional[torch.Tensor],
      sample_mode: bool = False,
      store_kv: bool = False,
  ) -> torch.Tensor:
    """Existing diffusion/sampler-compatible active-block entry point.

    Training calls ``forward_active`` explicitly so prefix and suffix ownership
    is unambiguous. During sampling, a fully denoised block is committed when
    ``store_kv=True``; repeated noisy calls never mutate the clean cache.
    """
    if not sample_mode:
      return self.forward_active(indices, sigma)

    logits = self.forward_active(
      indices,
      sigma,
      left_cache=self._sampling_left_cache,
      right_cache=self._sampling_right_cache)
    if store_kv:
      self._sampling_left_cache = self.prefill_left(
        indices, cache=self._sampling_left_cache, detach=True)
    return logits

