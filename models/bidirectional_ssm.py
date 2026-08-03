"""Leakage-safe bidirectional Mamba backbone for block diffusion.

There are two valid directions of context:

* de novo: a clean, timestep-free left cache plus a reverse scan confined to
  the current noisy block;
* C-a infilling: the same left cache plus a clean, timestep-free right cache
  that initializes the active block's reverse scan.

The clean target block is never used to construct either cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2_segment import Mamba2State, SegmentMamba2


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
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
    return self.mlp(embedding)


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
  ):
    super().__init__()
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
      backend=backend)
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

  def scan_active(
      self,
      x: torch.Tensor,
      left_state: Mamba2State,
      right_state: Mamba2State,
  ) -> torch.Tensor:
    normalized = self.mixer_norm(x)
    forward, _ = self.mixer.scan_segment(normalized, left_state)
    reverse, _ = self.mixer.scan_segment(
      torch.flip(normalized, dims=(1,)), right_state)
    reverse = torch.flip(reverse, dims=(1,))
    # The paper uses a direct sum: no zero-initialized route gate that can
    # silently disable either context direction.
    x = x + self.dropout(forward + reverse)
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
        backend=str(model.get("ssm_backend", "auto")))
      for _ in range(int(model.n_blocks))
    ])
    self.final_norm = RMSNorm(self.hidden_size)
    self.output = nn.Linear(self.hidden_size, vocab_size, bias=False)
    if bool(model.get("tie_word_embeddings", True)):
      self.output.weight = self.token_embedding.weight

    self._sampling_left_cache: Optional[DirectionalCache] = None
    self._sampling_right_cache: Optional[DirectionalCache] = None
    self._initialize_weights()

  def _initialize_weights(self):
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    for module in self.modules():
      if isinstance(module, nn.Linear) and module is not self.output:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

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

  def _boundary_caches(
      self,
      token_ids: torch.Tensor,
      block_size: int,
      direction: str,
  ) -> tuple[DirectionalCache, ...]:
    """Scans a clean sequence once, keeping the state entering each block.

    Entry ``i`` is the state produced by ``token_ids[:, :i * block_size]``, so
    entry 0 is the empty state. Total work equals one full-length scan: the
    blocks are consumed in order and each one continues the previous state.
    """
    if token_ids.ndim != 2:
      raise ValueError("Clean context must have shape [batch, length]")
    batch_size, length = token_ids.shape
    if block_size <= 0 or length % block_size:
      raise ValueError(
        f"Length ({length}) must be a positive multiple of the block size "
        f"({block_size})")
    num_blocks = length // block_size

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

