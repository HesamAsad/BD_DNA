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

