"""Causal Mamba-2 baseline for autoregressive and block-diffusion DNA LMs.

The backbone deliberately has one sequence direction only.  Under the BD3-LM
objective it scans a clean committed prefix into a recurrent cache and then
continues that scan through the noisy active block.  Under the AR objective it
is a standard next-nucleotide causal SSM with the same recurrent state.

Keeping both objectives in one backbone separates two questions that the
partial-bidirectionality model otherwise confounds:

* does a recurrent causal state help the DNA model; and
* does a reverse scan inside the active diffusion block add value?
"""

from __future__ import annotations

from typing import Optional

import torch

from .bidirectional_ssm import BidirectionalSSM, DirectionalCache


class UnidirectionalSSM(BidirectionalSSM):
  """Single-direction counterpart of :class:`BidirectionalSSM`.

  The module and parameter names intentionally match ``BidirectionalSSM``.
  The bidirectional implementation shares its Mamba weights across directions,
  so this gives an exactly parameter-matched baseline and also permits a trained
  BiSSM checkpoint to be evaluated with its reverse active-block path disabled.
  """

  def __init__(self, config, vocab_size: int):
    super().__init__(config, vocab_size)
    self.parameterization = str(config.algo.parameterization)
    right_probability = float(
      config.model.get("right_flank_probability", 0.0))
    if right_probability != 0.0:
      raise ValueError(
        "UnidirectionalSSM has no right-flank path; set "
        "model.right_flank_probability=0.0")

  def _scan_active(
      self,
      noisy_ids: torch.Tensor,
      sigma: Optional[torch.Tensor],
      left_cache: Optional[DirectionalCache],
  ) -> tuple[torch.Tensor, DirectionalCache]:
    if noisy_ids.ndim != 2:
      raise ValueError("Active sequence must have shape [batch, length]")
    batch_size = noisy_ids.shape[0]
    x = self.token_embedding(noisy_ids)
    if self.time_embedding is not None:
      if sigma is None:
        raise ValueError("sigma is required when time conditioning is enabled")
      if sigma.ndim != 1 or sigma.shape[0] != batch_size:
        raise ValueError("sigma must have shape [batch]")
      x = x + self.time_embedding(sigma)[:, None, :]

    if left_cache is None:
      left_cache = self._empty_cache(
        batch_size, x.device, x.dtype, "left")
    self._validate_cache(left_cache, batch_size, "left")

    # bf16 for the layer stack, mirroring models/dit.py. Without this the AR
    # objective ran this backbone entirely in FP32 -- see `_compute_autocast`.
    with self._compute_autocast(x):
      final_states = []
      for layer_index, layer in enumerate(self.layers):
        x, state = layer.scan_clean(x, left_cache.states[layer_index])
        final_states.append(state)
      logits = self.output(self.final_norm(x))
    final_length = (
      left_cache.length + noisy_ids.shape[1]
      if left_cache.length >= 0 else -1)
    final_cache = DirectionalCache(
      tuple(final_states), final_length, "left")
    return logits, final_cache

  def forward_active(
      self,
      noisy_block_ids: torch.Tensor,
      sigma: Optional[torch.Tensor],
      left_cache: Optional[DirectionalCache] = None,
      right_cache: Optional[DirectionalCache] = None,
  ) -> torch.Tensor:
    if right_cache is not None:
      raise ValueError(
        "UnidirectionalSSM cannot consume a right-direction cache")
    logits, _ = self._scan_active(noisy_block_ids, sigma, left_cache)
    return logits

  def forward(
      self,
      indices: torch.Tensor,
      sigma: Optional[torch.Tensor],
      sample_mode: bool = False,
      store_kv: bool = False,
  ) -> torch.Tensor:
    """Run a full causal sequence or continue the immutable sampling cache.

    The repository's AR sampler supplies the growing token prefix on every
    call.  Once a recurrent cache exists, only its final token is new.  The
    BD3-LM sampler instead supplies exactly one active block, which is committed
    only after it has been fully denoised.
    """
    use_cache = sample_mode or store_kv
    active = indices
    left_cache = None
    if use_cache:
      left_cache = self._sampling_left_cache
      if self.parameterization == "ar" and left_cache is not None:
        active = indices[:, -1:]

    logits, final_cache = self._scan_active(active, sigma, left_cache)
    if store_kv:
      self._sampling_left_cache = final_cache.detach()
    return logits
