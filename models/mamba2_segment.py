"""Segment-continuable Mamba-2 mixer.

The parameterization and initialization in this module are adapted from the
Apache-2.0 `state-spaces/mamba` Mamba2 implementation.  The important local
addition is an explicit, differentiable ``scan_segment`` interface:

    output, final_state = mixer.scan_segment(input, initial_state)

Unlike the upstream token-at-a-time inference cache, this interface can start
a whole diffusion block from an immutable prefix/suffix boundary state.  CUDA
runs use the upstream fused SSD chunk scan when it is installed; the small
PyTorch reference recurrence keeps cache semantics testable on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
  from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except (ImportError, OSError, RuntimeError):
  # Triton initializes its device driver while importing mamba-ssm 2.2.x.
  # Head/login nodes have no active GPU driver, so they intentionally fall
  # back to the reference scan even when the package is installed.
  mamba_chunk_scan_combined = None


@dataclass(frozen=True)
class Mamba2State:
  """Boundary state required to continue a Mamba-2 scan exactly."""

  conv: torch.Tensor
  ssm: torch.Tensor

  def clone(self) -> "Mamba2State":
    return Mamba2State(self.conv.clone(), self.ssm.clone())

  def detach(self) -> "Mamba2State":
    return Mamba2State(self.conv.detach(), self.ssm.detach())


def fused_mamba2_available() -> bool:
  return mamba_chunk_scan_combined is not None


class SegmentMamba2(nn.Module):
  """Mamba-2 mixer with explicit segment boundary input/output states.

  Args mirror the official Mamba-2 defaults used by the attached partial-
  bidirectionality paper. ``backend='auto'`` selects the fused SSD scan on
  CUDA and the readable reference scan elsewhere.
  """

  def __init__(
      self,
      d_model: int,
      d_state: int = 64,
      d_conv: int = 4,
      expand: int = 2,
      headdim: int = 64,
      chunk_size: int = 128,
      backend: str = "auto",
      bias: bool = False,
      conv_bias: bool = True,
      dt_min: float = 1e-3,
      dt_max: float = 1e-1,
      dt_init_floor: float = 1e-4,
      A_init_range: tuple[float, float] = (1.0, 16.0),
  ):
    super().__init__()
    if backend not in {"auto", "fused", "torch"}:
      raise ValueError(f"Unknown Mamba-2 backend: {backend}")
    if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
      raise ValueError("Mamba-2 dimensions must be positive")

    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = expand * d_model
    self.headdim = headdim
    if self.d_inner % headdim:
      raise ValueError(
        f"expand*d_model ({self.d_inner}) must divide headdim ({headdim})")
    self.nheads = self.d_inner // headdim
    self.chunk_size = chunk_size
    self.backend = backend
    self.conv_dim = self.d_inner + 2 * d_state

    # Official Mamba-2 order: [z, x, B, C, dt].
    projection_dim = 2 * self.d_inner + 2 * d_state + self.nheads
    self.in_proj = nn.Linear(d_model, projection_dim, bias=bias)
    self.conv1d = nn.Conv1d(
      self.conv_dim,
      self.conv_dim,
      kernel_size=d_conv,
      groups=self.conv_dim,
      padding=0,
      bias=conv_bias)

    dt = torch.exp(
      torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
      + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    self.dt_bias = nn.Parameter(inv_dt)
    self.dt_bias._no_weight_decay = True

    if A_init_range[0] <= 0 or A_init_range[1] < A_init_range[0]:
      raise ValueError(f"Invalid A initialization range: {A_init_range}")
    A = torch.empty(self.nheads).uniform_(*A_init_range)
    self.A_log = nn.Parameter(torch.log(A))
    self.A_log._no_weight_decay = True

    self.D = nn.Parameter(torch.ones(self.nheads))
    self.D._no_weight_decay = True
    self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
    self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

  def zero_state(
      self,
      batch_size: int,
      *,
      device: torch.device,
      dtype: torch.dtype,
  ) -> Mamba2State:
    return Mamba2State(
      conv=torch.zeros(
        batch_size, self.conv_dim, self.d_conv,
        device=device, dtype=dtype),
      ssm=torch.zeros(
        batch_size, self.nheads, self.headdim, self.d_state,
        device=device, dtype=dtype))

  def _select_backend(self, x: torch.Tensor) -> str:
    if self.backend == "torch":
      return "torch"
    if self.backend == "fused":
      if mamba_chunk_scan_combined is None:
        raise RuntimeError(
          "backend='fused' requires mamba-ssm. Install requirements.txt "
          "with --no-build-isolation after installing PyTorch.")
      if not x.is_cuda:
        raise RuntimeError("The fused Mamba-2 scan requires a CUDA tensor")
      return "fused"
    if x.is_cuda and mamba_chunk_scan_combined is not None:
      return "fused"
    return "torch"

  def _causal_conv(
      self,
      xBC: torch.Tensor,
      initial_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution continued from an immutable state.

    The cache stores the last ``d_conv`` raw projected inputs, matching the
    state layout in official Mamba-2. The first convolution result belongs to
    the old cache boundary and is dropped; the remaining results correspond
    one-for-one with the new segment.
    """
    raw = xBC.transpose(1, 2)
    history = torch.cat((initial_state.to(dtype=raw.dtype), raw), dim=-1)
    convolved = F.conv1d(
      history,
      self.conv1d.weight,
      self.conv1d.bias,
      groups=self.conv_dim)
    # `narrow` rather than fancy slicing: same values, but its backward is a
    # view-scatter instead of a full-tensor zeros+copy.
    convolved = convolved.narrow(-1, 1, raw.shape[-1]).transpose(1, 2)
    return F.silu(convolved), history.narrow(
      -1, history.shape[-1] - self.d_conv, self.d_conv)

  def _reference_scan(
      self,
      x: torch.Tensor,
      dt: torch.Tensor,
      B: torch.Tensor,
      C: torch.Tensor,
      initial_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Readable Mamba-2 diagonal selective recurrence from upstream ``step``."""
    A = -torch.exp(self.A_log.float())
    dt = F.softplus(dt.float() + self.dt_bias.float())
    state = initial_state
    outputs = []
    for position in range(x.shape[1]):
      dt_i = dt[:, position]
      x_i = x[:, position]
      B_i = B[:, position, 0]
      C_i = C[:, position, 0]
      dA = torch.exp(dt_i * A).to(dtype=x.dtype)
      dBx = torch.einsum(
        "bh,bn,bhp->bhpn",
        dt_i.to(dtype=x.dtype), B_i, x_i)
      state = state * dA[:, :, None, None] + dBx
      y_i = torch.einsum("bhpn,bn->bhp", state, C_i)
      y_i = y_i + self.D.to(dtype=x.dtype)[None, :, None] * x_i
      outputs.append(y_i)
    return torch.stack(outputs, dim=1), state

  def _fused_scan(
      self,
      x: torch.Tensor,
      dt: torch.Tensor,
      B: torch.Tensor,
      C: torch.Tensor,
      initial_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    result = mamba_chunk_scan_combined(
      x,
      dt,
      -torch.exp(self.A_log.float()),
      B,
      C,
      chunk_size=self.chunk_size,
      D=self.D,
      dt_bias=self.dt_bias,
      initial_states=initial_state,
      dt_softplus=True,
      return_final_states=True)
    return result

  def _zero_ssm(
      self,
      batch_size: int,
      device: torch.device,
      dtype: torch.dtype,
  ) -> torch.Tensor:
    return torch.zeros(
      batch_size, self.nheads, self.headdim, self.d_state,
      device=device, dtype=dtype)

  def _scan(
      self,
      x: torch.Tensor,
      dt: torch.Tensor,
      B: torch.Tensor,
      C: torch.Tensor,
      initial_ssm: Optional[torch.Tensor],
      backend: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch one selective scan, treating ``None`` as the zero state."""
    if backend == "fused":
      return self._fused_scan(x, dt, B, C, initial_ssm)
    if initial_ssm is None:
      initial_ssm = self._zero_ssm(x.shape[0], x.device, x.dtype)
    return self._reference_scan(x, dt, B, C, initial_ssm)

  def _gated_output(
      self,
      y: torch.Tensor,
      z: torch.Tensor,
  ) -> torch.Tensor:
    """Official Mamba-2 tail: SiLU gate, RMSNorm, output projection."""
    y = rearrange(y, "b l h p -> b l (h p)")
    y = y * F.silu(z)
    variance = y.float().pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(variance.to(dtype=y.dtype) + 1e-5)
    y = y * self.norm_weight.to(dtype=y.dtype)
    return self.out_proj(y)

  def _block_state_passing(
      self,
      dt: torch.Tensor,
      local_ssm: torch.Tensor,
      block_size: int,
  ) -> torch.Tensor:
    """Combine per-block local final states into per-block entry states.

    This is the standard SSD inter-chunk recurrence
    ``S[i] = decay[i-1] * S[i-1] + local[i-1]`` applied one level up, at block
    rather than chunk granularity.  Unrolling it into a single masked matmul
    keeps the whole cross-block carry in one kernel instead of a per-block
    Python loop.

    ``decay`` is a difference of prefix sums of ``dt``, and ``A`` is negative,
    so every exponent the mask keeps is non-positive; the clamp only guards
    the masked-out upper triangle, whose gradient the mask discards anyway.
    """
    batch_size, seqlen, _ = dt.shape
    num_seg = seqlen // block_size
    # Autocast would demote the matmul to bf16 and quantize the caches that
    # every block's denoiser is conditioned on; the recurrence is cheap enough
    # to always run in fp32.
    with torch.autocast(device_type=dt.device.type, enabled=False):
      A = -torch.exp(self.A_log.float())
      dt_eff = F.softplus(dt.float() + self.dt_bias.float())
      segment_dt = dt_eff.reshape(
        batch_size, num_seg, block_size, self.nheads).sum(dim=2)
      # cumulative[i] sums dt over the blocks strictly before block i.
      cumulative = F.pad(
        torch.cumsum(segment_dt, dim=1), (0, 0, 1, 0))
      # exponent[b, i, j, h] carries block j's final state to block i's entry.
      exponent = A * (
        cumulative[:, :, None, :] - cumulative[:, None, 1:, :])
      keep = (
        torch.arange(num_seg, device=dt.device)[None, :]
        < torch.arange(num_seg + 1, device=dt.device)[:, None])
      decay = torch.where(
        keep[None, :, :, None],
        torch.exp(exponent.clamp(max=0.0)),
        exponent.new_zeros(()))
      return torch.einsum(
        "bijh,bjhpn->bihpn", decay, local_ssm.float())

  def scan_with_block_boundaries(
      self,
      u: torch.Tensor,
      block_size: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan ``u`` from the zero state, keeping every block-boundary state.

    ``u`` must hold a whole number of blocks.  The returned state tensors have
    a leading block axis of length ``num_seg + 1``: entry ``i`` is the state a
    scan has reached after consuming ``u[:, :i * block_size]``, so entry 0 is
    the zero state and entry ``num_seg`` is the final state.

    This replaces a ``num_seg``-iteration loop of short ``scan_segment`` calls
    with two well-shaped scans -- one full-length pass for the outputs, one
    folded pass whose per-block final states the inter-block recurrence
    carries forward.  It computes the same function of the input; only the
    association order of the floating-point sums differs.
    """
    if u.ndim != 3 or u.shape[-1] != self.d_model:
      raise ValueError(
        f"Expected [batch, length, {self.d_model}], received {tuple(u.shape)}")
    batch_size, seqlen, _ = u.shape
    if block_size <= 0 or seqlen % block_size:
      raise ValueError(
        f"Length ({seqlen}) must be a positive multiple of the block size "
        f"({block_size})")
    num_seg = seqlen // block_size

    zxbcdt = self.in_proj(u)
    z, xBC, dt = torch.split(
      zxbcdt,
      [self.d_inner, self.conv_dim, self.nheads],
      dim=-1)

    # One full-length causal convolution from a zero state.  `history` retains
    # every raw projected input, so a block's convolution boundary state is a
    # strided window of it rather than a separate per-block convolution.
    raw = xBC.transpose(1, 2)
    history = torch.cat(
      (torch.zeros(
        batch_size, self.conv_dim, self.d_conv,
        device=raw.device, dtype=raw.dtype), raw), dim=-1)
    convolved = F.conv1d(
      history,
      self.conv1d.weight,
      self.conv1d.bias,
      groups=self.conv_dim)
    xBC = F.silu(convolved.narrow(-1, 1, seqlen).transpose(1, 2))
    # Window i spans history[..., i * block : i * block + d_conv], i.e. exactly
    # the d_conv raw inputs preceding token i * block; window 0 is the zero
    # state and window num_seg is the final state.
    conv_states = history.unfold(-1, self.d_conv, block_size).permute(
      0, 2, 1, 3).contiguous()

    x, B, C = torch.split(
      xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
    B = rearrange(B, "b l n -> b l 1 n")
    C = rearrange(C, "b l n -> b l 1 n")

    backend = self._select_backend(u)
    # (1) The true outputs, from one contiguous scan over the whole prefix.
    y, _ = self._scan(x, dt, B, C, None, backend)
    # (2) Every block's *local* final state, from one folded batched scan.
    def fold(tensor):
      return tensor.reshape(
        batch_size * num_seg, block_size, *tensor.shape[2:])
    _, local_ssm = self._scan(
      fold(x), fold(dt), fold(B), fold(C), None, backend)
    local_ssm = local_ssm.reshape(
      batch_size, num_seg, self.nheads, self.headdim, self.d_state)
    ssm_states = self._block_state_passing(dt, local_ssm, block_size)
    return self._gated_output(y, z), conv_states, ssm_states

  def scan_segment(
      self,
      u: torch.Tensor,
      initial_state: Optional[Mamba2State] = None,
  ) -> tuple[torch.Tensor, Mamba2State]:
    """Scan ``u`` from ``initial_state`` without mutating the input state."""
    if u.ndim != 3 or u.shape[-1] != self.d_model:
      raise ValueError(
        f"Expected [batch, length, {self.d_model}], received {tuple(u.shape)}")
    batch_size, seqlen, _ = u.shape
    if initial_state is None:
      initial_state = self.zero_state(
        batch_size, device=u.device, dtype=u.dtype)
    if initial_state.conv.shape != (
        batch_size, self.conv_dim, self.d_conv):
      raise ValueError(
        f"Invalid convolution state shape: {tuple(initial_state.conv.shape)}")
    expected_ssm = (batch_size, self.nheads, self.headdim, self.d_state)
    if initial_state.ssm.shape != expected_ssm:
      raise ValueError(f"Invalid SSM state shape: {tuple(initial_state.ssm.shape)}")
    if seqlen == 0:
      return u, initial_state

    zxbcdt = self.in_proj(u)
    z, xBC, dt = torch.split(
      zxbcdt,
      [self.d_inner, self.conv_dim, self.nheads],
      dim=-1)
    xBC, final_conv = self._causal_conv(xBC, initial_state.conv)
    x, B, C = torch.split(
      xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
    B = rearrange(B, "b l n -> b l 1 n")
    C = rearrange(C, "b l n -> b l 1 n")

    y, final_ssm = self._scan(
      x, dt, B, C, initial_state.ssm, self._select_backend(u))
    return self._gated_output(y, z), Mamba2State(final_conv, final_ssm)

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    output, _ = self.scan_segment(u)
    return output
