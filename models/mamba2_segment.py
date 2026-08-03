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
    history = torch.cat((initial_state, raw), dim=-1)
    convolved = F.conv1d(
      history,
      self.conv1d.weight,
      self.conv1d.bias,
      groups=self.conv_dim)
    convolved = convolved[:, :, 1:].transpose(1, 2)
    return F.silu(convolved), history[:, :, -self.d_conv:]

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

    if self._select_backend(u) == "fused":
      y, final_ssm = self._fused_scan(x, dt, B, C, initial_state.ssm)
    else:
      y, final_ssm = self._reference_scan(x, dt, B, C, initial_state.ssm)

    y = rearrange(y, "b l h p -> b l (h p)")
    # Official Mamba-2 default: RMSNorm after applying the SiLU gate.
    y = y * F.silu(z)
    variance = y.float().pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(variance.to(dtype=y.dtype) + 1e-5)
    y = y * self.norm_weight.to(dtype=y.dtype)
    output = self.out_proj(y)
    return output, Mamba2State(final_conv, final_ssm)

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    output, _ = self.scan_segment(u)
    return output
