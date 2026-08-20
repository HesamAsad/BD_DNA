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

try:
  from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except (ImportError, OSError, RuntimeError):
  rmsnorm_fn = None


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
      return_state: bool = True,
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Depthwise causal convolution continued from an immutable state.

    The cache stores the last ``d_conv`` raw projected inputs, matching the
    state layout in official Mamba-2.

    The obvious spelling, ``conv1d(cat(initial_state, raw))``, costs a whole
    extra activation: the concatenation is a fresh contiguous copy of the
    projected stream, and ``ConvolutionBackward0`` retains it. Measured at
    batch 4 / L=8192 that is 104.1 MiB per layer on top of the 201.5 MiB
    ``in_proj`` output the graph already holds -- the fused gated RMSNorm keeps
    ``z``, and ``z``, ``xBC`` and ``dt`` are views of one storage, so a
    convolution reading a *view* of that storage retains nothing new.

    So the body is a zero-padded convolution straight off the view. Only the
    first ``d_conv - 1`` outputs are then wrong (they are missing the boundary
    state), and those are recomputed by a second convolution over a
    ``2 * d_conv - 1`` column slice and written back in place. Verified
    bit-identical to the concatenated form in fp64 on CPU and fp32 on CUDA,
    gradients included, and measured at -104 MiB per layer: -1.22 GiB for
    uSSM-AR and -3.56 GiB for BiSSM-BD at batch 4 / L=8192.
    """
    raw = xBC.transpose(1, 2)
    seqlen = raw.shape[-1]
    weight, bias = self.conv1d.weight, self.conv1d.bias
    state = initial_state.to(dtype=raw.dtype)
    if seqlen < self.d_conv:
      # Too short to hold a whole window: the boundary state and the segment
      # both feed the returned cache, so keep the concatenated form.
      history = torch.cat((state, raw), dim=-1)
      convolved = F.conv1d(history, weight, bias, groups=self.conv_dim)
      # `narrow` rather than fancy slicing: same values, but its backward is a
      # view-scatter instead of a full-tensor zeros+copy.
      convolved = convolved.narrow(-1, 1, seqlen).transpose(1, 2)
      if not return_state:
        return F.silu(convolved), None
      return F.silu(convolved), history.narrow(
        -1, history.shape[-1] - self.d_conv, self.d_conv).contiguous()

    pad = self.d_conv - 1
    # Output j spans raw[j - pad .. j], zero-filled where the boundary state
    # belongs, so every output from `pad` on is already final.
    full = F.conv1d(raw, weight, bias, groups=self.conv_dim, padding=pad)
    head = F.conv1d(
      torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
      weight, bias, groups=self.conv_dim).narrow(-1, 1, pad)
    # A convolution's backward never reads its own output, so overwriting the
    # `pad` columns the zero padding got wrong only inserts a CopySlices node.
    # Writing in place rather than re-concatenating avoids a second full-width
    # copy that measured 0.26 ms per layer at batch 4 / L=8192.
    full.narrow(-1, 0, pad).copy_(head)
    convolved = F.silu(full.narrow(-1, 0, seqlen).transpose(1, 2))
    if not return_state:
      return convolved, None
    # `.contiguous()`: the returned state must not be a view of a full-length
    # tensor, or holding the cache pins the whole segment's projections --
    # 104 MiB per layer that a no-grad prefill would otherwise free.
    return (convolved,
            raw.narrow(-1, seqlen - self.d_conv, self.d_conv).contiguous())

  def _reverse_causal_conv(
      self,
      xBC: torch.Tensor,
      initial_state: torch.Tensor,
  ) -> torch.Tensor:
    """The reverse direction's causal conv, read off the *unflipped* stream.

    The reverse scan consumes ``flip(u)``, and ``in_proj`` is per-position, so
    its projection is exactly ``flip(in_proj(u))``. The convolution, however,
    is genuinely direction-dependent: forward output ``j`` spans
    ``raw[j-3 .. j]`` while the reverse direction's output at the same original
    position spans ``raw[j .. j+3]``. What makes them shareable anyway is that
    an anti-causal convolution is a causal convolution with a reversed kernel,

        conv(flip(raw), w)[t] = conv(raw, flip(w))[L - 1 - t]

    for a depthwise kernel. So this reads the same ``raw`` view the forward
    direction reads -- no flipped copy of the 3224-wide projection -- runs
    ``flip(weight)`` over it, and flips only the 1664-wide convolution output,
    which the caller has to materialise in reverse order for the scan anyway.

    The boundary is the mirror image of ``_causal_conv``'s: zero padding gets
    the last ``d_conv - 1`` outputs wrong instead of the first, and the right
    cache supplies the missing columns. That cache stores the flipped stream's
    trailing window, i.e. ``[raw[L+3], raw[L+2], raw[L+1], raw[L]]``, so
    ``flip(state)`` is literally the continuation of ``raw`` past the block.
    """
    raw = xBC.transpose(1, 2)
    seqlen = raw.shape[-1]
    weight = self.conv1d.weight.flip(-1)
    bias = self.conv1d.bias
    state = initial_state.to(dtype=raw.dtype)
    pad = self.d_conv - 1
    if seqlen < self.d_conv:
      # Too short for the padded form to have any already-final column; fall
      # back to the explicit flipped concatenation.
      history = torch.cat((state, torch.flip(raw, dims=(-1,))), dim=-1)
      convolved = F.conv1d(
        history, self.conv1d.weight, bias, groups=self.conv_dim)
      return F.silu(convolved.narrow(-1, 1, seqlen).transpose(1, 2))

    # Column `s + pad` of `full` is the reverse-direction output at original
    # position `s`; the last `pad` of them are missing the boundary state.
    full = F.conv1d(raw, weight, bias, groups=self.conv_dim, padding=pad)
    tail = F.conv1d(
      torch.cat((raw.narrow(-1, seqlen - pad, pad), state.flip(-1)), dim=-1),
      weight, bias, groups=self.conv_dim).narrow(-1, 0, pad)
    full.narrow(-1, seqlen, pad).copy_(tail)
    # Flip *before* the SiLU: `SiluBackward` saves its input, so whichever
    # tensor it reads stays resident. Feeding it the flipped copy lets the
    # (equally sized) convolution output be freed instead of both being held.
    return F.silu(
      torch.flip(full.narrow(-1, pad, seqlen).transpose(1, 2), dims=(1,)))

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
      return_final_state: bool = True,
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
      return_final_states=return_final_state)
    return result if return_final_state else (result, None)

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
      return_final_state: bool = True,
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch one selective scan, treating ``None`` as the zero state."""
    if backend == "fused":
      return self._fused_scan(
        x, dt, B, C, initial_ssm, return_final_state=return_final_state)
    if initial_ssm is None:
      initial_ssm = self._zero_ssm(x.shape[0], x.device, x.dtype)
    return self._reference_scan(x, dt, B, C, initial_ssm)

  def _gated_output(
      self,
      y: torch.Tensor,
      z: torch.Tensor,
  ) -> torch.Tensor:
    """Official Mamba-2 tail: SiLU gate, RMSNorm, output projection.

    The eager body below is correct but expensive to differentiate: autograd
    retains five intermediates per layer, two of them fp32 copies forced by
    `y.float().pow(2)`. Measured at batch 4 / L=8192 that is 672 MiB per layer
    where the fused kernel needs 96 -- it saves one tensor and recomputes the
    rest in its backward.

    `mamba_ssm`'s `rmsnorm_fn` is the same function: it applies SiLU to `z`
    internally, and `norm_before_gate=False` selects norm(x * silu(z)), which
    is exactly the order used here (layernorm_gated.py:26-27, :343). Verified
    numerically -- in fp32 the outputs agree to 9.5e-7 against a mean magnitude
    of 0.46, and the gradients w.r.t. y, z, norm_weight and out_proj agree to
    1e-10.

    Switching to it took the 12-layer stack from 29.221 to 22.471 GiB and from
    194,261 to 235,016 tokens/s -- less memory *and* more speed, so there is no
    trade to weigh. The eager path is kept for CPU and for the `torch` backend
    used by the unit tests, where the fused kernel is unavailable.
    """
    return self.out_proj(self._gated_norm(y, z))

  def _gated_norm(
      self,
      y: torch.Tensor,
      z: torch.Tensor,
  ) -> torch.Tensor:
    """``_gated_output`` without the output projection.

    Split out so a bidirectional layer can sum the two directions *before*
    projecting: ``out_proj`` is linear, so ``out_proj(a) + out_proj(b)`` and
    ``out_proj(a + b)`` compute the same function, but the second retains one
    ``[batch, length, d_inner]`` input for backward instead of two and runs one
    GEMM instead of two.
    """
    y = rearrange(y, "b l h p -> b l (h p)")
    if rmsnorm_fn is not None and y.is_cuda:
      return rmsnorm_fn(y, self.norm_weight, None, z=z, eps=1e-5,
                        norm_before_gate=False)
    y = y * F.silu(z)
    variance = y.float().pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(variance.to(dtype=y.dtype) + 1e-5)
    return y * self.norm_weight.to(dtype=y.dtype)

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

    # One full-length causal convolution from a zero state.  The boundary here
    # *is* zero, so a zero-padded convolution over a view of the `in_proj`
    # output is already exact -- no concatenated copy for the graph to retain
    # (see `_causal_conv`; same 104 MiB per layer).
    raw = xBC.transpose(1, 2)
    convolved = F.conv1d(
      raw,
      self.conv1d.weight,
      self.conv1d.bias,
      groups=self.conv_dim,
      padding=self.d_conv - 1)
    xBC = F.silu(convolved.narrow(-1, 0, seqlen).transpose(1, 2))
    # Window i holds the d_conv raw inputs preceding token i * block, so
    # window 0 is the zero state and window num_seg is the final state.  For
    # i >= 1 that is raw[..., i * block - d_conv : i * block]; block 0's window
    # is the only one that needs the zeros spelled out.
    zero_window = raw.new_zeros(batch_size, 1, self.conv_dim, self.d_conv)
    if block_size >= self.d_conv:
      windows = raw.narrow(
        -1, block_size - self.d_conv, seqlen - block_size + self.d_conv
      ).unfold(-1, self.d_conv, block_size).permute(0, 2, 1, 3)
      conv_states = torch.cat((zero_window, windows), dim=1)
    else:
      # Blocks shorter than the kernel make consecutive windows overlap into
      # the zero prefix; materialize it rather than special-case the strides.
      history = torch.cat((zero_window.squeeze(1), raw), dim=-1)
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

  def scan_bidirectional(
      self,
      u: torch.Tensor,
      left_state: Mamba2State,
      right_state: Mamba2State,
  ) -> torch.Tensor:
    """Forward + reverse scan of ``u`` sharing one ``in_proj`` and ``out_proj``.

    Equivalent to

        scan_segment(u, left)[0]
        + flip(scan_segment(flip(u), right)[0], 1)

    which is what ``BiMambaLayer.scan_active`` used to spell out. That form
    runs the whole mixer twice, and three of its four stages do not need to be:

    * ``in_proj`` is per-position, so the reverse direction's projection is a
      flip of the forward direction's. Computing it twice costs a second
      ``d_model x (2*d_inner + 2*d_state + nheads)`` GEMM *and* retains a
      second full-width projection for backward.
    * the flip of the layer input is itself an extra retained tensor, because
      ``in_proj`` saves whatever tensor it was handed.
    * ``out_proj`` is linear, so the two directions can be summed before it
      rather than after.

    Only the convolution and the scan are genuinely direction-dependent, and
    ``_reverse_causal_conv`` shows the convolution can still read the shared
    projection. The scan cannot: the SSD kernel is causal in memory order, so
    the reverse direction's operands must be materialised reversed.

    The final boundary states are deliberately not computed. Nothing consumes
    them -- an active block never continues a scan -- and asking the kernel for
    them also makes backward carry a ``dfinal_states`` term.
    """
    if self.out_proj.bias is not None:
      # This routine sums the two directions BEFORE `out_proj`, where the
      # two-call spelling sums after. Those agree only for a linear map: with a
      # bias the split form adds it twice and this form once, so the two would
      # silently disagree by exactly one bias term. `bias=False` is the
      # constructor default and every equivalence test uses it, so nothing in
      # the suite would catch that; fail loudly instead.
      raise NotImplementedError(
        "scan_bidirectional requires out_proj without bias; use "
        "model.bidirectional_impl='split' if a bias is needed")
    if u.ndim != 3 or u.shape[-1] != self.d_model:
      raise ValueError(
        f"Expected [batch, length, {self.d_model}], received {tuple(u.shape)}")
    batch_size, seqlen, _ = u.shape
    expected_conv = (batch_size, self.conv_dim, self.d_conv)
    expected_ssm = (batch_size, self.nheads, self.headdim, self.d_state)
    for name, state in (("left", left_state), ("right", right_state)):
      if tuple(state.conv.shape) != expected_conv:
        raise ValueError(
          f"Invalid {name} convolution state shape: {tuple(state.conv.shape)}")
      if tuple(state.ssm.shape) != expected_ssm:
        raise ValueError(
          f"Invalid {name} SSM state shape: {tuple(state.ssm.shape)}")
    if seqlen == 0:
      return u

    zxbcdt = self.in_proj(u)
    z, xBC, dt = torch.split(
      zxbcdt,
      [self.d_inner, self.conv_dim, self.nheads],
      dim=-1)
    backend = self._select_backend(u)

    def heads(convolved):
      x, B, C = torch.split(
        convolved, [self.d_inner, self.d_state, self.d_state], dim=-1)
      return (rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
              rearrange(B, "b l n -> b l 1 n"),
              rearrange(C, "b l n -> b l 1 n"))

    forward_xBC, _ = self._causal_conv(
      xBC, left_state.conv, return_state=False)
    x, B, C = heads(forward_xBC)
    forward_y, _ = self._scan(
      x, dt, B, C, left_state.ssm, backend, return_final_state=False)

    reverse_xBC = self._reverse_causal_conv(xBC, right_state.conv)
    x, B, C = heads(reverse_xBC)
    reverse_y, _ = self._scan(
      x, torch.flip(dt, dims=(1,)), B, C, right_state.ssm, backend,
      return_final_state=False)
    # Flip the scan output rather than `z`: both are one
    # ``[batch, length, d_inner]`` copy that the gated norm then retains, but
    # restoring the original order here is what lets the two directions share
    # `out_proj`, and it removes the flip of the projected output.
    reverse_y = torch.flip(reverse_y, dims=(1,))

    return self.out_proj(
      self._gated_norm(forward_y, z) + self._gated_norm(reverse_y, z))

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    output, _ = self.scan_segment(u)
    return output
