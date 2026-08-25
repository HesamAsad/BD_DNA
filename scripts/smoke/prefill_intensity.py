#!/usr/bin/env python3
"""H3: are the boundary-prefill kernels too small to occupy the GPU?

WHAT THIS TESTS. The measured throughput table has the signature of a FIXED
per-step cost: BiSSM's step time is 257.9 / 259.0 / 263.0 ms at L = 2048 /
4096 / 8192, i.e. FLAT while the arithmetic quadruples. H3 attributes that
fixed cost to LOW ARITHMETIC INTENSITY -- kernels running on tensors too small
to fill an H200, with the block-structured operations
(`_block_state_passing`'s [num_seg+1, num_seg] decay matmul, the conv-state
gather, the per-block folded scan) as the named suspects.

H3 makes a QUANTITATIVE, falsifiable prediction: the size-starved operations
must account for the ~258 ms floor. This script enumerates every tensor that
flows through `SegmentMamba2.scan_with_block_boundaries`
(models/mamba2_segment.py:456), `_block_state_passing`
(models/mamba2_segment.py:412) and `SegmentMamba2.scan_bidirectional`
(models/mamba2_segment.py:578) at the measured geometry, and puts a roofline
time on each: max(FLOPs / peak, bytes / bandwidth), floored at a per-launch
latency. Summing the size-starved rows and comparing against 258 ms is the
test.

NO CUDA IS USED. Every number here is closed-form from the shapes the code
computes; the shapes themselves are asserted against a real CPU forward of
`SegmentMamba2` under a shape-recording hook (`--verify`), so the arithmetic
is not a paper exercise.

VERDICT: H3 IS REFUTED for the length-dependent excess, on four counts.

 1. MAGNITUDE. Every size-starved op together, charged a generous 6 us floor
    each, is 6.49 ms of BiSSM's 257.94 ms step at L=2048 -- 2.5%. To carry the
    floor each of the 936 starved launches would need 276 us of GPU time on
    <=12.6 MB of operands, i.e. 46 GB/s, 1.0% of HBM.
 2. WRONG SIGN. The starved share GROWS with length (2.5 -> 4.8 -> 5.3%),
    because every block-shaped tensor grows with num_seg (7 -> 127). The
    unexplained factor SHRINKS (2.08 -> 1.44). With the per-launch floor set
    to zero -- pure arithmetic intensity -- the predicted BD/AR efficiency gap
    is FLAT in length (1.55 / 1.56 / 1.59), never decaying.
 3. OCCUPANCY IS PRESERVED BY THE FOLD, exactly. Folding blocks into the batch
    (diffusion.py:1232) makes the active pass's GEMM M = batch*num_blocks*block
    = batch*length = the AR arm's M, and makes the SSD scan's tile grid
    batch*length/chunk*nheads either way. Nothing heavy has a leading
    dimension of num_blocks.
 4. THE CONTROL. uSSM-AR has no block structure at all (0 starved kernels) and
    plateaus too: 63.30 -> 64.63 ms for 2.00x the FLOPs.

What the evidence points at instead is per-launch overhead: dispatch counts
are length-INDEPENDENT (scripts/smoke/launch_count_probe.py: 1746 / 7360 /
8152 aten dispatches for uSSM-AR / uSSM-BD / BiSSM, identical at L=2048 and
L=8192) and the three plateaus divided by those counts give 36.3 / 28.6 /
31.6 us per dispatch -- one constant across arms whose tensors differ 20x.

Usage:
  python scripts/smoke/prefill_intensity.py                 # tables
  python scripts/smoke/prefill_intensity.py --verify        # shapes vs code
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# ---- geometry: configs/model/small_bissm.yaml -----------------------------
D_MODEL = 768        # model.hidden_size
N_LAYERS = 12        # model.n_blocks
D_STATE = 64         # model.ssm_state_size
D_CONV = 4           # model.ssm_conv_size
EXPAND = 2           # model.ssm_expand
HEADDIM = 64         # model.ssm_head_dim
CHUNK = 128          # model.ssm_chunk_size
MLP_RATIO = 4.0
BLOCK = 256          # block_size, set by the launch scripts
D_INNER = EXPAND * D_MODEL                    # 1536
NHEADS = D_INNER // HEADDIM                   # 24
CONV_DIM = D_INNER + 2 * D_STATE              # 1664
PROJ_DIM = 2 * D_INNER + 2 * D_STATE + NHEADS  # 3224
MLP_HIDDEN = int(D_MODEL * MLP_RATIO)         # 3072

# ---- H200 SXM roofline ----------------------------------------------------
# 989.4 TFLOP/s dense BF16 tensor core, 4.8 TB/s HBM3e, 67 TFLOP/s FP32 on the
# CUDA cores (the fp32 figure matters: `_block_state_passing` disables autocast
# at models/mamba2_segment.py:433).
PEAK_BF16 = 989.4e12
PEAK_FP32 = 67.0e12
PEAK_TF32 = 494.7e12
BW = 4.8e12
# Efficiency haircuts. Deliberately GENEROUS to H3: the smaller these are, the
# more time the size-starved kernels are allowed to take.
GEMM_EFF = 0.70
BW_EFF = 0.80
# Floor for any single kernel: empty-kernel latency on an H200 is ~2 us; eager
# dispatch adds a few more. 6 us is charitable to H3.
LAUNCH_US = 6.0


@dataclass
class Op:
  name: str
  shape: str
  flops: float = 0.0
  bytes: float = 0.0
  dtype: str = "bf16"
  count: int = 1          # launches per layer per call
  starved: bool = False   # H3 flags this as a size-starved / block-shaped op
  note: str = ""

  @property
  def peak(self):
    return {"bf16": PEAK_BF16, "fp32": PEAK_FP32, "tf32": PEAK_TF32}[self.dtype]

  @property
  def ai(self):
    return self.flops / self.bytes if self.bytes else float("inf")

  @property
  def t_roof_us(self):
    t = max(self.flops / (self.peak * GEMM_EFF),
            self.bytes / (BW * BW_EFF))
    return max(t * 1e6, LAUNCH_US) * self.count


def bf16(n):
  return 2.0 * n


def fp32(n):
  return 4.0 * n


def prefill_ops(batch, length):
  """One layer of `scan_clean_with_boundaries` -> `scan_with_block_boundaries`.

  models/bidirectional_ssm.py:217 (layer) -> models/mamba2_segment.py:456.
  The prefix is (num_blocks - 1) * BLOCK: models/bidirectional_ssm.py:492.
  """
  nb = length // BLOCK
  pre = (nb - 1) * BLOCK
  nseg = pre // BLOCK               # == nb - 1
  rows = batch * pre                # token positions in the prefill
  ops: list[Op] = []
  A = ops.append

  A(Op("mixer_norm (rmsnorm_fn)", f"[{batch},{pre},{D_MODEL}]",
       bytes=2 * bf16(rows * D_MODEL)))
  A(Op("in_proj GEMM", f"[{rows},{D_MODEL}] @ [{D_MODEL},{PROJ_DIM}]",
       flops=2 * rows * D_MODEL * PROJ_DIM,
       bytes=bf16(rows * D_MODEL + D_MODEL * PROJ_DIM + rows * PROJ_DIM),
       note="mamba2_segment.py:483"))
  A(Op("conv1d depthwise", f"[{batch},{CONV_DIM},{pre}] k={D_CONV} g={CONV_DIM}",
       flops=2 * batch * CONV_DIM * pre * D_CONV,
       bytes=2 * bf16(batch * CONV_DIM * pre),
       note="mamba2_segment.py:494"))
  A(Op("silu(conv)", f"[{batch},{pre},{CONV_DIM}]",
       bytes=2 * bf16(batch * pre * CONV_DIM)))
  # (1) contiguous scan over the whole prefix, mamba2_segment.py:525
  A(Op("scan #1 full prefix", f"x[{batch},{pre},{NHEADS},{HEADDIM}] nchunk={pre//CHUNK}",
       flops=scan_flops(batch * pre), bytes=scan_bytes(batch, pre),
       count=1, note="mamba2_segment.py:525"))
  # (2) folded per-block scan, mamba2_segment.py:530
  A(Op("scan #2 folded blocks",
       f"x[{batch*nseg},{BLOCK},{NHEADS},{HEADDIM}] nchunk={BLOCK//CHUNK}",
       flops=scan_flops(batch * pre), bytes=scan_bytes(batch * nseg, BLOCK),
       count=1, starved=(BLOCK // CHUNK <= 2),
       note="mamba2_segment.py:530; leading dim is batch*num_seg"))
  # conv-state gather, mamba2_segment.py:502-512
  A(Op("conv_states zeros", f"[{batch},1,{CONV_DIM},{D_CONV}]",
       bytes=bf16(batch * CONV_DIM * D_CONV), starved=True,
       note="mamba2_segment.py:503"))
  A(Op("conv_states cat", f"[{batch},{nseg+1},{CONV_DIM},{D_CONV}]",
       bytes=2 * bf16(batch * (nseg + 1) * CONV_DIM * D_CONV), starved=True,
       note="mamba2_segment.py:509"))
  ops += block_state_passing_ops(batch, pre, nseg)
  A(Op("gated rmsnorm (fused)", f"[{batch},{pre},{D_INNER}]",
       bytes=3 * bf16(rows * D_INNER)))
  A(Op("out_proj GEMM", f"[{rows},{D_INNER}] @ [{D_INNER},{D_MODEL}]",
       flops=2 * rows * D_INNER * D_MODEL,
       bytes=bf16(rows * D_INNER + D_INNER * D_MODEL + rows * D_MODEL)))
  A(Op("residual add", f"[{rows},{D_MODEL}]", bytes=3 * bf16(rows * D_MODEL)))
  A(Op("mlp_norm", f"[{rows},{D_MODEL}]", bytes=2 * bf16(rows * D_MODEL)))
  A(Op("mlp fc1", f"[{rows},{D_MODEL}] @ [{D_MODEL},{MLP_HIDDEN}]",
       flops=2 * rows * D_MODEL * MLP_HIDDEN,
       bytes=bf16(rows * D_MODEL + rows * MLP_HIDDEN)))
  A(Op("gelu", f"[{rows},{MLP_HIDDEN}]", bytes=2 * bf16(rows * MLP_HIDDEN)))
  A(Op("mlp fc2", f"[{rows},{MLP_HIDDEN}] @ [{MLP_HIDDEN},{D_MODEL}]",
       flops=2 * rows * MLP_HIDDEN * D_MODEL,
       bytes=bf16(rows * MLP_HIDDEN + rows * D_MODEL)))
  A(Op("residual add", f"[{rows},{D_MODEL}]", bytes=3 * bf16(rows * D_MODEL)))
  return ops


def block_state_passing_ops(batch, pre, nseg):
  """models/mamba2_segment.py:412-455. Everything here is fp32 (autocast off
  at :433) and everything whose leading axis is `nseg` is what H3 accuses."""
  ops = []
  A = ops.append
  A(Op("bsp: exp(A_log)/neg", f"[{NHEADS}]", bytes=fp32(2 * NHEADS),
       dtype="fp32", count=2, starved=True, note=":435"))
  A(Op("bsp: dt.float()", f"[{batch},{pre},{NHEADS}]",
       bytes=bf16(batch * pre * NHEADS) + fp32(batch * pre * NHEADS),
       dtype="fp32", note=":436"))
  A(Op("bsp: softplus", f"[{batch},{pre},{NHEADS}]",
       bytes=2 * fp32(batch * pre * NHEADS), dtype="fp32", note=":436"))
  A(Op("bsp: segment sum", f"[{batch},{nseg},{BLOCK},{NHEADS}] -> [{batch},{nseg},{NHEADS}]",
       bytes=fp32(batch * pre * NHEADS), dtype="fp32", note=":437"))
  A(Op("bsp: cumsum", f"[{batch},{nseg},{NHEADS}]",
       bytes=2 * fp32(batch * nseg * NHEADS), dtype="fp32", starved=True,
       note=":441"))
  A(Op("bsp: pad", f"[{batch},{nseg+1},{NHEADS}]",
       bytes=2 * fp32(batch * (nseg + 1) * NHEADS), dtype="fp32", starved=True,
       note=":440"))
  A(Op("bsp: exponent sub+mul", f"[{batch},{nseg+1},{nseg},{NHEADS}]",
       bytes=2 * fp32(batch * (nseg + 1) * nseg * NHEADS), dtype="fp32",
       count=2, starved=True, note=":443"))
  A(Op("bsp: arange x2", f"[{nseg}],[{nseg+1}]", bytes=fp32(2 * nseg + 1),
       dtype="fp32", count=2, starved=True, note=":446"))
  A(Op("bsp: lt mask", f"[{nseg+1},{nseg}]", bytes=fp32(2 * (nseg + 1) * nseg),
       dtype="fp32", starved=True, note=":446"))
  A(Op("bsp: clamp+exp", f"[{batch},{nseg+1},{nseg},{NHEADS}]",
       bytes=2 * fp32(batch * (nseg + 1) * nseg * NHEADS), dtype="fp32",
       count=2, starved=True, note=":449"))
  A(Op("bsp: where", f"[{batch},{nseg+1},{nseg},{NHEADS}]",
       bytes=2 * fp32(batch * (nseg + 1) * nseg * NHEADS), dtype="fp32",
       starved=True, note=":449"))
  A(Op("bsp: local_ssm.float()", f"[{batch},{nseg},{NHEADS},{HEADDIM},{D_STATE}]",
       bytes=2 * fp32(batch * nseg * NHEADS * HEADDIM * D_STATE), dtype="fp32",
       note=":454"))
  # einsum "bijh,bjhpn->bihpn": permute decay -> [b,h,i,j], permute local_ssm
  # -> [b,h,j,p*n], bmm(batch=b*h, M=i, K=j, N=p*n), permute back.
  A(Op("bsp: einsum permute in", f"[{batch},{nseg},{NHEADS},{HEADDIM*D_STATE}]",
       bytes=2 * fp32(batch * nseg * NHEADS * HEADDIM * D_STATE), dtype="fp32",
       starved=True, note=":454 permute copy"))
  A(Op("bsp: einsum bmm",
       f"batch={batch*NHEADS} M={nseg+1} K={nseg} N={HEADDIM*D_STATE}",
       flops=2 * batch * NHEADS * (nseg + 1) * nseg * HEADDIM * D_STATE,
       bytes=fp32(batch * NHEADS * ((nseg + 1) * nseg
                                    + nseg * HEADDIM * D_STATE
                                    + (nseg + 1) * HEADDIM * D_STATE)),
       dtype="fp32", starved=True, note=":454 THE M=8 GEMM"))
  A(Op("bsp: einsum permute out",
       f"[{batch},{nseg+1},{NHEADS},{HEADDIM},{D_STATE}]",
       bytes=2 * fp32(batch * (nseg + 1) * NHEADS * HEADDIM * D_STATE),
       dtype="fp32", starved=True, note=":454 permute copy"))
  return ops


def scan_flops(tokens):
  """Per-token SSD cost, identical to scripts/eval/training_flops.py:158."""
  per = (2 * CHUNK * D_STATE
         + NHEADS * (2 * CHUNK * HEADDIM + 2 * D_STATE * HEADDIM
                     + 2 * D_STATE * HEADDIM)
         + 2 * NHEADS * HEADDIM * D_STATE // CHUNK)
  return per * tokens


def scan_bytes(batch, seqlen):
  """HBM traffic of one mamba_chunk_scan_combined call (fwd)."""
  nchunk = max(seqlen // CHUNK, 1)
  x = bf16(batch * seqlen * NHEADS * HEADDIM)
  bc = 2 * bf16(batch * seqlen * D_STATE)
  dt = bf16(batch * seqlen * NHEADS)
  y = bf16(batch * seqlen * NHEADS * HEADDIM)
  # chunk states + cumulative dA, fp32, written then read
  states = 2 * fp32(batch * nchunk * NHEADS * HEADDIM * D_STATE)
  cb = 2 * bf16(batch * nchunk * CHUNK * CHUNK)   # the [chunk,chunk] tile
  return x + bc + dt + y + states + cb


def active_ops(batch, length):
  """One layer of `scan_bidirectional` (models/mamba2_segment.py:578).

  The active pass folds blocks into the batch: rows = batch*num_blocks,
  seqlen = BLOCK (diffusion.py:1232-1234), so rows*BLOCK == batch*length.
  """
  nb = length // BLOCK
  rows = batch * nb * BLOCK       # == batch * length
  ops: list[Op] = []
  A = ops.append
  A(Op("act mixer_norm", f"[{batch*nb},{BLOCK},{D_MODEL}]",
       bytes=2 * bf16(rows * D_MODEL)))
  A(Op("act in_proj GEMM", f"[{rows},{D_MODEL}] @ [{D_MODEL},{PROJ_DIM}]",
       flops=2 * rows * D_MODEL * PROJ_DIM,
       bytes=bf16(rows * D_MODEL + rows * PROJ_DIM)))
  A(Op("act conv fwd+rev", f"[{batch*nb},{CONV_DIM},{BLOCK}] x2",
       flops=2 * 2 * rows * CONV_DIM * D_CONV,
       bytes=2 * 2 * bf16(rows * CONV_DIM), count=1,
       note="mamba2_segment.py:653,660 plus the d_conv-1 fixups"))
  A(Op("act scan fwd", f"x[{batch*nb},{BLOCK},{NHEADS},{HEADDIM}] nchunk={BLOCK//CHUNK}",
       flops=scan_flops(rows), bytes=scan_bytes(batch * nb, BLOCK),
       starved=(BLOCK // CHUNK <= 2), note="mamba2_segment.py:655"))
  A(Op("act flip dt/y", f"[{batch*nb},{BLOCK},*]",
       bytes=2 * bf16(rows * NHEADS) + 2 * bf16(rows * D_INNER), count=2))
  A(Op("act scan rev", f"x[{batch*nb},{BLOCK},{NHEADS},{HEADDIM}] nchunk={BLOCK//CHUNK}",
       flops=scan_flops(rows), bytes=scan_bytes(batch * nb, BLOCK),
       starved=(BLOCK // CHUNK <= 2), note="mamba2_segment.py:663"))
  A(Op("act gated rmsnorm x2", f"[{rows},{D_INNER}]",
       bytes=2 * 3 * bf16(rows * D_INNER)))
  A(Op("act out_proj GEMM", f"[{rows},{D_INNER}] @ [{D_INNER},{D_MODEL}]",
       flops=2 * rows * D_INNER * D_MODEL,
       bytes=bf16(rows * D_INNER + rows * D_MODEL)))
  A(Op("act mlp", f"[{rows},{D_MODEL}] <-> [{rows},{MLP_HIDDEN}]",
       flops=2 * 2 * rows * D_MODEL * MLP_HIDDEN,
       bytes=4 * bf16(rows * MLP_HIDDEN), count=1))
  return ops


def ar_ops(batch, length):
  """One layer of uSSM-AR: `scan_clean` -> `scan_segment` over the whole
  sequence (models/unidirectional_ssm.py:69, models/bidirectional_ssm.py:206).
  Every leading dimension here is batch*length -- the control for H3."""
  rows = batch * length
  ops: list[Op] = []
  A = ops.append
  A(Op("ar mixer_norm", f"[{batch},{length},{D_MODEL}]",
       bytes=2 * bf16(rows * D_MODEL)))
  A(Op("ar in_proj GEMM", f"[{rows},{D_MODEL}] @ [{D_MODEL},{PROJ_DIM}]",
       flops=2 * rows * D_MODEL * PROJ_DIM,
       bytes=bf16(rows * D_MODEL + rows * PROJ_DIM)))
  A(Op("ar conv1d", f"[{batch},{CONV_DIM},{length}]",
       flops=2 * rows * CONV_DIM * D_CONV, bytes=2 * bf16(rows * CONV_DIM)))
  A(Op("ar scan", f"x[{batch},{length},{NHEADS},{HEADDIM}] nchunk={length//CHUNK}",
       flops=scan_flops(rows), bytes=scan_bytes(batch, length)))
  A(Op("ar gated rmsnorm", f"[{rows},{D_INNER}]",
       bytes=3 * bf16(rows * D_INNER)))
  A(Op("ar out_proj GEMM", f"[{rows},{D_INNER}] @ [{D_INNER},{D_MODEL}]",
       flops=2 * rows * D_INNER * D_MODEL,
       bytes=bf16(rows * D_INNER + rows * D_MODEL)))
  A(Op("ar mlp", f"[{rows},{D_MODEL}] <-> [{rows},{MLP_HIDDEN}]",
       flops=2 * 2 * rows * D_MODEL * MLP_HIDDEN,
       bytes=4 * bf16(rows * MLP_HIDDEN)))
  return ops


def uni_active_ops(batch, length):
  """uSSM-BD active pass: one direction only (unidirectional_ssm.py:69)."""
  keep = []
  for o in active_ops(batch, length):
    if o.name in ("act scan rev", "act flip dt/y"):
      continue
    if o.name == "act conv fwd+rev":
      o = Op("act conv fwd", o.shape.replace(" x2", ""), o.flops / 2,
             o.bytes / 2, o.dtype, o.count, o.starved, o.note)
    if o.name == "act gated rmsnorm x2":
      o = Op("act gated rmsnorm", o.shape, o.flops, o.bytes / 2, o.dtype,
             o.count, o.starved, o.note)
    keep.append(o)
  return keep


def arm_roofline(arm, batch, length, ckpt=True):
  """(total FLOP fwd+bwd for `batch` sequences, roofline seconds, starved s)."""
  parts = {
    "ussm-ar": [(ar_ops(batch, length), 1.0)],
    # Non-reentrant checkpointing runs the prefill forward TWICE and its
    # backward once: 1 + 1 + 2 = 4 forward-equivalents where an unchecked pass
    # costs 3. `mult` scales the 3x already inside t_roof, hence 4/3.
    "ussm": [(prefill_ops(batch, length), 4.0 / 3.0 if ckpt else 1.0),
             (uni_active_ops(batch, length), 1.0)],
    "bissm": [(prefill_ops(batch, length), 4.0 / 3.0 if ckpt else 1.0),
              (active_ops(batch, length), 1.0)],
  }[arm]
  f = t = s = 0.0
  hbm = 0.0
  n_starved = 0
  for ops, mult in parts:
    for o in ops:
      # FLOPs are paid once per forward, not once per checkpoint recompute for
      # the purpose of the analytic FLOP column; time is paid `mult` times.
      f += o.flops * o.count * N_LAYERS * 3.0
      t += o.t_roof_us * 1e-6 * N_LAYERS * mult * 3.0
      hbm += o.bytes * o.count * N_LAYERS * mult * 3.0
      if o.starved:
        s += o.t_roof_us * 1e-6 * N_LAYERS * mult * 3.0
        n_starved += int(round(o.count * N_LAYERS * mult * 3))
  return f, t, s, hbm, n_starved


MEASURED_NT_S = {   # results/figures/scaling_data.json
  "ussm-ar": {2048: 64709.0, 4096: 126744.0, 8192: 213461.0,
              16384: 236029.0, 32768: 247728.0},
  "ussm": {2048: 19473.0, 4096: 38466.0, 8192: 74901.0,
           16384: 99141.0, 32768: 103151.0},
  "bissm": {2048: 15879.4, 4096: 31633.2, 8192: 62294.4,
            16384: 77109.1, 32768: 82424.7},
}


def summary(batch, lengths):
  print(f"\n{'='*104}")
  print("ARM ROOFLINE vs MEASURED STEP  (batch 2, fwd+bwd+AdamW step, "
        "checkpoint_boundary_prefill=on)")
  print("=" * 104)
  print(f"{'arm':<9}{'L':>7}{'roofline ms':>13}{'measured ms':>13}"
        f"{'gpu busy %':>12}{'starved ms':>12}{'starv/meas':>12}"
        f"{'HBM GB/step':>13}{'sust GB/s':>11}{'sust TF/s':>11}")
  roof = {}
  for arm in ("ussm-ar", "ussm", "bissm"):
    for L in lengths:
      if L not in MEASURED_NT_S[arm]:
        continue
      f, t, s, hbm, ns = arm_roofline(arm, batch, L)
      meas = batch * L / MEASURED_NT_S[arm][L]
      roof[(arm, L)] = (f, t, s, hbm, ns, meas)
      print(f"{arm:<9}{L:>7}{t*1e3:>13.2f}{meas*1e3:>13.2f}"
            f"{100*t/meas:>12.1f}{s*1e3:>12.2f}{100*s/meas:>11.1f}%"
            f"{hbm/1e9:>13.2f}{hbm/meas/1e9:>11.1f}{f/meas/1e12:>11.1f}")
  print("\nH3 stress test: time each size-starved launch would need in order "
        "to carry the whole step")
  print(f"{'arm':<9}{'L':>7}{'starved launches':>18}{'us each needed':>16}"
        f"{'us each modelled':>18}")
  for arm in ("ussm", "bissm"):
    for L in lengths:
      if (arm, L) not in roof:
        continue
      _, _, s, _, ns, meas = roof[(arm, L)]
      print(f"{arm:<9}{L:>7}{ns:>18}{meas/ns*1e6:>16.1f}"
            f"{s/ns*1e6:>18.1f}")
  # The discriminator. If the floor were size-starved GPU kernels, the cost
  # per launch would differ sharply between arms, because their tensors differ
  # sharply. If the floor is per-launch overhead, cost/launch is a constant.
  # Kernel counts from scripts/smoke/launch_count_probe.py (CPU dispatch
  # count, identical at L=2048 and L=8192 -- the count does not depend on
  # length, which is the whole point).
  kernels = {"ussm-ar": 1987, "ussm": 8030, "bissm": 9038}
  floor_ms = {"ussm-ar": 63.30, "ussm": 210.34, "bissm": 257.94}
  print("\nDISCRIMINATOR: cost per kernel launch at the plateau (L=2048)")
  print(f"{'arm':<9}{'kernels/step':>14}{'floor ms':>10}{'us/launch':>12}"
        f"{'starved kernels':>17}")
  for arm in ("ussm-ar", "ussm", "bissm"):
    ns = arm_roofline(arm, batch, 2048)[4]
    print(f"{arm:<9}{kernels[arm]:>14}{floor_ms[arm]:>10.2f}"
          f"{floor_ms[arm]*1e3/kernels[arm]:>12.2f}{ns:>17}")
  print("  A common us/launch across arms whose tensor shapes differ by 20x "
        "is\n  launch overhead, not occupancy.")
  print("\nH3's prediction for the BD/AR efficiency gap, from arithmetic "
        "intensity alone:")
  print(f"{'pair':<16}{'L':>7}{'U_roofline':>13}{'U_measured':>13}")
  for arm in ("bissm", "ussm"):
    for L in lengths:
      if (arm, L) not in roof:
        continue
      fb, tb = roof[(arm, L)][0], roof[(arm, L)][1]
      fa, ta = roof[("ussm-ar", L)][0], roof[("ussm-ar", L)][1]
      u_roof = (fa / ta) / (fb / tb)
      mb = batch * L / MEASURED_NT_S[arm][L]
      ma = batch * L / MEASURED_NT_S["ussm-ar"][L]
      u_meas = (fa / ma) / (fb / mb)
      print(f"{arm+'/ussm-ar':<16}{L:>7}{u_roof:>13.3f}{u_meas:>13.3f}")


def report(batch, length, grad_mult=3.0, ckpt_prefill=True):
  nb = length // BLOCK
  pre = (nb - 1) * BLOCK
  nseg = nb - 1
  print(f"\n{'='*100}")
  print(f"L={length}  batch={batch}  block={BLOCK}  num_blocks={nb}  "
        f"prefix={pre}  num_seg={nseg}")
  print("=" * 100)
  for title, ops, mult in (
      ("PREFILL  (per layer; "
       f"{'checkpointed: 2 fwd + 1 bwd = 4 fwd-equiv' if ckpt_prefill else '3 fwd-equiv'})",
       prefill_ops(batch, length), 4.0 / 3.0 if ckpt_prefill else 1.0),
      ("ACTIVE   (per layer)", active_ops(batch, length), 1.0)):
    print(f"\n-- {title}")
    print(f"{'op':<28}{'shape':<52}{'GFLOP':>9}{'MB':>9}{'AI':>8}"
          f"{'us':>9}  starved")
    tot_us = tot_f = 0.0
    starved_us = 0.0
    for o in ops:
      print(f"{o.name:<28}{o.shape:<52}{o.flops/1e9:>9.3f}{o.bytes/1e6:>9.2f}"
            f"{o.ai:>8.1f}{o.t_roof_us:>9.1f}  {'YES' if o.starved else ''}")
      tot_us += o.t_roof_us
      tot_f += o.flops * o.count
      if o.starved:
        starved_us += o.t_roof_us
    print(f"{'TOTAL/layer':<28}{'':<52}{tot_f/1e9:>9.3f}{'':>9}{'':>8}"
          f"{tot_us:>9.1f}")
    print(f"  x{N_LAYERS} layers x{mult*grad_mult:.2f} fwd-equiv = "
          f"{tot_us*N_LAYERS*mult*grad_mult/1000:>8.2f} ms   "
          f"of which size-starved: "
          f"{starved_us*N_LAYERS*mult*grad_mult/1000:>7.2f} ms")


def verify(batch, length):
  """Assert the shapes above are the ones the code actually produces."""
  import torch
  import models.mamba2_segment as m2
  nb = length // BLOCK
  pre = (nb - 1) * BLOCK
  seen = {}
  mixer = m2.SegmentMamba2(d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
                           expand=EXPAND, headdim=HEADDIM, chunk_size=CHUNK,
                           backend="torch")
  orig_bsp = m2.SegmentMamba2._block_state_passing
  orig_scan = m2.SegmentMamba2._scan

  def bsp(self, dt, local_ssm, block_size):
    seen["bsp.dt"] = tuple(dt.shape)
    seen["bsp.local_ssm"] = tuple(local_ssm.shape)
    out = orig_bsp(self, dt, local_ssm, block_size)
    seen["bsp.out"] = tuple(out.shape)
    return out

  def scan(self, x, dt, B, C, s, backend, return_final_state=True):
    # Shapes only: the CPU reference scan is a Python loop over positions and
    # would take minutes at this length. Return correctly shaped zeros.
    seen.setdefault("scan.x", []).append(tuple(x.shape))
    seen.setdefault("scan.dt", []).append(tuple(dt.shape))
    seen.setdefault("scan.B", []).append(tuple(B.shape))
    y = torch.zeros_like(x)
    final = torch.zeros(x.shape[0], self.nheads, self.headdim, self.d_state)
    return (y, final) if return_final_state else (y, None)

  m2.SegmentMamba2._block_state_passing = bsp
  m2.SegmentMamba2._scan = scan
  u = torch.randn(batch, pre, D_MODEL)
  with torch.no_grad():
    y, conv_states, ssm_states = mixer.scan_with_block_boundaries(u, BLOCK)
  m2.SegmentMamba2._block_state_passing = orig_bsp
  m2.SegmentMamba2._scan = orig_scan
  seen["conv_states"] = tuple(conv_states.shape)
  seen["ssm_states"] = tuple(ssm_states.shape)
  seen["ssm_states.dtype"] = str(ssm_states.dtype)
  print(f"\nVERIFY L={length} batch={batch} (real CPU forward):")
  for k, v in seen.items():
    print(f"  {k:<20} {v}")
  nseg = nb - 1
  assert seen["scan.x"][0] == (batch, pre, NHEADS, HEADDIM)
  assert seen["scan.x"][1] == (batch * nseg, BLOCK, NHEADS, HEADDIM)
  assert seen["bsp.local_ssm"] == (batch, nseg, NHEADS, HEADDIM, D_STATE)
  assert seen["conv_states"] == (batch, nseg + 1, CONV_DIM, D_CONV)
  assert seen["ssm_states"] == (batch, nseg + 1, NHEADS, HEADDIM, D_STATE)
  assert str(ssm_states.dtype) == "torch.float32", "bsp must stay fp32"
  print("  OK: every shape used by the roofline table is the code's own.")


if __name__ == "__main__":
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--batch", type=int, default=2)
  ap.add_argument("--lengths", default="2048,8192,32768")
  ap.add_argument("--verify", action="store_true")
  ap.add_argument("--summary-only", action="store_true")
  a = ap.parse_args()
  lengths = [int(v) for v in a.lengths.split(",")]
  if not a.summary_only:
    for L in lengths:
      report(a.batch, L)
  summary(a.batch, lengths)
  if a.verify:
    for L in [int(v) for v in a.lengths.split(",")][:2]:
      verify(a.batch, L)
