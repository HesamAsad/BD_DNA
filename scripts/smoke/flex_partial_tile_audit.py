#!/usr/bin/env python3
"""Does flex attention compute PARTIALLY masked tiles for the BD3LM mask?

H4 (Transformer-BD) proposes that the analytic attention denominator in
scripts/eval/training_flops.py:218 / scripts/eval/scaling_curves.py:110

    pairs = block^2 * nb * (nb + 1)

undercounts the real arithmetic, because flex_attention skips only FULLY
masked tiles and computes PARTIALLY masked ones in full.  If so the true work
is larger and part of the "unexplained" 1.20-1.26x is just a wrong
denominator.

This runs `create_block_mask` with the project's own `block_diff_mask`
(models/dit.py:31) on CPU -- no GPU needed -- and reads the returned
BlockMask's two counters:

    kv_num_blocks       tiles that are PARTIALLY unmasked (mask_mod is
                        evaluated inside the kernel; full arithmetic paid)
    full_kv_num_blocks  tiles that are ENTIRELY unmasked (no mask_mod)

`kv_num_blocks + full_kv_num_blocks` is therefore the number of tiles the
kernel actually touches, and each costs a full BLOCK_Q x BLOCK_KV tile of
work regardless of how much of it is masked out.

Usage:  python scripts/smoke/flex_partial_tile_audit.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402

from models.dit import block_diff_mask  # noqa: E402

BLOCK = 256


def audit(seqlen, block=BLOCK, tile=128):
  mask = create_block_mask(
    partial(block_diff_mask, block_size=block, n=seqlen),
    B=None, H=None, Q_LEN=seqlen * 2, KV_LEN=seqlen * 2,
    BLOCK_SIZE=(tile, tile),
    device="cpu",
    _compile=False)
  partial_tiles = int(mask.kv_num_blocks.sum())
  full_tiles = int(mask.full_kv_num_blocks.sum()) if mask.full_kv_num_blocks is not None else 0
  touched = partial_tiles + full_tiles

  nb = seqlen // block
  analytic_pairs = block * block * nb * (nb + 1)
  # every touched tile costs a whole tile of arithmetic
  kernel_pairs = touched * tile * tile
  return dict(
    seqlen=seqlen, tile=tile,
    partial_tiles=partial_tiles, full_tiles=full_tiles, touched=touched,
    total_tiles=(2 * seqlen // tile) ** 2,
    analytic_pairs=analytic_pairs, kernel_pairs=kernel_pairs,
    overcompute=kernel_pairs / analytic_pairs)


def flash_causal_audit(seqlen, tile=128):
  """Same question for the AR arm's flash-attention causal mask (the DENOMINATOR
  of the slowdown ratio).  Flash skips fully-masked kv blocks but computes the
  diagonal ones in full."""
  tokens = seqlen - 1                       # diffusion.py shifts by one
  analytic_pairs = tokens * (tokens + 1) // 2
  n = -(-tokens // tile)                    # ceil
  kernel_pairs = tile * tile * n * (n + 1) // 2
  return dict(seqlen=seqlen, analytic_pairs=analytic_pairs,
              kernel_pairs=kernel_pairs,
              overcompute=kernel_pairs / analytic_pairs)


if __name__ == "__main__":
  lengths = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192]
  print("Transformer-BD, flex block mask (block_size=256, Q=KV=2L)")
  print(f"{'L':>7} {'tiles':>8} {'touched':>8} {'PARTIAL':>8} {'full':>8} "
        f"{'analytic pairs':>16} {'kernel pairs':>16} {'over':>7}")
  for L in lengths:
    r = audit(L)
    print(f"{r['seqlen']:>7} {r['total_tiles']:>8} {r['touched']:>8} "
          f"{r['partial_tiles']:>8} {r['full_tiles']:>8} "
          f"{r['analytic_pairs']:>16,} {r['kernel_pairs']:>16,} "
          f"{r['overcompute']:>7.4f}")

  print("\nTransformer-AR, flash causal (the denominator of the ratio)")
  print(f"{'L':>7} {'analytic pairs':>16} {'kernel pairs':>16} {'over':>7}")
  for L in [2048, 4096, 8192, 16384, 32768]:
    r = flash_causal_audit(L)
    print(f"{r['seqlen']:>7} {r['analytic_pairs']:>16,} "
          f"{r['kernel_pairs']:>16,} {r['overcompute']:>7.4f}")
