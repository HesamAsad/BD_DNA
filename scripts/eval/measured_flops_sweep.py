#!/usr/bin/env python3
"""FLOPs actually dispatched, per arm, across sequence lengths.

`training_flops.py` derives FLOPs from hand-written formulas. Running the
counter against it showed those formulas are right for the Transformer -- the
shortfall equals the attention term to three significant figures -- and WRONG
for every SSM arm, understating by 26% at L=8192 and 35% at L=32768. So the
FLOPs panel of the scaling figure cannot be built from arithmetic.

This measures instead. One forward+backward per point under
`torch.utils.flop_counter.FlopCounterMode`, no timing loop, so it is cheap.

Two things the counter cannot do, both handled explicitly rather than silently:

* It is blind to flash/flex attention and to mamba-ssm's Triton SSD kernel,
  which dispatch outside aten. For the Transformer the missing piece is exactly
  the attention term and can be added back analytically; that reconstruction is
  reported separately from the raw count so the two never get confused.
* Transformer-BD's flex path fails under the counter outright
  ("Attempted to call function marked as skipped" -- flex is compiled). This
  falls back to `attn_backend=sdpa` for that arm, which computes the same
  attention with an aten op the counter can see. Different kernel, same
  arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import hydra
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main  # noqa: F401,E402 - registers the OmegaConf resolvers
from dataloader import DNATokenizer  # noqa: E402
from diffusion import Diffusion  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import scripts.eval.training_flops as tf  # noqa: E402

ARMS = {
  "bissm":   ("small_bissm", "bd3lm_bissm", False),
  "ussm-ar": ("small_ussm", "ar", False),
  "dit":     ("small", "bd3lm", True),
  "dit-ar":  ("small_ar_transformer", "ar", False),
}


def build(arm, length, block_size, batch):
  model_cfg, algo_cfg, needs_sdpa = ARMS[arm]
  is_ar = algo_cfg == "ar"
  overrides = [
    f"model={model_cfg}", f"algo={algo_cfg}", "data=carbon-prokaryote",
    f"model.length={length}", f"block_size={1 if is_ar else block_size}",
    f"loader.batch_size={batch}", f"loader.eval_batch_size={batch}",
    "loader.global_batch_size=64", "training.ema=0",
    "trainer.accumulate_grad_batches=1",
  ]
  if arm == "ussm-ar":
    overrides.append("algo.backbone=ussm")
  if arm == "bissm":
    overrides.append("model.active_blocks=all")
    overrides.append("model.checkpoint_boundary_prefill=true")
  if needs_sdpa:
    # flex is compiled and the counter cannot enter it; sdpa is the same
    # attention through an aten op it can see.
    overrides.append("model.attn_backend=sdpa")
  with hydra.initialize_config_dir(version_base=None,
                                   config_dir=str(REPO / "configs")):
    return hydra.compose(config_name="config", overrides=overrides)


def analytic_attention_tflop(arm, length, batch, n_layers=12):
  """The term the counter is structurally blind to, for the Transformer arms."""
  if arm == "dit-ar":
    tokens = length - 1
    pairs = tokens * (tokens + 1) // 2
    return tf.GRAD_MULT * 4 * n_layers * pairs * tf.D_AR * batch / 1e12
  if arm == "dit":
    nb = max(length // 256, 1)
    pairs = 256 * 256 * nb * (nb + 1)
    return tf.GRAD_MULT * 4 * n_layers * pairs * tf.D_BD * batch / 1e12
  return 0.0


def run(arm, length, block, batch, device):
  config = build(arm, length, block, batch)
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  x0 = torch.randint(8, 12, (batch, length), device=device)
  mask = torch.ones_like(x0)
  counter = FlopCounterMode(display=False)
  with counter:
    loss = model._loss(x0, mask)
    loss.loss.backward()
  counted = counter.get_total_flops() / 1e12
  del model, x0, mask, loss
  torch.cuda.empty_cache()
  return counted


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arms", default="bissm,dit,ussm-ar,dit-ar")
  parser.add_argument("--lengths", default="2048,4096,8192,16384,32768")
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--output", type=Path,
                      default=REPO / "results" / "sizing" / "measured_flops.json")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU")
  device = torch.device("cuda")
  rows = []
  print(f"{'arm':<9}{'L':>8}{'counted TF':>12}{'+attn TF':>10}{'total TF':>10}")
  print("-" * 49)
  for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
    for length in [int(v) for v in args.lengths.split(",")]:
      try:
        counted = run(arm, length, args.block_size, args.batch, device)
        attn = analytic_attention_tflop(arm, length, args.batch)
        rows.append({"arm": arm, "length": length, "batch": args.batch,
                     "counted_tflop": counted,
                     "analytic_attention_tflop": attn,
                     "total_tflop": counted + attn})
        print(f"{arm:<9}{length:>8}{counted:>12.2f}{attn:>10.2f}"
              f"{counted + attn:>10.2f}")
      except Exception as exc:  # noqa: BLE001
        rows.append({"arm": arm, "length": length,
                     "error": f"{type(exc).__name__}: {exc}"})
        print(f"{arm:<9}{length:>8}   FAILED {type(exc).__name__}: "
              f"{str(exc)[:44]}")
        torch.cuda.empty_cache()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("w", dir=args.output.parent,
                                   delete=False) as handle:
    json.dump({"batch": args.batch, "block_size": args.block_size,
               "note": ("counted_tflop is what FlopCounterMode dispatched "
                        "(blind to flash/flex attention and the Triton SSD "
                        "scan). analytic_attention_tflop adds back the "
                        "attention term for the Transformer arms only; it is "
                        "zero for the SSM arms, whose scan remains uncounted, "
                        "so their total is a LOWER bound."),
               "rows": rows}, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, args.output)
  print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
