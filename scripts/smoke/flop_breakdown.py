#!/usr/bin/env python
"""Per-module FLOP breakdown, to find where the SSM formula loses work.

`training_flops.py` undercounts the SSM arms by a constant ~1.35x at every
length, while reproducing the Transformer exactly. A constant factor means one
or more PER-TOKEN terms are missing or wrong, not a structural error -- so
counting per module and comparing term by term should localise it.

Uses FlopCounterMode's module breakdown against the analytic terms from
`ssm_terms()`. The Triton SSD scan dispatches outside aten and is invisible, so
the scan row is expected to read zero; every OTHER row should match.
"""
import argparse, json, os, sys
from pathlib import Path

import hydra, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import main  # noqa: F401 - registers resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion
from torch.utils.flop_counter import FlopCounterMode
import scripts.eval.training_flops as tf


def build(length, batch):
  with hydra.initialize_config_dir(version_base=None,
                                   config_dir=str(REPO / "configs")):
    return hydra.compose(config_name="config", overrides=[
      "model=small_ussm", "algo=ar", "algo.backbone=ussm", "block_size=1",
      "data=carbon-prokaryote", f"model.length={length}",
      f"loader.batch_size={batch}", f"loader.eval_batch_size={batch}",
      "loader.global_batch_size=64", "training.ema=0",
      "trainer.accumulate_grad_batches=1"])


def main_cli():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--length", type=int, default=2048)
  ap.add_argument("--batch", type=int, default=1)
  ap.add_argument("--output", type=Path,
                  default=REPO / "results" / "sizing" / "flop_breakdown.json")
  a = ap.parse_args()
  device = torch.device("cuda")
  model = Diffusion(build(a.length, a.batch), DNATokenizer()).to(device)
  model.train()
  x0 = torch.randint(8, 12, (a.batch, a.length), device=device)
  mask = torch.ones_like(x0)

  counter = FlopCounterMode(mods=model, depth=4, display=False)
  with counter:
    loss = model._loss(x0, mask)
    loss.loss.backward()
  counts = counter.get_flop_counts()

  # Sum by leaf module kind across the 12 layers.
  by_kind, total = {}, 0
  for module, ops in counts.items():
    s = sum(ops.values())
    if module == "Global":
      total = s
      continue
    name = module.split(".")[-1]
    by_kind[name] = by_kind.get(name, 0) + s

  tokens = a.batch * (a.length - 1)          # AR shifts by one
  n_layers, t = 12, tf.ssm_terms()
  analytic = {
    "in_proj":  tf.GRAD_MULT * n_layers * tokens * t["in_proj"],
    "conv1d":   tf.GRAD_MULT * n_layers * tokens * t["conv"],
    "out_proj": tf.GRAD_MULT * n_layers * tokens * t["out_proj"],
    "mlp":      tf.GRAD_MULT * n_layers * tokens * t["mlp"],
    "scan":     tf.GRAD_MULT * n_layers * tokens * t["scan"],
    "head":     tf.GRAD_MULT * tokens * t["head"],
  }
  print(f"uSSM-AR  L={a.length} batch={a.batch}  tokens={tokens}\n")
  print(f"{'counted module':<26}{'GFLOP':>12}")
  print("-" * 38)
  for k in sorted(by_kind, key=lambda k: -by_kind[k]):
    if by_kind[k]:
      print(f"{k:<26}{by_kind[k]/1e9:>12.2f}")
  print(f"{'TOTAL counted':<26}{total/1e9:>12.2f}")
  print(f"\n{'analytic term':<26}{'GFLOP':>12}")
  print("-" * 38)
  for k, v in sorted(analytic.items(), key=lambda kv: -kv[1]):
    print(f"{k:<26}{v/1e9:>12.2f}")
  an_total = sum(analytic.values())
  print(f"{'TOTAL analytic':<26}{an_total/1e9:>12.2f}")
  print(f"\ncounted / analytic = {total/an_total:.3f}")
  print(f"counted - analytic = {(total-an_total)/1e9:.2f} GFLOP")
  print(f"per token per layer: {(total-an_total)/(tokens*n_layers):,.0f} FLOP")

  a.output.parent.mkdir(parents=True, exist_ok=True)
  a.output.write_text(json.dumps(
    {"length": a.length, "batch": a.batch, "tokens": tokens,
     "counted_by_module": {k: v for k, v in by_kind.items() if v},
     "counted_total": total, "analytic": analytic,
     "analytic_total": an_total}, indent=2, sort_keys=True) + "\n")
  print(f"\nwrote {a.output}")


if __name__ == "__main__":
  main_cli()
