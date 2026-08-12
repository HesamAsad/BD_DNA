#!/usr/bin/env python
"""Correctness + speed check for the layer-major boundary-cache rewrite.

Runs at the production geometry (L=8192, block 256, 12 layers, width 768) on
one GPU under the fused Mamba-2 backend and BF16 autocast, and reports:

* max |layer-major - block-major| for every boundary state, forward and
  backward, so the rewrite is shown to be equivalent on the kernel that
  training actually uses rather than only on the CPU reference scan;
* steady-state wall clock and peak memory for both paths.

The block-major path is `_boundary_caches_sequential`, kept in the model as
the equivalence oracle.
"""

import argparse
import time

import torch
from omegaconf import OmegaConf

from models.bidirectional_ssm import BidirectionalSSM, stack_boundary_caches
from models.mamba2_segment import fused_mamba2_available


def build_model(args, device):
  config = OmegaConf.create({
    "block_size": args.block_size,
    "algo": {"time_conditioning": False},
    "model": {
      "hidden_size": args.hidden_size,
      "cond_dim": 128,
      "n_blocks": args.layers,
      "dropout": 0.0,
      "tie_word_embeddings": True,
      "ssm_state_size": 64,
      "ssm_conv_size": 4,
      "ssm_expand": 2,
      "ssm_head_dim": 64,
      "ssm_chunk_size": 128,
      "ssm_backend": args.backend,
      "mlp_ratio": 4.0,
    },
  })
  torch.manual_seed(0)
  return BidirectionalSSM(config, vocab_size=args.vocab).to(device)


def autocast(device):
  return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def boundary_loss(model, build, clean, noisy, args):
  """One BD3-LM-shaped step: build every cache, denoise every block."""
  num_blocks = clean.shape[1] // args.block_size
  with autocast(clean.device):
    cache = stack_boundary_caches(build(clean, args.block_size, "left"))
    logits = model.forward_active(
      noisy.reshape(noisy.shape[0] * num_blocks, args.block_size),
      None,
      left_cache=cache)
  return logits.float().square().mean(), cache


def compare(model, clean, noisy, args):
  results = {}
  for name, build in (("layer_major", model._boundary_caches),
                      ("block_major", model._boundary_caches_sequential)):
    model.zero_grad(set_to_none=True)
    loss, cache = boundary_loss(model, build, clean, noisy, args)
    loss.backward()
    results[name] = {
      "loss": loss.detach().float().item(),
      "ssm": [state.ssm.detach().float() for state in cache.states],
      "conv": [state.conv.detach().float() for state in cache.states],
      "grad": {n: p.grad.detach().float().clone()
               for n, p in model.named_parameters() if p.grad is not None},
    }
  new, old = results["layer_major"], results["block_major"]

  print(f"loss                 layer_major={new['loss']:.8f} "
        f"block_major={old['loss']:.8f} "
        f"rel_diff={abs(new['loss'] - old['loss']) / max(abs(old['loss']), 1e-12):.3e}")
  for field in ("ssm", "conv"):
    worst = max((a - b).abs().max().item()
                for a, b in zip(new[field], old[field]))
    scale = max(b.abs().max().item() for b in old[field])
    print(f"cache.{field:<15s} max_abs_diff={worst:.3e}  "
          f"(state max_abs={scale:.3e})")
  worst_name, worst_rel = None, 0.0
  for name, grad in old["grad"].items():
    denominator = max(grad.abs().max().item(), 1e-12)
    relative = (new["grad"][name] - grad).abs().max().item() / denominator
    if relative > worst_rel:
      worst_name, worst_rel = name, relative
  print(f"grad                 worst_rel_diff={worst_rel:.3e} at {worst_name}")
  return worst_rel


def benchmark(model, clean, noisy, args):
  print(f"\n{'path':<14s}{'fwd+bwd s':>12s}{'peak GiB':>11s}{'speedup':>10s}")
  baseline = None
  for name, build in (("block_major", model._boundary_caches_sequential),
                      ("layer_major", model._boundary_caches)):
    for _ in range(args.warmup):
      model.zero_grad(set_to_none=True)
      boundary_loss(model, build, clean, noisy, args)[0].backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.iters):
      model.zero_grad(set_to_none=True)
      boundary_loss(model, build, clean, noisy, args)[0].backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / args.iters
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    baseline = baseline if baseline is not None else elapsed
    print(f"{name:<14s}{elapsed:>12.4f}{peak:>11.2f}{baseline / elapsed:>9.2f}x")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--hidden-size", type=int, default=768)
  parser.add_argument("--layers", type=int, default=12)
  parser.add_argument("--vocab", type=int, default=16)
  parser.add_argument("--backend", default="auto")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--tolerance", type=float, default=5e-2)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise SystemExit("This benchmark needs a GPU")
  device = torch.device("cuda")
  print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
        f"fused_mamba2={fused_mamba2_available()}")
  print(f"L={args.length} block={args.block_size} "
        f"blocks={args.length // args.block_size} batch={args.batch_size} "
        f"width={args.hidden_size} layers={args.layers}\n")

  model = build_model(args, device)
  print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
  torch.manual_seed(1)
  clean = torch.randint(0, args.vocab, (args.batch_size, args.length),
                        device=device)
  noisy = torch.randint(0, args.vocab, (args.batch_size, args.length),
                        device=device)

  worst = compare(model, clean, noisy, args)
  benchmark(model, clean, noisy, args)
  if worst > args.tolerance:
    raise SystemExit(
      f"FAIL: worst relative gradient difference {worst:.3e} "
      f"exceeds {args.tolerance:.3e}")
  print("\nPASS: layer-major matches block-major within tolerance")


if __name__ == "__main__":
  main()
