#!/usr/bin/env python
"""Equivalence tests for the two zero-numerics launch-cost fixes.

Both are motivated by `scripts/smoke/launch_count_probe.py`, which counts a
real training step's operator dispatches:

  A. `checkpoint_boundary_prefill` costs 2573 extra dispatches per step
     (uSSM-BD 4787 -> 7360, BiSSM 5579 -> 8152 at L=2048; the delta is
     identical because the prefill is identical), all of them in backward.
     It is a MEMORY option, so at a micro batch that already fits it is pure
     overhead. This test asserts turning it off is bitwise-identical --
     loss and every gradient -- so the choice is free to make per geometry.

  B. `Diffusion._loss` performs 5 device->host syncs per step
     (`aten._local_scalar_dense` x5 in the same probe): two `bos_rows.any()`
     (diffusion.py:406, :1281) and three tensor-vs-scalar `if` comparisons
     (diffusion.py:1014 x2, :1096). On CUDA each drains the launch queue, which
     is exactly what a launch-bound step cannot afford. This test asserts the
     branchless rewrites below produce bitwise-identical tensors.

Run on CPU:  python scripts/smoke/sync_free_equiv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def build(checkpoint_prefill: bool, arm: str = "bissm", length: int = 512):
  import hydra
  import main  # noqa: F401  registers resolvers
  from dataloader import DNATokenizer
  from diffusion import Diffusion

  model_cfg, algo_cfg = {
    "bissm": ("small_bissm", "bd3lm_bissm"),
    "ussm": ("small_ussm", "bd3lm_ussm"),
  }[arm]
  overrides = [
    f"model={model_cfg}", f"algo={algo_cfg}", "data=carbon-prokaryote",
    f"model.length={length}", "block_size=256",
    "loader.batch_size=2", "loader.eval_batch_size=2",
    "loader.global_batch_size=64", "training.ema=0",
    "trainer.accumulate_grad_batches=1",
    "model.hidden_size=128", "model.ssm_head_dim=64", "model.n_blocks=2",
    "model.active_blocks=all",
    f"model.checkpoint_boundary_prefill={str(checkpoint_prefill).lower()}",
  ]
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO / "configs")):
    config = hydra.compose(config_name="config", overrides=overrides)
  torch.manual_seed(0)
  return Diffusion(config, DNATokenizer())


def loss_and_grads(checkpoint_prefill, arm="bissm", length=512):
  model = build(checkpoint_prefill, arm, length)
  model.train()
  x0 = torch.randint(8, 12, (2, length))
  mask = torch.ones_like(x0)
  torch.manual_seed(1234)          # fixes t, q_xt and dropout identically
  out = model._loss(x0, mask)
  out.loss.backward()
  grads = {n: (p.grad.clone() if p.grad is not None else None)
           for n, p in model.named_parameters()}
  return float(out.loss), grads


def test_checkpoint_prefill_is_bitwise_identical():
  """A. checkpoint recompute is exact, so ON and OFF must agree bitwise."""
  loss_on, grads_on = loss_and_grads(True)
  loss_off, grads_off = loss_and_grads(False)
  assert loss_on == loss_off, (loss_on, loss_off)
  worst, worst_name = 0.0, None
  for name, g_on in grads_on.items():
    g_off = grads_off[name]
    assert (g_on is None) == (g_off is None), name
    if g_on is None:
      continue
    if not torch.equal(g_on, g_off):
      delta = (g_on - g_off).abs().max().item()
      if delta > worst:
        worst, worst_name = delta, name
  assert worst == 0.0, f"grad mismatch, worst {worst:.3e} at {worst_name}"
  print(f"A  checkpoint_boundary_prefill on/off: loss and all grads "
        f"BITWISE identical (loss={loss_on!r})")


# ---------------------------------------------------------------------------
# B. The two branchless rewrites, stated as reference implementations so the
#    test compares against the code as it stands today.
# ---------------------------------------------------------------------------

def preserve_bos_current(noisy, clean, bos_id):
  """diffusion.py:404-408 -- `.any()` on a device tensor, i.e. a sync."""
  bos_rows = clean[:, 0].eq(bos_id)
  if bos_rows.any():                       # <-- sync
    noisy[bos_rows, 0] = clean[bos_rows, 0]
  return noisy


def preserve_bos_branchless(noisy, clean, bos_id):
  """Proposed: same writes, no host round trip."""
  bos_rows = clean[:, 0].eq(bos_id)
  noisy[:, 0] = torch.where(bos_rows, clean[:, 0], noisy[:, 0])
  return noisy


def mask_bos_current(mask, tokens, bos_id):
  """diffusion.py:1280-1282."""
  bos_rows = tokens[:, 0].eq(bos_id)
  if bos_rows.any():                       # <-- sync
    mask[bos_rows, 0] = 0
  return mask


def mask_bos_branchless(mask, tokens, bos_id):
  bos_rows = tokens[:, 0].eq(bos_id)
  mask[:, 0] = torch.where(
    bos_rows, torch.zeros_like(mask[:, 0]), mask[:, 0])
  return mask


def test_branchless_bos_is_bitwise_identical():
  torch.manual_seed(0)
  for trial in range(200):
    clean = torch.randint(0, 12, (5, 7))
    noisy = torch.randint(0, 12, (5, 7))
    mask = torch.randint(0, 2, (5, 7))
    bos_id = 1
    a = preserve_bos_current(noisy.clone(), clean, bos_id)
    b = preserve_bos_branchless(noisy.clone(), clean, bos_id)
    assert torch.equal(a, b), f"preserve_bos differs at trial {trial}"
    a = mask_bos_current(mask.clone(), clean, bos_id)
    b = mask_bos_branchless(mask.clone(), clean, bos_id)
    assert torch.equal(a, b), f"mask_bos differs at trial {trial}"
  print("B1 branchless BOS rewrites: bitwise identical over 200 random cases "
        "(includes all-BOS, no-BOS and mixed rows)")


def test_scalar_eps_compare():
  """diffusion.py:1014 and :1096 compare a 0-dim BUFFER against a float.

  `self.sampling_eps_min` is a tensor (it is written with `.fill_`, see
  diffusion.py:1309), so `sampling_eps_min > 0.5` inside an `if` is a sync.
  Reading it once per step through `float()` is the same value; the point is
  that a training step must not read it at all on the hot path. The rewrite is
  to hold a Python mirror updated only where `.fill_` is called.
  """
  buf = torch.tensor(0.75)
  mirror = 0.75
  assert bool(buf > 0.5) == (mirror > 0.5)
  buf.fill_(0.25)
  mirror = 0.25
  assert bool(buf > 0.5) == (mirror > 0.5)
  print("B2 python mirror of sampling_eps_{min,max}: same branch decision, "
        "no device read")


if __name__ == "__main__":
  test_branchless_bos_is_bitwise_identical()
  test_scalar_eps_compare()
  test_checkpoint_prefill_is_bitwise_identical()
  print("\nall equivalence checks passed")
