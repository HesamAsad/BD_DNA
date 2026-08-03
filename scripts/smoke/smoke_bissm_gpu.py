#!/usr/bin/env python
"""GPU acceptance smoke for the production BiSSM path."""

import copy

import torch
from omegaconf import OmegaConf

from models.bidirectional_ssm import BidirectionalSSM
from models.mamba2_segment import SegmentMamba2, fused_mamba2_available


def assert_close(name, actual, expected, atol=3e-2, rtol=3e-2):
  difference = (actual.float() - expected.float()).abs().max().item()
  print(f"{name}: max_abs_diff={difference:.6g}", flush=True)
  torch.testing.assert_close(
    actual.float(), expected.float(), atol=atol, rtol=rtol)


def mixer_smoke(device):
  torch.manual_seed(17)
  reference = SegmentMamba2(
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=32,
    chunk_size=16,
    backend="torch").to(device=device, dtype=torch.bfloat16)
  fused = copy.deepcopy(reference)
  fused.backend = "fused"
  x = torch.randn(2, 31, 64, device=device, dtype=torch.bfloat16)

  with torch.no_grad():
    expected, expected_state = reference.scan_segment(x)
    actual, actual_state = fused.scan_segment(x)
  assert_close("fused_vs_reference.output", actual, expected)
  assert_close("fused_vs_reference.ssm", actual_state.ssm, expected_state.ssm)
  assert_close("fused_vs_reference.conv", actual_state.conv, expected_state.conv)

  with torch.no_grad():
    full, full_state = fused.scan_segment(x)
    first, boundary = fused.scan_segment(x[:, :13])
    second, split_state = fused.scan_segment(x[:, 13:], boundary)
  assert_close("fused_segment.output", torch.cat((first, second), 1), full)
  assert_close("fused_segment.ssm", split_state.ssm, full_state.ssm)
  assert_close("fused_segment.conv", split_state.conv, full_state.conv)


def backbone_smoke(device):
  config = OmegaConf.create({
    "block_size": 64,
    "algo": {"time_conditioning": False},
    "model": {
      "hidden_size": 64,
      "cond_dim": 32,
      "n_blocks": 2,
      "dropout": 0.0,
      "tie_word_embeddings": True,
      "ssm_state_size": 16,
      "ssm_conv_size": 4,
      "ssm_expand": 2,
      "ssm_head_dim": 32,
      "ssm_chunk_size": 16,
      "ssm_backend": "fused",
      "mlp_ratio": 2.0,
    },
  })
  model = BidirectionalSSM(config, vocab_size=13).to(
    device=device, dtype=torch.bfloat16).train()
  prefix = torch.randint(0, 13, (2, 128), device=device)
  active = torch.randint(0, 13, (2, 64), device=device)
  suffix = torch.randint(0, 13, (2, 96), device=device)

  left = model.prefill_left(prefix)
  right = model.prefill_right(suffix)
  left_snapshot = left.clone()
  right_snapshot = right.clone()
  logits = model.forward_active(active, None, left, right)
  loss = logits.float().square().mean()
  loss.backward()

  assert logits.shape == (2, 64, 13)
  assert model.token_embedding.weight.grad is not None
  assert model.token_embedding.weight.grad.abs().sum() > 0
  for actual, expected in zip(left.states, left_snapshot.states):
    assert_close("left_cache.conv", actual.conv, expected.conv, atol=0, rtol=0)
    assert_close("left_cache.ssm", actual.ssm, expected.ssm, atol=0, rtol=0)
  for actual, expected in zip(right.states, right_snapshot.states):
    assert_close("right_cache.conv", actual.conv, expected.conv, atol=0, rtol=0)
    assert_close("right_cache.ssm", actual.ssm, expected.ssm, atol=0, rtol=0)
  print(
    f"backbone: loss={loss.item():.6g} cache_bytes={left.nbytes + right.nbytes}",
    flush=True)


def main():
  assert torch.cuda.is_available(), "CUDA is required"
  assert fused_mamba2_available(), "mamba-ssm fused scan is not importable"
  device = torch.device("cuda")
  print(
    f"torch={torch.__version__} gpu={torch.cuda.get_device_name(device)} "
    f"bf16={torch.cuda.is_bf16_supported()}", flush=True)
  mixer_smoke(device)
  backbone_smoke(device)
  print("BISSM_GPU_SMOKE_OK", flush=True)


if __name__ == "__main__":
  main()

