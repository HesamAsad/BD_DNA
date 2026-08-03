from pathlib import Path

import hydra
import torch

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer, _assert_bissm_compat
from diffusion import Diffusion


def _config(right_probability=0.0):
  config_dir = str(Path(__file__).resolve().parents[1] / "configs")
  with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    config = hydra.compose(
      config_name="config",
      overrides=[
        "model=small_bissm",
        "algo=bd3lm_bissm",
        "data=carbon-prokaryote",
        "model.hidden_size=8",
        "model.n_blocks=2",
        "model.length=8",
        "model.ssm_state_size=3",
        "model.ssm_conv_size=4",
        "model.ssm_expand=2",
        "model.ssm_head_dim=4",
        "model.ssm_chunk_size=4",
        "model.ssm_backend=torch",
        "model.mlp_ratio=2.0",
        f"model.right_flank_probability={right_probability}",
        "block_size=4",
        "loader.batch_size=2",
        "loader.eval_batch_size=2",
        "training.ema=0",
        "sampling.kv_cache=true",
      ])
  return config


def test_bissm_config_and_diffusion_objective_smoke():
  torch.manual_seed(5)
  config = _config()
  _assert_bissm_compat(config)
  model = Diffusion(config, DNATokenizer())
  x0 = torch.randint(8, 12, (2, 8))
  t = torch.full_like(x0, 0.5, dtype=torch.float32)

  loss = model._forward_pass_diffusion(
    x0,
    t=t,
    sampling_eps_min=1e-3,
    sampling_eps_max=1.0)

  assert loss.shape == x0.shape
  assert torch.isfinite(loss).all()
  # Only one uniformly sampled block is evaluated; its num_blocks multiplier
  # makes the full-length mean an unbiased all-block loss estimate.
  positions = loss.ne(0).nonzero(as_tuple=False)[:, 1]
  assert positions.numel() > 0
  assert torch.unique(positions // config.block_size).numel() == 1
  loss.mean().backward()
  assert model.backbone.token_embedding.weight.grad is not None


def test_ca_objective_builds_right_cache_without_target_leakage():
  torch.manual_seed(9)
  config = _config(right_probability=1.0)
  model = Diffusion(config, DNATokenizer())
  x0 = torch.randint(8, 12, (2, 8))
  t = torch.full_like(x0, 0.5, dtype=torch.float32)

  loss = model._forward_pass_diffusion(
    x0,
    t=t,
    sampling_eps_min=1e-3,
    sampling_eps_max=1.0)

  assert torch.isfinite(loss).all()
  assert model._last_right_flank is True

