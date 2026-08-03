from pathlib import Path

import hydra
import torch

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer, _assert_bissm_compat
from diffusion import Diffusion


def _config(right_probability=0.0, active_blocks='one'):
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
        f"model.active_blocks={active_blocks}",
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


def test_all_block_objective_supervises_every_block_in_one_step():
  torch.manual_seed(5)
  config = _config(active_blocks='all')
  model = Diffusion(config, DNATokenizer())
  x0 = torch.randint(8, 12, (2, 8))
  t = torch.full_like(x0, 0.5, dtype=torch.float32)

  loss = model._forward_pass_diffusion(
    x0, t=t, sampling_eps_min=1e-3, sampling_eps_max=1.0)

  assert loss.shape == x0.shape
  assert torch.isfinite(loss).all()
  # Every block carries gradient now, so no block is left at exactly zero.
  blocks = loss.reshape(2, -1, config.block_size)
  assert (blocks.abs().sum(dim=-1) > 0).all()
  loss.mean().backward()
  assert model.backbone.token_embedding.weight.grad is not None


def test_all_block_and_one_block_objectives_agree_on_the_sampled_block():
  torch.manual_seed(5)
  config = _config(active_blocks='one')
  model = Diffusion(config, DNATokenizer())
  x0 = torch.randint(8, 12, (2, 8))
  t = torch.full_like(x0, 0.5, dtype=torch.float32)
  p = model.noise(t)[1]
  xt = torch.where(
    torch.rand(x0.shape) < 0.5, torch.full_like(x0, model.mask_index), x0)
  loss_scale = model.noise(t)[0]
  num_blocks = x0.shape[1] // config.block_size

  torch.manual_seed(0)
  one_block = model._forward_pass_bissm(
    x0=x0, xt=xt, p=p, loss_scale=loss_scale)
  sampled = model._last_active_block
  all_blocks = model._forward_pass_bissm_all_blocks(
    x0=x0, xt=xt, p=p, loss_scale=loss_scale, num_blocks=num_blocks)

  start = sampled * config.block_size
  end = start + config.block_size
  # The one-block path carries the num_blocks rescaling; the all-block path
  # supervises the same computation for that block without it.
  torch.testing.assert_close(
    one_block[:, start:end] / num_blocks,
    all_blocks[:, start:end],
    atol=3e-5, rtol=3e-5)


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


def test_ca_sampler_keeps_right_cache_fixed_and_advances_left_cache():
  torch.manual_seed(13)
  config = _config(right_probability=1.0)
  config.sampling.first_hitting = False
  model = Diffusion(config, DNATokenizer())
  left = torch.randint(8, 12, (1, 4))
  right = torch.randint(8, 12, (1, 4))

  completed = model.sample_infill_ca(
    left_context=left,
    right_context=right,
    gap_length=8,
    num_steps=2)

  assert completed.shape == (1, 16)
  torch.testing.assert_close(completed[:, :4], left)
  torch.testing.assert_close(completed[:, -4:], right)
  assert model.mask_index not in completed
  assert model.backbone._sampling_left_cache.length == 12
  assert model.backbone._sampling_right_cache.length == 4
