from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer, _assert_bissm_compat
from diffusion import Diffusion
from models.bidirectional_ssm import BidirectionalSSM
from models.unidirectional_ssm import UnidirectionalSSM


def _unit_config(parameterization="subs", time_conditioning=False):
  return OmegaConf.create({
    "block_size": 4,
    "algo": {
      "parameterization": parameterization,
      "time_conditioning": time_conditioning,
    },
    "model": {
      "hidden_size": 8,
      "cond_dim": 8,
      "n_blocks": 2,
      "dropout": 0.0,
      "tie_word_embeddings": True,
      "right_flank_probability": 0.0,
      "ssm_state_size": 3,
      "ssm_conv_size": 4,
      "ssm_expand": 2,
      "ssm_head_dim": 4,
      "ssm_chunk_size": 4,
      "ssm_backend": "torch",
      "mlp_ratio": 2.0,
    },
  })


def _unit_model(parameterization="subs"):
  torch.manual_seed(17)
  return UnidirectionalSSM(
    _unit_config(parameterization=parameterization), vocab_size=13).eval()


def _integration_config(objective="bd3lm"):
  config_dir = str(Path(__file__).resolve().parents[1] / "configs")
  overrides = [
    "model=small_ussm",
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
    "loader.batch_size=2",
    "loader.eval_batch_size=2",
    "training.ema=0",
    "sampling.kv_cache=true",
  ]
  if objective == "bd3lm":
    overrides.extend(("algo=bd3lm_ussm", "block_size=4"))
  elif objective == "ar":
    overrides.extend(("algo=ar", "algo.backbone=ussm", "block_size=1"))
  else:
    raise ValueError(objective)
  with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    return hydra.compose(config_name="config", overrides=overrides)


def test_unidirectional_and_bidirectional_models_are_parameter_matched():
  config = _unit_config()
  uni = UnidirectionalSSM(config, vocab_size=13)
  bi = BidirectionalSSM(config, vocab_size=13)

  assert sum(p.numel() for p in uni.parameters()) == sum(
    p.numel() for p in bi.parameters())
  assert set(uni.state_dict()) == set(bi.state_dict())


def test_future_active_tokens_do_not_change_earlier_logits():
  model = _unit_model()
  active = torch.randint(0, 13, (2, 4))
  changed = active.clone()
  changed[:, -1] = (changed[:, -1] + 1) % 13

  original = model.forward_active(active, None)
  perturbed = model.forward_active(changed, None)

  torch.testing.assert_close(original[:, :-1], perturbed[:, :-1])
  assert not torch.allclose(original[:, -1], perturbed[:, -1])


def test_cached_causal_continuation_matches_one_shot_logits():
  model = _unit_model(parameterization="ar")
  tokens = torch.randint(0, 13, (2, 9))
  expected = model.forward_active(tokens, None)

  model.reset_kv_cache()
  pieces = []
  for end in range(1, tokens.shape[1] + 1):
    pieces.append(model(tokens[:, :end], None, store_kv=True))
  actual = torch.cat(pieces, dim=1)

  torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
  assert model._sampling_left_cache.length == tokens.shape[1]


def test_block_diffusion_cache_advances_only_when_block_is_committed():
  model = _unit_model()
  prefix = torch.randint(0, 13, (2, 4))
  active = torch.randint(0, 13, (2, 4))
  model._sampling_left_cache = model.prefill_left(prefix, detach=True)

  first = model(active, None, sample_mode=True)
  second = model(active, None, sample_mode=True)

  torch.testing.assert_close(first, second)
  assert model._sampling_left_cache.length == prefix.shape[1]

  committed = model(
    active, None, sample_mode=True, store_kv=True)
  torch.testing.assert_close(first, committed)
  assert model._sampling_left_cache.length == (
    prefix.shape[1] + active.shape[1])


def test_unidirectional_block_diffusion_objective_smoke():
  torch.manual_seed(23)
  config = _integration_config("bd3lm")
  _assert_bissm_compat(config)
  model = Diffusion(config, DNATokenizer())
  x0 = torch.randint(8, 12, (2, 8))
  t = torch.full_like(x0, 0.5, dtype=torch.float32)

  loss = model._forward_pass_diffusion(
    x0, t=t, sampling_eps_min=1e-3, sampling_eps_max=1.0)

  assert loss.shape == x0.shape
  assert torch.isfinite(loss).all()
  assert (loss.reshape(2, 2, 4).abs().sum(dim=-1) > 0).all()
  loss.mean().backward()
  assert model.backbone.token_embedding.weight.grad is not None


def test_unidirectional_autoregressive_objective_smoke():
  torch.manual_seed(29)
  config = _integration_config("ar")
  _assert_bissm_compat(config)
  model = Diffusion(config, DNATokenizer())
  x0 = torch.randint(8, 12, (2, 8))
  attention_mask = torch.ones_like(x0)

  loss = model._loss(x0, attention_mask)

  assert torch.isfinite(loss.loss)
  assert loss.nlls.shape == (2, 7)
  loss.loss.backward()
  assert model.backbone.token_embedding.weight.grad is not None
