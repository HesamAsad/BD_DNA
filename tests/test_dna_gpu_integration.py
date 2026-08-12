from pathlib import Path

import hydra
import pytest
import torch

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="requires a CUDA GPU")


def _config(arm, *, length=8, block_size=4):
  config_dir = str(Path(__file__).resolve().parents[1] / "configs")
  common = [
    "data=carbon-prokaryote",
    f"model.length={length}",
    "model.hidden_size=8",
    "model.cond_dim=8",
    "model.n_blocks=1",
    "model.dropout=0.0",
    "loader.batch_size=2",
    "loader.eval_batch_size=2",
    "loader.global_batch_size=2",
    "loader.eval_global_batch_size=2",
    "training.ema=0",
    "trainer.precision=bf16",
    "sampling.kv_cache=true",
    "sampling.num_sample_batches=1",
  ]
  if arm in {"ussm_ar", "transformer_ar"}:
    model = "small_ussm" if arm == "ussm_ar" else "small_ar_transformer"
    common.extend((f"model={model}", "algo=ar", "block_size=1"))
    if arm == "ussm_ar":
      common.append("algo.backbone=ussm")
  elif arm in {"ussm_bd", "bissm_bd", "transformer_bd"}:
    model = {
      "ussm_bd": "small_ussm",
      "bissm_bd": "small_bissm",
      "transformer_bd": "small",
    }[arm]
    algo = {
      "ussm_bd": "bd3lm_ussm",
      "bissm_bd": "bd3lm_bissm",
      "transformer_bd": "bd3lm",
    }[arm]
    common.extend((f"model={model}", f"algo={algo}",
                   f"block_size={block_size}"))
  else:
    raise ValueError(arm)

  if arm.startswith(("ussm", "bissm")):
    common.extend((
      "model.ssm_state_size=3",
      "model.ssm_conv_size=4",
      "model.ssm_expand=2",
      "model.ssm_head_dim=4",
      "model.ssm_chunk_size=4",
      "model.ssm_backend=torch",
      "model.mlp_ratio=2.0",
      "model.active_blocks=all",
      "model.right_flank_probability=0.0",
    ))
  else:
    common.extend(("model.n_heads=2", "model.attn_backend=sdpa"))

  with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    return hydra.compose(config_name="config", overrides=common)


def _output_layer(model):
  if hasattr(model.backbone, "output"):
    return model.backbone.output
  return model.backbone.output_layer.linear


@pytest.mark.parametrize(
  "arm", ["ussm_ar", "transformer_ar", "ussm_bd", "bissm_bd",
          "transformer_bd"])
def test_real_training_and_evaluation_objectives_use_bfloat16(arm):
  torch.manual_seed(41)
  model = Diffusion(_config(arm), DNATokenizer()).cuda()
  x0 = torch.randint(8, 12, (2, 8), device="cuda")
  attention_mask = torch.ones_like(x0)
  observed = []
  hook = _output_layer(model).register_forward_hook(
    lambda _module, _inputs, output: observed.append(output.dtype))

  model.train()
  train_loss = model._loss(x0, attention_mask).loss
  train_loss.backward()
  assert observed and set(observed) == {torch.bfloat16}

  observed.clear()
  model.eval()
  with torch.inference_mode():
    eval_loss = model._loss(x0, attention_mask).loss
  assert torch.isfinite(eval_loss)
  assert observed and set(observed) == {torch.bfloat16}
  hook.remove()


@pytest.mark.parametrize(
  ("arm", "length", "block_size", "num_steps"),
  [("ussm_ar", 300, 1, None), ("ussm_bd", 320, 32, 32)])
def test_native_dna_sampling_runs_beyond_256_tokens(
    arm, length, block_size, num_steps):
  torch.manual_seed(53)
  config = _config(arm, length=length, block_size=block_size)
  config.loader.eval_batch_size = 1
  config.loader.eval_global_batch_size = 1
  tokenizer = DNATokenizer()
  model = Diffusion(config, tokenizer).cuda().eval()
  observed = []
  hook = _output_layer(model).register_forward_hook(
    lambda _module, _inputs, output: observed.append(output.dtype))

  samples = model._sample(
    seqlen=length, num_steps=num_steps, batch_size_per_gpu=1)

  assert len(samples) == 1
  assert len(samples[0]) == length
  assert set(samples[0]).issubset(set(tokenizer.characters))
  assert observed and set(observed) == {torch.bfloat16}
  hook.remove()
