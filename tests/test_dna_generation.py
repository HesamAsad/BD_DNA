from pathlib import Path

import hydra
import torch

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion


def _config(objective="bd3lm", *, var_length=False):
  config_dir = str(Path(__file__).resolve().parents[1] / "configs")
  overrides = [
    "model=small_ussm",
    "data=carbon-prokaryote",
    "model.hidden_size=8",
    "model.n_blocks=1",
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
    f"sampling.var_length={str(var_length).lower()}",
  ]
  if objective == "ar":
    overrides.extend(("algo=ar", "algo.backbone=ussm", "block_size=1"))
  else:
    overrides.extend(("algo=bd3lm_ussm", "block_size=4"))
  with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    return hydra.compose(config_name="config", overrides=overrides)


def test_dna_generation_alphabet_excludes_every_special_token():
  tokenizer = DNATokenizer()

  assert tuple(tokenizer.convert_ids_to_tokens(
    tokenizer.generation_token_ids)) == tuple("ACGTN")
  assert not set(tokenizer.generation_token_ids).intersection(
    tokenizer.all_special_ids)


def test_special_free_dna_training_does_not_inject_bos_at_generation():
  tokenizer = DNATokenizer()
  ar_model = Diffusion(_config("ar"), tokenizer)
  bd_model = Diffusion(_config("bd3lm"), tokenizer)

  assert not ar_model.prepend_bos
  assert not bd_model.prepend_bos
  ar_tokens = ar_model._initialize_ar_tokens(32, 8)
  assert set(ar_tokens[:, 0].tolist()).issubset(
    set(tokenizer.generation_token_ids))
  assert tokenizer.bos_token_id not in ar_tokens[:, 0]
  diffusion_tokens = bd_model._initialize_diffusion_tokens(2, 8)
  assert torch.all(diffusion_tokens == bd_model.mask_index)


def test_ignore_bos_applies_only_to_rows_that_really_start_with_bos():
  tokenizer = DNATokenizer()
  model = Diffusion(_config("bd3lm"), tokenizer)
  a_id = tokenizer.convert_tokens_to_ids("A")
  x0 = torch.tensor([
    [a_id, a_id, a_id, a_id],
    [tokenizer.bos_token_id, a_id, a_id, a_id],
  ])
  xt = torch.full_like(x0, model.mask_index)

  assert model._bos_rows(x0).tolist() == [False, True]
  actual = model._preserve_observed_bos(xt, x0)
  assert actual[0, 0] == model.mask_index
  assert actual[1, 0] == tokenizer.bos_token_id


def test_first_nucleotide_remains_in_the_actual_evaluation_token_mask():
  tokenizer = DNATokenizer()
  model = Diffusion(_config("bd3lm"), tokenizer).eval()
  a_id = tokenizer.convert_tokens_to_ids("A")
  x0 = torch.full((2, 8), a_id, dtype=torch.long)
  x0[1, 0] = tokenizer.bos_token_id

  losses = model._loss(x0, torch.ones_like(x0))

  assert losses.token_mask[0, 0] == 1
  assert losses.token_mask[1, 0] == 0


def test_fixed_length_dna_is_not_rejected_after_256_low_entropy_tokens():
  tokenizer = DNATokenizer()
  model = Diffusion(_config("bd3lm"), tokenizer)
  a_id = tokenizer.convert_tokens_to_ids("A")
  sequence = torch.full((2, 300), a_id, dtype=torch.long)

  stop, actual = model._check_stop_conds(sequence)

  assert not stop
  torch.testing.assert_close(actual, sequence)


def test_variable_length_stopping_is_per_row_and_uses_first_eos():
  tokenizer = DNATokenizer()
  model = Diffusion(_config("bd3lm", var_length=True), tokenizer)
  a_id = tokenizer.convert_tokens_to_ids("A")
  c_id = tokenizer.convert_tokens_to_ids("C")
  eos = tokenizer.eos_token_id
  pad = tokenizer.pad_token_id
  partial = torch.tensor([
    [a_id, eos, c_id, c_id, c_id, c_id],
    [c_id, c_id, c_id, c_id, c_id, c_id],
  ])

  stop, sanitized = model._check_stop_conds(partial)
  assert not stop
  assert sanitized.shape == partial.shape
  assert sanitized[0].tolist() == [a_id, eos, pad, pad, pad, pad]
  torch.testing.assert_close(sanitized[1], partial[1])

  completed = partial.clone()
  completed[1, 3] = eos
  stop, sanitized = model._check_stop_conds(completed)
  assert stop
  assert sanitized.shape == (2, 4)
  assert sanitized[0].tolist() == [a_id, eos, pad, pad]
  assert sanitized[1, -1] == eos


def test_sampling_probabilities_are_limited_to_dna_content_alphabet():
  tokenizer = DNATokenizer()
  model = Diffusion(_config("bd3lm"), tokenizer)
  probabilities = torch.ones(2, 3, tokenizer.vocab_size)

  restricted = model._restrict_generation_probs(probabilities)

  allowed = set(tokenizer.generation_token_ids)
  for token_id in range(tokenizer.vocab_size):
    if token_id in allowed:
      assert torch.all(restricted[..., token_id] > 0)
    else:
      assert torch.all(restricted[..., token_id] == 0)
  torch.testing.assert_close(
    restricted.sum(dim=-1), torch.ones(2, 3))


def test_different_variable_length_batches_decode_without_tensor_concat():
  tokenizer = DNATokenizer()
  model = Diffusion(_config("bd3lm", var_length=True), tokenizer)
  a_id = tokenizer.convert_tokens_to_ids("A")
  c_id = tokenizer.convert_tokens_to_ids("C")
  batches = [
    torch.tensor([[a_id, tokenizer.eos_token_id]]),
    torch.tensor([[c_id, c_id, tokenizer.eos_token_id]]),
  ]

  assert model._decode_sample_batches(batches) == ["A", "CC"]
