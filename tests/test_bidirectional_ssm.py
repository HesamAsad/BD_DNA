import pytest
import torch
from omegaconf import OmegaConf

from models.bidirectional_ssm import BidirectionalSSM, stack_boundary_caches


def _config(time_conditioning=False):
  return OmegaConf.create({
    "block_size": 4,
    "algo": {"time_conditioning": time_conditioning},
    "model": {
      "hidden_size": 8,
      "cond_dim": 8,
      "n_blocks": 2,
      "dropout": 0.0,
      "tie_word_embeddings": True,
      "ssm_state_size": 3,
      "ssm_conv_size": 4,
      "ssm_expand": 2,
      "ssm_head_dim": 4,
      "ssm_chunk_size": 4,
      "ssm_backend": "torch",
      "mlp_ratio": 2.0,
    },
  })


def _model(time_conditioning=False):
  torch.manual_seed(11)
  return BidirectionalSSM(_config(time_conditioning), vocab_size=13).eval()


def _assert_cache_close(actual, expected):
  assert actual.length == expected.length
  assert actual.direction == expected.direction
  for actual_state, expected_state in zip(actual.states, expected.states):
    torch.testing.assert_close(actual_state.conv, expected_state.conv)
    torch.testing.assert_close(
      actual_state.ssm, expected_state.ssm, atol=3e-6, rtol=3e-6)


def test_sequential_commits_match_one_shot_clean_prefix():
  model = _model()
  prefix = torch.randint(0, 13, (2, 9))

  expected = model.prefill_left(prefix)
  first = model.prefill_left(prefix[:, :4])
  actual = model.prefill_left(prefix[:, 4:], first)

  _assert_cache_close(actual, expected)


def test_active_denoising_does_not_mutate_clean_caches():
  model = _model()
  left = model.prefill_left(torch.randint(0, 13, (2, 6)))
  right = model.prefill_right(torch.randint(0, 13, (2, 5)))
  left_snapshot = left.clone()
  right_snapshot = right.clone()

  noisy = torch.randint(0, 13, (2, 4))
  model.forward_active(noisy, None, left, right)
  model.forward_active(torch.flip(noisy, dims=(1,)), None, left, right)

  _assert_cache_close(left, left_snapshot)
  _assert_cache_close(right, right_snapshot)


def test_known_right_flank_changes_conditional_logits():
  model = _model()
  left = model.prefill_left(torch.randint(0, 13, (2, 6)))
  suffix = torch.randint(0, 13, (2, 7))
  right = model.prefill_right(suffix)
  noisy = torch.randint(0, 13, (2, 4))

  de_novo = model.forward_active(noisy, None, left_cache=left)
  infill = model.forward_active(
    noisy, None, left_cache=left, right_cache=right)

  assert not torch.allclose(de_novo, infill)


def test_reverse_active_scan_propagates_within_block():
  model = _model()
  noisy = torch.randint(0, 13, (1, 4))
  changed_future = noisy.clone()
  changed_future[:, -1] = (changed_future[:, -1] + 1) % 13

  original_logits = model.forward_active(noisy, None)
  changed_logits = model.forward_active(changed_future, None)

  # Position zero can only see the changed final token through the active
  # block's reverse scan.
  assert not torch.allclose(original_logits[:, 0], changed_logits[:, 0])


def test_timestep_only_changes_active_path_not_clean_cache():
  model = _model(time_conditioning=True)
  prefix = torch.randint(0, 13, (2, 6))
  cache = model.prefill_left(prefix)
  snapshot = cache.clone()
  noisy = torch.randint(0, 13, (2, 4))

  low_noise = model.forward_active(noisy, torch.tensor([0.1, 0.1]), cache)
  high_noise = model.forward_active(noisy, torch.tensor([0.9, 0.9]), cache)

  assert not torch.allclose(low_noise, high_noise)
  _assert_cache_close(cache, snapshot)


def test_left_boundaries_match_per_block_prefills():
  model = _model()
  clean = torch.randint(0, 13, (2, 12))

  boundaries = model.prefill_left_boundaries(clean, block_size=4)

  assert len(boundaries) == 3
  assert boundaries[0].length == 0
  for index in range(1, 3):
    expected = model.prefill_left(clean[:, :index * 4])
    _assert_cache_close(boundaries[index], expected)


def test_right_boundaries_hold_only_the_suffix_after_each_block():
  model = _model()
  clean = torch.randint(0, 13, (2, 12))

  boundaries = model.prefill_right_boundaries(clean, block_size=4)

  assert len(boundaries) == 3
  # The final block has no clean suffix; earlier blocks see everything to
  # their right but never their own tokens.
  assert boundaries[-1].length == 0
  for index in range(2):
    expected = model.prefill_right(clean[:, (index + 1) * 4:])
    _assert_cache_close(boundaries[index], expected)


def test_stacked_boundary_caches_denoise_all_blocks_at_once():
  model = _model()
  clean = torch.randint(0, 13, (2, 12))
  noisy = torch.randint(0, 13, (2, 12))
  block_size, num_blocks = 4, 3

  folded = model.forward_active(
    noisy.reshape(2 * num_blocks, block_size),
    None,
    left_cache=stack_boundary_caches(
      model.prefill_left_boundaries(clean, block_size)),
    right_cache=stack_boundary_caches(
      model.prefill_right_boundaries(clean, block_size)))
  folded = folded.reshape(2, num_blocks, block_size, -1)

  for index in range(num_blocks):
    start = index * block_size
    expected = model.forward_active(
      noisy[:, start:start + block_size],
      None,
      left_cache=model.prefill_left(clean[:, :start]),
      right_cache=model.prefill_right(clean[:, start + block_size:]))
    torch.testing.assert_close(
      folded[:, index], expected, atol=3e-5, rtol=3e-5)


def test_all_block_logits_never_depend_on_present_or_future_clean_tokens():
  model = _model()
  block_size, num_blocks = 4, 3
  clean = torch.randint(0, 13, (1, block_size * num_blocks))
  noisy = torch.randint(0, 13, (1, block_size * num_blocks))

  def all_block_logits(clean_ids):
    return model.forward_active(
      noisy.reshape(num_blocks, block_size),
      None,
      left_cache=stack_boundary_caches(
        model.prefill_left_boundaries(clean_ids, block_size)))

  reference = all_block_logits(clean)
  for changed_position in range(block_size, block_size * num_blocks):
    perturbed = clean.clone()
    perturbed[:, changed_position] = (
      perturbed[:, changed_position] + 1) % 13
    logits = all_block_logits(perturbed)
    # De novo, block i may only be moved by clean tokens strictly before it.
    first_affected = changed_position // block_size + 1
    torch.testing.assert_close(
      logits[:first_affected], reference[:first_affected])
    if first_affected < num_blocks:
      assert not torch.allclose(
        logits[first_affected:], reference[first_affected:])


def test_layer_major_boundaries_match_the_block_major_oracle():
  # The whole point of the layer-major rewrite: same states, two full-length
  # scans per layer instead of one short scan per block per layer.
  model = _model()
  clean = torch.randint(0, 13, (2, 20))

  for direction in ("left", "right"):
    actual = model._boundary_caches(clean, 4, direction)
    expected = model._boundary_caches_sequential(clean, 4, direction)
    assert len(actual) == len(expected) == 5
    for index, (got, want) in enumerate(zip(actual, expected)):
      assert got.length == want.length == index * 4
      _assert_cache_close(got, want)


def test_layer_major_boundaries_survive_a_single_block():
  # num_blocks == 1 scans nothing at all: the only block's cache is empty.
  model = _model()
  clean = torch.randint(0, 13, (2, 4))

  boundaries = model._boundary_caches(clean, 4, "left")
  assert len(boundaries) == 1
  _assert_cache_close(
    boundaries[0], model._boundary_caches_sequential(clean, 4, "left")[0])
  stacked = model.prefill_left_boundaries_stacked(clean, 4)
  _assert_cache_close(stacked, boundaries[0])


def test_stacked_boundaries_match_stacking_the_tuple():
  model = _model()
  clean = torch.randint(0, 13, (3, 16))

  for stacked, tupled in (
      (model.prefill_left_boundaries_stacked(clean, 4),
       stack_boundary_caches(model.prefill_left_boundaries(clean, 4))),
      (model.prefill_right_boundaries_stacked(clean, 4),
       stack_boundary_caches(model.prefill_right_boundaries(clean, 4)))):
    _assert_cache_close(stacked, tupled)


def test_boundary_caches_stay_differentiable_through_every_block():
  # The BD3-LM objective backprops into the clean prefill; the rewrite must
  # not quietly detach it, and every block that has context must receive
  # gradient.
  model = _model()
  clean = torch.randint(0, 13, (1, 16))

  boundaries = model.prefill_left_boundaries(clean, 4)
  # Block 0 has no context. It is now a slice of the stacked tensor rather
  # than a freshly allocated zero, so it carries an (inert) grad_fn; what has
  # to hold is that it is exactly zero and passes no gradient back.
  for state in boundaries[0].states:
    assert not state.ssm.any() and not state.conv.any()
  for cache in boundaries[1:]:
    assert cache.states[0].ssm.requires_grad
    assert cache.states[-1].conv.requires_grad

  model.zero_grad()
  sum(cache.states[-1].ssm.sum() for cache in boundaries[1:]).backward()
  assert model.token_embedding.weight.grad.abs().sum() > 0
  assert model.layers[0].mixer.in_proj.weight.grad.abs().sum() > 0


def test_layer_major_rewrite_reproduces_the_block_major_gradient():
  # Equivalence of the states is not enough: the BD3-LM objective backprops
  # through the prefill, so the rewrite has to reproduce the gradient too.
  model = _model()
  clean = torch.randint(0, 13, (2, 20))
  noisy = torch.randint(0, 13, (2, 20))
  block_size, num_blocks = 4, 5

  def loss_from(boundaries):
    logits = model.forward_active(
      noisy.reshape(2 * num_blocks, block_size),
      None,
      left_cache=stack_boundary_caches(boundaries))
    return logits.square().mean()

  grads = []
  for build in (model._boundary_caches, model._boundary_caches_sequential):
    model.zero_grad(set_to_none=True)
    loss_from(build(clean, block_size, "left")).backward()
    grads.append({
      name: parameter.grad.clone()
      for name, parameter in model.named_parameters()
      if parameter.grad is not None})

  actual, expected = grads
  assert set(actual) == set(expected) and actual
  for name in expected:
    torch.testing.assert_close(
      actual[name], expected[name], atol=2e-5, rtol=2e-5, msg=lambda m: f"{name}: {m}")


def test_boundary_impl_flag_selects_the_block_major_path():
  # The escape hatch has to route the real training entry points, both
  # directions, and produce the same caches as the default.
  clean = torch.randint(0, 13, (2, 16))
  fast = _model()
  slow = _model()
  slow.boundary_impl = "block_major"

  for left in (True, False):
    build = (lambda m: m.prefill_left_boundaries_stacked(clean, 4)) if left \
      else (lambda m: m.prefill_right_boundaries_stacked(clean, 4))
    _assert_cache_close(build(slow), build(fast))

  with pytest.raises(ValueError, match="boundary_impl"):
    config = _config()
    config.model.boundary_impl = "sideways"
    BidirectionalSSM(config, vocab_size=13)


def test_sampler_commits_only_when_store_flag_is_set():
  model = _model()
  block = torch.randint(0, 13, (2, 4))

  model.reset_kv_cache()
  model(block, None, sample_mode=True, store_kv=False)
  assert model._sampling_left_cache is None

  model(block, None, sample_mode=True, store_kv=True)
  assert model._sampling_left_cache.length == block.shape[1]
  first_cache = model._sampling_left_cache.clone()

  model(block, None, sample_mode=True, store_kv=False)
  _assert_cache_close(model._sampling_left_cache, first_cache)

