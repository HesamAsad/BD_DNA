"""Scoring-path parity between the dit and ussm backbones under `algo=ar`.

MaveDB's exact-AR estimator (`score_mavedb._exact_ar_losses`) is called with
the SAME code for both `uSSM-AR` and `Transformer-AR` -- no backbone branch
exists inside it. That makes it easy to assume the two backbones are scored
identically without ever checking the assumption against the real modules.
This file checks it directly: build a tiny real `dit` backbone and a tiny real
`ussm` backbone under `algo=ar`, feed each an IDENTICAL toy sequence through
the ACTUAL `score_mavedb` harness functions, and assert two things neither
existing test covers:

1. The number of scored tokens and the position-0 exclusion convention are
   identical across backbones (they are computed from the tokenizer/mask
   before either model is touched, so this is a floor, not a surprise -- but
   it is exactly what a length-normalisation or off-by-one bug would break).
2. Both backbones are STRICTLY CAUSAL under the harness's own forward call:
   perturbing the token at position p leaves every scored loss at position
   < p bit-for-bit unchanged. This is the invariant `_exact_ar_losses`'s
   next-token alignment assumes and that a masking or position-encoding bug
   in either backbone would violate.

CPU only. `attn_backend='sdpa'` on the causal DiT block routes to
`DDiTBlockCausal.cross_attn`, which calls `F.scaled_dot_product_attention(...,
is_causal=True)` when the adaLN condition is None (true for every `algo=ar`
block, since `causal_attention=True` forces `adaLN=False` unless
`model.adaln` is explicitly set) -- the same masking semantics as the
`flash_attn` backend the real checkpoints use, just a CPU-runnable kernel.
`ssm_backend='torch'` is the existing CPU fallback used by
tests/test_unidirectional_ssm.py.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion
from scripts.eval.dnahnet.score_mavedb import _exact_ar_losses, score_batch

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
MODEL_LENGTH = 24


def _ar_config(backbone):
  model_name = "small_ar_transformer" if backbone == "dit" else "small_ussm"
  overrides = [
    f"model={model_name}",
    "data=carbon-prokaryote",
    "algo=ar",
    f"algo.backbone={backbone}",
    "block_size=1",
    "model.hidden_size=16",
    "model.cond_dim=16",
    "model.n_blocks=2",
    f"model.length={MODEL_LENGTH}",
    "loader.batch_size=4",
    "loader.eval_batch_size=4",
    "training.ema=0",
    "sampling.kv_cache=true",
  ]
  if backbone == "dit":
    overrides += ["model.n_heads=2", "model.attn_backend=sdpa"]
  else:
    overrides += [
      "model.n_heads=2",
      "model.ssm_state_size=3",
      "model.ssm_conv_size=4",
      "model.ssm_expand=2",
      "model.ssm_head_dim=4",
      "model.ssm_chunk_size=4",
      "model.ssm_backend=torch",
      "model.mlp_ratio=2.0",
    ]
  with hydra.initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    return hydra.compose(config_name="config", overrides=overrides)


def _ar_model(backbone, seed):
  torch.manual_seed(seed)
  config = _ar_config(backbone)
  model = Diffusion(config, DNATokenizer())
  model.eval()
  model.backbone.eval()
  if backbone == "dit":
    # `DDiTFinalLayer.__init__` deliberately zero-inits its output
    # projection (models/dit.py, standard "zero-init the last layer" DiT
    # practice) so a freshly-trained model starts as a no-op. A trained
    # checkpoint has long since moved off that point, but a from-scratch
    # random init used only for this test has NOT -- verified directly: a
    # fresh `algo=ar backbone=dit` model's raw backbone output is exactly
    # 0.0 for every input, which trivially (and uninformatively) satisfies
    # any causality check. `backbone=ussm`'s output projection is tied to
    # the (non-zero, normal-initialised) token embedding, so it needs no
    # such fix. Break the symmetry here, deliberately, so the causality
    # assertions below exercise the real computation instead of a zeroed one.
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(model.backbone.output_layer.linear.bias, std=0.02)
  return model


def test_both_ar_backbones_construct_with_matched_geometry():
  """Sanity: the two configs actually agree on the fields the harness reads."""
  dit_model = _ar_model("dit", seed=0)
  ussm_model = _ar_model("ussm", seed=0)
  for model in (dit_model, ussm_model):
    assert str(model.parameterization) == "ar"
    assert int(model.config.block_size) == 1
    assert int(model.config.model.length) == MODEL_LENGTH
    assert model.cross_attn is False


def _toy_x0(tokenizer, batch=2, length=MODEL_LENGTH):
  torch.manual_seed(3)
  base_ids = torch.tensor(
    tokenizer.convert_tokens_to_ids(list("ACGT")), dtype=torch.long)
  return base_ids[torch.randint(0, 4, (batch, length))]


def test_exact_ar_losses_shape_and_zeroed_first_position_match_across_backbones():
  """Both real backbones obey the same contract the FakeAR unit test assumes."""
  tokenizer = DNATokenizer()
  x0 = _toy_x0(tokenizer)
  for backbone in ("dit", "ussm"):
    model = _ar_model(backbone, seed=1)
    with torch.inference_mode():
      losses = _exact_ar_losses(model, x0)
    assert losses.shape == x0.shape, backbone
    assert torch.equal(losses[:, 0], torch.zeros(x0.shape[0])), backbone
    assert torch.isfinite(losses).all(), backbone


def test_ar_backbones_are_strictly_causal_under_the_harness_forward_call():
  """Perturbing token p must not move any scored loss at position < p.

  `_exact_ar_losses` calls `model.forward(x0[:, :-1], sigma=None)` and reads
  `log_probs[:, i-1]` as the distribution for token i. If either backbone's
  mask, rotary position ids, or (for the SSM) scan direction let a later
  token leak into an earlier position's logits -- the exact failure mode a
  swapped mask or an off-by-one in the block-causal construction would cause
  -- this catches it directly on the real module, not a stand-in.
  """
  tokenizer = DNATokenizer()
  x0 = _toy_x0(tokenizer, batch=3, length=MODEL_LENGTH)
  n_id = tokenizer.convert_tokens_to_ids("N")
  for backbone in ("dit", "ussm"):
    model = _ar_model(backbone, seed=2)
    with torch.inference_mode():
      baseline = _exact_ar_losses(model, x0)
      for p in (1, MODEL_LENGTH // 2, MODEL_LENGTH - 1):
        perturbed = x0.clone()
        # Rotate the token at position p to something else in-vocab (avoid a
        # no-op if it already happened to equal n_id).
        perturbed[:, p] = torch.where(
          perturbed[:, p] == n_id, perturbed[:, p] - 1,
          torch.full_like(perturbed[:, p], n_id))
        assert not torch.equal(perturbed[:, p], x0[:, p])
        moved = _exact_ar_losses(model, perturbed)
        assert torch.equal(moved[:, :p], baseline[:, :p]), (
          f"{backbone}: perturbing position {p} changed a loss at an "
          f"earlier position -- causality violated")
        # And the converse, so this cannot pass vacuously on a network that
        # (for whatever reason -- e.g. an untrained, zero-initialised output
        # head) ignores its input everywhere: position p itself must move,
        # since its target-token lookup index changed.
        assert not torch.equal(moved[:, p], baseline[:, p]), (
          f"{backbone}: perturbing position {p} changed NO scored loss "
          f"at or after it -- the model appears input-insensitive, which "
          f"would make the causality check above vacuous")


def _paired_records(wt="ACGTACGTACGTACGTACGTAC", mut_pos=5):
  mut = list(wt)
  mut[mut_pos] = "T" if mut[mut_pos] != "T" else "A"
  mut = "".join(mut)
  return [{
    "score_set_urn": "urn:test", "score_set_title": "t", "target": "g",
    "accession": "acc#1", "hgvs_nt": f"c.{mut_pos + 1}X>Y", "hgvs_pro": "p.=",
    "experimental_score": 0.0, "wt_sequence": wt, "mut_sequence": mut,
    "license": "CC0", "source_url": "",
  }]


def test_scored_token_count_and_loss_type_match_across_backbones_on_score_batch():
  """The harness's own accounting (not just the raw forward call) agrees.

  `wt_scored_tokens`/`mut_scored_tokens` are computed from
  `token_mask_cpu`, which is built by `encode_dna` from the tokenizer alone
  before either backbone's `forward` runs -- so a mismatch here would
  implicate `score_batch`'s bookkeeping, not the model.
  """
  tokenizer = DNATokenizer()
  records = _paired_records()
  wt_len = len(records[0]["wt_sequence"])
  generator = torch.Generator(device="cpu").manual_seed(0)
  results = {}
  for backbone in ("dit", "ussm"):
    model = _ar_model(backbone, seed=4)
    with torch.inference_mode():
      scored = score_batch(
        model=model, tokenizer=tokenizer, records=records,
        model_length=MODEL_LENGTH, mc_samples=1, epsilon=1e-3,
        generator=generator, score_mode="nelbo")
    assert len(scored) == 1
    results[backbone] = scored[0]

  for backbone, row in results.items():
    assert row["loss_type"] == "exact_ar_nll", backbone
    # Position 0 is excluded from every AR score (see
    # `test_exact_ar_losses_shape_and_zeroed_first_position_match_across_backbones`),
    # so wt_len real tokens minus position 0 are scored.
    assert row["wt_scored_tokens"] == wt_len - 1, backbone
    assert row["mut_scored_tokens"] == wt_len - 1, backbone

  # The cross-backbone assertion the task calls for: identical scored-token
  # COUNTS and identical POSITION-0 convention, independent of which model
  # produced the (necessarily different, differently-initialised-weight)
  # likelihoods.
  assert (results["dit"]["wt_scored_tokens"]
          == results["ussm"]["wt_scored_tokens"])
  assert (results["dit"]["mut_scored_tokens"]
          == results["ussm"]["mut_scored_tokens"])
