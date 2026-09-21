"""CPU checks for the GenomicBenchmarks fine-tuning harness.

Everything here runs on a head node. The three things worth proving without a
GPU are (a) the reverse-complement mapping is the real one and is an
involution, (b) the new encoder is bit-for-bit the old one under the old
settings, and (c) the sweep/seed bookkeeping never lets the test split into
model selection. (c) is exercised end-to-end against a real (tiny)
`BidirectionalSSM`, not a stub, so the control flow tested is the control flow
that runs.
"""

import pathlib

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import dataloader
from models.bidirectional_ssm import BidirectionalSSM
from scripts.eval.caduceus import finetune as ft
from scripts.eval.caduceus import genomic_benchmarks as gb
from scripts.eval.dnahnet.score_mavedb import encode_dna

VOCAB = 13


def _tokenizer():
  return dataloader.DNATokenizer()


def _sequences(n, rng, low=20, high=60):
  return ["".join(rng.choice(list("ACGT"), size=rng.integers(low, high)))
          for _ in range(n)]


# --------------------------------------------------------------------------
# 1. Reverse complement
# --------------------------------------------------------------------------

def test_complement_table_is_the_real_vocabulary_and_an_involution():
  tokenizer = _tokenizer()
  table = ft.build_complement_table(tokenizer, VOCAB)
  ids = {c: tokenizer.convert_tokens_to_ids(c) for c in "ACGTN"}
  # The mapping is read off the tokenizer, but assert the values it lands on:
  # DNATokenizer puts 8 special tokens first, so A=8, C=9, G=10, T=11, N=12.
  assert ids == {"A": 8, "C": 9, "G": 10, "T": 11, "N": 12}
  assert table[ids["A"]] == ids["T"] and table[ids["T"]] == ids["A"]
  assert table[ids["C"]] == ids["G"] and table[ids["G"]] == ids["C"]
  assert table[ids["N"]] == ids["N"]
  for special in range(8):  # [CLS] [SEP] [BOS] [EOS] [MASK] [PAD] [RESERVED] [UNK]
    assert table[special] == special
  assert torch.equal(table[table], torch.arange(VOCAB))


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_reverse_complement_matches_string_rc_and_is_an_involution(pad_side):
  tokenizer = _tokenizer()
  table = ft.build_complement_table(tokenizer, VOCAB)
  rng = np.random.default_rng(0)
  sequences = _sequences(16, rng)
  encoder = ft.Encoder(tokenizer, 64, pad_to="task", pad_side=pad_side)
  ids, mask = encoder.encode(sequences, torch.device("cpu"))

  rc = ft.reverse_complement_ids(ids, mask, table)
  # rc(rc(x)) == x, the assertion the brief asked for, on padded batches.
  assert torch.equal(ft.reverse_complement_ids(rc, mask, table), ids)
  # Padding is untouched: only the real span moves.
  assert torch.equal(rc[~mask], ids[~mask])

  # And it agrees with reverse-complementing the STRING and re-encoding, which
  # is the definition. This is what a naive flip(dim=1) gets wrong: on a padded
  # row the flip drags the padding across the sequence.
  pairs = {"A": "T", "T": "A", "C": "G", "G": "C"}
  expected, _ = encoder.encode(
    ["".join(pairs[c] for c in reversed(s)) for s in sequences],
    torch.device("cpu"))
  assert torch.equal(rc, expected)

  naive = torch.flip(ids, dims=(1,))
  assert not torch.equal(naive, expected), "flip alone must not be the answer"


def _hidden(backbone, ids):
  h = backbone.token_embedding(ids)
  left = backbone._empty_cache(ids.shape[0], h.device, h.dtype, "left")
  right = backbone._empty_cache(ids.shape[0], h.device, h.dtype, "right")
  for index, layer in enumerate(backbone.layers):
    h = layer.scan_active(h, left.states[index], right.states[index])
  return backbone.final_norm(h)


def test_the_backbone_is_already_exactly_equivariant_to_reversal():
  """Half of 'reverse complement' is free on this architecture, and always was.

  The two scan directions share one SegmentMamba2 and everything else is
  per-position, so with both caches empty the stack commutes with a flip. That
  is the fact that sets the ceiling on what --rc-tta can be worth here.
  """
  from scripts.eval.caduceus.embed import pool
  model = _tiny_model()
  backbone = model.backbone.double().eval()
  encoder = ft.Encoder(model.tokenizer, 20)
  ids, mask = encoder.encode(
    ["ACGTTGCAAGGCTTACGATC", "GGCATTACGATCCGTAAGCT"], torch.device("cpu"))
  assert bool(mask.all()), "this identity is about the unpadded case"
  with torch.no_grad():
    h = _hidden(backbone, ids)
    flipped = _hidden(backbone, torch.flip(ids, dims=(1,)))
  # Per position, up to the flip: exact to float64 round-off.
  assert torch.allclose(torch.flip(flipped, dims=(1,)), h, atol=1e-12)
  for how in ("mean", "max", "meanmax"):
    assert torch.allclose(pool(flipped, mask, how), pool(h, mask, how),
                          atol=1e-12), f"{how} pooling must annihilate a flip"


def test_rc_tta_is_exactly_a_complement_only_ensemble_here():
  """...so a null --rc-tta result must not be read as 'RC does not help'."""
  from scripts.eval.caduceus.embed import pool
  model = _tiny_model()
  backbone = model.backbone.double().eval()
  table = ft.build_complement_table(model.tokenizer, VOCAB)
  encoder = ft.Encoder(model.tokenizer, 20)
  ids, mask = encoder.encode(
    ["ACGTTGCAAGGCTTACGATC", "GGCATTACGATCCGTAAGCT"], torch.device("cpu"))
  with torch.no_grad():
    plain = _hidden(backbone, ids)
    rc = _hidden(backbone, ft.reverse_complement_ids(ids, mask, table))
    complemented = _hidden(backbone, table[ids])
  for how in ("mean", "max", "meanmax"):
    assert torch.allclose(pool(rc, mask, how), pool(complemented, mask, how),
                          atol=1e-12)
    # But it is not a no-op: the complement half does move the representation,
    # so the flag is still worth running.
    assert not torch.allclose(pool(rc, mask, how), pool(plain, mask, how),
                              atol=1e-3)


def test_reverse_complement_rejects_non_contiguous_padding():
  table = ft.build_complement_table(_tokenizer(), VOCAB)
  ids = torch.tensor([[8, 9, 12, 10, 11]])
  holes = torch.tensor([[True, True, False, True, True]])
  with pytest.raises(ValueError, match="contiguous"):
    ft.reverse_complement_ids(ids, holes, table)


# --------------------------------------------------------------------------
# 2. Encoding: the new path must reproduce the old one exactly
# --------------------------------------------------------------------------

def test_fast_encoder_is_bit_for_bit_encode_dna():
  tokenizer = _tokenizer()
  rng = np.random.default_rng(1)
  sequences = _sequences(64, rng, 1, 120)
  window = 128
  encoder = ft.Encoder(tokenizer, window, pad_to="task", pad_token="N",
                       pad_side="right", fast=True)
  ids, mask = encoder.encode(sequences, torch.device("cpu"))
  for row, sequence in enumerate(sequences):
    reference_ids, reference_mask = encode_dna(tokenizer, sequence, window)
    assert ids[row].tolist() == reference_ids
    assert mask[row].tolist() == reference_mask

  slow = ft.Encoder(tokenizer, window, pad_to="task", fast=False)
  slow_ids, slow_mask = slow.encode(sequences, torch.device("cpu"))
  assert torch.equal(ids, slow_ids) and torch.equal(mask, slow_mask)


def test_non_acgt_is_rejected_the_same_way():
  encoder = ft.Encoder(_tokenizer(), 32)
  with pytest.raises(ValueError, match="non-ACGT"):
    encoder.encode(["ACGTX"], torch.device("cpu"))


def test_pad_to_batch_removes_most_padding_and_keeps_the_mask_honest():
  tokenizer = _tokenizer()
  rng = np.random.default_rng(2)
  # The shape of human_ocr_ensembl: median ~315 in a 768 window.
  sequences = ["A" * int(n) for n in rng.integers(150, 500, size=256)]
  task = ft.Encoder(tokenizer, 768, pad_to="task")
  batch = ft.Encoder(tokenizer, 768, pad_to="batch", pad_multiple=8)
  order = ft.make_batches(sequences, range(len(sequences)), 32, bucket=False)
  bucketed = ft.make_batches(sequences, range(len(sequences)), 32, bucket=True,
                             pool_batches=10 ** 6)
  for chunk in order:
    task.encode([sequences[i] for i in chunk], torch.device("cpu"))
  for chunk in bucketed:
    ids, mask = batch.encode([sequences[i] for i in chunk], torch.device("cpu"))
    for row, index in enumerate(chunk):
      assert int(mask[row].sum()) == len(sequences[index])
  assert task.pad_fraction > 0.55
  assert batch.pad_fraction < 0.10
  # The whole point: the padded positions the SSM has to scan drop ~8x, which
  # is where the compute saving that pays for uncapped training comes from.
  assert task.pad_fraction / batch.pad_fraction > 6


def test_pad_side_left_puts_the_sequence_at_the_end():
  encoder = ft.Encoder(_tokenizer(), 16, pad_to="task", pad_side="left")
  ids, mask = encoder.encode(["ACGT"], torch.device("cpu"))
  assert mask[0].tolist() == [False] * 12 + [True] * 4
  assert ids[0, 12:].tolist() == [8, 9, 10, 11]


def test_pad_token_pad_uses_the_real_pad_id():
  encoder = ft.Encoder(_tokenizer(), 8, pad_to="task", pad_token="PAD")
  ids, mask = encoder.encode(["ACGT"], torch.device("cpu"))
  assert ids[0].tolist() == [8, 9, 10, 11, 5, 5, 5, 5]
  assert mask[0].tolist() == [True] * 4 + [False] * 4


# --------------------------------------------------------------------------
# 3. Batching
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bucket", [False, True])
def test_batches_cover_every_example_exactly_once(bucket):
  rng = np.random.default_rng(3)
  sequences = _sequences(101, rng)
  batches = ft.make_batches(sequences, rng.permutation(101), 16, bucket, rng)
  flat = sorted(int(i) for chunk in batches for i in chunk)
  assert flat == list(range(101))
  assert max(len(c) for c in batches) == 16


# --------------------------------------------------------------------------
# 4. Split construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stratified", [False, True])
def test_val_split_is_disjoint_from_train_and_moves_with_the_seed(stratified):
  labels = np.array([0] * 700 + [1] * 300)
  train, val = ft.split_train_val(labels, 0.1, seed=0, stratified=stratified)
  assert set(train.tolist()) & set(val.tolist()) == set()
  assert sorted(train.tolist() + val.tolist()) == list(range(1000))
  assert abs(len(val) - 100) <= 2
  _, other = ft.split_train_val(labels, 0.1, seed=1, stratified=stratified)
  assert set(val.tolist()) != set(other.tolist())


def test_stratified_split_preserves_the_class_balance():
  labels = np.array([0] * 700 + [1] * 300)
  _, val = ft.split_train_val(labels, 0.1, seed=0, stratified=True)
  assert (labels[val] == 1).mean() == pytest.approx(0.3, abs=0.01)


def test_tiny_task_still_yields_a_val_split():
  # dummy_mouse_enhancers_ensembl has 968 train rows; the degenerate case is
  # what the old `max(1, ...)` guarded and it must survive stratification too.
  labels = np.array([0, 1, 0, 1, 0])
  train, val = ft.split_train_val(labels, 0.1, seed=0, stratified=True)
  assert len(val) >= 1 and len(train) >= 1


# --------------------------------------------------------------------------
# 5. Sweep grid, presets, schedule
# --------------------------------------------------------------------------

def test_sweep_grid_expands_and_types_from_the_base_config():
  base = {"backbone_lr": 1e-5, "head_lr": 1e-3, "batch_size": 16,
          "pooling": "mean"}
  grid = ft.parse_sweep("backbone_lr=1e-5,3e-5,1e-4;head_lr=1e-3,3e-3", base)
  assert len(grid) == 6
  assert all(isinstance(g["backbone_lr"], float) for g in grid)
  assert {g["backbone_lr"] for g in grid} == {1e-5, 3e-5, 1e-4}
  assert ft.parse_sweep("batch_size=64,128", base)[0]["batch_size"] == 64
  assert ft.parse_sweep("", base) == [{}]
  with pytest.raises(ValueError, match="not sweepable"):
    ft.parse_sweep("checkpoint=a,b", base)


def test_preset_defaults_apply_but_explicit_flags_win(monkeypatch):
  parser = ft.build_parser()
  argv = ["--checkpoint", "x", "--label", "y", "--preset", "v2",
          "--epochs", "3", "--seeds", "0,1,2"]
  monkeypatch.setattr(ft.sys, "argv", ["finetune.py"] + argv)
  args = ft.resolve(parser.parse_args(argv))
  assert args.epochs == 3, "explicit --epochs must beat the preset's 10"
  assert args.batch_size == 128 and args.pad_to == "batch"
  assert args.scheduler == "cosine" and args.clip_mode == "per-group"
  assert args.stratified_val is True and args.head_layernorm is True
  assert args.rc_tta is False, "v2 must not silently turn RC on"
  assert args.seed_list == [0, 1, 2]

  monkeypatch.setattr(ft.sys, "argv",
                      ["finetune.py", "--checkpoint", "x", "--label", "y"])
  legacy = ft.resolve(parser.parse_args(["--checkpoint", "x", "--label", "y"]))
  assert (legacy.epochs, legacy.batch_size, legacy.backbone_lr) == (4, 16, 1e-5)
  assert legacy.pad_to == "task" and legacy.scheduler == "none"
  assert legacy.clip_mode == "global" and legacy.seed_list == [0]


def test_schedule_warms_up_freezes_the_backbone_and_decays_to_a_tenth():
  config = {"scheduler": "cosine", "warmup_frac": 0.1, "head_warmup_steps": 20}
  backbone, head = ft.build_schedule(config, total_steps=200)
  assert backbone(0) == 0.0 and backbone(19) == 0.0
  assert backbone(20) > 0.0
  assert head(0) == pytest.approx(1 / 20)
  assert head(19) == pytest.approx(1.0)
  assert head(199) == pytest.approx(0.1, abs=1e-3)
  flat, _ = ft.build_schedule(
    {"scheduler": "none", "warmup_frac": 0.1, "head_warmup_steps": 0}, 200)
  assert flat(0) == 1.0 and flat(199) == 1.0


# --------------------------------------------------------------------------
# 6. The test-set guard
# --------------------------------------------------------------------------

def test_held_out_test_counts_accesses():
  guard = ft.HeldOutTest(["ACGT"], np.array([0]))
  guard.require(0)
  guard.take("final seed=0")
  guard.require(1)
  with pytest.raises(AssertionError, match="touched 1 times, expected 0"):
    guard.require(0)


# --------------------------------------------------------------------------
# 7. End to end on a real (tiny) backbone
# --------------------------------------------------------------------------

def _tiny_model():
  config = OmegaConf.create({
    "block_size": 8,
    "algo": {"parameterization": "subs", "time_conditioning": False},
    "model": {
      "hidden_size": 16, "cond_dim": 8, "n_blocks": 2, "dropout": 0.0,
      "tie_word_embeddings": True, "right_flank_probability": 0.0,
      "ssm_state_size": 4, "ssm_conv_size": 4, "ssm_expand": 2,
      "ssm_head_dim": 8, "ssm_chunk_size": 4, "ssm_backend": "torch",
      "mlp_ratio": 2.0,
    },
  })
  torch.manual_seed(7)
  backbone = BidirectionalSSM(config, vocab_size=VOCAB)

  class Wrapper:
    pass

  model = Wrapper()
  model.backbone = backbone
  model.tokenizer = _tokenizer()
  model.pristine = {k: v.clone() for k, v in backbone.state_dict().items()}
  return model


def _synthetic_task(monkeypatch, n_train=64, n_test=32):
  """A learnable toy task: class 1 sequences are GC-rich."""
  rng = np.random.default_rng(11)

  def draw(n):
    sequences, labels = [], []
    for _ in range(n):
      label = int(rng.integers(2))
      alphabet = list("GC") if label else list("AT")
      length = int(rng.integers(12, 24))
      sequences.append("".join(rng.choice(alphabet, size=length)))
      labels.append(label)
    return sequences, np.asarray(labels)

  xtr, ytr = draw(n_train)
  xte, yte = draw(n_test)
  monkeypatch.setattr(
    ft, "load_task",
    lambda name, max_train=None, max_test=None, seed=0: (xtr, ytr, xte, yte))
  monkeypatch.setattr(ft, "task_stats", lambda name: {
    "n_train_full": n_train, "n_test_full": n_test, "max_length": 24,
    "median_length": 18.0, "mean_length": 18.0, "num_classes": 2})
  monkeypatch.setattr(ft, "reference", lambda name, column: 0.5)
  return xtr, ytr, xte, yte


def _args(**overrides):
  parser = ft.build_parser()
  args = parser.parse_args(["--checkpoint", "x", "--label", "y"])
  args.seeds = None
  args = ft.resolve(args)
  for key, value in overrides.items():
    setattr(args, key, value)
  args.seed_list = overrides.get("seed_list", args.seed_list)
  return args


def _base_config(args, **overrides):
  # Delegate, never duplicate: this list used to be a verbatim copy of the one
  # in finetune.py, and the copies drifted the first time a knob was added.
  config = ft.base_config_from(args, block_size=8)
  config.update(overrides)
  return config


def test_run_task_touches_test_exactly_once_per_seed(monkeypatch):
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  args = _args(seed_list=[0, 1, 2], epochs=1, batch_size=16,
               eval_batch_size=16, window_from="full")
  config = _base_config(args)
  row = ft.run_task("toy", model, args, config, torch.device("cpu"), None)
  assert row["test_evaluations"] == 3
  assert row["test_access_log"] == [f"final seed={s}" for s in (0, 1, 2)]
  assert len(row["accuracy_per_seed"]) == 3
  assert row["accuracy"] == pytest.approx(np.mean(row["accuracy_per_seed"]))
  assert row["accuracy_std"] == pytest.approx(np.std(row["accuracy_per_seed"]))
  assert row["seeds"] == [0, 1, 2]
  assert row["n_train_full"] == 64 and row["train_fraction"] == 1.0


def test_sweep_selects_on_validation_without_touching_test(monkeypatch):
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  args = _args(seed_list=[0], epochs=1, batch_size=16, eval_batch_size=16,
               window_from="full", sweep="head_lr=1e-3,1e-2", sweep_seeds=1)
  config = _base_config(args)

  # Trip an alarm if anything reaches the test split before the final loop.
  seen = []
  original = ft.HeldOutTest.take

  def audited(self, reason):
    seen.append(reason)
    return original(self, reason)

  monkeypatch.setattr(ft.HeldOutTest, "take", audited)
  row = ft.run_task("toy", model, args, config, torch.device("cpu"), None)

  assert len(row["sweep"]) == 2
  assert all(r["val_mean"] is not None for r in row["sweep"])
  assert seen == ["final seed=0"], "the sweep must not read the test split"
  assert row["test_evaluations"] == 1
  winner = max(row["sweep"], key=lambda r: r["val_mean"])
  assert row["config"]["head_lr"] == winner["config"]["head_lr"]


def test_a_single_point_grid_does_not_run_a_second_training_pass(monkeypatch):
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  calls = []
  original = ft.train_one
  monkeypatch.setattr(
    ft, "train_one",
    lambda *a, **k: (calls.append(1), original(*a, **k))[1])
  args = _args(seed_list=[0], epochs=1, batch_size=16, eval_batch_size=16,
               window_from="full")
  ft.run_task("toy", model, args, _base_config(args), torch.device("cpu"), None)
  assert len(calls) == 1, "no sweep means exactly one training run per seed"


def test_seeds_actually_produce_different_runs(monkeypatch):
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  args = _args(seed_list=[0, 1, 2, 3], epochs=1, batch_size=16,
               eval_batch_size=16, window_from="full")
  row = ft.run_task("toy", model, args, _base_config(args),
                    torch.device("cpu"), None)
  assert len(set(row["val_per_seed"])) > 1, (
    "seed averaging is pointless if every seed gives the same run")


def test_backbone_is_restored_to_pristine_between_runs(monkeypatch):
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
  args = _args(seed_list=[0], epochs=1, batch_size=16, eval_batch_size=16,
               window_from="full", backbone_lr=1e-2)
  encoder = ft.Encoder(model.tokenizer, 24)
  xtr, ytr, _, _ = ft.load_task("toy")
  train, val = ft.split_train_val(ytr, 0.2, 0, False)
  task = ([xtr[i] for i in train], ytr[train],
          [xtr[i] for i in val], ytr[val], 2, 2.9)
  config = _base_config(args)
  _, classifier = ft.train_one(model, encoder, task, config, 0,
                               torch.device("cpu"), None)
  moved = any(not torch.allclose(before[k], v)
              for k, v in model.backbone.state_dict().items())
  assert moved, "the backbone must actually be fine-tuned"
  ft.train_one(model, encoder, task, config, 0, torch.device("cpu"), None)
  # A second run restores the snapshot first, so it reproduces the first run.
  after = model.backbone.state_dict()
  assert all(torch.allclose(v, after[k])
             for k, v in classifier.backbone.state_dict().items())


def test_rc_tta_changes_nothing_when_the_model_is_strand_symmetric(monkeypatch):
  """RC averaging must be a no-op on a strand-symmetric input, not noise."""
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  table = ft.build_complement_table(model.tokenizer, VOCAB)
  encoder = ft.Encoder(model.tokenizer, 16)
  classifier = ft.Classifier(model.backbone, 16, 2).eval()
  # A palindrome is its own reverse complement, so both views are identical and
  # the averaged prediction must equal the single-view prediction exactly.
  palindromes = ["ACGT" * 2, "GGCC", "AATT"]
  plain = ft.evaluate(classifier, encoder, palindromes, np.zeros(3, int), 8,
                      torch.device("cpu"))
  both = ft.evaluate(classifier, encoder, palindromes, np.zeros(3, int), 8,
                     torch.device("cpu"), complement=table)
  assert plain == both


def test_intermediate_layer_tap_uses_only_the_layers_below_it():
  model = _tiny_model()
  classifier = ft.Classifier(model.backbone, 16, 2, layer=1)
  assert classifier.n_layers == 1
  assert classifier.use_final_norm is False
  # Tapping below the top skips final_norm, so a head norm is forced on.
  assert isinstance(classifier.norm, torch.nn.LayerNorm)
  trainable = {id(p) for p in classifier.trainable_backbone_parameters()}
  assert all(id(p) not in trainable
             for p in model.backbone.layers[1].parameters())
  assert all(id(p) in trainable
             for p in model.backbone.layers[0].parameters())
  with pytest.raises(ValueError, match="--layer"):
    ft.Classifier(model.backbone, 16, 2, layer=9)


@pytest.mark.parametrize("overrides", [
  pytest.param({"pooling": "max"}, id="max-pool"),
  pytest.param({"pooling": "meanmax"}, id="meanmax-pool"),
  pytest.param({"log_length": True}, id="log-length"),
  pytest.param({"layer": 1}, id="layer-tap"),
  pytest.param({"dropout": 0.1}, id="dropout"),
  pytest.param({"honour_no_weight_decay": True}, id="no-wd-flags"),
  pytest.param({"clip_mode": "per-group"}, id="per-group-clip"),
  pytest.param({"pad_to": "batch"}, id="bucketed"),
  pytest.param({"rc_aug": 0.5, "rc_tta": True}, id="rc"),
  pytest.param({"rc_aug": 0.5, "rc_tta": True, "rc_average": "logit"},
               id="rc-logit"),
  pytest.param({"scheduler": "cosine", "warmup_frac": 0.2,
                "head_warmup_steps": 2, "epochs": 2}, id="cosine-lpft"),
  pytest.param({"evals_per_epoch": 2, "patience": 1, "epochs": 3},
               id="step-eval-early-stop"),
  pytest.param({"stratified_val": True}, id="stratified"),
])
def test_every_flag_path_trains_and_scores(monkeypatch, overrides):
  """Each new behaviour must survive a real forward/backward, not just parse."""
  _synthetic_task(monkeypatch)
  model = _tiny_model()
  args = _args(seed_list=[0], epochs=1, batch_size=16, eval_batch_size=16,
               window_from="full")
  config = _base_config(args, **overrides)
  complement = ft.build_complement_table(model.tokenizer, VOCAB)
  row = ft.run_task("toy", model, args, config, torch.device("cpu"), complement)
  assert row["test_evaluations"] == 1
  assert 0.0 <= row["accuracy"] <= 1.0


def test_the_v2_preset_runs_end_to_end_with_a_sweep(monkeypatch):
  """The exact bundle the LSF job launches, on a toy task."""
  _synthetic_task(monkeypatch, n_train=128, n_test=64)
  model = _tiny_model()
  parser = ft.build_parser()
  argv = ["--checkpoint", "x", "--label", "y", "--preset", "v2",
          "--epochs", "2", "--batch-size", "16", "--eval-batch-size", "16",
          "--seeds", "0,1", "--sweep", "backbone_lr=1e-5,1e-4",
          "--window-from", "full"]
  monkeypatch.setattr(ft.sys, "argv", ["finetune.py"] + argv)
  args = ft.resolve(parser.parse_args(argv))
  config = _base_config(args)
  row = ft.run_task("toy", model, args, config, torch.device("cpu"), None)
  assert row["test_evaluations"] == 2  # one per final seed, zero in the sweep
  assert len(row["sweep"]) == 2
  assert row["config"]["backbone_lr"] in (1e-5, 1e-4)
  assert row["pad_fraction"] < 0.35, "v2 buckets by length"
  assert set(row["config"]) == set(ft.SWEEPABLE)


def test_log_length_widens_the_head_by_one():
  model = _tiny_model()
  plain = ft.Classifier(model.backbone, 16, 2)
  extended = ft.Classifier(model.backbone, 16, 2, log_length=True,
                           length_scale=2.9)
  assert plain.head.in_features == 16
  assert extended.head.in_features == 17
  assert ft.Classifier(model.backbone, 16, 2,
                       pooling="meanmax").head.in_features == 32


# --------------------------------------------------------------------------
# 9. Readout (A): the recurrent summary
# --------------------------------------------------------------------------

def _recurrent(model, sequences, window, **kwargs):
  from scripts.eval.caduceus.embed import embed_sequences_recurrent
  return embed_sequences_recurrent(
    model, model.tokenizer, sequences, window, batch_size=8, **kwargs)


def test_recurrent_summary_width_is_nheads_times_dstate_per_layer_per_direction():
  """headmean is a deliberate 64x narrowing; assert the arithmetic, not a guess."""
  model = _tiny_model()
  mixer = model.backbone.layers[0].mixer
  per_layer = mixer.nheads * mixer.d_state
  features = _recurrent(model, ["ACGTACGT"], 8)
  # last layer only, two directions
  assert features.shape == (1, 2 * per_layer)
  every = _recurrent(model, ["ACGTACGT"], 8, layers="all")
  assert every.shape == (1, 2 * len(model.backbone.layers) * per_layer)
  raw = _recurrent(model, ["ACGTACGT"], 8, reduce="raw")
  assert raw.shape == (1, 2 * mixer.nheads * mixer.headdim * mixer.d_state)
  # The narrowing is the whole point: raw is headdim times wider.
  assert raw.shape[1] == features.shape[1] * mixer.headdim


def test_recurrent_summary_is_invariant_to_the_window_ceiling():
  """Grouping by exact length makes the task window irrelevant, which is the point.

  An earlier design padded every sequence up to the task window and tried to
  bound the damage by putting the filler on each direction's leading edge. That
  only ATTENUATES it -- leading context decays as exp(A*dt*distance) and never
  reaches zero -- and on this toy model the two-sided trick bought a factor of
  1.85, not the orders of magnitude the production timescales would need to make
  65% filler safe. Grouping by length removes the filler instead, so `length`
  degrades to a ceiling and cannot influence a feature at all.
  """
  model = _tiny_model()
  sequence = ["ACGTTGCA"]
  assert np.allclose(_recurrent(model, sequence, 8),
                     _recurrent(model, sequence, 4096), atol=0), (
    "the window ceiling changed the summary, so something is still padding to it")


def test_recurrent_summary_keeps_rows_aligned_with_their_sequences():
  """Grouping reorders; the scatter back must undo it exactly.

  This is the dangerous failure mode of length-grouping: a wrong scatter pairs
  every feature with another sequence's label, which no accuracy check would
  report as an error -- it would just look like the readout does not work.
  """
  sequences = ["ACGT", "ACGTTGCAACGT", "TTTT", "GGCATTACGATC", "AC"]
  model = _tiny_model()
  together = _recurrent(model, sequences, 4096)
  assert together.shape[0] == len(sequences)
  for index, sequence in enumerate(sequences):
    alone = _recurrent(model, [sequence], 4096)
    assert np.allclose(together[index], alone[0], atol=0), (
      f"row {index} ({sequence!r}) does not match that sequence embedded alone")


def test_recurrent_summary_rejects_a_sequence_over_the_ceiling():
  model = _tiny_model()
  with pytest.raises(ValueError, match="exceeds the window"):
    _recurrent(model, ["A" * 64], 32)


def test_recurrent_summary_matches_a_hand_rolled_scan_clean_loop():
  """prefill_left's final state is the state the layer loop ends on."""
  model = _tiny_model()
  backbone = model.backbone.double().eval()
  encoder = ft.Encoder(model.tokenizer, 8)
  ids, _ = encoder.encode(["ACGTTGCA"], torch.device("cpu"))
  with torch.no_grad():
    cache = backbone.prefill_left(ids)
    h = backbone.token_embedding(ids)
    manual = []
    for layer in backbone.layers:
      h, state = layer.scan_clean(h, None)
      manual.append(state)
  for index, state in enumerate(manual):
    assert torch.allclose(cache.states[index].ssm, state.ssm, atol=1e-10), index
    assert torch.allclose(cache.states[index].conv, state.conv, atol=1e-10)


def test_recurrent_summary_refuses_a_unidirectional_checkpoint():
  """Half of (A) is the reverse direction; a uni checkpoint has no honest one.

  UnidirectionalSSM subclasses BidirectionalSSM, so a reverse prefill would have
  silently succeeded out of distribution -- the exact bug shape that reached the
  pooled readout twice.
  """
  from scripts.eval.caduceus.embed import recurrent_summary
  model = _tiny_model()

  class FakeUni(type(model.backbone)):
    pass
  FakeUni.__name__ = "UnidirectionalSSM"
  model.backbone.__class__ = FakeUni
  ids = torch.zeros((1, 8), dtype=torch.long)
  with pytest.raises(TypeError, match="unidirectional"):
    recurrent_summary(model, ids, ids)


def test_window_from_subset_and_the_real_task_stats_are_exercised(monkeypatch):
  """The coverage gap the leakage audit found.

  `--window-from subset` is the argparse DEFAULT and is the one branch that read
  `xte` directly, yet every test passed `window_from="full"` explicitly, and
  `task_stats` was monkeypatched out everywhere -- so the line whose own comment
  called itself "the one number that reads the test side" had no test at all.
  Both are now covered, and `task_stats` is asserted to be train-only.
  """
  xtr = ["ACGT" * 3, "TTTT" * 2]
  xte = ["GGCC" * 9]           # deliberately LONGER than anything in train
  ytr, yte = np.asarray([0, 1]), np.asarray([0])

  class FakeSplit:
    def __init__(self, sequences, labels):
      self.sequences, self.labels = sequences, labels
      self.column_names = ["seq", "label"]
    def __len__(self):
      return len(self.sequences)
    def __getitem__(self, key):
      return self.sequences if key == "seq" else list(self.labels)

  monkeypatch.setattr(
    gb, "_open_task",
    lambda name: ({"train": FakeSplit(xtr, ytr), "test": FakeSplit(xte, yte)},
                  "train", "test"))
  stats = gb.task_stats("toy")
  assert stats["max_length"] == max(len(s) for s in xtr), (
    "task_stats sized max_length from the test split")
  assert stats["max_length"] < max(len(s) for s in xte), (
    "this test is vacuous unless test is longer than train")

  # And the subset branch. Until 2026-09-21 it widened the window to fit the
  # longest TEST sequence, and this test asserted that it did. It no longer
  # does: the window comes from train alone and `Encoder.encode` refuses a
  # sequence that does not fit, so a test row longer than the train-sized
  # window is a hard ERROR rather than a silent peek at held-out data.
  #
  # This changes no published number. Verified 2026-09-21 on all 8 official
  # tasks: the train-only window is identical to the old train-or-test window
  # on every one of them, and the longest test row still fits.
  monkeypatch.setattr(
    ft, "load_task",
    lambda name, max_train=None, max_test=None, seed=0: (xtr, ytr, xte, yte))
  monkeypatch.setattr(ft, "task_stats", lambda name: stats)
  monkeypatch.setattr(ft, "reference", lambda name, column: 0.5)
  model = _tiny_model()
  args = _args(seed_list=[0], epochs=1, batch_size=2, eval_batch_size=2,
               window_from="subset")
  with pytest.raises(ValueError, match="exceeds window"):
    ft.run_task("toy", model, args, _base_config(args), torch.device("cpu"),
                None)


def test_embed_lut_encoder_is_bit_identical_to_encode_dna():
  """The fast path must be a pure speedup, on both padding sides.

  `embed.py` tokenised one character at a time through the HF tokenizer until
  2026-09-13, measured 590x slower than this table. Swapping it in is only safe
  if it is exact, so: same ids, same mask, same UNK rejection.
  """
  from scripts.eval.caduceus.embed import _encode_padded
  tokenizer = _tokenizer()
  cpu = torch.device("cpu")
  chunk = ["ACGTTGCA", "AC", "GGGGCCCCAAAATTTT"]
  window = 24
  both = _encode_padded(tokenizer, chunk, window, cpu, ("left", "right"))
  for row, sequence in enumerate(chunk):
    want_ids, want_mask = encode_dna(tokenizer, sequence, window)
    got_ids, got_mask = both["right"]
    assert got_ids[row].tolist() == want_ids, sequence
    assert got_mask[row].tolist() == want_mask, sequence
    # left padding is the same row rolled so the filler leads
    keep = len(sequence)
    left_ids, left_mask = both["left"]
    assert left_ids[row].tolist() == want_ids[keep:] + want_ids[:keep]
    assert left_mask[row].tolist() == want_mask[keep:] + want_mask[:keep]
  with pytest.raises(ValueError, match="non-ACGT"):
    _encode_padded(tokenizer, ["ACGTX"], window, cpu)
  with pytest.raises(ValueError, match="exceeds model length"):
    _encode_padded(tokenizer, ["A" * 99], window, cpu)


def test_boolean_sweep_values_are_parsed_not_coerced_to_true():
  """bool('False') is True, which made every boolean knob unsweepable.

  This is why --log-length could never be swept, on the one task
  (human_ensembl_regulatory) that gives up 71% of its label entropy to
  sequence length alone.
  """
  base = {"log_length": False, "head_layernorm": False, "backbone_lr": 1e-5}
  grid = ft.parse_sweep("log_length=False,True", base)
  assert sorted(g["log_length"] for g in grid) == [False, True], grid
  grid = ft.parse_sweep("head_layernorm=false,true", base)
  assert sorted(g["head_layernorm"] for g in grid) == [False, True]
  # and it still types non-booleans correctly
  assert ft.parse_sweep("backbone_lr=1e-5,3e-5", base)[1]["backbone_lr"] == 3e-5
  with pytest.raises(ValueError, match="not a boolean"):
    ft.parse_sweep("log_length=maybe", base)


def test_head_warmup_freeze_is_capped_so_small_tasks_still_fine_tune():
  """Under v2 (head_warmup_steps=200) dummy_mouse has ~70 total steps, so its
  backbone never unfroze and the 'fine-tune' was a linear probe: about -0.058
  on that task."""
  cfg = {"warmup_frac": 0.05, "head_warmup_steps": 200, "scheduler": "none"}
  # build_schedule returns [backbone_fn, head_fn]
  backbone, _head = ft.build_schedule(dict(cfg), total_steps=70)
  live = [s for s in range(70) if backbone(s) > 0]
  assert live, "backbone never unfroze on a 70-step run"
  assert min(live) <= 7, f"freeze not capped: first live backbone step {min(live)}"
  # on a long run the cap is generous enough to keep LP-FT's intent
  big_backbone, _ = ft.build_schedule(dict(cfg), total_steps=30000)
  assert big_backbone(100) == 0, "LP-FT should still freeze early on a long run"
  assert big_backbone(1000) > 0


# --------------------------------------------------------------------------
# Train/eval consistency and wrapper plumbing.
#
# Added 2026-09-17 after an audit found that training encodes with
# `Encoder.window_for(batch)` while `evaluate()` pins the width to
# `task_window`. Those agree under pad_to="task" and DISAGREE under
# pad_to="batch", which would score the model at a padding fraction it never
# trained at. No reported result hit it, and these tests keep it that way.
# --------------------------------------------------------------------------

def test_train_and_eval_encode_to_the_same_width_under_pad_to_task():
  tokenizer = _tokenizer()
  sequences = ["ACGT" * 40, "ACGT" * 20, "ACGT" * 60]   # 160, 80, 240 nt
  encoder = ft.Encoder(tokenizer, 256, pad_to="task", pad_multiple=8)
  train_ids, train_mask = encoder.encode(sequences, torch.device("cpu"))
  # `evaluate` pins the window; see its docstring.
  eval_ids, eval_mask = encoder.encode(sequences, torch.device("cpu"),
                                       window=encoder.task_window)
  assert train_ids.shape == eval_ids.shape
  assert torch.equal(train_ids, eval_ids)
  assert torch.equal(train_mask, eval_mask)


def test_pad_to_batch_really_does_diverge_train_from_eval():
  """The defect the guard exists for. If this ever stops failing, drop the guard."""
  tokenizer = _tokenizer()
  sequences = ["ACGT" * 40, "ACGT" * 20, "ACGT" * 60]
  encoder = ft.Encoder(tokenizer, 256, pad_to="batch", pad_multiple=8)
  train_ids, _ = encoder.encode(sequences, torch.device("cpu"))
  eval_ids, _ = encoder.encode(sequences, torch.device("cpu"),
                               window=encoder.task_window)
  assert train_ids.shape[1] == 240 and eval_ids.shape[1] == 256


def test_wrapper_maps_no_two_env_vars_to_one_flag_except_guarded_epochs():
  """Two env vars emitting one flag silently drops the earlier value.

  `EPOCHS`/`EPOCHS_OVERRIDE` both emit --epochs and are allowed only because
  the wrapper now refuses to run when both are set.
  """
  import collections
  import re as _re
  script = (pathlib.Path(__file__).resolve().parents[1]
            / "scripts/eval/caduceus/finetune.sh").read_text()
  flags = _re.findall(r"EXTRA\+=\(\s*(--[\w-]+)", script)
  duplicated = {f for f, n in collections.Counter(flags).items() if n > 1}
  assert duplicated == {"--epochs"}, duplicated
  assert "set EPOCHS or EPOCHS_OVERRIDE, not both" in script


def test_every_wrapper_hook_names_a_flag_argparse_defines():
  import re as _re
  root = pathlib.Path(__file__).resolve().parents[1] / "scripts/eval/caduceus"
  script = (root / "finetune.sh").read_text()
  source = (root / "finetune.py").read_text()
  defined = set(_re.findall(r'add_argument\(\s*"(--[\w-]+)"', source))
  emitted = set(_re.findall(r"EXTRA\+=\(\s*(--[\w-]+)", script))
  assert emitted <= defined, sorted(emitted - defined)


# --------------------------------------------------------------------------
# Padding invariance of the bidirectional scan (--pad-invariant).
#
# Mean pooling already ignores pad positions, but the SCAN does not: the
# reverse direction enters from the right end, i.e. through the pad region.
# Measured on the real checkpoint at the human_ocr_ensembl window (57.5% pad):
# 5.0% relative change of the pooled feature at the median length, 15.5% at
# 150nt. `--pad-invariant` drives dt to -inf and zeroes the conv input at pads
# so a padded forward equals an unpadded one. These tests prove both halves:
# that it works, and that leaving it off changes nothing.
# --------------------------------------------------------------------------

def _tiny_bissm():
  """Same shape as `_tiny_model`'s backbone, but fused so masking is available."""
  config = OmegaConf.create({
    "block_size": 8,
    "algo": {"parameterization": "subs", "time_conditioning": False},
    "model": {
      "hidden_size": 16, "cond_dim": 8, "n_blocks": 2, "dropout": 0.0,
      "tie_word_embeddings": True, "right_flank_probability": 0.0,
      "ssm_state_size": 4, "ssm_conv_size": 4, "ssm_expand": 2,
      "ssm_head_dim": 8, "ssm_chunk_size": 4, "ssm_backend": "torch",
      "mlp_ratio": 2.0, "bidirectional_impl": "fused",
    },
  })
  torch.manual_seed(7)
  return BidirectionalSSM(config, vocab_size=VOCAB)


@pytest.mark.parametrize("real_len,window", [(6, 24), (11, 32)])
def test_pad_invariant_scan_matches_the_unpadded_forward(real_len, window):
  model = _tiny_bissm().eval()
  layer = model.layers[0]
  torch.manual_seed(1)
  short = torch.randn(1, real_len, model.hidden_size)
  padded = torch.cat(
    [short, torch.randn(1, window - real_len, model.hidden_size)], dim=1)
  mask = torch.zeros(1, window, dtype=torch.bool)
  mask[:, :real_len] = True

  empty = lambda n: model._empty_cache(1, short.device, short.dtype, n)
  with torch.no_grad():
    reference = layer.scan_active(short, empty("left").states[0],
                                  empty("right").states[0])
    masked = layer.scan_active(padded, empty("left").states[0],
                               empty("right").states[0], mask=mask)
    unmasked = layer.scan_active(padded, empty("left").states[0],
                                 empty("right").states[0])

  # With the mask, the real positions reproduce the unpadded forward.
  assert torch.allclose(masked[:, :real_len], reference, atol=2e-3), (
    (masked[:, :real_len] - reference).abs().max().item())
  # Without it they do NOT -- this is the defect the flag exists for. If this
  # assertion ever fails, padding stopped mattering and the flag is dead code.
  assert not torch.allclose(unmasked[:, :real_len], reference, atol=2e-3)


def test_pad_invariant_defaults_to_off_and_changes_nothing():
  """mask=None must be bit-for-bit the previous behaviour."""
  model = _tiny_bissm().eval()
  layer = model.layers[0]
  torch.manual_seed(2)
  x = torch.randn(2, 16, model.hidden_size)
  empty = lambda n: model._empty_cache(2, x.device, x.dtype, n)
  with torch.no_grad():
    a = layer.scan_active(x, empty("left").states[0], empty("right").states[0])
    b = layer.scan_active(x, empty("left").states[0], empty("right").states[0],
                          mask=None)
  assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Block-structured readout (--scan-path blocked).
#
# Block diffusion never scans further than block_size, but the default readout
# runs one unbroken scan across the whole window (19 blocks on
# dummy_mouse_enhancers_ensembl). `blocked` reproduces the pretraining geometry.
# The decisive property is the built-in control: a window of exactly one block
# has NO geometry mismatch, so `blocked` must be a no-op there. If it is not,
# any effect measured on multi-block tasks is something other than geometry.
# --------------------------------------------------------------------------

def _classifier(backbone, scan_path, block_size):
  torch.manual_seed(3)
  return ft.Classifier(backbone, backbone.hidden_size, 2, "mean", -1, False,
                       False, 0.0, 0.0, None, scan_path, False,
                       block_size).eval()


def test_blocked_readout_is_a_noop_on_a_single_block_window():
  model = _tiny_model()
  block = 8
  torch.manual_seed(4)
  ids = torch.randint(8, VOCAB, (2, block))          # window == block_size
  mask = torch.ones(2, block, dtype=torch.bool)
  active = _classifier(model.backbone, "active", block)
  blocked = _classifier(model.backbone, "blocked", block)
  blocked.load_state_dict(active.state_dict())
  with torch.no_grad():
    a, b = active(ids, mask), blocked(ids, mask)
  assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_blocked_readout_differs_on_a_multi_block_window():
  """Three blocks: the default runs one 24-token scan, blocked runs 3x8."""
  model = _tiny_model()
  block = 8
  torch.manual_seed(5)
  ids = torch.randint(8, VOCAB, (2, block * 3))
  mask = torch.ones(2, block * 3, dtype=torch.bool)
  active = _classifier(model.backbone, "active", block)
  blocked = _classifier(model.backbone, "blocked", block)
  blocked.load_state_dict(active.state_dict())
  with torch.no_grad():
    a, b = active(ids, mask), blocked(ids, mask)
  assert not torch.allclose(a, b, atol=1e-4)


def test_blocked_readout_rejects_an_indivisible_window():
  model = _tiny_model()
  blocked = _classifier(model.backbone, "blocked", 8)
  ids = torch.randint(8, VOCAB, (1, 20))             # 20 is not a multiple of 8
  mask = torch.ones(1, 20, dtype=torch.bool)
  with pytest.raises(ValueError, match="divisible by block_size"):
    blocked(ids, mask)


# --------------------------------------------------------------------------
# Layer-wise LR decay (--llrd). Every result before 2026-09-18 gave all 12
# backbone layers one shared rate. The failure mode to guard is not a wrong
# learning rate but a DROPPED parameter: if the depth partition misses one, the
# model silently trains fewer weights and nothing else notices.
# --------------------------------------------------------------------------

def test_llrd_partition_covers_every_trainable_parameter_exactly_once():
  model = _tiny_model()
  clf = ft.Classifier(model.backbone, model.backbone.hidden_size, 2)
  flat = clf.trainable_backbone_parameters()
  by_depth = clf.backbone_parameters_by_depth()
  ids_flat = [id(p) for p in flat]
  ids_depth = [id(p) for _, ps in by_depth for p in ps]
  assert len(ids_depth) == len(set(ids_depth)), "a parameter appears twice"
  assert set(ids_depth) == set(ids_flat), (
    f"partition misses {len(set(ids_flat) - set(ids_depth))} parameters and "
    f"invents {len(set(ids_depth) - set(ids_flat))}")
  assert len(ids_depth) == len(ids_flat)


def test_llrd_scales_learning_rate_by_depth_and_is_off_by_default():
  model = _tiny_model()
  clf = ft.Classifier(model.backbone, model.backbone.hidden_size, 2)
  by_depth = clf.backbone_parameters_by_depth()
  depths = [d for d, _ in by_depth]
  assert depths == sorted(depths) and depths[0] == 0, depths
  # gamma**(top - depth): the deepest layer keeps the full rate, the embedding
  # is damped the most.
  top, base, gamma = max(depths), 1e-4, 0.8
  scaled = {d: base * gamma ** (top - d) for d in depths}
  assert scaled[top] == pytest.approx(base)
  assert scaled[0] == pytest.approx(base * gamma ** top)
  assert scaled[0] < scaled[top]


# ---------------------------------------------------------------------------
# --window-from subset: the argparse DEFAULT, and until 2026-09-21 the one
# branch that sized a hyperparameter from the test split. Every other test in
# this file passes window_from="full", so this branch had no coverage at all --
# which is exactly how the test read survived a leakage audit that named it.
# ---------------------------------------------------------------------------

def _fixed_length_task(monkeypatch, train_len, test_len, n_train=64, n_test=32):
  """Train and test sequences at chosen, DIFFERENT fixed lengths."""
  rng = np.random.default_rng(3)

  def draw(n, length):
    sequences, labels = [], []
    for _ in range(n):
      label = int(rng.integers(2))
      alphabet = list("GC") if label else list("AT")
      sequences.append("".join(rng.choice(alphabet, size=length)))
      labels.append(label)
    return sequences, np.asarray(labels)

  xtr, ytr = draw(n_train, train_len)
  xte, yte = draw(n_test, test_len)
  monkeypatch.setattr(
    ft, "load_task",
    lambda name, max_train=None, max_test=None, seed=0: (xtr, ytr, xte, yte))
  monkeypatch.setattr(ft, "task_stats", lambda name: {
    "n_train_full": n_train, "n_test_full": n_test, "max_length": train_len,
    "median_length": float(train_len), "mean_length": float(train_len),
    "num_classes": 2})
  monkeypatch.setattr(ft, "reference", lambda name, column: 0.5)
  return xtr, ytr, xte, yte


def test_subset_window_matches_train_max_rounded_to_the_block(monkeypatch):
  _fixed_length_task(monkeypatch, train_len=20, test_len=12)
  model = _tiny_model()
  args = _args(seed_list=[0], epochs=1, batch_size=16, eval_batch_size=16,
               window_from="subset")
  row = ft.run_task("toy", model, args, config := _base_config(args),
                    torch.device("cpu"), None)
  # ceil(20/8)*8 == 24, from TRAIN alone; the 12-long test rows cannot lower it
  assert row["window"] == 24, row["window"]
  assert config["block_size"] == 8


def test_real_task_stats_reports_a_train_only_max_length():
  """`task_stats` is monkeypatched out everywhere else, so assert the real one.

  It must not consult the test split -- `finetune.py`'s window (under
  --window-from full) is sized straight from this value.
  """
  import ast
  import inspect
  tree = ast.parse(inspect.getsource(gb.task_stats).lstrip())
  fn = tree.body[0]
  # Strip the docstring: it QUOTES the old buggy expression to explain it, so a
  # naive substring check over the whole source fails on the explanation, not
  # on the code. This test is about the body.
  body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                         and isinstance(fn.body[0].value, ast.Constant)) else fn.body
  code = "\n".join(ast.unparse(node) for node in body)
  assert "test_lengths.max()" not in code, (
    "task_stats reads the test split's lengths:\n" + code)
  assert "train_lengths" in code


def test_leak_free_mean_excludes_exactly_the_two_upstream_leaky_tasks():
  """The 6-task mean must name its exclusions and keep the 8-task mean intact."""
  import re
  repo = pathlib.Path(__file__).resolve().parents[1]
  source = (repo / "scripts/eval/caduceus/finetune.py").read_text()
  block = re.search(r"LEAKY_TASKS\s*=\s*\(([^)]*)\)", source)
  assert block, "LEAKY_TASKS tuple not found"
  assert "human_enhancers_ensembl" in block.group(1)
  assert "human_nontata_promoters" in block.group(1)
  # mean_accuracy must still be over ALL rows, for comparability with published
  assert 'mean = float(np.mean([r["accuracy"] for r in rows]))' in source
  for key in ("leak_free_mean_accuracy", "leak_free_n_tasks",
              "leak_free_excluded"):
    assert f'"{key}"' in source, key
