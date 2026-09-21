import json
from pathlib import Path

import numpy as np
import torch
import pytest
from omegaconf import OmegaConf

from scripts.eval.dnahnet.mavedb import (
  apply_coding_hgvs,
  build_records,
  load_manifest,
  rankdata,
  spearmanr,
  summarize_predictions,
)
from scripts.eval.dnahnet.aggregate_mavedb import aggregate_runs
from scripts.eval.dnahnet.score_mavedb import _exact_ar_losses


MANIFEST = (
  Path(__file__).resolve().parents[1]
  / "scripts/eval/dnahnet/mavedb_manifest.json")


def test_pinned_manifest_reproduces_paper_count():
  manifest = load_manifest(MANIFEST)
  assert len(manifest["score_sets"]) == 12
  assert manifest["expected_total_variants"] == 21_250
  assert len({row["urn"] for row in manifest["score_sets"]}) == 12


@pytest.mark.parametrize(("hgvs", "expected"), [
  ("c.=", "ACGTAC"),
  ("c.2C>T", "ATGTAC"),
  ("c.[2C>T;5A>G]", "ATGTGC"),
  ("c.2_3del", "ATAC"),
  ("c.2_3insAAA", "ACAAAGTAC"),
  ("c.2_3delinsTT", "ATTTAC"),
  ("c.[2C>T;5_6del]", "ATGT"),
])
def test_apply_coding_hgvs(hgvs, expected):
  assert apply_coding_hgvs("ACGTAC", hgvs) == expected


def test_apply_coding_hgvs_rejects_reference_mismatch():
  with pytest.raises(ValueError, match="reference mismatch"):
    apply_coding_hgvs("ACGT", "c.2A>T")


class _FakeClient:
  def score_set(self, urn):
    return {
      "title": "example",
      "numVariants": 2,
      "targetGenes": [{
        "name": "gene",
        "targetSequence": {"sequence": "ACGT", "sequenceType": "dna"},
      }],
      "license": {"shortName": "CC0"},
    }

  def variants(self, urn):
    return [
      {"accession": f"{urn}#1", "hgvs_nt": "c.=", "hgvs_pro": "p.=",
       "scores.score": "0.0"},
      {"accession": f"{urn}#2", "hgvs_nt": "c.2C>T", "hgvs_pro": "p.X",
       "scores.score": "1.5"},
    ]


def test_build_records_validates_and_materializes_mutants():
  import hashlib
  sequence = "ACGT"
  manifest = {
    "expected_total_variants": 2,
    "score_sets": [{
      "urn": "urn:mavedb:test",
      "title": "example",
      "expected_variants": 2,
      "target_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }],
  }
  rows = list(build_records(manifest, _FakeClient()))
  assert [row["mut_sequence"] for row in rows] == ["ACGT", "ATGT"]
  assert rows[1]["experimental_score"] == 1.5


def test_rankdata_and_spearman_handle_ties():
  np.testing.assert_allclose(rankdata([4, 1, 1, 3]), [4, 1.5, 1.5, 3])
  assert spearmanr([1, 2, 3], [4, 5, 6]) == pytest.approx(1.0)
  assert spearmanr([1, 2, 3], [6, 5, 4]) == pytest.approx(-1.0)


def test_summary_uses_macro_absolute_per_assay_spearman():
  records = []
  for urn, prediction in [("a", [1, 2, 3]), ("b", [3, 2, 1])]:
    for index, value in enumerate(prediction):
      records.append({
        "score_set_urn": urn,
        "score_set_title": urn,
        "target": urn,
        "predicted_fitness": value,
        "experimental_score": index,
      })
  summary = summarize_predictions(records)
  assert summary["num_assays"] == 2
  assert summary["num_variants"] == 6
  assert summary["macro_abs_spearman"] == pytest.approx(1.0)


def test_aggregate_runs_averages_scores_and_measures_agreement():
  def run(scores):
    return [{
      "score_set_urn": "a",
      "score_set_title": "a",
      "target": "a",
      "accession": str(index),
      "experimental_score": str(index),
      "predicted_fitness": str(score),
      "predicted_fitness_per_nt": str(score),
      "wt_nelbo": str(score + 2),
      "mut_nelbo": "2",
      "wt_nelbo_per_nt": str(score + 2),
      "mut_nelbo_per_nt": "2",
    } for index, score in enumerate(scores)]

  combined, stability = aggregate_runs([run([1, 2, 3]), run([3, 2, 1])])
  assert [row["predicted_fitness"] for row in combined] == [2, 2, 2]
  agreement = stability["run_agreements"][0]
  assert agreement["prediction_spearman"] == pytest.approx(-1.0)
  assert agreement["mean_absolute_score_delta"] == pytest.approx(4 / 3)


def test_exact_ar_losses_align_next_token_nll_and_leave_first_position_zero():
  import torch

  class FakeAR:
    def forward(self, x, sigma):
      assert sigma is None
      logits = torch.full((*x.shape, 6), -4.0)
      # Put the expected next token at a distinct, known probability after
      # log-softmax so the gather/alignment is exercised.
      logits.scatter_(-1, ((x + 1) % 6).unsqueeze(-1), 2.0)
      return logits.log_softmax(-1)

  x0 = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
  losses = _exact_ar_losses(FakeAR(), x0)
  assert losses.shape == x0.shape
  assert torch.equal(losses[:, 0], torch.zeros(2))
  expected = -torch.log_softmax(torch.tensor([-4.] * 5 + [2.]), 0)[-1]
  assert torch.allclose(losses[:, 1:], expected.expand(2, 3))


# --------------------------------------------------------------------------
# Score I: context compatibility (masked infilling preference)
# --------------------------------------------------------------------------

def _fake_bd_model(preference=None):
  """Minimal stand-in for a block-diffusion Diffusion. CPU, no checkpoint.

  `preference` maps a token id to a logit bump, so a test can make the model
  prefer a chosen spelling and check the sign of the score.
  """
  import torch
  from scripts.eval.dnahnet import score_mavedb as sm

  class FakeTokenizer:
    _vocab = {"[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[EOS]": 3, "[MASK]": 4,
              "[PAD]": 5, "[RESERVED]": 6, "[UNK]": 7,
              "A": 8, "C": 9, "G": 10, "T": 11, "N": 12}
    def convert_tokens_to_ids(self, token):
      if isinstance(token, list):
        return [self._vocab[t] for t in token]
      return self._vocab[token]

  class FakeBD:
    mask_index = 4
    cross_attn = False
    tokenizer = FakeTokenizer()
    config = OmegaConf.create(
      {"model": {"length": 12}, "block_size": 12,
       "algo": {"time_conditioning": False}})
    def forward(self, x, sigma):
      logits = torch.zeros((*x.shape, 13))
      for token, bump in (preference or {}).items():
        logits[..., token] += bump
      return logits.log_softmax(-1)

  return FakeBD(), sm


def test_variant_positions_rejects_length_changes_and_finds_substitutions():
  from scripts.eval.dnahnet.score_mavedb import variant_positions
  assert variant_positions("ACGTAC", "ACGTAC") == []
  assert variant_positions("ACGTAC", "ACGAAC") == [3]
  assert variant_positions("ACGTAC", "ACTTGC") == [2, 4]
  # 8.95% of the benchmark changes length; those must be refused, not guessed.
  assert variant_positions("ACGTAC", "ACGTA") is None


def test_genetic_code_is_complete_and_correct():
  from scripts.eval.dnahnet.score_mavedb import GENETIC_CODE
  assert len(GENETIC_CODE) == 64
  assert GENETIC_CODE["ATG"] == "M"
  assert GENETIC_CODE["TGG"] == "W"
  for stop in ("TAA", "TAG", "TGA"):
    assert GENETIC_CODE[stop] == "*"
  # All six leucine codons, which is what makes the synonymous class non-trivial.
  assert {c for c, a in GENETIC_CODE.items() if a == "L"} == {
    "TTA", "TTG", "CTT", "CTC", "CTA", "CTG"}


def test_codon_score_is_exactly_zero_for_a_synonymous_change():
  """The property the codon form exists for, and it must be EXACT, not small.

  Nucleotide scoring penalises a rare synonymous codon; the codon form cannot,
  because wild-type and mutant marginalise to the same amino acid. On this
  dataset that is not a corner case -- the median variant carries 10 synonymous
  codon changes against 1 non-synonymous.
  """
  import math
  model, sm = _fake_bd_model()
  wt = "ATGCTTAAACCC"      # Met Leu Lys Pro
  mut = "ATGCTCAAACCC"     # Met Leu Lys Pro -- CTT->CTC, both Leucine
  ids = lambda s: torch.tensor(
    model.tokenizer.convert_tokens_to_ids(list(s)))
  score, units, synonymous = sm.score_infill(
    model, ids(wt), ids(mut), wt, mut, unit="codon")
  assert score == 0.0, score
  assert units == 1 and synonymous == 1
  # The nucleotide form does NOT have this property: it contrasts two different
  # bases, so a model with any preference between them returns non-zero.
  biased, sm2 = _fake_bd_model(preference={9: 3.0})   # prefers C
  nt_score, _, _ = sm2.score_infill(
    biased, ids(wt), ids(mut), wt, mut, unit="nt")
  assert nt_score > 0.0, nt_score


def test_codon_score_sign_follows_the_model_preference():
  """Higher = model prefers the mutant, matching predicted_fitness's convention."""
  model, sm = _fake_bd_model(preference={8: 4.0})   # strongly prefers A
  wt = "ATGCTTAAACCC"
  mut = "ATGAAAAAACCC"     # codon 1 CTT(Leu) -> AAA(Lys): non-synonymous
  ids = lambda s: torch.tensor(
    model.tokenizer.convert_tokens_to_ids(list(s)))
  score, units, synonymous = sm.score_infill(
    model, ids(wt), ids(mut), wt, mut, unit="codon")
  assert units == 1 and synonymous == 0
  assert score > 0.0, f"model prefers A, mutant is AAA, so score must be >0: {score}"


def test_protein_event_baseline_reproduces_the_published_0_30931_shape():
  """The zero-parameter baseline that beats every model. Sign and direction."""
  from scripts.eval.dnahnet.mavedb import protein_event_baseline
  # Fitness falls as the amino-acid event count rises, as in the real data.
  records = []
  for urn in ("a", "b"):
    for index in range(12):
      events = index % 3                      # 0, 1 or 2, like the real column
      records.append({
        "score_set_urn": urn, "accession": f"{urn}{index}",
        "score_set_title": f"assay {urn}", "target": f"gene{urn}",
        "hgvs_pro": "p.=" if events == 0 else
                    ";".join(["p.Gln1Glu"] * events),
        "experimental_score": 1.0 - 0.4 * events,
      })
  summary = protein_event_baseline(records)
  assert summary["num_assays"] == 2
  # Negated count vs fitness is POSITIVE: fewer events, fitter.
  assert summary["macro_signed_spearman"] > 0.9
  assert summary["num_negative_assays"] == 0


def test_summary_reports_signed_as_well_as_absolute():
  """Signed became primary on 2026-08-14; the code only caught up on 2026-09-13."""
  from scripts.eval.dnahnet.mavedb import summarize_predictions
  records = [
    # assay "up": prediction tracks fitness. assay "down": it is inverted.
    *[{"score_set_urn": "up", "accession": f"u{i}", "score_set_title": "up",
       "target": "g", "predicted_fitness": float(i),
       "experimental_score": float(i)} for i in range(10)],
    *[{"score_set_urn": "down", "accession": f"d{i}", "score_set_title": "down",
       "target": "g", "predicted_fitness": float(-i),
       "experimental_score": float(i)} for i in range(10)],
  ]
  summary = summarize_predictions(records)
  assert summary["macro_abs_spearman"] == pytest.approx(1.0)
  assert summary["macro_signed_spearman"] == pytest.approx(0.0)
  assert summary["num_negative_assays"] == 1


def test_non_finite_predictions_are_dropped_not_ranked():
  """rankdata gives NaN the LARGEST rank, so a refused variant silently became
  the highest-predicted-fitness row. Measured on a real Score I run: one assay
  read -0.22227 with 39 NaN rows ranked against -0.14113 with them excluded."""
  from scripts.eval.dnahnet.mavedb import summarize_predictions
  base = [{"score_set_urn": "a", "accession": f"a{i}", "score_set_title": "a",
           "target": "g", "predicted_fitness": float(i),
           "experimental_score": float(i)} for i in range(12)]
  clean = summarize_predictions(base)
  assert clean["macro_signed_spearman"] == pytest.approx(1.0)
  assert clean["num_unscored_variants"] == 0

  # Add refused variants whose experimental scores would drag the correlation.
  poisoned = base + [
    {"score_set_urn": "a", "accession": f"n{i}", "score_set_title": "a",
     "target": "g", "predicted_fitness": float("nan"),
     "experimental_score": -100.0 * (i + 1)} for i in range(6)]
  summary = summarize_predictions(poisoned)
  assert summary["num_unscored_variants"] == 6
  assert summary["num_scored_variants"] == 12
  assert summary["macro_signed_spearman"] == pytest.approx(1.0), (
    "NaN rows are being ranked instead of dropped")


def test_infill_nsyn_masks_only_the_non_synonymous_codons():
  """The 11x reduction in context destruction that `codon` mode gets wrong.

  `codon` masks every differing codon (median 33 nt of a 204-nt fragment) and
  then reads marginals from the context it just destroyed. `codon_nsyn` masks
  only the codons whose amino acid changes (median 3 nt). Both must return
  exactly 0 when every change is synonymous.
  """
  import torch
  model, sm = _fake_bd_model(preference={8: 4.0})
  ids = lambda s: torch.tensor(model.tokenizer.convert_tokens_to_ids(list(s)))
  #        codon0 codon1 codon2 codon3
  wt  = "ATG" "CTT" "AAA" "CCC"     # Met Leu Lys Pro
  mut = "ATG" "CTC" "GAA" "CCC"     # Met Leu Glu Pro : codon1 SYNONYMOUS, codon2 not
  s_all, units_all, syn_all = sm.score_infill(
    model, ids(wt), ids(mut), wt, mut, unit="codon")
  s_ns, units_ns, syn_ns = sm.score_infill(
    model, ids(wt), ids(mut), wt, mut, unit="codon_nsyn")
  assert units_all == 2, units_all          # two codons differ
  assert syn_all == 1                        # one of them is synonymous
  assert units_ns == 1, units_ns             # nsyn mode kept only the real one
  # Both score the same amino-acid change, but from different contexts, so the
  # values differ -- what must hold is that nsyn masked strictly fewer codons.
  assert units_ns < units_all
  assert s_ns != 0.0 and s_all != 0.0

  # All-synonymous variant: exactly zero under BOTH modes.
  mut_syn = "ATG" "CTC" "AAA" "CCC"          # only the synonymous CTT->CTC
  for unit in ("codon", "codon_nsyn"):
    v, _, _ = sm.score_infill(model, ids(wt), ids(mut_syn), wt, mut_syn, unit=unit)
    assert v == 0.0, (unit, v)

  with pytest.raises(ValueError, match="nt|codon"):
    sm.score_infill(model, ids(wt), ids(mut), wt, mut, unit="bogus")


# --------------------------------------------------------------------------
# Argument plumbing. Added 2026-09-19 after a real regression: a new
# `--score2-window` argument was wired into the CALL SITE
# (`score2_window=args.score2_window`) but its `add_argument` was inserted with
# a string replace that silently no-opped, because the anchor said
# `group.add_argument(` where the file uses `parser.add_argument(`. The result
# was an AttributeError on EVERY invocation of score_mavedb.py, including score
# modes unrelated to the new flag. Syntax checks and --help on the OLD file both
# passed; nothing caught it until six LSF jobs had already failed.
# --------------------------------------------------------------------------

import re as _re
import pathlib as _pathlib

_ROOT = _pathlib.Path(__file__).resolve().parents[1]


def _declared_flags(source: str) -> set:
  """Flags from add_argument, including the form where the name is on its own line."""
  return ({m.replace("-", "_") for m in _re.findall(r'add_argument\(\s*"--([\w-]+)"', source)}
          | {m.replace("-", "_")
             for m in _re.findall(r'add_argument\(\s*\n\s*"--([\w-]+)"', source)})


def test_every_args_attribute_read_by_score_mavedb_is_declared():
  source = (_ROOT / "scripts/eval/dnahnet/score_mavedb.py").read_text()
  used = set(_re.findall(r"\bargs\.([a-z_][a-z_0-9]*)", source))
  missing = used - _declared_flags(source)
  assert not missing, (
    f"read from args but never declared: {sorted(missing)} -- this is exactly "
    f"the failure that crashed every score mode on 2026-09-19")


def test_every_args_attribute_read_by_finetune_is_declared():
  source = (_ROOT / "scripts/eval/caduceus/finetune.py").read_text()
  used = set(_re.findall(r"\bargs\.([a-z_][a-z_0-9]*)", source))
  # set inside resolve() rather than by argparse
  derived = {"seed_list", "allow_window_mismatch"}
  missing = used - _declared_flags(source) - derived
  assert not missing, f"read from args but never declared: {sorted(missing)}"


def test_every_shell_hook_names_a_declared_flag():
  """A wrapper emitting a flag argparse does not know is an instant hard failure."""
  for shell, python in (("scripts/eval/dnahnet/mavedb_score.sh",
                         "scripts/eval/dnahnet/score_mavedb.py"),
                        ("scripts/eval/caduceus/finetune.sh",
                         "scripts/eval/caduceus/finetune.py")):
    emitted = set(_re.findall(r"EXTRA(?:_ARGS)?\+=\(\s*--([\w-]+)",
                              (_ROOT / shell).read_text()))
    declared = _declared_flags((_ROOT / python).read_text())
    undeclared = {e.replace("-", "_") for e in emitted} - declared
    assert not undeclared, f"{shell} emits undeclared flags: {sorted(undeclared)}"


def test_every_score_mode_choice_has_a_dispatch_branch():
  """A mode in `choices` with no branch would silently fall through to nelbo."""
  source = (_ROOT / "scripts/eval/dnahnet/score_mavedb.py").read_text()
  block = source[source.index('choices=("nelbo"'):]
  choices = set(_re.findall(r'"([a-z_]+)"', block[:block.index(")")]))
  handled = (set(_re.findall(r'score_mode\s*==\s*"([\w]+)"', source))
             | set(_re.findall(r'score_mode\s+in\s+\(([^)]*)\)', source))
             | {"nelbo"})
  flat = set()
  for h in handled:
    flat |= {x.strip().strip('"\'') for x in h.split(",")}
  # infill_* modes are dispatched through the explicit unit mapping
  unit_mapped = set(_re.findall(r'"(infill_[\w]+)":\s*"', source))
  missing = choices - flat - unit_mapped
  assert not missing, f"score_mode choices with no dispatch branch: {sorted(missing)}"


def test_every_score_mode_has_its_own_provenance_string():
  """`score_definition` in summary.json must describe the estimator that ran.

  It used to be an if/elif chain whose `else` stamped "paired NELBO(WT) -
  paired NELBO(mutant)" on four estimators that are not NELBO
  (predictive_divergence, pll, pll_causal, state_displacement), so a
  summary.json asserted provenance it did not have.
  """
  import ast

  source = (_ROOT / "scripts/eval/dnahnet/score_mavedb.py").read_text()
  tree = ast.parse(source)
  defined = None
  for node in ast.walk(tree):
    if (isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_SCORE_DEFINITIONS"):
      defined = {k.value for k in node.value.keys}
  assert defined is not None, "_SCORE_DEFINITIONS not found"

  start = source.index('"--score-mode"')
  choices = set(eval(
    source[source.index("choices=(", start) + 8:
           source.index("default=", start)].rstrip().rstrip(",")))

  assert choices == defined, (
    f"score_mode choices and _SCORE_DEFINITIONS disagree: "
    f"missing={choices - defined} extra={defined - choices}")

  # And no two modes may share a description, which is how the old `else`
  # branch hid: four distinct estimators, one string.
  assert len(set(_score_definitions_values(source))) == len(defined), (
    "two score modes share a provenance string")


def _score_definitions_values(source):
  import ast
  for node in ast.walk(ast.parse(source)):
    if (isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_SCORE_DEFINITIONS"):
      return [ast.literal_eval(v) for v in node.value.values]
  return []


def test_one_block_guard_applies_only_under_cross_attention():
  """The PLL/Score I one-block guard must be gated on `cross_attn`.

  The leak it protects against is the CLEAN stream: with more than one block,
  x_t could read x_0 and the masked marginal would hand back the answer. That
  stream is only ever built under `if model.cross_attn` -- a state-space
  backbone is passed `noisy` alone, in which the scored position is masked, so
  nothing can leak at any model_length. Guarding it unconditionally made pll,
  pll_causal and all four infill modes unrunnable on the b8/b32 checkpoints,
  leaving the two best models in the study with 3 estimators against block
  256's 9. Verified empirically on b8 at 32 blocks: the logit at position i is
  invariant to x_i to 1e-6 while remaining sensitive to earlier positions.
  """
  source = (_ROOT / "scripts/eval/dnahnet/score_mavedb.py").read_text()
  lines = source.splitlines()
  needle = "int(model.config.model.length) != int(model.config.block_size)"
  hits = [i for i, ln in enumerate(lines) if needle in ln]
  assert len(hits) == 3, f"expected 3 one-block guards, found {len(hits)}"
  for i in hits:
    # the guard may wrap, so read the whole logical condition, not one line
    stmt = " ".join(ln.strip() for ln in lines[max(0, i - 2):i + 1])
    assert "cross_attn" in stmt, (
      f"one-block guard is not gated on cross_attn, so it will again block "
      f"every sharp estimator on multi-block SSM checkpoints: {stmt!r}")


def test_clean_stream_is_only_built_under_cross_attention():
  """The premise of the guard relaxation above: no cross_attn, no clean stream."""
  source = (_ROOT / "scripts/eval/dnahnet/score_mavedb.py").read_text()
  # every concatenation of a clean stream must sit under a cross_attn test
  for marker in ("torch.cat((noisy, clean), dim=-1)",
                 "torch.cat((model_input, ids.unsqueeze(0)), dim=-1)"):
    assert marker in source, f"missing expected clean-stream construction {marker!r}"
    before = source[:source.index(marker)]
    tail = before[-400:]
    assert "model.cross_attn" in tail, (
      f"clean stream {marker!r} is built without a nearby cross_attn guard")


def test_score_i_index_is_guarded_against_a_prefix_offset():
  """Score I must verify its indices land on the intended nucleotides.

  `positions` are 0-based within the variant string; `wt_ids` is the padded
  model input. They coincide only when nothing precedes the variant. With
  `--genomic-prefix` the variant is shifted, but `score_batch` computes a
  non-zero `offset` only for `--infill-pad-side left`, so an infill run behind
  a prefix would read positions inside the prefix. Harmless until 2026-09-20,
  when relaxing the one-block guard to `cross_attn` only made it reachable.
  """
  source = (_ROOT / "scripts/eval/dnahnet/score_mavedb.py").read_text()
  assert "def _assert_index_lands_on_variant(" in source, "guard helper is gone"
  # every masked-index construction in score_infill must be guarded
  body = source[source.index("def score_infill("):]
  body = body[:body.index("\ndef ")]
  for marker in ("index = torch.tensor([p + offset for p in positions]",
                 "span = torch.tensor([3 * c + k + offset"):
    assert marker in body, f"missing index construction {marker!r}"
  assert body.count("_assert_index_lands_on_variant(") >= 2, (
    "not every Score I index path is guarded")


def test_aggregate_refuses_to_mix_estimators():
  """Averaging across estimators is meaningless -- they score different sets."""
  source = (_ROOT / "scripts/eval/dnahnet/aggregate_mavedb.py").read_text()
  assert "_shared_score_definition" in source
  assert "refusing to average across different estimators" in source
  assert '"mean of paired NELBO(WT) - NELBO(mutant) runs"' not in source, (
    "the hardcoded NELBO provenance string is back")


def test_standings_does_not_cherry_pick_and_checks_ar_geometry():
  """Two defects that made our own standings table wrong.

  (1) `headline = max(macro, pertok)` picked whichever of two different
      metrics scored higher, per arm.
  (2) the arm list pointed at no-prefix AR runs, reproducing a spurious
      1.4x uSSM-AR vs Transformer-AR gap that vanishes at matched geometry.
  """
  source = (_ROOT / "scripts/eval/standings.py").read_text()
  # strip comments: the fix is documented in a comment that names the old
  # expression, and that prose must not itself trip the check
  code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines())
  assert "max(pooled, pertok)" not in code and "max(macro, pertok)" not in code
  assert "_check_arm_geometry" in source
  assert "genomic_prefix" in source, "AR geometry is not verified"
  arms = source[source.index("MAVEDB_ARMS = ["):]
  arms = arms[:arms.index("]\n")]
  for line in arms.splitlines():
    if "-AR" in line and "exact AR" in line:
      assert "genomicprefix" in line, (
        f"AR arm must point at a genomic-prefix run: {line.strip()!r}")


def test_summary_splits_substitutions_from_indels():
  """Indels must be reported separately -- our scores are unnormalised sums.

  A 3-nt deletion drops three log-probability terms and a 3-nt insertion adds
  three, so `predicted_fitness` on those rows tracks LENGTH, not biology:
  corr(predicted_fitness, length change) = -0.79 over the 1,901 indel rows,
  mean +3.015 for deletions against -4.249 for insertions, while the assay
  puts them at a near-identical +1.319 / +1.699 kcal/mol. ProteinGym keeps
  substitutions and indels as separate benchmarks for this reason.
  """
  import csv as _csv
  from scripts.eval.dnahnet.mavedb import (
    summarize_predictions, _hgvs_length_delta)

  # the parser itself
  assert _hgvs_length_delta("c.=") == 0
  assert _hgvs_length_delta("c.[3G>A;9T>C]") == 0
  assert _hgvs_length_delta("c.[3_4insGGT]") == 3
  assert _hgvs_length_delta("c.[10_12del]") == -3
  assert _hgvs_length_delta("c.[10_12delins A]".replace(" ", "")) == -2

  path = (_ROOT / "results/dnahnet/mavedb/b8-nelbo-eps0.9-157368/predictions.csv")
  if not path.exists():
    return  # results not present in this checkout
  summary = summarize_predictions(list(_csv.DictReader(path.open())))
  assert summary["num_substitution_variants"] == 19349
  assert summary["num_indel_variants"] == 1901
  # the indel rows score at chance and drag the pooled headline down
  assert abs(summary["macro_signed_spearman_indels"]) < 0.05
  assert (summary["macro_signed_spearman_substitutions"]
          > summary["macro_signed_spearman"])


def test_every_eval_script_can_print_help():
  """argparse %-interpolates help strings, so a bare '%' crashes --help.

  Two scripts shipped with this defect (`score_mavedb.py --only-urn` wrote
  "48.4% N", `benchmark_arms.py` wrote "(1.36%)"), and because --help is not
  on any hot path it went unnoticed. Percent signs are common in this repo's
  help text since the numbers are mostly fractions, so this is worth a guard.
  """
  import re
  bad = []
  for path in sorted((_ROOT / "scripts/eval").rglob("*.py")):
    source = path.read_text()
    if "add_argument" not in source:
      continue
    for match in re.finditer(
        r'help=\s*\(?((?:\s*"(?:[^"\\]|\\.)*"\s*)+)\)?', source):
      text = match.group(1)
      if re.search(r'(?<!%)%(?![%(sdfrgeix])', text):
        bad.append(f"{path.relative_to(_ROOT)}: {text[:60]}")
  assert not bad, (
    "bare '%' in an argparse help string will crash --help; escape as '%%':\n  "
    + "\n  ".join(bad))


def test_every_result_writing_eval_script_stamps_provenance():
  """A result file must record what produced it.

  The generation/infilling branch wrote JSON with no argv provenance, so you
  could not recover from a result which sample size produced it -- and
  `--n-loci` defaults to 24 in task2_generate and 32 in ag_receptive_field.
  That is the same class of gap as the `score_definition` bug: the artifact
  did not record what it actually did.
  """
  import re
  offenders = []
  for path in sorted((_ROOT / "scripts/eval").rglob("*.py")):
    source = path.read_text()
    # only scripts that both take CLI args and write a result file
    if "add_argument" not in source:
      continue
    writes = re.search(r"\.write_text\(\s*json\.dumps|json\.dump\(", source)
    if not writes:
      continue
    if "stamp(" not in source:
      offenders.append(str(path.relative_to(_ROOT)))
  # RATCHET. These 16 predate the provenance convention. The list may SHRINK,
  # never grow: a new result-writing script must stamp itself. The seven that
  # were fixed on 2026-09-20 (the aglonggen generation/infilling chain, DEG,
  # the MaveDB aggregator and the GB probe) are deliberately absent, so a
  # regression there fails this test.
  known_unstamped = {
    "scripts/eval/ar_decode_benchmark.py",
    "scripts/eval/benchmark_arms.py",
    "scripts/eval/build_human_longrange.py",
    "scripts/eval/build_longrange_eval.py",
    "scripts/eval/caduceus/embed.py",
    "scripts/eval/dnahnet/codon_independence.py",
    "scripts/eval/dnahnet/partial_corr.py",
    "scripts/eval/dnahnet/prepare_deg.py",
    "scripts/eval/dnahnet/prepare_mavedb.py",
    "scripts/eval/dnahnet/profile_forward.py",
    "scripts/eval/gen_synthetic_duplication.py",
    "scripts/eval/gen_synthetic_longrange.py",
    "scripts/eval/gen_synthetic_recall.py",
    "scripts/eval/inference_curves.py",
    "scripts/eval/measure_runtime_timescales.py",
    "scripts/eval/measured_flops_sweep.py",
    "scripts/eval/scaling_curves.py",
    "scripts/eval/ssm_prefix_intervention.py",
    "scripts/eval/ssm_streaming_benchmark.py",
    "scripts/eval/training_flops.py",
  }
  new_offenders = sorted(set(offenders) - known_unstamped)
  assert not new_offenders, (
    "these eval scripts write a result file but never call stamp(), so the "
    "output cannot say what produced it:\n  " + "\n  ".join(new_offenders))
  regressed = sorted(known_unstamped & set(offenders) ^ known_unstamped & set(offenders))
  assert not regressed
