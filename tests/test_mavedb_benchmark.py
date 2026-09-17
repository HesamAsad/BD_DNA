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
