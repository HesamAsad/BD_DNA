import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval.dnahnet.mavedb import (
  apply_coding_hgvs,
  build_records,
  load_manifest,
  rankdata,
  spearmanr,
  summarize_predictions,
)


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
