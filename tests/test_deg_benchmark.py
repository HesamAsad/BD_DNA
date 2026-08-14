"""CPU-only unit tests for the dnaHNet DEG gene-essentiality harness.

Mirrors tests/test_mavedb_benchmark.py: no GPU, no network, no checkpoint.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval.dnahnet.deg import (
  KNOCKOUT_OFFSET,
  STOP_CASSETTE,
  WINDOW_LENGTH,
  EssentialGeneMatcher,
  apply_knockout,
  baseline_scores,
  build_window_record,
  centered_window_start,
  extract_window,
  feature_name_keys,
  gc_content,
  gene_name_aliases,
  hamming_positions,
  knockout_from_record,
  normalize_accessions,
  reverse_complement,
  roc_auc,
  sequence_identity,
  summarize_deg_predictions,
)
from scripts.eval.dnahnet.prepare_deg import validate_record
from scripts.eval.dnahnet.score_deg import (
  _assemble,
  assert_sign_convention,
  baseline_records,
)


MANIFEST = (
  Path(__file__).resolve().parents[1]
  / "scripts/eval/dnahnet/deg_manifest.json")

RNG = np.random.default_rng(20260814)


def _random_contig(length):
  return "".join(RNG.choice(list("ACGT"), size=length))


# ---------------------------------------------------------------------------
# Published protocol constants
# ---------------------------------------------------------------------------

def test_protocol_constants_match_the_paper():
  assert STOP_CASSETTE == "TAATAATAATAGTGA"
  assert len(STOP_CASSETTE) == 15
  # Five stop codons back to back, so a stop is hit in any reading frame.
  assert [STOP_CASSETTE[i:i + 3] for i in range(0, 15, 3)] == [
    "TAA", "TAA", "TAA", "TAG", "TGA"]
  assert WINDOW_LENGTH == 8192
  assert KNOCKOUT_OFFSET == 12
  # Every checkpoint block size in this repository must divide the window.
  for block_size in (1, 128, 256, 512):
    assert WINDOW_LENGTH % block_size == 0


def test_manifest_is_pinned_and_self_consistent():
  manifest = json.loads(MANIFEST.read_text())
  assert manifest["protocol"]["window_length"] == WINDOW_LENGTH
  assert manifest["protocol"]["stop_cassette"] == STOP_CASSETTE
  assert manifest["protocol"]["knockout_offset"] == KNOCKOUT_OFFSET
  assert manifest["num_organisms"] == len(manifest["organisms"])
  ids = [row["organism_id"] for row in manifest["organisms"]]
  assert len(set(ids)) == len(ids)
  for name, entry in manifest["sources"]["deg_files"].items():
    assert len(entry["sha256"]) == 64, name
    assert entry["bytes"] > 0, name
  # The organism-count divergence from the paper's 62 must stay documented.
  assert manifest["paper_reported"]["organisms"] == 62
  assert any(text.startswith("A1 organism count") for text in
             manifest["assumptions"])


# ---------------------------------------------------------------------------
# Knockout construction and offset arithmetic
# ---------------------------------------------------------------------------

def test_apply_knockout_is_a_length_preserving_substitution():
  window = "A" * 100
  mutated = apply_knockout(window, 12)
  assert len(mutated) == len(window)
  assert mutated[12:27] == STOP_CASSETTE
  assert mutated[:12] == window[:12]
  assert mutated[27:] == window[27:]
  # 'A' appears inside the cassette, so fewer than 15 positions change.
  assert hamming_positions(window, mutated) == sum(
    1 for base in STOP_CASSETTE if base != "A")


def test_apply_knockout_changes_exactly_fifteen_positions_when_disjoint():
  window = "C" * 100                       # C never occurs in the cassette
  mutated = apply_knockout(window, 40)
  assert hamming_positions(window, mutated) == 15
  changed = [i for i in range(100) if window[i] != mutated[i]]
  assert changed == list(range(40, 55))


def test_apply_knockout_rejects_a_cassette_that_does_not_fit():
  with pytest.raises(ValueError, match="does not fit"):
    apply_knockout("ACGT" * 5, 10)
  with pytest.raises(ValueError, match="does not fit"):
    apply_knockout("ACGT" * 10, -1)


def test_knockout_offset_is_twelve_bases_into_the_coding_sequence():
  contig = _random_contig(20000)
  gene_start, gene_end = 5000, 5600
  contig = contig[:gene_start] + "ATG" + contig[gene_start + 3:]
  geometry = build_window_record(contig, gene_start, gene_end, 1)
  window = geometry["window_sequence"]
  assert window[geometry["cds_offset"]:geometry["cds_offset"] + 3] == "ATG"
  assert geometry["cassette_start"] == geometry["cds_offset"] + KNOCKOUT_OFFSET
  knockout = apply_knockout(window, geometry["cassette_start"])
  # Codons 1-4 of the gene survive; the cassette occupies codons 5-9.
  assert knockout[geometry["cds_offset"]:geometry["cassette_start"]] == \
      window[geometry["cds_offset"]:geometry["cassette_start"]]
  assert knockout[
    geometry["cassette_start"]:geometry["cassette_start"] + 15] == \
      STOP_CASSETTE


def test_alternative_offset_reading_is_reachable_by_flag():
  contig = _random_contig(20000)
  default = build_window_record(contig, 5000, 5600, 1)
  alternative = build_window_record(contig, 5000, 5600, 1, knockout_offset=15)
  assert alternative["cassette_start"] - default["cassette_start"] == 3
  # Both readings keep the cassette in frame with the start codon.
  for geometry in (default, alternative):
    assert (geometry["cassette_start"] - geometry["cds_offset"]) % 3 == 0


# ---------------------------------------------------------------------------
# Window extraction and centring
# ---------------------------------------------------------------------------

def test_window_is_centred_on_the_gene_midpoint():
  contig = _random_contig(100000)
  start, clamped = centered_window_start(len(contig), 40000, 40600)
  assert not clamped
  assert start == (40000 + 40600) // 2 - WINDOW_LENGTH // 2
  window = extract_window(contig, start)
  assert len(window) == WINDOW_LENGTH
  assert window == contig[start:start + WINDOW_LENGTH]


def test_circular_contig_wraps_across_the_origin():
  contig = _random_contig(20000)
  # Gene straddling coordinate 0 is impossible for a simple location, but a
  # gene near the origin still needs the left half of its window from the end.
  geometry = build_window_record(contig, 100, 700, 1, circular=True)
  assert geometry["window_wrapped"]
  window = geometry["window_sequence"]
  assert len(window) == WINDOW_LENGTH
  start = geometry["window_start"]
  expected = "".join(contig[(start + i) % len(contig)]
                     for i in range(WINDOW_LENGTH))
  assert window == expected
  # Centring survives the wrap.
  assert (400 - WINDOW_LENGTH // 2) % len(contig) == start


def test_circular_wrap_at_the_far_end_of_the_contig():
  contig = _random_contig(20000)
  geometry = build_window_record(contig, 19300, 19900, 1, circular=True)
  assert geometry["window_wrapped"]
  window = geometry["window_sequence"]
  assert len(window) == WINDOW_LENGTH
  assert window[geometry["cds_offset"]:geometry["cds_offset"] + 6] == \
      contig[19300:19306]


def test_linear_contig_clamps_the_window_inward_and_keeps_the_length():
  contig = _random_contig(20000)
  geometry = build_window_record(contig, 100, 700, 1, circular=False)
  assert geometry["window_clamped"]
  assert not geometry["window_wrapped"]
  assert geometry["window_start"] == 0
  assert len(geometry["window_sequence"]) == WINDOW_LENGTH
  assert geometry["window_sequence"] == contig[:WINDOW_LENGTH]
  assert geometry["cds_offset"] == 100

  far = build_window_record(contig, 19300, 19900, 1, circular=False)
  assert far["window_clamped"]
  assert far["window_start"] == 20000 - WINDOW_LENGTH


def test_contig_shorter_than_the_window_is_rejected():
  with pytest.raises(ValueError, match="shorter than"):
    centered_window_start(4000, 100, 700)
  with pytest.raises(ValueError, match="shorter than"):
    extract_window(_random_contig(4000), 0)


def test_gene_longer_than_the_window_is_rejected_not_silently_identical():
  contig = _random_contig(60000)
  with pytest.raises(ValueError, match="outside the centred window"):
    build_window_record(contig, 20000, 30000, 1)


# ---------------------------------------------------------------------------
# Strand handling
# ---------------------------------------------------------------------------

def test_minus_strand_gene_is_reverse_complemented_into_gene_orientation():
  contig = list(_random_contig(40000))
  # Put a reverse-strand ATG at the gene's 3'-most plus coordinate.
  contig[20597:20600] = list(reverse_complement("ATG"))
  contig = "".join(contig)
  geometry = build_window_record(contig, 20000, 20600, -1, orientation="gene")
  window = geometry["window_sequence"]
  assert geometry["cassette"] == STOP_CASSETTE
  assert window[geometry["cds_offset"]:geometry["cds_offset"] + 3] == "ATG"
  # The window really is the reverse complement of the plus-strand window.
  plus = contig[geometry["window_start"]:
                geometry["window_start"] + WINDOW_LENGTH]
  assert window == reverse_complement(plus)
  knockout = apply_knockout(window, geometry["cassette_start"])
  assert knockout[
    geometry["cassette_start"]:geometry["cassette_start"] + 15] == \
      STOP_CASSETTE


def test_reference_orientation_writes_the_reverse_complement_cassette():
  contig = _random_contig(40000)
  geometry = build_window_record(
    contig, 20000, 20600, -1, orientation="reference")
  assert geometry["cassette"] == reverse_complement(STOP_CASSETTE)
  assert geometry["window_sequence"] == contig[
    geometry["window_start"]:geometry["window_start"] + WINDOW_LENGTH]
  knockout = knockout_from_record({
    "window_sequence": geometry["window_sequence"],
    "cassette_start": geometry["cassette_start"],
    "cassette": geometry["cassette"]})
  # Read back in gene orientation, the literal cassette reappears.
  gene_oriented = reverse_complement(knockout)
  position = WINDOW_LENGTH - geometry["cassette_start"] - 15
  assert gene_oriented[position:position + 15] == STOP_CASSETTE


def test_both_orientations_describe_the_same_edit():
  contig = _random_contig(40000)
  gene = build_window_record(contig, 20000, 20600, -1, orientation="gene")
  reference = build_window_record(
    contig, 20000, 20600, -1, orientation="reference")
  gene_ko = knockout_from_record(
    {"window_sequence": gene["window_sequence"],
     "cassette_start": gene["cassette_start"], "cassette": gene["cassette"]})
  reference_ko = knockout_from_record(
    {"window_sequence": reference["window_sequence"],
     "cassette_start": reference["cassette_start"],
     "cassette": reference["cassette"]})
  assert gene_ko == reverse_complement(reference_ko)


def test_build_window_record_rejects_a_bad_strand():
  with pytest.raises(ValueError, match="strand"):
    build_window_record(_random_contig(20000), 100, 700, 0)


# ---------------------------------------------------------------------------
# The record-level validation used by prepare_deg.py
# ---------------------------------------------------------------------------

def test_validate_record_accepts_a_well_formed_record():
  contig = _random_contig(40000)
  geometry = build_window_record(contig, 20000, 20600, 1)
  record = dict(geometry, locus_tag="t1", essential=1)
  record["window_sequence"] = geometry["window_sequence"]
  differences = validate_record(record, WINDOW_LENGTH, STOP_CASSETTE)
  assert 1 <= differences <= 15


def test_validate_record_rejects_a_no_op_knockout():
  """A window that already carries the cassette would make WT == knockout."""
  contig = _random_contig(40000)
  geometry = build_window_record(contig, 20000, 20600, 1)
  window = list(geometry["window_sequence"])
  start = geometry["cassette_start"]
  window[start:start + 15] = list(STOP_CASSETTE)
  record = dict(geometry, locus_tag="t1", essential=1)
  record["window_sequence"] = "".join(window)
  with pytest.raises(ValueError, match="identical to the wild type"):
    validate_record(record, WINDOW_LENGTH, STOP_CASSETTE)


def test_validate_record_rejects_a_non_binary_label():
  contig = _random_contig(40000)
  geometry = build_window_record(contig, 20000, 20600, 1)
  record = dict(geometry, locus_tag="t1", essential=2)
  with pytest.raises(ValueError, match="not binary"):
    validate_record(record, WINDOW_LENGTH, STOP_CASSETTE)


def test_validate_record_rejects_a_mislabelled_cassette_offset():
  contig = _random_contig(40000)
  geometry = build_window_record(contig, 20000, 20600, 1)
  record = dict(geometry, locus_tag="t1", essential=1)
  # apply_knockout writes at cassette_start, so a record claiming the cassette
  # sits elsewhere must not validate.
  record["cassette"] = reverse_complement(STOP_CASSETTE)
  with pytest.raises(ValueError, match="not TAATAATAATAGTGA"):
    validate_record(record, WINDOW_LENGTH, STOP_CASSETTE)


def test_validate_record_rejects_a_wrong_length_window():
  contig = _random_contig(40000)
  geometry = build_window_record(contig, 20000, 20600, 1)
  record = dict(geometry, locus_tag="t1", essential=1)
  record["window_sequence"] = record["window_sequence"][:-1]
  with pytest.raises(ValueError, match="window is"):
    validate_record(record, WINDOW_LENGTH, STOP_CASSETTE)


# ---------------------------------------------------------------------------
# Label parsing and essential-gene matching
# ---------------------------------------------------------------------------

def test_gene_name_aliases_split_deg_synonym_notation():
  assert gene_name_aliases("tmk/tdk") == {"tmk", "tdk"}
  assert gene_name_aliases("metS,metG") == {"mets", "metg"}
  assert gene_name_aliases("-") == set()
  assert gene_name_aliases("") == set()
  assert gene_name_aliases("dnaA") == {"dnaa"}


def test_feature_name_keys_cover_every_ncbi_name_field():
  qualifiers = {
    "gene": ["rpsL"], "locus_tag": ["BSU_00010"],
    "old_locus_tag": ["BSU00010"], "gene_synonym": ["strA/rpsL12"],
    "product": ["ignored"]}
  assert feature_name_keys(qualifiers) == {
    "rpsl", "bsu_00010", "bsu00010", "stra", "rpsl12"}


def test_normalize_accessions_handles_every_deg_separator():
  assert normalize_accessions("NC_000964") == ("NC_000964",)
  assert normalize_accessions("NC_002505,NC_002506") == (
    "NC_002505", "NC_002506")
  assert normalize_accessions("NC_007650;NC_007651") == (
    "NC_007650", "NC_007651")
  assert normalize_accessions("AM747720 AM747723") == (
    "AM747720", "AM747723")
  assert normalize_accessions("NC_003295.1; NC_003296.1") == (
    "NC_003295", "NC_003296")
  assert normalize_accessions("-") == ()


def test_sequence_identity_is_ungapped_and_length_sensitive():
  assert sequence_identity("ACGT" * 25, "ACGT" * 25) == 1.0
  left = "A" * 200
  right = "A" * 199 + "C"
  assert sequence_identity(left, right) == pytest.approx(0.995)
  assert sequence_identity("ACGT", "ACG") == 0.0


def test_matcher_labels_by_name_by_sequence_and_by_both():
  essential = _random_contig(300)
  matcher = EssentialGeneMatcher(["dnaA", "tmk/tdk"], [essential])
  assert matcher.match({"dnaa"}, "ACGT" * 10) == "name"
  assert matcher.match({"tdk"}, "ACGT" * 10) == "name"
  assert matcher.match(set(), essential) == "sequence"
  assert matcher.match({"dnaa"}, essential) == "both"
  assert matcher.match({"yfla"}, "ACGT" * 10) == "none"


def test_matcher_accepts_the_reverse_complement_and_99_percent_identity():
  essential = _random_contig(400)
  matcher = EssentialGeneMatcher([], [essential])
  assert matcher.match(set(), reverse_complement(essential)) == "sequence"
  # Three mismatches in 400 nt is 99.25% identity: above the >99% threshold.
  near = list(essential)
  for position in (10, 200, 390):
    near[position] = "ACGT"[("ACGT".index(near[position]) + 1) % 4]
  assert matcher.match(set(), "".join(near)) == "sequence"
  # Five mismatches is 98.75%: below it.
  far = list(essential)
  for position in (10, 100, 200, 300, 390):
    far[position] = "ACGT"[("ACGT".index(far[position]) + 1) % 4]
  assert matcher.match(set(), "".join(far)) == "none"


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_roc_auc_against_a_hand_computed_case():
  # Labels 1,0,1,0 with scores 4,3,2,1.
  # Positive scores {4, 2}; negative {3, 1}. Concordant pairs: (4>3), (4>1),
  # (2>1) = 3 of 4, so AUROC = 3/4.
  assert roc_auc([1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.75)
  assert roc_auc([1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)


def test_roc_auc_is_sign_sensitive_and_bounded():
  assert roc_auc([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0]) == 1.0
  assert roc_auc([0, 0, 1, 1], [1.0, 0.9, 0.1, 0.0]) == 0.0
  assert roc_auc([0, 1], [5.0, 5.0]) == pytest.approx(0.5)


def test_roc_auc_mid_ranks_ties():
  # One tie between a positive and a negative contributes exactly one half.
  # Positives {2, 1}, negatives {2, 0}: pairs (2 vs 2)=0.5, (2 vs 0)=1,
  # (1 vs 2)=0, (1 vs 0)=1 -> 2.5/4.
  assert roc_auc([1, 0, 1, 0], [2.0, 2.0, 1.0, 0.0]) == pytest.approx(0.625)


def test_roc_auc_matches_sklearn_on_random_data():
  from sklearn.metrics import roc_auc_score
  rng = np.random.default_rng(7)
  for _ in range(20):
    labels = rng.integers(0, 2, size=200)
    if labels.min() == labels.max():
      continue
    scores = rng.normal(size=200) + labels * 0.4
    scores = np.round(scores, 1)          # force ties
    assert roc_auc(labels, scores) == pytest.approx(
      roc_auc_score(labels, scores))


def test_roc_auc_returns_nan_for_a_single_class():
  assert np.isnan(roc_auc([1, 1, 1], [1.0, 2.0, 3.0]))
  assert np.isnan(roc_auc([0, 0], [1.0, 2.0]))


def test_roc_auc_rejects_non_finite_scores():
  with pytest.raises(ValueError, match="non-finite"):
    roc_auc([0, 1], [0.0, float("nan")])


# ---------------------------------------------------------------------------
# Summary: pooled vs macro, and the mandatory model-free baselines
# ---------------------------------------------------------------------------

def _summary_records(n_per_organism=40):
  """Two organisms with very different base rates and no within-organism signal."""
  rng = np.random.default_rng(3)
  records = []
  for organism, base_rate in (("orgA", 0.1), ("orgB", 0.9)):
    for index in range(n_per_organism):
      essential = int(index < base_rate * n_per_organism)
      records.append({
        "organism_id": organism,
        "essential": essential,
        "gene_length": float(rng.integers(300, 3000)),
        "window_gc": float(rng.random()),
        "gene_start": float(rng.integers(0, 4_000_000)),
        "essentiality_score": float(rng.normal()),
        "essentiality_score_local": float(rng.normal()),
        "essentiality_score_downstream": float(rng.normal()),
      })
  return records


def test_summary_reports_pooled_macro_and_the_sign_flipped_value():
  summary = summarize_deg_predictions(
    _summary_records(),
    score_fields=("essentiality_score", "essentiality_score_local"))
  assert summary["headline_metric"] == "macro mean per-organism AUROC"
  assert summary["num_organisms"] == 2
  assert summary["num_genes"] == 80
  assert len(summary["per_organism"]) == 2
  assert summary["macro_auroc_if_sign_flipped"] == pytest.approx(
    1.0 - summary["macro_auroc"])
  assert set(summary["score_variants"]) == {
    "essentiality_score", "essentiality_score_local"}


def test_organism_base_rate_baseline_exposes_the_pooling_confound():
  """The whole reason macro, not pooled, is the headline."""
  summary = summarize_deg_predictions(_summary_records())
  baseline = summary["baselines"]["organism_base_rate"]
  # Constant within an organism, so it cannot discriminate within one...
  assert baseline["macro_auroc"] == pytest.approx(0.5)
  # ...yet pooled it looks like a strong predictor, purely from base rates.
  assert baseline["pooled_auroc"] > 0.8


def test_random_control_lands_near_one_half():
  rng = np.random.default_rng(11)
  records = [{
    "organism_id": "org",
    "essential": int(rng.random() < 0.3),
    "gene_length": 1000.0,
    "window_gc": 0.5,
    "gene_start": 0.0,
    "essentiality_score": 0.0,
  } for _ in range(4000)]
  summary = summarize_deg_predictions(records)
  assert summary["baselines"]["random"]["pooled_auroc"] == pytest.approx(
    0.5, abs=0.05)
  # A constant model score must be exactly chance, never NaN.
  assert summary["pooled_auroc"] == pytest.approx(0.5)


def test_every_mandatory_baseline_is_reported():
  summary = summarize_deg_predictions(_summary_records())
  assert set(summary["baselines"]) >= {
    "gene_length", "negative_gene_length", "window_gc",
    "distance_from_contig_start", "organism_base_rate", "random"}


def test_baseline_scores_are_deterministic_under_a_seed():
  records = _summary_records()
  assert baseline_scores(records, seed=5) == baseline_scores(records, seed=5)
  assert (baseline_scores(records, seed=5)["random"]
          != baseline_scores(records, seed=6)["random"])


# ---------------------------------------------------------------------------
# Sign convention of the scorer
# ---------------------------------------------------------------------------

def test_score_assembly_makes_a_costlier_knockout_score_higher():
  assert_sign_convention()
  record = {
    "organism_id": "o", "organism_name": "o", "accession": "a",
    "locus_tag": "t", "gene_name": "g", "protein_id": "p",
    "gene_start": 0, "gene_end": 300, "strand": 1, "gene_length": 300,
    "essential": 1, "match_route": "name", "window_start": 0,
    "window_gc": 0.5, "cds_offset": 100, "cassette_start": 112,
  }
  hurt = _assemble(record, 100.0, 130.0, 1.0, 4.0, 5.0, 9.0, 8191, "ar")
  spared = _assemble(record, 100.0, 99.0, 1.0, 0.5, 5.0, 4.0, 8191, "ar")
  assert hurt["essentiality_score"] == pytest.approx(30.0)
  assert spared["essentiality_score"] == pytest.approx(-1.0)
  # And the resulting AUROC is 1.0, not 0.0, when essentials are the hurt ones.
  assert roc_auc([1, 0], [hurt["essentiality_score"],
                          spared["essentiality_score"]]) == 1.0


def test_baselines_only_rows_carry_zero_model_scores_but_real_features():
  contig = _random_contig(40000)
  geometry = build_window_record(contig, 20000, 20600, 1)
  record = {
    "organism_id": "o", "organism_name": "o", "accession": "a",
    "locus_tag": "t", "gene_name": "g", "protein_id": "p",
    "gene_start": 20000, "gene_end": 20600, "strand": 1, "gene_length": 600,
    "essential": 1, "match_route": "sequence",
    "window_start": geometry["window_start"],
    "window_gc": gc_content(geometry["window_sequence"]),
    "cds_offset": geometry["cds_offset"],
    "cassette_start": geometry["cassette_start"],
  }
  rows = baseline_records([record])
  assert rows[0]["essentiality_score"] == 0.0
  assert rows[0]["gene_length"] == 600
  assert 0.0 < rows[0]["window_gc"] < 1.0


def test_gc_content_and_hamming_helpers():
  assert gc_content("GGCC") == 1.0
  assert gc_content("ATAT") == 0.0
  assert gc_content("ACGT") == pytest.approx(0.5)
  assert hamming_positions("ACGT", "ACGA") == 1
  with pytest.raises(ValueError, match="equal lengths"):
    hamming_positions("ACGT", "ACG")


def test_reverse_complement_round_trips_and_handles_ambiguity():
  sequence = _random_contig(500)
  assert reverse_complement(reverse_complement(sequence)) == sequence
  assert reverse_complement("ACGTN") == "NACGT"
  assert reverse_complement("RY") == "RY"
