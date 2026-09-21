"""Reproducible MaveDB preparation helpers for the dnaHNet benchmark.

The dnaHNet paper reports 12 E. coli K-12 nucleotide-level score sets with
21,250 total variants, but does not list their accessions. The adjacent JSON
manifest pins the twelve MaveDB combined-score sets that reproduce that total
exactly. Keeping selection separate from the live search API prevents newly
published records from silently changing the benchmark.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


API_BASE = "https://api.mavedb.org/api/v1"
USER_AGENT = "bd3lms-dnahnet-benchmark/1.0"
DNA_RE = re.compile(r"^[ACGT]+$")
SUB_RE = re.compile(r"^(\d+)([ACGT])>([ACGT])$")
DELINS_RE = re.compile(r"^(\d+)(?:_(\d+))?delins([ACGT]+)$")
DEL_RE = re.compile(r"^(\d+)(?:_(\d+))?del(?:[ACGT]+)?$")
INS_RE = re.compile(r"^(\d+)_(\d+)ins([ACGT]+)$")


def sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
  with path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
  expected = int(manifest["expected_total_variants"])
  observed = sum(
    int(score_set["expected_variants"])
    for score_set in manifest["score_sets"])
  if expected != observed:
    raise ValueError(
      f"Manifest variant total is inconsistent: {expected} != {observed}")
  return manifest


class MaveDBClient:
  """Small read-only client with bounded retries and no extra dependency."""

  def __init__(
      self,
      api_base: str = API_BASE,
      timeout: float = 60.0,
      retries: int = 3,
  ):
    self.api_base = api_base.rstrip("/")
    self.timeout = timeout
    self.retries = retries

  def _get(self, path: str, query: Sequence[tuple[str, str]] = ()) -> bytes:
    url = f"{self.api_base}/{path.lstrip('/')}"
    if query:
      url = f"{url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(self.retries):
      try:
        with urllib.request.urlopen(
            request, timeout=self.timeout) as response:
          return response.read()
      except (urllib.error.URLError, TimeoutError) as error:
        last_error = error
        if attempt + 1 < self.retries:
          time.sleep(2 ** attempt)
    raise RuntimeError(f"MaveDB request failed for {url}") from last_error

  def score_set(self, urn: str) -> dict:
    data = self._get(f"score-sets/{urllib.parse.quote(urn, safe=':')}")
    return json.loads(data)

  def variants(self, urn: str) -> list[dict[str, str]]:
    data = self._get(
      f"score-sets/{urllib.parse.quote(urn, safe=':')}/variants/data",
      (("namespaces", "scores"),))
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def _parse_change(change: str, reference: str):
  """Returns a zero-based half-open replacement operation."""
  match = SUB_RE.fullmatch(change)
  if match:
    position, expected, replacement = match.groups()
    start = int(position) - 1
    if start < 0 or start >= len(reference):
      raise ValueError(f"Substitution outside reference: {change}")
    if reference[start] != expected:
      raise ValueError(
        f"HGVS reference mismatch for {change}: found {reference[start]}")
    return start, start + 1, replacement, change

  match = DELINS_RE.fullmatch(change)
  if match:
    first, last, replacement = match.groups()
    start = int(first) - 1
    end = int(last or first)
    return start, end, replacement, change

  match = DEL_RE.fullmatch(change)
  if match:
    first, last = match.groups()
    start = int(first) - 1
    end = int(last or first)
    return start, end, "", change

  match = INS_RE.fullmatch(change)
  if match:
    left, right, replacement = match.groups()
    left, right = int(left), int(right)
    if right != left + 1:
      raise ValueError(f"Insertion coordinates are not adjacent: {change}")
    return left, left, replacement, change

  raise ValueError(f"Unsupported coding HGVS operation: {change}")


def apply_coding_hgvs(reference: str, hgvs_nt: str) -> str:
  """Applies the substitution/indel subset used by the pinned score sets.

  MaveDB coordinates refer to the unchanged reference, so replacements are
  applied from right to left. This supports all 21,250 records in the pinned
  snapshot: substitutions, deletions, insertions and delins operations.
  """
  reference = reference.upper()
  if not DNA_RE.fullmatch(reference):
    raise ValueError("Reference sequence must contain only A/C/G/T")
  if hgvs_nt == "c.=":
    return reference
  if not hgvs_nt.startswith("c."):
    raise ValueError(f"Expected coding HGVS notation, received {hgvs_nt!r}")

  body = hgvs_nt[2:]
  if body.startswith("[") and body.endswith("]"):
    changes = body[1:-1].split(";")
  else:
    changes = [body]
  operations = [_parse_change(change, reference) for change in changes]

  occupied = []
  for start, end, _, change in operations:
    if start < 0 or end < start or end > len(reference):
      raise ValueError(f"HGVS coordinates outside reference: {change}")
    if start != end:
      for other_start, other_end in occupied:
        if max(start, other_start) < min(end, other_end):
          raise ValueError(f"Overlapping HGVS operations in {hgvs_nt}")
      occupied.append((start, end))

  result = reference
  for start, end, replacement, _ in sorted(
      operations, key=lambda item: (item[0], item[1]), reverse=True):
    result = result[:start] + replacement + result[end:]
  if not DNA_RE.fullmatch(result):
    raise ValueError(f"Mutation produced invalid DNA for {hgvs_nt}")
  return result


def build_records(
    manifest: Mapping,
    client: MaveDBClient,
) -> Iterator[dict]:
  """Downloads, validates and yields canonical benchmark records."""
  total = 0
  for expected in manifest["score_sets"]:
    urn = expected["urn"]
    metadata = client.score_set(urn)
    if metadata["title"] != expected["title"]:
      raise ValueError(f"Title changed for {urn}: {metadata['title']!r}")
    if int(metadata["numVariants"]) != int(expected["expected_variants"]):
      raise ValueError(f"Variant count changed for {urn}")
    genes = metadata["targetGenes"]
    if len(genes) != 1:
      raise ValueError(f"Expected exactly one target gene for {urn}")
    gene = genes[0]
    target = gene.get("targetSequence") or {}
    reference = str(target.get("sequence", "")).upper()
    if str(target.get("sequenceType", "")).lower() != "dna":
      raise ValueError(f"Expected a DNA target sequence for {urn}")
    if sha256_text(reference) != expected["target_sequence_sha256"]:
      raise ValueError(f"Target sequence changed for {urn}")

    rows = client.variants(urn)
    if len(rows) != int(expected["expected_variants"]):
      raise ValueError(f"Downloaded row count changed for {urn}")
    license_info = metadata.get("license") or {}
    for row in rows:
      score_text = row.get("scores.score")
      if score_text is None:
        raise ValueError(f"Missing scores.score in {urn}")
      score = float(score_text)
      if not math.isfinite(score):
        raise ValueError(f"Non-finite experimental score in {urn}")
      hgvs_nt = row["hgvs_nt"]
      yield {
        "score_set_urn": urn,
        "score_set_title": metadata["title"],
        "target": gene["name"],
        "accession": row["accession"],
        "hgvs_nt": hgvs_nt,
        "hgvs_pro": row.get("hgvs_pro"),
        "experimental_score": score,
        "wt_sequence": reference,
        "mut_sequence": apply_coding_hgvs(reference, hgvs_nt),
        "license": license_info.get("shortName"),
        "source_url": f"https://www.mavedb.org/score-sets/{urn}",
      }
      total += 1
  if total != int(manifest["expected_total_variants"]):
    raise ValueError(
      f"Prepared total does not match the paper: {total} != "
      f"{manifest['expected_total_variants']}")


def read_jsonl_gz(path: Path) -> Iterator[dict]:
  with gzip.open(path, "rt", encoding="utf-8") as handle:
    for line in handle:
      if line.strip():
        yield json.loads(line)


def rankdata(values: Sequence[float]) -> np.ndarray:
  """Average ranks with tie handling, equivalent to scipy.stats.rankdata."""
  values = np.asarray(values, dtype=np.float64)
  order = np.argsort(values, kind="mergesort")
  ranks = np.empty(len(values), dtype=np.float64)
  index = 0
  while index < len(values):
    end = index + 1
    while end < len(values) and values[order[end]] == values[order[index]]:
      end += 1
    ranks[order[index:end]] = (index + end - 1) / 2 + 1
    index = end
  return ranks


def spearmanr(left: Sequence[float], right: Sequence[float]) -> float:
  if len(left) != len(right) or len(left) < 2:
    raise ValueError("Spearman inputs must have equal length >= 2")
  left_rank, right_rank = rankdata(left), rankdata(right)
  if left_rank.std() == 0 or right_rank.std() == 0:
    return float("nan")
  return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _hgvs_length_delta(hgvs_nt) -> int:
  """Net nucleotide length change implied by a c. annotation (0 for subs)."""
  if not isinstance(hgvs_nt, str) or hgvs_nt.strip() in {"", "c.=", "n.=", "c.(=)"}:
    return 0
  body = hgvs_nt.strip()
  body = body[body.find("[") + 1:body.rfind("]")] if "[" in body else body[2:]
  delta = 0
  for change in (c.strip() for c in body.split(";") if c.strip()):
    match = DELINS_RE.fullmatch(change)
    if match:
      start = int(match.group(1))
      stop = int(match.group(2)) if match.group(2) else start
      delta += len(match.group(3)) - (stop - start + 1)
      continue
    match = DEL_RE.fullmatch(change)
    if match:
      start = int(match.group(1))
      stop = int(match.group(2)) if match.group(2) else start
      delta -= stop - start + 1
      continue
    match = INS_RE.fullmatch(change)
    if match:
      delta += len(match.group(3))
  return delta


def _substitution_only_metrics(all_records) -> dict:
  """Macro signed Spearman over length-preserving variants only.

  Returns the reference-correct benchmark (ProteinGym scores substitutions and
  indels separately) beside the pooled number, so both are always visible and
  neither can be quoted by accident.
  """
  subs, indels = [], []
  for record in all_records:
    # predictions.csv carries no sequences, so classify from the annotation
    (indels if _hgvs_length_delta(record.get("hgvs_nt")) else subs).append(record)

  def _macro(rows):
    groups: dict[str, list] = {}
    for row in rows:
      groups.setdefault(row["score_set_urn"], []).append(row)
    values = []
    for assay in groups.values():
      pred = [float(r["predicted_fitness"]) for r in assay
              if np.isfinite(float(r["predicted_fitness"]))]
      exp = [float(r["experimental_score"]) for r in assay
             if np.isfinite(float(r["predicted_fitness"]))]
      if len(pred) >= 10:
        values.append(spearmanr(pred, exp))
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")

  return {
    "macro_signed_spearman_substitutions": _macro(subs),
    "macro_signed_spearman_indels": _macro(indels),
    "num_substitution_variants": len(subs),
    "num_indel_variants": len(indels),
  }


def summarize_predictions(records: Iterable[Mapping]) -> dict:
  groups: dict[str, list[Mapping]] = {}
  all_records = list(records)
  for record in all_records:
    groups.setdefault(record["score_set_urn"], []).append(record)
  assays = []
  dropped = 0
  for urn, rows in groups.items():
    # DROP non-finite predictions; do not rank them. `rankdata` assigns NaN the
    # LARGEST rank, so a refused variant silently became a "highest predicted
    # fitness" row. Measured when Score I landed on 2026-09-13: one assay's
    # correlation read -0.22227 with 39 NaN rows ranked, against -0.14113 with
    # them excluded -- a 0.08 distortion from rows the scorer had explicitly
    # declined to score. Score I returns NaN for the 8.95% of pairs that change
    # length, which is the honest refusal; ranking it was not.
    rows = [r for r in rows if math.isfinite(float(r["predicted_fitness"]))]
    dropped += len(groups[urn]) - len(rows)
    if not rows:
      continue
    correlation = spearmanr(
      [float(row["predicted_fitness"]) for row in rows],
      [float(row["experimental_score"]) for row in rows])
    assays.append({
      "score_set_urn": urn,
      "title": rows[0]["score_set_title"],
      "target": rows[0]["target"],
      "n": len(rows),
      "spearman": correlation,
      "abs_spearman": abs(correlation),
    })
  assays.sort(key=lambda row: row["score_set_urn"])
  finite_records = [r for r in all_records
                    if math.isfinite(float(r["predicted_fitness"]))]
  pooled = spearmanr(
    [float(row["predicted_fitness"]) for row in finite_records],
    [float(row["experimental_score"]) for row in finite_records])
  finite_abs = [row["abs_spearman"] for row in assays
                if math.isfinite(row["abs_spearman"])]
  finite_signed = [row["spearman"] for row in assays
                   if math.isfinite(row["spearman"])]
  return {
    "num_assays": len(assays),
    "num_variants": len(all_records),
    # Reported, never silent: a score mode that declines some variants must say
    # how many, or the headline is computed on a different set than it claims.
    "num_unscored_variants": dropped,
    "num_scored_variants": len(all_records) - dropped,
    # SIGNED is the primary metric. The decision to make it primary was recorded
    # on 2026-08-14 -- all 12 assays share one direction, so discarding the sign
    # credits anti-correlation as skill -- but only the prose was updated; this
    # function kept returning `macro_abs_spearman` alone for another month.
    # Taking the absolute value inflates the block-diffusion arms about 1.5x and
    # the autoregressive arms 1.13x, so it NARROWS a real gap. Both are emitted;
    # quote the signed one.
    "macro_signed_spearman": float(np.mean(finite_signed)) if finite_signed
                             else float("nan"),
    "num_negative_assays": sum(1 for r in finite_signed if r < 0),
    # SUBSTITUTION-ONLY re-run of the same metric. ProteinGym keeps
    # substitutions (DMS_substitutions.csv) and indels (DMS_indels.csv) as
    # SEPARATE benchmarks and switches aggregation on an --indel_mode flag; we
    # were pooling them. That matters because our sequence scores are UNNORMALISED
    # SUMS over tokens, so a 3-nt deletion drops three log-probability terms and a
    # 3-nt insertion adds three. Measured on b8 eps=0.9: corr(predicted_fitness,
    # length change) = -0.79 over the 1,901 indel rows, mean predicted +3.015 for
    # deletions against -4.249 for insertions, while the ASSAY puts them at a
    # near-identical +1.319 and +1.699 kcal/mol. Those rows score at chance
    # (macro 0.0149) and drag the headline down ~0.010.
    # `partial_corr.py` was already immune because `len_delta` is one of its
    # controls; only the signed headline was contaminated.
    **_substitution_only_metrics(all_records),
    "macro_abs_spearman": float(np.mean(finite_abs)),
    "pooled_spearman": pooled,
    "pooled_abs_spearman": abs(pooled),
    "assays": assays,
  }


# --------------------------------------------------------------------------
# Stratified reporting: what this benchmark is actually measuring
# --------------------------------------------------------------------------

# BLOSUM62, embedded rather than imported so this module keeps its no-biopython,
# no-scipy property. Upper triangle by row, standard 20 AA order plus '*'.
_B62_ORDER = "ARNDCQEGHILKMFPSTWYV*"
_B62_ROWS = (
  "  4  -1  -2  -2   0  -1  -1   0  -2  -1  -1  -1  -1  -2  -1   1   0  -3  -2   0  -4",
  " -1   5   0  -2  -3   1   0  -2   0  -3  -2   2  -1  -3  -2  -1  -1  -3  -2  -3  -4",
  " -2   0   6   1  -3   0   0   0   1  -3  -3   0  -2  -3  -2   1   0  -4  -2  -3  -4",
  " -2  -2   1   6  -3   0   2  -1  -1  -3  -4  -1  -3  -3  -1   0  -1  -4  -3  -3  -4",
  "  0  -3  -3  -3   9  -3  -4  -3  -3  -1  -1  -3  -1  -2  -3  -1  -1  -2  -2  -1  -4",
  " -1   1   0   0  -3   5   2  -2   0  -3  -2   1   0  -3  -1   0  -1  -2  -1  -2  -4",
  " -1   0   0   2  -4   2   5  -2   0  -3  -3   1  -2  -3  -1   0  -1  -3  -2  -2  -4",
  "  0  -2   0  -1  -3  -2  -2   6  -2  -4  -4  -2  -3  -3  -2   0  -2  -2  -3  -3  -4",
  " -2   0   1  -1  -3   0   0  -2   8  -3  -3  -1  -2  -1  -2  -1  -2  -2   2  -3  -4",
  " -1  -3  -3  -3  -1  -3  -3  -4  -3   4   2  -3   1   0  -3  -2  -1  -3  -1   3  -4",
  " -1  -2  -3  -4  -1  -2  -3  -4  -3   2   4  -2   2   0  -3  -2  -1  -2  -1   1  -4",
  " -1   2   0  -1  -3   1   1  -2  -1  -3  -2   5  -1  -3  -1   0  -1  -3  -2  -2  -4",
  " -1  -1  -2  -3  -1   0  -2  -3  -2   1   2  -1   5   0  -2  -1  -1  -1  -1   1  -4",
  " -2  -3  -3  -3  -2  -3  -3  -3  -1   0   0  -3   0   6  -4  -2  -2   1   3  -1  -4",
  " -1  -2  -2  -1  -3  -1  -1  -2  -2  -3  -3  -1  -2  -4   7  -1  -1  -4  -3  -2  -4",
  "  1  -1   1   0  -1   0   0   0  -1  -2  -2   0  -1  -2  -1   4   1  -3  -2  -2  -4",
  "  0  -1   0  -1  -1  -1  -1  -2  -2  -1  -1  -1  -1  -2  -1   1   5  -2  -2   0  -4",
  " -3  -3  -4  -4  -2  -2  -3  -2  -2  -3  -2  -3  -1   1  -4  -3  -2  11   2  -3  -4",
  " -2  -2  -2  -3  -2  -1  -2  -3   2  -1  -1  -2  -1   3  -3  -2  -2   2   7  -1  -4",
  "  0  -3  -3  -3  -1  -2  -2  -3  -3   3   1  -2   1  -1  -2  -2   0  -3  -1   4  -4",
  " -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4  -4   1",
)


def _b62():
  table = {}
  for i, row in enumerate(_B62_ROWS):
    vals = [int(v) for v in row.split()]
    for j, v in enumerate(vals):
      table[(_B62_ORDER[i], _B62_ORDER[j])] = v
  return table


BLOSUM62 = _b62()

_THREE_TO_ONE = {
  "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
  "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
  "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
  "Tyr": "Y", "Val": "V", "Ter": "*",
}
_SUB = re.compile(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


def protein_substitutions(hgvs_pro):
  """[(wt_aa, mut_aa), ...] for simple substitutions; None if unparseable."""
  text = (hgvs_pro or "").strip()
  if text in {"", "NA", "p.=", "p.(=)"}:
    return []
  body = text[2:] if text.startswith("p.") else text
  out = []
  for event in body.strip("[]").split(";"):
    event = event.strip()
    if not event:
      continue
    match = _SUB.match(event)
    if match is None:
      return None
    wt, _, mut = match.groups()
    if wt not in _THREE_TO_ONE or mut not in _THREE_TO_ONE:
      return None
    out.append((_THREE_TO_ONE[wt], _THREE_TO_ONE[mut]))
  return out


def stratified_summary(records):
  """Per-stratum Spearman plus the BLOSUM62 baseline. Reported on every run.

  WHY THIS IS MANDATORY OUTPUT. The 21,250 variants carry exactly 0, 1 or 2
  non-synonymous changes, and the mean fitness of the 1-change group sits well
  above the 2-change group -- so the pooled macro Spearman is substantially a
  1-vs-2 mutation detector. Measured on this data: restricting to the nsyn == 1
  stratum (13,838 variants) costs uSSM-AR 14% of its score but costs every
  block-diffusion arm 53-59% of theirs, and in that stratum a zero-parameter
  BLOSUM62 lookup reaches +0.2384 -- positive on 12 of 12 assays, above
  uSSM-AR's 0.2279 and 4.5x above the best BD estimator's 0.0506.

  That stratum is where the benchmark genuinely asks "what does one amino-acid
  substitution do", so it is the number that says whether a model understands
  coding sequence. The pooled figure is kept for comparability with dnaHNet's
  published 0.3266, but it should never be quoted without this beside it.
  """
  usable, unparsed = [], 0
  for record in records:
    subs = protein_substitutions(record.get("hgvs_pro"))
    if subs is None:
      unparsed += 1
      continue
    usable.append((record, subs))

  def macro(rows, score_of):
    groups = {}
    for record, subs in rows:
      value = score_of(record, subs)
      if value is None or not math.isfinite(value):
        continue
      groups.setdefault(record["score_set_urn"], []).append(
        (value, float(record["experimental_score"])))
    per = [spearmanr([a for a, _ in v], [b for _, b in v])
           for v in groups.values() if len(v) >= 10]
    per = [r for r in per if math.isfinite(r)]
    if not per:
      return float("nan"), 0, 0
    return (float(np.mean(per)), len(per), sum(1 for r in per if r > 0))

  out = {"unparseable_hgvs_pro": unparsed}
  for name, keep in (("nsyn_eq_1", lambda n: n == 1),
                     ("nsyn_ge_2", lambda n: n >= 2),
                     ("all", lambda n: True)):
    rows = [(r, s) for r, s in usable if keep(len(s))]
    value, n_assays, n_pos = macro(
      rows, lambda r, s: float(r["predicted_fitness"]))
    out[name] = {"macro_signed_spearman": value, "n_variants": len(rows),
                 "n_assays": n_assays, "n_assays_positive": n_pos}
  single = [(r, s) for r, s in usable if len(s) == 1]
  value, n_assays, n_pos = macro(
    single, lambda r, s: float(BLOSUM62.get(s[0], float("nan"))))
  out["blosum62_baseline_nsyn_eq_1"] = {
    "macro_signed_spearman": value, "n_variants": len(single),
    "n_assays": n_assays, "n_assays_positive": n_pos,
    "note": "zero parameters, no DNA; the bar the nsyn==1 stratum must clear",
  }
  return out


def protein_event_baseline(records):
  """Zero-parameter baseline: count amino-acid events in `hgvs_pro`.

  THIS BEATS EVERY MODEL IN THE REPO and is the reason the benchmark's headline
  number must be read carefully. Macro signed Spearman +0.30931 against fitness,
  pointing the right way on 12 of 12 assays, versus uSSM-AR's +0.26402 (and
  dnaHNet's published 0.3266). It takes only three distinct values across all
  21,250 variants -- {0: 66, 1: 15739, 2: 5445} -- so it says little more than
  "wild type, single mutant, or double mutant".

  `partial_corr.py`'s docstring has claimed since 2026-08-17 that this baseline
  is "computed here"; it was not. What that script actually computed was the
  NUCLEOTIDE edit count, a much weaker +0.159, and it then used that as the
  control -- so the partial correlations were controlling for the wrong
  confound. Controlling for this one as well costs uSSM-AR only 5% of its
  partial rho (+0.21644 -> +0.20548), which is the evidence that its signal is
  not merely a mutation count.

  Returns the score convention "higher = fitter", i.e. the NEGATED count, so the
  correlation is positive like a model's.
  """
  scores = []
  for record in records:
    protein = (record.get("hgvs_pro") or "").strip()
    if protein in {"", "NA", "p.=", "p.(=)"}:
      count = 0
    else:
      body = protein[2:] if protein.startswith("p.") else protein
      count = len([e for e in body.strip("[]").split(";") if e.strip()])
    row = dict(record)
    row["predicted_fitness"] = float(-count)
    scores.append(row)
  return summarize_predictions(scores)
