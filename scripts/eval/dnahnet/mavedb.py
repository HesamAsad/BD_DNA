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


def summarize_predictions(records: Iterable[Mapping]) -> dict:
  groups: dict[str, list[Mapping]] = {}
  all_records = list(records)
  for record in all_records:
    groups.setdefault(record["score_set_urn"], []).append(record)
  assays = []
  for urn, rows in groups.items():
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
  pooled = spearmanr(
    [float(row["predicted_fitness"]) for row in all_records],
    [float(row["experimental_score"]) for row in all_records])
  finite_abs = [row["abs_spearman"] for row in assays
                if math.isfinite(row["abs_spearman"])]
  return {
    "num_assays": len(assays),
    "num_variants": len(all_records),
    "macro_abs_spearman": float(np.mean(finite_abs)),
    "pooled_spearman": pooled,
    "pooled_abs_spearman": abs(pooled),
    "assays": assays,
  }
