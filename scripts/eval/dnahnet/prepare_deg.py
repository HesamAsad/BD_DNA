#!/usr/bin/env python3
"""Assemble the pinned dnaHNet DEG gene-essentiality benchmark snapshot.

Mirrors ``prepare_mavedb.py``: a pinned manifest selects the sources, the
downloads are content-hashed, every record is validated, and the output is a
deterministic ``jsonl.gz`` plus a metadata JSON carrying per-organism counts
and hashes so the benchmark cannot drift.

Two modes:

  ``--emit-manifest``  bootstrap the organism selection from the DEG bacteria
                       table (writes ``deg_manifest.json``; run once, commit).
  (default)            build ``deg_essentiality.jsonl.gz`` from that manifest.

Everything here is CPU work and fits the head node's ~1.7 GB per-process
cgroup: downloads stream to disk and only one replicon is parsed at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.deg import (  # noqa: E402
  DEG_BACTERIA_FILES,
  DEG_DOWNLOAD_BASE,
  IDENTITY_THRESHOLD,
  KNOCKOUT_OFFSET,
  NCBI_EUTILS,
  STOP_CASSETTE,
  WINDOW_LENGTH,
  EssentialGeneMatcher,
  NCBIClient,
  apply_knockout,
  build_window_record,
  feature_name_keys,
  gc_content,
  hamming_positions,
  load_deg_annotations,
  load_deg_nucleotides,
  load_deg_organisms,
  sha256_file,
  write_jsonl_gz,
)

DEFAULT_MANIFEST = HERE / "deg_manifest.json"
DEFAULT_RAW = REPO / "data_cache/dnahnet/deg_raw"
DEFAULT_OUTPUT = REPO / "data_cache/dnahnet/deg_essentiality.jsonl.gz"

PAPER = "Ma et al. (2026), dnaHNet, zero-shot gene essentiality prediction"
PAPER_REPORTED = {
  "organisms": 62,
  "data_points": 185226,
  "window_bp": 8192,
  "stop_cassette": STOP_CASSETTE,
  "knockout_offset_bp_downstream_of_start_codon": 12,
  "metric": "AUROC",
}

ASSUMPTIONS = [
  "A1 organism count: DEG's bacteria table currently lists 66 essentiality "
  "experiments spanning 61 distinct organism names, 63 distinct nucleotide "
  "accessions and 55 distinct accession sets. None of those equals the "
  "paper's 62, and the paper does not publish its organism list, so the "
  "count is not exactly reproducible. This manifest defines one benchmark "
  "organism per distinct DEG accession set and unions the essential-gene "
  "labels of every DEG experiment sharing that genome.",
  "A2 sequence identity: the paper's '>99% sequence identity' was almost "
  "certainly a gapped BLAST search. No BLAST binary exists on this cluster, "
  "so matching is exact-or-ungapped (identical, or equal-length with <1% "
  "mismatching positions) on both strands, unioned with gene-name matching.",
  "A3 knockout offset: '12 bp downstream of the start codon' is read as a "
  "0-based offset from the first base of the start codon. The competing "
  "reading (12 bp after the start codon ends) gives 15. Both keep the "
  "cassette in frame; --knockout-offset switches between them.",
  "A4 strand: windows are emitted in gene orientation (reverse-complemented "
  "for minus-strand genes) so the cassette is literally TAATAATAATAGTGA at a "
  "fixed offset for every gene. --orientation reference keeps the plus "
  "strand and writes the reverse complement instead.",
  "A5 contig edges: bacterial replicons annotated 'circular' wrap around the "
  "origin, which preserves both the 8192 nt length and the centring. Linear "
  "contigs clamp the window inward, preserving length but not centring. "
  "Contigs shorter than 8192 nt are excluded and counted.",
  "A6 long genes: a gene longer than the window pushes the knockout site "
  "outside the centred window, which would make wild-type and knockout "
  "byte-identical. Those genes are excluded and counted.",
  "A7 gene set: only CDS features with a single-interval location are used. "
  "Origin-spanning and programmed-frameshift joins are excluded and counted.",
  "A8 aggregation: the paper reports one AUROC and does not say whether it "
  "is pooled or macro-averaged. Both are reported; the macro mean of "
  "per-organism AUROC is the headline because a pooled AUROC over organisms "
  "with different base rates can be driven by between-organism differences.",
]


def _atomic_json(path: Path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def download_deg_sources(raw_dir: Path, timeout: float = 300.0) -> dict:
  """Fetches the three DEG bacteria archives, returning name -> sha256/bytes."""
  import urllib.request

  raw_dir.mkdir(parents=True, exist_ok=True)
  digests = {}
  for name in DEG_BACTERIA_FILES:
    target = raw_dir / name
    if not target.exists():
      url = f"{DEG_DOWNLOAD_BASE}/{name}"
      print(f"[deg] downloading {url}", flush=True)
      request = urllib.request.Request(
        url, headers={"User-Agent": "bd3lms-dnahnet-benchmark/1.0"})
      temporary = target.with_suffix(target.suffix + ".partial")
      with urllib.request.urlopen(request, timeout=timeout) as response:
        with temporary.open("wb") as sink:
          while True:
            chunk = response.read(1 << 20)
            if not chunk:
              break
            sink.write(chunk)
      temporary.replace(target)
    digests[name] = {
      "sha256": sha256_file(target), "bytes": target.stat().st_size}
  return digests


def emit_manifest(raw_dir: Path, manifest_path: Path, timeout: float):
  """Bootstraps the pinned organism selection from the DEG bacteria table."""
  digests = download_deg_sources(raw_dir, timeout)
  organisms = load_deg_organisms(raw_dir / "deg_bacteria.csv.zip")
  annotations = load_deg_annotations(raw_dir / "deg_annotation_p.csv.zip")
  per_experiment = defaultdict(int)
  for row in annotations:
    per_experiment[row["deg_id"]] += 1

  grouped: dict[tuple[str, ...], dict] = {}
  skipped = []
  for entry in organisms:
    accessions = entry["accessions"]
    if not accessions:
      skipped.append({
        "deg_id": entry["deg_id"], "organism": entry["organism"],
        "reason": "DEG lists no nucleotide accession"})
      continue
    group = grouped.setdefault(accessions, {
      "organism_id": "+".join(accessions),
      "name": entry["organism"],
      "accessions": list(accessions),
      "deg_ids": [],
      "deg_essential_entries": 0,
    })
    group["deg_ids"].append(entry["deg_id"])
    group["deg_essential_entries"] += per_experiment.get(entry["deg_id"], 0)

  manifest = {
    "benchmark": "dnaHNet DEG bacterial gene-essentiality (zero-shot AUROC)",
    "paper": PAPER,
    "paper_reported": PAPER_REPORTED,
    "protocol": {
      "window_length": WINDOW_LENGTH,
      "knockout_offset": KNOCKOUT_OFFSET,
      "stop_cassette": STOP_CASSETTE,
      "identity_threshold": IDENTITY_THRESHOLD,
      "orientation": "gene",
    },
    "sources": {
      "deg_download_base": DEG_DOWNLOAD_BASE,
      "deg_files": digests,
      "ncbi_eutils": NCBI_EUTILS,
      "ncbi_rettype": "gbwithparts",
    },
    "organism_selection_rule": (
      "One benchmark organism per distinct DEG nucleotide-accession set "
      "(version-stripped, sorted). DEG experiments sharing a genome are "
      "merged and their essential-gene labels unioned."),
    "assumptions": ASSUMPTIONS,
    "skipped_deg_experiments": skipped,
    "num_deg_experiments": len(organisms),
    "num_organisms": len(grouped),
    "organisms": sorted(grouped.values(), key=lambda row: row["organism_id"]),
  }
  _atomic_json(manifest_path, manifest)
  print(json.dumps({
    "manifest": str(manifest_path),
    "num_deg_experiments": len(organisms),
    "num_organisms": len(grouped),
    "skipped": len(skipped),
  }, indent=2))


def load_manifest(path: Path) -> dict:
  with path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
  seen = {row["organism_id"] for row in manifest["organisms"]}
  if len(seen) != len(manifest["organisms"]):
    raise ValueError("Duplicate organism_id in the DEG manifest")
  if manifest["num_organisms"] != len(manifest["organisms"]):
    raise ValueError("DEG manifest organism count is inconsistent")
  return manifest


def verify_deg_sources(raw_dir: Path, manifest: dict, timeout: float):
  digests = download_deg_sources(raw_dir, timeout)
  pinned = manifest["sources"]["deg_files"]
  for name, expected in pinned.items():
    observed = digests[name]
    if observed["sha256"] != expected["sha256"]:
      raise ValueError(
        f"{name} changed upstream: {observed['sha256']} != "
        f"{expected['sha256']}. DEG published a new release; re-pin the "
        "manifest deliberately rather than letting the benchmark drift.")


def parse_replicon(path: Path):
  """Yields ``(record_id, topology, sequence, cds_features)`` for one file."""
  import gzip

  from Bio import SeqIO

  with gzip.open(path, "rt") as handle:
    for record in SeqIO.parse(handle, "genbank"):
      topology = str(record.annotations.get("topology", "linear")).lower()
      cds = [f for f in record.features if f.type == "CDS"]
      yield record.id, topology, record, cds


def build_organism_records(
    organism: dict,
    raw_dir: Path,
    client: NCBIClient,
    deg_names,
    deg_sequences,
    window_length: int,
    knockout_offset: int,
    orientation: str,
    identity_threshold: float,
    max_genes_per_organism: int | None,
):
  """Yields validated benchmark records for one organism plus its statistics."""
  matcher = EssentialGeneMatcher(
    deg_names, deg_sequences, identity_threshold=identity_threshold)
  stats = {
    "organism_id": organism["organism_id"],
    "name": organism["name"],
    "deg_ids": organism["deg_ids"],
    "accessions": [],
    "num_genes": 0,
    "num_essential": 0,
    "match_routes": defaultdict(int),
    "excluded": defaultdict(int),
    "deg_essential_entries": organism.get("deg_essential_entries", 0),
  }
  emitted = 0

  for accession in organism["accessions"]:
    destination = raw_dir / "ncbi" / f"{accession}.gb.gz"
    if not destination.exists():
      print(f"[deg]   fetching {accession} from NCBI", flush=True)
      client.fetch_genbank(accession, destination)
    accession_sha = sha256_file(destination)

    for record_id, topology, record, cds_features in parse_replicon(
        destination):
      contig = str(record.seq).upper()
      stats["accessions"].append({
        "requested": accession,
        "version": record_id,
        "length": len(contig),
        "topology": topology,
        "num_cds": len(cds_features),
        "sha256": accession_sha,
        "source_file": str(destination.resolve()),
      })
      circular = topology == "circular"
      if len(contig) < window_length:
        stats["excluded"]["contig_shorter_than_window"] += len(cds_features)
        continue

      for feature in cds_features:
        if len(feature.location.parts) != 1:
          stats["excluded"]["compound_location"] += 1
          continue
        strand = feature.location.strand
        if strand not in (1, -1):
          stats["excluded"]["unknown_strand"] += 1
          continue
        gene_start = int(feature.location.start)
        gene_end = int(feature.location.end)
        gene_length = gene_end - gene_start
        if gene_length < knockout_offset + len(STOP_CASSETTE):
          stats["excluded"]["gene_shorter_than_cassette_site"] += 1
          continue
        try:
          geometry = build_window_record(
            contig, gene_start, gene_end, strand,
            window_length=window_length,
            knockout_offset=knockout_offset,
            circular=circular,
            orientation=orientation)
        except ValueError:
          stats["excluded"]["knockout_site_outside_window"] += 1
          continue

        cds_sequence = str(feature.extract(record.seq)).upper()
        route = matcher.match(
          feature_name_keys(feature.qualifiers), cds_sequence)
        essential = int(route != "none")
        stats["match_routes"][route] += 1

        window = geometry["window_sequence"]
        yield {
          "organism_id": organism["organism_id"],
          "organism_name": organism["name"],
          "deg_ids": ",".join(organism["deg_ids"]),
          "accession": record_id,
          "contig_length": len(contig),
          "topology": topology,
          "locus_tag": (feature.qualifiers.get("locus_tag") or [""])[0],
          "gene_name": (feature.qualifiers.get("gene") or [""])[0],
          "protein_id": (feature.qualifiers.get("protein_id") or [""])[0],
          "product": (feature.qualifiers.get("product") or [""])[0][:200],
          "gene_start": gene_start,
          "gene_end": gene_end,
          "strand": int(strand),
          "gene_length": gene_length,
          "essential": essential,
          "match_route": route,
          "window_start": geometry["window_start"],
          "window_wrapped": geometry["window_wrapped"],
          "window_clamped": geometry["window_clamped"],
          "orientation": geometry["orientation"],
          "cds_offset": geometry["cds_offset"],
          "cassette_start": geometry["cassette_start"],
          "cassette": geometry["cassette"],
          "window_gc": round(gc_content(window), 6),
          "window_n_count": window.count("N"),
          "window_sequence": window,
        }
        stats["num_genes"] += 1
        stats["num_essential"] += essential
        emitted += 1
        if (max_genes_per_organism is not None
            and emitted >= max_genes_per_organism):
          return stats
  return stats


def validate_record(record: dict, window_length: int, cassette: str):
  """Every invariant the task specification asks to be checked, per record."""
  window = record["window_sequence"]
  if len(window) != window_length:
    raise ValueError(
      f"{record['locus_tag']}: window is {len(window)} nt, expected "
      f"{window_length}")
  knockout = apply_knockout(
    window, record["cassette_start"], record["cassette"])
  if len(knockout) != window_length:
    raise ValueError(f"{record['locus_tag']}: knockout changed the length")
  start = record["cassette_start"]
  differences = hamming_positions(window, knockout)
  if differences > len(cassette):
    raise ValueError(
      f"{record['locus_tag']}: knockout differs at {differences} positions, "
      f"expected at most {len(cassette)}")
  if differences == 0:
    raise ValueError(
      f"{record['locus_tag']}: knockout is identical to the wild type")
  # The substitution must be strictly confined to the cassette footprint. The
  # count can be below 15 because an AT-rich window sometimes already carries
  # some of the cassette's bases; it can never be above 15, and never outside.
  if (window[:start] != knockout[:start]
      or window[start + len(cassette):] != knockout[start + len(cassette):]):
    raise ValueError(
      f"{record['locus_tag']}: knockout differs outside the cassette")
  if knockout[start:start + len(cassette)] != record["cassette"]:
    raise ValueError(
      f"{record['locus_tag']}: cassette not present at the intended offset")
  if record["orientation"] == "gene" and record["cassette"] != cassette:
    raise ValueError(
      f"{record['locus_tag']}: gene-oriented cassette is not {cassette}")
  if record["essential"] not in (0, 1):
    raise ValueError(f"{record['locus_tag']}: label is not binary")
  return differences


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
  parser.add_argument("--knockout-offset", type=int, default=KNOCKOUT_OFFSET)
  parser.add_argument(
    "--orientation", choices=("gene", "reference"), default="gene")
  parser.add_argument(
    "--identity-threshold", type=float, default=IDENTITY_THRESHOLD)
  parser.add_argument(
    "--organism", action="append",
    help="Restrict to these organism_id values (repeatable). Use for the "
         "single-organism validation pass before scaling to all of them.")
  parser.add_argument(
    "--max-genes-per-organism", type=int,
    help="Cap emitted genes per organism (smoke subsets only)")
  parser.add_argument("--timeout", type=float, default=300.0)
  parser.add_argument(
    "--strict", action="store_true",
    help="Abort on the first organism that fails instead of recording it in "
         "the metadata's failed_organisms list and continuing")
  parser.add_argument(
    "--emit-manifest", action="store_true",
    help="Bootstrap deg_manifest.json from the DEG bacteria table and exit")
  args = parser.parse_args()

  if args.emit_manifest:
    emit_manifest(args.raw_dir, args.manifest, args.timeout)
    return

  if args.window_length % 2 or args.window_length <= 0:
    parser.error("window-length must be a positive even number")
  if args.knockout_offset < 0:
    parser.error("knockout-offset must be non-negative")

  manifest = load_manifest(args.manifest)
  verify_deg_sources(args.raw_dir, manifest, args.timeout)

  annotations = load_deg_annotations(args.raw_dir / "deg_annotation_p.csv.zip")
  nucleotides = load_deg_nucleotides(args.raw_dir / "DEG10.nt.gz")
  by_experiment = defaultdict(list)
  for row in annotations:
    by_experiment[row["deg_id"]].append(row)

  organisms = manifest["organisms"]
  if args.organism:
    wanted = set(args.organism)
    organisms = [row for row in organisms if row["organism_id"] in wanted]
    missing = wanted - {row["organism_id"] for row in organisms}
    if missing:
      parser.error(f"Unknown organism_id(s): {sorted(missing)}")
  if not organisms:
    parser.error("No organisms selected")

  client = NCBIClient(
    timeout=args.timeout, api_key=os.environ.get("NCBI_API_KEY"))
  per_organism: list[dict] = []
  failures: list[dict] = []
  totals = {"validated": 0, "difference_histogram": defaultdict(int)}

  def record_stream():
    for organism in organisms:
      print(
        f"[deg] {organism['organism_id']} ({organism['name']})", flush=True)
      deg_rows = [row for deg_id in organism["deg_ids"]
                  for row in by_experiment.get(deg_id, ())]
      generator = build_organism_records(
        organism=organism,
        raw_dir=args.raw_dir,
        client=client,
        deg_names=[row["gene_name"] for row in deg_rows],
        deg_sequences=[
          nucleotides.get(row["deg_accession"], "") for row in deg_rows],
        window_length=args.window_length,
        knockout_offset=args.knockout_offset,
        orientation=args.orientation,
        identity_threshold=args.identity_threshold,
        max_genes_per_organism=args.max_genes_per_organism)
      # Buffered per organism (~35 MB for a 4,000-gene genome) so that a
      # failure part-way through a genome emits nothing at all for it rather
      # than half of it. A failure is printed here and recorded in the
      # metadata under "failed_organisms"; it is never silently dropped.
      buffered, histogram, stats = [], defaultdict(int), None
      try:
        while True:
          try:
            record = next(generator)
          except StopIteration as stop:
            stats = stop.value
            break
          histogram[
            validate_record(record, args.window_length, STOP_CASSETTE)] += 1
          buffered.append(record)
      except Exception as error:  # noqa: BLE001 - recorded, never swallowed
        if args.strict:
          raise
        generator.close()
        print(
          f"[deg]   FAILED {organism['organism_id']}: "
          f"{type(error).__name__}: {error}", flush=True)
        failures.append({
          "organism_id": organism["organism_id"],
          "name": organism["name"],
          "accessions": organism["accessions"],
          "error": f"{type(error).__name__}: {error}",
        })
        continue

      for key, value in histogram.items():
        totals["difference_histogram"][key] += value
      totals["validated"] += len(buffered)
      stats["match_routes"] = dict(stats["match_routes"])
      stats["excluded"] = dict(stats["excluded"])
      stats["base_rate"] = (
        stats["num_essential"] / stats["num_genes"]
        if stats["num_genes"] else float("nan"))
      per_organism.append(stats)
      print(
        f"[deg]   genes={stats['num_genes']} "
        f"essential={stats['num_essential']} "
        f"base_rate={stats['base_rate']:.4f} "
        f"excluded={stats['excluded']}", flush=True)
      yield from buffered

  count, digest = write_jsonl_gz(args.output, record_stream())
  if count != totals["validated"]:
    raise AssertionError("Validated and written record counts disagree")
  if not count:
    raise ValueError("No records were produced")
  histogram = {
    str(key): totals["difference_histogram"][key]
    for key in sorted(totals["difference_histogram"])}
  if max(totals["difference_histogram"]) > len(STOP_CASSETTE):
    raise AssertionError("A knockout differed at too many positions")
  if min(totals["difference_histogram"]) < 1:
    raise AssertionError("A knockout was identical to its wild type")

  total_essential = sum(row["num_essential"] for row in per_organism)
  metadata = {
    "benchmark": manifest["benchmark"],
    "paper": manifest["paper"],
    "paper_reported": manifest["paper_reported"],
    "assumptions": manifest["assumptions"],
    "manifest": str(args.manifest.resolve()),
    "manifest_sha256": hashlib.sha256(
      args.manifest.read_bytes()).hexdigest(),
    "output": str(args.output.resolve()),
    "output_sha256": digest,
    "protocol": {
      "window_length": args.window_length,
      "knockout_offset": args.knockout_offset,
      "stop_cassette": STOP_CASSETTE,
      "identity_threshold": args.identity_threshold,
      "orientation": args.orientation,
    },
    "num_organisms": len(per_organism),
    "num_organisms_requested": len(organisms),
    "failed_organisms": failures,
    "num_genes": count,
    "num_essential": total_essential,
    "base_rate": total_essential / count,
    "knockout_difference_histogram": histogram,
    "prepared_at": datetime.now(timezone.utc).isoformat(),
    "per_organism": per_organism,
  }
  metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
  _atomic_json(metadata_path, metadata)
  print(json.dumps({
    key: metadata[key] for key in (
      "output", "output_sha256", "num_organisms", "num_genes",
      "num_essential", "base_rate", "knockout_difference_histogram",
      "num_organisms_requested", "failed_organisms")
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
