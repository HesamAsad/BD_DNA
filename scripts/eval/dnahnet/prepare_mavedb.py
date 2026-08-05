#!/usr/bin/env python3
"""Download the pinned 21,250-example dnaHNet MaveDB benchmark snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.mavedb import (  # noqa: E402
  MaveDBClient,
  build_records,
  load_manifest,
)


def _atomic_json(path: Path, value: dict):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def write_records(path: Path, records) -> tuple[int, str]:
  """Writes deterministic gzip JSONL and returns count plus SHA-256."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as raw:
    temporary = Path(raw.name)
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
      with io.TextIOWrapper(compressed, encoding="utf-8") as text:
        count = 0
        for record in records:
          text.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
          text.write("\n")
          count += 1
  digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
  os.replace(temporary, path)
  return count, digest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--manifest", type=Path, default=HERE / "mavedb_manifest.json")
  parser.add_argument(
    "--output", type=Path,
    default=REPO / "data_cache/dnahnet/mavedb_ecoli_k12_21250.jsonl.gz")
  parser.add_argument("--timeout", type=float, default=60.0)
  args = parser.parse_args()

  manifest = load_manifest(args.manifest)
  client = MaveDBClient(timeout=args.timeout)
  count, digest = write_records(
    args.output, build_records(manifest, client))
  metadata = {
    "benchmark": manifest["benchmark"],
    "paper": manifest["paper"],
    "manifest": str(args.manifest.resolve()),
    "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    "output": str(args.output.resolve()),
    "output_sha256": digest,
    "num_score_sets": len(manifest["score_sets"]),
    "num_variants": count,
    "downloaded_at": datetime.now(timezone.utc).isoformat(),
    "source_api": manifest["source_api"],
  }
  metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
  _atomic_json(metadata_path, metadata)
  print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
