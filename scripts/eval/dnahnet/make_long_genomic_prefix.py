#!/usr/bin/env python
"""Build longer real E. coli upstream prefixes for MaveDB scoring.

Scoring runs at model_length=256 and the MaveDB sequences are only 198-207 nt,
so a block-256 checkpoint sees the whole scored sequence as ONE block with an
empty cache, while a block-8 checkpoint sees 32 blocks mostly conditioned on a
populated one. That difference is a candidate explanation for the entire
block-size effect on this benchmark, and it is testable at inference with no
retraining: give the block-256 model a LONGER prefix so it too has many blocks.

`build_pair_tensors` is generic in prefix length -- the prefix is loss-masked
and the variant always lands in the final block -- so a prefix of 768 or 1792
nt at model_length 1024 or 2048 yields 4 or 8 blocks at block_size 256.

Each extended prefix KEEPS the original 256-nt prefix as its suffix, so the
variant's immediate context is byte-identical to the existing runs and only
more distant blocks are added.

Usage:
  python scripts/eval/dnahnet/make_long_genomic_prefix.py 768 1792
"""
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GENBANK = ROOT / "data_cache/dnahnet/deg_raw/ncbi/NC_000913.gb.gz"
BASE = ROOT / "data_cache/dnahnet/mavedb_genomic_prefix.json"


def genome() -> str:
  chunks, in_origin = [], False
  with gzip.open(GENBANK, "rt") as handle:
    for line in handle:
      if line.startswith("ORIGIN"):
        in_origin = True
        continue
      if in_origin:
        if line.startswith("//"):
          break
        chunks.append(re.sub(r"[^acgtnACGTN]", "", line))
  return "".join(chunks).upper()


def revcomp(seq: str) -> str:
  return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))


def build(total: int, ref: str, base: dict) -> dict:
  out = {}
  size = len(ref)
  for urn, prefix in base.items():
    start, strand = ref.find(prefix), "+"
    if start < 0:
      start, strand = ref.find(revcomp(prefix)), "-"
    if start < 0:
      raise SystemExit(f"{urn}: prefix not found in NC_000913 on either strand")
    extra = total - len(prefix)
    if strand == "+":
      begin = (start - extra) % size
      upstream = ref[begin:start] if begin < start else ref[begin:] + ref[:start]
      extended = upstream + prefix
    else:
      end = start + len(prefix)          # 5' of a minus-strand gene is downstream
      upstream = (ref[end:end + extra] if end + extra <= size
                  else ref[end:] + ref[:(end + extra) % size])
      extended = revcomp(upstream) + prefix
    if len(extended) != total or not extended.endswith(prefix):
      raise SystemExit(f"{urn}: built {len(extended)} nt, or lost the original suffix")
    if set(extended) - set("ACGT"):
      raise SystemExit(f"{urn}: non-ACGT in prefix: {set(extended) - set('ACGT')}")
    out[urn] = extended
  return out


def main(argv):
  lengths = [int(a) for a in argv[1:]] or [768, 1792]
  ref, base = genome(), json.loads(BASE.read_text())
  for total in lengths:
    out = build(total, ref, base)
    path = BASE.with_name(f"mavedb_genomic_prefix_{total}.json")
    path.write_text(json.dumps(out))
    gc = (sum(s.count("G") + s.count("C") for s in out.values())
          / sum(len(s) for s in out.values()))
    print(f"{path.name}: {len(out)} entries, {total} nt, GC {gc:.3f} "
          f"(NC_000913 is 0.508)")


if __name__ == "__main__":
  main(sys.argv)
