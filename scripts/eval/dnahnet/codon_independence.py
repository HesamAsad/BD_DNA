"""Measure the error in Score I's independent-per-position codon factorisation.

`score_infill(unit="codon")` takes a codon's probability as the PRODUCT of its
three positional marginals, all read from ONE forward pass with all three
positions masked:

    p_hat(c) = p(b1 | ctx) * p(b2 | ctx) * p(b3 | ctx)

That is the standard masked-marginal factorisation, but the three bases of a
codon are strongly dependent -- the genetic code is exactly a constraint over
triplets -- so the product can place mass on triplets the model would never
predict jointly. The exact joint is available by the chain rule at the cost of
three passes instead of one:

    p(c) = p(b1 | ctx) * p(b2 | ctx, b1) * p(b3 | ctx, b1, b2)

This script measures the gap on real variants and, more importantly, whether
the gap CHANGES THE SCORE -- an approximation that is numerically loose but
rank-preserving would be harmless for Spearman.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.eval.dnahnet.score_mavedb import (          # noqa: E402
    load_checkpoint_model, build_pair_tensors, GENETIC_CODE,
    variant_positions, _infill_logprobs)
from scripts.eval.dnahnet.mavedb import read_jsonl_gz     # noqa: E402


def codon_logprob_product(model, ids, start):
    """log p_hat(codon) for all 64 codons -- product of marginals, ONE pass."""
    span = torch.tensor([start, start + 1, start + 2], device=ids.device)
    rows = _infill_logprobs(model, ids, span).double()       # [3, vocab]
    out = {}
    for t in GENETIC_CODE:
        i = [model.tokenizer.convert_tokens_to_ids(b) for b in t]
        out[t] = float(rows[0, i[0]] + rows[1, i[1]] + rows[2, i[2]])
    return out


def codon_logprob_chain(model, ids, start):
    """log p(codon) by the chain rule within the codon -- EXACT, 3 passes/prefix."""
    out = {}
    # position 1: all three masked
    span3 = torch.tensor([start, start + 1, start + 2], device=ids.device)
    r1 = _infill_logprobs(model, ids, span3).double()[0]
    for b1 in "TCAG":
        i1 = model.tokenizer.convert_tokens_to_ids(b1)
        base = ids.clone(); base[start] = i1
        # position 2: b1 committed, 2 and 3 masked
        span2 = torch.tensor([start + 1, start + 2], device=ids.device)
        r2 = _infill_logprobs(model, base, span2).double()[0]
        for b2 in "TCAG":
            i2 = model.tokenizer.convert_tokens_to_ids(b2)
            base2 = base.clone(); base2[start + 1] = i2
            span1 = torch.tensor([start + 2], device=ids.device)
            r3 = _infill_logprobs(model, base2, span1).double()[0]
            for b3 in "TCAG":
                i3 = model.tokenizer.convert_tokens_to_ids(b3)
                out[b1 + b2 + b3] = float(r1[i1] + r2[i2] + r3[i3])
    return out


def aa_logsumexp(codon_logp, aa):
    v = [codon_logp[c] for c, a in GENETIC_CODE.items() if a == aa and c in codon_logp]
    if not v:
        return None
    t = torch.tensor(v, dtype=torch.float64)
    return float(torch.logsumexp(t, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-variants", type=int, default=300)
    ap.add_argument("--model-length", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config, step = load_checkpoint_model(
        Path(a.checkpoint), a.model_length, 16, device)
    model.tokenizer = tokenizer
    records = list(read_jsonl_gz(Path(a.data)))
    # keep only variants with EXACTLY ONE non-synonymous codon: the stratum the
    # codon form was designed for, and the one where a single codon's joint is
    # the whole score.
    keep = []
    for r in records:
        pos = variant_positions(r["wt_sequence"], r["mut_sequence"])
        if not pos:
            continue
        codons = sorted({p // 3 for p in pos})
        nsyn = [c for c in codons
                if GENETIC_CODE.get(r["wt_sequence"][3 * c:3 * c + 3].upper())
                != GENETIC_CODE.get(r["mut_sequence"][3 * c:3 * c + 3].upper())]
        if len(nsyn) == 1:
            keep.append((r, nsyn[0]))
        if len(keep) >= a.max_variants:
            break
    print(f"{len(keep)} single-non-synonymous-codon variants", flush=True)

    x0, _ = build_pair_tensors([r for r, _ in keep], tokenizer,
                               a.model_length, None)
    x0 = x0.to(device)

    rows = []
    with torch.no_grad():
        for i, (rec, codon) in enumerate(keep):
            start = 3 * codon
            wt_ids = x0[2 * i]
            prod = codon_logprob_product(model, wt_ids, start)
            chain = codon_logprob_chain(model, wt_ids, start)
            wt_c = rec["wt_sequence"][start:start + 3].upper()
            mut_c = rec["mut_sequence"][start:start + 3].upper()
            wt_aa, mut_aa = GENETIC_CODE.get(wt_c), GENETIC_CODE.get(mut_c)
            if wt_aa is None or mut_aa is None:
                continue
            s_prod = aa_logsumexp(prod, mut_aa)
            t_prod = aa_logsumexp(prod, wt_aa)
            s_chain = aa_logsumexp(chain, mut_aa)
            t_chain = aa_logsumexp(chain, wt_aa)
            if None in (s_prod, t_prod, s_chain, t_chain):
                continue
            # how far is the factorised codon distribution from the exact one?
            pj = np.array([prod[c] for c in sorted(prod)])
            cj = np.array([chain[c] for c in sorted(chain)])
            pj -= np.log(np.exp(pj).sum()); cj -= np.log(np.exp(cj).sum())
            kl = float((np.exp(cj) * (cj - pj)).sum())
            rows.append({"urn": rec["score_set_urn"],
                         "experimental_score": rec.get("experimental_score"),
                         "score_product": s_prod - t_prod,
                         "score_chain": s_chain - t_chain,
                         "kl_chain_vs_product": kl})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(keep)}", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.output, "w"))
    print(f"wrote {len(rows)} rows -> {a.output}")


if __name__ == "__main__":
    main()
