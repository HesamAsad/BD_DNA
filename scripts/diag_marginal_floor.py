"""CPU diagnostic: marginal-entropy floor vs the reported ~2.3-nat plateau.

Loads the SAME wrapped/grouped validation cache the training run consumed and
computes, on the EXACT scored targets:
  (1) token composition (ACGT / N / EOS / specials / PAD / MASK),
  (2) unigram cross-entropy H(unigram) in nats = the NELBO a marginal predictor
      would achieve under this MDLM(subs)+loglinear objective,
  (3) a Monte-Carlo simulation of the EXACT estimator in diffusion._loss with a
      constant marginal predictor: loss = -(1/t) * log p_uni(x0) at masked
      positions (subs => unmasked contribute 0), averaged as
      nlls.sum()/attention_mask.sum(). Confirms (2) numerically incl. the -1/t
      weighting, and reports single-batch variance.
  (4) reference floors: uniform-13 (ln13), uniform-ACGT (ln4).
  (5) doc-boundary / EOS density (hyp 3: wrap gluing unlearnable context).

No GPU, no model weights -- purely the data + objective arithmetic.
"""
import sys, math
import numpy as np
import datasets
import torch

CACHE = sys.argv[1] if len(sys.argv) > 1 else (
    "data_cache/carbon/carbon-prokaryote_validation_bs98496_wrapped_specialFalse_nf1.dat")
BLOCK_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 18
N_ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 50   # rows to load (val batch count)

# DNATokenizer ids
NAMES = {0:'[CLS]',1:'[SEP]',2:'[BOS]',3:'[EOS]',4:'[MASK]',5:'[PAD]',
         6:'[RSV]',7:'[UNK]',8:'A',9:'C',10:'G',11:'T',12:'N'}
MASK_INDEX = 4
ACGT = [8,9,10,11]

ds = datasets.load_from_disk(CACHE).with_format('numpy')
n = min(N_ROWS, len(ds))
ids = np.stack([ds[i]['input_ids'] for i in range(n)]).astype(np.int64)  # (n, L)
L = ids.shape[1]
print(f"cache={CACHE}\n  rows_used={n}/{len(ds)}  block_len(L)={L}  block_size(diff)={BLOCK_SIZE}")

# ignore_bos: val zeroes attention_mask[:,0]. attention_mask is otherwise all-ones.
scored = ids[:, 1:].reshape(-1)          # every position except col 0 is scored
total = scored.size

# (1) composition
print("\n[1] token composition over SCORED val positions (attention_mask=1, excl col0):")
vals, counts = np.unique(scored, return_counts=True)
comp = {int(v): int(c) for v, c in zip(vals, counts)}
for v in sorted(comp):
    print(f"    {NAMES.get(v,v):6s} id={v:2d}: {comp[v]:>12,d}  {100*comp[v]/total:6.3f}%")
acgt_frac = sum(comp.get(i,0) for i in ACGT)/total
print(f"    ACGT fraction = {acgt_frac:.5f}   non-ACGT = {1-acgt_frac:.5f}")

# (2) unigram cross-entropy of the scored targets (nats)
p = counts / counts.sum()
H_unigram = float(-(p * np.log(p)).sum())
print(f"\n[2] H(unigram) over scored targets = {H_unigram:.4f} nats  (ppl {math.exp(H_unigram):.3f})")
print(f"    --> a model that learned ONLY the marginal scores exactly this NELBO.")

# (4) reference floors
print(f"\n[4] reference floors:")
print(f"    uniform-13 (ln 13)        = {math.log(13):.4f} nats  (ppl {13:.1f})")
print(f"    uniform-ACGT (ln 4)       = {math.log(4):.4f} nats  (ppl {4:.1f})")
n_present = len([v for v in comp if comp[v] > 0])
print(f"    uniform over {n_present} present ids = {math.log(n_present):.4f} nats")
print(f"    REPORTED PLATEAU          ~ 2.30   nats  (ppl ~{math.exp(2.30):.1f})")

# (3) MC simulation of the EXACT diffusion._loss estimator w/ marginal predictor
# Build log p_uni over full vocab (id->logprob); ids not present -> -inf (never targets).
VOCAB = 13
log_puni = np.full(VOCAB, -1e9)
for v in comp:
    log_puni[v] = math.log(comp[v]/total)
log_puni_t = torch.tensor(log_puni)
x0 = torch.tensor(ids)                                   # (n, L)
# attention_mask: all ones, then ignore_bos zeroes col 0 (val, not training)
attn = torch.ones_like(x0, dtype=torch.float64)
attn[:, 0] = 0.0
num_blocks = L // BLOCK_SIZE
torch.manual_seed(0)

def estimator_once():
    # sample t per (row, block) in (eps, 1], mask w.p. t, score with marginal.
    t_b = torch.rand(n, num_blocks, dtype=torch.float64) * (1 - 1e-3) + 1e-3
    t = t_b.repeat_interleave(BLOCK_SIZE, dim=1)          # (n, L)
    move = torch.rand(n, L, dtype=torch.float64) <= t     # masked positions
    loss_scale = -1.0 / t
    log_p_theta = log_puni_t[x0]                          # marginal log-prob of TRUE token
    # subs: unmasked positions contribute 0; masked contribute loss_scale*log_p
    loss = torch.where(move, loss_scale * log_p_theta, torch.zeros_like(log_p_theta))
    nlls = loss * attn
    return (nlls.sum() / attn.sum()).item()

draws = np.array([estimator_once() for _ in range(200)])
print(f"\n[3] MC of EXACT estimator (-1/t weight, subs) with MARGINAL predictor:")
print(f"    mean over 200 t-draws = {draws.mean():.4f} nats   (target H_unigram={H_unigram:.4f})")
print(f"    per-draw std          = {draws.std():.4f}   (single-batch noise of the metric)")
print(f"    --> the -1/t weighting does NOT inflate the floor; marginal NELBO == H(unigram).")

# uniform-ACGT predictor through the same estimator (sanity = ln4 on ACGT targets)
log_unif_acgt = np.full(VOCAB, -1e9);
for i in ACGT: log_unif_acgt[i] = math.log(0.25)
lu = torch.tensor(log_unif_acgt)
def est_unif():
    t_b = torch.rand(n, num_blocks, dtype=torch.float64) * (1-1e-3) + 1e-3
    t = t_b.repeat_interleave(BLOCK_SIZE, dim=1)
    move = torch.rand(n, L, dtype=torch.float64) <= t
    log_p = lu[x0]                                         # -inf where target is N/EOS -> huge loss
    loss = torch.where(move, (-1.0/t)*log_p, torch.zeros_like(log_p))
    return ((loss*attn).sum()/attn.sum()).item()
# guard against -inf*: only meaningful if ~all targets ACGT
if 1-acgt_frac < 1e-3:
    du = np.array([est_unif() for _ in range(50)])
    print(f"    uniform-ACGT predictor through estimator = {du.mean():.4f} (expect ln4={math.log(4):.4f})")

# (5) EOS / doc-boundary density (hyp 3)
eos = int((ids == 3).sum())
print(f"\n[5] doc-gluing (hyp 3): [EOS]=3 count over loaded blocks = {eos}")
print(f"    EOS per block = {eos/n:.3f}  -> boundary density = {eos/total*100:.4f}% of positions")
print(f"    (val uses insert_valid_eos=False, so within-block doc joins are seamless &")
print(f"     boundary positions are a negligible fraction either way.)")

print("\n==== VERDICT INPUTS ====")
print(f"H_unigram(marginal floor) = {H_unigram:.3f} nats | plateau ~2.30 nats | gap = {2.30-H_unigram:+.3f} nats")
