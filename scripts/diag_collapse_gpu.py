"""GPU diagnostic on a trained checkpoint: confirm marginal collapse and fork
the 'why'. Produces the user's (b) collapse test and (c) noise-stratified loss.

All forwards are batch=1 (training used batch=1 at this length -> avoid OOM).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import omegaconf
import datasets

import diffusion, dataloader

CKPT = sys.argv[1]
VAL_CACHE = sys.argv[2]
NBLOCKS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.manual_seed(0)
dev = 'cuda'

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
hp = ck.get('hyper_parameters', {})
cfg = omegaconf.OmegaConf.create(hp.get('config', hp))
cfg.eval.checkpoint_path = CKPT
print(f"ckpt={CKPT}\n  step={ck.get('global_step')} len={cfg.model.length} block={cfg.block_size} "
      f"backbone={cfg.algo.backbone} mdlm_loss_scale={cfg.algo.mdlm_loss_scale}", flush=True)

tok = dataloader.get_tokenizer(cfg)
model = diffusion.Diffusion.load_from_checkpoint(
    CKPT, tokenizer=tok, config=cfg, strict=False, weights_only=False).to(dev)
model.eval()
if getattr(model, 'ema', None) is not None:
    model.ema.copy_to(model._get_parameters())
    print("  EMA weights applied.", flush=True)

L = cfg.model.length
MASK = model.mask_index
V = model.vocab_size

ds = datasets.load_from_disk(VAL_CACHE).with_format('numpy')
rows = np.stack([ds[i]['input_ids'] for i in range(NBLOCKS)]).astype(np.int64)
x0_all = torch.tensor(rows, device=dev)
flat = rows[:, 1:].reshape(-1)
binc = np.bincount(flat, minlength=V).astype(np.float64); binc[MASK] = 0
marg = torch.tensor(binc / binc.sum(), device=dev)
H_marg = float(-(marg[marg > 0] * marg[marg > 0].log()).sum())
print(f"  loaded {NBLOCKS} blocks; H_unigram(empirical)={H_marg:.4f} nats (ppl {math.exp(H_marg):.3f})\n", flush=True)

def fwd(xt, ctx_x0, t):
    p = torch.full((1, 1), float(t), device=dev)
    sigma = model._sigma_from_p(p)
    x_in = torch.cat((xt, ctx_x0), -1) if model.cross_attn else xt
    with torch.no_grad():
        return model.forward(x_in, sigma=sigma)   # (1,L,V) logprobs

print("[c] NOISE STRATIFICATION (per-masked-token, mean over blocks, batch=1)")
print(f"    {'t':>6} {'raw_CE':>9} {'pred_H':>8}   floor H_unigram={H_marg:.3f}")
for t in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98]:
    ces, hs = [], []
    for b in range(NBLOCKS):
        x0 = x0_all[b:b+1]
        moved = torch.rand(1, L, device=dev) <= t
        moved[:, 0] = False
        xt = torch.where(moved, torch.full_like(x0, MASK), x0)
        logp = fwd(xt, x0, t)
        lp = logp.gather(-1, x0.clamp(min=0)[..., None]).squeeze(-1)
        ces.append((-(lp[moved])).mean().item())
        ent = -(logp.exp() * logp).sum(-1)
        hs.append(ent[moved].mean().item())
        del logp, ent; torch.cuda.empty_cache()
    print(f"    {t:>6.2f} {np.mean(ces):>9.4f} {np.mean(hs):>8.4f}", flush=True)

print("\n[b] COLLAPSE / CONDITIONALITY at low noise (t=0.15, same masked positions)")
t = 0.15
moved = torch.rand(1, L, device=dev) <= t; moved[:, 0] = False
A = x0_all[0:1]; Bb = x0_all[1:2]
xtA = torch.where(moved, torch.full_like(A, MASK), A)
xtB = torch.where(moved, torch.full_like(Bb, MASK), Bb)
def kl(p, q):
    p = p.clamp_min(1e-12); q = q.clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(-1)
predA = fwd(xtA, A, t)[0][moved[0]].exp(); torch.cuda.empty_cache()
predB = fwd(xtB, Bb, t)[0][moved[0]].exp(); torch.cuda.empty_cache()
print(f"    mean KL(pred_A || marginal) = {kl(predA, marg[None]).mean().item():.4f}  (~0 => predicts marginal)")
print(f"    mean KL(pred_B || marginal) = {kl(predB, marg[None]).mean().item():.4f}")
print(f"    mean KL(pred_A || pred_B)   = {kl(predA, predB).mean().item():.4f}  (~0 => ignores which seq)")
print(f"    mean pred entropy A         = {(-(predA.clamp_min(1e-12).log()*predA).sum(-1)).mean().item():.4f}")
top = predA.mean(0).topk(6); print("    avg pred dist top tokens (id:prob):",
      {int(i): round(float(v),3) for v, i in zip(top.values, top.indices)})

print("\n    CONTEXT-SWAP (xt from A fixed; clean ctx A -> B):")
ps = fwd(xtA, A, t)[0][moved[0]].exp(); torch.cuda.empty_cache()
pw = fwd(xtA, Bb, t)[0][moved[0]].exp(); torch.cuda.empty_cache()
print(f"    mean KL(pred_ctxA || pred_ctxB) = {kl(ps, pw).mean().item():.5f} nats")
print(f"      ~0  => clean-context path is DEAD (conditioning never reaches head)")
print(f"      >0  => context is read but model still collapses (optimization)")
print("\nDONE", flush=True)
