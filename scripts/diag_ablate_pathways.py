"""GPU: on a PRE-COLLAPSE dual checkpoint, attribute the sub-floor gain to the
self-attention vs cross-attention pathway, and report cross-attn coverage.

Decisively answers the user's deliverable (c): "how much of the 1.24-vs-1.385
sub-floor gain does cross-attn carry vs the windowed self-attn?"  by ZEROING the
relevant residual output projection (cross contribution = gate_cross * out_proj,
so out_proj.weight=0 kills cross; self_proj.weight=0 kills windowed self-attn)
and re-measuring per-masked-token NLL.

Prediction (from the offline gate trajectory: gate_cross~0.03 vestigial, gate1
~0.21 carries the signal):
  zero cross_out  -> NLL ~ unchanged (cross is vestigial)
  zero self_proj  -> NLL jumps toward the marginal floor
  zero both       -> NLL == marginal floor  (MLP-only stack = unigram)

Run on an H200 (full-length batch=1 forward). All forwards no_grad.
"""
import os, sys, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import omegaconf
import datasets
import diffusion, dataloader

CKPT = sys.argv[1]
VAL_CACHE = sys.argv[2]
NBLOCKS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
dev = 'cuda'
torch.manual_seed(0)

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
hp = ck.get('hyper_parameters', {})
cfg = omegaconf.OmegaConf.create(hp.get('config', hp))
cfg.eval.checkpoint_path = CKPT
print(f"ckpt={CKPT}\n  step={ck.get('global_step')} len={cfg.model.length} block={cfg.block_size} "
      f"backbone={cfg.algo.backbone}", flush=True)

tok = dataloader.get_tokenizer(cfg)
model = diffusion.Diffusion.load_from_checkpoint(
    CKPT, tokenizer=tok, config=cfg, strict=False, weights_only=False).to(dev)
model.eval()
if getattr(model, 'ema', None) is not None:
    model.ema.copy_to(model._get_parameters())
    print("  EMA weights applied (matches val/nll metric).", flush=True)

L = cfg.model.length
MASK = model.mask_index
V = model.vocab_size
bb = model.backbone
nbl = len(bb.fine_blocks)

# snapshot the two projection weights/biases per block so we can ablate & restore
orig = {}
for i, blk in enumerate(bb.fine_blocks):
    orig[i] = (blk.self_proj.weight.data.clone(),
               blk.self_proj.bias.data.clone() if blk.self_proj.bias is not None else None,
               blk.cross_attn.out_proj.weight.data.clone(),
               blk.cross_attn.out_proj.bias.data.clone() if blk.cross_attn.out_proj.bias is not None else None)

def set_mode(zero_self=False, zero_cross=False):
    for i, blk in enumerate(bb.fine_blocks):
        sw, sb, cw, cb = orig[i]
        blk.self_proj.weight.data.copy_(torch.zeros_like(sw) if zero_self else sw)
        if sb is not None: blk.self_proj.bias.data.copy_(torch.zeros_like(sb) if zero_self else sb)
        blk.cross_attn.out_proj.weight.data.copy_(torch.zeros_like(cw) if zero_cross else cw)
        if cb is not None: blk.cross_attn.out_proj.bias.data.copy_(torch.zeros_like(cb) if zero_cross else cb)

# ---- data + marginal floor on the EXACT scored targets ----
ds = datasets.load_from_disk(VAL_CACHE).with_format('numpy')
rows = np.stack([ds[i]['input_ids'] for i in range(NBLOCKS)]).astype(np.int64)
x0_all = torch.tensor(rows, device=dev)
flat = rows[:, 1:].reshape(-1)
binc = np.bincount(flat, minlength=V).astype(np.float64); binc[MASK] = 0
H_marg = float(-(p := binc/binc.sum())[p > 0] @ np.log(p[p > 0]))
print(f"  H_unigram(empirical floor) = {H_marg:.4f} nats (ppl {math.exp(H_marg):.3f})\n", flush=True)

def fwd(xt, ctx_x0, t):
    pp = torch.full((1, 1), float(t), device=dev)
    sigma = model._sigma_from_p(pp)
    x_in = torch.cat((xt, ctx_x0), -1) if model.cross_attn else xt
    with torch.no_grad():
        return model.forward(x_in, sigma=sigma)

# fixed masked positions per block (shared across all ablations & a sweep of t)
TS = [0.05, 0.15, 0.3, 0.5, 0.8]
masks = {}
for b in range(NBLOCKS):
    m = torch.rand(1, L, device=dev) <= 0.15
    m[:, 0] = False
    masks[b] = m

def measure(label):
    out = {}
    for t in TS:
        ces = []
        for b in range(NBLOCKS):
            x0 = x0_all[b:b+1]; mv = masks[b]
            xt = torch.where(mv, torch.full_like(x0, MASK), x0)
            logp = fwd(xt, x0, t)
            lp = logp.gather(-1, x0.clamp(min=0)[..., None]).squeeze(-1)
            ces.append((-(lp[mv])).mean().item())
            del logp; torch.cuda.empty_cache()
        out[t] = float(np.mean(ces))
    print(f"  [{label:>16}] per-masked-token CE by t: " +
          " ".join(f"t={t}:{out[t]:.3f}" for t in TS) +
          f"   (floor {H_marg:.3f})", flush=True)
    return out

print("[c] PATHWAY ABLATION (per-masked-token CE; lower=more conditional)")
set_mode(); full = measure('full')
set_mode(zero_cross=True); noc = measure('cross OFF')
set_mode(zero_self=True);  nos = measure('self OFF')
set_mode(zero_self=True, zero_cross=True); non = measure('both OFF')
set_mode()  # restore

print("\n  --- gain attribution at t=0.15 (CE relative to floor) ---")
t = 0.15
g_full = H_marg - full[t]
print(f"    full sub-floor gain         = {g_full:+.4f} nats  (CE {full[t]:.3f} vs floor {H_marg:.3f})")
print(f"    lost when cross-attn OFF    = {noc[t]-full[t]:+.4f} nats  -> cross carries {100*(noc[t]-full[t])/max(g_full,1e-9):.1f}% of the gain")
print(f"    lost when self-attn OFF     = {nos[t]-full[t]:+.4f} nats  -> self  carries {100*(nos[t]-full[t])/max(g_full,1e-9):.1f}% of the gain")
print(f"    both OFF (expect ~= floor)  = CE {non[t]:.3f}   (floor {H_marg:.3f})")

# ---- cross-attn coverage (analytic from the mask; xt-half queries, M_OBC) ----
print("\n[c] CROSS-ATTN COVERAGE (analytic, xt-half queries under M_OBC)")
bs = cfg.block_size; kc = bb.k_coarse
nblk = L // bs
ncoarse = L // kc
# block i query sees coarse 0..(i*bs//kc - 1); block 0 -> none
keys_per_block = [max(i * bs // kc, 0) for i in range(nblk)]
fully_masked_rows = sum(bs for i in range(nblk) if keys_per_block[i] == 0)
print(f"    block_size={bs} k_coarse={kc} n_fine_blocks={nblk} n_coarse_tokens={ncoarse}")
print(f"    fully-masked (zero-filled) query rows = {fully_masked_rows}/{L} "
      f"= {100*fully_masked_rows/L:.3f}%  (= block 0 only)")
print(f"    mean coarse keys visible per fine block = {np.mean(keys_per_block):.1f} "
      f"(block1={keys_per_block[1]}, last={keys_per_block[-1]}, max possible={ncoarse})")
print("\nDONE", flush=True)
