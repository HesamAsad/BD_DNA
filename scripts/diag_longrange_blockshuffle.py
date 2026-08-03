"""Decisive GPU diagnostic: is the ~0 block-shuffle NLL effect a real 'model is
local' result, or an underpowered test over a vestigial coarse pathway?

Runs on the bigblock dual checkpoint (block_size=24576, window_blocks=1). Uses
the first NB blocks of the real long-range cache (carbon-prok-lr983) truncated to
L = NB*block_size, EMA weights, fixed masked positions for matched comparisons.

Three tests:
  (A) CONTEXT-SWAP KL on a fixed target block i, predictions split by within-block
      token DEPTH (boundary <=Wb nt from block start vs deep). Three swaps:
        swapNEIGHBOR : replace only block i-1  -> isolates the self-attn M_OBC path
                       (the ONLY cross-block self-attn signal at window_blocks=1).
                       H1 predicts boundary KL>0, deep KL~0.
        swapFAR      : replace blocks 0..i-2 (keep i-1, i) -> reachable ONLY via the
                       coarse stream. KL>0 here == coarse carries usable long-range
                       info to block i; KL~0 == coarse pathway not used (H2/H5).
        swapALL      : replace every block != i (combined).
  (B) PATHWAY ABLATION: zero cross_attn.out_proj (kills coarse cross-attn), and
      separately zero self_proj, re-measure per-masked-token CE. Quantifies how
      many nats the coarse pathway actually buys.
  (C) BLOCK-SHUFFLE delta-NLL vs within-block offset: contiguous vs a fixed
      whole-block permutation, per-masked-token NLL binned by distance-from-block-
      start. Reproduces the aggregate ~0 and localizes where (if anywhere) it bites.
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
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 10      # blocks to use
NWIN = int(sys.argv[4]) if len(sys.argv) > 4 else 6     # windows (rows)
dev = 'cuda'
torch.manual_seed(0)

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
hp = ck.get('hyper_parameters', {})
cfg = omegaconf.OmegaConf.create(hp.get('config', hp))
BS = cfg.block_size                      # 24576
KC = cfg.model.k_coarse                  # 6
L = NB * BS
cfg.model.length = L                     # rebuild masks at the truncated length
cfg.eval.checkpoint_path = CKPT
print(f"ckpt={CKPT}\n  step={ck.get('global_step')} block={BS} k_coarse={KC} "
      f"window_blocks={cfg.model.window_blocks} -> eval L={L} ({NB} blocks), NWIN={NWIN}", flush=True)

tok = dataloader.get_tokenizer(cfg)
model = diffusion.Diffusion.load_from_checkpoint(
    CKPT, tokenizer=tok, config=cfg, strict=False, weights_only=False).to(dev)
model.eval()
if getattr(model, 'ema', None) is not None:
    model.ema.copy_to(model._get_parameters())
    print("  EMA weights applied (matches val/nll).", flush=True)
MASK = model.mask_index
V = model.vocab_size
bb = model.backbone

# ---- data: first L nt of NWIN windows ----
ds = datasets.load_from_disk(VAL_CACHE).with_format('numpy')
rows = np.stack([ds[i]['input_ids'][:L] for i in range(NWIN)]).astype(np.int64)
X = torch.tensor(rows, device=dev)                      # (NWIN, L)
def as_blocks(x): return x.view(x.shape[0], NB, BS)

def fwd(xt, ctx_x0):
    p = torch.full((1, 1), 0.5, device=dev)             # t value irrelevant: time_cond off
    sigma = model._sigma_from_p(p)
    x_in = torch.cat((xt, ctx_x0), -1) if model.cross_attn else xt
    with torch.no_grad():
        return model.forward(x_in, sigma=sigma)         # (1,L,V) logprobs

def kl(p, q):
    p = p.clamp_min(1e-12); q = q.clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(-1)

# within-block offset of every position
off = (torch.arange(L, device=dev) % BS)

# ============================ (A) CONTEXT-SWAP KL =========================
print("\n========== (A) CONTEXT-SWAP KL on a fixed target block ==========")
i = NB // 2                                              # target block (middle)
t_mask = 0.3
# fixed masked positions inside the target block (same across all conditions/windows)
gmask = torch.zeros(1, L, dtype=torch.bool, device=dev)
blk_lo, blk_hi = i*BS, (i+1)*BS
rng = torch.Generator(device=dev).manual_seed(1)
m_in_blk = (torch.rand(BS, generator=rng, device=dev) <= t_mask)
m_in_blk[0] = False
gmask[0, blk_lo:blk_hi] = m_in_blk
mvec = gmask[0]                                          # (L,)
off_masked = off[mvec]                                   # offsets of scored tokens (within target block)

Wb = [64, 256, 1024]                                     # boundary widths (nt from block start)
def split_report(klv):
    # klv: (n_masked,) KL per scored token
    out = {}
    for w in Wb:
        bnd = off_masked < w
        out[f'bnd<{w}'] = (klv[bnd].mean().item(), int(bnd.sum()))
    deep = off_masked > 2048
    out['deep>2048'] = (klv[deep].mean().item(), int(deep.sum()))
    out['ALL'] = (klv.mean().item(), int(mvec.sum()))
    return out

conds = ['swapNEIGHBOR', 'swapFAR', 'swapALL']
agg = {c: [] for c in conds}
for w in range(NWIN):
    x0 = X[w:w+1].clone()
    xt = torch.where(gmask, torch.full_like(x0, MASK), x0)
    base = fwd(xt, x0)[0][mvec].exp()                    # (n_masked, V)
    src = X[(w+1) % NWIN].view(NB, BS)                   # donor blocks from another window
    for c in conds:
        ctx = x0.view(NB, BS).clone()
        if c == 'swapNEIGHBOR':
            ctx[i-1] = src[i-1]
        elif c == 'swapFAR':
            if i-1 > 0: ctx[:i-1] = src[:i-1]            # blocks 0..i-2
        elif c == 'swapALL':
            keep = ctx[i].clone(); ctx[:] = src; ctx[i] = keep
        ctx = ctx.view(1, L)
        # xt unchanged (target block content & mask identical); only clean ctx changes
        xt_c = torch.where(gmask, torch.full_like(ctx, MASK), ctx)
        pc = fwd(xt_c, ctx)[0][mvec].exp()
        agg[c].append(kl(base, pc))
        torch.cuda.empty_cache()
print(f"  target block i={i}, scored tokens/window={int(mvec.sum())}, masking t={t_mask}")
print(f"  KL(base || swapped) in nats; deep>2048 reachable ONLY via coarse stream\n")
print(f"  {'condition':>13} | " + " ".join(f"{k:>12}" for k in ['bnd<64','bnd<256','bnd<1024','deep>2048','ALL']))
for c in conds:
    klv = torch.cat(agg[c])
    # recompute split over concatenated windows: offsets repeat per window
    off_rep = off_masked.repeat(NWIN)
    def rep_split():
        d = {}
        for w_ in Wb: d[f'bnd<{w_}'] = klv[off_rep < w_].mean().item()
        d['deep>2048'] = klv[off_rep > 2048].mean().item()
        d['ALL'] = klv.mean().item()
        return d
    d = rep_split()
    print(f"  {c:>13} | " + " ".join(f"{d[k]:>12.5f}" for k in ['bnd<64','bnd<256','bnd<1024','deep>2048','ALL']))
print("  interpretation: swapNEIGHBOR boundary>0 & deep~0 => H1 (only boundary tokens")
print("    see cross-block via self-attn). swapFAR deep~0 => coarse carries no long-range.")

# ===================== (B) PATHWAY ABLATION (NLL) ========================
print("\n========== (B) PATHWAY ABLATION: per-masked-token CE ==========")
orig = {}
for j, blk in enumerate(bb.fine_blocks):
    orig[j] = (blk.self_proj.weight.data.clone(),
               blk.cross_attn.out_proj.weight.data.clone(),
               blk.cross_attn.out_proj.bias.data.clone())
def set_mode(zero_self=False, zero_cross=False):
    for j, blk in enumerate(bb.fine_blocks):
        sw, cw, cb = orig[j]
        blk.self_proj.weight.data.copy_(torch.zeros_like(sw) if zero_self else sw)
        blk.cross_attn.out_proj.weight.data.copy_(torch.zeros_like(cw) if zero_cross else cw)
        blk.cross_attn.out_proj.bias.data.copy_(torch.zeros_like(cb) if zero_cross else cb)
# fixed full-sequence masks per window
fmask = {}
for w in range(NWIN):
    m = torch.rand(1, L, device=dev) <= 0.3; m[:, 0] = False
    fmask[w] = m
def measure_ce():
    ces = []
    for w in range(NWIN):
        x0 = X[w:w+1]; mv = fmask[w]
        xt = torch.where(mv, torch.full_like(x0, MASK), x0)
        logp = fwd(xt, x0)
        lp = logp.gather(-1, x0[..., None]).squeeze(-1)
        ces.append((-(lp[mv])).mean().item())
        del logp; torch.cuda.empty_cache()
    return float(np.mean(ces))
set_mode(); full = measure_ce()
set_mode(zero_cross=True); noc = measure_ce()
set_mode(zero_self=True); nos = measure_ce()
set_mode()
print(f"  full CE                 = {full:.4f} nats")
print(f"  cross-attn OFF          = {noc:.4f}  (delta {noc-full:+.4f}  <- coarse pathway value)")
print(f"  self-attn  OFF          = {nos:.4f}  (delta {nos-full:+.4f})")

# ============= (C) BLOCK-SHUFFLE delta-NLL vs offset-in-block =============
print("\n========== (C) BLOCK-SHUFFLE delta-NLL vs within-block offset ==========")
perm = torch.randperm(NB, generator=torch.Generator(device=dev).manual_seed(7), device=dev)
print(f"  block permutation: {perm.tolist()}")
ts = [0.15, 0.3, 0.5]
# accumulate per-offset-bin NLL for contiguous vs shuffled
bins = [(0,64),(64,256),(256,1024),(1024,4096),(4096,12288),(12288,24576)]
acc = {tag: {b: [0.0,0] for b in bins} for tag in ['contig','shuf']}
agg_nll = {'contig': [], 'shuf': []}
for w in range(NWIN):
    base = X[w:w+1]
    shufd = as_blocks(base.clone())[:, perm, :].reshape(1, L)
    for tag, x0 in [('contig', base), ('shuf', shufd)]:
        for t in ts:
            g = torch.Generator(device=dev).manual_seed(1000*w + int(100*t))
            mv = (torch.rand(1, L, generator=g, device=dev) <= t); mv[:, 0] = False
            xt = torch.where(mv, torch.full_like(x0, MASK), x0)
            logp = fwd(xt, x0)
            nll = -logp.gather(-1, x0[..., None]).squeeze(-1)[0]    # (L,)
            sel = mv[0]
            agg_nll[tag].append(nll[sel].mean().item())
            o = off[sel]; n = nll[sel]
            for (lo, hi) in bins:
                m = (o >= lo) & (o < hi)
                if m.any():
                    acc[tag][(lo,hi)][0] += n[m].sum().item()
                    acc[tag][(lo,hi)][1] += int(m.sum())
            del logp; torch.cuda.empty_cache()
print(f"  aggregate mean NLL  contig={np.mean(agg_nll['contig']):.5f}  "
      f"shuf={np.mean(agg_nll['shuf']):.5f}  delta={np.mean(agg_nll['shuf'])-np.mean(agg_nll['contig']):+.5f}")
print(f"  (reproduces the ~0 aggregate finding if delta~0)\n")
print(f"  {'offset-bin':>14} | {'contig_NLL':>10} {'shuf_NLL':>10} {'delta':>9} {'n':>10}")
for (lo,hi) in bins:
    c = acc['contig'][(lo,hi)]; s = acc['shuf'][(lo,hi)]
    cm = c[0]/max(c[1],1); sm = s[0]/max(s[1],1)
    print(f"  [{lo:>6},{hi:>6}) | {cm:>10.5f} {sm:>10.5f} {sm-cm:>+9.5f} {c[1]:>10}")
print("  H1 confirmed if delta is concentrated in the first bin(s) and ~0 deep in the block.")
print("\nDONE", flush=True)
