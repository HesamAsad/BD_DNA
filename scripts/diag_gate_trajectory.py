"""OFFLINE (CPU) post-mortem of the dual-stream marginal collapse.

Because time_conditioning=False, the adaLN gates are CONSTANTS derived purely
from the weights:  t_cond = silu(sigma_map(0)); gate{1,cross,2} are chunks of
adaLN_modulation(t_cond).  So the *entire* gate trajectory across training can be
reconstructed from checkpoints alone -- no GPU, no data, no forward pass.

For each checkpoint (RAW weights and, if mappable, EMA weights) and each fine
block this prints:
  gate1   (self-attn residual gate)      mean / |mean| / L2     <- HYP 1
  gate_cr (cross-attn residual gate)     mean / |mean| / L2     <- HYP 1 (shared)
  gate2   (mlp residual gate)            mean / |mean| / L2     <- survives?
  ||W|| of self_proj / cross out_proj / adaLN_modulation / qkv  <- pathway death
  (optional) AdamW exp_avg_sq RMS of adaLN_modulation.weight    <- HYP 2 spike
and the stored sampling_eps_min/max (confirms var_min is a no-op).

Usage:
  python scripts/diag_gate_trajectory.py <ckpt_dir> [step1 step2 ...]
If no steps given, uses a strategic set spanning the collapse.
"""
import os, sys, glob, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('BD3LM_COMPILE_MASK', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import torch
import torch.nn.functional as F

CKPT_DIR = sys.argv[1]
STEPS = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else None
torch.set_grad_enabled(False)

# ----------------------------------------------------------------------------
# 1) Recover the backbone parameter ORDER (to index EMA shadow list by name).
#    Param order is independent of model.length, so instantiate a tiny clone.
# ----------------------------------------------------------------------------
NAME2IDX = None
try:
  import omegaconf, models.dit_dual as dd
  # find a hydra config next to the ckpt dir
  cfgp = None
  for c in [os.path.join(CKPT_DIR, '..', '.hydra', 'config.yaml'),
            os.path.join(os.path.dirname(CKPT_DIR.rstrip('/')), '.hydra', 'config.yaml')]:
    if os.path.exists(c):
      cfgp = c; break
  cfg = omegaconf.OmegaConf.load(cfgp)
  small = omegaconf.OmegaConf.create(omegaconf.OmegaConf.to_container(cfg, resolve=False))
  small.model.length = small.block_size  # tiny but same module structure
  bb = dd.DualStreamDIT(small, vocab_size=14)
  names = [n for n, p in bb.named_parameters() if p.requires_grad]
  NAME2IDX = {n: i for i, n in enumerate(names)}
  print(f"[order] recovered {len(names)} backbone param slots for EMA mapping", flush=True)
except Exception as e:
  print(f"[order] EMA mapping unavailable ({type(e).__name__}: {e}); RAW only", flush=True)


def find_ckpts(d):
  out = {}
  for f in glob.glob(os.path.join(d, '*.ckpt')):
    base = os.path.basename(f)
    m = re.search(r'(\d+)-(\d+)\.ckpt$', base)  # epoch-step.ckpt
    if m:
      out[int(m.group(2))] = f
  return out


def timestep_embedding_zero(dim=256):
  # timestep_embedding(t=0): cos(0)=1 for first half, sin(0)=0 for second half
  e = torch.zeros(1, dim)
  e[0, : dim // 2] = 1.0
  return e


def t_cond_from(sd, prefix):
  w0 = sd[f'{prefix}sigma_map.mlp.0.weight'].float()
  b0 = sd[f'{prefix}sigma_map.mlp.0.bias'].float()
  w2 = sd[f'{prefix}sigma_map.mlp.2.weight'].float()
  b2 = sd[f'{prefix}sigma_map.mlp.2.bias'].float()
  freq = timestep_embedding_zero(w0.shape[1])
  h = F.silu(F.linear(freq, w0, b0))
  t_emb = F.linear(h, w2, b2)
  return F.silu(t_emb)  # (1, cond_dim)


def gates_for_block(sd, prefix, i, t_cond, dim):
  W = sd[f'{prefix}fine_blocks.{i}.adaLN_modulation.weight'].float()
  b = sd[f'{prefix}fine_blocks.{i}.adaLN_modulation.bias'].float()
  mod = F.linear(t_cond, W, b)              # (1, 7*dim)
  ch = mod.squeeze(0).chunk(7, dim=-1)      # shift1,scale1,gate1,gate_cross,shift2,scale2,gate2
  g1, gc, g2 = ch[2], ch[3], ch[6]
  def stat(g):
    return (g.mean().item(), g.abs().mean().item(), g.norm().item())
  return stat(g1), stat(gc), stat(g2)


def wnorm(sd, key):
  return float(sd[key].float().norm()) if key in sd else float('nan')


def analyze_state(sd, prefix, label, n_blocks, dim):
  t_cond = t_cond_from(sd, prefix)
  rows = []
  for i in range(n_blocks):
    g1, gc, g2 = gates_for_block(sd, prefix, i, t_cond, dim)
    sp = wnorm(sd, f'{prefix}fine_blocks.{i}.self_proj.weight')
    cp = wnorm(sd, f'{prefix}fine_blocks.{i}.cross_attn.out_proj.weight')
    ada = wnorm(sd, f'{prefix}fine_blocks.{i}.adaLN_modulation.weight')
    qkv = wnorm(sd, f'{prefix}fine_blocks.{i}.qkv.weight')
    rows.append(dict(i=i, g1=g1, gc=gc, g2=g2, sp=sp, cp=cp, ada=ada, qkv=qkv))
  return rows


def summarize(rows, label):
  # aggregate |mean| of gates and proj norms across blocks (mean and block0/blockL)
  g1a = np.mean([r['g1'][1] for r in rows]); g1L2 = np.mean([r['g1'][2] for r in rows])
  gca = np.mean([r['gc'][1] for r in rows]); gcL2 = np.mean([r['gc'][2] for r in rows])
  g2a = np.mean([r['g2'][1] for r in rows]); g2L2 = np.mean([r['g2'][2] for r in rows])
  sp = np.mean([r['sp'] for r in rows]); cp = np.mean([r['cp'] for r in rows])
  ada = np.mean([r['ada'] for r in rows])
  print(f"   [{label}] mean over {len(rows)} blocks | "
        f"|gate1|={g1a:.4f} (L2 {g1L2:.3f})  |gate_cr|={gca:.4f} (L2 {gcL2:.3f})  "
        f"|gate2|={g2a:.4f} (L2 {g2L2:.3f}) | "
        f"||self_proj||={sp:.3f} ||cross_out||={cp:.3f} ||adaLN||={ada:.3f}", flush=True)
  return dict(g1=g1a, gc=gca, g2=g2a, sp=sp, cp=cp, ada=ada, g1L2=g1L2, gcL2=gcL2, g2L2=g2L2)


def ema_state_dict(ck, prefix='backbone.'):
  """Map EMA shadow_params list back to backbone.* names via NAME2IDX."""
  if NAME2IDX is None or 'ema' not in ck:
    return None
  shadow = ck['ema']['shadow_params']
  out = {}
  for name, idx in NAME2IDX.items():
    if idx < len(shadow):
      out[prefix + name] = shadow[idx]
  return out


def main():
  ckpts = find_ckpts(CKPT_DIR)
  steps = sorted(ckpts) if STEPS is None else [s for s in STEPS if s in ckpts]
  if STEPS is None:
    # strategic subset spanning the collapse if the full set is large
    want = [500, 1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 6500,
            7000, 8000, 9000, 10000, 12000]
    steps = [s for s in want if s in ckpts] or steps
  print(f"ckpt_dir={CKPT_DIR}\nsteps={steps}\n", flush=True)

  traj_raw, traj_ema = {}, {}
  for s in steps:
    f = ckpts[s]
    try:
      ck = torch.load(f, map_location='cpu', weights_only=False, mmap=True)
    except Exception:
      ck = torch.load(f, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    gstep = ck.get('global_step')
    eps_min = ck.get('sampling_eps_min'); eps_max = ck.get('sampling_eps_max')
    n_blocks = 1 + max(int(m.group(1)) for k in sd
                       for m in [re.match(r'backbone\.fine_blocks\.(\d+)\.', k)] if m)
    dim = sd['backbone.fine_blocks.0.self_proj.weight'].shape[0]
    em = eps_min.item() if torch.is_tensor(eps_min) else eps_min
    ex = eps_max.item() if torch.is_tensor(eps_max) else eps_max
    print(f"=== step {s} (global_step={gstep}) eps=({em},{ex}) blocks={n_blocks} dim={dim} ===", flush=True)
    rows = analyze_state(sd, 'backbone.', 'RAW', n_blocks, dim)
    traj_raw[s] = summarize(rows, 'RAW')
    # per-block gate1/gate_cr |mean| so we can see if specific blocks die
    print("       per-block |gate1| : " + " ".join(f"{r['g1'][1]:.3f}" for r in rows), flush=True)
    print("       per-block |gate_cr|: " + " ".join(f"{r['gc'][1]:.3f}" for r in rows), flush=True)
    print("       per-block |gate2| : " + " ".join(f"{r['g2'][1]:.3f}" for r in rows), flush=True)
    esd = ema_state_dict(ck)
    if esd is not None:
      try:
        erows = analyze_state(esd, 'backbone.', 'EMA', n_blocks, dim)
        traj_ema[s] = summarize(erows, 'EMA')
      except Exception as e:
        print(f"   [EMA] failed: {type(e).__name__}: {e}", flush=True)
    del ck, sd
    print('', flush=True)

  # compact trajectory table
  print("\n==== TRAJECTORY (RAW |gate| means across blocks) ====")
  print(f"{'step':>6} {'|gate1|':>9} {'|gate_cr|':>9} {'|gate2|':>9} "
        f"{'||self_p||':>10} {'||cross||':>10} {'||adaLN||':>10}")
  for s in steps:
    if s in traj_raw:
      t = traj_raw[s]
      print(f"{s:>6} {t['g1']:>9.4f} {t['gc']:>9.4f} {t['g2']:>9.4f} "
            f"{t['sp']:>10.3f} {t['cp']:>10.3f} {t['ada']:>10.3f}")
  if traj_ema:
    print("\n==== TRAJECTORY (EMA |gate| means across blocks) ====")
    print(f"{'step':>6} {'|gate1|':>9} {'|gate_cr|':>9} {'|gate2|':>9} "
          f"{'||self_p||':>10} {'||cross||':>10} {'||adaLN||':>10}")
    for s in steps:
      if s in traj_ema:
        t = traj_ema[s]
        print(f"{s:>6} {t['g1']:>9.4f} {t['gc']:>9.4f} {t['g2']:>9.4f} "
              f"{t['sp']:>10.3f} {t['cp']:>10.3f} {t['ada']:>10.3f}")
  print("\nDONE", flush=True)


if __name__ == '__main__':
  main()
