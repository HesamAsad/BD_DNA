"""Inspect a checkpoint dir: best val/nll + whether the dual fine-block self/cross
output projections left zero (deadlock broken)."""
import sys, os, glob, re, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = sys.argv[1]
def load(f):
    return torch.load(f, map_location='cpu', weights_only=False)
best = sorted(glob.glob(d + '/checkpoints/best*.ckpt'))
last = sorted(glob.glob(d + '/checkpoints/last*.ckpt'))
for f in best[:1]:
    ck = load(f); sc = None
    for v in ck.get('callbacks', {}).values():
        if isinstance(v, dict) and v.get('best_model_score') is not None: sc = float(v['best_model_score'])
    print(f"BEST {f.split('/')[-1]} step={ck.get('global_step')} best_val_nll={sc}  (unigram floor=1.382)")
if last:
    sd = load(last[0])['state_dict']
    blks = sorted(set(int(m.group(1)) for k in sd for m in [re.match(r'backbone\.fine_blocks\.(\d+)\.', k)] if m))
    def mn(key): return float(sd[key].float().norm()) if key in sd else float('nan')
    sp = [mn(f'backbone.fine_blocks.{i}.self_proj.weight') for i in blks]
    cp = [mn(f'backbone.fine_blocks.{i}.cross_attn.out_proj.weight') for i in blks]
    print(f"LAST {last[0].split('/')[-1]} step={load(last[0]).get('global_step')}")
    print(f"  self_proj.weight  norms (per block): " + " ".join(f"{x:.2f}" for x in sp))
    print(f"  cross out_proj.w  norms (per block): " + " ".join(f"{x:.2f}" for x in cp))
    print(f"  --> deadlock {'BROKEN (nonzero)' if (max(sp)>1e-4 and max(cp)>1e-4) else 'STILL PRESENT (zero)'}")
