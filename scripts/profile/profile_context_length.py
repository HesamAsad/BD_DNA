"""Profile BD3-LM GPU memory vs context length (batch=1), to find the longest
context that fits one GPU and chart how memory scales.

For each context length it rebuilds the real DIT backbone (with the BD3-LM flex
block mask), runs a forward + backward on a *dummy* batch (memory depends on
tensor shapes, not values, so no tokenized data is needed), and reports peak GPU
memory — stopping at the first OOM. With BD3LM_LOG_BLOCK_MEM=1 (set by the launch
script), dit.py also prints allocated/peak memory after each transformer block.

Note: BD3-LM cross-attention concatenates [x_t; x_0], so the backbone's working
sequence is 2*length. The DIT backbone holds the vast majority of memory; the
diffusion wrapper (noise schedule, metrics, EMA) is negligible by comparison.
"""
import gc
import os
import sys
import traceback

import torch
import torch._dynamo
from hydra import compose, initialize_config_dir

REPO = '/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms'
sys.path.insert(0, REPO)  # so `import dataloader`/`import models.dit` work from scripts/profile/

import dataloader
import models.dit

CONFIG_DIR = os.path.join(REPO, 'configs')

LENGTHS = [int(x) for x in os.environ.get(
    'PROFILE_LENGTHS',
    '4096,16384,65536,131072,262144,524288,1048576').replace(',', ' ').split()]
BLOCK_SIZE = int(os.environ.get('PROFILE_BLOCK_SIZE', '16'))
ATTN = os.environ.get('ATTN', 'flex')


def profile_one(cfg, length, vocab_size):
  """Build the DIT backbone at `length` and run one fwd+bwd; return peak GiB."""
  dit = models.dit.DIT(cfg, vocab_size=vocab_size).cuda().train()
  # cross-attn training input is length 2*L ([x_t; x_0]); sigma is reduced to
  # shape (batch,) by diffusion before the backbone, so mimic that here.
  idx = torch.randint(0, vocab_size, (1, 2 * length), device='cuda')
  sigma = torch.zeros(1, device='cuda')
  out = dit(idx, sigma)
  loss = out.float().sum()
  loss.backward()
  torch.cuda.synchronize()
  return torch.cuda.max_memory_allocated() / 2**30


def main():
  assert torch.cuda.is_available(), 'profiler needs a GPU'
  print(f'GPU: {torch.cuda.get_device_name(0)} '
        f'({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB) '
        f'| torch {torch.__version__} | attn={ATTN} | block_size={BLOCK_SIZE}',
        flush=True)
  vocab_size = dataloader.DNATokenizer().vocab_size  # 13 (mask token in-vocab)
  results = []
  with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    for length in LENGTHS:
      torch._dynamo.reset()  # fresh compile per length; avoids flex dynamic-shape inductor bug
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()
      cfg = compose(config_name='config', overrides=[
          'model=small', 'algo=bd3lm', 'data=carbon-prokaryote',
          f'model.length={length}', f'block_size={BLOCK_SIZE}',
          f'model.attn_backend={ATTN}',
          'loader.batch_size=1', 'loader.eval_batch_size=1',
          'loader.global_batch_size=1', 'loader.eval_global_batch_size=1',
          'wandb=null'])
      print(f'\n===== context length {length} (working seq {2 * length}) =====',
            flush=True)
      try:
        peak = profile_one(cfg, length, vocab_size)
        print(f'[RESULT] length={length:>9}  OK   peak={peak:6.1f} GiB',
              flush=True)
        results.append((length, f'{peak:.1f} GiB'))
      except Exception as e:
        # OOM can arrive as torch.cuda.OutOfMemoryError or a RuntimeError wrapped
        # by the TorchScript-fused kernels; treat both as the ceiling and stop.
        if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower():
          print(f'[RESULT] length={length:>9}  OOM', flush=True)
          results.append((length, 'OOM'))
          gc.collect(); torch.cuda.empty_cache()
          break
        print(f'[RESULT] length={length:>9}  ERROR {type(e).__name__}: {e}',
              flush=True)
        traceback.print_exc()
        results.append((length, f'ERROR ({type(e).__name__})'))
      gc.collect()
      torch.cuda.empty_cache()

  print('\n================ SUMMARY (batch=1) ================', flush=True)
  for length, status in results:
    print(f'  length={length:>9}  ->  {status}', flush=True)


if __name__ == '__main__':
  main()
