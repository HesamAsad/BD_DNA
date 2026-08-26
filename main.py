import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  text_samples = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', text_samples)
  print('Generative perplexity:',
        model.metrics.gen_ppl.compute())
  print('Entropy:', model.metrics.gen_entropy.compute())
  csv_path = config.sampling.logdir
  save_dict = {'gen_ppl': model.metrics.gen_ppls,
                'gen_nfes': model.metrics.gen_nfes,
                'gen_entropy': model.metrics.gen_entropies,
                'gen_lengths': model.metrics.gen_lengths,
                'samples': [[i] for i in text_samples],
                'seed': [config.seed for _ in range(len(text_samples))]}
  if config.sampling.var_length:
    print(text_samples)
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  # Log to Weights & Biases when `wandb` is configured, else a lightweight CSV
  # logger. We avoid `logger=None` (Lightning falls back to TensorBoardLogger,
  # which imports tensorflow) and `logger=False` (breaks LearningRateMonitor).
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  else:
    wandb_logger = L.pytorch.loggers.CSVLogger(
      save_dir=os.getcwd(), name='csv_logs')
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _io_dump(config, logger, tokenizer):
  """Dump real long-context input / tokens / decoded predictions so the forward
  pass can be eyeballed for validity at long L (not just trusted via NLL).

  For each requested mask fraction it noises a real validation sequence with the
  model's own q_xt, runs the EMA model forward, and writes to IO_DUMP_DIR:
    - report.txt   : masked-position accuracy OVERALL and PER POSITION-BIN along
                     the full length (uniform-across-bins => valid at long range),
                     plus the ACGT composition of predictions at masked sites.
    - windows.txt  : decoded truth / masked-input / prediction at the START,
                     MIDDLE and END of the sequence (manual eyeball).
    - decoded_seq0.txt : full decoded truth + prediction (FASTA-ish).
    - arrays.npz   : raw int16 token ids (x0, xt, pred) per fraction.
  Env tunables: IO_DUMP_NSEQ, IO_DUMP_FRACS, IO_DUMP_BINS, IO_DUMP_WIN, IO_DUMP_DIR.
  """
  import itertools
  import numpy as np
  logger.info('Starting IO dump.')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.eval()
  # Evaluate under the EMA weights (matches the validation/NLL path).
  if getattr(model, 'ema', None) is not None:
    model.ema.store(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
    model.ema.copy_to(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
  model.backbone.eval()

  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)

  n_seqs = int(os.environ.get('IO_DUMP_NSEQ', '1'))
  fracs = [float(x) for x in
           os.environ.get('IO_DUMP_FRACS', '0.15,0.5').split(',')]
  nbins = int(os.environ.get('IO_DUMP_BINS', '20'))
  win = int(os.environ.get('IO_DUMP_WIN', '120'))
  L_cfg = config.model.length
  out_dir = os.environ.get(
    'IO_DUMP_DIR', os.path.join(os.getcwd(), f'io_dump_L{L_cfg}'))
  os.makedirs(out_dir, exist_ok=True)

  batch = next(iter(valid_ds))
  x0 = batch['input_ids'][:n_seqs].to(model.device)
  B, L = x0.shape

  NUC = {8: 'A', 9: 'C', 10: 'G', 11: 'T', 12: 'N', 4: '.'}
  to_str = lambda ids: ''.join(NUC.get(int(i), '?') for i in ids)

  torch.manual_seed(config.seed)
  saved = {'x0': x0.to(torch.int16).cpu().numpy()}
  report = [
    f'IO DUMP  length={L}  n_seqs={B}  mask_fracs={fracs}  bins={nbins}',
    f'checkpoint={config.eval.checkpoint_path}',
    'x0=real validation DNA | xt=q_xt-noised input | '
    'pred=argmax model log p(x0|xt). Accuracy is over MASKED positions only '
    '(unmasked are copied by the subs parameterization). chance=0.25 (ACGT).',
    '']
  win_lines, dec_lines = [], []
  for frac in fracs:
    p = torch.full((B, 1), frac, device=model.device)
    sigma = model._sigma_from_p(p)
    xt = model.q_xt(x0, p)
    if model.ignore_bos:
      xt[:, 0] = x0[:, 0]
    x_input = torch.cat((xt, x0), dim=-1) if model.cross_attn else xt
    with torch.no_grad():
      logp = model.forward(x_input, sigma=sigma)  # (B, L, V) log-probs
    pred = logp.argmax(-1)

    masked = (xt == model.mask_index)
    n_masked = int(masked.sum())
    acc = float(((pred == x0) & masked).sum()) / max(n_masked, 1)
    pm = pred[masked]
    comp = {NUC.get(v, str(v)): f'{100*float((pm == v).sum())/max(len(pm),1):.1f}%'
            for v in [8, 9, 10, 11, 12]}

    pos = torch.arange(L, device=model.device)
    binid = pos * nbins // L
    bin_acc = []
    for b in range(nbins):
      mb = masked & (binid.unsqueeze(0) == b)
      tb = int(mb.sum())
      bin_acc.append(float(((pred == x0) & mb).sum()) / max(tb, 1))

    report += [
      f'=== mask_frac={frac}  (sigma={float(sigma[0,0]):.3f}) ===',
      f'  masked tokens={n_masked}  masked-accuracy={acc:.4f}',
      f'  pred composition @ masked: {comp}',
      f'  masked-accuracy by position bin (0=start .. {nbins-1}=end):',
      '    ' + ' '.join(f'{a:.3f}' for a in bin_acc), '']

    saved[f'xt_frac{frac}'] = xt.to(torch.int16).cpu().numpy()
    saved[f'pred_frac{frac}'] = pred.to(torch.int16).cpu().numpy()
    dec_lines += [f'>seq0_pred_frac{frac} len={L}', to_str(pred[0])]

    for tag, st in [('START', 0), ('MIDDLE', L // 2), ('END', max(L - win, 0))]:
      sl = slice(st, st + win)
      mk = masked[0, sl]
      match = ''.join(
        '^' if (mk[j] and x0[0, st + j] == pred[0, st + j])
        else ('x' if mk[j] else ' ') for j in range(int(mk.shape[0])))
      win_lines += [
        f'--- frac={frac} seq0 {tag} pos[{st}:{st+win}] ---',
        f'truth: {to_str(x0[0, sl])}',
        f'input: {to_str(xt[0, sl])}   (. = [MASK])',
        f'pred : {to_str(pred[0, sl])}',
        f'match: {match}   (^=correct@mask  x=wrong@mask)', '']

  np.savez_compressed(os.path.join(out_dir, 'arrays.npz'), **saved)
  open(os.path.join(out_dir, 'report.txt'), 'w').write('\n'.join(report))
  open(os.path.join(out_dir, 'windows.txt'), 'w').write('\n'.join(win_lines))
  open(os.path.join(out_dir, 'decoded_seq0.txt'), 'w').write(
    f'>seq0_truth len={L}\n{to_str(x0[0])}\n' + '\n'.join(dec_lines) + '\n')
  logger.info('\n'.join(report))
  logger.info(f'IO dump written to {out_dir}')


def _synth_copy_eval(config, logger, tokenizer):
  """Targeted long-range copy metric for the synthetic echo benchmark.

  Loads <data.valid> cache + <name>_echo_manifest.json, masks ONLY the echo TARGET
  spans (sources left clean), runs the EMA forward once, and reports exact
  per-nucleotide copy accuracy at the targets bucketed by gap and by within- vs
  cross-block, plus a random-span CONTROL (should be ~0.25). High cross-block
  accuracy == the model used >block_size context. See
  docs/synthetic_longrange_benchmark.md.  Env: SYNTH_NSEQ (default 64).
  """
  import itertools
  import json
  from collections import defaultdict
  import numpy as np
  import datasets as _ds
  logger.info('Starting synthetic copy eval.')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.eval()
  if getattr(model, 'ema', None) is not None:
    model.ema.store(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
    model.ema.copy_to(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
  model.backbone.eval()

  cache_dir, name, L = config.data.cache_dir, config.data.valid, config.model.length
  block = config.block_size
  path = os.path.join(cache_dir, f'{name}_validation_bs{L}_wrapped_specialFalse.dat')
  manifest = json.load(
    open(os.path.join(cache_dir, f'{name}_echo_manifest.json')))['echoes']
  data = _ds.load_from_disk(path).with_format('numpy')
  n_seqs = int(os.environ.get('SYNTH_NSEQ', '64'))
  rng = np.random.default_rng(0)

  hit, tot = defaultdict(int), defaultdict(int)   # keyed by ('within'|'cross', gap)
  ctrl_hit = ctrl_tot = 0
  for i in range(min(n_seqs, data.num_rows)):
    echoes = manifest[i]
    if not echoes:
      continue
    x0 = torch.tensor(data[i]['input_ids'], dtype=torch.long,
                      device=model.device).unsqueeze(0)
    tmask = torch.zeros_like(x0, dtype=torch.bool)
    occupied = np.zeros(L, bool)
    for e in echoes:
      tmask[0, e['target']:e['target'] + e['motif_len']] = True
      occupied[e['source']:e['source'] + e['motif_len']] = True
      occupied[e['target']:e['target'] + e['motif_len']] = True
    # control: equal number of random non-echo spans of the same motif length
    ctrl_spans = []
    mlen = echoes[0]['motif_len']
    for _ in range(len(echoes)):
      for _try in range(50):
        st = int(rng.integers(0, L - mlen))
        if not occupied[st:st + mlen].any():
          occupied[st:st + mlen] = True
          ctrl_spans.append(st)
          tmask[0, st:st + mlen] = True
          break
    xt = torch.where(tmask, model.mask_index, x0)
    p = torch.full((1, 1), float(tmask.float().mean()), device=model.device)
    sigma = model._sigma_from_p(p)
    x_input = torch.cat((xt, x0), dim=-1) if model.cross_attn else xt
    with torch.no_grad():
      pred = model.forward(x_input, sigma=sigma).argmax(-1)
    correct = (pred == x0)[0]
    for e in echoes:
      sl = slice(e['target'], e['target'] + e['motif_len'])
      key = ('within' if e['source'] // block == e['target'] // block else 'cross',
             e['gap'])
      hit[key] += int(correct[sl].sum())
      tot[key] += e['motif_len']
    for st in ctrl_spans:
      ctrl_hit += int(correct[st:st + mlen].sum())
      ctrl_tot += mlen

  report = [f'SYNTH COPY EVAL  name={name} L={L} block={block} seqs={i+1}',
            f'checkpoint={config.eval.checkpoint_path}',
            f'control (random spans) acc = {ctrl_hit/max(ctrl_tot,1):.4f}  (chance 0.25)',
            'copy accuracy at TARGET spans, by (placement, gap):']
  for key in sorted(tot, key=lambda k: (k[0], k[1])):
    report.append(f'  {key[0]:6s} gap={key[1]:7d}: acc={hit[key]/tot[key]:.4f}  '
                  f'(n={tot[key]})')
  for placement in ('within', 'cross'):
    h = sum(hit[k] for k in tot if k[0] == placement)
    t = sum(tot[k] for k in tot if k[0] == placement)
    if t:
      report.append(f'  ALL {placement}-block: acc={h/t:.4f} (n={t})')
  logger.info('\n'.join(report))
  print('\n'.join(report))


def _oracle_delta_eval(config, logger, tokenizer):
  """H_signal: how much predictive information lives BEYOND radius d of a target?

  For each target span T, compare two forwards that differ ONLY in the informativeness
  of the distal region -- never in the mask, so the mask fraction (and hence sigma) is
  identical and cannot confound:
    A  "true"      : x0 everywhere, T masked.
    B(d) "ablated" : x0 within [t-d, t+M+d] kept TRUE, everything beyond it token-SHUFFLED
                     (a permutation -> exact same composition, structure destroyed), T masked.
  Delta(d) = NLL_T(B(d)) - NLL_T(A)  >0 means context beyond d carries real information.
  Delta(0) is the total context value; Delta(d) -> 0 as d grows. The d at which Delta
  vanishes is the effective usable range. Scoring only T, so this is not diluted by the
  99% of positions that are locally predictable. See plan Stage 2 / 6.4.

  Env: ORACLE_NSEQ (default 32), ORACLE_NTGT (4), ORACLE_TGTLEN (256),
       ORACLE_RADII (comma list, default 0,256,1024,4096,16384).
  """
  import itertools
  import numpy as np
  import datasets as _ds
  logger.info('Starting oracle Delta(d) eval.')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.eval()
  if getattr(model, 'ema', None) is not None:
    model.ema.store(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
    model.ema.copy_to(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
  model.backbone.eval()

  cache_dir, name, L = config.data.cache_dir, config.data.valid, config.model.length
  path = os.path.join(cache_dir, f'{name}_validation_bs{L}_wrapped_specialFalse.dat')
  data = _ds.load_from_disk(path).with_format('numpy')
  n_seqs = int(os.environ.get('ORACLE_NSEQ', '32'))
  n_tgt = int(os.environ.get('ORACLE_NTGT', '4'))
  M = int(os.environ.get('ORACLE_TGTLEN', '256'))
  radii = [int(x) for x in os.environ.get(
    'ORACLE_RADII', '0,256,1024,4096,16384').split(',')]
  rng = np.random.default_rng(0)

  def nll_at(x0_t, tmask, distal_keep=None, donor=None):
    """Mean NLL (nats/nt) over tmask positions. distal_keep=None -> full true context.
    donor!=None -> LOCUS SWAP: replace the distal region with the SAME positions from a
    different genomic window (composition CHANGES), instead of permuting (composition
    preserved). This closes the loophole that a permutation hides isochore/GC-regime
    signal -- if Delta jumps under locus-swap where it was ~0 under permute, the distal
    value is compositional, not order-based."""
    xt = torch.where(tmask, model.mask_index, x0_t)
    inp = x0_t.clone()
    if distal_keep is not None:
      idx = (~distal_keep).nonzero(as_tuple=True)[1]
      if donor is not None:                          # locus swap (composition changes)
        inp[0, idx] = donor[0, idx]
      elif idx.numel() > 1:                          # permute (composition preserved)
        perm = idx[torch.randperm(idx.numel(), device=idx.device)]
        inp[0, idx] = x0_t[0, perm]
      xt = torch.where(tmask, model.mask_index, inp)
    p = torch.full((1, 1), float(tmask.float().mean()), device=model.device)
    sigma = model._sigma_from_p(p)
    x_input = torch.cat((xt, inp), dim=-1) if model.cross_attn else xt
    with torch.no_grad():
      logp = model.forward(x_input, sigma=sigma)
    tgt = torch.gather(logp, -1, x0_t[:, :, None]).squeeze(-1)
    return float(-(tgt[tmask]).mean())

  ablate = os.environ.get('ORACLE_ABLATE', 'permute')   # 'permute' | 'locus_swap'
  n_rows = min(n_seqs, data.num_rows)
  acc = {d: [] for d in radii}
  base_nlls = []
  for i in range(n_rows):
    x0 = torch.tensor(data[i]['input_ids'], dtype=torch.long,
                      device=model.device).unsqueeze(0)
    donor = None
    if ablate == 'locus_swap':                        # distal from a DIFFERENT window
      donor = torch.tensor(data[(i + 1) % n_rows]['input_ids'], dtype=torch.long,
                           device=model.device).unsqueeze(0)
    starts = [int(rng.integers(M, L - M)) for _ in range(n_tgt)]
    tmask = torch.zeros_like(x0, dtype=torch.bool)
    for s in starts:
      tmask[0, s:s + M] = True
    nll_true = nll_at(x0, tmask)
    base_nlls.append(nll_true)
    for d in radii:
      keep = torch.zeros_like(x0, dtype=torch.bool)
      for s in starts:
        keep[0, max(0, s - d):min(L, s + M + d)] = True
      acc[d].append(nll_at(x0, tmask, distal_keep=keep, donor=donor) - nll_true)

  rep = [f'ORACLE Delta(d)  name={name} L={L} seqs={n_rows} ablate={ablate} '
         f'targets/seq={n_tgt} target_len={M}',
         f'checkpoint={config.eval.checkpoint_path}',
         f'baseline NLL at targets (full true context) = {np.mean(base_nlls):.4f} nats',
         'Delta(d) = NLL(distal beyond d shuffled) - NLL(full true); >0 => that context matters']
  for d in radii:
    a = np.array(acc[d])
    sem = a.std() / max(np.sqrt(len(a)), 1)
    rep.append(f'  radius d={d:>6}: Delta={a.mean():+.4f} +/- {1.96*sem:.4f} nats (95% CI)')
  logger.info('\n'.join(rep))
  print('\n'.join(rep))


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  # Log to Weights & Biases when `wandb` is configured, else a lightweight CSV
  # logger. We avoid `logger=None` (Lightning falls back to TensorBoardLogger,
  # which imports tensorflow) and `logger=False` (breaks LearningRateMonitor).
  # ALWAYS attach a CSV logger, even when wandb is on. Lightning takes a list.
  # wandb is for watching a run live; the CSV is what survives locally and is
  # what any later plot of val/nll and val/ppl across arms will be built from.
  # Without this, a run with WANDB_MODE=online leaves NO metric file on disk.
  csv_logger = L.pytorch.loggers.CSVLogger(
    save_dir=os.getcwd(), name='csv_logs', flush_logs_every_n_steps=50)
  if config.get('wandb', None) is not None:
    wandb_logger = [
      L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config),
        ** config.wandb),
      csv_logger]
  else:
    wandb_logger = csv_logger

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.from_pretrained is not None and ckpt_path is None:
    logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
    # load pretraining checkpoint
    if 'kuleshov-group/' in config.training.from_pretrained:
      # load from hf
      model = diffusion.Diffusion(config, tokenizer=tokenizer)
      state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
          config.training.from_pretrained,
          trust_remote_code=True
      ).state_dict()
      model.load_state_dict(state_dict)
    else:
      model = diffusion.Diffusion.load_from_checkpoint(
        config.training.from_pretrained,
        tokenizer=tokenizer,
        config=config,
        strict=False)
    # add buffers for grid search
    model.register_buffer('sampling_eps_min', torch.tensor(
      config.training.sampling_eps_min))
    model.register_buffer('sampling_eps_max', torch.tensor(
      config.training.sampling_eps_max))
  else:
    logger.info(f'Initializing new model')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  # In DDP, Lightning re-executes this script once per rank (setting LOCAL_RANK)
  # and the model is built *before* Lightning assigns each rank its GPU. The
  # BD3-LM flex block mask is created on the current CUDA device in
  # DIT.__init__, so without this every rank would build on cuda:0 and collide
  # (fatal under gpu `mode=exclusive_process`). Pin each rank to its own GPU.
  if torch.cuda.is_available() and 'LOCAL_RANK' in os.environ:
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    config.wandb = None
    samples = generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'io_dump':
    config.wandb = None
    _io_dump(config, logger, tokenizer)
  elif config.mode == 'synth_copy_eval':
    config.wandb = None
    _synth_copy_eval(config, logger, tokenizer)
  elif config.mode == 'oracle_delta':
    config.wandb = None
    _oracle_delta_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()