import contextlib
import itertools
import math
import time
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict

import dataloader
import metrics
import models
import noise_schedule
import utils
import numpy as np
import itertools

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    # L_use: counterfactual coarse-route incentive (P2 / plan Stage 3). Off by
    # default; enable with algo.l_use_weight>0. margin is in nats.
    self.l_use_weight = float(self.config.algo.get('l_use_weight', 0.0))
    self.l_use_margin = float(self.config.algo.get('l_use_margin', 0.2))
    self._last_l_use = None
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    prepend_bos = self.config.sampling.get('prepend_bos', None)
    if prepend_bos is None:
      prepend_bos = bool(
        self.config.data.get('insert_train_special', True))
    self.prepend_bos = bool(prepend_bos)

    # DNA generation is sequence generation, not special-token generation.
    # Keep the generic text-LM behavior unchanged, while limiting DNA content
    # to A/C/G/T/N and admitting EOS only as a variable-length control token.
    generation_ids = getattr(self.tokenizer, 'generation_token_ids', None)
    if generation_ids is None:
      generation_mask = torch.ones(self.vocab_size, dtype=torch.bool)
    else:
      generation_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
      generation_mask[list(generation_ids)] = True
      if self.config.sampling.var_length:
        generation_mask[self.tokenizer.eos_token_id] = True
    self.register_buffer(
      '_generation_token_mask', generation_mask, persistent=False)

    # Reverse-complement data augmentation. Independent of the architectural
    # `model.rc_equivariant` path: this one only relabels training data, needs
    # no checkpoint change and costs one flip plus one gather per step. Absent
    # or 0.0 (the default) short-circuits before any RNG is drawn, so existing
    # configs keep their exact random stream.
    self.rc_augment_probability = float(
      self.config.data.get('rc_augment_probability', 0.0)
      if hasattr(self.config, 'data') else 0.0)
    if self.rc_augment_probability > 0:
      import models.rc_equivariance as rc_equivariance
      if not 0.0 < self.rc_augment_probability <= 1.0:
        raise ValueError(
          'data.rc_augment_probability must lie in (0, 1], got '
          f'{self.rc_augment_probability}')
      rc_equivariance.assert_rc_safe_special_tokens(self.config)
      self.register_buffer(
        '_complement_token_ids',
        rc_equivariance.complement_permutation(
          self.vocab_size, tokenizer=self.tokenizer),
        persistent=False)
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dit_dual':
      self.backbone = models.dit_dual.DualStreamDIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'bissm':
      self.backbone = models.bidirectional_ssm.BidirectionalSSM(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'ussm':
      self.backbone = models.unidirectional_ssm.UnidirectionalSSM(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      # Host mirrors, read only by `if` statements (see `_eps_host`). Kept in
      # step with the buffers at `on_load_checkpoint` and
      # `_clipped_schedule_search`, the only two places they are written.
      self._sampling_eps_min_host = float(self.sampling_eps_min.item())
      self._sampling_eps_max_host = float(self.sampling_eps_max.item())
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit', 'dit_dual', 'bissm', 'ussm'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
    if self.prepend_bos and self.tokenizer.bos_token_id is None:
      raise ValueError(
        'sampling.prepend_bos=True requires a tokenizer BOS token')
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
      self._sampling_eps_min_host = float(self.sampling_eps_min.item())
      self._sampling_eps_max_host = float(self.sampling_eps_max.item())
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _model_autocast_context(self):
    """Use the configured mixed precision in direct and Lightning calls.

    Lightning already supplies this context during fit/validate. Nesting the
    same context is harmless and also makes checkpoint evaluation and native
    sampling use the identical model dtype instead of silently falling back to
    FP32.
    """
    if self.device.type != 'cuda':
      return contextlib.nullcontext()
    precision = str(self.config.trainer.precision).lower()
    if precision.startswith('bf16'):
      return torch.autocast('cuda', dtype=torch.bfloat16)
    if precision in {'16', '16-mixed', 'fp16', 'fp16-mixed'}:
      return torch.autocast('cuda', dtype=torch.float16)
    return contextlib.nullcontext()

  def _bos_is_possible(self):
    """Whether any row *could* carry BOS, decided without touching the device.

    When this is False `_bos_rows` is the all-False vector by construction, so
    every downstream masked write is a no-op and can be skipped outright.
    """
    return bool(self.ignore_bos) and self.tokenizer.bos_token_id is not None

  def _bos_rows(self, tokens):
    """Rows whose first token really is BOS (rather than the first base)."""
    if not self._bos_is_possible():
      return torch.zeros(
        tokens.shape[0], dtype=torch.bool, device=tokens.device)
    return tokens[:, 0].eq(self.tokenizer.bos_token_id)

  def _preserve_observed_bos(self, noisy_tokens, clean_tokens):
    # `bos_rows.any()` is a device->host sync: it drains the launch queue in
    # the middle of a step whose cost is dominated by issuing launches. The
    # branchless form writes column 0 unconditionally, leaving non-BOS rows
    # holding exactly the value they already held, so the result is bitwise
    # identical for every row. See scripts/smoke/fix_equivalence.py::F2.
    if not self._bos_is_possible():
      return noisy_tokens
    bos_rows = self._bos_rows(clean_tokens)
    noisy_tokens[:, 0] = torch.where(
      bos_rows, clean_tokens[:, 0], noisy_tokens[:, 0])
    return noisy_tokens

  def _restrict_generation_probs(self, probabilities, current_tokens=None):
    """Remove non-alphabet symbols and renormalize sampling probabilities."""
    mask = self._generation_token_mask.to(probabilities.device)
    if mask.all():
      return probabilities
    restricted = probabilities * mask.to(probabilities.dtype)
    normalizer = restricted.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
      if current_tokens is None:
        raise RuntimeError(
          'The model assigned zero probability to every allowed generation token')
      # Substitution parameterization represents observed control tokens (for
      # example an explicitly requested BOS) as a point mass. Preserve those
      # already-observed positions without making BOS sampleable elsewhere.
      fallback = F.one_hot(
        current_tokens, num_classes=self.vocab_size).to(probabilities.dtype)
      restricted = torch.where(normalizer <= 0, fallback, restricted)
      normalizer = restricted.sum(dim=-1, keepdim=True)
    return restricted / normalizer

  def _initialize_ar_tokens(self, batch_size, sequence_length):
    if sequence_length <= 0:
      raise ValueError('Generation length must be positive')
    tokens = torch.zeros(
      (batch_size, sequence_length), dtype=torch.long, device=self.device)
    if self.prepend_bos:
      tokens[:, 0] = self.tokenizer.bos_token_id
      return tokens
    allowed = self._generation_token_mask.nonzero(as_tuple=False).squeeze(-1)
    # EOS is a stopping control token, never a valid context-free first base.
    if self.tokenizer.eos_token_id is not None:
      allowed = allowed[allowed != self.tokenizer.eos_token_id]
    if allowed.numel() == 0:
      raise RuntimeError('No valid non-EOS token is available to seed generation')
    choices = torch.randint(allowed.numel(), (batch_size,), device=self.device)
    tokens[:, 0] = allowed[choices]
    return tokens

  def _initialize_diffusion_tokens(self, batch_size, sequence_length):
    if sequence_length <= 0:
      raise ValueError('Generation length must be positive')
    tokens = self._sample_prior(batch_size, sequence_length).to(self.device)
    if self.prepend_bos:
      tokens[:, 0] = self.tokenizer.bos_token_id
    return tokens

  def forward(self, x, sigma, sample_mode=False, store_kv=False,
              coarse_ablate=None):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    # Deliberately FP32, overriding Lightning's bf16 AMP for the duration of
    # the backbone call. Every published Transformer number was produced this
    # way, and the DiT is not merely imprecise in bf16 here -- it fails to
    # train: under bf16 both the lr 3e-4/beta2 0.999 recipe (LSF 103274) and
    # the lr 1e-3/beta2 0.95 recipe (LSF 103182) plateau at val/nll ~1.342,
    # while the same checkpoints under FP32 reach 1.2464. Scoring an FP32
    # checkpoint in bf16 likewise shifts it 1.2458 -> 1.3619 (LSF 103283 vs
    # 103280). The BLOCK-DIFFUSION SSM arms are unaffected: they run through
    # `_forward_pass_bissm`, which never reaches this wrapper, and re-scoring
    # them across the change moves BiSSM by 1.2e-4 and uSSM-BD by 2.0e-5.
    #
    # WARNING -- this wrapper is NOT symmetric across backbones. `models/dit.py`
    # re-opens a bf16 autocast around its own block loop and output layer, so
    # the Transformer only runs its embedding, rotary and logit tail in FP32;
    # its blocks are bf16 either way. The SSM stack has no such re-entry, so
    # `algo=ar backbone=ussm` -- the one path that reaches this wrapper without
    # going through `_forward_pass_bissm` (see `_loss`) -- executes ENTIRELY in
    # FP32, at the H200's ~67 TFLOP/s FP32 ceiling instead of the bf16 tensor
    # cores. That is the bulk of uSSM-AR's 3.09x training-time gap against the
    # Transformer AR arm and is a measurement artifact, not an architecture
    # result. Fixing it means mirroring dit.py's inner bf16 autocast in the SSM
    # layer stack, which changes uSSM-AR's training precision and so requires
    # re-running that arm.
    #
    # If this is ever relaxed, it must be an explicit, benchmarked flag --
    # silently switching it re-baselines every Transformer row.
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name == 'bd3lm':
        bb_kwargs = dict(store_kv=store_kv, sample_mode=sample_mode)
        if coarse_ablate is not None:   # only the dual backbone accepts this
          bb_kwargs['coarse_ablate'] = coarse_ablate
        logits = self.backbone(x, sigma, **bb_kwargs)
      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn:
      x = x[:, :self.config.model.length]
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                      xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                        xt=x,
                                        sigma=sigma)
    return logits
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    loss_val = losses.loss.item()
    self.log(name='trainer/loss',
             value=loss_val,
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    self._log_train_telemetry(loss_val, batch['input_ids'].shape[0])
    return losses.loss

  def _log_train_telemetry(self, loss_val, micro_batch_size):
    """Log cluster-wide throughput/compute/quality telemetry, multi-GPU aware
    (scaled by accumulate_grad_batches * world_size). FLOPs per sequence:
    2*N*tokens (linear) + 4*n_layers*pairs*d (attention over the block-diffusion
    mask, ~L^2 attended pairs for cross-attn), times 3 for the backward pass.
    FlopCounterMode can't see the compiled flex kernel, hence the estimate."""
    if not hasattr(self, '_flops_per_micro'):
      L = self.config.model.length
      d = self.config.model.hidden_size
      n_layers = self.config.model.n_blocks
      n_params = sum(p.numel() for p in self.backbone.parameters())
      seq = 2 * L if self.cross_attn else L  # backbone sees [x_t; x_0] for cross-attn
      pairs = (L * L + 2 * L * self.block_size) if self.cross_attn \
          else (seq * (seq + 1) // 2)
      fwd = 2 * n_params * seq + 4 * n_layers * pairs * d
      self._flops_per_micro = 3 * fwd * micro_batch_size
      self._tokens_per_micro = micro_batch_size * L  # loss tokens per sequence
      self._flops_t0 = time.time()
    # bits per base: the loss is a NELBO upper bound on NLL (nats/token);
    # for ACGT, uniform random is 2.0, so lower is better.
    self.log('trainer/train_bpb', loss_val / 0.6931471805599453,
             on_step=True, on_epoch=False, sync_dist=True)
    # train NLL and PPL, per step. `self.metrics.train_nlls` has been updated
    # every step since forever and logged NOWHERE -- reset at epoch start,
    # accumulated, discarded. So the validation curves (valid/nll, valid/ppl,
    # valid/bpd) had no train-side counterpart at all.
    # Taken from loss_val rather than from that metric because the metric is a
    # RUNNING mean over the epoch: it lags, and at one epoch per run it would
    # flatten the whole curve. `losses.loss` is already the token-weighted mean
    # NLL (diffusion.py:1358), so this is the instantaneous value.
    self.log('trainer/train_nll', loss_val,
             on_step=True, on_epoch=False, sync_dist=True)
    # exp() of a NELBO for the BD arms, so this is an UPPER bound on perplexity
    # rather than perplexity -- the same caveat that applies to valid/ppl.
    # Clamped because an early diverging step overflows float32 above ~88 and a
    # single inf poisons the axis of every plot drawn from this CSV.
    self.log('trainer/train_ppl', math.exp(min(loss_val, 20.0)),
             on_step=True, on_epoch=False, sync_dist=True)
    scale = self.trainer.accumulate_grad_batches * self.trainer.world_size
    steps = max(self.trainer.global_step, 1)
    elapsed = max(time.time() - self._flops_t0, 1e-6)
    total_flop = self._flops_per_micro * scale * steps
    total_tok = self._tokens_per_micro * scale * steps
    log = lambda k, v: self.log(k, v, on_step=True, on_epoch=False,
                                rank_zero_only=True)
    log('trainer/total_pflop', total_flop / 1e15)
    log('trainer/pflop_per_s', total_flop / elapsed / 1e15)
    log('trainer/total_gtokens', total_tok / 1e9)
    log('trainer/tokens_per_s', total_tok / elapsed)
    if torch.cuda.is_available():
      log('trainer/gpu_mem_gb', torch.cuda.max_memory_allocated() / 2**30)
    # Live adaLN gate magnitudes for the dual stream (cheap: gates are weight-
    # constants under time_conditioning=false). gate_cross / gate_cross_over_self
    # track whether the COARSE (long-range) pathway is engaging or atrophying.
    if hasattr(self.backbone, 'gate_stats'):
      for k, v in self.backbone.gate_stats().items():
        log(f'gates/{k}', v)
    if getattr(self, '_last_l_use', None) is not None:
      log('trainer/l_use', self._last_l_use)

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    if self.var_min:
      # How many validation batches feed the clipped-schedule variance search.
      #
      # THIS BOUNDS THE DOMINANT COST OF VALIDATION FOR A BD ARM. The loop
      # below runs a SEPARATE forward pass per (eps_min, eps_max) window per
      # batch, and clip_search_widths=[0.3..0.9] at delta 0.05 makes ~64
      # windows. metrics.reset() calls init_valid_vars() at every
      # on_validation_epoch_start, so the accumulation counter resets EVERY
      # validation -- the hardcoded 100 was not a one-off warmup cost, it was
      # 100 x 64 = 6,400 extra forwards on every single validation. Measured at
      # ~10 minutes per validation, which over 544 validations is ~90 hours,
      # against ~12 hours for the same model with the search off.
      #
      # 100 batches is far more than the variance estimate needs: at batch 4
      # and 32 blocks per sequence, even 8 batches gives 1,024 block NELBOs per
      # window. Default stays 100 so existing runs reproduce exactly.
      clip_batches = int(getattr(
        self.config.algo, 'clip_search_batches', 100) or 100)
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
          # compute and record nelbo
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
          losses = Loss(
            nlls=losses_clip.nlls.clone(),
            token_mask=losses_clip.token_mask,
            loss=losses_clip.loss.clone())
        elif len(self.metrics.valid_vars[noise_clip_start]) < clip_batches:
          # elbo from clipped schedule (biased estimate)
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
        if len(self.metrics.valid_vars[noise_clip_start]) < clip_batches:
          # only report variance over `clip_search_batches` batches
          nlls = losses_clip.nlls
          self.metrics.valid_vars[noise_clip_start].append(
            nlls.reshape(
              nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt
  
  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(
      self, x, t, dt, p_x0=None, first_hitting=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if self.config.sampling.kv_cache:
        p_x0 = self.forward(x[:, -self.block_size:],
                        sigma_t,
                        sample_mode=True).to(torch.float64)
      else:   
        p_x0 = self.forward(x,
                          sigma_t,
                          sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._restrict_generation_probs(
        p_x0, current_tokens=x[:, -self.block_size:])
      p_x0 = self._nucleus_sample(p_x0)

    if first_hitting is None:
      first_hitting = self.config.sampling.first_hitting
    if first_hitting:
      x_block = _sample_categorical(p_x0)
      # randomly and uniformly select an index in the block (among masked tokens)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # compute kv cache if all tokens in a block are sampled
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new

  @torch.no_grad()
  def _ar_sampler(self, bsz, seqlen=None, context_len=None):
    """`context_len=None` means the model's full context.

    This defaulted to a bare 1024, so every AR sample -- DiT and uSSM alike --
    was conditioned on at most the last 1024 tokens regardless of
    config.model.length, which was available and ignored. Pass an explicit
    value only if you deliberately want a shorter window.
    """
    if context_len is None:
      context_len = int(self.config.model.length)
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    if seqlen is None:
      seqlen = self.num_tokens
    # The first position is BOS only if training used per-window special
    # tokens. DNA runs without BOS use an alphabet token as their seed.
    x = self._initialize_ar_tokens(bsz, seqlen)
    num_pred_tokens = seqlen - 1
    for i in tqdm(range(num_pred_tokens)):
      # Need one Gumbel draw per possible next token. Keeping this per-step
      # avoids allocating a sequence-length-by-vocabulary noise tensor.
      noise = (torch.distributions.Gumbel(0, 1)
               .sample((bsz, self.vocab_size))
               .to(self.device))
      next_logits = self.forward(
        x[:, :i + 1][:, -context_len:],
        None,
        store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)

      next_probs = self._restrict_generation_probs(next_logits.exp())
      next_logits = self._nucleus_sample(next_probs).log()
      y = (next_logits[:, -1] + noise).argmax(-1)
      x[:, i + 1] = y
      if self.config.sampling.var_length:
        stop, x = self._check_stop_conds(x)
        if stop:
          break
    return x
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.sampler == 'semi_ar' and seqlen % self.block_size:
      raise ValueError(
        f'seqlen ({seqlen}) must be divisible by block_size '
        f'({self.block_size}) for semi-AR sampling')
    samples = []
    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(
            batch_size_per_gpu, seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.gen_nfes.append(seqlen)
      return self._decode_sample_batches(samples)
    if self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=(seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          num_tries += 1
          if num_tries > 10 and sample_i is None:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    return self._decode_sample_batches(samples)

  def _decode_sample_batches(self, sample_batches):
    # Separate variable-length calls need not share the same truncated tensor
    # width, so decode each rectangular batch before combining Python strings.
    if self.config.sampling.var_length:
      decoded = []
      for batch in sample_batches:
        decoded.extend(self._decode_samples(batch))
      return decoded
    return self._decode_samples(torch.cat(sample_batches, dim=0))

  def _decode_samples(self, samples):
    is_dna = hasattr(self.tokenizer, 'generation_token_ids')
    return self.tokenizer.batch_decode(
      samples,
      skip_special_tokens=is_dna,
      clean_up_tokenization_spaces=False)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self.metrics.record_generative_perplexity(
      samples,
      self.config.model.length,
      self.config.loader.eval_batch_size,
      self.device)
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return self._restrict_generation_probs(
        model_output.exp(), current_tokens=x)
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._restrict_generation_probs(
      model_output.exp(), current_tokens=x)
    model_output = self._nucleus_sample(model_output)
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)


  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  @staticmethod
  def _eps_host(value, mirror):
    """Host-side value of a sampling-eps bound, for CONTROL FLOW only.

    `self.sampling_eps_{min,max}` are 0-d buffers, so `if eps_max >= 1` costs a
    device->host sync per call. `mirror` is the host copy maintained at every
    write site (`__init__`, `on_load_checkpoint`, `_clipped_schedule_search`),
    so the branch can be taken without draining the launch queue.

    The tensors remain the source of truth for the ARITHMETIC -- nothing here
    feeds a value into a computation, so no float32/float64 rounding of the
    bound can leak into `t`.
    """
    if mirror is not None:
      return mirror
    if value is None or not torch.is_tensor(value):
      return value
    return value.item()

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max,
      block_size=None, eps_min_host=None, eps_max_host=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    eps_max_h = self._eps_host(sampling_eps_max, eps_max_host)
    eps_min_h = self._eps_host(sampling_eps_min, eps_min_host)
    if eps_max_h >= 1 and eps_min_h >= 1:
      return torch.ones_like(t)
    # Deliberately still the TENSOR operands: switching to the host floats
    # would move `(max - min)` from float32 to float64-then-rounded and change
    # `t` in the last bit.
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _maybe_rc_augment(self, x0, attention_mask):
    """Per-row reverse-complement augmentation of a training batch.

    Applied to the *whole contiguous window* before ``_maybe_sub_sample``: for
    ``parameterization == 'ar'`` that method builds ``x0[:, :-1]`` /
    ``x0[:, 1:]``, so augmenting afterwards would misalign the AR targets by
    one position.  Training only -- validation NLL stays un-augmented so the
    numbers remain comparable across runs.

    The complement map fixes ``[MASK]``, so this commutes with the absorbing
    corruption in ``q_xt``; it also fixes ``[EOS]``, which the DNA corpora do
    emit at record boundaries (``configs/data/carbon-prokaryote.yaml:9``
    ``insert_train_eos: True``), so under ``rc`` an ``[EOS]`` stops being a
    terminator and becomes an initiator for the next contig.  That affects
    roughly one token per source record and is documented rather than fixed;
    swapping BOS/EOS would be the honest complement map for those two ids.
    """
    if not self.training or self.rc_augment_probability <= 0:
      return x0, attention_mask
    coin = torch.rand(
      x0.shape[0], device=x0.device) < self.rc_augment_probability
    flipped = self._complement_token_ids[torch.flip(x0, dims=(1,))]
    x0 = torch.where(coin[:, None], flipped, x0)
    attention_mask = torch.where(
      coin[:, None], torch.flip(attention_mask, dims=(1,)), attention_mask)
    return x0, attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None,
                              sampling_eps_max=None, eps_min_host=None,
                              eps_max_host=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max,
                         eps_min_host=eps_min_host,
                         eps_max_host=eps_max_host)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) # used for sedd

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    eps_min_h = self._eps_host(sampling_eps_min, eps_min_host)
    if eps_min_h is not None and eps_min_h > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    xt = self._preserve_observed_bos(xt, x0)

    if self.config.algo.backbone in {'bissm', 'ussm'}:
      return self._forward_pass_bissm(
        x0=x0, xt=xt, p=p, loss_scale=loss_scale)
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        model_output, sigma, xt, x0)

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta

    # L_use (P2 incentive): margin-ranking term that rewards the TRUE coarse memory
    # for beating a coarse-SHUFFLED counterfactual on masked tokens, so the
    # long-range route earns gradient the plain diffusion loss never gives it.
    # Self-targeting: where coarse is uninformative (random background) both CEs
    # match and softplus saturates with ~no gradient; only tokens the coarse route
    # can help (cross-block echoes) drive it. Second forward => ~2x step cost.
    if self.training and self.l_use_weight > 0 and self.cross_attn:
      shuf_output = self.forward(x_input, sigma=sigma, coarse_ablate='shuffle')
      log_p_shuf = torch.gather(
        input=shuf_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
      masked = (xt == self.mask_index).to(loss.dtype)
      l_use = F.softplus(
        (-log_p_theta) - (-log_p_shuf) + self.l_use_margin) * masked
      loss = loss + self.l_use_weight * l_use
      self._last_l_use = (l_use.sum() / masked.sum().clamp(min=1)).detach()
    return loss

  def _bissm_use_right_flank(self, device):
    right_probability = float(
      self.config.model.get('right_flank_probability', 0.0))
    if right_probability < 0 or right_probability > 1:
      raise ValueError("model.right_flank_probability must be in [0, 1]")
    return bool(
      right_probability == 1.0
      or (right_probability > 0.0
          and torch.rand((), device=device) < right_probability))

  def _forward_pass_bissm(self, x0, xt, p, loss_scale):
    """Block-diffusion objective evaluated through recurrent boundary caches.

    ``model.active_blocks='all'`` (default) supervises every block in one
    step, matching the Transformer's ``[x_t; x_0]`` all-block objective: one
    clean scan yields the boundary state entering each block, and the blocks
    are folded into the batch dimension so a single batched call denoises all
    of them. ``model.active_blocks='one'`` keeps the original estimator, which
    samples one block and rescales by the block count -- unbiased for the loss
    value, but it only ever puts gradient on 1/num_blocks of the tokens.
    """
    sequence_length = x0.shape[1]
    if sequence_length % self.block_size:
      raise ValueError(
        f"BiSSM requires sequence length ({sequence_length}) divisible by "
        f"block size ({self.block_size})")
    num_blocks = sequence_length // self.block_size

    active_blocks = str(self.config.model.get('active_blocks', 'all'))
    if active_blocks not in {'all', 'one'}:
      raise ValueError(
        f"model.active_blocks must be 'all' or 'one', got {active_blocks!r}")
    if active_blocks == 'all':
      return self._forward_pass_bissm_all_blocks(
        x0=x0, xt=xt, p=p, loss_scale=loss_scale, num_blocks=num_blocks)
    active_block = int(torch.randint(num_blocks, (), device=x0.device).item())
    start = active_block * self.block_size
    end = start + self.block_size

    clean_prefix = x0[:, :start]
    noisy_active = xt[:, start:end]
    clean_target = x0[:, start:end]
    with self._model_autocast_context():
      left_cache = self.backbone.prefill_left(clean_prefix)

    right_cache = None
    use_right = self._bissm_use_right_flank(x0.device)
    if use_right:
      # The target block is excluded by construction. An empty suffix at the
      # last block is legal and reduces to the de-novo reverse initial state.
      with self._model_autocast_context():
        right_cache = self.backbone.prefill_right(x0[:, end:])

    sigma_active = self._sigma_from_p(p[:, start:start + 1]).squeeze(-1)
    with self._model_autocast_context():
      logits = self.backbone.forward_active(
        noisy_active,
        sigma_active,
        left_cache=left_cache,
        right_cache=right_cache)
    log_scores = self._subs_parameterization(logits, noisy_active)
    log_p_theta = torch.gather(
      input=log_scores,
      dim=-1,
      index=clean_target[:, :, None]).squeeze(-1)
    active_loss = loss_scale[:, start:end] * log_p_theta

    loss = torch.zeros_like(loss_scale)
    loss[:, start:end] = active_loss * num_blocks
    self._last_active_block = active_block
    self._last_right_flank = use_right
    return loss

  def _forward_pass_bissm_all_blocks(self, x0, xt, p, loss_scale, num_blocks):
    """Supervise every block per step through folded boundary caches.

    Cost is one clean scan for the caches plus one batched active scan over
    the same number of tokens, i.e. a small constant factor over the single-
    block estimator, for ``num_blocks`` times as many supervised tokens.
    """
    batch_size, sequence_length = x0.shape
    block_size = self.block_size

    with self._model_autocast_context():
      left_cache = self.backbone.prefill_left_boundaries_stacked(
        x0, block_size)

    right_cache = None
    use_right = self._bissm_use_right_flank(x0.device)
    if use_right:
      # Entry i covers only the clean suffix strictly after block i, so no
      # block ever sees its own clean target.
      with self._model_autocast_context():
        right_cache = self.backbone.prefill_right_boundaries_stacked(
          x0, block_size)

    folded = (batch_size * num_blocks, block_size)
    noisy_blocks = xt.reshape(*folded)
    clean_blocks = x0.reshape(*folded)
    # One noise level per (sequence, block); take it at each block's first
    # position, matching `_sample_t`'s per-block schedule.
    sigma_blocks = self._sigma_from_p(
      p[:, ::block_size]).reshape(batch_size * num_blocks)

    with self._model_autocast_context():
      logits = self.backbone.forward_active(
        noisy_blocks,
        sigma_blocks,
        left_cache=left_cache,
        right_cache=right_cache)
    log_scores = self._subs_parameterization(logits, noisy_blocks)
    log_p_theta = torch.gather(
      input=log_scores,
      dim=-1,
      index=clean_blocks[:, :, None]).squeeze(-1)

    self._last_active_block = None
    self._last_right_flank = use_right
    return loss_scale * log_p_theta.reshape(batch_size, sequence_length)

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    eps_min_host = eps_max_host = None
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
      # Host mirrors of the two 0-d buffers, so the branches downstream never
      # read the device. Maintained wherever the buffers are written.
      eps_min_host = self._sampling_eps_min_host
      eps_max_host = self._sampling_eps_max_host
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    x0, attention_mask = self._maybe_rc_augment(x0, attention_mask)
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    attention_mask = attention_mask.clone()
    if self.parameterization == 'ar':
      output = self.forward(input_tokens, None)
      loss = - output.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(
        input_tokens,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max,
        eps_min_host=eps_min_host,
        eps_max_host=eps_max_host,)

    # Branchless for the same reason as `_preserve_observed_bos`: rows without
    # BOS are written back their own value, so the mask is bitwise unchanged.
    if self._bos_is_possible():
      bos_rows = self._bos_rows(input_tokens)
      attention_mask[:, 0] = torch.where(
        bos_rows,
        torch.zeros_like(attention_mask[:, 0]),
        attention_mask[:, 0])

    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)
      self._sampling_eps_min_host = float(sampling_eps_min_best)
      self._sampling_eps_max_host = float(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def _analytic_sampler(
    self, n_samples, num_steps, seqlen, eps=1e-5): 
    x = self._initialize_diffusion_tokens(n_samples, seqlen)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    for i in tqdm(range(num_steps), desc='step'):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      x = self._analytic_update(x=x, t=t, dt=dt)
    # denoising step 
    t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                  device=self.device)
    x = self._denoiser_update(x=x, t=t)
    
    _, x = self._check_stop_conds(x)
    return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=None):
    """`context_size=None` means the model's full context.

    Also defaulted to a bare 1024. This is the DEFAULT generation path --
    configs/config.yaml sets sampling.kv_cache: False, and with the cache off
    the model sees only this window for the whole run.
    """
    if context_size is None:
      context_size = int(self.config.model.length)
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
          
    mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
    if mdlm_semi_ar:
      # sliding window of length 512 for mdlm semi-ar decoding
      num_strides = self.config.model.length // 512
      num_strides -= 1

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)
    
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      # sample next block
      if stride_num == 0:
        x_accum = self._initialize_diffusion_tokens(
          n_samples, self.block_size)
      else:
        if mdlm_semi_ar:
          x = self._sample_prior(n_samples, 512).to(self.device)
        else:
          x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      # Snap the window start down to a block boundary so (a) the fine block grid
      # stays aligned with the global block structure and (b) the coarse k-mer
      # boundaries match training (start_idx % k_coarse == 0, since the dual
      # backbone requires block_size % k_coarse == 0). No-op when block_size
      # divides context_size (e.g. all single-block / power-of-two configs).
      start_idx -= start_idx % self.block_size
      fwd_idx = torch.arange(start_idx, end_idx)
      if mdlm_semi_ar and stride_num > 0: # MDLM
        fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
              
        elif not self.config.sampling.first_hitting:
          t = timesteps[i]

        p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      # Fixed-length sampling always completes the requested grid. For
      # variable length, EOS is checked after every completed block.
      if self.config.sampling.var_length:
        stop, x_accum = self._check_stop_conds(x_accum)
        if stop:
          break
    return x_accum, sampling_steps

  @torch.no_grad()
  def sample_infill_ca(
      self,
      left_context: torch.Tensor,
      right_context: torch.Tensor,
      gap_length: int,
      num_steps: int,
  ) -> torch.Tensor:
    """C-a infilling with fixed right belief and advancing left belief.

    Returns ``[left_context; generated_gap; right_context]``. The first
    implementation requires a whole number of diffusion blocks so neither
    padding nor observed target tokens can accidentally enter a boundary
    cache.
    """
    if self.config.algo.backbone != 'bissm':
      raise ValueError("sample_infill_ca requires algo.backbone=bissm")
    if left_context.ndim != 2 or right_context.ndim != 2:
      raise ValueError("left_context and right_context must be [batch, length]")
    if left_context.shape[0] != right_context.shape[0]:
      raise ValueError("left and right contexts must use the same batch size")
    if gap_length <= 0 or gap_length % self.block_size:
      raise ValueError(
        f"gap_length must be a positive multiple of block_size={self.block_size}")
    if num_steps <= 0:
      raise ValueError("num_steps must be positive")

    batch_size = left_context.shape[0]
    self.backbone.reset_kv_cache(eval_batch_size=batch_size)
    self.backbone._sampling_left_cache = self.backbone.prefill_left(
      left_context, detach=True)
    self.backbone.prepare_right_cache(right_context)
    fixed_right = self.backbone._sampling_right_cache.clone()

    generated_blocks = []
    dt = 1.0 / num_steps
    for _ in range(gap_length // self.block_size):
      active = self._sample_prior(batch_size, self.block_size).to(self.device)
      p_x0_cache = None
      for step in range(num_steps):
        if self.mask_index not in active:
          break
        # Use the fixed ancestral grid for batched C-a infilling. The optional
        # first-hitting sampler selects individual mask positions and is a
        # de-novo speed heuristic, not part of the C-a conditioning contract.
        t_value = 1.0 - step * dt
        t = torch.full(
          (batch_size, 1), t_value,
          device=self.device, dtype=self.dtype)
        p_x0_cache, active = self._ddpm_caching_update(
          x=active,
          t=t,
          dt=dt,
          p_x0=p_x0_cache,
          first_hitting=False)
      if self.mask_index in active:
        raise RuntimeError(
          "C-a sampler finished with masked tokens; increase num_steps or "
          "check the noise schedule")
      generated_blocks.append(active)
      # Committing the clean active block must not mutate the fixed right cache.
      for actual, expected in zip(
          self.backbone._sampling_right_cache.states, fixed_right.states):
        if not torch.equal(actual.conv, expected.conv) \
            or not torch.equal(actual.ssm, expected.ssm):
          raise RuntimeError("The fixed C-a right cache was mutated")

    gap = torch.cat(generated_blocks, dim=1)
    return torch.cat((left_context, gap, right_context), dim=1)

  def sample_infill_refined(
      self,
      left_context: torch.Tensor,
      right_context: torch.Tensor,
      gap_length: int,
      num_steps: int,
      passes: int = 2,
  ) -> torch.Tensor:
    """C-a infilling with a POSITIONALLY CORRECT right cache.

    THE BUG THIS FIXES. Training conditions the active block on

        right_cache = prefill_right(x0[:, end:])

    -- everything to the right of the block, which for an interior block
    INCLUDES the interior blocks after it. `sample_infill_ca` instead prefills
    the right cache from the committed flank alone and freezes it, so when the
    gap spans n blocks the right belief for block i starts (n-1-i) blocks too
    far away. At n=1 the two agree exactly; at n=8 block 0's right belief is
    displaced by 1,792 nt.

    That is not a subtle inefficiency. Measured on 24 loci at 16,384 nt
    (2026-09-02), the fraction of loci whose AlphaGenome RNA-seq MSE exceeds 10
    goes 0.00, 0.00, 0.33, 0.62 as the gap grows 256 -> 512 -> 1024 -> 2048,
    while the same model with an EMPTY right cache stays near 0.12. A wrong
    flank (`mismatch`) was as good as the true one, which is the signature of a
    cache that carries no usable information -- exactly what a displaced belief
    would look like.

    THE ATTEMPTED FIX. Bootstrap the interior left-to-right (the old
    behaviour), then sweep it `passes` times; on each sweep every block is
    re-denoised with

        left_cache  = prefill_left(left_context + blocks[:i])
        right_cache = prefill_right(blocks[i+1:] + right_context)

    which is the training contract exactly. Block-wise Gibbs sampling over the
    interior, costing `passes * n` extra prefills.

    *** IT DOES NOT WORK. DO NOT USE passes>0 WITHOUT READING THIS. ***

    Measured against the displaced sampler on identical loci and seed, 24 loci
    at 16,384 nt, passes=2 (2026-09-02). Median AlphaGenome RNA-seq MSE and
    the fraction of loci above 10:

        gap 1024   ca        2.6 / 0.33   ->    43.9 / 0.67
        gap 2048   ca       72.9 / 0.62   -> 1,972.4 / 0.79
        gap 2048   denovo    1.2 / 0.12   -> 4,675.2 / 0.83

    Every condition got worse, and `denovo` -- which the sweeps degraded from
    0.12 to 0.83 -- is the diagnostic case. Its `right_context` is empty, so
    refinement gave it a right cache built ENTIRELY from the model's own
    generated blocks. That is the mechanism: re-denoising a block against
    self-generated neighbours is a Gibbs chain whose conditionals are not
    accurate enough to be stable, so each sweep amplifies the previous sweep's
    errors instead of correcting them.

    So the positional displacement documented above is real, but it is NOT the
    cause of the multi-block pathology; correcting it while feeding the model
    its own output is strictly worse than leaving it displaced. The best
    configuration measured so far remains a single left-to-right pass with NO
    right cache at all (`sample_infill_ca` with an empty right context), which
    is also the configuration that conditions least on non-local information.

    Kept in the tree because the negative result is worth more than the code:
    it rules out the obvious mechanical explanation and points at the model's
    conditionals rather than the sampler's bookkeeping.

    Returns ``[left_context; generated_gap; right_context]``.
    """
    if self.config.algo.backbone != 'bissm':
      raise ValueError("sample_infill_refined requires algo.backbone=bissm")
    if gap_length <= 0 or gap_length % self.block_size:
      raise ValueError(
        f"gap_length must be a positive multiple of block_size={self.block_size}")
    if num_steps <= 0 or passes < 0:
      raise ValueError("num_steps must be positive and passes non-negative")

    batch_size = left_context.shape[0]
    n_blocks = gap_length // self.block_size
    dt = 1.0 / num_steps

    def denoise(left_cache, right_cache):
      """One block, from the prior, under explicitly supplied caches."""
      self.backbone._sampling_left_cache = left_cache
      self.backbone._sampling_right_cache = right_cache
      active = self._sample_prior(batch_size, self.block_size).to(self.device)
      p_x0 = None
      for step in range(num_steps):
        if self.mask_index not in active:
          break
        t = torch.full((batch_size, 1), 1.0 - step * dt,
                       device=self.device, dtype=self.dtype)
        # first_hitting is a de-novo speed heuristic and is not part of the
        # C-a conditioning contract; keep the fixed ancestral grid.
        p_x0, active = self._ddpm_caching_update(
          x=active, t=t, dt=dt, p_x0=p_x0, first_hitting=False)
      if self.mask_index in active:
        raise RuntimeError(
          "C-a sampler finished with masked tokens; increase num_steps")
      return active

    def left_of(blocks, i):
      parts = [left_context] + blocks[:i]
      return self.backbone.prefill_left(torch.cat(parts, dim=1), detach=True)

    def right_of(blocks, i):
      parts = blocks[i + 1:] + [right_context]
      return self.backbone.prefill_right(torch.cat(parts, dim=1))

    self.backbone.reset_kv_cache(eval_batch_size=batch_size)

    # Bootstrap: left-to-right, right cache = flank only. Identical to
    # sample_infill_ca, and the starting point the sweeps then correct.
    blocks: list[torch.Tensor] = []
    flank_right = self.backbone.prefill_right(right_context)
    for i in range(n_blocks):
      blocks.append(denoise(left_of(blocks, i), flank_right))

    # Sweeps: every block now sees the true training-time context on both sides.
    for _ in range(passes):
      for i in range(n_blocks):
        blocks[i] = denoise(left_of(blocks, i), right_of(blocks, i))

    gap = torch.cat(blocks, dim=1)
    return torch.cat((left_context, gap, right_context), dim=1)

  def _check_stop_conds(self, x):
    """Stop a variable-length batch once every row has emitted EOS.

    Rows that finish early are padded immediately after their first EOS. This
    keeps the tensor rectangular while preventing later sampled tokens from
    leaking into decoded DNA. Fixed-length generation never rejects a sample
    for low empirical entropy: the old threshold (4 nats) exceeded even the
    maximum entropy of the complete 13-token DNA vocabulary.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    if not self.config.sampling.var_length:
      return False, x
    eos_id = self.tokenizer.eos_token_id
    if eos_id is None:
      return False, x

    eos = x.eq(eos_id)
    has_eos = eos.any(dim=1)
    if not has_eos.any():
      return False, x

    first_eos = torch.where(
      has_eos,
      eos.to(torch.int64).argmax(dim=1),
      torch.full((x.shape[0],), x.shape[1] - 1, device=x.device))
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    after_eos = has_eos[:, None] & (positions > first_eos[:, None])
    x = x.clone()
    pad_id = self.tokenizer.pad_token_id
    if pad_id is None:
      pad_id = eos_id
    x[after_eos] = pad_id

    stop = bool(has_eos.all())
    if stop:
      x = x[:, :int(first_eos.max().item()) + 1]
    return stop, x
