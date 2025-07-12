# Based on score_sde_pytorch
# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def optimize_memory():
  """Optimise l'utilisation mémoire GPU."""
  if torch.cuda.is_available():
      # Vider le cache
      torch.cuda.empty_cache()
      
      # Collecter les déchets
      import gc
      gc.collect()
      
      # Pour les GPUs avec beaucoup de VRAM (H100 94GB, GH200 96GB)
      if torch.cuda.get_device_properties(0).total_memory > 80 * 1024**3:
          # Augmenter la fraction de mémoire réservée
          torch.cuda.set_per_process_memory_fraction(0.95)