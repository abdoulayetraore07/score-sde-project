## Modified from https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
## Will only work if https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py is in the same folder
## I removed all the stuff we do not need

#### Defaults
# config.sampling.method='adaptive', choices=['euler_maruyama_adaptive','adaptive']
# config.sampling.sampling_h_init=1e-2
# config.sampling.sampling_reltol=1e-2
# config.sampling.sampling_safety=0.9
# config.sampling.sampling_exp=0.9

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import torch
import abc
import sde_lib
import numpy as np
from datetime import datetime

_PREDICTORS = {}

def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_sampling_fn(config, sde, shape, eps, device):
  """Create a sampling function.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  predictor = get_predictor(config.sampling.method)

  sampling_fn = get_pc_sampler(sde = sde,
                               shape = shape,
                               predictor = predictor,
                               denoise = True,
                               eps = eps,
                               device = device,
                               abstol = config.sampling.sampling_abstol, 
                               reltol = config.sampling.sampling_reltol, 
                               safety = config.sampling.sampling_safety, 
                               exp = config.sampling.sampling_exp,
                               adaptive = config.sampling.method == "adaptive",
                               h_init = config.sampling.sampling_h_init)

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm - INTERFACE UNIFIÉE."""

  def __init__(self, sde, score_fn, probability_flow=False, **kwargs):
    # Interface standard compatible avec sampling.py
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.probability_flow = probability_flow
    
    # Paramètres spécifiques au fast sampling (extraits des kwargs)
    self.shape = kwargs.get('shape', None)
    self.eps = kwargs.get('eps', 1e-3)
    self.abstol = kwargs.get('abstol', 1e-2)
    self.reltol = kwargs.get('reltol', 1e-2)
    self.safety = kwargs.get('safety', 0.9)
    self.exp = kwargs.get('exp', 0.9)
    
    # Calculer le reverse SDE
    self.rsde = sde.reverse(score_fn, probability_flow)

  @abc.abstractmethod
  def update_fn(self, x, t, h=None, x_prev=None):
    """Update function - compatible avec les deux interfaces."""
    pass


@register_predictor(name='euler_maruyama_adaptive')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, **kwargs):
    super().__init__(sde, score_fn, probability_flow, **kwargs)

  def update_fn(self, x, t, h=None, x_prev=None):
    # AUCUNE FORMULE CHANGÉE - juste adaptation de l'interface
    if h is None:
        # Mode compatible sampling.py - utilise dt fixe
        dt = -1. / self.sde.N
        h = -dt
    
    # FORMULES ORIGINALES EXACTES - pas une virgule changée
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x - drift * h
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(h) * z
    return x, x_mean


@register_predictor(name='adaptive')
class AdaptivePredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, **kwargs):
    super().__init__(sde, score_fn, probability_flow, **kwargs)
    
    # Configuration adaptive - FORMULES EXACTES
    self.h_min = 1e-10
    self.t = sde.T
    self.error_use_prev = True
    
    # Calculer self.n à partir de shape ou valeur par défaut
    if self.shape is None:
        # Valeur par défaut intelligente - sera mise à jour au premier appel
        self.n = None
        self._shape_detected = False
    else:
        if isinstance(self.shape, (list, tuple)) and len(self.shape) >= 4:
            self.n = self.shape[1] * self.shape[2] * self.shape[3]
            self._shape_detected = True
        else:
            self.n = None
            self._shape_detected = False
    
    # FORMULE EXACTE norm_fn - pas une virgule changée
    def norm_fn(x):
        if self.n is None:
            # Auto-detect shape au premier appel
            if len(x.shape) == 4:  # [batch, channels, height, width]
                self.n = x.shape[1] * x.shape[2] * x.shape[3]
                self._shape_detected = True
            else:
                self.n = np.prod(x.shape[1:])  # fallback
        return torch.sqrt(torch.sum((x)**2, dim=(1,2,3), keepdim=True)/self.n)
    
    self.norm_fn = norm_fn

  def update_fn(self, x, t, h=None, x_prev=None):
    # Mode compatibility check
    if h is None:
        # Mode sampling.py - conversion automatique
        dt = -1. / self.sde.N  
        h = torch.ones(x.shape[0], device=x.device) * (-dt)
        if x_prev is None:
            x_prev = x
    
    # Auto-detect shape si pas encore fait
    if not getattr(self, '_shape_detected', False):
        if len(x.shape) == 4:
            self.n = x.shape[1] * x.shape[2] * x.shape[3]
            self._shape_detected = True
    
    # TOUTES LES FORMULES EXACTES - pas une virgule changée
    # ===================================================
    my_rsde = self.rsde.sde

    h_ = h[:, None, None, None] if h.dim() == 1 else h
    t_ = t[:, None, None, None] if t.dim() == 1 else t
    z = torch.randn_like(x)
    drift, diffusion = my_rsde(x, t)

    # Heun's method for SDE - FORMULES EXACTES
    K1_mean = -h_ * drift
    K1 = K1_mean + diffusion[:, None, None, None] * torch.sqrt(h_) * z

    drift_Heun, diffusion_Heun = my_rsde(x + K1, t - h)
    K2_mean = -h_*drift_Heun
    K2 = K2_mean + diffusion_Heun[:, None, None, None] * torch.sqrt(h_) * z
    E = 1/2*(K2 - K1) # local-error between EM and Heun
    
    # Extrapolate using the Heun's method result
    x_new = x + (1/2)*(K1 + K2)
    x_check = x + K1

    # Calculating the error-control - FORMULES EXACTES
    if self.error_use_prev:
      reltol_ctl = torch.maximum(torch.abs(x_prev), torch.abs(x_check))*self.reltol
    else:
      reltol_ctl = torch.abs(x_check)*self.reltol
    err_ctl = torch.clamp(reltol_ctl, min=self.abstol)

    # Normalizing for each sample separately - FORMULES EXACTES
    E_scaled_norm = self.norm_fn(E/err_ctl)

    # Accept or reject - FORMULES EXACTES
    accept = E_scaled_norm <= torch.ones_like(E_scaled_norm)
    x_final = torch.where(accept, x_new, x)
    x_prev_final = torch.where(accept, x_check, x_prev)
    t_final = torch.where(accept, t_ - h_, t_)

    # Change the step-size - FORMULES EXACTES
    h_max = torch.clamp(t_final - self.eps, min=0)
    E_pow = torch.where(h_ == 0, h_, torch.pow(E_scaled_norm, -self.exp))
    h_new = torch.minimum(h_max, self.safety*h_*E_pow)

    # Return compatible avec les deux interfaces
    if h is None:
        # Mode sampling.py - retour simple
        return x_final, x_final  # x_mean = x pour compatibilité
    else:
        # Mode fast sampling - retour complet
        return x_final, x_prev_final, t_final.reshape((-1)), h_new.reshape((-1))


# ==============================================================================
# Adapter shared_predictor_update_fn pour supporter les deux interfaces
# ==============================================================================

def shared_predictor_update_fn(x, t, h=None, sde=None, model=None, predictor=None, 
                              x_prev=None, shape=None, eps=1e-3, abstol=1e-2, 
                              reltol=1e-2, safety=.9, exp=0.9, **kwargs):
  """Wrapper unifié qui supporte les deux interfaces."""
  
  # Construire la score function
  if hasattr(model, '__call__'):
      score_fn = model  # déjà une fonction
  else:
      # Construire depuis le modèle
      continuous = kwargs.get('continuous', True)
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  
  # Créer le predictor avec tous les paramètres
  predictor_obj = predictor(
      sde=sde, 
      score_fn=score_fn, 
      probability_flow=kwargs.get('probability_flow', False),
      shape=shape,
      eps=eps,
      abstol=abstol,
      reltol=reltol,
      safety=safety,
      exp=exp
  )
  
  # Appeler update_fn avec les bons paramètres
  return predictor_obj.update_fn(x, t, h, x_prev)


# ==============================================================================
# Adapter get_pc_sampler pour unifier les interfaces
# ==============================================================================

def get_pc_sampler(sde, shape, predictor, denoise=True, device='cuda',
                   eps=1e-3, abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9, 
                   adaptive=False, h_init=1e-2):
  """Sampler unifié qui marche avec les deux interfaces."""
  
  # Wrapper unifié pour predictor_update_fn
  predictor_update_fn = functools.partial(
      shared_predictor_update_fn,
      sde=sde,
      shape=shape,
      predictor=predictor,
      eps=eps, 
      abstol=abstol, 
      reltol=reltol, 
      safety=safety, 
      exp=exp
  )

  def pc_sampler(model, prior=None):
    """Sampler unifié."""
    with torch.no_grad():
      # Initial sample
      if prior is None:
        x = sde.prior_sampling(shape).to(device)
      else:
        x = prior.to(device)
      
      if adaptive:
          # Mode adaptatif - FORMULES EXACTES
          h = torch.ones(shape[0]).to(device) * h_init
          t = torch.ones(shape[0]).to(device) * sde.T
          x_prev = x 
          N = 0
          
          while (torch.abs(t - eps) > 1e-6).any():
            x, x_prev, t, h = predictor_update_fn(x, t, h, x_prev=x_prev, model=model)
            N = N + 1
      else:
          # Mode standard - FORMULES EXACTES  
          timesteps = np.linspace(sde.T, eps, sde.N)
          h = timesteps - np.append(timesteps, 0)[1:]
          N = sde.N - 1

          for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0]).to(device) * t
            x, x_mean = predictor_update_fn(x, vec_t, h[i], model=model)

      # Denoising final - FORMULE EXACTE
      if denoise:
        eps_t = torch.ones(shape[0]).to(device) * eps
        u, std = sde.marginal_prob(x, eps_t)
        x = x + (std[:, None, None, None] ** 2) * model(x, eps_t)
      
      return x, N + 1

  return pc_sampler if adaptive else pc_sampler