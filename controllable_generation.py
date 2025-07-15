# Based on score_sde_pytorch
# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Controllable generation for score-based generative models."""

import functools
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
import torch.nn.functional as F
from PIL import Image
import os
import logging
import random

#Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
from sampling import (
    ReverseDiffusionPredictor, 
    LangevinCorrector, 
    EulerMaruyamaPredictor,
    AncestralSamplingPredictor,
    NoneCorrector,
    NonePredictor
)


def range_sigmas_active(labels):
   
  if labels[0].item()== 2 :        # Wild
    SIGMA_LIMIT_PRED_MIN = 0.01  
    SIGMA_LIMIT_PRED_MAX = 374   #100       
    SIGMA_MAX_CLASSIFIER = 50   #50
  elif labels[0].item()== 1 :    #Dog
    SIGMA_LIMIT_PRED_MIN = 0.01  
    SIGMA_LIMIT_PRED_MAX = 374   #40     
    SIGMA_MAX_CLASSIFIER = 50   #50
  elif labels[0].item()== 0 :    #Cat
    SIGMA_LIMIT_PRED_MIN = 0.01
    SIGMA_LIMIT_PRED_MAX = 374   #100     
    SIGMA_MAX_CLASSIFIER = 50   #50

  return SIGMA_MAX_CLASSIFIER, SIGMA_LIMIT_PRED_MIN, SIGMA_LIMIT_PRED_MAX 


# ✅ SUPPRESSION de la fonction shared_predictor_update_fn dupliquée
# (On utilise celle de sampling.py qui est complète)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)


def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=True,
                     denoise=True, eps=1e-5):
  """Create a Predictor-Corrector (PC) sampler for inpainting.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

  Returns:
    A inpainting function.
  """

  # ✅ UTILISATION de la fonction shared_predictor_update_fn de sampling.py
  from sampling import shared_predictor_update_fn
  
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                           sde=sde,
                                           shape=None,  # Sera défini dans pc_inpainter
                                           model=None,  # Will be passed in sampler
                                           predictor=predictor,
                                           probability_flow=probability_flow,
                                           continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           model=None,  # Will be passed in sampler
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)

  def get_inpaint_update_fn(update_fn):
    """Modify an update function to incorporate data information for inpainting."""

    def inpaint_update_fn(model, data, mask, x, t):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        masked_data_mean, std = sde.marginal_prob(data, vec_t)
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    return inpaint_update_fn

  predictor_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(model, data, mask):
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(data.shape).to(data.device)
      masked_data_mean, std = sde.marginal_prob(data, torch.ones(data.shape[0], device=data.device) * sde.T)
      masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
      x = x * (1. - mask) + masked_data * mask

      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)
        x, x_mean = predictor_inpaint_update_fn(model, data, mask, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_inpainter


def get_pc_colorizer(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=True,
                     denoise=True, eps=1e-5):
  """Create a PC sampler for colorization.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

  Returns:
    A colorization function.
  """

  # ✅ UTILISATION de la fonction shared_predictor_update_fn de sampling.py
  from sampling import shared_predictor_update_fn
  
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                           sde=sde,
                                           shape=None,  # Sera défini dans pc_colorizer
                                           model=None,  # Will be passed in sampler
                                           predictor=predictor,
                                           probability_flow=probability_flow,
                                           continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           model=None,  # Will be passed in sampler
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)

  def get_colorize_update_fn(update_fn):
    """Modify an update function to incorporate data information for colorization."""

    def colorize_update_fn(model, gray_scale_img, x, t):
      with torch.no_grad():
        vec_t = torch.ones(gray_scale_img.shape[0], device=gray_scale_img.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        # Project to gray scale
        if x.shape[1] > 1:
          # For color images, convert to grayscale using standard weights
          weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).reshape(1, 3, 1, 1)
          luminance = torch.sum(x * weights, dim=1, keepdim=True)
          x = x - luminance + gray_scale_img
          luminance_mean = torch.sum(x_mean * weights, dim=1, keepdim=True)
          x_mean = x_mean - luminance_mean + gray_scale_img
        else:
          # For grayscale input, directly constrain the single channel
          x = gray_scale_img
          x_mean = gray_scale_img
        return x, x_mean

    return colorize_update_fn

  predictor_colorize_update_fn = get_colorize_update_fn(predictor_update_fn)
  corrector_colorize_update_fn = get_colorize_update_fn(corrector_update_fn)

  def pc_colorizer(model, gray_scale_img):
    with torch.no_grad():
      # Initial sample 
      shape = gray_scale_img.shape
      if shape[1] == 1:
        # For single channel grayscale, expand to 3 channels for colorization
        shape = (shape[0], 3, shape[2], shape[3])
        gray_scale_img = gray_scale_img.repeat(1, 3, 1, 1)
      
      x = sde.prior_sampling(shape).to(gray_scale_img.device)

      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
        x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_colorizer



def apply_standard_strategy(classifier_grad, guidance_scale):
    """Stratégie 1: Standard baseline."""
    return classifier_grad * guidance_scale


def apply_adaptive_scale_strategy(classifier_grad, current_sigma, guidance_scale, adaptive_sigma_limit):
    """Stratégie 2: Adaptive scale linear variation."""
    sigma_max = 374.0

    adaptive_sigma_limit = 0.01
    # Zone adaptative: 500.0 à 400.0 linéairement
    a, b = 400.0, 500.0
    alpha = (sigma_max - current_sigma) / (sigma_max - adaptive_sigma_limit)
    adaptive_scale = a + (b - a) * alpha
    
    return classifier_grad * adaptive_scale 


def apply_truncation_strategy(x, noise_scale, labels, classifier, guidance_scale):
    """Stratégie 3: Truncation (méthode existante)."""
    
    current_sigma = noise_scale[0].item()
    SIGMA_MAX_CLASSIFIER, SIGMA_LIMIT_PRED_MIN, SIGMA_LIMIT_PRED_MAX = range_sigmas_active(labels)
   
    
    if current_sigma < SIGMA_LIMIT_PRED_MIN or current_sigma > SIGMA_LIMIT_PRED_MAX:
        # En dehors de la zone : pas de guidance
        return torch.zeros_like(x)
    
    # Dans la zone : utiliser le classifier avec sigma tronqué
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        noise_scale_clamped = torch.clamp(noise_scale, max=SIGMA_MAX_CLASSIFIER)
        logits = classifier(x, noise_scale_clamped)
        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
        loss = target_log_probs.sum()
        grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
    
    if random.random() < 0.01 :
       print(f"Troncature from {SIGMA_LIMIT_PRED_MIN} to {SIGMA_LIMIT_PRED_MAX} with classifier_max  = {SIGMA_MAX_CLASSIFIER}")
       print(f"Guidance_scale : {guidance_scale}")
       
    return grad.detach() * guidance_scale


def get_classifier_grad_fn(classifier, guidance_strategy="truncation", guidance_scale=500.0, adaptive_sigma_limit=20.0, reference_values=None):
    """Create gradient function for noise-dependent classifier with 4 strategies."""
    
    def classifier_grad_fn(x, noise_scale, labels):
        """Compute classifier gradient w.r.t. input with selected strategy."""
        current_sigma = noise_scale[0].item()
        
        if guidance_strategy == "standard":
            # Stratégie 1: Standard
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = classifier(x, noise_scale)
                log_probs = F.log_softmax(logits, dim=1)
                target_log_probs = log_probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
                loss = target_log_probs.sum()
                grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            return apply_standard_strategy(grad.detach(), guidance_scale)
            
        elif guidance_strategy == "adaptive_scale":
            # Stratégie 2: Adaptive scale
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = classifier(x, noise_scale)
                log_probs = F.log_softmax(logits, dim=1)
                target_log_probs = log_probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
                loss = target_log_probs.sum()
                grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            return apply_adaptive_scale_strategy(grad.detach(), current_sigma, guidance_scale, adaptive_sigma_limit)
            
        elif guidance_strategy == "truncation":
            # Stratégie 3: Truncation 
            return apply_truncation_strategy(x, noise_scale, labels, classifier, guidance_scale)
                
        else:
            raise ValueError(f"Stratégie inconnue: {guidance_strategy}")
    
    return classifier_grad_fn


def get_pc_conditional_sampler(sde, classifier, shape, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=True,
                             denoise=True, eps=1e-5, device='cuda', guidance_scale=500.0, 
                             guidance_strategy="standard", adaptive_sigma_limit=20.0):
    """Create a conditional sampler with automatic interface detection."""
     
    # Construire la fonction de classifier gradient
    reference_values = None
    
    classifier_grad_fn = get_classifier_grad_fn(
        classifier, 
        guidance_strategy, 
        guidance_scale, 
        adaptive_sigma_limit,
        reference_values
    )
    
    # DÉTECTION du type de predictor
    is_adaptive = predictor is not None and hasattr(predictor, '__name__') and predictor.__name__ == 'AdaptivePredictor'
    
    if is_adaptive:
        logging.info("🔄 Détection d'AdaptivePredictor - utilisation de la logique adaptative")
    
    def conditional_predictor_update_fn(x, t, labels, model, h=None, x_prev=None):
        """Predictor step with automatic interface handling."""
        # Combined score function
        def combined_score_fn(x_input, t_input):
            score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
            diffusion_score = score_fn(x_input, t_input)
            
            _, current_noise_scale = sde.marginal_prob(x_input, t_input)
            current_classifier_grad = classifier_grad_fn(x_input, current_noise_scale, labels)

            return diffusion_score + current_classifier_grad
        
        # CRÉATION du predictor avec la bonne interface selon le type
        if predictor is None:
            predictor_obj = NonePredictor(sde, combined_score_fn, shape, probability_flow)
            return predictor_obj.update_fn(x, t)
        else:
            if is_adaptive:
                # ✅ POUR AdaptivePredictor : interface complète
                predictor_obj = predictor(sde, combined_score_fn, shape, probability_flow, 
                                        eps=eps, abstol=0.01, reltol=0.01, 
                                        error_use_prev=True, norm="L2_scaled", safety=0.9, 
                                        sde_improved_euler=True, extrapolation=True, exp=0.9)
                # AdaptivePredictor retourne (x, x_prev, t, h)
                return predictor_obj.update_fn(x, t, h, x_prev)
            else:
                # ✅ POUR les autres predictors : interface standard
                predictor_obj = predictor(sde, combined_score_fn, shape, probability_flow)
                # Predictors standards retournent (x, x_mean)
                return predictor_obj.update_fn(x, t)
    
    def conditional_corrector_update_fn(x, t, labels, model):
        """Corrector step - interface standard."""
        def combined_score_fn(x_input, t_input):
            score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
            diffusion_score = score_fn(x_input, t_input)
            
            _, current_noise_scale = sde.marginal_prob(x_input, t_input)
            current_classifier_grad = classifier_grad_fn(x_input, current_noise_scale, labels)
            
            return diffusion_score + current_classifier_grad
        
        if corrector is None:
            corrector_obj = NoneCorrector(sde, combined_score_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, combined_score_fn, snr, n_steps)
        
        return corrector_obj.update_fn(x, t)
    
    def pc_conditional_sampler(model, labels):
        """Generate conditional samples with automatic method detection."""
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            
            if is_adaptive:
                logging.info("🔄 Utilisation du sampling adaptatif")
                h = torch.ones(shape[0], device=device) * 0.01  # h_init
                t = torch.ones(shape[0], device=device) * sde.T  # initial time
                x_prev = x.clone()
                
                N = 0  
                    
                while (torch.abs(t - eps) > 1e-6).any() :

                    x, x_mean = conditional_corrector_update_fn(x, t, labels, model)  
                    x_prev = x_mean  
                    
                    corrector_name = getattr(corrector, '__name__', 'none') if corrector is not None else 'none'
                    predictor_name = getattr(predictor, '__name__', 'none') if predictor is not None else 'none'
                    
                    if corrector_name != "none":  
                        N = N + 1
                        
                    x, x_prev, t, h = conditional_predictor_update_fn(x, t, labels, model, h, x_prev)
                    
                    if predictor_name != "none":  
                        N = N + 2
                    
                # Débruitage final pour AdaptivePredictor
                if denoise:
                    eps_t = torch.ones(shape[0], device=device) * eps
                    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
                    _, std = sde.marginal_prob(x, eps_t)  # ✅ FIX: syntaxe correcte
                    x = x + (std[:, None, None, None] ** 2) * score_fn(x, eps_t)
                
                return inverse_scaler(x)
                
            else:
                # ============ LOGIQUE STANDARD ============
                timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
                
                for i in range(sde.N):
                    t_step = timesteps[i]
                    vec_t = torch.ones(shape[0], device=device) * t_step
                    
                    # Corrector step
                    x, x_mean = conditional_corrector_update_fn(x, vec_t, labels, model)
                    
                    # Predictor step standard
                    x, x_mean = conditional_predictor_update_fn(x, vec_t, labels, model)
                
                return inverse_scaler(x_mean if denoise else x)
    
    return pc_conditional_sampler