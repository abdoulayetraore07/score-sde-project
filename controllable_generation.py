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

#Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
from .pc_sampler import (
    ReverseDiffusionPredictor, 
    LangevinCorrector, 
    EulerMaruyamaPredictor,
    AncestralSamplingPredictor,
    NoneCorrector,
    NonePredictor
)


def range_sigmas_active(labels):
   
  if labels[0].item()== 2 : # Wild
    SIGMA_LIMIT_PRED_MIN = 0.01  
    SIGMA_LIMIT_PRED_MAX = 100   
    SIGMA_MAX_CLASSIFIER = 50.   
  elif labels[0].item()== 1 : # Dog
    SIGMA_LIMIT_PRED_MIN = 0.01  
    SIGMA_LIMIT_PRED_MAX = 40  
    SIGMA_MAX_CLASSIFIER = 20 
  elif labels[0].item()== 0 :  #Cat
    SIGMA_LIMIT_PRED_MIN = 0.01
    SIGMA_LIMIT_PRED_MAX = 100
    SIGMA_MAX_CLASSIFIER = 50

  return SIGMA_MAX_CLASSIFIER, SIGMA_LIMIT_PRED_MIN, SIGMA_LIMIT_PRED_MAX 


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


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

  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                           sde=sde,
                                           predictor=predictor,
                                           probability_flow=probability_flow,
                                           continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
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

  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                           sde=sde,
                                           predictor=predictor,
                                           probability_flow=probability_flow,
                                           continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
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


def load_reference_classifier_values(classifier, device='cuda', reference_sigma=30.0):
    """Charge les images de référence et calcule les valeurs du classifier.
    
    Args:
        classifier: Le classifier AFHQ entraîné
        device: Device à utiliser
        reference_sigma: Valeur de sigma pour la référence
        
    Returns:
        dict: Valeurs de référence pour chaque classe
    """
    reference_dir = "assets/cond_gen_afhq_512"
    reference_values = {}
    
    class_mapping = {'cat': 0, 'dog': 1, 'wild': 2}
    
    if not os.path.exists(reference_dir):
        logging.warning(f"⚠️ Dossier de référence {reference_dir} non trouvé")
        # Valeurs par défaut si pas d'images de référence
        return {0: 1.0, 1: 1.0, 2: 1.0}
    
    for class_name, class_idx in class_mapping.items():
        # Chercher l'image de référence
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate_path = os.path.join(reference_dir, f"{class_name}{ext}")
            if os.path.exists(candidate_path):
                image_path = candidate_path
                break
        
        if image_path is None:
            logging.warning(f"⚠️ Image de référence pour {class_name} non trouvée")
            reference_values[class_idx] = 1.0
            continue
        
        try:
            # Charger et préprocesser l'image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Calculer la valeur de référence
            with torch.no_grad():
                sigma_tensor = torch.full((1,), reference_sigma, device=device)
                logits = classifier(image_tensor, sigma_tensor)
                log_probs = F.log_softmax(logits, dim=1)
                reference_values[class_idx] = log_probs[0, class_idx].item()
                
            logging.info(f"✅ Référence {class_name}: {reference_values[class_idx]:.4f}")
            
        except Exception as e:
            logging.warning(f"⚠️ Erreur chargement référence {class_name}: {e}")
            reference_values[class_idx] = 1.0
    
    return reference_values


def apply_standard_strategy(classifier_grad, guidance_scale):
    """Stratégie 1: Standard baseline."""
    return classifier_grad * guidance_scale


def apply_adaptive_scale_strategy(classifier_grad, current_sigma, guidance_scale, adaptive_sigma_limit):
    """Stratégie 2: Adaptive scale linear variation."""
    sigma_max = 784.0
    
    if current_sigma >= adaptive_sigma_limit:
        # Zone adaptative: 0.01 à 1.0 linéairement
        a, b = 0.01, 1.0
        alpha = (sigma_max - current_sigma) / (sigma_max - adaptive_sigma_limit)
        adaptive_scale = a + (b - a) * alpha
    else:
        # Zone constante: 1.0
        adaptive_scale = 1.0
    
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
    
    return grad.detach() * guidance_scale


def apply_amplification_strategy(x, noise_scale, labels, classifier, guidance_scale, reference_values, amplification_sigma_limit=30.0):
    """Stratégie 4: Amplification artificielle."""
    current_sigma = noise_scale[0].item()
    sigma_max = 784.0
    
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        
        if current_sigma >= amplification_sigma_limit:
            # Zone artificielle: interpolation vers référence
            alpha = (sigma_max - current_sigma) / (sigma_max - amplification_sigma_limit)
            
            # Valeur uniforme (équiprobable)
            uniform_log_prob = np.log(1.0 / 3.0)
            
            # Valeur de référence pour cette classe
            target_class = labels[0].item()
            target_log_prob = reference_values.get(target_class, uniform_log_prob)
            
            # Interpolation linéaire
            artificial_log_prob = (1 - alpha) * uniform_log_prob + alpha * target_log_prob
            
            # Créer un gradient artificiel vers cette valeur
            artificial_loss = artificial_log_prob * labels.shape[0]
            grad = torch.autograd.grad(artificial_loss, x, create_graph=False, retain_graph=False)[0]
            
        else:
            # Zone normale: utiliser le classifier réel
            logits = classifier(x, noise_scale)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_probs = log_probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
            loss = target_log_probs.sum()
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
    
    return grad.detach() * guidance_scale


def get_classifier_grad_fn(classifier, guidance_strategy="truncation", guidance_scale=1.0, adaptive_sigma_limit=50.0, reference_values=None):
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
            
        elif guidance_strategy == "amplification":
            # Stratégie 4: Amplification
            if reference_values is None:
                # Fallback vers standard si pas de référence
                logging.warning("⚠️ Pas de valeurs de référence, fallback vers standard")
                return get_classifier_grad_fn(classifier, "standard", guidance_scale, adaptive_sigma_limit)(x, noise_scale, labels)
            return apply_amplification_strategy(x, noise_scale, labels, classifier, guidance_scale, reference_values)
            
        else:
            raise ValueError(f"Stratégie inconnue: {guidance_strategy}")
    
    return classifier_grad_fn


def get_pc_conditional_sampler(sde, classifier, shape, predictor, corrector, inverse_scaler, snr,
                               n_steps=1, probability_flow=False, continuous=True, 
                               denoise=True, eps=1e-5, device='cuda', guidance_scale=1.0,
                               guidance_strategy="truncation", adaptive_sigma_limit=50.0):
    """Class-conditional sampling with PC samplers and 4 adaptive strategies."""
    
    # Charger les valeurs de référence pour la stratégie amplification
    reference_values = None
    if guidance_strategy == "amplification":
        reference_values = load_reference_classifier_values(classifier, device)
    
    # Create classifier gradient function with selected strategy
    classifier_grad_fn = get_classifier_grad_fn(
        classifier, 
        guidance_strategy, 
        guidance_scale, 
        adaptive_sigma_limit,
        reference_values
    )
    
    def conditional_predictor_update_fn(x, t, labels, model):
        """Predictor step with classifier guidance."""
        # Create combined score function with closure on labels
        def combined_score_fn(x_input, t_input):
            # Get diffusion score
            score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
            diffusion_score = score_fn(x_input, t_input)
            
            # Get classifier gradient with selected strategy
            _, current_noise_scale = sde.marginal_prob(x_input, t_input)
            current_classifier_grad = classifier_grad_fn(x_input, current_noise_scale, labels)
            
            # Combined score
            return diffusion_score + current_classifier_grad
        
        # Create predictor with combined score
        if predictor is None:
            predictor_obj = NonePredictor(sde, combined_score_fn, probability_flow)
        else:
            predictor_obj = predictor(sde, combined_score_fn, probability_flow)
        
        return predictor_obj.update_fn(x, t)
    
    def conditional_corrector_update_fn(x, t, labels, model):
        """Corrector step with classifier guidance."""
        # Create combined score function with closure on labels
        def combined_score_fn(x_input, t_input):
            # Get diffusion score
            score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
            diffusion_score = score_fn(x_input, t_input)
            
            # Get classifier gradient with selected strategy
            _, current_noise_scale = sde.marginal_prob(x_input, t_input)
            current_classifier_grad = classifier_grad_fn(x_input, current_noise_scale, labels)
            
            # Combined score
            return diffusion_score + current_classifier_grad
        
        # Create corrector with combined score
        if corrector is None:
            corrector_obj = NoneCorrector(sde, combined_score_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, combined_score_fn, snr, n_steps)
        
        return corrector_obj.update_fn(x, t)
    
    def pc_conditional_sampler(model, labels):
        """Generate class-conditional samples with selected strategy."""
        with torch.no_grad():
            # Initial sample from prior
            x = sde.prior_sampling(shape).to(device)
            
            # Time steps
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            
            # Sampling loop
            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=device) * t
                
                # Corrector step with guidance
                x, x_mean = conditional_corrector_update_fn(x, vec_t, labels, model)
                
                # Predictor step with guidance
                x, x_mean = conditional_predictor_update_fn(x, vec_t, labels, model)
            
            # Return final samples
            return inverse_scaler(x_mean if denoise else x)
    
    return pc_conditional_sampler