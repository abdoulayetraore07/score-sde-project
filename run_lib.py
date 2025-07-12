# Based on score_sde_pytorch
# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from PIL import Image
from tqdm import tqdm
import warnings
from datetime import datetime

# Enhanced evaluation imports
from evaluation import (
    load_dataset_stats, 
    save_dataset_statistics,
    compute_fid_from_samples,
    compute_inception_score
)

# pytorch-fid integration
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logging.warning("pytorch-fid not available. Install with: pip install pytorch-fid")


# Import du debug contrôlé
try:
    from debug import DebugConfig, get_debug_logger
    DEBUG_MODE = DebugConfig.ENABLE_DEBUG
    logger = get_debug_logger(__name__)
except ImportError:
    DEBUG_MODE = True
    logger = logging.getLogger(__name__)


FLAGS = flags.FLAGS


def count_parameters(model):
    """Compte et affiche le nombre de paramètres du modèle."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_summary(config, score_model):
    """Affiche un résumé complet du modèle et de la configuration."""

    if not DebugConfig.should_show_model_summary():
        return
    
    total_params, trainable_params = count_parameters(score_model)
    
    logging.info("="*80)
    logging.info("MODEL & CONFIGURATION SUMMARY")
    logging.info("="*80)
    
    # Model info
    logging.info("📊 MODEL ARCHITECTURE:")
    logging.info(f"  Model type: {config.model.name}")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    logging.info(f"  EMA rate: {config.model.ema_rate}")
    
    # Model specific params
    if hasattr(config.model, 'nf'):
        logging.info(f"  Base channels (nf): {config.model.nf}")
    if hasattr(config.model, 'ch_mult'):
        logging.info(f"  Channel multipliers: {config.model.ch_mult}")
    if hasattr(config.model, 'num_res_blocks'):
        logging.info(f"  Residual blocks: {config.model.num_res_blocks}")
    if hasattr(config.model, 'attn_resolutions'):
        logging.info(f"  Attention resolutions: {config.model.attn_resolutions}")
    
    # SDE info
    logging.info("\n🌊 SDE CONFIGURATION:")
    logging.info(f"  SDE type: {config.training.sde}")
    logging.info(f"  Continuous: {config.training.continuous}")
    if config.training.sde.lower() == 'vesde':
        logging.info(f"  Sigma min: {config.model.sigma_min}")
        logging.info(f"  Sigma max: {config.model.sigma_max}")
    else:
        logging.info(f"  Beta min: {config.model.beta_min}")
        logging.info(f"  Beta max: {config.model.beta_max}")
    logging.info(f"  Num scales: {config.model.num_scales}")
    
    # Training info
    logging.info("\n🎯 TRAINING CONFIGURATION:")
    logging.info(f"  Dataset: {config.data.dataset}")
    logging.info(f"  Image size: {config.data.image_size}x{config.data.image_size}")
    logging.info(f"  Batch size: {config.training.batch_size}")
    logging.info(f"  Training iterations: {config.training.n_iters:,}")
    logging.info(f"  Snapshot frequency: {config.training.snapshot_freq:,}")
    logging.info(f"  Likelihood weighting: {config.training.likelihood_weighting}")
    logging.info(f"  Reduce mean: {config.training.reduce_mean}")
    
    # Optimizer info
    logging.info("\n⚙️  OPTIMIZER CONFIGURATION:")
    logging.info(f"  Optimizer: {config.optim.optimizer}")
    logging.info(f"  Learning rate: {config.optim.lr}")
    logging.info(f"  Beta1: {config.optim.beta1}")
    logging.info(f"  Weight decay: {config.optim.weight_decay}")
    logging.info(f"  Gradient clip: {config.optim.grad_clip}")
    logging.info(f"  Warmup steps: {config.optim.warmup}")
    
    # Sampling info (if enabled)
    if config.training.snapshot_sampling:
        logging.info("\n🎨 SAMPLING CONFIGURATION:")
        logging.info(f"  Method: {config.sampling.method}")
        logging.info(f"  Predictor: {config.sampling.predictor}")
        logging.info(f"  Corrector: {config.sampling.corrector}")
        logging.info(f"  SNR: {config.sampling.snr}")
        logging.info(f"  Noise removal: {config.sampling.noise_removal}")
    
    logging.info("="*80)
    logging.info("")


def train(config, workdir):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # AFFICHAGE: Summary du modèle et configuration
    print_model_summary(config, score_model)

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)

        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                tf.io.gfile.makedirs(this_sample_dir)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)

                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout)


def parse_eval_resolutions(eval_resolutions_str, default_resolution):
    """Parse comma-separated resolutions string."""
    if not eval_resolutions_str:
        return [default_resolution]
    
    try:
        resolutions = [int(r.strip()) for r in eval_resolutions_str.split(',')]
        # Validation basique
        for r in resolutions:
            if r < 64 or r > 2048:
                raise ValueError(f"Resolution {r} out of range [64, 2048]")
        return resolutions
    except (ValueError, AttributeError) as e:
        logging.warning(f"Invalid resolution string '{eval_resolutions_str}': {e}")
        logging.warning(f"Using default resolution: {default_resolution}")
        return [default_resolution]


def prepare_samples_multi_resolution(samples, config, target_resolutions):
    """
    Prepare samples for multiple target resolutions.
    
    Args:
        samples: numpy array [N, H, W, C] at original resolution
        config: configuration object
        target_resolutions: list of target resolutions
        
    Returns:
        dict: {resolution: samples_array}
    """
    from evaluation import resize_samples_to_resolution
    
    samples_dict = {}
    original_resolution = config.data.image_size
    
    for target_res in target_resolutions:
        if target_res == original_resolution:
            # Pas de resize nécessaire
            samples_dict[target_res] = samples
        else:
            # Resize
            logging.info(f"🔄 Resizing samples: {original_resolution}px → {target_res}px")
            resized_samples = resize_samples_to_resolution(samples, target_res)
            samples_dict[target_res] = resized_samples
    
    return samples_dict


def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # ENHANCED: Prepare dataset statistics for FID computation
    dataset_stats = None
    if config.eval.enable_sampling and FID_AVAILABLE:
        logging.info("🎯 Preparing dataset statistics for FID computation...")
        
        # Try to load existing statistics
        dataset_stats = load_dataset_stats(config)
        
        if dataset_stats is None:
            # Compute dataset statistics if they don't exist
            logging.info("Computing dataset statistics (this may take a while)...")
            try:
                stats_file = save_dataset_statistics(config)
                if stats_file:
                    dataset_stats = load_dataset_stats(config)
                    logging.info(f"✅ Dataset statistics computed and saved")
                else:
                    logging.warning("Failed to compute dataset statistics")
            except Exception as e:
                logging.warning(f"Error computing dataset statistics: {e}")
        else:
            logging.info("✅ Loaded existing dataset statistics")

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                logging.error(f"Cannot load checkpoint {ckpt_path}.")
                continue

        ema.copy_to(score_model.parameters())
        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            all_losses = []
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            for i, batch in enumerate(eval_iter):
                eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                eval_batch = eval_batch.permute(0, 3, 1, 2)
                eval_batch = scaler(eval_batch)
                eval_loss = eval_step(state, eval_batch)
                all_losses.append(eval_loss.item())
                if (i + 1) % 1000 == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = np.asarray(all_losses)
            with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
                fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            logging.info(f"Computing bits/dim on {config.eval.bpd_dataset} dataset...")
            # Build the likelihood computation function when likelihood is enabled
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler,
                                                          hutchinson_type=config.eval.hutchinson_type,
                                                          rtol=config.eval.rtol, atol=config.eval.atol)
            bpds = []
            bpd_iter = iter(ds_bpd)
            for repeat in range(bpd_num_repeats):
                bpd_round = []
                for batch_id in range(len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    bpd = likelihood_fn(score_model, eval_batch)[0]
                    bpd = bpd.detach().cpu().numpy().reshape(-1)
                    bpd_round.extend(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpd_round))))
                    bpd_round = np.asarray(bpd_round)

                bpds.append(bpd_round)

            bpds = np.asarray(bpds)
            with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_bpd.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, bpds)
                fout.write(io_buffer.getvalue())

        # Generate samples for FID/IS computation.
        if config.eval.enable_sampling:
            # Create unique sample directory
            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            tf.io.gfile.makedirs(this_sample_dir)
            
            # NOUVEAU: Génération SEULEMENT par classe (pas de génération aléatoire)
            all_samples = []  # Sera rempli par la génération conditionnelle
            
            # NOUVEAU: Support génération par classe pour AFHQ
            samples_by_class = None
            if config.data.dataset == 'AFHQ':
                
                logging.info("🎯 Generating class-conditional samples for AFHQ...")
                
                try:
                    # Importer les fonctions nécessaires
                    from controllable_generation import get_pc_conditional_sampler
                    from models.afhq_classifier import create_afhq_classifier
                    
                    # Charger le classifier AFHQ
                    classifier_dir = 'experiments/afhq_classifier'
                    if os.path.exists(classifier_dir):
                        checkpoints = [f for f in os.listdir(classifier_dir) if f.endswith('.pth')]
                        if checkpoints:
                            latest_checkpoint = sorted(checkpoints)[-1]
                            checkpoint_path = os.path.join(classifier_dir, latest_checkpoint)
                            
                            logging.info(f"📊 Loading AFHQ classifier: {checkpoint_path}")
                            
                            classifier = create_afhq_classifier(pretrained=False, freeze_backbone=False, embedding_size=128)
                            checkpoint = torch.load(checkpoint_path, map_location=config.device)
                            classifier.load_state_dict(checkpoint['model_state_dict'])
                            classifier = classifier.to(config.device)
                            classifier.eval()
                            
                            # Configuration génération conditionnelle
                            guidance_scale = getattr(config.eval, 'guidance_scale', 1.0)

                            # Répartition intelligente du nombre total de samples
                            total_samples = config.eval.num_samples
                            samples_per_class_base = total_samples // 3
                            remainder = total_samples % 3

                            # Répartir le reste sur les premières classes
                            class_sample_counts = {
                                'cat': samples_per_class_base + (1 if remainder > 0 else 0),
                                'dog': samples_per_class_base + (1 if remainder > 1 else 0), 
                                'wild': samples_per_class_base
                            }

                            logging.info(f"📊 Sample distribution: cat={class_sample_counts['cat']}, dog={class_sample_counts['dog']}, wild={class_sample_counts['wild']} (total={sum(class_sample_counts.values())})")
                            

                            class_mapping = {'cat': 0, 'dog': 1, 'wild': 2}
                            samples_by_class = {}
                            
                            # Générer pour chaque résolution
                            eval_resolutions = getattr(config.eval, 'resolutions', [config.data.image_size])
                            
                            for resolution in eval_resolutions:
                                samples_by_class[resolution] = {}
                                
                                for class_name, class_idx in class_mapping.items():
                                    current_class_count = class_sample_counts[class_name]
                                    logging.info(f"🔄 Generating {current_class_count} samples for class '{class_name}' at {resolution}px...")
                                    
                                    # Setup conditional sampler pour cette résolution
                                    sampling_shape = (current_class_count, config.data.num_channels, resolution, resolution)
                                    
                                    conditional_sampler = get_pc_conditional_sampler(
                                        sde=sde,
                                        classifier=classifier,
                                        shape=sampling_shape,
                                        predictor=sampling.get_predictor(config.sampling.predictor.lower()),
                                        corrector=sampling.get_corrector(config.sampling.corrector.lower()),
                                        inverse_scaler=inverse_scaler,
                                        snr=config.sampling.snr,
                                        n_steps=config.sampling.n_steps_each,
                                        probability_flow=config.sampling.probability_flow,
                                        continuous=config.training.continuous,
                                        denoise=config.sampling.noise_removal,
                                        eps=sampling_eps,
                                        device=config.device,
                                        guidance_scale=guidance_scale
                                    )
                                    
                                    # Labels pour cette classe
                                    labels = torch.full((current_class_count,), class_idx, dtype=torch.long, device=config.device)
                                    
                                    # Générer
                                    ema.store(score_model.parameters())
                                    ema.copy_to(score_model.parameters())
                                    
                                    class_samples = conditional_sampler(score_model, labels)
                                    
                                    ema.restore(score_model.parameters())
                                    
                                    # Convertir et stocker
                                    class_samples_np = np.clip(
                                        class_samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 
                                        0, 255
                                    ).astype(np.uint8)
                                    
                                    # Redimensionner si nécessaire
                                    if resolution != config.data.image_size:
                                        from evaluation import resize_samples_to_resolution
                                        class_samples_np = resize_samples_to_resolution(
                                            class_samples_np, resolution
                                        )
                                    
                                    samples_by_class[resolution][class_name] = class_samples_np
                                    
                                    logging.info(f"✅ Generated {len(class_samples_np)} samples for {class_name}")
                            
                            logging.info("✅ Class-conditional generation completed")
                            
                        else:
                            logging.warning("⚠️  No AFHQ classifier checkpoint found - using global FID")
                    else:
                        logging.warning("⚠️  AFHQ classifier directory not found - using global FID")
                        
                except Exception as e:
                    logging.warning(f"⚠️  Error in class-conditional generation: {e}")
                    logging.warning("Falling back to global FID")
                    samples_by_class = None


            if samples_by_class:
                logging.info("💾 Saving class-specific samples...")
                
                # Créer dossiers par classe
                generated_images_dir = os.path.join(this_sample_dir, "generated_images")
                tf.io.gfile.makedirs(generated_images_dir)
                
                combined_samples = []
                
                for resolution in eval_resolutions:
                    if resolution in samples_by_class:
                        for class_name, class_samples in samples_by_class[resolution].items():
                            # Dossier pour cette classe
                            class_dir = os.path.join(generated_images_dir, class_name)
                            tf.io.gfile.makedirs(class_dir)
                            
                            # Sauvegarder images individuelles
                            for idx, sample in enumerate(class_samples):
                                img = Image.fromarray(sample.astype(np.uint8))
                                img_path = os.path.join(class_dir, f"sample_{idx+1:05d}.png")
                                with tf.io.gfile.GFile(img_path, "wb") as f:
                                    img.save(f, format='PNG')
                            
                            # Sauvegarder archive NumPy par classe
                            class_npz_path = os.path.join(this_sample_dir, f"class_samples_{class_name}.npz")
                            with tf.io.gfile.GFile(class_npz_path, "wb") as fout:
                                io_buffer = io.BytesIO()
                                np.savez_compressed(io_buffer, samples=class_samples)
                                fout.write(io_buffer.getvalue())
                            
                            # Ajouter aux samples combinés (pour résolution native uniquement)
                            if resolution == config.data.image_size:
                                combined_samples.extend(class_samples)
                            
                            logging.info(f"✅ Saved {len(class_samples)} samples for class '{class_name}' at {resolution}px")
                
                # Créer all_samples pour les autres calculs
                all_samples = np.array(combined_samples) if combined_samples else np.array([])
                
                # Sauvegarder samples combinés
                if len(all_samples) > 0:
                    combined_dir = os.path.join(generated_images_dir, "combined")
                    tf.io.gfile.makedirs(combined_dir)
                    
                    # Images combinées
                    for idx, sample in enumerate(all_samples):
                        img = Image.fromarray(sample.astype(np.uint8))
                        img_path = os.path.join(combined_dir, f"sample_{idx+1:05d}.png")
                        with tf.io.gfile.GFile(img_path, "wb") as f:
                            img.save(f, format='PNG')
                    
                    # Archive NumPy combinée
                    combined_npz_path = os.path.join(this_sample_dir, "combined_samples.npz")
                    with tf.io.gfile.GFile(combined_npz_path, "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, samples=all_samples)
                        fout.write(io_buffer.getvalue())
                    
                    logging.info(f"✅ Saved {len(all_samples)} combined samples")

            else:
                # Fallback si pas de génération par classe (ne devrait jamais arriver pour AFHQ)
                logging.warning("⚠️  No class-conditional generation - this should not happen for AFHQ")
                all_samples = np.array([])

                
            # NOUVEAU: Support multi-résolution
            eval_resolutions = getattr(config.eval, 'resolutions', [config.data.image_size])
            use_official_stats = getattr(config.eval, 'use_official_stats', True)
            
            logging.info(f"🎯 Évaluation multi-résolution: {eval_resolutions}")
            
            # Préparer échantillons pour toutes les résolutions
            samples_dict = prepare_samples_multi_resolution(all_samples, config, eval_resolutions)
            
            # ENHANCED: Compute FID multi-resolution avec Clean-FID
            fid_results = {}
            if config.eval.enable_sampling:
                # Configuration FID
                use_clean_fid = getattr(config.eval, 'use_clean_fid', True)
                fid_mode = getattr(config.eval, 'fid_mode', 'clean')  # 'clean' ou 'legacy_pytorch'
                dataset_name = getattr(config.eval, 'dataset_name', 'afhq')  # 'afhq', 'afhq_cat', etc.
                
                # NOUVEAU: Import et appel avec samples_by_class
                from evaluation import compute_fid_multi_resolution
                
                fid_results = compute_fid_multi_resolution(
                    samples_dict, 
                    config, 
                    eval_resolutions,
                    use_clean_fid=use_clean_fid,
                    mode=fid_mode,
                    dataset_name=dataset_name,
                    samples_by_class=samples_by_class  # NOUVEAU PARAMÈTRE
                )
                
                # Log FID results
                logging.info("📊 FID Results:")
                for resolution, result in fid_results.items():
                    if isinstance(result, dict) and 'overall' in result:
                        # Résultats par classe
                        logging.info(f"   Resolution {resolution}px:")
                        logging.info(f"      Overall FID: {result['overall']:.4f}")
                        for cls, score in result['by_class'].items():
                            count = result['class_counts'][cls]
                            logging.info(f"      {cls}: {score:.4f} ({count} samples)")
                    else:
                        # FID global simple
                        logging.info(f"   Resolution {resolution}px: {result:.4f}")

            # ENHANCED: Compute Inception Score
            if config.eval.enable_sampling and len(all_samples) > 0:
                logging.info("🎯 Computing Inception Score...")
                
                try:
                    is_mean, is_std = compute_inception_score(all_samples, device=config.device)
                    logging.info(f"✅ Inception Score: {is_mean:.4f} ± {is_std:.4f}")
                except Exception as e:
                    logging.warning(f"⚠️  Error computing Inception Score: {e}")
                    is_mean, is_std = -1, -1

            # Save FID and IS results
            if config.eval.enable_sampling:
                results_dict = {
                    'ckpt': ckpt,
                    'num_samples': config.eval.num_samples,
                    'fid_results': fid_results,
                    'inception_score_mean': is_mean if 'is_mean' in locals() else -1,
                    'inception_score_std': is_std if 'is_std' in locals() else -1,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Sauvegarder les résultats
                with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_metrics.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, **results_dict)
                    fout.write(io_buffer.getvalue())

        ema.restore(score_model.parameters())