#!/usr/bin/env python3
"""
Script de controllable generation pour les modèles pré-entraînés.
VERSION CORRIGÉE: Interface de modification des paramètres externalisée
Supporte inpainting et colorization avec masques créatifs.
"""

import sys
import os
import argparse
import torch
import numpy as np
import logging
from datetime import datetime
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import warnings

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

sys.path.append('.')

# Imports pour les modèles
from models import ncsnpp, ddpm, ncsnv2
import run_lib
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sampling
import sde_lib
import datasets
import losses
from utils import restore_checkpoint
from torchvision.utils import make_grid, save_image
from controllable_generation import get_pc_inpainter, get_pc_colorizer, get_pc_conditional_sampler
from models.afhq_classifier import create_afhq_classifier
from tools.checkpoint_selector import select_afhq_checkpoint, copy_checkpoint_to_meta


# ✅ IMPORT DU MODULE EXTERNE
from tools.config_modifier import interactive_config_modification, print_config_summary
from tools.seed_utils import set_seeds_for_sampling, apply_seed_before_generation, log_seed_info, set_global_seed_policy
# Activer les seeds fixes
set_global_seed_policy(True)

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


def create_samples_structure():
    """Crée la structure des dossiers samples selon les spécifications."""
    models = ['church', 'celebahq_256', 'ffhq_1024', 'afhq_512']
    base_tasks = ['general', 'inpainting', 'colorization']
    
    logging.info("🔧 Création de la structure des dossiers samples...")
    
    for model in models:
        for task in base_tasks:
            os.makedirs(f"samples/{model}/{task}", exist_ok=True)
        
        # Seul AFHQ a le dossier controllable
        if model == 'afhq_512':
            os.makedirs(f"samples/{model}/controllable", exist_ok=True)
    
    logging.info("✅ Structure des dossiers créée")


def load_afhq_classifier(device='cuda'):
    """Charge le classifier AFHQ entraîné avec paramètres CORRECTS."""
    
    # Chercher le dernier checkpoint
    classifier_dir = 'experiments/afhq_classifier'
    
    if not os.path.exists(classifier_dir):
        raise FileNotFoundError(f"Dossier classifier non trouvé: {classifier_dir}")
    
    # Lister les checkpoints
    checkpoints = [f for f in os.listdir(classifier_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        raise FileNotFoundError(f"Aucun checkpoint classifier trouvé dans {classifier_dir}")
    
    # Prendre le plus récent
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(classifier_dir, latest_checkpoint)
    
    logging.info(f"📊 Chargement du classifier: {checkpoint_path}")
    
    # Créer et charger le modèle
    classifier = create_afhq_classifier(pretrained=False, freeze_backbone=False, embedding_size=128)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.to(device)
    classifier.eval()
    
    accuracy = checkpoint.get('val_acc', 'Unknown')
    logging.info(f"✅ Classifier chargé - Accuracy: {accuracy:.2f}%")
    
    return classifier


def load_config(model_name):
    """Charge la configuration selon le modèle."""
    if model_name == 'church':
        from configs.ve import church_ncsnpp_continuous
        return church_ncsnpp_continuous.get_config()
    elif model_name == 'celebahq_256':
        from configs.ve.celebahq_256_ncsnpp_continuous import get_config as get_celebahq_256_config
        return get_celebahq_256_config()
    elif model_name == 'ffhq_1024':
        from configs.ve.ffhq_1024_ncsnpp_continuous import get_config as get_ffhq_1024_config
        return get_ffhq_1024_config()
    elif model_name == 'afhq_512':
        from configs.ve.afhq_512_ncsnpp_continuous import get_config as get_afhq_512_config
        return get_afhq_512_config()
    else:
        raise ValueError(f"Modèle '{model_name}' non supporté. Utilisez: church, celebahq_256, ffhq_1024, afhq_512")


def load_model(model_name, interactive=True):
    """Charge le modèle pré-entraîné avec sélection AFHQ."""
    checkpoint_paths = {
        'church': "experiments/church_ncsnpp_continuous/checkpoints-meta/checkpoint.pth",
        'celebahq_256': "experiments/celebahq_256_ncsnpp_continuous/checkpoints-meta/checkpoint.pth", 
        'ffhq_1024': "experiments/ffhq_1024_ncsnpp_continuous/checkpoints-meta/checkpoint.pth",
        'afhq_512': "experiments/afhq_512_ncsnpp_continuous/checkpoints-meta/checkpoint.pth"
    }
    
    if model_name not in checkpoint_paths:
        raise ValueError(f"Modèle '{model_name}' non supporté.")
    
    # Sélection interactive pour AFHQ
    if model_name == 'afhq_512':
        selected_checkpoint = select_afhq_checkpoint(model_name, interactive=interactive)
        if selected_checkpoint is None:
            logging.error("❌ Aucun checkpoint AFHQ sélectionné")
            return None, None, None, None
        
        # Copier vers checkpoints-meta pour utilisation
        copy_checkpoint_to_meta(selected_checkpoint, model_name)
    
    checkpoint_path = checkpoint_paths[model_name]
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        return None, None, None, None
    
    logging.info(f"📊 Chargement du modèle {model_name}...")
    config = load_config(model_name)
    
    # ✅ AFFICHAGE INITIAL
    print_config_summary(config, model_name, 4, checkpoint_path, modified=False)
    
    # ✅ MODIFICATION INTERACTIVE EXTERNALISÉE
    config_modified = interactive_config_modification(config)
    
    # ✅ AFFICHAGE FINAL SI MODIFIÉ
    if config_modified:
        print_config_summary(config, model_name, 4, checkpoint_path, modified=True)
                

    # Créer le modèle
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    # Charger le checkpoint
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = restore_checkpoint(checkpoint_path, state, device=config.device)
        logging.info("✅ Modèle chargé avec succès")
        return config, score_model, ema, state
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement: {e}")
        return None, None, None, None

 
def setup_sde(config):
    """Configure le SDE selon la configuration."""
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
        raise NotImplementedError(f"SDE {config.training.sde} non supporté.")
    
    return sde, sampling_eps


def create_creative_mask(image_size, mask_percentage, mask_shape='random'):
    """Crée un masque créatif avec formes variées."""
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Types de masques possibles
    mask_types = ['square', 'rectangle', 'triangle', 'eight', 'random_patches', 'circle']
    
    if mask_shape == 'random':
        mask_type = random.choice(mask_types)
    else:
        mask_type = mask_shape
    
    if mask_type == 'square':
        # Carré aléatoire
        size = int(image_size * np.sqrt(mask_percentage / 100))
        x = random.randint(0, max(1, image_size - size))
        y = random.randint(0, max(1, image_size - size))
        mask[y:y+size, x:x+size] = 1.0
        
    elif mask_type == 'rectangle':
        # Rectangle aléatoire
        area = int(image_size * image_size * mask_percentage / 100)
        aspect_ratio = random.uniform(0.3, 3.0)
        w = int(np.sqrt(area * aspect_ratio))
        h = int(area / w)
        w = min(w, image_size)
        h = min(h, image_size)
        x = random.randint(0, max(1, image_size - w))
        y = random.randint(0, max(1, image_size - h))
        mask[y:y+h, x:x+w] = 1.0
        
    elif mask_type == 'triangle':
        # Triangle aléatoire
        img = Image.fromarray((mask * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
        # Points du triangle
        center_x = random.randint(image_size//4, 3*image_size//4)
        center_y = random.randint(image_size//4, 3*image_size//4)
        size = int(image_size * np.sqrt(mask_percentage / 100) * 0.8)
        
        points = [
            (center_x, center_y - size//2),
            (center_x - size//2, center_y + size//2),
            (center_x + size//2, center_y + size//2)
        ]
        draw.polygon(points, fill=255)
        mask = np.array(img) / 255.0
        
    elif mask_type == 'eight':
        # Forme de "8" avec proportions réalistes
        img = Image.fromarray((mask * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        center_x = image_size // 2
        center_y = image_size // 2
        
        # Radius proportionnel au pourcentage de masque
        radius = int(image_size * np.sqrt(mask_percentage / 100) * 0.25)
        
        # Espacements pour un "8" bien formé
        vertical_offset = int(radius * 1.1)
        
        # Cercle supérieur (légèrement plus petit)
        top_radius = int(radius * 0.9)
        draw.ellipse([
            center_x - top_radius,
            center_y - vertical_offset - top_radius//2,
            center_x + top_radius,
            center_y - vertical_offset + top_radius//2
        ], fill=255)
        
        # Cercle inférieur (légèrement plus grand)
        bottom_radius = int(radius * 1.1)
        draw.ellipse([
            center_x - bottom_radius,
            center_y + vertical_offset - bottom_radius//2,
            center_x + bottom_radius,
            center_y + vertical_offset + bottom_radius//2
        ], fill=255)
        
        # Connexion centrale pour un vrai "8"
        connection_width = int(radius * 0.6)
        connection_height = int(radius * 0.3)
        draw.ellipse([
            center_x - connection_width,
            center_y - connection_height,
            center_x + connection_width,
            center_y + connection_height
        ], fill=255)
    
        mask = np.array(img) / 255.0
            
    elif mask_type == 'circle':
        # Cercle aléatoire
        img = Image.fromarray((mask * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
        center_x = random.randint(image_size//4, 3*image_size//4)
        center_y = random.randint(image_size//4, 3*image_size//4)
        radius = int(image_size * np.sqrt(mask_percentage / 100) * 0.5)
        
        draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], fill=255)
        mask = np.array(img) / 255.0
        
    elif mask_type == 'random_patches':
        # Patches aléatoires
        num_patches = random.randint(3, 8)
        total_area = int(image_size * image_size * mask_percentage / 100)
        patch_area = total_area // num_patches
        
        for _ in range(num_patches):
            size = int(np.sqrt(patch_area))
            x = random.randint(0, max(1, image_size - size))
            y = random.randint(0, max(1, image_size - size))
            mask[y:y+size, x:x+size] = 1.0
    
    # CORRECTION: Inverser le masque pour que les formes géométriques soient à reconstruire
    mask = 1.0 - mask  # Les zones 1 deviennent 0 (à reconstruire) et vice versa
    
    return torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]


def load_test_image(model_name, image_id):
    """Charge l'image de test depuis le dossier assets."""
    assets_dir = f"assets/cond_gen_{model_name}"
    
    if not os.path.exists(assets_dir):
        raise FileNotFoundError(f"Dossier {assets_dir} non trouvé")
    
    # Lister les images
    image_files = [f for f in os.listdir(assets_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
               and not f.startswith('._')]  # Filtrer les fichiers cachés macOS
    
    if not image_files:
        raise FileNotFoundError(f"Aucune image trouvée dans {assets_dir}")
    
    if image_id > len(image_files):
        raise ValueError(f"Image ID {image_id} non disponible. Max: {len(image_files)}")
    
    image_path = os.path.join(assets_dir, sorted(image_files)[image_id-1])
    logging.info(f"📷 Chargement de l'image: {image_path}")
    
    return image_path



def test_inpainting(model_name, image_id, mask_type, mask_percentage, num_samples, mask_shape='random', interactive=True):
    """Test de l'inpainting."""
    logging.info("🎨 Test d'inpainting...")
    
    # Charger le modèle avec sélection
    config, score_model, ema, state = load_model(model_name, interactive=interactive)

    if config is None:
        return
    
    # Setup SDE
    sde, sampling_eps = setup_sde(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Charger l'image
    image_path = load_test_image(model_name, image_id)
    
    # Préparer l'image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((config.data.image_size, config.data.image_size))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(config.device)  # [1, 3, H, W]
    
    # Normaliser l'image
    scaler = datasets.get_data_scaler(config)
    image_tensor = scaler(image_tensor)
    
    # Déterminer le pourcentage de masque
    if mask_percentage is None:
        if mask_type == 'light':
            mask_percentage = random.uniform(10, 25)
        elif mask_type == 'heavy':
            mask_percentage = random.uniform(50, 75)
        else:  # random
            if random.random() < 0.5:
                mask_percentage = random.uniform(10, 25)
            else:
                mask_percentage = random.uniform(50, 75)
    
    logging.info(f"🎭 Masque: {mask_percentage:.1f}%")
    
    # Créer le masque
    mask = create_creative_mask(config.data.image_size, mask_percentage, mask_shape)
    mask = mask.to(config.device)
    
    # Setup inpainting
    inpainting_fn = get_pc_inpainter(
        sde=sde,
        predictor=sampling.get_predictor(config.sampling.predictor.lower()),
        corrector=sampling.get_corrector(config.sampling.corrector.lower()),
        inverse_scaler=inverse_scaler,
        snr=config.sampling.snr,
        n_steps=config.sampling.n_steps_each,
        probability_flow=config.sampling.probability_flow,
        continuous=config.training.continuous,
        denoise=config.sampling.noise_removal,
        eps=sampling_eps
    )
    
    # Utiliser EMA
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    
    # Générer les échantillons
    results = []
    
    # Image originale
    orig_img = inverse_scaler(image_tensor).squeeze(0)
    results.append(orig_img)
    
    # Image masquée (pour visualisation) - NOIR dans les zones à reconstruire
    masked_img = image_tensor * mask + torch.zeros_like(image_tensor) * (1 - mask)
    masked_img = inverse_scaler(masked_img).squeeze(0)
    results.append(masked_img)
    
    # Générer les reconstructions
    logging.info(f"🔄 Génération de {num_samples} échantillons...")
    for i in tqdm(range(num_samples), desc="Inpainting"):
        reconstructed = inpainting_fn(score_model, image_tensor, mask)
        results.append(reconstructed.squeeze(0))
    
    ema.restore(score_model.parameters())
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"samples/{model_name}/inpainting"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer la grille
    grid = make_grid(results, nrow=len(results), padding=2, normalize=True)
    
    # Sauvegarder la grille
    grid_path = os.path.join(output_dir, f"inpainting_{model_name}_{timestamp}_mask_{mask_percentage:.0f}pct.png")
    save_image(grid, grid_path)

    # Convertir les résultats en numpy pour sauvegarde NPZ
    results_np = []
    for result in results:
        img_np = np.clip(result.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        results_np.append(img_np)

    # Sauvegarder en NPZ
    npz_path = os.path.join(output_dir, f"inpainting_{model_name}_{timestamp}_mask_{mask_percentage:.0f}pct.npz")
    np.savez_compressed(npz_path, 
                        samples=np.array(results_np),
                        model=model_name,
                        task='inpainting',
                        mask_percentage=mask_percentage,
                        mask_shape=mask_shape)

    # Calculer la taille
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    logging.info(f"📦 Archive NPZ sauvegardée: {os.path.basename(npz_path)} ({npz_size:.1f} MB)")  
    logging.info(f"✅ Résultats sauvegardés: {grid_path}")



def test_colorization(model_name, image_id, num_samples, interactive=True):
    """Test de la colorization."""
    logging.info("🌈 Test de colorization...")
    
    # Charger le modèle avec sélection
    config, score_model, ema, state = load_model(model_name, interactive=interactive)

    if config is None:
        return
    
    # Setup SDE
    sde, sampling_eps = setup_sde(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Charger l'image
    image_path = load_test_image(model_name, image_id)
    
    # Préparer l'image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((config.data.image_size, config.data.image_size))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(config.device)  # [1, 3, H, W]
    
    # Normaliser l'image
    scaler = datasets.get_data_scaler(config)
    image_tensor = scaler(image_tensor)
    
    # Convertir en noir et blanc
    gray_image = torch.mean(image_tensor, dim=1, keepdim=True)
    gray_image = gray_image.repeat(1, 3, 1, 1)  # Répéter sur les 3 canaux
    
    # Setup colorization
    colorization_fn = get_pc_colorizer(
        sde=sde,
        predictor=sampling.get_predictor(config.sampling.predictor.lower()),
        corrector=sampling.get_corrector(config.sampling.corrector.lower()),
        inverse_scaler=inverse_scaler,
        snr=config.sampling.snr,
        n_steps=config.sampling.n_steps_each,
        probability_flow=config.sampling.probability_flow,
        continuous=config.training.continuous,
        denoise=config.sampling.noise_removal,
        eps=sampling_eps
    )
    
    # Utiliser EMA
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    
    # Générer les échantillons
    results = []
    
    # Image originale
    orig_img = inverse_scaler(image_tensor).squeeze(0)
    results.append(orig_img)
    
    # Image en noir et blanc
    gray_img = inverse_scaler(gray_image).squeeze(0)
    results.append(gray_img)
    
    # Générer les colorizations
    logging.info(f"🔄 Génération de {num_samples} échantillons...")
    for i in tqdm(range(num_samples), desc="Colorization"):
        colorized = colorization_fn(score_model, gray_image)
        results.append(colorized.squeeze(0))
    
    ema.restore(score_model.parameters())
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"samples/{model_name}/colorization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer la grille
    grid = make_grid(results, nrow=len(results), padding=2, normalize=True)
    
    # Sauvegarder la grille
    grid_path = os.path.join(output_dir, f"colorization_{model_name}_{timestamp}.png")
    save_image(grid, grid_path)

    # Convertir les résultats en numpy pour sauvegarde NPZ
    results_np = []
    for result in results:
        img_np = np.clip(result.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        results_np.append(img_np)

    # Sauvegarder en NPZ
    npz_path = os.path.join(output_dir, f"colorization_{model_name}_{timestamp}.npz")
    np.savez_compressed(npz_path,
                        samples=np.array(results_np),
                        model=model_name,
                        task='colorization')

    # Calculer la taille
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    logging.info(f"📦 Archive NPZ sauvegardée: {os.path.basename(npz_path)} ({npz_size:.1f} MB)")
    
    logging.info(f"✅ Résultats sauvegardés: {grid_path}")



def test_controllable(model_name, image_id, class_name, num_samples, guidance_scale=1.0, interactive=True):
    """Test de la génération conditionnelle par classe (AFHQ seulement)."""
    
    if model_name != 'afhq_512':
        logging.error("❌ Génération par classe disponible seulement pour AFHQ_512")
        return
    
    logging.info(f"🎯 Test de génération conditionnelle - Classe: {class_name}")
    
    # Classes AFHQ
    class_mapping = {'cat': 0, 'dog': 1, 'wild': 2}
    
    if class_name not in class_mapping:
        logging.error(f"❌ Classe {class_name} non supportée. Utilisez: cat, dog, wild")
        return
    
    class_idx = class_mapping[class_name]
    
    # Charger le modèle de diffusion avec sélection
    config, score_model, ema, state = load_model(model_name, interactive=interactive)

    if config is None:
        return
    
    # Charger le classifier
    try:
        classifier = load_afhq_classifier(config.device)
    except Exception as e:
        logging.error(f"❌ Erreur chargement classifier: {e}")
        logging.info("💡 Entraînez d'abord le classifier avec: python train_afhq_classifier.py")
        return
    
    # Setup SDE
    sde, sampling_eps = setup_sde(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Récupérer les paramètres de stratégie depuis la config
    guidance_strategy = getattr(config, 'guidance_strategy', 'truncation')
    adaptive_sigma_limit = getattr(config, 'adaptive_sigma_limit', 50.0)
    
    logging.info(f"🚀 Stratégie de guidance: {guidance_strategy}")
    if guidance_strategy == "adaptive_scale":
        logging.info(f"📏 Limite sigma adaptative: {adaptive_sigma_limit}")

    # Setup conditional sampler avec stratégie
    conditional_sampler = get_pc_conditional_sampler(
        sde=sde,
        classifier=classifier,
        shape=(num_samples, config.data.num_channels, config.data.image_size, config.data.image_size),
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
        guidance_scale=guidance_scale,
        guidance_strategy=guidance_strategy,
        adaptive_sigma_limit=adaptive_sigma_limit
    )
    
    # Utiliser EMA
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    
    # Configurer les seeds
    seeds_used = set_seeds_for_sampling(num_samples)
    log_seed_info(num_samples, seeds_used)

    logging.info(f"🔄 Génération de {num_samples} échantillons de classe '{class_name}'...")

    # Labels pour toutes les images (même classe)
    labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=config.device)

    # MODIFICATION: Génération image par image avec seeds fixes
    if seeds_used is not None:
        # Génération avec seeds fixes
        samples_list = []
        for i in range(num_samples):
            used_seed = apply_seed_before_generation(i, seeds_used)
            single_label = labels[i:i+1]  # Un seul label
            
            # Sampler pour une seule image
            single_sampler = get_pc_conditional_sampler(
                sde=sde,
                classifier=classifier,
                shape=(1, config.data.num_channels, config.data.image_size, config.data.image_size),
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
                guidance_scale=guidance_scale,
                guidance_strategy=guidance_strategy,
                adaptive_sigma_limit=adaptive_sigma_limit
            )
            
            single_sample = single_sampler(score_model, single_label)
            samples_list.append(single_sample)
        
        # Concaténer tous les samples
        samples = torch.cat(samples_list, dim=0)
    else:
        # Génération normale (batch complet)
        samples = conditional_sampler(score_model, labels)

    SIGMA_MAX_CLASSIFIER, SIGMA_LIMIT_PRED_MIN, SIGMA_LIMIT_PRED_MAX = range_sigmas_active(labels)
    
    ema.restore(score_model.parameters())
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"samples/{model_name}/controllable"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer la grille
    grid = make_grid(samples, nrow=int(np.sqrt(num_samples)), padding=2, normalize=True)
    
    # Nom de fichier avec info stratégie
    guidance_str = f"_guidance{guidance_scale:.1f}" if guidance_scale != 1.0 else ""
    strategy_str = f"_{guidance_strategy}"
    if guidance_strategy == "adaptive_scale":
        strategy_str += f"_limit{adaptive_sigma_limit:.0f}"
    
    grid_path = os.path.join(output_dir, f"controllable_{model_name}_{class_name}{guidance_str}{strategy_str}_{timestamp}.png")
    save_image(grid, grid_path)

    # Convertir les échantillons en numpy pour sauvegarde NPZ
    samples_np = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)

    # Sauvegarder en NPZ avec info stratégie
    npz_path = os.path.join(output_dir, f"controllable_{model_name}_{class_name}{guidance_str}{strategy_str}_{timestamp}.npz")
    np.savez_compressed(npz_path,
                        samples=samples_np,
                        model=model_name,
                        task='controllable',
                        class_name=class_name,
                        class_idx=class_idx,
                        guidance_scale=guidance_scale,
                        guidance_strategy=guidance_strategy,
                        adaptive_sigma_limit=adaptive_sigma_limit)

    # Calculer la taille
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    logging.info(f"📦 Archive NPZ sauvegardée: {os.path.basename(npz_path)} ({npz_size:.1f} MB)")
    
    logging.info(f"✅ Résultats sauvegardés: {grid_path}")




#Inpaiting, colorization et controllable
"""
python pretrained_controllable_gen.py church inpainting --mask-type light --mask-shape square --mask-percentage 20.0 --image-id 1 ; python pretrained_controllable_gen.py church inpainting --mask-type light --mask-shape rectangle --mask-percentage 80.0 --image-id 5 ;python pretrained_controllable_gen.py church colorization  --image-id 3; python pretrained_controllable_gen.py church colorization --image-id 10 ;sample=16;
python pretrained_controllable_gen.py afhq_512 controllable --class wild --num-samples $sample --guidance-scale 1.0


python pretrained_controllable_gen.py celebahq_256 inpainting --mask-type light --mask-shape circle --mask-percentage 20.0 --image-id  6; python pretrained_controllable_gen.py celebahq_256 inpainting --mask-type light --mask-shape random_patches --mask-percentage 80.0 --image-id 10 ;python pretrained_controllable_gen.py celebahq_256 colorization --image-id  3; python pretrained_controllable_gen.py celebahq_256 colorization  --image-id 5;sample=16;
python pretrained_controllable_gen.py afhq_512 controllable --class cat --num-samples $sample --guidance-scale 1.0


python pretrained_controllable_gen.py ffhq_1024 inpainting --mask-type light --mask-shape random_patches --mask-percentage 20.0 --image-id 7; python pretrained_controllable_gen.py ffhq_1024 inpainting --mask-type light --mask-shape eight --mask-percentage 80.0 --image-id 3 ;python pretrained_controllable_gen.py ffhq_1024 colorization  --image-id 10; python pretrained_controllable_gen.py ffhq_1024 colorization  --image-id 9;sample=16;
python pretrained_controllable_gen.py afhq_512 controllable --class dog --num-samples $sample --guidance-scale 1.0;


python pretrained_sampling.py afhq_512 25; python pretrained_controllable_gen.py afhq_512 inpainting --mask-type light --mask-shape triangle --mask-percentage 20.0 --image-id 26; python pretrained_controllable_gen.py afhq_512 inpainting --mask-type light --mask-shape eight --mask-percentage 80.0 --image-id 6 ;python pretrained_controllable_gen.py afhq_512 colorization  --image-id 9; python pretrained_controllable_gen.py afhq_512 colorization  --image-id 3

"""


def main():
    parser = argparse.ArgumentParser(description='Test de controllable generation pour modèles pré-entraînés')
    parser.add_argument('model', choices=['church', 'celebahq_256', 'ffhq_1024', 'afhq_512'],
                       help='Modèle à utiliser')
    parser.add_argument('task', choices=['inpainting', 'colorization', 'controllable'], 
                       help='Type de test à effectuer')
    parser.add_argument('--mask-type', choices=['light', 'heavy', 'random'], default='random',
                       help='Type de masque pour inpainting (défaut: random)')
    parser.add_argument('--mask-shape', choices=['square', 'rectangle', 'triangle', 'eight', 'circle', 'random_patches', 'random'], 
                       default='random', help='Forme du masque (défaut: random)')
    parser.add_argument('--mask-percentage', type=float,
                       help='Pourcentage exact du masque (surcharge mask-type)')
    parser.add_argument('--image-id', type=int, default=1,
                       help='ID de l\'image à utiliser (défaut: 1)')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Nombre d\'échantillons à générer (défaut: 4)')
    parser.add_argument('--class', dest='class_name', choices=['cat', 'dog', 'wild'], 
                       help='Classe pour génération conditionnelle (AFHQ seulement)')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                       help='Guidance scaling factor (défaut: 1.0)')
    parser.add_argument('--auto', action='store_true',
                       help='Mode automatique (pas de sélection interactive pour AFHQ)')
    
    args = parser.parse_args()
    
    # Mode interactif ou automatique
    interactive = not args.auto
    
    # Validation des arguments
    if args.task == 'controllable':
        if args.model != 'afhq_512':
            logging.error("❌ --task controllable disponible seulement avec --model afhq_512")
            return 1
        if not args.class_name:
            logging.error("❌ --class requis pour --task controllable (cat/dog/wild)")
            return 1
    
    # Créer la structure des dossiers
    create_samples_structure()
    
    # Affichage des informations
    logging.info("="*60)
    logging.info("🎯 CONTROLLABLE GENERATION")
    logging.info("="*60)
    logging.info(f"Modèle: {args.model}")
    logging.info(f"Tâche: {args.task}")
    if args.task == 'controllable':
        logging.info(f"Classe: {args.class_name}")
        logging.info(f"Guidance scale: {args.guidance_scale}")
    else:
        logging.info(f"Image: #{args.image_id}")
    logging.info(f"Échantillons: {args.num_samples}")
    if args.task == 'inpainting':
        logging.info(f"Type masque: {args.mask_type}")
        logging.info(f"Forme masque: {args.mask_shape}")
        if args.mask_percentage:
            logging.info(f"Pourcentage masque: {args.mask_percentage}%")
    if args.auto:
        logging.info("🤖 Mode automatique activé")
    logging.info("")
    
    try:
        if args.task == 'inpainting':
            test_inpainting(args.model, args.image_id, args.mask_type, 
                          args.mask_percentage, args.num_samples, args.mask_shape, interactive)
        elif args.task == 'colorization':
            test_colorization(args.model, args.image_id, args.num_samples, interactive)
        elif args.task == 'controllable':  
            test_controllable(args.model, args.image_id, args.class_name, args.num_samples, args.guidance_scale, interactive)
            
    except Exception as e:
        logging.error(f"❌ Erreur: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    logging.info("🎉 Test terminé avec succès!")
    return 0


if __name__ == "__main__":
    main()