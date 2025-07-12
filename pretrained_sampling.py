#!/usr/bin/env python3
"""
Script de sampling qui réutilise directement les fonctions existantes du codebase.
Pas de duplication de code - utilise run_lib.py et evaluation.py directement.
"""
import sys
import torch
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(levelname)s - %(filename)s - %(asctime)s - %(message)s')

sys.path.append('.')

# Imports obligatoires pour l'enregistrement des modèles
from models import ncsnpp, ddpm, ncsnv2

# Réutilisation directe des fonctions existantes
import run_lib
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sampling
import sde_lib
import datasets
import losses
from utils import restore_checkpoint
from tools.checkpoint_selector import select_afhq_checkpoint, copy_checkpoint_to_meta

# ✅ IMPORT DU MODULE EXTERNE
from tools.config_modifier import interactive_config_modification, print_config_summary


def print_sampling_summary(config, model_name, num_samples, checkpoint_path, new_config = False):
    """Affiche un résumé de la configuration de sampling."""
    
    logging.info("="*80)
    logging.info("EDITED SAMPLING CONFIGURATION SUMMARY")  if new_config else logging.info("SAMPLING CONFIGURATION SUMMARY")
    logging.info("="*80)
    logging.info(f"🎯 Model: {model_name}")
    logging.info(f"📁 Checkpoint: {checkpoint_path}")
    logging.info(f"🖼️  Samples to generate: {num_samples}")
    logging.info(f"📐 Resolution: {config.data.image_size}x{config.data.image_size}")
    logging.info(f"🎨 Sampling method: {config.sampling.method}")
    logging.info(f"📊 Predictor: {config.sampling.predictor}")
    logging.info(f"🔧 Corrector: {config.sampling.corrector}")
    logging.info(f"📈 SNR: {config.sampling.snr}")
    logging.info(f"🧹 Denoising: {config.sampling.noise_removal}")
    logging.info("="*80)
    logging.info("")



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
        raise ValueError(f"Modèle '{model_name}' non supporté. Utilisez 'church' ou 'celebahq_256' ou 'ffhq_1024 ou afhq_512'")


def quick_sample_from_checkpoint(model_name, checkpoint_path, num_samples=8, output_dir=None):
    """
    Génère des samples en réutilisant directement les fonctions de run_lib.py
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    logging.info(f"🎯 Chargement de la configuration pour {model_name}")
    config = load_config(model_name)
    print_sampling_summary(config, model_name, num_samples, checkpoint_path)

    # ✅ MODIFICATION INTERACTIVE EXTERNALISÉE
    config_modified = interactive_config_modification(config)
    
    # ✅ AFFICHAGE FINAL SI MODIFIÉ
    if config_modified:
        print_sampling_summary(config, model_name, num_samples, checkpoint_path, new_config=True)

    # Dossier de sortie
    import os
    if output_dir is None:
        output_dir = f"samples/{model_name}/general/samples_{model_name}_{num_samples}imgs_{timestamp}"
        
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"📦 Création du modèle...")
    
    # RÉUTILISATION: Créer le modèle exactement comme run_lib.py
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    # RÉUTILISATION: Charger le checkpoint exactement comme run_lib.py
    logging.info(f"📊 Chargement du checkpoint: {checkpoint_path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = restore_checkpoint(checkpoint_path, state, device=config.device)
        logging.info("✅ Checkpoint chargé avec succès")
        logging.info(f"   Step: {state['step']:,}")
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement: {e}")
        return None
    
    # RÉUTILISATION: Setup SDE exactement comme run_lib.py (lignes 76-85)
    logging.info("⚙️ Configuration du SDE...")
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
    
    # RÉUTILISATION: Inverse scaler comme run_lib.py
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # RÉUTILISATION: Fonction de sampling comme run_lib.py
    sampling_shape = (num_samples, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    
    # RÉUTILISATION: Génération exactement comme run_lib.py (lignes 150-152)
    logging.info("🎨 Génération des samples...")
    logging.info(f"   Méthode: {config.sampling.method}")
    logging.info(f"   Steps par sample: {sde.N}")

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())

    # Timer pour estimer le temps
    import time
    start_time = time.time()

    sample, nfe = sampling_fn(score_model)

    elapsed_time = time.time() - start_time
    ema.restore(score_model.parameters())

    logging.info(f"✅ Génération terminée en {elapsed_time:.1f}s")
    logging.info(f"   Temps par image: {elapsed_time/num_samples:.1f}s")
    
    logging.info(f"✅ Génération terminée! NFE: {nfe}")
    
    # RÉUTILISATION: Sauvegarde exactement comme run_lib.py (lignes 157-166)
    logging.info("💾 Sauvegarde...")
    
    # Traitement identique à run_lib.py
    from torchvision.utils import make_grid, save_image
    
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = make_grid(sample, nrow, padding=2)
    sample_np = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    
    # Sauvegarde grille
    base_name = f"{model_name}_{config.data.image_size}px_{timestamp}"
    grid_path = os.path.join(output_dir, f"{base_name}_grid.png")
    save_image(image_grid, grid_path)
    
    # Sauvegarde numpy
    np_path = os.path.join(output_dir, f"{base_name}_samples.npz")
    np.savez_compressed(np_path, samples=sample_np, nfe=nfe, model=model_name)
    
    # Images individuelles
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    for i, img in enumerate(sample):
        save_image(img, os.path.join(individual_dir, f"sample_{i:03d}.png"))
    
    # Calculer la taille des fichiers
    grid_size = os.path.getsize(grid_path) / (1024*1024)  # En MB
    npz_size = os.path.getsize(np_path) / (1024*1024)

    logging.info("")
    logging.info("="*60)
    logging.info("✨ SAMPLING COMPLETED SUCCESSFULLY!")
    logging.info("="*60)
    logging.info(f"📁 Output directory: {output_dir}/")
    logging.info(f"🖼️  Images generated: {len(sample)}")
    logging.info(f"📐 Resolution: {config.data.image_size}x{config.data.image_size}")
    logging.info(f"🔢 NFE (function evaluations): {nfe:,}")
    logging.info(f"📊 Files saved:")
    logging.info(f"   - Grid image: {os.path.basename(grid_path)} ({grid_size:.1f} MB)")
    logging.info(f"   - NumPy archive: {os.path.basename(np_path)} ({npz_size:.1f} MB)")
    logging.info(f"   - Individual images: {individual_dir}/")
    logging.info("="*60)
    
    return output_dir


def sample_pretrained_model(model_name, num_samples=4, output_dir=None, interactive=True):
    """
    Interface simple qui utilise les chemins de checkpoint par défaut avec sélection AFHQ.
    """
    # Chemins par défaut basés sur la structure standard
    checkpoint_paths = {
        'church': "experiments/church_ncsnpp_continuous/checkpoints-meta/checkpoint.pth",
        'celebahq_256': "experiments/celebahq_256_ncsnpp_continuous/checkpoints-meta/checkpoint.pth", 
        'ffhq_1024': "experiments/ffhq_1024_ncsnpp_continuous/checkpoints-meta/checkpoint.pth",
        'afhq_512': "experiments/afhq_512_ncsnpp_continuous/checkpoints-meta/checkpoint.pth" 
    }
    
    if model_name not in checkpoint_paths:
        raise ValueError(f"Modèle '{model_name}' non supporté. Disponibles: {list(checkpoint_paths.keys())}")
    
    # NOUVEAU: Sélection interactive pour AFHQ
    if model_name == 'afhq_512':
        selected_checkpoint = select_afhq_checkpoint(model_name, interactive=interactive)
        if selected_checkpoint is None:
            logging.error("❌ Aucun checkpoint AFHQ sélectionné")
            return None
        
        # Copier vers checkpoints-meta pour utilisation
        copy_checkpoint_to_meta(selected_checkpoint, model_name)
    
    checkpoint_path = checkpoint_paths[model_name]
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        logging.info("💡 Vérifiez que le modèle est bien entraîné ou spécifiez un autre chemin")
        return None
    
    return quick_sample_from_checkpoint(model_name, checkpoint_path, num_samples, output_dir)



def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        logging.info("Usage: python pretrained_sampling.py [church|celebahq_256|ffhq_1024|afhq_512] [num_samples] [output_dir] [--auto]")
        logging.info("  --auto: Mode automatique (pas de sélection interactive pour AFHQ)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Vérifier le mode automatique
    interactive = '--auto' not in sys.argv
    
    # Si un checkpoint custom est fourni (5ème argument)
    if len(sys.argv) > 4 and not sys.argv[4].startswith('--'):
        checkpoint_path = sys.argv[4]
        result_dir = quick_sample_from_checkpoint(model_name, checkpoint_path, num_samples, output_dir)
    else:
        result_dir = sample_pretrained_model(model_name, num_samples, output_dir, interactive=interactive)
    
    if result_dir:
        print(f"\n✨ Terminé! Résultats dans: {result_dir}")
    else:
        print("\n❌ Échec de la génération")
        sys.exit(1)


if __name__ == "__main__":
    main()