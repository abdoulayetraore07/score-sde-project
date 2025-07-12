#!/usr/bin/env python3
"""
Script pour calculer les likelihoods de toutes les images dans assets/cond_gen_afhq_512/
"""
import sys
import torch
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
from PIL import Image
import pandas as pd
import glob

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(levelname)s - %(filename)s - %(asctime)s - %(message)s')

sys.path.append('.')

# Imports nécessaires
from models import ncsnpp, ddpm, ncsnv2
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
import datasets
import losses
from utils import restore_checkpoint
from likelihood import get_likelihood_fn

# Fixer le seed pour reproductibilité
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(model_name):
    """Charge la configuration selon le modèle."""
    if model_name == 'afhq_512':
        from configs.ve.afhq_512_ncsnpp_continuous import get_config as get_afhq_512_config
        return get_afhq_512_config()
    else:
        raise ValueError(f"Modèle '{model_name}' non supporté. Utilisez 'afhq_512'")

def load_and_preprocess_image(image_path, target_size=512):
    """Charge et préprocess une image pour le calcul de likelihood."""
    try:
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        
        # Redimensionner si nécessaire
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.LANCZOS)
        
        # Convertir en tensor et normaliser
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # Normaliser vers [-1, 1] (standard pour les modèles de diffusion)
        image_tensor = image_tensor * 2.0 - 1.0
        
        return image_tensor
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {image_path}: {e}")
        return None

def calculate_likelihoods_for_directory(model_name, checkpoint_path, images_dir, output_file=None):
    """
    Calcule les likelihoods pour toutes les images dans un dossier.
    """
    set_seed(42)  # Fixer le seed
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    logging.info(f"🎯 Chargement de la configuration pour {model_name}")
    config = load_config(model_name)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Créer le modèle
    logging.info(f"📦 Création du modèle...")
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    # Charger le checkpoint
    logging.info(f"📊 Chargement du checkpoint: {checkpoint_path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = restore_checkpoint(checkpoint_path, state, device=config.device)
        logging.info("✅ Checkpoint chargé avec succès")
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement: {e}")
        return None
    
    # Setup SDE
    logging.info("⚙️ Configuration du SDE...")
    if config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} non supporté pour likelihood.")
    
    # Inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Créer la fonction de likelihood
    logging.info("🧮 Création de la fonction de likelihood...")
    likelihood_fn = get_likelihood_fn(sde, inverse_scaler, 
                                     hutchinson_type='Rademacher',
                                     rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5)
    
    # Mettre le modèle en mode évaluation
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    score_model.eval()
    
    # Trouver toutes les images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    if not image_paths:
        logging.error(f"❌ Aucune image trouvée dans {images_dir}")
        return None
    
    logging.info(f"📸 Trouvé {len(image_paths)} images à traiter")
    
    # Préparer le fichier de résultats
    if output_file is None:
        output_file = f"likelihood_results_{model_name}_{timestamp}.csv"
    
    results = []
    
    # Calculer les likelihoods
    for image_path in tqdm(image_paths, desc="Calcul des likelihoods"):
        image_name = os.path.basename(image_path)
        
        # Charger et préprocesser l'image
        image_tensor = load_and_preprocess_image(image_path, config.data.image_size)
        if image_tensor is None:
            continue
        
        image_tensor = image_tensor.to(config.device)
        
        try:
            # Calculer la likelihood
            with torch.no_grad():
                bpd, z, nfe = likelihood_fn(score_model, image_tensor)
            
            # Convertir en valeurs Python
            bpd_value = bpd.item() if torch.is_tensor(bpd) else bpd
            nfe_value = nfe
            
            results.append({
                'image_name': image_name,
                'image_path': image_path,
                'likelihood_bpd': bpd_value,
                'nfe': nfe_value,
                'timestamp': timestamp
            })
            
            logging.info(f"✅ {image_name}: {bpd_value:.4f} bits/dim (NFE: {nfe_value})")
            
        except Exception as e:
            logging.error(f"❌ Erreur pour {image_name}: {e}")
            results.append({
                'image_name': image_name,
                'image_path': image_path,
                'likelihood_bpd': np.nan,
                'nfe': np.nan,
                'timestamp': timestamp,
                'error': str(e)
            })
    
    # Restaurer les paramètres
    ema.restore(score_model.parameters())
    
    # Sauvegarder les résultats
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Statistiques
    valid_results = df[~df['likelihood_bpd'].isna()]
    if len(valid_results) > 0:
        mean_bpd = valid_results['likelihood_bpd'].mean()
        std_bpd = valid_results['likelihood_bpd'].std()
        min_bpd = valid_results['likelihood_bpd'].min()
        max_bpd = valid_results['likelihood_bpd'].max()
        
        logging.info("")
        logging.info("="*80)
        logging.info("📊 RÉSULTATS DES LIKELIHOODS")
        logging.info("="*80)
        logging.info(f"📁 Dossier traité: {images_dir}")
        logging.info(f"🖼️  Images traitées: {len(valid_results)}/{len(image_paths)}")
        logging.info(f"📊 Likelihood moyenne: {mean_bpd:.4f} ± {std_bpd:.4f} bits/dim")
        logging.info(f"📈 Min: {min_bpd:.4f} bits/dim")
        logging.info(f"📉 Max: {max_bpd:.4f} bits/dim")
        logging.info(f"💾 Résultats sauvés: {output_file}")
        logging.info("="*80)
        
        return output_file
    else:
        logging.error("❌ Aucun résultat valide obtenu")
        return None

def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        logging.info("Usage: python calculate_likelihoods.py [model_name] [checkpoint_path] [images_dir] [output_file]")
        logging.info("  model_name: afhq_512")
        logging.info("  checkpoint_path: chemin vers le checkpoint")
        logging.info("  images_dir: dossier contenant les images (défaut: assets/cond_gen_afhq_512/)")
        logging.info("  output_file: fichier CSV de sortie (optionnel)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "experiments/afhq_512_ncsnpp_continuous/checkpoints-meta/checkpoint.pth"
    images_dir = sys.argv[3] if len(sys.argv) > 3 else "assets/cond_gen_afhq_512/"
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(images_dir):
        logging.error(f"❌ Dossier d'images non trouvé: {images_dir}")
        sys.exit(1)
    
    result_file = calculate_likelihoods_for_directory(model_name, checkpoint_path, images_dir, output_file)
    
    if result_file:
        print(f"\n✨ Terminé! Résultats dans: {result_file}")
    else:
        print("\n❌ Échec du calcul des likelihoods")
        sys.exit(1)

if __name__ == "__main__":
    main()