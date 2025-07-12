#!/usr/bin/env python3
"""
Script pour figer les seeds par image avec option on/off.
"""

import torch
import numpy as np
import random

# Configuration globale
FIXED_SEEDS = [65, 103, 487, 987]  # Seeds fixes pour les 4 premières images
USE_FIXED_SEEDS = False  # Mettre False pour désactiver

def set_seeds_for_sampling(num_samples, use_fixed=None):
    """
    Configure les seeds pour le sampling.
    
    Args:
        num_samples: nombre d'images à générer
        use_fixed: True/False pour forcer, None pour utiliser USE_FIXED_SEEDS
    
    Returns:
        Liste des seeds utilisés
    """
    global USE_FIXED_SEEDS, FIXED_SEEDS
    
    if use_fixed is None:
        use_fixed = USE_FIXED_SEEDS
    
    if not use_fixed:
        # Seeds aléatoires normaux
        return None
    
    # Utiliser les seeds fixes pour les premières images
    seeds_used = []
    for i in range(num_samples):
        if i < len(FIXED_SEEDS):
            seed = FIXED_SEEDS[i]
        else:
            # Pour les images au-delà de 4, utiliser un pattern
            seed = FIXED_SEEDS[i % len(FIXED_SEEDS)] + (i // len(FIXED_SEEDS)) * 1000
        
        seeds_used.append(seed)
    
    return seeds_used

def apply_seed_before_generation(sample_idx, seeds_list=None):
    """
    Applique le seed avant de générer une image spécifique.
    
    Args:
        sample_idx: index de l'image (0, 1, 2, 3...)
        seeds_list: liste des seeds à utiliser (si None, pas de seed fixe)
    """
    if seeds_list is None:
        return None
    
    if sample_idx < len(seeds_list):
        seed = seeds_list[sample_idx]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed
    
    return None

def set_global_seed_policy(use_fixed_seeds):
    """
    Active/désactive l'utilisation des seeds fixes globalement.
    
    Args:
        use_fixed_seeds: True pour activer, False pour désactiver
    """
    global USE_FIXED_SEEDS
    USE_FIXED_SEEDS = use_fixed_seeds

def log_seed_info(num_samples, seeds_used):
    """
    Affiche les informations sur les seeds utilisés.
    """
    import logging
    
    if seeds_used is None:
        logging.info("🎲 Seeds: ALÉATOIRES (non figés)")
    else:
        logging.info(f"🎲 Seeds figés: {seeds_used[:min(4, len(seeds_used))]}")
        if len(seeds_used) > 4:
            logging.info(f"   ... et {len(seeds_used)-4} autres")