#!/usr/bin/env python3
"""
Script pour la gestion des seeds avec génération batch.
"""

import torch
import numpy as np
import random

# Configuration globale
FIXED_SEEDS = [65, 103, 487, 987, 136, 29, 42, 37]  # Seeds fixes pour les 8 premières images

def configure_batch_seeds(num_samples, seed_mode='fixed'):
    """
    Configure les seeds pour génération batch (RAPIDE).
    
    Args:
        num_samples: nombre d'images à générer
        seed_mode: 'fixed', 'random', ou liste de seeds
    
    Returns:
        tuple: (batch_seed, seeds_info)
            - batch_seed: int pour torch.manual_seed() ou None
            - seeds_info: liste des seeds individuels pour debug
    """
    if seed_mode == 'random':
        return None, None
    
    elif seed_mode == 'fixed':
        # Utiliser les seeds fixes existants
        base_seeds = FIXED_SEEDS  # [65, 103, 487, 987, 136, 29, 42, 37]
        
        # Créer la liste complète des seeds
        full_seeds = []
        for i in range(num_samples):
            if i < len(base_seeds):
                full_seeds.append(base_seeds[i])
            else:
                # Pattern pour images supplémentaires
                base_idx = i % len(base_seeds)
                offset = (i // len(base_seeds)) * 1000
                full_seeds.append(base_seeds[base_idx] + offset)
        
        # Calculer UN seed unique pour tout le batch
        # Méthode déterministe simple
        combined_seed = sum(full_seeds) % 2147483647  # Max int32
        
        return combined_seed, full_seeds
    
    elif isinstance(seed_mode, list):
        # Seeds personnalisés
        if len(seed_mode) >= num_samples:
            user_seeds = seed_mode[:num_samples]
        else:
            # Répéter les seeds si pas assez
            user_seeds = (seed_mode * ((num_samples // len(seed_mode)) + 1))[:num_samples]
        
        combined_seed = sum(user_seeds) % 2147483647
        return combined_seed, user_seeds
    
    else:
        return None, None

def apply_batch_seeds(batch_seed, seeds_info=None):
    """
    Applique les seeds pour génération batch.
    
    Args:
        batch_seed: seed à appliquer ou None
        seeds_info: liste des seeds pour logging
    """
    import logging
    
    if batch_seed is not None:
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        np.random.seed(batch_seed)
        random.seed(batch_seed)
        
        logging.info(f"🎲 Seeds fixes - Batch seed: {batch_seed}")
        if seeds_info:
            displayed_seeds = seeds_info[:8]
            if len(seeds_info) > 8:
                logging.info(f"🎲 Seeds individuels: {displayed_seeds}... (+{len(seeds_info)-8} autres)")
            else:
                logging.info(f"🎲 Seeds individuels: {displayed_seeds}")
    else:
        logging.info("🎲 Seeds: ALÉATOIRES")

def log_seed_strategy(seed_mode, num_samples):
    """Affiche la stratégie de seeds utilisée."""
    import logging
    
    if seed_mode == 'random':
        logging.info("🎯 Stratégie: Seeds aléatoires")
    elif seed_mode == 'fixed':
        logging.info("🎯 Stratégie: Seeds fixes (reproductible)")
    elif isinstance(seed_mode, list):
        logging.info(f"🎯 Stratégie: Seeds personnalisés ({len(seed_mode)} fournis)")
    
    logging.info(f"🎯 Images à générer: {num_samples}")