#!/usr/bin/env python3
"""
Script pour gérer les statistiques dataset avec Clean-FID.
Clean-FID gère automatiquement AFHQ, mais ce script permet des customisations.
"""

import sys
import os
import logging
import argparse

sys.path.append('.')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_clean_fid_availability():
    """Test si Clean-FID est disponible."""
    try:
        from cleanfid import fid
        logging.info("✅ Clean-FID disponible")
        return True
    except ImportError:
        logging.error("❌ Clean-FID non installé. Installer avec: pip install clean-fid")
        return False

def list_available_datasets():
    """Liste les datasets supportés par Clean-FID."""
    datasets = {
        'afhq_cat': [256, 512],
        'afhq_dog': [256, 512], 
        'afhq_wild': [256, 512],
        'ffhq': [256, 1024],
        'cifar10': [32],
        'lsun_church': [256]
        # ... autres datasets
    }
    
    logging.info("📊 Datasets Clean-FID disponibles:")
    for dataset, resolutions in datasets.items():
        logging.info(f"   {dataset}: {resolutions}px")

def create_custom_stats(dataset_path, custom_name, mode="clean"):
    """Créer des stats personnalisées avec Clean-FID."""
    try:
        from cleanfid import fid
        
        logging.info(f"🔄 Création stats personnalisées: {custom_name}")
        logging.info(f"   Dataset path: {dataset_path}")
        logging.info(f"   Mode: {mode}")
        
        fid.make_custom_stats(custom_name, dataset_path, mode=mode)
        
        logging.info("✅ Stats personnalisées créées avec succès")
        return True
        
    except Exception as e:
        logging.error(f"❌ Erreur création stats: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Gestion statistiques Clean-FID')
    parser.add_argument('--list', action='store_true', help='Lister datasets disponibles')
    parser.add_argument('--custom-name', help='Nom pour stats personnalisées')
    parser.add_argument('--dataset-path', help='Chemin dataset pour stats personnalisées')
    parser.add_argument('--mode', default='clean', choices=['clean', 'legacy_pytorch'])
    
    args = parser.parse_args()
    setup_logging()
    
    if not test_clean_fid_availability():
        return 1
    
    if args.list:
        list_available_datasets()
        return 0
    
    if args.custom_name and args.dataset_path:
        success = create_custom_stats(args.dataset_path, args.custom_name, args.mode)
        return 0 if success else 1
    
    logging.info("💡 Usage:")
    logging.info("  --list                    : Lister datasets disponibles")
    logging.info("  --custom-name NAME        : Créer stats personnalisées")
    logging.info("  --dataset-path PATH       : Chemin dataset")
    logging.info("  --mode {clean,legacy}     : Mode Clean-FID")
    
    return 0

if __name__ == "__main__":
    exit(main())