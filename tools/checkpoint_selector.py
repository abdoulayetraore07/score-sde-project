#!/usr/bin/env python3
"""
Utilitaire pour sélectionner les checkpoints AFHQ selon les itérations.
"""

import os
import re
import logging
from typing import Dict, Optional, Tuple
import random

USE_CUSTOM = False
RANDOM_CHOICE = False

def list_afhq_checkpoints(checkpoint_dir: str = "experiments/afhq_512_ncsnpp_continuous/checkpoints") -> Dict[int, str]:
    """
    Liste tous les checkpoints AFHQ disponibles.
    
    Returns:
        Dict[iterations, filepath]: Mapping itérations -> chemin du fichier
    """
    checkpoints = {}
    
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            
            # Extraire le numéro d'itération
            if filename == 'checkpoint.pth':
                # Checkpoint par défaut = 520k itérations
                checkpoints[520000] = filepath
            elif match := re.match(r'checkpoint_(\d+)\.pth', filename):
                # checkpoint_X.pth = X * 20000 itérations
                iter_num = int(match.group(1))
                iterations = iter_num * 20000
                checkpoints[iterations] = filepath
    
    return checkpoints


def select_afhq_checkpoint(model_name: str, interactive: bool = True) -> Optional[str]:
    """
    Sélectionne un checkpoint AFHQ.
    
    Args:
        model_name: Nom du modèle (doit contenir 'afhq')
        interactive: Si True, demande à l'utilisateur
        
    Returns:
        Chemin vers le checkpoint sélectionné, ou None si aucun
    """
   
    if 'afhq' not in model_name.lower():
        # Pas un modèle AFHQ, utiliser le comportement par défaut
        return None
    
    checkpoints = list_afhq_checkpoints()
    
    if not checkpoints:
        logging.warning("❌ Aucun checkpoint AFHQ trouvé")
        return None
    
    if len(checkpoints) == 1 :
        # Un seul checkpoint disponible
        iterations, filepath = list(checkpoints.items())[0]
        logging.info(f"✅ Checkpoint unique trouvé ou par défaut: {iterations:,} itérations")
        return filepath
    
    if not interactive or not USE_CUSTOM:
        # Mode non-interactif, prendre le plus récent
        if not RANDOM_CHOICE :
            max_iter = max(checkpoints.keys())
            logging.info(f"🔄 Mode auto: utilisation du checkpoint le plus récent ({max_iter:,} itérations)")
            return checkpoints[max_iter]
        else :
            iter = random.choice([500000,520000])
            logging.info(f"🔄 Mode auto: utilisation du checkpoint aléatoire ({iter:,} itérations)")
            return checkpoints[iter]
    
    # Mode interactif - demander à l'utilisateur
    print("\n" + "="*60)
    print("🎯 SÉLECTION DU CHECKPOINT AFHQ")
    print("="*60)
    print("Checkpoints disponibles:")
    
    # Trier par itérations
    sorted_checkpoints = sorted(checkpoints.items())
    
    for i, (iterations, filepath) in enumerate(sorted_checkpoints, 1):
        filename = os.path.basename(filepath)
        print(f"  {i}. {filename:<20} → {iterations:>8,} itérations")
    
    print(f"  0. Annuler")
    print("")
    
    while True:
        try:
            choice = input("Choisissez un checkpoint (numéro): ").strip()
            
            if choice == '0':
                print("❌ Sélection annulée")
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sorted_checkpoints):
                iterations, filepath = sorted_checkpoints[choice_idx]
                filename = os.path.basename(filepath)
                print(f"✅ Checkpoint sélectionné: {filename} ({iterations:,} itérations)")
                print("="*60)
                return filepath
            else:
                print(f"❌ Choix invalide. Entrez un numéro entre 1 et {len(sorted_checkpoints)}")
                
        except ValueError:
            print("❌ Veuillez entrer un numéro valide")
        except KeyboardInterrupt:
            print("\n❌ Sélection annulée")
            return None


def get_checkpoint_info(checkpoint_path: str) -> Tuple[int, str]:
    """
    Extrait les informations d'un checkpoint.
    
    Returns:
        (iterations, filename)
    """
    filename = os.path.basename(checkpoint_path)
    
    if filename == 'checkpoint.pth':
        return 520000, filename
    elif match := re.match(r'checkpoint_(\d+)\.pth', filename):
        iter_num = int(match.group(1))
        iterations = iter_num * 20000
        return iterations, filename
    else:
        return 0, filename


def copy_checkpoint_to_meta(checkpoint_path: str, model_name: str):
    """
    Copie le checkpoint sélectionné vers le dossier checkpoints-meta.
    """
    if 'afhq' not in model_name.lower():
        return
    
    meta_dir = "experiments/afhq_512_ncsnpp_continuous/checkpoints-meta"
    meta_checkpoint = os.path.join(meta_dir, "checkpoint.pth")
    
    os.makedirs(meta_dir, exist_ok=True)
    
    # Copier le checkpoint sélectionné
    import shutil
    shutil.copy2(checkpoint_path, meta_checkpoint)
    
    iterations, filename = get_checkpoint_info(checkpoint_path)
    logging.info(f"📋 Checkpoint actif: {filename} ({iterations:,} itérations)")
    logging.info(f"📁 Copié vers: {meta_checkpoint}")


if __name__ == "__main__":
    # Test de la sélection
    selected = select_afhq_checkpoint("afhq_512", interactive=True)
    if selected:
        copy_checkpoint_to_meta(selected, "afhq_512")
        print(f"✅ Prêt à utiliser: {selected}")
    else:
        print("❌ Aucun checkpoint sélectionné")