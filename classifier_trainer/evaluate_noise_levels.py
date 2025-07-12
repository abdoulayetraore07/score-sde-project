#!/usr/bin/env python3
"""
Script séparé pour évaluer l'accuracy vs noise scale du classifier AFHQ
MODIFIÉ: Utilise les 7 niveaux optimisés (5 linspace + 2 zone critique)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import logging
from tqdm import tqdm
import argparse

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Ajouter le chemin
sys.path.append('.')

# Imports locaux
from models.afhq_classifier import create_afhq_classifier
from sde_lib import VESDE


SIGMA_MAX = 500


class AFHQEvalDataset(Dataset):
    """Dataset AFHQ simple pour évaluation."""
    
    def __init__(self, data_dir, split='val', transform=None, sde=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sde = sde
        
        self.classes = ['cat', 'dog', 'wild']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()
        
        logging.info(f"📊 Dataset {split}: {len(self.samples)} images")
    
    def _load_samples(self):
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)


def load_trained_classifier(checkpoint_path, device):
    """Charge le classifier entraîné."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    logging.info(f"📊 Chargement du classifier: {checkpoint_path}")
    
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Créer le modèle
    model = create_afhq_classifier(
        pretrained=False,  # On charge nos poids
        freeze_backbone=False,
        embedding_size=128
    )
    
    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 'Unknown')
    epoch = checkpoint.get('epoch', 'Unknown')
    
    logging.info(f"✅ Classifier chargé - Epoch: {epoch}")
    logging.info(f"   → Val Acc: {val_acc:.2f}%")
    
    return model


def get_optimal_noise_levels(sigma_min=0.01, sigma_max=SIGMA_MAX):
    """Génère les 12 niveaux optimisés (5 linspace + 7 zone critique)."""
    # 5 points linspace sur range complet
    base_sigmas = np.linspace(sigma_min, sigma_max, 5)
    
    # 7 points zone critique comme vous voulez
    critical_sigmas = [7.0, 16, 20, 30, 40, 50, 200]
    
    # Combiner et trier
    all_sigmas = sorted(list(base_sigmas) + critical_sigmas)
    
    return all_sigmas  # 12 niveaux total


def evaluate_by_noise_levels(model, dataset, sde, device, num_samples=1000, custom_noise_levels=None):
    """Évalue l'accuracy du classifier par niveau de bruit."""
    model.eval()
    
    # Utiliser les niveaux optimisés ou custom
    if custom_noise_levels is not None:
        noise_levels = custom_noise_levels
        logging.info("📊 Utilisation des niveaux de bruit personnalisés")
    else:
        noise_levels = get_optimal_noise_levels(sde.sigma_min, sde.sigma_max)
        logging.info("📊 Utilisation des 12 niveaux optimisés (5 linspace + 7 zone critique)")
    
    results = {}
    
    logging.info(f"📊 Évaluation sur {len(noise_levels)} niveaux de bruit...")
    logging.info(f"   → Niveaux: {[f'{n:.2f}' for n in noise_levels]}")
    
    # Prendre un échantillon random du dataset
    total_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), total_samples, replace=False)
    
    for noise_scale in tqdm(noise_levels, desc="Noise levels"):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for idx in indices:
                # Charger image et label originaux
                img_path, label = dataset.samples[idx]
                image = Image.open(img_path).convert('RGB')
                image = dataset.transform(image)
                
                # Calculer le temps correspondant à ce noise_scale
                # Pour VESDE: noise_scale = sigma_min * (sigma_max/sigma_min)^t
                # Donc: t = log(noise_scale/sigma_min) / log(sigma_max/sigma_min)
                sigma_min, sigma_max = sde.sigma_min, sde.sigma_max
                if noise_scale <= sigma_min:
                    t = torch.tensor([0.001], dtype=torch.float32)
                elif noise_scale >= sigma_max:
                    t = torch.tensor([0.999], dtype=torch.float32)          
                else:
                    t_val = np.log(noise_scale / sigma_min) / np.log(sigma_max / sigma_min)
                    t = torch.tensor([t_val], dtype=torch.float32)
                
                # Appliquer CE niveau de bruit spécifique
                mean, current_noise_scale = sde.marginal_prob(image.unsqueeze(0), t)
                noise = torch.randn_like(mean)
                perturbed_image = mean + current_noise_scale[:, None, None, None] * noise
                
                # Prédiction
                perturbed_image = perturbed_image.to(device)
                current_noise_scale = current_noise_scale.to(device)
                
                outputs = model(perturbed_image, current_noise_scale)
                _, predicted = torch.max(outputs.data, 1)
                
                correct += (predicted.item() == label)
                total += 1
        
        accuracy = 100. * correct / total
        results[noise_scale] = accuracy
        logging.info(f"  Noise scale {noise_scale:8.2f}: {accuracy:5.1f}%")
    
    return results


def plot_accuracy_vs_noise_optimized(results, save_path, title_suffix=""):
    """Plot la courbe accuracy vs noise scale avec les niveaux optimisés."""
    noise_scales = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(14, 8))
    
    # Couleurs différentes pour les différents types de points
    base_points = []
    critical_points = []
    
    # Identifier les points critiques (approximativement)
    for i, noise in enumerate(noise_scales):
        if 5 < noise < 10 or 1500 < noise < 2500:  # Zone critique
            critical_points.append((noise, accuracies[i]))
        else:
            base_points.append((noise, accuracies[i]))
    
    # Plot des points de base
    if base_points:
        base_x, base_y = zip(*base_points)
        plt.semilogx(base_x, base_y, 'bo-', linewidth=3, markersize=10, 
                    markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2,
                    label='Points linspace')
    
    # Plot des points critiques
    if critical_points:
        crit_x, crit_y = zip(*critical_points)
        plt.semilogx(crit_x, crit_y, 'ro-', linewidth=3, markersize=12, 
                    markerfacecolor='orange', markeredgecolor='red', markeredgewidth=2,
                    label='Zone critique (t=0.4, t=0.8)')
    
    # Ligne de connexion pour toute la courbe
    plt.semilogx(noise_scales, accuracies, 'k--', linewidth=1, alpha=0.5)
    
    plt.xlabel('Noise Scale (σ)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title(f'AFHQ Classifier: Accuracy vs Noise Scale{title_suffix}\n'
              f'12 Niveaux Optimisés (5 linspace + 7 zone critique)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 105)
    plt.xlim(0.008, max(noise_scales) * 1.2)
    
    # Annotations pour tous les points
    for i, (noise_scale, acc) in enumerate(zip(noise_scales, accuracies)):
        color = 'red' if (5 < noise_scale < 10 or 1500 < noise_scale < 2500) else 'blue'
        plt.annotate(f'{acc:.1f}%', 
                    (noise_scale, acc),
                    textcoords="offset points", 
                    xytext=(0, 15), 
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    color=color, fontweight='bold')
    
    # Zones colorées pour performance
    plt.axhspan(90, 100, alpha=0.1, color='green', label='Excellent (90-100%)')
    plt.axhspan(70, 90, alpha=0.1, color='orange', label='Good (70-90%)')
    plt.axhspan(0, 70, alpha=0.1, color='red', label='Poor (<70%)')
    
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📈 Courbe sauvée: {save_path}")


def evaluate_by_class_and_noise(model, dataset, sde, device, num_samples_per_class=200, custom_noise_levels=None):
    """Évalue l'accuracy par classe ET par niveau de bruit."""
    model.eval()
    
    # Utiliser les niveaux optimisés ou custom
    if custom_noise_levels is not None:
        noise_levels = custom_noise_levels
    else:
        # Utiliser seulement 5 niveaux clés pour l'analyse par classe
        all_levels = get_optimal_noise_levels(sde.sigma_min, sde.sigma_max)
        noise_levels = [all_levels[0], all_levels[1], all_levels[3], all_levels[5], all_levels[6]]  # 5 niveaux représentatifs
    
    classes = ['cat', 'dog', 'wild']
    results = {cls: {} for cls in classes}
    
    logging.info("📊 Évaluation par classe et niveau de bruit...")
    logging.info(f"   → Niveaux testés: {[f'{n:.2f}' for n in noise_levels]}")
    
    # Séparer les échantillons par classe
    samples_by_class = {0: [], 1: [], 2: []}
    for idx, (img_path, label) in enumerate(dataset.samples):
        samples_by_class[label].append(idx)
    
    for class_idx, class_name in enumerate(classes):
        logging.info(f"🎯 Évaluation classe: {class_name}")
        
        # Échantillonner pour cette classe
        class_indices = samples_by_class[class_idx]
        selected_indices = np.random.choice(
            class_indices, 
            min(num_samples_per_class, len(class_indices)), 
            replace=False
        )
        
        for noise_scale in noise_levels:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for idx in selected_indices:
                    # Charger image
                    img_path, label = dataset.samples[idx]
                    image = Image.open(img_path).convert('RGB')
                    image = dataset.transform(image)
                    
                    # Calculer temps pour ce noise_scale
                    sigma_min, sigma_max = sde.sigma_min, sde.sigma_max
                    if noise_scale <= sigma_min:
                        t = torch.tensor([0.001], dtype=torch.float32)
                    elif noise_scale >= sigma_max:
                        t = torch.tensor([0.999], dtype=torch.float32)
                    else:
                        t_val = np.log(noise_scale / sigma_min) / np.log(sigma_max / sigma_min)
                        t = torch.tensor([t_val], dtype=torch.float32)
                    
                    # Perturber
                    mean, current_noise_scale = sde.marginal_prob(image.unsqueeze(0), t)
                    noise = torch.randn_like(mean)
                    perturbed_image = mean + current_noise_scale[:, None, None, None] * noise
                    
                    # Prédiction
                    perturbed_image = perturbed_image.to(device)
                    current_noise_scale = current_noise_scale.to(device)
                    
                    outputs = model(perturbed_image, current_noise_scale)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    correct += (predicted.item() == label)
                    total += 1
            
            accuracy = 100. * correct / total
            results[class_name][noise_scale] = accuracy
            logging.info(f"  {class_name} @ σ={noise_scale:8.2f}: {accuracy:5.1f}%")
    
    return results


def compute_weighted_score(results):
    """Calcule le score pondéré comme dans les autres scripts."""
    noise_levels = sorted(results.keys())
    accuracies = [results[noise] for noise in noise_levels]
    
    # VOS 12 poids si 12 niveaux, sinon uniforme
    if len(accuracies) == 12:
        weights = [0.25, 0.25, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05]
        weights_base = np.sum(weights)  # = 2.25
        weighted_score = sum(w * acc for w, acc in zip(weights, accuracies)) / weights_base
    else:
        # Fallback: poids uniformes
        weights = [1.0/len(accuracies)] * len(accuracies)
        weighted_score = sum(w * acc for w, acc in zip(weights, accuracies))
    
    return weighted_score


def main():
    """Script principal d'évaluation."""
    
    parser = argparse.ArgumentParser(description='Évaluation AFHQ Classifier avec niveaux optimisés')
    parser.add_argument('--data-dir', default='data/afhq', help='Dossier dataset AFHQ')
    parser.add_argument('--classifier-path', default='experiments/afhq_classifier/afhq_classifier.pth',
                       help='Chemin vers le checkpoint du classifier')
    parser.add_argument('--output-dir', default='experiments/afhq_classifier',
                       help='Dossier de sortie pour les résultats')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Nombre d\'images à tester (défaut: 1000)')
    parser.add_argument('--device', default='auto',
                       help='Device à utiliser (auto/cuda/cpu)')
    parser.add_argument('--custom-levels', nargs='+', type=float,
                       help='Niveaux de bruit personnalisés (optionnel)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'image_size': 512,
        'device': 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device,
        'classifier_path': args.classifier_path,
        'output_dir': args.output_dir,
        'num_samples': args.num_samples,
    }
    
    logging.info("🎯 AFHQ Classifier - Évaluation avec Niveaux Optimisés")
    logging.info("=" * 60)
    logging.info(f"Device: {config['device']}")
    logging.info(f"Classifier: {config['classifier_path']}")
    logging.info(f"Échantillons: {config['num_samples']}")
    
    # Setup SDE
    sde = VESDE(sigma_min=0.01, sigma_max=SIGMA_MAX, N=1000)
    logging.info("✅ SDE configuré: VESDE (σ_min=0.01, σ_max=500)")
    
    # Afficher les niveaux qui seront testés
    if args.custom_levels:
        test_levels = sorted(args.custom_levels)
        logging.info(f"📊 Niveaux personnalisés: {[f'{n:.2f}' for n in test_levels]}")
    else:
        test_levels = get_optimal_noise_levels(sde.sigma_min, sde.sigma_max)
        logging.info(f"📊 Niveaux optimisés: {[f'{n:.2f}' for n in test_levels]}")
    
    # Dataset de validation
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    dataset = AFHQEvalDataset(
        config['data_dir'], 
        split='val', 
        transform=val_transform,
        sde=sde
    )
    
    # Charger le classifier
    device = torch.device(config['device'])
    model = load_trained_classifier(config['classifier_path'], device)
    
    # Évaluation 1: Accuracy vs Noise Scale
    logging.info("\n📊 1. Évaluation globale par niveau de bruit...")
    noise_results = evaluate_by_noise_levels(
        model, dataset, sde, device, config['num_samples'], args.custom_levels
    )
    
    # Calculer le score pondéré
    weighted_score = compute_weighted_score(noise_results)
    logging.info(f"📊 Score pondéré global: {weighted_score:.2f}%")
    
    # Plot principal
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(config['output_dir'], f'accuracy_vs_noise_optimized_{timestamp}.png')
    title_suffix = f" ({len(test_levels)} niveaux)"
    plot_accuracy_vs_noise_optimized(noise_results, plot_path, title_suffix)
    
    # Évaluation 2: Par classe et par bruit
    logging.info("\n📊 2. Évaluation par classe et niveau de bruit...")
    class_results = evaluate_by_class_and_noise(model, dataset, sde, device, 200, args.custom_levels)
    
    # Sauvegarder tous les résultats
    results_path = os.path.join(config['output_dir'], f'noise_analysis_optimized_{timestamp}.txt')
    with open(results_path, 'w') as f:
        f.write("=== AFHQ Classifier - Évaluation avec Niveaux Optimisés ===\n\n")
        
        f.write("1. NIVEAUX TESTÉS\n")
        f.write("Type: 5 linspace [0.01, 500] + 7 zone critique\n")
        for noise_scale in sorted(noise_results.keys()):
            f.write(f"σ={noise_scale:.2f}\n")
        
        f.write(f"\n2. SCORE PONDÉRÉ GLOBAL: {weighted_score:.2f}%\n")
        
        f.write("\n3. RÉSULTATS GLOBAUX\n")
        f.write("Noise Scale\tAccuracy (%)\n")
        for noise_scale in sorted(noise_results.keys()):
            acc = noise_results[noise_scale]
            f.write(f"{noise_scale:.2f}\t\t{acc:.1f}\n")
        
        f.write("\n4. RÉSULTATS PAR CLASSE\n")
        for class_name, class_data in class_results.items():
            f.write(f"\n{class_name.upper()}:\n")
            for noise_scale in sorted(class_data.keys()):
                acc = class_data[noise_scale]
                f.write(f"  σ={noise_scale:.2f}: {acc:.1f}%\n")
    
    logging.info(f"📊 Résultats complets sauvés: {results_path}")
    
    # Résumé
    logging.info("\n🎯 RÉSUMÉ:")
    logging.info(f"  Score pondéré: {weighted_score:.2f}%")
    logging.info(f"  Meilleure accuracy: {max(noise_results.values()):.1f}%")
    logging.info(f"  Accuracy à σ_max (500): {noise_results.get(max(noise_results.keys()), 'N/A'):.1f}%")
    logging.info(f"  Courbe générée: {plot_path}")
    logging.info(f"  Niveaux testés: {len(test_levels)}")
    logging.info("✅ Évaluation terminée!")


if __name__ == "__main__":
    main()