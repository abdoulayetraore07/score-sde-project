#!/usr/bin/env python3
"""
Test matrice de confusion du classifier AFHQ par niveau de bruit
VERSION MODERNISÉE: 7 niveaux optimisés + weighted score + AFHQDatasetOptimized
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os
from tqdm import tqdm
import logging
import argparse
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# Ajouter le chemin
sys.path.append('.')

# Imports locaux
from models.afhq_classifier import create_afhq_classifier
from sde_lib import VESDE

sigma_max = 500


class ImprovedAugmentations:
    """Augmentations améliorées comme dans les autres scripts."""
    
    @staticmethod
    def get_transforms(image_size, split='train'):
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ])
        else:  # val/test
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])


class AFHQDatasetOptimized(Dataset):
    """Dataset AFHQ optimisé comme dans les autres scripts."""
    
    def __init__(self, data_dir, split='train', transform=None, sde=None, noise_scheduler=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sde = sde
        self.noise_scheduler = noise_scheduler
        
        self.classes = ['cat', 'dog', 'wild']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()
        
        logging.info(f"📊 Dataset {split}: {len(self.samples)} images")
        for i, cls in enumerate(self.classes):
            count = sum(1 for _, label in self.samples if label == i)
            logging.info(f"   → {cls}: {count} images")
    
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
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Charger image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.sde is not None:
            # Sample random time
            t = torch.rand(1) * (self.sde.T - 1e-5) + 1e-5
            
            # Perturber l'image avec SDE
            mean, noise_scale = self.sde.marginal_prob(image.unsqueeze(0), t)
            noise = torch.randn_like(mean)
            perturbed_image = mean + noise_scale[:, None, None, None] * noise
            
            return perturbed_image.squeeze(0), noise_scale.squeeze(0), label
        else:
            # Pour test sans perturbation auto
            return image, torch.tensor([0.0]), label


def find_latest_checkpoint(save_dir):
    """Trouve le dernier checkpoint comme dans les autres scripts."""
    pattern = os.path.join(save_dir, "afhq_classifier*.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Prendre le plus récent
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_classifier(checkpoint_path=None, device='cuda'):
    """Charge le classifier entraîné."""
    if checkpoint_path is None:
        classifier_dir = 'experiments/afhq_classifier'
        checkpoint_path = find_latest_checkpoint(classifier_dir)
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"Aucun checkpoint trouvé dans {classifier_dir}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    logging.info(f"📊 Chargement du classifier: {os.path.basename(checkpoint_path)}")
    
    classifier = create_afhq_classifier(
        pretrained=False,
        freeze_backbone=False,
        embedding_size=128
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.to(device)
    classifier.eval()
    
    accuracy = checkpoint.get('val_acc', 0)
    weighted_score = checkpoint.get('weighted_score', 0)
    epoch = checkpoint.get('epoch', 0)
    
    logging.info(f"✅ Classifier chargé:")
    logging.info(f"   → Epoch: {epoch}")
    logging.info(f"   → Val accuracy: {accuracy:.2f}%")
    logging.info(f"   → Weighted score: {weighted_score:.2f}%")
    
    return classifier


def get_optimal_noise_levels(sigma_min=0.01, sigma_max=sigma_max):
    """Génère les 12 niveaux optimisés (5 linspace + 7 zone critique)."""
    # 5 points linspace sur range complet
    base_sigmas = np.linspace(sigma_min, sigma_max, 5)
    
    # 7 points zone critique 
    critical_sigmas = [7.0, 16, 20, 30, 40, 50, 200]
    
    # Combiner et trier
    all_sigmas = sorted(list(base_sigmas) + critical_sigmas)
    
    return all_sigmas


def compute_weighted_score(results):
    """Calcule le score pondéré comme dans les autres scripts."""
    noise_levels = sorted(results.keys())
    accuracies = [results[noise] for noise in noise_levels]
    
    # VOS 12 poids si 12 niveaux
    if len(accuracies) == 12:
        weights = [0.25, 0.25, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05]
        weights_base = np.sum(weights)  # = 2.25
        weighted_score = sum(w * acc for w, acc in zip(weights, accuracies)) / weights_base
    else:
        # Fallback
        weights = [1.0/len(accuracies)] * len(accuracies)
        weighted_score = sum(w * acc for w, acc in zip(weights, accuracies))
    
    return weighted_score


def test_classifier_by_noise_levels(classifier, dataset, sde, device='cuda', num_samples=1000):
    """Test classifier sur les 7 niveaux optimisés."""
    
    classifier.eval()
    
    # Utiliser les 7 niveaux optimisés
    noise_levels = get_optimal_noise_levels(sde.sigma_min, sde.sigma_max)
    
    logging.info(f"📊 Test sur {len(noise_levels)} niveaux optimisés:")
    logging.info(f"   → Niveaux: {[f'{n:.2f}' for n in noise_levels]}")
    logging.info(f"   → Échantillons: {num_samples}")
    
    # Prendre un échantillon random du dataset
    total_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), total_samples, replace=False)
    
    results = {}
    class_names = ['cat', 'dog', 'wild']
    
    for noise_scale in tqdm(noise_levels, desc="Niveaux de bruit"):
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for idx in indices:
                # Charger image et label originaux
                img_path, label = dataset.samples[idx]
                image = Image.open(img_path).convert('RGB')
                image = dataset.transform(image)
                
                # Calculer le temps correspondant à ce noise_scale
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
                
                logits = classifier(perturbed_image, current_noise_scale)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu().numpy()[0])
                all_labels.append(label)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Calculs metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        
        # Distribution des prédictions
        pred_dist = np.bincount(all_preds, minlength=3) / len(all_preds) * 100
        true_dist = np.bincount(all_labels, minlength=3) / len(all_labels) * 100
        
        results[noise_scale] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': np.array(all_probs),
            'pred_distribution': pred_dist,
            'true_distribution': true_dist
        }
        
        logging.info(f"  σ={noise_scale:8.2f}: {accuracy:5.1f}% | Cat:{pred_dist[0]:.1f}% Dog:{pred_dist[1]:.1f}% Wild:{pred_dist[2]:.1f}%")
    
    return results, class_names



def plot_confusion_matrices_modern(results, class_names, save_path):
    """Plot matrices de confusion avec style moderne pour 12 niveaux."""
    
    noise_levels = sorted(results.keys())
    n_levels = len(noise_levels)
    
    # Grille 3x4 pour 12 niveaux au lieu de 2x4
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
        
    # Style moderne compatible
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
            # Appliquer style manuel
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    for i, noise_level in enumerate(noise_levels):
        cm = results[noise_level]['confusion_matrix']
        accuracy = results[noise_level]['accuracy']
        
        # Normaliser par ligne (true labels)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Couleurs modernes
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[i], cbar_kws={'shrink': 0.8})
        
        # Titre avec informations
        is_critical = (5 < noise_level < 10 or 1500 < noise_level < 2500)
        title_color = 'red' if is_critical else 'blue'
        marker = " 🎯" if is_critical else ""
        
        axes[i].set_title(f'σ={noise_level:.2f}{marker}\nAcc: {accuracy:.1f}%', 
                         fontweight='bold', color=title_color)
        axes[i].set_xlabel('Predicted', fontweight='bold')
        axes[i].set_ylabel('True', fontweight='bold')
    
    # Cacher le dernier subplot (7 niveaux dans grille 2x4)
    if n_levels < 8:
        axes[7].set_visible(False)

    plt.suptitle('AFHQ Classifier - Matrices de Confusion par Niveau de Bruit\n'
                f'12 Niveaux Optimisés (5 linspace + 7 zone critique)', 
                fontsize=16, fontweight='bold', y=0.98)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📈 Matrices de confusion sauvées: {save_path}")


def plot_accuracy_vs_noise_modern(results, save_path):
    """Plot accuracy vs noise level avec style moderne."""
    
    noise_levels = sorted(results.keys())
    accuracies = [results[nl]['accuracy'] for nl in noise_levels]
    
    plt.figure(figsize=(14, 8))
    
    # Style moderne compatible
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
            # Appliquer style manuel
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    # Identifier points critiques
    base_points = []
    critical_points = []
    
    for i, noise in enumerate(noise_levels):
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
                    label='Zone critique 🎯')
    
    # Ligne de connexion
    plt.semilogx(noise_levels, accuracies, 'k--', linewidth=1, alpha=0.5)
    
    # Annotations
    for noise_level, acc in zip(noise_levels, accuracies):
        color = 'red' if (5 < noise_level < 10 or 1500 < noise_level < 2500) else 'blue'
        plt.annotate(f'{acc:.1f}%', 
                    (noise_level, acc),
                    textcoords="offset points", 
                    xytext=(0, 15), 
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    color=color, fontweight='bold')
    
    plt.xlabel('Noise Scale (σ)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('AFHQ Classifier: Accuracy vs Noise Scale\n'
              '7 Niveaux Optimisés + Matrices de Confusion', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.xlim(0.008, max(noise_levels) * 1.2)
    plt.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📈 Courbe accuracy vs noise sauvée: {save_path}")


def analyze_bias_detailed(results, class_names):
    """Analyse détaillée du biais comme dans les autres scripts."""
    
    logging.info("\n" + "="*60)
    logging.info("🔍 ANALYSE DÉTAILLÉE DU BIAIS DU CLASSIFIER")
    logging.info("="*60)
    
    # Calculer le weighted score
    weighted_score = compute_weighted_score({nl: data['accuracy'] for nl, data in results.items()})
    logging.info(f"📊 Score pondéré global: {weighted_score:.2f}%")
    
    # Analyse par niveau
    for noise_level, data in results.items():
        pred_dist = data['pred_distribution']
        true_dist = data['true_distribution']
        accuracy = data['accuracy']
        
        is_critical = (5 < noise_level < 10 or 1500 < noise_level < 2500)
        marker = " 🎯" if is_critical else ""
        
        logging.info(f"\nσ={noise_level:.2f}{marker} (Acc: {accuracy:.1f}%):")
        logging.info(f"  Distribution vraie: cat={true_dist[0]:.1f}%, dog={true_dist[1]:.1f}%, wild={true_dist[2]:.1f}%")
        logging.info(f"  Distribution pred:  cat={pred_dist[0]:.1f}%, dog={pred_dist[1]:.1f}%, wild={pred_dist[2]:.1f}%")
        
        # Biais par classe
        bias = pred_dist - true_dist
        for i, class_name in enumerate(class_names):
            if abs(bias[i]) > 5:
                direction = "sur-représentée" if bias[i] > 0 else "sous-représentée"
                logging.info(f"  ⚠️  {class_name} {direction} de {abs(bias[i]):.1f}%")
    
    # Statistiques globales
    all_accuracies = [data['accuracy'] for data in results.values()]
    logging.info(f"\n📊 STATISTIQUES GLOBALES:")
    logging.info(f"  Accuracy max: {max(all_accuracies):.1f}%")
    logging.info(f"  Accuracy min: {min(all_accuracies):.1f}%")
    logging.info(f"  Accuracy moyenne: {np.mean(all_accuracies):.1f}%")
    logging.info(f"  Écart-type: {np.std(all_accuracies):.1f}%")
    
    # Biais moyen sur tous les niveaux
    avg_pred_dist = np.mean([data['pred_distribution'] for data in results.values()], axis=0)
    avg_true_dist = np.mean([data['true_distribution'] for data in results.values()], axis=0)
    
    logging.info(f"\n🎯 BIAIS MOYEN GLOBAL:")
    logging.info(f"  Distribution vraie: cat={avg_true_dist[0]:.1f}%, dog={avg_true_dist[1]:.1f}%, wild={avg_true_dist[2]:.1f}%")
    logging.info(f"  Distribution pred:  cat={avg_pred_dist[0]:.1f}%, dog={avg_pred_dist[1]:.1f}%, wild={avg_pred_dist[2]:.1f}%")
    
    avg_bias = avg_pred_dist - avg_true_dist
    for i, class_name in enumerate(class_names):
        if abs(avg_bias[i]) > 2:
            direction = "favorisée" if avg_bias[i] > 0 else "défavorisée"
            logging.info(f"  🎯 {class_name} globalement {direction} de {abs(avg_bias[i]):.1f}%")


def save_detailed_report(results, class_names, output_dir, weighted_score):
    """Sauvegarde rapport détaillé comme dans les autres scripts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'confusion_analysis_report_{timestamp}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== AFHQ Classifier - Analyse Matrices de Confusion ===\n")
        f.write("VERSION MODERNISÉE: 7 niveaux optimisés + weighted score\n\n")
        
        f.write("1. CONFIGURATION\n")
        f.write("Type: 5 linspace [0.01, 784] + 2 zone critique (t=0.4, t=0.8)\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Total échantillons: {len(results[list(results.keys())[0]]['labels'])}\n")
        
        f.write(f"\n2. SCORE PONDÉRÉ GLOBAL: {weighted_score:.2f}%\n")
        
        f.write("\n3. RÉSULTATS DÉTAILLÉS PAR NIVEAU\n")
        f.write("Noise Scale\tAccuracy\tCat%\tDog%\tWild%\tBiais\n")
        
        for noise_level in sorted(results.keys()):
            data = results[noise_level]
            pred_dist = data['pred_distribution']
            true_dist = data['true_distribution']
            bias = pred_dist - true_dist
            max_bias = max(abs(bias))
            
            is_critical = (5 < noise_level < 10 or 1500 < noise_level < 2500)
            marker = "*" if is_critical else " "
            
            f.write(f"{noise_level:.2f}{marker}\t\t{data['accuracy']:.1f}%\t\t")
            f.write(f"{pred_dist[0]:.1f}\t{pred_dist[1]:.1f}\t{pred_dist[2]:.1f}\t{max_bias:.1f}\n")
        
        f.write("\n4. MATRICES DE CONFUSION\n")
        for noise_level in sorted(results.keys()):
            cm = results[noise_level]['confusion_matrix']
            f.write(f"\nσ={noise_level:.2f}:\n")
            f.write("     \tCat\tDog\tWild\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}\t{cm[i,0]}\t{cm[i,1]}\t{cm[i,2]}\n")
        
        f.write(f"\n5. TIMESTAMP: {timestamp}\n")
    
    logging.info(f"📊 Rapport détaillé sauvé: {report_path}")
    return report_path


def main():
    """Test complet du classifier avec matrices de confusion modernisées."""
    
    parser = argparse.ArgumentParser(description='Test matrices de confusion AFHQ Classifier modernisé')
    parser.add_argument('--data-dir', default='data/afhq', help='Dossier dataset AFHQ')
    parser.add_argument('--classifier-path', default=None,
                       help='Chemin vers checkpoint (auto si None)')
    parser.add_argument('--output-dir', default='experiments/afhq_classifier',
                       help='Dossier de sortie')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Nombre d\'échantillons à tester')
    parser.add_argument('--device', default='auto',
                       help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    device = torch.device(device)
    
    logging.info("🧪 TEST MATRICES DE CONFUSION - VERSION MODERNISÉE")
    logging.info("="*60)
    logging.info(f"📁 Data dir: {args.data_dir}")
    logging.info(f"📁 Output dir: {args.output_dir}")
    logging.info(f"🔧 Device: {device}")
    logging.info(f"📊 Échantillons: {args.num_samples}")
    logging.info(f"🎯 Niveaux: 7 optimisés (5 linspace + 2 zone critique)")
    
    # Setup SDE comme dans les autres scripts
    sde = VESDE(sigma_min=0.01, sigma_max=sigma_max, N=1000)
    logging.info("✅ SDE configuré: VESDE")
    
    # Charger classifier
    classifier = load_classifier(args.classifier_path, device)
    
    # Dataset validation avec transforms modernes
    val_transform = ImprovedAugmentations.get_transforms(512, 'val')
    val_dataset = AFHQDatasetOptimized(
        args.data_dir, 
        split='val', 
        transform=val_transform,
        sde=None  # Pas de perturbation auto
    )
    
    # Test par niveaux de bruit
    logging.info("\n🔄 Début du test par niveaux de bruit...")
    start_time = datetime.now()
    
    results, class_names = test_classifier_by_noise_levels(
        classifier, val_dataset, sde, device, args.num_samples
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculer weighted score
    weighted_score = compute_weighted_score({nl: data['accuracy'] for nl, data in results.items()})
    
    logging.info(f"\n✅ Test terminé en {duration:.1f}s")
    logging.info(f"📊 Score pondéré: {weighted_score:.2f}%")
    
    # Générer visualisations modernes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Matrices de confusion
    confusion_path = os.path.join(args.output_dir, f'confusion_matrices_modern_{timestamp}.png')
    plot_confusion_matrices_modern(results, class_names, confusion_path)
    
    # Courbe accuracy
    accuracy_path = os.path.join(args.output_dir, f'accuracy_vs_noise_modern_{timestamp}.png')
    plot_accuracy_vs_noise_modern(results, accuracy_path)
    
    # Analyse détaillée du biais
    analyze_bias_detailed(results, class_names)
    
    # Rapport détaillé
    report_path = save_detailed_report(results, class_names, args.output_dir, weighted_score)
    
    # Résumé final
    logging.info("\n🎉 ANALYSE TERMINÉE!")
    logging.info("="*60)
    logging.info(f"📊 Score pondéré: {weighted_score:.2f}%")
    logging.info(f"📈 Matrices: {confusion_path}")
    logging.info(f"📈 Courbe: {accuracy_path}")
    logging.info(f"📄 Rapport: {report_path}")
    logging.info(f"⏱️  Durée: {duration:.1f}s")
    logging.info(f"🎯 7 niveaux testés (5 linspace + 2 zone critique)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())