#!/usr/bin/env python3
"""
Script dédié pour évaluer la courbe accuracy vs noise level avec BEAUCOUP plus de points.
Génère une courbe très lisse et détaillée pour analyse fine du classifier AFHQ.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import os
import sys
import logging
from datetime import datetime
import argparse

sys.path.append('.')
import sde_lib
from models.afhq_classifier import create_afhq_classifier
from train_afhq_classifier import AFHQDatasetYangSong  # Updated import

SIGMA_MAX = 500


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def find_latest_checkpoint(save_dir):
    import glob
    pattern = os.path.join(save_dir, "afhq_classifier.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0]


def load_trained_classifier(checkpoint_path, device):
    logging.info(f"📊 Chargement du classifier: {os.path.basename(checkpoint_path)}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = create_afhq_classifier(
        pretrained=False,
        freeze_backbone=False,
        embedding_size=128
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 0)
    epoch = checkpoint.get('epoch', 0)
    
    logging.info(f"✅ Modèle chargé:")
    logging.info(f"   → Epoch: {epoch}")
    logging.info(f"   → Val accuracy: {val_acc:.2f}%")
    logging.info(f"   → Architecture: ResNet-50")
    logging.info(f"   → Embedding dim: 128")
    
    return model, checkpoint.get('config', {})


def create_vesde_from_config(config):
    sde_params = config.get('sde_params', {})
    
    sigma_min = sde_params.get('sigma_min', 0.01)
    sigma_max = sde_params.get('sigma_max', SIGMA_MAX)
    N = sde_params.get('N', 2000)
    
    vesde = sde_lib.VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=N)
    
    logging.info(f"🌊 VESDE créé:")
    logging.info(f"   → σ_min: {sigma_min}")
    logging.info(f"   → σ_max: {sigma_max}")
    logging.info(f"   → N: {N}")
    
    return vesde


def evaluate_noise_levels_detailed(model, dataset, vesde, device, 
                                 num_points=100, samples_per_point=200,
                                 noise_range=(0.01, SIGMA_MAX)):
    """
    Évalue l'accuracy par niveau de bruit avec BEAUCOUP plus de points.
    
    Args:
        model: Classifier pré-entraîné
        dataset: Dataset AFHQ 
        vesde: SDE VESDE
        device: Device CUDA
        num_points: Nombre de points sur la courbe (défaut 100 vs 20 avant)
        samples_per_point: Échantillons par point (défaut 200 vs 100 avant)
        noise_range: Range des niveaux de bruit (min, max)
    
    Returns:
        results: Liste de (noise_level, accuracy, std_dev)
    """
    # Générer les niveaux de bruit avec plus de points
    noise_levels = np.linspace(noise_range[0], noise_range[1], num_points)
    
    model.eval()
    results = []
    
    logging.info(f"📊 Évaluation détaillée par niveau de bruit:")
    logging.info(f"   → Nombre de points: {num_points}")
    logging.info(f"   → Échantillons par point: {samples_per_point}")
    logging.info(f"   → Range: [{noise_range[0]:.3f}, {noise_range[1]:.3f}]")
    logging.info(f"   → Total évaluations: {num_points * samples_per_point:,}")
    
    # Préparer les indices d'échantillons
    total_samples = len(dataset)
    
    for i, noise_level in enumerate(tqdm(noise_levels, desc="Noise levels")):
        accuracies_for_level = []
        
        # Échantillonner plusieurs fois pour avoir une estimation plus robuste
        num_batches = 5  # Diviser en petits batches pour éviter OOM
        samples_per_batch = samples_per_point // num_batches
        
        for batch_idx in range(num_batches):
            # Sample des indices aléatoires
            indices = np.random.choice(total_samples, samples_per_batch, replace=False)
            
            batch_correct = 0
            batch_total = 0
            
            with torch.no_grad():
                for idx in indices:
                    try:
                        # Charger image originale
                        img_path, label = dataset.samples[idx]
                        transform = dataset.transform
                        image = transform(Image.open(img_path).convert('RGB'))
                        
                        # Appliquer niveau de bruit spécifique
                        t = torch.tensor([noise_level], device=device)
                        mean, std = vesde.marginal_prob(image.unsqueeze(0), t)
                        noise = torch.randn_like(image)
                        perturbed_image = mean.squeeze(0) + std.squeeze() * noise
                        
                        # Prédiction
                        perturbed_image = perturbed_image.unsqueeze(0).to(device)
                        outputs = model(perturbed_image, t)
                        _, predicted = torch.max(outputs, 1)
                        
                        batch_correct += (predicted.item() == label)
                        batch_total += 1
                        
                    except Exception as e:
                        # Skip en cas d'erreur
                        continue
            
            if batch_total > 0:
                batch_accuracy = 100.0 * batch_correct / batch_total
                accuracies_for_level.append(batch_accuracy)
        
        # Calculer statistiques pour ce niveau
        if accuracies_for_level:
            mean_accuracy = np.mean(accuracies_for_level)
            std_accuracy = np.std(accuracies_for_level)
        else:
            mean_accuracy = 0.0
            std_accuracy = 0.0
        
        results.append((noise_level, mean_accuracy, std_accuracy))
        
        # Log progress régulièrement
        if (i + 1) % (num_points // 10) == 0:
            logging.info(f"   → {i+1}/{num_points} points traités, dernier: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")
    
    logging.info("✅ Évaluation détaillée terminée")
    return results


def plot_detailed_noise_curve(results, save_path, title_suffix=""):
    """
    Génère un plot détaillé de la courbe accuracy vs noise.
    
    Args:
        results: Liste de (noise_level, accuracy, std_dev)
        save_path: Chemin de sauvegarde
        title_suffix: Suffixe pour le titre
    """
    noise_levels, accuracies, std_devs = zip(*results)
    
    # Configuration du style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Courbe principale avec barre d'erreur
    ax.errorbar(noise_levels, accuracies, yerr=std_devs, 
                color='#2E86AB', linewidth=2.5, marker='o', markersize=4,
                capsize=3, capthick=1, elinewidth=1, alpha=0.8,
                label=f'Accuracy (±std, {len(noise_levels)} points)')
    
    # Zone d'incertitude
    accuracies_np = np.array(accuracies)
    std_devs_np = np.array(std_devs)
    ax.fill_between(noise_levels, 
                    accuracies_np - std_devs_np, 
                    accuracies_np + std_devs_np,
                    alpha=0.2, color='#2E86AB', label='±1 std')
    
    # Ligne de tendance lissée
    from scipy.ndimage import gaussian_filter1d
    smoothed_acc = gaussian_filter1d(accuracies, sigma=2)
    ax.plot(noise_levels, smoothed_acc, '--', color='#A23B72', linewidth=2, 
            alpha=0.7, label='Tendance lissée')
    
    # Annotations importantes
    max_acc_idx = np.argmax(accuracies)
    ax.annotate(f'Max: {accuracies[max_acc_idx]:.1f}%\n@t={noise_levels[max_acc_idx]:.3f}',
                xy=(noise_levels[max_acc_idx], accuracies[max_acc_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Points spéciaux
    low_noise_acc = accuracies[0]
    high_noise_acc = accuracies[-1]
    ax.annotate(f'Low noise: {low_noise_acc:.1f}%', 
                xy=(noise_levels[0], low_noise_acc),
                xytext=(20, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax.annotate(f'High noise: {high_noise_acc:.1f}%', 
                xy=(noise_levels[-1], high_noise_acc),
                xytext=(-60, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Configuration des axes
    ax.set_xlabel('Noise Level (t)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'AFHQ Classifier: Accuracy vs Noise Level{title_suffix}\n'
                f'Détaillé: {len(noise_levels)} points', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0.01, SIGMA_MAX)
    ax.set_ylim(0, min(100, max(accuracies) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Statistiques dans le coin
    stats_text = f"Points: {len(noise_levels)}\n"
    stats_text += f"Max acc: {max(accuracies):.1f}%\n"
    stats_text += f"Min acc: {min(accuracies):.1f}%\n"
    stats_text += f"Range: {max(accuracies) - min(accuracies):.1f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📊 Courbe détaillée sauvée: {save_path}")


def analyze_noise_curve(results):
    """Analyse statistique de la courbe."""
    noise_levels, accuracies, std_devs = zip(*results)
    
    # Statistiques de base
    max_acc = max(accuracies)
    min_acc = min(accuracies)
    max_acc_idx = np.argmax(accuracies)
    optimal_noise = noise_levels[max_acc_idx]
    
    # Trouver le seuil de dégradation (quand accuracy chute de 10%)
    degradation_threshold = max_acc * 0.9
    degradation_idx = None
    for i, acc in enumerate(accuracies):
        if acc < degradation_threshold:
            degradation_idx = i
            break
    
    # Pente moyenne de dégradation
    mid_idx = len(accuracies) // 2
    slope = (accuracies[-1] - accuracies[mid_idx]) / (noise_levels[-1] - noise_levels[mid_idx])
    
    analysis = {
        'max_accuracy': max_acc,
        'min_accuracy': min_acc,
        'accuracy_range': max_acc - min_acc,
        'optimal_noise_level': optimal_noise,
        'degradation_threshold_noise': noise_levels[degradation_idx] if degradation_idx else 1.0,
        'degradation_slope': slope,
        'low_noise_accuracy': accuracies[0],
        'high_noise_accuracy': accuracies[-1],
        'std_mean': np.mean(std_devs),
        'std_max': max(std_devs)
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Évaluation détaillée courbe accuracy vs noise')
    parser.add_argument('--checkpoint-dir', default='experiments/afhq_classifier',
                       help='Dossier contenant les checkpoints')
    parser.add_argument('--data-dir', default='data/afhq',
                       help='Dossier dataset AFHQ')
    parser.add_argument('--num-points', type=int, default=100,
                       help='Nombre de points sur la courbe (défaut: 100)')
    parser.add_argument('--samples-per-point', type=int, default=200,
                       help='Échantillons par point (défaut: 200)')
    parser.add_argument('--output-dir', default=None,
                       help='Dossier de sortie (défaut: même que checkpoint-dir)')
    parser.add_argument('--noise-min', type=float, default=0.01,
                       help='Niveau de bruit minimum (défaut: 0.01)')
    parser.add_argument('--noise-max', type=float, default=SIGMA_MAX,
                       help='Niveau de bruit maximum (défaut: 500.0)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("🎯 ÉVALUATION DÉTAILLÉE - Accuracy vs Noise Level")
    logging.info("=" * 60)
    logging.info(f"📁 Checkpoint dir: {args.checkpoint_dir}")
    logging.info(f"📁 Data dir: {args.data_dir}")
    logging.info(f"📊 Points sur courbe: {args.num_points}")
    logging.info(f"📊 Échantillons/point: {args.samples_per_point}")
    logging.info(f"📊 Range bruit: [{args.noise_min:.3f}, {args.noise_max:.3f}]")
    logging.info("")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        logging.info(f"   → GPU: {gpu_name} ({gpu_memory}GB)")
    
    # Trouver le checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint_path:
        logging.error(f"❌ Aucun checkpoint trouvé dans {args.checkpoint_dir}")
        logging.info("💡 Vérifiez le dossier ou entraînez d'abord le classifier")
        return 1
    
    # Charger le modèle
    model, config = load_trained_classifier(checkpoint_path, device)
    
    # Créer le SDE
    vesde = create_vesde_from_config(config)
    
    # Préparer le dataset de validation - Updated to use AFHQDatasetYangSong
    import torchvision.transforms as transforms
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = AFHQDatasetYangSong(
        args.data_dir,
        split='val',
        transform=val_transform,
        sde=vesde
    )
    
    logging.info(f"📊 Dataset validation: {len(val_dataset)} images")
    
    # Évaluation détaillée
    logging.info("🔄 Début de l'évaluation détaillée...")
    start_time = datetime.now()
    
    results = evaluate_noise_levels_detailed(
        model=model,
        dataset=val_dataset,
        vesde=vesde,
        device=device,
        num_points=args.num_points,
        samples_per_point=args.samples_per_point,
        noise_range=(args.noise_min, args.noise_max)
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logging.info(f"✅ Évaluation terminée en {duration:.1f}s")
    
    # Analyse des résultats
    analysis = analyze_noise_curve(results)
    
    logging.info("📊 ANALYSE DES RÉSULTATS:")
    logging.info(f"   → Accuracy max: {analysis['max_accuracy']:.2f}%")
    logging.info(f"   → Noise optimal: {analysis['optimal_noise_level']:.3f}")
    logging.info(f"   → Accuracy range: {analysis['accuracy_range']:.2f}%")
    logging.info(f"   → Low noise acc: {analysis['low_noise_accuracy']:.2f}%")
    logging.info(f"   → High noise acc: {analysis['high_noise_accuracy']:.2f}%")
    logging.info(f"   → Dégradation slope: {analysis['degradation_slope']:.2f}%/unit")
    logging.info(f"   → Std moyenne: {analysis['std_mean']:.2f}%")
    
    # Dossier de sortie
    output_dir = args.output_dir or args.checkpoint_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder la courbe
    plot_path = os.path.join(output_dir, f'accuracy_vs_noise_detailed_{args.num_points}pts_{timestamp}.png')
    title_suffix = f" ({args.num_points} points, {args.samples_per_point} samples/pt)"
    
    plot_detailed_noise_curve(results, plot_path, title_suffix)
    
    # Sauvegarder les données
    data_path = os.path.join(output_dir, f'noise_curve_data_{args.num_points}pts_{timestamp}.npz')
    noise_levels, accuracies, std_devs = zip(*results)
    
    np.savez_compressed(data_path, 
                       noise_levels=noise_levels,
                       accuracies=accuracies,
                       std_devs=std_devs,
                       analysis=analysis,
                       config=config)
    
    logging.info(f"💾 Données sauvées: {data_path}")
    
    # Comparaison avec version basique (20 points)
    if args.num_points > 20:
        improvement_factor = args.num_points / 20
        logging.info(f"📈 Amélioration résolution: {improvement_factor:.1f}x plus de points")
        logging.info(f"📈 Total évaluations: {args.num_points * args.samples_per_point:,}")
    
    logging.info("")
    logging.info("🎉 ÉVALUATION DÉTAILLÉE TERMINÉE!")
    logging.info("=" * 60)
    logging.info(f"📊 Courbe détaillée: {plot_path}")
    logging.info(f"💾 Données: {data_path}")
    logging.info(f"📈 {args.num_points} points vs 20 points standard")
    logging.info(f"⏱️  Durée: {duration:.1f}s")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())