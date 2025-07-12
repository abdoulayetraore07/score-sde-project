#!/usr/bin/env python3
"""
AFHQ Classifier Training - VERSION CORRIGÉE
✅ sigma_max = 300 (au lieu de 784)
✅ Suppression du weighted_score
✅ Validation simple et claire
✅ Garde le progressive fine-tuning et resume
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Ajouter le chemin
sys.path.append('.')

# Imports locaux
from models.afhq_classifier import create_afhq_classifier
from sde_lib import VESDE

sigma_max = 500


class AFHQDatasetYangSong(Dataset):
    """Dataset AFHQ avec noise_scale comme Yang Song."""
    
    def __init__(self, data_dir, split='train', transform=None, sde=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sde = sde
        
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
        
        # Sample random time EXACTEMENT comme Yang Song
        t = torch.rand(1) * (self.sde.T - 1e-5) + 1e-5  # [eps, T]
        
        # Perturber l'image avec SDE
        mean, noise_scale = self.sde.marginal_prob(image.unsqueeze(0), t)
        noise = torch.randn_like(mean)
        perturbed_image = mean + noise_scale[:, None, None, None] * noise
        
        # RETOURNER noise_scale (pas t) comme Yang Song
        return perturbed_image.squeeze(0), noise_scale.squeeze(0), label


def train_epoch_yang_song(model, dataloader, criterion, optimizer, device, epoch):
    """Training epoch EXACTEMENT comme Yang Song."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    for batch_idx, (images, noise_scales, labels) in enumerate(pbar):
        images = images.to(device)
        noise_scales = noise_scales.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass avec noise_scales
        outputs = model(images, noise_scales)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        
        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        # Stats
        running_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress
        acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    class_acc = {cls: 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0 for cls in class_total}
    for cls, acc in class_acc.items():
        logging.info(f"📈 Train Accuracy classe {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    return epoch_loss, epoch_acc


def validate_yang_song(model, dataloader, criterion, device, classes=['cat', 'dog', 'wild']):
    """Validation EXACTEMENT comme Yang Song."""
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, noise_scales, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            noise_scales = noise_scales.to(device)
            labels = labels.to(device)
            
            outputs = model(images, noise_scales)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    class_acc = {
        cls: 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in class_total
    }
    for cls, acc in class_acc.items():
        logging.info(f"📈 Val Accuracy classe {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    return val_loss, val_acc, all_preds, all_labels


def test_at_noise_level(model, dataset, noise_level, device, max_samples=500):
    """Test le modèle à un niveau de bruit spécifique avec échantillonnage équilibré."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    # Echantillonnage équilibré par classe
    samples_per_class = max_samples // 3  # 166 par classe
    
    # Séparer les indices par classe
    cat_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    dog_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
    wild_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 2]
    
    # Échantillonner équitablement
    selected_indices = []
    selected_indices.extend(np.random.choice(cat_indices, min(samples_per_class, len(cat_indices)), replace=False))
    selected_indices.extend(np.random.choice(dog_indices, min(samples_per_class, len(dog_indices)), replace=False))
    selected_indices.extend(np.random.choice(wild_indices, min(samples_per_class, len(wild_indices)), replace=False))
    
    np.random.shuffle(selected_indices)
    
    with torch.no_grad():
        for idx in selected_indices:
            # Charger image et label
            img_path, label = dataset.samples[idx]
            image = Image.open(img_path).convert('RGB')
            image = dataset.transform(image)
            
            # Appliquer le niveau de bruit fixe
            images = image.unsqueeze(0).to(device)
            labels = torch.tensor([label]).to(device)
            
            batch_size = images.size(0)
            noise_scales = torch.full((batch_size,), noise_level, device=device)
            
            outputs = model(images, noise_scales)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Analyser la distribution
    pred_dist = np.bincount(all_predictions, minlength=3) / len(all_predictions) * 100
    true_dist = np.bincount(all_labels, minlength=3) / len(all_labels) * 100
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    logging.info(f"    σ={noise_level:6.2f}: Acc={accuracy:5.1f}% | Pred: cat={pred_dist[0]:.1f}% dog={pred_dist[1]:.1f}% wild={pred_dist[2]:.1f}% | True: cat={true_dist[0]:.1f}% dog={true_dist[1]:.1f}% wild={true_dist[2]:.1f}%")
    
    return accuracy

def simple_noise_validation(model, dataset, device):  # ✅ Passer dataset au lieu de dataloader
    """Validation simple sur 8 niveaux de bruit raisonnables."""
    test_sigmas = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # ✅ Corrigé 
    
    logging.info("🧪 Test sur 8 niveaux de bruit:")
    accuracies = []
    
    for sigma in test_sigmas:
        acc = test_at_noise_level(model, dataset, sigma, device)  # ✅ Passer dataset
        accuracies.append(acc)
    
    avg_score = np.mean(accuracies)
    logging.info(f"📊 Score moyen sur 8 niveaux: {avg_score:.2f}%")
    
    return avg_score, accuracies


def save_model_simple(model, epoch, val_acc, config, is_best=False):
    """Sauvegarde simple sans weighted_score."""
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Historique avec timestamp
    history_dir = os.path.join(save_dir, 'checkpoints_history')
    os.makedirs(history_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint_data = {
        'model_state_dict': model_to_save.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'config': config,
        'timestamp': timestamp
    }
    
    # Sauvegarde historique
    history_path = os.path.join(history_dir, f'afhq_classifier_epoch{epoch:03d}_{val_acc:.2f}_{timestamp}.pth')
    torch.save(checkpoint_data, history_path)
    logging.info(f"📂 Modèle sauvé: {os.path.basename(history_path)}")
    
    # Meilleur modèle
    if is_best:
        best_path = os.path.join(save_dir, 'afhq_classifier.pth')
        torch.save(checkpoint_data, best_path)
        logging.info(f"🏆 NOUVEAU MEILLEUR modèle: val_acc={val_acc:.2f}%")


def main():
    """Training corrigé avec paramètres corrects."""
    
    # Config finale
    config = {
        'data_dir': 'data/afhq',
        'image_size': 512,
        'batch_size': 64,
        'num_epochs': 300,
        'lr_head': 1e-4,
        'lr_backbone': 5e-6,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'save_dir': 'experiments/afhq_classifier',
        'progressive_epochs': [20, 40, 80],
        'embedding_size': 128,
    }
    
    logging.info("🔥 AFHQ Classifier Training - VERSION CORRIGÉE")
    logging.info(f"Device: {config['device']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Total epochs: {config['num_epochs']}")
    
    # ✅ SDE CORRIGÉ: sigma_max = 300 (au lieu de 784)
    sde = VESDE(sigma_min=0.01, sigma_max=sigma_max, N=1000)
    logging.info("✅ SDE configuré: VESDE (sigma_max=300 - CORRIGÉ)")
    
    # Datasets
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    train_dataset = AFHQDatasetYangSong(
        config['data_dir'], 
        split='train', 
        transform=train_transform,
        sde=sde
    )
    
    val_dataset = AFHQDatasetYangSong(
        config['data_dir'], 
        split='val', 
        transform=val_transform,
        sde=sde
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Modèle
    device = torch.device(config['device'])
    model = create_afhq_classifier(
        pretrained=True,
        freeze_backbone=False,
        embedding_size=config['embedding_size']
    ).to(device)

    # Charger le modèle pré-entraîné CLEAN si disponible
    clean_checkpoint_path = 'experiments/afhq_classifier_clean/afhq_classifier_clean.pth'
    if os.path.exists(clean_checkpoint_path):
        logging.info(f"🔄 Chargement du modèle pré-entraîné CLEAN: {clean_checkpoint_path}")
        
        clean_checkpoint = torch.load(clean_checkpoint_path, map_location=device)
        clean_acc = clean_checkpoint.get('val_acc', 0)
        
        # Charger SEULEMENT les poids du backbone et classifier
        model_dict = model.state_dict()
        clean_dict = clean_checkpoint['model_state_dict']
        
        updated_dict = {}
        for key, value in clean_dict.items():
            if key.startswith('backbone.') or key.startswith('classifier.'):
                updated_dict[key] = value
        
        model_dict.update(updated_dict)
        model.load_state_dict(model_dict)
        
        logging.info(f"✅ Modèle pré-entraîné chargé! Accuracy clean: {clean_acc:.2f}%")
    else:
        logging.warning("⚠️  Modèle clean non trouvé, démarrage à zéro")
        
    logging.info("✅ Modèle créé: ResNet-50 + Gaussian Fourier")
    
    # Multi-GPU si disponible
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logging.info(f"🔥 Multi-GPU: {torch.cuda.device_count()} GPUs")
    
    # Class weighting
    class_counts = [5153, 4739, 4738]  
    total = sum(class_counts)
    class_weights = torch.tensor([total/(3*count) for count in class_counts])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Setup optimizer parameters
    if hasattr(model, 'module'):
        time_params = list(model.module.time_conditioning.parameters()) + \
                     list(model.module.time_proj.parameters()) + \
                     list(model.module.classifier.parameters()) + \
                     list(model.module.fourier_proj.parameters())
        backbone_params = list(model.module.backbone.parameters())
    else:
        time_params = list(model.time_conditioning.parameters()) + \
                     list(model.time_proj.parameters()) + \
                     list(model.classifier.parameters()) + \
                     list(model.fourier_proj.parameters())
        backbone_params = list(model.backbone.parameters())
    
    # ======================================================================
    # LOGIQUE DE RESUME SIMPLIFIÉE
    # ======================================================================
    
    checkpoint_path = os.path.join(config['save_dir'], 'afhq_classifier.pth')
    start_epoch = 1
    best_val_acc = 0.0  # ✅ SIMPLIFIÉ: utilise val_acc normale
    
    # Optimizer initial
    optimizer = torch.optim.AdamW([
        {'params': time_params, 'lr': config['lr_head']},
    ], weight_decay=config['weight_decay'])
    
    # Scheduler initial
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    
    # Variables de tracking pour fine-tuning
    layer4_unfrozen = False
    layer3_unfrozen = False
    
    if os.path.exists(checkpoint_path):
        logging.info(f"🔄 RESUME: Chargement du checkpoint {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Restaurer le modèle
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            # Restaurer les informations de training
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0)  # ✅ SIMPLIFIÉ
            
            logging.info(f"✅ RESUME réussi!")
            logging.info(f"   → Reprend à l'epoch: {start_epoch}")
            logging.info(f"   → Meilleur val accuracy: {best_val_acc:.2f}%")
            
            # Restaurer l'état du fine-tuning
            if start_epoch > config['progressive_epochs'][2]:  # Après epoch 80
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbone_layers(['layer2', 'layer3', 'layer4'])
                else:
                    model.unfreeze_backbone_layers(['layer2', 'layer3', 'layer4'])
                
                optimizer = torch.optim.AdamW([
                    {'params': time_params, 'lr': config['lr_head']},
                    {'params': backbone_params, 'lr': config['lr_backbone']},
                ], weight_decay=config['weight_decay'])
                
                layer4_unfrozen = True
                layer3_unfrozen = True
                logging.info("🔓 Layer2, Layer3 et Layer4 déjà dégelés (restauré)")

            elif start_epoch > config['progressive_epochs'][1]:  # Après epoch 40
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbone_layers(['layer3', 'layer4'])
                else:
                    model.unfreeze_backbone_layers(['layer3', 'layer4'])
                
                optimizer = torch.optim.AdamW([
                    {'params': time_params, 'lr': config['lr_head']},
                    {'params': backbone_params, 'lr': config['lr_backbone']},
                ], weight_decay=config['weight_decay'])
                
                layer4_unfrozen = True
                layer3_unfrozen = True
                logging.info("🔓 Layer3 et Layer4 déjà dégelés (restauré)")

            elif start_epoch > config['progressive_epochs'][0]:  # Après epoch 20
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbone_layers(['layer4'])
                else:
                    model.unfreeze_backbone_layers(['layer4'])
                
                optimizer = torch.optim.AdamW([
                    {'params': time_params, 'lr': config['lr_head']},
                    {'params': backbone_params, 'lr': config['lr_backbone']},
                ], weight_decay=config['weight_decay'])
                
                layer4_unfrozen = True
                logging.info("🔓 Layer4 déjà dégelé (restauré)")
            
            # Recréer scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['num_epochs'], eta_min=1e-6
            )
            
        except Exception as e:
            logging.warning(f"⚠️ Erreur lors du resume: {e}")
            logging.info("🔄 Redémarrage à zéro...")
            start_epoch = 1
            best_val_acc = 0.0
    else:
        logging.info("🆕 Nouveau training - aucun checkpoint trouvé")
    
    # ======================================================================
    # TRAINING LOOP PRINCIPAL
    # ======================================================================
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    logging.info(f"🔥 {'Reprise' if start_epoch > 1 else 'Début'} entraînement avec paramètres corrigés...")
    
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        
        # Fine-tuning progressif
        if epoch in config['progressive_epochs']:
            
            if epoch == config['progressive_epochs'][0] and not layer4_unfrozen:  # epoch 20
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbone_layers(['layer4'])
                else:
                    model.unfreeze_backbone_layers(['layer4'])
                
                optimizer = torch.optim.AdamW([
                    {'params': time_params, 'lr': config['lr_head']},
                    {'params': backbone_params, 'lr': config['lr_backbone']},
                ], weight_decay=config['weight_decay'])
                
                layer4_unfrozen = True
                logging.info("🔓 PHASE 2: Layer4 dégelé")
            
            elif epoch == config['progressive_epochs'][1] and not layer3_unfrozen:  # epoch 40
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbone_layers(['layer3', 'layer4'])
                else:
                    model.unfreeze_backbone_layers(['layer3', 'layer4'])
                
                layer3_unfrozen = True
                logging.info("🔓 PHASE 3: Layer3 dégelé")
            
            elif epoch == config['progressive_epochs'][2]:  # epoch 80
                if hasattr(model, 'module'):
                    model.module.unfreeze_backbone_layers(['layer2', 'layer3', 'layer4'])
                else:
                    model.unfreeze_backbone_layers(['layer2', 'layer3', 'layer4'])
                
                logging.info("🔓 PHASE 4: Layer2 dégelé (fine-tuning complet)")
        
        # Training
        train_loss, train_acc = train_epoch_yang_song(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # ✅ VALIDATION SIMPLE
        val_loss, val_acc, all_preds, all_labels = validate_yang_song(
            model, val_loader, criterion, device
        )
        
        # ✅ TEST OPTIONNEL sur 3 niveaux (tous les 5 epochs)
        if epoch % 10 == 0:
            simple_noise_validation(model, val_dataset, device)
        
        # Scheduler step
        scheduler.step()
        
        # ✅ LOGGING SIMPLIFIÉ
        logging.info(f"Epoch {epoch}/{config['num_epochs']}:")
        logging.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logging.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        logging.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # ✅ SAUVEGARDE SIMPLIFIÉE
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_model_simple(model, epoch, val_acc, config, is_best=True)
            logging.info(f"🏆 NOUVEAU RECORD: {val_acc:.2f}%")
        else:
            # Sauvegarder tous les 20 epochs
            if epoch % 20 == 0:
                save_model_simple(model, epoch, val_acc, config, is_best=False)
            logging.info(f"📊 Val Acc: {val_acc:.2f}% (meilleur: {best_val_acc:.2f}%)")
    
    logging.info(f"🎉 Training terminé! Meilleur val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()