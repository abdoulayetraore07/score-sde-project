#!/usr/bin/env python3
"""
ÉTAPE 1: Pré-entraînement du classifier AFHQ SANS BRUIT
Pour calibrer parfaitement le modèle sur cat/dog/wild avant d'ajouter noise conditioning
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


class AFHQCleanDataset(Dataset):
    """Dataset AFHQ SANS bruit pour pré-entraînement."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.classes = ['cat', 'dog', 'wild']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()
        self.class_weights = self._compute_class_weights()
        
        logging.info(f"📊 Dataset {split}: {len(self.samples)} images")
        for i, cls in enumerate(self.classes):
            count = sum(1 for _, label in self.samples if label == i)
            weight = self.class_weights[i]
            logging.info(f"   → {cls}: {count} images (weight: {weight:.3f})")
    
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
    
    def _compute_class_weights(self):
        """Calcule les poids pour équilibrer les classes."""
        class_counts = [0, 0, 0]
        for _, label in self.samples:
            class_counts[label] += 1
        
        total_samples = sum(class_counts)
        weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (len(self.classes) * count)
                weights.append(weight)
            else:
                weights.append(1.0)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_weights(self):
        """Retourne les poids des classes pour le loss."""
        return self.class_weights
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Charger image SANS BRUIT
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AFHQCleanClassifier(nn.Module):
    """
    Version SIMPLIFIÉE du classifier AFHQ SANS noise conditioning
    Utilise la même architecture mais sans les parties time/noise
    """
    
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Même backbone que votre classifier principal
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        self.feature_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone initial
        if freeze_backbone:
            self._freeze_backbone()
        
        # Classifier head SIMPLE (sans time conditioning)
        self.classifier = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.feature_dim, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialisation
        self._initialize_weights()
    
    def _freeze_backbone(self):
        """Freeze toutes les couches du backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logging.info("🧊 Backbone ResNet-50 gelé")
    
    def _initialize_weights(self):
        """Initialisation comme Yang Song."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass SIMPLE sans noise conditioning."""
        # Features du backbone ResNet
        image_features = self.backbone(x)  # [batch, 2048]
        
        # Classification directe
        logits = self.classifier(image_features)
        
        return logits
    
    def unfreeze_backbone_layers(self, layer_names=['layer4']):
        """Fine-tuning progressif du backbone."""
        unfrozen_params = 0
        
        for layer_name in layer_names:
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                logging.info(f"🔓 {layer_name} dégelée")
            else:
                logging.warning(f"⚠️  Couche {layer_name} non trouvée")
        
        logging.info(f"📊 {unfrozen_params:,} paramètres dégelés")
        return unfrozen_params


def train_epoch_clean(model, dataloader, criterion, optimizer, device, epoch):
    """Training epoch SANS bruit."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass SIMPLE
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
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
        logging.info(f"📈 Train Accur. classe {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    return epoch_loss, epoch_acc


def validate_clean(model, dataloader, criterion, device, classes=['cat', 'dog', 'wild']):
    """Validation SANS bruit."""
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
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
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    class_acc = {
        cls: 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in class_total
    }
    for cls, acc in class_acc.items():
        logging.info(f"📈 Val Accur. classe {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    return val_loss, val_acc


def main():
    """Pré-entraînement SANS bruit."""
    
    # Config conservatrice pour pré-entraînement
    config = {
        'data_dir': 'data/afhq',
        'image_size': 512,
        'batch_size': 64,
        'num_epochs': 50,  # Plus court que votre training principal
        'lr_head': 1e-3,   # Un peu plus élevé car pas de bruit
        'lr_backbone': 2e-5,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'save_dir': 'experiments/afhq_classifier_clean',
        'progressive_epochs': [10, 20],  # Dégeler plus tôt
    }
    
    logging.info("🔥 AFHQ Classifier Pré-entraînement SANS BRUIT")
    logging.info("🎯 OBJECTIF: Calibrer le modèle sur cat/dog/wild avant noise conditioning")
    logging.info(f"Device: {config['device']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Epochs: {config['num_epochs']}")
    
    # Datasets SANS bruit
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    train_dataset = AFHQCleanDataset(
        config['data_dir'], 
        split='train', 
        transform=train_transform
    )
    
    val_dataset = AFHQCleanDataset(
        config['data_dir'], 
        split='val', 
        transform=val_transform
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
    
    # Modèle SIMPLE
    device = torch.device(config['device'])
    model = AFHQCleanClassifier(
        num_classes=3,
        pretrained=True,
        freeze_backbone=True  # Commence gelé
    ).to(device)
    
    logging.info("✅ Modèle CLEAN créé: ResNet-50 SANS noise conditioning")
    
    # Loss avec class weighting
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logging.info(f"⚖️  Class weights: {class_weights}")
    
    # Optimizer initial (head seulement)
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=config['lr_head'], weight_decay=config['weight_decay'])
    
    # Scheduler simple
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    
    # Training
    os.makedirs(config['save_dir'], exist_ok=True)
    best_acc = 0.0
    layer4_unfrozen = False
    
    logging.info("🔥 Début pré-entraînement CLEAN...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        
        # Fine-tuning progressif
        if epoch == config['progressive_epochs'][0] and not layer4_unfrozen:
            # Dégeler layer4
            model.unfreeze_backbone_layers(['layer4'])
            
            # Ajouter backbone params à l'optimizer
            backbone_params = list(model.backbone.layer4.parameters())
            optimizer.add_param_group({'params': backbone_params, 'lr': config['lr_backbone']})
            
            layer4_unfrozen = True
            logging.info("🔓 Layer4 dégelé et ajouté à l'optimizer")
        
        # Training
        train_loss, train_acc = train_epoch_clean(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = validate_clean(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        logging.info(f"Epoch {epoch}/{config['num_epochs']}:")
        logging.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logging.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        logging.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'config': config,
                'model_type': 'clean_classifier'
            }, os.path.join(config['save_dir'], 'afhq_classifier_clean.pth'))
            
            logging.info(f"💾 Nouveau meilleur modèle CLEAN sauvé: {val_acc:.2f}%")
    
    logging.info(f"🎉 Pré-entraînement CLEAN terminé! Meilleure accuracy: {best_acc:.2f}%")
    logging.info(f"📁 Modèle sauvé: {config['save_dir']}/afhq_classifier_clean.pth")
    logging.info("")
    logging.info("🚀 PROCHAINE ÉTAPE:")
    logging.info("   1. Utilisez ce modèle comme point de départ")
    logging.info("   2. Chargez-le dans votre training avec noise conditioning")
    logging.info("   3. Fine-tunez avec learning rate plus bas")


if __name__ == "__main__":
    # Import nécessaire
    import torchvision.models as models
    main()