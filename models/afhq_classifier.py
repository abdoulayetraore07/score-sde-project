#!/usr/bin/env python3
"""
AFHQ Classifier - COPIE EXACTE de Yang Song
Conditioning sur noise_scale (sigmas) comme dans wideresnet_noise_conditional.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import math
import logging


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels - COPIE EXACTE Yang Song."""
    
    def __init__(self, embedding_size=128, scale=16.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        
        # W fixe comme dans JAX version
        self.register_buffer('W', torch.randn(embedding_size) * scale)
        
    def forward(self, x):
        """
        Args:
            x: noise_scale [batch] (sigmas)
        Returns:
            embeddings [batch, embedding_size * 2]
        """
        # EXACTEMENT comme Yang Song : jnp.log(sigmas)
        x_proj = torch.log(x)[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AFHQClassifier(nn.Module):
    """
    AFHQ Classifier EXACTEMENT comme Yang Song le ferait
    - ResNet-50 backbone 
    - Gaussian Fourier embedding sur noise_scale
    - Conditioning EXACTEMENT comme wideresnet_noise_conditional.py
    """
    
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False, embedding_size=128):
        super().__init__()
        
        # ResNet-50 backbone (suffisant)
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        self.feature_dim = self.backbone.fc.in_features  # 512 pour ResNet-50
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone initial
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
        
        # Gaussian Fourier embedding EXACTEMENT comme Yang Song
        self.embedding_size = embedding_size
        self.fourier_proj = GaussianFourierProjection(embedding_size=embedding_size, scale=16.0)
        
        # Time conditioning EXACTEMENT comme Yang Song
        # temb = nn.Dense(128 * 4)(temb)
        # temb = nn.Dense(128 * 4)(nn.swish(temb))
        self.time_conditioning = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size * 4),  # 128*2 -> 128*4
            nn.SiLU(),  # swish
            nn.Linear(embedding_size * 4, embedding_size * 4),  # 128*4 -> 128*4
        )
        
        # Classifier head - SIMPLE comme Yang Song
        self.classifier = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.feature_dim, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Projection time -> feature pour injection
        self.time_proj = nn.Linear(embedding_size * 4, self.feature_dim)
        
        # Initialisation
        self._initialize_weights()
    
    def _freeze_backbone(self):
        """Freeze toutes les couches du backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logging.info("🧊 Backbone ResNet-50 gelé")
    
    def _initialize_weights(self):
        """Initialisation comme Yang Song."""
        for m in [self.time_conditioning, self.classifier, self.time_proj]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, sigmas):
        """
        Forward pass EXACTEMENT comme Yang Song
        
        Args:
            x: Images [batch, 3, height, width] - perturbées par SDE
            sigmas: Noise scales [batch] - DIRECTEMENT les sigmas du SDE
            
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # Gaussian Fourier embedding EXACTEMENT comme Yang Song
        temb = self.fourier_proj(sigmas)  # [batch, embedding_size * 2]
        
        # Time conditioning EXACTEMENT comme Yang Song  
        temb = self.time_conditioning(temb)  # [batch, embedding_size * 4]
        
        # Features du backbone ResNet
        image_features = self.backbone(x)  # [batch, 512]
        
        # Projection time embedding vers feature space
        time_features = self.time_proj(temb)  # [batch, 512]
        
        # Fusion additive comme Yang Song (simple addition)
        conditioned_features = image_features + time_features
        
        # Classification finale
        logits = self.classifier(conditioned_features)
        
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


def create_afhq_classifier(pretrained=True, freeze_backbone=True, embedding_size=128):
    """
    Factory function pour créer le classifier AFHQ EXACTEMENT comme Yang Song
    
    Args:
        pretrained: Utiliser ResNet-50 pré-entraîné
        freeze_backbone: Freeze le backbone pour training rapide
        embedding_size: Dimension Gaussian Fourier (128 comme Yang Song)
        
    Returns:
        model: AFHQClassifier instance
    """
    return AFHQClassifier(
        num_classes=3,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        embedding_size=embedding_size
    )