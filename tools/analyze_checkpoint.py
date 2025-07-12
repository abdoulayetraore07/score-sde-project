#!/usr/bin/env python3
"""
Script pour analyser EXACTEMENT ce qui est dans le checkpoint
et voir les paramètres bizarres
"""

import torch
import os

def analyze_afhq_checkpoint():
    """Analyse détaillée du checkpoint pour voir ce qui a été entraîné."""
    
    checkpoint_path = 'experiments/afhq_classifier/afhq_classifier.pth'
    
    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint non trouvé")
        return
    
    print("🔍 Analyse du checkpoint AFHQ...")
    
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Voir ce qu'il contient
    print(f"\n📋 Clés dans le checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Analyser le state_dict
    state_dict = checkpoint['model_state_dict']
    
    print(f"\n🏗️ Architecture détectée:")
    print(f"Nombre total de clés: {len(state_dict.keys())}")
    
    # Identifier le backbone
    backbone_keys = [k for k in state_dict.keys() if 'backbone' in k]
    print(f"\n🔧 Backbone layers: {len(backbone_keys)}")
    
    # Chercher des indices sur le type de ResNet
    for key, tensor in state_dict.items():
        if 'backbone.fc' in key or 'backbone.layer4' in key:
            print(f"  {key}: {tensor.shape}")
            
        # Feature dimensions importantes
        if 'classifier.0.weight' in key:
            print(f"\n📊 Dimension d'entrée classifier: {tensor.shape}")
            input_dim = tensor.shape[1]
            if input_dim == 512:
                print("  → ResNet-18/34 détecté (512 features)")
            elif input_dim == 2048:
                print("  → ResNet-50/101/152 détecté (2048 features)")
            else:
                print(f"  → Architecture inconnue ({input_dim} features)")
    
    # Time conditioning
    time_keys = [k for k in state_dict.keys() if 'time' in k]
    print(f"\n⏰ Time conditioning layers: {len(time_keys)}")
    for key in time_keys[:5]:  # Premiers 5
        tensor = state_dict[key]
        print(f"  {key}: {tensor.shape}")
    
    # Classifier architecture
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
    print(f"\n🎯 Classifier layers: {len(classifier_keys)}")
    for key in sorted(classifier_keys):
        tensor = state_dict[key]
        print(f"  {key}: {tensor.shape}")
    
    # Embedding dimension
    for key, tensor in state_dict.items():
        if 'time_conditioning' in key and 'weight' in key:
            print(f"\n🔢 Embedding dimension détectée: {tensor.shape[1]}")
            break
    
    # Informations training
    if 'val_acc' in checkpoint:
        print(f"\n📈 Accuracy validation: {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"📅 Epoch sauvé: {checkpoint['epoch']}")
    if 'config' in checkpoint:
        print(f"⚙️ Config sauvée: {list(checkpoint['config'].keys())}")
    
    # Détecter les "paramètres bizarres"
    print(f"\n🚨 PARAMÈTRES BIZARRES DÉTECTÉS:")
    
    # Vérifier si c'est vraiment ResNet-50
    resnet50_expected_keys = [
        'backbone.layer1.0.conv3.weight',
        'backbone.layer1.0.bn3.weight', 
        'backbone.layer2.0.conv3.weight',
        'backbone.layer3.0.conv3.weight',
        'backbone.layer4.0.conv3.weight'
    ]
    
    resnet18_expected_keys = [
        'backbone.layer1.0.conv2.weight',
        'backbone.layer1.1.conv2.weight'
    ]
    
    has_resnet50_keys = any(key in state_dict for key in resnet50_expected_keys)
    has_resnet18_keys = any(key in state_dict for key in resnet18_expected_keys)
    
    if has_resnet50_keys:
        print("  ✅ ResNet-50 confirmé")
    elif has_resnet18_keys:
        print("  ⚠️ ResNet-18 détecté (pas ResNet-50!)")
    else:
        print("  ❓ Architecture backbone inconnue")
    
    # Vérifier les dimensions attendues vs réelles
    print(f"\n🔄 SOLUTION RECOMMANDÉE:")
    
    # Essayer de détecter la config exacte utilisée
    try:
        classifier_input_dim = state_dict['classifier.0.weight'].shape[1]
        time_embedding_dim = None
        
        for key, tensor in state_dict.items():
            if 'time_conditioning' in key and len(tensor.shape) == 2:
                time_embedding_dim = tensor.shape[1] if 'weight' in key else tensor.shape[0]
                break
        
        print(f"  1. Feature dimension: {classifier_input_dim}")
        print(f"  2. Embedding dimension: {time_embedding_dim}")
        
        if classifier_input_dim == 2048:
            print("  3. Utilise: backbone='resnet50'")
        else:
            print("  3. Utilise: backbone='resnet18'")
            
        print(f"  4. Utilise: embedding_dim={time_embedding_dim}")
        
    except Exception as e:
        print(f"  ❌ Erreur analyse: {e}")

if __name__ == "__main__":
    analyze_afhq_checkpoint()