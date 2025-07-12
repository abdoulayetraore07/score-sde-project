#!/usr/bin/env python3
"""
Calculate σ_max for dataset training
Usage: !python calculate_sigma_max.py --dataset_path /path/to/dataset --samples_per_class 100
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import os

def calculate_sigma_max_for_training(dataset, samples_per_class=100, num_classes=3):
    """
    Calculate σ_max for your dataset to use in training
    """
    
    # Get labels
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_labels = np.array(dataset.labels)
    else:
        all_labels = []
        print("Extracting labels from dataset...")
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label)
        all_labels = np.array(all_labels)
    
    print(f"Dataset has {len(all_labels)} images")
    print(f"Classes found: {np.unique(all_labels)}")
    
    # Sample indices
    sampled_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(all_labels == class_id)[0]
        print(f"Class {class_id}: {len(class_indices)} images")
        
        if len(class_indices) >= samples_per_class:
            selected = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            selected = class_indices
            print(f"⚠️  Warning: Only {len(selected)} images available for class {class_id}")
        
        sampled_indices.extend(selected)
    
    print(f"\n📊 Sampled {len(sampled_indices)} images total")
    
    # Get flattened data
    print("Loading and flattening images...")
    sampled_data = []
    for i, idx in enumerate(sampled_indices):
        if i % 50 == 0:
            print(f"Processing {i}/{len(sampled_indices)}")
            
        img, _ = dataset[idx]
        if torch.is_tensor(img):
            img = img.numpy()
        sampled_data.append(img.flatten())
    
    sampled_data = np.array(sampled_data)
    print(f"Final data shape: {sampled_data.shape}")
    
    # Calculate distances and find max
    print("🔄 Computing maximum Euclidean distance...")
    distances = euclidean_distances(sampled_data)
    sigma_max = np.max(distances)
    
    print(f"\n🎯 RESULT:")
    print(f"σ_max = {sigma_max:.6f}")
    print(f"\nUse this value in your model training!")
    
    return sigma_max

def main():
    parser = argparse.ArgumentParser(description='Calculate σ_max for dataset')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to your dataset folder')
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='Number of samples per class (default: 100)')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of classes (default: 3 for AFHQ)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size (default: 256)')
    
    args = parser.parse_args()
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = ImageFolder(root=args.dataset_path, transform=transform)
    
    # Calculate sigma_max
    sigma_max = calculate_sigma_max_for_training(
        dataset, 
        samples_per_class=args.samples_per_class,
        num_classes=args.num_classes
    )
    
    # Save result
    result_file = "sigma_max_result.txt"
    with open(result_file, 'w') as f:
        f.write(f"sigma_max = {sigma_max:.6f}\n")
    
    print(f"\n💾 Result saved to {result_file}")

if __name__ == "__main__":
    main()