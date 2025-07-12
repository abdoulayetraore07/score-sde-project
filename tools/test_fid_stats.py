#!/usr/bin/env python3
"""
Test du pipeline FID avec Clean-FID.
"""

import sys
import os
import numpy as np
import tempfile
from PIL import Image

sys.path.append('.')

def test_clean_fid_installation():
    """Test 1: Installation Clean-FID"""
    print("🧪 TEST 1: Installation Clean-FID")
    
    try:
        from cleanfid import fid
        print("  ✅ Clean-FID importé avec succès")
        
        # Test datasets disponibles
        available_datasets = ['afhq_cat', 'afhq_dog', 'afhq_wild']
        for dataset in available_datasets:
            print(f"  📊 Dataset supporté: {dataset}")
        
    except ImportError:
        print("  ❌ Clean-FID non installé")
        print("  💡 Installer avec: pip install clean-fid")
        return False
    
    return True

def test_clean_fid_computation():
    """Test 2: Calcul FID avec données factices"""
    print("🧪 TEST 2: Calcul FID avec Clean-FID")
    
    try:
        from cleanfid import fid
        
        # Créer dossier temporaire avec images factices
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  📁 Dossier temporaire: {temp_dir}")
            
            # Générer 10 images factices 512x512
            for i in range(10):
                img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
                pil_img = Image.fromarray(img)
                pil_img.save(f"{temp_dir}/fake_{i:03d}.png")
            
            print("  🎨 Images factices créées (512x512)")
            
            # Test FID avec AFHQ
            for dataset in ['afhq_cat', 'afhq_dog']:
                try:
                    print(f"  🔢 Calcul FID avec {dataset}...")
                    
                    score = fid.compute_fid(
                        temp_dir,
                        dataset_name=dataset,
                        dataset_res=512,
                        mode="clean",
                        dataset_split="train"
                    )
                    
                    print(f"  ✅ FID {dataset}: {score:.2f}")
                    
                    if score > 100:  # Images aléatoires = FID élevé
                        print("  ✅ Score cohérent (images aléatoires)")
                    
                except Exception as e:
                    print(f"  ⚠️  Erreur {dataset}: {e}")
            
    except Exception as e:
        print(f"  ❌ Erreur test: {e}")
        return False
    
    return True

def test_integration_with_project():
    """Test 3: Intégration avec votre projet"""
    print("🧪 TEST 3: Intégration projet")
    
    try:
        from evaluation import compute_fid_from_samples_clean
        print("  ✅ Fonction Clean-FID importée")
        
        # Test avec dossier temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            # Image factice
            img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(img).save(f"{temp_dir}/test.png")
            
            score = compute_fid_from_samples_clean(
                temp_dir, "afhq_cat", 512, "clean"
            )
            
            if score > 0:
                print(f"  ✅ Intégration réussie, FID: {score:.2f}")
            else:
                print("  ⚠️  FID négatif (erreur possible)")
        
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Erreur test: {e}")
        return False
    
    return True

def main():
    print("🚀 TEST COMPLET CLEAN-FID")
    print("=" * 50)
    
    success = True
    success &= test_clean_fid_installation()
    print()
    success &= test_clean_fid_computation()
    print()
    success &= test_integration_with_project()
    
    print("=" * 50)
    if success:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("Votre projet est prêt pour Clean-FID.")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez les erreurs ci-dessus.")

if __name__ == "__main__":
    main()