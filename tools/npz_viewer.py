#!/usr/bin/env python3
"""
Analyseur et Visualiseur NPZ/NPY Complet - Version Autonome (Corrigée)
Analyse et visualise les fichiers NPZ et NPY contenant des images générées par IA
Supporte fichiers individuels et dossiers entiers
Compatible avec tous les formats NPZ/NPY d'images IA
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class NPZAnalyzer:
    def __init__(self):
        self.supported_extensions = ['.npz', '.npy', '.np']
        
    def analyze_npz_file(self, filepath):
        """Analyse un fichier NPZ/NPY et retourne ses informations"""
        try:
            filepath = Path(filepath)
            
            # Charger le fichier selon son extension
            if filepath.suffix.lower() in ['.npy', '.np']:
                # Fichier NPY/NP - contient un seul array
                array = np.load(filepath, allow_pickle=True)
                data_dict = {'array': array}
                is_single_array = True
            else:
                # Fichier NPZ - peut contenir plusieurs arrays
                data = np.load(filepath, allow_pickle=True)
                
                # Vérifier si c'est un array direct ou un dictionnaire
                if isinstance(data, np.ndarray):
                    # Le fichier contient un seul array (sauvé avec np.save puis renommé en .npz)
                    data_dict = {'array': data}
                    is_single_array = True
                elif hasattr(data, 'files'):
                    # Le fichier contient plusieurs arrays (sauvé avec np.savez)
                    data_dict = {key: data[key] for key in data.files}
                    is_single_array = False
                else:
                    # Cas particulier - essayer de traiter comme un array unique
                    data_dict = {'array': data}
                    is_single_array = True
            
            info = {
                'filepath': str(filepath),
                'filename': filepath.name,
                'file_size': filepath.stat().st_size,
                'is_single_array': is_single_array,
                'arrays': {}
            }
            
            print(f"\n📁 Analyse de: {filepath}")
            print(f"📏 Taille: {info['file_size'] / 1024:.2f} KB")
            print(f"📋 Type: {'Array unique (.npy/.np)' if is_single_array else 'Multi-arrays (.npz)'}")
            print("🔍 Contenu du fichier:")
            
            for key, array in data_dict.items():
                
                # Gestion des objets Python et strings
                if array.dtype == 'O':  # Object dtype
                    try:
                        # Essayer de convertir en string si c'est un objet
                        if array.size == 1:
                            obj_content = array.item()
                            print(f"  🗝️  Clé: '{key}' (Objet Python)")
                            print(f"     📋 Type: {type(obj_content)}")
                            print(f"     📝 Contenu: {str(obj_content)[:100]}{'...' if len(str(obj_content)) > 100 else ''}")
                        else:
                            print(f"  🗝️  Clé: '{key}' (Array d'objets)")
                            print(f"     📐 Shape: {array.shape}")
                            print(f"     📝 Premier élément: {str(array.flat[0])[:50]}{'...' if len(str(array.flat[0])) > 50 else ''}")
                    except:
                        print(f"  🗝️  Clé: '{key}' (Objet non lisible)")
                        print(f"     📐 Shape: {array.shape}")
                    continue
                
                # Gestion des arrays numériques normaux
                info['arrays'][key] = {
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'size': array.size,
                    'min': float(np.min(array)) if array.size > 0 and np.issubdtype(array.dtype, np.number) else None,
                    'max': float(np.max(array)) if array.size > 0 and np.issubdtype(array.dtype, np.number) else None,
                    'mean': float(np.mean(array)) if array.size > 0 and np.issubdtype(array.dtype, np.number) else None
                }
                
                print(f"  🗝️  Clé: '{key}'")
                print(f"     📐 Shape: {array.shape}")
                print(f"     🏷️  Type: {array.dtype}")
                
                if np.issubdtype(array.dtype, np.number):
                    print(f"     📊 Min/Max/Mean: {np.min(array):.3f} / {np.max(array):.3f} / {np.mean(array):.3f}")
                
                # Détection du type de données
                if np.issubdtype(array.dtype, np.number):
                    if len(array.shape) >= 3:
                        if array.shape[-1] in [3, 4]:
                            print(f"     🖼️  Probablement des images (RGB/RGBA)")
                        elif len(array.shape) == 4:
                            print(f"     🖼️  Probablement un batch d'images ({array.shape[0]} images)")
                        elif array.shape[0] in [3, 4] and len(array.shape) == 3:
                            print(f"     🖼️  Probablement une image (format CHW)")
                    elif len(array.shape) == 2:
                        print(f"     🖼️  Probablement une image en niveaux de gris")
                    elif len(array.shape) == 1:
                        print(f"     📋 Probablement des métadonnées ou paramètres")
                else:
                    print(f"     📝 Données non-numériques")
            
            # Fermer le fichier NPZ si nécessaire
            if not is_single_array and hasattr(data, 'close'):
                data.close()
                
            return info
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_images_from_npz(self, filepath, output_dir, individual=True, grid=False):
        """Extrait et sauvegarde les images d'un fichier NPZ/NPY"""
        try:
            filepath = Path(filepath)
            
            # Charger le fichier selon son extension
            if filepath.suffix.lower() in ['.npy', '.np']:
                # Fichier NPY/NP - contient un seul array
                array = np.load(filepath, allow_pickle=True)
                data_dict = {'array': array}
                is_single_array = True
            else:
                # Fichier NPZ - peut contenir plusieurs arrays
                data = np.load(filepath, allow_pickle=True)
                
                # Vérifier si c'est un array direct ou un dictionnaire
                if isinstance(data, np.ndarray):
                    # Le fichier contient un seul array
                    data_dict = {'array': data}
                    is_single_array = True
                elif hasattr(data, 'files'):
                    # Le fichier contient plusieurs arrays
                    data_dict = {key: data[key] for key in data.files}
                    is_single_array = False
                else:
                    # Cas particulier
                    data_dict = {'array': data}
                    is_single_array = True
            
            filename_base = filepath.stem
            
            # Créer le dossier de sortie
            output_path = Path(output_dir) / filename_base
            output_path.mkdir(parents=True, exist_ok=True)
            
            images_found = 0
            all_images = []
            metadata = {
                'source_file': str(filepath),
                'file_type': 'np/npy' if filepath.suffix.lower() in ['.npy', '.np'] else 'npz',
                'is_single_array': is_single_array,
                'arrays_info': {},
                'extraction_info': {}
            }
            
            for key, array in data_dict.items():
                
                # Ignorer les objets non-numériques
                if array.dtype == 'O':
                    try:
                        obj_content = array.item() if array.size == 1 else array
                        metadata['arrays_info'][key] = {
                            'type': 'object',
                            'content': str(obj_content)[:200],
                            'shape': list(array.shape)
                        }
                    except:
                        metadata['arrays_info'][key] = {
                            'type': 'object',
                            'content': 'Non-readable object',
                            'shape': list(array.shape)
                        }
                    continue
                
                # Traiter seulement les arrays numériques
                if not np.issubdtype(array.dtype, np.number):
                    continue
                
                # Détecter si c'est une image ou un batch d'images
                if len(array.shape) >= 2:
                    images = self._process_array_to_images(array, key)
                    
                    if images:
                        all_images.extend(images)
                        
                        if individual:
                            for i, img_array in enumerate(images):
                                if len(images) == 1:
                                    img_filename = f"{filename_base}_{key}.png"
                                else:
                                    img_filename = f"{filename_base}_{key}_{i:03d}.png"
                                
                                img_path = output_path / img_filename
                                success = self._save_image_array(img_array, img_path)
                                if success:
                                    images_found += 1
                        
                        metadata['extraction_info'][key] = {
                            'images_extracted': len(images),
                            'image_shape': list(images[0].shape) if images else None
                        }
                
                # Sauvegarder les infos de l'array
                metadata['arrays_info'][key] = {
                    'shape': list(array.shape),
                    'dtype': str(array.dtype),
                    'min': float(np.min(array)) if array.size > 0 else None,
                    'max': float(np.max(array)) if array.size > 0 else None,
                    'mean': float(np.mean(array)) if array.size > 0 else None
                }
            
            # Créer une grille si demandé
            if grid and all_images:
                grid_path = output_path / f"{filename_base}_grid.png"
                self._create_image_grid(all_images, grid_path)
                metadata['grid_created'] = True
            
            metadata['images_extracted'] = images_found
            
            # Sauvegarder les métadonnées
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Fermer le fichier NPZ si nécessaire
            if not is_single_array and hasattr(data, 'close'):
                data.close()
                
            print(f"✅ {images_found} images extraites vers {output_path}")
            return images_found
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _process_array_to_images(self, array, key_name):
        """Convertit un array numpy en liste d'images"""
        images = []
        
        try:
            # Copie pour éviter de modifier l'original
            work_array = array.copy()
            
            # Normaliser les valeurs si nécessaire
            if work_array.dtype in [np.float32, np.float64]:
                if np.max(work_array) <= 1.0 and np.min(work_array) >= 0.0:
                    # Valeurs entre 0-1, normaliser vers 0-255
                    work_array = (work_array * 255).astype(np.uint8)
                elif np.max(work_array) <= 2.0 and np.min(work_array) >= -1.0:
                    # Valeurs entre -1 et 1, normaliser vers 0-255
                    work_array = ((work_array + 1.0) * 127.5).astype(np.uint8)
                else:
                    # Valeurs arbitraires, clipper vers 0-255
                    work_array = np.clip(work_array, 0, 255).astype(np.uint8)
            elif work_array.dtype == np.uint8:
                # Déjà en bon format
                pass
            else:
                # Autres types, essayer de convertir
                work_array = np.clip(work_array.astype(np.float32), 0, 255).astype(np.uint8)
            
            if len(work_array.shape) == 4:  # Batch d'images (N, H, W, C) ou (N, C, H, W)
                for i in range(work_array.shape[0]):
                    img = work_array[i]
                    
                    # Vérifier le format (C, H, W) vs (H, W, C)
                    if img.shape[0] in [1, 3, 4] and img.shape[0] < min(img.shape[1], img.shape[2]):
                        # Format (C, H, W) -> (H, W, C)
                        img = np.transpose(img, (1, 2, 0))
                    
                    if len(img.shape) == 3 and img.shape[-1] in [3, 4]:  # RGB ou RGBA
                        images.append(img)
                    elif len(img.shape) == 3 and img.shape[-1] == 1:  # Mono
                        images.append(img.squeeze(-1))
                    elif len(img.shape) == 2:  # Déjà 2D
                        images.append(img)
                        
            elif len(work_array.shape) == 3:  # Une seule image
                # Vérifier le format (C, H, W) vs (H, W, C)
                if work_array.shape[0] in [1, 3, 4] and work_array.shape[0] < min(work_array.shape[1], work_array.shape[2]):
                    # Format (C, H, W) -> (H, W, C)
                    work_array = np.transpose(work_array, (1, 2, 0))
                
                if work_array.shape[-1] in [3, 4]:  # RGB ou RGBA
                    images.append(work_array)
                elif work_array.shape[-1] == 1:  # Mono
                    images.append(work_array.squeeze(-1))
                else:
                    # Essayer comme image 2D si les dimensions sont cohérentes
                    if len(work_array.shape) == 3:
                        # Prendre le premier canal si multiples
                        images.append(work_array[:, :, 0])
                    
            elif len(work_array.shape) == 2:  # Image en niveaux de gris
                images.append(work_array)
                
        except Exception as e:
            print(f"⚠️  Erreur lors du traitement de l'array '{key_name}': {e}")
            
        return images
    
    def _save_image_array(self, img_array, filepath):
        """Sauvegarde un array numpy comme image"""
        try:
            if len(img_array.shape) == 2:  # Niveaux de gris
                img = Image.fromarray(img_array, mode='L')
            elif len(img_array.shape) == 3:
                if img_array.shape[-1] == 3:  # RGB
                    img = Image.fromarray(img_array, mode='RGB')
                elif img_array.shape[-1] == 4:  # RGBA
                    img = Image.fromarray(img_array, mode='RGBA')
                else:
                    return False
            else:
                return False
                
            img.save(filepath)
            return True
        except Exception as e:
            print(f"⚠️  Erreur lors de la sauvegarde de {filepath}: {e}")
            return False
    
    def _create_image_grid(self, images, output_path):
        """Crée une grille d'images"""
        if not images:
            return
            
        # Calculer la disposition de la grille
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        
        # Déterminer la taille des images
        sample_img = images[0]
        if len(sample_img.shape) == 2:
            img_h, img_w = sample_img.shape
            n_channels = 1
        else:
            img_h, img_w, n_channels = sample_img.shape
        
        # Créer la grille
        if n_channels == 1:
            grid = np.ones((rows * img_h, cols * img_w), dtype=np.uint8) * 255  # Fond blanc
        else:
            grid = np.ones((rows * img_h, cols * img_w, n_channels), dtype=np.uint8) * 255
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            
            start_row = row * img_h
            end_row = start_row + img_h
            start_col = col * img_w
            end_col = start_col + img_w
            
            # Redimensionner l'image si nécessaire
            if img.shape[:2] != (img_h, img_w):
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((img_w, img_h))
                img = np.array(img_pil)
            
            if len(grid.shape) == 2:
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2).astype(np.uint8)
                grid[start_row:end_row, start_col:end_col] = img
            else:
                if len(img.shape) == 2:
                    img = np.stack([img] * n_channels, axis=2)
                grid[start_row:end_row, start_col:end_col] = img
        
        success = self._save_image_array(grid, output_path)
        if success:
            print(f"🖼️  Grille créée: {output_path}")
    
    def process_path(self, input_path, output_dir, individual=True, grid=False, analyze_only=False):
        """Traite un fichier ou un dossier"""
        input_path = Path(input_path)
        
        if input_path.is_file():
            if input_path.suffix.lower() in self.supported_extensions:
                self.analyze_npz_file(input_path)
                if not analyze_only:
                    self.extract_images_from_npz(input_path, output_dir, individual, grid)
            else:
                print(f"❌ Extension non supportée: {input_path.suffix}")
                print(f"📋 Extensions supportées: {', '.join(self.supported_extensions)}")
                
        elif input_path.is_dir():
            # Chercher les fichiers NPZ, NPY et NP
            supported_files = []
            for ext in self.supported_extensions:
                supported_files.extend(list(input_path.glob(f"*{ext}")))
            
            if not supported_files:
                print(f"❌ Aucun fichier NPZ/NPY/NP trouvé dans {input_path}")
                return
                
            print(f"📂 Traitement de {len(supported_files)} fichiers...")
            total_images = 0
            
            for file_path in supported_files:
                print(f"\n{'='*60}")
                self.analyze_npz_file(file_path)
                if not analyze_only:
                    images_count = self.extract_images_from_npz(file_path, output_dir, individual, grid)
                    total_images += images_count
            
            if not analyze_only:
                print(f"\n🎉 Total: {total_images} images extraites!")
        else:
            print(f"❌ Chemin invalide: {input_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyseur et Visualiseur NPZ/NPY/NP Complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Analyser un fichier NPZ, NPY ou NP (sans extraction)
  python npz_analyzer.py chemin/vers/fichier.npz --analyze-only
  python npz_analyzer.py chemin/vers/fichier.npy --analyze-only
  python npz_analyzer.py chemin/vers/fichier.np --analyze-only

  # Extraire images individuelles d'un fichier
  python npz_analyzer.py chemin/vers/fichier.npz --output ./images --individual
  python npz_analyzer.py chemin/vers/fichier.npy --output ./images --individual
  python npz_analyzer.py chemin/vers/fichier.np --output ./images --individual

  # Créer une grille d'images d'un fichier
  python npz_analyzer.py chemin/vers/fichier.npz --output ./images --grid

  # Traiter un dossier entier avec images individuelles ET grille
  python npz_analyzer.py chemin/vers/dossier/ --output ./extractions --individual --grid

  # Analyser tous les fichiers d'un dossier
  python npz_analyzer.py chemin/vers/dossier/ --analyze-only
        """)
    
    parser.add_argument("input", help="Chemin vers le fichier NPZ/NPY/NP ou le dossier contenant les fichiers")
    parser.add_argument("--output", "-o", default="./npz_extracted", 
                        help="Dossier de sortie pour les images extraites (défaut: ./npz_extracted)")
    parser.add_argument("--individual", "-i", action="store_true", 
                        help="Extraire les images individuellement")
    parser.add_argument("--grid", "-g", action="store_true", 
                        help="Créer une grille avec toutes les images")
    parser.add_argument("--analyze-only", "-a", action="store_true", 
                        help="Analyser seulement sans extraire les images")
    
    args = parser.parse_args()
    
    # Valeurs par défaut si aucune option d'extraction n'est spécifiée
    if not args.analyze_only and not args.individual and not args.grid:
        args.individual = True
    
    print("🚀 Analyseur NPZ/NPY/NP - Démarrage...")
    print(f"📁 Entrée: {args.input}")
    
    if not args.analyze_only:
        print(f"📤 Sortie: {args.output}")
        print(f"🖼️  Images individuelles: {'✅' if args.individual else '❌'}")
        print(f"🔳 Grille d'images: {'✅' if args.grid else '❌'}")
    
    analyzer = NPZAnalyzer()
    analyzer.process_path(args.input, args.output, args.individual, args.grid, args.analyze_only)
    
    print("✨ Terminé!")

if __name__ == "__main__":
    main()