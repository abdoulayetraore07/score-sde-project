#!/usr/bin/env python3
"""
Analyseur et Visualiseur NPZ/NPY Complet - Version Autonome (Corrigée)
Analyse et visualise les fichiers NPZ et NPY contenant des images générées par IA
Supporte fichiers individuels et dossiers entiers
Compatible avec tous les formats NPZ/NPY d'images IA
NOUVEAU: Création de grilles d'images à partir de dossiers d'images
NOUVEAU: Extraction d'images individuelles depuis une grille
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
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif']
        
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
    
    def split_grid_image(self, grid_path, output_dir=None, rows=None, cols=None, 
                        cell_width=512, cell_height=512, padding=0):
        """
        Extrait les images individuelles d'une grille d'images
        
        Args:
            grid_path (str): Chemin vers l'image de grille
            output_dir (str): Dossier de sortie (défaut: grid_path_split/)
            rows (int): Nombre de lignes dans la grille (optionnel si cell_width/height fournis)
            cols (int): Nombre de colonnes dans la grille (optionnel si cell_width/height fournis)
            cell_width (int): Largeur de chaque cellule en pixels (défaut: 512)
            cell_height (int): Hauteur de chaque cellule en pixels (défaut: 512)
            padding (int): Pixels de padding à ignorer autour de chaque cellule
        """
        try:
            grid_path = Path(grid_path)
            
            if not grid_path.exists():
                print(f"❌ Le fichier {grid_path} n'existe pas")
                return False
            
            # Charger l'image de grille
            grid_img = Image.open(grid_path)
            grid_array = np.array(grid_img)
            
            print(f"📁 Extraction depuis: {grid_path}")
            print(f"📐 Taille de la grille: {grid_img.width}x{grid_img.height}")
            
            # Définir le dossier de sortie
            if output_dir is None:
                output_dir = grid_path.parent / f"{grid_path.stem}_split"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculer rows et cols si pas fournis
            if rows is None:
                rows = grid_img.height // cell_height
            if cols is None:
                cols = grid_img.width // cell_width
            
            print(f"🔳 Grille calculée: {rows} lignes × {cols} colonnes")
            print(f"📏 Taille des cellules: {cell_width}x{cell_height}")
            
            extracted_count = 0
            metadata = {
                'source_grid': str(grid_path),
                'grid_dimensions': f"{rows}x{cols}",
                'cell_size': f"{cell_width}x{cell_height}",
                'padding': padding,
                'extracted_images': []
            }
            
            # Extraire chaque cellule
            for row in range(rows):
                for col in range(cols):
                    # Calculer les coordonnées de la cellule
                    x1 = col * cell_width + padding
                    y1 = row * cell_height + padding
                    x2 = x1 + cell_width - (2 * padding)
                    y2 = y1 + cell_height - (2 * padding)
                    
                    # Vérifier que la cellule est dans l'image
                    if x2 > grid_img.width or y2 > grid_img.height:
                        continue
                    
                    # Extraire la cellule
                    if len(grid_array.shape) == 3:  # Image couleur
                        cell = grid_array[y1:y2, x1:x2, :]
                    else:  # Image en niveaux de gris
                        cell = grid_array[y1:y2, x1:x2]
                    
                    # Sauvegarder la cellule
                    cell_filename = f"cell_{row:02d}_{col:02d}.png"
                    cell_path = output_dir / cell_filename
                    
                    success = self._save_image_array(cell, cell_path)
                    if success:
                        extracted_count += 1
                        metadata['extracted_images'].append({
                            'filename': cell_filename,
                            'position': f"row_{row}_col_{col}",
                            'coordinates': [x1, y1, x2, y2],
                            'size': [x2-x1, y2-y1]
                        })
                        
                        if extracted_count % 5 == 0:
                            print(f"   📝 Extraites: {extracted_count} images...")
            
            # Sauvegarder les métadonnées
            metadata_path = output_dir / 'split_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ {extracted_count} images extraites vers {output_dir}")
            print(f"📄 Métadonnées: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de la grille: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_grid_from_folder(self, folder_path, output_path=None, max_images=None, 
                               cols=None, image_size=(256, 256), background_color=(255, 255, 255)):
        """
        Crée une grille d'images à partir d'un dossier contenant des images
        
        Args:
            folder_path (str): Chemin vers le dossier contenant les images
            output_path (str): Chemin de sortie pour la grille (défaut: dossier_source/grid.png)
            max_images (int): Nombre maximum d'images à inclure (défaut: toutes)
            cols (int): Nombre de colonnes (défaut: calculé automatiquement)
            image_size (tuple): Taille de redimensionnement des images (largeur, hauteur)
            background_color (tuple): Couleur de fond RGB (défaut: blanc)
        """
        try:
            folder_path = Path(folder_path)
            
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"❌ Le dossier {folder_path} n'existe pas ou n'est pas un dossier")
                return False
            
            # Chercher toutes les images dans le dossier
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(list(folder_path.glob(f"*{ext}")))
                image_files.extend(list(folder_path.glob(f"*{ext.upper()}")))
            
            if not image_files:
                print(f"❌ Aucune image trouvée dans {folder_path}")
                print(f"📋 Extensions supportées: {', '.join(self.image_extensions)}")
                return False
            
            # Trier les fichiers par nom pour un ordre cohérent
            image_files.sort()
            
            # Limiter le nombre d'images si nécessaire
            if max_images and len(image_files) > max_images:
                image_files = image_files[:max_images]
                print(f"📊 Limitation à {max_images} images sur {len(image_files)} trouvées")
            
            n_images = len(image_files)
            print(f"🖼️  Traitement de {n_images} images...")
            
            # Calculer la disposition de la grille
            if cols is None:
                cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            
            print(f"🔳 Grille: {rows} lignes × {cols} colonnes")
            
            # Créer la grille
            grid_width = cols * image_size[0]
            grid_height = rows * image_size[1]
            
            # Créer l'image de fond
            if len(background_color) == 3:
                grid = Image.new('RGB', (grid_width, grid_height), background_color)
            else:
                grid = Image.new('RGBA', (grid_width, grid_height), background_color)
            
            # Placer chaque image dans la grille
            for i, img_path in enumerate(image_files):
                try:
                    # Charger et redimensionner l'image
                    img = Image.open(img_path)
                    
                    # Convertir en RGB si nécessaire
                    if img.mode == 'RGBA' and len(background_color) == 3:
                        # Créer un fond blanc pour les images RGBA
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])  # Utiliser le canal alpha comme masque
                        img = background
                    elif img.mode != 'RGB' and len(background_color) == 3:
                        img = img.convert('RGB')
                    elif img.mode != 'RGBA' and len(background_color) == 4:
                        img = img.convert('RGBA')
                    
                    # Redimensionner l'image en gardant les proportions
                    img.thumbnail(image_size, Image.Resampling.LANCZOS)
                    
                    # Centrer l'image dans la cellule
                    cell_x = (i % cols) * image_size[0]
                    cell_y = (i // cols) * image_size[1]
                    
                    # Calculer la position pour centrer l'image
                    paste_x = cell_x + (image_size[0] - img.width) // 2
                    paste_y = cell_y + (image_size[1] - img.height) // 2
                    
                    # Coller l'image dans la grille
                    if img.mode == 'RGBA':
                        grid.paste(img, (paste_x, paste_y), img)
                    else:
                        grid.paste(img, (paste_x, paste_y))
                    
                    if (i + 1) % 10 == 0:
                        print(f"   📝 Traité: {i + 1}/{n_images} images")
                        
                except Exception as e:
                    print(f"⚠️  Erreur avec {img_path}: {e}")
                    continue
            
            # Définir le chemin de sortie
            if output_path is None:
                output_path = folder_path / "grid.png"
            else:
                output_path = Path(output_path)
            
            # Sauvegarder la grille
            grid.save(output_path, format='PNG')
            
            # Créer un fichier de métadonnées
            metadata = {
                'source_folder': str(folder_path),
                'grid_file': str(output_path),
                'total_images': n_images,
                'grid_dimensions': f"{rows}x{cols}",
                'image_size': image_size,
                'background_color': background_color,
                'images_processed': [str(img.name) for img in image_files]
            }
            
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Grille créée: {output_path}")
            print(f"📄 Métadonnées: {metadata_path}")
            print(f"📐 Taille finale: {grid.width}x{grid.height} pixels")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la création de la grille: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
        description="Analyseur et Visualiseur NPZ/NPY/NP Complet + Créateur de grilles d'images + Extracteur de grilles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Analyser un fichier NPZ, NPY ou NP (sans extraction)
  python npz_viewer.py chemin/vers/fichier.npz --analyze-only
  python npz_viewer.py chemin/vers/fichier.npy --analyze-only
  python npz_viewer.py chemin/vers/fichier.np --analyze-only

  # Extraire images individuelles d'un fichier
  python npz_viewer.py chemin/vers/fichier.npz --output ./images --individual
  python npz_viewer.py chemin/vers/fichier.npy --output ./images --individual
  python npz_viewer.py chemin/vers/fichier.np --output ./images --individual

  # Créer une grille d'images d'un fichier
  python npz_viewer.py chemin/vers/fichier.npz --output ./images --grid

  # Traiter un dossier entier avec images individuelles ET grille
  python npz_viewer.py chemin/vers/dossier/ --output ./extractions --individual --grid

  # Analyser tous les fichiers d'un dossier
  python npz_viewer.py chemin/vers/dossier/ --analyze-only

  # NOUVEAU: Créer une grille à partir d'un dossier d'images
  python npz_viewer.py --image-grid chemin/vers/dossier_images/ --output ./grille.png
  python npz_viewer.py --image-grid chemin/vers/dossier_images/ --output ./grille.png --max-images 100 --cols 10 --image-size 128 128

  # NOUVEAU: Extraire images individuelles d'une grille
  python npz_viewer.py --split-grid chemin/vers/grille.png --cell-width 512 --cell-height 512
  python npz_viewer.py --split-grid grille_1030x2058.png --cell-width 515 --cell-height 515 --output ./extraites/
        """)
    
    parser.add_argument("input", nargs='?', help="Chemin vers le fichier NPZ/NPY/NP ou le dossier contenant les fichiers")
    parser.add_argument("--output", "-o", default="./npz_extracted", 
                        help="Dossier de sortie pour les images extraites (défaut: ./npz_extracted)")
    parser.add_argument("--individual", "-i", action="store_true", 
                        help="Extraire les images individuellement")
    parser.add_argument("--grid", "-g", action="store_true", 
                        help="Créer une grille avec toutes les images")
    parser.add_argument("--analyze-only", "-a", action="store_true", 
                        help="Analyser seulement sans extraire les images")
    
    # OPTIONS pour la création de grille d'images
    parser.add_argument("--image-grid", type=str, 
                        help="Créer une grille à partir d'un dossier d'images existantes")
    parser.add_argument("--max-images", type=int, 
                        help="Nombre maximum d'images à inclure dans la grille")
    parser.add_argument("--cols", type=int, 
                        help="Nombre de colonnes dans la grille (défaut: calculé automatiquement)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], 
                        help="Taille de redimensionnement des images [largeur hauteur] (défaut: 256 256)")
    parser.add_argument("--bg-color", type=int, nargs=3, default=[255, 255, 255], 
                        help="Couleur de fond RGB [R G B] (défaut: 255 255 255 pour blanc)")
    
    # NOUVELLES OPTIONS pour l'extraction depuis grille
    parser.add_argument("--split-grid", type=str, 
                        help="Extraire les images individuelles d'une grille d'images")
    parser.add_argument("--rows", type=int, 
                        help="Nombre de lignes dans la grille (optionnel)")
    parser.add_argument("--cell-width", type=int, default=512,
                        help="Largeur de chaque cellule en pixels (défaut: 512)")
    parser.add_argument("--cell-height", type=int, default=512,
                        help="Hauteur de chaque cellule en pixels (défaut: 512)")
    parser.add_argument("--padding", type=int, default=0,
                        help="Pixels de padding à ignorer autour de chaque cellule (défaut: 0)")
    
    args = parser.parse_args()
    
    analyzer = NPZAnalyzer()
    
    # Mode spécial: extraction depuis grille
    if args.split_grid:
        print("🚀 Mode extraction depuis grille - Démarrage...")
        print(f"📁 Grille source: {args.split_grid}")
        
        output_dir = args.output if args.output != "./npz_extracted" else None
        
        success = analyzer.split_grid_image(
            grid_path=args.split_grid,
            output_dir=output_dir,
            rows=args.rows,
            cols=args.cols,
            cell_width=args.cell_width,
            cell_height=args.cell_height,
            padding=args.padding
        )
        
        if success:
            print("✨ Extraction réussie!")
        else:
            print("❌ Échec de l'extraction")
        return
    
    # Mode spécial: création de grille à partir d'un dossier d'images
    if args.image_grid:
        print("🚀 Mode création de grille d'images - Démarrage...")
        print(f"📁 Dossier source: {args.image_grid}")
        
        # Paramètres optionnels
        output_path = args.output if args.output != "./npz_extracted" else None
        
        success = analyzer.create_grid_from_folder(
            folder_path=args.image_grid,
            output_path=output_path,
            max_images=args.max_images,
            cols=args.cols,
            image_size=tuple(args.image_size),
            background_color=tuple(args.bg_color)
        )
        
        if success:
            print("✨ Grille créée avec succès!")
        else:
            print("❌ Échec de la création de la grille")
        return
    
    # Vérifier qu'un input est fourni pour les autres modes
    if not args.input:
        parser.error("Un chemin d'entrée est requis (sauf pour --image-grid ou --split-grid)")
    
    # Valeurs par défaut si aucune option d'extraction n'est spécifiée
    if not args.analyze_only and not args.individual and not args.grid:
        args.individual = True
    
    print("🚀 Analyseur NPZ/NPY/NP - Démarrage...")
    print(f"📁 Entrée: {args.input}")
    
    if not args.analyze_only:
        print(f"📤 Sortie: {args.output}")
        print(f"🖼️  Images individuelles: {'✅' if args.individual else '❌'}")
        print(f"🔳 Grille d'images: {'✅' if args.grid else '❌'}")
    
    analyzer.process_path(args.input, args.output, args.individual, args.grid, args.analyze_only)
    
    print("✨ Terminé!")

if __name__ == "__main__":
    main()