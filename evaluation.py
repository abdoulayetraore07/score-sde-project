# Based on score_sde_pytorch
# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Enhanced evaluation functions with robust FID/IS computation."""

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_hub as tfhub
import torch
import os
import logging
from PIL import Image
from tqdm import tqdm

# FID/IS imports
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    from torchvision import transforms
    import scipy.linalg
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logging.warning("pytorch-fid not available. Install with: pip install pytorch-fid")

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
    if inceptionv3:
        return tfhub.load(
            'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
    else:
        return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(config, use_official=True, force_recompute=False, use_clean_fid=True):
    """
    Load dataset statistics - Clean-FID par défaut, fallback si nécessaire.
    """
    if use_clean_fid:
        # Clean-FID gère automatiquement les stats, pas besoin de les charger
        logging.info("✅ Using Clean-FID (no manual stats loading needed)")
        return "clean-fid-managed"
    
    else :
        logging.info("📊 Fallback to manual stats loading...")
        dataset = config.data.dataset
        image_size = config.data.image_size
        
        # Déterminer le nom de fichier selon le dataset
        if dataset == 'CIFAR10':
            filename = 'assets/stats/cifar10_stats.npz'
        elif dataset == 'CELEBA':
            filename = 'assets/stats/celeba_stats.npz'
        elif dataset == 'LSUN':
            filename = f'assets/stats/lsun_{config.data.category}_{image_size}_stats.npz'
        elif dataset == 'AFHQ':
            filename = f'assets/stats/afhq_{image_size}_stats.npz'
        elif dataset == 'FFHQ':
            filename = f'assets/stats/ffhq_{image_size}_stats.npz'
        else:
            raise ValueError(f'Dataset {dataset} stats not supported.')
        
        # Si force_recompute, supprimer le fichier existant
        if force_recompute and tf.io.gfile.exists(filename):
            logging.info(f"🔄 Force recompute: suppression {filename}")
            tf.io.gfile.remove(filename)
        
        # Essayer de charger les stats existantes
        if tf.io.gfile.exists(filename):
            logging.info(f"📊 Chargement stats existantes: {filename}")
            try:
                with tf.io.gfile.GFile(filename, 'rb') as fin:
                    stats = np.load(fin)
                    
                    # Adapter le format si nécessaire (CleanFID vs notre format)
                    if 'mu' in stats and 'sigma' in stats:
                        # Notre format standard
                        logging.info(f"✅ Stats format standard chargées: mu {stats['mu'].shape}, sigma {stats['sigma'].shape}")
                        return stats
                    elif 'inception_features' in stats:
                        # Format CleanFID - convertir
                        logging.info("🔄 Conversion format CleanFID vers format standard...")
                        features = stats['inception_features']
                        mu = np.mean(features, axis=0)
                        sigma = np.cov(features, rowvar=False)
                        converted_stats = {'mu': mu, 'sigma': sigma}
                        
                        # Sauvegarder le format converti pour la prochaine fois
                        try:
                            with tf.io.gfile.GFile(filename, 'wb') as fout:
                                np.savez_compressed(fout, mu=mu, sigma=sigma)
                            logging.info("✅ Format converti et sauvegardé")
                        except Exception as e:
                            logging.warning(f"⚠️  Impossible de sauvegarder format converti: {e}")
                        
                        return converted_stats
                    else:
                        logging.warning(f"⚠️  Format stats non reconnu dans {filename}")
                        # Continuer vers téléchargement/calcul
                        
            except Exception as e:
                logging.warning(f"⚠️  Erreur chargement {filename}: {e}")
                # Continuer vers téléchargement/calcul
        
        # Stats non trouvées ou invalides
        logging.info(f"📭 Stats non trouvées pour {dataset} {image_size}px")
        
        # Essayer de télécharger les stats officielles si demandé
        if use_official:
            logging.info("🌐 Tentative téléchargement stats officielles...")
            downloaded_file = download_official_stats(dataset, image_size)
            
            if downloaded_file:
                # Réessayer de charger les stats téléchargées
                try:
                    with tf.io.gfile.GFile(downloaded_file, 'rb') as fin:
                        stats = np.load(fin)
                        
                    if 'inception_features' in stats:
                        # Convertir format CleanFID
                        features = stats['inception_features']
                        mu = np.mean(features, axis=0)
                        sigma = np.cov(features, rowvar=False)
                        
                        # Sauvegarder au format standard
                        try:
                            with tf.io.gfile.GFile(downloaded_file, 'wb') as fout:
                                np.savez_compressed(fout, mu=mu, sigma=sigma)
                        except Exception as e:
                            logging.warning(f"⚠️  Sauvegarde format standard échouée: {e}")
                        
                        return {'mu': mu, 'sigma': sigma}
                    elif 'mu' in stats and 'sigma' in stats:
                        return stats
                        
                except Exception as e:
                    logging.warning(f"⚠️  Erreur chargement stats téléchargées: {e}")
        
        # Fallback: calcul local des stats (si dataset supporté)
        if dataset == 'AFHQ':
            logging.info("🔄 Fallback: calcul local des statistiques...")
            logging.info("⏱️  Cela peut prendre 10-30 minutes selon la taille du dataset...")
            
            try:
                stats_file = save_dataset_statistics(config, output_dir='assets/stats')
                if stats_file:
                    # Recharger les stats calculées
                    with tf.io.gfile.GFile(stats_file, 'rb') as fin:
                        stats = np.load(fin)
                        logging.info("✅ Stats calculées localement avec succès")
                        return stats
            except Exception as e:
                logging.error(f"❌ Erreur calcul local des stats: {e}")
        
        # Échec total
        logging.warning(f"❌ Impossible d'obtenir les stats pour {dataset} {image_size}px")
        logging.info("💡 Options disponibles:")
        logging.info("  1. Télécharger manuellement les stats")
        logging.info("  2. Calculer avec: python compute_dataset_stats.py afhq_512")
        logging.info("  3. Vérifier que le dataset AFHQ est bien dans data/afhq/")
        
        return None


def compute_fid_multi_resolution(samples_dict, config, target_resolutions=[512], 
                               use_clean_fid=True, mode="clean", dataset_name="afhq", 
                               samples_by_class=None):
    """
    Compute FID for multiple resolutions - Enhanced avec Clean-FID et support par classe.
    
    Args:
        samples_dict: Dict {resolution: samples_array}
        samples_by_class: Dict {resolution: {class_name: samples_array}} pour FID par classe
    """
    if not target_resolutions:
        target_resolutions = [config.data.image_size]
    
    results = {}
    
    for resolution in target_resolutions:
        logging.info(f"📊 Computing FID for {resolution}px...")
        
        if use_clean_fid:
            # NOUVEAU: Support FID par classe
            if samples_by_class and resolution in samples_by_class:
                logging.info("🎯 Computing class-specific FID...")
                class_results = {}
                
                for class_name, class_samples in samples_by_class[resolution].items():
                    # Créer dossier temporaire pour cette classe
                    samples_dir = f"temp_samples_{resolution}px_{class_name}"
                    os.makedirs(samples_dir, exist_ok=True)
                    
                    # Sauvegarder échantillons de cette classe
                    for idx, sample in enumerate(class_samples):  
                        img = Image.fromarray(sample.astype(np.uint8))
                        img.save(f"{samples_dir}/sample_{idx:04d}.png")
                    
                    # Calculer FID pour cette classe spécifique
                    dataset_mapping = {
                        'cat': 'afhq_cat',
                        'dog': 'afhq_dog', 
                        'wild': 'afhq_wild'
                    }
                    clean_dataset = dataset_mapping.get(class_name, f'afhq_{class_name}')
                    
                    fid_score = compute_fid_from_samples_clean(
                        samples_dir, clean_dataset, resolution, mode
                    )
                    
                    # Récupérer le score (compute_fid_from_samples_clean retourne un dict)
                    if isinstance(fid_score, dict):
                        class_results[class_name] = fid_score[clean_dataset]
                    else:
                        class_results[class_name] = fid_score
                    
                    logging.info(f"✅ FID {class_name}: {class_results[class_name]:.4f}")
                    
                    # Nettoyer
                    import shutil
                    shutil.rmtree(samples_dir, ignore_errors=True)
                
                # Calculer FID moyen pondéré
                class_counts = {k: len(v) for k, v in samples_by_class[resolution].items()}
                total_samples = sum(class_counts.values())
                
                weighted_fid = sum(
                    class_results[cls] * (class_counts[cls] / total_samples) 
                    for cls in class_results
                )
                
                results[resolution] = {
                    'overall': weighted_fid,
                    'by_class': class_results,
                    'class_counts': class_counts
                }
                
                logging.info(f"📊 Resolution {resolution}px FID Summary:")
                for cls, score in class_results.items():
                    logging.info(f"   → {cls}: {score:.4f} ({class_counts[cls]} samples)")
                logging.info(f"   → Weighted Average: {weighted_fid:.4f}")
            
            else:
                # ANCIEN: FID global (fallback)
                logging.info("🌍 Computing global FID...")
                samples_dir = f"temp_samples_{resolution}px"
                os.makedirs(samples_dir, exist_ok=True)
                
                if resolution in samples_dict:
                    # Sauvegarder images
                    samples = samples_dict[resolution]
                    for idx, sample in enumerate(samples):  
                        img = Image.fromarray(sample.astype(np.uint8))
                        img.save(f"{samples_dir}/sample_{idx:04d}.png")
                    
                    # Calculer FID avec Clean-FID (mode ancien)
                    if isinstance(dataset_name, list):
                        # Plusieurs datasets - prendre le premier ou faire moyenne
                        primary_dataset = dataset_name[0]
                    else:
                        primary_dataset = dataset_name
                    
                    fid_score = compute_fid_from_samples_clean(
                        samples_dir, primary_dataset, resolution, mode
                    )
                    
                    if isinstance(fid_score, dict):
                        results[resolution] = fid_score[primary_dataset]
                    else:
                        results[resolution] = fid_score
                    
                    # Nettoyer
                    import shutil
                    shutil.rmtree(samples_dir, ignore_errors=True)
                else:
                    logging.warning(f"⚠️  No samples for {resolution}px")
                    results[resolution] = -1
        else:

            if not FID_AVAILABLE:
                logging.warning("pytorch-fid not available. Returning empty results.")
                return {}
            
            results = {}
            
            for resolution in target_resolutions:
                if resolution not in samples_dict:
                    logging.warning(f"⚠️  Samples {resolution}px non disponibles, ignoré")
                    continue
                
                logging.info(f"📊 Calcul FID pour résolution {resolution}px...")
                
                # Créer config temporaire pour cette résolution
                temp_config = type(config)(config.copy())
                temp_config.data.image_size = resolution
                
                # Charger les stats pour cette résolution
                stats = load_dataset_stats(temp_config, use_official=True)
                
                if stats is not None:
                    try:
                        fid_score = compute_fid_from_samples(
                            samples_dict[resolution], 
                            stats, 
                            device=config.device
                        )
                        results[resolution] = fid_score
                        logging.info(f"✅ FID {resolution}px: {fid_score:.4f}")
                    except Exception as e:
                        logging.error(f"❌ Erreur calcul FID {resolution}px: {e}")
                        results[resolution] = -1
                else:
                    logging.warning(f"⚠️  Stats {resolution}px non disponibles")
                    results[resolution] = -1
                
            pass
            
        return results


def resize_samples_to_resolution(samples, target_resolution):
    """
    Resize samples to target resolution for multi-resolution FID.
    
    Args:
        samples: numpy array [N, H, W, C] in [0, 255]
        target_resolution: target size (e.g., 256)
        
    Returns:
        numpy array: resized samples [N, target_resolution, target_resolution, C]
    """
    from PIL import Image
    
    if samples.shape[1] == target_resolution and samples.shape[2] == target_resolution:
        return samples
    
    resized_samples = []
    
    logging.info(f"🔄 Resize {samples.shape[1]}px → {target_resolution}px pour {len(samples)} images...")
    
    for i in tqdm(range(len(samples)), desc=f"Resize to {target_resolution}px"):
        # Convertir en PIL Image
        img = Image.fromarray(samples[i].astype(np.uint8))
        
        # Resize avec high-quality resampling
        img_resized = img.resize((target_resolution, target_resolution), Image.LANCZOS)
        
        # Reconvertir en numpy
        resized_samples.append(np.array(img_resized))
    
    return np.array(resized_samples)


def compute_fid_from_samples(samples, dataset_stats=None, device='cuda', 
                           use_clean_fid=True, samples_dir=None, 
                           dataset_name='afhq', resolution=512, mode='clean'):
    """
    Compute FID - Clean-FID par défaut, fallback vers ancienne méthode.
    """
    if use_clean_fid and samples_dir:
        return compute_fid_from_samples_clean(samples_dir, dataset_name, resolution, mode, device)
    else:
        logging.warning("⚠️  Using legacy FID computation")

        if not FID_AVAILABLE:
            logging.warning("pytorch-fid not available. Returning -1.")
            return -1.0
        
        # Initialize Inception model
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        
        # Preprocess samples
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        activations = []
        batch_size = 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), batch_size), desc="Computing FID"):
                batch_samples = samples[i:i+batch_size]
                batch_tensors = []
                
                for sample in batch_samples:
                    # Ensure sample is in [0, 255] uint8
                    if sample.dtype != np.uint8:
                        sample = np.clip(sample, 0, 255).astype(np.uint8)
                    
                    tensor = transform(sample)
                    batch_tensors.append(tensor)
                
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    pred = model(batch_tensor)[0]
                    
                    # Flatten spatial dimensions
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
                    
                    activations.append(pred.cpu().numpy().reshape(pred.size(0), -1))
        
        # Compute sample statistics
        activations = np.concatenate(activations, axis=0)
        mu_samples = np.mean(activations, axis=0)
        sigma_samples = np.cov(activations, rowvar=False)
        
        # Compute FID
        fid_value = calculate_frechet_distance(
            dataset_stats['mu'], dataset_stats['sigma'],
            mu_samples, sigma_samples
        )
        
        return fid_value 



def compute_fid_from_samples_clean(samples_dir, dataset_names, resolution, mode="clean", device='cuda'):
    """
    Compute FID using Clean-FID - SUPPORT MULTI-DATASETS.
    
    Args:
        dataset_names: Liste ['afhq_cat', 'afhq_dog', 'afhq_wild'] ou string unique
    Returns:
        dict: {dataset_name: fid_score} OU float si un seul dataset
    """
    try:
        from cleanfid import fid
        
        # Support liste de datasets ET string unique
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        results = {}
        
        for dataset_name in dataset_names:
            dataset_mapping = {
                'afhq': 'afhq_cat',
                'afhq_cat': 'afhq_cat',
                'afhq_dog': 'afhq_dog', 
                'afhq_wild': 'afhq_wild'
            }
            
            clean_dataset = dataset_mapping.get(dataset_name.lower(), 'afhq_cat')
            
            logging.info(f"🎯 Computing Clean-FID: {clean_dataset} @ {resolution}px")
            
            score = fid.compute_fid(
                samples_dir,
                dataset_name=clean_dataset,
                dataset_res=resolution,
                mode=mode,
                dataset_split="train"
            )
            
            results[clean_dataset] = score
            logging.info(f"✅ Clean-FID {clean_dataset}: {score:.4f}")
        
        # RETOUR COHÉRENT
        if len(results) == 1:
            # Un seul dataset - retourner le score directement
            return list(results.values())[0]
        else:
            # Plusieurs datasets - retourner le dict
            return results
        
    except Exception as e:
        logging.error(f"❌ Clean-FID error: {e}")
        if isinstance(dataset_names, str):
            return -1.0
        else:
            return {name: -1.0 for name in dataset_names}
        

    
def compute_inception_score(samples, device='cuda', splits=10):
    """
    Compute Inception Score from generated samples.
    
    Args:
        samples: Generated samples as numpy array [N, H, W, C] in [0, 255]
        device: Device to use
        splits: Number of splits for computing IS
        
    Returns:
        is_mean, is_std: Inception Score mean and standard deviation
    """
    if not FID_AVAILABLE:
        logging.warning("pytorch-fid not available. Returning -1, -1.")
        return -1.0, -1.0
    
    # Initialize Inception model for IS (needs logits)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[1008]]).to(device)  # logits
    model.eval()
    
    # Preprocess samples
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get predictions
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Computing IS"):
            batch_samples = samples[i:i+batch_size]
            batch_tensors = []
            
            for sample in batch_samples:
                # Ensure sample is in [0, 255] uint8
                if sample.dtype != np.uint8:
                    sample = np.clip(sample, 0, 255).astype(np.uint8)
                
                tensor = transform(sample)
                batch_tensors.append(tensor)
            
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(device)
                pred = model(batch_tensor)[0]
                predictions.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Compute IS
    N = predictions.shape[0]
    split_size = N // splits
    
    scores = []
    for i in range(splits):
        part = predictions[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        kl_divs = []
        for j in range(part.shape[0]):
            pyx = part[j, :]
            kl_div = np.sum(pyx * np.log(pyx / py + 1e-16))
            kl_divs.append(kl_div)
        scores.append(np.exp(np.mean(kl_divs)))
    
    return np.mean(scores), np.std(scores)


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
    """Returns a function that can be as a classifier function.

    Copied from tfgan but avoid loading the model each time calling _classifier_fn

    Args:
        output_fields: A string, list, or `None`. If present, assume the module
        outputs a dictionary, and select this field.
        inception_model: A model loaded from TFHub.
        return_tensor: If `True`, return a single tensor instead of a dictionary.

    Returns:
        A one-argument function that takes an image Tensor and returns outputs.
    """
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]

    def _classifier_fn(images):
        output = inception_model(images)
        if output_fields is not None:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output = list(output.values())[0]
        return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

    return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
    """Running the inception network. Assuming input is within [0, 255]."""
    if not inceptionv3:
        inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
    else:
        inputs = tf.cast(inputs, tf.float32) / 255.

    return {
        'pool_3': inception_model(inputs),
        'logits': None if inceptionv3 else inception_model(inputs)
    }


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
    """Distribute the inception network computation to all available TPUs.

    Args:
        input_tensor: The input images. Assumed to be within [0, 255].
        inception_model: The inception network model obtained from `tfhub`.
        num_batches: The number of batches used for dividing the input.
        inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

    Returns:
        A dictionary with key `pool_3` and `logits`, representing the pool_3 and
        logits of the inception network respectively.
    """
    num_tpus = jax.local_device_count()
    input_tensors = tf.split(input_tensor, num_tpus, axis=0)
    pool3 = []
    logits = [] if not inceptionv3 else None
    device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
    for i, tensor in enumerate(input_tensors):
        with tf.device(device_format.format(i)):
            tensor_on_device = tf.identity(tensor)
            res = run_inception_jit(
                tensor_on_device, inception_model, num_batches=num_batches,
                inceptionv3=inceptionv3)

            if not inceptionv3:
                pool3.append(res['pool_3'])
                logits.append(res['logits'])  # pytype: disable=attribute-error
            else:
                pool3.append(res)

    with tf.device('/CPU'):
        return {
            'pool_3': tf.concat(pool3, axis=0),
            'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
        }


def save_dataset_statistics(config, output_dir='assets/stats'):
    """
    Compute and save dataset statistics for FID computation.
    
    Args:
        config: Configuration object
        output_dir: Directory to save statistics
        
    Returns:
        str: Path to saved statistics file, or None if failed
    """
    if not FID_AVAILABLE:
        logging.warning("pytorch-fid not available. Cannot compute dataset statistics.")
        return None
        
    import tensorflow as tf
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    dataset = config.data.dataset
    image_size = config.data.image_size
    
    # Create output directory
    tf.io.gfile.makedirs(output_dir)
    
    # Determine output filename
    if dataset == 'CIFAR10':
        filename = os.path.join(output_dir, 'cifar10_stats.npz')
    elif dataset == 'CELEBA':
        filename = os.path.join(output_dir, 'celeba_stats.npz')
    elif dataset == 'LSUN':
        filename = os.path.join(output_dir, f'lsun_{config.data.category}_{image_size}_stats.npz')
    elif dataset == 'AFHQ':
        filename = os.path.join(output_dir, f'afhq_{image_size}_stats.npz')
    elif dataset == 'FFHQ':
        filename = os.path.join(output_dir, f'ffhq_{image_size}_stats.npz')
    else:
        logging.error(f'Dataset {dataset} not supported for stats computation.')
        return None
    
    try:
        # Load dataset
        import datasets
        train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
        
        # Initialize Inception model
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(config.device)
        model.eval()
        
        # Preprocess transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Compute features
        all_features = []
        batch_size = 32
        
        logging.info(f"Computing dataset statistics for {dataset} {image_size}px...")
        logging.info("This may take 10-30 minutes depending on dataset size...")
        
        # Process dataset in batches
        data_iter = iter(train_ds)
        num_batches = len(train_ds)
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing stats"):
                try:
                    batch = next(data_iter)
                    images = batch['image']._numpy()
                    
                    # Convert from [0,1] to [0,255] if needed
                    if images.max() <= 1.0:
                        images = (images * 255).astype(np.uint8)
                    else:
                        images = images.astype(np.uint8)
                    
                    batch_features = []
                    for img in images:
                        # Convert to PIL and apply transform
                        if img.shape[-1] == 3:  # HWC format
                            img_tensor = transform(img)
                        else:  # CHW format
                            img = img.transpose(1, 2, 0)  # Convert to HWC
                            img_tensor = transform(img)
                        
                        batch_features.append(img_tensor)
                    
                    if batch_features:
                        batch_tensor = torch.stack(batch_features).to(config.device)
                        
                        # Get Inception features
                        features = model(batch_tensor)[0]
                        
                        # Flatten spatial dimensions
                        if features.size(2) != 1 or features.size(3) != 1:
                            features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))
                        
                        features = features.cpu().numpy().reshape(features.size(0), -1)
                        all_features.append(features)
                        
                except StopIteration:
                    break
                except Exception as e:
                    logging.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        if not all_features:
            logging.error("No features computed. Check dataset loading.")
            return None
            
        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        logging.info(f"Computed features shape: {all_features.shape}")
        
        # Compute statistics
        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)
        
        # Save statistics
        with tf.io.gfile.GFile(filename, 'wb') as fout:
            np.savez_compressed(fout, mu=mu, sigma=sigma)
        
        logging.info(f"✅ Dataset statistics saved to: {filename}")
        logging.info(f"   μ shape: {mu.shape}")
        logging.info(f"   Σ shape: {sigma.shape}")
        
        return filename
        
    except Exception as e:
        logging.error(f"Error computing dataset statistics: {e}")
        return None


def download_official_stats(dataset, image_size):
    """
    Download official pre-computed statistics if available.
    
    Args:
        dataset: Dataset name
        image_size: Image resolution
        
    Returns:
        str: Path to downloaded file, or None if not available
    """
    # This is a placeholder - you would implement actual download logic here
    # For now, just return None to indicate no official stats available
    logging.info(f"Official stats download not implemented for {dataset} {image_size}px")
    return None