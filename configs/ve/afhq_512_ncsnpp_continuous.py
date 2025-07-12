"""Training NCSN++ on AFHQ with VE SDE - Auto-Adaptive Config."""

import torch
import platform
from configs.default_lsun_configs import get_default_configs


def detect_gpu_config():
    """Détecte automatiquement la configuration GPU et système."""
    config_info = {
        'gpu_count': 0,
        'gpu_memory_gb': 0,
        'gpu_names': [],
        'is_arm64': False,
        'total_memory_gb': 0,
        'compute_caps': []
    }
    
    # Détection architecture
    config_info['is_arm64'] = platform.machine().lower() in ['aarch64', 'arm64']
    
    if torch.cuda.is_available():
        config_info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(config_info['gpu_count']):
            gpu_name = torch.cuda.get_device_name(i)
            config_info['gpu_names'].append(gpu_name)
            
            props = torch.cuda.get_device_properties(i)
            gpu_memory = props.total_memory // (1024**3)  # GB
            config_info['gpu_memory_gb'] = max(config_info['gpu_memory_gb'], gpu_memory)
            config_info['total_memory_gb'] += gpu_memory
            config_info['compute_caps'].append(f"{props.major}.{props.minor}")
    
    return config_info


def get_optimal_batch_sizes(gpu_config):
    """Calcule les batch sizes optimaux selon la configuration."""
    gpu_count = gpu_config['gpu_count']
    gpu_memory = gpu_config['gpu_memory_gb']
    is_arm64 = gpu_config['is_arm64']
    
    # Batch size par GPU selon la mémoire (AFHQ 512x512)
    if gpu_memory >= 90:  # GH200, H100 80GB+
        per_gpu_batch = 32
    elif gpu_memory >= 70:  # A100 80GB
        per_gpu_batch = 8
    elif gpu_memory >= 40:  # A100 40GB, V100 32GB
        per_gpu_batch = 8
    elif gpu_memory >= 20:  # RTX series
        per_gpu_batch = 4
    else:  # Petit GPU
        per_gpu_batch = 2
    
    # Ajustement pour ARM64 (plus conservateur)
    if is_arm64:
        per_gpu_batch = min(per_gpu_batch, 32)
    
    # Calcul batch size total
    total_batch_size = per_gpu_batch * max(1, gpu_count)
    eval_batch_size = min(total_batch_size, 64)  # Cap pour l'évaluation
    
    return {
        'train_batch_size': total_batch_size,
        'eval_batch_size': eval_batch_size,
        'per_gpu_batch': per_gpu_batch
    }


def get_training_params(gpu_config):
    """Ajuste les paramètres d'entraînement selon la configuration."""
    snapshot_freq = 20000
    eval_samples = 2  
    return {
        'snapshot_freq': snapshot_freq,
        'eval_samples': eval_samples
    }


def get_config():
    """Configuration auto-adaptative pour AFHQ."""
    # Détection automatique
    gpu_config = detect_gpu_config()
    batch_config = get_optimal_batch_sizes(gpu_config)
    training_config = get_training_params(gpu_config)
    
    # Affichage de la configuration détectée
    print("🔍 Configuration GPU détectée:")
    print(f"   → GPUs: {gpu_config['gpu_count']} x {gpu_config['gpu_memory_gb']}GB")
    print(f"   → Noms: {', '.join(gpu_config['gpu_names'])}")
    print(f"   → Architecture: {'ARM64' if gpu_config['is_arm64'] else 'x86_64'}")
    print(f"   → Compute: {', '.join(set(gpu_config['compute_caps']))}")
    print(f"   → Mémoire totale: {gpu_config['total_memory_gb']}GB")
    print("")
    print("⚙️  Configuration optimisée:")
    print(f"   → Batch size training: {batch_config['train_batch_size']} ({batch_config['per_gpu_batch']}/GPU)")
    print(f"   → Batch size eval: {batch_config['eval_batch_size']}")
    print(f"   → Snapshot freq: {training_config['snapshot_freq']}")
    print(f"   → Eval samples: {training_config['eval_samples']}")
    print("")
    
    # Configuration de base
    config = get_default_configs()
    
    # === TRAINING ===
    training = config.training
    training.sde = 'vesde'
    training.continuous = True
    training.n_iters = 750001
    training.snapshot_freq = training_config['snapshot_freq']
    training.batch_size = batch_config['train_batch_size']
    training.reduce_mean = True
    
    # === SAMPLING ===
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'
    sampling.snr = 0.16
    
    # === DATA ===
    data = config.data
    data.dataset = 'AFHQ'
    data.image_size = 512
    data.afhq_path = "data/afhq"
    
    # === EVALUATION ===
    evaluate = config.eval
    evaluate.begin_ckpt = 15
    evaluate.end_ckpt = 23
    evaluate.batch_size = batch_config['eval_batch_size']
    evaluate.num_samples = training_config['eval_samples']
    evaluate.enable_sampling = True  
    
    evaluate.use_clean_fid = True  # Utiliser Clean-FID par défaut
    evaluate.fid_mode = 'clean'  # ou 'legacy_pytorch' pour comparaison
    evaluate.dataset_name = ['afhq_cat', 'afhq_dog', 'afhq_wild'] # for metrics 
    evaluate.multi_resolution = True  # Évaluer plusieurs résolutions
    evaluate.resolutions = [256, 512]  # Résolutions à évaluer

    evaluate.enable_class_conditional = True  # Activer génération par classe
    evaluate.guidance_scale = 1.0  # Force du guidance classifier (1.0 = normal)

    # === MODEL ===
    model = config.model
    model.name = 'ncsnpp'
    model.sigma_max = 784
    model.scale_by_sigma = True
    model.num_scales = 2000
    model.ema_rate = 0.9999
    model.dropout = 0.
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 16
    model.ch_mult = (1, 2, 4, 8, 16, 32, 32)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    
    # === NOUVELLES CONFIGURATIONS POUR LES STRATÉGIES DE GUIDANCE ===
    # Configuration par défaut des stratégies
    config.guidance_strategy = "standard"  # standard, adaptive_scale, truncation, amplification
    config.guidance_scale = 1.0  # Échelle de guidance de base
    config.adaptive_sigma_limit = 50.0  # Limite sigma pour stratégie adaptive_scale
    
    # === DEVICE AUTO ===
    # Utilise tous les GPUs disponibles ou CPU
    if torch.cuda.is_available():
        config.device = torch.device('cuda')
        print(f"   → Device: CUDA ({gpu_config['gpu_count']} GPUs)")
    else:
        config.device = torch.device('cpu')
        print(f"   → Device: CPU")
    
    print("✅ Configuration auto-adaptative prête !")
    return config


# Fonction utilitaire pour forcer une configuration spécifique
def get_config_override(force_batch_size=None, force_eval_samples=None):
    """Version avec override manuel des paramètres."""
    config = get_config()
    
    if force_batch_size is not None:
        config.training.batch_size = force_batch_size
        config.eval.batch_size = min(force_batch_size, 64)
        print(f"🔧 Override: batch_size forcé à {force_batch_size}")
    
    if force_eval_samples is not None:
        config.eval.num_samples = force_eval_samples
        print(f"🔧 Override: eval_samples forcé à {force_eval_samples}")
    
    return config