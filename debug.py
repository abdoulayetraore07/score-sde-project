#!/usr/bin/env python3
"""
Configuration centralisée pour le debugging - Score SDE PyTorch
Permet d'activer/désactiver facilement tous les logs pour le debug
"""

import os
import warnings
import logging
import sys

class DebugConfig:
    """Configuration centralisée pour le debugging."""
    
    # 🎛️ CONTROLES PRINCIPAUX - Changez ces valeurs pour debug
    ENABLE_DEBUG = True  # ✅ ACTIVER pour debug, ❌ False pour production
    ENABLE_VERBOSE_TRAINING = True  # Logs détaillés pendant training
    ENABLE_TENSORFLOW_LOGS = True   # Logs TensorFlow
    ENABLE_PYTORCH_LOGS = True      # Logs PyTorch
    ENABLE_COMPILATION_LOGS = True  # Logs compilation CUDA
    ENABLE_PROGRESS_BARS = True     # Progress bars détaillées
    ENABLE_MODEL_SUMMARY = True     # Résumé du modèle au démarrage
    
    # 📊 NIVEAUX DE LOG
    LOG_LEVEL = logging.INFO if ENABLE_DEBUG else logging.WARNING
    
    @classmethod
    def setup_global_logging(cls):
        """Configure les logs globaux selon la configuration."""
             
        if cls.ENABLE_DEBUG:
            print("🐛 DEBUG MODE ACTIVÉ - Logs détaillés")
            print(f"   → Training verbose: {cls.ENABLE_VERBOSE_TRAINING}")
            print(f"   → TensorFlow logs: {cls.ENABLE_TENSORFLOW_LOGS}")
            print(f"   → PyTorch logs: {cls.ENABLE_PYTORCH_LOGS}")
            print(f"   → CUDA compilation: {cls.ENABLE_COMPILATION_LOGS}")
            print("")
        
        # Configuration des warnings
        if not cls.ENABLE_DEBUG:
            warnings.filterwarnings("ignore")
            os.environ['PYTHONWARNINGS'] = 'ignore'
        else:
            warnings.filterwarnings("default")
        warnings.filterwarnings("ignore", category=ResourceWarning)
        # Configuration TensorFlow
        if not cls.ENABLE_TENSORFLOW_LOGS:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        
        # Configuration PyTorch
        if not cls.ENABLE_PYTORCH_LOGS:
            logging.getLogger('torch').setLevel(logging.ERROR)
            logging.getLogger('torchvision').setLevel(logging.ERROR)
        
        # Configuration compilation CUDA
        if not cls.ENABLE_COMPILATION_LOGS:
            logging.getLogger('torch.utils.cpp_extension').setLevel(logging.ERROR)
            logging.getLogger('ninja').setLevel(logging.ERROR)
        
        # Configuration logging principal
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if cls.ENABLE_DEBUG else '%(levelname)s - %(message)s'
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        return cls.ENABLE_DEBUG

    @classmethod
    def get_logger(cls, name):
        """Retourne un logger configuré."""
        logger = logging.getLogger(name)
        logger.setLevel(cls.LOG_LEVEL)
        return logger
    
    @classmethod
    def should_show_progress(cls):
        """Indique si on doit montrer les progress bars détaillées."""
        return cls.ENABLE_DEBUG and cls.ENABLE_PROGRESS_BARS
    
    @classmethod
    def should_show_model_summary(cls):
        """Indique si on doit montrer le résumé du modèle."""
        return cls.ENABLE_DEBUG and cls.ENABLE_MODEL_SUMMARY
    
    @classmethod
    def should_verbose_training(cls):
        """Indique si on doit avoir des logs détaillés pendant training."""
        return cls.ENABLE_DEBUG and cls.ENABLE_VERBOSE_TRAINING

# Configuration au niveau module
DEBUG_MODE = DebugConfig.ENABLE_DEBUG

def setup_debug():
    """Fonction utilitaire pour setup rapide."""
    return DebugConfig.setup_global_logging()

def get_debug_logger(name):
    """Fonction utilitaire pour obtenir un logger."""
    return DebugConfig.get_logger(name)

# Auto-setup si importé
if __name__ != "__main__":
    DebugConfig.setup_global_logging()

if __name__ == "__main__":
    print("🎛️ CONFIGURATION DEBUG - Score SDE PyTorch")
    print("=" * 50)
    print(f"Debug activé: {DebugConfig.ENABLE_DEBUG}")
    print(f"Training verbose: {DebugConfig.ENABLE_VERBOSE_TRAINING}")
    print(f"TensorFlow logs: {DebugConfig.ENABLE_TENSORFLOW_LOGS}")
    print(f"PyTorch logs: {DebugConfig.ENABLE_PYTORCH_LOGS}")
    print(f"Compilation logs: {DebugConfig.ENABLE_COMPILATION_LOGS}")
    print(f"Progress bars: {DebugConfig.ENABLE_PROGRESS_BARS}")
    print(f"Model summary: {DebugConfig.ENABLE_MODEL_SUMMARY}")
    print("")
    print("💡 Pour changer la config, éditez les variables en haut du fichier")
    print("💡 ENABLE_DEBUG = False pour désactiver tout rapidement")