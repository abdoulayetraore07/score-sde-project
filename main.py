# Based on score_sde_pytorch
# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");


# GUIDE INSTALLATION SUR LAMBDA CLOUD

"""

# Zipper (comme vous faisiez déjà)
cd Downloads
tar -czf projet_score_light.tar.gz \
  --exclude='experiments/*/checkpoints-meta/*' \
  --exclude='experiments/*/checkpoints/*' \
  --exclude='*/__pycache__/*' \
  --exclude='*.pyc' \
  --exclude='score_sde_pytorch/data/*' \
  score_sde_pytorch/

# Transférer
scp projet_score_light.tar.gz ubuntu@[IP]:~

# ÉTAPE 1: Décompression du projet
#echo ""
#echo "📁 ÉTAPE 1: Décompression du projet..."
tar -xzf projet_score_light.tar.gz
cd score_sde_pytorch
chmod +x setup_arm64.sh
./setup_arm64.sh

#echo "✅ Projet décompressé"

# Rendre executable 
chmod +x setup_arm64.sh
./setup_arm64.sh

# Le code sauve maintenant les images dans generated_images/
# Pour calculer FID :

pip install pytorch-fid Pillow

python -m pytorch_fid path/to/real_dataset path/to/generated_images --device cuda


# Recuperer les checkpoints
tar -czf ~/checkpoint_afhq_backup.tar.gz -C ~/score_sde_pytorch/experiments/afhq_512_ncsnpp_continuous/checkpoints-meta checkpoint.pth
scp ubuntu@192.222.51.0:~/checkpoint_afhq_backup.tar.gz /Users/abdoulayetraore/Downloads/score_sde_pytorch/experiments/afhq_512_ncsnpp_continuous/
cd /Users/abdoulayetraore/Downloads/score_sde_pytorch/experiments/afhq_512_ncsnpp_continuous/
tar -xzf checkpoint_afhq_backup.tar.gz


"""


# IMPORT DEBUG SYSTEM
try:
    from debug import DebugConfig, get_debug_logger
    DEBUG_MODE = DebugConfig.setup_global_logging()
    logger = get_debug_logger(__name__)
except ImportError:
    import logging
    DEBUG_MODE = True
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("debug.py non trouvé - mode debug basique")

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torch
import tensorflow as tf
import sys
from datetime import datetime

# Supprimer les warnings dès le démarrage
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Réduire les logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_logging(workdir=None):
    """Configure le logging selon debug.py."""
    if DEBUG_MODE:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        if workdir:
            file_handler = logging.FileHandler(os.path.join(workdir, 'training_debug.log'))
            file_handler.setFormatter(logging.Formatter(log_format))
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            root_logger = logging.getLogger()
            root_logger.handlers = []
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.INFO)
            logger.info(f"📝 Logs debug: {workdir}/training_debug.log")


def log_eval_configuration(config):
    """Log la configuration d'évaluation multi-résolution."""
    if hasattr(config.eval, 'resolutions') and hasattr(config.eval, 'use_official_stats'):
        logging.info("📊 CONFIGURATION ÉVALUATION:")
        logging.info(f"  Résolutions: {config.eval.resolutions}")
        logging.info(f"  Stats officielles: {config.eval.use_official_stats}")
        
        # Prédictions de temps
        num_resolutions = len(config.eval.resolutions)
        if num_resolutions > 1:
            logging.info(f"  ⏱️  Temps estimé: ~{num_resolutions * 5} min (FID multi-résolution)")
        else:
            logging.info(f"  ⏱️  Temps estimé: ~5 min (FID single résolution)")


def print_banner(mode):
    """Affiche une bannière au démarrage."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                   SCORE-BASED SDE MODELS                  ║
    ║                    PyTorch Implementation                 ║
    ║                  🆕 Multi-Resolution FID                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    logging.info(banner)
    logging.info(f"🚀 Mode: {mode.upper()}")
    logging.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Afficher les nouvelles capacités si en mode eval
    if mode == "eval":
        logging.info("🆕 Nouvelles fonctionnalités:")
        logging.info("  📊 FID multi-résolution (256px + 512px)")
        logging.info("  🌐 Auto-download stats officielles")
        logging.info("  ⚙️  Configuration flexible par ligne de commande")
    
    logging.info("")



config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("eval_resolutions", None,
                    "Comma-separated list of resolutions for multi-resolution FID evaluation (e.g., '256,512'). "
                    "Default: use config image size only.")
flags.DEFINE_boolean("use_official_stats", True,
                     "Whether to use official pre-computed statistics when available. "
                     "If False, compute statistics locally from dataset.")
flags.DEFINE_boolean("use_clean_fid", True,
                     "Use Clean-FID for evaluation (recommended). "
                     "If False, fallback to legacy FID computation.")
flags.DEFINE_enum("fid_mode", "clean", ["clean", "legacy_pytorch"],
                  "Clean-FID mode: 'clean' (recommended) or 'legacy_pytorch' (for comparison)")
flags.DEFINE_string("dataset_name", "afhq_cat,afhq_dog,afhq_wild", 
                    "Dataset name(s) for FID (comma-separated): 'afhq_cat,afhq_dog,afhq_wild'")
flags.mark_flags_as_required(["workdir", "config", "mode"])


FLAGS = flags.FLAGS




def main(argv):
  # Importer datetime ici si pas déjà fait globalement
  from datetime import datetime
  
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Configurer le logging proprement
    setup_logging(FLAGS.workdir)
    # Afficher la bannière
    print_banner(FLAGS.mode)

    # Log checkpoint AFHQ si nécessaire
    if DEBUG_MODE and 'afhq' in str(FLAGS.config).lower():
        checkpoint_path = os.path.join(FLAGS.workdir, "checkpoints-meta", "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            size = os.path.getsize(checkpoint_path) / (1024*1024)
            logger.info(f"✅ Checkpoint AFHQ trouvé: {size:.1f}MB")
        else:
            logger.warning(f"⚠️  Checkpoint AFHQ manquant: {checkpoint_path}")

    # Log les paths importants
    logging.info(f"📁 Working directory: {FLAGS.workdir}")
    logging.info(f"📄 Config file: {FLAGS.config}")
    logging.info("")
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)

  elif FLAGS.mode == "eval":
    # Configurer le logging pour eval
    setup_logging()
    print_banner(FLAGS.mode)
    
    logging.info(f"📁 Working directory: {FLAGS.workdir}")
    logging.info(f"📂 Eval folder: {FLAGS.eval_folder}")
    config = FLAGS.config

    # Parser les résolutions d'évaluation
    if FLAGS.eval_resolutions:
        from run_lib import parse_eval_resolutions
        eval_resolutions = parse_eval_resolutions(FLAGS.eval_resolutions, config.data.image_size)
        logging.info(f"🎯 Résolutions d'évaluation: {eval_resolutions}")
    else:
        eval_resolutions = [config.data.image_size]
        logging.info(f"🎯 Résolution d'évaluation: {eval_resolutions[0]}px (défaut)")

    # Configuration stats officielles
    use_official_stats = FLAGS.use_official_stats
    logging.info(f"📊 Stats officielles: {'✅ Activées' if use_official_stats else '❌ Désactivées (calcul local)'}")

    if FLAGS.dataset_name:
        dataset_names = [name.strip() for name in FLAGS.dataset_name.split(',')]
        config.eval.dataset_name = dataset_names
        logging.info(f"🏷️  Datasets FID: {dataset_names}")
    else:
        config.eval.dataset_name = ['afhq_cat']  # Défaut
        logging.info(f"🏷️  Dataset FID: ['afhq_cat'] (défaut)")
    
    # Ajouter les paramètres à la config
    config.eval.resolutions = eval_resolutions
    config.eval.use_official_stats = use_official_stats
    config.eval.use_clean_fid = FLAGS.use_clean_fid
    config.eval.fid_mode = FLAGS.fid_mode

    logging.info("")
    
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)

  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
