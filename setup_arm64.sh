#!/bin/bash

# 🚀 Setup FINAL Score SDE PyTorch pour GH200 ARM64
# Script testé et validé - Version finale fonctionnelle
# Basé sur l'installation réelle réussie

set -e

echo "🎯 Score SDE PyTorch - Setup FINAL pour GH200 ARM64"
echo "====================================================="
echo "GPU: NVIDIA GH200 480GB (94GB VRAM utilisable)"
echo "CUDA: 12.8"
echo "Architecture: ARM64"
echo "Ubuntu: 22.04"
echo ""

# Détection automatique de la configuration GPU
echo "🔍 Détection de la configuration système..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✅ $GPU_COUNT GPU(s) détecté(s):"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | nl -w2 -s': ' | while read line; do
        echo "   → GPU $line"
    done
    
    COMPUTE_CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | sort -u | tr '\n' ';' | sed 's/;$//')
    echo "✅ Compute Capabilities: $COMPUTE_CAPS"
else
    echo "⚠️  nvidia-smi non disponible"
    GPU_COUNT=1
    COMPUTE_CAPS="9.0"
fi

# ===========================================
# ÉTAPE 1: DÉPENDANCES SYSTÈME
# ===========================================

echo ""
echo "🔧 ÉTAPE 1: Installation des dépendances système..."

sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev pybind11-dev wget unzip

echo "✅ Dépendances système installées"

# ===========================================
# ÉTAPE 2: DÉPENDANCES PYTHON ESSENTIELLES
# ===========================================

echo ""
echo "📦 ÉTAPE 2: Installation dépendances Python..."

# Installation des packages manquants identifiés lors des tests
pip install tqdm ml-collections absl-py gdown pytorch-fid Pillow scipy pybind11
pip install tensorflow-datasets tensorflow-hub
pip install clean-fid
pip install seaborn
pip install questionary


# CRITIQUE: Downgrade NumPy pour compatibilité extensions CUDA
echo "🔧 Downgrade NumPy pour compatibilité..."
pip install "numpy<2"

echo "✅ Dépendances Python installées"

# ===========================================
# ÉTAPE 3: COMPILATION EXTENSIONS CUDA
# ===========================================

echo ""
echo "🔨 ÉTAPE 3: Compilation extensions CUDA pour GH200..."

if [ -d "op" ]; then
    cd op
    
    # Configuration variables CUDA pour GH200
    export CUDA_HOME="/usr/local/cuda"
    export TORCH_CUDA_ARCH_LIST="9.0"  # GH200 Compute Capability
    export FORCE_CUDA="1"
    
    echo "✅ Variables CUDA configurées:"
    echo "   CUDA_HOME: $CUDA_HOME"
    echo "   TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    
    # Nettoyage complet
    rm -rf build/ *.so __pycache__/ 2>/dev/null || true
    
    # Compilation
    echo "  → Compilation pour Compute Capability 9.0 (GH200)..."
    python setup.py build_ext --inplace
    
    cd ..
    echo "✅ Extensions CUDA compilées"
else
    echo "❌ Dossier 'op' non trouvé"
    exit 1
fi

# ===========================================
# ÉTAPE 4: TÉLÉCHARGEMENT CHECKPOINTS
# ===========================================

echo ""
echo "📦 ÉTAPE 4: Téléchargement et installation checkpoints..."

# 4.1 Structure dossiers
mkdir -p experiments/{church,celebahq_256,ffhq_1024,afhq_512}_ncsnpp_continuous/{checkpoints,checkpoints-meta}

## 4.2 Church checkpoint
#echo "  → Church checkpoint..."
#gdown --folder "https://drive.google.com/drive/folders/1zVChA0HrnJU66Jkt4P6KOnlREhBMc4Yh" --quiet 2>/dev/null || echo "    ⚠️ Church échoué"
#if [ -d "church_ncsnpp_continuous" ]; then
#    find church_ncsnpp_continuous -name "*.pth" -exec cp {} experiments/church_ncsnpp_continuous/checkpoints-meta/checkpoint.pth \; 2>/dev/null || true
#    find church_ncsnpp_continuous -name "*.pth" -exec cp {} experiments/church_ncsnpp_continuous/checkpoints/ \; 2>/dev/null || true
#    rm -rf church_ncsnpp_continuous/
#    echo "    ✅ Church installé"
#fi


# 4.3 AFHQ checkpoints (TOUS les checkpoints)
echo "  → AFHQ checkpoints..."
gdown --folder "https://drive.google.com/drive/folders/1xSoDpHiZSoBToPlJ2rdu2t2fXzzIQLQe" --quiet 2>/dev/null || echo "    ⚠️ AFHQ échoué"

if [ -d "afhq_512_ncsnpp_continuous" ]; then
    mkdir -p experiments/afhq_512_ncsnpp_continuous/checkpoints/
    
    # Installer TOUS les checkpoints avec leurs noms d'origine
    for checkpoint_file in afhq_512_ncsnpp_continuous/checkpoint*.pth; do
        if [ -f "$checkpoint_file" ]; then
            filename=$(basename "$checkpoint_file")
            cp "$checkpoint_file" "experiments/afhq_512_ncsnpp_continuous/checkpoints/$filename"
            echo "    ✅ Installé: $filename"
            
            # Extraire le numéro d'itération
            if [[ "$filename" =~ checkpoint_([0-9]+)\.pth ]]; then
                iter_num=${BASH_REMATCH[1]}
                iterations=$((iter_num * 20000))
                echo "       → Itérations: ${iterations:0}"
            elif [ "$filename" = "checkpoint.pth" ]; then
                echo "       → Itérations: 520000 (défaut)"
            fi
        fi
    done
    
    # Copier le checkpoint par défaut vers checkpoints-meta
    if [ -f "afhq_512_ncsnpp_continuous/checkpoint.pth" ]; then
        cp "afhq_512_ncsnpp_continuous/checkpoint.pth" experiments/afhq_512_ncsnpp_continuous/checkpoints-meta/checkpoint.pth
        echo "    ✅ Checkpoint par défaut (520k iter) installé"
    fi
    
    rm -rf afhq_512_ncsnpp_continuous/
fi

# 4.4 FFHQ_1024 et Celeab_256 checkpoints
#echo "  → FFHQ_1024 et Celeba_256 checkpoints..."
#for model in "celebahq_256_ncsnpp_continuous:19VJ7UZTE-ytGX6z5rl-tumW9c0Ps3itk" "ffhq_1024_ncsnpp_continuous:1ZqLNr_kH0o9DxvwSlrQPMmkrhEnXhBm2"; do
#    name=$(echo $model | cut -d: -f1)
#    id=$(echo $model | cut -d: -f2)
    
#    gdown --folder "https://drive.google.com/drive/folders/$id" --quiet 2>/dev/null || continue
    
#    if [ -d "$name" ]; then
#        find "$name" -name "*.pth" -exec cp {} "experiments/$name/checkpoints-meta/checkpoint.pth" \; 2>/dev/null || true
#        find "$name" -name "*.pth" -exec cp {} "experiments/$name/checkpoints/" \; 2>/dev/null || true
#        rm -rf "$name/"
#        echo "    ✅ $name installé"
#    fi
#done


# 4.5 AFHQ classifier checkpoint
echo "  → AFHQ classifier checkpoint..."
gdown --folder "https://drive.google.com/drive/u/0/folders/1p7l_4MHG7gAzdnc8oq2haORZCtHd2QN1" --quiet 2>/dev/null || echo "    ⚠️ AFHQ classifier échoué"
if [ -d "afhq_classifier" ]; then

    if [ -f "afhq_classifier/afhq_classifier.pth" ]; then
        cp afhq_classifier/afhq_classifier.pth experiments/afhq_classifier/afhq_classifier.pth
        echo "    ✅ AFHQ classifier 1 installé"
    fi
    if [ -f "afhq_classifier/afhq_classifier_2.pth" ]; then
        cp afhq_classifier/afhq_classifier_2.pth experiments/afhq_classifier/afhq_classifier_2.pth
        echo "    ✅ AFHQ classifier 2 installé"
    fi
    rm -rf afhq_classifier/
fi


# ===========================================
# ÉTAPE 5: DATASET AFHQ
# ===========================================

echo ""
echo "🐕 ÉTAPE 5: Installation dataset AFHQ..."

mkdir -p data
cd data/

if [ ! -d "afhq" ]; then
    echo "  → Téléchargement AFHQ (1.5GB)..."
    wget https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip --progress=bar --timeout=120 2>/dev/null || {
        echo "    ⚠️ Téléchargement AFHQ échoué"
        echo "    💡 Téléchargez manuellement si nécessaire"
    }
    
    if [ -f "afhq.zip" ]; then
        echo "  → Extraction..."
        unzip -q afhq.zip && rm afhq.zip
        echo "    ✅ AFHQ dataset extrait"
    fi
else
    echo "  ✅ Dataset AFHQ déjà présent"
fi

cd ..





# ===========================================
# ÉTAPE 6: TESTS DE VALIDATION
# ===========================================

echo ""
echo "🧪 ÉTAPE 6: Tests de validation..."

echo "  → Test environnement GPU..."
python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'✅ VRAM: {torch.cuda.get_device_properties(i).total_memory // 1024**3}GB')
" 2>/dev/null || echo "⚠️  Test Python échoué"

echo "  → Test imports projet..."
python3 -c "
try:
    from op import upfirdn2d
    from op.fused_act import fused_leaky_relu, FusedLeakyReLU
    print('✅ Extensions CUDA OK')
except Exception as e:
    print(f'⚠️  Extensions CUDA: {e}')

try:
    from models import ncsnpp, ddpm, ncsnv2
    print('✅ Models import OK')
except Exception as e:
    print(f'❌ Models error: {e}')

try:
    from configs.ve.afhq_512_ncsnpp_continuous import get_config
    config = get_config()
    print(f'✅ Config auto-détection: {config.training.batch_size} batch size')
except Exception as e:
    print(f'❌ Config error: {e}')
" 2>/dev/null || echo "⚠️  Test imports échoué"

# ===========================================
# VÉRIFICATION FINALE
# ===========================================

echo ""
echo "🔍 VÉRIFICATION FINALE..."

# Checkpoints
echo "📦 Checkpoints installés:"
for checkpoint in "church" "afhq_512" "celebahq_256" "ffhq_1024"; do
    if [ -f "experiments/${checkpoint}_ncsnpp_continuous/checkpoints-meta/checkpoint.pth" ]; then
        CHECKPOINT_SIZE=$(du -h "experiments/${checkpoint}_ncsnpp_continuous/checkpoints-meta/checkpoint.pth" | cut -f1)
        echo "  ✅ $checkpoint: $CHECKPOINT_SIZE"
    else
        echo "  ⚠️ $checkpoint: manquant"
    fi
done

# Dataset
if [ -d "data/afhq/train" ]; then
    TRAIN_COUNT=$(find data/afhq/train -name "*.jpg" | wc -l)
    VAL_COUNT=$(find data/afhq/val -name "*.jpg" | wc -l)
    echo "🐕 Dataset AFHQ: $TRAIN_COUNT train + $VAL_COUNT val images"
else
    echo "⚠️ Dataset AFHQ manquant"
fi

echo ""
echo "🎉 INSTALLATION TERMINÉE - GH200 OPTIMISÉE !"
echo "============================================="
echo ""
echo "🚀 Configuration finale:"
echo "   GPU: NVIDIA GH200 480GB (94GB VRAM utilisable)"
echo "   CUDA: 12.8 + Compute 9.0"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Error')"
echo "   Extensions: Compilées avec succès"
echo "   NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Error') (< 2.0 pour compatibilité)"
echo ""
echo "📋 Commandes de test disponibles:"
echo ""
echo "🎨 SAMPLING :"
echo "   python pretrained_sampling.py church 4"
echo "   python pretrained_sampling.py celebahq_256 4"
echo "   python pretrained_sampling.py ffhq_1024 4"
echo "   python pretrained_sampling.py afhq_512 4"
echo ""
echo "🔥 TRAINING AFHQ (utilise 94GB VRAM) :"
echo "   python main.py --config configs/ve/afhq_512_ncsnpp_continuous.py --workdir experiments/afhq_512_ncsnpp_continuous --mode train"
echo ""
echo "📊 MONITORING GPU pendant training:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "💪 Avec 94GB de VRAM, vous pouvez entraîner des modèles énormes !"
echo "✨ Setup terminé - Bon training ! 🚀"