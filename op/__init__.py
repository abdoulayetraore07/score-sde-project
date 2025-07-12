import warnings
import logging
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F

# Supprimer tous les warnings lors de l'import
warnings.filterwarnings("ignore")
logging.getLogger('torch.utils.cpp_extension').setLevel(logging.ERROR)

# Détecter Mac M1/M2/M3
def is_apple_silicon():
    return (platform.system() == 'Darwin' and 
            (platform.processor() == 'arm' or 'arm64' in platform.machine().lower()))

# Détecter si CUDA est disponible
def has_cuda():
    return torch.cuda.is_available()

# Implémentation native upfirdn2d sans extensions CUDA
def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """Implémentation native sans extensions CUDA"""
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)

# Sur Mac M1, utiliser les fallbacks PyTorch
if is_apple_silicon():
    print("🍎 Mac M1/M2 détecté - Utilisation des fallbacks PyTorch")
    
    # Fallback pour upfirdn2d
    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
        """Fallback upfirdn2d pour Mac M1 avec MPS"""
        # Utiliser l'implémentation native directement
        return upfirdn2d_native(
            input, kernel, up, up, down, down, 
            pad[0], pad[1], pad[0], pad[1]
        )
    
    # Fallback pour fused_leaky_relu
    def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
        """Fallback fused_leaky_relu pour Mac M1"""
        if bias is not None:
            # Adapter les dimensions du bias
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            input = input + bias.view(1, bias.shape[0], *rest_dim)
        
        return F.leaky_relu(input, negative_slope=negative_slope) * scale
    
    # Fallback pour FusedLeakyReLU
    class FusedLeakyReLU(nn.Module):
        def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(channel))
            self.negative_slope = negative_slope
            self.scale = scale

        def forward(self, input):
            return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
    
    print("✅ Fallbacks PyTorch chargés pour Mac M1")

else:
    # Sur autres plateformes, essayer de charger les extensions CUDA
    try:
        from .fused_act import FusedLeakyReLU, fused_leaky_relu
        from .upfirdn2d import upfirdn2d
        print("✅ Extensions CUDA chargées")
        
    except ImportError as e:
        print("⚠️  Extensions CUDA non compilées. Compilation en cours...")
        print("   Cela peut prendre quelques minutes la première fois.")
        
        # Essayer de compiler
        import subprocess
        import sys
        import os
        
        op_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Compiler silencieusement
            subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=op_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print("✅ Compilation terminée avec succès!")
            
            # Réessayer l'import
            from .fused_act import FusedLeakyReLU, fused_leaky_relu
            from .upfirdn2d import upfirdn2d
            
        except subprocess.CalledProcessError as e:
            print("❌ Erreur lors de la compilation des extensions CUDA")
            print("   Utilisation des fallbacks PyTorch")
            
            # Fallbacks identiques à Mac M1
            def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
                return upfirdn2d_native(
                    input, kernel, up, up, down, down, 
                    pad[0], pad[1], pad[0], pad[1]
                )
            
            def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
                if bias is not None:
                    rest_dim = [1] * (input.ndim - bias.ndim - 1)
                    input = input + bias.view(1, bias.shape[0], *rest_dim)
                return F.leaky_relu(input, negative_slope=negative_slope) * scale
            
            class FusedLeakyReLU(nn.Module):
                def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
                    super().__init__()
                    self.bias = nn.Parameter(torch.zeros(channel))
                    self.negative_slope = negative_slope
                    self.scale = scale

                def forward(self, input):
                    return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)