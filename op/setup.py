from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import warnings

# Import du debug system
try:
    import sys
    sys.path.append('..')
    from debug import DebugConfig
    DEBUG_MODE = DebugConfig.ENABLE_COMPILATION_LOGS
except ImportError:
    DEBUG_MODE = False

# Configuration warnings selon debug
if not DEBUG_MODE:
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings("ignore")

class QuietBuildExtension(BuildExtension):
    """Extension de build qui contrôle les logs selon debug.py."""
    def build_extensions(self):
        for ext in self.extensions:
            if DEBUG_MODE:
                ext.extra_compile_args = ['-std=c++17', '-O3']
                if hasattr(ext, 'nvcc_extra_compile_args'):
                    ext.nvcc_extra_compile_args = ['--expt-relaxed-constexpr']
            else:
                ext.extra_compile_args = ['-std=c++17', '-O3', '-w']
                if hasattr(ext, 'nvcc_extra_compile_args'):
                    ext.nvcc_extra_compile_args = ['-w', '--expt-relaxed-constexpr']
        
        if not DEBUG_MODE:
            import sys, io
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = io.StringIO()
            try:
                super().build_extensions()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                print("✅ Extensions CUDA compilées")
        else:
            print("🔨 Compilation CUDA avec logs détaillés...")
            super().build_extensions()
            print("✅ Extensions CUDA compilées avec debug")

setup(
    name='op_extensions',
    ext_modules=[
        CUDAExtension(
            name='upfirdn2d_op',
            sources=['upfirdn2d.cpp', 'upfirdn2d_kernel.cu'],
            extra_compile_args={'cxx': ['-std=c++17', '-O3'], 'nvcc': ['--expt-relaxed-constexpr']}
        ),
        CUDAExtension(
            name='fused_bias_act_op',
            sources=['fused_bias_act.cpp', 'fused_bias_act_kernel.cu'],
            extra_compile_args={'cxx': ['-std=c++17', '-O3'], 'nvcc': ['--expt-relaxed-constexpr']}
        ),
    ],
    cmdclass={'build_ext': QuietBuildExtension},
    zip_safe=False,
    verbose=DEBUG_MODE
)