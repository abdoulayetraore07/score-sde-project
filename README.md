# From Noise to Images: Implementing Score-Based Generative Models using Stochastic Differential Equations

This repository contains the implementation and research results from a comprehensive study on score-based generative models using Stochastic Differential Equations (SDEs). The work was conducted as part of a research internship at the University of Torino, Department of Mathematics "Giuseppe Peano."

## Overview

Score-based generative models represent a significant advancement in generative modeling, enabling high-quality image generation through the estimation of data distribution gradients (score functions). This project explores the continuous-time formulation of diffusion processes using SDEs, building upon the foundational work of Song et al. (ICLR 2021).

### Key Features

- **Complete VE-SDE Implementation**: Full implementation of Variance Exploding Stochastic Differential Equations for image generation
- **Multi-Resolution Training**: Experiments across multiple datasets and resolutions (256×256 to 1024×1024)
- **Custom AFHQ-512 Model**: Trained from scratch with 520,000 iterations
- **Adaptive Classifier Guidance**: Three novel strategies for controllable generation
- **Comprehensive Downstream Tasks**: Image inpainting, colorization, and class-conditional generation

## Architecture

The implementation is based on the NCSN++ continuous architecture with VE-SDE formulation:

```
Forward SDE:  dx = √(d[σ²(t)]/dt) dw
Reverse SDE:  dx = [f(x,t) - g(t)²∇ₓlog pₜ(x)]dt + g(t)dw̄
```

Where the score function ∇ₓlog pₜ(x) is learned through denoising score matching.

## Datasets

The project evaluates performance across four high-quality datasets:

- **FFHQ-1024**: High-resolution human faces (1024×1024)
- **AFHQ-512**: Animal faces with three categories - cats, dogs, wild animals (512×512)
- **CelebA-HQ-256**: Celebrity faces (256×256)  
- **LSUN Church-256**: Architectural imagery (256×256)

## Sample Results

### Unconditional Generation

The model generates diverse, high-quality samples across all tested domains:

#### AFHQ-512 (Custom Trained Model)
Our custom-trained AFHQ-512 model demonstrates excellent inter-class diversity and intra-class coherence across animal categories.

#### FFHQ-1024 (Pre-trained)
High-resolution human face generation with realistic facial features, though occasional symmetry challenges are observed at very high resolutions.

#### Architectural Generation
LSUN Church samples showcase the model's capability for structured content generation beyond organic forms.

### Downstream Applications

#### Image Inpainting
Coherent completion of masked regions while maintaining consistency with surrounding context. The approach works across various masking strategies including geometric shapes and irregular patterns.

#### Colorization  
Transformation of grayscale images to realistic color versions using orthogonal transformation in YUV color space. The method preserves luminance information while generating plausible chrominance values.

#### Class-Conditional Generation
Controllable generation using classifier guidance with three adaptive strategies:

1. **Amplified Guidance Scale** (λ = 500): Most effective approach
2. **Adaptive Linear Scaling**: Marginal improvements in detail quality  
3. **Sigma Truncation**: Validates theoretical assumptions about classifier effectiveness zones

## Technical Contributions

### Adaptive Classifier Guidance
This work introduces three novel strategies to address the challenge of classifier ineffectiveness across different noise levels:

- **Strategy 1**: Amplified guidance scale to balance score function and classifier gradient contributions
- **Strategy 2**: Linearly decreasing guidance scale based on noise level analysis
- **Strategy 3**: Restricted classifier usage to effective noise ranges through sigma truncation

### Training Insights
- Successfully trained AFHQ-512 model from scratch with limited computational resources
- Demonstrated that 512×512 resolution provides optimal quality-coherence trade-offs
- Identified computational and scalability challenges for high-resolution generation

## Implementation Details

### Sampling Configuration
- **Predictor**: Reverse-diffusion sampler (Yang Song et al.)
- **Corrector**: Langevin dynamics with SNR = 0.16
- **Steps**: 2000 discretization steps for high-resolution datasets
- **Architecture**: NCSN++ continuous with EMA rate = 0.9999

### Training Configuration
- **Noise Schedule**: Geometric progression from σ_min = 0.01 to dataset-specific σ_max (=375 for AFHQ-512)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Monitoring**: TensorBoard-based progress tracking without FID computation due to resource constraints

## Project Structure

```
score-sde-project/
├── models/              # Model architectures and definitions
├── configs/             # Configuration files for different datasets
├── sampling.py          # Sampling algorithms and strategies
├── sde_lib.py          # SDE implementations (VE, VP, sub-VP)
├── datasets.py         # Dataset handling and preprocessing
├── run_lib.py          # Training and evaluation orchestration
├── samples/            # Generated samples and results
│   └── afhq_512/
│       └── general/
│           └── reverse_diffusion_base/  # Sample outputs
└── README.md
```

## Key Findings

### Strengths
- Excellent sample diversity without mode collapse
- Successful multi-resolution scaling (256×256 to 1024×1024)
- Effective downstream task performance using unified framework
- Robust performance across different visual domains

### Limitations
- High computational requirements (€300 for AFHQ-512 training)
- Symmetry challenges at very high resolutions (1024×1024)
- Classifier guidance requires dataset-specific hyperparameter tuning
- Slow sampling speed compared to other generative approaches

## Requirements

- PyTorch >= 1.8.0
- CUDA-capable GPU (recommended: V100 or better)
- ml_collections for configuration management
- Standard ML libraries: numpy, scipy, PIL, matplotlib

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{traore2025score,
  title={From Noise to Images: Implementing Score-Based Generative Models using Stochastic Differential Equations},
  author={Abdoulaye Traore},
  year={2025},
  institution={University of Torino, ENSTA Paris},
  type={Research Internship Report}
}
```

## Acknowledgments

This work was conducted at the University of Torino, Department of Mathematics "Giuseppe Peano," under the supervision of Professor Elena Issoglio and Professor Francesco Russo (ENSTA Paris). The research was supported by the Erasmus+ program.

Special thanks to Yang Song et al. for their foundational work on score-based generative modeling and for making their codebase available, which served as the foundation for this implementation.

## Related Work

- **Song et al. (2021)**: Score-Based Generative Modeling through Stochastic Differential Equations
- **Ho et al. (2020)**: Denoising Diffusion Probabilistic Models  
- **Song & Ermon (2019)**: Generative Modeling by Estimating Gradients of the Data Distribution
- **Jolicoeur-Martineau et al. (2021)**: Gotta Go Fast When Generating Data with Score-Based Models

## License

This project is available for academic and research purposes. Please refer to the original licenses of the base codebase and datasets used.
