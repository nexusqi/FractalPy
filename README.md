# Fractal Neural Field (DEQ-AGI) Prototype

A novel, biologically plausible computing architecture that replaces traditional "layers" with a continuous, dynamical **Fractal Neural Field** on a toroidal topology.

## 🎯 Overview

This project implements an experimental neural architecture that fundamentally differs from standard deep learning approaches:

- **No Transformers**: Uses Deep Equilibrium Models (DEQ) instead of transformer layers
- **No Attention**: Fixed-point solvers replace self-attention mechanisms
- **Complex-Valued Networks**: Works with complex numbers instead of real-valued weights
- **Continuous Dynamics**: Field-based evolution instead of discrete layer-by-layer processing
- **Fractal Structure**: Weight-sharing across iterations creates self-similar patterns

## 🏗️ Architecture

### Core Components

#### 1. **Fractal Kernel** (`src/kernel.py`)
- **Complex-valued 3×3 convolutions** with periodic padding (torus topology)
- **ModReLU/Cardioid activations** for complex numbers
- **Cellular update rule**: `Z_new = (1-α)·Z_old + α·σ(Conv(Z_old) + InputInjection)`
- **Spectral normalization** for stability

#### 2. **Fixed-Point Solver** (`src/solver.py`)
Three methods for finding equilibrium states:
- **Naive Solver**: Simple iterative updates using `jax.lax.scan`
- **Anderson Acceleration**: Jacobian-free acceleration using residual history
- **Multigrid V-Cycle**: Multi-resolution coarse-to-fine solving

#### 3. **Fractal Field Model** (`src/model.py`)
- **Encoder**: Real image → Complex hidden state
- **Core**: Fixed-point solver with fractal kernel (evolves to attractor)
- **Readout**: Complex state → Class logits

### Key Concepts

**Deep Equilibrium Models (DEQ)**: Instead of fixed-depth networks, the model finds a fixed point where `f(z*) = z*`. This allows:
- Adaptive depth (converges when ready)
- Memory efficiency (no need to store intermediate states)
- Global information flow (all positions interact through field dynamics)

**Toroidal Topology**: Periodic boundary conditions create a torus, enabling:
- No edge effects
- Circular information flow
- Symmetric spatial processing

**Complex-Valued Networks**: Using complex numbers provides:
- Phase information (rotation in complex plane)
- Richer representations
- Natural handling of periodic patterns

## 📦 Project Structure

```
FractalAtlas/
├── src/
│   ├── kernel.py          # Complex-valued convolution & fractal update rule
│   ├── solver.py          # Fixed-point finding algorithms
│   └── model.py           # Complete DEQ architecture
│
├── fractal_atlas_project/ # Text encoding experiments
│   ├── fractal_decoder_v*.py  # Coordinate-based text decoders
│   ├── train_single_book_v*.py
│   ├── train_key_v*.py
│   └── data/              # Text books for training
│
├── train_mnist.py         # Main training script
├── visualize.py           # Field evolution visualization
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training on MNIST

```bash
python train_mnist.py
```

The training script will:
- Download MNIST dataset automatically
- Save checkpoints after each epoch to `trained_model.eqx`
- Save the best model (highest test accuracy) to `trained_model_best.eqx`
- Save epoch-specific checkpoints to `checkpoints/epoch_N.eqx`
- Automatically resume from last checkpoint if interrupted

### Visualization

```bash
python visualize.py
```

This creates `evolution_fractal.png` showing:
- **Magnitude evolution**: How field intensity changes over iterations
- **Phase evolution**: How field phase (color) evolves to attractor
- **Crystallization process**: The convergence from input to fixed point

## 🔬 Fractal Atlas: Coordinate-Based Text Encoding

The `fractal_atlas_project/` directory contains experiments with **coordinate-based text encoding**:

### Concept
Instead of token embeddings, the model learns: **coordinate x ∈ [0,1] → symbol**

- **Fourier Features**: Octave-based frequencies for multi-scale representation
- **HashGrid 1D**: Instant-NGP style hashed grid encoding
- **Fractal Kernel**: Weight-sharing across T iterations
- **Sparse Updates**: 90% of steps can be static (efficiency)
- **Latent Keys**: Compact vectors (32-64 dim) per book for library compression

### Usage

```bash
# Train master kernel
python -m fractal_atlas_project.train_single_book_v7

# Train book-specific key
python -m fractal_atlas_project.train_key_v7

# Inference
python -m fractal_atlas_project.infer_single_book_v7
```

See `fractal_atlas_project/README_ru.md` for detailed documentation (in Russian).

## 🧪 Technical Details

### Hyperparameters (MNIST)

- **Hidden channels**: 16 (complex)
- **Spatial size**: 28×28 (preserved)
- **Solver steps**: 30 (BPTT depth)
- **Solver method**: "naive", "anderson", or "multigrid"
- **Learning rate**: 1e-3 (Adam optimizer)
- **Batch size**: 32

### Model Size

The "Nano" version targets <20k parameters:
- Encoder: Conv2D (1 → 16 channels)
- Kernel: Complex Conv2D (16 → 16 channels, 3×3)
- Readout: Linear (16 → 10 classes)

### Dependencies

- **JAX** (≥0.4.20): Automatic differentiation & JIT compilation
- **Equinox** (≥0.11.0): Neural network library for JAX
- **Optax** (≥0.1.7): Optimizers
- **NumPy** (≥1.24.0): Numerical computing
- **Matplotlib** (≥3.7.0): Visualization

## 🎓 How It Works

### Forward Pass

1. **Encode**: Real image → Complex hidden state `z₀`
2. **Inject**: Use encoded state as persistent input injection
3. **Iterate**: Apply fractal kernel repeatedly:
   ```
   z_{t+1} = (1-α)·z_t + α·σ(Conv(z_t) + injection)
   ```
4. **Converge**: Stop when `||z_{t+1} - z_t|| < ε` (fixed point)
5. **Readout**: Extract magnitude → global pooling → logits

### Backward Pass

Uses **implicit differentiation** through fixed point:
- No need to unroll all iterations
- Memory-efficient gradient computation
- Handled automatically by JAX's autodiff

## 🔍 Key Differences from Standard Architectures

| Standard Approach | Fractal Neural Field |
|-------------------|---------------------|
| Discrete layers | Continuous field evolution |
| Fixed depth | Adaptive depth (convergence-based) |
| Real-valued weights | Complex-valued weights |
| Attention mechanisms | Fixed-point solvers |
| Token embeddings | Coordinate-based encoding (Atlas) |
| Sequential processing | Parallel field updates |

## 📊 Results

The model is trained on MNIST digit classification. Check `training_log.txt` for training history.

## 🚧 Future Work

- [ ] Extend to larger image datasets (CIFAR-10, ImageNet)
- [ ] Multi-scale fractal kernels
- [ ] Attention-like mechanisms using field interactions
- [ ] Quantization (8-bit/4-bit) for deployment
- [ ] ONNX export for inference
- [ ] Web demo for visualization

## 📚 References

- **Deep Equilibrium Models**: [Bai et al., 2019](https://arxiv.org/abs/1909.01377)
- **Complex-Valued Neural Networks**: [Trabelsi et al., 2018](https://arxiv.org/abs/1705.09792)
- **Anderson Acceleration**: [Anderson, 1965](https://doi.org/10.1090/qam/99999)

## 📝 License

See repository for license information.

## 👥 Authors

Serhii & Sasha — Fractal Atlas project

---

**Note**: This is experimental research code. The architecture is novel and may require tuning for different tasks.
