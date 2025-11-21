# Fractal Neural Field (DEQ-AGI) Prototype

A novel, biologically plausible computing architecture that replaces traditional "layers" with a continuous, dynamical **Fractal Neural Field** on a toroidal topology.

## Architecture

- **Kernel**: Complex-valued cellular update rule with periodic padding (torus topology)
- **Solver**: Fixed-point finding using Anderson Acceleration and Multigrid methods
- **Model**: Deep Equilibrium Model (DEQ) that evolves hidden state towards attractor

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_mnist.py
```

The script will:
- Save checkpoints after each epoch to `trained_model.eqx`
- Save the best model (highest test accuracy) to `trained_model_best.eqx`
- Save epoch-specific checkpoints to `checkpoints/epoch_N.eqx`
- Automatically resume from last checkpoint if interrupted

## Visualization

```bash
python visualize.py
```

This will create `evolution_fractal.png` showing how the complex field evolves from input to attractor.

## Project Structure

- `src/kernel.py` - The physics/dynamics of the cell
- `src/solver.py` - Fixed-point finding algorithms
- `src/model.py` - The encapsulating Fractal Field architecture
- `train_mnist.py` - Training loop for MNIST classification

