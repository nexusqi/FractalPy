"""
Visualization script for Fractal Neural Field dynamics.

This script visualizes how the complex field evolves from input to attractor,
showing the "crystallization" process.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import equinox as eqx
import gzip
import os

from src.model import FractalFieldClassifier
from src.solver import naive_solver
from src.kernel import spectral_normalize


def load_mnist_sample(digit: int = 5, split: str = "test"):
    """
    Load a single MNIST sample of the specified digit.
    
    Args:
        digit: Digit to load (0-9)
        split: Dataset split ("train" or "test")
    
    Returns:
        Tuple of (image, label) where image is [28, 28]
    """
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    if split == "train":
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
        num_samples = 60000
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"
        num_samples = 10000
    
    data_dir = "mnist_data"
    images_path = os.path.join(data_dir, images_file)
    labels_path = os.path.join(data_dir, labels_file)
    
    # Load images
    with gzip.open(images_path, 'rb') as f:
        f.read(16)  # Skip header
        buf = f.read(28 * 28 * num_samples)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(num_samples, 28, 28) / 255.0
    
    # Load labels
    with gzip.open(labels_path, 'rb') as f:
        f.read(8)  # Skip header
        buf = f.read(num_samples)
        labels = np.frombuffer(buf, dtype=np.uint8)
    
    # Find first occurrence of the digit
    digit_indices = np.where(labels == digit)[0]
    if len(digit_indices) == 0:
        raise ValueError(f"Digit {digit} not found in {split} set")
    
    idx = digit_indices[0]
    return images[idx], labels[idx]


def load_model(model_path: str = None):
    """
    Load a trained model.
    
    Args:
        model_path: Path to saved model. If None, tries best model first, then regular model.
    
    Returns:
        Loaded model
    """
    # Try to find model if path not specified
    if model_path is None:
        # Try V1 Nano models first
        nano_best = "trained_model_v1_nano_best.eqx"
        nano_regular = "trained_model_v1_nano.eqx"
        # Fallback to V0 models
        v0_best = "trained_model_best.eqx"
        v0_regular = "trained_model.eqx"
        
        if os.path.exists(nano_best):
            model_path = nano_best
            print(f"Using V1 Nano best model: {nano_best}")
        elif os.path.exists(nano_regular):
            model_path = nano_regular
            print(f"Using V1 Nano model: {nano_regular}")
        elif os.path.exists(v0_best):
            model_path = v0_best
            print(f"Using V0 best model: {v0_best}")
        elif os.path.exists(v0_regular):
            model_path = v0_regular
            print(f"Using V0 model: {v0_regular}")
        else:
            raise FileNotFoundError(
                f"No model file found. Please train the model first with train_mnist.py"
            )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Please train the model first with train_mnist.py"
        )
    
    # Model hyperparameters (must match training)
    # Try to detect model version from path
    is_v1_nano = "v1_nano" in model_path or "nano" in model_path.lower()
    
    if is_v1_nano:
        # V1 Nano model
        in_channels = 1
        hidden_channels = 16
        spatial_size = (28, 28)
        num_classes = 10
        
        key = jax.random.PRNGKey(42)
        model = FractalFieldClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            spatial_size=spatial_size,
            num_classes=num_classes,
            key=key,
            alpha_init=0.1,
            activation="modrelu",
            solver_method="naive",
            num_steps=30,
            use_spectral_norm=True
        )
    else:
        # V0 model (legacy)
        input_dim = 28 * 28
        hidden_channels = 16
        spatial_size = (28, 28)
        num_classes = 10
        
        key = jax.random.PRNGKey(42)
        # For V0, we need to handle old signature - but this won't work
        # Let's assume V1 for now
        in_channels = 1
        model = FractalFieldClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            spatial_size=spatial_size,
            num_classes=num_classes,
            key=key,
            alpha_init=0.1,
            activation="modrelu",
            solver_method="naive",
            num_steps=30,
            use_spectral_norm=True
        )
    
    # Load weights
    model = eqx.tree_deserialise_leaves(model_path, model)
    print(f"Model loaded from {model_path}")
    
    return model


def capture_evolution(model: FractalFieldClassifier, x: jnp.ndarray, capture_steps: list):
    """
    Run the solver and capture states at specified steps.
    
    Args:
        model: The fractal field model
        x: Input image [1, 784]
        capture_steps: List of step indices to capture (e.g., [0, 5, 10, 30])
    
    Returns:
        Dictionary mapping step -> complex state [1, C, H, W]
    """
    # Encode input
    z_init = model.encoder(x)  # [1, C, H, W]
    input_injection = z_init
    
    # Create kernel function
    if model.use_spectral_norm:
        normalized_weight = spectral_normalize(model.kernel.conv.weight)
        normalized_kernel = eqx.tree_at(
            lambda k: k.conv.weight,
            model.kernel,
            normalized_weight
        )
        kernel_fn = lambda z, inj: normalized_kernel(z, inj)
    else:
        kernel_fn = lambda z, inj: model.kernel(z, inj)
    
    # Run solver with state capture
    max_step = max(capture_steps)
    _, _, captured_states = naive_solver(
        kernel_fn,
        z_init,
        input_injection,
        num_steps=max_step + 1,
        capture_steps=capture_steps
    )
    
    # Add initial state
    captured_states[0] = z_init
    
    return captured_states


def visualize_fractal_evolution(captured_states: dict, output_path: str = "evolution_fractal.png"):
    """
    Visualize the evolution of the complex field.
    
    Args:
        captured_states: Dictionary mapping step -> complex state [1, C, H, W]
        output_path: Path to save the visualization
    """
    steps = sorted(captured_states.keys())
    num_steps = len(steps)
    
    # Get shape info
    z_sample = captured_states[steps[0]]
    B, C, H, W = z_sample.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4 * num_steps))
    gs = GridSpec(num_steps, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, step in enumerate(steps):
        z = captured_states[step]
        z = z[0]  # Remove batch dimension: [C, H, W]
        
        # Average across channels to get single [H, W] representation
        z_avg = jnp.mean(z, axis=0)  # [H, W] complex
        
        # Extract magnitude and phase
        magnitude = jnp.abs(z_avg)  # [H, W]
        phase = jnp.angle(z_avg)  # [H, W]
        
        # Plot magnitude (brightness)
        ax_mag = fig.add_subplot(gs[i, 0])
        im_mag = ax_mag.imshow(
            np.array(magnitude),
            cmap='hot',
            aspect='auto',
            origin='lower',
            interpolation='bilinear'
        )
        ax_mag.set_title(f'Step {step}: Magnitude (Brightness)', fontsize=12, fontweight='bold')
        ax_mag.set_xlabel('Width')
        ax_mag.set_ylabel('Height')
        plt.colorbar(im_mag, ax=ax_mag, label='Magnitude')
        
        # Plot phase (color) - use HSV color space
        ax_phase = fig.add_subplot(gs[i, 1])
        # Normalize phase to [0, 1] for HSV
        phase_norm = (phase + jnp.pi) / (2 * jnp.pi)  # [-π, π] -> [0, 1]
        # Create HSV image: H=phase, S=1, V=magnitude (normalized)
        magnitude_norm = magnitude / (jnp.max(magnitude) + 1e-8)
        hsv_image = np.stack([
            np.array(phase_norm),
            np.ones_like(phase_norm),
            np.array(magnitude_norm)
        ], axis=-1)
        rgb_image = mcolors.hsv_to_rgb(hsv_image)
        
        im_phase = ax_phase.imshow(
            rgb_image,
            aspect='auto',
            origin='lower',
            interpolation='bilinear'
        )
        ax_phase.set_title(f'Step {step}: Phase (Color) + Magnitude (Brightness)', 
                          fontsize=12, fontweight='bold')
        ax_phase.set_xlabel('Width')
        ax_phase.set_ylabel('Height')
        
        # Add colorbar for phase
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, 
                                   norm=plt.Normalize(vmin=-jnp.pi, vmax=jnp.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_phase)
        cbar.set_label('Phase (radians)', rotation=270, labelpad=15)
    
    plt.suptitle('Fractal Neural Field Evolution: Crystallization to Attractor', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    print("=" * 60)
    print("Fractal Neural Field Visualization")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading trained model...")
    try:
        model = load_model()  # Will try best model first, then regular
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Load sample image (digit 5)
    print("\n2. Loading MNIST sample (digit 5)...")
    image, label = load_mnist_sample(digit=5, split="test")
    print(f"   Loaded digit {label} from test set")
    
    # Reshape to NCHW format and add batch dimension
    x = jnp.array(image.reshape(1, 1, 28, 28))  # [1, 1, 28, 28]
    
    # Capture evolution at specific steps
    capture_steps = [0, 5, 10, 30]
    print(f"\n3. Running solver and capturing states at steps {capture_steps}...")
    captured_states = capture_evolution(model, x, capture_steps)
    
    # Get prediction
    print("\n4. Computing prediction...")
    logits, _ = model(x)
    pred = int(jnp.argmax(logits[0]))
    print(f"   Model prediction: {pred} (true label: {label})")
    
    # Visualize
    print("\n5. Creating visualization...")
    visualize_fractal_evolution(captured_states, "evolution_fractal.png")
    
    print("\n" + "=" * 60)
    print("Visualization complete! Check evolution_fractal.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

