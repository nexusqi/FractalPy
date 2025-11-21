"""
Fractal Neural Field Model - Complete DEQ Architecture.

This module assembles the full model:
- Encoder: Real image -> Complex hidden state
- Core: Fixed-point solver with fractal kernel
- Readout: Complex state -> Class logits
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx

from .kernel import FractalKernel, spectral_normalize
from .solver import solve_fixed_point


class Encoder(eqx.Module):
    """
    Lightweight convolutional encoder: Real image -> Complex hidden state.
    
    V1 "Nano": Uses Conv2D instead of Linear to drastically reduce parameters.
    """
    weight: jnp.ndarray  # [out_channels, in_channels, 3, 3]
    bias: jnp.ndarray    # [out_channels]
    hidden_channels: int
    
    def __init__(self, in_channels: int, hidden_channels: int, key: jax.random.PRNGKey):
        """
        Initialize lightweight convolutional encoder.
        
        Args:
            in_channels: Input channels (1 for grayscale MNIST)
            hidden_channels: Number of output complex channels
            key: JAX random key
        """
        # Initialize Conv2D weights: kernel_size=3, padding=1
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (in_channels * 9))
        self.weight = jax.random.normal(key, (hidden_channels, in_channels, 3, 3)) * scale
        self.bias = jax.random.normal(key, (hidden_channels,)) * 0.01
        self.hidden_channels = hidden_channels
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encode input to complex hidden state.
        
        Args:
            x: Input image [B, 1, H, W] (NCHW format)
        
        Returns:
            Complex hidden state [B, hidden_channels, H, W]
        """
        # Apply convolution with padding=1 to preserve spatial dimensions
        # Use circular padding for torus topology
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='wrap')
        
        # Apply convolution: [B, 1, H+2, W+2] -> [B, hidden_channels, H, W]
        x_real = jax.lax.conv_general_dilated(
            x_padded,  # [B, in_channels, H+2, W+2]
            self.weight,  # [out_channels, in_channels, 3, 3]
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        # Add bias
        x_real = x_real + self.bias[None, :, None, None]
        
        # Initialize imaginary part as zeros
        # This creates a complex field where Real = conv output, Imag = 0
        x_imag = jnp.zeros_like(x_real)
        
        # Combine to complex: [B, hidden_channels, H, W]
        z = x_real + 1j * x_imag
        
        return z


class Readout(eqx.Module):
    """
    Reads out class logits from complex equilibrium state.
    """
    linear: eqx.nn.Linear
    
    def __init__(self, hidden_channels: int, num_classes: int, key: jax.random.PRNGKey):
        """
        Initialize readout layer.
        
        Args:
            hidden_channels: Number of complex channels
            num_classes: Number of output classes
            key: JAX random key
        """
        # Input: hidden_channels (after global average pooling)
        self.linear = eqx.nn.Linear(hidden_channels, num_classes, key=key)
    
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Readout from complex state to logits.
        
        Args:
            z: Complex equilibrium state [B, C, H, W]
        
        Returns:
            Class logits [B, num_classes]
        """
        # Take magnitude of complex state
        z_magnitude = jnp.abs(z)  # [B, C, H, W]
        
        # Global average pooling
        z_pooled = jnp.mean(z_magnitude, axis=(2, 3))  # [B, C]
        
        # Project to logits
        # Equinox Linear computes weight @ x, so we need x.T
        # z_pooled: [B, C] -> z_pooled.T: [C, B]
        # weight: [num_classes, C]
        # weight @ z_pooled.T: [num_classes, B] -> transpose to [B, num_classes]
        logits = (self.linear.weight @ z_pooled.T).T + self.linear.bias  # [B, num_classes]
        
        return logits


class FractalFieldClassifier(eqx.Module):
    """
    Complete Fractal Neural Field Classifier.
    
    Architecture:
    1. Encoder: Real image -> Complex hidden state
    2. Core: Fixed-point solver with fractal kernel
    3. Readout: Complex state -> Class logits
    """
    encoder: Encoder
    kernel: FractalKernel
    readout: Readout
    solver_method: str = "naive"
    num_steps: int = 30
    use_spectral_norm: bool = True
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        spatial_size: Tuple[int, int],
        num_classes: int,
        key: jax.random.PRNGKey,
        alpha_init: float = 0.1,
        activation: str = "modrelu",
        solver_method: str = "naive",
        num_steps: int = 30,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the complete model (V1 "Nano").
        
        Args:
            in_channels: Input channels (1 for grayscale MNIST)
            hidden_channels: Number of complex channels
            spatial_size: Spatial size (H, W) - kept for compatibility
            num_classes: Number of output classes
            alpha_init: Initial step size for kernel
            activation: Activation function ("modrelu" or "cardioid")
            solver_method: Solver method ("naive", "anderson", "multigrid")
            num_steps: Number of solver steps
            use_spectral_norm: Whether to apply spectral normalization
            key: JAX random key
        """
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Lightweight Conv2D encoder: [B, 1, H, W] -> [B, hidden_channels, H, W]
        self.encoder = Encoder(in_channels, hidden_channels, key1)
        self.kernel = FractalKernel(
            hidden_channels, key2, alpha_init, activation
        )
        self.readout = Readout(hidden_channels, num_classes, key3)
        self.solver_method = solver_method
        self.num_steps = num_steps
        self.use_spectral_norm = use_spectral_norm
    
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image [B, 1, H, W] (NCHW format, not flattened)
        
        Returns:
            Tuple of (logits, convergence_history)
            - logits: [B, num_classes]
            - convergence_history: [num_steps] (convergence deltas)
        """
        # Encode input to complex hidden state
        z_init = self.encoder(x)  # [B, C, H, W]
        
        # Use encoded state as input injection
        input_injection = z_init
        
        # Create kernel function
        # Apply spectral normalization if enabled (functionally, without mutating)
        if self.use_spectral_norm:
            # Create normalized kernel for this forward pass
            normalized_weight = spectral_normalize(self.kernel.conv.weight)
            # Create a new kernel with normalized weights using eqx.tree_at
            normalized_kernel = eqx.tree_at(
                lambda k: k.conv.weight,
                self.kernel,
                normalized_weight
            )
            
            def kernel_fn(z, inj):
                return normalized_kernel(z, inj)
        else:
            def kernel_fn(z, inj):
                return self.kernel(z, inj)
        
        # Solve for fixed point
        z_equilibrium, convergence_history = solve_fixed_point(
            kernel_fn,
            z_init,
            input_injection,
            method=self.solver_method,
            num_steps=self.num_steps
        )
        
        # Readout to logits
        logits = self.readout(z_equilibrium)
        
        return logits, convergence_history
    
    def get_convergence_metric(self, convergence_history: jnp.ndarray) -> float:
        """
        Compute convergence metric from history.
        
        Args:
            convergence_history: [num_steps] convergence deltas
        
        Returns:
            Final convergence delta (stability measure)
        """
        return float(jnp.mean(convergence_history[-5:]))  # Average of last 5 steps

