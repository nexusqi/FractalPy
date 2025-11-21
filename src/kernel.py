"""
Fractal Neural Field Kernel - Complex-valued cellular update rule.

This module implements the core dynamics of the fractal field:
- Complex-valued 3x3 convolution with periodic padding (torus topology)
- ModReLU activation for complex numbers
- State update with learnable step size
"""

from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx


def periodic_pad(x: jnp.ndarray, pad_width: int) -> jnp.ndarray:
    """
    Apply periodic (wrap) padding to simulate toroidal topology.
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        pad_width: Padding width (typically 1 for 3x3 conv)
    
    Returns:
        Padded tensor of shape [B, C, H+2*pad_width, W+2*pad_width]
    """
    # For 2D images, pad along last two dimensions
    # mode='wrap' creates periodic boundary conditions
    return jnp.pad(x, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='wrap')


def mod_relu(z: jnp.ndarray, bias: float = 0.0) -> jnp.ndarray:
    """
    ModReLU activation for complex numbers.
    
    Formula: ModReLU(z) = z * ReLU(|z| + b) / |z|
    
    Args:
        z: Complex tensor
        bias: Bias term (default 0.0)
    
    Returns:
        Activated complex tensor
    """
    magnitude = jnp.abs(z)
    # Avoid division by zero
    magnitude_safe = jnp.maximum(magnitude, 1e-8)
    # ReLU(|z| + b)
    relu_magnitude = jnp.maximum(magnitude + bias, 0.0)
    # Normalize and scale
    return z * (relu_magnitude / magnitude_safe)


def cardioid(z: jnp.ndarray) -> jnp.ndarray:
    """
    Cardioid activation for complex numbers (alternative to ModReLU).
    
    Formula: Cardioid(z) = z * (1 + cos(phase(z))) / 2
    
    Args:
        z: Complex tensor
    
    Returns:
        Activated complex tensor
    """
    phase = jnp.angle(z)
    return z * (1 + jnp.cos(phase)) / 2


class ComplexConv2D(eqx.Module):
    """
    Complex-valued 2D Convolution with periodic padding.
    """
    weight: jnp.ndarray  # Complex weights [out_channels, in_channels, 3, 3]
    bias: jnp.ndarray    # Complex bias [out_channels]
    
    def __init__(self, in_channels: int, out_channels: int, key: jax.random.PRNGKey):
        """
        Initialize complex convolution weights.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            key: JAX random key
        """
        # Initialize as complex: real + imag
        # AGGRESSIVE INITIALIZATION to prevent vanishing gradients
        key_real, key_imag = jax.random.split(key, 2)
        # Xavier/Glorot initialization with higher variance
        scale = jnp.sqrt(2.0 / (in_channels * 9)) * 1.5  # Boosted scale
        weight_real = jax.random.normal(key_real, (out_channels, in_channels, 3, 3)) * scale
        weight_imag = jax.random.normal(key_imag, (out_channels, in_channels, 3, 3)) * scale
        self.weight = weight_real + 1j * weight_imag
        
        # Initialize bias with higher variance
        bias_real = jax.random.normal(key_real, (out_channels,)) * 0.1  # Increased from 0.01
        bias_imag = jax.random.normal(key_imag, (out_channels,)) * 0.1
        self.bias = bias_real + 1j * bias_imag
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply complex convolution with periodic padding.
        
        Args:
            x: Input complex tensor [B, C, H, W]
        
        Returns:
            Convolved complex tensor [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        out_channels = self.weight.shape[0]
        
        # Apply periodic padding (pad_width=1 for 3x3 conv)
        x_padded = periodic_pad(x, pad_width=1)
        
        # Manual convolution with periodic padding
        # For complex numbers, we need to handle real and imaginary parts separately
        # conv(z) = conv(real(z)) - conv(imag(z)) + 1j * (conv(real(z)) + conv(imag(z)))
        # Actually, simpler: treat as real convolution on [real, imag] channels
        
        # Split into real and imaginary parts
        x_real = jnp.real(x_padded)
        x_imag = jnp.imag(x_padded)
        
        # Stack real and imag as separate channels
        x_stacked = jnp.concatenate([x_real, x_imag], axis=1)  # [B, 2*C, H+2, W+2]
        
        # Split weights into real and imaginary parts
        w_real = jnp.real(self.weight)  # [out_channels, in_channels, 3, 3]
        w_imag = jnp.imag(self.weight)
        
        # Stack weights: [real, -imag] for real output, [imag, real] for imag output
        w_real_out = jnp.concatenate([w_real, -w_imag], axis=1)  # [out_channels, 2*in_channels, 3, 3]
        w_imag_out = jnp.concatenate([w_imag, w_real], axis=1)
        
        # Apply convolution (using JAX's conv_general_dilated)
        # We'll use a simpler approach: manual sliding window
        output_real = self._conv2d(x_stacked, w_real_out)  # [B, out_channels, H, W]
        output_imag = self._conv2d(x_stacked, w_imag_out)
        
        # Combine back to complex
        output = output_real + 1j * output_imag
        
        # Add bias
        output = output + self.bias[None, :, None, None]
        
        return output
    
    def _conv2d(self, x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """
        Manual 2D convolution implementation.
        
        Args:
            x: Input [B, C, H, W]
            w: Weights [out_channels, in_channels, 3, 3]
        
        Returns:
            Output [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        out_channels = w.shape[0]
        
        # Use JAX's conv_general_dilated for efficiency
        # But we need to handle the padding manually since we already padded
        # Actually, let's use lax.conv_general_dilated with valid padding
        # since we already applied periodic padding
        
        # Reshape for conv: [B, C, H, W] -> [B, C, H, W]
        # Use 'valid' padding since we already padded
        output = jax.lax.conv_general_dilated(
            x,  # [B, C, H, W]
            w,  # [out_channels, in_channels, 3, 3]
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        return output


class FractalKernel(eqx.Module):
    """
    The core fractal update rule for the neural field.
    
    Implements: Z_new = (1 - α) * Z_old + α * σ(Conv(Z_old) + InputInjection)
    """
    conv: ComplexConv2D
    alpha: jnp.ndarray  # Learnable step size [1] or per-channel [C]
    activation: str = "modrelu"  # "modrelu" or "cardioid"
    modrelu_bias: jnp.ndarray  # Learnable bias for ModReLU
    
    def __init__(
        self,
        channels: int,
        key: jax.random.PRNGKey,
        alpha_init: float = 0.1,
        activation: str = "modrelu",
        modrelu_bias: float = 0.0
    ):
        """
        Initialize the fractal kernel.
        
        Args:
            channels: Number of channels (input = output)
            alpha_init: Initial step size value
            activation: Activation function ("modrelu" or "cardioid")
            modrelu_bias: Bias for ModReLU
            key: JAX random key
        """
        conv_key, _ = jax.random.split(key)
        self.conv = ComplexConv2D(channels, channels, conv_key)
        # Use provided alpha_init (can be 0.5 for faster dynamics)
        self.alpha = jnp.array(alpha_init)
        self.activation = activation
        self.modrelu_bias = jnp.array(modrelu_bias)  # Make learnable
    
    def __call__(self, z: jnp.ndarray, input_injection: jnp.ndarray) -> jnp.ndarray:
        """
        Apply one step of the fractal update rule.
        
        Args:
            z: Current hidden state [B, C, H, W] (complex)
            input_injection: Input injection [B, C, H, W] (complex)
        
        Returns:
            Updated hidden state [B, C, H, W] (complex)
        """
        # Apply convolution
        conv_out = self.conv(z)
        
        # Add input injection
        combined = conv_out + input_injection
        
        # Apply activation
        if self.activation == "modrelu":
            activated = mod_relu(combined, bias=self.modrelu_bias)
        elif self.activation == "cardioid":
            activated = cardioid(combined)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Update rule: Z_new = (1 - α) * Z_old + α * σ(Conv(Z_old) + InputInjection)
        z_new = (1 - self.alpha) * z + self.alpha * activated
        
        return z_new


def spectral_normalize(weight: jnp.ndarray, max_singular_value: float = 1.0) -> jnp.ndarray:
    """
    Apply spectral normalization to keep Lipschitz constant ≈ 1.
    
    IMPROVED VERSION: More aggressive normalization for stability.
    
    Args:
        weight: Complex weight tensor [out_channels, in_channels, 3, 3]
        max_singular_value: Maximum allowed singular value (default 1.0)
    
    Returns:
        Normalized weight tensor
    """
    # For complex matrices, compute spectral norm
    # Reshape weight to 2D: [out_channels, in_channels * 3 * 3]
    w_reshaped = weight.reshape(weight.shape[0], -1)
    
    # For complex weights, compute norm of the real representation
    w_real = jnp.real(w_reshaped)
    w_imag = jnp.imag(w_reshaped)
    
    # Stack real and imaginary parts for spectral norm calculation
    # [real, -imag] and [imag, real] for proper complex matrix norm
    w_combined = jnp.concatenate([w_real, -w_imag, w_imag, w_real], axis=1)
    
    # Compute spectral norm using power iteration (more iterations for accuracy)
    u = jnp.ones((w_combined.shape[0],)) / jnp.sqrt(w_combined.shape[0])
    for _ in range(10):  # Increased iterations for better accuracy
        v = w_combined.T @ u
        v_norm = jnp.linalg.norm(v)
        v = v / (v_norm + 1e-10)
        u = w_combined @ v
        u_norm = jnp.linalg.norm(u)
        u = u / (u_norm + 1e-10)
    
    # Compute spectral norm (largest singular value)
    sigma = jnp.linalg.norm(w_combined @ v)
    
    # ALWAYS normalize to enforce strict Lipschitz constraint
    # This prevents oscillation by keeping energy bounded
    scale = max_singular_value / (sigma + 1e-10)
    weight = weight * scale
    
    return weight

