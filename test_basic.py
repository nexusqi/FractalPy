"""
Basic test script to verify the implementation works.
"""

import jax
import jax.numpy as jnp
from src.kernel import FractalKernel, mod_relu
from src.solver import naive_solver
from src.model import FractalFieldClassifier

print("Testing Fractal Neural Field implementation...")
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())

# Test ModReLU
print("\n1. Testing ModReLU...")
z_test = jnp.array([1.0 + 1.0j, -0.5 + 0.5j])
z_activated = mod_relu(z_test, bias=0.0)
print(f"Input: {z_test}")
print(f"Output: {z_activated}")
print("✓ ModReLU works")

# Test Kernel
print("\n2. Testing FractalKernel...")
key = jax.random.PRNGKey(42)
kernel = FractalKernel(channels=4, key=key, alpha_init=0.1)
z_init = jax.random.normal(key, (2, 4, 8, 8)) + 1j * jax.random.normal(key, (2, 4, 8, 8))
input_inj = jax.random.normal(key, (2, 4, 8, 8)) + 1j * jax.random.normal(key, (2, 4, 8, 8))
z_new = kernel(z_init, input_inj)
print(f"Input shape: {z_init.shape}")
print(f"Output shape: {z_new.shape}")
print("✓ FractalKernel works")

# Test Solver
print("\n3. Testing Naive Solver...")
def kernel_fn(z, inj):
    return kernel(z, inj)

z_final, deltas, _ = naive_solver(kernel_fn, z_init, input_inj, num_steps=5)
print(f"Final state shape: {z_final.shape}")
print(f"Convergence deltas: {deltas}")
print("✓ Naive Solver works")

# Test Model
print("\n4. Testing FractalFieldClassifier...")
model = FractalFieldClassifier(
    input_dim=784,
    hidden_channels=8,
    spatial_size=(28, 28),
    num_classes=10,
    key=key
)
x_test = jax.random.normal(key, (2, 784))
logits, conv_history = model(x_test)
print(f"Input shape: {x_test.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Convergence history shape: {conv_history.shape}")
print("✓ FractalFieldClassifier works")

print("\n✅ All basic tests passed!")

