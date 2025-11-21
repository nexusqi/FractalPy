"""
Training script for Fractal Neural Field on MNIST.

This script trains the DEQ model to classify MNIST digits using attractor dynamics.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Tuple
import equinox as eqx
import urllib.request
import gzip
import os

from src.model import FractalFieldClassifier

# Configure JAX for CPU (as requested)
# JAX will automatically use CPU if Metal is not available
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())


def load_mnist_numpy(batch_size: int = 32, split: str = "train"):
    """
    Load MNIST dataset directly from files (no TensorFlow required).
    
    Args:
        batch_size: Batch size
        split: Dataset split ("train" or "test")
    
    Returns:
        Generator yielding (images, labels) batches
    """
    # URLs for MNIST dataset
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    if split == "train":
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
        num_samples = 60000
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"
        num_samples = 10000
    
    # Download if not exists
    data_dir = "mnist_data"
    os.makedirs(data_dir, exist_ok=True)
    
    images_path = os.path.join(data_dir, images_file)
    labels_path = os.path.join(data_dir, labels_file)
    
    if not os.path.exists(images_path):
        print(f"Downloading {images_file}...")
        urllib.request.urlretrieve(base_url + images_file, images_path)
    
    if not os.path.exists(labels_path):
        print(f"Downloading {labels_file}...")
        urllib.request.urlretrieve(base_url + labels_file, labels_path)
    
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
    
    # Shuffle if training
    if split == "train":
        indices = np.random.permutation(num_samples)
        images = images[indices]
        labels = labels[indices]
    
    # Create batches
    num_batches = num_samples // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Reshape to NCHW format: [B, 28, 28] -> [B, 1, 28, 28]
        # Add channel dimension for Conv2D encoder
        batch_images = batch_images.reshape(batch_size, 1, 28, 28)
        
        yield batch_images, batch_labels


def loss_fn(model: FractalFieldClassifier, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[float, dict]:
    """
    Compute loss and metrics.
    
    Args:
        model: The fractal field model
        x: Input images [B, 784]
        y: Labels [B]
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    logits, convergence_history = model(x)
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    
    # Accuracy
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == y).mean()
    
    # Convergence metric
    convergence_delta = jnp.mean(convergence_history[-5:])
    
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "convergence_delta": convergence_delta
    }
    
    return loss, metrics


def train_step(
    model: FractalFieldClassifier,
    opt_state: optax.OptState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    optimizer: optax.GradientTransformation
) -> Tuple[FractalFieldClassifier, optax.OptState, dict]:
    """
    Perform one training step.
    
    Args:
        model: The fractal field model
        opt_state: Optimizer state
        x: Input images [B, 784]
        y: Labels [B]
        optimizer: Optax optimizer
    
    Returns:
        Tuple of (updated_model, updated_opt_state, metrics)
    """
    # Compute loss and gradients using Equinox filter
    def loss_fn_wrapper(m):
        return loss_fn(m, x, y)
    
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn_wrapper, has_aux=True)(model)
    
    # Update model
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, metrics


def load_model_checkpoint(model_path: str, in_channels: int, hidden_channels: int, 
                         spatial_size: tuple, num_classes: int, key: jax.random.PRNGKey):
    """
    Load model from checkpoint if exists, otherwise create new.
    
    Returns:
        model, start_epoch
    """
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        # Initialize model structure
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
        
        # Try to load epoch info
        epoch_file = model_path.replace(".eqx", "_epoch.txt")
        start_epoch = 0
        if os.path.exists(epoch_file):
            with open(epoch_file, 'r') as f:
                start_epoch = int(f.read().strip())
        print(f"Resuming from epoch {start_epoch}")
        return model, start_epoch
    else:
        print("No checkpoint found, initializing new model...")
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
        return model, 0


def save_model_checkpoint(model, epoch: int, model_path: str, is_best: bool = False):
    """Save model checkpoint."""
    # Save main checkpoint
    eqx.tree_serialise_leaves(model_path, model)
    
    # Save epoch info
    epoch_file = model_path.replace(".eqx", "_epoch.txt")
    with open(epoch_file, 'w') as f:
        f.write(str(epoch))
    
    # Save best model separately
    if is_best:
        best_path = model_path.replace(".eqx", "_best.eqx")
        eqx.tree_serialise_leaves(best_path, model)
        print(f"  ✓ Best model saved to {best_path}")


def main():
    """Main training loop."""
    # Hyperparameters
    batch_size = 32  # Start with smaller batch for quick test
    num_epochs = 3  # Quick test first
    learning_rate = 1e-3
    in_channels = 1  # Grayscale MNIST
    hidden_channels = 16
    spatial_size = (28, 28)  # Keep original spatial size
    num_classes = 10
    num_steps = 30  # Number of solver steps (BPTT depth)
    solver_method = "naive"  # "naive", "anderson", or "multigrid"
    
    # Model checkpoint paths
    model_path = "trained_model_v1_nano.eqx"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize or load model
    key = jax.random.PRNGKey(42)
    model, start_epoch = load_model_checkpoint(
        model_path, in_channels, hidden_channels, spatial_size, num_classes, key
    )
    
    # Calculate and print parameter count
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(model) if isinstance(x, jnp.ndarray))
    print(f"\n{'='*60}")
    print(f"V1 'Nano' Model Parameter Count: {total_params:,}")
    print(f"{'='*60}")
    
    if total_params > 50_000:
        print(f"⚠️  WARNING: Parameter count ({total_params:,}) exceeds 50k target!")
    elif total_params > 20_000:
        print(f"⚠️  WARNING: Parameter count ({total_params:,}) exceeds 20k goal!")
    else:
        print(f"✅ SUCCESS: Parameter count ({total_params:,}) is under 20k goal!")
    print(f"{'='*60}\n")
    
    # Initialize optimizer
    # Create a dummy gradient structure to initialize optimizer
    def dummy_loss(m):
        return 0.0
    
    _, dummy_grads = eqx.filter_value_and_grad(dummy_loss)(model)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(dummy_grads)
    
    # Load training data
    print("Loading MNIST dataset...")
    print("(This may take a moment on first run - downloading data)")
    train_data = list(load_mnist_numpy(batch_size=batch_size, split="train"))
    print(f"Loaded {len(train_data)} training batches")
    test_data = list(load_mnist_numpy(batch_size=batch_size, split="test"))
    print(f"Loaded {len(test_data)} test batches")
    
    # Limit batches for quick test
    train_data = train_data[:10]  # Only first 10 batches for quick test
    test_data = test_data[:5]  # Only first 5 batches for quick test
    
    print(f"Using {len(train_data)} training batches and {len(test_data)} test batches")
    print("\nStarting training...\n")
    
    # Track best test accuracy
    best_test_accuracy = 0.0
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_convergence = 0.0
        num_batches = 0
        
        # Train on batches
        for batch_idx, (images, labels) in enumerate(train_data):
            # Convert to JAX arrays
            x = jnp.array(images)
            y = jnp.array(labels)
            
            # Training step
            model, opt_state, metrics = train_step(model, opt_state, x, y, optimizer)
            
            # Accumulate metrics
            epoch_loss += float(metrics["loss"])
            epoch_accuracy += float(metrics["accuracy"])
            epoch_convergence += float(metrics["convergence_delta"])
            num_batches += 1
            
            # Print progress every batch for quick test
            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_data)}, "
                    f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                    f"Conv: {metrics['convergence_delta']:.6f}"
                )
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_convergence = epoch_convergence / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Accuracy: {avg_accuracy:.4f}")
        print(f"  Convergence Delta: {avg_convergence:.6f}")
        
        # Evaluate on test set
        test_loss = 0.0
        test_accuracy = 0.0
        test_batches = 0
        
        for images, labels in test_data:  # Evaluate on all test batches
            x = jnp.array(images)
            y = jnp.array(labels)
            
            loss, metrics = loss_fn(model, x, y)
            test_loss += float(metrics["loss"])
            test_accuracy += float(metrics["accuracy"])
            test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        avg_test_accuracy = test_accuracy / test_batches
        
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Test Accuracy: {avg_test_accuracy:.4f}")
        
        # Check if this is the best model
        is_best = avg_test_accuracy > best_test_accuracy
        if is_best:
            best_test_accuracy = avg_test_accuracy
            print(f"  🎯 New best test accuracy: {best_test_accuracy:.4f}")
        
        # Save checkpoint after each epoch
        print(f"\nSaving checkpoint (epoch {epoch+1})...")
        save_model_checkpoint(model, epoch + 1, model_path, is_best=is_best)
        
        # Also save epoch-specific checkpoint
        epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.eqx")
        eqx.tree_serialise_leaves(epoch_checkpoint, model)
        print(f"  ✓ Epoch checkpoint saved to {epoch_checkpoint}")
        
        print("-" * 60)
    
    # Final save
    print(f"\nFinal model saved to {model_path}")
    print(f"Best model (accuracy {best_test_accuracy:.4f}) saved to {model_path.replace('.eqx', '_best.eqx')}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

