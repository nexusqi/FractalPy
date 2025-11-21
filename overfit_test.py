"""
Overfit Experiment: Test if 2.6k parameter model can memorize 64 images.

This script trains on a SINGLE batch for 100 epochs to check if the model
can achieve Loss -> 0 (perfect memorization).
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import equinox as eqx
import time

from src.model import FractalFieldClassifier


def loss_fn(model, x, y):
    """Compute loss and metrics."""
    logits, convergence_history = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == y).mean()
    convergence_delta = jnp.mean(convergence_history[-5:])
    return loss, {"loss": loss, "accuracy": accuracy, "convergence_delta": convergence_delta}


def train_step(model, opt_state, x, y, optimizer):
    """Perform one training step with gradient monitoring."""
    def loss_fn_wrapper(m):
        return loss_fn(m, x, y)
    
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn_wrapper, has_aux=True)(model)
    
    # Calculate gradient norm for monitoring
    # Handle both real and complex gradients
    grad_norm = 0.0
    for g in jax.tree_util.tree_leaves(grads):
        if isinstance(g, jnp.ndarray):
            # For complex numbers, take magnitude squared
            if jnp.iscomplexobj(g):
                grad_norm += jnp.sum(jnp.abs(g) ** 2)
            else:
                grad_norm += jnp.sum(g * g)
    grad_norm = jnp.sqrt(grad_norm)
    metrics["grad_norm"] = grad_norm
    
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, metrics


def load_single_batch(batch_size=64):
    """Load only the first batch from MNIST training set."""
    import gzip
    import os
    import urllib.request
    
    data_dir = "mnist_data"
    images_file = "train-images-idx3-ubyte.gz"
    labels_file = "train-labels-idx1-ubyte.gz"
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    images_path = os.path.join(data_dir, images_file)
    labels_path = os.path.join(data_dir, labels_file)
    
    # Download if needed
    if not os.path.exists(images_path):
        print(f"Downloading {images_file}...")
        urllib.request.urlretrieve(base_url + images_file, images_path)
    if not os.path.exists(labels_path):
        print(f"Downloading {labels_file}...")
        urllib.request.urlretrieve(base_url + labels_file, labels_path)
    
    # Load first batch_size images
    with gzip.open(images_path, 'rb') as f:
        f.read(16)  # Skip header
        buf = f.read(28 * 28 * batch_size)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(batch_size, 28, 28) / 255.0
    
    with gzip.open(labels_path, 'rb') as f:
        f.read(8)  # Skip header
        buf = f.read(batch_size)
        labels = np.frombuffer(buf, dtype=np.uint8)
    
    # Reshape to NCHW: [B, 28, 28] -> [B, 1, 28, 28]
    images = images.reshape(batch_size, 1, 28, 28)
    
    return images, labels


def main():
    """Overfit experiment on single batch."""
    print("=" * 70)
    print("OVERFIT EXPERIMENT: Can 2.6k parameter model memorize 64 images?")
    print("=" * 70)
    
    # Hyperparameters - STABILIZED to reduce oscillation
    batch_size = 64
    num_epochs = 300  # Reduced for focused test
    learning_rate = 0.002  # Reduced from 0.005 - need precision, not speed
    print_every = 50  # Print every 50 epochs
    
    # Load ONLY the first batch
    print(f"\n1. Loading first batch ({batch_size} images)...")
    images, labels = load_single_batch(batch_size)
    x = jnp.array(images)  # [64, 1, 28, 28]
    y = jnp.array(labels)  # [64]
    print(f"   ✅ Loaded batch: {x.shape}, labels: {y.shape}")
    
    # Initialize model
    print("\n2. Initializing V1 Nano model...")
    key = jax.random.PRNGKey(42)
    model = FractalFieldClassifier(
        in_channels=1,
        hidden_channels=16,
        spatial_size=(28, 28),
        num_classes=10,
        key=key,
        alpha_init=0.2,  # Reduced from 0.5 - slow down time flow to prevent overshooting
        activation="modrelu",
        solver_method="naive",
        num_steps=10,  # Reduced for faster overfit test
        use_spectral_norm=True
    )
    
    # Count parameters
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(model) if isinstance(x, jnp.ndarray))
    print(f"   ✅ Model parameters: {total_params:,}")
    
    # Initialize optimizer - Use AdamW for better weight decay
    def dummy_loss(m):
        return 0.0
    _, dummy_grads = eqx.filter_value_and_grad(dummy_loss)(model)
    optimizer = optax.adamw(learning_rate, weight_decay=1e-5)
    opt_state = optimizer.init(dummy_grads)
    
    print(f"\n3. Training on SINGLE batch for {num_epochs} epochs...")
    print(f"   Goal: Conv < 0.1 while Accuracy climbs to 90%!")
    print(f"   Strategy: Spectral Normalization + Reduced LR/Alpha for stability")
    print(f"   Printing every {print_every} epochs\n")
    print("-" * 70)
    
    start_time = time.time()
    
    # Training loop on single batch
    for epoch in range(num_epochs):
        # Train on the same batch
        model, opt_state, metrics = train_step(model, opt_state, x, y, optimizer)
        
        # Print every print_every epochs
        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - start_time
            grad_norm_val = metrics.get('grad_norm', jnp.array(0.0))
            grad_norm = float(jnp.real(grad_norm_val)) if jnp.iscomplexobj(grad_norm_val) else float(grad_norm_val)
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {metrics['loss']:.6f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"Grad Norm: {grad_norm:.6f} | "
                f"Conv: {metrics['convergence_delta']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # Warn about gradient issues
            if grad_norm < 1e-8:
                print(f"  ⚠️  WARNING: Gradients are VANISHING! (norm < 1e-8)")
            elif grad_norm > 100.0:
                print(f"  ⚠️  WARNING: Gradients are EXPLODING! (norm > 100)")
            elif grad_norm < 0.01:
                print(f"  ⚠️  WARNING: Gradients are very small (norm < 0.01)")
    
    total_time = time.time() - start_time
    
    # Final evaluation with gradient check
    print("-" * 70)
    print("\n4. Final Results:")
    final_loss, final_metrics = loss_fn(model, x, y)
    
    # Get final gradient norm
    def final_loss_wrapper(m):
        loss, _ = loss_fn(m, x, y)
        return loss
    _, final_grads = eqx.filter_value_and_grad(final_loss_wrapper)(model)
    final_grad_norm = 0.0
    for g in jax.tree_util.tree_leaves(final_grads):
        if isinstance(g, jnp.ndarray):
            # Handle complex gradients
            if jnp.iscomplexobj(g):
                final_grad_norm += jnp.sum(jnp.abs(g) ** 2)
            else:
                final_grad_norm += jnp.sum(g * g)
    final_grad_norm = jnp.sqrt(final_grad_norm)
    
    # Convert JAX arrays to Python floats using .item()
    final_loss_val = float(final_loss.item()) if hasattr(final_loss, 'item') else float(final_loss)
    final_acc_val = float(final_metrics['accuracy'].item()) if hasattr(final_metrics['accuracy'], 'item') else float(final_metrics['accuracy'])
    final_grad_norm_val = float(final_grad_norm.item()) if hasattr(final_grad_norm, 'item') else float(final_grad_norm)
    if jnp.iscomplexobj(final_grad_norm):
        final_grad_norm_val = float(jnp.real(final_grad_norm).item())
    
    print(f"   Final Loss: {final_loss_val:.6f}")
    print(f"   Final Accuracy: {final_acc_val:.4f} ({final_acc_val*100:.1f}%)")
    print(f"   Final Grad Norm: {final_grad_norm_val:.6f}")
    print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Verdict
    print("\n" + "=" * 70)
    if final_loss_val < 0.01:
        print("🎉🎉🎉 EPIC SUCCESS: Model CAN memorize! Loss -> 0")
        print("   The 2.6k parameter model has enough capacity!")
        print("   🍺🍺🍺 TIME TO CELEBRATE!!!")
    elif final_loss_val < 0.1:
        print("✅ EXCELLENT: Model is learning very well (Loss < 0.1)")
        print("   Model can memorize with more training or slight adjustments.")
    elif final_loss_val < 1.0:
        print("✅ GOOD: Model is learning (Loss < 1.0)")
        print("   May need more capacity or different hyperparameters.")
    elif final_acc_val >= 0.80:
        print("🎉 SUCCESS: Accuracy >= 80%! The 'Nano' brain is working!")
        print(f"   Loss: {final_loss_val:.4f}, Accuracy: {final_acc_val*100:.1f}%")
        print("   🍺 TIME TO CELEBRATE!!!")
    elif final_acc_val >= 0.50:
        print("✅ IMPROVING: Accuracy >= 50%! Model is learning well!")
        print(f"   Loss: {final_loss_val:.4f}, Accuracy: {final_acc_val*100:.1f}%")
        print("   Keep training - we're getting there!")
    elif final_loss_val < 2.0:
        print("⚠️  IMPROVING: Loss dropped below 2.0 (better than random)")
        print(f"   Loss: {final_loss_val:.4f}, Accuracy: {final_acc_val*100:.1f}%")
        print("   Model is learning, but slowly. Gradients are flowing.")
    else:
        print("❌ FAIL: Model cannot memorize (Loss > 2.0)")
        if final_grad_norm_val < 0.01:
            print("   ⚠️  Problem: VANISHING GRADIENTS (grad_norm < 0.01)")
        elif final_grad_norm_val > 100.0:
            print("   ⚠️  Problem: EXPLODING GRADIENTS (grad_norm > 100)")
        else:
            print("   Model may be too small or architecture needs adjustment.")
    print("=" * 70)


if __name__ == "__main__":
    main()

