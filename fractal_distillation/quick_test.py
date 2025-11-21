"""
Quick test script to verify installation and basic functionality.
"""

import torch
from fractal_student import FractalStudent

def test_fractal_student():
    """Test that FractalStudent can be created and run forward pass."""
    print("🧪 Testing FractalStudent...")
    
    # Create model
    vocab_size = 1000
    student = FractalStudent(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_fractal_steps=5,  # Fewer steps for quick test
        alpha=0.5
    )
    
    print(f"   ✅ Model created")
    print(f"   📊 Parameters: {student.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"   🔄 Running forward pass...")
    with torch.no_grad():
        logits = student(input_ids)
    
    print(f"   ✅ Forward pass successful")
    print(f"   📐 Output shape: {logits.shape} (expected: [{batch_size}, {seq_len}, {vocab_size}])")
    
    assert logits.shape == (batch_size, seq_len, vocab_size), "Wrong output shape!"
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_fractal_student()

