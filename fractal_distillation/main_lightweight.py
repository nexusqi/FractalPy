"""
LIGHTWEIGHT VERSION - Optimized for 8GB RAM
Minimal test to prove the concept works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import gc

from fractal_student import FractalStudent

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

# Force garbage collection
gc.collect()
if device.type == "mps":
    torch.mps.empty_cache()

# ============================================================================
# DATA LOADING (MINIMAL)
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset."""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 32):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


def load_minimal_data(tokenizer, num_samples: int = 50, max_length: int = 32):
    """Load minimal dummy data."""
    texts = [
        "Once upon a time, there was a king.",
        "The sun rose in the east.",
        "In the forest, secrets lay hidden.",
        "A wise philosopher said: knowledge is treasure.",
        "The stars twinkled in the night sky.",
        "A long time ago in a galaxy far away.",
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
    ] * (num_samples // 10 + 1)
    texts = texts[:num_samples]
    return TextDataset(texts, tokenizer, max_length=max_length)


# ============================================================================
# TEACHER MODEL (Qwen) - Memory Optimized
# ============================================================================

def load_teacher_model():
    """Load Qwen with memory optimizations."""
    print("\n📖 Loading Teacher Model (Qwen2.5-0.5B)...")
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with maximum memory optimization
    print("   Loading model in float16 with low memory...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,  # Manual control
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Move to device
    if device.type == "mps":
        model = model.to(device)
        torch.mps.empty_cache()
    
    # Freeze teacher
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"   ✅ Teacher loaded on {next(model.parameters()).device}")
    print(f"   📊 Teacher parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   📚 Teacher vocab_size: {model.config.vocab_size}")
    
    return model, tokenizer


# ============================================================================
# STUDENT MODEL (Tiny Fractal)
# ============================================================================

def create_student_model(vocab_size: int):
    """Create minimal fractal student."""
    print("\n🎓 Creating Tiny Fractal Student...")
    
    student = FractalStudent(
        vocab_size=vocab_size,
        hidden_dim=64,  # MINIMAL: 64 instead of 256
        num_fractal_steps=3,  # MINIMAL: 3 instead of 10
        alpha=0.5,
        max_seq_len=32,  # MINIMAL: 32 instead of 128
        pad_token_id=0
    ).to(device)
    
    num_params = student.get_num_parameters()
    print(f"   ✅ Student created on {device}")
    print(f"   📊 Student parameters: {num_params:,}")
    print(f"   💾 Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if device.type == "mps":
        torch.mps.empty_cache()
    
    return student


# ============================================================================
# TRAINING LOOP (Memory Optimized)
# ============================================================================

def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    num_epochs: int = 1,  # MINIMAL: 1 epoch
    temperature: float = 2.0,
    learning_rate: float = 1e-4
):
    """Train with memory optimizations."""
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    criterion_ce = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    student.train()
    
    print("\n🚀 Starting Training (LIGHTWEIGHT MODE)...")
    print(f"   Temperature: {temperature}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}")
    print("-" * 60)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Teacher forward (with memory clearing)
            with torch.no_grad():
                if device.type == "mps":
                    torch.mps.empty_cache()
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits.detach()
            
            # Student forward
            student_logits = student(input_ids)
            
            # Shift for next-token prediction
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            
            # Reshape
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))
            
            # Loss
            student_probs = F.log_softmax(shift_logits / temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
            kl_loss = criterion_kl(student_probs, teacher_probs)
            ce_loss = criterion_ce(shift_logits, shift_labels)
            loss = kl_loss + 0.1 * ce_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Memory cleanup
            if device.type == "mps":
                torch.mps.empty_cache()
            del teacher_logits, student_logits, shift_logits, shift_teacher_logits
            gc.collect()
            
            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}"
            })
            
            if global_step % 5 == 0:
                print(f"\nStep {global_step}: Loss={loss.item():.4f}, KL={kl_loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"\n✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        print("-" * 60)
    
    return student


# ============================================================================
# GENERATION
# ============================================================================

def generate_text(student: nn.Module, tokenizer, prompt: str = "Once upon", max_length: int = 20):
    """Generate text."""
    student.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = student(generated)
            next_token_logits = logits[0, -1, :] / 0.8
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("FRACTAL KNOWLEDGE DISTILLATION (LIGHTWEIGHT - 8GB RAM)")
    print("=" * 60)
    
    # Load teacher
    teacher, tokenizer = load_teacher_model()
    vocab_size = teacher.config.vocab_size
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create tiny student
    student = create_student_model(vocab_size=vocab_size)
    
    # Load minimal data
    print("\n📦 Loading minimal training data...")
    dataset = load_minimal_data(tokenizer, num_samples=50, max_length=32)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # batch_size=1 for minimal memory
    print(f"   Batches: {len(dataloader)}")
    
    # Train (1 epoch, minimal)
    trained_student = train_distillation(
        teacher=teacher,
        student=student,
        dataloader=dataloader,
        tokenizer=tokenizer,
        num_epochs=1,  # Just 1 epoch for proof of concept
        temperature=2.0,
        learning_rate=1e-4
    )
    
    # Save
    save_path = "fractal_student_lightweight.pt"
    torch.save({
        "model_state_dict": trained_student.state_dict(),
        "vocab_size": vocab_size,
        "hidden_dim": trained_student.hidden_dim,
    }, save_path)
    print(f"\n💾 Model saved to: {save_path}")
    
    # Test generation
    print("\n🎨 Testing Generation...")
    print("-" * 60)
    
    test_prompts = ["Once upon", "The king"]
    for prompt in test_prompts:
        generated = generate_text(trained_student, tokenizer, prompt, max_length=15)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    print("=" * 60)
    print("✅ Proof of Concept Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

