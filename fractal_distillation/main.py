"""
Fractal Knowledge Distillation - Main Training Script
Distills Qwen2.5-0.5B into a tiny Fractal Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import re
import gc

from fractal_student import FractalStudent


# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")
print(f"   MPS available: {torch.backends.mps.is_available()}")


# ============================================================================
# DATA LOADING
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize
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


def load_training_data(tokenizer, num_samples: int = 1000, max_length: int = 128):
    """Load text from input.txt file and split into samples."""
    input_file = os.path.join(os.path.dirname(__file__), "input.txt")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}. Please create input.txt with training text.")
    
    print(f"📚 Loading text from {input_file}...")
    
    # Read the entire text file
    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read().strip()
    
    if len(full_text) == 0:
        raise ValueError("Input file is empty!")
    
    print(f"   Loaded {len(full_text)} characters from file")
    
    # Split text into sentences (simple approach: split by periods, exclamation, question marks)
    import re
    sentences = re.split(r'[.!?]+\s+', full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]  # Filter very short sentences
    
    if len(sentences) == 0:
        # If no sentences found, split by newlines or just use the whole text
        sentences = [full_text]
    
    # Create samples by taking sentences (or chunks if we need more samples)
    texts = []
    if len(sentences) >= num_samples:
        texts = sentences[:num_samples]
    else:
        # If we have fewer sentences than needed, repeat them or split the text into chunks
        # Split the full text into overlapping chunks
        chunk_size = len(full_text) // num_samples
        for i in range(num_samples):
            start = i * chunk_size
            end = start + chunk_size + 50  # Add some overlap
            if end > len(full_text):
                end = len(full_text)
            chunk = full_text[start:end].strip()
            if len(chunk) > 10:  # Only add non-empty chunks
                texts.append(chunk)
    
    print(f"   Created {len(texts)} text samples")
    
    return TextDataset(texts, tokenizer, max_length=max_length)


# ============================================================================
# TEACHER MODEL (Qwen)
# ============================================================================

def load_teacher_model():
    """Load Qwen2.5-0.5B as frozen teacher."""
    print("\n📖 Loading Teacher Model (Qwen2.5-0.5B)...")
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in float16 to save memory
    print("   Loading model in float16...")
    # Use low_cpu_mem_usage to reduce RAM usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device.type != "mps" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Reduce RAM usage during loading
    )
    
    # Move to device if not using device_map
    if device.type == "mps":
        model = model.to(device)
    
    # Freeze teacher
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"   ✅ Teacher loaded on {next(model.parameters()).device}")
    print(f"   📊 Teacher parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   📚 Teacher vocab_size: {model.config.vocab_size}")
    
    return model, tokenizer


# ============================================================================
# STUDENT MODEL (Fractal)
# ============================================================================

def create_student_model(vocab_size: int):
    """Create fractal student network."""
    print("\n🎓 Creating Fractal Student...")
    
    student = FractalStudent(
        vocab_size=vocab_size,
        hidden_dim=128,  # Reduced: 256->128 for memory
        num_fractal_steps=5,  # Reduced: 10->5 for speed
        alpha=0.5,
        max_seq_len=64,  # Reduced: 128->64
        pad_token_id=0
    ).to(device)
    
    num_params = student.get_num_parameters()
    print(f"   ✅ Student created on {device}")
    print(f"   📊 Student parameters: {num_params:,}")
    print(f"   💾 Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return student


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    num_epochs: int = 10,
    temperature: float = 2.0,
    learning_rate: float = 1e-4
):
    """Train student via knowledge distillation."""
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    criterion_ce = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    student.train()
    
    print("\n🚀 Starting Training...")
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
            input_ids = batch["input_ids"].to(device)  # [B, Seq]
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass through teacher (with memory optimization)
            with torch.no_grad():
                # Clear cache before teacher forward
                if device.type == "mps":
                    torch.mps.empty_cache()
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits  # [B, Seq, Vocab]
                # Detach to free memory
                teacher_logits = teacher_logits.detach()
            
            # Forward pass through student
            student_logits = student(input_ids)  # [B, Seq, Vocab]
            
            # Shift for next-token prediction
            # Teacher: predict next token from previous tokens
            shift_logits = student_logits[..., :-1, :].contiguous()  # [B, Seq-1, Vocab]
            shift_labels = input_ids[..., 1:].contiguous()  # [B, Seq-1]
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()  # [B, Seq-1, Vocab]
            
            # Reshape for loss calculation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # [B*(Seq-1), Vocab]
            shift_labels = shift_labels.view(-1)  # [B*(Seq-1)]
            shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))  # [B*(Seq-1), Vocab]
            
            # Knowledge Distillation Loss (KL Divergence)
            student_probs = F.log_softmax(shift_logits / temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
            kl_loss = criterion_kl(student_probs, teacher_probs)
            
            # Optional: Add cross-entropy loss for ground truth
            ce_loss = criterion_ce(shift_logits, shift_labels)
            
            # Combined loss
            loss = kl_loss + 0.1 * ce_loss  # Weight KL more heavily
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute gradient norm (before cleanup)
            total_norm = 0.0
            for p in student.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Logging (before cleanup)
            loss_val = loss.item()
            kl_val = kl_loss.item()
            ce_val = ce_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "kl": f"{kl_val:.4f}",
                "grad_norm": f"{total_norm:.3f}"
            })
            
            # Print detailed stats every 10 steps
            if global_step % 10 == 0:
                print(f"\nStep {global_step}:")
                print(f"  Loss: {loss_val:.4f} (KL: {kl_val:.4f}, CE: {ce_val:.4f})")
                print(f"  Grad Norm: {total_norm:.3f}")
            
            # Clear cache after each step (MPS memory management)
            if device.type == "mps":
                torch.mps.empty_cache()
            
            # Explicit cleanup
            del teacher_logits, student_logits, shift_logits, shift_teacher_logits, shift_labels
            del student_probs, teacher_probs, kl_loss, ce_loss, loss
            gc.collect()
            
            # Logging
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"\n✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        print("-" * 60)
    
    return student


# ============================================================================
# GENERATION FUNCTION
# ============================================================================

def generate_text(
    student: nn.Module,
    tokenizer,
    prompt: str = "Once upon a time",
    max_length: int = 50,
    temperature: float = 0.8
):
    """Generate text using the fractal student."""
    student.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = student(generated)  # [1, Seq, Vocab]
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature  # [Vocab]
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("FRACTAL KNOWLEDGE DISTILLATION (M3 EDITION)")
    print("=" * 60)
    
    # Load teacher
    teacher, tokenizer = load_teacher_model()
    # Get vocab size from tokenizer or model config
    try:
        vocab_size = len(tokenizer.get_vocab())
    except:
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    
    # Also get from model config to be sure
    if hasattr(teacher.config, 'vocab_size'):
        vocab_size = teacher.config.vocab_size
    
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create student with SAME vocab_size as teacher
    student = create_student_model(vocab_size=vocab_size)
    
    # Load data (500 samples for more training data)
    print("\n📦 Loading training data...")
    dataset = load_training_data(tokenizer, num_samples=500, max_length=64)  # 500 samples for more data
    # Disable multiprocessing to avoid hangs (num_workers=0)
    # Start with batch_size=2, can reduce to 1 if memory issues
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)  # 500/2 = 250 batches
    print(f"   Batches: {len(dataloader)} ({len(dataloader)} batches per epoch)")
    
    # Train (30 epochs for better learning)
    trained_student = train_distillation(
        teacher=teacher,
        student=student,
        dataloader=dataloader,
        tokenizer=tokenizer,
        num_epochs=30,  # 30 epochs for better convergence
        temperature=2.0,
        learning_rate=1e-4
    )
    
    # Save model
    save_path = "fractal_student_distilled.pt"
    torch.save({
        "model_state_dict": trained_student.state_dict(),
        "vocab_size": vocab_size,
        "hidden_dim": trained_student.hidden_dim,
    }, save_path)
    print(f"\n💾 Model saved to: {save_path}")
    
    # Test generation
    print("\n🎨 Testing Generation...")
    print("-" * 60)
    
    test_prompts = [
        "Once upon a time",
        "The king said",
        "In the beginning",
    ]
    
    for prompt in test_prompts:
        generated = generate_text(trained_student, tokenizer, prompt, max_length=30)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

