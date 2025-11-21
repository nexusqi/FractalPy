"""
Fractal Student Network - PyTorch Implementation
Simulates complex-valued operations using split real/imaginary parts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """
    Linear layer that simulates complex number multiplication.
    
    Treats first half of hidden_dim as Real, second half as Imaginary.
    Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # We need 4 linear layers for complex multiplication
        # Real part: W_rr @ real + W_ri @ imag
        # Imag part: W_ir @ real + W_ii @ imag
        self.W_rr = nn.Linear(in_dim // 2, out_dim // 2, bias=False)
        self.W_ri = nn.Linear(in_dim // 2, out_dim // 2, bias=False)
        self.W_ir = nn.Linear(in_dim // 2, out_dim // 2, bias=False)
        self.W_ii = nn.Linear(in_dim // 2, out_dim // 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # Initialize with small weights for stability
        for layer in [self.W_rr, self.W_ri, self.W_ir, self.W_ii]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Seq, Hidden] where Hidden is even
        Split into real and imaginary parts
        """
        B, Seq, Hidden = x.shape
        assert Hidden % 2 == 0, "Hidden dimension must be even for complex operations"
        
        # Split into real and imaginary
        real = x[..., :Hidden//2]  # [B, Seq, Hidden//2]
        imag = x[..., Hidden//2:]   # [B, Seq, Hidden//2]
        
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        # But we're doing matrix multiplication, so:
        # Real output = W_rr @ real - W_ri @ imag
        # Imag output = W_ir @ real + W_ii @ imag
        real_out = self.W_rr(real) - self.W_ri(imag)
        imag_out = self.W_ir(real) + self.W_ii(imag)
        
        # Concatenate back
        output = torch.cat([real_out, imag_out], dim=-1)  # [B, Seq, Hidden]
        output = output + self.bias
        return output


class SplitLeakyReLU(nn.Module):
    """
    Apply LeakyReLU separately to Real and Imaginary parts.
    """
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Seq, Hidden]
        """
        return self.leaky_relu(x)


class FractalCore(nn.Module):
    """
    The fractal core that applies recursive transformations.
    Uses weight-sharing: same transformation applied T times.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_steps: int = 10,
        alpha: float = 0.5,
        clamp_range: tuple = (-3.0, 3.0)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.alpha = alpha
        self.clamp_range = clamp_range
        
        # Single fractal block (weight-sharing across iterations)
        self.fractal_block = nn.Sequential(
            ComplexLinear(hidden_dim, hidden_dim * 2),
            SplitLeakyReLU(),
            ComplexLinear(hidden_dim * 2, hidden_dim),
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply fractal transformation T times.
        
        Args:
            h: [B, Seq, Hidden] - initial hidden state
        
        Returns:
            h_final: [B, Seq, Hidden] - transformed state after T iterations
        """
        # Iterative fractal update
        for _ in range(self.num_steps):
            # Compute update
            h_update = self.fractal_block(h)
            h_update = torch.tanh(h_update)
            
            # Mix old and new: Z_new = (1-α)Z_old + α·σ(Layer(Z_old))
            h = (1.0 - self.alpha) * h + self.alpha * h_update
            
            # Normalize
            h = self.norm(h)
            
            # CRITICAL SAFETY: Clamp to prevent exploding gradients
            h = torch.clamp(h, self.clamp_range[0], self.clamp_range[1])
        
        return h


class FractalStudent(nn.Module):
    """
    Complete Fractal Student Network for Knowledge Distillation.
    
    Architecture:
    1. Embeddings: token -> hidden_dim
    2. Fractal Core: recursive complex-valued transformations
    3. Head: hidden -> vocab_size
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_fractal_steps: int = 10,
        alpha: float = 0.5,
        max_seq_len: int = 512,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        
        # Ensure hidden_dim is even for complex operations
        if hidden_dim % 2 != 0:
            hidden_dim += 1
            self.hidden_dim = hidden_dim
        
        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        
        # Fractal core
        self.fractal_core = FractalCore(
            hidden_dim=hidden_dim,
            num_steps=num_fractal_steps,
            alpha=alpha
        )
        
        # Output head
        self.head = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fractal network.
        
        Args:
            input_ids: [B, Seq] - token indices
        
        Returns:
            logits: [B, Seq, Vocab] - output logits
        """
        # Embed tokens
        h = self.embeddings(input_ids)  # [B, Seq, Hidden]
        
        # Apply fractal core
        h = self.fractal_core(h)  # [B, Seq, Hidden]
        
        # Project to vocabulary
        logits = self.head(h)  # [B, Seq, Vocab]
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

