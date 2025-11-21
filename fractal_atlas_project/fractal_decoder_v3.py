# fractal_decoder_v3.py
# FractalDecoder V3 — стабильная и согласованная версия

import math
from typing import Optional

import torch
import torch.nn as nn


class FractalDecoder(nn.Module):
    """
    FINAL V3:
    координата x + (опционально) latent z -> скрытое h -> T фрактальных шагов -> логиты
    """

    def __init__(
        self,
        vocab_size: int,
        fourier_frequencies: int = 32,
        hidden_dim: int = 512,
        num_fractal_steps: int = 48,
        alpha: float = 0.5,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.K = fourier_frequencies
        self.hidden_dim = hidden_dim
        self.num_fractal_steps = num_fractal_steps
        self.alpha = alpha
        self.latent_dim = latent_dim

        # вход: [x] + sin/cos + z?
        in_dim = 1 + 2 * self.K
        if latent_dim is not None:
            in_dim += latent_dim

        # MLP: in → h0
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # fractal block f(h)
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    # Fourier features
    def _fourier_features(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        k = torch.arange(1, self.K + 1, device=x.device, dtype=x.dtype)
        arg = 2 * math.pi * x * k

        sinf = torch.sin(arg)
        cosf = torch.cos(arg)

        return torch.cat([x, sinf, cosf], dim=-1)

    # forward pass
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        feats = self._fourier_features(x)

        if z is not None:
            feats = torch.cat([feats, z], dim=-1)

        h = self.mlp_in(feats)

        for _ in range(self.num_fractal_steps):
            h_update = torch.tanh(self.fractal_block(h))
            h = (1 - self.alpha) * h + self.alpha * h_update

        return self.linear_out(h)

    def forward_argmax(self, x, z=None):
        return torch.argmax(self.forward(x, z), dim=-1)