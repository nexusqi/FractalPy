# fractal_decoder_v8.py
# FractalDecoder V8: Fourier + HashGrid (Instant-NGP style) + latent key z
#
# Идеи:
#  - Логарифмическая сетка частот до очень высоких (≈ 1e4): "микроскоп" для резких переходов.
#  - Hash Encoding: многослойная хеш-сетка координат -> плотный вектор признаков.
#  - Один и тот же fractal_block применяется num_steps раз (weight sharing).
#  - Разреженные обновления (sparse_updates) через dropout по h_update.

import math
from typing import Optional

import torch
import torch.nn as nn


class FractalDecoderV8(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        # Fourier
        fourier_frequencies: int = 48,
        max_fourier_freq: float = 1e4,
        # Фрактальное ядро
        hidden_dim: int = 256,
        num_steps: int = 64,
        alpha: float = 0.5,
        # Латентный ключ
        latent_dim: int = 32,
        # Sparse updates
        sparse_updates: bool = True,
        sparse_p: float = 0.9,
        # HashGrid параметры
        hash_levels: int = 12,
        hash_dim: int = 4,
        hash_base_resolution: float = 16.0,
        hash_growth: float = 1.5,
        hash_table_size: int = 2 ** 14,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # ---- Fourier ----
        self.K = fourier_frequencies
        self.max_fourier_freq = max_fourier_freq

        # Логарифмическая сетка частот: от 1 до max_fourier_freq
        # log10(1) = 0, log10(max_fourier_freq) -> верхняя граница
        log_max = math.log10(self.max_fourier_freq)
        freqs = torch.logspace(
            start=0.0,
            end=log_max,
            steps=self.K,
            base=10.0,
            dtype=torch.float32,
        )
        self.register_buffer("fourier_freqs", freqs, persistent=False)

        # ---- HashGrid ----
        self.hash_levels = hash_levels
        self.hash_dim = hash_dim
        self.hash_table_size = hash_table_size
        self.hash_base_resolution = hash_base_resolution
        self.hash_growth = hash_growth

        # Разрешения решётки по уровням
        level_ids = torch.arange(self.hash_levels, dtype=torch.float32)
        resolutions = self.hash_base_resolution * (self.hash_growth ** level_ids)
        self.register_buffer("hash_resolutions", resolutions, persistent=False)

        # Одна и та же таблица размера hash_table_size для всех уровней
        # (как в Instant-NGP — хеширование в ограниченный буфер)
        self.hash_tables = nn.ModuleList(
            [
                nn.Embedding(self.hash_table_size, self.hash_dim)
                for _ in range(self.hash_levels)
            ]
        )

        # ---- Прочие параметры ----
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.sparse_updates = sparse_updates
        self.update_dropout = nn.Dropout(p=sparse_p)

        # ---- Размер входа: Fourier + HashGrid (+ latent) ----
        fourier_dim = 1 + 2 * self.K          # x + sin + cos
        hash_dim_total = self.hash_levels * self.hash_dim
        in_dim = fourier_dim + hash_dim_total
        if self.latent_dim > 0:
            in_dim += self.latent_dim

        # MLP для инициализации скрытого состояния
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Фрактальный блок f(h) — общий для всех шагов
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Выходной слой
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    # Fourier features
    # ------------------------------------------------------------------
    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] или [B,1] в диапазоне [0,1]
        -> [B, 1 + 2K]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B,1]

        # x: [B,1], freqs: [K]
        # 2π f_k x -> [B,K]
        arg = 2.0 * math.pi * x * self.fourier_freqs  # broadcast

        sin_feat = torch.sin(arg)  # [B,K]
        cos_feat = torch.cos(arg)  # [B,K]

        feats = torch.cat([x, sin_feat, cos_feat], dim=-1)
        return feats  # [B, 1+2K]

    # ------------------------------------------------------------------
    # HashGrid encoding (1D Instant-NGP style)
    # ------------------------------------------------------------------
    def _hash_1d(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Простейшая 32-битная хеш-функция для 1D индексов.
        idx: [B,1] (long)
        -> [B,1] (long в [0, hash_table_size))
        """
        # умножаем на "магическое" число и берём по модулю размера таблицы
        return (idx * 2654435761) % self.hash_table_size

    def _hashgrid_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] или [B,1] в [0,1]
        -> [B, hash_levels * hash_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B,1]

        B = x.size(0)
        encodings = []

        for level in range(self.hash_levels):
            res = self.hash_resolutions[level]  # скаляр

            # Координата в решётке данного уровня
            pos = x * res  # [B,1]
            idx_left = torch.floor(pos).long()
            idx_right = idx_left + 1

            # Нормализуем индексы в диапазон [0, hash_table_size)
            h_left = self._hash_1d(idx_left)
            h_right = self._hash_1d(idx_right)

            # Достаём embeddings
            table = self.hash_tables[level]
            emb_left = table(h_left.view(B))   # [B, hash_dim]
            emb_right = table(h_right.view(B))

            # Линейная интерполяция
            t = (pos - idx_left.float()).clamp(0.0, 1.0)  # [B,1]
            enc = (1.0 - t) * emb_left + t * emb_right    # broadcast по последней оси

            encodings.append(enc)

        # Конкатенация по уровням
        return torch.cat(encodings, dim=-1)  # [B, hash_levels * hash_dim]

    # ------------------------------------------------------------------
    # Основной forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B] или [B,1]
        z: [B, latent_dim] или None
        """
        # Fourier + HashGrid
        fourier_feats = self._fourier_features(x)      # [B, 1+2K]
        hash_feats = self._hashgrid_encoding(x)        # [B, L*D]
        feats = torch.cat([fourier_feats, hash_feats], dim=-1)  # [B, ...]

        B = feats.size(0)

        # Латентный ключ
        if self.latent_dim > 0:
            if z is None:
                z = torch.zeros(
                    B,
                    self.latent_dim,
                    device=feats.device,
                    dtype=feats.dtype,
                )
            elif z.dim() == 1:
                z = z.unsqueeze(0).expand(B, -1)
            elif z.dim() == 2 and z.size(0) == 1:
                z = z.expand(B, -1)

            feats = torch.cat([feats, z], dim=-1)

        # Инициализация скрытого состояния
        h = self.mlp_in(feats)  # [B, hidden_dim]

        # Фрактальные шаги
        for _ in range(self.num_steps):
            h_update = self.fractal_block(h)
            h_update = torch.tanh(h_update)

            if self.sparse_updates:
                h_update = self.update_dropout(h_update)

            h = (1.0 - self.alpha) * h + self.alpha * h_update

        # Логиты
        logits = self.linear_out(h)  # [B, vocab_size]
        return logits

    def forward_argmax(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(x, z=z)
        return torch.argmax(logits, dim=-1)