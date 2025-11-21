# fractal_decoder_v7.py
# FractalDecoder V7 FINAL:
#   x ∈ [0, 1] + latent key z  → logits по символам
#
# Идеи внутри:
#   • High-Freq Fourier: частоты до очень больших значений (экспоненциальная сетка)
#   • HashGrid 1D (Instant-NGP style): хэшированная решётка по координате
#   • Weight-sharing: один фрактальный блок, применяемый T раз
#   • Sparse updates: разреженные обновления h (dropout по h_update)
#
# Квантование весов (8/4 bit) можно делать отдельным шагом при экспорте.

import math
from typing import Optional

import torch
import torch.nn as nn


# ----------------------- HashGrid 1D ----------------------- #


class HashGrid1D(nn.Module):
    """
    Упрощённый 1D HashGrid encoder в духе Instant-NGP.

    Идея:
      Для каждой координаты x ∈ [0,1]:
        • на каждом уровне l имеем свою "решётку" разрешения R_l
        • берём два соседних узла (i0, i1), линейно интерполируем их эмбеддинги
        • индексы узлов хэшируем в таблицу фиксированного размера

    Параметры:
        n_levels      – число уровней решётки
        n_features    – число фич на уровне
        log2_hash_size – log2 размера хэш-таблицы (например 14 → таблица 16384)
        base_resolution – начальное разрешение R_0
        per_level_scale – во сколько раз растёт разрешение на каждом уровне
    """

    def __init__(
        self,
        n_levels: int = 12,
        n_features: int = 2,
        log2_hash_size: int = 14,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features = n_features
        self.hash_size = 1 << log2_hash_size  # 2^log2_hash_size

        # Разрешение по уровням: R_l = base_resolution * per_level_scale^l
        resolutions = []
        cur_res = float(base_resolution)
        for _ in range(n_levels):
            resolutions.append(int(round(cur_res)))
            cur_res *= per_level_scale
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.int32))

        # Одна embedding-таблица на все уровни (как в Instant-NGP)
        self.embeddings = nn.Embedding(self.hash_size, n_levels * n_features)
        nn.init.uniform_(self.embeddings.weight, -1e-4, 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] или [B,1] в диапазоне [0, 1]
        return: [B, n_levels * n_features]
        """
        if x.dim() == 2 and x.size(-1) == 1:
            x = x[..., 0]  # [B,1] → [B]
        elif x.dim() != 1:
            raise ValueError("HashGrid1D ожидает x формы [B] или [B,1]")

        B = x.size(0)
        x = x.clamp(0.0, 1.0)

        feats_per_level = []
        for lvl in range(self.n_levels):
            res = int(self.resolutions[lvl].item())

            # Координата в "ячейках" уровня
            pos = x * res  # [B]
            i0 = torch.floor(pos).to(torch.int64)  # [B]
            i1 = i0 + 1
            w = (pos - i0.to(pos.dtype)).unsqueeze(-1)  # [B,1]

            # Хэшируем индексы (1D случай очень простой)
            h0 = (i0 % self.hash_size).to(torch.int64)
            h1 = (i1 % self.hash_size).to(torch.int64)

            # Эмбеддинги обоих узлов
            emb_all = self.embeddings.weight.view(
                self.hash_size, self.n_levels, self.n_features
            )
            e0 = emb_all[h0, lvl]  # [B, n_features]
            e1 = emb_all[h1, lvl]  # [B, n_features]

            # Линейная интерполяция
            emb = (1.0 - w) * e0 + w * e1  # [B, n_features]
            feats_per_level.append(emb)

        feats = torch.cat(feats_per_level, dim=-1)  # [B, n_levels * n_features]
        return feats


# -------------------- FractalDecoder V7 -------------------- #


class FractalDecoderV7(nn.Module):
    """
    FractalDecoder V7.

    Параметры:
        vocab_size          – размер словаря
        fourier_frequencies – число Fourier-частот K
        fourier_base        – базовая частота
        fourier_growth      – множитель частоты между уровнями
        hash_levels         – число уровней HashGrid
        hash_features       – число фич на уровень в HashGrid
        hash_log2_size      – log2 размера хэш-таблицы
        hash_base_res       – начальное разрешение HashGrid
        hash_per_level_scale – рост разрешения по уровням
        hidden_dim          – размер скрытого состояния h
        num_steps           – число фрактальных шагов T
        alpha               – коэффициент смешивания (0 < alpha <= 1)
        latent_dim          – размер латентного ключа z (0 → не используется)
        sparse_updates      – включать разреженные обновления
        sparse_p            – вероятность зануления обновления (0.9 → 90% шагов статичны)
    """

    def __init__(
        self,
        vocab_size: int,
        fourier_frequencies: int = 32,
        fourier_base: float = 1.0,
        fourier_growth: float = 3.5,
        hash_levels: int = 12,
        hash_features: int = 2,
        hash_log2_size: int = 14,
        hash_base_res: int = 16,
        hash_per_level_scale: float = 2.0,
        hidden_dim: int = 256,
        num_steps: int = 48,
        alpha: float = 0.5,
        latent_dim: int = 32,
        sparse_updates: bool = True,
        sparse_p: float = 0.9,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.K = fourier_frequencies
        self.fourier_base = fourier_base
        self.fourier_growth = fourier_growth

        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.sparse_updates = sparse_updates
        self.sparse_p = sparse_p

        # --- частоты для High-Freq Fourier ---
        ks = torch.arange(self.K, dtype=torch.float32)
        freqs = self.fourier_base * (self.fourier_growth ** ks)  # [K]
        self.register_buffer("fourier_freqs", freqs, persistent=False)

        # --- HashGrid encoder ---
        self.hashgrid = HashGrid1D(
            n_levels=hash_levels,
            n_features=hash_features,
            log2_hash_size=hash_log2_size,
            base_resolution=hash_base_res,
            per_level_scale=hash_per_level_scale,
        )
        self.hash_feat_dim = hash_levels * hash_features

        # Размер входа: x + Fourier + Hash + (опционально) z
        in_dim = 1 + 2 * self.K + self.hash_feat_dim
        if self.latent_dim > 0:
            in_dim += self.latent_dim

        # MLP для инициализации скрытого состояния h0
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Фрактальный блок (один, T раз)
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Dropout для разреженных обновлений
        self.update_dropout = nn.Dropout(p=self.sparse_p)

        # Выходной слой
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    # ------------- вспомогательные методы ------------- #

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] или [B,1] в диапазоне [0,1]
        -> [B, 1 + 2K]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] → [B,1]
        elif x.dim() != 2 or x.size(-1) != 1:
            raise ValueError("Ожидается x формы [B] или [B,1]")

        # x: [B,1]
        arg = 2.0 * math.pi * x * self.fourier_freqs  # [B, K]
        sin_feat = torch.sin(arg)
        cos_feat = torch.cos(arg)

        feats = torch.cat([x, sin_feat, cos_feat], dim=-1)  # [B, 1+2K]
        return feats

    # ----------------- основной forward ----------------- #

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B] или [B,1] – координаты
        z: [B, latent_dim] или [latent_dim] или None

        Если latent_dim > 0 и z is None → используется нулевой ключ (мастер-книга).
        """
        if x.dim() == 1:
            x_in = x.unsqueeze(-1)  # [B] → [B,1]
        elif x.dim() == 2 and x.size(-1) == 1:
            x_in = x
        else:
            raise ValueError("x должен быть [B] или [B,1]")

        B = x_in.size(0)

        # 1) Fourier + Hash
        fourier_feats = self._fourier_features(x_in)  # [B, 1+2K]
        hash_feats = self.hashgrid(x_in[..., 0])      # [B, hash_feat_dim]
        feats = torch.cat([fourier_feats, hash_feats], dim=-1)  # [B, 1+2K+hash]

        # 2) Латентный ключ
        if self.latent_dim > 0:
            if z is None:
                z_b = torch.zeros(
                    B,
                    self.latent_dim,
                    device=feats.device,
                    dtype=feats.dtype,
                )
            else:
                if z.dim() == 1:
                    z_b = z.unsqueeze(0).expand(B, -1)
                elif z.dim() == 2 and z.size(0) == 1:
                    z_b = z.expand(B, -1)
                elif z.dim() == 2 and z.size(0) == B:
                    z_b = z
                else:
                    raise ValueError("Неверная форма z")
            feats = torch.cat([feats, z_b], dim=-1)

        # 3) Инициализация h0
        h = self.mlp_in(feats)  # [B, hidden_dim]

        # 4) Фрактальное обновление
        for _ in range(self.num_steps):
            h_update = self.fractal_block(h)
            h_update = torch.tanh(h_update)

            if self.sparse_updates:
                h_update = self.update_dropout(h_update)

            h = (1.0 - self.alpha) * h + self.alpha * h_update

        # 5) Логиты
        logits = self.linear_out(h)  # [B, vocab_size]
        return logits

    def forward_argmax(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(x, z=z)
        return torch.argmax(logits, dim=-1)