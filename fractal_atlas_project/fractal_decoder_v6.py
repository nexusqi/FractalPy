# fractal_decoder_v6.py
# FractalDecoder V6: x + hash-encoding + high-freq Fourier + (опциональный) z -> логиты
#
# Добавлено к V5:
# 1) Очень высокие частоты в Fourier-фичах (до ~1e4), чтобы модель видела резкие переходы.
# 2) Hash Encoding (в стиле Instant-NGP) для координаты x (1D hash-grid с multi-resolution).
#
# Идея:
#   - Fourier даёт "гладкий" но высокочастотный сигнал.
#   - Hash-grid даёт дискретную, но очень expressive эмбеддинговую сетку.
#
# Здесь мы НЕ включаем sparse_updates по умолчанию (чтобы сначала добиться нормального обучения).
# Если нужно, их можно включить обратно через аргументы.

import math
from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------- #
# 1D Hash Grid для координаты x в [0, 1] (упрощённый Instant-NGP)
# ----------------------------------------------------------- #

class HashGrid1D(nn.Module):
    """
    Упрощённый 1D hash-grid из Instant-NGP.

    x ∈ [0, 1] -> concat эмбеддингов с нескольких уровней.
    На каждом уровне:
        - есть своя дискретизация (resolution R_l)
        - есть своя hash-таблица эмбеддингов размера hashmap_size
        - мы берём две соседние ячейки (floor/ceil) и линейно интерполируем эмбеддинги.

    Параметры:
        num_levels     — число уровней (L)
        min_res        — минимальное разрешение R_min
        max_res        — максимальное разрешение R_max
        embedding_dim  — размер эмбеддинга на уровень
        hashmap_size   — размер hash-таблицы на уровень
    """

    def __init__(
        self,
        num_levels: int = 8,
        min_res: int = 16,
        max_res: int = 1024,
        embedding_dim: int = 8,
        hashmap_size: int = 2**14,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.embedding_dim = embedding_dim
        self.hashmap_size = hashmap_size

        # Разрешения по уровням (геометрическая прогрессия)
        # R_l = R_min * (R_max / R_min)^(l / (L-1))
        if num_levels == 1:
            resolutions = torch.tensor([min_res], dtype=torch.int32)
        else:
            resolutions_f = torch.logspace(
                start=math.log10(float(min_res)),
                end=math.log10(float(max_res)),
                steps=num_levels,
            )
            resolutions = torch.round(resolutions_f).to(torch.int32)

        self.register_buffer("resolutions", resolutions, persistent=False)

        # Одна большая таблица эмбеддингов на уровень (hashmap)
        # Можно сделать отдельную на каждый уровень, здесь именно так:
        self.tables = nn.ModuleList([
            nn.Embedding(hashmap_size, embedding_dim) for _ in range(num_levels)
        ])

        # Инициализация эмбеддингов
        for emb in self.tables:
            nn.init.uniform_(emb.weight, a=-1e-4, b=1e-4)

    @staticmethod
    def _hash(indices: torch.Tensor, level: int, hashmap_size: int) -> torch.Tensor:
        """
        Простейший hash для 1D индекса + level.
        indices: [B]
        return: [B] с числами 0..hashmap_size-1
        """
        # Берём 32-битное целое, немного перемешиваем и берём mod.
        # Важно: всё делаем на целых, чтобы не было NaN.
        prime1 = 73856093
        prime2 = 19349663
        x = indices.int()
        h = x * prime1 ^ (level * prime2)
        h = h % hashmap_size
        return h.long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] или [B, 1], значения в диапазоне [0, 1].
        return: [B, num_levels * embedding_dim]
        """
        if x.dim() == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)  # [B,1] -> [B]
        elif x.dim() != 1:
            raise ValueError("HashGrid1D ожидает x формы [B] или [B,1]")

        B = x.size(0)
        x_clamped = x.clamp(0.0, 1.0)

        features_per_level = []

        for lvl in range(self.num_levels):
            R = int(self.resolutions[lvl].item())  # скаляр
            # Координата в "ячейках"
            t = x_clamped * (R - 1)       # [B]
            i0 = torch.floor(t).long()    # [B]
            i1 = torch.clamp(i0 + 1, max=R - 1)  # [B]
            w = (t - i0.float()).unsqueeze(-1)   # [B,1]

            # Hash индексы для двух соседних ячеек
            h0 = self._hash(i0, lvl, self.hashmap_size)  # [B]
            h1 = self._hash(i1, lvl, self.hashmap_size)  # [B]

            table = self.tables[lvl]  # nn.Embedding
            e0 = table(h0)  # [B, D]
            e1 = table(h1)  # [B, D]

            # Линейная интерполяция
            e = e0 * (1.0 - w) + e1 * w  # [B, D]
            features_per_level.append(e)

        # Конкатим все уровни: [B, L*D]
        return torch.cat(features_per_level, dim=-1)


# ----------------------------------------------------------- #
# FractalDecoder V6
# ----------------------------------------------------------- #

class FractalDecoderV6(nn.Module):
    """
    FractalDecoder V6.

    x ∈ [0,1], опционально z ∈ R^latent_dim.
    Входные фичи:
        [x,
         sin(ω_k x), cos(ω_k x) для k=1..K  (ω_k логарифмически от 1 до max_freq)
         hash_grid(x),
         (опционально) z]

    Всё это идёт в небольшой MLP, потом через фрактальный блок T раз.
    """

    def __init__(
        self,
        vocab_size: int,
        # Fourier
        fourier_frequencies: int = 32,
        max_fourier_freq: float = 1e4,  # до sin(10000 x)
        # Hash-grid
        hash_num_levels: int = 8,
        hash_min_res: int = 16,
        hash_max_res: int = 1024,
        hash_embedding_dim: int = 8,
        hash_map_size: int = 2**14,
        # Fractal core
        hidden_dim: int = 256,
        num_steps: int = 48,
        alpha: float = 0.5,
        # Latent key
        latent_dim: int = 32,
        # Sparse updates (по умолчанию выключены, чтобы сначала добиться стабильного обучения)
        sparse_updates: bool = False,
        sparse_p: float = 0.9,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.K = fourier_frequencies
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.sparse_updates = sparse_updates
        self.sparse_p = sparse_p

        # -------- Fourier частоты (логарифмически от 1 до max_fourier_freq) --------
        if self.K > 0:
            freqs = torch.logspace(
                start=0.0,  # 10^0 = 1
                end=math.log10(max_fourier_freq),
                steps=self.K,
            )
        else:
            freqs = torch.zeros(0)
        self.register_buffer("fourier_freqs", freqs, persistent=False)

        # -------- Hash-grid 1D --------
        self.hash_grid = HashGrid1D(
            num_levels=hash_num_levels,
            min_res=hash_min_res,
            max_res=hash_max_res,
            embedding_dim=hash_embedding_dim,
            hashmap_size=hash_map_size,
        )
        hash_feat_dim = hash_num_levels * hash_embedding_dim

        # -------- Размер входного вектора --------
        in_dim = 1  # сама координата x
        if self.K > 0:
            in_dim += 2 * self.K  # sin/cos
        in_dim += hash_feat_dim  # hash-encoding
        if self.latent_dim > 0:
            in_dim += self.latent_dim

        # -------- MLP для инициализации скрытого состояния --------
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # -------- Фрактальный блок (один и тот же для всех T шагов) --------
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Dropout для разреженных обновлений (можно выключить)
        self.update_dropout = nn.Dropout(p=self.sparse_p)

        # -------- Выходной слой --------
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    # ---------------- Fourier features ---------------- #

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] или [B,1]
        -> [B, 2K]  (если K == 0 -> тензор размера [B,0])
        """
        if self.K == 0:
            return x.new_zeros(x.shape[0], 0)

        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B,1]

        # self.fourier_freqs: [K]
        # 2π ω_k x  -> [B, K]
        arg = 2.0 * math.pi * x * self.fourier_freqs  # broadcast

        sin_feat = torch.sin(arg)
        cos_feat = torch.cos(arg)
        return torch.cat([sin_feat, cos_feat], dim=-1)  # [B, 2K]

    # ---------------- Основной forward ---------------- #

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: координаты [B] или [B,1] в [0,1]
        z: опциональный латентный ключ [B, latent_dim] или [latent_dim]
        """
        # Приводим x к [B]
        if x.dim() == 2 and x.size(-1) == 1:
            x_vec = x.squeeze(-1)  # [B]
        elif x.dim() == 1:
            x_vec = x
        else:
            raise ValueError("x должен быть формы [B] или [B,1]")

        B = x_vec.size(0)
        x_norm = x_vec.clamp(0.0, 1.0)

        # 1) Базовые фичи: сама координата
        base_x = x_norm.unsqueeze(-1)  # [B,1]

        # 2) Fourier-фичи (высокие частоты)
        fourier_feat = self._fourier_features(x_norm)  # [B, 2K] или [B,0]

        # 3) Hash-encoding
        hash_feat = self.hash_grid(x_norm)  # [B, hash_feat_dim]

        feats = [base_x, fourier_feat, hash_feat]

        # 4) Латентный ключ
        if self.latent_dim > 0:
            if z is None:
                z_full = torch.zeros(
                    B,
                    self.latent_dim,
                    device=x_norm.device,
                    dtype=x_norm.dtype,
                )
            else:
                if z.dim() == 1:
                    z_full = z.unsqueeze(0).expand(B, -1)
                elif z.dim() == 2 and z.size(0) == 1:
                    z_full = z.expand(B, -1)
                elif z.dim() == 2 and z.size(0) == B:
                    z_full = z
                else:
                    raise ValueError("Неверная форма z")
            feats.append(z_full)

        feats_cat = torch.cat(feats, dim=-1)  # [B, in_dim]

        # 5) Инициализация h0
        h = self.mlp_in(feats_cat)  # [B, hidden_dim]

        # 6) Фрактальные шаги
        for _ in range(self.num_steps):
            h_update = self.fractal_block(h)
            h_update = torch.tanh(h_update)

            if self.sparse_updates:
                h_update = self.update_dropout(h_update)

            h = (1.0 - self.alpha) * h + self.alpha * h_update

        # 7) Выход
        logits = self.linear_out(h)
        return logits

    def forward_argmax(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(x, z=z)
        return torch.argmax(logits, dim=-1)