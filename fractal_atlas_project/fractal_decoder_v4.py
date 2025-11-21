# fractal_decoder_v4.py
# Фрактальное ядро V4 с упором на компрессию:
#  - weight-sharing по глубине (один и тот же блок крутится T раз)
#  - multi-scale Fourier-признаки (октавы)
#  - sparse updates (обновляем только часть нейронов)
#  - spectral bottleneck (контроль максимальной частоты)
#
# Интерфейс максимально похож на V3, чтобы было легко переехать.

import math
from typing import Optional

import torch
import torch.nn as nn


class FractalDecoder(nn.Module):
    """
    FractalDecoder V4 (compress-oriented).

    Параметры:
        vocab_size:           размер словаря (кол-во символов)
        fourier_frequencies:  сколько БАЗОВЫХ частот в одном октавном блоке
        num_octaves:          сколько октав (масштабов) использовать
        hidden_dim:           размер скрытого состояния h
        num_fractal_steps:    число фрактальных шагов T
        alpha:                коэффициент смешивания (0 < alpha <= 1)
        latent_dim:           размер латентного ключа z (если None – ключ не используется)
        sparse_fraction:      доля нейронов, которые реально обновляются (0.0..1.0)
                              0.0  -> полнообновляемая сеть (как раньше)
                              0.1  -> обновляем 10% нейронов, остальное просто копируется
        high_freq_scale:      множитель для старших октав (spectral bottleneck).
                              Например, 0.5 ослабляет вклад самых высоких частот.
    """

    def __init__(
        self,
        vocab_size: int,
        fourier_frequencies: int = 8,
        num_octaves: int = 4,
        hidden_dim: int = 256,
        num_fractal_steps: int = 32,
        alpha: float = 0.6,
        latent_dim: Optional[int] = None,
        sparse_fraction: float = 0.0,
        high_freq_scale: float = 1.0,
    ):
        super().__init__()

        # --------- Базовые параметры ---------
        self.vocab_size = vocab_size
        self.K_base = fourier_frequencies
        self.num_octaves = num_octaves
        self.hidden_dim = hidden_dim
        self.num_fractal_steps = num_fractal_steps
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.sparse_fraction = float(max(0.0, min(1.0, sparse_fraction)))
        self.high_freq_scale = high_freq_scale

        # Общее число частот = K_base * num_octaves
        self.total_K = self.K_base * self.num_octaves

        # --------- Multi-scale / octave параметры ---------
        # Для каждой октавы будет свой learnable scale (масштаб влияния).
        # Это даёт модели возможность сама решить, сколько хранить в глобальных
        # структурах, а сколько в локальных.
        self.octave_gains = nn.Parameter(
            torch.ones(self.num_octaves, dtype=torch.float32)
        )

        # --------- Размер входного вектора ---------
        # [x] + [sin/cos для всех частот] + [опциональный z]
        in_dim = 1 + 2 * self.total_K
        if latent_dim is not None:
            in_dim += latent_dim

        # --------- Входной MLP: feats -> h0 ---------
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # --------- Фрактальный блок f(h) (weight-sharing по глубине) ---------
        # Один и тот же блок применяется num_fractal_steps раз.
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Нормировка (чуть стабилизируем динамику по глубине)
        self.norm = nn.LayerNorm(hidden_dim)

        # --------- Выход: h_T -> логиты ---------
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------ #
    # Fourier + multi-scale (octave) признаки
    # ------------------------------------------------------------------ #

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Превращает координату x ∈ [0,1] в multi-scale Fourier-признаки.

        x: тензор формы [B] или [B, 1]
        возвращает: [B, 1 + 2*total_K]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B, 1]

        B = x.size(0)
        device = x.device
        dtype = x.dtype

        # Базовые частоты: 1..K_base
        base_k = torch.arange(1, self.K_base + 1, device=device, dtype=dtype)  # [K_base]

        sin_feats = []
        cos_feats = []

        for octave in range(self.num_octaves):
            # Масштаб октавы: 2^octave
            scale = 2.0 ** octave

            # k_oct = base_k * scale
            k_oct = base_k * scale  # [K_base]

            # 2π k x
            arg = 2.0 * math.pi * x * k_oct  # broadcast -> [B, K_base]

            sin_oct = torch.sin(arg)
            cos_oct = torch.cos(arg)

            # Применяем learnable gain для данной октавы
            gain = self.octave_gains[octave]

            # Доп. spectral bottleneck: старшие октавы можем ослаблять
            if self.high_freq_scale != 1.0 and octave == self.num_octaves - 1:
                gain = gain * self.high_freq_scale

            sin_feats.append(gain * sin_oct)
            cos_feats.append(gain * cos_oct)

        # Конкатенируем все октавы по частотному измерению
        sin_all = torch.cat(sin_feats, dim=-1)  # [B, total_K]
        cos_all = torch.cat(cos_feats, dim=-1)  # [B, total_K]

        feats = torch.cat([x, sin_all, cos_all], dim=-1)  # [B, 1 + 2*total_K]
        return feats

    # ------------------------------------------------------------------ #
    # Основной проход
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: координаты, [B] или [B, 1]
        z: опциональный латентный ключ, [B, latent_dim]

        Возвращает логиты [B, vocab_size].
        """
        # 1) Fourier + multi-scale признаки
        feats = self._fourier_features(x)  # [B, 1 + 2*total_K]

        # 2) Если есть latent-ключ, приклеиваем его
        if z is not None:
            if z.dim() == 1:
                z = z.unsqueeze(-1)  # [B] -> [B, 1]
            feats = torch.cat([feats, z], dim=-1)  # [B, 1 + 2*total_K + latent_dim]

        # 3) Инициализируем скрытое состояние
        h = self.mlp_in(feats)  # [B, hidden_dim]

        # 4) Фрактальное обновление с опциональными "sparse updates"
        if self.sparse_fraction <= 0.0:
            # Полное обновление, как в V3
            for _ in range(self.num_fractal_steps):
                h_update = torch.tanh(self.fractal_block(h))
                h = (1.0 - self.alpha) * h + self.alpha * h_update
                h = self.norm(h)
        else:
            # Обновляем только часть нейронов
            B, D = h.shape
            k = max(1, int(D * self.sparse_fraction))  # сколько нейронов обновляем

            for _ in range(self.num_fractal_steps):
                h_update = torch.tanh(self.fractal_block(h))  # [B, D]

                # Строим один и тот же маск по фичам для всех объектов батча:
                idx = torch.randperm(D, device=h.device)[:k]
                mask = torch.zeros(D, device=h.device, dtype=h.dtype)
                mask[idx] = 1.0  # обновляем только эти координаты

                # Расширяем маску до [B, D]
                mask_batch = mask.unsqueeze(0).expand(B, D)

                # h_new = (1 - alpha)*h + alpha*h_update, но только на активных координатах
                h = (1.0 - self.alpha) * h + self.alpha * h_update * mask_batch
                h = self.norm(h)

        # 5) Логиты по словарю
        logits = self.linear_out(h)
        return logits

    def forward_argmax(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Удобный хелпер: координаты (и опциональный ключ) -> индексы символов."""
        logits = self.forward(x, z=z)
        return torch.argmax(logits, dim=-1)