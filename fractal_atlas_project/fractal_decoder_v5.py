# fractal_decoder_v5.py
# FractalDecoder V5: координата x + латентный ключ z -> логиты по символам
#
# В этом ядре зашиты идеи для компрессии:
# 1) Weight-sharing по глубине: один и тот же fractal_block применяется T раз (самоподобие).
# 2) Multi-scale representation: частоты по шкале октав (exp-последовательность).
# 3) Sparse updates: большинство шагов ничего не меняют (dropout по обновлению).
# 4) Spectral bottleneck: ограниченное число частот, растущих по шкале (низкие/высокие).
# 5) Latent key z: маленький вектор, который «разворачивает» текст внутри фиксированного ядра.
#
# Квантование (8-bit/4-bit) и более сложная регуляризация весов можно
# сделать как отдельный шаг при сохранении / экспорте.

import math
from typing import Optional

import torch
import torch.nn as nn


class FractalDecoderV5(nn.Module):
    """
    FractalDecoder V5.

    Параметры:
        vocab_size          - размер словаря (кол-во уникальных символов)
        fourier_frequencies - количество частот (K) для Fourier-признаков
        hidden_dim          - размер скрытого состояния h
        num_steps           - число фрактальных шагов T
        alpha               - коэффициент смешивания старого и нового состояния (0 < alpha <= 1)
        latent_dim          - размер латентного ключа z (0 -> не используется)
        sparse_updates      - включать разреженные обновления (dropout по h_update)
        sparse_p            - вероятность зануления обновления (напр. 0.9 -> 90% шагов почти статичны)
        base_frequency      - базовая частота для октав
        frequency_growth    - множитель между соседними частотами (октавы)
    """

    def __init__(
        self,
        vocab_size: int,
        fourier_frequencies: int = 24,
        hidden_dim: int = 256,
        num_steps: int = 48,
        alpha: float = 0.5,
        latent_dim: int = 32,
        sparse_updates: bool = True,
        sparse_p: float = 0.9,
        base_frequency: float = 1.0,
        frequency_growth: float = 1.5,
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
        self.base_frequency = base_frequency
        self.frequency_growth = frequency_growth

        # --- Спектральная сетка (октавы) ---
        # Массив частот: f_k = base * growth^k
        # Это создает multi-scale представление (низкие + высокие частоты).
        k = torch.arange(self.K, dtype=torch.float32)
        freqs = self.base_frequency * (self.frequency_growth ** k)  # [K]
        self.register_buffer("freqs", freqs, persistent=False)

        # Размер входного вектора:
        #   [x] + [sin(f_k x), cos(f_k x)] для k=1..K => 1 + 2K
        in_dim = 1 + 2 * self.K
        if self.latent_dim > 0:
            in_dim += self.latent_dim

        # MLP, который инициализирует скрытое состояние h0
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Фрактальный блок f(h) — ОДИН для всех T шагов (weight-sharing по глубине)
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Dropout для разреженных обновлений
        self.update_dropout = nn.Dropout(p=self.sparse_p)

        # Выходной слой: h_T -> логиты по словарю
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    # ----------------------------------------------------------- #
    # Вспомогательные методы
    # ----------------------------------------------------------- #

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Превращает координату x ∈ [0,1] в Fourier-признаки с октавными частотами.

        x: тензор формы [B] или [B, 1]
        return: тензор формы [B, 1 + 2K]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B, 1]

        # x: [B, 1]
        # freqs: [K]
        # 2π f_k x -> [B, K]
        arg = 2.0 * math.pi * x * self.freqs  # broadcasting: [B,1] * [K] -> [B,K]

        sin_feat = torch.sin(arg)  # [B, K]
        cos_feat = torch.cos(arg)  # [B, K]

        feats = torch.cat([x, sin_feat, cos_feat], dim=-1)  # [B, 1+2K]
        return feats

    # ----------------------------------------------------------- #
    # Основной проход
    # ----------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Прямой проход через фрактальный декодер.

        x: координаты, [B] или [B, 1]
        z: (опционально) латентный ключ, [B, latent_dim]
           Если latent_dim > 0 и z=None -> используется нулевой ключ.
        """
        # 1) Fourier-фичи по координате
        feats = self._fourier_features(x)  # [B, 1+2K]
        B = feats.size(0)

        # 2) Латентный ключ
        if self.latent_dim > 0:
            if z is None:
                # Нулевой ключ: мастер-книга или стандартное состояние
                z = torch.zeros(
                    B,
                    self.latent_dim,
                    device=feats.device,
                    dtype=feats.dtype,
                )
            elif z.dim() == 1:
                # [latent_dim] -> [1, latent_dim] -> broadcast
                z = z.unsqueeze(0).expand(B, -1)
            elif z.dim() == 2 and z.size(0) == 1:
                z = z.expand(B, -1)

            feats = torch.cat([feats, z], dim=-1)  # [B, 1+2K+latent_dim]

        # 3) Инициализация скрытого состояния
        h = self.mlp_in(feats)  # [B, hidden_dim]

        # 4) Фрактальное обновление T шагов
        for _ in range(self.num_steps):
            h_update = self.fractal_block(h)
            h_update = torch.tanh(h_update)

            if self.sparse_updates:
                # 90% обновлений почти статичны -> огромная экономия и
                # имитация разреженной динамики
                h_update = self.update_dropout(h_update)

            # Смесь старого и нового состояния
            h = (1.0 - self.alpha) * h + self.alpha * h_update

        # 5) Логиты по словарю
        logits = self.linear_out(h)  # [B, vocab_size]
        return logits

    def forward_argmax(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Удобный хелпер: координаты (и ключ) -> индексы символов.
        """
        logits = self.forward(x, z=z)
        return torch.argmax(logits, dim=-1)