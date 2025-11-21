# fractal_decoder_v2.py
# Простой, но полностью обучаемый фрактальный декодер.
# Координата x -> Fourier-признаки -> фрактальные итерации -> логиты по словарю.

import math
import torch
import torch.nn as nn


class FractalDecoder(nn.Module):
    """
    FractalDecoder V2

    Идея:
    - На входе одна координата x ∈ [0,1].
    - Преобразуем её в набор Fourier-признаков: [x, sin(2πkx), cos(2πkx)].
    - Прогоняем через небольшой MLP, чтобы получить "семя" скрытого состояния h0.
    - Дальше несколько итераций фрактального обновления h:
        h_{k+1} = (1 - α) * h_k + α * tanh(W h_k + b)
      (т.е. learnable contraction mapping).
    - После T шагов декодируем h_T в логиты по словарю.

    Это по-прежнему "координата -> символ",
    но уже более мощный нелинейный фрактальный слой.
    """

    def __init__(
        self,
        vocab_size: int,
        fourier_frequencies: int = 8,
        hidden_dim: int = 128,
        num_fractal_steps: int = 5,
        alpha: float = 0.5,
    ):
        """
        :param vocab_size: размер словаря (число уникальных символов)
        :param fourier_frequencies: сколько частот использовать в Fourier-признаках
        :param hidden_dim: размер скрытого состояния h
        :param num_fractal_steps: сколько итераций фрактального обновления T
        :param alpha: коэффициент смешивания старого и нового h (0 < α <= 1)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.K = fourier_frequencies
        self.hidden_dim = hidden_dim
        self.T = num_fractal_steps

        # Коэффициент "контракции" (смеси старого и нового состояния)
        self.alpha = alpha

        # Размер входного признакового вектора:
        # [x] + [sin(2πkx), cos(2πkx)] для k=1..K
        # итого 1 + 2K
        in_dim = 1 + 2 * self.K

        # Небольшой MLP, который инициализирует скрытое состояние h0
        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Один и тот же "фрактальный" слой, который мы применяем T раз
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)

        # Выходной слой: скрытое состояние -> логиты по словарю
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Превращает координату x ∈ [0,1] в Fourier-признаки.

        :param x: тензор формы [B, 1] или [B]
        :return: тензор формы [B, 1 + 2K]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B, 1]

        # x: [B, 1]
        # Создаем частоты k = 1..K
        k = torch.arange(1, self.K + 1, device=x.device, dtype=x.dtype)  # [K]

        # 2π k x  -> broadcast: [B,1] * [K] -> [B,K]
        arg = 2.0 * math.pi * x * k

        sin_feat = torch.sin(arg)  # [B, K]
        cos_feat = torch.cos(arg)  # [B, K]

        # Конкатенируем: x, sin(..), cos(..)
        feats = torch.cat([x, sin_feat, cos_feat], dim=-1)  # [B, 1+2K]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: координаты в диапазоне [0,1], тензор формы [B] или [B,1]
        :return: логиты по символам, форма [B, vocab_size]
        """
        # 1) Fourier-признаки
        feats = self._fourier_features(x)  # [B, 1+2K]

        # 2) Инициализация скрытого состояния
        h = self.mlp_in(feats)  # [B, hidden_dim]

        # 3) Фрактальное обновление T шагов
        for _ in range(self.T):
            h_update = torch.tanh(self.linear_hidden(h))
            h = (1.0 - self.alpha) * h + self.alpha * h_update

        # 4) Логиты по словарю
        logits = self.linear_out(h)  # [B, vocab_size]
        return logits

    def forward_argmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Удобный хелпер: координаты -> индексы символов (argmax по логитам).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)