# fractal_decoder.py
# Реализация фрактального декодера V1 на PyTorch.
# Все комментарии — на русском.

import torch
import torch.nn as nn


# ---------------------------------------------------------
# FourierFeatures — спектральное расширение одной координаты
# ---------------------------------------------------------
class FourierFeatures(nn.Module):
    def __init__(self, k_max: int = 100):
        """k_max — максимальная частота (число гармоник).

        Для каждой частоты k мы добавляем sin(kx) и cos(kx),
        в итоге размерность = k_max * 2.
        """
        super().__init__()
        self.k_max = k_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: тензор формы [B, 1] с координатами (например, x = i / N).

        Возвращает тензор формы [B, 2 * k_max] со значениями sin(kx), cos(kx).
        """
        feats = []
        for k in range(1, self.k_max + 1):
            feats.append(torch.sin(k * x))
            feats.append(torch.cos(k * x))
        return torch.cat(feats, dim=-1)


# ---------------------------------------------------------
# FractalDecoder — базовый фрактальный декодер (без ключей)
# ---------------------------------------------------------
class FractalDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        k_max: int = 100,
        vocab_size: int = 128,
        alpha: float = 0.2,
        T: int = 50,
    ):
        """Простой фрактальный декодер V1.

        Параметры:
        - d_model: размер скрытого состояния z.
        - k_max: максимальная частота для FourierFeatures.
        - vocab_size: размер алфавита (кол-во возможных символов).
        - alpha: коэффициент смешивания в итерации (1 - alpha) * z + alpha * f(z).
        - T: количество итераций фрактального ядра.
        """
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ff = FourierFeatures(k_max)

        d_in = k_max * 2  # столько фич даёт FourierFeatures
        self.ff2state = nn.Linear(d_in, d_model)

        # Одно и то же фрактальное ядро повторяется T раз
        self.frac_w = nn.Linear(d_model, d_model)

        # Финальный слой, выдающий распределение по символам
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели.

        x: [B, 1] — координата (например, i / N).

        Возвращает:
        - logits: [B, vocab_size] — логиты по алфавиту.
        """
        ff = self.ff(x)           # [B, 2 * k_max]
        z = self.ff2state(ff)     # [B, d_model]

        for _ in range(self.T):
            h = self.frac_w(z)    # [B, d_model]
            a = torch.relu(h)
            z = (1.0 - self.alpha) * z + self.alpha * a

        logits = self.out(z)
        return logits


if __name__ == "__main__":
    # Небольшой smoke-тест: прогон одной точки
    model = FractalDecoder()
    x = torch.tensor([[0.001]], dtype=torch.float32)
    logits = model(x)
    print("Форма логитов:", logits.shape)
    print("Пример логитов:", logits[0, :10].detach())
