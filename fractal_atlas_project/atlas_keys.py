# atlas_keys.py
# Заготовка под «Фрактальный Атлас».
# Идея: один мастер-фрактал + разные латентные ключи для разных «книг».

from typing import Optional

import torch
import torch.nn as nn

from .fractal_decoder import FractalDecoder


class FractalDecoderWithKey(nn.Module):
    """Расширение FractalDecoder с латентным «ключом книги».

    Идея:
    - есть общий фрактальный декодер (мастер-фрактал),
    - для каждой «книги» или «слота» есть латентный вектор-ключ,
    - ключ слегка деформирует координату x (scale + bias),
      тем самым «подстраивая» одну и ту же фрактальную структуру
      под разные данные.
    """

    def __init__(
        self,
        d_model: int = 256,
        k_max: int = 100,
        vocab_size: int = 128,
        num_steps: int = 8,
        alpha: float = 0.5,
        num_slots: int = 16,
        key_dim: int = 32,
        freeze_decoder: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.num_slots = int(num_slots)
        self.key_dim = int(key_dim)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Базовый фрактальный декодер — общий для всех слотов
        self.decoder = FractalDecoder(
            d_model=d_model,
            k_max=k_max,
            vocab_size=vocab_size,
            num_steps=num_steps,
            alpha=alpha,
        ).to(self.device)

        # Эмбеддинги ключей для каждого слота
        self.key_embed = nn.Embedding(self.num_slots, self.key_dim)

        # Небольшие головы, которые превращают ключ в scale и bias для координаты
        self.scale_head = nn.Linear(self.key_dim, 1)
        self.bias_head = nn.Linear(self.key_dim, 1)

        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        self._reset_parameters()

        self.to(self.device)

    def _reset_parameters(self) -> None:
        """Инициализация эмбеддингов и голов."""
        nn.init.normal_(self.key_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        slot_ids: torch.Tensor | int,
    ) -> torch.Tensor:
        """Координаты + id слота -> логиты.

        x: [B, 1] или [B] — координаты.
        slot_ids: либо скаляр int, либо тензор формы [B] со слотами.
        """
        # Переносим на нужное устройство
        x = x.to(self.device)

        # Приводим slot_ids к тензору [B]
        if isinstance(slot_ids, int):
            # Один и тот же слот для всего батча
            if x.dim() == 1:
                batch = x.size(0)
            else:
                batch = x.size(0)
            slot_ids = torch.full(
                (batch,),
                slot_ids,
                dtype=torch.long,
                device=self.device,
            )
        elif isinstance(slot_ids, torch.Tensor):
            if slot_ids.dim() == 0:
                slot_ids = slot_ids.unsqueeze(0)
            slot_ids = slot_ids.to(dtype=torch.long, device=self.device)
        else:
            raise TypeError("slot_ids должен быть int или torch.Tensor")

        # Эмбеддинг ключей: [B, key_dim]
        key_emb = self.key_embed(slot_ids)  # [B, key_dim]

        # Преобразуем ключ в scale и bias
        scale = self.scale_head(key_emb)  # [B, 1]
        bias = self.bias_head(key_emb)    # [B, 1]

        # x может быть [B] или [B,1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Мягко деформируем координату:
        # x' = x * (1 + 0.1 * tanh(scale)) + 0.1 * tanh(bias)
        x_prime = x * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)

        # Прогоняем через общий фрактальный декодер
        logits = self.decoder(x_prime)
        return logits


class SimpleFractalAtlas(nn.Module):
    """Простейший «Фрактальный Атлас».

    Обёртка над FractalDecoderWithKey, чтобы логически отделить
    понятие «атласа» от конкретной реализации ключей.
    """

    def __init__(
        self,
        d_model: int = 256,
        k_max: int = 100,
        vocab_size: int = 128,
        num_steps: int = 8,
        alpha: float = 0.5,
        num_slots: int = 16,
        key_dim: int = 32,
        freeze_decoder: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.inner = FractalDecoderWithKey(
            d_model=d_model,
            k_max=k_max,
            vocab_size=vocab_size,
            num_steps=num_steps,
            alpha=alpha,
            num_slots=num_slots,
            key_dim=key_dim,
            freeze_decoder=freeze_decoder,
            device=device,
        )

    def decode(self, x: torch.Tensor, slot_ids: torch.Tensor | int) -> torch.Tensor:
        """Внешний метод декодирования: координаты + слот -> логиты."""
        return self.inner(x, slot_ids)


# ---------------------------------------------------------
# Пример: оптимизация одного ключа под маленький набор точек
# ---------------------------------------------------------
def example_optimize_single_key():
    """Небольшой пример: учим один ключ под 3 точки."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Допустим, у нас есть 3 координаты и целевые символы
    x_coords = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32, device=device)
    # Здесь vocab_size=128, поэтому просто берём какие-то индексы токенов
    y_targets = torch.tensor([10, 20, 30], dtype=torch.long, device=device)

    vocab_size = 128
    num_slots = 8
    slot_id = 3  # будем учить именно этот слот

    atlas = SimpleFractalAtlas(
        d_model=256,
        k_max=64,
        vocab_size=vocab_size,
        num_steps=6,
        alpha=0.5,
        num_slots=num_slots,
        key_dim=32,
        freeze_decoder=True,  # фиксируем веса декодера, учим только ключ
        device=device,
    ).to(device)

    # Оптимизируем только параметры ключей (и их голов)
    params_to_optimize = [
        p for name, p in atlas.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    atlas.train()
    for step in range(100):
        optimizer.zero_grad()
        logits = atlas.decode(x_coords, slot_id)  # [3, vocab_size]
        loss = criterion(logits, y_targets)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Шаг {step}: loss={loss.item():.4f}")

    print("Готово. Ключ для слота", slot_id, "обучен на заданные точки.")


if __name__ == "__main__":
    # Простейший запуск примера оптимизации одного ключа
    example_optimize_single_key()