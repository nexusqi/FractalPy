# train_key_v4.py
# V4: "майним" ТОЛЬКО ключ для НОВОЙ книги, мастер-фрактал заморожен.
#
# Идея:
#  - Берём уже обученный фрактал (FractalDecoder V3) + словарь символов.
#  - Замораживаем все веса модели.
#  - Создаём новый latent-вектор z_new для новой книги.
#  - Обучаем ТОЛЬКО z_new, чтобы по координатам x фрактал воспроизводил новый текст.
#
# На выходе:
#  - master-чекпоинт остаётся прежним (fractal_text_decoder_v3.pt)
#  - новый ключ сохраняем в отдельный файл: latent_key_book_v4_new.pt

import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v3 import FractalDecoder


# ----------------------------- Датасет -----------------------------


@dataclass
class TextDatasetWithFixedVocab(Dataset):
    """
    Датасет для новой книги, но со СТРОГО фиксированным словарём (stoi/itos)
    из мастер-чекпоинта.
    """

    text: str
    stoi: dict
    device: torch.device

    def __post_init__(self):
        if len(self.text) == 0:
            raise ValueError("Пустой текст для обучения ключа")

        # Проверяем, что ВСЕ символы есть в словаре мастер-модели
        unknown_chars = sorted({ch for ch in self.text if ch not in self.stoi})
        if unknown_chars:
            raise ValueError(
                "В тексте найдены символы, которых нет в словаре мастер-модели: "
                + repr(unknown_chars)
            )

        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        n = len(self.text)

        # Координаты x равномерно на [0, 1]
        if n == 1:
            xs = torch.tensor([0.0], dtype=torch.float32)
        else:
            xs = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)

        ys_idx = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        self.xs = xs.to(self.device)  # [N]
        self.ys = ys_idx.to(self.device)  # [N]

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        # x: [1], y: scalar
        x = self.xs[idx].unsqueeze(-1)
        y = self.ys[idx]
        return x, y


def load_text(path: str) -> str:
    """Простая загрузка текста из файла."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip("\n")


# --------------------- Диагностика декодирования ---------------------


def decode_first_n(model, latent_z, dataset: TextDatasetWithFixedVocab, n: int = 10):
    """Показываем первые n позиций, чтобы видеть прогресс."""
    model.eval()
    with torch.no_grad():
        n = min(n, len(dataset))
        for i in range(n):
            x, y_true = dataset[i]
            x_in = x.unsqueeze(0)  # [1,1]
            z_in = latent_z.unsqueeze(0)  # [1, latent_dim]

            logits = model(x_in, z_in)  # [1, vocab_size]
            pred_idx = torch.argmax(logits, dim=-1).item()

            ch_true = dataset.itos[y_true.item()]
            ch_pred = dataset.itos[pred_idx]

            print(
                f"i={i}, x={float(x.item()):.3f}, "
                f"истинный='{ch_true}', предсказанный='{ch_pred}'"
            )


def decode_full_text(model, latent_z, dataset: TextDatasetWithFixedVocab) -> str:
    """Полная реконструкция текста по координатам + ключу."""
    model.eval()
    preds_chars = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            x_in = x.unsqueeze(0)  # [1,1]
            z_in = latent_z.unsqueeze(0)  # [1, latent_dim]
            logits = model(x_in, z_in)
            pred_idx = torch.argmax(logits, dim=-1).item()
            ch_pred = dataset.itos[pred_idx]
            preds_chars.append(ch_pred)
    return "".join(preds_chars)


# --------------------------- Основной скрипт ---------------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # 1) Путь к мастер-чекпоинту (обученному V3)
    master_ckpt_path = os.path.join(here, "fractal_text_decoder_v3.pt")
    if not os.path.exists(master_ckpt_path):
        raise FileNotFoundError(
            f"Не найден мастер-чекпоинт: {master_ckpt_path}\n"
            f"Сначала запусти train_single_book_v3.py, чтобы его создать."
        )

    # 2) Путь к НОВОЙ книге, для которой майним ключ
    #    (положи сюда свой новый текст)
    new_book_path = os.path.join(here, "book_v4_new.txt")
    if not os.path.exists(new_book_path):
        raise FileNotFoundError(
            f"Не найден файл новой книги: {new_book_path}\n"
            f"Создай его и положи туда текст."
        )

    # 3) Девайс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # 4) Загружаем мастер-чекпоинт
    ckpt = torch.load(master_ckpt_path, map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    # ВАЖНО: гиперпараметры должны совпадать с train_single_book_v3.py
    vocab_size = len(stoi)
    fourier_frequencies = 32
    hidden_dim = 512
    num_fractal_steps = 32
    alpha = 0.6
    latent_dim = 64

    # 5) Создаём фрактал и загружаем веса
    model = FractalDecoder(
        vocab_size=vocab_size,
        fourier_frequencies=fourier_frequencies,
        hidden_dim=hidden_dim,
        num_fractal_steps=num_fractal_steps,
        alpha=alpha,
        latent_dim=latent_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("Мастер-веса успешно загружены.")

    # 6) Замораживаем ВСЕ параметры модели
    for p in model.parameters():
        p.requires_grad = False

    # 7) Создаём НОВЫЙ latent-вектор для новой книги
    latent_z_new = nn.Parameter(torch.zeros(latent_dim, device=device))
    print(f"Создан новый ключ z_new размерности {latent_dim}.")

    # 8) Гиперпараметры обучения ключа
    num_epochs = 800         # можно увеличить, если хочется сильнее зазубрить
    batch_size = 128
    lr = 5e-3
    weight_decay = 1e-6
    print_every = 50

    # 9) Загружаем текст новой книги и строим датасет
    text_new = load_text(new_book_path)
    print(f"Длина новой книги: {len(text_new)} символов")

    dataset = TextDatasetWithFixedVocab(text=text_new, stoi=stoi, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [latent_z_new],
        lr=lr,
        weight_decay=weight_decay,
    )

    # ------------------ Тренинг: учим ТОЛЬКО ключ ------------------

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for x_batch, y_batch in dataloader:
            # x_batch: [B, 1]
            # y_batch: [B]

            # Расширяем один и тот же ключ на батч
            z_batch = latent_z_new.unsqueeze(0).expand(x_batch.size(0), -1)

            logits = model(x_batch, z_batch)  # [B, vocab_size]
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = y_batch.size(0)
            total_loss += float(loss.item()) * bsz
            total_tokens += bsz
            total_correct += int((torch.argmax(logits, dim=-1) == y_batch).sum())

        avg_loss = total_loss / total_tokens
        avg_acc = total_correct / total_tokens

        print(f"[V4 key] Эпоха {epoch:03d}: loss = {avg_loss:.4f}, acc = {avg_acc:.3f}")

        if epoch % print_every == 0 or epoch == 1:
            print("-" * 60)
            decode_first_n(model, latent_z_new, dataset, n=10)
            print("-" * 60)

    # ------------------ Финальная проверка ------------------

    print("=" * 60)
    print("Оригинальный текст новой книги:")
    print(text_new[:500])
    if len(text_new) > 500:
        print("... [обрезано]")
    print("-" * 60)
    print("Текст, который восстанавливает мастер-фрактал + новый ключ:")
    decoded_new = decode_full_text(model, latent_z_new, dataset)
    print(decoded_new[:500])
    if len(decoded_new) > 500:
        print("... [обрезано]")
    print("=" * 60)

    # ------------------ Сохраняем НОВЫЙ ключ ------------------

    key_path = os.path.join(here, "latent_key_book_v4_new.pt")
    torch.save(
        {
            "book_name": "book_v4_new",
            "latent_z": latent_z_new.detach().cpu(),
            "latent_dim": latent_dim,
            "stoi": stoi,
            "itos": itos,
        },
        key_path,
    )
    print(f"Новый ключ сохранён в: {key_path}")
    print("Мастер-чекпоинт НЕ изменён (он общий для всех книг).")


if __name__ == "__main__":
    main()