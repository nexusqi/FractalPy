# train_single_book_v2.py
# Обучение FractalDecoder V3 на одной "книге" (строке/фрагменте текста).
# Используем 2D-координаты: (позиция, индекс предложения).

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v2 import FractalDecoder


# ---------------------------------------------------------
# Датасет: координата (x_pos, x_sent) -> индекс символа
# ---------------------------------------------------------

@dataclass
class TextFractalDatasetV2(Dataset):
    """
    Датасет для фрактального текстового декодера V2.

    Для каждого символа строим:
        x_pos  = i / (N - 1)            (позиция в тексте)
        x_sent = s_i / (S - 1)          (номер предложения, если S > 1, иначе 0)
    """

    text: str
    device: torch.device

    def __post_init__(self):
        if len(self.text) == 0:
            raise ValueError("Пустой текст для обучения")

        # Строим словарь символов
        chars = sorted(set(self.text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        n = len(self.text)

        # --- 1) Нормированная позиция в тексте ---
        if n == 1:
            pos = torch.tensor([0.0], dtype=torch.float32)
        else:
            pos = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)  # [N]

        # --- 2) Индекс предложения для каждого символа ---
        sent_indices = []
        sent_id = 0
        for ch in self.text:
            sent_indices.append(sent_id)
            if ch in ".!? \n":  # простое разбиение на "предложения"
                # увеличиваем индекс, но только если не подряд идущие пробелы/переводы
                if ch in ".!?":
                    sent_id += 1

        if len(sent_indices) == 0:
            sent_indices = [0]

        sent_indices_t = torch.tensor(sent_indices, dtype=torch.float32)  # [N]
        num_sents = int(sent_indices_t.max().item()) + 1

        if num_sents <= 1:
            sent_norm = torch.zeros_like(sent_indices_t)
        else:
            sent_norm = sent_indices_t / float(num_sents - 1)  # [N]

        # --- 3) Координаты: (pos, sent_norm) ---
        coords = torch.stack([pos, sent_norm], dim=-1)  # [N, 2]

        # --- 4) Индексы целевых символов ---
        ys_idx = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        self.xs = coords.to(self.device)   # [N, 2]
        self.ys = ys_idx.to(self.device)   # [N]

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        x = self.xs[idx]     # [2]
        y = self.ys[idx]     # []
        return x, y


# ---------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------

def load_text(path: str) -> str:
    """Читает текст из файла и слегка его чистит."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.strip("\n")
    return text


def decode_first_n(model, dataset: TextFractalDatasetV2, n: int = 10):
    """Небольшая диагностика: показываем первые n позиций."""
    model.eval()
    with torch.no_grad():
        n = min(n, len(dataset))
        for i in range(n):
            x, y_true = dataset[i]   # x: [2], y_true: []
            x_in = x.unsqueeze(0)    # [1, 2]
            logits = model(x_in)     # [1, vocab_size]
            pred_idx = torch.argmax(logits, dim=-1).item()

            ch_true = dataset.itos[y_true.item()]
            ch_pred = dataset.itos[pred_idx]

            print(
                f"i={i}, coords=({float(x[0]):.3f}, {float(x[1]):.3f}), "
                f"истинный='{ch_true}', предсказанный='{ch_pred}'"
            )


def decode_full_text(model, dataset: TextFractalDatasetV2) -> str:
    """Прогоняем всю строку и собираем предсказанные символы."""
    model.eval()
    preds_chars = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]          # [2]
            x_in = x.unsqueeze(0)      # [1, 2]
            logits = model(x_in)       # [1, vocab_size]
            pred_idx = torch.argmax(logits, dim=-1).item()
            ch_pred = dataset.itos[pred_idx]
            preds_chars.append(ch_pred)
    return "".join(preds_chars)


# ---------------------------------------------------------
# Основной скрипт обучения
# ---------------------------------------------------------

def main():
    # Путь к тексту
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "data", "example_text.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Не найден файл с текстом: {data_path}")

    # Девайс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Гиперпараметры обучения
    num_epochs = 8000           # можно крутить ещё выше
    batch_size = 256
    lr = 3e-4
    weight_decay = 1e-6
    max_grad_norm = 1.0
    print_every = 200

    # Загружаем текст
    text = load_text(data_path)
    print(f"Длина текста: {len(text)}")

    dataset = TextFractalDatasetV2(text=text, device=device)
    print(f"Размер словаря: {dataset.vocab_size}")

    print("Частоты символов в датасете:")
    with torch.no_grad():
        class_counts = torch.bincount(dataset.ys, minlength=dataset.vocab_size)
        for ch, idx in dataset.stoi.items():
            cnt = int(class_counts[idx].item())
            print(f"  '{ch}': {cnt}")
    print("-" * 60)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Создаём модель (FractalDecoder V3)
    model = FractalDecoder(
        vocab_size=dataset.vocab_size,
        coord_dim=2,             # т.к. датасет отдаёт (pos, sent_id)
        fourier_frequencies=32,
        hidden_dim=512,
        num_fractal_steps=32,
        alpha=0.6,
        dropout=0.1,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )

    # Цикл обучения
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for x_batch, y_batch in dataloader:
            # x_batch, y_batch уже на нужном девайсе
            logits = model(x_batch)        # [B, vocab_size]
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            # градиент-клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            bsz = y_batch.size(0)
            total_loss += float(loss.item()) * bsz
            total_tokens += bsz
            total_correct += int((torch.argmax(logits, dim=-1) == y_batch).sum())

        scheduler.step()

        avg_loss = total_loss / total_tokens
        avg_acc = total_correct / total_tokens

        print(f"Эпоха {epoch:04d}: loss = {avg_loss:.4f}, acc = {avg_acc:.3f}")

        if epoch % print_every == 0 or epoch == 1:
            print("-" * 60)
            decode_first_n(model, dataset, n=20)
            print("-" * 60)

    # Финальный вывод реконструкции
    print("=" * 60)
    print("Оригинальный текст:")
    print(text)
    print("-" * 60)
    print("Текст, который восстанавливает модель:")
    decoded = decode_full_text(model, dataset)
    print(decoded)
    print("=" * 60)

    # Сохраняем веса
    ckpt_path = os.path.join(here, "fractal_text_decoder_v3.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        },
        ckpt_path,
    )
    print(f"Модель сохранена в: {ckpt_path}")


if __name__ == "__main__":
    main()