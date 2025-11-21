# train_single_book_v4.py
# Обучение FractalDecoder V4 на одной "книге" (строке текста).
#
# V4-ядро заточено под компрессию:
#   - weight-sharing по глубине (один и тот же блок крутится T раз)
#   - multi-scale Fourier-признаки (октавы)
#   - sparse updates (обновляем только часть нейронов)
#   - spectral bottleneck (контроль высоких частот)
#   - (опционально) weight-entropy regularization

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v4 import FractalDecoder


# ---------------------------------------------------------
# Датасет: координата x -> индекс символа
# ---------------------------------------------------------


@dataclass
class TextFractalDataset(Dataset):
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

        # Координаты x равномерно на [0, 1]
        if n == 1:
            xs = torch.tensor([0.0], dtype=torch.float32)
        else:
            xs = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)

        ys_idx = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        self.xs = xs.to(self.device)       # [N]
        self.ys = ys_idx.to(self.device)   # [N]

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        x = self.xs[idx].unsqueeze(-1)  # [1]
        y = self.ys[idx]
        return x, y


# ---------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------


def load_text(path: str) -> str:
    """Читает текст из файла и немного чистит."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip("\n")


def decode_first_n(model, dataset: TextFractalDataset, n: int = 10):
    """Диагностика: показываем первые n позиций."""
    model.eval()
    with torch.no_grad():
        n = min(n, len(dataset))
        for i in range(n):
            x, y_true = dataset[i]
            x_in = x.unsqueeze(0)  # [1,1]
            logits = model(x_in)   # [1, vocab_size]
            pred_idx = torch.argmax(logits, dim=-1).item()

            ch_true = dataset.itos[y_true.item()]
            ch_pred = dataset.itos[pred_idx]

            print(
                f"i={i}, x={float(x.item()):.3f}, "
                f"истинный='{ch_true}', предсказанный='{ch_pred}'"
            )


def decode_full_text(model, dataset: TextFractalDataset) -> str:
    """Прогоняем всю строку и собираем предсказанные символы."""
    model.eval()
    preds_chars = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            x_in = x.unsqueeze(0)      # [1,1]
            logits = model(x_in)       # [1, vocab_size]
            pred_idx = torch.argmax(logits, dim=-1).item()
            ch_pred = dataset.itos[pred_idx]
            preds_chars.append(ch_pred)
    return "".join(preds_chars)


# -------- weight-entropy regularization (опционально) --------


def weight_entropy_penalty(model: torch.nn.Module) -> torch.Tensor:
    """
    Naказываем модель за «разбухание» весов.
    Чем равномернее веса, тем выше энтропия — мы её слегка штрафуем.
    """
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p.view(-1))
    w = torch.cat(params)  # [N]

    w_abs = torch.abs(w)
    w_sum = w_abs.sum() + 1e-8
    p = w_abs / w_sum

    entropy = -(p * (p + 1e-8).log()).sum()
    return entropy


# ---------------------------------------------------------
# Основной скрипт обучения
# ---------------------------------------------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # ====== ВЫБОР КНИГИ ДЛЯ ОБУЧЕНИЯ ======
    # Можешь менять файл тут:
    # data_path = os.path.join(here, "data", "book_v3.txt")
    data_path = os.path.join(here, "data", "book_v4_new.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Не найден файл с текстом: {data_path}")

    # Девайс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # ====== Гиперпараметры V4 ======
    config = {
        "fourier_frequencies": 8,    # базовых частот в октаве
        "num_octaves": 4,            # сколько октав (масштабов)
        "hidden_dim": 256,
        "num_fractal_steps": 32,     # T
        "alpha": 0.6,
        "latent_dim": None,          # для одной книги ключ не нужен
        "sparse_fraction": 0.2,      # доля обновляемых нейронов (0.0 = как V3)
        "high_freq_scale": 0.5,      # поджимаем самую высокую октаву
    }

    # Тренировочные параметры
    num_epochs = 800
    batch_size = 128
    lr = 3e-4
    weight_decay = 1e-4
    print_every = 40

    # Коэффициент для энтропийного регуляризатора
    use_entropy_reg = True
    lambda_entropy = 1e-4

    # ====== Загружаем текст ======
    text = load_text(data_path)
    print(f"Длина текста: {len(text)}")

    dataset = TextFractalDataset(text=text, device=device)
    print(f"Размер словаря: {dataset.vocab_size}")

    print("Частоты символов в датасете:")
    with torch.no_grad():
        class_counts = torch.bincount(dataset.ys, minlength=dataset.vocab_size)
        for ch, idx in dataset.stoi.items():
            cnt = int(class_counts[idx].item())
            print(f"  '{ch}': {cnt}")
    print("-" * 60)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ====== Создаём модель V4 ======
    model = FractalDecoder(
        vocab_size=dataset.vocab_size,
        **config,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # -------------------------------------------------
    # Цикл обучения
    # -------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for x_batch, y_batch in dataloader:
            # x_batch, y_batch уже на правильном device (внутри датасета)
            logits = model(x_batch)  # [B, vocab_size]
            ce_loss = criterion(logits, y_batch)

            if use_entropy_reg:
                entropy_loss = weight_entropy_penalty(model)
                loss = ce_loss + lambda_entropy * entropy_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = y_batch.size(0)
            total_loss += float(loss.item()) * bsz
            total_tokens += bsz
            total_correct += int((torch.argmax(logits, dim=-1) == y_batch).sum())

        avg_loss = total_loss / total_tokens
        avg_acc = total_correct / total_tokens

        print(f"Эпоха {epoch:03d}: loss = {avg_loss:.4f}, acc = {avg_acc:.3f}")

        if epoch % print_every == 0 or epoch == 1:
            print("-" * 60)
            decode_first_n(model, dataset, n=20)
            print("-" * 60)

    # ====== Финальная проверка ======
    print("=" * 60)
    print("Оригинальный текст (первые 500 символов):")
    print(text[:500])
    print("-" * 60)
    print("Текст, который восстанавливает модель (первые 500 символов):")
    decoded = decode_full_text(model, dataset)
    print(decoded[:500])
    print("=" * 60)

    # ====== Сохранение модели ======
    ckpt_path = os.path.join(here, "fractal_text_decoder_v4.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "config": config,
        },
        ckpt_path,
    )
    print(f"Модель V4 сохранена в: {ckpt_path}")


if __name__ == "__main__":
    main()