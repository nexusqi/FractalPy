# train_single_book_v5.py
# Обучение FractalDecoder V5 на одной книге.
#
# Результат: мастер-фрактал + общий словарь.
# Дальше мы будем замораживать это ядро и обучать только latent-ключи z
# для новых книг (см. train_key_v5.py).

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v5 import FractalDecoderV5


# ------------------------- Dataset -------------------------


@dataclass
class TextFractalDataset(Dataset):
    text: str
    device: torch.device

    def __post_init__(self):
        if len(self.text) == 0:
            raise ValueError("Пустой текст для обучения")

        chars = sorted(set(self.text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        n = len(self.text)

        # Равномерные координаты на [0, 1]
        if n == 1:
            xs = torch.tensor([0.0], dtype=torch.float32)
        else:
            xs = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)

        ys_idx = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        self.xs = xs.to(self.device)     # [N]
        self.ys = ys_idx.to(self.device) # [N]

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        x = self.xs[idx].unsqueeze(-1)  # [1]
        y = self.ys[idx]
        return x, y


# ---------------------- Utils ----------------------


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip("\n")


def decode_first_n(model, dataset: TextFractalDataset, n: int = 20):
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
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            x_in = x.unsqueeze(0)
            logits = model(x_in)
            pred_idx = torch.argmax(logits, dim=-1).item()
            preds.append(dataset.itos[pred_idx])
    return "".join(preds)


# ------------------- Training Script -------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # Файл мастер-книги для ядра V5
    data_path = os.path.join(here, "data", "book_v5_master.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Не найден файл с текстом мастер-книги: {data_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Конфиг V5 (запишем его в чекпоинт)
    config = {
        "fourier_frequencies": 24,
        "hidden_dim": 256,
        "num_steps": 48,
        "alpha": 0.5,
        "latent_dim": 32,          # z_dim = 32
        "sparse_updates": True,
        "sparse_p": 0.9,
        "base_frequency": 1.0,
        "frequency_growth": 1.5,
    }

    num_epochs = 800
    batch_size = 128
    lr = 3e-4
    weight_decay = 1e-4
    weight_entropy_lambda = 1e-5   # лёгкая регуляризация весов
    print_every = 40

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

    # Модель V5
    model = FractalDecoderV5(
        vocab_size=dataset.vocab_size,
        **config,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ----------------- Training Loop -----------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for x_batch, y_batch in dataloader:
            logits = model(x_batch)       # z по умолчанию = 0
            ce_loss = criterion(logits, y_batch)

            # Простая "entropy-like" регуляризация весов
            if weight_entropy_lambda > 0.0:
                reg = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        reg = reg + (p.pow(2)).mean()
                loss = ce_loss + weight_entropy_lambda * reg
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = y_batch.size(0)
            total_loss += float(ce_loss.item()) * bsz
            total_tokens += bsz
            total_correct += int((torch.argmax(logits, dim=-1) == y_batch).sum())

        avg_loss = total_loss / total_tokens
        avg_acc = total_correct / total_tokens

        print(f"Эпоха {epoch:03d}: loss = {avg_loss:.4f}, acc = {avg_acc:.3f}")

        if epoch % print_every == 0 or epoch == 1:
            print("-" * 60)
            decode_first_n(model, dataset, n=20)
            print("-" * 60)

    # ----------------- Final Check & Save -----------------
    print("=" * 60)
    print("Оригинальный текст (первые 500 символов):")
    print(text[:500])
    print("-" * 60)
    print("Текст, который восстанавливает модель (первые 500 символов):")
    decoded = decode_full_text(model, dataset)
    print(decoded[:500])
    print("=" * 60)

    ckpt_path = os.path.join(here, "fractal_text_decoder_v5.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "config": config,
        },
        ckpt_path,
    )
    print(f"Модель V5 сохранена в: {ckpt_path}")


if __name__ == "__main__":
    main()