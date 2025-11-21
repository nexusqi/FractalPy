# train_key_v7.py
# Обучаем ТОЛЬКО latent-ключ z для новой книги,
# используя замороженное фрактальное ядро V7.

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v7 import FractalDecoderV7


# ---------------- Dataset с фиксированным словарём ---------------- #


@dataclass
class TextDatasetWithFixedVocab(Dataset):
    text: str
    stoi: dict
    device: torch.device

    def __post_init__(self):
        if len(self.text) == 0:
            raise ValueError("Пустой текст для обучения")

        # Проверяем, что все символы есть в словаре мастер-книги
        missing = sorted({ch for ch in set(self.text) if ch not in self.stoi})
        if missing:
            raise ValueError(f"В тексте есть символы вне словаря мастер-книги: {missing}")

        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        n = len(self.text)
        if n == 1:
            xs = torch.tensor([0.0], dtype=torch.float32)
        else:
            xs = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)

        ys_idx = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        self.xs = xs.to(self.device)
        self.ys = ys_idx.to(self.device)

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        x = self.xs[idx].unsqueeze(-1)  # [1]
        y = self.ys[idx]
        return x, y


# ---------------- Utils ---------------- #


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip("\n")


def decode_first_n(model, dataset: TextDatasetWithFixedVocab, z: torch.Tensor, n: int = 20):
    model.eval()
    with torch.no_grad():
        n = min(n, len(dataset))
        for i in range(n):
            x, y_true = dataset[i]
            x_in = x.unsqueeze(0)
            logits = model(x_in, z=z)
            pred_idx = torch.argmax(logits, dim=-1).item()

            ch_true = dataset.itos[y_true.item()]
            ch_pred = dataset.itos[pred_idx]

            print(
                f"i={i}, x={float(x.item()):.3f}, "
                f\"истинный='{ch_true}', предсказанный='{ch_pred}'\"
            )


# ---------------- Training Script ---------------- #


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # 1. Грузим мастер-фрактал
    ckpt_path = os.path.join(here, "fractal_text_decoder_v7.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Не найден чекпоинт мастер-книги: {ckpt_path}\n"
            f"Сначала запусти train_single_book_v7.py"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt["config"]
    vocab_size = len(stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # 2. Текст новой книги
    data_path = os.path.join(here, "data", "book_v7_new.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Не найден файл новой книги: {data_path}"
        )

    text = load_text(data_path)
    print(f"Длина текста новой книги: {len(text)}")

    dataset = TextDatasetWithFixedVocab(text=text, stoi=stoi, device=device)
    print(f"Словарь (мастер): {len(stoi)} символов")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 3. Восстанавливаем модель и замораживаем веса
    model = FractalDecoderV7(
        vocab_size=vocab_size,
        **config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    latent_dim = config.get("latent_dim", 32)
    if latent_dim <= 0:
        raise ValueError("latent_dim <= 0 в конфиге – нечему обучаться!")

    # 4. Обучаем только latent-ключ z
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=3e-2)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 400
    print_every = 20

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for x_batch, y_batch in dataloader:
            logits = model(x_batch, z=z)
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

        print(f"[KEY] Эпоха {epoch:03d}: loss = {avg_loss:.4f}, acc = {avg_acc:.3f}")

        if epoch % print_every == 0 or epoch == 1:
            print("-" * 60)
            decode_first_n(model, dataset, z=z, n=20)
            print("-" * 60)

    # 5. Сохраняем ключ
    key_path = os.path.join(here, "latent_key_book_v7.pt")
    torch.save({"z": z.detach().cpu(), "config": config, "stoi": stoi, "itos": itos}, key_path)
    print(f"Latent-ключ для книги сохранён в: {key_path}")


if __name__ == "__main__":
    main()