# infer_single_book_v3.py
# Инференс для FractalDecoder V3: реконструируем текст по координатам [0,1].

import os
import torch

from .fractal_decoder_v3 import FractalDecoder


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip("\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(here, "data", "book_v3.txt")
    ckpt_path = os.path.join(here, "fractal_text_decoder_v3.pt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Не найден файл с текстом: {data_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Не найден чекпоинт модели: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    text = load_text(data_path)
    n = len(text)
    print(f"Длина текста: {n}")

    ckpt = torch.load(ckpt_path, map_location=device)

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt["config"]

    vocab_size = len(itos)
    print(f"Размер словаря (из чекпоинта): {vocab_size}")

    # ВАЖНО: создаём модель с теми же параметрами, что и при обучении
    model = FractalDecoder(
        vocab_size=vocab_size,
        **config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Координаты [0,1] для каждой позиции
    if n == 1:
        xs = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        xs = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32, device=device)
    xs = xs.unsqueeze(-1)  # [N, 1]

    preds_chars = []
    with torch.no_grad():
        # Можно батчить, но для простоты сделаем всё разом
        logits = model(xs)  # [N, vocab_size]
        pred_idx = torch.argmax(logits, dim=-1)  # [N]

        for i in range(n):
            idx = int(pred_idx[i].item())
            ch = itos[idx]
            preds_chars.append(ch)

    decoded_text = "".join(preds_chars)

    print("=" * 60)
    print("Оригинальный текст (первые 1000 символов):")
    print(text[:1000])
    print("-" * 60)
    print("Реконструированный текст (первые 1000 символов):")
    print(decoded_text[:1000])
    print("=" * 60)


if __name__ == "__main__":
    main()