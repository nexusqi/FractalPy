# infer_single_book_v5.py
# Инференс для FractalDecoder V5:
#  - можно использовать либо нулевой ключ (мастер-книга),
#  - либо загруженный latent_key_book_v5.pt для новой книги.

import os

import torch

from .fractal_decoder_v5 import FractalDecoderV5


def decode_book_with_key(
    model: FractalDecoderV5,
    text_len: int,
    itos: dict,
    z: torch.Tensor | None,
    device: torch.device,
) -> str:
    model.eval()
    preds = []
    with torch.no_grad():
        if text_len == 1:
            xs = torch.tensor([0.0], dtype=torch.float32, device=device)
        else:
            xs = torch.linspace(0.0, 1.0, steps=text_len, dtype=torch.float32, device=device)

        for i in range(text_len):
            x = xs[i].unsqueeze(0).unsqueeze(-1)  # [1,1]
            logits = model(x, z=z)
            idx = torch.argmax(logits, dim=-1).item()
            preds.append(itos[idx])
    return "".join(preds)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, "fractal_text_decoder_v5.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Не найден мастер-чекпоинт V5: {ckpt_path}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt["config"]
    vocab_size = len(stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    model = FractalDecoderV5(
        vocab_size=vocab_size,
        **config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Длину текста возьмём из выбранного файла
    # По умолчанию — мастер-книга
    master_book_path = os.path.join(here, "data", "book_v5_master.txt")
    if not os.path.exists(master_book_path):
        raise FileNotFoundError(f"Не найден текст мастер-книги: {master_book_path}")

    with open(master_book_path, "r", encoding="utf-8") as f:
        master_text = f.read().strip("\n")

    text_len = len(master_text)

    # Попробуем сначала без ключа (z=None) — мастер-книга
    print("=" * 60)
    print("Мастер-книга (нулевой ключ, первые 500 символов):")
    decoded_master = decode_book_with_key(
        model=model,
        text_len=text_len,
        itos=itos,
        z=None,
        device=device,
    )
    print(decoded_master[:500])
    print("=" * 60)

    # Если есть латентный ключ новой книги — попробуем и его
    key_path = os.path.join(here, "latent_key_book_v5.pt")
    if os.path.exists(key_path):
        key_ckpt = torch.load(key_path, map_location=device)
        z = key_ckpt["z"].to(device)  # [1, latent_dim]

        # Возьмём длину новой книги из файла, который указан в чекпоинте ключа
        book_path = key_ckpt.get("book_path", None)
        if book_path is not None and os.path.exists(book_path):
            with open(book_path, "r", encoding="utf-8") as f:
                new_text = f.read().strip("\n")
            new_len = len(new_text)
        else:
            # fallback: длина мастер-текста
            new_len = text_len

        print("Книга по латентному ключу (первые 500 символов):")
        decoded_new = decode_book_with_key(
            model=model,
            text_len=new_len,
            itos=itos,
            z=z,
            device=device,
        )
        print(decoded_new[:500])
        print("=" * 60)
    else:
        print("latent_key_book_v5.pt не найден — вывод только мастер-книги.")


if __name__ == "__main__":
    main()