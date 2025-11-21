# infer_single_book_v7.py
# Инференс для FractalDecoder V7.
#
# Сейчас:
#   - грузим мастер-ядро V7
#   - читаем текст мастер-книги
#   - смотрим, как фрактал восстанавливает первые 500 символов
#
# Позже сюда можно будет добавить:
#   - загрузку latent-ключа для новой книги (latent_key_book_v7.pt)
#   - развёртку другой книги тем же ядром.

import os

import torch

from .fractal_decoder_v7 import FractalDecoderV7
from .train_single_book_v7 import TextFractalDataset, load_text, decode_full_text


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ckpt_path = os.path.join(here, "fractal_text_decoder_v7.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Не найден чекпоинт V7: {ckpt_path}")

    data_path = os.path.join(here, "data", "book_v5_master.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Не найден текст мастер-книги: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    print("=" * 60)

    ckpt = torch.load(ckpt_path, map_location=device)

    config = ckpt["config"]
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    vocab_size = len(stoi)

    model = FractalDecoderV7(
        vocab_size=vocab_size,
        **config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    text = load_text(data_path)
    dataset = TextFractalDataset(text=text, device=device)

    # Перепишем словарь датасета на тот, что в чекпоинте (на всякий случай)
    dataset.stoi = stoi
    dataset.itos = itos
    dataset.vocab_size = len(stoi)

    print("Мастер-книга (первые 500 символов):")
    print(text[:500])
    print("-" * 60)

    decoded = decode_full_text(model, dataset)
    print("Текст, который восстанавливает V7 (первые 500 символов):")
    print(decoded[:500])
    print("=" * 60)

    # Заготовка под latent-ключ (Atlas V2)
    latent_path = os.path.join(here, "latent_key_book_v7.pt")
    if os.path.exists(latent_path):
        print("Найден latent_key_book_v7.pt – позже сюда добавим разворот другой книги.")
    else:
        print("latent_key_book_v7.pt не найден – показываем только мастер-книгу.")


if __name__ == "__main__":
    main()