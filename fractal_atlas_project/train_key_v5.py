# train_key_v5.py
# Обучение ЛАТЕНТНОГО КЛЮЧА z для новой книги на фиксированном фрактальном ядре V5.
#
# Шаги:
# 1) Загружаем мастер-ядро V5 + словарь (fractal_text_decoder_v5.pt).
# 2) Загружаем текст новой книги (book_v5_new.txt), МАПИМ его через тот же stoi.
# 3) Замораживаем ВСЕ веса модели, создаём один обучаемый вектор z размерности 32.
# 4) Обучаем z так, чтобы FractalDecoderV5(x, z) восстанавливал новый текст.
# 5) Сохраняем z в отдельный файл latent_key_book_v5.pt.

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v5 import FractalDecoderV5


# ------------------------- Dataset -------------------------


@dataclass
class TextDatasetWithFixedVocab(Dataset):
    text: str
    stoi: dict
    device: torch.device

    def __post_init__(self):
        if len(self.text) == 0:
            raise ValueError("Пустой текст для обучения ключа")

        # Фильтруем символы, которых нет в словаре мастер-модели
        filtered_chars = []
        skipped = 0
        for ch in self.text:
            if ch in self.stoi:
                filtered_chars.append(ch)
            else:
                skipped += 1

        if skipped > 0:
            print(f"ВНИМАНИЕ: пропущено {skipped} символов, которых нет в словаре мастер-модели.")

        if not filtered_chars:
            raise ValueError("После фильтрации текст пуст — нет ни одного символа из словаря.")

        self.text = "".join(filtered_chars)

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


# ---------------------- Utils ----------------------


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip("\n")


def decode_first_n_with_key(
    model: FractalDecoderV5,
    dataset: TextDatasetWithFixedVocab,
    itos: dict,
    z: torch.Tensor,
    n: int = 20,
):
    model.eval()
    with torch.no_grad():
        n = min(n, len(dataset))
        for i in range(n):
            x, y_true = dataset[i]
            x_in = x.unsqueeze(0)  # [1,1]
            logits = model(x_in, z=z)   # [1, vocab_size]
            pred_idx = torch.argmax(logits, dim=-1).item()

            ch_true = itos[y_true.item()]
            ch_pred = itos[pred_idx]

            print(
                f"i={i}, x={float(x.item()):.3f}, "
                f"истинный='{ch_true}', предсказанный='{ch_pred}'"
            )


# ------------------- Training Script -------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # 1) Загружаем мастер-ядро
    ckpt_path = os.path.join(here, "fractal_text_decoder_v5.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Не найден мастер-чекпоинт V5: {ckpt_path}. "
            f"Сначала запусти train_single_book_v5.py, чтобы его создать."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt["config"]
    vocab_size = len(stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # 2) Загружаем текст НОВОЙ книги
    new_book_path = os.path.join(here, "data", "book_v5_new.txt")
    if not os.path.exists(new_book_path):
        raise FileNotFoundError(
            f"Не найден файл новой книги: {new_book_path}"
        )

    text_new = load_text(new_book_path)
    print(f"Длина новой книги (до фильтрации): {len(text_new)}")

    dataset = TextDatasetWithFixedVocab(
        text=text_new,
        stoi=stoi,
        device=device,
    )
    print(f"Длина новой книги (после фильтрации): {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 3) Восстанавливаем модель V5 и замораживаем её веса
    model = FractalDecoderV5(
        vocab_size=vocab_size,
        **config,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    latent_dim = config.get("latent_dim", 32)
    if latent_dim <= 0:
        raise ValueError("latent_dim в конфиге V5 <= 0, ключи обучать невозможно.")

    # 4) Создаём обучаемый латентный ключ z
    # Один вектор, который будем бродкастить для любых батчей.
    z = torch.zeros(1, latent_dim, device=device, dtype=torch.float32)
    z = torch.nn.Parameter(z)

    # Оптимизируем ТОЛЬКО z
    lr = 1e-2
    z_l2_lambda = 1e-4   # лёгкая L2-регуляризация на самом ключе
    num_epochs = 1500
    print_every = 50

    optimizer = torch.optim.Adam([z], lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # ----------------- Training Loop (только для z) -----------------
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for x_batch, y_batch in dataloader:
            # bsz x 1 -> bsz x latent_dim
            bsz = x_batch.size(0)
            z_batch = z.expand(bsz, -1)

            logits = model(x_batch, z=z_batch)
            ce_loss = criterion(logits, y_batch)

            # Регуляризация ключа (чтобы он не взрывался)
            if z_l2_lambda > 0.0:
                reg = (z.pow(2)).mean()
                loss = ce_loss + z_l2_lambda * reg
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(ce_loss.item()) * bsz
            total_tokens += bsz
            total_correct += int((torch.argmax(logits, dim=-1) == y_batch).sum())

        avg_loss = total_loss / total_tokens
        avg_acc = total_correct / total_tokens

        print(f"[KEY] Эпоха {epoch:04d}: loss = {avg_loss:.4f}, acc = {avg_acc:.3f}")

        if epoch % print_every == 0 or epoch == 1:
            print("-" * 60)
            decode_first_n_with_key(model, dataset, itos, z=z, n=20)
            print("-" * 60)

    # 5) Сохраняем ключ
    key_path = os.path.join(here, "latent_key_book_v5.pt")
    torch.save(
        {
            "z": z.detach().cpu(),
            "latent_dim": latent_dim,
            "config": config,
            "book_path": new_book_path,
        },
        key_path,
    )
    print(f"Латентный ключ для новой книги сохранён в: {key_path}")


if __name__ == "__main__":
    main()