# train_single_book_v3.py (FINAL)

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .fractal_decoder_v3 import FractalDecoder


@dataclass
class TextFractalDataset(Dataset):
    text: str
    device: torch.device

    def __post_init__(self):
        chars = sorted(set(self.text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        self.vocab_size = len(self.stoi)

        n = len(self.text)
        xs = torch.linspace(0, 1, n, dtype=torch.float32)
        ys = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)

        self.xs = xs.to(self.device)
        self.ys = ys.to(self.device)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx].unsqueeze(-1), self.ys[idx]


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip("\n")


def decode_first_n(model, dataset, n=20):
    model.eval()
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            x, y = dataset[i]
            pred = model.forward_argmax(x.unsqueeze(0)).item()
            print(f"{i:03d} | true='{dataset.itos[y.item()]}' pred='{dataset.itos[pred]}'")


def decode_full_text(model, dataset):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            pred = model.forward_argmax(x.unsqueeze(0)).item()
            out.append(dataset.itos[pred])
    return "".join(out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "data", "book_v3.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    config = dict(
        fourier_frequencies=32,
        hidden_dim=512,
        num_fractal_steps=48,
        alpha=0.55,
        latent_dim=None,     # пока OFF
    )

    text = load_text(data_path)
    dataset = TextFractalDataset(text, device)

    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = FractalDecoder(
        vocab_size=dataset.vocab_size,
        **config
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    EPOCHS = 600

    for ep in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0
        tot_acc = 0
        tot = 0

        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pred = torch.argmax(logits, dim=-1)
            tot_loss += loss.item() * x.size(0)
            tot_acc += (pred == y).sum().item()
            tot += x.size(0)

        print(f"Epoch {ep:04d}: loss={tot_loss/tot:.4f} acc={tot_acc/tot:.3f}")

        if ep % 40 == 0:
            decode_first_n(model, dataset)

    decoded = decode_full_text(model, dataset)
    print("\n=== DECODED SAMPLE ===\n", decoded[:400])

    torch.save(
        dict(
            model_state_dict=model.state_dict(),
            config=config,
            stoi=dataset.stoi,
            itos=dataset.itos,
        ),
        os.path.join(here, "fractal_text_decoder_v3.pt")
    )


if __name__ == "__main__":
    main()