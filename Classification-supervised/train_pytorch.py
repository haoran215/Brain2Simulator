"""
train_pytorch.py
================
Train a rate-regime SNN proxy on MNIST and save weights for transfer to the
Brian2 MSN evaluator.

Architecture (matches the 200-neuron rate-coded MSN classifier):

    784 input "rates"  ──[ W (200x784) ]──>  200 hidden "rates"
                                              │
                                              ▼  partition into 10 groups of 20
                                            (group-mean rate per class)
                                              │
                                              ▼  argmax → predicted digit

The hidden activation is the differentiable rate proxy of one MSN neuron:

    rate(I) = F_MAX · sigmoid(I)         [Hz]

where F_MAX ≈ 8 Hz is the saturation rate of Wu et al. 2023's MSN. This
captures the bounded, monotone shape of the MSN I-F curve in the spiking
window. Surrogate-gradient training in PyTorch then optimizes weights for
the *group-mean rate* readout — exactly what the Brian2 simulator counts.

Output
------
Classification/weights.npz with keys:
    W           (200, 784)  float32 — synaptic weight matrix (signed)
    F_MAX                    float  — Hz, used for rate-to-current conversion
    train_acc, test_acc      float  — sanity check on transfer ceiling
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


N_PIX        = 28 * 28
N_HID        = 200
N_CLASSES    = 10
PER_CLASS    = N_HID // N_CLASSES   # 20 neurons per class

F_MAX        = 8.0    # Hz — MSN saturation rate (depol-block onset)
TRAIN_TEMP   = 0.3    # softmax temperature; CE on rates needs sharpening
                      # because rates are bounded in [0, F_MAX].


class RateSNN(nn.Module):
    """Differentiable rate proxy of the 200-neuron MSN classifier.

    No bias term: the bias would map onto a tonic current I_0 in the MSN,
    which complicates the operating-point analysis. We let the synaptic
    weights carry the full discriminative signal.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_PIX, N_HID, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h     = self.fc(x.view(-1, N_PIX))            # (B, 200)  pre-activation
        rates = F_MAX * torch.sigmoid(h)              # (B, 200)  in Hz
        # group-average readout: mean firing rate per class group of 20
        logits = rates.view(-1, N_CLASSES, PER_CLASS).mean(dim=2)   # (B, 10)
        return logits, rates


def evaluate(net: RateSNN, loader: DataLoader, device: str) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = net(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.numel()
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--epochs',   type=int,   default=10)
    ap.add_argument('--batch',    type=int,   default=128)
    ap.add_argument('--lr',       type=float, default=3e-3)
    ap.add_argument('--seed',     type=int,   default=0)
    ap.add_argument('--data',     default='Classification/data')
    ap.add_argument('--out',      default='Classification/weights.npz')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {device}")

    tfm = transforms.ToTensor()      # pixels in [0, 1]
    train_ds = datasets.MNIST(args.data, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(args.data, train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=512,         shuffle=False, num_workers=2)

    net = RateSNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"\n{'epoch':>5}  {'loss':>8}  {'train':>7}  {'test':>7}  {'time':>6}")
    print('-' * 44)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits, _ = net(x)
            loss = F.cross_entropy(logits / TRAIN_TEMP, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
        running /= len(train_ds)

        train_acc = evaluate(net, train_dl, device)
        test_acc  = evaluate(net, test_dl,  device)
        print(f"{ep:5d}  {running:8.4f}  {100*train_acc:6.2f}%  {100*test_acc:6.2f}%  "
              f"{time.time()-t0:5.1f}s")

    # ── Save weights ─────────────────────────────────────────────────────────
    W = net.fc.weight.detach().cpu().numpy().astype(np.float32)   # (200, 784)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             W           = W,
             F_MAX       = np.float32(F_MAX),
             N_HID       = np.int32(N_HID),
             N_CLASSES   = np.int32(N_CLASSES),
             PER_CLASS   = np.int32(PER_CLASS),
             test_acc    = np.float32(test_acc),
             train_acc   = np.float32(train_acc))
    print(f"\nweights saved → {out_path}")
    print(f"  W: shape={W.shape}  range=[{W.min():.3f}, {W.max():.3f}]  "
          f"mean|W|={np.abs(W).mean():.4f}")
    print(f"  rate-proxy test acc: {100*test_acc:.2f}%   "
          f"(this is the conversion ceiling for the MSN run)")


if __name__ == '__main__':
    main()
