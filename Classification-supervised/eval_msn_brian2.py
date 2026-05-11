"""
eval_msn_brian2.py
==================
Evaluate the trained 784→100 weight matrix on MNIST in the *real* Brian2
MSN simulator. Each test image is presented for T seconds; pixels drive
784 Poisson sources whose spikes feed 100 MSN neurons via signed synapses
(positive weights → I_exc inlet, negative → I_inh inlet). The 100 outputs
are partitioned into 10 groups of 10; the group with the highest spike
count wins.

Usage
-----
    python Classification-supervised/eval_msn_brian2.py --n 200
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brian2 import (
    Network, PoissonGroup, Synapses, SpikeMonitor,
    defaultclock, ms, us, second, Hz, amp, seed,
)

from msn_neuron import MSNParams, make_msn


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_mnist_test(n: int, data_dir: str, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[n,784] in [0,1], y[n] int) sampled from MNIST test set."""
    from torchvision import datasets, transforms
    ds = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    X = np.stack([ds[i][0].numpy().reshape(-1) for i in idx]).astype(np.float32)
    y = np.array([ds[i][1] for i in idx], dtype=np.int64)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────

def _signed_synapse(source, target, target_var: str, tau_s: float, name: str):
    """Build a Synapses object with the new cascade-on-synapse cascade,
    writing Is2 into a named (summed) inlet on the post group.
    Returns the Synapses (not yet connected; caller does syn.connect(...))."""
    model = f"""
        dIs1/dt = -Is1 / tau_s1                  : amp (clock-driven)
        dIs2/dt = (-Is2 + Is1) / tau_s2          : amp (clock-driven)
        {target_var}_post = Is2                  : amp (summed)
        w : amp
    """
    syn = Synapses(source, target,
                   model=model,
                   on_pre='Is1 += w',
                   method='euler',
                   namespace={'tau_s1': tau_s*second, 'tau_s2': tau_s*second},
                   name=name)
    return syn


def build_network(
    W: np.ndarray,                 # (N_HID, N_PIX), signed
    params: MSNParams,
    weight_scale: float,           # A per unit W
    tau_s: float = 200e-3,         # synaptic cascade τ (s)
) -> tuple[Network, PoissonGroup, SpikeMonitor]:
    """784 Poisson → 100 MSN.  W>0 routes to I_exc, W<0 to I_inh.

    One exc Synapses group and one inh Synapses group target G — Pattern A.
    """
    N_HID, N_PIX = W.shape

    P = PoissonGroup(N_PIX, rates=np.zeros(N_PIX) * Hz, name='inp')
    G = make_msn(params=params, N=N_HID, name='msn_out')
    G.I_0 = 0 * amp                # all drive comes from synaptic input

    # W[k, p]: pixel p → output k.  Source = pixel (i), target = output (j).
    pos = np.argwhere(W > 0)
    neg = np.argwhere(W < 0)

    syn_e = _signed_synapse(P, G, target_var='I_exc', tau_s=tau_s, name='syn_exc')
    syn_e.connect(i=pos[:, 1].astype(int), j=pos[:, 0].astype(int))
    syn_e.w = W[pos[:, 0], pos[:, 1]] * weight_scale * amp
    syn_e.Is1 = 0 * amp
    syn_e.Is2 = 0 * amp

    syn_i = _signed_synapse(P, G, target_var='I_inh', tau_s=tau_s, name='syn_inh')
    syn_i.connect(i=neg[:, 1].astype(int), j=neg[:, 0].astype(int))
    syn_i.w = (-W[neg[:, 0], neg[:, 1]]) * weight_scale * amp
    syn_i.Is1 = 0 * amp
    syn_i.Is2 = 0 * amp

    sp = SpikeMonitor(G, name='sp_out')

    net = Network(P, G, syn_e, syn_i, sp)
    return net, P, sp


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--weights',      default='Classification/weights.npz')
    ap.add_argument('--data',         default='Classification/data')
    ap.add_argument('--n',            type=int,   default=500)
    ap.add_argument('--T',            type=float, default=0.5)
    ap.add_argument('--lambda_max',   type=float, default=200.0)
    ap.add_argument('--weight_scale', type=float, default=5e-7)
    ap.add_argument('--tau_s',        type=float, default=200e-3,
                    help='synaptic cascade τ_s1 = τ_s2 (s)')
    ap.add_argument('--seed',         type=int,   default=0)
    ap.add_argument('--out',          default='Classification/eval_results.png')
    args = ap.parse_args()

    seed(args.seed)
    np.random.seed(args.seed)
    defaultclock.dt = 50 * us

    # ── Load weights ─────────────────────────────────────────────────────────
    blob       = np.load(args.weights)
    W          = blob['W'].astype(np.float32)
    N_HID      = int(blob['N_HID'])
    N_CLASSES  = int(blob['N_CLASSES'])
    PER_CLASS  = int(blob['PER_CLASS'])
    ann_acc    = float(blob['test_acc'])

    print(f"weights: {W.shape}  range=[{W.min():+.3f}, {W.max():+.3f}]  "
          f"mean|W|={np.abs(W).mean():.4f}")
    print(f"rate-proxy (PyTorch) ceiling: {100*ann_acc:.2f}%")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_mnist_test(args.n, args.data, args.seed)
    print(f"loaded {len(X)} MNIST test images")

    # ── Build network ─────────────────────────────────────────────────────────
    params = MSNParams()          # tau_s now lives on the synapse
    print(params.summary())

    net, P, sp = build_network(W, params, args.weight_scale, tau_s=args.tau_s)
    net.store('init')

    # Sanity check: typical input current to a hidden neuron under mean field.
    h_mean = (W @ X.mean(axis=0))
    I_mean = h_mean * args.lambda_max * args.tau_s * args.weight_scale
    I_min, I_max = params.operating_window()
    print(f"mean-field check: <I_in> ∈ [{I_mean.min()*1e6:+.1f}, "
          f"{I_mean.max()*1e6:+.1f}] µA   "
          f"(MSN window: [{I_min*1e6:.1f}, {I_max*1e6:.1f}] µA)")

    # ── Loop over test images ─────────────────────────────────────────────────
    print(f"\nrunning {args.n} trials × T_present={args.T*1e3:.0f} ms ...")
    T_run = args.T * second

    preds = np.zeros(args.n, dtype=np.int64)
    rates = np.zeros((args.n, N_HID), dtype=np.float32)

    t_start = time.time()
    for k, img in enumerate(X):
        net.restore('init')
        P.rates = img * args.lambda_max * Hz
        net.run(T_run)

        idx = np.asarray(sp.i[:])
        counts = np.bincount(idx, minlength=N_HID)
        rates[k] = counts / args.T
        grouped = counts.reshape(N_CLASSES, PER_CLASS).sum(axis=1)
        preds[k] = int(np.argmax(grouped))

        if (k + 1) % 20 == 0 or k == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / (k + 1) * (args.n - k - 1)
            running_acc = (preds[:k + 1] == y[:k + 1]).mean()
            print(f"  trial {k+1:4d}/{args.n}   "
                  f"running acc = {100*running_acc:5.2f}%   "
                  f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s")

    acc = (preds == y).mean()
    total = time.time() - t_start
    print(f"\nMSN classification accuracy: {100*acc:.2f}%   "
          f"(rate-proxy ceiling: {100*ann_acc:.2f}%)")
    print(f"total simulation time: {total:.1f} s "
          f"({total/args.n*1e3:.1f} ms/trial)")

    plot_results(y, preds, rates, args, acc, ann_acc, params)


def plot_results(y, preds, rates, args, acc, ann_acc, params):
    """Confusion matrix + mean group rate per true class."""
    N_CLASSES = 10
    PER_CLASS = rates.shape[1] // N_CLASSES

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for ti, pi in zip(y, preds):
        cm[ti, pi] += 1

    grouped_rate = rates.reshape(-1, N_CLASSES, PER_CLASS).mean(axis=2)
    rate_by_true = np.zeros((N_CLASSES, N_CLASSES))
    for c in range(N_CLASSES):
        m = (y == c)
        if m.any():
            rate_by_true[c] = grouped_rate[m].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xlabel('predicted'); ax.set_ylabel('true digit')
    ax.set_title(f'Confusion matrix  —  MSN accuracy = {100*acc:.2f}%')
    vmax = cm.max()
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > vmax/2 else 'black',
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045)

    ax = axes[1]
    im = ax.imshow(rate_by_true, cmap='magma')
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xlabel('output class group'); ax.set_ylabel('true digit')
    ax.set_title('Mean group firing rate (Hz)')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = rate_by_true[i, j]
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    color='white' if v < rate_by_true.max()*0.5 else 'black',
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, label='Hz')

    fig.suptitle(
        f'MSN rate-regime classifier — 100 neurons (10 × 10 per class)   '
        f'τ_s={args.tau_s*1e3:.0f} ms, T={args.T*1e3:.0f} ms, '
        f'λ_max={args.lambda_max:.0f} Hz, '
        f'w_scale={args.weight_scale*1e9:.0f} nA/unit   '
        f'(ceiling {100*ann_acc:.2f}%)',
        fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"figure → {args.out}")


if __name__ == '__main__':
    main()
