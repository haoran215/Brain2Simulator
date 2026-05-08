"""
plot_raster.py
==============
Raster plot of the 100-neuron MSN classifier responding to one MNIST image
per digit (0-9). Useful for *seeing* the rate-coded readout: the 10-neuron
group whose row-band lights up most should match the input digit.

Reuses the Brian2 network from eval_msn_brian2.py (784 Poisson -> 100 MSN,
weights split into excitatory/inhibitory synapses).

Usage
-----
    uv run python Classification/plot_raster.py
    uv run python Classification/plot_raster.py --T 0.5 --lambda_max 200
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brian2 import (
    PoissonGroup, SpikeMonitor, Network,
    defaultclock, ms, us, second, Hz, amp, seed,
)

from msn_neuron import MSNParams
from eval_msn_brian2 import build_network  # type: ignore


def pick_one_per_class(data_dir: str, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[10,784], y[10]) — one MNIST test image per class 0..9."""
    from torchvision import datasets, transforms
    ds = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(rng_seed)
    labels = np.array([ds[i][1] for i in range(len(ds))])
    chosen_idx = []
    for c in range(10):
        cand = np.where(labels == c)[0]
        chosen_idx.append(int(rng.choice(cand)))
    X = np.stack([ds[i][0].numpy().reshape(-1) for i in chosen_idx]).astype(np.float32)
    y = np.array([ds[i][1] for i in chosen_idx], dtype=np.int64)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description='MSN classifier raster plot')
    ap.add_argument('--weights',      default='Classification/weights.npz')
    ap.add_argument('--data',         default='Classification/data')
    ap.add_argument('--T',            type=float, default=0.5)
    ap.add_argument('--lambda_max',   type=float, default=200.0)
    ap.add_argument('--weight_scale', type=float, default=5e-7)
    ap.add_argument('--tau_s',        type=float, default=200e-3)
    ap.add_argument('--seed',         type=int,   default=0)
    ap.add_argument('--out',          default='Classification/raster_per_digit.png')
    args = ap.parse_args()

    seed(args.seed)
    np.random.seed(args.seed)
    defaultclock.dt = 50 * us

    blob       = np.load(args.weights)
    W          = blob['W'].astype(np.float32)
    N_HID      = int(blob['N_HID'])
    N_CLASSES  = int(blob['N_CLASSES'])
    PER_CLASS  = int(blob['PER_CLASS'])

    X, y = pick_one_per_class(args.data, args.seed)
    print(f"selected one test image per class: y={y.tolist()}")

    params = MSNParams(tau_s1=args.tau_s, tau_s2=args.tau_s)
    net, P, sp = build_network(W, params, args.weight_scale)
    net.store('init')

    T_run = args.T * second

    spike_data = []   # list of (times[ms], indices) for each digit
    group_counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)  # rows: input digit, cols: group

    t_start = time.time()
    for k, img in enumerate(X):
        net.restore('init')
        P.rates = img * args.lambda_max * Hz
        net.run(T_run)

        t_spk = np.asarray(sp.t[:]) / (1e-3)        # seconds -> ms (Brian2 quantity)
        i_spk = np.asarray(sp.i[:])
        spike_data.append((t_spk.copy(), i_spk.copy()))

        counts = np.bincount(i_spk, minlength=N_HID)
        group_counts[k] = counts.reshape(N_CLASSES, PER_CLASS).sum(axis=1)
        pred = int(np.argmax(group_counts[k]))
        print(f"  digit {y[k]} -> predicted {pred}   "
              f"group spikes: {group_counts[k].tolist()}")
    print(f"sim total: {time.time()-t_start:.1f}s")

    plot(X, y, spike_data, group_counts, args, params)


def plot(X, y, spike_data, group_counts, args, params):  # noqa: ARG001
    N_CLASSES = 10
    PER_CLASS = 10
    N_HID     = 100
    T_ms      = args.T * 1e3

    cmap = plt.get_cmap('tab10')
    group_colors = [cmap(c) for c in range(N_CLASSES)]   # neuron k in group g = k // PER_CLASS

    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[1.0, 1.0, 1.15],
        width_ratios=[1, 1, 1, 1, 1, 1.25],
        hspace=0.55, wspace=0.35,
    )

    # ── Row 0 + 1: 10 raster panels (one per digit) ──────────────────────────
    raster_axes = []
    for k in range(N_CLASSES):
        r, c = divmod(k, 5)
        ax = fig.add_subplot(gs[r, c])
        raster_axes.append(ax)
        t_spk, i_spk = spike_data[k]

        # color each spike by the group of its neuron
        groups = (i_spk // PER_CLASS).astype(int)
        for g in range(N_CLASSES):
            m = (groups == g)
            if m.any():
                ax.scatter(t_spk[m], i_spk[m], s=3.0, marker='|',
                           color=group_colors[g], linewidths=0.8,
                           rasterized=True)

        # shade winning group's row-band
        pred = int(np.argmax(group_counts[k]))
        ax.axhspan(pred*PER_CLASS - 0.5,
                   (pred+1)*PER_CLASS - 0.5,
                   facecolor=group_colors[pred], alpha=0.10, zorder=-1)
        # shade true group's row-band (lighter outline)
        true = int(y[k])
        ax.axhline(true*PER_CLASS - 0.5,    color='k', lw=0.4, alpha=0.3)
        ax.axhline((true+1)*PER_CLASS - 0.5, color='k', lw=0.4, alpha=0.3)

        ok = '✓' if pred == true else '✗'
        ax.set_title(f'digit {true}  →  pred {pred} {ok}',
                     fontsize=10,
                     color=('darkgreen' if pred == true else 'crimson'))
        ax.set_xlim(0, T_ms)
        ax.set_ylim(-1, N_HID)
        ax.set_yticks([5 + g*PER_CLASS for g in range(N_CLASSES)])
        ax.set_yticklabels([str(g) for g in range(N_CLASSES)], fontsize=7)
        if c == 0:
            ax.set_ylabel('output group')
        if r == 1:
            ax.set_xlabel('time (ms)')
        ax.tick_params(axis='x', labelsize=7)

    # ── Right column on rows 0–1: small color legend / readout strip ─────────
    ax_leg = fig.add_subplot(gs[0:2, 5])
    ax_leg.set_axis_off()
    handles = []
    for g in range(N_CLASSES):
        handles.append(plt.Line2D([0], [0], marker='|', linestyle='',
                                  color=group_colors[g], markersize=10,
                                  markeredgewidth=2,
                                  label=f'group {g}  (neurons {g*PER_CLASS}-{(g+1)*PER_CLASS-1})'))
    ax_leg.legend(handles=handles, loc='center left', frameon=False,
                  title='Output group → class',
                  fontsize=9, title_fontsize=10)

    # ── Row 2 (left): summary heatmap of group spike counts per input ────────
    ax_hm = fig.add_subplot(gs[2, 0:3])
    im = ax_hm.imshow(group_counts, cmap='magma', aspect='auto')
    ax_hm.set_xticks(range(N_CLASSES)); ax_hm.set_xticklabels([str(c) for c in range(N_CLASSES)])
    ax_hm.set_yticks(range(N_CLASSES)); ax_hm.set_yticklabels([str(int(c)) for c in y])
    ax_hm.set_xlabel('output group (class)')
    ax_hm.set_ylabel('input digit')
    ax_hm.set_title(f'group spike counts over T = {T_ms:.0f} ms')
    vmax = group_counts.max()
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = group_counts[i, j]
            ax_hm.text(j, i, str(v), ha='center', va='center',
                       fontsize=8,
                       color='white' if v < vmax*0.5 else 'black')
        # mark winner with a yellow box
        jw = int(np.argmax(group_counts[i]))
        ax_hm.add_patch(plt.Rectangle((jw-0.5, i-0.5), 1, 1,
                                       fill=False, edgecolor='cyan', lw=1.6))
    fig.colorbar(im, ax=ax_hm, fraction=0.04, label='spikes')

    # ── Row 2 (right): the 10 input images ───────────────────────────────────
    ax_img = fig.add_subplot(gs[2, 3:6])
    ax_img.set_axis_off()
    grid = np.ones((28, 28*N_CLASSES + (N_CLASSES-1)*2)) * 0.0
    for k in range(N_CLASSES):
        x0 = k * (28 + 2)
        grid[:, x0:x0+28] = X[k].reshape(28, 28)
    ax_img.imshow(grid, cmap='gray_r', aspect='equal')
    ax_img.set_title('input images (one per digit)', fontsize=10)
    for k in range(N_CLASSES):
        x0 = k * (28 + 2) + 14
        ax_img.text(x0, 32, f'{y[k]}', ha='center', va='top', fontsize=10)

    fig.suptitle(
        f'MSN raster — 200 neurons (10×20 per class)   '
        f'T={T_ms:.0f} ms,  λ_max={args.lambda_max:.0f} Hz,  '
        f'τ_s={args.tau_s*1e3:.0f} ms,  '
        f'w_scale={args.weight_scale*1e9:.0f} nA/unit',
        fontsize=12, fontweight='bold', y=0.995)

    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"figure -> {args.out}")


if __name__ == '__main__':
    main()
