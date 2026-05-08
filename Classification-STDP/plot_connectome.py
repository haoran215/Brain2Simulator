"""
plot_connectome.py
==================
4-panel topology figure for the unsupervised STDP MSN classifier:

  (a) Input → E weight matrix  (N × 784)   — rows sorted by assigned digit
  (b) E → I 1:1 identity matrix
  (c) I → E lateral inhibition (all-but-self)
  (d) Bar chart of how many E neurons were assigned to each digit

Usage
-----
    uv run python Classification-STDP/plot_connectome.py --N 100
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--N',       type=int, default=100)
    ap.add_argument('--weights', default=None)
    ap.add_argument('--eval',    default=None)
    ap.add_argument('--fig',     default=None)
    args = ap.parse_args()

    if args.weights is None:
        args.weights = f'Classification-STDP/weights_N{args.N}.npz'
    if args.eval is None:
        args.eval = f'Classification-STDP/eval_N{args.N}.npz'
    if args.fig is None:
        args.fig = f'Classification-STDP/connectome_N{args.N}.png'

    blob = np.load(args.weights, allow_pickle=True)
    W = blob['W'].astype(np.float32)            # (N, 784)
    cfg = json.loads(str(blob['config']))
    w_max = cfg['w_max']
    w_e2i = cfg['w_e2i']
    w_i2e = cfg['w_i2e']
    N = W.shape[0]

    eval_blob = np.load(args.eval, allow_pickle=True)
    assigned = eval_blob['assigned_label']
    test_acc = float(eval_blob['test_acc'])

    order = np.argsort(assigned, kind='stable')
    W_sorted = W[order]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

    # (a) Input → E
    ax_a = fig.add_subplot(gs[0, :2])
    im = ax_a.imshow(W_sorted, aspect='auto', cmap='hot',
                     vmin=0, vmax=w_max, interpolation='nearest')
    ax_a.set_title(f'(a) Input → E  weight matrix  ({N}×784, sorted by assigned digit)')
    ax_a.set_xlabel('pixel index (0–783)')
    ax_a.set_ylabel('E neuron (sorted)')
    boundaries = np.where(np.diff(np.sort(assigned)))[0] + 1
    for b in boundaries:
        ax_a.axhline(b - 0.5, color='cyan', lw=0.6, alpha=0.7)
    plt.colorbar(im, ax=ax_a, fraction=0.025)

    # (b) E → I  (1:1 identity)
    ax_b = fig.add_subplot(gs[0, 2])
    EtoI = np.eye(N, dtype=np.float32) * w_e2i * 1e6   # µA
    im_b = ax_b.imshow(EtoI, cmap='Reds', vmin=0,
                       vmax=max(w_e2i, 1e-9) * 1e6, interpolation='nearest')
    ax_b.set_title(f'(b) E → I (1:1)\nfixed kick = {w_e2i*1e6:.1f} µA')
    ax_b.set_xlabel('I neuron')
    ax_b.set_ylabel('E neuron')
    plt.colorbar(im_b, ax=ax_b, fraction=0.04)

    # (c) I → E  (all-but-self lateral)
    ax_c = fig.add_subplot(gs[1, 0])
    ItoE = w_i2e * 1e6 * (1 - np.eye(N, dtype=np.float32))   # µA
    im_c = ax_c.imshow(ItoE, cmap='Blues', vmin=0,
                       vmax=max(w_i2e, 1e-9) * 1e6, interpolation='nearest')
    ax_c.set_title(f'(c) I → E lateral inhibition\nfixed kick = {w_i2e*1e6:.1f} µA')
    ax_c.set_xlabel('E neuron')
    ax_c.set_ylabel('I neuron')
    plt.colorbar(im_c, ax=ax_c, fraction=0.04)

    # (d) Assignment histogram
    ax_d = fig.add_subplot(gs[1, 1:])
    counts = np.bincount(assigned, minlength=10)
    ax_d.bar(np.arange(10), counts, color='tab:purple', alpha=0.85)
    for i, c in enumerate(counts):
        ax_d.text(i, c + 0.1, str(int(c)), ha='center', fontsize=10)
    ax_d.set_xticks(range(10))
    ax_d.set_xlabel('assigned digit')
    ax_d.set_ylabel('# E neurons')
    ax_d.set_title(f'(d) E neurons per digit  —  test accuracy = {100*test_acc:.1f}%')

    fig.suptitle(f'STDP-MSN connectome  (N={N}, '
                 f'plastic input + fixed E↔I WTA)', y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120, bbox_inches='tight')
    print(f"[done] connectome → {args.fig}")


if __name__ == '__main__':
    main()
