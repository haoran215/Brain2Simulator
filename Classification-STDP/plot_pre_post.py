"""
plot_pre_post.py
================
Side-by-side connection diagram of the network BEFORE and AFTER training,
driven by the W_history snapshots in a weights_*.npz file.

Layout
------
Two panels (PRE | POST):
    left:   28×28 input pixel grid (one dot per pixel)
    right:  vertical column of K sampled E neurons
    lines:  per-neuron, "active" pixel → neuron (alpha ∝ weight)

The K neurons are picked as the ones whose weight vector changed the
most between the first and last snapshot — those carry the visible signal
of what STDP/BSF learned.

Usage
-----
    uv run python Classification-STDP/plot_pre_post.py \
        --weights Classification-STDP/weights_bsf_N100.npz \
        --K 12
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw_panel(ax, W_pixel_to_E, K_idx, *, title, w_thresh, line_color):
    """W_pixel_to_E: (N, 784) row-normalised so max=1 per row.
    K_idx: indices of N E neurons to draw."""
    K = len(K_idx)

    # ── pixel grid: 28×28 dots, x∈[0, 1], y∈[0, 1] ────────────────────────
    pix_x = np.repeat(np.linspace(0.05, 0.40, 28), 28)
    pix_y = np.tile(np.linspace(0.95, 0.05, 28), 28)
    ax.scatter(pix_x, pix_y, s=3, color='#888', alpha=0.55, zorder=2)

    # ── E neuron column: x=0.85, y spread evenly ──────────────────────────
    e_x = np.full(K, 0.85)
    e_y = np.linspace(0.92, 0.08, K)
    ax.scatter(e_x, e_y, s=160, facecolor='#FFB3B3', edgecolor='k',
               lw=1.2, zorder=4)
    for k, (yk, idx) in enumerate(zip(e_y, K_idx)):
        ax.text(0.93, yk, f'E{idx}', fontsize=8, va='center')

    # ── connection lines, alpha ∝ weight ──────────────────────────────────
    for k, idx in enumerate(K_idx):
        w_row = W_pixel_to_E[idx]                  # (784,)
        active = np.where(w_row > w_thresh)[0]
        if len(active) == 0:
            continue
        # Cap drawn lines per neuron to keep render light.
        if len(active) > 60:
            top = np.argsort(w_row[active])[-60:]
            active = active[top]
        for p in active:
            alpha = float(min(1.0, w_row[p])) ** 1.4
            ax.plot([pix_x[p], e_x[k]],
                    [pix_y[p], e_y[k]],
                    color=line_color, lw=0.5, alpha=alpha * 0.6, zorder=3)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    # legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
               markersize=4, label='784 input pixels'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFB3B3',
               markeredgecolor='k', markersize=10, label='E neurons (sample)'),
        Line2D([0], [0], color=line_color, lw=1.2,
               label='plastic synapse  (alpha ∝ w)'),
    ]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9, frameon=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--K',       type=int, default=12,
                    help='number of E neurons to sample (most-changed first)')
    ap.add_argument('--w_thresh', type=float, default=0.10,
                    help='draw lines only for w > thresh*max(w)')
    ap.add_argument('--fig',     default=None)
    args = ap.parse_args()

    if args.fig is None:
        stem = pathlib.Path(args.weights).stem.replace('weights_', '')
        args.fig = f'Classification-STDP/pre_post_{stem}.png'

    blob = np.load(args.weights, allow_pickle=True)
    W_history = blob['W_history']                 # (n_snap, N, 784)
    cfg = json.loads(str(blob['config']))
    rule = 'BSF' if 'w_jump' in cfg else 'pair-STDP'
    snap_steps = blob['snapshot_steps']
    n_trained = int(blob['n_trained']) if 'n_trained' in blob.files else snap_steps[-1]
    N = W_history.shape[1]

    W_pre  = W_history[0].astype(np.float32)
    W_post = W_history[-1].astype(np.float32)

    # Normalise each panel by its own max so colour scales are visually
    # comparable even if w_pre and w_post have very different magnitudes.
    def row_norm(W):
        m = W.max(axis=1, keepdims=True)
        m = np.where(m > 1e-9, m, 1.0)
        return W / m

    W_pre_n  = row_norm(W_pre)
    W_post_n = row_norm(W_post)

    # Pick K neurons with the largest L2 change between pre and post.
    delta = np.linalg.norm(W_post - W_pre, axis=1)
    K_idx = np.argsort(delta)[::-1][:args.K]
    K_idx = np.sort(K_idx)        # render in ascending neuron order

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    draw_panel(axes[0], W_pre_n, K_idx,
               title=f'PRE-train  (image 0 / {n_trained})',
               w_thresh=args.w_thresh,
               line_color='#888')
    draw_panel(axes[1], W_post_n, K_idx,
               title=f'POST-train  (image {n_trained})',
               w_thresh=args.w_thresh,
               line_color='#1A6FB0')

    fig.suptitle(
        f'Network connectivity before vs. after training '
        f'({rule}, N={N}, showing top-{args.K} most-changed E neurons)',
        fontsize=13, fontweight='bold', y=1.0)

    fig.tight_layout()
    fig.savefig(args.fig, dpi=140, bbox_inches='tight')
    print(f"[done] pre/post diagram → {args.fig}")

    # ── small companion: per-sampled-neuron 28×28 receptive field, pre+post,
    # and the delta. This is the "what actually changed" view.
    fig2, axes2 = plt.subplots(3, args.K, figsize=(1.4 * args.K, 4.4))
    for c, idx in enumerate(K_idx):
        rf_pre  = W_pre[idx].reshape(28, 28)
        rf_post = W_post[idx].reshape(28, 28)
        rf_d    = rf_post - rf_pre
        for r, (rf, cmap, vlim) in enumerate([
                (rf_pre,  'hot',     (0, max(rf_pre.max(), 1e-9))),
                (rf_post, 'hot',     (0, max(rf_post.max(), 1e-9))),
                (rf_d,    'RdBu_r',  (-np.abs(rf_d).max() or 1e-9,
                                       np.abs(rf_d).max() or 1e-9)),
        ]):
            ax = axes2[r, c] if args.K > 1 else axes2[r]
            ax.imshow(rf, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(['pre', 'post', 'Δ'][r], fontsize=10,
                              rotation=0, labelpad=18, va='center')
            if r == 0:
                ax.set_title(f'E{idx}', fontsize=8)

    fig2.suptitle(
        f'Receptive fields of the {args.K} most-changed E neurons '
        f'(rows: pre / post / Δ)', fontsize=11, fontweight='bold')
    fig2.tight_layout()
    fig2_path = args.fig.replace('.png', '_RFs.png')
    fig2.savefig(fig2_path, dpi=140, bbox_inches='tight')
    print(f"[done] RF strip → {fig2_path}")


if __name__ == '__main__':
    main()
