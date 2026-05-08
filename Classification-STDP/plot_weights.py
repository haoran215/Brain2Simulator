"""
plot_weights.py
===============
Receptive-field visualisation for the unsupervised STDP MSN classifier.

Produces:
  1. weights_static_N{N}.png   — 4-snapshot grid (0%, 25%, 50%, 100%) so
     you can see Gaussian noise resolve into digit prototypes.
  2. weights_anim_N{N}.{mp4|gif} — full animation across all snapshots.
  3. training_trace_N{N}.png   — mean |w|, frac@w_max, and mean firing
     rate vs training image index.

Usage
-----
    uv run python Classification-STDP/plot_weights.py --N 100
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def tile(W: np.ndarray) -> np.ndarray:
    """Tile (N, 784) into a (rows*28, cols*28) image, padded with NaN
    (rendered transparent by imshow with the chosen cmap.set_bad)."""
    N = W.shape[0]
    cols = int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))
    out = np.full((rows * 28, cols * 28), np.nan, dtype=np.float32)
    for k in range(N):
        r, c = divmod(k, cols)
        out[r*28:(r+1)*28, c*28:(c+1)*28] = W[k].reshape(28, 28)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--N',         type=int, default=100)
    ap.add_argument('--weights',   default=None)
    ap.add_argument('--out_dir',   default='Classification-STDP')
    ap.add_argument('--fps',       type=int, default=5)
    ap.add_argument('--gif_fallback', action='store_true',
                    help='write GIF even if ffmpeg is available')
    args = ap.parse_args()

    if args.weights is None:
        args.weights = f'Classification-STDP/weights_N{args.N}.npz'

    blob = np.load(args.weights, allow_pickle=True)
    W_history = blob['W_history'].astype(np.float32)         # (S, N, 784)
    snap_steps = blob['snapshot_steps']
    rate_history = blob['spike_history']
    cfg = json.loads(str(blob['config']))
    w_max = cfg['w_max']
    N = W_history.shape[1]
    S = W_history.shape[0]
    print(f"loaded W_history shape={W_history.shape} from {args.weights}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap('hot').copy()
    cmap.set_bad((0, 0, 0, 0))

    # ── Static figure ──
    pick = sorted(set([0, S // 4, S // 2, S - 1]))
    while len(pick) < 4:
        pick.append(min(pick[-1] + 1, S - 1))
        pick = sorted(set(pick))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    for k, s in enumerate(pick):
        ax[k].imshow(tile(W_history[s]), cmap=cmap, vmin=0, vmax=w_max,
                     interpolation='nearest')
        pct = 100 * snap_steps[s] / max(1, snap_steps[-1])
        ax[k].set_title(f'snapshot {s}/{S-1}  '
                        f'image {snap_steps[s]} ({pct:.0f}%)')
        ax[k].set_xticks([]); ax[k].set_yticks([])
    fig.suptitle(f'STDP receptive fields — N={N}')
    fig.tight_layout()
    static_path = out_dir / f'weights_static_N{N}.png'
    fig.savefig(static_path, dpi=120)
    plt.close(fig)
    print(f"[static] {static_path}")

    # ── Training trace ──
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    mean_w = W_history.mean(axis=(1, 2))
    frac_max = (W_history > 0.95 * w_max).mean(axis=(1, 2))
    ax[0].plot(snap_steps, mean_w, lw=2)
    ax[0].set_ylabel('<|w|>')
    ax[1].plot(snap_steps, frac_max, lw=2, color='tab:orange')
    ax[1].set_ylabel(f'frac w > 0.95 w_max')
    ax[2].plot(snap_steps, rate_history, lw=2, color='tab:green')
    ax[2].set_ylabel('mean E rate (Hz)')
    ax[2].set_xlabel('training image #')
    fig.suptitle(f'STDP training trace — N={N}')
    fig.tight_layout()
    trace_path = out_dir / f'training_trace_N{N}.png'
    fig.savefig(trace_path, dpi=120)
    plt.close(fig)
    print(f"[trace]  {trace_path}")

    # ── Animation ──
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(tile(W_history[0]), cmap=cmap, vmin=0, vmax=w_max,
                   interpolation='nearest')
    title = ax.set_title('')
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        im.set_data(tile(W_history[i]))
        title.set_text(f'image {snap_steps[i]} '
                       f'({100*snap_steps[i]/max(1,snap_steps[-1]):.0f}%)')
        return im, title

    anim = FuncAnimation(fig, update, frames=S, interval=1000//args.fps,
                         blit=False)
    try:
        if not args.gif_fallback:
            mp4_path = out_dir / f'weights_anim_N{N}.mp4'
            anim.save(mp4_path, writer=FFMpegWriter(fps=args.fps), dpi=120)
            print(f"[anim]  {mp4_path}")
        else:
            raise RuntimeError("forced gif")
    except Exception as e:
        gif_path = out_dir / f'weights_anim_N{N}.gif'
        anim.save(gif_path, writer=PillowWriter(fps=args.fps), dpi=100)
        print(f"[anim]  {gif_path}  (mp4 failed: {e})")
    plt.close(fig)


if __name__ == '__main__':
    main()
