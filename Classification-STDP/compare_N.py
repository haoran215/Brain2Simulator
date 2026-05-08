"""
compare_N.py
============
Drive the full STDP pipeline for several network sizes (default
N ∈ {50, 100, 150}) and aggregate the results into a single comparison
figure + JSON.

Pipeline per N:
    1. python train_stdp.py    --N N --n_images <n_images> ...
    2. python eval_stdp.py     --N N --n_label <n_label> --n_test <n_test>
    3. python plot_weights.py  --N N
    4. python plot_connectome.py --N N

Aggregation:
    Classification-STDP/comparison.json  per-N {test_acc, train_seconds,
                                                 mean_w, frac_w_max,
                                                 receptive_field_corr}
    Classification-STDP/comparison.png   bar charts of all of the above

Usage
-----
    uv run python Classification-STDP/compare_N.py \
        --Ns 50 100 150 --n_images 60000 --n_label 10000 --n_test 10000

This is an *orchestrator* — it shells out to the per-step scripts so
each script remains independently runnable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_step(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"step failed (exit {res.returncode}): {' '.join(cmd)}")


def receptive_field_corr(W: np.ndarray, assigned: np.ndarray,
                         data_dir: str) -> float:
    """Mean Pearson correlation between each E neuron's normalised weight
    vector and the average MNIST image of its assigned digit. A crude
    'how digit-like is the receptive field' score in [-1, 1]."""
    from torchvision import datasets, transforms
    ds = datasets.MNIST(data_dir, train=True, download=False,
                        transform=transforms.ToTensor())
    digit_means = np.zeros((10, 784), dtype=np.float32)
    counts = np.zeros(10, dtype=np.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        digit_means[y] += x.numpy().reshape(-1)
        counts[y] += 1
        if i >= 10_000:                  # 10k is plenty for a mean image
            break
    digit_means /= np.maximum(counts[:, None], 1)
    corrs = []
    for k in range(W.shape[0]):
        a = W[k] - W[k].mean()
        b = digit_means[assigned[k]] - digit_means[assigned[k]].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        corrs.append(float((a @ b) / denom) if denom > 1e-9 else 0.0)
    return float(np.mean(corrs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ns',         type=int, nargs='+', default=[50, 100, 150])
    ap.add_argument('--n_images',   type=int, default=60_000)
    ap.add_argument('--n_label',    type=int, default=10_000)
    ap.add_argument('--n_test',     type=int, default=10_000)
    ap.add_argument('--data',       default='Classification/data')
    ap.add_argument('--codegen',    default='numpy', choices=['numpy', 'cython'])
    ap.add_argument('--curve_n',    type=int, default=400,
                    help='per-snapshot label/test subset size for learning curves')
    ap.add_argument('--out_json',   default='Classification-STDP/comparison.json')
    ap.add_argument('--out_fig',    default='Classification-STDP/comparison.png')
    ap.add_argument('--out_curves', default='Classification-STDP/comparison_curves.png')
    ap.add_argument('--skip_train', action='store_true',
                    help='reuse existing weights_N{N}.npz files')
    ap.add_argument('--skip_eval',  action='store_true',
                    help='reuse existing eval_N{N}.npz files')
    args = ap.parse_args()

    py = sys.executable
    results = {}
    for N in args.Ns:
        wfile = pathlib.Path(f'Classification-STDP/weights_N{N}.npz')
        efile = pathlib.Path(f'Classification-STDP/eval_N{N}.npz')

        if not args.skip_train or not wfile.exists():
            t0 = time.time()
            run_step([py, 'Classification-STDP/train_stdp.py',
                      '--N', str(N), '--n_images', str(args.n_images),
                      '--data', args.data, '--codegen', args.codegen])
            train_seconds = time.time() - t0
        else:
            train_seconds = float('nan')
            print(f"[skip] reusing {wfile}")

        if not args.skip_eval or not efile.exists():
            run_step([py, 'Classification-STDP/eval_stdp.py',
                      '--N', str(N), '--n_label', str(args.n_label),
                      '--n_test', str(args.n_test),
                      '--data', args.data, '--codegen', args.codegen])

        run_step([py, 'Classification-STDP/plot_weights.py',    '--N', str(N)])
        run_step([py, 'Classification-STDP/plot_connectome.py', '--N', str(N)])
        run_step([py, 'Classification-STDP/learning_curve.py',  '--N', str(N),
                  '--n_label', str(args.curve_n), '--n_test', str(args.curve_n),
                  '--data', args.data, '--codegen', args.codegen])

        wblob = np.load(wfile, allow_pickle=True)
        eblob = np.load(efile, allow_pickle=True)
        W = wblob['W'].astype(np.float32)
        cfg = json.loads(str(wblob['config']))
        results[N] = {
            'test_acc':       float(eblob['test_acc']),
            'train_seconds':  train_seconds,
            'mean_w':         float(W.mean()),
            'frac_w_max':     float((W > 0.95 * cfg['w_max']).mean()),
            'rf_corr':        receptive_field_corr(W, eblob['assigned_label'],
                                                   args.data),
            'assigned_hist':  np.bincount(eblob['assigned_label'],
                                          minlength=10).tolist(),
        }
        print(f"\n[N={N}]  acc={100*results[N]['test_acc']:.2f}%  "
              f"<|w|>={results[N]['mean_w']:.3f}  "
              f"frac@max={results[N]['frac_w_max']:.2%}  "
              f"rf_corr={results[N]['rf_corr']:.3f}  "
              f"train={results[N]['train_seconds']:.0f}s")

    # ── Persist ──
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Comparison figure ──
    Ns = sorted(results.keys())
    metrics = [('test_acc',      'test accuracy',   lambda v: 100 * v, '%'),
               ('rf_corr',       'mean RF corr',    lambda v: v,      ''),
               ('frac_w_max',    'frac w @ w_max',  lambda v: 100 * v, '%'),
               ('train_seconds', 'train wallclock', lambda v: v / 60,  'min')]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for k, (key, title, transform, unit) in enumerate(metrics):
        a = ax[k // 2, k % 2]
        vals = [transform(results[N][key]) for N in Ns]
        a.bar([str(N) for N in Ns], vals, color='tab:blue', alpha=0.85)
        for x, v in enumerate(vals):
            a.text(x, v, f'{v:.2f}{unit}', ha='center', va='bottom', fontsize=9)
        a.set_title(f'{title}')
        a.set_xlabel('N (E-neurons)')
        a.set_ylabel(unit if unit else title)
    fig.suptitle('STDP-MSN classifier — N comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=120)

    # ── Combined learning-curve overlay ──
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.8))
    palette = plt.get_cmap('viridis')(np.linspace(0.2, 0.85, len(Ns)))
    for color, N in zip(palette, Ns):
        cpath = pathlib.Path(f'Classification-STDP/learning_curve_N{N}.npz')
        if not cpath.exists():
            print(f"[warn] missing {cpath}; skipping in overlay")
            continue
        cblob = np.load(cpath)
        ax2[0].plot(cblob['snapshot_steps'], 100 * cblob['accuracy'],
                    marker='o', lw=2, color=color, label=f'N={N}')
        ax2[1].plot(cblob['snapshot_steps'], cblob['assignment_entropy'],
                    marker='^', lw=2, color=color, label=f'N={N}')
    ax2[0].axhline(10, ls='--', color='gray', lw=0.8, label='chance (10%)')
    ax2[0].set_xlabel('training image #'); ax2[0].set_ylabel('test accuracy (%)')
    ax2[0].set_title('Learning curves')
    ax2[0].legend(); ax2[0].grid(alpha=0.3)
    ax2[1].axhline(np.log2(10), ls='--', color='gray', lw=0.8, label='uniform')
    ax2[1].set_xlabel('training image #')
    ax2[1].set_ylabel('assignment entropy (bits)')
    ax2[1].set_title('Specialisation across digits')
    ax2[1].legend(); ax2[1].grid(alpha=0.3)
    fig2.suptitle('STDP-MSN learning curves — N comparison', fontsize=13)
    fig2.tight_layout()
    fig2.savefig(args.out_curves, dpi=120)
    print(f"\n[done] {args.out_json}\n[done] {args.out_fig}\n[done] {args.out_curves}")


if __name__ == '__main__':
    main()
