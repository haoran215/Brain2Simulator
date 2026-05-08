"""
eval_stdp.py
============
Post-hoc readout for the unsupervised STDP classifier.

Stage 1 (label-assign): freeze the trained 784→E weight matrix, replay a
labelled subset of the *training* set, count per-(neuron, class) spikes,
and assign each E neuron the digit class that fired it most.

Stage 2 (test): replay the test set, sum spikes per assigned class, and
report accuracy + confusion matrix. Labels are used only at this stage —
they never enter the weight update.

Usage
-----
    uv run python Classification-STDP/eval_stdp.py --N 100 \
        --n_label 10000 --n_test 10000

Output
------
Classification-STDP/eval_N{N}.npz with keys:
    assigned_label   shape (N,)   int — class assigned to each E neuron
    test_acc         float        accuracy on the test slice
    confusion        (10, 10)     row=true digit, col=predicted
Plus eval_N{N}.png with the confusion matrix.
"""

from __future__ import annotations

import argparse
import json
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
    defaultclock, ms, us, second, Hz, amp, prefs, seed,
)

from msn_neuron import MSNParams, make_msn


def load_mnist(split: str, n: int, data_dir: str, rng_seed: int):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(data_dir, train=(split == 'train'), download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(rng_seed)
    n = min(n, len(ds))
    idx = rng.permutation(len(ds))[:n]
    X = np.stack([ds[i][0].numpy().reshape(-1) for i in idx]).astype(np.float32)
    y = np.array([ds[i][1] for i in idx], dtype=np.int64)
    return X, y


def build_frozen_network(W: np.ndarray, theta: np.ndarray, cfg: dict):
    """Same topology as training but plastic synapses are frozen
    (model='w : amp', no on_post)."""
    N = W.shape[0]
    params = MSNParams(tau_s1=cfg['tau_s'], tau_s2=cfg['tau_s'])

    P = PoissonGroup(784, rates=np.zeros(784) * Hz, name='inp')
    G_E = make_msn(N, params=params, name='E')
    G_I = make_msn(N, params=params, name='I')
    G_E.I_0 = (-theta) * amp                     # carry over learned θ
    G_I.I_0 = 0 * amp

    syn_in_e = Synapses(P, G_E, model='w : amp',
                        on_pre='Is1_exc_post += w', name='syn_in_e')
    syn_in_e.connect(True)
    syn_i = np.array(syn_in_e.i[:], dtype=np.int64)
    syn_j = np.array(syn_in_e.j[:], dtype=np.int64)
    syn_in_e.w = (W[syn_j, syn_i] * cfg['w_unit']) * amp

    syn_e_i = Synapses(G_E, G_I, on_pre=f"Is1_exc_post += {cfg['w_e2i']}*amp",
                       name='syn_e_i')
    syn_e_i.connect(j='i')

    syn_i_e = Synapses(G_I, G_E, on_pre=f"Is1_inh_post += {cfg['w_i2e']}*amp",
                       name='syn_i_e')
    syn_i_e.connect(condition='i != j')

    sp_E = SpikeMonitor(G_E, name='sp_E')

    net = Network(P, G_E, G_I, syn_in_e, syn_e_i, syn_i_e, sp_E)
    return net, P, G_E, sp_E


def replay(net, P, sp_E, X, *, T_present, T_rest, lambda_max, N) -> np.ndarray:
    """Return spike counts shape (n_images, N), one row per image."""
    counts = np.zeros((len(X), N), dtype=np.int32)
    last = sp_E.count[:].copy()
    for i, x in enumerate(X):
        P.rates = (x * lambda_max) * Hz
        net.run(T_present * second)
        P.rates = np.zeros(784) * Hz
        net.run(T_rest * second)
        now = sp_E.count[:].copy()
        counts[i] = now - last
        last = now
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--N',          type=int, default=100)
    ap.add_argument('--weights',    default=None,
                    help='default: Classification-STDP/weights_N{N}.npz')
    ap.add_argument('--out',        default=None,
                    help='default: Classification-STDP/eval_N{N}.npz')
    ap.add_argument('--fig',        default=None,
                    help='default: Classification-STDP/eval_N{N}.png')
    ap.add_argument('--data',       default='Classification/data')
    ap.add_argument('--n_label',    type=int, default=10_000)
    ap.add_argument('--n_test',     type=int, default=10_000)
    ap.add_argument('--T_present',  type=float, default=0.35)
    ap.add_argument('--T_rest',     type=float, default=0.15)
    ap.add_argument('--seed',       type=int, default=1)
    ap.add_argument('--dt',         type=float, default=50e-6)
    ap.add_argument('--codegen',    default='numpy', choices=['numpy', 'cython'])
    args = ap.parse_args()

    if args.weights is None:
        args.weights = f'Classification-STDP/weights_N{args.N}.npz'
    if args.out is None:
        args.out = f'Classification-STDP/eval_N{args.N}.npz'
    if args.fig is None:
        args.fig = f'Classification-STDP/eval_N{args.N}.png'

    seed(args.seed)
    np.random.seed(args.seed)
    defaultclock.dt = args.dt * second
    if args.codegen != 'numpy':
        prefs.codegen.target = args.codegen

    blob = np.load(args.weights, allow_pickle=True)
    W      = blob['W'].astype(np.float32)              # (N, 784) dimensionless
    theta  = blob['theta'].astype(np.float64)
    cfg    = json.loads(str(blob['config']))
    N      = W.shape[0]
    print(f"loaded W:{W.shape} theta:{theta.shape} from {args.weights}")

    net, P, G_E, sp_E = build_frozen_network(W, theta, cfg)
    net.store('init')

    # ── Stage 1: label assignment on a labelled training subset ──
    print(f"[label] replaying {args.n_label} training images")
    X_lab, y_lab = load_mnist('train', args.n_label, args.data, args.seed)
    t0 = time.time()
    counts_lab = replay(net, P, sp_E, X_lab,
                        T_present=args.T_present, T_rest=args.T_rest,
                        lambda_max=cfg['lambda_max'], N=N)
    print(f"  replay wall={time.time()-t0:.1f}s  "
          f"<spikes/image>={counts_lab.sum(axis=1).mean():.1f}")

    # mean rate per (neuron, class)
    per_class = np.zeros((N, 10), dtype=np.float64)
    for c in range(10):
        mask = y_lab == c
        if mask.any():
            per_class[:, c] = counts_lab[mask].mean(axis=0)
    assigned_label = per_class.argmax(axis=1)
    print(f"[label] assignment histogram: "
          f"{np.bincount(assigned_label, minlength=10).tolist()}")

    # ── Stage 2: test ──
    print(f"[test] replaying {args.n_test} test images")
    net.restore('init')
    X_te, y_te = load_mnist('test', args.n_test, args.data, args.seed + 1)
    counts_te = replay(net, P, sp_E, X_te,
                       T_present=args.T_present, T_rest=args.T_rest,
                       lambda_max=cfg['lambda_max'], N=N)

    class_scores = np.zeros((len(X_te), 10), dtype=np.float64)
    for c in range(10):
        members = np.where(assigned_label == c)[0]
        if len(members):
            class_scores[:, c] = counts_te[:, members].sum(axis=1) / len(members)
        else:
            class_scores[:, c] = -1.0    # no neuron assigned this digit
    pred = class_scores.argmax(axis=1)
    acc = float((pred == y_te).mean())
    print(f"[test] accuracy = {100*acc:.2f}%")

    confusion = np.zeros((10, 10), dtype=np.int32)
    for t, p in zip(y_te, pred):
        confusion[t, p] += 1

    out = pathlib.Path(args.out)
    np.savez(out,
             assigned_label = assigned_label,
             test_acc       = np.float32(acc),
             confusion      = confusion,
             counts_test    = counts_te,
             y_test         = y_te)
    print(f"[done] eval → {out}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    im = ax[0].imshow(confusion, cmap='viridis')
    ax[0].set_title(f'STDP-MSN N={N}   test acc {100*acc:.1f}%')
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('true')
    ax[0].set_xticks(range(10)); ax[0].set_yticks(range(10))
    plt.colorbar(im, ax=ax[0])
    ax[1].bar(range(10), np.bincount(assigned_label, minlength=10))
    ax[1].set_title('neurons per assigned digit')
    ax[1].set_xlabel('digit')
    ax[1].set_ylabel('# E-neurons')
    ax[1].set_xticks(range(10))
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print(f"[done] figure → {args.fig}")


if __name__ == '__main__':
    main()
