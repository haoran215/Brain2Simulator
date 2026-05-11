"""
learning_curve.py
=================
Build the unsupervised STDP learning curve: accuracy vs training-image
count. For each weight snapshot saved by `train_stdp.py`, we

  1. build the network with that snapshot's `W` frozen (no STDP),
  2. label-assign on a small training subset,
  3. evaluate on a small test subset,

and plot accuracy against `snapshot_steps`.

The eval subsets are intentionally small (default 400 label + 400 test) so
that the per-snapshot replay stays a few minutes; the final headline
accuracy is still produced separately by `eval_stdp.py` on the full set.

Usage
-----
    uv run python Classification-STDP/learning_curve.py --N 100

    # smaller subsets if you have many snapshots:
    uv run python Classification-STDP/learning_curve.py --N 100 \
        --n_label 200 --n_test 200

Output
------
Classification-STDP/learning_curve_N{N}.npz with keys:
    snapshot_steps       (S,)   image idx of each snapshot
    accuracy             (S,)   test accuracy at each snapshot
    dead_frac            (S,)   fraction of E neurons that fired 0 spikes
                                during label-assign
    assignment_entropy   (S,)   Shannon entropy (bits) of the per-class
                                assignment distribution; log2(10) ≈ 3.32
                                is uniform, lower means collapse

Classification-STDP/learning_curve_N{N}.png with three stacked panels.
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


def build_frozen(W: np.ndarray, theta: np.ndarray, cfg: dict, w_unit_amp: float):
    """Frozen-weight clone of the trained network. Cascade lives on each
    Synapses object; one exc and one inh per target group (Pattern A)."""
    N = W.shape[0]
    Cm = cfg.get('Cm', 10e-7)
    params = MSNParams(Cm=Cm)
    tau_s = cfg['tau_s']

    P = PoissonGroup(784, rates=np.zeros(784) * Hz, name='inp')
    G_E = make_msn(params=params, N=N, name='E')
    G_I = make_msn(params=params, N=N, name='I')
    G_E.I_0 = (-theta) * amp
    G_I.I_0 = 0 * amp

    def _cascade(target_var):
        return f"""
            dIs1/dt = -Is1 / tau_s1                : amp (clock-driven)
            dIs2/dt = (-Is2 + Is1) / tau_s2        : amp (clock-driven)
            {target_var}_post = Is2                : amp (summed)
            w : amp
        """

    ns = {'tau_s1': tau_s*second, 'tau_s2': tau_s*second}

    syn_in_e = Synapses(P, G_E, model=_cascade('I_exc'),
                        on_pre='Is1 += w', method='euler',
                        namespace=ns, name='syn_in_e')
    syn_in_e.connect(True)
    syn_i = np.array(syn_in_e.i[:], dtype=np.int64)
    syn_j = np.array(syn_in_e.j[:], dtype=np.int64)
    syn_in_e.w = (W[syn_j, syn_i] * w_unit_amp) * amp
    syn_in_e.Is1 = 0 * amp
    syn_in_e.Is2 = 0 * amp

    syn_e_i = Synapses(G_E, G_I, model=_cascade('I_exc'),
                       on_pre='Is1 += w', method='euler',
                       namespace=ns, name='syn_e_i')
    syn_e_i.connect(j='i')
    syn_e_i.w = cfg['w_e2i'] * amp
    syn_e_i.Is1 = 0 * amp
    syn_e_i.Is2 = 0 * amp

    syn_i_e = Synapses(G_I, G_E, model=_cascade('I_inh'),
                       on_pre='Is1 += w', method='euler',
                       namespace=ns, name='syn_i_e')
    syn_i_e.connect(condition='i != j')
    syn_i_e.w = cfg['w_i2e'] * amp
    syn_i_e.Is1 = 0 * amp
    syn_i_e.Is2 = 0 * amp

    sp_E = SpikeMonitor(G_E, name='sp_E')
    return Network(P, G_E, G_I, syn_in_e, syn_e_i, syn_i_e, sp_E), P, sp_E


def replay(net, P, sp_E, X, *, T_present, T_rest, lambda_max, N) -> np.ndarray:
    counts = np.zeros((len(X), N), dtype=np.int32)
    last = sp_E.count[:].copy()
    zero = np.zeros(784, dtype=np.float64)
    for i, x in enumerate(X):
        P.rates = (x * lambda_max) * Hz
        net.run(T_present * second)
        P.rates = zero * Hz
        net.run(T_rest * second)
        now = sp_E.count[:].copy()
        counts[i] = now - last
        last = now
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--N',          type=int, default=100)
    ap.add_argument('--weights',    default=None)
    ap.add_argument('--out_npz',    default=None)
    ap.add_argument('--out_fig',    default=None)
    ap.add_argument('--data',       default='Classification/data')
    ap.add_argument('--n_label',    type=int, default=400)
    ap.add_argument('--n_test',     type=int, default=400)
    ap.add_argument('--T_present',  type=float, default=0.35)
    ap.add_argument('--T_rest',     type=float, default=0.15)
    ap.add_argument('--seed',       type=int, default=2)
    ap.add_argument('--dt',         type=float, default=50e-6)
    ap.add_argument('--codegen',    default='numpy', choices=['numpy', 'cython'])
    args = ap.parse_args()

    if args.weights is None:
        args.weights = f'Classification-STDP/weights_N{args.N}.npz'
    if args.out_npz is None:
        args.out_npz = f'Classification-STDP/learning_curve_N{args.N}.npz'
    if args.out_fig is None:
        args.out_fig = f'Classification-STDP/learning_curve_N{args.N}.png'

    np.random.seed(args.seed)
    defaultclock.dt = args.dt * second
    if args.codegen != 'numpy':
        prefs.codegen.target = args.codegen

    blob = np.load(args.weights, allow_pickle=True)
    W_history = blob['W_history'].astype(np.float32)         # (S, N, 784)
    snap_steps = blob['snapshot_steps']
    theta = blob['theta'].astype(np.float64)
    cfg = json.loads(str(blob['config']))
    S = W_history.shape[0]
    N = W_history.shape[1]
    w_unit = cfg['w_unit']
    print(f"loaded W_history shape={W_history.shape} from {args.weights}")
    print(f"evaluating {S} snapshots × ({args.n_label} label + {args.n_test} test) images")

    X_lab, y_lab = load_mnist('train', args.n_label, args.data, args.seed)
    X_te,  y_te  = load_mnist('test',  args.n_test,  args.data, args.seed + 1)

    accuracy            = np.zeros(S, dtype=np.float32)
    dead_frac           = np.zeros(S, dtype=np.float32)
    assignment_entropy  = np.zeros(S, dtype=np.float32)

    for s in range(S):
        seed(args.seed + s)
        t0 = time.time()
        net, P, sp_E = build_frozen(W_history[s], theta, cfg, w_unit)
        net.store('init')

        # label-assign
        counts_lab = replay(net, P, sp_E, X_lab,
                            T_present=args.T_present, T_rest=args.T_rest,
                            lambda_max=cfg['lambda_max'], N=N)
        dead_frac[s] = float((counts_lab.sum(axis=0) == 0).mean())
        per_class = np.zeros((N, 10), dtype=np.float64)
        for c in range(10):
            mask = y_lab == c
            if mask.any():
                per_class[:, c] = counts_lab[mask].mean(axis=0)
        assigned = per_class.argmax(axis=1)
        hist = np.bincount(assigned, minlength=10).astype(np.float64)
        p = hist / max(hist.sum(), 1)
        nz = p[p > 0]
        assignment_entropy[s] = float(-(nz * np.log2(nz)).sum())

        # test
        net.restore('init')
        counts_te = replay(net, P, sp_E, X_te,
                           T_present=args.T_present, T_rest=args.T_rest,
                           lambda_max=cfg['lambda_max'], N=N)
        scores = np.zeros((len(X_te), 10), dtype=np.float64)
        for c in range(10):
            members = np.where(assigned == c)[0]
            scores[:, c] = (counts_te[:, members].sum(axis=1) / len(members)
                            if len(members) else -1.0)
        pred = scores.argmax(axis=1)
        accuracy[s] = float((pred == y_te).mean())

        print(f"  [snap {s+1:2d}/{S}  img {snap_steps[s]:6d}]  "
              f"acc={100*accuracy[s]:5.2f}%  "
              f"dead={dead_frac[s]:.0%}  "
              f"H={assignment_entropy[s]:.2f}/3.32 bits  "
              f"({time.time()-t0:.1f}s)")

    np.savez(args.out_npz,
             snapshot_steps     = snap_steps,
             accuracy           = accuracy,
             dead_frac          = dead_frac,
             assignment_entropy = assignment_entropy)
    print(f"[done] curve → {args.out_npz}")

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(snap_steps, 100 * accuracy, marker='o', lw=2)
    ax[0].axhline(10, ls='--', color='gray', lw=0.8, label='chance (10%)')
    ax[0].set_ylabel('test accuracy (%)')
    ax[0].set_title(f'STDP-MSN learning curve  —  N={N}  '
                    f'(quick eval: {args.n_label}+{args.n_test} images)')
    ax[0].legend(loc='lower right')
    ax[0].grid(alpha=0.3)
    ax[1].plot(snap_steps, 100 * dead_frac, marker='s', lw=2, color='tab:red')
    ax[1].set_ylabel('dead E neurons (%)')
    ax[1].grid(alpha=0.3)
    ax[2].plot(snap_steps, assignment_entropy, marker='^', lw=2, color='tab:green')
    ax[2].axhline(np.log2(10), ls='--', color='gray', lw=0.8,
                  label='uniform (log₂10 ≈ 3.32)')
    ax[2].set_ylabel('assignment entropy (bits)')
    ax[2].set_xlabel('training image #')
    ax[2].legend(loc='lower right')
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=120)
    print(f"[done] figure → {args.out_fig}")


if __name__ == '__main__':
    main()
