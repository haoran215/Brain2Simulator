"""
train_stdp.py
=============
Unsupervised STDP training of an MSN MNIST classifier (Diehl & Cook 2015,
adapted to the Wu et al. 2023 memristive spiking neuron).

Architecture
------------
    784 Poisson  ──[ W (N×784), plastic STDP ]──▶  N E-MSN
                                                     │  one-to-one fixed
                                                     ▼
                                                   N I-MSN
                                                     │  all-to-all-except-self fixed
                                                     ▼
                                                back to E (lateral inhibition)

Per-E-neuron homeostatic threshold θ is implemented as a slow per-image
update of the MSN tonic bias I_0: θ grows on each spike and decays between
images, and is fed in as a negative current. This avoids modifying
msn_neuron.py.

STDP rule (Diehl & Cook 2015 eq. 7 — LTP gated by recent presynaptic
activity, plus an unconditional w^μ decay on every postsynaptic spike):

    dapre/dt  = -apre /τ_pre        τ_pre  = 20 ms
    dapost/dt = -apost/τ_post       τ_post = 20 ms

    on pre :  apre  += 1
              w     -= η_pre * apost                        (post-trace LTD)

    on post:  apost += 1
              w     += η_post * apre - η_post * x_tar * w^μ
                       ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
                       LTP on        unconditional decay
                       active        — drives competitive
                       inputs        specialisation

Output
------
Classification-STDP/weights_N{N}.npz with keys:
    W                final weight matrix, shape (N, 784), dimensionless in [0, w_max]
    W_history        snapshots, shape (n_snapshots, N, 784)
    snapshot_steps   image index at which each snapshot was taken
    theta            final homeostatic θ (A), shape (N,)
    spike_history    per-snapshot mean firing rate (Hz)
    n_trained        number of training images shown
    config           dict of all hyperparameters
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
from brian2 import (
    Network, NeuronGroup, PoissonGroup, Synapses, SpikeMonitor,
    defaultclock, ms, us, second, Hz, amp, prefs, seed,
)

from msn_neuron import MSNParams, make_msn


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_mnist_train(n: int, data_dir: str, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[n,784] in [0,1], y[n]). Order is randomised by rng_seed."""
    from torchvision import datasets, transforms
    ds = datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(rng_seed)
    n = min(n, len(ds))
    idx = rng.permutation(len(ds))[:n]
    X = np.stack([ds[i][0].numpy().reshape(-1) for i in idx]).astype(np.float32)
    y = np.array([ds[i][1] for i in idx], dtype=np.int64)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────

def build_network(
    N: int,
    *,
    params: MSNParams,
    w_unit: float,        # A — amps per unit dimensionless w
    w_max: float,         # dimensionless ceiling
    eta_pre: float,
    eta_post: float,
    mu: float,
    x_tar: float,         # presynaptic-trace target for competitive LTP
    tau_pre: float,       # s
    tau_post: float,      # s
    w_e2i: float,         # A — fixed kick E→I
    w_i2e: float,         # A — fixed kick I→E (lateral)
    seed_init: int,
):
    """Build the network and return (Net, P, G_E, G_I, syn_in_e, sp_E, sp_I)."""
    rng = np.random.default_rng(seed_init)

    P = PoissonGroup(784, rates=np.zeros(784) * Hz, name='inp')
    G_E = make_msn(N, params=params, name='E')
    G_I = make_msn(N, params=params, name='I')
    G_E.I_0 = 0 * amp
    G_I.I_0 = 0 * amp

    stdp_model = '''
        w : 1
        dapre/dt  = -apre /tau_pre  : 1 (event-driven)
        dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    on_pre = '''
        Is1_exc_post += w * w_unit
        apre += 1
        w = clip(w - eta_pre * apost, 0, w_max)
    '''
    # Diehl & Cook 2015 eq. 7: LTP scaled by recent presynaptic activity,
    # plus an unconditional w^μ decay on every postsynaptic spike. The
    # decay term is what drives competitive specialisation — without it,
    # active synapses saturate to w_max and stay there.
    on_post = '''
        apost += 1
        w = clip(w + eta_post * apre - eta_post * x_tar * w**mu, 0, w_max)
    '''
    syn_in_e = Synapses(
        P, G_E,
        model=stdp_model, on_pre=on_pre, on_post=on_post,
        namespace=dict(
            tau_pre  = tau_pre  * second,
            tau_post = tau_post * second,
            w_unit   = w_unit   * amp,
            w_max    = w_max,
            eta_pre  = eta_pre,
            eta_post = eta_post,
            mu       = mu,
            x_tar    = x_tar,
        ),
        name='syn_in_e',
    )
    syn_in_e.connect(True)                          # 784 × N all-to-all
    syn_in_e.w = rng.uniform(0.0, 0.3, size=784 * N)

    syn_e_i = Synapses(G_E, G_I, on_pre=f'Is1_exc_post += {w_e2i}*amp', name='syn_e_i')
    syn_e_i.connect(j='i')                          # 1:1

    syn_i_e = Synapses(G_I, G_E, on_pre=f'Is1_inh_post += {w_i2e}*amp', name='syn_i_e')
    syn_i_e.connect(condition='i != j')             # all-but-self lateral

    sp_E = SpikeMonitor(G_E, name='sp_E')
    sp_I = SpikeMonitor(G_I, name='sp_I')

    net = Network(P, G_E, G_I, syn_in_e, syn_e_i, syn_i_e, sp_E, sp_I)
    return net, P, G_E, G_I, syn_in_e, sp_E, sp_I


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    seed(args.seed)
    np.random.seed(args.seed)
    defaultclock.dt = args.dt * second
    if args.codegen != 'numpy':
        prefs.codegen.target = args.codegen

    print(f"[config] N={args.N} dt={args.dt*1e6:.0f}µs  codegen={args.codegen}")

    # ── Data ──
    X, y = load_mnist_train(args.n_images, args.data, args.seed)
    print(f"[data] {len(X)} MNIST training images loaded")

    # ── Network ──
    # Cm controls max firing rate: f_max ≈ 1 / (Cm·(Rm_hi + Ra)). The
    # paper default Cm = 1 µF gives ~8 Hz, which doesn't generate enough
    # post-spikes per presentation for spike-driven STDP to work. We pick
    # Cm so f_max lands around the 100–200 Hz regime where the algorithm
    # was originally tuned.
    params = MSNParams(Cm=args.Cm, tau_s1=args.tau_s, tau_s2=args.tau_s)
    print(params.summary())

    net, P, G_E, G_I, syn_in_e, sp_E, sp_I = build_network(
        args.N,
        params      = params,
        w_unit      = args.w_unit,
        w_max       = args.w_max,
        eta_pre     = args.eta_pre,
        eta_post    = args.eta_post,
        mu          = args.mu,
        x_tar       = args.x_tar,
        tau_pre     = args.tau_pre,
        tau_post    = args.tau_post,
        w_e2i       = args.w_e2i,
        w_i2e       = args.w_i2e,
        seed_init   = args.seed,
    )

    # ── Mean-field sanity check ──
    mean_active = float((X > 0).sum(axis=1).mean())
    mean_pixel  = float(X[X > 0].mean()) if (X > 0).any() else 0.0
    init_w_mean = 0.15
    expected_I  = mean_active * mean_pixel * args.lambda_max * args.tau_s * init_w_mean * args.w_unit
    I_min, I_max = params.operating_window()
    print(f"[mean-field] expected <I_in> ≈ {expected_I*1e6:.2f} µA  "
          f"(window [{I_min*1e6:.1f}, {I_max*1e6:.0f}] µA)")
    if not (I_min < expected_I < I_max):
        print("  ⚠  expected current is OUTSIDE the MSN spiking window — "
              "consider tuning --w_unit or --lambda_max")

    # ── Homeostatic θ as Python-side bookkeeping ──
    theta = np.zeros(args.N)                        # in same units as w_unit (A)
    decay = float(np.exp(-(args.T_present + args.T_rest) / args.tau_theta))

    # ── Snapshot bookkeeping ──
    # Pull the source/target index arrays once; Brian2's internal storage
    # order is implementation-defined, so we always scatter via (j, i).
    syn_i = np.array(syn_in_e.i[:], dtype=np.int64)   # pixel index
    syn_j = np.array(syn_in_e.j[:], dtype=np.int64)   # E neuron index

    def snapshot_W() -> np.ndarray:
        W = np.zeros((args.N, 784), dtype=np.float32)
        W[syn_j, syn_i] = np.asarray(syn_in_e.w[:], dtype=np.float32)
        return W

    n_snapshots = max(2, args.n_images // args.snapshot_every + 2)
    W_history = np.zeros((n_snapshots, args.N, 784), dtype=np.float32)
    snap_steps = np.zeros(n_snapshots, dtype=np.int64)
    rate_history = np.zeros(n_snapshots, dtype=np.float32)
    snap_idx = 0
    W_history[snap_idx] = snapshot_W()
    snap_steps[snap_idx] = 0
    snap_idx += 1

    # ── Main loop ──
    print(f"[train] starting: {args.n_images} images × "
          f"({args.T_present*1e3:.0f}ms present + {args.T_rest*1e3:.0f}ms rest)")
    if args.norm_every > 0:
        print(f"[norm]  divisive L1 every {args.norm_every} images, "
              f"per-neuron sum target = {args.norm_target}")
    t_wall_0 = time.time()
    last_count_E = np.zeros(args.N, dtype=np.int64)
    cumulative_spikes = 0

    def renormalise():
        """Per-E-neuron L1 rescaling. Each row of W is scaled so its
        sum equals norm_target. This is what actually drives bimodality
        — STDP alone potentiates faster than its decay term can keep up
        in the MSN spike-rate regime, so weights saturate without it."""
        w_arr = np.asarray(syn_in_e.w[:], dtype=np.float64)
        W_mat = np.zeros((args.N, 784), dtype=np.float64)
        W_mat[syn_j, syn_i] = w_arr
        sums = W_mat.sum(axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-9)
        W_mat *= args.norm_target / sums
        np.clip(W_mat, 0.0, args.w_max, out=W_mat)
        syn_in_e.w[:] = W_mat[syn_j, syn_i]

    zero_rates = np.zeros(784, dtype=np.float64)
    for img_idx in range(args.n_images):
        x = X[img_idx]
        img_rates = x * args.lambda_max
        max_rescales = 3

        for attempt in range(max_rescales + 1):
            # Apply current θ as (negative) tonic bias on E
            G_E.I_0 = (-theta) * amp

            # Capture spike count at start
            counts_before = sp_E.count[:].copy()

            # Present
            P.rates = img_rates * Hz
            net.run(args.T_present * second)
            counts_after = sp_E.count[:].copy()
            n_spikes = int((counts_after - counts_before).sum())

            # Rest period — zero rates so traces decay, no learning happens
            P.rates = zero_rates * Hz
            net.run(args.T_rest * second)

            if n_spikes >= args.min_spikes_per_image or attempt == max_rescales:
                break
            img_rates = img_rates + 32.0      # bump and re-present

        # Update homeostatic θ from per-image spike counts
        spikes_this_image = sp_E.count[:].copy() - last_count_E
        last_count_E = sp_E.count[:].copy()
        theta = theta * decay + args.theta_plus * spikes_this_image
        cumulative_spikes += int(spikes_this_image.sum())

        # Divisive L1 weight normalisation
        if args.norm_every > 0 and (img_idx + 1) % args.norm_every == 0:
            renormalise()

        # Snapshot
        if (img_idx + 1) % args.snapshot_every == 0 and snap_idx < n_snapshots:
            W_history[snap_idx] = snapshot_W()
            snap_steps[snap_idx] = img_idx + 1
            mean_rate = cumulative_spikes / (args.N * (img_idx + 1) *
                                              (args.T_present + args.T_rest))
            rate_history[snap_idx] = mean_rate
            snap_idx += 1

        if (img_idx + 1) % args.log_every == 0:
            elapsed = time.time() - t_wall_0
            mean_rate = cumulative_spikes / (args.N * (img_idx + 1) *
                                              (args.T_present + args.T_rest))
            W_now = np.array(syn_in_e.w[:])
            print(f"[{img_idx+1:6d}/{args.n_images}]  "
                  f"<rate>={mean_rate:5.2f} Hz  "
                  f"<|w|>={W_now.mean():.3f}  "
                  f"frac@max={(W_now > 0.95*args.w_max).mean():.2%}  "
                  f"<θ>={theta.mean()*1e6:.2f} µA  "
                  f"elapsed={elapsed:6.1f}s")

    # Final snapshot if we missed it
    if snap_idx < n_snapshots:
        W_history[snap_idx] = snapshot_W()
        snap_steps[snap_idx] = args.n_images
        snap_idx += 1

    W_history    = W_history[:snap_idx]
    snap_steps   = snap_steps[:snap_idx]
    rate_history = rate_history[:snap_idx]

    # ── Save ──
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        W              = W_history[-1],
        W_history      = W_history,
        snapshot_steps = snap_steps,
        theta          = theta,
        spike_history  = rate_history,
        n_trained      = np.int64(args.n_images),
        config         = json.dumps(vars(args)),
    )
    print(f"\n[done] wallclock={time.time()-t_wall_0:.1f}s  "
          f"weights → {out}  (n_snapshots={len(W_history)})")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--N',              type=int,   default=100,
                    help='number of E neurons (also size of I)')
    ap.add_argument('--n_images',       type=int,   default=60_000,
                    help='number of training images to present')
    ap.add_argument('--data',           default='Classification/data')
    ap.add_argument('--out',            default=None,
                    help='default: Classification-STDP/weights_N{N}.npz')

    # presentation
    ap.add_argument('--T_present',      type=float, default=0.35)
    ap.add_argument('--T_rest',         type=float, default=0.15)
    ap.add_argument('--lambda_max',     type=float, default=63.75,
                    help='Hz, peak Poisson rate at pixel intensity 1.0')
    ap.add_argument('--min_spikes_per_image', type=int, default=5)

    # synapse / STDP
    ap.add_argument('--w_max',          type=float, default=1.0)
    ap.add_argument('--w_unit',         type=float, default=1.0e-7,
                    help='A per unit w (operating-point knob)')
    ap.add_argument('--eta_pre',        type=float, default=1e-4,
                    help='post-trace LTD on pre-spike')
    ap.add_argument('--eta_post',       type=float, default=1e-3,
                    help='LTP-and-decay magnitude on post-spike. Smaller '
                         'than canonical 1e-2 because at ~200 Hz post-rate '
                         'each presentation produces many more updates.')
    ap.add_argument('--mu',             type=float, default=0.2,
                    help='soft-bound exponent on w^μ in the decay term')
    ap.add_argument('--x_tar',          type=float, default=0.4,
                    help='target presynaptic trace — synapse equilibrium is '
                         'w^μ ≈ apre/x_tar; tune in concert with normalisation')
    ap.add_argument('--tau_pre',        type=float, default=20e-3)
    ap.add_argument('--tau_post',       type=float, default=20e-3)

    # divisive L1 normalisation — actually drives bimodality
    ap.add_argument('--norm_every',     type=int,   default=10,
                    help='renormalise input weights every K images '
                         '(0 disables; canonical Diehl-Cook trick)')
    ap.add_argument('--norm_target',    type=float, default=78.0,
                    help='per-E-neuron L1 weight sum target after rescaling')

    # WTA
    ap.add_argument('--w_e2i',          type=float, default=50e-6,
                    help='A; E→I 1:1 fixed kick (must drive I above I_min)')
    ap.add_argument('--w_i2e',          type=float, default=30e-6,
                    help='A; I→E lateral inhibitory kick')

    # homeostasis
    ap.add_argument('--theta_plus',     type=float, default=0.05e-6,
                    help='A added to θ per E spike. Scaled down vs. last '
                         'iteration because post-rate is now ~25× higher.')
    ap.add_argument('--tau_theta',      type=float, default=1e7,
                    help='s, slow homeostatic decay (Diehl & Cook value 1e7)')

    # MSN dynamics
    ap.add_argument('--Cm',             type=float, default=0.05e-6,
                    help='F, membrane capacitance. 1e-6 = paper default '
                         '(~8 Hz max); 0.05e-6 lifts max rate to ~200 Hz '
                         'so STDP gets enough post-spikes per presentation.')
    ap.add_argument('--tau_s',          type=float, default=200e-3)

    # bookkeeping
    ap.add_argument('--snapshot_every', type=int,   default=500)
    ap.add_argument('--log_every',      type=int,   default=100)
    ap.add_argument('--seed',           type=int,   default=0)
    ap.add_argument('--dt',             type=float, default=50e-6,
                    help='integrator timestep (s)')
    ap.add_argument('--codegen',        default='numpy',
                    choices=['numpy', 'cython'],
                    help="'cython' is much faster but needs a working compiler")

    args = ap.parse_args()
    if args.out is None:
        args.out = f'Classification-STDP/weights_N{args.N}.npz'
    train(args)


if __name__ == '__main__':
    main()
