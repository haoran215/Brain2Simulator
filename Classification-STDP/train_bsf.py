"""
train_bsf.py
============
Unsupervised Brader-Senn-Fusi (2007) "stop-learning" training of an MSN
MNIST classifier. Same network as `train_stdp.py` (784 Poisson → N E-MSN
↔ N I-MSN with WTA + homeostatic θ); the only thing that changes is the
plastic input synapse rule.

BSF vs pair-STDP — why this is the right rule for an on-chip story
------------------------------------------------------------------
* Each plastic synapse carries a continuous internal X ∈ [0, X_max] but
  the **delivered current is binary**: w_eff = w_jump · 𝟙[X > θ_X].
  This is the natural memristor mapping — one bit per synapse, SET/RESET
  writes only.
* Jumps are **pre-spike-driven** and gated by two analog quantities that
  are locally available on-chip: the post-neuron membrane voltage Vm and
  a leaky calcium variable C (a low-pass of post-spike history):
      LTP jump if  Vm_post > θ_V  AND  θ_lo_p ≤ C < θ_hi_p
      LTD jump if  Vm_post ≤ θ_V  AND  θ_lo_d ≤ C < θ_hi_d
      otherwise   no update — "stop-learning"
* The C-window is the homeostat: silent neurons (C below the floor)
  cannot learn, and saturating neurons (C above the ceiling) also cannot
  learn. This is the cure for the "everyone potentiates uniformly"
  failure that pair-STDP hit on this network.
* Calcium dynamics: dC/dt = -C/τ_C; on each post-spike, C += J_C. The
  per-synapse copy stays in lock-step with the post-neuron because all
  synapses onto the same post-neuron see the same on_post events.

Implementation notes
--------------------
* X has spike-driven jumps only — no continuous bistable drift. Lazy
  drift would need either clock-driven integration (78400 synapses ×
  every dt = expensive) or hand-rolled time-since-last-event book-
  keeping. The bistable rails dominate behaviour on MNIST timescales,
  so this is a justified simplification for a first run.
* C is per-synapse but `(event-driven)` — Brian2 advances it lazily at
  pre/post events using the closed-form exponential decay. Cheap.
* No L1 normalisation. The C-window does that job; reintroducing
  Diehl-Cook L1 would defeat the whole point.

Output format matches train_stdp.py for plug-and-play plotting:
    Classification-STDP/weights_bsf_N{N}.npz
        W              effective binary weights, shape (N, 784) ∈ {0, w_jump}
        X              internal X state, shape (N, 784)
        W_history      effective binary snapshots, (n_snapshots, N, 784)
        snapshot_steps image index per snapshot
        theta          final homeostatic θ (A)
        spike_history  per-snapshot mean firing rate
        n_trained      images shown
        config         hyperparam dict
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
    defaultclock, ms, us, second, Hz, amp, volt, prefs, seed,
)

from msn_neuron import MSNParams, make_msn


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_mnist_train(n: int, data_dir: str, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
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
    # BSF synapse
    w_jump: float,        # A — current delivered when X > theta_X (binary readout)
    X_max: float,         # X ceiling
    theta_X: float,       # readout threshold (bistable midpoint)
    a_LTP: float,         # X jump up
    a_LTD: float,         # X jump down
    theta_V: float,       # V — post-membrane voltage threshold
    tau_C: float,         # s — calcium time constant
    J_C: float,           # calcium increment per post-spike
    theta_lo_p: float,    # calcium pot window low
    theta_hi_p: float,    # calcium pot window high
    theta_lo_d: float,    # calcium dep window low
    theta_hi_d: float,    # calcium dep window high
    # Fixed WTA
    w_e2i: float,
    w_i2e: float,
    seed_init: int,
):
    rng = np.random.default_rng(seed_init)

    P = PoissonGroup(784, rates=np.zeros(784) * Hz, name='inp')
    G_E = make_msn(N, params=params, name='E')
    G_I = make_msn(N, params=params, name='I')
    G_E.I_0 = 0 * amp
    G_I.I_0 = 0 * amp

    bsf_model = '''
        X : 1
        dC/dt = -C/tau_C : 1 (event-driven)
    '''
    # Pre-spike: deliver binary effective current, then BSF jump conditional
    # on post Vm and post C. Note: Vm_post is read at the spike instant, so
    # the synapse sees the post-neuron's voltage just before any reset rule
    # at the same step would fire — exactly the BSF semantics.
    on_pre = '''
        Is1_exc_post += w_jump * int(X > theta_X)
        pot_gate = int(Vm_post > theta_V) * int(C >= theta_lo_p) * int(C < theta_hi_p)
        dep_gate = int(Vm_post <= theta_V) * int(C >= theta_lo_d) * int(C < theta_hi_d)
        X = clip(X + a_LTP * pot_gate - a_LTD * dep_gate, 0.0, X_max)
    '''
    # Post-spike: increment calcium. No direct X change at post-spike in BSF
    # — all jumps happen at pre-spike. The post-spike just feeds the C trace.
    on_post = '''
        C += J_C
    '''

    syn_in_e = Synapses(
        P, G_E,
        model=bsf_model, on_pre=on_pre, on_post=on_post,
        namespace=dict(
            tau_C       = tau_C * second,
            w_jump      = w_jump * amp,
            theta_V     = theta_V * volt,
            theta_X     = theta_X,
            X_max       = X_max,
            a_LTP       = a_LTP,
            a_LTD       = a_LTD,
            J_C         = J_C,
            theta_lo_p  = theta_lo_p,
            theta_hi_p  = theta_hi_p,
            theta_lo_d  = theta_lo_d,
            theta_hi_d  = theta_hi_d,
        ),
        name='syn_in_e',
    )
    syn_in_e.connect(True)
    # Heterogeneous init around the bistable midpoint so WTA has something
    # to arbitrate from step zero (the third STDP failure was symmetry that
    # never broke). 30% start "ON", 70% start "OFF" with a Gaussian spread.
    n_syn = 784 * N
    init_X = rng.normal(loc=theta_X, scale=0.25, size=n_syn)
    init_X = np.clip(init_X, 0.0, X_max).astype(np.float64)
    syn_in_e.X = init_X
    syn_in_e.C = 0.0

    syn_e_i = Synapses(G_E, G_I, on_pre=f'Is1_exc_post += {w_e2i}*amp', name='syn_e_i')
    syn_e_i.connect(j='i')

    syn_i_e = Synapses(G_I, G_E, on_pre=f'Is1_inh_post += {w_i2e}*amp', name='syn_i_e')
    syn_i_e.connect(condition='i != j')

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

    print(f"[config] N={args.N} dt={args.dt*1e6:.0f}µs  codegen={args.codegen}  rule=BSF")

    X_data, y = load_mnist_train(args.n_images, args.data, args.seed)
    print(f"[data] {len(X_data)} MNIST training images loaded")

    params = MSNParams(Cm=args.Cm, tau_s1=args.tau_s, tau_s2=args.tau_s)
    print(params.summary())

    net, P, G_E, G_I, syn_in_e, sp_E, sp_I = build_network(
        args.N,
        params      = params,
        w_jump      = args.w_jump,
        X_max       = args.X_max,
        theta_X     = args.theta_X,
        a_LTP       = args.a_LTP,
        a_LTD       = args.a_LTD,
        theta_V     = args.theta_V,
        tau_C       = args.tau_C,
        J_C         = args.J_C,
        theta_lo_p  = args.theta_lo_p,
        theta_hi_p  = args.theta_hi_p,
        theta_lo_d  = args.theta_lo_d,
        theta_hi_d  = args.theta_hi_d,
        w_e2i       = args.w_e2i,
        w_i2e       = args.w_i2e,
        seed_init   = args.seed,
    )

    # Mean-field sanity check: assume ~50% of synapses ON at init
    mean_active = float((X_data > 0).sum(axis=1).mean())
    mean_pixel  = float(X_data[X_data > 0].mean()) if (X_data > 0).any() else 0.0
    p_on = 0.5
    expected_I = mean_active * mean_pixel * args.lambda_max * args.tau_s * p_on * args.w_jump
    I_min, I_max = params.operating_window()
    print(f"[mean-field] expected <I_in> ≈ {expected_I*1e6:.2f} µA  "
          f"(window [{I_min*1e6:.1f}, {I_max*1e6:.0f}] µA)")
    if not (I_min < expected_I < I_max):
        print("  ⚠  expected current is OUTSIDE the MSN spiking window — "
              "tune --w_jump or --lambda_max")

    # Homeostatic θ: same Python-side bookkeeping as STDP path.
    theta = np.zeros(args.N)
    decay = float(np.exp(-(args.T_present + args.T_rest) / args.tau_theta))

    syn_i = np.array(syn_in_e.i[:], dtype=np.int64)
    syn_j = np.array(syn_in_e.j[:], dtype=np.int64)

    def snapshot_W() -> np.ndarray:
        """Effective binary weight matrix in amps-equivalent (we store the
        dimensionless 0/1 readout × 1.0; downstream plotters treat it like
        the STDP W in [0, w_max=1])."""
        X_arr = np.asarray(syn_in_e.X[:], dtype=np.float32)
        W_eff = (X_arr > args.theta_X).astype(np.float32)
        W = np.zeros((args.N, 784), dtype=np.float32)
        W[syn_j, syn_i] = W_eff
        return W

    def snapshot_X() -> np.ndarray:
        X_arr = np.asarray(syn_in_e.X[:], dtype=np.float32)
        Xm = np.zeros((args.N, 784), dtype=np.float32)
        Xm[syn_j, syn_i] = X_arr
        return Xm

    n_snapshots = max(2, args.n_images // args.snapshot_every + 2)
    W_history = np.zeros((n_snapshots, args.N, 784), dtype=np.float32)
    snap_steps = np.zeros(n_snapshots, dtype=np.int64)
    rate_history = np.zeros(n_snapshots, dtype=np.float32)
    snap_idx = 0
    W_history[snap_idx] = snapshot_W()
    snap_steps[snap_idx] = 0
    snap_idx += 1

    print(f"[train] starting: {args.n_images} images × "
          f"({args.T_present*1e3:.0f}ms present + {args.T_rest*1e3:.0f}ms rest)")
    t_wall_0 = time.time()
    last_count_E = np.zeros(args.N, dtype=np.int64)
    cumulative_spikes = 0

    zero_rates = np.zeros(784, dtype=np.float64)
    for img_idx in range(args.n_images):
        x = X_data[img_idx]
        img_rates = x * args.lambda_max
        max_rescales = 3

        for attempt in range(max_rescales + 1):
            G_E.I_0 = (-theta) * amp
            counts_before = sp_E.count[:].copy()

            P.rates = img_rates * Hz
            net.run(args.T_present * second)
            counts_after = sp_E.count[:].copy()
            n_spikes = int((counts_after - counts_before).sum())

            P.rates = zero_rates * Hz
            net.run(args.T_rest * second)

            if n_spikes >= args.min_spikes_per_image or attempt == max_rescales:
                break
            img_rates = img_rates + 32.0

        spikes_this_image = sp_E.count[:].copy() - last_count_E
        last_count_E = sp_E.count[:].copy()
        theta = theta * decay + args.theta_plus * spikes_this_image
        cumulative_spikes += int(spikes_this_image.sum())

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
            X_now = np.array(syn_in_e.X[:])
            frac_on = float((X_now > args.theta_X).mean())
            print(f"[{img_idx+1:6d}/{args.n_images}]  "
                  f"<rate>={mean_rate:5.2f} Hz  "
                  f"<X>={X_now.mean():.3f}  "
                  f"frac_ON={frac_on:.2%}  "
                  f"<θ>={theta.mean()*1e6:.2f} µA  "
                  f"elapsed={elapsed:6.1f}s")

    if snap_idx < n_snapshots:
        W_history[snap_idx] = snapshot_W()
        snap_steps[snap_idx] = args.n_images
        snap_idx += 1

    W_history    = W_history[:snap_idx]
    snap_steps   = snap_steps[:snap_idx]
    rate_history = rate_history[:snap_idx]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        W              = W_history[-1],
        X              = snapshot_X(),
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
    ap = argparse.ArgumentParser(description="BSF unsupervised MSN MNIST trainer")
    ap.add_argument('--N',              type=int,   default=100)
    ap.add_argument('--n_images',       type=int,   default=60_000)
    ap.add_argument('--data',           default='Classification/data')
    ap.add_argument('--out',            default=None)

    # presentation
    ap.add_argument('--T_present',      type=float, default=0.35)
    ap.add_argument('--T_rest',         type=float, default=0.15)
    ap.add_argument('--lambda_max',     type=float, default=63.75)
    ap.add_argument('--min_spikes_per_image', type=int, default=5)

    # BSF synapse — current delivery
    ap.add_argument('--w_jump',         type=float, default=1.0e-7,
                    help='A delivered per pre-spike when X > theta_X (binary readout)')
    ap.add_argument('--X_max',          type=float, default=1.0)
    ap.add_argument('--theta_X',        type=float, default=0.5)

    # BSF synapse — jump magnitudes
    ap.add_argument('--a_LTP',          type=float, default=0.10,
                    help='X increment on potentiation event')
    ap.add_argument('--a_LTD',          type=float, default=0.10,
                    help='X decrement on depression event')

    # BSF synapse — gates
    ap.add_argument('--theta_V',        type=float, default=0.7,
                    help='V; post-Vm threshold splitting LTP from LTD '
                         '(MSN Vth=1.5V, ~0.5*Vth puts split near upstroke)')
    ap.add_argument('--tau_C',          type=float, default=60e-3,
                    help='s; calcium decay time constant')
    ap.add_argument('--J_C',            type=float, default=1.0,
                    help='calcium increment per post-spike')
    ap.add_argument('--theta_lo_p',     type=float, default=3.0,
                    help='LTP gate: lower calcium bound')
    ap.add_argument('--theta_hi_p',     type=float, default=13.0,
                    help='LTP gate: upper calcium bound (above = stop-learning)')
    ap.add_argument('--theta_lo_d',     type=float, default=3.0,
                    help='LTD gate: lower calcium bound')
    ap.add_argument('--theta_hi_d',     type=float, default=4.0,
                    help='LTD gate: upper calcium bound')

    # WTA
    ap.add_argument('--w_e2i',          type=float, default=50e-6)
    ap.add_argument('--w_i2e',          type=float, default=5e-6,
                    help='A; lateral inhibition. Lower than STDP path (30µA) — '
                         'we want winners to emerge, not be crushed.')

    # homeostasis
    ap.add_argument('--theta_plus',     type=float, default=0.05e-6)
    ap.add_argument('--tau_theta',      type=float, default=1e7)

    # MSN dynamics
    ap.add_argument('--Cm',             type=float, default=0.05e-6)
    ap.add_argument('--tau_s',          type=float, default=200e-3)

    # bookkeeping
    ap.add_argument('--snapshot_every', type=int,   default=500)
    ap.add_argument('--log_every',      type=int,   default=100)
    ap.add_argument('--seed',           type=int,   default=0)
    ap.add_argument('--dt',             type=float, default=50e-6)
    ap.add_argument('--codegen',        default='numpy', choices=['numpy', 'cython'])

    args = ap.parse_args()
    if args.out is None:
        args.out = f'Classification-STDP/weights_bsf_N{args.N}.npz'
    train(args)


if __name__ == '__main__':
    main()
