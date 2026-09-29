"""
bench_mnist.py
==============
MSN (discrete switch) vs MSN_HH (bistable gate, docs/MSN_HH_GATE_PLAN.md)
on MNIST: accuracy as a feature extractor, and simulation cost.

Network: 784 pixels → 100 hidden MSN neurons (fixed random projection) →
ridge-regression readout on hidden spike counts.  Identical weights, images
and readout for every model, so only the neuron model differs.

Part 0  sanity   — Brian2 F–I of each model at a few currents
Part A  accuracy — current-driven: each image sets a constant input current
                   I_j = I_MID + I_SPAN·z_j (z = standardised projection);
                   all images simulated in parallel as one NeuronGroup
Part B  cost     — full spiking net: 784 PoissonGroup → make_synapse (alpha
                   cascade) → 100 neurons, one image; wall-clock per
                   simulated second, extrapolated to one MNIST epoch

    uv run python MNIST_test/bench_mnist.py
"""

import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
from brian2 import (prefs, start_scope, defaultclock, Network, SpikeMonitor,
                    PoissonGroup, second, amp, Hz)

from msn_neuron import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse
from msn_bistable import make_msn_bistable

prefs.codegen.target = 'cython'
prefs.logging.file_log = False

PARAMS = MSNParams.from_json(os.path.join(_HERE, '..', 'configs', 'neuron_hardware_fit.json'))
N_HID, N_TRAIN, N_TEST = 100, 500, 200
T_PRESENT = 50e-3                   # s per image (part A)
T_B = 20e-3                         # s simulated in part B
I_MID, I_SPAN = 60e-6, 15e-6        # A — maps projection into the 40–95 µA window
RIDGE = 10.0
SEED = 0

# model name → (factory, dt [s], kwargs)
MODELS = {
    'MSN (discrete, dt=1 µs)':           (make_msn,          1e-6,   {}),
    'MSN_HH (τx=1 µs, dt=0.1 µs)':       (make_msn_bistable, 0.1e-6, dict(tau_x=1e-6)),
    'MSN_HH (τx=10 µs, dt=1 µs)':        (make_msn_bistable, 1e-6,   dict(tau_x=10e-6, k_V=20e-3, k_I=2e-6)),
}


# ── data ─────────────────────────────────────────────────────────────────────

def load_mnist():
    from torchvision.datasets import MNIST
    root = os.path.join(_HERE, 'data')
    tr, te = MNIST(root, train=True, download=True), MNIST(root, train=False, download=True)
    rng = np.random.default_rng(SEED)
    itr = rng.choice(len(tr), N_TRAIN, replace=False)
    ite = rng.choice(len(te), N_TEST, replace=False)
    X = lambda ds, idx: ds.data.numpy()[idx].reshape(len(idx), -1).astype(float) / 255
    y = lambda ds, idx: ds.targets.numpy()[idx]
    return X(tr, itr), y(tr, itr), X(te, ite), y(te, ite)


def ridge_acc(Ftr, ytr, Fte, yte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A = np.c_[(Ftr - mu) / sd, np.ones(len(Ftr))]
    B = np.c_[(Fte - mu) / sd, np.ones(len(Fte))]
    Y = np.eye(10)[ytr]
    Wr = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ Y)
    return float(np.mean((B @ Wr).argmax(1) == yte))


def analytic_rate(I, p=PARAMS):
    to, tc = p.Cm * (p.Rm_hi + p.Ra), p.Cm * (p.Rm_lo + p.Ra)
    Vh, Vo, Vc = p.I_hold * (p.Rm_lo + p.Ra), I * (p.Rm_hi + p.Ra), I * (p.Rm_lo + p.Ra)
    with np.errstate(all='ignore'):
        T = to * np.log((Vo - Vh) / (Vo - p.Vth)) + tc * np.log((p.Vth - Vc) / (Vh - Vc))
    return np.where((I > p.Vth / (p.Rm_hi + p.Ra)) & (I < p.I_hold), 1 / T, 0.0)


# ── simulation helpers ───────────────────────────────────────────────────────

def run_current_driven(factory, dt, kw, I_in, T):
    """Constant current per neuron. Returns (spike counts, wall s excl. compile)."""
    start_scope()
    defaultclock.dt = dt * second
    G = factory(len(I_in), params=PARAMS, name='g', **kw)
    G.I_0 = I_in * amp
    mon = SpikeMonitor(G, record=False)
    net = Network(G, mon)
    net.run(10 * dt * second)                  # warm-up: code generation / compile
    c0 = np.array(mon.count)
    t0 = time.perf_counter()
    net.run(T * second)
    return np.array(mon.count) - c0, time.perf_counter() - t0


def run_full_snn(factory, dt, kw, rates, W, T):
    """Poisson pixels → alpha-cascade synapses → 100 hidden. Returns (counts, wall s)."""
    start_scope()
    defaultclock.dt = dt * second
    P = PoissonGroup(784, rates=rates * Hz, name='px')
    G = factory(N_HID, params=PARAMS, name='g', **kw)
    G.I_0 = I_MID * amp
    sp = SynapseParams(kind='exc', weight=0.0, tau_s1=5e-3, tau_s2=5e-3)
    S = make_synapse(P, G, params=sp, connect=True, name='syn')
    S.w = W[np.array(S.i[:]), np.array(S.j[:])] * amp
    mon = SpikeMonitor(G, record=False)
    net = Network(P, G, S, mon)
    net.run(10 * dt * second)
    c0 = np.array(mon.count)
    t0 = time.perf_counter()
    net.run(T * second)
    return np.array(mon.count) - c0, time.perf_counter() - t0


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out = {'config': dict(N_HID=N_HID, N_TRAIN=N_TRAIN, N_TEST=N_TEST,
                          T_PRESENT=T_PRESENT, T_B=T_B, I_MID=I_MID, I_SPAN=I_SPAN)}

    # Part 0 — sanity F–I
    I_chk = np.array([45, 52.5, 67.9, 80.1, 91.5, 94]) * 1e-6
    print('Part 0  F–I sanity (Hz) at', (I_chk * 1e6).round(1), 'µA')
    print(f'  {"analytic":34s}', analytic_rate(I_chk).round(0))
    out['sanity'] = {'I_uA': (I_chk * 1e6).tolist(), 'analytic': analytic_rate(I_chk).tolist()}
    for name, (fac, dt, kw) in MODELS.items():
        c, _ = run_current_driven(fac, dt, kw, I_chk, 0.3)
        print(f'  {name:34s}', (c / 0.3).round(0))
        out['sanity'][name] = (c / 0.3).tolist()

    # Part A — accuracy (current-driven)
    Xtr, ytr, Xte, yte = load_mnist()
    rng = np.random.default_rng(SEED)
    Wp = rng.normal(0, 1, (784, N_HID)) / np.sqrt(784)
    Htr, Hte = Xtr @ Wp, Xte @ Wp
    mu, sd = Htr.mean(0), Htr.std(0)
    Itr, Ite = I_MID + I_SPAN * (Htr - mu) / sd, I_MID + I_SPAN * (Hte - mu) / sd
    I_all = np.r_[Itr, Ite].ravel()             # (images × hidden) neurons in one group

    print(f'\nPart A  {N_TRAIN} train / {N_TEST} test images, {N_HID} hidden, '
          f'{T_PRESENT*1e3:.0f} ms each ({len(I_all)} neurons simulated in parallel)')
    out['A'] = {}
    acc_px = ridge_acc(Xtr, ytr, Xte, yte)
    acc_an = ridge_acc(analytic_rate(Itr), ytr, analytic_rate(Ite), yte)
    print(f'  {"ridge on raw pixels":34s} acc = {acc_px:.3f}')
    print(f'  {"analytic MSN rates (T → ∞)":34s} acc = {acc_an:.3f}')
    out['A']['pixels'] = {'acc': acc_px}
    out['A']['analytic'] = {'acc': acc_an}
    for name, (fac, dt, kw) in MODELS.items():
        c, wall = run_current_driven(fac, dt, kw, I_all, T_PRESENT)
        C = c.reshape(N_TRAIN + N_TEST, N_HID).astype(float)
        acc = ridge_acc(C[:N_TRAIN], ytr, C[N_TRAIN:], yte)
        print(f'  {name:34s} acc = {acc:.3f}   wall = {wall:7.1f} s   '
              f'mean rate = {C.mean() / T_PRESENT:5.1f} Hz')
        out['A'][name] = {'acc': acc, 'wall_s': wall, 'mean_rate_Hz': C.mean() / T_PRESENT}

    # Part B — cost of a full spiking network (one image)
    lam = Xtr[0] * 63.75                         # Hz, Diehl & Cook input scaling
    Wsyn = Wp * (I_SPAN / (lam @ Wp * 5e-3).std())
    epoch_sim_s = 60_000 * 0.35                  # one MNIST epoch at 350 ms/image
    print(f'\nPart B  full SNN 784 Poisson → {N_HID} (78 400 alpha synapses), '
          f'{T_B*1e3:.0f} ms simulated')
    out['B'] = {}
    for name, (fac, dt, kw) in MODELS.items():
        c, wall = run_full_snn(fac, dt, kw, lam, Wsyn, T_B)
        per_s = wall / T_B
        print(f'  {name:34s} wall = {wall:6.1f} s  → {per_s:7.0f} s per simulated s'
              f'  → one epoch ≈ {per_s * epoch_sim_s / 86400:7.1f} days   '
              f'({c.sum()} hidden spikes)')
        out['B'][name] = {'wall_s': wall, 'wall_per_sim_s': per_s,
                          'epoch_days': per_s * epoch_sim_s / 86400, 'spikes': int(c.sum())}

    with open(os.path.join(_HERE, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nsaved', os.path.join(_HERE, 'results.json'))
