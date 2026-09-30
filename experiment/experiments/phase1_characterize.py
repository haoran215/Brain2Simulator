"""
phase1_characterize.py
======================
Phase 1 of "Software Exploration of Memristive Spiking Neurons and STDP":
characterise the MSN as a computational element, against LIF-matched and
LIF-generic (experiment/models/neurons.py), under identical input currents.

No synapses, no learning.  Every input is an injected current I(t) written to
the I_exc inlet each timestep, on top of a tonic bias I_0.

  P1  F–I curve, rheobase, rate ceiling
  P2  sustained step: first-spike latency, spike-frequency adaptation
  P3  slow triangular ramp: hysteresis (onset/offset, block/release)
  P4  paired pulses: temporal integration window (timing sensitivity)
  P5  pulse trains (following fidelity) and subthreshold bursts
  P6  irregular (OU-noise) input: rate, CV, spike-time reliability
  P7  history: response to a probe after conditioning (firing / block)
  P8  perturbation memory: how long a small kick changes the future

    uv run python experiment/experiments/phase1_characterize.py [--parts P1,P4]
"""

from __future__ import annotations

import argparse, json, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiment'))

import numpy as np
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import (prefs, start_scope, defaultclock, Network, SpikeMonitor,
                    StateMonitor, TimedArray, second, amp, volt)

from models.neurons import MODELS, make_neuron, hw_params, lif_config

prefs.codegen.target = 'cython'
prefs.logging.file_log = False
prefs.logging.console_log_level = 'ERROR'

DT = 10e-6                      # s — chosen in fi_dt_check
P = hw_params()
I_MIN, I_MAX = P.operating_window()
I_BIAS_SUB = 30e-6              # A — subthreshold bias for pulse experiments
PW = 0.1e-3                     # s — injected pulse width
OUT = os.path.join(_ROOT, 'experiment', 'results', 'phase1')
COLORS = {'lif_generic': '#2a78d6', 'lif_matched': '#eb6834', 'msn': '#1baf7a'}
LABELS = {'lif_generic': 'LIF-generic', 'lif_matched': 'LIF-matched', 'msn': 'MSN'}
SEED = 1


# ── simulation core ──────────────────────────────────────────────────────────

def capacitance(kind: str) -> float:
    return P.Cm if kind == 'msn' else lif_config(kind, P)['C']


def simulate(kind, T, bias, stim=None, record_v=None, v_dt=None):
    """Run N = len(bias) independent neurons for T seconds.

    bias : (N,) tonic current [A]
    stim : (n_steps, N) extra current [A] on the DT grid, or None
    Returns (list of spike-time arrays [s], Vm array (n_rec, n_t) or None, t_v).
    """
    start_scope()
    defaultclock.dt = DT * second
    N = len(bias)
    G = make_neuron(kind, N, P)
    G.I_0 = np.asarray(bias) * amp
    # start at the open-state resting voltage of the bias (identical R in all models),
    # capped below threshold, so pulse experiments do not ride on the initial charge-up
    G.Vm = np.minimum(np.asarray(bias) * (P.Rm_hi + P.Ra), 0.95 * P.Vth) * volt
    objs = [G]
    if stim is not None:
        G.namespace['stim_ta'] = TimedArray(np.asarray(stim) * amp, dt=DT * second)
        G.run_regularly('I_exc = stim_ta(t, i)', when='start')
    sm = SpikeMonitor(G)
    objs.append(sm)
    vm = None
    if record_v is not None:
        vm = StateMonitor(G, 'Vm', record=record_v, dt=(v_dt or DT) * second)
        objs.append(vm)
    Network(*objs).run(T * second)
    ti, ii = np.asarray(sm.t / second), np.asarray(sm.i)
    order = np.argsort(ii, kind='stable')
    ti, ii = ti[order], ii[order]
    bounds = np.searchsorted(ii, np.arange(N + 1))
    spikes = [ti[bounds[k]:bounds[k + 1]] for k in range(N)]
    if vm is None:
        return spikes, None, None
    return spikes, np.asarray(vm.Vm / volt), np.asarray(vm.t / second)


def steps(T):
    n = np.rint(np.asarray(T) / DT).astype(int)
    return int(n) if n.ndim == 0 else n


def add_pulse(stim, col, t_on, amp_A, width=PW):
    a = steps(t_on)
    stim[a:a + steps(width), col] += amp_A


def ou(n, tau, sigma, rng):
    """Ornstein–Uhlenbeck noise, unit-free, stationary std = sigma."""
    a = np.exp(-DT / tau)
    b = sigma * np.sqrt(1 - a * a)
    z = b * rng.standard_normal(n)
    z[0] = sigma * rng.standard_normal()
    return lfilter([1.0], [1.0, -a], z)


def rate(sp, t0, t1):
    return np.sum((sp >= t0) & (sp < t1)) / (t1 - t0)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches='tight')
    plt.close(fig)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


# ── P1  F–I ──────────────────────────────────────────────────────────────────

def p1_fi():
    I = np.linspace(0, 150e-6, 76)
    I_fine = np.arange(38e-6, 42e-6, 0.05e-6)
    T, skip = 1.0, 0.1
    res = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in MODELS:
        sp, _, _ = simulate(kind, T, np.concatenate([I, I_fine]))
        r = np.array([rate(s, skip, T) for s in sp])
        rI, rf = r[:len(I)], r[len(I):]
        rheo = float(I_fine[np.argmax(rf > 0)]) if np.any(rf > 0) else None
        firing = rI > 0
        block = None
        if kind == 'msn':
            above = np.where((I > rheo) & ~firing)[0] if rheo else []
            block = float(I[above[0]]) if len(above) else None
        res[kind] = dict(I=I.tolist(), rate=rI.tolist(), rheobase=rheo,
                         max_rate=float(rI.max()), I_at_max=float(I[np.argmax(rI)]),
                         block_onset=block)
        ax.plot(I * 1e6, rI, '-', color=COLORS[kind], lw=2, label=LABELS[kind])
    ax.axvline(I_MAX * 1e6, color='0.5', ls=':', lw=1)
    ax.text(I_MAX * 1e6 + 1, ax.get_ylim()[1] * 0.9, 'I_hold', color='0.4', fontsize=8)
    ax.set(xlabel='input current (µA)', ylabel='firing rate (Hz)', title='P1  F–I curve')
    style(ax); ax.legend(frameon=False)
    savefig(fig, 'P1_fi.png')
    return res


# ── P2  sustained step / adaptation ──────────────────────────────────────────

def p2_step():
    levels = np.array([45e-6, 65e-6, 85e-6, 110e-6])
    t_on, T = 0.05, 1.05
    stim = np.zeros((steps(T), len(levels)))
    stim[steps(t_on):, :] = levels
    res = {}
    fig, axs = plt.subplots(1, len(levels), figsize=(13, 3.4), sharey=True)
    for kind in MODELS:
        sp, _, _ = simulate(kind, T, np.zeros(len(levels)), stim)
        out = []
        for k, s in enumerate(sp):
            s = s[s >= t_on]
            if len(s) < 3:
                out.append(dict(I=levels[k], n=len(s), latency=float(s[0] - t_on) if len(s) else None,
                                f_initial=None, f_ss=None, adaptation=None))
                continue
            isi = np.diff(s)
            f0 = 1 / isi[0]
            fss = rate(s, T - 0.5, T)
            out.append(dict(I=levels[k], n=len(s), latency=float(s[0] - t_on), f_initial=float(f0),
                            f_ss=float(fss), adaptation=float(1 - fss / f0)))
            axs[k].plot((s[1:] - t_on) * 1e3, 1 / isi, '.', ms=3, color=COLORS[kind], label=LABELS[kind])
        res[kind] = out
    for k, a in enumerate(axs):
        a.set(title=f'{levels[k]*1e6:.0f} µA step', xlabel='time since onset (ms)')
        style(a)
    axs[0].set_ylabel('instantaneous rate (Hz)')
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('P2  Sustained step response (1/ISI)', y=1.02)
    savefig(fig, 'P2_step.png')
    return res


# ── P3  hysteresis ramp ──────────────────────────────────────────────────────

def p3_ramp():
    I_top = 150e-6
    durations = [0.4, 4.0]
    res = {}
    fig, axs = plt.subplots(1, len(durations), figsize=(11, 3.8), sharey=True)
    for d_idx, D in enumerate(durations):
        n = steps(D)
        half = n // 2
        ramp = np.concatenate([np.linspace(0, I_top, half), np.linspace(I_top, 0, n - half)])
        for kind in MODELS:
            sp, _, _ = simulate(kind, D, np.zeros(1), ramp[:, None])
            s = sp[0]
            I_at = lambda t: ramp[np.minimum(steps(t), n - 1)]
            up, dn = s[s < D / 2], s[s >= D / 2]
            entry = dict(
                onset_up=float(I_at(up[0])) if len(up) else None,
                last_up=float(I_at(up[-1])) if len(up) else None,
                first_down=float(I_at(dn[0])) if len(dn) else None,
                offset_down=float(I_at(dn[-1])) if len(dn) else None,
            )
            res.setdefault(kind, {})[f'{D}s'] = entry
            if len(s) > 1:
                isi = np.diff(s)
                mid = s[1:]
                sgn = np.where(mid < D / 2, 1, -1)
                same_side = (s[:-1] < D / 2) == (s[1:] < D / 2)   # drop the ISI spanning the top
                for sg, ls in [(1, '-'), (-1, '--')]:
                    m = (sgn == sg) & same_side
                    axs[d_idx].plot(I_at(mid[m]) * 1e6, 1 / isi[m], ls, color=COLORS[kind], lw=1.5,
                                    label=f'{LABELS[kind]} {"up" if sg > 0 else "down"}')
        axs[d_idx].set(title=f'triangular ramp 0→150→0 µA over {D:g} s', xlabel='input current (µA)')
        axs[d_idx].axvline(I_MAX * 1e6, color='0.5', ls=':', lw=1)
        style(axs[d_idx])
    axs[0].set_ylabel('instantaneous rate (Hz)')
    axs[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle('P3  Ramp up (solid) vs ramp down (dashed)', y=1.02)
    savefig(fig, 'P3_ramp.png')
    return res


# ── P4  paired pulses → integration window ───────────────────────────────────

K_GRID = np.round(np.arange(0.20, 1.40, 0.01), 3)
DTS_PAIR = np.array([0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]) * 1e-3


def q_estimate(kind):
    return capacitance(kind) * (P.Vth - I_BIAS_SUB * (P.Rm_hi + P.Ra))


def threshold_k(sp):
    fired = np.array([len(s) > 0 for s in sp])
    return float(K_GRID[np.argmax(fired)]) if fired.any() else None


def p4_pairs():
    t1, tail = 0.01, 0.03
    res = {}
    Q1 = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in MODELS:
        Qe = q_estimate(kind)
        amps = K_GRID * Qe / PW
        # single pulse
        T = t1 + tail
        stim = np.zeros((steps(T), len(K_GRID)))
        for c, a in enumerate(amps):
            add_pulse(stim, c, t1, a)
        sp, _, _ = simulate(kind, T, np.full(len(K_GRID), I_BIAS_SUB), stim)
        k1 = threshold_k(sp)
        Q1[kind] = k1 * Qe
        ratios = []
        for dt_pair in DTS_PAIR:
            T = t1 + dt_pair + tail
            stim = np.zeros((steps(T), len(K_GRID)))
            for c, a in enumerate(amps):
                add_pulse(stim, c, t1, a)
                add_pulse(stim, c, t1 + dt_pair, a)
            sp, _, _ = simulate(kind, T, np.full(len(K_GRID), I_BIAS_SUB), stim)
            kp = threshold_k(sp)
            ratios.append(kp / k1 if kp is not None else None)
        ratios = np.array(ratios, dtype=float)
        # integration window: Δt where the ratio crosses 0.75 (half-way between perfect 0.5 and none 1.0)
        above = np.where(ratios >= 0.75)[0]
        if len(above) and above[0] > 0:
            j = above[0]
            x0, x1 = np.log(DTS_PAIR[j - 1]), np.log(DTS_PAIR[j])
            y0, y1 = ratios[j - 1], ratios[j]
            win = float(np.exp(x0 + (0.75 - y0) * (x1 - x0) / (y1 - y0)))
        else:
            win = None
        res[kind] = dict(Q1_nC=Q1[kind] * 1e9, dt_pair=DTS_PAIR.tolist(), ratio=ratios.tolist(),
                         integration_window=win)
        ax.plot(DTS_PAIR * 1e3, ratios, 'o-', ms=4, color=COLORS[kind], lw=2, label=LABELS[kind])
    ax.axhline(0.5, color='0.6', ls=':', lw=1); ax.axhline(1.0, color='0.6', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set(xlabel='inter-pulse interval Δt (ms)', ylabel='pair threshold / single threshold',
           title='P4  Paired-pulse summation (0.5 = perfect, 1 = none)')
    style(ax); ax.legend(frameon=False)
    savefig(fig, 'P4_pairs.png')
    return res, Q1


# ── P5  pulse trains & bursts ────────────────────────────────────────────────

def p5_trains(Q1):
    freqs = np.array([10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500])
    T_train, t0 = 0.2, 0.01
    k_max = 10
    burst_isi = np.array([1, 2, 5, 10]) * 1e-3
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind in MODELS:
        # (a) suprathreshold trains: fraction of pulses that produce a spike
        T = t0 + T_train + 0.01
        stim = np.zeros((steps(T), len(freqs)))
        n_pulses = []
        for c, f in enumerate(freqs):
            ts = t0 + np.arange(0, T_train, 1 / f)
            n_pulses.append(len(ts))
            for t in ts:
                add_pulse(stim, c, t, 1.5 * Q1[kind] / PW)
        sp, _, _ = simulate(kind, T, np.full(len(freqs), I_BIAS_SUB), stim)
        fidelity = np.array([len(s) / n for s, n in zip(sp, n_pulses)])
        axs[0].plot(freqs, fidelity, 'o-', ms=4, lw=2, color=COLORS[kind], label=LABELS[kind])
        # (b) subthreshold bursts: minimum number of 0.35·Q1 pulses to fire
        cols = [(isi, k) for isi in burst_isi for k in range(1, k_max + 1)]
        T = t0 + k_max * burst_isi.max() + 0.03
        stim = np.zeros((steps(T), len(cols)))
        for c, (isi, k) in enumerate(cols):
            for j in range(k):
                add_pulse(stim, c, t0 + j * isi, 0.35 * Q1[kind] / PW)
        sp, _, _ = simulate(kind, T, np.full(len(cols), I_BIAS_SUB), stim)
        k_needed = []
        for isi in burst_isi:
            ks = [k for (i2, k), s in zip(cols, sp) if i2 == isi and len(s) > 0]
            k_needed.append(min(ks) if ks else None)
        axs[1].plot(burst_isi * 1e3, [k if k else np.nan for k in k_needed], 'o-', ms=5, lw=2,
                    color=COLORS[kind], label=LABELS[kind])
        res[kind] = dict(freqs=freqs.tolist(), fidelity=fidelity.tolist(),
                         burst_isi=burst_isi.tolist(), pulses_needed=k_needed)
    axs[0].set_xscale('log')
    axs[0].set(xlabel='pulse frequency (Hz)', ylabel='output spikes / input pulse',
               title='P5a  Following of suprathreshold trains (1.5·Q₁)')
    axs[1].set(xlabel='intra-burst interval (ms)', ylabel='pulses needed to fire',
               title=f'P5b  Burst detection (0.35·Q₁ pulses, >{k_max} = never)')
    for a in axs:
        style(a)
    axs[0].legend(frameon=False)
    savefig(fig, 'P5_trains.png')
    return res


# ── P6  irregular input: reliability ─────────────────────────────────────────

def schreiber(sp, T, sigma=1e-3):
    t = np.arange(0, T, 0.1e-3)
    vecs = []
    for s in sp:
        v = np.zeros_like(t)
        for x in s:
            v += np.exp(-0.5 * ((t - x) / sigma) ** 2)
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
    if len(vecs) < 2:
        return None
    V = np.array(vecs)
    C = V @ V.T
    m = len(vecs)
    return float((C.sum() - m) / (m * (m - 1)))


def p6_irregular():
    mus = np.array([45e-6, 65e-6, 85e-6])
    sigma, sig_ind, tau = 20e-6, 4e-6, 2e-3
    n_trials, T, skip = 20, 2.0, 0.1
    n = steps(T)
    rng = np.random.default_rng(SEED)
    frozen = [ou(n, tau, sigma, rng) for _ in mus]
    private = [[ou(n, tau, sig_ind, rng) for _ in range(n_trials)] for _ in mus]
    stim = np.zeros((n, len(mus) * n_trials))
    for m in range(len(mus)):
        for r in range(n_trials):
            stim[:, m * n_trials + r] = frozen[m] + private[m][r]
    bias = np.repeat(mus, n_trials)
    res = {}
    fig, axs = plt.subplots(1, len(MODELS), figsize=(13, 3.6), sharey=True)
    for a_idx, kind in enumerate(MODELS):
        sp, _, _ = simulate(kind, T, bias, stim)
        out = []
        for m, mu in enumerate(mus):
            trials = [s[s >= skip] - skip for s in sp[m * n_trials:(m + 1) * n_trials]]
            rates = [len(s) / (T - skip) for s in trials]
            cvs = [np.std(np.diff(s)) / np.mean(np.diff(s)) for s in trials if len(s) > 3]
            out.append(dict(mu=mu, rate=float(np.mean(rates)),
                            cv=float(np.mean(cvs)) if cvs else None,
                            reliability=schreiber(trials, T - skip)))
            for r, s in enumerate(trials):
                w = s[s < 0.3]
                axs[a_idx].plot(w * 1e3, np.full(len(w), m * (n_trials + 4) + r), '|',
                                color=COLORS[kind], ms=3)
        res[kind] = out
        axs[a_idx].set(title=LABELS[kind], xlabel='time (ms)')
        axs[a_idx].set_yticks([m * (n_trials + 4) + n_trials / 2 for m in range(len(mus))])
        axs[a_idx].set_yticklabels([f'μ={mu*1e6:.0f} µA' for mu in mus])
        axs[a_idx].spines[['top', 'right']].set_visible(False)
    fig.suptitle('P6  Frozen OU input, 20 trials with private noise (first 300 ms)', y=1.02)
    savefig(fig, 'P6_irregular.png')
    return res


# ── P7  history: probe after conditioning ────────────────────────────────────

def p7_history():
    I_probe, T_probe = 50e-6, 0.1
    t_pre, T_cond = 0.1, 0.2
    conds = {'firing (65 µA)': 65e-6, 'block (120 µA)': 120e-6}
    gaps = np.array([0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]) * 1e-3
    # the conditioning length is jittered over n_phase values so that the phase of the
    # last conditioning spike is averaged out (otherwise latency reflects that phase)
    n_phase, jitter = 20, 1.25e-3
    res = {}
    fig, axs = plt.subplots(1, len(conds), figsize=(11, 3.8), sharey=True)
    for kind in MODELS:
        T = t_pre + T_cond + n_phase * jitter + gaps.max() + T_probe
        stim = np.zeros((steps(T), 1))
        stim[steps(t_pre):, 0] = I_probe - I_BIAS_SUB          # control: probe from rest
        sp, _, _ = simulate(kind, T, np.array([I_BIAS_SUB]), stim)
        control = float(sp[0][sp[0] >= t_pre][0] - t_pre)
        out = dict(control_latency=control)
        for c_i, c in enumerate(conds):
            cols = [(g, k) for g in gaps for k in range(n_phase)]
            stim = np.zeros((steps(T), len(cols)))
            probe_on = []
            for c_idx, (g, k) in enumerate(cols):
                t_end = t_pre + T_cond + k * jitter
                stim[steps(t_pre):steps(t_end), c_idx] = conds[c] - I_BIAS_SUB
                t_p = t_end + g
                stim[steps(t_p):steps(t_p + T_probe), c_idx] = I_probe - I_BIAS_SUB
                probe_on.append(t_p)
            sp, _, _ = simulate(kind, T, np.full(len(cols), I_BIAS_SUB), stim)
            lat = np.array([(s[(s >= t_p) & (s < t_p + T_probe)][:1] - t_p).tolist() or [np.nan]
                            for s, t_p in zip(sp, probe_on)]).reshape(len(gaps), n_phase)
            ratio = lat / control
            dev = np.nanmean(np.abs(ratio - 1), axis=1)
            ok = np.where(dev < 0.02)[0]
            mem = float(gaps[ok[0]]) if len(ok) and np.all(dev[ok[0]:] < 0.02) else None
            out[c] = dict(gaps=gaps.tolist(), ratio_mean=np.nanmean(ratio, 1).tolist(),
                          ratio_min=np.nanmin(ratio, 1).tolist(), ratio_max=np.nanmax(ratio, 1).tolist(),
                          mean_abs_dev=dev.tolist(), memory_2pct=mem)
            axs[c_i].plot(gaps * 1e3, dev, 'o-', ms=4, lw=2, color=COLORS[kind], label=LABELS[kind])
        res[kind] = out
    for c_i, c in enumerate(conds):
        axs[c_i].axhline(0.02, color='0.6', ls=':', lw=1)
        axs[c_i].set_xscale('log'); axs[c_i].set_yscale('log'); axs[c_i].set_ylim(1e-4, 3)
        axs[c_i].set(title=f'after 200 ms {c}', xlabel='gap before probe (ms)')
        style(axs[c_i])
    axs[0].set_ylabel('mean |latency/control − 1| over phases')
    axs[0].legend(frameon=False)
    fig.suptitle('P7  Response to a 50 µA probe after conditioning (bias 30 µA)', y=1.02)
    savefig(fig, 'P7_history.png')
    return res


# ── P8  perturbation memory ──────────────────────────────────────────────────

def p8_perturb(Q1):
    # (μ, σ): the third regime never reaches I_hold (μ + 5σ < 95 µA) — control for the latch
    mus = {'fluctuation-driven (μ=35, σ=15 µA)': (35e-6, 15e-6),
           'mean-driven (μ=65, σ=15 µA)': (65e-6, 15e-6),
           'mean-driven, below I_hold (μ=55, σ=7 µA)': (55e-6, 7e-6)}
    tau = 2e-3
    n_seeds, T, t_kick, tol = 30, 1.0, 0.3, 0.05e-3
    n = steps(T)
    rng = np.random.default_rng(SEED + 1)
    noises = {m: [ou(n, tau, mus[m][1], rng) for _ in range(n_seeds)] for m in mus}
    res = {}
    fig, axs = plt.subplots(1, len(mus), figsize=(15, 3.8), sharey=True)
    for kind in MODELS:
        kick = 0.1 * Q1[kind] / PW
        dV0 = 0.1 * Q1[kind] / capacitance(kind)
        cols, bias = [], []
        stim = np.zeros((n, 2 * n_seeds * len(mus)))
        for m_i, m in enumerate(mus):
            for s_i in range(n_seeds):
                for pert in (0, 1):
                    c = len(cols)
                    stim[:, c] = noises[m][s_i]
                    if pert:
                        add_pulse(stim, c, t_kick, kick)
                    cols.append((m, s_i, pert))
                    bias.append(mus[m][0])
        sp, V, tv = simulate(kind, T, np.array(bias), stim, record_v=True, v_dt=0.05e-3)
        out = {}
        for m_i, m in enumerate(mus):
            conv, dvs = [], []
            for s_i in range(n_seeds):
                a = 2 * (m_i * n_seeds + s_i)
                s0, s1 = sp[a], sp[a + 1]
                s0, s1 = s0[s0 >= t_kick], s1[s1 >= t_kick]
                un = [x for x in s0 if not np.any(np.abs(s1 - x) < tol)] + \
                     [x for x in s1 if not np.any(np.abs(s0 - x) < tol)]
                conv.append(max(un) - t_kick if un else 0.0)
                dvs.append(np.abs(V[a + 1] - V[a]))
            dv = np.mean(dvs, axis=0) / dV0
            post = tv >= t_kick
            pk = np.argmax(dv[post])
            below = np.where(dv[post][pk:] < 0.01)[0]
            t_dv = float(tv[post][pk + below[0]] - t_kick) if len(below) else None
            frac_latched = float(np.mean([np.mean(mus[m][0] + noises[m][s] > P.I_hold) for s in range(n_seeds)]))
            conv = np.array(conv)
            out[m] = dict(n_seeds=n_seeds, spikes_changed_frac=float(np.mean(conv > 0)),
                          convergence_median=float(np.median(conv)),
                          convergence_p90=float(np.percentile(conv, 90)),
                          convergence_max=float(conv.max()),
                          dV_below_1pct=t_dv, frac_time_above_I_hold=frac_latched,
                          rate=float(np.mean([len(sp[2 * (m_i * n_seeds + s)]) / T for s in range(n_seeds)])))
            axs[m_i].plot((tv[post] - t_kick) * 1e3, np.maximum(dv[post], 1e-6), color=COLORS[kind],
                          lw=1.5, label=LABELS[kind])
        res[kind] = out
    for m_i, m in enumerate(mus):
        axs[m_i].set_yscale('log'); axs[m_i].set_xlim(0, 200); axs[m_i].set_ylim(1e-5, 20)
        axs[m_i].set(title=m, xlabel='time after kick (ms)')
        style(axs[m_i])
    axs[0].set_ylabel('mean |ΔVm| / kick ΔV')
    axs[0].legend(frameon=False)
    fig.suptitle('P8  How long a 0.1·Q₁ kick changes the trajectory (30 frozen-noise seeds)', y=1.02)
    savefig(fig, 'P8_perturb.png')
    return res


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='P1,P2,P3,P4,P5,P6,P7,P8')
    parts = ap.parse_args().parts.split(',')
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'phase1_results.json')
    results = json.load(open(path)) if os.path.exists(path) else {}
    results['config'] = dict(dt=DT, seed=SEED, codegen='cython', msn_params=P.__dict__,
                             lif_matched=lif_config('lif_matched', P),
                             lif_generic=lif_config('lif_generic', P),
                             I_bias_sub=I_BIAS_SUB, pulse_width=PW)
    Q1 = None
    for name, fn in [('P1', p1_fi), ('P2', p2_step), ('P3', p3_ramp), ('P4', None),
                     ('P5', None), ('P6', p6_irregular), ('P7', p7_history), ('P8', None)]:
        if name not in parts:
            continue
        t0 = time.perf_counter()
        if name == 'P4':
            results['P4'], Q1 = p4_pairs()
        elif name in ('P5', 'P8'):
            if Q1 is None:
                Q1 = {k: v['Q1_nC'] * 1e-9 for k, v in results['P4'].items()}
            results[name] = (p5_trains if name == 'P5' else p8_perturb)(Q1)
        else:
            results[name] = fn()
        print(f'{name} done in {time.perf_counter() - t0:.1f} s', flush=True)
        with open(path, 'w') as fh:
            json.dump(results, fh, indent=1, default=float)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
