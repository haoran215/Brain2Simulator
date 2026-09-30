"""
fi_dt_check.py
==============
Step 1 of the MSN + STDP + MNIST plan: pick the simulation timestep.

The discrete MSN (msn_neuron.make_msn) with the hardware-fit parameters
(configs/neuron_hardware_fit.json) is driven by constant currents across its
spiking window.  For each dt the Brian2 F–I curve and closed-state duration
(spike width) are compared against the analytic MSN solution and against
the dt = 1 µs reference used in MNIST_test/bench_mnist.py.

Analytic MSN (piecewise-linear RC, I_in < I_hold):
    V_r     = I_hold·(Rm_lo + Ra)                  reopen voltage
    T_open  = τ_open  · ln((I_in·(Rm_hi+Ra) − V_r) / (I_in·(Rm_hi+Ra) − Vth))
    T_close = τ_close · ln((Vth − I_in·(Rm_lo+Ra)) / (V_r − I_in·(Rm_lo+Ra)))
    f       = 1 / (T_open + T_close)

    uv run python experiment/experiments/fi_dt_check.py
"""

import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import (prefs, start_scope, defaultclock, Network, SpikeMonitor,
                    StateMonitor, second, amp)

from msn_neuron import MSNParams, make_msn

prefs.codegen.target = 'cython'
prefs.logging.file_log = False

PARAMS_PATH = os.path.join(_ROOT, 'configs', 'neuron_hardware_fit.json')
PARAMS = MSNParams.from_json(PARAMS_PATH)
I_IN = np.array([42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 93]) * 1e-6   # A
DTS = [1e-6, 5e-6, 10e-6, 20e-6, 50e-6]                                    # s
T_SIM, T_SKIP = 0.5, 0.05                                                   # s
OUT_DIR = os.path.join(_ROOT, 'experiment', 'results', 'fi_dt_check')


def analytic(p: MSNParams, I):
    tau_o, tau_c = p.time_constants()
    V_r = p.I_hold * (p.Rm_lo + p.Ra)
    Vinf_o, Vinf_c = I * (p.Rm_hi + p.Ra), I * (p.Rm_lo + p.Ra)
    f = np.zeros_like(I)
    t_close = np.full_like(I, np.nan)
    ok = (Vinf_o > p.Vth) & (I < p.I_hold)
    T_o = tau_o * np.log((Vinf_o[ok] - V_r) / (Vinf_o[ok] - p.Vth))
    T_c = tau_c * np.log((p.Vth - Vinf_c[ok]) / (V_r - Vinf_c[ok]))
    f[ok] = 1 / (T_o + T_c)
    t_close[ok] = T_c
    return f, t_close


def simulate(dt):
    start_scope()
    defaultclock.dt = dt * second
    G = make_msn(len(I_IN), params=PARAMS)
    G.I_0 = I_IN * amp
    sm = SpikeMonitor(G)
    st = StateMonitor(G, 's', record=True, dt=dt * second)
    net = Network(G, sm, st)
    t0 = time.perf_counter()
    net.run(T_SIM * second)
    wall = time.perf_counter() - t0

    t, i = sm.t / second, np.asarray(sm.i)
    rate = np.array([np.sum((i == k) & (t >= T_SKIP)) for k in range(len(I_IN))]) / (T_SIM - T_SKIP)
    # closed time per spike = time with s=1 / number of spikes (after transient)
    keep = (st.t / second) >= T_SKIP
    closed = st.s[:, keep].sum(axis=1) * dt
    n = np.array([np.sum((i == k) & (t >= T_SKIP)) for k in range(len(I_IN))])
    t_close = np.where(n > 0, closed / np.maximum(n, 1), np.nan)
    return rate, t_close, wall


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(PARAMS.summary())
    f_an, tc_an = analytic(PARAMS, I_IN)

    res = {}
    for dt in DTS:
        rate, tc, wall = simulate(dt)
        res[dt] = (rate, tc, wall)
        print(f"dt={dt*1e6:4.0f} µs  wall={wall:6.2f} s")

    ref_rate = res[1e-6][0]
    lines = ["| I_in (µA) | analytic | " + " | ".join(f"dt={dt*1e6:.0f} µs" for dt in DTS) + " |",
             "|---:|---:|" + "---:|" * len(DTS)]
    for k, I in enumerate(I_IN):
        lines.append(f"| {I*1e6:.0f} | {f_an[k]:.0f} | " +
                     " | ".join(f"{res[dt][0][k]:.0f}" for dt in DTS) + " |")
    lines.append("")
    lines.append("| dt (µs) | max |Δf| vs analytic (Hz) | max |Δf| vs dt=1 µs (Hz) | "
                 "mean spike width (µs) | wall per sim. s (s) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for dt in DTS:
        rate, tc, wall = res[dt]
        lines.append(f"| {dt*1e6:.0f} | {np.max(np.abs(rate - f_an)):.1f} | "
                     f"{np.max(np.abs(rate - ref_rate)):.1f} | {np.nanmean(tc)*1e6:.0f} | "
                     f"{wall / T_SIM:.2f} |")
    lines.append(f"\nanalytic mean spike width: {np.nanmean(tc_an)*1e6:.0f} µs")
    table = "\n".join(lines)
    print(table)

    with open(os.path.join(OUT_DIR, 'fi_dt_check.md'), 'w') as fh:
        fh.write(f"params: {os.path.relpath(PARAMS_PATH, _ROOT)}\n"
                 f"T_sim={T_SIM}s (first {T_SKIP}s discarded), codegen=cython\n\n{table}\n")
    with open(os.path.join(OUT_DIR, 'fi_dt_check.json'), 'w') as fh:
        json.dump(dict(params=PARAMS.__dict__, I_in=I_IN.tolist(),
                       analytic=f_an.tolist(), analytic_t_close=tc_an.tolist(),
                       runs={f"{dt:g}": dict(rate=r.tolist(), t_close=t.tolist(), wall=w)
                             for dt, (r, t, w) in res.items()}), fh, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(I_IN * 1e6, f_an, 'k-', lw=2, label='analytic')
    for dt in DTS:
        ax[0].plot(I_IN * 1e6, res[dt][0], 'o--', ms=4, label=f'dt={dt*1e6:.0f} µs')
    ax[0].set(xlabel='I_in (µA)', ylabel='rate (Hz)', title='MSN F–I vs timestep')
    ax[0].legend(fontsize=8)
    for dt in DTS:
        ax[1].plot(I_IN * 1e6, res[dt][1] * 1e6, 'o--', ms=4, label=f'dt={dt*1e6:.0f} µs')
    ax[1].plot(I_IN * 1e6, tc_an * 1e6, 'k-', lw=2, label='analytic')
    ax[1].set(xlabel='I_in (µA)', ylabel='closed time / spike (µs)', title='Spike width')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fi_dt_check.png'), dpi=130)
    print(f"saved {OUT_DIR}")


if __name__ == '__main__':
    main()
