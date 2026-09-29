"""
proto_gate.py
=============
Standalone NumPy prototype for docs/MSN_HH_GATE_PLAN.md.

Compares, on the hardware-fit parameters (configs/neuron_hardware_fit.json):

  discrete   current MSN hysteretic switch (Stage 0 reference)
  bistable   Stage-2 gate  τx dx/dt = -x(x-θ)(x-1),
             θ = θ0 - aV·σ((V-Vth)/kV) + aI·σ((I_hold-I_M)/kI)   (adopted)
  or_gate    first-order gate, x∞ = σ_V OR σ_I          (rejected: depolarised rest)
  alpha_beta α(V) opens, β(I_M) closes                    (rejected: gate stalls at x≈1e-3)
  vonly      HH-style gate  τ dx/dt = x∞(V) - x           (no-go: never spikes)

against the analytical MSN F–I (README §6.2) and the measured F–I
(experiment_data_compare/IFcurve.xlsx, values copied below).

Does not import msn_neuron.py and changes nothing in the library.

    uv run python docs/gate_proto/proto_gate.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
P = json.loads((HERE.parents[1] / 'configs/neuron_hardware_fit.json').read_text())
Cm, Ra, Rhi, Rlo, Vth, Ih = (P[k] for k in ('Cm', 'Ra', 'Rm_hi', 'Rm_lo', 'Vth', 'I_hold'))
G_lo, G_hi = 1 / Rhi, 1 / Rlo          # memristor conductance, open / closed

# Measured F–I (IFcurve.xlsx): I [µA], f [Hz]
I_HW = np.array([41.3, 42.4, 43.6, 47.6, 52.5, 54.0, 56.0, 58.6, 61.0, 63.4,
                 65.8, 67.9, 71.4, 80.1, 91.5, 95.3, 97.7, 99.9, 103.4])
F_HW = np.array([0, 0, 49.4, 98.0, 117.0, 120.5, 128.44, 138.58, 144.9, 151.4,
                 159.0, 164.9, 173.1, 194.7, 212.0, 215.0, 211.0, 56.6, 0])

DT, T_END, T_SKIP = 0.1e-6, 0.15, 0.03   # s


def sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))   # overflow-safe logistic


def analytic(I):
    to, tc = Cm * (Rhi + Ra), Cm * (Rlo + Ra)
    Vh, Vo, Vc = Ih * (Rlo + Ra), I * (Rhi + Ra), I * (Rlo + Ra)
    with np.errstate(all='ignore'):
        T = to * np.log((Vo - Vh) / (Vo - Vth)) + tc * np.log((Vth - Vc) / (Vh - Vc))
    return np.where((I > Vth / (Rhi + Ra)) & (I < Ih), 1 / T, 0.0)


def channel_current(V, x, interp='G'):
    """I_M through the memristor in series with Ra.

    interp='G'  conductance linear in x (HH convention, thyristor ‖ R)
    interp='R'  resistance linear in x  (current msn_neuron.py form)
    Identical for x ∈ {0, 1}; differ for fractional x.
    """
    if interp == 'G':
        G = G_lo + x * (G_hi - G_lo)
        return V * G / (1 + Ra * G)
    return V / ((1 - x) * Rhi + x * Rlo + Ra)


# bistable-gate shape: θ = θ0 − aV·σ_V + aI·σ_I  (see MSN_HH_GATE_PLAN.md §4)
TH0, A_V, A_I, X_FLOOR = 0.5, 2.0, 1.0, 1e-6


def simulate(I, mode, kV=None, kI=None, rate=None, trace_idx=None, interp='G'):
    """Vectorised Euler over currents I [A]. Returns rates [Hz] (+ trace)."""
    n = len(I)
    V, x = np.zeros(n), np.zeros(n)
    up = np.zeros(n, bool)
    cnt = np.zeros(n); t_first = np.full(n, np.nan); t_last = np.full(n, np.nan)
    tr = []
    for k in range(int(T_END / DT)):
        t = k * DT
        IM = channel_current(V, x, interp)
        V += DT * (I - IM) / Cm
        if mode == 'discrete':                 # hysteretic event switch (Stage 0)
            x = np.where(x < 0.5, np.where(V > Vth, 1.0, 0.0),
                         np.where(IM < Ih, 0.0, 1.0))
        elif mode == 'or_gate':                # rejected: x∞ = V-trigger OR current-latch
            x_inf = 1 - (1 - sig((V - Vth) / kV)) * (1 - sig((IM - Ih) / kI))
            x += DT * rate * (x_inf - x)
        elif mode == 'alpha_beta':             # rejected: α(V) opens, β(I_M) closes
            a = rate * sig((V - Vth) / kV)
            b = rate * sig((Ih - IM) / kI)
            x += DT * (a * (1 - x) - b * x)
        elif mode == 'bistable':               # Stage 2 (adopted): self-latching cubic gate
            th = TH0 - A_V * sig((V - Vth) / kV) + A_I * sig((Ih - IM) / kI)
            x += DT * rate * (-x * (x - th) * (x - 1))
            x = np.clip(x, X_FLOOR, 1 - X_FLOOR)
        elif mode == 'vonly':                  # HH-style, V-only rates
            x += DT * rate * (sig((V - Vth) / kV) - x)
        on = x > 0.5
        new = on & ~up & (t > T_SKIP)
        cnt += new
        t_first = np.where(new & np.isnan(t_first), t, t_first)
        t_last = np.where(new, t, t_last)
        up = on
        if trace_idx is not None and k % 5 == 0:
            tr.append((t, V[trace_idx], x[trace_idx], IM[trace_idx] * Ra))
    with np.errstate(all='ignore'):
        f = np.where(cnt >= 3, (cnt - 1) / (t_last - t_first), 0.0)
    return (f, np.array(tr)) if trace_idx is not None else f


if __name__ == '__main__':
    I = np.unique(np.concatenate([I_HW, [44, 46, 85, 93, 94, 96, 99, 101]])) * 1e-6
    i68 = int(np.argmin(abs(I - 67.9e-6)))

    runs = {}
    runs['analytic'] = analytic(I)
    runs['discrete'], tr_d = simulate(I, 'discrete', trace_idx=i68, interp='R')
    configs = {
        'bistable steep  (kV=5 mV, kI=0.5 µA, τx=1 µs)': (5e-3, 0.5e-6, 1e6),
        'bistable medium (kV=20 mV, kI=2 µA, τx=1 µs)': (20e-3, 2e-6, 1e6),
        'bistable soft   (kV=50 mV, kI=5 µA, τx=5 µs)': (50e-3, 5e-6, 2e5),
    }
    tr_k = None
    for name, (kV, kI, r) in configs.items():
        if 'steep' in name:
            runs[name], tr_k = simulate(I, 'bistable', kV, kI, r, trace_idx=i68, interp='R')
        else:
            runs[name] = simulate(I, 'bistable', kV, kI, r, interp='R')
    # negative controls
    runs['bistable medium, G-linear'] = simulate(I, 'bistable', 20e-3, 2e-6, 1e6, interp='G')
    runs['OR-gate steep, G-linear'] = simulate(I, 'or_gate', 5e-3, 0.5e-6, 1e6, interp='G')
    runs['V-only gate (kV=20 mV, 1/µs)'] = simulate(I, 'vonly', 20e-3, None, 1e6, interp='G')

    # ── table ────────────────────────────────────────────────────────────
    hw = dict(zip(np.round(I_HW, 1), F_HW))
    names = list(runs)
    print('I(µA)   hw   ' + '  '.join(f'[{j}]' for j in range(len(names))))
    for j, nm in enumerate(names):
        print(f'  [{j}] {nm}')
    for i, Ii in enumerate(I):
        h = hw.get(round(Ii * 1e6, 1), np.nan)
        print(f'{Ii*1e6:6.1f} {h:5.0f} ' + ' '.join(f'{runs[nm][i]:5.0f}' for nm in names))

    wr = (I_HW > 43) & (I_HW < 93)
    sel = np.isin(np.round(I * 1e6, 1), np.round(I_HW[wr], 1))
    for nm in names:
        rmse = np.sqrt(np.mean((runs[nm][sel] - F_HW[wr]) ** 2))
        dev = np.max(abs(runs[nm][sel] - runs['analytic'][sel]))
        print(f'{nm:45s} working-range RMSE vs hw = {rmse:6.1f} Hz,'
              f' max |Δ| vs analytic = {dev:6.1f} Hz')

    # ── figure ───────────────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(I_HW, F_HW, 'ko', label='hardware (IFcurve.xlsx)')
    a1.plot(I * 1e6, runs['analytic'], 'k--', lw=1, label='analytic MSN')
    a1.plot(I * 1e6, runs['discrete'], 'x', color='gray', label='discrete MSN (Stage 0)')
    for nm, c in zip(configs, ['C0', 'C1', 'C2']):
        a1.plot(I * 1e6, runs[nm], '.-', color=c, label=' '.join(nm.split()))
    a1.plot(I * 1e6, runs['V-only gate (kV=20 mV, 1/µs)'], 's', mfc='none', color='C3',
            label='V-only HH gate (0 Hz)')
    a1.set(xlabel='I_in (µA)', ylabel='rate (Hz)', title='F–I: discrete switch vs continuous gate')
    a1.legend(fontsize=7)

    t0 = tr_d[:, 0]
    w = (t0 > 0.1) & (t0 < 0.1 + 3 / max(runs['discrete'][i68], 1))
    a2.plot(tr_d[w, 0] * 1e3, tr_d[w, 3], color='gray', lw=2, label='Vout discrete')
    a2.plot(tr_k[w, 0] * 1e3, tr_k[w, 3], 'C0', lw=1, label='Vout bistable (steep)')
    a2.plot(tr_k[w, 0] * 1e3, tr_k[w, 1], 'C0:', lw=1, label='Vm bistable')
    a2b = a2.twinx()
    a2b.plot(tr_k[w, 0] * 1e3, tr_k[w, 2], 'C3', lw=0.8, alpha=0.6)
    a2b.set_ylabel('gate x', color='C3')
    a2.set(xlabel='t (ms)', ylabel='V', title=f'Traces at I_in = {I[i68]*1e6:.1f} µA')
    a2.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(HERE / 'gate_proto_fi.png', dpi=130)
    print('saved', HERE / 'gate_proto_fi.png')
