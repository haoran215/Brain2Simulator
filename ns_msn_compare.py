"""
ns_msn_compare.py
=================
Side-by-side comparison of three single-neuron models in this repo.

  SET A — aLIF        (modelcopare.py SET A)
          series  Rm+Ra,  fixed Rm = Rm_hi,  threshold+reset rule (Vm→0).
          NO spike shape; refractoriness = t_ref parameter.

  SET B — Thyristor   (modelcopare.py SET B)
          parallel  ga ∥ gg,  threshold+reset to Vr = IH/ga ≠ 0.
          NO spike shape; refractoriness = tn parameter.

  SET C — MSN         (ns_msn_v1.py — this thread)
          series  Rm+Ra,  HYSTERETIC Rm (state machine Rm_hi ↔ Rm_lo).
          REAL spike shape: Cm discharges through Rm_lo+Ra during closed phase.
          NO t_ref; refractoriness emerges from τ_close = Cm·(Rm_lo+Ra).
          Wu et al. 2023 §2, Fig. 1e + Fig. 2 (paper-faithful).

  (MSBN — not implemented here. Requires an extra Cs+Rs second compartment
   to support bursting; Wu et al. §3 / Fig. 3.)

Plot rows
  1. Full Vm traces — qualitative shape across the operating range.
  2. Single-spike zoom — instant reset vs real waveform.
  3. I-F curves — three native scales (parameters span 4 orders of magnitude).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ─── SET A — aLIF (paper Eqs. 9–12) ──────────────────────────────────────────
A_Cm     = 40.15e-9
A_Ra     = 10e3
A_Rm_hi  = 98e3
A_Vth    = 4.0
A_Vr     = 0.0
A_tref   = 3e-3
A_Rtot   = A_Rm_hi + A_Ra
A_taum   = A_Cm * A_Rtot
A_I_drive = 55e-6           # → ~127 Hz

# ─── SET B — Thyristor working fit ───────────────────────────────────────────
B_C      = 3.3e-7
B_ga     = 1/2.2e3
B_gg     = 1/680e3
B_IT     = 4.95e-6
B_IH     = 105e-6
B_VT     = B_IT / B_gg      # 3.366 V
B_Vr     = B_IH / B_ga      # 0.231 V
B_tn     = 60e-3
B_taum   = B_C / (B_ga + B_gg)
B_Irheo  = B_VT * (B_ga + B_gg)
B_I_drive = B_Irheo * 1.5

# ─── SET C — MSN (paper Fig. 2) ──────────────────────────────────────────────
C_Cm     = 10e-7
C_Ra     = 47
C_Rm_hi  = 100e3
C_Rm_lo  = 500
C_Vth    = 1.5
C_Ihold  = 100e-6
C_I_drive = 92.4e-6
C_taum_hi = C_Cm * (C_Rm_hi + C_Ra)
C_taum_lo = C_Cm * (C_Rm_lo + C_Ra)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Simulations                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def sim_aLIF():
    start_scope()
    defaultclock.dt = 0.01*ms
    Cm     = A_Cm   * farad
    Rtot   = A_Rtot * ohm
    Vth_q  = A_Vth  * volt
    Vr_q   = A_Vr   * volt
    Iinj_q = A_I_drive * amp
    eqs = 'dVm/dt = (-Vm/Rtot + Iinj_q) / Cm : volt'
    G = NeuronGroup(1, eqs, threshold='Vm > Vth_q',
                    reset='Vm = Vr_q', refractory=A_tref*second, method='euler')
    G.Vm = 0*volt
    sm = StateMonitor(G, 'Vm', record=True, dt=0.01*ms)
    sp = SpikeMonitor(G)
    run(150*ms)
    return np.array(sm.t/ms), np.array(sm.Vm[0]/volt), np.array(sp.t/ms)

def sim_thyristor():
    start_scope()
    defaultclock.dt = 0.05*ms
    C     = B_C  * farad
    ga    = B_ga * siemens
    gg    = B_gg * siemens
    VT_q  = B_VT * volt
    Vr_q  = B_Vr * volt
    Iinj_q = B_I_drive * amp
    eqs = 'dVm/dt = (-(ga+gg)*Vm + Iinj_q) / C : volt'
    G = NeuronGroup(1, eqs, threshold='Vm > VT_q',
                    reset='Vm = Vr_q', refractory=B_tn*second, method='euler')
    G.Vm = B_Vr*volt
    sm = StateMonitor(G, 'Vm', record=True, dt=0.05*ms)
    sp = SpikeMonitor(G)
    run(800*ms)
    return np.array(sm.t/ms), np.array(sm.Vm[0]/volt), np.array(sp.t/ms)

def sim_MSN():
    start_scope()
    defaultclock.dt = 1*us
    Cm     = C_Cm    * farad
    Ra     = C_Ra    * ohm
    Rm_hi  = C_Rm_hi * ohm
    Rm_lo  = C_Rm_lo * ohm
    Vth_q  = C_Vth   * volt
    Ihold_q = C_Ihold * amp
    Iin_q  = C_I_drive * amp
    eqs = '''
    dVm/dt = (Iin_q - Vm/(Rm_S + Ra)) / Cm   : volt
    Rm_S   = (1 - s)*Rm_hi + s*Rm_lo          : ohm
    I_M    = Vm / (Rm_S + Ra)                 : amp
    Vout   = Vm * Ra / (Rm_S + Ra)            : volt
    s      : 1
    '''
    G = NeuronGroup(1, eqs,
                    threshold='Vm > Vth_q and s < 0.5',
                    reset='s = 1',
                    events={'reopen': 'I_M < Ihold_q and s > 0.5'},
                    method='euler')
    G.run_on_event('reopen', 's = 0')
    G.Vm = 0*volt
    G.s  = 0
    sm = StateMonitor(G, ['Vm', 'Vout'], record=True, dt=2*us)
    sp = SpikeMonitor(G)
    run(300*ms)
    return (np.array(sm.t/ms),
            np.array(sm.Vm[0]/volt),
            np.array(sm.Vout[0]/volt),
            np.array(sp.t/ms))

print("Simulating SET A (aLIF) …")
A_t, A_Vm, A_sp = sim_aLIF()
print(f"  {len(A_sp)} spikes in 150 ms  →  {len(A_sp)/0.150:.0f} Hz")

print("Simulating SET B (Thyristor) …")
B_t, B_Vm, B_sp = sim_thyristor()
print(f"  {len(B_sp)} spikes in 800 ms  →  {len(B_sp)/0.800:.1f} Hz")

print("Simulating SET C (MSN) …")
C_t, C_Vm, C_Vout, C_sp = sim_MSN()
print(f"  {len(C_sp)} spikes in 300 ms  →  {len(C_sp)/0.300:.1f} Hz")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ I-F curves (analytical)                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def IF_aLIF(I_arr):
    f = np.zeros_like(I_arr)
    for k, I in enumerate(I_arr):
        Vss = I*A_Rtot
        if Vss <= A_Vth: continue
        tc = -A_taum * np.log(1 - (A_Vth - A_Vr)/(Vss - A_Vr))
        f[k] = 1.0/(tc + A_tref)
    return f

def IF_thyristor(I_arr):
    f = np.zeros_like(I_arr)
    for k, I in enumerate(I_arr):
        Vss = I/(B_ga + B_gg)
        if Vss <= B_VT: continue
        tc = B_taum * np.log((Vss - B_Vr)/(Vss - B_VT))
        f[k] = 1.0/(tc + B_tn)
    return f

def IF_MSN(I_arr):
    f = np.zeros_like(I_arr)
    R_o = C_Rm_hi + C_Ra
    R_c = C_Rm_lo + C_Ra
    tau_o = C_Cm * R_o
    tau_c = C_Cm * R_c
    Vreop = C_Ihold * R_c
    for k, I in enumerate(I_arr):
        Vss_o = I*R_o
        Vss_c = I*R_c
        if Vss_o <= C_Vth or I >= C_Ihold: continue
        t_o = -tau_o * np.log(1 - C_Vth/Vss_o)
        t_c = -tau_c * np.log((Vreop - Vss_c)/(C_Vth - Vss_c))
        f[k] = 1.0/(t_o + t_c)
    return f

A_Isweep = np.linspace(0, 160e-6, 1500)
B_Isweep = np.linspace(0, 8e-3,   1500)
C_Isweep = np.linspace(0, 110e-6, 1500)
A_f = IF_aLIF(A_Isweep)
B_f = IF_thyristor(B_Isweep)
C_f = IF_MSN(C_Isweep)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Plot                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
C_A = '#2980B9'   # aLIF blue
C_B = '#E74C3C'   # thyristor red
C_C = '#16A085'   # MSN teal

fig = plt.figure(figsize=(20, 13))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.32,
                        height_ratios=[1.2, 1.0, 1.1])

# ─── Row 0: Vm traces ────────────────────────────────────────────────────────
for col, (t, Vm, sp, c, name, vth, vr, drive_label) in enumerate([
    (A_t, A_Vm, A_sp, C_A, 'SET A — aLIF',
     A_Vth, A_Vr,  f'I = {A_I_drive*1e6:.0f} µA'),
    (B_t, B_Vm, B_sp, C_B, 'SET B — Thyristor',
     B_VT, B_Vr,  f'I = {B_I_drive*1e3:.2f} mA'),
    (C_t, C_Vm, C_sp, C_C, 'SET C — MSN (new)',
     C_Vth, None, f'I = {C_I_drive*1e6:.1f} µA'),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t, Vm, color=c, lw=0.8)
    ax.axhline(vth, color='dimgray', ls='--', lw=1, label=f'Vth = {vth:.2f} V')
    if vr is not None:
        ax.axhline(vr, color='green', ls=':', lw=1,
                   label=f'Vreset = {vr*1e3:.0f} mV' if vr*1e3 < 100
                         else f'Vreset = {vr:.2f} V')
    rate = len(sp)/((t[-1]-t[0])/1000)
    ax.set_title(f'{name}\nVm(t)  |  {drive_label}  →  f = {rate:.1f} Hz',
                 fontsize=10, fontweight='bold', color=c)
    ax.set_xlabel('t (ms)'); ax.set_ylabel('Vm (V)')
    ax.legend(fontsize=8, loc='upper right')

# ─── Row 1: single-spike zoom ────────────────────────────────────────────────
def zoom_around(t, y, t_event, before, after):
    m = (t >= t_event - before) & (t <= t_event + after)
    return t[m] - t_event, y[m]

# SET A — instant reset (Vm jumps, then re-rises)
ax = fig.add_subplot(gs[1, 0])
if len(A_sp):
    tz, yz = zoom_around(A_t, A_Vm, A_sp[0], 2.0, 6.0)
    ax.plot(tz, yz, color=C_A, lw=1.6)
ax.axvline(0, color='k', ls=':', lw=1, label='spike event')
ax.set_title('aLIF — "spike" is just a reset rule\n(no waveform; Vm jumps to 0)',
             fontsize=9, fontweight='bold', color=C_A)
ax.set_xlabel('t − t_spike (ms)'); ax.set_ylabel('Vm (V)')
ax.legend(fontsize=8)

# SET B — instant reset to Vr
ax = fig.add_subplot(gs[1, 1])
if len(B_sp):
    tz, yz = zoom_around(B_t, B_Vm, B_sp[0], 5, 60)
    ax.plot(tz, yz, color=C_B, lw=1.6)
ax.axvline(0, color='k', ls=':', lw=1, label='spike event')
ax.set_title('Thyristor — reset to Vr=IH/ga\n(no waveform; non-zero reset)',
             fontsize=9, fontweight='bold', color=C_B)
ax.set_xlabel('t − t_spike (ms)'); ax.set_ylabel('Vm (V)')
ax.legend(fontsize=8)

# SET C — REAL spike: show Vout (paper Fig. 2 right inset)
ax = fig.add_subplot(gs[1, 2])
if len(C_sp):
    tz, yz = zoom_around(C_t, C_Vout*1e3, C_sp[0], 1, 30)
    ax.plot(tz, yz, color=C_C, lw=1.8, label='Vout (paper trace)')
ax.axvline(0, color='k', ls=':', lw=1, label='close event')
ax.set_title('MSN — REAL spike waveform (Vout)\nemerges from Cm·(Rm_lo+Ra) discharge',
             fontsize=9, fontweight='bold', color=C_C)
ax.set_xlabel('t − t_spike (ms)'); ax.set_ylabel('Vout (mV)')
ax.legend(fontsize=8, loc='upper right')

# ─── Row 2: I-F curves ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.plot(A_Isweep*1e6, A_f, color=C_A, lw=2.2)
ax.fill_between(A_Isweep*1e6, A_f, alpha=0.15, color=C_A)
ax.axhline(1/A_tref, color=C_A, ls=':', lw=1,
           label=f'f_max=1/t_ref={1/A_tref:.0f} Hz')
ax.set_xlabel('I (µA)'); ax.set_ylabel('f (Hz)')
ax.set_title(f'aLIF I-F   (t_ref={A_tref*1e3:.0f} ms)',
             fontsize=10, fontweight='bold', color=C_A)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[2, 1])
ax.plot(B_Isweep*1e3, B_f, color=C_B, lw=2.2)
ax.fill_between(B_Isweep*1e3, B_f, alpha=0.15, color=C_B)
ax.axhline(1/B_tn, color=C_B, ls=':', lw=1,
           label=f'f_max=1/tn={1/B_tn:.1f} Hz')
ax.set_xlabel('I (mA)'); ax.set_ylabel('0.9 (Hz)')
ax.set_title(f'Thyristor I-F   (tn={B_tn*1e3:.0f} ms)',
             fontsize=10, fontweight='bold', color=C_B)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[2, 2])
ax.plot(C_Isweep*1e6, C_f, color=C_C, lw=2.2)
ax.fill_between(C_Isweep*1e6, C_f, alpha=0.15, color=C_C)
ax.axvline(C_Ihold*1e6, color='red', ls=':', lw=1,
           label=f'I_hold={C_Ihold*1e6:.0f} µA (depol block)')
ax.set_xlabel('I (µA)'); ax.set_ylabel('f (Hz)')
ax.set_title('MSN I-F   (no t_ref; emerges from τ_close)',
             fontsize=10, fontweight='bold', color=C_C)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# ─── Suptitle / annotation table ────────────────────────────────────────────
fig.suptitle(
    'Three single-neuron models in this repo  —  '
    'aLIF (abstract)  vs  Thyristor (parallel topology)  vs  MSN (memristor switching)\n'
    'Key distinction: aLIF & Thyristor BOTH use Brian2 threshold+reset (no spike shape).  '
    'MSN replaces the reset rule with explicit Rm hysteresis → real spike waveform.',
    fontsize=12, fontweight='bold', y=1.005)

out_path = 'ns_msn_compare.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Summary table                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  STRUCTURAL COMPARISON")
print("="*78)
rows = [
    ("Topology",       "Series Rm+Ra",        "Parallel ga∥gg",       "Series Rm+Ra"),
    ("Rm during spike","fixed = Rm_hi",       "(no Rm)",               "STATE: Rm_hi↔Rm_lo"),
    ("Spike emit",     "threshold+reset",     "threshold+reset",       "natural close event"),
    ("Vm during spike","INSTANT jump to Vr=0","INSTANT jump to Vr",   "CONTINUOUS discharge"),
    ("Spike shape",    "none",                "none",                  "real (paper Fig. 2)"),
    ("Refractory",     f"t_ref={A_tref*1e3:.0f} ms (param)",
                       f"tn={B_tn*1e3:.0f} ms (param)",
                       f"τ_close={C_taum_lo*1e3:.2f} ms (emergent)"),
    ("Cm",             f"{A_Cm*1e9:.1f} nF",
                       f"{B_C*1e9:.0f} nF",
                       f"{C_Cm*1e6:.0f} µF"),
    ("Time scale",     "ms",                  "ms",                    "ms (paper Fig. 2)"),
    ("Source file",    "modelcopare.py",      "modelcopare.py",        "ns_msn_v1.py"),
]
print(f"  {'':<22}{'SET A — aLIF':<24}{'SET B — Thyristor':<25}{'SET C — MSN'}")
print("-"*78)
for r in rows:
    print(f"  {r[0]:<22}{r[1]:<24}{r[2]:<25}{r[3]}")
print("="*78)
print("  Note: MSBN (bursting) needs a second Cs+Rs compartment — not in this repo yet.")
print("="*78)
