"""ns_msn_compare.py — MSN only

This script runs and plots only the MSN model (SET C) from the original
comparison script. It produces a three-row figure showing:
  1) Full Vm trace, 2) single-spike zoom (Vout waveform), 3) I-F curve.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *
# Prefer Cython code generation for speed (requires system dev headers + Cython)
from brian2 import prefs
prefs.codegen.target = 'cython'

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
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
#
# Only the MSN (SET C) parameters and routines are kept below.

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

print("Simulating SET C (MSN) …")
C_t, C_Vm, C_Vout, C_sp = sim_MSN()
print(f"  {len(C_sp)} spikes in 300 ms  →  {len(C_sp)/0.300:.1f} Hz")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ I-F curves (analytical)                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
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

C_Isweep = np.linspace(0, 110e-6, 1500)
C_f = IF_MSN(C_Isweep)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Plot                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
C_A = '#2980B9'   # aLIF blue
C_B = '#E74C3C'   # thyristor red
C_C = '#16A085'   # MSN teal

fig = plt.figure(figsize=(8, 10))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45, wspace=0.2,
                        height_ratios=[1.2, 0.9, 1.1])

# Row 0: full Vm trace (MSN)
ax = fig.add_subplot(gs[0, 0])
ax.plot(C_t, C_Vm, color='#16A085', lw=1.0)
ax.axhline(C_Vth, color='dimgray', ls='--', lw=1, label=f'Vth = {C_Vth:.2f} V')
rate = len(C_sp)/((C_t[-1]-C_t[0])/1000)
ax.set_title(f'MSN  |  I = {C_I_drive*1e6:.1f} µA  →  f = {rate:.1f} Hz',
             fontsize=10, fontweight='bold', color='#16A085')
ax.set_xlabel('t (ms)'); ax.set_ylabel('Vm (V)')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# Row 1: single-spike zoom (Vout)
def zoom_around(t, y, t_event, before, after):
    m = (t >= t_event - before) & (t <= t_event + after)
    return t[m] - t_event, y[m]

ax = fig.add_subplot(gs[1, 0])
if len(C_sp):
    tz, yz = zoom_around(C_t, C_Vout*1e3, C_sp[0], 1, 30)
    ax.plot(tz, yz, color='#16A085', lw=1.6, label='Vout (mV)')
ax.axvline(0, color='k', ls=':', lw=1, label='close event')
ax.set_title('MSN — spike waveform (Vout) from Cm·(Rm_lo+Ra) discharge',
             fontsize=9, fontweight='bold', color='#16A085')
ax.set_xlabel('t − t_spike (ms)'); ax.set_ylabel('Vout (mV)')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# ─── Row 2: I-F curves ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.plot(C_Isweep*1e6, C_f, color='#16A085', lw=2.2)
ax.fill_between(C_Isweep*1e6, C_f, alpha=0.15, color='#16A085')
ax.axvline(C_Ihold*1e6, color='red', ls=':', lw=1,
           label=f'I_hold={C_Ihold*1e6:.0f} µA (depol block)')
ax.set_xlabel('I (µA)'); ax.set_ylabel('f (Hz)')
ax.set_title('MSN I-F   (emerges from τ_close)',
             fontsize=10, fontweight='bold', color='#16A085')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# ─── Suptitle / annotation table ────────────────────────────────────────────
fig.suptitle(
    'MSN (memristor switching)\n'
    'MSN replaces the reset rule with explicit Rm hysteresis → real spike waveform.',
    fontsize=12, fontweight='bold', y=1.005)

out_path = 'ns_msn_only.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Summary table                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\nMSN-only plot complete.")
