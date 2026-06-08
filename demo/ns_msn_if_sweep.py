"""
ns_msn_if_sweep.py
==================
I-F characterisation of the MSN model (inherits from ns_msn_v1.py).

For each value of the constant drive current I_in, runs the
paper-faithful memristor model and measures the steady-state mean
firing rate.  Compares with the analytical I-F formula derived from the
two-state dynamics.

─── Inheritance ─────────────────────────────────────────────────────────────
  ns_test.py / spike_Ra_sweep.py:  Rm fixed → analytical I-F was just LIF.
  ns_msn_v1.py:                    Rm hysteretic → real spike shape, single
                                   trace at one I_in.
  THIS FILE:                       Rm hysteretic → I-F across many I_in,
                                   exposing both excitability onsets that
                                   Wu et al. discuss in §2.

─── Operating window (Wu et al. 2023, §2) ───────────────────────────────────
  I_min = Vth / (Rm_hi + Ra)       (rheobase — open state must reach Vth)
  I_max = I_hold                    (closed state must fall to I_hold or it
                                     stays latched → depolarisation block)

─── Period decomposition ────────────────────────────────────────────────────
  T(I_in) = t_open(I_in) + t_close(I_in)
  t_open  = -τ_open  · ln(1 - Vth / [I_in·(Rm_hi+Ra)])
  t_close = -τ_close · ln[ (V_reopen - V_ss^c) / (Vth - V_ss^c) ]
  with V_ss^c = I_in·(Rm_lo+Ra),  V_reopen = I_hold·(Rm_lo+Ra),
       τ_open  = Cm·(Rm_hi+Ra),    τ_close = Cm·(Rm_lo+Ra).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

seed(42)
defaultclock.dt = 1*us

# ─── Parameters (identical to ns_msn_v1.py) ──────────────────────────────────
Cm_F      = 10e-7
Ra_ohm    = 47
Rm_hi_ohm = 100e3
Rm_lo_ohm = 500
Vth_V     = 2.0
I_hold_A  = 100e-6

# ─── Sweep configuration ─────────────────────────────────────────────────────
I_in_sweep_uA = np.concatenate([
    np.linspace(  5,  20,  6),       # below & at rheobase
    np.linspace( 22,  95, 25),       # main spiking range — fine grid
    np.linspace( 99, 110,  6),       # near & past depol-block
])
N      = len(I_in_sweep_uA)
T_run  = 800*ms
T_skip = 100*ms                       # discard initial transient

# ─── Brian2 model ────────────────────────────────────────────────────────────
Cm    = Cm_F      * farad
Ra    = Ra_ohm    * ohm
Rm_hi = Rm_hi_ohm * ohm
Rm_lo = Rm_lo_ohm * ohm
Vth   = Vth_V     * volt
I_hold = I_hold_A * amp

eqs = '''
dVm/dt = (I_in - Vm / (Rm_S + Ra)) / Cm   : volt
Rm_S   = (1 - s)*Rm_hi + s*Rm_lo           : ohm
I_M    = Vm / (Rm_S + Ra)                  : amp
I_in   : amp
s      : 1
'''

neurons = NeuronGroup(
    N, eqs,
    threshold='Vm > Vth and s < 0.5',
    reset='s = 1',
    events={'reopen': 'I_M < I_hold and s > 0.5'},
    method='euler',
)
neurons.run_on_event('reopen', 's = 0')
neurons.Vm   = 0*volt
neurons.s    = 0
neurons.I_in = I_in_sweep_uA * uA

sp_mon = SpikeMonitor(neurons)

print(f"Sweeping {N} drive currents from "
      f"{I_in_sweep_uA[0]:.1f} → {I_in_sweep_uA[-1]:.1f} µA, "
      f"each for {T_run/ms:.0f} ms ...")
run(T_run, report='text')

# ─── Numerical f from spike trains ───────────────────────────────────────────
f_num = np.zeros(N)
for i in range(N):
    t_sp = np.array(sp_mon.t[sp_mon.i == i] / ms)
    t_sp = t_sp[t_sp > T_skip/ms]
    if len(t_sp) >= 2:
        f_num[i] = 1000.0 / np.mean(np.diff(t_sp))
    elif len(t_sp) <= 1:
        f_num[i] = 0.0

# ─── Analytical I-F ──────────────────────────────────────────────────────────
def period_analytic(I_A):
    R_open  = Rm_hi_ohm + Ra_ohm
    R_close = Rm_lo_ohm + Ra_ohm
    Vss_o   = I_A * R_open
    Vss_c   = I_A * R_close
    Vreop   = I_hold_A * R_close
    if Vss_o <= Vth_V:
        return np.inf                       # quiescent (below rheobase)
    if I_A >= I_hold_A:
        return np.inf                       # depol block
    tau_o = Cm_F * R_open
    tau_c = Cm_F * R_close
    t_o = -tau_o * np.log(1 - Vth_V/Vss_o)
    t_c = -tau_c * np.log((Vreop - Vss_c) / (Vth_V - Vss_c))
    return t_o + t_c

I_dense_uA = np.linspace(0.5, 130, 2000)
f_anal = np.array([1.0/period_analytic(I*1e-6) for I in I_dense_uA])

I_min_uA = Vth_V / (Rm_hi_ohm + Ra_ohm) * 1e6     # rheobase
I_max_uA = I_hold_A * 1e6                          # depol block onset

# Quick analytical breakdown for plot-2: time spent in each phase
t_open_a, t_close_a = [], []
for I in I_dense_uA:
    R_open  = Rm_hi_ohm + Ra_ohm
    R_close = Rm_lo_ohm + Ra_ohm
    Vss_o = I*1e-6 * R_open
    Vss_c = I*1e-6 * R_close
    Vreop = I_hold_A * R_close
    if Vss_o <= Vth_V or I*1e-6 >= I_hold_A:
        t_open_a.append(np.nan); t_close_a.append(np.nan); continue
    t_open_a.append(-Cm_F*R_open  * np.log(1 - Vth_V/Vss_o))
    t_close_a.append(-Cm_F*R_close* np.log((Vreop-Vss_c)/(Vth_V-Vss_c)))
t_open_a  = np.array(t_open_a) * 1e3
t_close_a = np.array(t_close_a) * 1e3

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         gridspec_kw=dict(wspace=0.3))

# Left: I-F curve
ax = axes[0]
ax.plot(I_dense_uA, f_anal, color='#34495E', lw=2.2, label='analytical',
        zorder=2)
ax.plot(I_in_sweep_uA, f_num, 'o', ms=6, color='#E67E22',
        label='Brian2 numerical', zorder=3)

ax.axvspan(0, I_min_uA, alpha=0.10, color='gray')
ax.axvspan(I_max_uA, I_dense_uA[-1], alpha=0.10, color='red')
ax.axvline(I_min_uA, color='gray', ls=':', lw=1.2)
ax.axvline(I_max_uA, color='red',  ls=':', lw=1.2)

ymax_for_text = max(np.nanmax(f_anal[np.isfinite(f_anal)]),
                    np.nanmax(f_num)) * 1.10
ax.text(I_min_uA + 1, ymax_for_text*0.55,
        f'I_min = {I_min_uA:.1f} µA\n= Vth/(Rm_hi+Ra)',
        fontsize=8.5, color='dimgray',
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
ax.text(I_max_uA - 1, ymax_for_text*0.55,
        f'I_max = {I_max_uA:.0f} µA\n= I_hold',
        fontsize=8.5, color='#C0392B', ha='right',
        bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#E74C3C', alpha=0.85))
ax.text(I_min_uA*0.4, ymax_for_text*0.85, 'quiescent\n(below rheobase)',
        fontsize=9, color='gray', ha='center', style='italic')
ax.text((I_max_uA+I_dense_uA[-1])/2, ymax_for_text*0.85,
        'depolarisation\nblock',
        fontsize=9, color='#C0392B', ha='center', style='italic')

ax.set_xlabel('I_in  (µA)', fontsize=11)
ax.set_ylabel('Firing rate f  (Hz)', fontsize=11)
ax.set_title('MSN I-F curve  (type-1 excitability, two-sided onsets)',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_xlim(0, I_dense_uA[-1])
ax.set_ylim(0, ymax_for_text)

# Right: period decomposition (where the time goes)
ax = axes[1]
ax.plot(I_dense_uA, t_open_a,  color='#2980B9', lw=2.0,
        label='t_open (rise: leak through Rm_hi)')
ax.plot(I_dense_uA, t_close_a, color='#27AE60', lw=2.0,
        label='t_close (discharge through Rm_lo)')
ax.plot(I_dense_uA, t_open_a + t_close_a, color='k', lw=1.4, ls='--',
        label='T = 1/f')
ax.axvline(I_min_uA, color='gray', ls=':', lw=1.2)
ax.axvline(I_max_uA, color='red',  ls=':', lw=1.2)
ax.axvspan(0, I_min_uA, alpha=0.10, color='gray')
ax.axvspan(I_max_uA, I_dense_uA[-1], alpha=0.10, color='red')
ax.set_xlabel('I_in  (µA)', fontsize=11)
ax.set_ylabel('Time per cycle  (ms)', fontsize=11)
ax.set_title('Period decomposition: open vs closed phase',
             fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.25, which='both')
ax.set_xlim(0, I_dense_uA[-1])

fig.suptitle(
    rf'MSN I-F sweep  |  $C_m={Cm_F*1e6:.0f}\,\mu F$,  '
    rf'$R_m^{{hi}}={Rm_hi_ohm/1e3:.0f}\,k\Omega$,  '
    rf'$R_m^{{lo}}={Rm_lo_ohm:.0f}\,\Omega$,  '
    rf'$R_a={Ra_ohm}\,\Omega$,  '
    rf'$V_{{th}}={Vth_V}\,V$,  '
    rf'$I_{{hold}}={I_hold_A*1e6:.0f}\,\mu A$',
    fontsize=12, fontweight='bold', y=1.02)

out_path = 'ns_msn_if_sweep.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")

print("\nNumerical I-F samples:")
for I, f in zip(I_in_sweep_uA, f_num):
    print(f"  I = {I:6.1f} µA  →  f = {f:7.2f} Hz")
