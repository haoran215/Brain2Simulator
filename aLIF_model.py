"""
aLIF_model.py — Memristor aLIF Neuron (Eqs. 9-12), standalone reference
========================================================================
Standalone counterpart of modules/neuron.py.  Same solver, same equations,
inline so the math is visible.  Use the modules for production work; use
this file when you want to read or tweak the formulation.

Tuning principle
----------------
You measure (I_min, I_max) on the device and pick (f_min, f_max) targets.
The solver then computes  (Cm, t_ref)  so that
    f(I_min) = f_min     and     f(I_max) = f_max
simultaneously, using the LIF I-F formula
    f(I) = 1 / [ -tau_m · ln(1 - Vth / Vm_ss(I)) + t_ref ]
with Vm_ss(I) = (I + I_0)·(R_m^hi + R_a) and tau_m = Cm·(R_m^hi+R_a).

Two quantities that are easy to confuse — read this once
--------------------------------------------------------
  f at I_max     : firing rate at the OPERATING upper bound (= f_max by
                   construction).  This is what the neuron actually does.
  1 / t_ref      : ASYMPTOTIC ceiling of the I-F curve as I → ∞ (where
                   t_cross → 0).  Not an operational rate — the neuron
                   never sees I high enough to reach it.  With sensible
                   targets (this file: 50→100 Hz over 40→100 µA) the
                   asymptote sits ~2× above f_max.  If your targets push
                   it to ~10× f_max, you've picked targets too close to
                   the feasibility boundary (see below).

Feasibility constraint
----------------------
For t_ref > 0, the LIF I-F formula requires
        B / A   ≤   f_min / f_max
with    A = -ln(1 - Vth/Vm_ss(I_min))    (curve shape near rheobase)
        B = -ln(1 - Vth/Vm_ss(I_max))    (curve shape at upper bound).
At equality the solver gives t_ref = 0 (and asymptote = ∞).  As your
target ratio f_min/f_max moves further above B/A, t_ref grows, the
asymptote drops, and the solved Cm grows — i.e. parameters become more
"rational".  The assertion below catches the violating case explicitly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *

seed(42)
defaultclock.dt = 0.05*ms
duration        = 10*second

# =============================================================================
# [USER CHOICE 1]  Which synaptic stage drives Vm?
# =============================================================================
#   'Is1' -> exponential decay  (one leaky stage)  — AMPA-like, feedforward
#   'Is2' -> alpha function     (two leaky stages) — NMDA-like, recurrent
NETWORK_MODE = 'Is2'

# =============================================================================
# [USER SPECS]  I-F operating targets  (rational defaults)
# =============================================================================
# Operating currents — measure these on your device.
I_min_uA =  40.0     # µA  — lower operating point
I_max_uA = 100.0     # µA  — upper operating point

# Desired firing rates at those currents.
# 50/100 Hz over 40→100 µA gives a non-degenerate t_ref (~5 ms) and an
# asymptotic ceiling 1/t_ref ≈ 200 Hz, only ~2× f_max.  Earlier defaults
# (70/200 Hz) sat at the feasibility edge B/A ≈ 0.334 ≈ f_min/f_max =
# 0.35, squeezing t_ref to ~0.33 ms and inflating 1/t_ref to ~3 kHz.
f_min_Hz =  50.0
f_max_Hz = 100.0

# =============================================================================
# Circuit parameters  (thyristor hardware reference)
# =============================================================================
Ra_ohm    =  2200    # Ω    axon-hillock resistor
Rm_hi_ohm = 100e3    # Ω    memristor HIGH (open switch — quiescent)
Rm_lo_ohm = 100      # Ω    memristor LOW  (closed switch — only used by MSN)
Vthresh_V =  4.0     # V    spike threshold (thyristor gate)

Rs_ohm    =  10e3
Cs_F      = 1000e-9
# → tau_s1 = tau_s2 = Rs*Cs = 10 ms

Iw_exc_uA = 20.0
Iw_inh_uA = 30.0

# Tonic bias.
#   I_0 = 0     → silent without synaptic input
#   I_0 > 0     → closer to threshold; reduces synaptic drive needed
#   I_0 ≥ I_min → tonic spiking even without input (oscillator)
I_0_uA    = 15.0

# =============================================================================
# Analytical solver: find (tau_m, t_ref) → Cm from two (I, f) targets
# =============================================================================
R_tot_ohm = Rm_hi_ohm + Ra_ohm
Vm_ss_min = (I_min_uA*1e-6 + I_0_uA*1e-6) * R_tot_ohm
Vm_ss_max = (I_max_uA*1e-6 + I_0_uA*1e-6) * R_tot_ohm

assert Vm_ss_min > Vthresh_V, (
    f"Vm_ss(I_min+I_0) = {Vm_ss_min:.3f} V <= Vth = {Vthresh_V} V — "
    f"neuron can't fire at I_min!")
assert Vm_ss_max > Vthresh_V, (
    f"Vm_ss(I_max+I_0) = {Vm_ss_max:.3f} V <= Vth = {Vthresh_V} V — "
    f"neuron can't fire at I_max!")

A = -np.log(1 - Vthresh_V / Vm_ss_min)   # > 0
B = -np.log(1 - Vthresh_V / Vm_ss_max)   # > 0
assert A > B > 0

ratio_BA   = B / A
ratio_fmf  = f_min_Hz / f_max_Hz

if ratio_BA > ratio_fmf:
    raise AssertionError(
        f"Infeasible targets:  B/A = {ratio_BA:.3f}  >  f_min/f_max = "
        f"{ratio_fmf:.3f}.\nWith these (Vth, Rm, Ra, I_0), no positive "
        f"t_ref can satisfy both targets simultaneously.\nFix by either:\n"
        f"  - raising f_min  (→ larger f_min/f_max),\n"
        f"  - lowering f_max (→ larger f_min/f_max),\n"
        f"  - lowering R_total (→ smaller B/A — Vth becomes a bigger\n"
        f"    fraction of Vm_ss, curve gets steeper),\n"
        f"  - reducing I_0    (→ smaller B/A by the same logic).")

tau_m_s = (1/f_min_Hz - 1/f_max_Hz) / (A - B)
tc_min  = tau_m_s * A
tc_max  = tau_m_s * B
t_ref_s = 1/f_min_Hz - tc_min

assert t_ref_s > 0, (
    f"Solved t_ref = {t_ref_s*1e3:.3f} ms ≤ 0 — should not happen if the "
    f"feasibility check above passed.  Numerical issue?")

Cm_F = tau_m_s / R_tot_ohm

# Verify
f_check_min = 1/(tc_min + t_ref_s)
f_check_max = 1/(tc_max + t_ref_s)
f_asymptote = 1/t_ref_s   # the I → ∞ ceiling — NOT the rate at I_max

print("=" * 72)
print("  Solved aLIF parameters from I-F targets")
print("=" * 72)
print(f"  Inputs (specs):")
print(f"    Vth     = {Vthresh_V:.1f} V    Ra = {Ra_ohm/1e3:.2f} kΩ    "
      f"Rm_hi = {Rm_hi_ohm/1e3:.0f} kΩ")
print(f"    R_total = Rm_hi + Ra = {R_tot_ohm/1e3:.2f} kΩ")
print(f"    I_0     = {I_0_uA:.1f} µA   (tonic bias)")
print(f"    target  f({I_min_uA:.0f} µA) = {f_min_Hz:.0f} Hz   "
      f"f({I_max_uA:.0f} µA) = {f_max_Hz:.0f} Hz")
print()
print(f"  Feasibility:")
print(f"    Vm_ss(I_min+I_0) = {Vm_ss_min:.2f} V   "
      f"Vm_ss(I_max+I_0) = {Vm_ss_max:.2f} V")
print(f"    A = -ln(1-Vth/Vm_ss_min) = {A:.3f}    "
      f"B = -ln(1-Vth/Vm_ss_max) = {B:.3f}")
print(f"    B/A = {ratio_BA:.3f}   f_min/f_max = {ratio_fmf:.3f}   "
      f"({'feasible' if ratio_BA < ratio_fmf else 'TIGHT'})")
print()
print(f"  Solved:")
print(f"    tau_m = {tau_m_s*1e3:.2f} ms     "
      f"Cm    = {Cm_F*1e9:.2f} nF     t_ref = {t_ref_s*1e3:.2f} ms")
print()
print(f"  Verify (from solved params):")
print(f"    f at I_min ({I_min_uA:.0f} µA) = {f_check_min:.1f} Hz   "
      f"(target {f_min_Hz:.0f})")
print(f"    f at I_max ({I_max_uA:.0f} µA) = {f_check_max:.1f} Hz   "
      f"(target {f_max_Hz:.0f})")
print()
print(f"  Reference (do not confuse with f at I_max):")
print(f"    1/t_ref = {f_asymptote:.0f} Hz  ← asymptotic I-F ceiling at I→∞.")
print(f"    Operationally f never reaches this; with target f_max = "
      f"{f_max_Hz:.0f} Hz")
print(f"    the asymptote sits {f_asymptote/f_max_Hz:.1f}× above the "
      f"upper operating bound.")
print()
print(f"  Synapse: tau_s1 = tau_s2 = {Rs_ohm*Cs_F*1e3:.0f} ms   "
      f"Iw_exc = {Iw_exc_uA:.0f} µA   Iw_inh = {Iw_inh_uA:.0f} µA")
print(f"  Mode:    NETWORK_MODE = '{NETWORK_MODE}'")
print("=" * 72)

# =============================================================================
# Brian2 quantities
# =============================================================================
Cm      = Cm_F      * farad
Ra      = Ra_ohm    * ohm
Rm_hi   = Rm_hi_ohm * ohm
Rm_lo   = Rm_lo_ohm * ohm
Vthresh = Vthresh_V * volt
Vreset  = 0         * volt
t_ref   = t_ref_s   * second
Rs      = Rs_ohm    * ohm
Cs      = Cs_F      * farad
tau_s1  = Rs * Cs
tau_s2  = Rs * Cs
Iw_exc  = Iw_exc_uA * uA
Iw_inh  = Iw_inh_uA * uA
I_0     = I_0_uA    * uA

# =============================================================================
# Analytical I-F curve  (with I_0 bias, depol-block above I_max)
# =============================================================================
I_sweep_uA = np.linspace(0, I_max_uA * 1.6, 3000)
Vm_ss_arr  = (I_sweep_uA*1e-6 + I_0_uA*1e-6) * R_tot_ohm
f_IF       = np.zeros_like(I_sweep_uA)
firing     = (Vm_ss_arr > Vthresh_V) & (I_sweep_uA <= I_max_uA)
tc_arr     = -tau_m_s * np.log(1 - Vthresh_V / Vm_ss_arr[firing])
f_IF[firing] = 1.0 / (tc_arr + t_ref_s)
# Above I_max: f stays 0 (depol-block placeholder — pure aLIF doesn't really
# do depol-block; this is the convention from the paper / module solver).

# =============================================================================
# Poisson inputs (sinusoidally modulated rates)
# =============================================================================
rate_res     = 100*ms
n_steps      = int(duration / rate_res)
t_vals       = np.arange(n_steps) * float(rate_res/second)

exc_rates    = (70 + 50*np.sin(2*np.pi*0.15*t_vals)) * Hz       # 20-120 Hz
inh_rates    = (50 + 30*np.sin(2*np.pi*0.15*t_vals + np.pi)) * Hz   # 20-80 Hz
exc_rate_arr = TimedArray(exc_rates, dt=rate_res)
inh_rate_arr = TimedArray(inh_rates, dt=rate_res)
exc_input    = PoissonGroup(2, rates='exc_rate_arr(t)')
inh_input    = PoissonGroup(2, rates='inh_rate_arr(t)')

# =============================================================================
# Neuron equations  (Eqs. 9-12)
# =============================================================================
if NETWORK_MODE == 'Is1':
    eqs = '''
    dVm/dt      = (-Vm/(Rm_S+Ra) + Is1_exc - Is1_inh + I_0) / Cm   : volt
    dIs1_exc/dt = -Is1_exc / tau_s1                                  : amp
    dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                      : amp
    dIs1_inh/dt = -Is1_inh / tau_s1                                  : amp
    dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                      : amp
    Vpost       = Vm * Ra / (Rm_S + Ra)                              : volt
    Rm_S        : ohm
    '''
    I_syn_label = 'Is1 (exponential / AMPA-like)'
else:
    eqs = '''
    dVm/dt      = (-Vm/(Rm_S+Ra) + Is2_exc - Is2_inh + I_0) / Cm   : volt
    dIs1_exc/dt = -Is1_exc / tau_s1                                  : amp
    dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                      : amp
    dIs1_inh/dt = -Is1_inh / tau_s1                                  : amp
    dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                      : amp
    Vpost       = Vm * Ra / (Rm_S + Ra)                              : volt
    Rm_S        : ohm
    '''
    I_syn_label = 'Is2 (alpha function / NMDA-like)'

neurons = NeuronGroup(2, model=eqs,
                      threshold='Vm > Vthresh',
                      reset='Vm = Vreset',
                      refractory=t_ref,
                      method='euler')
neurons.Vm      = Vreset
neurons.Rm_S    = Rm_hi
neurons.Is1_exc = 0*uA;  neurons.Is2_exc = 0*uA
neurons.Is1_inh = 0*uA;  neurons.Is2_inh = 0*uA

# =============================================================================
# Synapses (Iw·δ injected into Is1 — Eq. 11)
# =============================================================================
exc_syn = Synapses(exc_input, neurons, on_pre='Is1_exc_post += Iw_exc')
exc_syn.connect(j='i')
inh_syn = Synapses(inh_input, neurons, on_pre='Is1_inh_post += Iw_inh')
inh_syn.connect(j='i')

# =============================================================================
# Monitors
# =============================================================================
sp_exc  = SpikeMonitor(exc_input)
sp_inh  = SpikeMonitor(inh_input)
sp_neur = SpikeMonitor(neurons)
st_mon  = StateMonitor(neurons,
                       ['Vm', 'Is1_exc', 'Is1_inh', 'Is2_exc', 'Is2_inh'],
                       record=True, dt=0.5*ms)

print(f"\nRunning {duration/second:.0f} s simulation [mode='{NETWORK_MODE}'] ...")
run(duration, report='text')

n0 = int(np.sum(sp_neur.i == 0))
n1 = int(np.sum(sp_neur.i == 1))
print(f"\n  Exc input spikes : {len(sp_exc.t)}")
print(f"  Inh input spikes : {len(sp_inh.t)}")
print(f"  Neuron 0 spikes  : {n0}  ({n0/float(duration/second):.1f} Hz mean)")
print(f"  Neuron 1 spikes  : {n1}  ({n1/float(duration/second):.1f} Hz mean)")

# =============================================================================
# Plotting
# =============================================================================
t_s   = st_mon.t / second
dur_s = float(duration / second)

C_exc = '#27AE60'; C_inh = '#E74C3C'
C_n0  = '#2980B9'; C_n1  = '#E67E22'
C_I0  = '#F39C12'

fig = plt.figure(figsize=(20, 20))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.68, wspace=0.40,
                        height_ratios=[1, 1, 1.2, 1.5, 1.4])

# ── Panel 0 (full): Input raster ─────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.eventplot(sp_exc.t[sp_exc.i==0]/second, lineoffsets=2,
              linelengths=0.55, colors=C_exc, linewidths=0.5, label='Exc (20-120 Hz)')
ax0.eventplot(sp_inh.t[sp_inh.i==0]/second, lineoffsets=1,
              linelengths=0.55, colors=C_inh, linewidths=0.5, label='Inh (20-80 Hz)')
ax0.set_xlim(0, dur_s); ax0.set_ylim(0.3, 2.7)
ax0.set_yticks([1, 2]); ax0.set_yticklabels(['Inh', 'Exc'], fontsize=9)
ax0.set_title('Poisson Input Spikes (channel 0; sinusoidally modulated rates)',
              fontsize=11, fontweight='bold')
ax0.set_ylabel('Channel'); ax0.legend(loc='upper right', fontsize=9)
ax0r = ax0.twinx()
t_r  = np.arange(n_steps) * float(rate_res/second)
ax0r.fill_between(t_r, exc_rates/Hz, alpha=0.12, color=C_exc)
ax0r.fill_between(t_r, inh_rates/Hz, alpha=0.12, color=C_inh)
ax0r.set_ylabel('Rate (Hz)', fontsize=8, color='gray')
ax0r.set_ylim(0, 160); ax0r.tick_params(labelsize=8, colors='gray')

# ── Panel 1 (full): Neural spike raster ──────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    mask = sp_neur.i == ni
    if mask.any():
        ax1.eventplot(sp_neur.t[mask]/second, lineoffsets=ni+1,
                      linelengths=0.55, colors=c, linewidths=0.7, label=lbl)
ax1.set_xlim(0, dur_s); ax1.set_ylim(0.3, 2.7)
ax1.set_yticks([1, 2]); ax1.set_yticklabels(['N0', 'N1'], fontsize=9)
ax1.set_title(
    f'Neural Output Spikes  '
    f'(N0: {n0} sp @ {n0/dur_s:.1f} Hz   |   N1: {n1} sp @ {n1/dur_s:.1f} Hz)\n'
    f'Mode: {I_syn_label}     I_0 = {I_0_uA:.1f} µA',
    fontsize=10, fontweight='bold')
ax1.set_ylabel('Neuron'); ax1.legend(loc='upper right', fontsize=9)

# ── Panel 2: Vm traces ───────────────────────────────────────────────────────
Vm_I0_alone = I_0_uA*1e-6 * R_tot_ohm
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    ax = fig.add_subplot(gs[2, ni])
    ax.plot(t_s, st_mon.Vm[ni]/volt, color=c, lw=0.6, label='Vm(t)')
    ax.axhline(Vthresh_V, color='dimgray', ls='--', lw=1.1,
               label=f'Vthresh = {Vthresh_V:.0f} V')
    ax.axhline(Vm_I0_alone, color=C_I0, ls=':', lw=1.1,
               label=f'Vm_ss from I_0 alone = {Vm_I0_alone:.2f} V')
    mask = sp_neur.i == ni
    if mask.any():
        ax.vlines(sp_neur.t[mask]/second, Vthresh_V, Vthresh_V + 0.3,
                  colors='k', lw=0.8, zorder=5, label='Spikes')
    ax.set_title(
        f'{lbl} — Membrane Potential\n'
        rf'$\tau_m = C_m(R_m^{{hi}}+R_a) = {tau_m_s*1e3:.2f}\ ms$   '
        rf'$I_0 = {I_0_uA:.1f}\ \mu A$',
        fontsize=9, fontweight='bold')
    ax.set_ylabel('Vm  (V)'); ax.set_xlabel('Time  (s)')
    ax.set_xlim(0, dur_s)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)

# ── Panel 3: Is1 and Is2 ─────────────────────────────────────────────────────
for ni, lbl in enumerate(['Neuron 0', 'Neuron 1']):
    ax = fig.add_subplot(gs[3, ni])
    Is1e = st_mon.Is1_exc[ni]/uA;  Is1i = st_mon.Is1_inh[ni]/uA
    Is2e = st_mon.Is2_exc[ni]/uA;  Is2i = st_mon.Is2_inh[ni]/uA
    ax.plot(t_s,  Is1e, color=C_exc, lw=0.7, ls='--', alpha=0.7, label='Is1_exc (exp)')
    ax.plot(t_s, -Is1i, color=C_inh, lw=0.7, ls='--', alpha=0.7, label='-Is1_inh (exp)')
    ax.plot(t_s,  Is2e, color=C_exc, lw=1.3, ls='-',              label='Is2_exc (alpha)')
    ax.plot(t_s, -Is2i, color=C_inh, lw=1.3, ls='-',              label='-Is2_inh (alpha)')
    if NETWORK_MODE == 'Is1':
        ax.fill_between(t_s,  Is1e, alpha=0.15, color=C_exc)
        ax.fill_between(t_s, -Is1i, alpha=0.15, color=C_inh)
        net = Is1e - Is1i
    else:
        ax.fill_between(t_s,  Is2e, alpha=0.15, color=C_exc)
        ax.fill_between(t_s, -Is2i, alpha=0.15, color=C_inh)
        net = Is2e - Is2i
    ax.plot(t_s, net, color='k', lw=0.9, ls=':', alpha=0.7,
            label=f'Net {NETWORK_MODE} → Vm')
    ax.axhline(I_min_uA, color='purple',      ls='-.', lw=0.9, alpha=0.7,
               label=f'I_min = {I_min_uA:.0f} µA → {f_min_Hz:.0f} Hz')
    ax.axhline(I_max_uA, color='saddlebrown', ls='-.', lw=0.9, alpha=0.7,
               label=f'I_max = {I_max_uA:.0f} µA → {f_max_Hz:.0f} Hz')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_title(
        f'{lbl} — Synaptic Currents Is1 & Is2\n'
        rf'Dashed=Is1 (exp, $\tau_{{s1}}={Rs_ohm*Cs_F*1e3:.0f}\ ms$)   '
        rf'Solid=Is2 (alpha, $\tau_{{s2}}={Rs_ohm*Cs_F*1e3:.0f}\ ms$)   '
        f'[Filled = drives Vm, mode={NETWORK_MODE}]',
        fontsize=9, fontweight='bold')
    ax.set_ylabel('Current (µA)'); ax.set_xlabel('Time (s)')
    ax.set_xlim(0, dur_s)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9, ncol=2)

# ── Panel 4 (full): I-F curve ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4, :])
firing_plot = f_IF > 0
block_plot  = I_sweep_uA > I_max_uA

ax4.plot(I_sweep_uA[firing_plot], f_IF[firing_plot],
         color='#8E44AD', lw=2.5, label='I-F curve (analytical)')
ax4.fill_between(I_sweep_uA[firing_plot], f_IF[firing_plot],
                 alpha=0.13, color='#8E44AD')
ax4.plot(I_sweep_uA[block_plot], np.zeros(block_plot.sum()),
         color=C_inh, lw=2.5, label='Beyond I_max (depol-block convention)')

# Calibration markers
for I_cal, f_cal, col in [(I_min_uA, f_min_Hz, 'navy'),
                           (I_max_uA, f_max_Hz, 'saddlebrown')]:
    ax4.plot(I_cal, f_cal, 'o', ms=10, color=col, zorder=7,
             label=f'f({I_cal:.0f} µA) = {f_cal:.0f} Hz  ← user spec')
    ax4.vlines(I_cal, 0, f_cal, colors=col, ls='--', lw=1.2, alpha=0.6)
    ax4.hlines(f_cal, 0, I_cal, colors=col, ls='--', lw=1.2, alpha=0.6)

# ASYMPTOTIC ceiling (NOT f at I_max — explicitly labelled)
ax4.axhline(f_asymptote, color='gray', ls=':', lw=1.2,
            label=f'1/t_ref = {f_asymptote:.0f} Hz  (asymptote at I→∞, '
                  f'NOT f at I_max)')

ax4.axvspan(0,         I_min_uA, alpha=0.05, color='gray', label='Sub-threshold')
ax4.axvspan(I_min_uA,  I_max_uA, alpha=0.07, color='gold', label='Operating range')
ax4.axvspan(I_max_uA,  I_sweep_uA[-1]+1, alpha=0.07, color='red',
            label='Beyond operating range')

ax4.text(2, f_asymptote*0.92,
         f'Solved:  Cm = {Cm_F*1e9:.1f} nF,   '
         f'τ_m = {tau_m_s*1e3:.2f} ms,   '
         f't_ref = {t_ref_s*1e3:.2f} ms\n'
         f'asymptote/f_max = {f_asymptote/f_max_Hz:.1f}×    '
         f'(rational: 1.5–3×;   weird: ≥10×)',
         fontsize=8.5, color='#333333',
         bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

ax4.set_xlabel('Synaptic current  I_syn  (µA)   [I_0 adds on top]', fontsize=10)
ax4.set_ylabel('Firing rate  f  (Hz)', fontsize=10)
ax4.set_title(
    rf'I-F Curve  —  $f(I)=\left[t_{{ref}}-\tau_m\ln\!\left(1-\frac{{V_{{thr}}}}{{(I+I_0)(R_m^{{hi}}+R_a)}}\right)\right]^{{-1}}$'
    '\n'
    rf'Pinned to  $f({I_min_uA:.0f}\,\mu A)={f_min_Hz:.0f}\,Hz$  and  '
    rf'$f({I_max_uA:.0f}\,\mu A)={f_max_Hz:.0f}\,Hz$    '
    rf'$\Rightarrow$  $C_m={Cm_F*1e9:.1f}$ nF, $t_{{ref}}={t_ref_s*1e3:.2f}$ ms',
    fontsize=9.5, fontweight='bold')
ax4.set_xlim(0, I_max_uA * 1.6)
ax4.set_ylim(0, f_asymptote * 1.15)
ax4.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
ax4.grid(axis='y', alpha=0.25)

fig.suptitle(
    'Memristor aLIF Neuron — Brian2 standalone reference\n'
    rf'$\tau_m={tau_m_s*1e3:.2f}\ ms$   '
    rf'$\tau_{{s1}}=\tau_{{s2}}={Rs_ohm*Cs_F*1e3:.0f}\ ms$   '
    rf'$I_{{min}}={I_min_uA:.0f}\ \mu A\ @\ {f_min_Hz:.0f}\ Hz$   '
    rf'$I_{{max}}={I_max_uA:.0f}\ \mu A\ @\ {f_max_Hz:.0f}\ Hz$   '
    rf'$I_0={I_0_uA:.1f}\ \mu A$   mode={NETWORK_MODE}',
    fontsize=11, fontweight='bold', y=0.99)

out_path = 'output/aLIF_model.png'
plt.savefig(out_path, dpi=120)
print(f"\nFigure saved → {out_path}")
