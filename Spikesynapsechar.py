"""
Memristor aLIF — Single Spike & Synapse Characterisation
=========================================================
Corrected parameters derived to satisfy:
   f(40 µA)  = 70 Hz   (lower operating point)
   f(100 µA) = 200 Hz  (upper operating point, near depol-block)
   Rheobase  = 37 µA   (below I_min — neuron is sub-threshold)

Two focused panels
------------------
  A. Single spike shape
       Constant I0 = 55 µA injected (midrange), high-res Vm trace.
       Shows sub-threshold RC charging, spike threshold, reset.
       Also plots the analytical solution for comparison.

  B. Synaptic current shape (one pre-spike activation)
       One pre-synaptic spike at t = 20 ms, no further input.
       Shows Is1(t) — exponential decay (Eq. 11)
            Is2(t) — alpha function    (Eq. 12)
       with analytical overlays.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *
import builtins

# ──────────────────────────────────────────────────────────────────────────────
# Model parameters  (corrected to match f(40µA)=70Hz, f(100µA)=200Hz)
# ──────────────────────────────────────────────────────────────────────────────
Cm      =  40.15*nF    # membrane capacitance
Ra      =  10*kohm     # axon-hillock resistor
Rm_hi   =  98*kohm     # memristor HIGH resistance (open  — quiescent)
Rm_lo   = 100*ohm      # memristor LOW  resistance (closed — spike)
                       #   R_total = Rm_hi + Ra = 108 kΩ

# ── Explicit time constants ───────────────────────────────────────────────────
tau_m   = Cm * (Rm_hi + Ra)   # = 40.15 nF × 108 kΩ = 4.336 ms  ← membrane τ

Rs      = 10*kohm      # synaptic leak resistor
Cs      = 1000*nF      # synaptic capacitance
tau_s1  = Rs * Cs      # = 10 ms  first  leaky-integration stage  (Eq. 11)
tau_s2  = Rs * Cs      # = 10 ms  second leaky-integration stage  (Eq. 12)

# ── Spiking ───────────────────────────────────────────────────────────────────
Vthresh =   4*volt
Vreset  =   0*volt
t_ref   =   3*ms       # absolute refractory period → f_max = 333 Hz

# ── I-F operating range ───────────────────────────────────────────────────────
I_min   =  40*uA       # lower operating point  → f = 70  Hz
I_max   = 100*uA       # upper operating point  → f = 200 Hz (depol-block onset)
I_rheo  = float(Vthresh/(Rm_hi+Ra))  # rheobase ≈ 37.0 µA

# ── Synaptic weight ───────────────────────────────────────────────────────────
Iw_exc  = 150*uA       # single-spike kick into Is1 (Eq. 11)

# ──────────────────────────────────────────────────────────────────────────────
# Analytical helpers
# ──────────────────────────────────────────────────────────────────────────────
tau_m_f  = float(tau_m  / second)
tau_s1_f = float(tau_s1 / second)
tau_s2_f = float(tau_s2 / second)
Vth_f    = float(Vthresh / volt)
t_ref_f  = float(t_ref   / second)
R_tot_f  = float((Rm_hi + Ra) / ohm)

def vm_analytical(t_arr, I0_A):
    """Sub-threshold Vm charge from 0 (after reset), constant I0."""
    Vm_ss = I0_A * R_tot_f
    return Vm_ss * (1 - np.exp(-t_arr / tau_m_f))

def t_cross_analytical(I0_A):
    """Time from reset to Vthresh crossing."""
    Vm_ss = I0_A * R_tot_f
    if Vm_ss <= Vth_f:
        return np.inf
    return -tau_m_f * np.log(1 - Vth_f / Vm_ss)

def Is1_analytical(t_arr, Iw_A, tau_s):
    """Is1: exponential decay after single spike kick."""
    out = np.zeros_like(t_arr)
    mask = t_arr >= 0
    out[mask] = Iw_A * np.exp(-t_arr[mask] / tau_s)
    return out

def Is2_analytical(t_arr, Iw_A, tau_s1, tau_s2):
    """Is2: alpha function if tau_s1==tau_s2, else double-exp."""
    out = np.zeros_like(t_arr)
    mask = t_arr >= 0
    t    = t_arr[mask]
    if abs(tau_s1 - tau_s2) < 1e-12:
        # true alpha function: Is2 = (Iw/tau_s) * t * exp(-t/tau_s)
        out[mask] = (Iw_A / tau_s1) * t * np.exp(-t / tau_s1)
    else:
        out[mask] = (Iw_A * tau_s2 / (tau_s1 - tau_s2)) * \
                    (np.exp(-t/tau_s1) - np.exp(-t/tau_s2))
    return out

print("=" * 58)
print("  Memristor aLIF — parameter summary (corrected)")
print("=" * 58)
print(f"  Cm        = {Cm/nF:.2f} nF")
print(f"  Ra        = {Ra/kohm:.0f} kΩ")
print(f"  Rm_hi     = {Rm_hi/kohm:.0f} kΩ   R_total={float((Rm_hi+Ra)/kohm):.0f} kΩ")
print(f"  tau_m     = Cm*(Rm_hi+Ra) = {tau_m/ms:.3f} ms  ← membrane τ")
print(f"  Rs        = {Rs/kohm:.0f} kΩ")
print(f"  Cs        = {Cs/nF:.0f} nF")
print(f"  tau_s1    = Rs*Cs = {tau_s1/ms:.0f} ms  ← 1st synaptic τ (Eq.11)")
print(f"  tau_s2    = Rs*Cs = {tau_s2/ms:.0f} ms  ← 2nd synaptic τ (Eq.12)")
print(f"  Vthresh   = {Vthresh/volt:.0f} V")
print(f"  t_ref     = {t_ref/ms:.0f} ms   → f_max = {1/float(t_ref/second):.0f} Hz")
print(f"  Rheobase  = {I_rheo*1e6:.2f} µA  (f→0, below I_min)")
print(f"  I_min     = {I_min/uA:.0f} µA  → f = 70 Hz")
print(f"  I_max     = {I_max/uA:.0f} µA  → f = 200 Hz (depol-block onset)")
print(f"  Iw_exc    = {Iw_exc/uA:.0f} µA")
print("=" * 58)

# Verification of I-F at operating points
for I_uA, f_tgt in [(40, 70), (100, 200)]:
    tc = t_cross_analytical(I_uA*1e-6)
    f  = 1/(tc + t_ref_f)
    print(f"  f({I_uA}µA) = {f:.1f} Hz  (target {f_tgt} Hz)")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PART A — Single spike shape
# ══════════════════════════════════════════════════════════════════════════════
# Inject constant I0 = 55 µA (midrange),  record a few ISIs at high resolution.

I_tonic = 55*uA
tc_55   = t_cross_analytical(float(I_tonic/amp))  # time from reset to spike
f_55    = 1 / (tc_55 + t_ref_f)
print(f"Tonic injection for spike shape: I0 = 55 µA")
print(f"  Vm_ss = {float(I_tonic/amp)*R_tot_f:.3f} V   "
      f"t_cross = {tc_55*1e3:.3f} ms   f = {f_55:.1f} Hz")

start_scope()
defaultclock.dt = 0.01*ms   # high resolution for spike shape

spike_eqs = '''
dVm/dt = (-Vm/(Rm_hi+Ra) + I_inj) / Cm : volt
I_inj : amp (shared)
'''
ng_spike = NeuronGroup(1, spike_eqs,
                       threshold='Vm > Vthresh',
                       reset='Vm = Vreset',
                       refractory=t_ref,
                       method='euler')
ng_spike.Vm     = Vreset
ng_spike.I_inj  = I_tonic

sm_spike = StateMonitor(ng_spike, 'Vm', record=True, dt=0.01*ms)
sp_spike = SpikeMonitor(ng_spike)

# Run for 5 ISIs to get settled, then capture
run(5/f_55 * second)
t0_capture = float(defaultclock.t/second)
run(4/f_55 * second)

print(f"  Spikes detected: {len(sp_spike.t)}")

# ── Analytical single ISI ─────────────────────────────────────────────────────
ISI  = 1/f_55
t_isi = np.linspace(0, ISI, 3000)

# Phase 1: refractory (Vm = 0 for t_ref)
mask_ref  = t_isi <= t_ref_f
# Phase 2: charging
mask_chg  = (t_isi > t_ref_f) & (t_isi <= t_ref_f + tc_55)
# Phase 3: spike (just a marker)

vm_analytic = np.zeros_like(t_isi)
vm_analytic[mask_chg] = vm_analytical(t_isi[mask_chg] - t_ref_f,
                                       float(I_tonic/amp))

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Synaptic current shape (one pre-synaptic spike)
# ══════════════════════════════════════════════════════════════════════════════
start_scope()
defaultclock.dt = 0.05*ms

SIM_DUR  = 200*ms
T_SPIKE  = 20*ms    # single pre-synaptic spike time

# Fake pre-synaptic group that fires exactly once via SpikeGeneratorGroup
pre_group = SpikeGeneratorGroup(1, [0], [T_SPIKE])

# Post-synaptic neuron: only track synaptic variables, no ongoing drive
syn_eqs = '''
dIs1/dt = -Is1 / tau_s1  : amp
dIs2/dt = (-Is2 + Is1) / tau_s2 : amp
'''
ng_syn = NeuronGroup(1, syn_eqs, method='euler')
ng_syn.Is1 = 0*uA
ng_syn.Is2 = 0*uA

syn = Synapses(pre_group, ng_syn, on_pre='Is1_post += Iw_exc')
syn.connect()

sm_syn = StateMonitor(ng_syn, ['Is1', 'Is2'], record=True, dt=0.05*ms)

run(SIM_DUR)

# Analytical synaptic currents
t_after_spike = np.linspace(0, float(SIM_DUR/second) - float(T_SPIKE/second), 4000)
Is1_ana = Is1_analytical(t_after_spike, float(Iw_exc/amp), tau_s1_f)
Is2_ana = Is2_analytical(t_after_spike, float(Iw_exc/amp), tau_s1_f, tau_s2_f)

# Peak of alpha function: at t = tau_s, value = Iw/e
t_peak_s = tau_s2_f
Is2_peak = float(Iw_exc/amp) / tau_s1_f * t_peak_s * np.exp(-t_peak_s/tau_s2_f)

print(f"\nSynaptic shape (single spike at t={float(T_SPIKE/ms):.0f} ms):")
print(f"  Is1 peak   = {float(Iw_exc/uA):.0f} µA  at t=0 (instantaneous kick)")
print(f"  Is2 peak   = {Is2_peak*1e6:.2f} µA  at t=tau_s={tau_s2_f*1e3:.0f} ms (alpha peak)")
print(f"  Is2 formula: Iw/tau_s × t × exp(−t/tau_s)   [alpha function]")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
C_vm   = '#2980B9'   # blue
C_ana  = '#E74C3C'   # red  — analytical
C_Is1  = '#27AE60'   # green
C_Is2  = '#8E44AD'   # purple
C_ref  = '#BDC3C7'   # light grey — refractory shading

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.60, wspace=0.40,
                        height_ratios=[1.6, 1.6, 1.2])

# ─────────────────────────────────────────────────────────────────────────────
# Panel A-left: Full simulated Vm trace (captured window)
# ─────────────────────────────────────────────────────────────────────────────
ax_full = fig.add_subplot(gs[0, 0])

t_full = sm_spike.t / second
vm_full = sm_spike.Vm[0] / volt

# Only plot from capture start
mask_cap = t_full >= t0_capture
t_plot   = (t_full[mask_cap] - t0_capture) * 1e3   # ms
vm_plot  = vm_full[mask_cap]

ax_full.plot(t_plot, vm_plot, color=C_vm, lw=0.8, label='Vm(t)  — simulated')
ax_full.axhline(Vth_f, color='dimgray', ls='--', lw=1.2,
                label=f'Vthresh = {Vth_f:.0f} V')
# Draw spike bars for each detected spike in window
for ts in sp_spike.t / second:
    if ts >= t0_capture:
        tx = (ts - t0_capture) * 1e3
        ax_full.vlines(tx, Vth_f, Vth_f + 0.3, colors='k', lw=1.2, zorder=5)

ax_full.set_title(f'Panel A — Vm trace (I₀ = 55 µA,  f = {f_55:.0f} Hz)\n'
                  rf'$\tau_m = C_m(R_m[hi]+R_a) = {tau_m/ms:.2f}$ ms,   '
                  rf'$t_{{ref}} = {t_ref/ms:.0f}$ ms',
                  fontsize=9, fontweight='bold')
ax_full.set_ylabel('Vm  (V)')
ax_full.set_xlabel('Time  (ms)')
ax_full.legend(fontsize=8, loc='upper right')

# ─────────────────────────────────────────────────────────────────────────────
# Panel A-right: Zoom into ONE ISI with analytical overlay
# ─────────────────────────────────────────────────────────────────────────────
ax_zoom = fig.add_subplot(gs[0, 1])

# Find first full spike in captured window
sp_in_window = [ts for ts in sp_spike.t/second if ts > t0_capture + 0.001]
if len(sp_in_window) >= 2:
    t_sp1 = sp_in_window[0]
    t_sp2 = sp_in_window[1]
    mask_isi = (t_full >= t_sp1) & (t_full <= t_sp2 + 0.001)
    t_isi_plot  = (t_full[mask_isi] - t_sp1) * 1e3
    vm_isi_plot = vm_full[mask_isi]
    ax_zoom.plot(t_isi_plot, vm_isi_plot, color=C_vm, lw=1.4,
                 label='Vm(t) — simulated', zorder=3)

# Analytical overlay on the same ISI window
t_isi_ms = t_isi * 1e3  # convert to ms
ax_zoom.plot(t_isi_ms, vm_analytic, color=C_ana, lw=1.4, ls='--',
             alpha=0.85, label='Analytical: $V_{ss}(1-e^{-t/\\tau_m})$', zorder=4)

# Annotate phases
isi_ms = ISI * 1e3
ref_ms = t_ref_f * 1e3
tc_ms  = tc_55 * 1e3
Vm_ss_55 = float(I_tonic/amp) * R_tot_f

ax_zoom.axvspan(0, ref_ms, alpha=0.12, color=C_ref, label=f'Refractory ({ref_ms:.1f} ms)')
ax_zoom.axhline(Vth_f, color='dimgray', ls='--', lw=1.2, label=f'Vthresh = {Vth_f:.0f} V')
ax_zoom.axhline(Vm_ss_55, color='orange', ls=':', lw=1.2,
                label=f'Vm_ss = I₀·R_total = {Vm_ss_55:.2f} V')

# Spike marker at threshold crossing
ax_zoom.vlines(ref_ms + tc_ms, Vth_f, Vth_f + 0.32,
               colors='k', lw=1.8, label='Spike', zorder=5)

# Dimension annotations
ax_zoom.annotate('', xy=(ref_ms + tc_ms, 0.1), xytext=(ref_ms, 0.1),
                 arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1.2))
ax_zoom.text(ref_ms + tc_ms/2, 0.2,
             rf'$t_{{cross}}={tc_ms:.1f}$ ms', ha='center', fontsize=8, color='steelblue')

ax_zoom.annotate('', xy=(ref_ms, -0.15), xytext=(0, -0.15),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax_zoom.text(ref_ms/2, -0.27, rf'$t_{{ref}}={ref_ms:.0f}$ ms',
             ha='center', fontsize=8, color='gray')

ax_zoom.set_xlim(-0.3, isi_ms + 0.5)
ax_zoom.set_ylim(-0.4, builtins.max(float(Vm_ss_55), float(Vth_f)) * 1.12)
ax_zoom.set_title('Panel A (zoom) — One ISI: Subthreshold charging\n'
                  rf'$V_m(t)=I_0(R_m[hi]+R_a)\cdot(1-e^{{-t/\tau_m}})$   '
                  rf'ISI = $t_{{cross}}+t_{{ref}}$ = {isi_ms:.1f} ms  →  f = {f_55:.0f} Hz',
                  fontsize=9, fontweight='bold')
ax_zoom.set_ylabel('Vm  (V)')
ax_zoom.set_xlabel('Time within ISI  (ms)')
ax_zoom.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B-left: Is1 — exponential decay
# ─────────────────────────────────────────────────────────────────────────────
ax_Is1 = fig.add_subplot(gs[1, 0])

t_syn_ms  = sm_syn.t / ms
Is1_sim   = sm_syn.Is1[0] / uA
t_ana_ms  = (t_after_spike + float(T_SPIKE/second)) * 1e3   # absolute time

ax_Is1.fill_between(t_syn_ms, Is1_sim, alpha=0.18, color=C_Is1)
ax_Is1.plot(t_syn_ms, Is1_sim, color=C_Is1, lw=1.4, label='Is1(t) — simulated')
ax_Is1.plot(t_ana_ms, Is1_ana*1e6, color='k', lw=1.2, ls='--', alpha=0.75,
            label=r'$I_w \cdot e^{-t/\tau_{s1}}$  — analytical')
ax_Is1.axvline(float(T_SPIKE/ms), color='red', ls=':', lw=1.2,
               label='Pre-synaptic spike')

# Mark 1/e decay
t_e   = float(T_SPIKE/ms) + tau_s1_f * 1e3
Is1_e = float(Iw_exc/uA) * np.exp(-1)
ax_Is1.annotate(f'1/e = {Is1_e:.0f} µA at t=τ_s1',
                xy=(t_e, Is1_e), xytext=(t_e+10, Is1_e+15),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='k'))

ax_Is1.set_title('Panel B — Is1(t): 1st leaky-integration stage  (Eq. 11)\n'
                 rf'$\tau_s \dot{{I}}_{{s1}} = -I_{{s1}} + I_w\delta(t_{{sp}})$  '
                 rf'→  $I_{{s1}}(t)=I_w\,e^{{-t/\tau_{{s1}}}}$,   '
                 rf'$\tau_{{s1}}=R_sC_s={tau_s1/ms:.0f}$ ms',
                 fontsize=9, fontweight='bold')
ax_Is1.set_ylabel('Is1  (µA)')
ax_Is1.set_xlabel('Time  (ms)')
ax_Is1.set_xlim(0, float(SIM_DUR/ms))
ax_Is1.legend(fontsize=8, loc='upper right', framealpha=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B-right: Is2 — alpha function
# ─────────────────────────────────────────────────────────────────────────────
ax_Is2 = fig.add_subplot(gs[1, 1])

Is2_sim = sm_syn.Is2[0] / uA

ax_Is2.fill_between(t_syn_ms, Is2_sim, alpha=0.18, color=C_Is2)
ax_Is2.plot(t_syn_ms, Is2_sim, color=C_Is2, lw=1.4, label='Is2(t) — simulated')
ax_Is2.plot(t_ana_ms, Is2_ana*1e6, color='k', lw=1.2, ls='--', alpha=0.75,
            label=r'$\frac{I_w}{\tau_s}\,t\,e^{-t/\tau_s}$  — alpha function')
ax_Is2.axvline(float(T_SPIKE/ms), color='red', ls=':', lw=1.2,
               label='Pre-synaptic spike')

# Mark alpha peak (at t = tau_s after spike)
t_peak_ms  = float(T_SPIKE/ms) + tau_s2_f*1e3
Is2_pk_uA  = Is2_peak * 1e6
ax_Is2.axvline(t_peak_ms, color='gray', ls='--', lw=0.9, alpha=0.7)
ax_Is2.annotate(f'Peak: {Is2_pk_uA:.1f} µA\n@ t = τ_s2 = {tau_s2_f*1e3:.0f} ms',
                xy=(t_peak_ms, Is2_pk_uA),
                xytext=(t_peak_ms + 15, Is2_pk_uA - 8),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='k'))

ax_Is2.set_title('Panel B — Is2(t): 2nd leaky-integration stage  (Eq. 12)\n'
                 rf'$\tau_s\dot{{I}}_{{s2}}=-I_{{s2}}+I_{{s1}}$  '
                 rf'→  $I_{{s2}}(t)=\frac{{I_w}}{{\tau_s}}\,t\,e^{{-t/\tau_s}}$  '
                 rf'[alpha function],   $\tau_{{s2}}={tau_s2/ms:.0f}$ ms',
                 fontsize=9, fontweight='bold')
ax_Is2.set_ylabel('Is2  (µA)')
ax_Is2.set_xlabel('Time  (ms)')
ax_Is2.set_xlim(0, float(SIM_DUR/ms))
ax_Is2.legend(fontsize=8, loc='upper right', framealpha=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# Panel C (full-width): I-F curve (corrected) with depolarisation block
# ─────────────────────────────────────────────────────────────────────────────
ax_IF = fig.add_subplot(gs[2, :])

I_sweep_uA = np.linspace(0, 160, 4000)
I_sweep_A  = I_sweep_uA * 1e-6
Vm_ss_arr  = I_sweep_A * R_tot_f
f_IF       = np.zeros_like(I_sweep_A)

# Normal firing: rheobase < I < I_max
band  = (Vm_ss_arr > Vth_f) & (I_sweep_uA <= float(I_max/uA))
tc_arr = -tau_m_f * np.log(1 - Vth_f / Vm_ss_arr[band])
f_IF[band] = 1.0 / (tc_arr + t_ref_f)
# Above I_max: depolarisation block → f = 0

f_max_hz = 1.0 / t_ref_f
ax_IF.plot(I_sweep_uA[band], f_IF[band], color='#8E44AD', lw=2.5,
           label='I-F curve  (corrected)')
ax_IF.fill_between(I_sweep_uA[band], f_IF[band], alpha=0.13, color='#8E44AD')
ax_IF.plot(I_sweep_uA[I_sweep_uA > float(I_max/uA)],
           np.zeros(int(np.sum(I_sweep_uA > float(I_max/uA)))),
           color='#E74C3C', lw=2.5, label='Depolarisation block  (f=0)')

# Calibration points
for I_cal, f_cal, col in [(40, 70, 'navy'), (100, 200, 'darkred')]:
    ax_IF.plot(I_cal, f_cal, 'o', ms=9, color=col, zorder=6,
               label=f'f({I_cal} µA) = {f_cal} Hz  ✓')
    ax_IF.vlines(I_cal, 0, f_cal, colors=col, ls='--', lw=1.2, alpha=0.7)
    ax_IF.hlines(f_cal, 0, I_cal, colors=col, ls='--', lw=1.2, alpha=0.7)

ax_IF.axvline(I_rheo*1e6, color='gray', ls=':', lw=1.2,
              label=f'Rheobase = {I_rheo*1e6:.1f} µA  (f→0)')
ax_IF.axhline(f_max_hz, color='gray', ls='-.', lw=1.0,
              label=f'f_max = 1/t_ref = {f_max_hz:.0f} Hz')
ax_IF.axvspan(float(I_min/uA), float(I_max/uA), alpha=0.07, color='gold',
              label='Operating range  (I_min – I_max)')
ax_IF.text(float(I_max/uA)+3, 30,
           'Capacitor saturated\nRm[hi] stuck open\nf = 0',
           fontsize=8, color='#C0392B',
           bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#E74C3C', alpha=0.9))

ax_IF.set_xlabel('Tonic current  I₀  (µA)', fontsize=10)
ax_IF.set_ylabel('Firing rate  f  (Hz)', fontsize=10)
ax_IF.set_title(
    'Panel C — Corrected I-F Curve\n'
    rf'$f(I_0)=\left[t_{{ref}}-\tau_m\ln(1-V_{{thr}}/I_0R_{{total}})\right]^{{-1}}$   '
    rf'$\tau_m={tau_m/ms:.2f}$ ms   $t_{{ref}}={t_ref/ms:.0f}$ ms   '
    rf'$R_{{total}}={float((Rm_hi+Ra)/kohm):.0f}$ kΩ',
    fontsize=9, fontweight='bold')
ax_IF.set_xlim(0, 160)
ax_IF.set_ylim(0, f_max_hz * 1.18)
ax_IF.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
ax_IF.grid(axis='y', alpha=0.25)

# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Memristor aLIF — Single Spike Shape & Synaptic Current Characterisation\n'
    rf'$C_m={Cm/nF:.2f}$ nF   $R_m[hi]={Rm_hi/kohm:.0f}$ kΩ   $R_a={Ra/kohm:.0f}$ kΩ   '
    rf'$\tau_m={tau_m/ms:.2f}$ ms   '
    rf'$\tau_{{s1}}=\tau_{{s2}}=R_sC_s={tau_s1/ms:.0f}$ ms   '
    rf'$t_{{ref}}={t_ref/ms:.0f}$ ms',
    fontsize=11, fontweight='bold', y=1.005)

out_path = 'spike_synapse_characterisation.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")