"""
Brian2 Demo – Memristor aLIF Neuron (Eqs. 9–12)  v3
=====================================================
Hardware-scale parameters (uA currents, kΩ resistances).

Changes vs v2
-------------
  • Rs, Cs, tau_s1, tau_s2 — physically explicit synaptic RC parameters
    (tau_s1 = Rs*Cs for the first stage, tau_s2 = Rs*Cs for the second;
     kept equal here but now independently adjustable to model
     AMPA vs NMDA by changing tau_s2)
  • I-F curve now has a hard upper cutoff at I_max:
      - Below I_min  → sub-threshold, no firing  (Vm_ss < Vthresh)
      - I_min–I_max  → normal spiking range
      - Above I_max  → depolarisation block: capacitor always full,
        memristor stuck in Rm_hi (open), Vpost ≈ 0 → no output spike
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *

# ── Reproducibility & timestep ────────────────────────────────────────────────
seed(42)
defaultclock.dt = 0.05*ms

# ── Simulation duration ───────────────────────────────────────────────────────
duration = 10*second

# =============================================================================
# Model parameters
# =============================================================================

# ── Membrane ──────────────────────────────────────────────────────────────────
Cm      = 100*nF       # membrane capacitance
Ra      =  2.2*kohm     # axon-hillock resistor
Rm_hi   =  100*kohm     # memristor HIGH resistance (open  — quiescent)
Rm_lo   = 100*ohm      # memristor LOW  resistance (closed — spike)
                       #   Rm_hi >> Ra → Vpost ≈ 0  (open switch)
                       #   Rm_lo << Ra → Vpost ≈ Vm (action potential)

# ── Explicit membrane time constant ──────────────────────────────────────────
#    Governs how fast Vm charges toward its steady state.
#    In the quiescent state (Rm_S = Rm_hi):
tau_m   = Cm * (Rm_hi + Ra)      # = 100 nF × 100 kΩ = 10 ms

# ── Spiking ──────────────────────────────────────────────────────────────────
Vthresh =   4*volt     # spike threshold
Vreset  =   0*volt     # post-spike reset
t_ref   =   2*ms       # absolute refractory period

# ── Synapse  (explicit RC parameters, Eqs. 11–12) ────────────────────────────
Rs      =  10*kohm     # synaptic leak resistor
Cs      =  1000*nF     # synaptic capacitance
tau_s1  = Rs * Cs      # 1st leaky-integration stage τ = Rs·Cs = 10 ms
tau_s2  = Rs * Cs      # 2nd leaky-integration stage τ = Rs·Cs = 10 ms
                       # ↑ set tau_s2 > tau_s1 to switch from
                       #   exponential (AMPA) to alpha-function (NMDA) kinetics

Iw_exc  = 150*uA       # excitatory synaptic weight  (Iw in Eq. 11)
Iw_inh  = 100*uA       # inhibitory synaptic weight

# ── I-F operating range ───────────────────────────────────────────────────────
# I_min — rheobase: Vm_ss = I_min·(Rm_hi+Ra) = Vthresh
#          → I_min = Vthresh/(Rm_hi+Ra) = 4V/100kΩ = 40 µA
# I_max — depolarisation-block threshold: above this the capacitor charges
#          faster than the memristor can commutate, Cm stays saturated,
#          Rm_S stays in Rm_hi (open switch) → Vpost ≈ 0 → no spike output.
I_min   =  40*uA
I_max   = 100*uA

# =============================================================================
# Parameter summary
# =============================================================================
print("=" * 58)
print("  Memristor aLIF — parameter summary")
print("=" * 58)
print(f"  Cm        = {Cm/nF:.0f} nF")
print(f"  Ra        = {Ra/kohm:.0f} kΩ")
print(f"  Rm_hi     = {Rm_hi/kohm:.0f} kΩ   (open  switch — quiescent)")
print(f"  Rm_lo     = {Rm_lo/ohm:.0f} Ω    (closed switch — spike)")
print(f"  tau_m     = Cm*(Rm_hi+Ra) = {tau_m/ms:.1f} ms  ← membrane τ")
print(f"  Rs        = {Rs/kohm:.0f} kΩ")
print(f"  Cs        = {Cs/nF:.0f} nF")
print(f"  tau_s1    = Rs*Cs = {tau_s1/ms:.1f} ms  ← 1st synaptic stage")
print(f"  tau_s2    = Rs*Cs = {tau_s2/ms:.1f} ms  ← 2nd synaptic stage")
print(f"  Vthresh   = {Vthresh/volt:.0f} V")
print(f"  t_ref     = {t_ref/ms:.0f} ms   →  f_max = {1/float(t_ref/second):.0f} Hz")
print(f"  I_min     = {I_min/uA:.0f} µA  (rheobase)")
print(f"  I_max     = {I_max/uA:.0f} µA  (depolarisation-block onset)")
print(f"  Iw_exc    = {Iw_exc/uA:.0f} µA")
print(f"  Iw_inh    = {Iw_inh/uA:.0f} µA")
print("=" * 58)

# =============================================================================
# Analytical I-F curve  with depolarisation-block cutoff
# =============================================================================
# Standard LIF inter-spike interval (after reset Vm=0):
#   Vm(t)   = Vm_ss · (1 − exp(−t/tau_m)),   Vm_ss = I0·(Rm_hi+Ra)
#   t_cross = −tau_m · ln(1 − Vthresh/Vm_ss)
#   f(I0)   = 1 / (t_cross + t_ref)          for I_min < I0 < I_max
#
# Above I_max: depolarisation block → f = 0
#   The capacitor charges to Vm_ss >> Vthresh before the memristor can
#   commutate back to Rm_lo.  The device is stuck in Rm_hi (open), so
#   Vpost = Vm·Ra/(Rm_hi+Ra) ≈ 0 — no action potential is produced.

I_sweep_uA  = np.linspace(0, float(1.6 * I_max/uA), 3000)
I_sweep_A   = I_sweep_uA * 1e-6

Vm_ss_vals  = I_sweep_A * float((Rm_hi + Ra)/ohm)
Vth_f       = float(Vthresh / volt)
tau_m_f     = float(tau_m   / second)
t_ref_f     = float(t_ref   / second)
I_min_uA    = float(I_min / uA)
I_max_uA    = float(I_max / uA)

f_IF = np.zeros_like(I_sweep_A)

# Normal firing band: I_min < I < I_max
band = (Vm_ss_vals > Vth_f) & (I_sweep_uA <= I_max_uA)
t_cross       = -tau_m_f * np.log(1.0 - Vth_f / Vm_ss_vals[band])
f_IF[band]    = 1.0 / (t_cross + t_ref_f)

# Above I_max: depolarisation block — f stays 0 (already initialised to 0)

# =============================================================================
# Time-varying Poisson inputs  (20–120 Hz exc,  20–80 Hz inh)
# =============================================================================
rate_res     = 100*ms
n_steps      = int(duration / rate_res)
t_vals       = np.arange(n_steps) * float(rate_res / second)

exc_rates    = (70 + 50 * np.sin(2*np.pi * 0.15 * t_vals)) * Hz
inh_rates    = (50 + 30 * np.sin(2*np.pi * 0.15 * t_vals + np.pi)) * Hz

exc_rate_arr = TimedArray(exc_rates, dt=rate_res)
inh_rate_arr = TimedArray(inh_rates, dt=rate_res)

exc_input    = PoissonGroup(2, rates='exc_rate_arr(t)')
inh_input    = PoissonGroup(2, rates='inh_rate_arr(t)')

# =============================================================================
# Neuron equations  (Eqs. 9–12, explicit tau_s1 / tau_s2)
# =============================================================================
eqs = '''
dVm/dt      = (-Vm / (Rm_S + Ra) + Is2_exc - Is2_inh) / Cm : volt
dIs1_exc/dt = -Is1_exc / tau_s1                              : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                  : amp
dIs1_inh/dt = -Is1_inh / tau_s1                              : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                  : amp
Vpost       = Vm * Ra / (Rm_S + Ra)                          : volt
Rm_S        : ohm
'''

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
# Synapses  — Iw·δ(t_spike) injects into Is1 (Eq. 11)
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
st_mon  = StateMonitor(neurons, ['Vm', 'Is2_exc', 'Is2_inh'],
                       record=True, dt=0.5*ms)

# =============================================================================
# Run
# =============================================================================
print("\nRunning 10 s simulation …")
run(duration, report='text')

n0 = int(np.sum(sp_neur.i == 0))
n1 = int(np.sum(sp_neur.i == 1))
print(f"\n  Exc input spikes : {len(sp_exc.t)}")
print(f"  Inh input spikes : {len(sp_inh.t)}")
print(f"  Neuron 0 spikes  : {n0}  ({n0/10:.1f} Hz mean)")
print(f"  Neuron 1 spikes  : {n1}  ({n1/10:.1f} Hz mean)")

# =============================================================================
# Plotting
# =============================================================================
t_s   = st_mon.t / second
dur_s = float(duration / second)

C_exc = '#27AE60'
C_inh = '#E74C3C'
C_n0  = '#2980B9'
C_n1  = '#E67E22'

fig = plt.figure(figsize=(20, 20))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.68, wspace=0.38,
                        height_ratios=[1, 1, 1.2, 1.2, 1.5])

# ── Panel 0: Input raster ─────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
exc_mask0 = sp_exc.i == 0
inh_mask0 = sp_inh.i == 0
ax0.eventplot(sp_exc.t[exc_mask0]/second, lineoffsets=2, linelengths=0.55,
              colors=C_exc, linewidths=0.5, label='Exc (20–120 Hz)')
ax0.eventplot(sp_inh.t[inh_mask0]/second, lineoffsets=1, linelengths=0.55,
              colors=C_inh, linewidths=0.5, label='Inh (20–80 Hz)')
ax0.set_xlim(0, dur_s); ax0.set_ylim(0.3, 2.7)
ax0.set_yticks([1, 2]); ax0.set_yticklabels(['Inh', 'Exc'], fontsize=9)
ax0.set_title('Poisson Input Spikes  (channel 0 shown; both neurons share same rate profile)',
              fontsize=11, fontweight='bold')
ax0.set_ylabel('Channel')
ax0.legend(loc='upper right', fontsize=9, framealpha=0.85)
ax0r = ax0.twinx()
t_r = np.arange(n_steps) * float(rate_res/second)
ax0r.fill_between(t_r, exc_rates/Hz, alpha=0.12, color=C_exc)
ax0r.fill_between(t_r, inh_rates/Hz, alpha=0.12, color=C_inh)
ax0r.set_ylabel('Rate (Hz)', fontsize=8, color='gray')
ax0r.set_ylim(0, 160); ax0r.tick_params(labelsize=8, colors='gray')

# ── Panel 1: Neural spike raster ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    mask = sp_neur.i == ni
    if mask.any():
        ax1.eventplot(sp_neur.t[mask]/second, lineoffsets=ni+1,
                      linelengths=0.55, colors=c, linewidths=0.7, label=lbl)
ax1.set_xlim(0, dur_s); ax1.set_ylim(0.3, 2.7)
ax1.set_yticks([1, 2]); ax1.set_yticklabels(['N0', 'N1'], fontsize=9)
ax1.set_title(f'Neural Output Spikes  '
              f'(N0: {n0} sp @ {n0/10:.1f} Hz   |   N1: {n1} sp @ {n1/10:.1f} Hz)',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('Neuron')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.85)

# ── Panels 2: Vm per neuron ───────────────────────────────────────────────────
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    ax = fig.add_subplot(gs[2, ni])
    ax.plot(t_s, st_mon.Vm[ni]/volt, color=c, lw=0.6, label='Vm(t)')
    ax.axhline(float(Vthresh/volt), color='dimgray', ls='--', lw=1.0,
               label=f'Vthresh = {Vthresh/volt:.0f} V')
    mask = sp_neur.i == ni
    if mask.any():
        ax.vlines(sp_neur.t[mask]/second,
                  float(Vthresh/volt), float(Vthresh/volt) + 0.3,
                  colors='k', lw=0.8, label='Spikes', zorder=5)
    ax.set_title(
        f'{lbl} — Membrane Potential  Vm(t)\n'
        rf'$\tau_m = C_m(R_m[hi]+R_a)={tau_m/ms:.0f}$ ms   '
        rf'$\tau_{{s1}}=\tau_{{s2}}=R_sC_s={tau_s1/ms:.0f}$ ms',
        fontsize=9, fontweight='bold')
    ax.set_ylabel('Vm  (V)')
    ax.set_xlim(0, dur_s)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)

# ── Panels 3: Synaptic currents Is2 ──────────────────────────────────────────
for ni, lbl in enumerate(['Neuron 0', 'Neuron 1']):
    ax = fig.add_subplot(gs[3, ni])
    exc_cur = st_mon.Is2_exc[ni] / uA
    inh_cur = st_mon.Is2_inh[ni] / uA
    net_cur = exc_cur - inh_cur
    ax.fill_between(t_s,  exc_cur, alpha=0.2, color=C_exc)
    ax.fill_between(t_s, -inh_cur, alpha=0.2, color=C_inh)
    ax.plot(t_s,  exc_cur, color=C_exc, lw=0.7, label='Is2_exc  (+)')
    ax.plot(t_s, -inh_cur, color=C_inh, lw=0.7, label='−Is2_inh  (−)')
    ax.plot(t_s,  net_cur, color='k',   lw=0.8, ls=':', alpha=0.8, label='Net Is2')
    ax.axhline(I_min_uA,  color='purple', ls='-.', lw=1.1, alpha=0.8,
               label=f'I_min = {I_min_uA:.0f} µA')
    ax.axhline(I_max_uA,  color='saddlebrown', ls='-.', lw=1.1, alpha=0.8,
               label=f'I_max = {I_max_uA:.0f} µA')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_title(f'{lbl} — Synaptic Currents  Is2(t)\n'
                 rf'$\tau_{{s1}}=R_sC_s={tau_s1/ms:.0f}$ ms   '
                 rf'$\tau_{{s2}}=R_sC_s={tau_s2/ms:.0f}$ ms   '
                 rf'$I_w^{{exc}}={Iw_exc/uA:.0f}$ µA   $I_w^{{inh}}={Iw_inh/uA:.0f}$ µA',
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('Current  (µA)')
    ax.set_xlabel('Time  (s)')
    ax.set_xlim(0, dur_s)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85, ncol=2)

# ── Panel 4 (full-width): I-F curve with depolarisation block ────────────────
ax4 = fig.add_subplot(gs[4, :])

# Normal firing region
band_plot  = (I_sweep_uA >= I_min_uA) & (I_sweep_uA <= I_max_uA)
above_plot = I_sweep_uA >  I_max_uA

ax4.plot(I_sweep_uA[band_plot],  f_IF[band_plot],
         color='#8E44AD', lw=2.5, label='Normal firing  (I-F curve)')
ax4.fill_between(I_sweep_uA[band_plot], f_IF[band_plot],
                 alpha=0.15, color='#8E44AD')

# Depolarisation-block region (f = 0 above I_max)
ax4.plot(I_sweep_uA[above_plot], f_IF[above_plot],
         color='#E74C3C', lw=2.5, ls='-',
         label='Depolarisation block  (f = 0)')
ax4.fill_between(I_sweep_uA[above_plot], 0,
                 max(f_IF) if max(f_IF) > 0 else 130,
                 alpha=0.07, color='#E74C3C')

# Vertical markers
f_max_hz = 1.0 / t_ref_f
ax4.axvline(I_min_uA, color='navy',       ls='--', lw=1.5,
            label=f'I_min = {I_min_uA:.0f} µA  (rheobase,  '
                  f'Vm_ss=Vthresh)')
ax4.axvline(I_max_uA, color='saddlebrown', ls='--', lw=1.5,
            label=f'I_max = {I_max_uA:.0f} µA  (depol-block onset)')

# Shade sub-threshold, operating, and block zones
ax4.axvspan(0,          I_min_uA, alpha=0.05, color='gray',   label='Sub-threshold')
ax4.axvspan(I_min_uA,   I_max_uA, alpha=0.08, color='gold',   label='Operating range')
ax4.axvspan(I_max_uA, I_sweep_uA[-1], alpha=0.07, color='red', label='Block region')

# f_max ceiling
ax4.axhline(f_max_hz, color='gray', ls=':', lw=1.2,
            label=f'f_max = 1/t_ref = {f_max_hz:.0f} Hz')

# Annotate peak
f_peak     = float(np.max(f_IF))
I_at_peak  = float(I_sweep_uA[np.argmax(f_IF)])
ax4.annotate(f'Peak: {f_peak:.0f} Hz @ {I_at_peak:.0f} µA',
             xy=(I_at_peak, f_peak),
             xytext=(I_at_peak - 20, f_peak - 15),
             fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.2),
             color='#8E44AD', fontweight='bold')

# Annotate block region
ax4.text(I_max_uA + 5, f_max_hz * 0.4,
         'Capacitor saturated\nRm[hi] always open\nVpost ≈ 0  →  no spike',
         fontsize=8, color='#C0392B',
         bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec='#E74C3C', alpha=0.9))

ax4.set_xlabel('Injected current  I₀  (µA)', fontsize=10)
ax4.set_ylabel('Firing rate  f  (Hz)', fontsize=10)
ax4.set_title(
    'Current–Frequency (I-F) Curve  with Depolarisation Block\n'
    r'$f(I_0)=\left[t_{ref}-\tau_m\ln\!\left(1-\frac{V_{thr}}'
    r'{I_0(R_m[hi]+R_a)}\right)\right]^{-1}$  for  $I_{min}<I_0\leq I_{max}$;'
    r'   $f=0$  for  $I_0>I_{max}$',
    fontsize=10, fontweight='bold')
ax4.set_xlim(0, float(1.6 * I_max_uA))
ax4.set_ylim(0, f_max_hz * 1.2)
ax4.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
ax4.grid(axis='y', alpha=0.25)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    'Memristor aLIF Neuron — Brian2 Simulation  (Eqs. 9–12)  v3\n'
    rf'$\tau_m={tau_m/ms:.0f}$ ms   '
    rf'$\tau_{{s1}}=\tau_{{s2}}=R_sC_s={tau_s1/ms:.0f}$ ms   '
    rf'$I_{{min}}={I_min_uA:.0f}$ µA   $I_{{max}}={I_max_uA:.0f}$ µA   '
    rf'$I_w^{{exc}}={Iw_exc/uA:.0f}$ µA   $I_w^{{inh}}={Iw_inh/uA:.0f}$ µA',
    fontsize=11, fontweight='bold', y=1.005)

out_path = 'neuron_sim_v3.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
