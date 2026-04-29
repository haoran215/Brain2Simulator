"""
Brian2 Demo – Memristor aLIF Neuron (Eqs. 9–12)
================================================
Hardware-scale parameters (uA currents, kΩ resistances).

Key additions vs v1
-------------------
  • tau_m  = Cm*(Rm_hi+Ra) — explicit model parameter (not just a comment)
  • I_min  = 40 uA         — rheobase  (Vm_ss = I_min*(Rm_hi+Ra) = Vthresh)
  • I_max  = 100 uA        — saturation limit for I-F curve
  • I-F curve plotted analytically from the integrate-and-fire closed form
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
# Model parameters  (all SI units via Brian2)
# =============================================================================

# Membrane
Cm      = 100*nF       # membrane capacitance
Ra      =  10*kohm     # axon-hillock resistor
Rm_hi   =  90*kohm     # memristor HIGH resistance (open switch  — quiescent)
Rm_lo   = 100*ohm      # memristor LOW  resistance (closed switch — spike)
                       # Rm_hi >> Ra  →  Vpost ≈ 0 (quiescent)
                       # Rm_lo << Ra  →  Vpost ≈ Vm (action potential)

# ── Explicit membrane time constant ──────────────────────────────────────────
#    In quiescent state (Rm_S = Rm_hi):
tau_m   = Cm * (Rm_hi + Ra)   # = 100 nF × 100 kΩ = 10 ms

# Spiking
Vthresh =   4*volt     # threshold voltage
Vreset  =   0*volt     # post-spike reset
t_ref   =   8*ms       # absolute refractory period

# Synapse
tau_s   =  10*ms       # double leaky-integration time constant
Iw_exc  = 150*uA       # excitatory synaptic weight  (Eq. 11 impulse)
Iw_inh  = 100*uA       # inhibitory synaptic weight

# ── I-F curve operating range ─────────────────────────────────────────────────
# I_min is the rheobase: smallest tonic I0 that causes sustained firing.
#   Vm_ss = I_min × (Rm_hi + Ra)  →  I_min = Vthresh / (Rm_hi + Ra)
#         = 4 V / 100 kΩ = 40 µA                      ✓
# I_max is the upper bound tested; rate is limited by t_ref above this.
#   f_max = 1/t_ref = 1/8ms = 125 Hz
I_min   =  40*uA
I_max   = 100*uA

# =============================================================================
# Print parameter summary
# =============================================================================
print("=" * 55)
print("  Neuron parameter summary")
print("=" * 55)
print(f"  Cm        = {Cm/nF:.0f} nF")
print(f"  Ra        = {Ra/kohm:.0f} kΩ")
print(f"  Rm_hi     = {Rm_hi/kohm:.0f} kΩ")
print(f"  Rm_lo     = {Rm_lo/ohm:.0f} Ω")
print(f"  tau_m     = Cm*(Rm_hi+Ra) = {tau_m/ms:.1f} ms   ← explicit")
print(f"  tau_s     = {tau_s/ms:.0f} ms")
print(f"  Vthresh   = {Vthresh/volt:.0f} V")
print(f"  t_ref     = {t_ref/ms:.0f} ms")
print(f"  I_min     = {I_min/uA:.0f} µA  (rheobase)")
print(f"  I_max     = {I_max/uA:.0f} µA  (saturation)")
print(f"  f_max     = 1/t_ref = {1/float(t_ref/second):.0f} Hz")
print(f"  Iw_exc    = {Iw_exc/uA:.0f} µA")
print(f"  Iw_inh    = {Iw_inh/uA:.0f} µA")
print("=" * 55)

# =============================================================================
# Analytical I-F curve
# =============================================================================
# After each spike reset (Vm=0):
#   Vm(t) = Vm_ss * (1 - exp(-t/tau_m)),   Vm_ss = I0*(Rm_hi+Ra)
#
# Time to reach Vthresh:
#   t_cross = -tau_m * ln(1 - Vthresh/Vm_ss)   [only when Vm_ss > Vthresh]
#
# Steady-state firing rate:
#   f(I0) = 1 / (t_cross + t_ref)

I_sweep_uA  = np.linspace(0, float(1.6*I_max/uA), 2000)   # µA
I_sweep_A   = I_sweep_uA * 1e-6                             # Amperes

Vm_ss_vals  = I_sweep_A * float((Rm_hi + Ra)/ohm)           # Volts
Vth_float   = float(Vthresh/volt)
tau_m_float = float(tau_m/second)
t_ref_float = float(t_ref/second)

f_IF        = np.zeros_like(I_sweep_A)
above       = Vm_ss_vals > Vth_float
t_cross     = -tau_m_float * np.log(1.0 - Vth_float / Vm_ss_vals[above])
f_IF[above] = 1.0 / (t_cross + t_ref_float)

# =============================================================================
# Poisson input groups  (time-varying 20–120 Hz excitatory, 20–80 Hz inhibitory)
# =============================================================================
rate_res = 100*ms
n_steps  = int(duration / rate_res)
t_vals   = np.arange(n_steps) * float(rate_res/second)

exc_rates    = (70 + 50*np.sin(2*np.pi*0.15*t_vals)) * Hz   # 20–120 Hz
inh_rates    = (50 + 30*np.sin(2*np.pi*0.15*t_vals + np.pi)) * Hz  # 20–80 Hz

exc_rate_arr = TimedArray(exc_rates, dt=rate_res)
inh_rate_arr = TimedArray(inh_rates, dt=rate_res)

exc_input    = PoissonGroup(2, rates='exc_rate_arr(t)')
inh_input    = PoissonGroup(2, rates='inh_rate_arr(t)')

# =============================================================================
# Neuron model  (Eqs. 9–12)
# =============================================================================
# Eq (9)  Cm·dVm/dt   = -Vm/(Rm_S+Ra) + Is2_exc - Is2_inh
# Eq (10) Vpost       = Vm·Ra/(Rm_S+Ra)
# Eq (11) τs·dIs1/dt  = -Is1  [+ Iw·δ(t_sp) via on_pre]
# Eq (12) τs·dIs2/dt  = -Is2 + Is1
#
# Is1/Is2 split into exc and inh so we can track them separately.

eqs = '''
dVm/dt      = (-Vm / (Rm_S + Ra) + Is2_exc - Is2_inh) / Cm : volt
dIs1_exc/dt = -Is1_exc / tau_s                               : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s                   : amp
dIs1_inh/dt = -Is1_inh / tau_s                               : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s                   : amp
Vpost       = Vm * Ra / (Rm_S + Ra)                          : volt
Rm_S        : ohm
'''

neurons = NeuronGroup(2, model=eqs,
                      threshold='Vm > Vthresh',
                      reset='Vm = Vreset',
                      refractory=t_ref,
                      method='euler')

neurons.Vm      = Vreset
neurons.Rm_S    = Rm_hi     # start in open-switch (quiescent) state
neurons.Is1_exc = 0*uA
neurons.Is2_exc = 0*uA
neurons.Is1_inh = 0*uA
neurons.Is2_inh = 0*uA

# =============================================================================
# Synapses  —  on_pre implements Iw·δ(t_sp) in Eq (11)
# =============================================================================
exc_syn = Synapses(exc_input, neurons, on_pre='Is1_exc_post += Iw_exc')
exc_syn.connect(j='i')   # Poisson[0]→Neuron[0], Poisson[1]→Neuron[1]

inh_syn = Synapses(inh_input, neurons, on_pre='Is1_inh_post += Iw_inh')
inh_syn.connect(j='i')

# =============================================================================
# Monitors
# =============================================================================
sp_exc  = SpikeMonitor(exc_input)
sp_inh  = SpikeMonitor(inh_input)
sp_neur = SpikeMonitor(neurons)
st_mon  = StateMonitor(neurons, ['Vm', 'Is2_exc', 'Is2_inh'], record=True, dt=0.5*ms)

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
# Plotting  (5 panels)
# =============================================================================
t_s   = st_mon.t / second
dur_s = float(duration / second)

C_exc = '#27AE60'    # green
C_inh = '#E74C3C'    # red
C_n0  = '#2980B9'    # blue
C_n1  = '#E67E22'    # orange

fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.65, wspace=0.38,
                        height_ratios=[1, 1, 1.2, 1.2, 1.3])

# ── Panel 0 (full-width): Input raster + rate overlay ────────────────────────
ax0 = fig.add_subplot(gs[0, :])

# Filter by neuron index (0 only, for clarity — both receive same rate profile)
exc_mask0 = sp_exc.i == 0
inh_mask0 = sp_inh.i == 0
ax0.eventplot(sp_exc.t[exc_mask0]/second, lineoffsets=2, linelengths=0.55,
              colors=C_exc, linewidths=0.5, label='Exc (20–120 Hz)')
ax0.eventplot(sp_inh.t[inh_mask0]/second, lineoffsets=1, linelengths=0.55,
              colors=C_inh, linewidths=0.5, label='Inh (20–80 Hz)')
ax0.set_xlim(0, dur_s); ax0.set_ylim(0.3, 2.7)
ax0.set_yticks([1, 2]); ax0.set_yticklabels(['Inh', 'Exc'], fontsize=9)
ax0.set_title('Poisson Input Spikes  (one channel shown; both neurons identical source)',
              fontsize=11, fontweight='bold')
ax0.set_ylabel('Channel'); ax0.legend(loc='upper right', fontsize=9, framealpha=0.85)

ax0r = ax0.twinx()
t_r = np.arange(n_steps) * float(rate_res/second)
ax0r.fill_between(t_r, exc_rates/Hz, alpha=0.12, color=C_exc)
ax0r.fill_between(t_r, inh_rates/Hz, alpha=0.12, color=C_inh)
ax0r.set_ylabel('Rate (Hz)', fontsize=8, color='gray')
ax0r.set_ylim(0, 160); ax0r.tick_params(labelsize=8, colors='gray')

# ── Panel 1 (full-width): Neural spike raster ─────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    mask = sp_neur.i == ni
    if mask.any():
        ax1.eventplot(sp_neur.t[mask]/second, lineoffsets=ni+1,
                      linelengths=0.55, colors=c, linewidths=0.7, label=lbl)
spk_counts = [n0, n1]
ax1.set_xlim(0, dur_s); ax1.set_ylim(0.3, 2.7)
ax1.set_yticks([1, 2]); ax1.set_yticklabels(['N0', 'N1'], fontsize=9)
ax1.set_title(f'Neural Output Spikes  '
              f'(N0: {n0} sp @ {n0/10:.1f} Hz,   N1: {n1} sp @ {n1/10:.1f} Hz)',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('Neuron'); ax1.legend(loc='upper right', fontsize=9, framealpha=0.85)

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
    ax.set_title(f'{lbl} — Membrane Potential  Vm(t)\n'
                 rf'$\tau_m = C_m(R_m[hi]+R_a) = {tau_m/ms:.0f}$ ms',
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('Vm  (V)')
    ax.set_xlim(0, dur_s)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)

# ── Panels 3: Synaptic currents Is2 ──────────────────────────────────────────
for ni, lbl in enumerate(['Neuron 0', 'Neuron 1']):
    ax = fig.add_subplot(gs[3, ni])
    exc_cur = st_mon.Is2_exc[ni]/uA
    inh_cur = st_mon.Is2_inh[ni]/uA
    net_cur = exc_cur - inh_cur
    ax.fill_between(t_s,  exc_cur, alpha=0.25, color=C_exc)
    ax.fill_between(t_s, -inh_cur, alpha=0.25, color=C_inh)
    ax.plot(t_s,  exc_cur, color=C_exc, lw=0.7, label='Is2_exc  (+)')
    ax.plot(t_s, -inh_cur, color=C_inh, lw=0.7, label='−Is2_inh  (−)')
    ax.plot(t_s,  net_cur, color='k',   lw=0.8, ls=':', alpha=0.8, label='Net Is2')
    # Mark I_min and I_max lines
    ax.axhline(float(I_min/uA), color='purple', ls='-.', lw=0.9, alpha=0.7,
               label=f'I_min = {I_min/uA:.0f} µA')
    ax.axhline(float(I_max/uA), color='brown',  ls='-.', lw=0.9, alpha=0.7,
               label=f'I_max = {I_max/uA:.0f} µA')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_title(f'{lbl} — Synaptic Currents  Is2(t)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Current  (µA)')
    ax.set_xlabel('Time  (s)')
    ax.set_xlim(0, dur_s)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85, ncol=2)

# ── Panel 4 (full-width): I-F curve ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[4, :])

ax4.plot(I_sweep_uA, f_IF, color='#8E44AD', lw=2.5, label='Analytical I-F curve')
ax4.fill_between(I_sweep_uA, f_IF, alpha=0.15, color='#8E44AD')

# Mark I_min and I_max
ax4.axvline(float(I_min/uA), color='navy',  ls='--', lw=1.5,
            label=f'I_min = {I_min/uA:.0f} µA  (rheobase)')
ax4.axvline(float(I_max/uA), color='darkred', ls='--', lw=1.5,
            label=f'I_max = {I_max/uA:.0f} µA')

# Shade operating region
ax4.axvspan(float(I_min/uA), float(I_max/uA), alpha=0.08, color='gold',
            label='Hardware operating range')

# Mark f_max (1/t_ref)
f_max = 1.0 / float(t_ref/second)
ax4.axhline(f_max, color='gray', ls=':', lw=1.2, label=f'f_max = 1/t_ref = {f_max:.0f} Hz')

# Annotate key points
f_at_Imax = float(f_IF[np.argmin(np.abs(I_sweep_uA - float(I_max/uA)))])
ax4.annotate(f'{f_at_Imax:.0f} Hz @ I_max',
             xy=(float(I_max/uA), f_at_Imax),
             xytext=(float(I_max/uA)+5, f_at_Imax+5),
             fontsize=8, arrowprops=dict(arrowstyle='->', color='k'))

ax4.set_xlabel('Injected current  I₀  (µA)', fontsize=10)
ax4.set_ylabel('Firing rate  f  (Hz)', fontsize=10)
ax4.set_title(
    r'Current–Frequency (I-F) Curve   '
    r'$f(I_0) = \left[t_{ref} - \tau_m \ln\!\left(1 - \frac{V_{thr}}{I_0(R_m[hi]+R_a)}\right)\right]^{-1}$',
    fontsize=10, fontweight='bold')
ax4.set_xlim(0, float(1.6*I_max/uA))
ax4.set_ylim(0, f_max * 1.15)
ax4.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax4.grid(axis='y', alpha=0.3)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    'Memristor aLIF Neuron — Brian2 Simulation  (Eqs. 9–12)\n'
    rf'$\tau_m={tau_m/ms:.0f}$ ms,  $\tau_s={tau_s/ms:.0f}$ ms,  '
    rf'$I_{{min}}={I_min/uA:.0f}$ µA,  $I_{{max}}={I_max/uA:.0f}$ µA,  '
    rf'$I_w^{{exc}}={Iw_exc/uA:.0f}$ µA,  $I_w^{{inh}}={Iw_inh/uA:.0f}$ µA',
    fontsize=11, fontweight='bold', y=1.005)

out_path = 'neuron_sim_v2.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")