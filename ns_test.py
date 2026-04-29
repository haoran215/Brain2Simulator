"""
Brian2 Demo – Memristor aLIF Neuron (Eqs. 9–12)
=================================================
• 2 neurons, each receiving 1 excitatory + 1 inhibitory Poisson synapse
• Synaptic weight modelled as Iw (current kick into Is1 stage)
• Poisson rates vary sinusoidally between 20–120 Hz (exc) / 20–80 Hz (inh)
• Plots: input raster | neural raster | Vm traces | Is2 synaptic currents
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

# ── Physical parameters ───────────────────────────────────────────────────────
Cm      = 100*pF       # membrane capacitance
Ra      = 10*Mohm      # axon-hillock resistor
Rm_hi   = 100*Mohm     # memristor HIGH resistance (open switch – quiescent)
Rm_lo   = 1*Mohm       # memristor LOW resistance  (closed – spike)
tau_s   = 10*ms        # synaptic time constant
I0      = 0*pA         # tonic bias current (zero – driven purely by synapses)
Vthresh = 12*mV        # spike threshold (on Vm)
Vreset  = 0*mV         # post-spike reset
t_ref   = 8*ms         # absolute refractory period

# tau_m = Cm*(Rm_hi+Ra) ≈ 100pF * 110 MΩ = 11 ms  (reasonable membrane decay)

# ── Synaptic weights Iw ───────────────────────────────────────────────────────
Iw_exc = 350*pA        # excitatory weight   (positive kick to Is1_exc)
Iw_inh = 350*pA        # inhibitory weight   (balanced; effect is subtractive)

# ── Time-varying Poisson rates ─────────────────────────────────────────────────
#   Exc: 20–120 Hz sinusoidal @ 0.15 Hz carrier
#   Inh: 20– 80 Hz sinusoidal @ 0.15 Hz, π phase-shifted
rate_res = 100*ms
n_steps  = int(duration / rate_res)
t_vals   = np.arange(n_steps) * float(rate_res / second)

exc_rates = (70 + 50 * np.sin(2*np.pi*0.15*t_vals)) * Hz   # 20–120 Hz
inh_rates = (50 + 30 * np.sin(2*np.pi*0.15*t_vals + np.pi)) * Hz  # 20–80 Hz

exc_rate_arr = TimedArray(exc_rates, dt=rate_res)
inh_rate_arr = TimedArray(inh_rates, dt=rate_res)

# ── Poisson input groups ──────────────────────────────────────────────────────
exc_input = PoissonGroup(2, rates='exc_rate_arr(t)')
inh_input = PoissonGroup(2, rates='inh_rate_arr(t)')

# ── Neuron model (Eqs. 9–12) ──────────────────────────────────────────────────
#
#  Eq (9)  Cm dVm/dt   = -Vm/(Rm_S+Ra) + Is2_exc - Is2_inh + I0
#  Eq (11) τs dIs1/dt  = -Is1   [+ Iw·δ(t_spike) via Synapse.on_pre]
#  Eq (12) τs dIs2/dt  = -Is2 + Is1
#
#  Note: separate (exc, inh) copies of (Is1, Is2) so we can visualise them.
#        Vpost = Vm*Ra/(Rm_S+Ra) — Eq (10), computed for reference but not
#        used as the spike trigger here (threshold is on Vm directly).

eqs = '''
dVm/dt      = (-Vm / (Rm_S + Ra) + Is2_exc - Is2_inh + I0) / Cm : volt
dIs1_exc/dt = -Is1_exc / tau_s                                     : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s                         : amp
dIs1_inh/dt = -Is1_inh / tau_s                                     : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s                         : amp
Vpost       = Vm * Ra / (Rm_S + Ra)                                : volt
Rm_S        : ohm
'''

neurons = NeuronGroup(2, model=eqs,
                      threshold='Vm > Vthresh',
                      reset='Vm = Vreset',
                      method='euler', refractory=t_ref)

# Initial conditions
neurons.Vm      = Vreset
neurons.Rm_S    = Rm_hi   # start in high-resistance (quiescent) state
neurons.Is1_exc = 0*pA
neurons.Is2_exc = 0*pA
neurons.Is1_inh = 0*pA
neurons.Is2_inh = 0*pA

# ── Synapses: on each pre-spike kick Is1 by Iw ───────────────────────────────
# Implements the δ-impulse term in Eq (11): τs dIs1/dt = -Is1 + Iw·δ(t_sp)
exc_syn = Synapses(exc_input, neurons, on_pre='Is1_exc_post += Iw_exc')
exc_syn.connect(j='i')   # one-to-one: Poisson[0]→N0, Poisson[1]→N1

inh_syn = Synapses(inh_input, neurons, on_pre='Is1_inh_post += Iw_inh')
inh_syn.connect(j='i')   # one-to-one: Poisson[0]→N0, Poisson[1]→N1

# ── Monitors ──────────────────────────────────────────────────────────────────
sp_exc  = SpikeMonitor(exc_input)
sp_inh  = SpikeMonitor(inh_input)
sp_neur = SpikeMonitor(neurons)
st_mon  = StateMonitor(neurons, ['Vm', 'Is2_exc', 'Is2_inh'], record=True, dt=0.5*ms)

# ── Run ───────────────────────────────────────────────────────────────────────
print("Running 10 s simulation …")
run(duration, report='text')

n0_spikes = int(np.sum(sp_neur.i == 0))
n1_spikes = int(np.sum(sp_neur.i == 1))
print(f"  Exc input spikes : {len(sp_exc.t)}")
print(f"  Inh input spikes : {len(sp_inh.t)}")
print(f"  Neuron 0 spikes  : {n0_spikes}  ({n0_spikes/10:.1f} Hz)")
print(f"  Neuron 1 spikes  : {n1_spikes}  ({n1_spikes/10:.1f} Hz)")

# ── Colour palette ────────────────────────────────────────────────────────────
C_exc = '#27AE60'    # green  – excitatory
C_inh = '#E74C3C'    # red    – inhibitory
C_n0  = '#2980B9'    # blue   – neuron 0
C_n1  = '#E67E22'    # orange – neuron 1

t_s    = st_mon.t / second
dur_s  = float(duration / second)

# ── Figure layout: 4 rows × 2 cols ───────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.60, wspace=0.38)

# ── Row 0 (full-width): Poisson input raster ──────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.eventplot(sp_exc.t/second, lineoffsets=2, linelengths=0.55,
              colors=C_exc, linewidths=0.6, label='Excitatory (20–120 Hz)')
ax0.eventplot(sp_inh.t/second, lineoffsets=1, linelengths=0.55,
              colors=C_inh, linewidths=0.6, label='Inhibitory (20–80 Hz)')
ax0.set_xlim(0, dur_s)
ax0.set_ylim(0.3, 2.7)
ax0.set_yticks([1, 2]); ax0.set_yticklabels(['Inh input', 'Exc input'], fontsize=9)
ax0.set_title('Poisson Input Spikes  (time-varying rate, sinusoidal modulation)',
              fontsize=11, fontweight='bold')
ax0.set_ylabel('Channel')
ax0.legend(loc='upper right', fontsize=9, framealpha=0.8)
# Overlay instantaneous rate as shaded background
ax0_r = ax0.twinx()
t_rate = np.arange(n_steps) * float(rate_res/second)
ax0_r.fill_between(t_rate, exc_rates/Hz, alpha=0.12, color=C_exc)
ax0_r.fill_between(t_rate, inh_rates/Hz, alpha=0.12, color=C_inh)
ax0_r.set_ylabel('Rate (Hz)', fontsize=8, color='gray')
ax0_r.set_ylim(0, 160); ax0_r.tick_params(labelsize=8, colors='gray')

# ── Row 1 (full-width): Neural spike raster ───────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    mask = sp_neur.i == ni
    if mask.any():
        ax1.eventplot(sp_neur.t[mask]/second, lineoffsets=ni+1,
                      linelengths=0.55, colors=c, linewidths=0.8, label=lbl)
ax1.set_xlim(0, dur_s)
ax1.set_ylim(0.3, 2.7)
ax1.set_yticks([1, 2]); ax1.set_yticklabels(['Neuron 0', 'Neuron 1'], fontsize=9)
ax1.set_title(f'Neural Output Spikes  '
              f'(N0: {n0_spikes} spikes @ {n0_spikes/10:.1f} Hz,  '
              f'N1: {n1_spikes} spikes @ {n1_spikes/10:.1f} Hz)',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('Neuron')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.8)

# ── Rows 2–3: per-neuron Vm and Is2 ──────────────────────────────────────────
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):

    # -- Membrane potential (row 2) -------------------------------------------
    ax_v = fig.add_subplot(gs[2, ni])
    ax_v.plot(t_s, st_mon.Vm[ni]/mV, color=c, lw=0.7, label='Vm')
    ax_v.axhline(Vthresh/mV, color='dimgray', ls='--', lw=1.0,
                 label=f'Threshold  ({Vthresh/mV:.0f} mV)')
    mask = sp_neur.i == ni
    if mask.any():
        ax_v.vlines(sp_neur.t[mask]/second,
                    Vthresh/mV, Vthresh/mV + 2.5,
                    colors='k', lw=0.9, label='Spike', zorder=5)
    ax_v.set_title(f'{lbl} — Membrane Potential  Vm(t)', fontsize=10, fontweight='bold')
    ax_v.set_ylabel('Vm  (mV)')
    ax_v.set_xlim(0, dur_s)
    ax_v.legend(fontsize=7, loc='upper right', framealpha=0.8)

    # -- Synaptic currents Is2 (row 3) ----------------------------------------
    ax_s = fig.add_subplot(gs[3, ni])
    ax_s.plot(t_s,  st_mon.Is2_exc[ni]/pA,   color=C_exc, lw=0.7, label='Is2_exc  (+)')
    ax_s.plot(t_s, -st_mon.Is2_inh[ni]/pA,  color=C_inh, lw=0.7, label='−Is2_inh  (−)')
    net = (st_mon.Is2_exc[ni] - st_mon.Is2_inh[ni]) / pA
    ax_s.plot(t_s, net, color='k', lw=0.5, ls=':', alpha=0.7, label='Net Is2')
    ax_s.axhline(0, color='k', lw=0.4, ls=':')
    ax_s.set_title(f'{lbl} — Synaptic Currents  Is2(t)', fontsize=10, fontweight='bold')
    ax_s.set_ylabel('Current  (pA)')
    ax_s.set_xlabel('Time  (s)')
    ax_s.set_xlim(0, dur_s)
    ax_s.legend(fontsize=7, loc='upper right', framealpha=0.8)

fig.suptitle(
    'Memristor aLIF Neuron — Brian2 Simulation\n'
    r'$C_m\,\dot{V}_m = -V_m/(R_m[S]+R_a) + I_{s2}^{exc} - I_{s2}^{inh} + I_0$'
    r'     $\tau_s\,\dot{I}_{s1} = -I_{s1}+I_w\,\delta(t_{sp})$'
    r'     $\tau_s\,\dot{I}_{s2} = -I_{s2}+I_{s1}$',
    fontsize=11, fontweight='bold', y=1.01)

out_path = 'neuron_sim.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Figure saved → {out_path}")