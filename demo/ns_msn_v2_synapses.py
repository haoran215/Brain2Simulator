"""
ns_msn_v2_synapses.py
=====================
MSN core (ns_msn_v1.py) + Is1/Is2 cascade synapses (from ns_test.py).

─── Inheritance ─────────────────────────────────────────────────────────────
  ns_test.py:           aLIF + Is1 (exp) → Is2 (alpha) cascade synapses,
                        Poisson exc/inh inputs.  Threshold+reset spike,
                        t_ref refractory.  No spike shape.
  ns_msn_v1.py:         MSN — paper-faithful memristor spike shape (no syn).
  ns_msn_if_sweep.py:   MSN I-F characterisation (no syn).
  ns_msn_compare.py:    aLIF vs Thyristor vs MSN side by side (no syn).

  THIS FILE:            MSN + synapses.
                        - Drops threshold-reset rule (uses ns_msn_v1's
                          hysteretic Rm state machine instead).
                        - Drops t_ref (refractoriness emerges from τ_close).
                        - Keeps Is1 (exp) → Is2 (alpha) cascade from ns_test.
                        - Keeps Poisson Pre-syn driver from ns_test.

─── Membrane equation ──────────────────────────────────────────────────────
  Cm·dVm/dt = I_0 + Is2_exc − Is2_inh − Vm/(Rm_S + Ra)

  with Rm_S hysteretically switching Rm_hi ↔ Rm_lo per the thyristor rule.

─── Synapse cascade  (Eqs. 11–12, ns_test.py) ──────────────────────────────
  τ_s1 · dIs1/dt = −Is1               + Iw·δ(t − t_pre)   (on_pre)
  τ_s2 · dIs2/dt = −Is2 + Is1          (passive cascade)

  NETWORK_MODE = 'Is2'  → alpha-shaped current drives Vm  (NMDA-like)
  NETWORK_MODE = 'Is1'  → exponential                       (AMPA-like)

─── Parameter scale ────────────────────────────────────────────────────────
  MSN operates at I_min ≈ 9 µA, I_max = I_hold = 100 µA (paper Fig. 2).
  We pick I_0 in the middle of this band and synaptic weights such that
  the cumulative input modulates the neuron across most of the I-F curve
  without crossing into depolarisation-block.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *

seed(42)
defaultclock.dt = 10*us            # spike width is ~5 ms → 500 steps/spike

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ MSN hardware parameters (paper Fig. 2 — same as ns_msn_v1.py)           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
Cm_F      = 10e-6
Ra_ohm    = 47
Rm_hi_ohm = 100e3
Rm_lo_ohm = 500
Vth_V     = 0.9
I_hold_A  = 100e-6

# ─── Tonic bias (background drive) ──────────────────────────────────────────
I_0_uA    = 30.0           # µA — places neuron a bit above rheobase

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Synapse parameters                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
Rs_ohm    = 10e3
Cs_F      = 5e-6           # ↑ from 1 µF → τ_s = 50 ms (matches MSN's slow τ_m)
Iw_exc_uA = 6.0            # µA per pre-syn spike (cumulative many spikes)
Iw_inh_uA = 6.0
NETWORK_MODE = 'Is2'        # 'Is1' (exp) or 'Is2' (alpha) drives Vm

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Poisson drivers (sinusoidally modulated rates)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
duration  = 3*second
rate_res  = 100*ms
n_steps   = int(duration / rate_res)
t_vals    = np.arange(n_steps) * float(rate_res/second)

exc_rates = (60 + 40*np.sin(2*np.pi*0.5*t_vals)) * Hz
inh_rates = (40 + 25*np.sin(2*np.pi*0.5*t_vals + np.pi)) * Hz

exc_rate_arr = TimedArray(exc_rates, dt=rate_res)
inh_rate_arr = TimedArray(inh_rates, dt=rate_res)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Brian2 quantities                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
Cm     = Cm_F      * farad
Ra     = Ra_ohm    * ohm
Rm_hi  = Rm_hi_ohm * ohm
Rm_lo  = Rm_lo_ohm * ohm
Vth    = Vth_V     * volt
I_hold = I_hold_A  * amp
I_0    = I_0_uA    * uA
Rs     = Rs_ohm    * ohm
Cs     = Cs_F      * farad
tau_s1 = Rs * Cs
tau_s2 = Rs * Cs
Iw_exc = Iw_exc_uA * uA
Iw_inh = Iw_inh_uA * uA

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Equations                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if NETWORK_MODE == 'Is1':
    I_syn_term = 'Is1_exc - Is1_inh'
    syn_label  = 'Is1 (exponential / AMPA-like)'
else:
    I_syn_term = 'Is2_exc - Is2_inh'
    syn_label  = 'Is2 (alpha function / NMDA-like)'

eqs = f'''
dVm/dt      = (I_0 + {I_syn_term} - Vm/(Rm_S + Ra)) / Cm   : volt
Rm_S        = (1 - s)*Rm_hi + s*Rm_lo                       : ohm
I_M         = Vm / (Rm_S + Ra)                              : amp
Vout        = Vm * Ra / (Rm_S + Ra)                         : volt
dIs1_exc/dt = -Is1_exc / tau_s1                             : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                 : amp
dIs1_inh/dt = -Is1_inh / tau_s1                             : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                 : amp
s           : 1
'''

N = 2
neurons = NeuronGroup(
    N, model=eqs,
    threshold='Vm > Vth and s < 0.5',
    reset='s = 1',
    events={'reopen': 'I_M < I_hold and s > 0.5'},
    method='euler',
)
neurons.run_on_event('reopen', 's = 0')

neurons.Vm = 0*volt
neurons.s  = 0
neurons.Is1_exc = 0*uA;  neurons.Is2_exc = 0*uA
neurons.Is1_inh = 0*uA;  neurons.Is2_inh = 0*uA

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Poisson inputs + synapses                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
exc_input = PoissonGroup(N, rates='exc_rate_arr(t)')
inh_input = PoissonGroup(N, rates='inh_rate_arr(t)')

exc_syn = Synapses(exc_input, neurons, on_pre='Is1_exc_post += Iw_exc')
exc_syn.connect(j='i')
inh_syn = Synapses(inh_input, neurons, on_pre='Is1_inh_post += Iw_inh')
inh_syn.connect(j='i')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Monitors                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sp_exc  = SpikeMonitor(exc_input)
sp_inh  = SpikeMonitor(inh_input)
sp_neur = SpikeMonitor(neurons)
st_mon  = StateMonitor(neurons,
                       ['Vm', 'Vout', 's',
                        'Is1_exc', 'Is2_exc', 'Is1_inh', 'Is2_inh'],
                       record=True, dt=200*us)

print("=" * 68)
print("  MSN v2 — paper-faithful memristor + synapses")
print("=" * 68)
print(f"  Cm        = {Cm_F*1e6:.0f} µF      Ra      = {Ra_ohm} Ω")
print(f"  Rm_hi     = {Rm_hi_ohm/1e3:.0f} kΩ    Rm_lo   = {Rm_lo_ohm} Ω")
print(f"  Vth       = {Vth_V:.2f} V    I_hold  = {I_hold_A*1e6:.0f} µA")
print(f"  I_0       = {I_0_uA:.1f} µA  (tonic bias)")
print(f"  τ_s1 = τ_s2 = Rs·Cs = {Rs_ohm*Cs_F*1e3:.0f} ms")
print(f"  Iw_exc    = {Iw_exc_uA:.1f} µA   Iw_inh = {Iw_inh_uA:.1f} µA")
print(f"  exc rate  = 20–100 Hz (sinusoidal)")
print(f"  inh rate  = 15–65  Hz (sinusoidal, antiphase)")
print(f"  mode      = '{NETWORK_MODE}'  ({syn_label})")
print(f"  duration  = {duration/second:.1f} s")
print("=" * 68)

print("\nRunning ...")
run(duration, report='text')

n0 = int(np.sum(sp_neur.i == 0))
n1 = int(np.sum(sp_neur.i == 1))
print(f"\n  Exc input spikes : {len(sp_exc.t)}")
print(f"  Inh input spikes : {len(sp_inh.t)}")
print(f"  Neuron 0 spikes  : {n0}  →  {n0/float(duration/second):.2f} Hz")
print(f"  Neuron 1 spikes  : {n1}  →  {n1/float(duration/second):.2f} Hz")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Plot                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
t_s   = np.array(st_mon.t / second)
dur_s = float(duration / second)

C_exc = '#27AE60'; C_inh = '#E74C3C'
C_n0  = '#2980B9'; C_n1  = '#E67E22'

fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.65, wspace=0.30,
                        height_ratios=[1.0, 1.0, 1.2, 1.4, 1.4])

# ─── Row 0: Poisson input raster + rate profile ──────────────────────────────
ax = fig.add_subplot(gs[0, :])
ax.eventplot(np.array(sp_exc.t[sp_exc.i==0]/second), lineoffsets=2,
             linelengths=0.55, colors=C_exc, linewidths=0.5)
ax.eventplot(np.array(sp_inh.t[sp_inh.i==0]/second), lineoffsets=1,
             linelengths=0.55, colors=C_inh, linewidths=0.5)
ax.set_xlim(0, dur_s); ax.set_ylim(0.3, 2.7)
ax.set_yticks([1, 2]); ax.set_yticklabels(['Inh', 'Exc'], fontsize=9)
ax.set_title(f'Poisson input (channel 0) — sinusoidally modulated rates',
             fontsize=11, fontweight='bold')
ax.set_ylabel('Channel')
axr = ax.twinx()
axr.fill_between(t_vals, exc_rates/Hz, alpha=0.12, color=C_exc, label='Exc rate')
axr.fill_between(t_vals, inh_rates/Hz, alpha=0.12, color=C_inh, label='Inh rate')
axr.set_ylabel('Rate (Hz)', fontsize=8, color='gray')
axr.tick_params(labelsize=8, colors='gray'); axr.set_ylim(0, 110)
axr.legend(fontsize=8, loc='upper right')

# ─── Row 1: Neuron output raster ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
for ni, c, lbl in zip([0,1], [C_n0, C_n1], ['Neuron 0', 'Neuron 1']):
    mask = sp_neur.i == ni
    if mask.any():
        ax.eventplot(np.array(sp_neur.t[mask]/second), lineoffsets=ni+1,
                     linelengths=0.55, colors=c, linewidths=0.7, label=lbl)
ax.set_xlim(0, dur_s); ax.set_ylim(0.3, 2.7)
ax.set_yticks([1, 2]); ax.set_yticklabels(['N0', 'N1'], fontsize=9)
ax.set_title(
    f'MSN output spikes  '
    f'(N0: {n0} sp @ {n0/dur_s:.2f} Hz   |   N1: {n1} sp @ {n1/dur_s:.2f} Hz)\n'
    f'mode={syn_label}     I_0 = {I_0_uA:.1f} µA',
    fontsize=10, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)

# ─── Row 2: Vm traces ────────────────────────────────────────────────────────
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    ax = fig.add_subplot(gs[2, ni])
    ax.plot(t_s, st_mon.Vm[ni]/volt, color=c, lw=0.6)
    ax.axhline(Vth_V, color='dimgray', ls='--', lw=1, label=f'Vth = {Vth_V:.2f} V')
    mask = sp_neur.i == ni
    if mask.any():
        ax.vlines(np.array(sp_neur.t[mask]/second), Vth_V, Vth_V+0.1,
                  colors='k', lw=0.8, zorder=5, label='close events')
    ax.set_title(f'{lbl} — Vm(t)  (no reset rule; spike shape emerges)',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Vm (V)')
    ax.set_xlim(0, dur_s); ax.legend(fontsize=8, loc='upper right')

# ─── Row 3: Synaptic currents (Is1 dashed, Is2 solid) ────────────────────────
for ni, lbl in enumerate(['Neuron 0', 'Neuron 1']):
    ax = fig.add_subplot(gs[3, ni])
    Is1e = np.array(st_mon.Is1_exc[ni]/uA); Is2e = np.array(st_mon.Is2_exc[ni]/uA)
    Is1i = np.array(st_mon.Is1_inh[ni]/uA); Is2i = np.array(st_mon.Is2_inh[ni]/uA)

    ax.plot(t_s,  Is1e, color=C_exc, lw=0.6, ls='--', alpha=0.6, label='Is1_exc')
    ax.plot(t_s, -Is1i, color=C_inh, lw=0.6, ls='--', alpha=0.6, label='-Is1_inh')
    ax.plot(t_s,  Is2e, color=C_exc, lw=1.2, ls='-', label='Is2_exc')
    ax.plot(t_s, -Is2i, color=C_inh, lw=1.2, ls='-', label='-Is2_inh')

    if NETWORK_MODE == 'Is2':
        net = Is2e - Is2i
    else:
        net = Is1e - Is1i
    ax.fill_between(t_s, net, alpha=0.15, color='gray')
    ax.plot(t_s, net, color='k', lw=0.8, ls=':', alpha=0.7,
            label=f'Net {NETWORK_MODE} → Vm')

    # operating window markers
    I_min_uA_calc = Vth_V/(Rm_hi_ohm+Ra_ohm)*1e6
    ax.axhline( I_hold_A*1e6 - I_0_uA, color='red',    ls='-.', lw=0.8, alpha=0.6,
               label=f'I_syn = I_hold-I_0 = {I_hold_A*1e6-I_0_uA:.0f} µA')
    ax.axhline( I_min_uA_calc - I_0_uA, color='gray',  ls='-.', lw=0.8, alpha=0.6,
               label=f'I_syn = I_min-I_0 = {I_min_uA_calc-I_0_uA:.1f} µA')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_title(f'{lbl} — synaptic currents (Is1 dashed, Is2 solid)',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Current (µA)')
    ax.set_xlim(0, dur_s); ax.legend(fontsize=7, loc='upper right', ncol=2)

# ─── Row 4: Vout (paper-style trace) ─────────────────────────────────────────
for ni, (c, lbl) in enumerate(zip([C_n0, C_n1], ['Neuron 0', 'Neuron 1'])):
    ax = fig.add_subplot(gs[4, ni])
    ax.plot(t_s, st_mon.Vout[ni]*1e3/volt, color=c, lw=0.7)
    ax.set_title(f'{lbl} — Vout  (= Vm·Ra/(Rm_S+Ra))   ←   externally measured spikes',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Vout (mV)')
    ax.set_xlim(0, dur_s)

fig.suptitle(
    f'MSN v2 — paper-faithful memristor + synapses  |  '
    f'I_0={I_0_uA:.0f} µA, τ_s={Rs_ohm*Cs_F*1e3:.0f} ms, '
    f'Iw={Iw_exc_uA:.1f}/{Iw_inh_uA:.1f} µA (exc/inh), '
    f'mode={NETWORK_MODE}',
    fontsize=12, fontweight='bold', y=0.995)

out_path = 'demo/ns_msn_v2_synapses.png'
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
