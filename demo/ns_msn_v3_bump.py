"""
ns_msn_v3_bump.py
=================
Single MSN + one self-excitatory synapse — minimal "bump" test.

Inheritance:
  ns_msn_v2_synapses.py — multi-neuron + Poisson, no recurrence

  THIS FILE:
      one MSN with I_0 just below rheobase (silent on its own)
      one self-excitatory synapse  (neuron 0 → neuron 0)
      one external input pulse at t = t_pulse  (single spike via SpikeGen)

Behaviour we expect (the bump):
  Pulse  →  Is2_exc kick of ~Iw_input/e  →  total drive crosses I_min
         →  first MSN spike
         →  self-syn adds Iw_recur to Is1_exc
         →  more spikes, while  I_0 + Is2_recur > I_min
         →  recurrent current eventually decays  →  firing stops
         →  bump dissipated.

Key tuning ratio:
  Iw_recur · f · τ_s    vs    I_min − I_0
      LARGER  → persistent firing (latched)
      SMALLER → bump fades (this file targets here, slightly subcritical)
"""
#%%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

prefs.codegen.target = 'numpy'   # pure-Python backend, avoids C++ compiler
defaultclock.dt = 10*us

# ─── Parameters ──────────────────────────────────────────────────────────────
params = MSNParams()                      # intrinsic params (no tau here)
tau_s_val = 200e-3                        # synapse cascade τ (s)
print(params.summary())
I_min, I_max = params.operating_window()

I_0_val      = 0.4 * I_min     # tonic bias current, just below rheobase (silent on its own)
I_pulse_val  = 0.65 * I_min     # 1s boost: I_0 + I_pulse
Iw_recur_val = 15e-7            # self-excitation sustains firing after pulse
t_pulse      = 2.0*second       # let Vm settle before triggering
t_pulse_dur  = 0.5*second       # pulse duration
T_run        = 10.0*second

# ─── Build neuron + self-excitatory synapse ───────────────────────────────────
neuron = make_msn(N=1, params=params, name='msn_pop')
neuron.I_0 = I_0_val * amp

# Cascade τ now lives on the synapse, not the neuron.
syn_recur = make_synapse(
    source=neuron, target=neuron,
    params=SynapseParams(kind='exc', weight=Iw_recur_val,
                         tau_s1=tau_s_val, tau_s2=tau_s_val),
    connect='i == j', name='syn_recur',
)

# 1-second step current pulse via network_operation
@network_operation(when='start')
def apply_pulse(t):
    if t_pulse <= t < t_pulse + t_pulse_dur:
        neuron.I_0[0] = (I_0_val + I_pulse_val) * amp
    else:
        neuron.I_0[0] = I_0_val * amp

# ─── Monitors ────────────────────────────────────────────────────────────────
sp_mon = SpikeMonitor(neuron)
st_mon = StateMonitor(neuron, ['Vm', 'Vout', 's', 'I_0'],
                      record=True, dt=200*us)
# Cascade is on the synapse; record Is1/Is2 from the synapse object.
syn_mon = StateMonitor(syn_recur, ['Is1', 'Is2'], record=True, dt=200*us)

I_during = I_0_val + I_pulse_val
deficit  = I_min - I_0_val
f_crit   = deficit / (Iw_recur_val * tau_s_val)

print(f"\nExperiment:")
print(f"  I_0          = {I_0_val*1e6:.2f} µA  ({I_0_val/I_min*100:.0f}% of rheobase — subthreshold)")
print(f"  I_during     = {I_during*1e6:.2f} µA  ({I_during/I_min*100:.0f}% of rheobase — {'ABOVE' if I_during > I_min else 'below'} I_min)")
print(f"  Iw_recur     = {Iw_recur_val*1e6:.0f} µA   (self-excitatory)")
print(f"  Pulse window : t = {t_pulse/second:.1f}s → {(t_pulse+t_pulse_dur)/second:.1f}s  ({t_pulse_dur/second:.0f} s)")
print(f"  Deficit I_min-I_0 = {deficit*1e6:.2f} µA  |  sustain for f > {f_crit:.2f} Hz")
print(f"  Duration     = {T_run/second:.1f} s")
print()

run(T_run, report='text')

n_sp = len(sp_mon.t)
print(f"\n  Output spikes: {n_sp}")
if n_sp:
    times_ms = np.array(sp_mon.t / ms)
    t_p_ms   = float(t_pulse / ms)
    t_end_ms = float((t_pulse + t_pulse_dur) / ms)
    pre_sp   = times_ms[times_ms <  t_p_ms]
    dur_sp   = times_ms[(times_ms >= t_p_ms) & (times_ms < t_end_ms)]
    post_sp  = times_ms[times_ms >= t_end_ms]
    print(f"  Before pulse : {len(pre_sp)} spikes")
    print(f"  During pulse : {len(dur_sp)} spikes")
    print(f"  After pulse  : {len(post_sp)} spikes  ({'SUSTAINED' if len(post_sp) > 0 else 'SILENT'})")
    if len(post_sp) >= 2:
        f_post  = 1000.0 / np.mean(np.diff(post_sp))
        sustain = Iw_recur_val * f_post * tau_s_val
        print(f"  Post-pulse mean rate: {f_post:.1f} Hz  |  "
              f"Iw·f·τ_s = {sustain*1e6:.2f} µA  vs  deficit = {deficit*1e6:.2f} µA  "
              f"→  {'SUSTAIN' if sustain > deficit else 'FADE'}")

# ─── Plot ────────────────────────────────────────────────────────────────────
t_ms  = np.array(st_mon.t / ms)
Vm    = np.array(st_mon.Vm[0]   / volt)
Vout  = np.array(st_mon.Vout[0] / volt)
Is1e  = np.array(syn_mon.Is1[0] / uA)   # cascade now lives on the synapse
Is2e  = np.array(syn_mon.Is2[0] / uA)
I0_v  = np.array(st_mon.I_0[0] / uA)
s_v   = np.array(st_mon.s[0])

t_p_ms   = float(t_pulse / ms)
t_end_ms = float((t_pulse + t_pulse_dur) / ms)

fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True,
                         gridspec_kw=dict(hspace=0.40))

def shade(ax):
    ax.axvspan(t_p_ms, t_end_ms, alpha=0.13, color='tomato', label='1 s pulse')

# (0) Input current
ax = axes[0]
ax.plot(t_ms, I0_v, color='C1', lw=1.2)
ax.axhline(I_min*1e6, color='gray', ls='--', lw=1,
           label=f'I_min = {I_min*1e6:.1f} µA (rheobase)')
ax.axhline(I_0_val*1e6, color='steelblue', ls=':', lw=1,
           label=f'baseline I_0 = {I_0_val*1e6:.2f} µA')
shade(ax)
ax.set_ylabel('I_0 (µA)')
ax.set_title('Step input — subthreshold baseline + 1 s pulse above rheobase',
             fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# (1) Vm
ax = axes[1]
ax.plot(t_ms, Vm, color='C0', lw=0.7)
ax.axhline(params.Vth, color='dimgray', ls='--', lw=1,
           label=f'Vth = {params.Vth:.2f} V')
shade(ax)
for ts in sp_mon.t / ms:
    ax.vlines(ts, params.Vth, params.Vth + 0.08, colors='k', lw=0.8, zorder=5)
ax.set_ylabel('Vm (V)')
ax.set_title(
    f'Membrane voltage  |  I_0={I_0_val*1e6:.1f} µA (subthr.), '
    f'Iw_rec={Iw_recur_val*1e6:.0f} µA, τ_s={tau_s_val*1e3:.0f} ms  →  {n_sp} spikes',
    fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# (2) Vout
ax = axes[2]
ax.plot(t_ms, Vout*1e3, color='C2', lw=0.7)
shade(ax)
ax.set_ylabel('Vout (mV)')
ax.set_title('Vout — output spike train', fontweight='bold')

# (3) Currents
ax = axes[3]
ax.plot(t_ms, Is1e, color='C3', lw=0.6, ls='--', alpha=0.7, label='Is1_exc')
ax.plot(t_ms, Is2e, color='C3', lw=1.4, label='Is2_exc (recurrent)')
ax.plot(t_ms, Is2e + I0_v, color='k', lw=0.9, ls=':',
        label='Is2 + I_0 (total drive)')
ax.axhline(I_min*1e6, color='gray', ls='-.', lw=1,
           label=f'I_min = {I_min*1e6:.1f} µA')
ax.axhline(I_max*1e6, color='red', ls='-.', lw=1,
           label=f'I_max = {I_max*1e6:.0f} µA')
shade(ax)
ax.set_ylabel('current (µA)')
ax.set_title('Synaptic + bias current — does Is2+I_0 stay above I_min after pulse?',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)

# (4) Memristor state
ax = axes[4]
ax.plot(t_ms, s_v, color='C4', lw=0.6, drawstyle='steps-post')
shade(ax)
ax.set_ylabel('s')
ax.set_xlabel('t (ms)')
ax.set_title('Memristor state (0=open, 1=closed)', fontweight='bold')
ax.set_ylim(-0.1, 1.1)

fig.suptitle(
    'MSN 1 s-pulse test  —  pulse triggers activity; '
    'subthreshold I_0 + self-recurrence sustains it',
    fontsize=12, fontweight='bold', y=1.005)
plt.show()
out_path = 'demo/ns_msn_v3_bump.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")

# %%
