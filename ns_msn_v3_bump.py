"""
ns_msn_v3_bump.py
=================
Single MSN + one self-excitatory synapse — minimal "bump" test.

Inheritance:
  msn_lib.py            — compact module: MSNParams, make_msn, make_synapse
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

Key tuning ratio (see msn_lib.py §6):
  Iw_recur · f · τ_s    vs    I_min − I_0
      LARGER  → persistent firing (latched)
      SMALLER → bump fades (this file targets here, slightly subcritical)
"""
#%%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *
from msn_lib import MSNParams, make_msn, make_synapse

defaultclock.dt = 10*us

# ─── Parameters ──────────────────────────────────────────────────────────────
# Use τ_s = 500 ms to give synaptic current enough lifetime to charge the
# slow membrane (τ_open = 1 s).  Pulse after baseline settles (t > 3·τ_open).
params = MSNParams(tau_s1=500e-3, tau_s2=500e-3)
print(params.summary())
I_min, I_max = params.operating_window()

I_0_val      = 0.85 * I_min     # 85% of rheobase: silent, near threshold
Iw_input_val = 30e-6            # ΔVm ≈ Iw·τ_s/Cm = 1.5 V (overkill — will fire)
Iw_recur_val = 6e-6             # marginal: a few self-spikes then fade
t_pulse      = 2.5*second       # let Vm baseline settle near I_0·R_total
T_run        = 6.0*second

# ─── Build neuron + synapses via msn_lib ─────────────────────────────────────
neuron = make_msn(N=1, params=params, name='msn_pop')
neuron.I_0 = I_0_val * amp

# Self-excitatory recurrent connection: neuron 0 → neuron 0
syn_recur = make_synapse(
    source=neuron, target=neuron,
    kind='exc', weight=Iw_recur_val,
    connect='i == j', name='syn_recur',
)

# External input pulse: single spike at t_pulse via SpikeGenerator
gen = SpikeGeneratorGroup(1, [0], [t_pulse], name='input_gen')
syn_input = make_synapse(
    source=gen, target=neuron,
    kind='exc', weight=Iw_input_val,
    connect=True, name='syn_input',
)

# ─── Monitors ────────────────────────────────────────────────────────────────
sp_mon = SpikeMonitor(neuron)
st_mon = StateMonitor(neuron, ['Vm', 'Vout', 's', 'Is1_exc', 'Is2_exc'],
                      record=True, dt=200*us)

print(f"\nExperiment:")
print(f"  I_0          = {I_0_val*1e6:.2f} µA  ({I_0_val/I_min*100:.0f}% of rheobase)")
print(f"  Iw_input     = {Iw_input_val*1e6:.0f} µA   (one pulse at t={t_pulse/ms:.0f} ms)")
print(f"  Iw_recur     = {Iw_recur_val*1e6:.0f} µA   (self-excitatory)")
print(f"  Trigger margin: e·(I_min−I_0) = "
      f"{2.718*(I_min-I_0_val)*1e6:.2f} µA  →  Iw_input must exceed this")
print(f"  Sustain check: Iw_recur·f·τ_s vs I_min−I_0 = "
      f"{(I_min-I_0_val)*1e6:.2f} µA  (compare after we know f)")
print(f"  Duration     = {T_run/second:.1f} s")
print()

run(T_run, report='text')

n_sp = len(sp_mon.t)
print(f"\n  Output spikes: {n_sp}")
if n_sp:
    times_ms = np.array(sp_mon.t/ms)
    if n_sp >= 2:
        print(f"  ISIs (ms): {np.diff(times_ms).round(1).tolist()}")
        f_in_burst = 1000.0 / np.mean(np.diff(times_ms))
        print(f"  Mean f during burst: {f_in_burst:.1f} Hz")
        # post-hoc sustainability check
        sustain_term = Iw_recur_val * f_in_burst * params.tau_s2
        deficit_term = I_min - I_0_val
        print(f"  Iw·f·τ_s = {sustain_term*1e6:.2f} µA  vs  "
              f"I_min−I_0 = {deficit_term*1e6:.2f} µA  →  "
              f"{'SUSTAIN' if sustain_term > deficit_term else 'FADE'}")

# ─── Plot ────────────────────────────────────────────────────────────────────
t_ms = np.array(st_mon.t / ms)
Vm   = np.array(st_mon.Vm[0]   / volt)
Vout = np.array(st_mon.Vout[0] / volt)
Is1e = np.array(st_mon.Is1_exc[0] / uA)
Is2e = np.array(st_mon.Is2_exc[0] / uA)
s_v  = np.array(st_mon.s[0])

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                         gridspec_kw=dict(hspace=0.35))

# (0) Vm
ax = axes[0]
ax.plot(t_ms, Vm, color='C0', lw=0.7)
ax.axhline(params.Vth, color='dimgray', ls='--', lw=1,
           label=f'Vth = {params.Vth:.2f} V')
ax.axvline(t_pulse/ms, color='red', ls=':', lw=1.4, label='input pulse')
for ts in sp_mon.t/ms:
    ax.vlines(ts, params.Vth, params.Vth+0.08, colors='k', lw=0.8, zorder=5)
ax.set_ylabel('Vm (V)')
ax.set_title(
    f'Bump test  |  I_0={I_0_val*1e6:.1f} µA (subthr.), '
    f'Iw_in={Iw_input_val*1e6:.0f} µA, Iw_rec={Iw_recur_val*1e6:.0f} µA, '
    f'τ_s={params.tau_s1*1e3:.0f} ms  →  {n_sp} output spikes',
    fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# (1) Vout
ax = axes[1]
ax.plot(t_ms, Vout*1e3, color='C2', lw=0.7)
ax.axvline(t_pulse/ms, color='red', ls=':', lw=1.4)
ax.set_ylabel('Vout (mV)')
ax.set_title('Vout — externally measured spike train', fontweight='bold')

# (2) Is1, Is2, total drive
ax = axes[2]
ax.plot(t_ms, Is1e, color='C3', lw=0.6, ls='--', alpha=0.7, label='Is1_exc')
ax.plot(t_ms, Is2e, color='C3', lw=1.4,                      label='Is2_exc')
ax.plot(t_ms, Is2e + I_0_val*1e6, color='k', lw=0.9, ls=':',
        label='Is2 + I_0 (total drive)')
ax.axhline(I_min*1e6, color='gray',  ls='-.', lw=1,
           label=f'I_min = {I_min*1e6:.1f} µA')
ax.axhline(I_max*1e6, color='red',   ls='-.', lw=1,
           label=f'I_max = {I_max*1e6:.0f} µA')
ax.axhline(I_0_val*1e6, color='green', ls=':', lw=1,
           label=f'I_0 = {I_0_val*1e6:.2f} µA')
ax.axvline(t_pulse/ms, color='red', ls=':', lw=1.4)
ax.set_ylabel('current (µA)')
ax.set_title('Synaptic + bias current — does Is2+I_0 stay above I_min?',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)

# (3) memristor state
ax = axes[3]
ax.plot(t_ms, s_v, color='C4', lw=0.6, drawstyle='steps-post')
ax.axvline(t_pulse/ms, color='red', ls=':', lw=1.4)
ax.set_ylabel('s'); ax.set_xlabel('t (ms)')
ax.set_title('Memristor state (0=open, 1=closed)', fontweight='bold')
ax.set_ylim(-0.1, 1.1)

fig.suptitle(
    'MSN single-neuron bump test (msn_lib demo)  —  pulse triggers self-excit. burst',
    fontsize=12, fontweight='bold', y=1.005)
plt.show()
out_path = '/home/haoran/Projects/Brain2simulator/ns_msn_v3_bump.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")

# %%
