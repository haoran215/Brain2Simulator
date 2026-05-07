"""
ns_msn_wta_demo.py
==================
Two-neuron Winner-Take-All (WTA) demonstration.

Circuit
───────
        ┌──────────────────────────────────────┐
        │          mutual inhibition           │
        │    ┌─────────────┐                   │
   D1 ──┤ N1 │─────────────────────── inh ──► N2 ├── D2(t)
        │    └─────────────┘                   │
        │    ◄─── inh ─────────────────────────┤
        └──────────────────────────────────────┘

Each neuron: MSN (Memristive Spiking Neuron), series Rm + Ra topology.
Each inhibitory edge adds Iw_inh to Is1_inh of the opponent.

WTA phases
──────────
  Phase 1  t = 0 → 3 s   N1 driven above rheobase (D1 = 1.3·I_min).
                           N2 subthreshold (D2 = 0).
                           N1 fires, builds Is2_inh on N2.

  Phase 2  t = 3 → 7 s   N2 receives strong drive (D2 = 3.3·I_min).
                           N2 fires faster than N1, rapidly builds
                           inhibition on N1.  N1 suppressed within ~1 s.
                           N2 wins.

  Phase 3  t = 7 → 10 s  N2 drive removed.  Is2_inh on N1 decays (τ_s).
                           N1 recovers and wins again.

External input types shown
──────────────────────────
  (a) Constant tonic bias  — G.I_0 = value * amp
  (b) Step-current pulse   — network_operation changes I_0 at runtime
  (c) Poisson background   — PoissonGroup + make_synapse (commented out,
                              shown at the bottom as a pattern reference)

Inheritance
───────────
  msn_neuron.py    — MSNParams, make_msn
  msn_synapse.py   — SynapseParams, make_synapse
  configs/         — JSON parameter files
"""
#%%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *

from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 0. Backend + clock                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
prefs.codegen.target = 'numpy'   # remove once build-essential is installed
start_scope()
defaultclock.dt = 10 * us


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. Load parameters from JSON                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ── Neurons ──────────────────────────────────────────────────────────────────
# Both neurons share the same hardware spec (same JSON file).
# Different populations would load different files.
neuron_params = MSNParams.from_json('configs/neuron_default.json')
print(neuron_params.summary())

I_min, I_max = neuron_params.operating_window()
print(f"\n  I_min (rheobase)     = {I_min*1e6:.3f} µA")
print(f"  I_max (depol block)  = {I_max*1e6:.0f} µA\n")

# ── Synapses ─────────────────────────────────────────────────────────────────
inh_params = SynapseParams.from_json('configs/synapse_default.json', key='inh')
exc_params  = SynapseParams.from_json('configs/synapse_default.json', key='exc')
print(inh_params.summary())
print(exc_params.summary(), '\n')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. Build neuron populations                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Separate NeuronGroups so each has a readable name.
# For a large homogeneous population use make_msn(N=100, ...).
N1 = make_msn(N=1, params=neuron_params, name='N1')
N2 = make_msn(N=1, params=neuron_params, name='N2')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. External inputs                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── (a) Constant tonic bias I_0 ─────────────────────────────────────────────
# Set once; persists for the whole run unless overridden by (b) or (c).
# Regime reference:
#   I_0 < I_min          → silent (needs synaptic push)
#   I_min < I_0 < I_max  → spontaneously firing
#   I_0 > I_max          → depolarisation block (avoid)

D1 = 1.3 * I_min          # 19.5 µA — N1 fires spontaneously at ~7 Hz
D2_off = 0.0              # µA — N2 silent on its own
D2_on  = 3.3 * I_min      # 49.5 µA — N2 fires fast when switched on

N1.I_0 = D1      * amp    # constant; never changes
N2.I_0 = D2_off  * amp    # starts silent; boosted by (b) below

# ─── (b) Step-current pulse via network_operation ────────────────────────────
# A network_operation runs every timestep and can read/write neuron state.
# Here it implements a rectangular drive pulse on N2.
t_on  = 3.0 * second
t_off = 7.0 * second

@network_operation(when='start')
def wta_pulse(t):
    """Switch N2's drive on at t_on, off at t_off."""
    if t_on <= t < t_off:
        N2.I_0[0] = D2_on  * amp
    else:
        N2.I_0[0] = D2_off * amp

# ─── (c) Poisson background input — PATTERN REFERENCE ────────────────────────
# Uncomment this block to add stochastic excitatory background to N1.
# Each Poisson spike adds exc_params.weight to N1's Is1_exc.
#
# poisson_N1 = PoissonGroup(1, rates=50 * Hz)
# syn_poisson_N1 = make_synapse(
#     source  = poisson_N1,
#     target  = N1,
#     params  = exc_params,
#     connect = 'i == j',         # neuron 0 of Poisson → neuron 0 of N1
#     name    = 'syn_pois_N1',
# )
# Steady-state extra drive: <Is2_exc> ≈ Iw * rate * tau_s2
#   = 6e-6 * 50 * 0.2 = 60 µA  (would push N1 well above threshold)
# Tune `rates` or `weight` in synapse_exc.json to get the desired mean.


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. Recurrent synapses — WTA mutual inhibition                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# N1 → N2: inhibitory.  connect=True because src ≠ tgt (all-to-all = one edge).
# N2 → N1: inhibitory.  Symmetric weights here; asymmetric is easy:
#   syn_12.w = 15e-6 * amp   # after creation

syn_N1_to_N2 = make_synapse(
    source  = N1,
    target  = N2,
    params  = inh_params,
    connect = True,           # one source neuron, one target neuron
    name    = 'syn_N1_N2',
)

syn_N2_to_N1 = make_synapse(
    source  = N2,
    target  = N1,
    params  = inh_params,
    connect = True,
    name    = 'syn_N2_N1',
)

# Per-edge weight is addressable after construction — useful for plasticity:
#   syn_N1_to_N2.w = new_value * amp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. Monitors                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# StateMonitor dt=1ms keeps memory manageable for a 10 s run.
rec_vars = ['Vm', 'Vout', 'Is2_exc', 'Is2_inh', 'I_0']

st_N1 = StateMonitor(N1, rec_vars, record=True, dt=1*ms)
st_N2 = StateMonitor(N2, rec_vars, record=True, dt=1*ms)

sp_N1 = SpikeMonitor(N1)
sp_N2 = SpikeMonitor(N2)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 6. Run                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
T_run = 10.0 * second

print("─" * 60)
print(f"Running {T_run/second:.0f} s  (dt={defaultclock.dt/us:.0f} µs = "
      f"{int(T_run / defaultclock.dt):,} steps) …")
print("─" * 60)

run(T_run, report='text')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 7. Summary                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
t_on_ms  = float(t_on  / ms)
t_off_ms = float(t_off / ms)

def phase_counts(sp_mon, label):
    t_ms = np.array(sp_mon.t / ms)
    pre   = t_ms[t_ms < t_on_ms]
    pulse = t_ms[(t_ms >= t_on_ms) & (t_ms < t_off_ms)]
    post  = t_ms[t_ms >= t_off_ms]
    print(f"  {label:3s}  total={len(t_ms):4d}  "
          f"phase1={len(pre):3d}  phase2={len(pulse):3d}  phase3={len(post):3d}")

print("\nSpike counts by phase:")
print(f"  {'':3s}  {'total':>5}  {'ph1(N1 wins)':>12}  "
      f"{'ph2(N2 wins)':>12}  {'ph3(N1 rec.)':>12}")
phase_counts(sp_N1, 'N1')
phase_counts(sp_N2, 'N2')
print()

# Estimate mean firing rates in each phase
def mean_rate_hz(sp_mon, t_start_ms, t_end_ms):
    t_ms = np.array(sp_mon.t / ms)
    n = np.sum((t_ms >= t_start_ms) & (t_ms < t_end_ms))
    dur = (t_end_ms - t_start_ms) / 1000.0
    return n / dur if dur > 0 else 0.0

print("Mean rates (Hz) by phase:")
print(f"  {'':3s}  {'ph1 [0→3s]':>12}  {'ph2 [3→7s]':>12}  {'ph3 [7→10s]':>12}")
for label, sp in [('N1', sp_N1), ('N2', sp_N2)]:
    r1 = mean_rate_hz(sp, 0,       t_on_ms)
    r2 = mean_rate_hz(sp, t_on_ms, t_off_ms)
    r3 = mean_rate_hz(sp, t_off_ms, T_run/ms)
    print(f"  {label:3s}  {r1:>12.1f}  {r2:>12.1f}  {r3:>12.1f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 8. Plot                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
t_ms   = np.array(st_N1.t / ms)

Vm1    = np.array(st_N1.Vm[0]    / volt)
Vm2    = np.array(st_N2.Vm[0]    / volt)
Vout1  = np.array(st_N1.Vout[0]  / volt) * 1e3   # mV
Vout2  = np.array(st_N2.Vout[0]  / volt) * 1e3
Iinh1  = np.array(st_N1.Is2_inh[0] / amp) * 1e6  # µA  (inhibition ON N1)
Iinh2  = np.array(st_N2.Is2_inh[0] / amp) * 1e6  # µA  (inhibition ON N2)
I0_1   = np.array(st_N1.I_0[0]   / amp) * 1e6
I0_2   = np.array(st_N2.I_0[0]   / amp) * 1e6

sp1_ms = np.array(sp_N1.t / ms)
sp2_ms = np.array(sp_N2.t / ms)

C1 = '#2980B9'    # N1 blue
C2 = '#E74C3C'    # N2 red

fig = plt.figure(figsize=(16, 16))
gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.50)

def shade_phases(ax):
    """Shade the three WTA phases."""
    ax.axvspan(0,         t_on_ms,  alpha=0.06, color='steelblue', label='Phase 1: N1 wins')
    ax.axvspan(t_on_ms,   t_off_ms, alpha=0.06, color='tomato',    label='Phase 2: N2 wins')
    ax.axvspan(t_off_ms,  T_run/ms, alpha=0.06, color='steelblue')


# ─── Panel 0: Input drives ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.plot(t_ms, I0_1, color=C1, lw=1.5, label='N1  I_0 (constant)')
ax.plot(t_ms, I0_2, color=C2, lw=1.5, label='N2  I_0 (step pulse)')
ax.axhline(I_min*1e6, color='k', ls='--', lw=1,
           label=f'I_min = {I_min*1e6:.1f} µA (rheobase)')
shade_phases(ax)
ax.set_ylabel('Drive I_0 (µA)')
ax.set_title('External drives — (a) constant I_0  +  (b) step pulse via network_operation',
             fontweight='bold')
ax.legend(fontsize=8, loc='center right', ncol=2)


# ─── Panel 1: Membrane voltage ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1])
ax.plot(t_ms, Vm1, color=C1, lw=0.7, alpha=0.9, label='N1  Vm')
ax.plot(t_ms, Vm2, color=C2, lw=0.7, alpha=0.9, label='N2  Vm')
ax.axhline(neuron_params.Vth, color='k', ls='--', lw=1,
           label=f'Vth = {neuron_params.Vth:.2f} V')
shade_phases(ax)
for ts in sp1_ms:
    ax.vlines(ts, neuron_params.Vth, neuron_params.Vth + 0.07, colors=C1, lw=0.6)
for ts in sp2_ms:
    ax.vlines(ts, neuron_params.Vth, neuron_params.Vth + 0.07, colors=C2, lw=0.6)
ax.set_ylabel('Vm (V)')
ax.set_title('Membrane voltage — spikes shown as tick marks at threshold',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper left', ncol=3)


# ─── Panel 2: Output spike trains (Vout) ────────────────────────────────────
ax = fig.add_subplot(gs[2])
ax.plot(t_ms, Vout1, color=C1, lw=0.7, alpha=0.9, label='N1  Vout')
ax.plot(t_ms, Vout2, color=C2, lw=0.7, alpha=0.9, label='N2  Vout')
shade_phases(ax)
ax.set_ylabel('Vout (mV)')
ax.set_title('Output voltage — real spike waveform from Cm discharge through Rm_lo + Ra',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper left')


# ─── Panel 3: Inhibitory currents (Is2_inh on each neuron) ──────────────────
ax = fig.add_subplot(gs[3])
ax.plot(t_ms, Iinh1, color=C1, lw=1.2,
        label='Is2_inh on N1  (from N2 spikes)')
ax.plot(t_ms, Iinh2, color=C2, lw=1.2,
        label='Is2_inh on N2  (from N1 spikes)')
ax.axhline(0, color='k', lw=0.5)
# Annotate the suppression threshold for N1
ax.axhline(D1 * 1e6, color=C1, ls=':', lw=1,
           label=f'N1 drive D1 = {D1*1e6:.1f} µA — inh above this silences N1')
ax.axhline(D2_on * 1e6, color=C2, ls=':', lw=1,
           label=f'N2 drive D2 = {D2_on*1e6:.1f} µA — inh above this silences N2')
shade_phases(ax)
ax.set_ylabel('Is2_inh (µA)')
ax.set_title('Inhibitory synaptic current — cascade build-up shows WTA competition',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper left', ncol=2)


# ─── Panel 4: Spike rasters ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[4])
ax.scatter(sp1_ms, np.ones_like(sp1_ms) * 1, color=C1,
           s=10, marker='|', linewidths=1.0, label='N1')
ax.scatter(sp2_ms, np.ones_like(sp2_ms) * 0, color=C2,
           s=10, marker='|', linewidths=1.0, label='N2')
shade_phases(ax)
ax.set_yticks([0, 1])
ax.set_yticklabels(['N2', 'N1'])
ax.set_xlabel('t (ms)')
ax.set_ylabel('Neuron')
ax.set_title(f'Spike raster — N1: {len(sp1_ms)} spikes,  N2: {len(sp2_ms)} spikes',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right')

for ax_ in fig.get_axes():
    ax_.set_xlim(0, T_run / ms)
    ax_.grid(axis='x', alpha=0.20)

fig.suptitle(
    'Two-neuron WTA (mutual inhibition)\n'
    'Phase 1: N1 active, suppresses N2  |  '
    'Phase 2: N2 stronger drive, wins  |  '
    'Phase 3: N2 off, N1 recovers',
    fontsize=12, fontweight='bold', y=1.002)

out_path = 'ns_msn_wta_demo.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Appendix: Poisson input pattern  (not executed — reference only)       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# To replace the step-current pulse with Poisson-driven excitation:
#
#   from brian2 import PoissonGroup, Hz
#
#   bg_N2 = PoissonGroup(1, rates=200 * Hz)
#   syn_bg_N2 = make_synapse(
#       source  = bg_N2,
#       target  = N2,
#       params  = exc_params,           # SynapseParams.from_json('configs/synapse_default.json', key='exc')
#       connect = 'i == j',
#       name    = 'syn_bg_N2',
#   )
#   # Steady-state extra drive: <Is2_exc> ≈ weight * rate * tau_s2
#   #   = 6e-6 A * 200 Hz * 0.2 s = 240 µA  (very strong — reduce weight or rate)
#   # Tune either exc_params.weight or the Poisson rate until the desired
#   # mean drive lands in (I_min, I_max).
#
# To add background noise to BOTH neurons simultaneously:
#
#   bg_all = PoissonGroup(2, rates=50 * Hz)
#   syn_bg = make_synapse(
#       source  = bg_all,
#       target  = N1,         # could be a combined NeuronGroup instead
#       params  = exc_params,
#       connect = 'i == j',   # Poisson neuron 0 → pop neuron 0, etc.
#       name    = 'syn_bg_all',
#   )
# %%