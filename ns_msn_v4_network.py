"""
ns_msn_v4_network.py
====================
Small recurrent network of 20 MSN neurons — proof that the modular code
(msn_lib.py) scales beyond a single cell.

Inheritance:
  msn_lib.py             — MSNParams, make_msn, make_synapse
  ns_msn_v3_bump.py      — single neuron + self-loop  (passed → scale up)
  THIS FILE              — N=20 ring with local recurrent excitation

─── Topology ────────────────────────────────────────────────────────────────
  N = 20 MSN neurons indexed 0..19 in a 1D ring (periodic boundary).
  Local recurrent excitation: each neuron i projects to i±1 and i±2
  (modulo N, no self-loop).  Total recurrent edges = 4·N = 80.

─── Stimulation ─────────────────────────────────────────────────────────────
  All neurons share a subthreshold tonic bias I_0 = 0.85·I_min.
  At t = t_pulse we deliver a single input spike to a 2-neuron "cue"
  cluster (neurons 9, 10) via a SpikeGeneratorGroup + make_synapse.

─── What we expect ─────────────────────────────────────────────────────────
  - Pre-cue: silence (subthreshold).
  - Cue lands → neurons 9, 10 fire.
  - Their spikes feed Is1_exc of neurons 7..12 via the ring.
  - Activity spreads to a small neighbourhood (a "bump").
  - As recurrent current decays (τ_s=500ms), the bump dissipates.

  Each post-syn neuron now has up to 4 incoming recurrent edges, so the
  per-edge weight is scaled DOWN (~1/4 of the single-neuron self-loop
  case) to land in the same operating regime.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *
from msn_lib import MSNParams, make_msn, make_synapse

seed(42)
defaultclock.dt = 10*us

# ─── Parameters ──────────────────────────────────────────────────────────────
params = MSNParams(tau_s1=500e-3, tau_s2=500e-3)
print(params.summary())
I_min, I_max = params.operating_window()

N            = 20
I_0_val      = 0.85 * I_min        # subthreshold
Iw_input_val = 30e-6                # cue weight (same as v3)
Iw_recur_val = 2.0e-6               # per recurrent edge (scaled ~1/4 of v3)
cue_idx      = [9, 10]              # centre of the ring
t_pulse      = 2.5 * second
T_run        = 8.0 * second

# ─── Build the population ────────────────────────────────────────────────────
neurons = make_msn(N=N, params=params, name='ring')
neurons.I_0 = I_0_val * amp

# Local recurrent excitation on a periodic ring: i → i±1, i±2  (mod N).
# Connection condition: i != j AND wrapped distance ∈ {1, 2}.
ring_cond = (
    f'i != j and '
    f'(abs(i-j) == 1 or abs(i-j) == 2 or '
    f' abs(i-j) == {N-1} or abs(i-j) == {N-2})'
)
syn_recur = make_synapse(
    source=neurons, target=neurons,
    kind='exc', weight=Iw_recur_val,
    connect=ring_cond, name='ring_recur',
)
print(f"Recurrent edges: {len(syn_recur)}  (expected 4·N = {4*N})")

# ─── External input cue: 1 spike to neurons in `cue_idx` ─────────────────────
gen = SpikeGeneratorGroup(
    N,
    indices=np.array(cue_idx),
    times=np.array([float(t_pulse/second)] * len(cue_idx)) * second,
    name='cue_gen',
)
syn_input = make_synapse(
    source=gen, target=neurons,
    kind='exc', weight=Iw_input_val,
    connect='i == j', name='syn_input',
)

# ─── Monitors ────────────────────────────────────────────────────────────────
sp_mon = SpikeMonitor(neurons)
st_mon = StateMonitor(neurons, ['Vm', 'Vout', 'Is2_exc'],
                      record=True, dt=2*ms)

print(f"\nExperiment:")
print(f"  N            = {N} neurons")
print(f"  I_0          = {I_0_val*1e6:.2f} µA  ({I_0_val/I_min*100:.0f}% rheobase)")
print(f"  Iw_input     = {Iw_input_val*1e6:.0f} µA  to neurons {cue_idx}")
print(f"  Iw_recur     = {Iw_recur_val*1e6:.1f} µA  per edge (4 in-degree)")
print(f"  τ_s          = {params.tau_s1*1e3:.0f} ms")
print(f"  t_pulse      = {t_pulse/second:.1f} s,   T_run = {T_run/second:.1f} s")
print()

run(T_run, report='text')

# ─── Spike statistics ────────────────────────────────────────────────────────
spike_t = np.array(sp_mon.t / second)
spike_i = np.array(sp_mon.i)

print(f"\n  Total spikes: {len(spike_t)}")
post_pulse = spike_t > float(t_pulse/second)
print(f"  Spikes after cue: {post_pulse.sum()}")
unique_active = np.unique(spike_i[post_pulse])
print(f"  Active neurons (post-cue): {sorted(unique_active.tolist())}")
if post_pulse.any():
    last_spike = spike_t[post_pulse].max()
    print(f"  Last spike: t = {last_spike:.2f} s   "
          f"(bump duration ~ {(last_spike - float(t_pulse/second)):.2f} s)")

# ─── Plot ────────────────────────────────────────────────────────────────────
t_s = np.array(st_mon.t / second)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.40,
                        height_ratios=[1.0, 1.4, 1.0])

# (0) Raster — spikes across the ring vs time
ax = fig.add_subplot(gs[0, 0])
ax.scatter(spike_t, spike_i, s=18, c='C3', marker='|', linewidths=1.4)
ax.axvline(float(t_pulse/second), color='red', ls=':', lw=1.2, label='cue pulse')
for ci in cue_idx:
    ax.axhline(ci, color='gold', ls='-', lw=0.5, alpha=0.4)
ax.set_xlim(0, float(T_run/second))
ax.set_ylim(-0.5, N-0.5)
ax.set_xlabel('t (s)'); ax.set_ylabel('neuron index')
ax.set_title(
    f'Spike raster — {N} MSN ring, local exc (±1, ±2), Iw_recur={Iw_recur_val*1e6:.1f} µA  '
    f'→ {post_pulse.sum()} post-cue spikes across {len(unique_active)} neurons',
    fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)

# (1) Heatmap of Is2_exc over time × neuron index
ax = fig.add_subplot(gs[1, 0])
Is2 = np.array(st_mon.Is2_exc / uA)        # shape (N, T)
im = ax.imshow(Is2, aspect='auto', origin='lower',
               extent=[0, float(T_run/second), -0.5, N-0.5],
               cmap='magma', interpolation='nearest')
ax.axvline(float(t_pulse/second), color='cyan', ls=':', lw=1.2)
for ci in cue_idx:
    ax.axhline(ci, color='cyan', ls='-', lw=0.5, alpha=0.5)
ax.set_xlabel('t (s)'); ax.set_ylabel('neuron index')
ax.set_title('Is2_exc heatmap — recurrent current spreading through the ring',
             fontsize=11, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label('Is2_exc (µA)', fontsize=9)

# (2) Population spike count per 100 ms bin
ax = fig.add_subplot(gs[2, 0])
bin_edges = np.arange(0, float(T_run/second) + 0.1, 0.1)
counts, _ = np.histogram(spike_t, bins=bin_edges)
bin_centres = 0.5*(bin_edges[:-1] + bin_edges[1:])
ax.bar(bin_centres, counts, width=0.09, color='C0', alpha=0.85)
ax.axvline(float(t_pulse/second), color='red', ls=':', lw=1.2, label='cue pulse')
ax.set_xlim(0, float(T_run/second))
ax.set_xlabel('t (s)'); ax.set_ylabel('spikes / 100 ms')
ax.set_title('Population spike count', fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)

fig.suptitle(
    f'MSN recurrent ring  ({N} neurons, msn_lib demo)  —  '
    f'cued bump propagates locally and dissipates',
    fontsize=12, fontweight='bold', y=1.005)

out_path = '/home/haoran/Projects/Brain2simulator/ns_msn_v4_network.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
