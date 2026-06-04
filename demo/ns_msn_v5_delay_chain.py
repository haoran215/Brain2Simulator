"""
ns_msn_v5_delay_chain.py
========================
Four-neuron feedforward delay chain with self-terminating bursts.

Topology
────────
Poisson ──(exc, alpha)──► N1 ──(exc, alpha)──► N2 ──(exc, alpha)──► N3
                                                                      │
                                                       (exc, alpha) ──┤──► N4
                                                       (inh, alpha) ──┘

Every neuron has:
  • fast self-excitatory alpha loop   (sustains burst once started)
  • slow self-inhibitory alpha loop   (terminates burst — the key new feature)

Burst mechanism per neuron
──────────────────────────
  1. External drive (Poisson / chain) pushes Vm above Vth → burst begins.
  2. Fast self-exc sustains firing while slow self-inh Is2 builds up.
  3. When Is2_inh reaches the threshold gap (I_drive − I_gt), firing stops.
  4. Is2_inh decays (τ=100 ms) while Is2_exc decays fast (τ=5 ms).
  5. After ~100 ms silence the neuron is available for the next burst.

Chain delay
───────────
τ_chain = 10 ms → each stage fires ~8 ms after the previous one starts,
well within the ~160 ms burst window.  N4 fires ~24 ms after N1.
Multiple burst cycles are visible in the 1.2 s simulation.

Plots
─────
Fig 1: Raster (all 4) + Vm traces
Fig 2: Synaptic step-1 (Is1) and step-2 (Is2) per inlet, for all neurons
       — especially the self-inh Is1 vs Is2 lag, and N4 net current
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from brian2 import *
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

seed(42)
prefs.codegen.target = 'numpy'
defaultclock.dt = 10 * us

# ── Hardware parameters ───────────────────────────────────────────────────────

p = MSNParams()
I_GT = p.I_gt          # ≈ 32 µA  (rheobase ≈ gate trigger current)
print(p.summary())
print(f"I_gt = {I_GT*1e6:.2f} µA")

T_SIM = 1200e-3        # s   (long enough to see 4-5 burst cycles)

# Tonic bias
I_0_N1   = 0.95 * I_GT   # N1: just below threshold; Poisson + self-exc push it over
I_0_REST = 0.90 * I_GT   # N2, N3, N4: close to threshold; small chain input fires them

# Poisson drive → N1 (moderate background — contributes ~12 µA steady-state)
RATE_POISSON = 500        # Hz
W_POISSON    = 5e-6       # A
TAU_POISSON  = 5e-3       # s

# Chain links N1→N2, N2→N3  (tau_chain=10 ms → ~8 ms inter-stage onset delay)
W_CHAIN   = 15e-6         # A
TAU_CHAIN = 10e-3         # s

# N3 → N4 (exc + inh, both alpha; exc weight > inh → net excitatory)
W_EXC_N4  = 20e-6         # A
W_INH_N4  = 8e-6          # A   (<  W_EXC_N4)
TAU_N3_N4 = 10e-3         # s

# Fast self-excitation (sustains burst once started)
W_SELF_EXC   = 5e-6       # A
TAU_SELF_EXC = 5e-3       # s

# Slow self-inhibition (terminates burst — key new feature)
# Is2_ss = W * f * tau.  At f≈116 Hz, Is2_ss ≈ 23 µA > threshold gap ≈ 11 µA
# → burst lasts ~160 ms, silence ~100 ms, period ~260 ms
W_SELF_INH   = 2e-6       # A
TAU_SELF_INH = 100e-3     # s   (10× slower than self-exc → clear burst shape)

# Recording dt (0.5 ms — enough to resolve synaptic dynamics)
REC_DT = 0.5e-3           # s

# ── Network construction ──────────────────────────────────────────────────────

start_scope()

N1 = make_msn(1, params=p,
              exc_inlets=('I_exc_poisson', 'I_exc_self'),
              inh_inlets=('I_inh_self',),
              name='N1')
N2 = make_msn(1, params=p,
              exc_inlets=('I_exc_n1', 'I_exc_self'),
              inh_inlets=('I_inh_self',),
              name='N2')
N3 = make_msn(1, params=p,
              exc_inlets=('I_exc_n2', 'I_exc_self'),
              inh_inlets=('I_inh_self',),
              name='N3')
# N4 receives exc AND inh from N3, plus its own slow self-inh
N4 = make_msn(1, params=p,
              exc_inlets=('I_exc_n3', 'I_exc_self'),
              inh_inlets=('I_inh_n3', 'I_inh_self'),
              name='N4')

N1.I_0 = I_0_N1   * amp
N2.I_0 = I_0_REST * amp
N3.I_0 = I_0_REST * amp
N4.I_0 = I_0_REST * amp

# Poisson source
poisson_src = PoissonGroup(1, rates=RATE_POISSON * Hz)

# ── Synapses ──────────────────────────────────────────────────────────────────

def alpha(kind, w, tau, target_var, name):
    src_map = {
        'syn_poisson_n1':  (poisson_src, N1),
        'syn_n1_self_exc': (N1, N1),
        'syn_n1_self_inh': (N1, N1),
        'syn_n1_n2':       (N1, N2),
        'syn_n2_self_exc': (N2, N2),
        'syn_n2_self_inh': (N2, N2),
        'syn_n2_n3':       (N2, N3),
        'syn_n3_self_exc': (N3, N3),
        'syn_n3_self_inh': (N3, N3),
        'syn_n3_n4_exc':   (N3, N4),
        'syn_n3_n4_inh':   (N3, N4),
        'syn_n4_self_exc': (N4, N4),
        'syn_n4_self_inh': (N4, N4),
    }
    src, tgt = src_map[name]
    conn = True if src is poisson_src else 'i == j'
    return make_synapse(src, tgt,
                        params=SynapseParams(kind=kind, weight=w,
                                             tau_s1=tau, tau_s2=tau,
                                             cascade='alpha',
                                             target_var=target_var),
                        connect=conn, name=name)

# Chain + cross-neuron synapses
syn_poisson_n1 = alpha('exc', W_POISSON,    TAU_POISSON,    'I_exc_poisson', 'syn_poisson_n1')
syn_n1_n2      = alpha('exc', W_CHAIN,      TAU_CHAIN,      'I_exc_n1',      'syn_n1_n2')
syn_n2_n3      = alpha('exc', W_CHAIN,      TAU_CHAIN,      'I_exc_n2',      'syn_n2_n3')
syn_n3_n4_exc  = alpha('exc', W_EXC_N4,    TAU_N3_N4,      'I_exc_n3',      'syn_n3_n4_exc')
syn_n3_n4_inh  = alpha('inh', W_INH_N4,    TAU_N3_N4,      'I_inh_n3',      'syn_n3_n4_inh')

# Fast self-excitation (sustains burst)
syn_n1_self_exc = alpha('exc', W_SELF_EXC, TAU_SELF_EXC,   'I_exc_self',    'syn_n1_self_exc')
syn_n2_self_exc = alpha('exc', W_SELF_EXC, TAU_SELF_EXC,   'I_exc_self',    'syn_n2_self_exc')
syn_n3_self_exc = alpha('exc', W_SELF_EXC, TAU_SELF_EXC,   'I_exc_self',    'syn_n3_self_exc')
syn_n4_self_exc = alpha('exc', W_SELF_EXC, TAU_SELF_EXC,   'I_exc_self',    'syn_n4_self_exc')

# Slow self-inhibition (terminates burst)
syn_n1_self_inh = alpha('inh', W_SELF_INH, TAU_SELF_INH,   'I_inh_self',    'syn_n1_self_inh')
syn_n2_self_inh = alpha('inh', W_SELF_INH, TAU_SELF_INH,   'I_inh_self',    'syn_n2_self_inh')
syn_n3_self_inh = alpha('inh', W_SELF_INH, TAU_SELF_INH,   'I_inh_self',    'syn_n3_self_inh')
syn_n4_self_inh = alpha('inh', W_SELF_INH, TAU_SELF_INH,   'I_inh_self',    'syn_n4_self_inh')

# ── Monitors ──────────────────────────────────────────────────────────────────

spk_N1 = SpikeMonitor(N1)
spk_N2 = SpikeMonitor(N2)
spk_N3 = SpikeMonitor(N3)
spk_N4 = SpikeMonitor(N4)

vm_N1  = StateMonitor(N1, 'Vm', record=True, dt=REC_DT*second)
vm_N2  = StateMonitor(N2, 'Vm', record=True, dt=REC_DT*second)
vm_N3  = StateMonitor(N3, 'Vm', record=True, dt=REC_DT*second)
vm_N4  = StateMonitor(N4, 'Vm', record=True, dt=REC_DT*second)

# Synaptic cascade state (Is1 = step 1, Is2 = step 2) on every synapse
sm_poisson      = StateMonitor(syn_poisson_n1,  ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n1_n2        = StateMonitor(syn_n1_n2,       ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n2_n3        = StateMonitor(syn_n2_n3,       ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n3_exc       = StateMonitor(syn_n3_n4_exc,   ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n3_inh       = StateMonitor(syn_n3_n4_inh,   ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n1_self_exc  = StateMonitor(syn_n1_self_exc, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n2_self_exc  = StateMonitor(syn_n2_self_exc, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n3_self_exc  = StateMonitor(syn_n3_self_exc, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n4_self_exc  = StateMonitor(syn_n4_self_exc, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n1_self_inh  = StateMonitor(syn_n1_self_inh, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n2_self_inh  = StateMonitor(syn_n2_self_inh, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n3_self_inh  = StateMonitor(syn_n3_self_inh, ['Is1', 'Is2'], record=True, dt=REC_DT*second)
sm_n4_self_inh  = StateMonitor(syn_n4_self_inh, ['Is1', 'Is2'], record=True, dt=REC_DT*second)

# ── Run ───────────────────────────────────────────────────────────────────────

net = Network(collect())
net.run(T_SIM * second, report='text', report_period=0.1*second)

print(f"\nSpike counts — N1:{spk_N1.num_spikes}  N2:{spk_N2.num_spikes}"
      f"  N3:{spk_N3.num_spikes}  N4:{spk_N4.num_spikes}")

# ── Plotting helpers ──────────────────────────────────────────────────────────

COLORS = {'N1': '#e74c3c', 'N2': '#3498db', 'N3': '#27ae60', 'N4': '#8e44ad'}
T_MS   = T_SIM * 1e3   # simulation end in ms

def is12(ax, mon, label, color, ls_is2='--', scale=1e6):
    """Plot Is1 (solid) and Is2 (dashed) from a synapse StateMonitor."""
    t = mon.t / ms
    ax.plot(t, mon.Is1[0] * scale, color=color,       lw=1.0, label=f'{label} Is1')
    ax.plot(t, mon.Is2[0] * scale, color=color, ls=ls_is2, lw=1.0, label=f'{label} Is2')
    ax.set_ylabel('Is (µA)')
    ax.set_xlim([0, T_MS])
    ax.legend(fontsize=7, loc='upper right')

# ── Figure 1: Raster + Vm ─────────────────────────────────────────────────────

fig1, axes = plt.subplots(5, 1, figsize=(12, 11),
                           gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
fig1.suptitle('Delay chain: Poisson → N1 → N2 → N3 → N4', fontsize=13)

# Raster
ax = axes[0]
spk_data = [('N1', spk_N1), ('N2', spk_N2), ('N3', spk_N3), ('N4', spk_N4)]
for i, (name, sm) in enumerate(spk_data):
    t_sp = sm.t / ms
    ax.scatter(t_sp, np.full_like(t_sp, i), c=COLORS[name], s=5, label=name)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['N1', 'N2', 'N3', 'N4'])
ax.set_xlim([0, T_MS])
ax.set_ylabel('Neuron')
ax.set_title('Spike raster')
ax.legend(loc='upper right', markerscale=3, fontsize=8)

# Vm traces
for ax, (name, vm, color) in zip(axes[1:],
        [('N1', vm_N1, COLORS['N1']), ('N2', vm_N2, COLORS['N2']),
         ('N3', vm_N3, COLORS['N3']), ('N4', vm_N4, COLORS['N4'])]):
    ax.plot(vm.t / ms, vm.Vm[0], color=color, lw=0.7)
    ax.axhline(p.Vth, color='k', ls=':', lw=0.8, alpha=0.6)
    ax.set_xlim([0, T_MS])
    ax.set_ylabel('Vm (V)')
    ax.set_title(f'{name} membrane voltage  (dotted = Vth = {p.Vth:.1f} V)')

axes[-1].set_xlabel('time (ms)')
fig1.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), 'ns_msn_v5_raster_vm.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved: {out1}")

# ── Figure 2: Synaptic cascade activities ─────────────────────────────────────
# Row 0: chain inputs    — Poisson→N1, N1→N2, N2→N3, N3→N4(exc+inh)
# Row 1: fast self-exc   — sustains burst (τ=5 ms)
# Row 2: slow self-inh   — terminates burst (τ=100 ms)  ← key new row
# Row 3: N4 net current  — exc − inh_n3 − self_inh + self_exc vs threshold

fig2 = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(4, 4, figure=fig2, hspace=0.6, wspace=0.35)

# ── Row 0: chain inputs ───────────────────────────────────────────────────────
ax = fig2.add_subplot(gs[0, 0])
is12(ax, sm_poisson, 'Poisson→N1', '#e67e22')
ax.set_title(f'N1 input: Poisson (τ={TAU_POISSON*1e3:.0f} ms)')

ax = fig2.add_subplot(gs[0, 1])
is12(ax, sm_n1_n2, 'N1→N2', COLORS['N1'])
ax.set_title(f'N2 input: N1→N2 (τ={TAU_CHAIN*1e3:.0f} ms)')

ax = fig2.add_subplot(gs[0, 2])
is12(ax, sm_n2_n3, 'N2→N3', COLORS['N2'])
ax.set_title(f'N3 input: N2→N3 (τ={TAU_CHAIN*1e3:.0f} ms)')

ax = fig2.add_subplot(gs[0, 3])
is12(ax, sm_n3_exc, f'exc w={W_EXC_N4*1e6:.0f}µA', COLORS['N3'])
is12(ax, sm_n3_inh, f'inh w={W_INH_N4*1e6:.0f}µA', '#c0392b', ls_is2=':')
ax.set_title(f'N4 input from N3 (τ={TAU_N3_N4*1e3:.0f} ms)')

# ── Row 1: fast self-excitation per neuron ────────────────────────────────────
for col, (name, sm) in enumerate([('N1', sm_n1_self_exc), ('N2', sm_n2_self_exc),
                                   ('N3', sm_n3_self_exc), ('N4', sm_n4_self_exc)]):
    ax = fig2.add_subplot(gs[1, col])
    is12(ax, sm, f'{name} self-exc', COLORS[name])
    ax.set_title(f'{name} self-exc  (τ={TAU_SELF_EXC*1e3:.0f} ms, fast — sustains burst)')

# ── Row 2: slow self-inhibition per neuron ────────────────────────────────────
for col, (name, sm) in enumerate([('N1', sm_n1_self_inh), ('N2', sm_n2_self_inh),
                                   ('N3', sm_n3_self_inh), ('N4', sm_n4_self_inh)]):
    ax = fig2.add_subplot(gs[2, col])
    is12(ax, sm, f'{name} self-inh', COLORS[name])
    ax.set_title(f'{name} self-inh  (τ={TAU_SELF_INH*1e3:.0f} ms, slow — kills burst)')

# ── Row 3: N4 net current ─────────────────────────────────────────────────────
ax_net = fig2.add_subplot(gs[3, :])
t = sm_n3_exc.t / ms
is2_exc      =  sm_n3_exc.Is2[0]      * 1e6
is2_inh_n3   =  sm_n3_inh.Is2[0]      * 1e6
is2_self_exc =  sm_n4_self_exc.Is2[0] * 1e6
is2_self_inh =  sm_n4_self_inh.Is2[0] * 1e6
net = is2_exc - is2_inh_n3 + is2_self_exc - is2_self_inh

ax_net.plot(t, is2_exc,      color=COLORS['N3'], lw=1.0, label='exc Is2 (N3→N4)')
ax_net.plot(t, -is2_inh_n3,  color='#c0392b',    lw=1.0, ls='--', label='−inh Is2 (N3→N4)')
ax_net.plot(t, is2_self_exc, color=COLORS['N4'], lw=1.0, ls=':',  label='self-exc Is2')
ax_net.plot(t, -is2_self_inh,color='#7f8c8d',    lw=1.0, ls='-.',  label='−self-inh Is2 (burst killer)')
ax_net.plot(t, net,           color='k',          lw=1.8, label='net Is2 total')
ax_net.axhline(0, color='gray', lw=0.5)
gap = (I_GT - I_0_REST) * 1e6
ax_net.axhline(gap, color='k', lw=0.8, ls=':', alpha=0.6,
               label=f'threshold gap = {gap:.1f} µA')
ax_net.set_ylabel('Is (µA)')
ax_net.set_xlim([0, T_MS])
ax_net.set_xlabel('time (ms)')
ax_net.set_title('N4 net synaptic input (Is2) — burst fires when net > threshold gap, killed by self-inh')
ax_net.legend(fontsize=7, loc='upper right', ncol=3)

fig2.suptitle('Synaptic cascade states: Is1 = step 1 (solid), Is2 = step 2 (dashed)', fontsize=13)
out2 = os.path.join(os.path.dirname(__file__), 'ns_msn_v5_synaptic.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

plt.close('all')
