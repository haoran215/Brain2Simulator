"""
ns_msn_compass_demo.py
======================
A minimal 1-D Drosophila head-direction (compass) circuit built from MSN
neurons and the split synapse library.  Two heading states only — "Right"
and "Left" — so the ring attractor collapses to a two-unit winner-take-all
(WTA) network that *flips* its winner each time a sensory pulse arrives.

Biology → simplification
─────────────────────────
Ellipsoid Body (E)  → EB_L, EB_R  : the heading "bump" (the WTA pair)
                      EB_GI        : the Gall / global inhibitor that makes
                                     EB_L, EB_R compete (winner-take-all)
Protocerebral Bridge (P) → PB_l, PB_r : position-copy / routing relay

Connectivity (mirrors panel d)
──────────────────────────────
  EB_L ──exc──► EB_L                 self-excitation  (latches the bump)
  EB_R ──exc──► EB_R                 self-excitation
  EB_L ──exc──► EB_GI                drive the global inhibitor
  EB_R ──exc──► EB_GI
  EB_GI ─inh──► EB_L                 global inhibition  ─┐
  EB_GI ─inh──► EB_R                                     ├ WTA
  EB_L ──exc──► PB_l                 position copy (SAME side)
  EB_R ──exc──► PB_r
  PB_r ──exc──► EB_L                 cross-activation (OPPOSITE side)
  PB_l ──exc──► EB_R

Inputs (both modelled as external *currents*, per the spec)
───────────────────────────────────────────────────────────
  • Every neuron carries a subthreshold tonic bias I_0  (silent at rest).
  • A brief initial current "seed" to EB_R selects the starting winner.
  • A "sensory pulse" is a brief current step delivered to BOTH PBs.

How a turn happens
──────────────────
Say EB_R is the winner: EB_R latches, drives EB_GI (which suppresses EB_L),
and copies to PB_r — so PB_r is the *primed* relay.  When a sensory pulse
hits both PBs, the primed PB_r fires hardest and cross-activates EB_L.  EB_L
wins the refreshed WTA, drives EB_GI to silence EB_R, and copies to PB_l.
The heading has flipped R → L.  The next pulse flips it back.

Run
───
    uv run python demo/ns_msn_compass_demo.py
produces  demo/ns_msn_compass_demo.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brian2 import *
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

seed(1)
prefs.codegen.target = 'cython'  # use Cython for speed; fallback to 'numpy' if Cython isn't available
defaultclock.dt = 10 * us

# ── Hardware parameters ─────────────────────────────────────────────────────
p     = MSNParams()
I_GT  = p.I_gt                  # ≈ 32 µA rheobase (gate trigger current)
print(p.summary())
print(f"I_gt (rheobase) = {I_GT*1e6:.2f} µA")

T_SIM = 2.4                     # s

# ── Tonic bias: every neuron sits just below threshold (silent at rest) ─────
I0_EB = 0.90 * I_GT             # EB_L / EB_R resting bias
I0_GI = 0.80 * I_GT            # global inhibitor: a little lower, needs EB drive
I0_PB = 0.90 * I_GT            # PB relays

# ── Initial "seed" current that selects the starting winner (EB_R) ──────────
SEED_I   = 1.8 * I_GT
SEED_END = 80e-3                # s

# ── Sensory pulses: brief current step to BOTH PBs (the "turn" command) ─────
PULSE_TIMES = [0.7, 1.3, 1.9]   # s
PULSE_DUR   = 60e-3             # s
PULSE_I     = 0.16 * I_GT       # added on top of I0_PB (kept modest on purpose:
                                #  alone it is sub-threshold; only the COPY-primed
                                #  PB crosses threshold and fires a burst)

# ── Synaptic weights (A) and time constants (s) ─────────────────────────────
# EB self-excitation — latches the active bump
W_SELF,  TAU_SELF  = 11e-6, 12e-3
# EB → EB_GI — drive the global inhibitor
W_EBGI,  TAU_EBGI  = 14e-6, 8e-3
# EB_GI → EB — global inhibition (the WTA arbiter)
W_GIEB,  TAU_GIEB  = 9e-6,  8e-3
# EB → PB — position copy (same side)
W_COPY,  TAU_COPY  = 9e-6,  12e-3
# PB → EB — cross-activation (opposite side); the turn signal path
W_CROSS, TAU_CROSS = 10e-6, 12e-3

# Recording dt
REC_DT = 0.5e-3

# ── Build network ───────────────────────────────────────────────────────────
start_scope()

# Ellipsoid body WTA pair: each gets self-exc + cross-exc, and GI inhibition
EB_L = make_msn(1, params=p,
                exc_inlets=('I_exc_self', 'I_exc_cross'),
                inh_inlets=('I_inh_gi',), name='EB_L')
EB_R = make_msn(1, params=p,
                exc_inlets=('I_exc_self', 'I_exc_cross'),
                inh_inlets=('I_inh_gi',), name='EB_R')
# Global inhibitor: excited separately by each EB (two inlets — Brian2 needs
# one Synapses per (inlet, group))
EB_GI = make_msn(1, params=p,
                 exc_inlets=('I_exc_l', 'I_exc_r'),
                 inh_inlets=(), name='EB_GI')
# Protocerebral-bridge relays: receive the EB copy; sensory pulse is a current
PB_l = make_msn(1, params=p, exc_inlets=('I_exc_eb',), inh_inlets=(), name='PB_l')
PB_r = make_msn(1, params=p, exc_inlets=('I_exc_eb',), inh_inlets=(), name='PB_r')

EB_L.I_0  = I0_EB * amp
EB_R.I_0  = I0_EB * amp
EB_GI.I_0 = I0_GI * amp
PB_l.I_0  = I0_PB * amp
PB_r.I_0  = I0_PB * amp


def syn(src, tgt, kind, w, tau, target_var, name):
    return make_synapse(src, tgt,
                        params=SynapseParams(kind=kind, weight=w,
                                             tau_s1=tau, tau_s2=tau,
                                             cascade='alpha',
                                             target_var=target_var),
                        connect=True, name=name)

# self-excitation (latch)
s_LL = syn(EB_L, EB_L, 'exc', W_SELF, TAU_SELF, 'I_exc_self', 'syn_EBL_self')
s_RR = syn(EB_R, EB_R, 'exc', W_SELF, TAU_SELF, 'I_exc_self', 'syn_EBR_self')
# EB → GI
s_LGI = syn(EB_L, EB_GI, 'exc', W_EBGI, TAU_EBGI, 'I_exc_l', 'syn_EBL_GI')
s_RGI = syn(EB_R, EB_GI, 'exc', W_EBGI, TAU_EBGI, 'I_exc_r', 'syn_EBR_GI')
# GI → EB (global inhibition)
s_GIL = syn(EB_GI, EB_L, 'inh', W_GIEB, TAU_GIEB, 'I_inh_gi', 'syn_GI_EBL')
s_GIR = syn(EB_GI, EB_R, 'inh', W_GIEB, TAU_GIEB, 'I_inh_gi', 'syn_GI_EBR')
# EB → PB copy (same side)
s_LPl = syn(EB_L, PB_l, 'exc', W_COPY, TAU_COPY, 'I_exc_eb', 'syn_EBL_PBl')
s_RPr = syn(EB_R, PB_r, 'exc', W_COPY, TAU_COPY, 'I_exc_eb', 'syn_EBR_PBr')
# PB → EB cross (opposite side)
s_PrL = syn(PB_r, EB_L, 'exc', W_CROSS, TAU_CROSS, 'I_exc_cross', 'syn_PBr_EBL')
s_PlR = syn(PB_l, EB_R, 'exc', W_CROSS, TAU_CROSS, 'I_exc_cross', 'syn_PBl_EBR')

# ── External currents: seed (start) + sensory pulses ────────────────────────
def in_pulse(t):
    tt = float(t / second)
    return any(t0 <= tt < t0 + PULSE_DUR for t0 in PULSE_TIMES)

@network_operation(when='start')
def drive(t):
    # initial seed → EB_R becomes the starting winner
    EB_R.I_0 = (SEED_I if t < SEED_END * second else I0_EB) * amp
    # sensory pulse → both PB relays (symmetric current step)
    pulse = PULSE_I if in_pulse(t) else 0.0
    PB_l.I_0 = (I0_PB + pulse) * amp
    PB_r.I_0 = (I0_PB + pulse) * amp

# ── Monitors ────────────────────────────────────────────────────────────────
groups = [('EB_L', EB_L), ('EB_R', EB_R), ('EB_GI', EB_GI),
          ('PB_l', PB_l), ('PB_r', PB_r)]
spk = {n: SpikeMonitor(g) for n, g in groups}
vm  = {n: StateMonitor(g, 'Vout', record=True, dt=REC_DT*second) for n, g in groups}

# diagnostic synaptic currents
sm_cross_L = StateMonitor(s_PrL, 'Is2', record=True, dt=REC_DT*second)  # PB_r→EB_L
sm_cross_R = StateMonitor(s_PlR, 'Is2', record=True, dt=REC_DT*second)  # PB_l→EB_R
sm_gi_L    = StateMonitor(s_GIL, 'Is2', record=True, dt=REC_DT*second)  # GI→EB_L

# ── Run ─────────────────────────────────────────────────────────────────────
# Build the Network EXPLICITLY.  collect() inspects the caller namespace and
# would miss the monitors held inside the `spk`/`vm` dict comprehensions
# (comprehensions have their own scope) — they would silently never run.
synapses = [s_LL, s_RR, s_LGI, s_RGI, s_GIL, s_GIR, s_LPl, s_RPr, s_PrL, s_PlR]
monitors = list(spk.values()) + list(vm.values()) + [sm_cross_L, sm_cross_R, sm_gi_L]
net = Network(EB_L, EB_R, EB_GI, PB_l, PB_r, *synapses, drive, *monitors)
net.run(T_SIM * second, report='text', report_period=0.5*second)

print("\nSpike counts:")
for n, _ in groups:
    print(f"  {n:5s}: {spk[n].num_spikes}")

# ── Decode heading from EB_L vs EB_R firing (sliding-window rate) ────────────
def rate_trace(sm, t_edges, win=40e-3):
    """Sliding firing rate (Hz) of a SpikeMonitor on a regular time grid."""
    ts = np.asarray(sm.t / second)
    r  = np.zeros_like(t_edges)
    for k, tc in enumerate(t_edges):
        r[k] = np.sum((ts > tc - win) & (ts <= tc)) / win
    return r

t_grid = np.arange(0, T_SIM, REC_DT)
rL = rate_trace(spk['EB_L'], t_grid)
rR = rate_trace(spk['EB_R'], t_grid)
heading = np.where(rR > rL, +1.0, np.where(rL > rR, -1.0, 0.0))  # +1=Right, −1=Left

# ── Plot ────────────────────────────────────────────────────────────────────
COLORS = {'EB_L': '#2980b9', 'EB_R': '#c0392b', 'EB_GI': '#27ae60',
          'PB_l': '#8e44ad', 'PB_r': '#e67e22'}
T_MS = T_SIM * 1e3


def mark_pulses(ax):
    for t0 in PULSE_TIMES:
        ax.axvspan(t0*1e3, (t0+PULSE_DUR)*1e3, color='gold', alpha=0.25, lw=0)


fig, axes = plt.subplots(5, 1, figsize=(13, 12),
                         gridspec_kw={'height_ratios': [2, 1.4, 1.4, 1.2, 1.2]})
fig.suptitle('1-D Drosophila compass: WTA heading flips on each sensory pulse',
             fontsize=14)

# (0) raster
ax = axes[0]
order = ['EB_L', 'EB_R', 'EB_GI', 'PB_l', 'PB_r']
for i, n in enumerate(order):
    ts = spk[n].t / ms
    ax.scatter(ts, np.full_like(ts, i), c=COLORS[n], s=6, label=n)
mark_pulses(ax)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlim([0, T_MS]); ax.set_ylim([-0.6, len(order)-0.4])
ax.set_ylabel('neuron'); ax.set_title('Spike raster  (gold = sensory pulse)')
ax.invert_yaxis()

# (1) decoded heading
ax = axes[1]
ax.fill_between(t_grid*1e3, heading, 0, where=heading > 0, color=COLORS['EB_R'],
                alpha=0.6, step='pre', label='Right (EB_R)')
ax.fill_between(t_grid*1e3, heading, 0, where=heading < 0, color=COLORS['EB_L'],
                alpha=0.6, step='pre', label='Left (EB_L)')
mark_pulses(ax)
ax.set_ylim([-1.3, 1.3]); ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(['Left', '—', 'Right'])
ax.set_xlim([0, T_MS]); ax.set_ylabel('heading')
ax.set_title('Decoded heading state (winner of EB_L vs EB_R)')
ax.legend(loc='upper right', fontsize=8)

# (2) EB_L / EB_R Vout
ax = axes[2]
ax.plot(vm['EB_R'].t/ms, vm['EB_R'].Vout[0], color=COLORS['EB_R'], lw=0.7, label='EB_R')
ax.plot(vm['EB_L'].t/ms, vm['EB_L'].Vout[0], color=COLORS['EB_L'], lw=0.7, label='EB_L')
mark_pulses(ax)
ax.set_xlim([0, T_MS]); ax.set_ylabel('Vout (V)')
ax.set_title('Ellipsoid-body WTA pair'); ax.legend(loc='upper right', fontsize=8)

# (3) PB relays Vout
ax = axes[3]
ax.plot(vm['PB_r'].t/ms, vm['PB_r'].Vout[0], color=COLORS['PB_r'], lw=0.7, label='PB_r')
ax.plot(vm['PB_l'].t/ms, vm['PB_l'].Vout[0], color=COLORS['PB_l'], lw=0.7, label='PB_l')
mark_pulses(ax)
ax.set_xlim([0, T_MS]); ax.set_ylabel('Vout (V)')
ax.set_title('Protocerebral-bridge relays (copy of the active EB)')
ax.legend(loc='upper right', fontsize=8)

# (4) cross-activation currents that drive the flip
ax = axes[4]
ax.plot(sm_cross_L.t/ms, sm_cross_L.Is2[0]*1e6, color=COLORS['EB_L'], lw=1.0,
        label='PB_r → EB_L  (cross)')
ax.plot(sm_cross_R.t/ms, sm_cross_R.Is2[0]*1e6, color=COLORS['EB_R'], lw=1.0,
        label='PB_l → EB_R  (cross)')
ax.plot(sm_gi_L.t/ms, sm_gi_L.Is2[0]*1e6, color=COLORS['EB_GI'], lw=0.9, ls='--',
        label='GI → EB  (inhibition)')
mark_pulses(ax)
ax.set_xlim([0, T_MS]); ax.set_ylabel('Is2 (µA)'); ax.set_xlabel('time (ms)')
ax.set_title('Cross-activation vs global inhibition (the flip mechanism)')
ax.legend(loc='upper right', fontsize=8)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'ns_msn_compass_demo.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close('all')
