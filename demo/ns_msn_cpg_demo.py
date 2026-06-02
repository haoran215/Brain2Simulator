"""
demo/ns_msn_cpg_demo.py
=======================
Central Pattern Generator (CPG) demo: two MSN populations (10 each)
with all-to-all mutual inhibition + all-to-all self-inhibition
(spike-frequency adaptation).  Autonomous half-centre oscillator.

Circuit
───────
  Pop A (10 MSN) ──[fast, W_INH  τ_m]──▷ Pop B   mutual cross-inh.
  Pop B (10 MSN) ──[fast, W_INH  τ_m]──▷ Pop A   mutual cross-inh.
  Pop A (10 MSN) ──[slow, W_SELF τ_s]──▷ Pop A   self-inh. / adaptation
  Pop B (10 MSN) ──[slow, W_SELF τ_s]──▷ Pop B   self-inh. / adaptation

  Both populations: constant I_0 = 40 µA  (> I_min ≈ 32.15 µA)

CPG mechanism (autonomous)
──────────────────────────
  1. A fires first (initial Vm asymmetry).  Fast mutual inhibition
     (τ_m = 50 ms) suppresses B immediately.
  2. While A fires, its slow self-inhibition (τ_s = 3 s) builds up.
     Fixed-point analysis: I_self_A* ≈ 7.8 µA  (where f_A ≈ 20–26 Hz).
  3. Release condition: A's mutual inhibition on B drops below B's
     drive margin (7.85 µA) when f_A < 26.2 Hz.
     → B is released and starts firing.
  4. B fires → fast mutual inhibition silences A completely.
     B's self-inhibition starts from 0, cycle repeats.
  Expected half-period ≈ 2–4 s  (≈ τ_s × 0.8).

Key: I_hold = I_max (depol-block onset) = thyristor holding current.

Inheritance: demo/ns_msn_wta_demo.py, README §3–5
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from brian2 import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from msn_neuron import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
prefs.codegen.target = 'cython'
defaultclock.dt = 10 * us
seed(42)

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def _gauss_smooth(arr, sigma_bins):
    hw = int(4 * sigma_bins)
    x  = np.arange(-hw, hw + 1)
    k  = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(arr.astype(float), k, mode='same')

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
p = MSNParams()
I_min, I_max = p.operating_window()   # I_max = I_hold (thyristor holding current)
tau_o, tau_c = p.time_constants()

N         = 10          # neurons per population

# Both populations always driven above I_min
I_0_DRIVE  = 40e-6      # A   = I_min + 7.85 µA  (spontaneously active)
SIM_TIME   = 20         # s

# Fast mutual inhibition (A→B, B→A)
# At f ≈ 97 Hz: total_inh = N × W_INH × f × τ_m = 10 × 600 nA × 97 × 50 ms ≈ 29 µA
# >> drive margin 7.85 µA → complete WTA suppression ✓
W_INH      = 600e-9     # A  per synapse
TAU_S_INH_M = 50e-3     # s  (fast, sets the transition delay ≈ 65 ms)

# Slow self-inhibition / adaptation (A→A, B→B)
# Fixed-point target: I_self* ≈ 7.8 µA  (at f_A ≈ 26 Hz, just past release threshold)
# Requires: W_SELF × τ_s × N × f_fp = 7.8 µA
#   → W_SELF × 3 × 10 × 26.2 Hz = 7.8 µA  → W_SELF ≈ 10 nA
# Active half-period ≈ τ_s × 0.8 ≈ 2.4 s
W_SELF     = 10e-9      # A  per synapse
TAU_S_SELF = 3.0        # s  (slow adaptation timescale)

MON_DT     = 1 * ms

# ─────────────────────────────────────────────────────────────────────────────
# Build network
# ─────────────────────────────────────────────────────────────────────────────
start_scope()

# Two inhibitory inlets: one for cross-inhibition, one for self-inhibition.
# Brian2 requires one Synapses writer per (inlet, target_group) pair.
A = make_msn(N, p,
             exc_inlets=('I_exc',),
             inh_inlets=('I_inh_mut', 'I_inh_self'),
             name='popA')
B = make_msn(N, p,
             exc_inlets=('I_exc',),
             inh_inlets=('I_inh_mut', 'I_inh_self'),
             name='popB')

A.I_exc = 0 * amp
B.I_exc = 0 * amp

# Fast mutual inhibition (A→B writes to B's I_inh_mut, B→A to A's I_inh_mut)
_mut_A = SynapseParams(kind='inh', weight=W_INH,
                       tau_s1=TAU_S_INH_M, tau_s2=TAU_S_INH_M,
                       target_var='I_inh_mut', cascade='alpha')
_mut_B = SynapseParams(kind='inh', weight=W_INH,
                       tau_s1=TAU_S_INH_M, tau_s2=TAU_S_INH_M,
                       target_var='I_inh_mut', cascade='alpha')

# Slow self-inhibition / adaptation (A→A to A's I_inh_self, B→B to B's)
_self_A = SynapseParams(kind='inh', weight=W_SELF,
                        tau_s1=TAU_S_SELF, tau_s2=TAU_S_SELF,
                        target_var='I_inh_self', cascade='alpha')
_self_B = SynapseParams(kind='inh', weight=W_SELF,
                        tau_s1=TAU_S_SELF, tau_s2=TAU_S_SELF,
                        target_var='I_inh_self', cascade='alpha')

syn_A_to_B = make_synapse(A, B, _mut_A,  connect=True, name='syn_A_to_B')
syn_B_to_A = make_synapse(B, A, _mut_B,  connect=True, name='syn_B_to_A')
syn_A_to_A = make_synapse(A, A, _self_A, connect=True, name='syn_A_to_A')
syn_B_to_B = make_synapse(B, B, _self_B, connect=True, name='syn_B_to_B')

# Monitors (set up once; active throughout all run() segments)
spk_A   = SpikeMonitor(A)
spk_B   = SpikeMonitor(B)
state_A = StateMonitor(A, ['I_inh_mut', 'I_inh_self'], record=True, dt=MON_DT)
state_B = StateMonitor(B, ['I_inh_mut', 'I_inh_self'], record=True, dt=MON_DT)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
drive_margin  = I_0_DRIVE - I_min              # 7.85 µA
mut_inh_97    = N * W_INH  * 97 * TAU_S_INH_M # mutual inh at 97 Hz
self_inh_fp   = N * W_SELF * 26.2 * TAU_S_SELF # self-inh fixed point at release f
loop_gain_mut = 5e6 * N * W_INH * TAU_S_INH_M
f_release     = drive_margin / (N * W_INH * TAU_S_INH_M)

print("=" * 66)
print("CPG Demo — autonomous half-centre oscillator (self-inhibition)")
print("=" * 66)
print(f"  I_min = {I_min*1e6:.2f} µA   I_max = I_hold = {I_max*1e6:.0f} µA")
print(f"  Both populations: I_0 = {I_0_DRIVE*1e6:.0f} µA "
      f"(+{drive_margin*1e6:.2f} µA above rheobase)")
print(f"  Mutual inh at 97 Hz: {mut_inh_97*1e6:.1f} µA  >> "
      f"drive margin {drive_margin*1e6:.2f} µA  (loop gain ≈ {loop_gain_mut:.1f})")
print(f"  Release threshold:   f_A < {f_release:.1f} Hz "
      f"(I_self > {(I_0_DRIVE - I_min - 1e-9)*1e6:.2f} µA ≈ drive margin)")
print(f"  Self-inh fixed point (at {f_release:.0f} Hz): "
      f"{self_inh_fp*1e6:.2f} µA  ≈ drive margin ✓")
print(f"  Expected half-period ≈ {TAU_S_SELF * 0.8:.1f} s  "
      f"(τ_self = {TAU_S_SELF:.0f} s)")
print(f"\nRunning {SIM_TIME} s (autonomous oscillation) ...")

# ─────────────────────────────────────────────────────────────────────────────
# Autonomous simulation — both populations always driven
# ─────────────────────────────────────────────────────────────────────────────
A.I_0 = I_0_DRIVE * amp
B.I_0 = I_0_DRIVE * amp

# Small Vm asymmetry so A fires first and breaks symmetry
A.Vm  = p.Vth * 0.45 * volt

run(SIM_TIME * second, report='text')

print(f"\n  Pop A: {spk_A.num_spikes} spikes  "
      f"({spk_A.num_spikes/SIM_TIME/N:.0f} Hz avg)")
print(f"  Pop B: {spk_B.num_spikes} spikes  "
      f"({spk_B.num_spikes/SIM_TIME/N:.0f} Hz avg)")
total = spk_A.num_spikes + spk_B.num_spikes
print(f"  A fraction: {spk_A.num_spikes/max(total,1):.2f}  "
      f"(0.50 = perfect alternation)")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Connectome schematic
# ─────────────────────────────────────────────────────────────────────────────
fig_sch, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(
    'CPG Connectome — Two MSN Populations (N = 10 each)\n'
    'All-to-all mutual inhibition (fast)  +  All-to-all self-inhibition / adaptation (slow)',
    fontsize=11, fontweight='bold', pad=10)

cA = np.array([3.2, 2.9])
cB = np.array([8.8, 2.9])
r  = 1.05
theta_ring = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, N, endpoint=False)
posA = np.column_stack([cA[0] + r*np.cos(theta_ring), cA[1] + r*np.sin(theta_ring)])
posB = np.column_stack([cB[0] + r*np.cos(theta_ring), cB[1] + r*np.sin(theta_ring)])

# All-to-all connections (faint)
for i in range(N):
    for j in range(N):
        ax.plot([posA[i,0], posB[j,0]], [posA[i,1], posB[j,1]],
                c='#E53935', alpha=0.045, lw=0.5, zorder=1)

# Population halos
for c, col in [(cA,'#1565C0'), (cB,'#B71C1C')]:
    ax.add_patch(Circle(c, r+0.22, fill=False, ec=col, lw=2, alpha=0.35, ls='--', zorder=2))

# Neuron dots
for idx, (x, y) in enumerate(posA):
    ax.add_patch(Circle((x,y), 0.13, fc='#1E88E5', ec='white', lw=0.8, zorder=4))
    ax.text(x, y, str(idx), ha='center', va='center',
            fontsize=5, color='white', fontweight='bold', zorder=5)
for idx, (x, y) in enumerate(posB):
    ax.add_patch(Circle((x,y), 0.13, fc='#E53935', ec='white', lw=0.8, zorder=4))
    ax.text(x, y, str(idx), ha='center', va='center',
            fontsize=5, color='white', fontweight='bold', zorder=5)

# Population labels
ax.text(cA[0], 1.55, 'Pop A\n10 × MSN', ha='center', va='top',
        fontsize=10, color='#1565C0', fontweight='bold')
ax.text(cB[0], 1.55, 'Pop B\n10 × MSN', ha='center', va='top',
        fontsize=10, color='#B71C1C', fontweight='bold')

# Inhibitory arrows
mid_x = (cA[0]+cB[0]) / 2
ax.annotate('', xy=(cB[0]-r-0.28, 3.3), xytext=(cA[0]+r+0.28, 3.3),
            arrowprops=dict(arrowstyle='-|>', color='#C62828', lw=2.5, mutation_scale=16), zorder=6)
ax.text(mid_x, 3.58, 'Inhibition  A → B', ha='center', fontsize=9,
        color='#C62828', fontweight='bold')
ax.annotate('', xy=(cA[0]+r+0.28, 2.5), xytext=(cB[0]-r-0.28, 2.5),
            arrowprops=dict(arrowstyle='-|>', color='#C62828', lw=2.5, mutation_scale=16), zorder=6)
ax.text(mid_x, 2.17, 'Inhibition  B → A', ha='center', fontsize=9,
        color='#C62828', fontweight='bold')

# Constant I_0 input boxes
def drive_box(ax, cx, cy, label, col):
    ax.add_patch(mpatches.FancyBboxPatch((cx-0.80, cy-0.40), 1.6, 0.8,
        boxstyle='round,pad=0.06', fc='#E8F5E9', ec=col, lw=1.8, zorder=3))
    ax.text(cx, cy, label, ha='center', va='center', fontsize=7.5,
            color=col, fontweight='bold', zorder=5)

drive_box(ax, cA[0]-r-1.0, cA[1],
          f'Constant I₀\n= {I_0_DRIVE*1e6:.0f} µA', '#1565C0')
drive_box(ax, cB[0]+r+1.0, cB[1],
          f'Constant I₀\n= {I_0_DRIVE*1e6:.0f} µA', '#B71C1C')
ax.annotate('', xy=(cA[0]-r-0.22, cA[1]), xytext=(cA[0]-r-0.58, cA[1]),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.2), zorder=6)
ax.annotate('', xy=(cB[0]+r+0.22, cB[1]), xytext=(cB[0]+r+0.58, cB[1]),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=2.2), zorder=6)

# Self-inhibition loops (curved arrows on each population ring)
from matplotlib.patches import Arc
for cx, cy, col in [(cA[0], cA[1], '#0D47A1'), (cB[0], cB[1], '#7F0000')]:
    arc = Arc((cx, cy + r + 0.32), width=0.9, height=0.55,
              angle=0, theta1=200, theta2=340, color=col, lw=2.0, zorder=6)
    ax.add_patch(arc)
    ax.annotate('', xy=(cx + 0.38, cy + r + 0.18),
                xytext=(cx + 0.39, cy + r + 0.34),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.5, mutation_scale=10), zorder=7)
    ax.text(cx, cy + r + 0.70, f'Self-inh\n(adapt.)', ha='center',
            fontsize=7.5, color=col, fontstyle='italic')

# Info box
info = (f'Mutual inh:  100 syn A→B + 100 syn B→A  |  '
        f'w = {W_INH*1e9:.0f} nA,  τ_m = {TAU_S_INH_M*1e3:.0f} ms  (loop gain ≈ {loop_gain_mut:.1f})\n'
        f'Self-inh:    100 syn A→A + 100 syn B→B  |  '
        f'w = {W_SELF*1e9:.0f} nA,  τ_s = {TAU_S_SELF:.0f} s   (adaptation / fatigue)')
ax.text(6.0, 0.32, info, ha='center', va='bottom', fontsize=8.0,
        bbox=dict(boxstyle='round,pad=0.4', fc='#FFFDE7', ec='#F9A825', lw=1.3))

fig_sch.tight_layout()
sch_path = os.path.join(OUTDIR, 'ns_msn_cpg_connectome.png')
fig_sch.savefig(sch_path, dpi=150, bbox_inches='tight')
print(f"\nSchematic → {sch_path}")
plt.close(fig_sch)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Simulation results
# ─────────────────────────────────────────────────────────────────────────────
t_bins = np.arange(0, SIM_TIME + 1e-6, 0.001)
t_mid  = 0.5 * (t_bins[:-1] + t_bins[1:])


def pop_rate(spk_mon, N_pop, sigma_ms=80):
    counts, _ = np.histogram(spk_mon.t / second, bins=t_bins)
    return _gauss_smooth(counts / (N_pop * 0.001), sigma_bins=sigma_ms)


rA = pop_rate(spk_A, N)
rB = pop_rate(spk_B, N)

t_st     = state_A.t / second
mut_A    = np.mean(state_A.I_inh_mut  / amp, axis=0) * 1e6   # µA mutual on A (from B)
mut_B    = np.mean(state_B.I_inh_mut  / amp, axis=0) * 1e6   # µA mutual on B (from A)
self_A   = np.mean(state_A.I_inh_self / amp, axis=0) * 1e6   # µA adaptation in A
self_B   = np.mean(state_B.I_inh_self / amp, axis=0) * 1e6   # µA adaptation in B

fig_res, axes = plt.subplots(
    3, 1, figsize=(13, 8.5), sharex=True,
    gridspec_kw={'height_ratios': [2.2, 1.5, 1.5], 'hspace': 0.07})

# ── Raster ────────────────────────────────────────────────────────────────────
ax_r = axes[0]
tA_ = spk_A.t/second;  iA_ = spk_A.i
tB_ = spk_B.t/second;  iB_ = spk_B.i
ax_r.scatter(tB_, iB_,         s=2, c='#E53935', alpha=0.55, lw=0, label='Pop B')
ax_r.scatter(tA_, iA_ + N + 1, s=2, c='#1E88E5', alpha=0.55, lw=0, label='Pop A')
ax_r.axhline(N+0.5, color='#424242', lw=0.8, ls='--', alpha=0.4)
ax_r.set_ylim(-0.8, 2*N+2.0)
ax_r.set_yticks([N/2 - 0.5, N + N/2 + 0.5])
ax_r.set_yticklabels(['Pop B', 'Pop A'], fontsize=10)
ax_r.set_ylabel('Neuron', fontsize=10)
ax_r.set_title(
    'CPG — Autonomous Half-Centre Oscillator  '
    '(mutual inh. τ=50 ms  +  self-inh./adapt. τ=3 s)',
    fontsize=11, fontweight='bold')
ax_r.legend(loc='upper right', markerscale=4, fontsize=9)

# ── Population rates ──────────────────────────────────────────────────────────
ax_f = axes[1]
ax_f.fill_between(t_mid, rA, alpha=0.25, color='#1E88E5')
ax_f.fill_between(t_mid, rB, alpha=0.25, color='#E53935')
ax_f.plot(t_mid, rA, color='#1565C0', lw=1.5, label='Pop A rate')
ax_f.plot(t_mid, rB, color='#B71C1C', lw=1.5, label='Pop B rate')
ax_f.set_ylabel('Rate (Hz)', fontsize=10)
ax_f.legend(loc='upper right', fontsize=9)

# ── Inhibitory currents: mutual (solid) and self-inh/adaptation (dashed) ──────
ax_i = axes[2]
ax_i.fill_between(t_st, mut_A,  alpha=0.18, color='#1E88E5')
ax_i.fill_between(t_st, mut_B,  alpha=0.18, color='#E53935')
ax_i.plot(t_st, mut_A,  color='#1565C0', lw=1.5, label='Mutual inh on A  (from B, fast)')
ax_i.plot(t_st, mut_B,  color='#B71C1C', lw=1.5, label='Mutual inh on B  (from A, fast)')
ax_i.plot(t_st, self_A, color='#1565C0', lw=1.5, ls='--', label='Self-inh in A  (adapt., slow)')
ax_i.plot(t_st, self_B, color='#B71C1C', lw=1.5, ls='--', label='Self-inh in B  (adapt., slow)')
suppress_thr = drive_margin * 1e6
ax_i.axhline(suppress_thr, color='k', ls=':', lw=1.1,
             label=f'Drive margin  ({suppress_thr:.2f} µA = I₀−I_min)')
ax_i.set_ylabel('Current (µA)', fontsize=10)
ax_i.set_xlabel('Time (s)', fontsize=10)
ax_i.legend(loc='upper right', fontsize=8.0, ncol=2)

for ax in axes:
    ax.set_xlim(0, SIM_TIME)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

fig_res.tight_layout()
res_path = os.path.join(OUTDIR, 'ns_msn_cpg_results.png')
fig_res.savefig(res_path, dpi=150, bbox_inches='tight')
print(f"Results   → {res_path}")
plt.close(fig_res)
