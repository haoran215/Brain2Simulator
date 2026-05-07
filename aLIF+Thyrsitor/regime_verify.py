import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

tau_s = 10e-3   # s
Iw    = 20e-6   # A

# Alpha function
def alpha(t, Iw, tau_s):
    out = np.zeros_like(t)
    m = t >= 0
    out[m] = (Iw/tau_s) * t[m] * np.exp(-t[m]/tau_s)
    return out

# Superimpose N spikes at rate f
def Is2_train(f_hz, duration_s, tau_s, Iw, dt=0.05e-3):
    t = np.arange(0, duration_s, dt)
    Is2 = np.zeros_like(t)
    ISI = 1.0/f_hz
    spike_times = np.arange(ISI, duration_s, ISI)
    for ts in spike_times:
        t_rel = t - ts
        Is2 += alpha(t_rel, Iw, tau_s)
    return t, Is2

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

colours = {'temporal': '#E74C3C', 'boundary': '#F39C12', 'rate': '#27AE60'}
cases   = [
    (70,  'temporal', 'f = 70 Hz  (ISI=14.3ms > τ_s=10ms)\n→ TEMPORAL regime'),
    (100, 'boundary', 'f = 100 Hz  (ISI=10ms = τ_s=10ms)\n→ BOUNDARY'),
    (200, 'rate',     'f = 200 Hz  (ISI=5ms < τ_s=10ms)\n→ RATE regime'),
]

# Row 0: Is2 waveforms for single spike + train
for col, (f_hz, regime, title) in enumerate(cases):
    ax = fig.add_subplot(gs[0, col])
    t, Is2 = Is2_train(f_hz, 120e-3, tau_s, Iw)
    ISI = 1/f_hz

    ax.fill_between(t*1e3, Is2*1e6, alpha=0.2, color=colours[regime])
    ax.plot(t*1e3, Is2*1e6, color=colours[regime], lw=1.4)

    # Mark spike times
    spike_times_ms = np.arange(ISI, 120e-3, ISI)*1e3
    ax.vlines(spike_times_ms, 0, -0.5, colors='k', lw=1.0, label='Spikes')

    # Mark tau_s
    ax.axvline(ISI*1e3, color='gray', ls=':', lw=1.0, alpha=0.6)

    # DC level annotation
    DC = Iw * f_hz * tau_s * 1e6
    ax.axhline(DC, color=colours[regime], ls='--', lw=1.2, alpha=0.7,
               label=f'DC = Iw·f·τ_s = {DC:.1f} µA')

    ax.set_title(title, fontsize=9, fontweight='bold', color=colours[regime])
    ax.set_xlabel('Time  (ms)'); ax.set_ylabel('Is2  (µA)')
    ax.set_xlim(0, 120); ax.set_ylim(-1, max(DC*1.3, Iw/np.e*1e6*1.3))
    ax.legend(fontsize=7.5)

# Row 1: Overlap ratio and DC/peak across frequency range
ax_overlap = fig.add_subplot(gs[1, :2])
f_range = np.linspace(10, 400, 1000)
ISI_range = 1/f_range * 1e3  # ms
ratio_range = ISI_range / (tau_s*1e3)  # ISI/tau_s
DC_range = Iw * f_range * tau_s  # A
peak_single = Iw/np.e                  # A
overlap_range = DC_range / peak_single

ax_overlap.plot(f_range, ratio_range, color='steelblue', lw=2.0,
                label='ISI/τ_s  (>1 = temporal, <1 = rate)')
ax_overlap.plot(f_range, overlap_range, color='darkorange', lw=2.0, ls='--',
                label='DC/peak_single  (>1 = heavy overlap)')
ax_overlap.axhline(1.0, color='k', ls=':', lw=1.2)
ax_overlap.axvline(1/tau_s, color='purple', ls='--', lw=1.5,
                   label=f'Crossover: f = 1/τ_s = {1/tau_s:.0f} Hz')

# Mark operating points
for f_pt, col, lbl in [(70, colours['temporal'], 'I_min=40µA'),
                        (200, colours['rate'],     'I_max=100µA')]:
    ax_overlap.axvline(f_pt, color=col, ls='-', lw=1.2, alpha=0.7)
    ax_overlap.text(f_pt+4, 3.8, f'{f_pt}Hz\n{lbl}', fontsize=7.5, color=col,
                    bbox=dict(boxstyle='round', fc='white', ec=col, alpha=0.8))

ax_overlap.fill_betweenx([0, 5], 0, 1/tau_s, alpha=0.06, color='red',
                          label='Temporal zone (ISI > τ_s)')
ax_overlap.fill_betweenx([0, 5], 1/tau_s, 400, alpha=0.06, color='green',
                          label='Rate zone (ISI < τ_s)')
ax_overlap.set_xlabel('Firing rate  f  (Hz)', fontsize=10)
ax_overlap.set_ylabel('Ratio', fontsize=10)
ax_overlap.set_title('Regime Transition  —  ISI/τ_s and DC overlap vs firing rate\n'
                     'Both operating points (70Hz, 200Hz) straddle the crossover at 100Hz  ✓',
                     fontsize=9, fontweight='bold')
ax_overlap.set_xlim(10, 400); ax_overlap.set_ylim(0, 5)
ax_overlap.legend(fontsize=8, loc='upper right')
ax_overlap.grid(alpha=0.25)

# Row 1 col 2: Is1 vs Is2 shape comparison (why Is2 is the right filter)
ax_shape = fig.add_subplot(gs[1, 2])
t_single = np.linspace(0, 80e-3, 2000)
Is1_single = Iw * np.exp(-t_single/tau_s)
Is2_single = (Iw/tau_s) * t_single * np.exp(-t_single/tau_s)
ax_shape.plot(t_single*1e3, Is1_single*1e6, color='#8E44AD', lw=1.8,
              label='Is1 (exp, one stage)')
ax_shape.plot(t_single*1e3, Is2_single*1e6, color='#16A085', lw=1.8,
              label='Is2 (alpha, two stages)')
ax_shape.axvline(tau_s*1e3, color='gray', ls='--', lw=1.0, label=f'τ_s = {tau_s*1e3:.0f} ms')
ax_shape.text(tau_s*1e3+1, Iw/np.e*1e6*0.5,
              f'Is2 peaks at τ_s\n= {tau_s*1e3:.0f} ms', fontsize=7.5)
ax_shape.set_xlabel('Time  (ms)'); ax_shape.set_ylabel('Current  (µA)')
ax_shape.set_title('Single-spike shape\nIs1 vs Is2 — why Is2 sets the coding regime',
                   fontsize=9, fontweight='bold')
ax_shape.legend(fontsize=8)
ax_shape.set_xlim(0, 80)

# Row 2: Planned modular architecture (text diagram)
ax_arch = fig.add_subplot(gs[2, :])
ax_arch.axis('off')
ax_arch.set_xlim(0, 1); ax_arch.set_ylim(0, 1)

arch_text = """
PLANNED MODULAR ARCHITECTURE  (JSON-configured SNN)
═══════════════════════════════════════════════════════════════════════════════════════════════════

  config.json                        Brian2 Python modules
  ───────────────                    ──────────────────────────────────────────────────────
  {                                  neuron.py        →  NeuronModule(params)
    "neuron":  { Cm, Ra, Rm_hi,                           NeuronGroup(eqs, threshold, reset, t_ref)
                 Vthresh, t_ref,     synapse.py       →  SynapseModule(params)
                 I_0, tau_m }                              Synapses(on_pre: Is1 += Iw·δ)
    "synapse": { Rs, Cs, tau_s1,    network.py       →  NetworkModule(json)
                 tau_s2, Iw,                               connects neuron ↔ synapse modules
                 mode: Is1|Is2 }                           builds adjacency from "connections" list
    "input":   { type: Poisson,     regime.py        →  auto-detect from f vs 1/tau_s
                 rate: [20,120],                           if ISI < tau_s  → rate_mode  → reservoir
                 n: 2 }                                    if ISI > tau_s  → temporal_mode → STDP
    "network": {                     learning.py      →  STDPModule  or  ReservoirReadout
      "connections": [                                      STDP:     on_pre/on_post weight update
        {pre:0, post:1, w:1.0},                            Reservoir: fixed weights + linear readout
        {pre:1, post:0, w:0.8} ]
    },                               RATE REGIME (f > 1/τ_s = 100Hz):
    "learning": {                      → Is2 pulses overlap → smooth DC current → rate signal
      "rule": "STDP"|"reservoir",      → Reservoir: random recurrent weights, Is2 readout
      "A_plus": 0.01,                  → Brian2: StateMonitor(Is2) + sklearn LinearRegression ✓
      "A_minus": 0.012,
      "tau_plus": 20e-3,             TEMPORAL REGIME (f < 1/τ_s = 100Hz):
      "tau_minus": 20e-3 }             → Is2 pulses resolved individually → spike timing matters
  }                                    → STDP: Δw = A+ exp(-Δt/τ+) or -A- exp(Δt/τ-)
                                        → Brian2: Synapses(on_pre+on_post) + SpikeMonitor ✓

  SAME neuron equations for both regimes.  Only learning rule + readout differ.
"""

ax_arch.text(0.01, 0.98, arch_text, transform=ax_arch.transAxes,
             fontsize=8.5, va='top', family='monospace',
             bbox=dict(boxstyle='round', fc='#F8F9FA', ec='#BDC3C7', alpha=0.95))

fig.suptitle(
    'Rate vs Temporal Coding Regime Verification  —  τ_s = 10 ms,  crossover at f = 1/τ_s = 100 Hz\n'
    'Operating range: 70 Hz (temporal) → 100 Hz (boundary) → 200 Hz (rate)   '
    '— both regimes accessible by design  ✓',
    fontsize=11, fontweight='bold', y=1.01)

plt.savefig('/mnt/user-data/outputs/regime_verification.png', dpi=130, bbox_inches='tight')
print("Saved.")