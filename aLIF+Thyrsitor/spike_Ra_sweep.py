"""
spike_Ra_sweep.py
=================
Visualises how Ra (axon-hillock resistor) shapes:
  1. Single spike waveform  (sub-threshold rise → threshold crossing → reset)
  2. Tonic firing pattern   (constant DC drive, multiple spikes)
  3. Analytical I-F curves  (firing rate vs. I₀ for each Ra)

Model: Memristor aLIF  (Eqs. 9-12, same as ns_test_claude.py)
  Cm dVm/dt = -Vm / (Rm_hi + Ra) + I_0
Cm and t_ref are fixed from the solved parameters at Ra_base = 2.2 kΩ.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import *

prefs.codegen.target = 'numpy'

# ── Fixed hardware parameters ─────────────────────────────────────────────────
Rm_hi_ohm = 100_000   # Ω   memristor HIGH state (quiescent)
Vthresh_V = 4.0       # V   spike threshold
Vreset_V  = 0.0       # V   post-spike reset

# ── Solve Cm and t_ref from the base model (Ra = 2.2 kΩ) ─────────────────────
Ra_base_ohm  = 2_200
I_min_A      = 40e-6;  I_max_A = 100e-6
f_min_Hz     = 70.0;   f_max_Hz = 200.0
I_0_solve_A  = 15e-6   # tonic bias used only in the solver
R_tot_base   = Rm_hi_ohm + Ra_base_ohm
Vm_ss_mn     = (I_min_A + I_0_solve_A) * R_tot_base
Vm_ss_mx     = (I_max_A + I_0_solve_A) * R_tot_base
lmn  = np.log(1 - Vthresh_V / Vm_ss_mn)
lmx  = np.log(1 - Vthresh_V / Vm_ss_mx)
tau_m_base_s = (1/f_min_Hz - 1/f_max_Hz) / (-(lmn - lmx))
t_ref_s      = 1/f_min_Hz - (-tau_m_base_s * lmn)
Cm_F         = tau_m_base_s / R_tot_base

print(f"Fixed Cm = {Cm_F*1e9:.3f} nF,  t_ref = {t_ref_s*1e3:.4f} ms")

# ── Ra sweep values ───────────────────────────────────────────────────────────
Ra_values = [220, 1_000, 2_200, 5_000, 10_000, 22_000, 50_000]   # Ω
Ra_labels  = [f'{r/1e3:.2g} kΩ' for r in Ra_values]
cmap       = plt.cm.plasma
colors     = cmap(np.linspace(0.1, 0.88, len(Ra_values)))

# ── Simulation settings ───────────────────────────────────────────────────────
DT_US       = 50      # µs  timestep
REC_DT_MS   = 0.05    # ms  monitor interval

I_single_A  = 50e-6   # A   drive for single-spike panel (50 µA)
I_tonic_A   = 70e-6   # A   drive for tonic panel (70 µA)
DUR_TONIC   = 300     # ms

# ── Run simulations ───────────────────────────────────────────────────────────
results_single = {}
results_tonic  = {}

for Ra_ohm in Ra_values:
    Rtot = Rm_hi_ohm + Ra_ohm

    # — Single spike (run until first spike + 30 ms buffer) ———————————————
    Vm_ss  = I_single_A * Rtot
    tau_m  = Cm_F * Rtot
    if Vm_ss > Vthresh_V:
        tc_s = -tau_m * np.log(1 - Vthresh_V / Vm_ss)
    else:
        tc_s = 999.0
    dur_single_s = min(tc_s + 0.030, 0.200)   # first spike + 30 ms, max 200 ms

    start_scope()
    defaultclock.dt = DT_US * usecond
    Cm      = Cm_F   * farad
    Ra      = Ra_ohm * ohm
    Rm_hi   = Rm_hi_ohm * ohm
    Vthresh = Vthresh_V * volt
    Vreset  = Vreset_V  * volt
    t_ref   = t_ref_s   * second
    I_0     = I_single_A * amp

    eqs = '''
    dVm/dt = (-Vm / (Rm_hi + Ra) + I_0) / Cm : volt
    '''
    grp = NeuronGroup(1, eqs, threshold='Vm > Vthresh', reset='Vm = Vreset',
                      refractory=t_ref, method='euler')
    grp.Vm = Vreset
    sp_mon = SpikeMonitor(grp)
    st_mon = StateMonitor(grp, 'Vm', record=True, dt=REC_DT_MS * ms)
    run(dur_single_s * second)

    results_single[Ra_ohm] = {
        't_ms'    : np.array(st_mon.t / ms),
        'Vm_V'    : np.array(st_mon.Vm[0] / volt),
        'sp_ms'   : np.array(sp_mon.t / ms),
        'tau_m_ms': tau_m * 1e3,
        'Vm_ss_V' : Vm_ss,
        'rheo_uA' : Vthresh_V / Rtot * 1e6,
    }

    # — Tonic spiking ——————————————————————————————————————————————————————
    start_scope()
    defaultclock.dt = DT_US * usecond
    Cm      = Cm_F   * farad
    Ra      = Ra_ohm * ohm
    Rm_hi   = Rm_hi_ohm * ohm
    Vthresh = Vthresh_V * volt
    Vreset  = Vreset_V  * volt
    t_ref   = t_ref_s   * second
    I_0     = I_tonic_A * amp

    grp = NeuronGroup(1, eqs, threshold='Vm > Vthresh', reset='Vm = Vreset',
                      refractory=t_ref, method='euler')
    grp.Vm = Vreset
    sp_mon = SpikeMonitor(grp)
    st_mon = StateMonitor(grp, 'Vm', record=True, dt=REC_DT_MS * ms)
    run(DUR_TONIC * ms)

    n_sp = len(sp_mon.t)
    results_tonic[Ra_ohm] = {
        't_ms' : np.array(st_mon.t / ms),
        'Vm_V' : np.array(st_mon.Vm[0] / volt),
        'sp_ms': np.array(sp_mon.t / ms),
        'n_sp' : n_sp,
        'f_Hz' : n_sp / (DUR_TONIC * 1e-3),
    }
    print(f"  Ra={Ra_ohm/1e3:.2g} kΩ: "
          f"rheobase={results_single[Ra_ohm]['rheo_uA']:.1f} µA  "
          f"τ_m={results_single[Ra_ohm]['tau_m_ms']:.2f} ms  "
          f"f(70µA)={results_tonic[Ra_ohm]['f_Hz']:.1f} Hz")

# ── Analytical I-F curves ─────────────────────────────────────────────────────
I_sweep_uA = np.linspace(0, 200, 2000)
IF_curves  = {}
for Ra_ohm in Ra_values:
    Rtot   = Rm_hi_ohm + Ra_ohm
    tau_m  = Cm_F * Rtot
    Vm_ss  = I_sweep_uA * 1e-6 * Rtot
    f_arr  = np.zeros_like(I_sweep_uA)
    mask   = Vm_ss > Vthresh_V
    tc     = -tau_m * np.log(1 - Vthresh_V / Vm_ss[mask])
    f_arr[mask] = 1.0 / (tc + t_ref_s)
    IF_curves[Ra_ohm] = f_arr

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig,
                        hspace=0.55, wspace=0.38,
                        height_ratios=[1.0, 1.2, 1.0])

ax_spike = fig.add_subplot(gs[0, :])    # full-width: single spike shape
ax_tonic = fig.add_subplot(gs[1, :])    # full-width: tonic traces (offset)
ax_if    = fig.add_subplot(gs[2, :])    # full-width: I-F curves

# ── Panel 1: Single spike shape ───────────────────────────────────────────────
for i, Ra_ohm in enumerate(Ra_values):
    r  = results_single[Ra_ohm]
    t  = r['t_ms']
    Vm = r['Vm_V']
    sp = r['sp_ms']
    tau_m = r['tau_m_ms']
    Vss   = r['Vm_ss_V']

    label = (f"Ra={Ra_ohm/1e3:.2g} kΩ  "
             f"τ_m={tau_m:.1f} ms  "
             f"Vm_ss={Vss:.2f} V")
    ax_spike.plot(t, Vm, color=colors[i], lw=2.0, label=label)

    # mark spike time
    if len(sp):
        ax_spike.axvline(sp[0], color=colors[i], ls=':', lw=1.0, alpha=0.6)

ax_spike.axhline(Vthresh_V, color='k', ls='--', lw=1.4,
                 label=f'Vthresh = {Vthresh_V:.0f} V')
ax_spike.set_xlabel('Time  (ms)', fontsize=11)
ax_spike.set_ylabel('Vm  (V)', fontsize=11)
ax_spike.set_title(
    f'Single Spike Shape — Ra sweep  '
    f'(I₀ = {I_single_A*1e6:.0f} µA,  Cm = {Cm_F*1e9:.1f} nF,  Rm_hi = {Rm_hi_ohm/1e3:.0f} kΩ)\n'
    r'$C_m \dot{V}_m = -V_m/(R_{m,hi}+R_a) + I_0$    '
    r'Larger $R_a$ → higher $V_{ss}$ → faster approach to threshold',
    fontsize=11, fontweight='bold')
ax_spike.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax_spike.grid(axis='y', alpha=0.25)
ax_spike.set_xlim(left=0)

# ── Panel 2: Tonic firing (vertically offset per trace) ───────────────────────
OFFSET = Vthresh_V * 1.7   # V between traces
offset_V = 0.0

for i, Ra_ohm in enumerate(Ra_values):
    r  = results_tonic[Ra_ohm]
    t  = r['t_ms']
    Vm = r['Vm_V'] + offset_V
    n  = r['n_sp']
    f  = r['f_Hz']

    label = f"Ra={Ra_ohm/1e3:.2g} kΩ  →  {n} spikes @ {f:.0f} Hz"
    ax_tonic.plot(t, Vm, color=colors[i], lw=1.2, label=label)
    ax_tonic.axhline(Vthresh_V + offset_V,
                     color=colors[i], ls='--', lw=0.7, alpha=0.5)
    ax_tonic.text(DUR_TONIC + 2, Vthresh_V + offset_V - 0.3,
                  f'{f:.0f} Hz', fontsize=8, color=colors[i], va='top')
    offset_V += OFFSET

ax_tonic.set_xlabel('Time  (ms)', fontsize=11)
ax_tonic.set_ylabel('Vm  (V)  [traces offset]', fontsize=11)
ax_tonic.set_title(
    f'Tonic Firing — Ra sweep  '
    f'(I₀ = {I_tonic_A*1e6:.0f} µA,  duration = {DUR_TONIC} ms)\n'
    r'Higher $R_a$ → higher $V_{ss}/V_{thr}$ ratio → shorter ISI → higher firing rate',
    fontsize=11, fontweight='bold')
ax_tonic.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax_tonic.set_xlim(0, DUR_TONIC + 10)

# ── Panel 3: Analytical I-F curves ───────────────────────────────────────────
for i, Ra_ohm in enumerate(Ra_values):
    f_arr = IF_curves[Ra_ohm]
    rheo  = results_single[Ra_ohm]['rheo_uA']
    ax_if.plot(I_sweep_uA, f_arr / 1, color=colors[i], lw=2.0,
               label=f"Ra={Ra_ohm/1e3:.2g} kΩ  (rheobase≈{rheo:.0f} µA)")

ax_if.axvline(I_single_A * 1e6, color='gray', ls=':', lw=1.2,
              label=f'I₀ = {I_single_A*1e6:.0f} µA  (single spike panel)')
ax_if.axvline(I_tonic_A * 1e6,  color='k',    ls=':', lw=1.2,
              label=f'I₀ = {I_tonic_A*1e6:.0f} µA  (tonic panel)')
ax_if.set_xlabel('Drive current  I₀  (µA)', fontsize=11)
ax_if.set_ylabel('Firing rate  f  (Hz)', fontsize=11)
ax_if.set_title(
    'I-F Curves (analytical)  —  Ra sweep\n'
    r'$f = \left[t_{ref} - \tau_m \ln\!\left(1 - V_{thr}/(I_0(R_{m,hi}+R_a))\right)\right]^{-1}$'
    r'    Higher $R_a$ → lower rheobase & higher f for same $I_0$',
    fontsize=11, fontweight='bold')
ax_if.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax_if.set_xlim(0, 200)
ax_if.set_ylim(0)
ax_if.grid(axis='y', alpha=0.25)

# ── Colorbar (Ra scale) ───────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(vmin=Ra_values[0]/1e3, vmax=Ra_values[-1]/1e3))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_spike, ax_tonic, ax_if],
                    orientation='vertical', fraction=0.015, pad=0.01)
cbar.set_label('Ra  (kΩ)', fontsize=10)

fig.suptitle(
    'Memristor aLIF Neuron — Effect of Axon-Hillock Resistance Ra\n'
    rf'Cm = {Cm_F*1e9:.1f} nF,  t_ref = {t_ref_s*1e3:.2f} ms,  '
    rf'Rm_hi = {Rm_hi_ohm/1e3:.0f} kΩ  (quiescent state)',
    fontsize=13, fontweight='bold', y=1.01)

out_path = '/home/haoran/Projects/Brain2simulator/spike_Ra_sweep.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
