"""
Memristor aLIF  vs  Thyristor Hardware — Model Comparison
==========================================================

Three parameter sets:
  SET A — aLIF abstract model (corrected: f(40µA)=70Hz, f(100µA)=200Hz)
  SET B — Thyristor hardware WORKING FIT  (scaled for simulation)
  SET C — Thyristor hardware ACTUAL components (C=10µF, Ra=49Ω, Rg=100kΩ)

Key structural differences uncovered:
  1. TOPOLOGY  : aLIF uses SERIES  Rm_hi + Ra  (voltage divider → Vpost)
                 Thyristor uses PARALLEL  ga ∥ gg  (conductance-based LIF)
  2. RESET     : aLIF  Vreset = 0 V  (abstract)
                 Thyristor  Vr = IH/ga ≠ 0  (physical holding-current latch)
  3. THRESHOLD : aLIF  Vthresh fitted to I-F targets
                 Thyristor  VT = IT/gg  (gate threshold current / leak conductance)
  4. f_max     : aLIF  1/t_ref = 333 Hz
                 Thyristor  1/tn  = 16.7 Hz   (50× slower — dominated by tn=60ms)
  5. tau_m     : aLIF  4.34 ms  vs  Thyristor  0.72 ms  (membrane charges faster)
  6. tau_s     : aLIF  10 ms  vs  Thyristor  60 ms  (synaptic filter 6× slower)
"""

import numpy as np
import builtins
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from brian2 import *

# ══════════════════════════════════════════════════════════════════════════════
# SET A  — aLIF corrected  (paper model, Eqs. 9–12)
# ══════════════════════════════════════════════════════════════════════════════
# Membrane  (SERIES topology: Rm_hi + Ra)
Cm_A      = 40.15e-9 * farad   # membrane capacitance
Ra_A      = 10e3     * ohm     # axon-hillock resistor
Rm_hi_A   = 98e3     * ohm     # memristor HIGH resistance (open switch)
Rm_lo_A   = 100      * ohm     # memristor LOW  resistance (closed switch)
tau_m_A   = Cm_A * (Rm_hi_A + Ra_A)   # 4.336 ms — membrane τ (explicit!)
# Spiking
Vth_A     = 4.0      * volt    # spike threshold  (fitted to I-F targets)
Vr_A      = 0.0      * volt    # reset (abstract zero reset)
t_ref_A   = 3e-3     * second  # refractory → f_max = 333 Hz
# Synapse  (Rs·Cs time constants)
Rs_A      = 10e3     * ohm
Cs_A      = 1000e-9  * farad
tau_s1_A  = Rs_A * Cs_A        # 10 ms — 1st leaky-integration stage
tau_s2_A  = Rs_A * Cs_A        # 10 ms — 2nd leaky-integration stage
Iw_A      = 150e-6   * amp     # synaptic weight (Iw in Eq. 11)
# I-F operating range
I_min_A   = 40e-6    * amp     # → 70  Hz
I_max_A   = 100e-6   * amp     # → 200 Hz  (depol-block onset)

# ══════════════════════════════════════════════════════════════════════════════
# SET B  — Thyristor hardware WORKING FIT  (your reference code)
# ══════════════════════════════════════════════════════════════════════════════
# Membrane  (PARALLEL topology: ga ∥ gg)
C_B       = 3.3e-7   * farad   # membrane capacitance
ga_B      = 1/2.2e3  * siemens # axon-hillock conductance   Ra = 2.2 kΩ
gg_B      = 1/680e3  * siemens # membrane leak conductance  Rg = 680 kΩ
tau_m_B   = C_B / (ga_B + gg_B)          # 0.724 ms — membrane τ
# Thyristor-derived thresholds
IT_B      = 4.95e-6  * amp     # gate threshold current
IH_B      = 105e-6   * amp     # holding current
VT_B      = IT_B / gg_B        # 3.366 V — threshold (= IT/gg)
Vr_B      = IH_B / ga_B        # 0.231 V — reset    (= IH/ga, NON-ZERO!)
V0_B      = 1e-4     * volt    # thyristor sensitivity
tn_B      = 60e-3    * second  # gate time scale → t_ref analogue
Id_B      = 200e-6   * amp     # spike current delta (= Iw)

# ══════════════════════════════════════════════════════════════════════════════
# SET C  — Thyristor ACTUAL component values (commented block in your code)
# ══════════════════════════════════════════════════════════════════════════════
C_C       = 10e-6    * farad   # actual PCB capacitor
ga_C      = 1/(2*49) * siemens # Ra = 2×49 = 98 Ω  (×2 fitting factor)
gg_C      = 1/100e3  * siemens # Rg = 100 kΩ
tau_m_C   = C_C / (ga_C + gg_C)
IT_C      = 2.6e-6   * amp
IH_C      = 120e-6   * amp
VT_C      = IT_C / gg_C        # 0.260 V
Vr_C      = IH_C / ga_C        # 0.01176 V = 11.76 mV
Id_C      = 200e-6   * amp
tau_C     = 60e-3    * second  # synapse time constant

# ══════════════════════════════════════════════════════════════════════════════
# Print parameter comparison table
# ══════════════════════════════════════════════════════════════════════════════
def fmt(val, unit='', scale=1, decimals=3):
    return f"{val*scale:.{decimals}f} {unit}"

print("=" * 80)
print("  PARAMETER COMPARISON")
print("=" * 80)
rows = [
    ("Topology",         "SERIES Rm+Ra",             "PARALLEL ga∥gg",           "PARALLEL ga∥gg"),
    ("Capacitance",      f"{Cm_A/nF:.1f} nF",         f"{C_B/nF:.0f} nF",          f"{C_C/uF:.0f} µF"),
    ("Ra (axon-hillock)",f"{Ra_A/kohm:.0f} kΩ",        f"{1/float(ga_B/siemens):.0f} Ω", f"{1/float(ga_C/siemens):.0f} Ω"),
    ("Rg / Rm_hi",       f"{Rm_hi_A/kohm:.0f} kΩ",    f"{1/float(gg_B/siemens)/1e3:.0f} kΩ", f"{1/float(gg_C/siemens)/1e3:.0f} kΩ"),
    ("tau_m  ← EXPLICIT",f"{tau_m_A/ms:.3f} ms",      f"{tau_m_B/ms:.3f} ms",      f"{tau_m_C/ms:.3f} ms"),
    ("Vthreshold",        f"{float(Vth_A/volt):.3f} V (fitted)",f"{float(VT_B/volt):.3f} V = IT/gg",f"{float(VT_C/volt):.3f} V = IT/gg"),
    ("Vreset",            f"{float(Vr_A/volt):.3f} V (abstract)", f"{float(Vr_B/volt)*1e3:.1f} mV = IH/ga", f"{float(Vr_C/volt)*1e3:.2f} mV = IH/ga"),
    ("IT (gate thresh.)", "—",                         f"{float(IT_B/amp)*1e6:.2f} µA", f"{float(IT_C/amp)*1e6:.1f} µA"),
    ("IH (holding curr.)", "—",                        f"{float(IH_B/amp)*1e6:.0f} µA", f"{float(IH_C/amp)*1e6:.0f} µA"),
    ("t_ref / tn",        f"{float(t_ref_A/ms):.0f} ms",f"{float(tn_B/ms):.0f} ms", "n/a"),
    ("f_max = 1/t_ref",   f"{1/float(t_ref_A/second):.0f} Hz", f"{1/float(tn_B/second):.1f} Hz","—"),
    ("tau_s1 (synapse 1)",f"{float(tau_s1_A/ms):.0f} ms", f"{float(tn_B/ms):.0f} ms (tn)", f"{float(tau_C/ms):.0f} ms (tau)"),
    ("tau_s2 (synapse 2)",f"{float(tau_s2_A/ms):.0f} ms", f"{float(tn_B/ms):.0f} ms (tn)", f"{float(tau_C/ms):.0f} ms (tau)"),
    ("Iw / Id",           f"{float(Iw_A/amp)*1e6:.0f} µA", f"{float(Id_B/amp)*1e6:.0f} µA", f"{float(Id_C/amp)*1e6:.0f} µA"),
]
hdr = f"  {'Parameter':<26} {'SET A  aLIF (abstract)':<26} {'SET B  Thyristor fit':<26} SET C  Thyristor actual"
print(hdr); print("-"*80)
for r in rows:
    mark = " ◄" if "EXPLICIT" in r[0] or "Vreset" in r[0] or "topology" in r[0].lower() else ""
    print(f"  {r[0]:<26} {r[1]:<26} {r[2]:<26} {r[3]}{mark}")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# Analytical I-F curves
# ══════════════════════════════════════════════════════════════════════════════
def f_aLIF_IF(I_arr, tau_m, R_total, Vth, Vreset, t_ref, I_max_block=np.inf):
    """aLIF I-F: series topology, threshold Vth, reset Vreset."""
    f_out = np.zeros_like(I_arr)
    for k, I0 in enumerate(I_arr):
        if I0 > I_max_block:
            continue   # depolarisation block
        Vm_ss = I0 * R_total
        if Vm_ss <= Vth:
            continue
        tc = -tau_m * np.log(1 - (Vth - Vreset) / (Vm_ss - Vreset))
        f_out[k] = 1.0 / (tc + t_ref)
    return f_out

def f_thyristor_IF(I_arr, tau_m, ga, gg, VT, Vr, t_ref):
    """Thyristor I-F: parallel topology, reset to Vr."""
    f_out = np.zeros_like(I_arr)
    for k, I0 in enumerate(I_arr):
        Vm_ss = I0 / (ga + gg)
        if Vm_ss <= VT:
            continue
        tc = tau_m * np.log((Vm_ss - Vr) / (Vm_ss - VT))
        f_out[k] = 1.0 / (tc + t_ref)
    return f_out

# Sweep ranges (different scales for each model)
I_A_sweep = np.linspace(0, 160e-6, 2000)
I_B_sweep = np.linspace(0, 8e-3,   2000)

fA = f_aLIF_IF(I_A_sweep,
               float(tau_m_A/second),
               float((Rm_hi_A+Ra_A)/ohm),
               float(Vth_A/volt),
               float(Vr_A/volt),
               float(t_ref_A/second),
               I_max_block=float(I_max_A/amp))

fB = f_thyristor_IF(I_B_sweep,
                    float(tau_m_B/second),
                    float(ga_B/siemens),
                    float(gg_B/siemens),
                    float(VT_B/volt),
                    float(Vr_B/volt),
                    float(tn_B/second))

# ══════════════════════════════════════════════════════════════════════════════
# Brian2 simulations for spike shape comparison
# ══════════════════════════════════════════════════════════════════════════════

# ── SET A simulation (aLIF, I0=55µA for ~127Hz) ──────────────────────────────
start_scope()
defaultclock.dt = 0.01*ms

eqs_A = '''
dVm/dt = (-Vm/(Rm_hi_A+Ra_A) + I_inj) / Cm_A : volt
I_inj  : amp (shared)
'''
ng_A = NeuronGroup(1, eqs_A, threshold='Vm > Vth_A',
                   reset='Vm = Vr_A', refractory=t_ref_A, method='euler')
ng_A.Vm     = 0*volt
ng_A.I_inj  = 55e-6*amp
sm_A = StateMonitor(ng_A, 'Vm', record=True, dt=0.01*ms)
sp_A = SpikeMonitor(ng_A)
run(200*ms)
t_A_ms  = sm_A.t / ms
vm_A_V  = sm_A.Vm[0] / volt

# ── SET B simulation (Thyristor fit) ─────────────────────────────────────────
# Membrane equation: C dVm/dt = -(ga+gg)*Vm + I_syn
# Reset to Vr_B (NON-ZERO — the crucial difference)
# t_ref = tn_B = 60ms,  I0 chosen to give ~10Hz (just above rheobase)
start_scope()
defaultclock.dt = 0.1*ms

# Rheobase for SET B
I_rheo_B = float(VT_B/volt) * (float(ga_B/siemens) + float(gg_B/siemens))
I_drive_B = I_rheo_B * 1.5  # 50% above rheobase

eqs_B = '''
dVm/dt = (-(ga_B+gg_B)*Vm + I_inj) / C_B : volt
I_inj  : amp (shared)
'''
ng_B = NeuronGroup(1, eqs_B, threshold='Vm > VT_B',
                   reset='Vm = Vr_B', refractory=tn_B, method='euler')
ng_B.Vm     = float(Vr_B/volt)*volt
ng_B.I_inj  = I_drive_B*amp
sm_B = StateMonitor(ng_B, 'Vm', record=True, dt=0.1*ms)
sp_B = SpikeMonitor(ng_B)
run(1500*ms)
t_B_ms  = sm_B.t / ms
vm_B_V  = sm_B.Vm[0] / volt

fA_drive = len(sp_A.t)/0.2
fB_drive = len(sp_B.t)/1.5
print(f"\nSET A simulation: I0=55µA → {fA_drive:.0f} Hz  (t_ref=3ms)")
print(f"SET B simulation: I0={I_drive_B*1e3:.2f}mA → {fB_drive:.1f} Hz  (tn=60ms)")

# ── Synaptic current: one spike, compare tau_s ────────────────────────────────
t_spike_ms = 20.0
t_syn_ms   = np.linspace(0, 300, 6000)  # ms after spike
t_rel      = (t_syn_ms - t_spike_ms) / 1e3  # seconds, spike at t=0

def alpha_fn(t_rel, Iw, tau_s):
    out = np.zeros_like(t_rel)
    m = t_rel > 0
    out[m] = (Iw / tau_s) * t_rel[m] * np.exp(-t_rel[m] / tau_s)
    return out

Is2_A_syn = alpha_fn(t_rel, float(Iw_A/amp), float(tau_s2_A/second))
Is2_B_syn = alpha_fn(t_rel, float(Id_B/amp), float(tn_B/second))

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
C_setA = '#2980B9'   # blue
C_setB = '#E74C3C'   # red
C_setC = '#27AE60'   # green
C_ana  = 'k'

fig = plt.figure(figsize=(22, 24))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.60, wspace=0.40,
                        height_ratios=[1.4, 1.4, 1.4, 1.2])

# ─────────────────────────────────────────────────────────────────────────────
# Row 0: Circuit topology diagrams  (text-based circuit sketches)
# ─────────────────────────────────────────────────────────────────────────────
ax_cA = fig.add_subplot(gs[0, 0])
ax_cB = fig.add_subplot(gs[0, 1])

for ax in (ax_cA, ax_cB):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

# SET A circuit description
cA_lines = [
    r"$\bf{SET\ A — aLIF\ (Series\ topology)}$",
    "",
    r"   Cm  $\frac{dV_m}{dt} = \frac{-V_m}{R_m[S]+R_a} + I_{s2} + I_0$",
    "",
    r"   Circuit:  $V_m$ → [ $R_m[S]$ ] → [ $R_a$ ] → GND",
    r"   $V_{post}=V_m\cdot\frac{R_a}{R_m[S]+R_a}$  (voltage divider)",
    "",
    r"   $R_m[hi]=98\ k\Omega \gg R_a=10\ k\Omega$  →  $V_{post}\approx0$  (silent)",
    r"   $R_m[lo]=100\ \Omega   \ll R_a$              →  $V_{post}\approx V_m$  (spike)",
    "",
    rf"   $\tau_m = C_m(R_m[hi]+R_a) = {float(tau_m_A/ms):.2f}\ ms$",
    rf"   $V_{{thr}} = {float(Vth_A/volt):.1f}\ V$  (fitted)",
    rf"   $V_{{reset}} = 0\ V$  (abstract)",
    rf"   $t_{{ref}} = {float(t_ref_A/ms):.0f}\ ms$  →  $f_{{max}}={1/float(t_ref_A/second):.0f}\ Hz$",
    rf"   $\tau_{{s1}}=\tau_{{s2}}=R_sC_s={float(tau_s1_A/ms):.0f}\ ms$",
]
for i, line in enumerate(cA_lines):
    ax_cA.text(0.05, 0.95 - i*0.065, line, transform=ax_cA.transAxes,
               fontsize=9, va='top', family='monospace' if line.startswith(' ') else 'sans-serif')
ax_cA.set_title('SET A — aLIF Abstract Model  (paper Eqs. 9–12)',
                fontsize=10, fontweight='bold', color=C_setA)
ax_cA.patch.set_facecolor('#EBF5FB')
ax_cA.patch.set_alpha(0.5)

# SET B circuit description
cB_lines = [
    r"$\bf{SET\ B — Thyristor\ Hardware\ (Parallel\ topology)}$",
    "",
    r"   $C\frac{dV_m}{dt} = -(g_a + g_g)\cdot V_m + I_{syn}$",
    "",
    r"   Circuit:  $V_m$ ← $C$ ← [ $g_a = 1/R_a$ ] ∥ [ $g_g = 1/R_g$ ] → GND",
    r"   Thyristor fires when $V_m > V_T = I_T/g_g$",
    "",
    r"   Key: HOLDING CURRENT $I_H$ latches thyristor ON until",
    r"   $V_m < V_r = I_H/g_a$  →  NON-ZERO reset voltage!",
    "",
    rf"   $\tau_m = C/(g_a+g_g) \approx C\cdot R_a = {float(tau_m_B/ms):.3f}\ ms$",
    rf"   $V_T = I_T/g_g = {float(VT_B/volt):.3f}\ V$  (physical)",
    rf"   $V_r = I_H/g_a = {float(Vr_B/volt)*1e3:.1f}\ mV$  (NON-ZERO!)",
    rf"   $t_n = {float(tn_B/ms):.0f}\ ms$  →  $f_{{max}}={1/float(tn_B/second):.1f}\ Hz$",
    rf"   $\tau_{{syn}} = t_n = {float(tn_B/ms):.0f}\ ms$  (6× slower than SET A)",
]
for i, line in enumerate(cB_lines):
    ax_cB.text(0.05, 0.95 - i*0.065, line, transform=ax_cB.transAxes,
               fontsize=9, va='top', family='monospace' if line.startswith(' ') else 'sans-serif')
ax_cB.set_title('SET B — Thyristor Hardware (Working Fit)',
                fontsize=10, fontweight='bold', color=C_setB)
ax_cB.patch.set_facecolor('#FDEDEC')
ax_cB.patch.set_alpha(0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1: Simulated Vm traces
# ─────────────────────────────────────────────────────────────────────────────
ax_vA = fig.add_subplot(gs[1, 0])
ax_vB = fig.add_subplot(gs[1, 1])

# SET A — show 200ms
ax_vA.plot(t_A_ms, vm_A_V, color=C_setA, lw=0.7, label='Vm(t)')
ax_vA.axhline(float(Vth_A/volt), color='dimgray', ls='--', lw=1.1,
              label=f'Vthresh = {float(Vth_A/volt):.1f} V')
ax_vA.axhline(float(Vr_A/volt), color='green', ls=':', lw=1.0,
              label=f'Vreset = {float(Vr_A/volt):.1f} V')
for ts in sp_A.t/ms:
    ax_vA.vlines(ts, float(Vth_A/volt), float(Vth_A/volt)+0.3,
                 colors='k', lw=0.9, zorder=5)
ax_vA.set_xlim(0, 200)
ax_vA.set_title(f'SET A — Vm(t)   I₀=55 µA   f={fA_drive:.0f} Hz\n'
                rf'$\tau_m={float(tau_m_A/ms):.2f}$ ms,  '
                rf'$t_{{ref}}={float(t_ref_A/ms):.0f}$ ms,  '
                rf'$V_{{reset}}=0$ V',
                fontsize=9, fontweight='bold')
ax_vA.set_ylabel('Vm  (V)'); ax_vA.set_xlabel('Time (ms)')
ax_vA.legend(fontsize=8, loc='upper right')

# SET B — show 1500ms
ax_vB.plot(t_B_ms, vm_B_V, color=C_setB, lw=0.8, label='Vm(t)')
ax_vB.axhline(float(VT_B/volt), color='dimgray', ls='--', lw=1.1,
              label=f'VT = IT/gg = {float(VT_B/volt):.3f} V')
ax_vB.axhline(float(Vr_B/volt), color='green', ls=':', lw=1.2,
              label=f'Vr = IH/ga = {float(Vr_B/volt)*1e3:.1f} mV  ← non-zero!')
for ts in sp_B.t/ms:
    ax_vB.vlines(ts, float(VT_B/volt), float(VT_B/volt)+0.15,
                 colors='k', lw=0.9, zorder=5)
ax_vB.set_xlim(0, 1500)
ax_vB.set_title(f'SET B — Vm(t)   I₀={I_drive_B*1e3:.2f} mA   f={fB_drive:.1f} Hz\n'
                rf'$\tau_m={float(tau_m_B/ms):.3f}$ ms,  '
                rf'$t_n={float(tn_B/ms):.0f}$ ms,  '
                rf'$V_r=I_H/g_a={float(Vr_B/volt)*1e3:.1f}$ mV',
                fontsize=9, fontweight='bold')
ax_vB.set_ylabel('Vm  (V)'); ax_vB.set_xlabel('Time (ms)')
ax_vB.legend(fontsize=8, loc='upper right')

# ─────────────────────────────────────────────────────────────────────────────
# Row 2: Synaptic current shape comparison
# ─────────────────────────────────────────────────────────────────────────────
ax_sA = fig.add_subplot(gs[2, 0])
ax_sB = fig.add_subplot(gs[2, 1])

# SET A synaptic current (Is2 alpha, tau_s=10ms)
ax_sA.fill_between(t_syn_ms, Is2_A_syn*1e6, alpha=0.2, color=C_setA)
ax_sA.plot(t_syn_ms, Is2_A_syn*1e6, color=C_setA, lw=1.6,
           label=rf'Is2(t) — alpha fn  $\tau_s={float(tau_s2_A/ms):.0f}$ ms')
ax_sA.axvline(t_spike_ms, color='red', ls=':', lw=1.2, label='Pre-syn spike')
t_peak_A = t_spike_ms + float(tau_s2_A/ms)
pk_A = float(Iw_A/amp) / float(tau_s2_A/second) * float(tau_s2_A/second) * np.exp(-1) * 1e6
ax_sA.plot(t_peak_A, pk_A, 'o', ms=7, color=C_setA)
ax_sA.annotate(f'Peak {pk_A:.1f} µA\n@ t=τ_s={float(tau_s2_A/ms):.0f} ms',
               xy=(t_peak_A, pk_A), xytext=(t_peak_A+30, pk_A-8),
               fontsize=8, arrowprops=dict(arrowstyle='->', color='k'))
ax_sA.set_title('SET A — Synaptic Current Is2(t)\n'
                rf'$I_w={float(Iw_A/amp)*1e6:.0f}\ \mu A$,  '
                rf'$\tau_{{s}}=R_sC_s={float(tau_s2_A/ms):.0f}$ ms  →  fast AMPA-like',
                fontsize=9, fontweight='bold')
ax_sA.set_ylabel('Is2  (µA)'); ax_sA.set_xlabel('Time (ms)')
ax_sA.set_xlim(0, 300); ax_sA.legend(fontsize=8)

# SET B synaptic current (alpha, tau_s=60ms)
ax_sB.fill_between(t_syn_ms, Is2_B_syn*1e6, alpha=0.2, color=C_setB)
ax_sB.plot(t_syn_ms, Is2_B_syn*1e6, color=C_setB, lw=1.6,
           label=rf'Is2(t) — alpha fn  $\tau_s=t_n={float(tn_B/ms):.0f}$ ms')
ax_sB.axvline(t_spike_ms, color='red', ls=':', lw=1.2, label='Pre-syn spike')
t_peak_B = t_spike_ms + float(tn_B/ms)
pk_B = float(Id_B/amp) / float(tn_B/second) * float(tn_B/second) * np.exp(-1) * 1e6
ax_sB.plot(t_peak_B, pk_B, 'o', ms=7, color=C_setB)
ax_sB.annotate(f'Peak {pk_B:.1f} µA\n@ t=τ_s={float(tn_B/ms):.0f} ms',
               xy=(t_peak_B, pk_B), xytext=(t_peak_B-80, pk_B+5),
               fontsize=8, arrowprops=dict(arrowstyle='->', color='k'))
ax_sB.set_title('SET B — Synaptic Current Is2(t)\n'
                rf'$I_d={float(Id_B/amp)*1e6:.0f}\ \mu A$,  '
                rf'$\tau_{{syn}}=t_n={float(tn_B/ms):.0f}$ ms  →  slow NMDA-like',
                fontsize=9, fontweight='bold')
ax_sB.set_ylabel('Is2  (µA)'); ax_sB.set_xlabel('Time (ms)')
ax_sB.set_xlim(0, 300); ax_sB.legend(fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Row 3 (full-width): I-F curve comparison
# ─────────────────────────────────────────────────────────────────────────────
ax_IF = fig.add_subplot(gs[3, :])

ax2 = ax_IF.twiny()   # second x-axis for SET B (mA scale)

# SET A on bottom x-axis (µA)
band_A = (I_A_sweep > 0) & (I_A_sweep <= float(I_max_A/amp))
blk_A  = I_A_sweep > float(I_max_A/amp)
lA1, = ax_IF.plot(I_A_sweep[band_A]*1e6, fA[band_A],
                  color=C_setA, lw=2.5, label='SET A — aLIF (µA, bottom axis)')
ax_IF.fill_between(I_A_sweep[band_A]*1e6, fA[band_A], alpha=0.12, color=C_setA)
ax_IF.plot(I_A_sweep[blk_A]*1e6, fA[blk_A], color=C_setA, lw=2.5, ls=':', alpha=0.5)
# Calibration markers
for I_cal, f_cal in [(40e-6, 70), (100e-6, 200)]:
    ax_IF.plot(I_cal*1e6, f_cal, 'o', ms=9, color=C_setA, zorder=7)
    ax_IF.annotate(f'{f_cal}Hz @ {I_cal*1e6:.0f}µA',
                   xy=(I_cal*1e6, f_cal), xytext=(I_cal*1e6+3, f_cal+8),
                   fontsize=8, color=C_setA)

# SET B on top x-axis (mA)
lB1, = ax2.plot(I_B_sweep*1e3, fB, color=C_setB, lw=2.5, ls='--',
                label='SET B — Thyristor fit (mA, top axis)')
ax2.fill_between(I_B_sweep*1e3, fB, alpha=0.10, color=C_setB)
ax2.set_xlabel('SET B Injected current  I₀  (mA)', fontsize=9, color=C_setB)
ax2.tick_params(colors=C_setB)
ax2.spines['top'].set_color(C_setB)

# Reference lines
ax_IF.axhline(1/float(t_ref_A/second), color=C_setA, ls=':', lw=1.0, alpha=0.7,
              label=f'SET A  f_max = {1/float(t_ref_A/second):.0f} Hz')
ax_IF.axhline(1/float(tn_B/second), color=C_setB, ls=':', lw=1.0, alpha=0.7,
              label=f'SET B  f_max = {1/float(tn_B/second):.1f} Hz')

# Depol block annotation
ax_IF.axvspan(float(I_max_A/amp)*1e6, 160, alpha=0.07, color='red')
ax_IF.text(float(I_max_A/amp)*1e6+1, 280, 'SET A\ndepol\nblock',
           fontsize=7.5, color='#C0392B',
           bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#E74C3C', alpha=0.9))

# Combined legend
lines = [lA1, lB1]
labels = [l.get_label() for l in lines]
ax_IF.legend(handles=lines + ax_IF.lines[2:6],
             labels=[str(l.get_label()) for l in lines + ax_IF.lines[2:6]],
             fontsize=8, loc='center right', framealpha=0.9)
ax_IF.set_xlabel('SET A Injected current  I₀  (µA)', fontsize=9, color=C_setA)
ax_IF.set_ylabel('Firing rate  f  (Hz)', fontsize=10)
ax_IF.tick_params(axis='x', colors=C_setA)
ax_IF.spines['bottom'].set_color(C_setA)
ax_IF.set_xlim(0, 160); ax_IF.set_ylim(0, 360)
ax_IF.set_title(
    'I-F Curve Comparison — SET A (aLIF) vs SET B (Thyristor fit)\n'
    r'SET A: $f=\left[t_{ref}-\tau_m\ln(1-V_{thr}/I_0 R_{tot})\right]^{-1}$   '
    r'SET B: $f=\left[t_n+\tau_m\ln\frac{V_{ss}-V_r}{V_{ss}-V_T}\right]^{-1}$,  '
    r'$V_{ss}=I_0/(g_a+g_g)$',
    fontsize=9, fontweight='bold')
ax_IF.grid(axis='y', alpha=0.25)

# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Memristor aLIF  vs  Thyristor Hardware — Full Model Comparison\n'
    'Blue = SET A (abstract aLIF, series topology, Vreset=0)     '
    'Red = SET B (thyristor fit, parallel topology, Vr=IH/ga≠0)',
    fontsize=12, fontweight='bold', y=1.002)

out_path = 'model_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")