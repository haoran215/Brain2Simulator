"""
ns_msn_v1.py
============
Memristive Spiking Neuron (MSN) — single-cell, paper-faithful spike shape.

Reference: Wu, Wang, Schneegans, Stoliar, Rozenberg.
"Bursting dynamics in a spiking neuron with a memristive voltage-gated
channel." Neuromorph. Comput. Eng. 3 044008 (2023).  Section 2, Fig. 1e, Fig. 2.

─── Inheritance from earlier files in this repo ─────────────────────────────
  ns_test.py (v5):
      Rm_S held FIXED at Rm_hi.  Spike timing produced by Brian2's
      threshold='Vm > Vthresh' / reset='Vm = Vreset' rule (instantaneous
      reset).  The simulation has no spike SHAPE — only spike timing.
      Cm and t_ref were solved from (I_min, f_min, I_max, f_max) targets.

  spike_Ra_sweep.py:
      Same model, swept Ra.  Confirmed that within a fixed-Rm model the
      spike "shape" is just the RC charging curve being yanked back to 0.
      No memristor commutation, no real spike.

  THIS FILE (ns_msn_v1.py):
      Rm_S becomes a STATE VARIABLE that hysteretically commutes between
      Rm_hi (open) and Rm_lo (closed).  The "spike" is the fast Cm
      discharge through Rm_lo + Ra during the closed phase — exactly the
      mechanism in Wu et al. Fig. 2.  No t_ref, no Vm reset rule;
      refractoriness emerges from the discharge time constant.

─── Memristor state machine (per Wu et al. §2) ──────────────────────────────
      s = 0  (open):    Rm_S = Rm_hi     (linear leak through ~100 kΩ)
      s = 1  (closed):  Rm_S = Rm_lo     (fast discharge through tens of Ω)

      open  → closed:   when V across M (≈ Vm here, since Rm_hi ≫ Ra)
                        exceeds the thyristor threshold Vth.
      closed → open:    when current through M drops below the holding
                        current I_hold.

  Brian2 wiring:
      threshold='Vm > Vth and s < 0.5'   → fires the close event;
                                           reset='s = 1' flips state.
                                           (Vm is NOT reset here.)
      events={'reopen': 'I_M < I_hold and s > 0.5'}
                                         → fires the reopen event;
                                           run_on_event flips s = 0.

  This makes the threshold "spike" event coincide with the moment Vout
  begins to rise — which is what downstream synapses should see.

─── Parameters: Wu et al. Fig. 2 (path A) ───────────────────────────────────
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

seed(42)
defaultclock.dt = 1*us              # fine grid: spike width is ~1 ms

# ─── Hardware parameters (Wu et al. 2023, Fig. 2) ────────────────────────────
Cm_F      = 10e-6        # F   membrane capacitor
Ra_ohm    = 47           # Ω   load resistor (paper "Rload")
Rm_hi_ohm = 100e3        # Ω   memristor open-state resistance
Rm_lo_ohm = 500          # Ω   memristor closed-state resistance
Vth_V     = 0.9          # V   thyristor close threshold (paper Fig. 2 inset)
I_hold_A  = 100e-6       # A   holding current (paper Fig. 2 inset)

I_in_A    = 92.4e-6      # A   constant drive (paper Fig. 2 caption)
T_run     = 300*ms       # long enough to capture several spikes

# ─── Brian2 quantities ────────────────────────────────────────────────────────
Cm     = Cm_F      * farad
Ra     = Ra_ohm    * ohm
Rm_hi  = Rm_hi_ohm * ohm
Rm_lo  = Rm_lo_ohm * ohm
Vth    = Vth_V     * volt
I_hold = I_hold_A  * amp
I_in   = I_in_A    * amp

# ─── Equations ────────────────────────────────────────────────────────────────
# Vm     : voltage at the top of Cm (≈ anode of M; cathode side sits across Ra)
# s      : 0 (open) or 1 (closed); controlled by threshold + custom event below
# Rm_S   : two-state memristor resistance  (subexpression of s)
# I_M    : current through the memristor branch
# Vout   : voltage across Ra — this is the externally-measured spike pulse
eqs = '''
dVm/dt = (I_in - Vm / (Rm_S + Ra)) / Cm        : volt
Rm_S   = (1 - s)*Rm_hi + s*Rm_lo                : ohm
I_M    = Vm / (Rm_S + Ra)                       : amp
Vout   = Vm * Ra / (Rm_S + Ra)                  : volt
s      : 1
'''

neurons = NeuronGroup(
    1, model=eqs,
    threshold='Vm > Vth and s < 0.5',     # open → closed (and emits a spike)
    reset='s = 1',                         # flip state; do NOT reset Vm
    events={'reopen': 'I_M < I_hold and s > 0.5'},
    method='euler',
)
neurons.run_on_event('reopen', 's = 0')

neurons.Vm = 0*volt
neurons.s  = 0                            # start open

# ─── Monitors ─────────────────────────────────────────────────────────────────
sp_mon = SpikeMonitor(neurons)
st_mon = StateMonitor(neurons, ['Vm', 'Vout', 's', 'Rm_S', 'I_M'],
                      record=True, dt=2*us)

print("=" * 64)
print("  MSN v1 — paper-faithful memristor spike shape")
print("=" * 64)
print(f"  Cm        = {Cm_F*1e6:.1f} µF")
print(f"  Ra        = {Ra_ohm:.0f} Ω    (load resistor / paper Rload)")
print(f"  Rm_hi     = {Rm_hi_ohm/1e3:.0f} kΩ")
print(f"  Rm_lo     = {Rm_lo_ohm:.0f} Ω")
print(f"  Vth       = {Vth_V:.2f} V")
print(f"  I_hold    = {I_hold_A*1e6:.0f} µA")
print(f"  I_in      = {I_in_A*1e6:.1f} µA")
print(f"  τ_open    = Cm·(Rm_hi+Ra) = {Cm_F*(Rm_hi_ohm+Ra_ohm)*1e3:.1f} ms")
print(f"  τ_closed  = Cm·(Rm_lo+Ra) = {Cm_F*(Rm_lo_ohm+Ra_ohm)*1e3:.2f} ms")
print(f"  Vm_ss(open)   = I_in·(Rm_hi+Ra) = {I_in_A*(Rm_hi_ohm+Ra_ohm):.2f} V")
print(f"  V_open_thresh = I_hold·(Rm_lo+Ra) = "
      f"{I_hold_A*(Rm_lo_ohm+Ra_ohm)*1e3:.1f} mV  "
      f"(Vm must drop below this to reopen)")
print("=" * 64)

print(f"\nRunning {T_run/ms:.0f} ms ...")
run(T_run, report='text')

n_sp = len(sp_mon.t)
print(f"\n  Spikes (close events): {n_sp}")
if n_sp >= 2:
    isi = np.diff(sp_mon.t/ms)
    print(f"  ISI: mean {isi.mean():.2f} ms,  std {isi.std():.2f} ms")
    print(f"  Mean firing rate: {1000/isi.mean():.1f} Hz")

# ─── Plot ─────────────────────────────────────────────────────────────────────
t_ms   = np.array(st_mon.t / ms)
Vm_V   = np.array(st_mon.Vm[0] / volt)
Vout_V = np.array(st_mon.Vout[0] / volt)
s_vec  = np.array(st_mon.s[0])
I_M_uA = np.array(st_mon.I_M[0] / uA)
Rm_S_a = np.array(st_mon.Rm_S[0] / ohm)
V_M_V  = Vm_V * Rm_S_a / (Rm_S_a + Ra_ohm)        # voltage across M only

fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30)

# (0,0) Vm trace (full duration)
ax = fig.add_subplot(gs[0, 0])
ax.plot(t_ms, Vm_V, 'C0', lw=0.7)
ax.axhline(Vth_V, color='k', ls='--', lw=1, label=f'Vth = {Vth_V:.2f} V')
ax.set_xlabel('t (ms)'); ax.set_ylabel('Vm (V)')
ax.set_title('Membrane voltage Vm  (across Cm)', fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# (0,1) Vout trace (full duration) — the paper's Fig. 2 trace
ax = fig.add_subplot(gs[0, 1])
ax.plot(t_ms, Vout_V*1e3, 'C2', lw=0.7)
ax.set_xlabel('t (ms)'); ax.set_ylabel('Vout (mV)')
ax.set_title('Vout = Vm·Ra/(Rm_S+Ra)   ←   paper Fig. 2 trace',
             fontweight='bold')

# (1,0) Memristor state s
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_ms, s_vec, 'C3', lw=0.7, drawstyle='steps-post')
ax.set_xlabel('t (ms)'); ax.set_ylabel('s   (0=open, 1=closed)')
ax.set_title('Memristor state s(t)', fontweight='bold')
ax.set_ylim(-0.1, 1.1)

# (1,1) Single-spike zoom (paper Fig. 2 right inset analogue)
ax = fig.add_subplot(gs[1, 1])
if n_sp > 0:
    t_sp = float(sp_mon.t[0]/ms)
    mask = (t_ms > t_sp - 1.0) & (t_ms < t_sp + 8.0)
    ax.plot(t_ms[mask] - t_sp, Vout_V[mask]*1e3, 'C2', lw=1.6)
    ax.axvline(0, color='gray', ls=':', lw=1, label='close event')
    ax.set_xlabel('t − t_spike (ms)'); ax.set_ylabel('Vout (mV)')
    ax.set_title(f'Single spike shape (zoom on first spike)',
                 fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
else:
    ax.text(0.5, 0.5, 'no spikes — increase T_run or I_in',
            ha='center', va='center', transform=ax.transAxes)

# (2,0) I_M (current through memristor branch)
ax = fig.add_subplot(gs[2, 0])
ax.plot(t_ms, I_M_uA, 'C4', lw=0.5)
ax.axhline(I_hold_A*1e6, color='k', ls='--', lw=1,
           label=f'I_hold = {I_hold_A*1e6:.0f} µA')
ax.set_xlabel('t (ms)'); ax.set_ylabel('I_M (µA)')
ax.set_title('Memristor current I_M', fontweight='bold')
ax.set_yscale('symlog', linthresh=10)
ax.legend(fontsize=9, loc='upper right')

# (2,1) I–V orbit of the memristor (paper Fig. 2 main panel analogue)
ax = fig.add_subplot(gs[2, 1])
ax.plot(V_M_V, I_M_uA*1e-3, color='C5', lw=0.4, alpha=0.6)
ax.set_xlabel('V across M  (V)'); ax.set_ylabel('I through M  (mA)')
ax.set_title('I–V orbit of M  (cf. paper Fig. 2 top-left inset)',
             fontweight='bold')

fig.suptitle(
    f'MSN v1 — paper-faithful memristor (Wu et al. 2023) | '
    f'I_in = {I_in_A*1e6:.1f} µA, {n_sp} spikes in {T_run/ms:.0f} ms'
    + (f', f̄ = {1000/np.diff(sp_mon.t/ms).mean():.1f} Hz' if n_sp >= 2 else ''),
    fontsize=12, fontweight='bold', y=0.995)

out_path = '/home/haoran/Projects/Brain2simulator/ns_msn_v1.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
