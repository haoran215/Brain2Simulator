"""
ns_msn_pacemaker.py
===================
Single-neuron self-excitation pacemaker / burster test.

Inheritance:
  demo/ns_msn_v3_bump.py — single MSN + one self-excitatory synapse (a single,
                           fading "bump").

  THIS FILE
  ─────────
  Two INDEPENDENT single-MSN experiments (one NeuronGroup of 2, every synapse a
  self-loop via 'i == j', so the neurons never talk to each other).  Each is
  kick-started by ONE 200 ms pulse and then left with NO external drive:

      N0  "pure self-excitation"            : one (strong) exc self-synapse
      N1  "self-exc + slow self-inhibition" : a (moderate) exc self-synapse for a
                                              standing drive, plus a faster-
                                              recovering inhibitory self-synapse
                                              (spike-frequency adaptation).

  Goal: test whether self-excitation can form a persistent pacemaker that fires
  periodic BURSTS with no ongoing external drive.

Hypothesis under test (user)
────────────────────────────
  A 200 ms pulse seeds an alpha synaptic current that integrates but stays just
  subthreshold; over ~tau_s it rises across the rheobase I_min and re-triggers a
  burst; each burst reloads the synaptic current for the next one -> a periodic,
  self-sustained pacemaker.  Membrane fast (tau_m = Cm*(Rm+Ra) ~ 6 ms with
  Cm = 0.1 uF); synapse slow (tau_s >> tau_m).

Two routes to bursting (what the two neurons show)
──────────────────────────────────────────────────
  N0  Pure positive feedback drives the cell up to the depolarisation-block
      ceiling (I_hold).  The block latch itself removes drive and terminates the
      burst -> periodic bursting pinned at I_hold.  So self-excitation ALONE can
      pace, using the hardware holding-current as the burst terminator.

  N1  Adaptation route: a slower self-inhibition builds during a burst, pushes
      the net drive below I_min, and terminates the burst BELOW the depol-block
      ceiling.  For the rhythm to sustain, inhibition must recover faster than
      excitation persists (tau_inh < tau_exc), so the gap ends and firing
      restarts.  Cleaner, lower-rate, more controllable bursts.

Note on signs (correction to the "-Is -> +Is" framing)
──────────────────────────────────────────────────────
  This is a CURRENT-based model; excitation is already +I_exc in dVm/dt:

      Cm dVm/dt = I_0 + I_exc - I_inh - Vm/(Rm(s)+Ra)

  There is no "-Is" to flip.  The positive feedback IS an excitatory self-synapse
  (kind='exc').  The only negative terms are the leak -Vm/(Rm+Ra) and inhibition.

Iterating on this file
──────────────────────
  All knobs are in the TUNABLES block.  Running prints a per-neuron VERDICT
  (fade / latch / tonic / periodic-bursting) with burst statistics and saves
  ns_msn_pacemaker.png.  Edit knobs, re-run, read the verdict.
"""
#%%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

prefs.codegen.target = 'numpy'   # pure-Python backend, avoids C++ compiler
defaultclock.dt = 20*us          # resolves tau_close (~221 us) with ~11 steps

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ TUNABLES  (edit + re-run; read the VERDICT printout)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
I0_frac    = 0.80      # I_0 = I0_frac * I_min   (subthreshold bias, both neurons)

# N0 — pure self-excitation (depol-block bursting).  6 uA gives the cleanest,
#      most regular rhythm (~6 spikes/burst, ~23 ms, CV ~0.14).  Lowering the
#      weight lengthens each burst (each spike nudges the drive over I_hold more
#      gently) but the rhythm turns irregular below ~5 uA (4 uA -> CV ~0.7); the
#      zoom figure below is the clean way to see the short-burst microstructure.
N0_Iw_exc  = 6e-6      # A
N0_tau_exc = 150e-3    # s   (>> tau_m ~ 6 ms)

# N1 — self-excitation (same strong drive as N0, so the depol-block ceiling
#      bounds it and prevents wind-up) + fast-recovering self-inhibition that
#      spaces the bursts out (spike-frequency adaptation).
N1_Iw_exc  = 6e-6      # A     same as N0
N1_tau_exc = 150e-3    # s     same as N0
N1_Iw_inh  = 5e-6      # A     carves gaps without extinguishing the drive
N1_tau_inh = 100e-3    # s     MUST be < N1_tau_exc so the gap can end -> restart

# Pulse + run
pulse_frac = 0.70      # I_pulse = pulse_frac * I_min  (added during the pulse)
t_pulse    = 0.5       # s   pulse onset (baseline settles first)
pulse_dur  = 0.2       # s   200 ms kick
T_run      = 8.0       # s   total simulation time

REC_DT     = 40*us     # neuron state recording resolution (fine enough to
                       # resolve individual ~0.5 ms spikes in the zoom panel)
GAP_ABS    = 30e-3     # s   min silent gap to call two clusters separate bursts
ZOOM_WIN   = (2.0, 2.5)  # s  window for the N0 burst-microstructure zoom figure

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Build                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
params = MSNParams()
print(params.summary())
I_min, I_max = params.operating_window()
I0_val    = I0_frac    * I_min
I_pulse   = pulse_frac * I_min

# One group of 2 neurons; every synapse is a self-loop, so N0 and N1 are two
# independent single-neuron experiments living in the same Network.  Brian2
# allows only ONE summed-writer per (inlet, group), so each self-loop gets its
# own named inlet; inlets with no writer for a given neuron simply stay 0.
G = make_msn(N=2, params=params,
             exc_inlets=('I_exc0', 'I_exc1'), inh_inlets=('I_inh1',),
             name='pacemaker')
G.I_0 = I0_val * amp

syn_exc0 = make_synapse(G, G,
    SynapseParams(kind='exc', weight=N0_Iw_exc, tau_s1=N0_tau_exc, tau_s2=N0_tau_exc,
                  target_var='I_exc0'),
    connect='i == j and i == 0', name='syn_exc0')      # N0 self-excitation
syn_exc1 = make_synapse(G, G,
    SynapseParams(kind='exc', weight=N1_Iw_exc, tau_s1=N1_tau_exc, tau_s2=N1_tau_exc,
                  target_var='I_exc1'),
    connect='i == j and i == 1', name='syn_exc1')      # N1 self-excitation
syn_inh1 = make_synapse(G, G,
    SynapseParams(kind='inh', weight=N1_Iw_inh, tau_s1=N1_tau_inh, tau_s2=N1_tau_inh,
                  target_var='I_inh1'),
    connect='i == j and i == 1', name='syn_inh1')      # N1 self-inhibition (slow)

# 200 ms step pulse on both neurons
@network_operation(when='start')
def apply_pulse(t):
    if t_pulse*second <= t < (t_pulse + pulse_dur)*second:
        G.I_0 = (I0_val + I_pulse) * amp
    else:
        G.I_0 = I0_val * amp

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Monitors                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sp_mon   = SpikeMonitor(G)
st_mon   = StateMonitor(G, ['Vm', 'Vout', 's'], record=True, dt=REC_DT)
exc0_mon = StateMonitor(syn_exc0, ['Is2'], record=True, dt=1*ms)   # edge0 -> N0
exc1_mon = StateMonitor(syn_exc1, ['Is2'], record=True, dt=1*ms)   # edge0 -> N1
inh1_mon = StateMonitor(syn_inh1, ['Is2'], record=True, dt=1*ms)   # edge0 -> N1

print(f"\nSetup")
print(f"  I_min = {I_min*1e6:.1f} uA   I_max = I_hold = {I_max*1e6:.0f} uA")
print(f"  I_0   = {I0_val*1e6:.2f} uA  ({I0_frac*100:.0f}% of I_min, subthreshold)")
print(f"  pulse = +{I_pulse*1e6:.2f} uA for {pulse_dur*1e3:.0f} ms "
      f"(total {(I0_val+I_pulse)*1e6:.1f} uA = {(I0_val+I_pulse)/I_min:.2f} x I_min)")
print(f"  N0 exc : Iw={N0_Iw_exc*1e6:.2f} uA  tau={N0_tau_exc*1e3:.0f} ms")
print(f"  N1 exc : Iw={N1_Iw_exc*1e6:.2f} uA  tau={N1_tau_exc*1e3:.0f} ms")
print(f"  N1 inh : Iw={N1_Iw_inh*1e6:.2f} uA  tau={N1_tau_inh*1e3:.0f} ms  "
      f"({'OK: tau_inh<tau_exc' if N1_tau_inh < N1_tau_exc else 'WARN: tau_inh>=tau_exc'})")
print()

run(T_run*second, report='text')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Analysis                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def detect_bursts(times_s, gap):
    """Split sorted spike times into clusters separated by gaps > `gap` (s)."""
    if len(times_s) == 0:
        return []
    bursts, cur = [], [times_s[0]]
    for t in times_s[1:]:
        if t - cur[-1] > gap:
            bursts.append(np.array(cur)); cur = [t]
        else:
            cur.append(t)
    bursts.append(np.array(cur))
    return bursts


def verdict(times_s, s_final, gap_abs):
    """Classify post-pulse activity of one neuron. Returns (label, stats)."""
    t_end = t_pulse + pulse_dur
    post  = times_s[times_s >= t_end]
    st = dict(n_post=len(post), n_bursts=0, spk_per_burst=0.0,
              ibi_mean=0.0, ibi_cv=float('nan'), rate=0.0, last=0.0)
    if len(post) == 0:
        return "FADE (silent right after pulse)", st
    st['last'] = float(post[-1])
    st['rate'] = len(post) / (T_run - t_end)

    # Adaptive gap: a real inter-burst silence is much longer than a typical
    # intra-burst ISI.  Merge on the larger of an absolute floor and 4x median ISI.
    isis = np.diff(post)
    gap  = max(gap_abs, 4.0 * np.median(isis)) if len(isis) else gap_abs
    bursts = detect_bursts(post, gap)
    st['n_bursts']      = len(bursts)
    st['spk_per_burst'] = float(np.mean([len(b) for b in bursts]))
    if len(bursts) >= 2:
        starts = np.array([b[0] for b in bursts])
        ibi = np.diff(starts)
        st['ibi_mean'] = float(np.mean(ibi))
        st['ibi_cv']   = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) else float('nan')

    # Did it survive to (near) the end?  A pacemaker must not die early.
    sustained = st['last'] >= 0.70 * T_run
    if not sustained:
        tag = "LATCH (depol block)" if s_final > 0.5 else "FADE"
        return f"{tag} — {len(bursts)} burst(s), silent from {st['last']:.1f}s", st

    multi = st['spk_per_burst'] >= 2.0
    if st['n_bursts'] >= 3 and st['ibi_cv'] < 0.35:
        return ("PERIODIC BURSTING" if multi
                else "TONIC pacemaker (regular single spikes)"), st
    if st['n_bursts'] >= 3:
        return "IRREGULAR bursting (sustained)", st
    return "TONIC (persistent firing, no gaps)", st


labels = ["N0 pure self-excitation", "N1 self-exc + slow self-inhibition"]
all_t  = np.array(sp_mon.t / second)
all_i  = np.array(sp_mon.i)
s_fin  = [float(st_mon.s[k][-1]) for k in range(2)]

print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
results = []
for k in range(2):
    tk = np.sort(all_t[all_i == k])
    lab, st = verdict(tk, s_fin[k], GAP_ABS)
    results.append((lab, st))
    print(f"\n{labels[k]}")
    print(f"  total spikes      : {int(np.sum(all_i == k))}")
    print(f"  after-pulse spikes: {st['n_post']}   (mean rate {st['rate']:.1f} Hz)")
    print(f"  bursts            : {st['n_bursts']}   "
          f"({st['spk_per_burst']:.1f} spikes/burst)")
    if st['n_bursts'] >= 2:
        print(f"  inter-burst intvl : {st['ibi_mean']*1e3:.0f} ms   "
              f"(CV {st['ibi_cv']:.2f})   -> {1/st['ibi_mean']:.2f} bursts/s")
    print(f"  last spike        : {st['last']:.2f} s / {T_run:.0f} s")
    print(f"  >> {lab}")
print("\n" + "=" * 74)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Plot                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
t_ms   = np.array(st_mon.t / ms)
te0_ms = np.array(exc0_mon.t / ms)
te1_ms = np.array(exc1_mon.t / ms)
ti1_ms = np.array(inh1_mon.t / ms)
tp0, tp1 = t_pulse * 1e3, (t_pulse + pulse_dur) * 1e3

def shade(ax):
    ax.axvspan(tp0, tp1, alpha=0.15, color='tomato', label='200 ms pulse')

fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True,
                         gridspec_kw=dict(hspace=0.42))

# (0) raster ------------------------------------------------------------------
ax = axes[0]
for k, col in zip((0, 1), ('C0', 'C3')):
    tk = all_t[all_i == k] * 1e3
    ax.vlines(tk, k - 0.35, k + 0.35, color=col, lw=0.8)
shade(ax)
ax.set_yticks([0, 1]); ax.set_yticklabels(['N0\npure exc', 'N1\nexc+inh'])
ax.set_ylim(-0.6, 1.6)
ax.set_ylabel('spikes')
ax.set_title(f"Spike raster   |   N0: {results[0][0]}    N1: {results[1][0]}",
             fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# (1) N0 Vout -----------------------------------------------------------------
ax = axes[1]
ax.plot(t_ms, st_mon.Vout[0] / volt, color='C0', lw=0.6)
shade(ax)
ax.set_ylabel('N0 Vout (V)')
ax.set_title('N0 (pure self-excitation) — output spikes', fontweight='bold')

# (2) N0 currents -------------------------------------------------------------
ax = axes[2]
is2e0 = exc0_mon.Is2[0] / uA
ax.plot(te0_ms, np.full_like(te0_ms, I0_val * 1e6), color='steelblue', ls=':',
        lw=1, label=f'I_0 = {I0_val*1e6:.1f} uA')
ax.plot(te0_ms, is2e0, color='C2', lw=1.3, label='Is2_exc')
ax.plot(te0_ms, I0_val * 1e6 + is2e0, color='k', lw=0.9, ls='--',
        label='I_0 + Is2_exc (drive)')
ax.axhline(I_min * 1e6, color='gray', ls='-.', lw=1, label=f'I_min = {I_min*1e6:.0f} uA')
ax.axhline(I_max * 1e6, color='red',  ls='-.', lw=1, label=f'I_max = {I_max*1e6:.0f} uA')
shade(ax)
ax.set_ylabel('current (uA)')
ax.set_title('N0 drive — positive feedback pins at the depol-block ceiling',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)

# (3) N1 Vout -----------------------------------------------------------------
ax = axes[3]
ax.plot(t_ms, st_mon.Vout[1] / volt, color='C3', lw=0.6)
shade(ax)
ax.set_ylabel('N1 Vout (V)')
ax.set_title('N1 (self-exc + slow self-inhibition) — output spikes', fontweight='bold')

# (4) N1 currents -------------------------------------------------------------
ax = axes[4]
is2e1 = exc1_mon.Is2[0] / uA
is2i1 = inh1_mon.Is2[0] / uA
net1  = I0_val * 1e6 + is2e1 - np.interp(te1_ms, ti1_ms, is2i1)
ax.plot(te1_ms, is2e1, color='C2', lw=1.3, label='Is2_exc')
ax.plot(ti1_ms, is2i1, color='C1', lw=1.3, label='Is2_inh (fast recovery)')
ax.plot(te1_ms, net1, color='k', lw=0.9, ls='--',
        label='I_0 + Is2_exc - Is2_inh (net drive)')
ax.axhline(I_min * 1e6, color='gray', ls='-.', lw=1, label=f'I_min = {I_min*1e6:.0f} uA')
ax.axhline(I_max * 1e6, color='red',  ls='-.', lw=1, label=f'I_max = {I_max*1e6:.0f} uA')
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
shade(ax)
ax.set_ylabel('current (uA)')
ax.set_xlabel('t (ms)')
ax.set_title('N1 drive — slow inhibition chops firing into bursts (adaptation)',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)

fig.suptitle('MSN self-excitation pacemaker — pure exc (N0) vs exc + slow inh (N1)',
             fontsize=13, fontweight='bold', y=1.005)
plt.show()
out_path = 'demo/ns_msn_pacemaker.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nFigure saved -> {out_path}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Zoom on N0 burst microstructure                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
z0, z1 = ZOOM_WIN
zt0, zt1 = z0 * 1e3, z1 * 1e3
mV = (t_ms   >= zt0) & (t_ms   <= zt1)
mE = (te0_ms >= zt0) & (te0_ms <= zt1)
is2e0z = exc0_mon.Is2[0] / uA

figz, axz = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                         gridspec_kw=dict(hspace=0.28))

# (0) N0 Vout with individual spikes marked
ax = axz[0]
ax.plot(t_ms[mV], (st_mon.Vout[0] / volt)[mV], color='C0', lw=1.0)
zk = all_t[(all_i == 0) & (all_t >= z0) & (all_t <= z1)] * 1e3
ax.vlines(zk, 2.03, 2.15, color='k', lw=0.9)
ax.axhline(params.Vth, color='dimgray', ls='--', lw=1, label=f'Vth = {params.Vth:.1f} V')
ax.set_ylabel('N0 Vout (V)')
ax.set_title(f'N0 zoom  [{z0:.2f}–{z1:.2f} s]  —  individual spikes within each burst '
             f'({len(zk)} spikes shown)', fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# (1) N0 drive over the same window
ax = axz[1]
ax.plot(te0_ms[mE], np.full(int(mE.sum()), I0_val * 1e6), color='steelblue',
        ls=':', lw=1, label=f'I_0 = {I0_val*1e6:.1f} uA')
ax.plot(te0_ms[mE], is2e0z[mE], color='C2', lw=1.4, label='Is2_exc')
ax.plot(te0_ms[mE], I0_val * 1e6 + is2e0z[mE], color='k', lw=1.0, ls='--',
        label='I_0 + Is2_exc (drive)')
ax.axhline(I_min * 1e6, color='gray', ls='-.', lw=1, label=f'I_min = {I_min*1e6:.0f} uA')
ax.axhline(I_max * 1e6, color='red',  ls='-.', lw=1,
           label=f'I_max = I_hold = {I_max*1e6:.0f} uA')
ax.set_ylabel('current (uA)')
ax.set_xlabel('t (ms)')
ax.set_title('N0 drive during the zoom window', fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)

figz.suptitle('N0 pure self-excitation — burst microstructure (zoom)',
              fontsize=12, fontweight='bold', y=1.01)
plt.savefig('demo/ns_msn_pacemaker_N0_zoom.png', dpi=200, bbox_inches='tight')
print("Zoom figure saved -> demo/ns_msn_pacemaker_N0_zoom.png")

# %%
