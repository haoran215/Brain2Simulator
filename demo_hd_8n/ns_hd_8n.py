"""
ns_hd_8n.py
===========
8-direction head-direction (HD) network built from MSN neurons.
Same wiring, parameters and tests as the 4-direction network of
demo/ns_hd_4n.py, with the ring enlarged to 8 positions.

Neurons (25)
────────────
  EB1..EB8            ellipsoid-body ring — holds the heading bump
  GI                  global inhibitor
  PB1..PB8_ccw        protocerebral-bridge relays, counter-clockwise shift
  PB1..PB8_cw         protocerebral-bridge relays, clockwise shift

Connectivity (56 synapses)
──────────────────────────
  EBk      ──exc──► EBk            self-excitation (the bump)
  EBk      ──exc──► GI             drive the global inhibitor
  GI       ──inh──► EBk            global inhibition (WTA)
  EBk      ──exc──► PBk_ccw        position copy
  EBk      ──exc──► PBk_cw         position copy
  PBk_ccw  ──exc──► EB(k-1)        shifted return  (PB1_ccw → EB8)
  PBk_cw   ──exc──► EB(k+1)        shifted return  (PB1_cw  → EB2)

Every neuron carries a subthreshold tonic bias I_0.  PBk_ccw / PBk_cw copy the
activity of EBk (with a synaptic delay): the copy current lifts them just above
rheobase, so they fire at a low rate while EBk holds the bump.  Their shifted
return onto EB(k∓1) is then too weak, on its own, to move the bump.

Turning: a velocity current pulse is delivered to the whole PB_cw (or PB_ccw)
population.  Only the PB that already receives the EB copy is lifted well above
threshold; its faster firing pushes EB(k±1) over threshold and the bump moves
one position.

Run
───
    uv run python demo_hd_8n/ns_hd_8n.py --stage connections | bump | wta | turn
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brian2 import *
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

prefs.codegen.target = 'numpy'
defaultclock.dt = 10 * us

N_DIR = 8                       # number of heading directions

# ── Hardware parameters ─────────────────────────────────────────────────────
# Cm is raised from the 100 nF default to 133 nF (100 nF ∥ 33 nF on the PCB).
# Firing rates scale ~1/Cm, which brings f_max from ~240 Hz down to the
# ~185 Hz of Fig2.2 without narrowing the 32–100 µA current window.
CM = 133e-9
p = MSNParams(Cm=CM)
I_MIN, I_MAX = p.operating_window()

# ── Tonic bias (all subthreshold) ───────────────────────────────────────────
I0_EB = 24.1e-6                 # 75% of I_min
I0_GI = 6.1e-6                  # 19% of I_min — GI is driven by the EBs
I0_PB = 5.5e-6                  # 17% of I_min — low: the EB copy and the velocity pulse do the work

# ── Synaptic weights (A) and time constants (s) ─────────────────────────────
# Tuned against Fig2.1 / Fig2.2 under the constraint that the winner and GI
# never stop firing when a pulse is removed.  Self-excitation is slow, EB → GI
# is fast: the inhibition loop must react faster than the self-excitation loop.
W_SELF, TAU_SELF = 2.394e-6, 150e-3    # EB → EB   self-excitation
W_EBGI, TAU_EBGI = 14.06e-6, 30e-3     # EB → GI
W_GIEB, TAU_GIEB = 1.503e-6, 100e-3    # GI → EB   global inhibition
# W_COPY is set so that I0_PB + peak copy current stays below I_hold (100 µA)
# while the EB is driven by a pulse: above I_hold the PB goes into
# depolarisation block and stops firing.
W_COPY, TAU_COPY = 4.9e-6, 100e-3      # EB → PB   position copy (PB follows EB, ~τ later)
W_SHIFT, TAU_SHIFT = 1.4e-6, 100e-3    # PB → EB   shifted return (moves the bump only with velocity input)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Network                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_network():
    """Build the 25-neuron HD network.

    Returns (groups, pathways):
      groups   : dict name → NeuronGroup          ('EB', 'GI', 'PB_ccw', 'PB_cw')
      pathways : list of dicts, one per Synapses object, with keys
                 syn, src, tgt, kind, tau, inlet, label
    """
    start_scope()

    # One named inlet per incoming pathway (Brian2: one Synapses per inlet).
    EB = make_msn(N_DIR, params=p,
                  exc_inlets=('I_exc_self', 'I_exc_ccw', 'I_exc_cw'),
                  inh_inlets=('I_inh_gi',), name='EB')
    GI = make_msn(1, params=p, exc_inlets=('I_exc_eb',), inh_inlets=(), name='GI')
    PB_ccw = make_msn(N_DIR, params=p, exc_inlets=('I_exc_eb',), inh_inlets=(),
                      name='PB_ccw')
    PB_cw = make_msn(N_DIR, params=p, exc_inlets=('I_exc_eb',), inh_inlets=(),
                     name='PB_cw')

    EB.I_0     = I0_EB * amp
    GI.I_0     = I0_GI * amp
    PB_ccw.I_0 = I0_PB * amp
    PB_cw.I_0  = I0_PB * amp

    groups = {'EB': EB, 'GI': GI, 'PB_ccw': PB_ccw, 'PB_cw': PB_cw}

    # (src, tgt, kind, weight, tau, inlet, connect, label)
    spec = [
        ('EB',     'EB',     'exc', W_SELF,  TAU_SELF,  'I_exc_self', 'i == j',
         'EB self-excitation'),
        ('EB',     'GI',     'exc', W_EBGI,  TAU_EBGI,  'I_exc_eb',   True,
         'EB -> GI'),
        ('GI',     'EB',     'inh', W_GIEB,  TAU_GIEB,  'I_inh_gi',   True,
         'GI -> EB global inhibition'),
        ('EB',     'PB_ccw', 'exc', W_COPY,  TAU_COPY,  'I_exc_eb',   'i == j',
         'EB -> PB_ccw copy'),
        ('EB',     'PB_cw',  'exc', W_COPY,  TAU_COPY,  'I_exc_eb',   'i == j',
         'EB -> PB_cw copy'),
        ('PB_ccw', 'EB',     'exc', W_SHIFT, TAU_SHIFT, 'I_exc_ccw',
         f'j == (i + {N_DIR - 1}) % {N_DIR}', 'PB_ccw -> EB (k-1)'),
        ('PB_cw',  'EB',     'exc', W_SHIFT, TAU_SHIFT, 'I_exc_cw',
         f'j == (i + 1) % {N_DIR}', 'PB_cw -> EB (k+1)'),
    ]

    pathways = []
    for src, tgt, kind, w, tau, inlet, connect, label in spec:
        syn = make_synapse(
            groups[src], groups[tgt],
            params=SynapseParams(kind=kind, weight=w, tau_s1=tau, tau_s2=tau,
                                 cascade='alpha', target_var=inlet),
            connect=connect, name=f'syn_{src}_to_{tgt}',
        )
        pathways.append(dict(syn=syn, src=src, tgt=tgt, kind=kind, tau=tau,
                             inlet=inlet, label=label))
    return groups, pathways


# ── Naming: one global index per neuron, 1-based labels as in the figure ────
GROUP_ORDER = ['EB', 'GI', 'PB_ccw', 'PB_cw']

def neuron_name(group, k):
    if group == 'GI':
        return 'GI'
    if group == 'EB':
        return f'EB{k + 1}'
    return f'PB{k + 1}_{group[3:]}'

def global_index(groups):
    """dict (group, k) → row/column in the full connectivity matrix."""
    idx, n = {}, 0
    for g in GROUP_ORDER:
        for k in range(groups[g].N):
            idx[(g, k)] = n
            n += 1
    return idx


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Stage 1 — connection check                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def stage_connections():
    groups, pathways = build_network()
    idx   = global_index(groups)
    names = [neuron_name(g, k) for (g, k) in idx]
    n_tot = len(names)

    print(p.summary())
    print(f"\nNeurons ({n_tot}) and tonic bias:")
    for g in GROUP_ORDER:
        for k in range(groups[g].N):
            i0 = float(groups[g].I_0[k] / amp)
            print(f"  {neuron_name(g, k):8s}  I_0 = {i0*1e6:5.2f} µA "
                  f"({i0/I_MIN*100:.0f}% of I_min = {I_MIN*1e6:.1f} µA)")

    # Edges are read back from the built Synapses objects, not from the spec.
    W = np.zeros((n_tot, n_tot))          # signed weight (µA): + exc, − inh
    n_edges = 0
    print(f"\n{'pre':8s}    {'post':8s}  type  weight(µA)  tau(ms)  inlet")
    print('─' * 60)
    for pw in pathways:
        syn = pw['syn']
        print(f"# {pw['label']}")
        for i, j, w in zip(np.array(syn.i), np.array(syn.j), np.array(syn.w / amp)):
            pre, post = neuron_name(pw['src'], i), neuron_name(pw['tgt'], j)
            sign = 1.0 if pw['kind'] == 'exc' else -1.0
            W[idx[(pw['src'], i)], idx[(pw['tgt'], j)]] = sign * w * 1e6
            print(f"{pre:8s} →  {post:8s}  {pw['kind']}   {w*1e6:8.2f}  "
                  f"{pw['tau']*1e3:7.0f}  {pw['inlet']}")
            n_edges += 1
    print('─' * 60)
    print(f"Total synapses: {n_edges}  "
          f"(exc: {int(np.sum(W > 0))}, inh: {int(np.sum(W < 0))})")

    # ── Connectivity matrix ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 13.5))
    vmax = np.abs(W).max()
    im = ax.imshow(W, cmap='bwr', vmin=-vmax, vmax=vmax)
    for r in range(n_tot):
        for c in range(n_tot):
            if W[r, c] != 0:
                ax.text(c, r, f'{abs(W[r, c]):.1f}', ha='center', va='center',
                        fontsize=7)
    ax.set_xticks(range(n_tot)); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(n_tot)); ax.set_yticklabels(names)
    ax.set_xlabel('post-synaptic (target)')
    ax.set_ylabel('pre-synaptic (source)')
    # separators between EB | GI | PB_ccw | PB_cw
    edge = 0
    for g in GROUP_ORDER[:-1]:
        edge += groups[g].N
        ax.axhline(edge - 0.5, color='k', lw=0.8)
        ax.axvline(edge - 0.5, color='k', lw=0.8)
    ax.set_title(f'HD network connectivity — {n_edges} synapses\n'
                 'red = excitatory, blue = inhibitory, number = weight (µA)',
                 fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='signed weight (µA)')
    out = os.path.join(os.path.dirname(__file__), 'ns_hd_8n_connections.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Simulation helper                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

REC_DT = 0.5e-3                 # s, state-monitor sampling

def simulate(pulses, T, pb_pulses=()):
    """Run the network for T seconds with step-current pulses.

    pulses    : list of (t_start [s], duration [s], EB index (0-based), amplitude [A])
                — added on top of I0_EB on one EB neuron.
    pb_pulses : list of (t_start [s], duration [s], 'cw' | 'ccw', amplitude [A])
                — velocity input, added on top of I0_PB on ALL neurons of
                that PB population.

    The run is split at every pulse edge and I_0 is rewritten between
    segments (no per-step Python callback).

    Returns dict with groups, spike monitors `spk`, state monitors `st`.
    """
    prefs.codegen.target = 'cython'
    groups, pathways = build_network()
    EB = groups['EB']

    spk = {g: SpikeMonitor(groups[g]) for g in GROUP_ORDER}
    st = {
        'EB': StateMonitor(EB, ['Vm', 'I_0', 'I_exc_self', 'I_exc_ccw',
                                'I_exc_cw', 'I_inh_gi'],
                           record=True, dt=REC_DT*second),
        'GI': StateMonitor(groups['GI'], ['I_exc_eb'], record=True,
                           dt=REC_DT*second),
        'PB_ccw': StateMonitor(groups['PB_ccw'], ['I_exc_eb', 'I_0'], record=True,
                               dt=REC_DT*second),
        'PB_cw': StateMonitor(groups['PB_cw'], ['I_exc_eb', 'I_0'], record=True,
                              dt=REC_DT*second),
    }
    # per-source EB → GI currents (the GI inlet only holds their sum)
    syn_ebgi = next(pw['syn'] for pw in pathways
                    if pw['src'] == 'EB' and pw['tgt'] == 'GI')
    st['EB_GI'] = StateMonitor(syn_ebgi, 'Is2', record=True, dt=REC_DT*second)
    ebgi_src = np.array(syn_ebgi.i)          # edge → source EB index
    # explicit Network: monitors live in dicts, collect() would miss them
    net = Network(*groups.values(), *[pw['syn'] for pw in pathways],
                  *spk.values(), *st.values())

    edges = sorted({0.0, T} | {x for t0, d, *_ in (*pulses, *pb_pulses)
                               for x in (t0, t0 + d)})
    for t_a, t_b in zip(edges[:-1], edges[1:]):
        on = lambda t0, d: t0 <= t_a < t0 + d
        i_eb = np.full(N_DIR, I0_EB)
        for t0, d, k, amp_ in pulses:
            if on(t0, d):
                i_eb[k] += amp_
        EB.I_0 = i_eb * amp
        for side in ('cw', 'ccw'):
            v = 0.0
            for t0, d, sd, a_ in pb_pulses:
                if sd == side and on(t0, d):
                    v += a_
            groups[f'PB_{side}'].I_0 = (I0_PB + v) * amp
        net.run((t_b - t_a) * second)

    return dict(groups=groups, spk=spk, st=st, ebgi_src=ebgi_src)


def spikes_of(sm, k):
    """Spike times (s) of neuron k in a SpikeMonitor."""
    return np.asarray(sm.t / second)[np.asarray(sm.i) == k]

def max_isi(ts, t_a, t_b):
    """Longest silent interval (ms) of spike times ts within [t_a, t_b]."""
    w = ts[(ts >= t_a) & (ts <= t_b)]
    return np.max(np.diff(np.concatenate(([t_a], w, [t_b])))) * 1e3

def mean_rate(ts, t_a, t_b):
    """Mean firing rate (Hz) of spike times ts within [t_a, t_b)."""
    return np.sum((ts >= t_a) & (ts < t_b)) / (t_b - t_a)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Stage 2 — EB bump                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

EB_COLORS = ['#f2c200', '#5fbf00', '#1f6fff', '#e020e0',   # EB1..4 as in Fig3.1
             '#ff7f0e', '#00a5a5', '#8c564b', '#6a3dbd']   # EB5..8
GI_COLOR  = '#d62728'

def stage_bump():
    T, t_p, dur, I_p, k_p = 5.0, 1.0, 0.5, 30e-6, 0
    res = simulate([(t_p, dur, k_p, I_p)], T)
    spk, st = res['spk'], res['st']
    t_end = t_p + dur

    print(f"\nBump test: {I_p*1e6:.0f} µA × {dur:.1f} s pulse on EB{k_p+1} "
          f"at t = {t_p:.1f} s   (I_0_EB = {I0_EB*1e6:.1f} µA, "
          f"I_min = {I_MIN*1e6:.1f} µA)")
    print(f"\n{'neuron':8s}  before  during   after   rate after (last 2 s, Hz)")
    for g in GROUP_ORDER:
        for k in range(res['groups'][g].N):
            ts = spikes_of(spk[g], k)
            print(f"{neuron_name(g, k):8s}  {np.sum(ts < t_p):6d}  "
                  f"{np.sum((ts >= t_p) & (ts < t_end)):6d}  "
                  f"{np.sum(ts >= t_end):6d}   {mean_rate(ts, T - 2, T):6.1f}")

    # firing gap after pulse removal: longest silence from pulse end onward
    print(f"\nLongest silent interval after the pulse is removed:")
    print(f"  EB{k_p+1}  {max_isi(spikes_of(spk['EB'], k_p), t_end, T):6.1f} ms")
    print(f"  GI   {max_isi(spikes_of(spk['GI'], 0), t_end, T):6.1f} ms")

    # steady-state currents over the last 2 s
    t = np.asarray(st['EB'].t / second)
    late = t >= T - 2
    e = st['EB']
    i_self = np.mean(e.I_exc_self[k_p][late] / uA)
    i_inh  = np.mean(e.I_inh_gi[k_p][late] / uA)
    i_gi   = np.mean(st['GI'].I_exc_eb[0][late] / uA)
    i_pb   = np.mean(st['PB_cw'].I_exc_eb[k_p][late] / uA)
    print(f"\nSteady-state currents (mean over last 2 s):")
    print(f"  EB{k_p+1} self-excitation   {i_self:6.2f} µA")
    print(f"  GI → EB inhibition    {i_inh:6.2f} µA")
    print(f"  EB{k_p+1} total drive       {I0_EB*1e6 + i_self - i_inh:6.2f} µA   "
          f"(I_0 + self − inh;  I_min = {I_MIN*1e6:.1f}, I_max = {I_MAX*1e6:.0f})")
    print(f"  EB → GI excitation    {i_gi:6.2f} µA")
    print(f"  EB{k_p+1} → PB{k_p+1} copy        {i_pb:6.2f} µA   "
          f"(PB total {I0_PB*1e6 + i_pb:.2f} µA vs I_min {I_MIN*1e6:.1f})")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True,
                             gridspec_kw=dict(hspace=0.35,
                                              height_ratios=[1, 3.2, 1.3, 1.6]))
    def shade(ax):
        ax.axvspan(t_p, t_end, alpha=0.13, color='tomato')

    # (0) input
    ax = axes[0]
    for k in range(N_DIR):
        ax.plot(t, e.I_0[k] / uA, color=EB_COLORS[k], lw=1.2, label=f'EB{k+1}')
    ax.axhline(I_MIN*1e6, color='gray', ls='--', lw=1,
               label=f'I_min = {I_MIN*1e6:.1f} µA')
    shade(ax)
    ax.set_ylabel('I_0 (µA)')
    ax.set_title(f'Input — subthreshold I_0 on every EB, '
                 f'{I_p*1e6:.0f} µA pulse on EB{k_p+1}', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=9)

    # (1) raster of all 25 neurons
    ax = axes[1]
    row, labels = 0, []
    for g in GROUP_ORDER:
        for k in range(res['groups'][g].N):
            ts = spikes_of(spk[g], k)
            c = EB_COLORS[k] if g == 'EB' else GI_COLOR if g == 'GI' else 'gray'
            ax.vlines(ts, row - 0.4, row + 0.4, color=c, lw=0.4)
            labels.append(neuron_name(g, k))
            row += 1
    shade(ax)
    ax.set_yticks(range(row)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(row - 0.5, -0.5)
    ax.set_title('Spike raster — all 25 neurons', fontweight='bold')

    # (2) Vm of the pulsed EB
    ax = axes[2]
    ax.plot(t, e.Vm[k_p] / volt, color=EB_COLORS[k_p], lw=0.5)
    ax.axhline(p.Vth, color='dimgray', ls='--', lw=1, label=f'Vth = {p.Vth:.1f} V')
    shade(ax)
    ax.set_ylabel('Vm (V)')
    ax.set_title(f'EB{k_p+1} membrane voltage', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (3) currents on the pulsed EB
    ax = axes[3]
    i_s = e.I_exc_self[k_p] / uA
    i_i = e.I_inh_gi[k_p] / uA
    ax.plot(t, i_s, color='C3', lw=1.2, label='self-excitation')
    ax.plot(t, i_i, color='C0', lw=1.2, label='GI → EB inhibition')
    ax.plot(t, e.I_0[k_p] / uA + i_s - i_i, color='k', lw=0.9, ls=':',
            label='total drive (I_0 + self − inh)')
    ax.plot(t, st['GI'].I_exc_eb[0] / uA, color=GI_COLOR, lw=1.0, alpha=0.6,
            label='EB → GI excitation')
    ax.axhline(I_MIN*1e6, color='gray', ls='-.', lw=1, label='I_min')
    ax.axhline(I_MAX*1e6, color='red', ls='-.', lw=1, label='I_max')
    shade(ax)
    ax.set_ylabel('current (µA)'); ax.set_xlabel('t (s)')
    ax.set_title(f'Synaptic currents on EB{k_p+1} — total drive stays above '
                 'I_min after the pulse', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=3)

    out = os.path.join(os.path.dirname(__file__), 'ns_hd_8n_bump.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Stage 3 — WTA between every pair of EB neurons                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Fig2.1c protocol: 0.5 s pulse every 2.5 s.  The pulse is 30 µA rather than
# the figure's 35 µA: with the PBs copying the EBs, the pulsed EB also receives
# PB shift current, and at 35 µA the summed EB → GI drive pushes GI past I_hold.
# The target sequence is an
# Euler circuit of the complete directed graph on 8 nodes, so every ordered
# pair (EBa → EBb) is switched exactly once (56 transitions).
def euler_circuit(n):
    """Closed walk over 1..n that uses every ordered pair (a, b), a ≠ b, once."""
    out = {a: [b for b in range(n, 0, -1) if b != a] for a in range(1, n + 1)}
    stack, walk = [1], []
    while stack:                                # Hierholzer
        if out[stack[-1]]:
            stack.append(out[stack[-1]].pop())
        else:
            walk.append(stack.pop())
    return walk[::-1]

WTA_SEQ    = euler_circuit(N_DIR)                 # 1-based EB labels, 57 pulses
WTA_T0     = 0.5        # s, first pulse
WTA_PERIOD = 2.5        # s
WTA_DUR    = 0.5        # s
WTA_I      = 30e-6      # A

# Figure targets (µA, Hz) read off Fig2.1c / Fig2.2b
TARGETS = {
    'self-exc steady (µA)': 35, 'self-exc peak (µA)': 55,
    'EB→GI steady (µA)':    45, 'EB→GI peak (µA)':    75,
    'GI→EB steady (µA)':    20, 'GI→EB peak (µA)':    27,
    'winner rate steady (Hz)': 85, 'winner rate peak (Hz)': 185,
    'GI rate steady (Hz)':    100, 'GI rate peak (Hz)':     185,
    'transition time (ms)':   500,
}

def smooth_rate(ts, t_grid, sigma=30e-3):
    """Gaussian-kernel firing rate (Hz) of spike times ts on t_grid."""
    r = np.zeros_like(t_grid)
    for s_ in ts:
        lo, hi = np.searchsorted(t_grid, [s_ - 4*sigma, s_ + 4*sigma])
        r[lo:hi] += np.exp(-0.5 * ((t_grid[lo:hi] - s_) / sigma) ** 2)
    return r / (sigma * np.sqrt(2 * np.pi))


def stage_wta():
    n_p    = len(WTA_SEQ)
    starts = WTA_T0 + WTA_PERIOD * np.arange(n_p)
    T      = WTA_T0 + WTA_PERIOD * n_p
    res = simulate([(t0, WTA_DUR, k - 1, WTA_I) for t0, k in zip(starts, WTA_SEQ)], T)
    spk, st, e = res['spk'], res['st'], res['st']['EB']
    t = np.asarray(e.t / second)

    eb_ts  = [spikes_of(spk['EB'], k) for k in range(N_DIR)]
    gi_ts  = spikes_of(spk['GI'], 0)
    pb_ts  = {g: [spikes_of(spk[g], k) for k in range(N_DIR)]
              for g in ('PB_ccw', 'PB_cw')}
    i_copy = np.array([st['PB_cw'].I_exc_eb[k] / uA for k in range(N_DIR)])
    # shifted return onto each EB (from PB_ccw(k+1) and PB_cw(k-1))
    i_shift = np.array([(e.I_exc_ccw[k] + e.I_exc_cw[k]) / uA for k in range(N_DIR)])
    r_pb = np.array([smooth_rate(ts, t) for ts in pb_ts['PB_cw']])
    i_self = np.array([e.I_exc_self[k] / uA for k in range(N_DIR)])
    i_inh  = np.asarray(e.I_inh_gi[0] / uA)          # identical on all EBs
    i_ebgi = np.zeros((N_DIR, len(t)))
    for edge, k in enumerate(res['ebgi_src']):
        i_ebgi[k] = st['EB_GI'].Is2[edge] / uA
    r_eb = np.array([smooth_rate(ts, t) for ts in eb_ts])
    r_gi = smooth_rate(gi_ts, t)

    # ── Per-epoch measurements ─────────────────────────────────────────────
    # steady = last 1 s before the next pulse; peak = pulse window + 0.3 s
    print(f"\nWTA test: {WTA_I*1e6:.0f} µA × {WTA_DUR:.1f} s pulses every "
          f"{WTA_PERIOD:.1f} s, {n_p} pulses, sequence "
          f"EB{' '.join(map(str, WTA_SEQ))}")
    print(f"\n{'#':>2} {'switch':9s} {'ok':3s} {'t_trans':>7s} | "
          f"{'f_win':>5s} {'f_GI':>5s} {'self':>5s} {'EB→GI':>5s} {'GI→EB':>5s} | "
          f"{'pk f':>5s} {'pk fGI':>6s} {'pk self':>7s} {'pk EB→GI':>8s} {'pk GI→EB':>8s} | "
          f"{'gap win':>7s} {'gap GI':>6s}")
    print(f"{'':16s} {'(ms)':>7s} | {'(Hz)':>5s} {'(Hz)':>5s} {'(µA)':>5s} "
          f"{'(µA)':>5s} {'(µA)':>5s} | {'':38s} | {'(ms)':>7s} {'(ms)':>6s}")
    rows, n_ok = [], 0
    for n, (t0, lab) in enumerate(zip(starts, WTA_SEQ)):
        k = lab - 1
        sa, sb = t0 + WTA_PERIOD - 1.0, t0 + WTA_PERIOD
        ss = (t >= sa) & (t < sb)
        pk = (t >= t0) & (t < t0 + WTA_DUR + 0.3)
        rates = [mean_rate(ts, sa, sb) for ts in eb_ts]
        ok = rates[k] > 5 and all(rates[m] == 0 for m in range(N_DIR) if m != k)
        n_ok += ok
        if n == 0:
            sw, t_tr = f"— → EB{lab}", np.nan
        else:
            old = WTA_SEQ[n - 1] - 1
            sw = f"EB{old+1} → EB{lab}"
            # transition time: pulse onset → last spike of the old winner
            last = eb_ts[old][eb_ts[old] < sb]
            t_tr = (last.max() - t0) * 1e3 if len(last) else np.nan
        row = dict(f=rates[k], fG=mean_rate(gi_ts, sa, sb),
                   s=i_self[k][ss].mean(), eg=i_ebgi[k][ss].mean(),
                   ge=i_inh[ss].mean(), pf=r_eb[k][pk].max(), pfG=r_gi[pk].max(),
                   ps=i_self[k][pk].max(), peg=i_ebgi[k][pk].max(),
                   pge=i_inh[pk].max(), tt=t_tr,
                   # longest silence from 50 ms after pulse onset to next pulse
                   gap=max_isi(eb_ts[k], t0 + 0.05, sb),
                   gapG=max_isi(gi_ts, t0 + 0.05, sb))
        rows.append(row)
        print(f"{n+1:2d} {sw:9s} {'yes' if ok else 'NO ':3s} {t_tr:7.0f} | "
              f"{row['f']:5.1f} {row['fG']:5.1f} {row['s']:5.1f} {row['eg']:5.1f} "
              f"{row['ge']:5.1f} | {row['pf']:5.0f} {row['pfG']:6.0f} "
              f"{row['ps']:7.1f} {row['peg']:8.1f} {row['pge']:8.1f} | "
              f"{row['gap']:7.1f} {row['gapG']:6.1f}")
    print(f"\nCorrect single winner after {n_ok}/{n_p} pulses.")

    # ── PB copy of the EB activity ─────────────────────────────────────────
    # on/off delay = PBk first/last spike relative to EBk first/last spike
    print(f"\nPB copy:  {'#':>2} {'EB':3s} {'ok':3s} {'f_ccw':>5s} {'f_cw':>5s} "
          f"{'copy':>5s} {'PB tot':>6s} {'shift':>5s} {'on dly':>6s} {'off dly':>7s}")
    print(f"{'':17s} {'(Hz)':>5s} {'(Hz)':>5s} {'(µA)':>5s} {'(µA)':>6s} "
          f"{'(µA)':>5s} {'(ms)':>6s} {'(ms)':>7s}")
    n_pb_ok = 0
    for n, (t0, lab) in enumerate(zip(starts, WTA_SEQ)):
        k = lab - 1
        sa, sb = t0 + WTA_PERIOD - 1.0, t0 + WTA_PERIOD
        ss = (t >= sa) & (t < sb)
        f_pb = {g: [mean_rate(ts, sa, sb) for ts in pb_ts[g]] for g in pb_ts}
        ok = all(f_pb[g][k] > 5 and
                 all(f_pb[g][m] == 0 for m in range(N_DIR) if m != k)
                 for g in pb_ts)
        n_pb_ok += ok
        eb_w = eb_ts[k][(eb_ts[k] >= t0) & (eb_ts[k] < sb + WTA_PERIOD)]
        pb_w = pb_ts['PB_cw'][k][(pb_ts['PB_cw'][k] >= t0) &
                                 (pb_ts['PB_cw'][k] < sb + WTA_PERIOD)]
        on  = (pb_w.min() - eb_w.min()) * 1e3 if len(pb_w) and len(eb_w) else np.nan
        off = (pb_w.max() - eb_w.max()) * 1e3 if len(pb_w) and len(eb_w) else np.nan
        if n == n_p - 1:
            off = np.nan                       # bump still on at end of run
        nb = (k + 1) % N_DIR                   # a neighbour that receives the shift
        rows[n].update(fpb=f_pb['PB_cw'][k], cp=i_copy[k][ss].mean(),
                       sh=i_shift[nb][ss].mean(), on=on, off=off)
        print(f"{'':10s}{n+1:2d} EB{lab} {'yes' if ok else 'NO ':3s} "
              f"{f_pb['PB_ccw'][k]:5.1f} {f_pb['PB_cw'][k]:5.1f} "
              f"{rows[n]['cp']:5.1f} {I0_PB*1e6 + rows[n]['cp']:6.1f} "
              f"{rows[n]['sh']:5.1f} {on:6.0f} {off:7.0f}")
    print(f"PBk_ccw and PBk_cw active only with EBk after {n_pb_ok}/{n_p} pulses.")
    print(f"Peak shift current onto any EB: {i_shift.max():.1f} µA   "
          f"(a silent EB needs I_0 + shift − inh > I_min = {I_MIN*1e6:.1f} µA)")
    print(f"Peak PB drive (I_0 + copy): {I0_PB*1e6 + i_copy.max():.1f} µA   "
          f"(depolarisation block above I_hold = {I_MAX*1e6:.0f} µA)")
    print(f"Longest silent interval after a pulse starts: winner "
          f"{max(r['gap'] for r in rows):.1f} ms, GI {max(r['gapG'] for r in rows):.1f} ms")

    # ── Target vs measured ─────────────────────────────────────────────────
    avg = lambda key: np.nanmean([r[key] for r in rows[1:]])   # skip start-up
    measured = {
        'self-exc steady (µA)': avg('s'),  'self-exc peak (µA)': avg('ps'),
        'EB→GI steady (µA)':    avg('eg'), 'EB→GI peak (µA)':    avg('peg'),
        'GI→EB steady (µA)':    avg('ge'), 'GI→EB peak (µA)':    avg('pge'),
        'winner rate steady (Hz)': avg('f'),  'winner rate peak (Hz)': avg('pf'),
        'GI rate steady (Hz)':     avg('fG'), 'GI rate peak (Hz)':     avg('pfG'),
        'transition time (ms)':    avg('tt'),
    }
    print(f"\n{'quantity':26s} {'figure':>7s} {'sim':>7s} {'diff':>7s}")
    for key, tgt in TARGETS.items():
        m = measured[key]
        print(f"{key:26s} {tgt:7.0f} {m:7.1f} {(m - tgt)/tgt*100:+6.0f}%")

    # ── Figure 1: Fig2.1c-style overview ───────────────────────────────────
    fig, axes = plt.subplots(7, 1, figsize=(60, 24), sharex=True,
                             gridspec_kw=dict(hspace=0.30,
                                              height_ratios=[1, 3.5, 1.6, 1.6, 1.4,
                                                             1.4, 1.4]))
    ax = axes[0]
    for k in range(N_DIR):
        ax.plot(t, e.I_0[k] / uA - I0_EB*1e6, color=EB_COLORS[k], lw=1.2,
                label=f'pulse to EB{k+1}')
    ax.set_ylabel('input (µA)')
    ax.set_title('Input pulses (on top of I_0)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=8)

    ax = axes[1]
    row, labels = 0, []
    for g in GROUP_ORDER:
        for k in range(res['groups'][g].N):
            ts = spikes_of(spk[g], k)
            c = EB_COLORS[k] if g == 'EB' else GI_COLOR if g == 'GI' else 'gray'
            ax.vlines(ts, row - 0.4, row + 0.4, color=c, lw=0.3)
            labels.append(neuron_name(g, k))
            row += 1
    ax.set_yticks(range(row)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(row - 0.5, -0.5)
    ax.set_title('Spike raster — all 25 neurons', fontweight='bold')

    for ax, cur, title in ((axes[2], i_self, 'EB self-excitatory current'),
                           (axes[3], i_ebgi, 'EB → GI excitatory current')):
        for k in range(N_DIR):
            ax.plot(t, cur[k], color=EB_COLORS[k], lw=1.0, label=f'EB{k+1}')
        ax.set_ylabel('current (µA)')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=8)

    ax = axes[4]
    ax.plot(t, i_inh, color='k', lw=1.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('current (µA)'); ax.set_xlabel('time (s)')
    ax.set_title('GI → EB inhibitory current (same on all 8 EBs)', fontweight='bold')
    ax.set_xlabel('')

    ax = axes[5]
    for k in range(N_DIR):
        ax.plot(t, i_copy[k], color=EB_COLORS[k], lw=1.0,
                label=f'EB{k+1} → PB{k+1}')
    ax.axhline((I_MIN - I0_PB)*1e6, color='gray', ls='-.', lw=1,
               label='PB threshold (I_min − I_0)')
    ax.set_ylabel('current (µA)')
    ax.set_title('EB → PB copy current (same on PBk_ccw and PBk_cw)',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=9)

    ax = axes[6]
    for k in range(N_DIR):
        ax.plot(t, i_shift[k], color=EB_COLORS[k], lw=1.0, label=f'onto EB{k+1}')
    ax.set_ylabel('current (µA)'); ax.set_xlabel('time (s)')
    ax.set_title('PB → EB shifted-return current (PB_ccw(k+1) + PB_cw(k−1) onto EBk)',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=8)

    for ax in axes:
        for t0 in starts:
            ax.axvline(t0, color='gray', ls='--', lw=0.5, alpha=0.6)
    out = os.path.join(os.path.dirname(__file__), 'ns_hd_8n_wta.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")

    # ── Figure 2: Fig2.2b-style firing rates around every switch ───────────
    fig, axes = plt.subplots(7, 8, figsize=(34, 23), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.30, wspace=0.08))
    for n, ax in zip(range(1, n_p), axes.flat):
        t0 = starts[n]
        old, new = WTA_SEQ[n - 1] - 1, WTA_SEQ[n] - 1
        w = (t >= t0 - 1) & (t < t0 + 1.5)
        ax.plot(t[w] - t0, r_eb[old][w], color=EB_COLORS[old], lw=2,
                label=f'EB{old+1}')
        ax.plot(t[w] - t0, r_eb[new][w], color=EB_COLORS[new], lw=2,
                label=f'EB{new+1}')
        ax.plot(t[w] - t0, r_gi[w], color=GI_COLOR, lw=2, label='GI')
        ax.plot(t[w] - t0, r_pb[old][w], color=EB_COLORS[old], lw=1.5, ls='--',
                label=f'PB{old+1}_cw')
        ax.plot(t[w] - t0, r_pb[new][w], color=EB_COLORS[new], lw=1.5, ls='--',
                label=f'PB{new+1}_cw')
        ax.axvspan(0, WTA_DUR, color='gray', alpha=0.15)
        ax.set_title(f'EB{old+1} → EB{new+1}   '
                     f'(transition {rows[n]["tt"]:.0f} ms)', fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left')
    for ax in axes[-1]:
        ax.set_xlabel('time from pulse onset (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('firing rate (Hz)')
    fig.suptitle('Firing rates around each of the 56 EB → EB switches '
                 '(grey = pulse, dashed = PB copy)', fontweight='bold', y=0.905)
    out = os.path.join(os.path.dirname(__file__), 'ns_hd_8n_wta_rates.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Stage 4 — turning via PB velocity input                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# One EB pulse seeds the bump on EB1.  After that every pulse is a velocity
# input to a whole PB population: 8 × CW (EB1→2→…→8→1), then 8 × CCW back.
TURN_SEED   = (0.5, 0.5, 0, 30e-6)        # t0, dur, EB index, amplitude
TURN_T0     = 3.5        # s, first velocity pulse — EB1 holds the bump for 3.5 s first
TURN_PERIOD = 2.5        # s
TURN_DUR    = 0.5        # s
TURN_I      = 25e-6      # A — I0_PB + TURN_I stays below I_min without the copy
TURN_SEQ    = ['cw'] * N_DIR + ['ccw'] * N_DIR

def stage_turn():
    n_p    = len(TURN_SEQ)
    starts = TURN_T0 + TURN_PERIOD * np.arange(n_p)
    T      = TURN_T0 + TURN_PERIOD * n_p
    res = simulate([TURN_SEED], T,
                   pb_pulses=[(t0, TURN_DUR, sd, TURN_I)
                              for t0, sd in zip(starts, TURN_SEQ)])
    spk, st, e = res['spk'], res['st'], res['st']['EB']
    t = np.asarray(e.t / second)

    eb_ts = [spikes_of(spk['EB'], k) for k in range(N_DIR)]
    gi_ts = spikes_of(spk['GI'], 0)
    pb_ts = {g: [spikes_of(spk[g], k) for k in range(N_DIR)]
             for g in ('PB_ccw', 'PB_cw')}
    i_inh = np.asarray(e.I_inh_gi[0] / uA)
    i_sh  = {'cw':  np.array([e.I_exc_cw[k]  / uA for k in range(N_DIR)]),
             'ccw': np.array([e.I_exc_ccw[k] / uA for k in range(N_DIR)])}
    r_eb = np.array([smooth_rate(ts, t) for ts in eb_ts])
    r_gi = smooth_rate(gi_ts, t)
    r_pb = {g: np.array([smooth_rate(ts, t) for ts in pb_ts[g]]) for g in pb_ts}

    def winner(t_a, t_b):
        """0-based index of the single active EB in [t_a, t_b), else None."""
        rates = [mean_rate(ts, t_a, t_b) for ts in eb_ts]
        act = [k for k, r in enumerate(rates) if r > 5]
        return act[0] if len(act) == 1 else None

    print(f"\nTurning test: bump seeded on EB1, then {TURN_I*1e6:.0f} µA × "
          f"{TURN_DUR:.1f} s velocity pulses to the whole PB_cw / PB_ccw population")
    print(f"  PB without copy during pulse: I_0 + pulse = "
          f"{(I0_PB + TURN_I)*1e6:.1f} µA  (I_min = {I_MIN*1e6:.1f} µA → silent)")
    k_now = winner(TURN_T0 - 1.0, TURN_T0)
    print(f"  Seed: winner before first velocity pulse = "
          f"{'EB%d' % (k_now + 1) if k_now is not None else 'none / several'}")

    print(f"\n{'#':>2} {'dir':3s} {'expected':11s} {'got':4s} {'ok':3s} "
          f"{'t_on':>5s} {'t_off':>5s} | {'f_new':>5s} {'f_GI':>5s} | "
          f"{'PB base':>7s} {'PB puls':>7s} {'sh base':>7s} {'sh peak':>7s} | "
          f"{'other PB':>8s} {'gap GI':>6s}")
    print(f"{'':27s} {'(ms)':>5s} {'(ms)':>5s} | {'(Hz)':>5s} {'(Hz)':>5s} | "
          f"{'(Hz)':>7s} {'(Hz)':>7s} {'(µA)':>7s} {'(µA)':>7s} | "
          f"{'spikes':>8s} {'(ms)':>6s}")
    n_ok, k_old = 0, 0
    for n, (t0, sd) in enumerate(zip(starts, TURN_SEQ)):
        step  = 1 if sd == 'cw' else -1
        k_new = (k_old + step) % N_DIR
        sa, sb = t0 + TURN_PERIOD - 1.0, t0 + TURN_PERIOD
        got = winner(sa, sb)
        ok  = got == k_new
        n_ok += ok
        g   = f'PB_{sd}'
        pls = (t >= t0) & (t < t0 + TURN_DUR)
        pre = (t >= t0 - 1.0) & (t < t0)
        # t_on : pulse onset → first spike of the new EB
        # t_off: pulse onset → last spike of the old EB
        first = eb_ts[k_new][eb_ts[k_new] >= t0]
        last  = eb_ts[k_old][(eb_ts[k_old] >= t0) & (eb_ts[k_old] < sb)]
        t_on  = (first.min() - t0) * 1e3 if len(first) else np.nan
        t_off = (last.max() - t0) * 1e3 if len(last) else np.nan
        # PBs of the pulsed population that do NOT hold the copy must stay silent
        other = int(np.sum([np.sum((pb_ts[g][m] >= t0) & (pb_ts[g][m] < t0 + TURN_DUR))
                            for m in range(N_DIR) if m not in (k_old, k_new)]))
        print(f"{n+1:2d} {sd:3s} EB{k_old+1} → EB{k_new+1}   "
              f"{'EB%d' % (got + 1) if got is not None else '—':4s} "
              f"{'yes' if ok else 'NO ':3s} {t_on:5.0f} {t_off:5.0f} | "
              f"{mean_rate(eb_ts[k_new], sa, sb):5.1f} {mean_rate(gi_ts, sa, sb):5.1f} | "
              f"{mean_rate(pb_ts[g][k_old], t0 - 1.0, t0):7.1f} "
              f"{mean_rate(pb_ts[g][k_old], t0, t0 + TURN_DUR):7.1f} "
              f"{i_sh[sd][k_new][pre].mean():7.1f} {i_sh[sd][k_new][pls].max():7.1f} | "
              f"{other:8d} {max_isi(gi_ts, t0, sb):6.1f}")
        k_old = k_new if got is None else got
    # depolarisation-block check: peak PB drive, and the longest pause of any
    # PB between its first and last spike of each active period
    i_pb = max(np.max((st[g].I_exc_eb[k] + st[g].I_0[k]) / uA)
               for g in pb_ts for k in range(N_DIR))
    pb_gap = max(np.max(d[d < 1.0]) for g in pb_ts for ts in pb_ts[g]
                 for d in [np.diff(ts)] if len(d)) * 1e3
    print(f"\nPeak PB drive (I_0 + copy + velocity): {i_pb:.1f} µA   "
          f"(depolarisation block above I_hold = {I_MAX*1e6:.0f} µA)")
    print(f"Longest pause of a PB while it is copying: {pb_gap:.1f} ms")
    print(f"\nBump moved exactly one position in the commanded direction on "
          f"{n_ok}/{n_p} velocity pulses.")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(24, 23), sharex=True,
                             gridspec_kw=dict(hspace=0.30,
                                              height_ratios=[1, 3.8, 1.5, 1.5,
                                                             1.5, 1.5]))
    ax = axes[0]
    ax.plot(t, e.I_0[TURN_SEED[2]] / uA - I0_EB*1e6, color=EB_COLORS[0], lw=1.2,
            label='seed pulse to EB1')
    ax.plot(t, st['PB_cw'].I_0[0] / uA - I0_PB*1e6, color='k', lw=1.4,
            label='velocity → all PB_cw')
    ax.plot(t, st['PB_ccw'].I_0[0] / uA - I0_PB*1e6, color='gray', lw=1.4, ls='--',
            label='velocity → all PB_ccw')
    ax.set_ylabel('input (µA)')
    ax.set_title('Inputs (on top of I_0): one EB seed, then velocity pulses to the PBs',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=3)

    ax = axes[1]
    row, labels = 0, []
    for g in GROUP_ORDER:
        for k in range(res['groups'][g].N):
            ts = spikes_of(spk[g], k)
            c = GI_COLOR if g == 'GI' else EB_COLORS[k]
            ax.vlines(ts, row - 0.4, row + 0.4, color=c, lw=0.3)
            labels.append(neuron_name(g, k))
            row += 1
    ax.set_yticks(range(row)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(row - 0.5, -0.5)
    ax.set_title('Spike raster — all 25 neurons', fontweight='bold')

    ax = axes[2]
    for k in range(N_DIR):
        ax.plot(t, r_eb[k], color=EB_COLORS[k], lw=1.4, label=f'EB{k+1}')
    ax.plot(t, r_gi, color=GI_COLOR, lw=1.0, alpha=0.7, label='GI')
    ax.set_ylabel('rate (Hz)')
    ax.set_title('EB and GI firing rates', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=9)

    for ax, g in ((axes[3], 'PB_cw'), (axes[4], 'PB_ccw')):
        for k in range(N_DIR):
            ax.plot(t, r_pb[g][k], color=EB_COLORS[k], lw=1.4,
                    label=neuron_name(g, k))
        ax.set_ylabel('rate (Hz)')
        ax.set_title(f'{g} firing rates — copy of EB, boosted by the velocity pulse',
                     fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=8)

    ax = axes[5]
    for k in range(N_DIR):
        ax.plot(t, i_sh['cw'][k], color=EB_COLORS[k], lw=1.2,
                label=f'PB{(k-1) % N_DIR + 1}_cw → EB{k+1}')
        ax.plot(t, i_sh['ccw'][k], color=EB_COLORS[k], lw=1.2, ls='--',
                label=f'PB{(k+1) % N_DIR + 1}_ccw → EB{k+1}')
    ax.plot(t, i_inh, color='k', lw=0.9, alpha=0.6, label='GI → EB inhibition')
    ax.set_ylabel('current (µA)'); ax.set_xlabel('time (s)')
    ax.set_title('PB → EB shifted-return currents (solid = CW, dashed = CCW)',
                 fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=9)

    for ax in axes:
        for t0, sd in zip(starts, TURN_SEQ):
            ax.axvspan(t0, t0 + TURN_DUR, color='gray', alpha=0.12, lw=0)
    out = os.path.join(os.path.dirname(__file__), 'ns_hd_8n_turn.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")


STAGES = {'connections': stage_connections, 'bump': stage_bump, 'wta': stage_wta,
          'turn': stage_turn}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--stage', choices=STAGES, default='connections')
    STAGES[ap.parse_args().stage]()
