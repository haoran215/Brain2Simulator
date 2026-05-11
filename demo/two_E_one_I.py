"""
two_E_one_I.py
==============
2 excitatory MSN neurons + 1 inhibitory MSN neuron — the simplest motif
where mutual interaction between two competing E units, mediated by a
common inhibitor, can produce non-trivial dynamics.

Goal
────
Investigate whether E1 and E2 settle into:
  - synchronous spiking (both fire together),
  - a fixed point (one wins, the other is silenced),
  - or an alternating oscillation (E1 active, then E2 active, ...).

Timescale hierarchy (the hypothesis to test)
────────────────────────────────────────────
  τ_self-E (thyristor close)   ~ 5 ms      intrinsic to the MSN
  τ_E→I    (fast exc to I)     ~ 10 ms     syn_E_to_I.json
  τ_I→E    (slow global inh)   ~ 150 ms    syn_I_to_E.json
  τ_E↔E    (slow mutual coupling) ~ 500 ms syn_E_to_E.json  ← sweep knob

Modes
─────
  baseline    1E + 1I (E2 silenced)               — periodic E ↔ I oscillation
  no_coupling 2E + 1I, no E↔E                     — do they sync via shared I?
  mutual_exc  2E + 1I + slow mutual EXCITATION
  mutual_inh  2E + 1I + slow mutual INHIBITION

Usage
─────
  python demo/two_E_one_I.py --mode baseline
  python demo/two_E_one_I.py --mode mutual_exc --w_mutual 3e-6
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make repo root importable when running this script from the repo root or
# from inside the demo/ folder.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from brian2 import (
    start_scope, defaultclock,
    StateMonitor, SpikeMonitor, Network,
    prefs, ms, us, second, amp, volt, uA,
)

from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CLI                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--mode',
                   choices=['baseline', 'no_coupling', 'mutual_exc', 'mutual_inh'],
                   default='baseline',
                   help='which experimental condition to run')
    p.add_argument('--w_mutual', type=float, default=None,
                   help='override the mutual-coupling weight (A); '
                        'default = value in syn_E_to_E*.json')
    p.add_argument('--T', type=float, default=3.0,
                   help='simulation duration (s)')
    p.add_argument('--I0_E', type=float, default=30e-6,
                   help='tonic bias on E neurons (A); slightly above I_min')
    p.add_argument('--I0_I', type=float, default=12e-6,
                   help='tonic bias on I neuron (A); sits just below I_min so '
                        'E spikes can push it over threshold')
    p.add_argument('--out', type=str, default=None,
                   help='output figure path; default: demo/two_E_one_I_<mode>.png')
    return p.parse_args()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Build the network                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def build_network(args):
    """Return (E, I, syns, monitors) for the requested mode."""
    cfg_dir = os.path.join(_REPO_ROOT, 'configs')

    E_params = MSNParams.from_json(os.path.join(cfg_dir, 'E_neuron.json'))
    I_params = MSNParams.from_json(os.path.join(cfg_dir, 'I_neuron.json'))

    # E group of 2 neurons; I group of 1.  Mutual-inh mode needs a second
    # inh inlet on E so global-inh and mutual-inh don't clobber each other.
    if args.mode == 'mutual_inh':
        inh_inlets_E = ('I_inh_global', 'I_inh_mutual')
    else:
        inh_inlets_E = ('I_inh',)

    E = make_msn(params=[E_params, E_params], name='E',
                 inh_inlets=inh_inlets_E)
    I = make_msn(params=I_params, name='I')

    # Tonic biases.  Baseline mode silences E2 by zeroing its bias.
    if args.mode == 'baseline':
        E.I_0 = np.array([args.I0_E, 0.0]) * amp
    else:
        E.I_0 = np.array([args.I0_E, args.I0_E]) * amp
    I.I_0 = args.I0_I * amp

    # ── Synapses ──────────────────────────────────────────────────────────
    syns = {}

    # E → I  (fast exc): both E neurons project onto I via one Synapses
    # object so the (summed) inlet on I has exactly one writer.
    p_E_to_I = SynapseParams.from_json(os.path.join(cfg_dir, 'syn_E_to_I.json'))
    syns['E_to_I'] = make_synapse(E, I, params=p_E_to_I,
                                  connect=True, name='syn_E_to_I')

    # I → E  (slow inh): one I neuron onto both E neurons.
    p_I_to_E = SynapseParams.from_json(os.path.join(cfg_dir, 'syn_I_to_E.json'))
    if args.mode == 'mutual_inh':
        # Route global inhibition to the dedicated 'global' inlet so it
        # doesn't conflict with mutual inhibition on the 'mutual' inlet.
        p_I_to_E.target_var = 'I_inh_global'
    syns['I_to_E'] = make_synapse(I, E, params=p_I_to_E,
                                  connect=True, name='syn_I_to_E')

    # E ↔ E  (mutual coupling): only present in coupled modes.
    if args.mode == 'mutual_exc':
        p_E_to_E = SynapseParams.from_json(
            os.path.join(cfg_dir, 'syn_E_to_E.json'))
        if args.w_mutual is not None:
            p_E_to_E.weight = args.w_mutual
        syns['E_to_E'] = make_synapse(E, E, params=p_E_to_E,
                                      connect='i != j', name='syn_E_to_E')
    elif args.mode == 'mutual_inh':
        p_E_to_E = SynapseParams.from_json(
            os.path.join(cfg_dir, 'syn_E_to_E_inh.json'))
        if args.w_mutual is not None:
            p_E_to_E.weight = args.w_mutual
        p_E_to_E.target_var = 'I_inh_mutual'    # dedicated inlet
        syns['E_to_E'] = make_synapse(E, E, params=p_E_to_E,
                                      connect='i != j', name='syn_E_to_E')

    # ── Monitors ──────────────────────────────────────────────────────────
    rec_vars_E = ['Vm', 'Vout', 'I_exc', 'I_inh', 'I_0']
    rec_vars_I = ['Vm', 'Vout', 'I_exc', 'I_0']
    monitors = {
        'st_E': StateMonitor(E, rec_vars_E, record=True, dt=200*us),
        'st_I': StateMonitor(I, rec_vars_I, record=True, dt=200*us),
        'sp_E': SpikeMonitor(E),
        'sp_I': SpikeMonitor(I),
    }

    return E, I, syns, monitors


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Plot                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_results(args, E_params, monitors, out_path):
    st_E = monitors['st_E']; st_I = monitors['st_I']
    sp_E = monitors['sp_E']; sp_I = monitors['sp_I']

    t_ms = np.array(st_E.t / ms)
    Vm_E1 = np.array(st_E.Vm[0] / volt)
    Vm_E2 = np.array(st_E.Vm[1] / volt)
    Vm_I  = np.array(st_I.Vm[0] / volt)
    Iinh_E1 = np.array(st_E.I_inh[0] / amp) * 1e6
    Iinh_E2 = np.array(st_E.I_inh[1] / amp) * 1e6
    Iexc_E1 = np.array(st_E.I_exc[0] / amp) * 1e6
    Iexc_E2 = np.array(st_E.I_exc[1] / amp) * 1e6
    Iexc_I  = np.array(st_I.I_exc[0]  / amp) * 1e6

    sp_E_t = np.array(sp_E.t / ms)
    sp_E_i = np.array(sp_E.i)
    sp_I_t = np.array(sp_I.t / ms)

    C_E1 = '#2980B9'   # blue
    C_E2 = '#E74C3C'   # red
    C_I  = '#16A085'   # teal

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.50,
                           height_ratios=[1.4, 1.0, 1.0, 1.0, 0.8])

    # Row 0: Vm traces
    ax = fig.add_subplot(gs[0])
    ax.plot(t_ms, Vm_E1, color=C_E1, lw=0.6, label='E1', alpha=0.85)
    ax.plot(t_ms, Vm_E2, color=C_E2, lw=0.6, label='E2', alpha=0.85)
    ax.plot(t_ms, Vm_I,  color=C_I,  lw=0.6, label='I',  alpha=0.85)
    ax.axhline(E_params.Vth, color='dimgray', ls='--', lw=1,
               label=f'Vth = {E_params.Vth:.2f} V')
    ax.set_ylabel('Vm (V)')
    ax.set_title(f'mode = {args.mode}  |  Vm traces  '
                 f'(E1={len(sp_E_t[sp_E_i==0])} sp, '
                 f'E2={len(sp_E_t[sp_E_i==1])} sp, '
                 f'I={len(sp_I_t)} sp over {args.T:.1f} s)',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=4)

    # Row 1: E inhibition input
    ax = fig.add_subplot(gs[1])
    ax.plot(t_ms, Iinh_E1, color=C_E1, lw=1.0, label='I_inh on E1')
    ax.plot(t_ms, Iinh_E2, color=C_E2, lw=1.0, label='I_inh on E2')
    ax.set_ylabel('I_inh (µA)')
    ax.set_title('Inhibitory current onto E neurons (from I, plus mutual-inh if active)',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # Row 2: E excitation input
    ax = fig.add_subplot(gs[2])
    ax.plot(t_ms, Iexc_E1, color=C_E1, lw=1.0, label='I_exc on E1')
    ax.plot(t_ms, Iexc_E2, color=C_E2, lw=1.0, label='I_exc on E2')
    ax.set_ylabel('I_exc (µA)')
    ax.set_title('Excitatory current onto E neurons (mutual E↔E, zero in baseline/no_coupling)',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # Row 3: I excitation input (from both E neurons)
    ax = fig.add_subplot(gs[3])
    ax.plot(t_ms, Iexc_I, color=C_I, lw=1.0, label='I_exc on I')
    ax.set_ylabel('I_exc on I (µA)')
    ax.set_title('Excitatory current onto I (driven by E1+E2)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # Row 4: raster
    ax = fig.add_subplot(gs[4])
    ax.scatter(sp_E_t[sp_E_i==0], np.full((sp_E_i==0).sum(), 2),
               color=C_E1, s=12, marker='|', linewidths=1.2, label='E1')
    ax.scatter(sp_E_t[sp_E_i==1], np.full((sp_E_i==1).sum(), 1),
               color=C_E2, s=12, marker='|', linewidths=1.2, label='E2')
    ax.scatter(sp_I_t, np.full(len(sp_I_t), 0),
               color=C_I, s=12, marker='|', linewidths=1.2, label='I')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['I', 'E2', 'E1'])
    ax.set_xlabel('t (ms)')
    ax.set_title('Spike raster', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=3)

    for a in fig.get_axes():
        a.set_xlim(0, args.T * 1000)
        a.grid(axis='x', alpha=0.2)

    fig.suptitle(
        f"2 E + 1 I motif — mode = {args.mode}"
        + (f"   w_mutual = {args.w_mutual*1e6:.2f} µA" if args.w_mutual else ""),
        fontsize=13, fontweight='bold', y=1.002)

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\nFigure saved → {out_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Main                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    args = parse_args()
    out_path = args.out or os.path.join(
        _REPO_ROOT, 'demo', f'two_E_one_I_{args.mode}.png')

    prefs.codegen.target = 'numpy'   # pure-Python backend
    start_scope()
    defaultclock.dt = 20 * us

    cfg_dir = os.path.join(_REPO_ROOT, 'configs')
    E_params = MSNParams.from_json(os.path.join(cfg_dir, 'E_neuron.json'))
    I_min, I_max = E_params.operating_window()
    print(E_params.summary())
    print(f"\nE_min={I_min*1e6:.2f} µA, I_max={I_max*1e6:.0f} µA")
    print(f"Mode: {args.mode}")
    print(f"  I_0 on E: {args.I0_E*1e6:.2f} µA  "
          f"({args.I0_E/I_min*100:.0f}% of rheobase — "
          f"{'above' if args.I0_E > I_min else 'BELOW'} I_min)")
    print(f"  I_0 on I: {args.I0_I*1e6:.2f} µA  "
          f"({args.I0_I/I_min*100:.0f}% of rheobase — "
          f"{'above' if args.I0_I > I_min else 'below'} I_min)")
    if args.w_mutual is not None:
        print(f"  w_mutual override: {args.w_mutual*1e6:.2f} µA")

    E, I, syns, monitors = build_network(args)
    print(f"  Synapse groups: {list(syns.keys())}")
    print()

    # Use an explicit Network so synapses/monitors stored in dicts are
    # actually picked up (Brian2's magic scan can't see into containers).
    net = Network(E, I, *syns.values(), *monitors.values())
    net.run(args.T * second, report='text')

    # Summary stats
    sp_E_t = np.array(monitors['sp_E'].t / ms)
    sp_E_i = np.array(monitors['sp_E'].i)
    sp_I_t = np.array(monitors['sp_I'].t / ms)
    print(f"\n  Spikes: E1={int((sp_E_i==0).sum())}, "
          f"E2={int((sp_E_i==1).sum())}, I={len(sp_I_t)}")

    plot_results(args, E_params, monitors, out_path)


if __name__ == '__main__':
    main()
