"""
neurons.py
==========
The three neuron models compared throughout the MSN + STDP exploration.

All three are current-based and expose the same interface as msn_neuron.make_msn:
state variable Vm, inlets I_0 (tonic bias), I_exc, I_inh [A], per-neuron Vth,
and a spike event at threshold crossing.  An identical input current therefore
means an identical experiment; only the neuron model changes.

  msn          msn_neuron.make_msn with configs/neuron_hardware_fit.json.
               Open (charging) state: τ_open = Cm(Rm_hi+Ra) = 6.31 ms.
               Spike: Vm > Vth closes the switch (s = 1), Cm discharges
               through Rm_lo + Ra; reopens when I_M < I_hold.  Latches
               (depolarisation block) for I_in > I_hold = 95 µA.

  lif_matched  Same R = Rm_hi+Ra, Cm, Vth as the MSN.  Instant reset to the
               MSN reopen voltage V_r = I_hold(Rm_lo+Ra) and a fixed refractory
               period equal to the MSN closed time at mid-window (65 µA).
               What remains different from the MSN: no latching above I_hold,
               and a fixed (not input-dependent) spike duration with Vm clamped
               instead of discharging.  This is the "MSN without the memristive
               mechanism" control.

  lif_generic  Textbook LIF put on the same current scale (same R and Vth, so the
               same rheobase 39.6 µA) but with conventional dynamics:
               τ_m = 20 ms, reset to 0 V, t_ref = 2 ms.
"""

from __future__ import annotations

import math
import os

from brian2 import NeuronGroup, farad, ohm, volt, amp, second

from msn_neuron import MSNParams, make_msn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HW_PARAMS_PATH = os.path.join(_ROOT, 'configs', 'neuron_hardware_fit.json')

LIF_GENERIC_TAU = 20e-3     # s
LIF_GENERIC_TREF = 2e-3     # s
LIF_MATCHED_I_REF = 65e-6   # A — current at which the MSN closed time sets t_ref


def hw_params() -> MSNParams:
    return MSNParams.from_json(HW_PARAMS_PATH)


def msn_reopen_voltage(p: MSNParams) -> float:
    return p.I_hold * (p.Rm_lo + p.Ra)


def msn_closed_time(p: MSNParams, I: float) -> float:
    """Analytic MSN closed-state duration at constant input I (< I_hold) [s]."""
    _, tau_c = p.time_constants()
    Vc = I * (p.Rm_lo + p.Ra)
    return tau_c * math.log((p.Vth - Vc) / (msn_reopen_voltage(p) - Vc))


def lif_config(kind: str, p: MSNParams | None = None) -> dict:
    """SI parameters of a LIF variant (for logging and for the factory)."""
    p = p or hw_params()
    R = p.Rm_hi + p.Ra
    if kind == 'lif_matched':
        return dict(R=R, C=p.Cm, Vth=p.Vth, V_reset=msn_reopen_voltage(p),
                    t_ref=msn_closed_time(p, LIF_MATCHED_I_REF))
    if kind == 'lif_generic':
        return dict(R=R, C=LIF_GENERIC_TAU / R, Vth=p.Vth, V_reset=0.0,
                    t_ref=LIF_GENERIC_TREF)
    raise ValueError(kind)


_LIF_EQS = """
dVm/dt = (I_0 + I_exc - I_inh - Vm/R) / C : volt (unless refractory)
I_exc : amp
I_inh : amp
I_0   : amp
Vth   : volt
"""


def make_lif(N: int, kind: str, p: MSNParams | None = None, name: str | None = None) -> NeuronGroup:
    c = lif_config(kind, p)
    G = NeuronGroup(
        N, _LIF_EQS,
        threshold='Vm > Vth',
        reset='Vm = V_reset',
        refractory=c['t_ref'] * second,
        method='euler',
        namespace=dict(R=c['R'] * ohm, C=c['C'] * farad, V_reset=c['V_reset'] * volt),
        name=name or kind,
    )
    G.Vm = 0 * volt
    G.I_0 = G.I_exc = G.I_inh = 0 * amp
    G.Vth = c['Vth'] * volt
    return G


def make_neuron(kind: str, N: int, p: MSNParams | None = None, name: str | None = None) -> NeuronGroup:
    """Factory for 'msn', 'lif_matched', 'lif_generic'."""
    p = p or hw_params()
    if kind == 'msn':
        return make_msn(N, params=p, name=name or 'msn')
    return make_lif(N, kind, p, name=name)


MODELS = ('lif_generic', 'lif_matched', 'msn')
