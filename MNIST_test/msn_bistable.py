"""
msn_bistable.py
===============
Brian2 factory for the Stage-2 "MSN_HH" neuron of docs/MSN_HH_GATE_PLAN.md:
the discrete memristor switch s is replaced by a continuous, self-latching
gate x with bistable (cubic) kinetics.

    tau_x dx/dt = -x (x - θ)(x - 1) + eps_x (1 - 2x)
    θ = 0.5 - 2·σ((Vm - Vth)/k_V) + σ((I_hold - I_M)/k_I)
    Rm_S = (1 - x)·Rm_hi + x·Rm_lo          (resistance-linear, as in msn_neuron)

eps_x is a tiny leak that keeps x off the exact fixed points 0 and 1
(the prototype clipped instead).  Exposes the same inlets (I_exc, I_inh, I_0)
and spike event as make_msn, so make_synapse works unchanged.

Local to MNIST_test/ — the library (msn_neuron.py) is not modified.
"""

from __future__ import annotations

from brian2 import NeuronGroup, farad, ohm, volt, amp, second

from msn_neuron import MSNParams


EQS = """
dVm/dt = (I_0 + I_exc - I_inh - Vm/(Rm_S + Ra)) / Cm              : volt
dx/dt  = (-x*(x - theta)*(x - 1) + eps_x*(1 - 2*x)) / tau_x        : 1
theta  = 0.5 - 2/(1 + exp(-(Vm - Vth)/k_V)) + 1/(1 + exp(-(I_hold - I_M)/k_I)) : 1
Rm_S   = (1 - x)*Rm_hi + x*Rm_lo                                    : ohm
I_M    = Vm / (Rm_S + Ra)                                           : amp
Vout   = Vm * Ra / (Rm_S + Ra)                                      : volt
I_exc  : amp
I_inh  : amp
I_0    : amp
Vth    : volt
I_hold : amp
"""


def make_msn_bistable(
    N: int,
    params: MSNParams | None = None,
    k_V: float = 5e-3,       # V
    k_I: float = 0.5e-6,     # A
    tau_x: float = 1e-6,     # s
    eps_x: float = 1e-6,
    name: str = 'msn_hh',
) -> NeuronGroup:
    """NeuronGroup of N bistable-gate MSNs (needs dt ≲ tau_x / 10)."""
    if params is None:
        params = MSNParams()
    G = NeuronGroup(
        N, EQS,
        threshold  = 'x > 0.5',
        refractory = 'x > 0.1',      # one spike per closing edge
        method     = 'euler',
        namespace  = dict(
            Cm=params.Cm * farad, Ra=params.Ra * ohm,
            Rm_hi=params.Rm_hi * ohm, Rm_lo=params.Rm_lo * ohm,
            k_V=k_V * volt, k_I=k_I * amp, tau_x=tau_x * second, eps_x=eps_x,
        ),
        name = name,
    )
    G.Vm = 0 * volt
    G.x = eps_x
    G.I_0 = G.I_exc = G.I_inh = 0 * amp
    G.Vth = params.Vth * volt
    G.I_hold = params.I_hold * amp
    return G
