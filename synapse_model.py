"""
synapse_model.py
----------------
Core memristive synapse model for Brain2Simulator.

The synapse conductance follows an exponential decay, while the synaptic
weight *w* represents the memristance state (normalised between ``w_min``
and ``w_max``).  Spike-timing-dependent plasticity (STDP) traces are also
included so that weight updates can be applied in downstream task branches.

Parameters are loaded from parameters.json so that they can be easily
modified without touching the model code.
"""

import json
import os

from brian2 import (
    Synapses,
    NeuronGroup,
    ms,
    mV,
    siemens,
)

_PARAMS_FILE = os.path.join(os.path.dirname(__file__), "parameters.json")


def load_params(params_file: str = _PARAMS_FILE) -> dict:
    """Load parameters from a JSON file."""
    with open(params_file, "r") as f:
        return json.load(f)


def create_synapses(
    source: NeuronGroup,
    target: NeuronGroup,
    params: dict | None = None,
    connect: str | bool = True,
) -> Synapses:
    """Create a Brian2 Synapses object implementing the memristive synapse model.

    The synapse model consists of:

    * An exponentially decaying conductance *g* that contributes a current
      ``I_syn = w * g_max * g * (V_target - E_rev)`` to the post-synaptic
      neuron (the post-synaptic current must be wired up in the target
      NeuronGroup equations by the caller).
    * A synaptic weight *w* that represents the normalised memristance state
      (0 = high resistance / off, 1 = low resistance / on).
    * Pre- and post-synaptic STDP eligibility traces (*apre* / *apost*) that
      can be used for weight updates in task-specific branches.

    Parameters
    ----------
    source : NeuronGroup
        Pre-synaptic neuron group.
    target : NeuronGroup
        Post-synaptic neuron group.
    params : dict, optional
        Full parameter dictionary (as returned by :func:`load_params`).  If
        *None*, parameters are loaded from ``parameters.json``.
    connect : str or bool, optional
        Connectivity specification forwarded to ``Synapses.connect()``.
        Defaults to ``True`` (all-to-all).

    Returns
    -------
    Synapses
        A Brian2 Synapses object ready to be used in a Network.
    """
    if params is None:
        params = load_params()

    sp = params["synapse"]
    stdp = params["stdp"]

    tau_syn = sp["tau_syn"] * ms
    w_min = sp["w_min"]
    w_max = sp["w_max"]
    g_max = sp["g_max"] * siemens
    tau_pre = stdp["tau_pre"] * ms
    tau_post = stdp["tau_post"] * ms
    A_pre = stdp["A_pre"]
    A_post = stdp["A_post"]

    model = """
    w          : 1                    # normalised memristance state
    dg/dt      = -g / tau_syn : siemens (clock-driven)  # exponentially decaying synaptic conductance
    dapre/dt   = -apre / tau_pre   : 1 (event-driven)
    dapost/dt  = -apost / tau_post : 1 (event-driven)
    """

    on_pre = """
    g += w * g_max
    apre += A_pre
    w = clip(w + apost, w_min, w_max)
    """

    on_post = """
    apost += A_post
    w = clip(w + apre, w_min, w_max)
    """

    synapses = Synapses(
        source,
        target,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        namespace={
            "tau_syn": tau_syn,
            "g_max": g_max,
            "tau_pre": tau_pre,
            "tau_post": tau_post,
            "A_pre": A_pre,
            "A_post": A_post,
            "w_min": w_min,
            "w_max": w_max,
        },
    )

    synapses.connect(connect)
    synapses.w = sp["w_init"]
    return synapses
