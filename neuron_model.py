"""
neuron_model.py
---------------
Core leaky integrate-and-fire (LIF) neuron model for Brain2Simulator.

Parameters are loaded from parameters.json so that they can be easily
modified without touching the model code.
"""

import json
import os

from brian2 import (
    NeuronGroup,
    ms,
    mV,
    Mohm,
    nA,
    defaultclock,
    start_scope,
)

_PARAMS_FILE = os.path.join(os.path.dirname(__file__), "parameters.json")


def load_params(params_file: str = _PARAMS_FILE) -> dict:
    """Load parameters from a JSON file."""
    with open(params_file, "r") as f:
        return json.load(f)


def create_neuron_group(N: int, params: dict | None = None) -> NeuronGroup:
    """Create a Brian2 NeuronGroup implementing the LIF neuron model.

    Parameters
    ----------
    N : int
        Number of neurons in the group.
    params : dict, optional
        Dictionary containing neuron parameters. If *None*, parameters are
        loaded from ``parameters.json``.

    Returns
    -------
    NeuronGroup
        A Brian2 NeuronGroup with the LIF dynamics.
    """
    if params is None:
        params = load_params()

    p = params["neuron"]

    V_rest = p["V_rest"] * mV
    V_threshold = p["V_threshold"] * mV
    V_reset = p["V_reset"] * mV
    R_m = p["R_m"] * Mohm
    tau_m = p["tau_m"] * ms
    I_ext = p["I_ext"] * nA
    refractory = p["refractory_period"] * ms

    eqs = """
    dV/dt = (-(V - V_rest) + R_m * I_ext) / tau_m : volt (unless refractory)
    """

    group = NeuronGroup(
        N,
        model=eqs,
        threshold="V >= V_threshold",
        reset="V = V_reset",
        refractory=refractory,
        method="euler",
        namespace={
            "V_rest": V_rest,
            "V_threshold": V_threshold,
            "V_reset": V_reset,
            "R_m": R_m,
            "tau_m": tau_m,
            "I_ext": I_ext,
        },
    )

    group.V = V_rest
    return group
