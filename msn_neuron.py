"""
msn_neuron.py
=============
MSN hardware model: MSNParams dataclass and make_msn factory.

Neuron half of the split library.  Synapse counterpart: msn_synapse.py.

    from msn_neuron import MSNParams, make_msn

Architecture change (cascade-on-synapse)
─────────────────────────────────────────
The synaptic cascade ODE (Is1 → Is2) lives on the Synapses object now,
not on the neuron — see msn_synapse.py.  The neuron exposes summed
inlets that synapses write into:

    I_exc : amp   ← written by ONE exc Synapses group (via 'summed')
    I_inh : amp   ← written by ONE inh Synapses group

Multiple pathways with different time constants
────────────────────────────────────────────────
When more than one synapse type of the same kind converges on a neuron
group (e.g. fast E→I exc AND slow mutual E↔E exc onto the same E
neuron), declare extra named inlets at construction:

    E = make_msn(params=[p_E1, p_E2], name='E',
                 inh_inlets=('I_inh_global', 'I_inh_mutual'))

Each Synapses then targets one inlet via SynapseParams.target_var.  The
neuron's Vm ODE always uses the totals I_exc and I_inh; when multiple
inlets are declared, the total becomes a subexpression summing them.

Per-neuron heterogeneity
─────────────────────────
Intrinsic params (Cm, Ra, Rm_hi, Rm_lo, Vth, I_hold) are state variables,
so each neuron in a NeuronGroup can carry its own values.  Load one JSON
per neuron and pass the list:

    p1 = MSNParams.from_json('configs/E1.json')
    p2 = MSNParams.from_json('configs/E2.json')
    E  = make_msn(params=[p1, p2], name='E')   # N=2, heterogeneous

Migration note
──────────────
tau_s1/tau_s2 moved to SynapseParams.  Legacy JSON keys are accepted but
ignored — load synaptic time constants from synapse configs instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from brian2 import NeuronGroup, farad, ohm, volt, amp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class MSNParams:
    """Intrinsic hardware parameters for one MSN neuron.

    All values in SI units.  Defaults reproduce Wu et al. 2023 Fig. 2.

    Hardware (fixed by the physical device)
    ────────────────────────────────────────
    Cm       Membrane capacitor                   [F]
    Ra       Load resistor (paper "Rload")         [Ω]
    Rm_hi    Memristor open-state resistance       [Ω]
    Rm_lo    Memristor closed-state resistance     [Ω]
    Vth      Thyristor close threshold             [V]
    I_hold   Holding current (reopen threshold)    [A]

    Note: synaptic time constants (tau_s1, tau_s2) live on the SYNAPSE
    object now.  See msn_synapse.SynapseParams.
    """

    Cm:     float = 10e-7      # F     (1 µF, paper value)
    Ra:     float = 47.0       # Ω
    Rm_hi:  float = 100e3      # Ω
    Rm_lo:  float = 500.0      # Ω
    Vth:    float = 1.5        # V
    I_hold: float = 100e-6     # A

    # ── JSON I/O ──────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> 'MSNParams':
        """Load parameters from a JSON file.  Unknown keys (and the legacy
        tau_s1/tau_s2 keys, which now belong to SynapseParams) are silently
        ignored so that old config files keep loading."""
        with open(path) as f:
            raw = json.load(f)
        known = {k: raw[k] for k in raw if not k.startswith('_')}
        for legacy in ('tau_s1', 'tau_s2'):
            known.pop(legacy, None)
        return cls(**known)

    def to_json(self, path: str) -> None:
        """Save parameters to a JSON file (SI units, no Brian2 objects)."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    # ── Derived quantities ────────────────────────────────────────────────────

    def operating_window(self) -> tuple[float, float]:
        """(I_min, I_max) in amps — the spiking current window.

        I_min  rheobase         = Vth / (Rm_hi + Ra)
        I_max  depol-block onset = I_hold
        """
        return self.Vth / (self.Rm_hi + self.Ra), self.I_hold

    def time_constants(self) -> tuple[float, float]:
        """(τ_open, τ_close) in seconds — intrinsic membrane time constants.

        τ_open   = Cm * (Rm_hi + Ra)
        τ_close  = Cm * (Rm_lo + Ra)
        """
        return (self.Cm * (self.Rm_hi + self.Ra),
                self.Cm * (self.Rm_lo + self.Ra))

    def summary(self) -> str:
        I_min, I_max = self.operating_window()
        tau_o, tau_c = self.time_constants()
        return (
            f"MSNParams:\n"
            f"  Cm={self.Cm*1e6:.2f} µF   Ra={self.Ra:.1f} Ω\n"
            f"  Rm_hi={self.Rm_hi/1e3:.0f} kΩ   Rm_lo={self.Rm_lo:.0f} Ω\n"
            f"  Vth={self.Vth:.3f} V   I_hold={self.I_hold*1e6:.0f} µA\n"
            f"  → I_min={I_min*1e6:.3f} µA   I_max={I_max*1e6:.0f} µA\n"
            f"  → τ_open={tau_o*1e3:.1f} ms   τ_close={tau_c*1e3:.2f} ms"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Equations                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# All intrinsic params (Cm, Ra, Rm_hi, Rm_lo, Vth, I_hold) are state
# variables so each neuron in the group can be configured independently
# from JSON.
#
# Synaptic inlets — each is a (summed) target written by ONE Synapses
# group (Brian2 limitation: multiple summed writers to the same variable
# overwrite each other).  To support multiple pathways of the same kind
# onto one neuron group (e.g. global-inh + mutual-inh), declare multiple
# inlets at construction time and route each Synapses to a unique one.
# Defaults are 'I_exc' and 'I_inh'.  When extra inlets are declared,
# I_exc and I_inh become subexpressions that sum the inlets.

def build_msn_eqs(exc_inlets: Sequence[str] = ('I_exc',),
                  inh_inlets: Sequence[str] = ('I_inh',)) -> str:
    """Construct the MSN equation string with the requested summed inlets.

    Each inlet appears as a state variable `<name> : amp`; Synapses target
    one inlet via `target_var` in SynapseParams.  The neuron's Vm ODE
    always uses the totals `I_exc` and `I_inh`.

    Conventions
    -----------
    - Single exc inlet named 'I_exc' (the default): no extra alias needed.
    - Multiple exc inlets: must all be distinct names, none of them
      'I_exc' (which becomes the derived subexpression `I_exc = sum...`).
    - Same rules for inh inlets / 'I_inh'.
    """
    exc_inlets = tuple(exc_inlets) if exc_inlets else ('I_exc',)
    inh_inlets = tuple(inh_inlets) if inh_inlets else ('I_inh',)

    if len(exc_inlets) > 1 and 'I_exc' in exc_inlets:
        raise ValueError(
            "Multi-inlet exc must use distinct names, not 'I_exc' "
            "(which becomes the derived sum)."
        )
    if len(inh_inlets) > 1 and 'I_inh' in inh_inlets:
        raise ValueError(
            "Multi-inlet inh must use distinct names, not 'I_inh'."
        )

    exc_inlet_decls = "\n".join(f"{n} : amp" for n in exc_inlets)
    inh_inlet_decls = "\n".join(f"{n} : amp" for n in inh_inlets)
    exc_alias = "" if exc_inlets == ('I_exc',) else \
                f"I_exc = {' + '.join(exc_inlets)} : amp"
    inh_alias = "" if inh_inlets == ('I_inh',) else \
                f"I_inh = {' + '.join(inh_inlets)} : amp"

    eqs = f"""
dVm/dt   = (I_0 + I_exc - I_inh - Vm/(Rm_S + Ra)) / Cm   : volt
Rm_S     = (1 - s)*Rm_hi + s*Rm_lo                        : ohm
I_M      = Vm / (Rm_S + Ra)                               : amp
Vout     = Vm * Ra / (Rm_S + Ra)                          : volt
{exc_alias}
{inh_alias}
{exc_inlet_decls}
{inh_inlet_decls}
I_0      : amp
s        : 1
Cm       : farad (constant)
Ra       : ohm   (constant)
Rm_hi    : ohm   (constant)
Rm_lo    : ohm   (constant)
Vth      : volt  (constant)
I_hold   : amp   (constant)
"""
    return eqs


# Default equation set — single I_exc + I_inh inlet (most networks use this).
MSN_EQS = build_msn_eqs()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Factory                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_msn(params: 'MSNParams | Sequence[MSNParams] | None' = None,
             N: int | None = None,
             name: str = 'msn',
             exc_inlets: Sequence[str] = ('I_exc',),
             inh_inlets: Sequence[str] = ('I_inh',)) -> NeuronGroup:
    """Build a NeuronGroup of MSN neurons.

    Parameters
    ----------
    params : MSNParams, list[MSNParams], or None
        - Single MSNParams: broadcast to all N neurons (homogeneous group).
        - List of MSNParams: one per neuron, length sets N (heterogeneous).
        - None: defaults to one neuron with paper defaults.
    N : int or None
        Number of neurons.  Required when params is a single MSNParams and
        N>1.  Must equal len(params) when params is a list.
    name : str
        Brian2 group name — must be unique per start_scope().
    exc_inlets, inh_inlets : tuple[str, ...]
        Names of the (summed) synaptic inlets to declare on the neuron.
        Defaults are ('I_exc',) and ('I_inh',).  Pass extra names when
        multiple distinct pathways of the same kind need to target this
        group (e.g. ('I_inh', 'I_inh_mutual') so global-inh and mutual-inh
        can both write without 'summed' overwrite).

    Returns
    -------
    NeuronGroup with state variables:
        Vm, Vout, I_M, Rm_S            — circuit quantities
        s                              — memristor state (0=open, 1=closed)
        I_0                            — per-neuron tonic bias [A]
        <inlet names>                  — summed synaptic inlets [A]
        Cm, Ra, Rm_hi, Rm_lo, Vth, I_hold — per-neuron intrinsic params

    All dynamic state initialised to 0.  Per-neuron intrinsic params are
    initialised from `params`.
    """
    if params is None:
        params = MSNParams()
    if isinstance(params, MSNParams):
        if N is None:
            N = 1
        params_list = [params] * N
    else:
        params_list = list(params)
        if N is not None and N != len(params_list):
            raise ValueError(
                f"N={N} disagrees with len(params)={len(params_list)}"
            )
        N = len(params_list)

    eqs = build_msn_eqs(exc_inlets=exc_inlets, inh_inlets=inh_inlets)

    G = NeuronGroup(
        N, eqs,
        threshold = 'Vm > Vth and s < 0.5',
        reset     = 's = 1',
        events    = {'reopen': 'I_M < I_hold and s > 0.5'},
        method    = 'euler',
        name      = name,
    )
    G.run_on_event('reopen', 's = 0')

    G.Vm  = 0 * volt
    G.s   = 0
    G.I_0 = 0 * amp
    for inlet in tuple(exc_inlets) + tuple(inh_inlets):
        setattr(G, inlet, 0 * amp)

    G.Cm     = np.array([p.Cm     for p in params_list]) * farad
    G.Ra     = np.array([p.Ra     for p in params_list]) * ohm
    G.Rm_hi  = np.array([p.Rm_hi  for p in params_list]) * ohm
    G.Rm_lo  = np.array([p.Rm_lo  for p in params_list]) * ohm
    G.Vth    = np.array([p.Vth    for p in params_list]) * volt
    G.I_hold = np.array([p.I_hold for p in params_list]) * amp

    return G


def msn_from_json(paths: 'str | Sequence[str]', name: str = 'msn') -> NeuronGroup:
    """Convenience loader: build a NeuronGroup directly from JSON path(s).

    Examples
    --------
        I = msn_from_json('configs/I_neuron.json', name='I')         # N=1
        E = msn_from_json(['configs/E1.json', 'configs/E2.json'],    # N=2
                          name='E')
    """
    if isinstance(paths, str):
        return make_msn(params=MSNParams.from_json(paths), name=name)
    return make_msn(params=[MSNParams.from_json(p) for p in paths], name=name)
