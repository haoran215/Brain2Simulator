"""
msn_synapse.py
==============
Synaptic connection model: SynapseParams dataclass and make_synapse factory.

Synapse half of the split library.  Neuron counterpart: msn_neuron.py.

    from msn_synapse import SynapseParams, make_synapse

Architecture change (cascade-on-synapse)
─────────────────────────────────────────
Each Synapses object now owns its own filter cascade:

    Is1 → Is2   (two-stage exponential)
    on_pre:    Is1 += w
    summed:    <target_var>_post = Is2

Biologically this matches the receptor view: AMPA, NMDA, GABA-A, GABA-B
each have their own kinetics; the postsynaptic neuron just sees the
summed current.  Different receptor types onto the same neuron are
modelled as separate Synapses objects, each writing to a distinct named
inlet on the target (see msn_neuron.make_msn(exc_inlets=, inh_inlets=)).

Migration note
──────────────
- tau_s1, tau_s2 now live in SynapseParams (used to be on MSNParams).
- The neuron's Is1_exc / Is2_exc / Is1_inh / Is2_inh state variables
  no longer exist.  Synapses write to summed inlets I_exc / I_inh
  (or user-named inlets) instead.  Replace recordings like
  `StateMonitor(G, 'Is2_exc')` with either `StateMonitor(G, 'I_exc')`
  (the total) or `StateMonitor(syn_obj, 'Is2')` (per-edge).
- Raw `on_pre='Is1_exc_post += w'` strings in custom Synapses break.
  Either use make_synapse() (which sets up the cascade for you) or write
  the cascade inline in your custom Synapses model (see _syn_eqs below).

Brian2 limitation
─────────────────
Only ONE Synapses object may write to a given (target_var, target_group)
pair via 'summed' (later writes overwrite earlier ones).  When multiple
pathways of the same kind converge on one target group, declare extra
named inlets via make_msn(..., exc_inlets=..., inh_inlets=...) and set
SynapseParams.target_var on each Synapses to a distinct inlet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

from brian2 import Synapses, amp, second


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class SynapseParams:
    """Intrinsic parameters of one synapse / receptor type.

    weight     Iw added to Is1 per pre-synaptic spike            [A]
               Future hardware: resistance of a non-volatile memristor.
    kind       'exc' → writes an exc inlet (depolarising);
               'inh' → writes an inh inlet (hyperpolarising).
    tau_s1     First-stage filter time constant                  [s]
    tau_s2     Second-stage filter time constant                 [s]
    delay      Synaptic transmission delay                       [s]
    target_var Optional explicit inlet name on the post group
               (e.g. 'I_inh_mutual').  If None, defaults to
               'I_exc' or 'I_inh' based on kind.  Must match an
               inlet declared on the target via make_msn(...,
               exc_inlets=..., inh_inlets=...).
    """

    weight:     float       = 6e-6     # A
    kind:       str         = 'exc'    # 'exc' | 'inh'
    tau_s1:     float       = 200e-3   # s
    tau_s2:     float       = 200e-3   # s
    delay:      float       = 0.0      # s
    target_var: str | None  = None

    def __post_init__(self):
        if self.kind not in ('exc', 'inh'):
            raise ValueError(f"kind must be 'exc' or 'inh', got {self.kind!r}")

    # ── JSON I/O ──────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str, key: str | None = None) -> 'SynapseParams':
        """Load parameters from a JSON file.

        Parameters
        ----------
        path : path to the JSON file
        key  : if the JSON contains a dict of named synapse types
               (e.g. {"exc": {...}, "inh": {...}}), pass the key to
               select one entry.  If None, the top-level object is used.

        Keys starting with '_' are treated as documentation and ignored.
        """
        with open(path) as f:
            raw = json.load(f)
        if key is not None:
            raw = raw[key]
        known = {k: raw[k] for k in raw if not k.startswith('_')}
        return cls(**known)

    def to_json(self, path: str) -> None:
        """Save parameters to a JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def summary(self) -> str:
        s = (f"SynapseParams:\n"
             f"  kind={self.kind}   weight={self.weight*1e6:.3f} µA\n"
             f"  tau_s1={self.tau_s1*1e3:.1f} ms   tau_s2={self.tau_s2*1e3:.1f} ms")
        if self.delay > 0:
            s += f"\n  delay={self.delay*1e3:.1f} ms"
        if self.target_var:
            s += f"\n  target_var={self.target_var}"
        return s


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Equations (cascade lives on the synapse)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _syn_eqs(target_var: str) -> str:
    """Build the synapse model string.  target_var is the post inlet name.

    Custom Synapses (e.g. with STDP traces) can copy this body and add
    their own trace variables:

        model = '''
            dIs1/dt = -Is1 / tau_s1                : amp (clock-driven)
            dIs2/dt = (-Is2 + Is1) / tau_s2        : amp (clock-driven)
            I_exc_post = Is2                       : amp (summed)
            w     : amp
            dApre/dt  = -Apre  / tau_pre  : 1 (event-driven)
            dApost/dt = -Apost / tau_post : 1 (event-driven)
        '''
    """
    return f"""
        dIs1/dt = -Is1 / tau_s1                  : amp (clock-driven)
        dIs2/dt = (-Is2 + Is1) / tau_s2          : amp (clock-driven)
        {target_var}_post = Is2                  : amp (summed)
        w : amp
    """


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Factory                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_synapse(
    source,
    target,
    params: SynapseParams | None = None,
    connect: 'str | bool' = 'i == j',
    name: str = 'syn',
) -> Synapses:
    """Connect source to target MSN group.

    Parameters
    ----------
    source  : SpikeSource — NeuronGroup, PoissonGroup, SpikeGeneratorGroup, …
    target  : NeuronGroup built by make_msn
    params  : SynapseParams (excitatory defaults if None).  Carries kind,
              weight, tau_s1, tau_s2, delay, target_var.
    connect : Brian2 connection condition string, or True for all-to-all.
    name    : Brian2 object name — must be unique per start_scope()

    Returns
    -------
    Synapses with per-edge weight `w` [A] and per-edge cascade state
    `Is1`, `Is2` [A].  Use `syn.w = ...` for heterogeneous weights.
    """
    if params is None:
        params = SynapseParams()

    # Default target is I_exc or I_inh; user may override to a named inlet
    # (e.g. 'I_inh_mutual') if the target group declared one via make_msn.
    target_var = params.target_var or f"I_{params.kind}"

    syn = Synapses(
        source, target,
        model     = _syn_eqs(target_var),
        on_pre    = 'Is1 += w',
        method    = 'euler',
        namespace = {
            'tau_s1': params.tau_s1 * second,
            'tau_s2': params.tau_s2 * second,
        },
        name      = name,
    )

    if connect is True:
        syn.connect()
    else:
        syn.connect(condition=connect)

    syn.w = params.weight * amp
    syn.Is1 = 0 * amp
    syn.Is2 = 0 * amp

    if params.delay > 0:
        syn.delay = params.delay * second

    return syn
