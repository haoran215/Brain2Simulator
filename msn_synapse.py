"""
msn_synapse.py
==============
Synaptic connection model: SynapseParams dataclass and make_synapse factory.

Synapse half of the split library.  Neuron counterpart: msn_neuron.py.

    from msn_synapse import SynapseParams, make_synapse

Each Synapses object owns its own filter cascade:

    Is1 → Is2 (two-stage exponential), with per-type tau_s1, tau_s2.
    on_pre:  Is1 += w
    summed:  <target_var>_post = Is2

Biologically this matches the receptor view: AMPA, NMDA, GABA-A, GABA-B
each have their own kinetics; the postsynaptic neuron just sees the
summed current.  Different receptor types onto the same neuron are
modelled as separate Synapses objects with different SynapseParams,
each writing to a distinct named inlet on the target.

Brian2 limitation
─────────────────
Only ONE Synapses object may write to a given (target_var, target_group)
pair via 'summed' (later writes overwrite earlier ones).  When multiple
pathways of the same kind converge on one target group, declare extra
named inlets via make_msn(..., exc_inlets=..., inh_inlets=...) and set
SynapseParams.target_var on each Synapses to a distinct inlet.

See METHODOLOGY.md §3.1 and §4.4 for cascade dynamics and settling times.
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

    weight     Iw added to Is1 per pre-synaptic spike          [A]
               Future hardware: resistance of a non-volatile memristor.
    kind       'exc' → writes an exc inlet (depolarising);
               'inh' → writes an inh inlet (hyperpolarising).
    tau_s1     First-stage filter time constant                [s]
    tau_s2     Second-stage filter time constant               [s]
    delay      Synaptic transmission delay                     [s]
    target_var Optional explicit inlet name on the post group
               (e.g. 'I_inh_mutual').  If None, defaults to
               'I_exc' or 'I_inh' based on kind.  Must match an
               inlet declared on the target via make_msn(...,
               exc_inlets=..., inh_inlets=...).
    """

    weight:     float       = 10e-6     # A
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
    def from_json(cls, path: str, key: str | None = None) -> SynapseParams:
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
    """Build the synapse model string.  target_var is the post inlet name."""
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
    connect: str | bool = 'i == j',
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

              Common patterns
              ───────────────
              'i == j'           1-to-1 (also self-loops when src is tgt)
              'i != j'           all-to-all, no self-loops
              True               all-to-all including self-loops
              'rand() < 0.1'     random sparse, 10% probability

              NOTE: topology is a NETWORK property — not stored in
              SynapseParams.  Pass it here, not in the config file.

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
