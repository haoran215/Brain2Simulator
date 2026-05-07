"""
msn_synapse.py
==============
Synaptic connection model: SynapseParams dataclass and make_synapse factory.

Synapse half of the split library.  Neuron counterpart: msn_neuron.py.

    from msn_synapse import SynapseParams, make_synapse

Parameters can be round-tripped through JSON:

    params = SynapseParams.from_json('configs/synapse_inh.json')
    params.to_json('configs/my_synapse.json')

Design note — what belongs in SynapseParams vs make_synapse()
──────────────────────────────────────────────────────────────
  SynapseParams holds INTRINSIC synapse properties:
    weight   The current kick Iw added to Is1 per pre-spike  [A]
    kind     Whether it targets Is1_exc or Is1_inh           'exc'|'inh'
    delay    Axonal transmission delay                        [s]

  make_synapse() takes TOPOLOGY as an argument, not a stored param:
    connect  Brian2 condition string ('i==j', 'rand()<0.1', …)
    name     Brian2 object name

  tau_s1 / tau_s2 are POST-synaptic neuron properties set via MSNParams.
  They are NOT stored here.

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
    """Intrinsic parameters of one synapse type.

    weight   Iw added to Is1_{kind} per pre-synaptic spike  [A]
             Future hardware: this is the resistance of a non-volatile
             memristive device (see METHODOLOGY.md §10.3).
    kind     'exc' → targets Is1_exc;  'inh' → targets Is1_inh
    delay    Synaptic transmission delay                     [s]
    """

    weight: float = 6e-6     # A
    kind:   str   = 'exc'    # 'exc' | 'inh'
    delay:  float = 0.0      # s

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

        Examples
        --------
        # single-type file
        params = SynapseParams.from_json('configs/synapse_default.json', key='inh')

        # or load the flat top-level dict
        params = SynapseParams.from_json('configs/synapse_flat.json')
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
             f"  kind={self.kind}   weight={self.weight*1e6:.3f} µA")
        if self.delay > 0:
            s += f"   delay={self.delay*1e3:.1f} ms"
        return s


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
    params  : SynapseParams  (excitatory 6 µA defaults if None)
    connect : Brian2 connection condition string, or True for all-to-all.

              Common patterns
              ───────────────
              'i == j'           1-to-1  (also self-loops when src is tgt)
              'i != j'           all-to-all, no self-loops
              True               all-to-all including self-loops
              'rand() < 0.1'     random sparse, 10% probability
              'abs(i-j) <= 2'    local band (ring topology)
              'abs(i-j) <= 2 and i != j'   local band, no self-loops

              NOTE: topology is a NETWORK property — it is not stored in
              SynapseParams.  Pass it here, not in the config file.

    name    : Brian2 object name — must be unique per start_scope()

    Returns
    -------
    Synapses  with per-edge weight variable `w` [A].
    `w` is addressable for plasticity or manual hetero-weighting:

        syn.w = 10e-6 * amp              # uniform scalar
        syn.w = np.random.normal(...)    # heterogeneous array
        syn.w['i==0'] = 20e-6 * amp     # subset assignment
    """
    if params is None:
        params = SynapseParams()

    target_var = f'Is1_{params.kind}_post'

    syn = Synapses(
        source, target,
        model  = 'w : amp',
        on_pre = f'{target_var} += w',
        name   = name,
    )

    if connect is True:
        syn.connect()
    else:
        syn.connect(condition=connect)

    syn.w = params.weight * amp

    if params.delay > 0:
        syn.delay = params.delay * second

    return syn