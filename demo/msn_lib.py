"""
msn_lib.py
==========
Compatibility shim — re-exports from the canonical split library:

    msn_neuron.py    — MSNParams, make_msn
    msn_synapse.py   — SynapseParams, make_synapse

Older demos in this folder import from `msn_lib`.  The cascade now lives on
the synapse (per-receptor τ), so demos that record `Is2_exc` from the
neuron need to record it from the synapse object instead.  See the
updated v3/v4 demos for examples.

For new code, import directly from msn_neuron / msn_synapse.
"""

from __future__ import annotations

# Make the parent dir importable when running scripts inside demo/.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from msn_neuron  import MSNParams, make_msn, msn_from_json   # noqa: F401
from msn_synapse import SynapseParams
from msn_synapse import make_synapse as _make_synapse_new


def make_synapse(source, target, kind='exc', weight=6e-6,
                 connect='i==j', delay=None, name='syn',
                 tau_s1=200e-3, tau_s2=200e-3, target_var=None):
    """Back-compat wrapper around msn_synapse.make_synapse().

    Old signature took (kind, weight) inline; now SynapseParams owns them
    along with tau_s1/tau_s2 (previously per-neuron).  Both styles work
    through this shim:

        # Old call (still works):
        make_synapse(src, tgt, kind='exc', weight=10e-6, connect='i!=j')

        # Preferred call (msn_synapse directly):
        from msn_synapse import SynapseParams, make_synapse
        p = SynapseParams(weight=10e-6, kind='exc', tau_s1=0.2, tau_s2=0.2)
        make_synapse(src, tgt, params=p, connect='i!=j')
    """
    params = SynapseParams(
        weight     = weight,
        kind       = kind,
        tau_s1     = tau_s1,
        tau_s2     = tau_s2,
        delay      = delay if delay is not None else 0.0,
        target_var = target_var,
    )
    return _make_synapse_new(source, target, params=params,
                             connect=connect, name=name)
