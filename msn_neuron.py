"""
msn_neuron.py
=============
MSN hardware model: MSNParams dataclass and make_msn factory.

Neuron half of the split library.  Synapse counterpart: msn_synapse.py.

    from msn_neuron import MSNParams, make_msn

Parameters can be round-tripped through JSON:

    params = MSNParams.from_json('configs/neuron_default.json')
    params.to_json('configs/my_params.json')

See METHODOLOGY.md §3–5 for physics, equations, and tuning guide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

from brian2 import NeuronGroup, farad, ohm, volt, amp, second


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class MSNParams:
    """Hardware parameters for one MSN population.

    All values in SI units.  Defaults reproduce Wu et al. 2023 Fig. 2.

    Hardware (fixed by the physical device)
    ────────────────────────────────────────
    Cm       Membrane capacitor                   [F]
    Ra       Load resistor (paper "Rload")         [Ω]
    Rm_hi    Memristor open-state resistance       [Ω]
    Rm_lo    Memristor closed-state resistance     [Ω]
    Vth      Thyristor close threshold             [V]
    I_hold   Holding current (reopen threshold)   [A]

    Synaptic filter (can differ per population type)
    ─────────────────────────────────────────────────
    tau_s1   Is1 decay time constant   [s]
    tau_s2   Is2 driven time constant  [s]

    Note on heterogeneity
    ─────────────────────
    Currently tau_s1 / tau_s2 are namespace constants shared across all N
    neurons in a NeuronGroup.  Planned: promote to per-neuron state variables
    so that `G.tau_s1 = np.random.normal(...)` becomes possible (see
    METHODOLOGY.md §10.5).
    """

    Cm:     float = 10e-7      # F     (1 µF, paper value)
    Ra:     float = 47.0       # Ω
    Rm_hi:  float = 100e3      # Ω
    Rm_lo:  float = 500.0      # Ω
    Vth:    float = 1.5        # V
    I_hold: float = 100e-6     # A
    tau_s1: float = 200e-3     # s
    tau_s2: float = 200e-3     # s

    # ── JSON I/O ──────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> MSNParams:
        """Load parameters from a JSON file.  Unknown keys are silently ignored
        so that JSON files can carry documentation fields (e.g. '_units')."""
        with open(path) as f:
            raw = json.load(f)
        known = {k: raw[k] for k in raw if not k.startswith('_')}
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
        """(τ_open, τ_close) in seconds.

        τ_open   = Cm * (Rm_hi + Ra)   charging time constant (~100 ms)
        τ_close  = Cm * (Rm_lo + Ra)   spike width time constant (~5 ms)
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
            f"  tau_s1={self.tau_s1*1e3:.0f} ms   tau_s2={self.tau_s2*1e3:.0f} ms\n"
            f"  → I_min={I_min*1e6:.3f} µA   I_max={I_max*1e6:.0f} µA\n"
            f"  → τ_open={tau_o*1e3:.1f} ms   τ_close={tau_c*1e3:.2f} ms"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Equations                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MSN_EQS = """
dVm/dt      = (I_0 + Is2_exc - Is2_inh - Vm/(Rm_S + Ra)) / Cm  : volt
Rm_S        = (1 - s)*Rm_hi + s*Rm_lo                           : ohm
I_M         = Vm / (Rm_S + Ra)                                  : amp
Vout        = Vm * Ra / (Rm_S + Ra)                             : volt
dIs1_exc/dt = -Is1_exc / tau_s1                                 : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                     : amp
dIs1_inh/dt = -Is1_inh / tau_s1                                 : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                    : amp
I_0         : amp
s           : 1
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Factory                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_msn(N: int,
             params: MSNParams | None = None,
             name: str = 'msn') -> NeuronGroup:
    """Build a NeuronGroup of N MSN neurons.

    Parameters
    ----------
    N      : number of neurons
    params : MSNParams  (Wu et al. 2023 defaults if None)
    name   : Brian2 group name — must be unique per start_scope()

    Returns
    -------
    NeuronGroup with state variables:
        Vm, Vout, I_M, Rm_S   — circuit quantities
        s                     — memristor state (0=open, 1=closed)
        I_0                   — per-neuron tonic bias [A]
        Is1_exc, Is2_exc      — excitatory synaptic cascade [A]
        Is1_inh, Is2_inh      — inhibitory synaptic cascade [A]

    All initialised to 0.  Set per-neuron bias AFTER construction:

        G.I_0 = 18e-6 * amp                      # scalar: same for all
        G.I_0 = np.array([18e-6, 0.0]) * amp     # array:  per-neuron
    """
    if params is None:
        params = MSNParams()

    namespace = dict(
        Cm     = params.Cm     * farad,
        Ra     = params.Ra     * ohm,
        Rm_hi  = params.Rm_hi  * ohm,
        Rm_lo  = params.Rm_lo  * ohm,
        Vth    = params.Vth    * volt,
        I_hold = params.I_hold * amp,
        tau_s1 = params.tau_s1 * second,
        tau_s2 = params.tau_s2 * second,
    )

    G = NeuronGroup(
        N, MSN_EQS,
        threshold = 'Vm > Vth and s < 0.5',
        reset     = 's = 1',
        events    = {'reopen': 'I_M < I_hold and s > 0.5'},
        method    = 'euler',
        namespace = namespace,
        name      = name,
    )
    G.run_on_event('reopen', 's = 0')

    G.Vm      = 0 * volt
    G.s       = 0
    G.I_0     = 0 * amp
    G.Is1_exc = 0 * amp
    G.Is2_exc = 0 * amp
    G.Is1_inh = 0 * amp
    G.Is2_inh = 0 * amp

    return G