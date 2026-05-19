"""
msn_neuron.py
=============
MSN hardware model: MSNParams dataclass and make_msn factory.

Neuron half of the split library.  Synapse counterpart: msn_synapse.py.

    from msn_neuron import MSNParams, make_msn

Parameters can be round-tripped through JSON:

    params = MSNParams.from_json('configs/neuron_default.json')
    params.to_json('configs/my_params.json')

Hardware calibration (Dec 2025)
────────────────────────────────
Default parameters are calibrated against 35 P0118MA thyristors measured
at Rm=680 kΩ, Ra=2.2 kΩ, Cm=0.1 µF (All_Sample.json).

Key derivations:
  Vth  = 2.0 V         ← user-selected representative spike amplitude
                          (dataset mean 1.73 V, range 1.1–2.8 V)
  Rm_hi = 60 kΩ        ← effective off-state membrane resistance.
                          NOTE: this is NOT the 680 kΩ gate-to-anode resistor.
                          The thyristor's own anode-cathode off-state impedance
                          (~60 kΩ) dominates and is what sets τ_open.
  Rm_lo  = 10 Ω        ← thyristor ≈ short circuit when conducting
  I_hold = 80 µA       ← median I_sat across 35 devices
  Ra     = 2200 Ω      ← hardware load resistor (was 47 Ω in earlier code)
  Cm     = 100 nF      ← membrane capacitor, confirmed by spike width

Self-consistency check:
  I_min  = Vth/(Rm_hi+Ra)         = 32 µA    ← data median 33 µA ✓
  τ_open = Cm*(Rm_hi+Ra)          = 6.2 ms
  τ_close= Cm*(Rm_lo+Ra)          = 221 µs
  t_spike= τ_close*ln(Vth/V_rest) ≈ 0.54 ms  ← measured ~0.5 ms ✓
  f_max  at I_sat=80 µA           ≈ 253 Hz    ← data median 245 Hz ✓

See METHODOLOGY.md §3–5 for physics, equations, and tuning guide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

from brian2 import NeuronGroup, farad, ohm, volt, amp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class MSNParams:
    """Hardware parameters for one MSN population.

    All values in SI units.  Defaults calibrated to the Dec 2025
    35-device P0118MA dataset (Ra=2.2 kΩ, Rm=680 kΩ, Cm=100 nF).

    Hardware (fixed by the physical device)
    ────────────────────────────────────────
    Cm       Membrane capacitor                              [F]
    Ra       Load resistor                                   [Ω]
    Rm_hi    Effective open-state membrane resistance        [Ω]
             (thyristor off-state impedance, NOT gate Rm=680 kΩ)
    Rm_lo    Effective closed-state membrane resistance      [Ω]
             (thyristor ≈ short circuit → ~10 Ω)
    Vth      Spike threshold on Vm                          [V]
             Set to 2.0 V (≈ Vout peak amplitude, since Rm_lo≈0)
    I_hold   Reopen current — I_M below this reopens thyristor [A]
             Calibrated to median I_sat from 35 devices.

    Synaptic filter time constants (tau_s1, tau_s2) belong to each
    Synapses object (SynapseParams), not to the neuron.  The neuron
    exposes summed inlets I_exc and I_inh; each Synapses object writes
    its filtered Is2 into the appropriate inlet.

    Device variability
    ──────────────────
    Vth and I_hold are promoted to per-neuron state variables inside the
    NeuronGroup so that msn_variability.apply_variability(G) can set
    per-neuron values drawn from the measured device distribution.
    All neurons are initialised to the scalar defaults.
    """

    Cm:     float = 100e-9    # F     (100 nF = 0.1 µF)
    Ra:     float = 2200.0    # Ω     (2.2 kΩ load resistor)
    Rm_hi:  float = 60_000.0  # Ω     (60 kΩ effective off-state)
    Rm_lo:  float = 10.0      # Ω     (thyristor ~short when conducting)
    Vth:    float = 2.0       # V     (threshold; ≈ Vout peak amplitude)
    I_hold: float = 80e-6     # A     (80 µA, median I_sat 35-device set)

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

        I_min  rheobase          = Vth / (Rm_hi + Ra)
        I_max  depol-block onset = I_hold
        """
        return self.Vth / (self.Rm_hi + self.Ra), self.I_hold

    def time_constants(self) -> tuple[float, float]:
        """(τ_open, τ_close) in seconds.

        τ_open   = Cm * (Rm_hi + Ra)   charging time constant
        τ_close  = Cm * (Rm_lo + Ra)   spike-width time constant
        """
        return (self.Cm * (self.Rm_hi + self.Ra),
                self.Cm * (self.Rm_lo + self.Ra))

    def summary(self) -> str:
        I_min, I_max = self.operating_window()
        tau_o, tau_c = self.time_constants()
        import math
        # spike width: τ_close * ln(Vth / (I_hold*(Rm_lo+Ra)))
        V_rest = self.I_hold * (self.Rm_lo + self.Ra)
        t_spike = tau_c * math.log(self.Vth / V_rest) if V_rest < self.Vth else float('nan')
        return (
            f"MSNParams (hardware-calibrated Dec 2025):\n"
            f"  Cm={self.Cm*1e9:.0f} nF   Ra={self.Ra:.0f} Ω\n"
            f"  Rm_hi={self.Rm_hi/1e3:.0f} kΩ (eff. off-state)   "
            f"Rm_lo={self.Rm_lo:.0f} Ω (thyristor on)\n"
            f"  Vth={self.Vth:.3f} V   I_hold={self.I_hold*1e6:.0f} µA\n"
            f"  → I_min={I_min*1e6:.1f} µA   I_hold={I_max*1e6:.0f} µA\n"
            f"  → τ_open={tau_o*1e3:.2f} ms   τ_close={tau_c*1e6:.0f} µs   "
            f"t_spike≈{t_spike*1e3:.2f} ms"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Equations                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _build_msn_eqs(
    exc_inlets: tuple[str, ...],
    inh_inlets: tuple[str, ...],
) -> str:
    """Build the Brian2 equation string for a given set of named inlets.

    Each inlet becomes a plain `amp` parameter; the corresponding Synapses
    object writes to it via `<inlet>_post = Is2 : amp (summed)`.  Brian2
    forbids two Synapses objects from writing to the same (inlet, group)
    pair, so assign one unique inlet name per (receptor_type, pathway).

    Examples
    --------
    Default (single exc + single inh):
        exc_inlets=('I_exc',), inh_inlets=('I_inh',)

    AMPA + NMDA excitation, GABA-A + GABA-B inhibition:
        exc_inlets=('I_ampa', 'I_nmda'),
        inh_inlets=('I_gaba_a', 'I_gaba_b')
    """
    exc_term  = ' + '.join(exc_inlets) if exc_inlets else '0*amp'
    inh_term  = ('(' + ' + '.join(inh_inlets) + ')') if inh_inlets else '0*amp'
    inlet_vars = '\n'.join(f'{v} : amp' for v in (*exc_inlets, *inh_inlets))
    return f"""
dVm/dt = (I_0 + {exc_term} - {inh_term} - Vm/(Rm_S + Ra)) / Cm  : volt
Rm_S   = (1 - s)*Rm_hi + s*Rm_lo                                 : ohm
I_M    = Vm / (Rm_S + Ra)                                        : amp
Vout   = Vm * Ra / (Rm_S + Ra)                                   : volt
{inlet_vars}
I_0    : amp
s      : 1
Vth    : volt
I_hold : amp
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Factory                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_msn(
    N: int,
    params: MSNParams | None = None,
    exc_inlets: tuple[str, ...] = ('I_exc',),
    inh_inlets: tuple[str, ...] = ('I_inh',),
    name: str = 'msn',
) -> NeuronGroup:
    """Build a NeuronGroup of N MSN neurons.

    Parameters
    ----------
    N           : number of neurons
    params      : MSNParams  (calibrated Dec 2025 defaults if None)
    exc_inlets  : names of excitatory current inlets — one per receptor type.
                  Default ('I_exc',).  Each inlet is a plain Brian2 parameter
                  [A] that a Synapses object writes to via (summed).
                  Brian2 allows only ONE Synapses object per (inlet, group)
                  pair, so use one unique name per receptor pathway, e.g.:
                      exc_inlets=('I_ampa', 'I_nmda')
    inh_inlets  : names of inhibitory current inlets.  Default ('I_inh',).
                  e.g.: inh_inlets=('I_gaba_a', 'I_gaba_b')
    name        : Brian2 group name — must be unique per start_scope()

    Returns
    -------
    NeuronGroup with state variables:
        Vm, Vout, I_M, Rm_S   — circuit quantities
        s                     — memristor state (0=open, 1=closed)
        I_0                   — per-neuron tonic bias [A]
        <exc_inlets>          — excitatory synaptic inlets [A]
        <inh_inlets>          — inhibitory synaptic inlets [A]
        Vth                   — per-neuron spike threshold [V]
        I_hold                — per-neuron reopen current [A]

    Usage with multiple receptor types
    -----------------------------------
        neurons = make_msn(N=100,
                           exc_inlets=('I_ampa', 'I_nmda'),
                           inh_inlets=('I_gaba_a', 'I_gaba_b'))

        ampa = SynapseParams(kind='exc', weight=5e-6,
                             tau_s1=2e-3, tau_s2=5e-3, target_var='I_ampa')
        nmda = SynapseParams(kind='exc', weight=3e-6,
                             tau_s1=50e-3, tau_s2=100e-3, target_var='I_nmda')
        make_synapse(pre, neurons, params=ampa, connect=..., name='ampa')
        make_synapse(pre, neurons, params=nmda, connect=..., name='nmda')
    """
    if params is None:
        params = MSNParams()

    eqs = _build_msn_eqs(exc_inlets, inh_inlets)

    namespace = dict(
        Cm    = params.Cm    * farad,
        Ra    = params.Ra    * ohm,
        Rm_hi = params.Rm_hi * ohm,
        Rm_lo = params.Rm_lo * ohm,
    )

    G = NeuronGroup(
        N, eqs,
        threshold = 'Vm > Vth and s < 0.5',
        reset     = 's = 1',
        events    = {'reopen': 'I_M < I_hold and s > 0.5'},
        method    = 'euler',
        namespace = namespace,
        name      = name,
    )
    G.run_on_event('reopen', 's = 0')

    G.Vm  = 0 * volt
    G.s   = 0
    G.I_0 = 0 * amp
    for v in (*exc_inlets, *inh_inlets):
        setattr(G, v, 0 * amp)

    G.Vth    = params.Vth    * volt
    G.I_hold = params.I_hold * amp

    return G
