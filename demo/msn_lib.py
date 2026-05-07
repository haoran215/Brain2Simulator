"""
msn_lib.py
==========
Reusable Brian2 modules for the Memristive Spiking Neuron (MSN) and its
Is1/Is2 cascade synapses (Wu et al. 2023, Neuromorph. Comput. Eng. 3 044008).

Public API
----------
    MSNParams        — hardware parameter container (dataclass)
    make_msn(N)      — build a NeuronGroup of N MSN neurons
    make_synapse()   — wire a SpikeSource into Is1_{exc,inh} of an MSN group

Used by
-------
    ns_msn_v3_bump.py        — single neuron + 1 self-excitatory synapse
    (and any future scripts in the inheritance chain)


══════════════════ TUNING GUIDE  (read before changing anything) ════════════

1. Hardware parameters (set by the device — usually fixed)
─────────────────────────────────────────────────────────────────────────────
   Cm     : membrane capacitor                       default 10 µF
   Ra     : load resistor (paper "Rload")            default 47 Ω
   Rm_hi  : memristor open-state resistance          default 100 kΩ
   Rm_lo  : memristor closed-state resistance        default 500 Ω
   Vth    : thyristor close threshold                default 0.9 V
   I_hold : holding current (open-state threshold)   default 100 µA

2. Derived quantities (do NOT tune; they fall out of (1))
─────────────────────────────────────────────────────────────────────────────
   I_min   = Vth/(Rm_hi+Ra)        rheobase, ~9 µA       (left edge of I-F)
   I_max   = I_hold                depol block, 100 µA   (right edge of I-F)
   τ_open  = Cm·(Rm_hi+Ra)         charge τ, ~1 s
   τ_close = Cm·(Rm_lo+Ra)         spike width, ~5 ms
   f_max   ≈ 8 Hz at I_in=92 µA  (dominated by τ_open)

3. Tonic bias I_0 (per-neuron — set after construction with `G.I_0 = ...`)
─────────────────────────────────────────────────────────────────────────────
   I_0 ∈ (0, I_min)         silent on its own; needs synaptic input to fire
   I_0 ∈ (I_min, I_max)     spontaneously firing
   I_0 > I_max              latched closed → depolarisation block

4. Synaptic weight Iw and time constants τ_s1, τ_s2
─────────────────────────────────────────────────────────────────────────────
   Each pre-syn spike adds Iw to Is1 (instantaneous kick).
   Is1 → Is2 is a passive cascade.

   For τ_s1 = τ_s2 = τ_s, Is2(t) for ONE spike is the alpha function:
       Is2(t)       = (Iw/τ_s)·t·exp(-t/τ_s)
       Is2_peak     = Iw/e  ≈  0.37·Iw   at t = τ_s

   For continuous Poisson input at rate λ (with λ·τ_s ≫ 1):
       <Is2>       ≈  Iw·λ·τ_s

   Choose τ_s comparable to the target ISI:
   - τ_s ≪ ISI → spike-like blips, no integration  (bad for slow MSN)
   - τ_s ~ ISI  → integration window matches firing timescale
   - τ_s ≫ ISI → smooth low-pass filter (effectively a DC offset)

   Default τ_s1 = τ_s2 = 200 ms — chosen to match MSN's slow ISI (~125 ms
   at I≈90 µA).  If you change Cm or the operating I_in, retune τ_s.

5. Triggering a spike from subthreshold I_0 with ONE pre-syn pulse
─────────────────────────────────────────────────────────────────────────────
   Need:  I_0 + Is2_peak  >  I_min
          I_0 + Iw/e      >  I_min
          Iw              >  e·(I_min − I_0)   ≈  2.72·(I_min − I_0)

   Example: I_0 = 7 µA, I_min = 9 µA  →  Iw > 5.4 µA  to trigger.
            Use Iw = 30 µA for a strong, robust trigger.

6. Self-sustained vs. transient firing (the "bump" regime)
─────────────────────────────────────────────────────────────────────────────
   For a self-excitatory connection of strength Iw_recur, ISI 1/f, and
   τ_s, the steady-state cumulative Is2 (geometric sum across past spikes)
   approaches:
       Is2_steady  ≈  Iw_recur · f · τ_s

   - Iw_recur · f · τ_s  >  I_min − I_0   →  persistent firing (latched)
   - Iw_recur · f · τ_s  <  I_min − I_0   →  transient: bump forms then dies
   - Right at the boundary: marginally self-sustaining, sensitive to noise

   The bump regime is the slightly-subcritical case: enough recurrent gain
   to sustain a few spikes, not enough to lock on permanently.


══════════════════════ CONNECTION PATTERNS  ════════════════════════════════

   make_synapse(src, tgt, kind, weight, connect=...)

   `connect` is a Brian2 condition string (or True for all-to-all):

      'i == j'        1-to-1  (and self-loops if src is tgt)
      'i != j'        all-to-all except self
      True            all-to-all (including self)
      'rand() < 0.1'  random with 10% probability
      'abs(i-j)<=2'   local connectivity (band of width 2)

──────────────────────────────────────────────────────────────────────────"""

from dataclasses import dataclass
from brian2 import (
    NeuronGroup, Synapses,
    farad, ohm, volt, amp, second,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Parameters                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@dataclass
class MSNParams:
    """Hardware parameters of the Memristive Spiking Neuron.

    Defaults match Wu et al. 2023 Fig. 2.  All quantities are SI floats;
    `make_msn` wraps them in Brian2 units.
    """
    Cm:     float = 10e-7        # F
    Ra:     float = 47.0         # Ω   (paper "Rload")
    Rm_hi:  float = 100e3        # Ω   open-state memristor
    Rm_lo:  float = 500.0        # Ω   closed-state memristor
    Vth:    float = 1.5          # V   thyristor close threshold
    I_hold: float = 100e-6       # A   holding current (reopen threshold)
    tau_s1: float = 200e-3       # s   synapse stage 1 (Is1)
    tau_s2: float = 200e-3       # s   synapse stage 2 (Is2)

    def operating_window(self):
        """(I_min, I_max) in amps."""
        I_min = self.Vth / (self.Rm_hi + self.Ra)
        return I_min, self.I_hold

    def time_constants(self):
        """(τ_open, τ_close) in seconds."""
        return (self.Cm * (self.Rm_hi + self.Ra),
                self.Cm * (self.Rm_lo + self.Ra))

    def summary(self) -> str:
        I_min, I_max = self.operating_window()
        tau_o, tau_c = self.time_constants()
        return (
            f"MSN params:\n"
            f"  Cm={self.Cm*1e6:.1f} µF, Ra={self.Ra:.0f} Ω, "
            f"Rm_hi={self.Rm_hi/1e3:.0f} kΩ, Rm_lo={self.Rm_lo:.0f} Ω\n"
            f"  Vth={self.Vth:.2f} V, I_hold={self.I_hold*1e6:.0f} µA\n"
            f"  τ_s1={self.tau_s1*1e3:.0f} ms, τ_s2={self.tau_s2*1e3:.0f} ms\n"
            f"  → I_min={I_min*1e6:.2f} µA, I_max={I_max*1e6:.0f} µA\n"
            f"  → τ_open={tau_o*1e3:.0f} ms, τ_close={tau_c*1e3:.2f} ms"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Master equation set                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MSN_EQS = """
dVm/dt   = (I_0 + Is2_exc - Is2_inh - Vm/(Rm_S + Ra)) / Cm   : volt
Rm_S     = (1 - s)*Rm_hi + s*Rm_lo                            : ohm
I_M      = Vm / (Rm_S + Ra)                                   : amp
Vout     = Vm * Ra / (Rm_S + Ra)                              : volt
dIs1_exc/dt = -Is1_exc / tau_s1                                : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                    : amp
dIs1_inh/dt = -Is1_inh / tau_s1                                : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                    : amp
I_0      : amp
s        : 1
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Factories                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def make_msn(N, params=None, name='msn'):
    """Build a NeuronGroup of N MSN neurons.

    Parameters
    ----------
    N      : int          number of neurons
    params : MSNParams or None    (uses defaults if None)
    name   : str          Brian2 group name

    Returns
    -------
    NeuronGroup with state vars Vm, Vout, I_M, Rm_S, s, I_0,
    Is1_exc, Is2_exc, Is1_inh, Is2_inh.

    All initialised to 0 (s=0 means open).  Set per-neuron tonic bias
    via  `G.I_0 = <amp-quantity-or-array>`.
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
        threshold='Vm > Vth and s < 0.5',
        reset='s = 1',
        events={'reopen': 'I_M < I_hold and s > 0.5'},
        method='euler',
        namespace=namespace,
        name=name,
    )
    G.run_on_event('reopen', 's = 0')
    G.Vm = 0*volt
    G.s  = 0
    G.I_0 = 0*amp
    G.Is1_exc = 0*amp; G.Is2_exc = 0*amp
    G.Is1_inh = 0*amp; G.Is2_inh = 0*amp
    return G


def make_synapse(source, target, kind='exc', weight=6e-6,
                 connect='i==j', delay=None, name='syn'):
    """Connect a SpikeSource to an MSN group's Is1_{exc|inh}.

    Each pre-syn spike instantaneously adds `weight` (amps) to the post
    target's Is1_{kind}.  The MSN's Is1→Is2 cascade then shapes the
    transmitted current.

    Parameters
    ----------
    source  : SpikeSource (NeuronGroup, PoissonGroup, SpikeGeneratorGroup, …)
    target  : NeuronGroup built by make_msn
    kind    : 'exc' or 'inh'
    weight  : float, in amps (added to Is1 per pre-syn spike)
    connect : Brian2 condition string, or True for all-to-all
    delay   : float (s) or None
    name    : str

    Returns
    -------
    Synapses
    """
    if kind not in ('exc', 'inh'):
        raise ValueError(f"kind must be 'exc' or 'inh', got {kind!r}")
    target_var = f'Is1_{kind}_post'
    syn = Synapses(
        source, target,
        model='w : amp',
        on_pre=f'{target_var} += w',
        name=name,
    )
    if connect is True:
        syn.connect()
    else:
        syn.connect(condition=connect)
    syn.w = weight * amp
    if delay is not None:
        syn.delay = delay * second
    return syn
