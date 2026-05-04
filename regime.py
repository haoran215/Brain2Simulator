"""
regime.py  —  Coding Regime Detector
=====================================
Determines whether the network operates in:

  TEMPORAL regime  (1/f > tau_s  →  ISI > tau_s)
    Each synaptic alpha pulse decays significantly before the next spike.
    Individual spike timing carries information.
    Recommended learning rule: STDP.

  RATE regime  (1/f < tau_s  →  ISI < tau_s)
    Alpha pulses overlap and accumulate into a smooth DC current.
    The rate (not timing) of spikes carries information.
    Recommended learning rule: Reservoir computing + linear readout.

  BOUNDARY  (1/f ≈ tau_s  →  ISI ≈ tau_s)
    Transition zone. Both spike timing and rate are informative.

Crossover frequency = 1 / tau_s
  For default params: tau_s = Rs*Cs = 10 kΩ × 1 µF = 10 ms → crossover = 100 Hz
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RegimeResult:
    f_Hz        : float
    tau_s_ms    : float
    ISI_ms      : float
    ISI_tau_ratio : float
    regime      : Literal['temporal', 'boundary', 'rate']
    overlap     : float     # DC / single_peak  (>1 = heavy overlap)
    residual_pct: float     # % of Is2 remaining at next spike onset
    recommendation : str

    def __str__(self):
        lines = [
            "=" * 54,
            "  Regime Detection",
            "=" * 54,
            f"  f          = {self.f_Hz:.1f} Hz",
            f"  tau_s      = {self.tau_s_ms:.1f} ms",
            f"  ISI        = {self.ISI_ms:.2f} ms",
            f"  ISI/tau_s  = {self.ISI_tau_ratio:.2f}  "
            f"({'> 1' if self.ISI_tau_ratio > 1 else '< 1' if self.ISI_tau_ratio < 1 else '= 1'})",
            f"  Overlap    = DC/peak = {self.overlap:.2f}",
            f"  Residual @ next spike = {self.residual_pct:.1f}%",
            f"  ── Regime: {self.regime.upper()} ──────────────",
            f"  → {self.recommendation}",
            "=" * 54,
        ]
        return "\n".join(lines)


def detect_regime(f_Hz        : float,
                  tau_s_ms    : float,
                  Iw          : float = 1.0,
                  boundary_tol: float = 0.2) -> RegimeResult:
    """
    Determine the coding regime for a given firing rate and tau_s.

    Parameters
    ----------
    f_Hz         : expected or measured mean firing rate (Hz)
    tau_s_ms     : synaptic time constant (ms)  =  Rs * Cs
    Iw           : synaptic weight (arbitrary units, for overlap calculation)
    boundary_tol : ISI/tau_s within [1-tol, 1+tol] is classified as 'boundary'

    Returns
    -------
    RegimeResult dataclass
    """
    import numpy as np

    ISI_ms    = 1e3 / f_Hz                # ms
    tau_s_s   = tau_s_ms * 1e-3
    ISI_s     = ISI_ms   * 1e-3
    ratio     = ISI_ms / tau_s_ms         # ISI / tau_s

    # DC level from a regular spike train: by convolution theorem
    # Is2_DC = Iw * f * tau_s  (integral of alpha fn = Iw*tau_s, times f)
    DC        = Iw * f_Hz * tau_s_s
    peak      = Iw / np.e                 # peak of single alpha fn at t=tau_s
    overlap   = DC / peak                 # > 1 = heavy overlap

    # Is2 residual at next spike (how much remains from previous spike)
    residual  = (Iw / tau_s_s) * ISI_s * np.exp(-ISI_s / tau_s_s)
    residual_pct = (residual / peak) * 100

    if ratio > 1.0 + boundary_tol:
        regime = 'temporal'
        rec    = ("STDP learning recommended.\n"
                  "  Each spike is individually resolved.\n"
                  "  Use SpikeMonitor for timing-based weight updates.\n"
                  "  Brian2: build_stdp_synapse() in synapse.py")
    elif ratio < 1.0 - boundary_tol:
        regime = 'rate'
        rec    = ("Reservoir computing recommended.\n"
                  "  Pulses overlap — smooth rate signal emerges.\n"
                  "  Use StateMonitor(Is2) + linear readout.\n"
                  "  Brian2: ReservoirReadout in learning.py")
    else:
        regime = 'boundary'
        rec    = ("Boundary zone — both rate and timing are informative.\n"
                  "  Consider both STDP and reservoir approaches.\n"
                  "  Spike timing gives finer classification resolution.")

    return RegimeResult(
        f_Hz          = f_Hz,
        tau_s_ms      = tau_s_ms,
        ISI_ms        = ISI_ms,
        ISI_tau_ratio = ratio,
        regime        = regime,
        overlap       = overlap,
        residual_pct  = residual_pct,
        recommendation= rec,
    )


def check_operating_range(f_min_Hz    : float,
                           f_max_Hz    : float,
                           tau_s_ms    : float,
                           Iw          : float = 1.0) -> None:
    """
    Print regime analysis for both I-F operating points.
    Useful for sanity-checking config before building a network.
    """
    crossover = 1e3 / tau_s_ms
    print(f"\nCrossover frequency: 1/tau_s = {crossover:.0f} Hz")
    print(f"tau_s = {tau_s_ms:.0f} ms\n")

    for f, label in [(f_min_Hz, f'f_min ({f_min_Hz:.0f} Hz)'),
                     (crossover, 'crossover'),
                     (f_max_Hz, f'f_max ({f_max_Hz:.0f} Hz)')]:
        r = detect_regime(f, tau_s_ms, Iw)
        print(f"  {label:<22}  ISI={r.ISI_ms:.1f}ms  "
              f"ISI/tau_s={r.ISI_tau_ratio:.2f}  "
              f"overlap={r.overlap:.2f}  → {r.regime.upper()}")
    print()


if __name__ == '__main__':
    # Quick demo
    check_operating_range(70, 200, 10.0)
    print(detect_regime(70, 10.0))
    print(detect_regime(200, 10.0))