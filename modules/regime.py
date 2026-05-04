"""
regime.py  —  Coding Regime Detector
======================================
Classifies the operating regime based on:
    ISI = 1/f  vs  tau_s  (synaptic time constant)

  ISI > tau_s  (f < 1/tau_s)  →  TEMPORAL coding
  ISI = tau_s  (f = 1/tau_s)  →  BOUNDARY
  ISI < tau_s  (f > 1/tau_s)  →  RATE coding

Physical intuition
------------------
  Is2 (alpha function) is a low-pass filter with cutoff ~1/tau_s.
  - Rate regime:     pulses overlap → Is2 carries a smooth DC signal → rate code
  - Temporal regime: pulses resolve individually → spike timing matters → temporal code
  - Crossover freq:  f_cross = 1 / tau_s
"""

class RegimeDetector:
    """Classifies operating regime given tau_s (ms)."""

    TEMPORAL = 'temporal'
    RATE     = 'rate'
    BOUNDARY = 'boundary'
    MARGIN   = 0.20   # ±20% around crossover = boundary zone

    def __init__(self, tau_s_ms: float):
        self.tau_s_ms   = tau_s_ms
        self.f_cross_Hz = 1000.0 / tau_s_ms   # Hz

    # ------------------------------------------------------------------
    def classify(self, f_hz: float) -> str:
        """Return 'temporal', 'rate', or 'boundary' for a given rate."""
        ratio = f_hz / self.f_cross_Hz        # ISI/tau_s inverse: f*tau_s
        if   ratio < (1.0 - self.MARGIN):
            return self.TEMPORAL
        elif ratio > (1.0 + self.MARGIN):
            return self.RATE
        else:
            return self.BOUNDARY

    def isi_ms(self, f_hz: float) -> float:
        return 1000.0 / f_hz

    def overlap_ratio(self, f_hz: float, Iw_uA: float = 1.0) -> float:
        """DC / single-peak ratio.  >1 means heavy overlap (rate regime)."""
        import math
        DC    = Iw_uA * f_hz * (self.tau_s_ms / 1000.0)
        peak  = Iw_uA / math.e
        return DC / peak

    # ------------------------------------------------------------------
    def report(self, f_min_Hz: float, f_max_Hz: float,
               Iw_uA: float = 20.0) -> None:
        """Print a full regime summary for the operating range."""
        print("=" * 62)
        print(f"  Regime Analysis   tau_s = {self.tau_s_ms:.1f} ms"
              f"   f_cross = {self.f_cross_Hz:.0f} Hz")
        print("=" * 62)
        checkpoints = [
            (f_min_Hz,                  "f_min"),
            (self.f_cross_Hz,           "crossover (f = 1/tau_s)"),
            (f_max_Hz,                  "f_max"),
        ]
        for f, label in checkpoints:
            reg  = self.classify(f)
            isi  = self.isi_ms(f)
            ov   = self.overlap_ratio(f, Iw_uA)
            sym  = ">>>" if reg == self.RATE else ("<<<" if reg == self.TEMPORAL else "===")
            print(f"  {sym} {label:<28}  f={f:6.1f} Hz  "
                  f"ISI={isi:6.2f} ms  overlap={ov:.2f}x  [{reg}]")
        print()
        print(f"  Learning recommendation:")
        reg_lo = self.classify(f_min_Hz)
        reg_hi = self.classify(f_max_Hz)
        if reg_hi == self.RATE:
            print(f"    f_max={f_max_Hz}Hz is in RATE regime "
                  f"→  Reservoir Computing  (Is2 DC readout)")
        if reg_lo == self.TEMPORAL:
            print(f"    f_min={f_min_Hz}Hz is in TEMPORAL regime "
                  f"→  STDP  (spike-timing-dependent plasticity)")
        print("=" * 62)
