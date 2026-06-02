"""
msn_variability.py
==================
Per-neuron device variability for MSN populations.

Draws Vth and I_hold from the measured distributions of 35 P0118MA
thyristors (All_Sample.json, Dec 2025 dataset, Rm=680 kΩ, Ra=2.2 kΩ,
Cm=0.1 µF).

Typical use
───────────
    from msn_neuron     import MSNParams, make_msn
    from msn_variability import apply_variability

    params = MSNParams()
    G = make_msn(N=20, params=params)
    G.I_0 = 40e-6 * amp

    apply_variability(G, seed=42)          # scatter Vth and I_hold
    # or
    apply_variability(G, seed=42, scale=0.5)   # half the hardware spread

After apply_variability():
    G.Vth    is an array of shape (N,)  [V]   — per-neuron thresholds
    G.I_hold is an array of shape (N,)  [A]   — per-neuron reopen currents

Design notes
────────────
- Vth and I_hold are already per-neuron state variables in MSN_EQS.
  This file just assigns non-uniform values to them.
- scale=0.0 restores the uniform (identical device) case.
- Distributions are clipped to the measured hardware range so no neuron
  gets parameters outside the observed envelope.
- Vth and I_hold are sampled independently.  In the real hardware they
  are weakly correlated (both scale with I_GT), but independence is a
  conservative, simpler assumption for population studies.
"""

from __future__ import annotations

import numpy as np
from brian2 import NeuronGroup, volt, amp


# ── Measured population statistics (35 P0118MA devices, Dec 2025) ─────────────
#
#   Vth    ← amp_mean_bins column (Vout peak ≈ Vth when Rm_lo≈0)
#   I_hold ← I_sat column.  I_sat is the measured depolarisation-block onset
#             current, which equals the thyristor holding current: when I_in
#             rises to I_sat the closed-state steady-state current I_M ≈ I_in
#             never drops back below I_hold, so the neuron stays latched.

DEVICE_STATS: dict[str, dict] = {
    'Vth': {
        'mean': 1.73,   # V
        'std':  0.56,   # V
        'lo':   1.10,   # V  (dataset minimum)
        'hi':   2.80,   # V  (dataset maximum)
    },
    'I_hold': {
        'mean': 77.4e-6,   # A
        'std':  16.8e-6,   # A
        'lo':   56.8e-6,   # A
        'hi':  109.5e-6,   # A
    },
}


def apply_variability(
    G: NeuronGroup,
    seed: int | None = None,
    scale: float = 1.0,
) -> None:
    """Assign per-neuron Vth and I_hold sampled from the device distribution.

    Parameters
    ----------
    G     : NeuronGroup built by make_msn
    seed  : random seed for reproducibility (None → unseeded)
    scale : multiplier on the standard deviation.
              1.0  → full hardware variability  (default)
              0.5  → half spread
              0.0  → identical devices (restores uniform defaults)

    Modifies G.Vth and G.I_hold in place.  Both are already per-neuron
    state variables allocated by make_msn(); this function just assigns
    non-uniform values.

    Raises
    ------
    AttributeError if G does not have Vth or I_hold state variables
    (i.e. G was not built by make_msn from msn_neuron.py).
    """
    if not hasattr(G, 'Vth') or not hasattr(G, 'I_hold'):
        raise AttributeError(
            "G must be a NeuronGroup built by make_msn() from msn_neuron.py. "
            "Vth and I_hold state variables not found."
        )

    rng = np.random.default_rng(seed)
    N   = len(G)

    if scale == 0.0:
        # Restore the uniform-device case without touching the existing values
        return

    for key, attr, unit in [('Vth', 'Vth', volt), ('I_hold', 'I_hold', amp)]:
        s = DEVICE_STATS[key]
        samples = rng.normal(loc=s['mean'], scale=s['std'] * scale, size=N)
        samples = np.clip(samples, s['lo'], s['hi'])
        setattr(G, attr, samples * unit)


def device_summary() -> str:
    """Return a human-readable summary of the device statistics."""
    s_v  = DEVICE_STATS['Vth']
    s_ih = DEVICE_STATS['I_hold']
    return (
        f"Device population statistics (35 × P0118MA, Dec 2025):\n"
        f"  Vth    : mean={s_v['mean']:.2f} V,  "
        f"std={s_v['std']:.2f} V,  "
        f"range=[{s_v['lo']:.2f}, {s_v['hi']:.2f}] V\n"
        f"  I_hold : mean={s_ih['mean']*1e6:.1f} µA, "
        f"std={s_ih['std']*1e6:.1f} µA, "
        f"range=[{s_ih['lo']*1e6:.1f}, {s_ih['hi']*1e6:.1f}] µA"
    )
