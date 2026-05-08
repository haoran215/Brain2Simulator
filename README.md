# Brain2Simulator — Single-neuron model comparison

This branch is a side-by-side comparison of single-neuron Brian2 models implementing the same memristor-style spiking neuron at different levels of abstraction. The two questions being asked of every model are:

1. **What is the mathematical form?** — state equations, parameter set, spike rule.
2. **How does it spike?** — `Vm(t)` waveform, single-spike shape, I-F curve, refractory mechanism.

> Branch siblings: `main` (split MSN library + WTA demo), `MSN` and `aLIF` (legacy snapshots). For the production neuron/synapse API check out `main`.

---

## 1. Models

| Model | Topology | Spike mechanism | Refractory | Real waveform? | Status |
|---|---|---|---|---|---|
| **aLIF** | series `Rm + Ra`, `Rm = Rm_hi` fixed | `threshold='Vm > Vth'` + reset `Vm → 0` | `t_ref` parameter | no — instantaneous reset | implemented |
| **Thyristor** | parallel `ga ∥ gg` | threshold + reset to `Vr = IH/ga ≠ 0` | `tn` parameter | no — instantaneous reset | implemented |
| **MSN** (Wu et al. 2023 [^1]) | series `Rm(s) + Ra`, hysteretic `Rm` | natural close event when `Vm > Vth`; reopen when `IM < I_hold` | emergent `τ_close = Cm·(Rm_lo + Ra)` | yes — `Cm` discharge through `Rm_lo + Ra` | implemented |
| **Hodgkin–Huxley** | conductance-based with `m, h, n` gates | continuous voltage-gated currents | emergent from gating dynamics | yes | **planned** |

[^1]: J. Wu, K. Wang, O. Schneegans, P. Stoliar, M. Rozenberg, *Bursting dynamics in a spiking neuron with a memristive voltage-gated channel*, Neuromorph. Comput. Eng. **3**, 044008 (2023).

### 1.1 Mathematical form

**aLIF** (paper Eqs. 9–12, abstract):

$$
C_m\frac{dV_m}{dt} = -\frac{V_m}{R_m^{hi} + R_a} + I_0 + I_{syn}
\quad ; \quad
V_m > V_{th} \Rightarrow V_m \leftarrow 0,\; \text{hold for } t_{ref}
$$

**Thyristor** (parallel-conductance fit):

$$
C\frac{dV_m}{dt} = -(g_a + g_g)\,V_m + I
\quad ; \quad
V_m > V_T \Rightarrow V_m \leftarrow V_r = I_H / g_a,\; \text{hold for } t_n
$$

**MSN** (memristor as state variable `s ∈ {0, 1}`):

$$
C_m\frac{dV_m}{dt} = I_0 + I_{syn} - \frac{V_m}{R_m(s) + R_a}
\quad,\quad
R_m(s) = (1-s)R_m^{hi} + sR_m^{lo}
$$

| Transition | Condition | Effect |
|---|---|---|
| Open → Closed | `Vm > Vth` and `s = 0` | `s ← 1`; emits Brian2 spike |
| Closed → Open | `IM < I_hold` and `s = 1` | `s ← 0` |

`Vm` is **not** reset; the output `Vout = Vm·Ra/(Rm + Ra)` discharges through `Rm_lo + Ra` while `s = 1`, producing a real spike waveform of width `τ_close ≈ Cm·(Rm_lo + Ra)`.

### 1.2 Spiking performance — what to look for

| Axis | aLIF | Thyristor | MSN |
|---|---|---|---|
| Spike shape | none (sawtooth) | none (sawtooth, non-zero reset) | continuous discharge waveform |
| Refractory origin | parameter `t_ref` | parameter `tn` | emergent `τ_close` |
| Firing-rate ceiling | `1/t_ref` (parameter cap) | `1/tn` (parameter cap) | depol-block cliff at `I = I_hold` (physical) |
| I-F left edge | hard rheobase | hard rheobase | type-1 (continuous from zero) |
| Synaptic input port | `Is1`/`Is2` cascade | `Is1`/`Is2` cascade | `Is1`/`Is2` cascade |

The synaptic cascade (`Is1` → `Is2`) is shared across all three to make the comparison clean: only the membrane / spike-generation block varies.

---

## 2. Files

```
alif_ns_test.py             aLIF network demo — 2 neurons + sinusoidal Poisson
alif_Spikesynapsechar.py    Single-spike + synapse impulse response (aLIF)
alif_regime_verify.py       Temporal vs rate regime of the Is1/Is2 cascade
alif_thy_msn_compare.py     Side-by-side aLIF / Thyristor / MSN
METHODOLOGY.md              MSN-side methodology document (reference)
```

| Script | What it produces |
|---|---|
| `alif_ns_test.py` | `alif_ns_test.png` — Poisson rasters, output rasters, `Vm`, `Is2_exc`, `Is2_inh` |
| `alif_Spikesynapsechar.py` | `alif_Spikesynapsechar.png` — single-spike RC charge + alpha-function fit |
| `alif_regime_verify.py` | `regime_verification.png` — temporal vs rate regime of `Iw·f·τ_s` |
| `alif_thy_msn_compare.py` | `alif_thy_msn_compare.png` — 3×3 grid: row 1 `Vm` traces, row 2 single-spike zoom, row 3 I-F curves |

### 2.1 Two parameterisation styles per model

The aLIF lives in two flavours on this branch, and the same split will be applied to Thyristor and MSN as their characterisation/network scripts come in:

- **Canonical** — fixed paper/hardware parameters. Used by [`alif_Spikesynapsechar.py`](alif_Spikesynapsechar.py) and SET A of [`alif_thy_msn_compare.py`](alif_thy_msn_compare.py): `Ra=10 kΩ`, `Cm=40.15 nF`, `t_ref=3 ms`. This is the reference point for the comparison figure.
- **Tunable** — user specifies operating targets `(I_min, f_min, I_max, f_max)` and the script solves for `Cm` and `t_ref` analytically. Used by [`alif_ns_test.py`](alif_ns_test.py). This is the intended template for the network/demo scripts of the other models, so each can be retuned to a chosen operating range without touching the canonical comparison.

Treat the network demo and the comparison figure as answering different questions: the network demo shows behaviour at a chosen operating range, the comparison figure pins models to their native scales.

---

## 3. Quick start

Python ≥ 3.12, `brian2`, `numpy`, `matplotlib`.

```bash
uv sync
uv run python alif_thy_msn_compare.py    # main comparison figure
uv run python alif_ns_test.py            # network demo
uv run python alif_Spikesynapsechar.py   # single-spike characterisation
uv run python alif_regime_verify.py      # cascade regime analysis
```

The comparison script (`alif_thy_msn_compare.py`) is the central artefact: one figure summarises all three models on the two axes above.

---

## 4. The comparison figure

[`alif_thy_msn_compare.py`](alif_thy_msn_compare.py) lays out a 3×3 grid:

| Row | Content | What it shows |
|---|---|---|
| 1 | `Vm(t)` over the full run | Qualitative shape: aLIF/Thyristor sawtooth vs MSN continuous discharge |
| 2 | Single-spike zoom (centred on first spike) | aLIF/Thyristor: instantaneous reset (no waveform). MSN: real `Vout` shape |
| 3 | Analytical I-F curves on each model's native current scale | Parameter-cap ceiling (aLIF, Thyristor) vs depol-block cliff (MSN) |

Parameters span ~4 orders of magnitude across the three models (paper-faithful values, not retuned), so each column uses its own scale. A summary table is printed to stdout when the script is run.

---

## 5. Synaptic cascade (shared across models)

Each pre-synaptic spike adds `I_w` to a first-stage current `Is1`, which drives a second-stage current `Is2` via a passive cascade:

$$
\tau_{s1}\frac{dI_{s1}}{dt} = -I_{s1} + I_w \sum_{t_k}\delta(t-t_k)
\qquad
\tau_{s2}\frac{dI_{s2}}{dt} = -I_{s2} + I_{s1}
$$

For `τ_s1 = τ_s2 = τ_s`, one pre-spike yields the alpha function `(I_w/τ_s)·t·e^{-t/τ_s}`, peaking at `I_w/e ≈ 0.37·I_w` at `t = τ_s`. Under continuous Poisson drive at rate `λ` the steady-state mean is `⟨I_{s2}⟩ ≈ I_w·λ·τ_s`.

[`alif_regime_verify.py`](alif_regime_verify.py) plots three regimes:

- `τ_s ≪ ISI` — temporal: distinct alpha-function bumps, no integration.
- `τ_s ~ ISI` — boundary: bumps starting to overlap.
- `τ_s ≫ ISI` — rate: `Is2` smooths to the DC level `I_w·f·τ_s`.

Although run in the aLIF script, this analysis applies to all three membrane models since the cascade is identical.

---

## 6. Adding a fourth model (Hodgkin–Huxley)

Planned addition. Drop-in scaffold:

```python
# hh_compare.py
def sim_HH():
    eqs = '''
    dV/dt  = (I - g_Na*m**3*h*(V-E_Na) - g_K*n**4*(V-E_K) - g_L*(V-E_L)) / Cm  : volt
    dm/dt  = alpha_m*(1-m) - beta_m*m   : 1
    dh/dt  = alpha_h*(1-h) - beta_h*h   : 1
    dn/dt  = alpha_n*(1-n) - beta_n*n   : 1
    '''
    G = NeuronGroup(1, eqs, threshold='V > 0*mV', refractory='V > 0*mV',
                    method='exponential_euler')
    ...
```

Then add a fourth column to the comparison figure: `Vm(t)`, single-spike zoom, I-F curve. The synaptic cascade above already plugs in via `I += Is2_exc - Is2_inh`.

---

## 7. License

MIT — see [LICENSE](LICENSE).
