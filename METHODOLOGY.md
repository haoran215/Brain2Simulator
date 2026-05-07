# Methodology: From abstract aLIF to a physically-faithful MSN simulation model

This document records the design choices, derivations, validation steps, and code organisation behind the Brian2 simulation of the Memristive Spiking Neuron (MSN) of Wu et al. 2023 [^1] in this repository. It is intended as the methodology section of a forthcoming write-up.

[^1]: J. Wu, K. Wang, O. Schneegans, P. Stoliar, M. Rozenberg, *Bursting dynamics in a spiking neuron with a memristive voltage-gated channel*, Neuromorph. Comput. Eng. **3**, 044008 (2023).

---

## 1. Motivation: what was wrong with the previous model

Previous repository state (files: [`ns_test.py`](ns_test.py), [`spike_Ra_sweep.py`](spike_Ra_sweep.py), [`modelcopare.py`](modelcopare.py)) implemented an "aLIF" abstraction in which the spike was an instantaneous Brian2 reset rule:

$$
C_m \frac{dV_m}{dt} = -\frac{V_m}{R_m^{\text{hi}} + R_a} + I_{\text{syn}} + I_0
$$

with `threshold='Vm > Vth'` triggering `reset='Vm = 0'` and a parameter `t_ref` providing the refractory period. `R_m^{\text{hi}}` was held *fixed* at the open-state value.

The consequences:

1. **No spike shape.** `V_m(t)` is a sawtooth — exponential rise to threshold, instantaneous drop to zero. The externally measurable signal `Vout = V_m \cdot R_a/(R_m+R_a)` is a single-sample spike, not a waveform.
2. **`t_ref` is unphysical.** The refractory period was treated as a free parameter and used (together with `C_m`) to fit user-specified `(I_{\min}, f_{\min})` and `(I_{\max}, f_{\max})` targets. There is no `t_ref` in the actual hardware.
3. **No depolarisation-block mechanism.** The model fired arbitrarily fast at high `I`. The real device latches at `I_{\text{in}} > I_{\text{hold}}`.
4. **Spike-shape sweeps were uninformative.** [`spike_Ra_sweep.py`](spike_Ra_sweep.py) varied `R_a` but produced no spike shape because the memristor never switched.

For network simulation of memristor hardware these limitations matter: synaptic events should be triggered by physically realisable spikes, the firing-rate ceiling should arise from a physical bound, and the spike width should set a physical timescale.

---

## 2. The physical reference (Wu et al. 2023)

Wu et al. introduce a **two-terminal memristor** `M` made of a thyristor `T` in parallel with a resistor `R` between its anode and gate. The macroscopic device exhibits a hysteretic two-state I–V characteristic (their Fig. 2, top-left inset):

- **Open state** `s = 0`: high resistance `R_m^{\text{hi}}` (paper: ≈ 100 kΩ, dominated by the parallel resistor).
- **Closed state** `s = 1`: low resistance `R_m^{\text{lo}}` (paper: tens-to-hundreds of Ω, thyristor in conduction).
- **Open → Closed** when the anode–cathode voltage exceeds a threshold `V_{\text{th}}` (≈ 0.9 V).
- **Closed → Open** when the current through `M` falls below a holding current `I_{\text{hold}}` (≈ 100 µA).

The single-compartment Memristive Spiking Neuron (MSN, paper §2, Fig. 1e) has:

- input current `I_{\text{in}}` injected at the top node `V_m`,
- membrane capacitor `C_m` to ground,
- memristor `M` in series with a load resistor `R_a` (paper "Rload") between `V_m` and ground.

The externally measured spike is `V_{\text{out}} = V_m \cdot R_a / (R_m + R_a)` (the voltage across `R_a`).

---

## 3. Model equations

The membrane equation is unchanged in form, but `R_m` becomes a state variable:

$$
C_m \frac{dV_m}{dt} = I_0 + I_{\text{syn}}^{\text{exc}} - I_{\text{syn}}^{\text{inh}} - \frac{V_m}{R_m(s) + R_a}
$$

with the **memristor state** `s ∈ {0, 1}`:

$$
R_m(s) = (1 - s) R_m^{\text{hi}} + s \, R_m^{\text{lo}}
$$

Switching is hysteretic and event-driven:

| Transition | Condition | Effect |
|---|---|---|
| Open → Closed | `V_m > V_{\text{th}}` and `s = 0` | `s ← 1`; emit spike to downstream synapses |
| Closed → Open | `I_M < I_{\text{hold}}` and `s = 1` | `s ← 0` |

where `I_M = V_m/(R_m + R_a)`.

`V_m` is **not reset** at the close event. The "spike" is the natural fast discharge of `C_m` through `R_m^{\text{lo}} + R_a` while `s = 1`, with width

$$
\tau_{\text{close}} = C_m (R_m^{\text{lo}} + R_a)
$$

This produces a real `V_{\text{out}}` waveform whose shape is set by the circuit, not by a reset rule.

### 3.1 Synaptic cascade (kept from the previous model)

Each pre-synaptic spike at neuron *j* contributes an instantaneous kick to a first-stage synaptic current `I_{s1}^{(j)}`, which feeds a second-stage current `I_{s2}^{(j)}` via a passive cascade:

$$
\tau_{s1} \frac{dI_{s1}}{dt} = -I_{s1} + I_w \sum_{t_k} \delta(t - t_k)
$$

$$
\tau_{s2} \frac{dI_{s2}}{dt} = -I_{s2} + I_{s1}
$$

For `\tau_{s1} = \tau_{s2} = \tau_s`, the response of `I_{s2}` to one pre-synaptic spike is the alpha function `(I_w/\tau_s) \cdot t \cdot e^{-t/\tau_s}`, peaking at `I_w/e` at `t = \tau_s`.

The neuron sees `I_{\text{syn}} = I_{s2}^{\text{exc}} - I_{s2}^{\text{inh}}` (in `NETWORK_MODE='Is2'`) or the corresponding `I_{s1}` difference.

### 3.2 Why this is not strictly LIF

Pure leaky integrate-and-fire is *defined* by the threshold-and-reset rule on `V_m`. In the MSN, `V_m` is continuous through the spike, the spike has a finite width, and "refractoriness" emerges from the discharge time constant rather than from an explicit `t_ref`. This places the MSN between LIF and Hodgkin–Huxley: a single voltage-gated channel (the memristor) replaces the abstraction of a magic reset rule. The macroscopic firing pattern is still LIF-like — type-1 excitability, integrate-then-fire — but the spike-generation mechanism is physical.

---

## 4. Implementation in Brian2

### 4.1 Why Brian2 (after considering scipy)

scipy's `solve_ivp` with event detection is the natural framework for hybrid continuous/discrete systems and gives sub-step accuracy on the switch instants. It was the initial recommendation for fitting a *single* spike shape. However:

- The end goal is **large-scale network simulation** with synapses and possibly plasticity. Brian2 is purpose-built for this.
- Brian2 supports **custom events** through `events={'name': '...'}` + `run_on_event()`. This provides exactly the mechanism needed for the close→open transition without porting back from scipy later.
- A 1 µs to 10 µs timestep is sufficient to resolve the spike shape (`\tau_{\text{close}} \approx 5 ms`) and the switch instants.

The model lives entirely in Brian2.

### 4.2 NeuronGroup specification

```python
eqs = """
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

G = NeuronGroup(
    N, eqs,
    threshold='Vm > Vth and s < 0.5',     # open → closed (also emits spike)
    reset='s = 1',                         # do NOT reset Vm
    events={'reopen': 'I_M < I_hold and s > 0.5'},
    method='euler',
)
G.run_on_event('reopen', 's = 0')
```

The Brian2 `threshold` event serves a dual role: it triggers the memristor close transition *and* fires the spike that downstream synapses listen to. This is physically correct — the moment the memristor closes is exactly when `V_{\text{out}}` begins to rise, which is what a downstream synapse should see.

### 4.3 Synapse specification

Each synapse simply adds to the post-synaptic `I_{s1}`:

```python
syn = Synapses(source, target,
               model='w : amp',
               on_pre='Is1_exc_post += w')
syn.connect(condition='i == j')   # any Brian2 connection pattern
syn.w = weight * amp
```

With `w` declared per-synapse, individual edge weights are addressable, so the same machinery extends naturally to plasticity and heterogeneous networks.

---

## 5. Parameter choices and tuning principles

### 5.1 Hardware parameters (Wu et al. 2023, Fig. 2)

| Symbol | Value | Description |
|---|---:|---|
| `C_m` | 10 µF | membrane capacitor |
| `R_a` | 47 Ω | load resistor (paper "Rload") |
| `R_m^{\text{hi}}` | 100 kΩ | open-state resistance |
| `R_m^{\text{lo}}` | 500 Ω | closed-state resistance |
| `V_{\text{th}}` | 0.9 V | thyristor close threshold |
| `I_{\text{hold}}` | 100 µA | holding current |

### 5.2 Derived quantities (read-only — they fall out of §5.1)

| Quantity | Formula | Value |
|---|---|---:|
| Rheobase | `I_{\min} = V_{\text{th}} / (R_m^{\text{hi}} + R_a)` | 9.0 µA |
| Depol-block onset | `I_{\max} = I_{\text{hold}}` | 100 µA |
| Open-state τ | `\tau_{\text{open}} = C_m (R_m^{\text{hi}} + R_a)` | 1.0 s |
| Closed-state τ (spike width) | `\tau_{\text{close}} = C_m (R_m^{\text{lo}} + R_a)` | 5.47 ms |
| Empirical f at I = 92 µA | (from sim) | ~ 8 Hz |

### 5.3 Tonic bias `I_0` (per-neuron, set after construction)

| Regime | Behaviour |
|---|---|
| `I_0 \in (0, I_{\min})` | silent on its own; needs synaptic input to fire |
| `I_0 \in (I_{\min}, I_{\max})` | spontaneously firing |
| `I_0 > I_{\max}` | latched closed → depolarisation block |

### 5.4 Synaptic weights and time constants

For `\tau_{s1} = \tau_{s2} = \tau_s`, the response of `I_{s2}` to a single pre-synaptic spike is `(I_w/\tau_s) \cdot t \cdot e^{-t/\tau_s}`, peaking at `I_w/e \approx 0.37 I_w` at `t = \tau_s`. For continuous Poisson input at rate `\lambda` (with `\lambda \tau_s \gg 1`), the steady-state mean is `\langle I_{s2} \rangle \approx I_w \lambda \tau_s`.

`\tau_s` should be **comparable to the target ISI**:
- `\tau_s \ll \text{ISI}`: spike-like blips with no integration (poor coupling to the slow MSN membrane).
- `\tau_s \sim \text{ISI}`: integration window matches the firing timescale.
- `\tau_s \gg \text{ISI}`: smooth low-pass — effectively a DC offset.

The MSN at paper-faithful `C_m = 10` µF has ISI ≈ 100–200 ms in its operating range. The library default is `\tau_s = 200` ms; specific scripts (bump test, ring) override to 500 ms to ensure synaptic events outlive the membrane charging time.

### 5.5 Trigger and sustain inequalities

These two ratios drive most of the parameter selection.

**Trigger** (one pulse drives the neuron from subthreshold `I_0` to a single output spike):

$$
I_0 + I_{s2}^{\text{peak}} > I_{\min}
\;\Rightarrow\;
I_w > e \cdot (I_{\min} - I_0)
$$

**Sustain** (self-excitatory recurrence locks on vs fades):

$$
I_w^{\text{recur}} \cdot f \cdot \tau_s
\quad
\genfrac{}{}{0pt}{0}{>}{<}
\quad
I_{\min} - I_0
$$

`>`: persistent (latched) firing. `<`: transient — bump fades. The "bump" regime in §7 sits just below this boundary.

---

## 6. Validation

The model is validated through four self-contained scripts, each addressing a specific claim.

### 6.1 Single trace ([`ns_msn_v1.py`](ns_msn_v1.py) → [`ns_msn_v1.png`](ns_msn_v1.png))

Drives one MSN with constant `I_{\text{in}} = 92.4` µA (paper's Fig. 2 caption). Verifies:

- `V_m` is continuous through the spike (no instantaneous drop).
- `V_{\text{out}}` shows ~80 mV peak, ~5 ms wide pulses (paper Fig. 2 right inset gives ~150 mV / ~3 ms — see §8 on remaining mismatches).
- Memristor state `s(t)` toggles cleanly on the predicted events.
- The `(V_M, I_M)` orbit shows the same triangular topology as the paper's Fig. 2 main panel.

Result: 2 spikes in 300 ms → 8 Hz, consistent with the analytical prediction.

### 6.2 I–F characterisation ([`ns_msn_if_sweep.py`](ns_msn_if_sweep.py) → [`ns_msn_if_sweep.png`](ns_msn_if_sweep.png))

Sweeps 37 drive currents from 5 to 110 µA in a single Brian2 NeuronGroup of N = 37 (vectorised parallel simulation). Compares numerical firing rates against the analytical formula:

$$
T(I_{\text{in}})
= -\tau_{\text{open}} \ln\!\left(1 - \frac{V_{\text{th}}}{I_{\text{in}} \cdot (R_m^{\text{hi}} + R_a)}\right)
- \tau_{\text{close}} \ln\!\left(\frac{(I_{\text{hold}} - I_{\text{in}})(R_m^{\text{lo}}+R_a)}{V_{\text{th}}-I_{\text{in}}(R_m^{\text{lo}}+R_a)}\right)
$$

with `f = 1/T`, valid for `I_{\min} < I_{\text{in}} < I_{\max}`.

Results confirmed:
- **Type-1 excitability** at the left onset (`f \to 0` as `I \to I_{\min}`).
- **Depolarisation-block cliff** at the right onset (`f` drops abruptly to 0 at `I = I_{\text{hold}}`), which is the *physical* mechanism — not a parameter ceiling.
- Numerical points sit slightly above the analytical curve due to single-step overshoot at the close threshold; tightening `dt` reduces this.

### 6.3 Three-model comparison ([`ns_msn_compare.py`](ns_msn_compare.py) → [`ns_msn_compare.png`](ns_msn_compare.png))

Side-by-side simulation of three models at their own native parameter scales:

| Set | Topology | Spike mechanism | Refractory | Spike shape |
|---|---|---|---|---|
| **A — aLIF** | series `R_m + R_a` | threshold + reset (Vm → 0) | `t_{\text{ref}}` parameter | none |
| **B — Thyristor** | parallel `g_a \parallel g_g` | threshold + reset to `V_r = I_H/g_a \neq 0` | `t_n` parameter | none |
| **D — MSN** | series `R_m + R_a` | hysteretic `R_m` state machine | emergent (`\tau_{\text{close}}`) | real |

Sets A and B are the previous abstractions from [`modelcopare.py`](modelcopare.py); set D is the new model. The figure shows that A and B have *no spike waveform* (instant jumps at the spike instant), while D has a continuous discharge waveform. Their I–F curves are also qualitatively different: A and B saturate at `1/t_{\text{ref}}` and `1/t_n` parameter ceilings; D drops to zero at `I_{\text{hold}}` (depol block).

### 6.4 Note on remaining quantitative mismatches with the paper

Two known discrepancies, both expected and tunable:

| Quantity | This model | Paper Fig. 2 | Cause |
|---|---|---|---|
| Spike width | ~5 ms | ~1–3 ms | `R_m^{\text{lo}}` fitted to 500 Ω; the real thyristor on-state has a near-constant forward voltage `V_{\text{on}} \approx 0.65` V plus small dynamic resistance. The two-state linear `R_m` abstraction gives an exponential decay rather than a constant-V plateau. |
| Vout peak | ~80 mV | ~150 mV | `V_{\text{out}}^{\text{peak}} = V_{\text{th}} R_a / (R_m^{\text{lo}} + R_a)`; tuning `R_m^{\text{lo}}` and `R_a` together adjusts both width and peak. |

A future refinement is a *thyristor-style* closed state in which `V_M = V_{\text{on}} + I_M R_m^{\text{lo}}`, replacing the linear `R_m^{\text{lo}}` with a clipped diode model. This is one extra subexpression in the equations and would close most of the gap.

---

## 7. Modular code organisation

After validation, the neuron and synapse construction were factored into a reusable module ([`msn_lib.py`](msn_lib.py)) with three public entries:

```python
from msn_lib import MSNParams, make_msn, make_synapse

params  = MSNParams(tau_s1=500e-3, tau_s2=500e-3)
neurons = make_msn(N=20, params=params)
neurons.I_0 = 0.85 * I_min * amp

syn = make_synapse(
    source=neurons, target=neurons,
    kind='exc', weight=2e-6,
    connect='abs(i-j) <= 2 and i != j',     # local recurrent
)
```

`MSNParams` is a dataclass containing the six hardware parameters plus the two synaptic time constants, with helper methods `operating_window()`, `time_constants()`, and `summary()`. Defaults reproduce Wu et al. Fig. 2.

The module's docstring contains a tuning guide (sections 1–6 above) and a list of common Brian2 connection patterns (`'i == j'`, `'i != j'`, `True`, `'rand() < 0.1'`, etc.), so all downstream scripts inherit the same parameter conventions and connection vocabulary.

---

## 8. Network-level demonstrations

Two scripts validate that the modular core scales without surprises.

### 8.1 Single neuron + self-loop bump ([`ns_msn_v3_bump.py`](ns_msn_v3_bump.py))

One MSN with `I_0 = 0.85 \cdot I_{\min} = 7.65` µA (subthreshold) and one self-excitatory edge of weight `I_w^{\text{recur}} = 6` µA. A single external pulse at `t = 2.5` s (delivered via a `SpikeGeneratorGroup` and a second `make_synapse` of weight 30 µA) triggers the first spike. Each output spike then feeds the self-loop, sustaining the bump until accumulated `I_{s2}` decays below `I_{\min} - I_0`.

Result: 3 spikes with ISIs growing 598 → 760 ms (i.e. the firing decelerates), and a 4th attempt that stalls before reaching `V_{\text{th}}`. The bump fades in roughly `2 \tau_s` after the cue. This is the "marginally subcritical" regime predicted by §5.5.

### 8.2 20-neuron ring ([`ns_msn_v4_network.py`](ns_msn_v4_network.py))

Twenty MSN neurons in a 1-D periodic ring with local recurrent excitation: each neuron `i` projects to `i \pm 1` and `i \pm 2` (mod 20), giving 4 incoming edges per neuron (80 total). Per-edge weight is scaled to `I_w^{\text{recur}} = 2` µA — roughly 1/4 of the self-loop case to keep the per-neuron summed input in the same regime. All 20 neurons receive subthreshold `I_0 = 0.85 \cdot I_{\min}`. A two-neuron cue at `i = 9, 10` is delivered at `t = 2.5` s.

Result:
- 8 post-cue spikes, all in neurons `\{8, 9, 10, 11\}` — i.e. confined to a one-neuron neighbourhood of the cue.
- The `I_{s2}^{\text{exc}}` heatmap shows a localised activity packet of width ~5 neurons that fades over ~2 s.
- Far-from-cue neurons (0–5, 14–19) never activate.

This is the canonical motif of a ring-attractor / working-memory network. With a Hebbian or STDP rule on the recurrent edges, the bump location becomes learnable — the natural next experiment.

---

## 9. File map and inheritance

```
Pre-existing (aLIF / Thyristor abstractions)
─────────────────────────────────────────────
  ns_test.py                  abstract aLIF + Poisson syn (no spike shape)
  spike_Ra_sweep.py           Ra sweep on the abstract aLIF
  modelcopare.py              aLIF vs Thyristor side-by-side

New (paper-faithful MSN)
─────────────────────────────────────────────
  ns_msn_v1.py                single trace, paper Fig. 2 reproduction
  ns_msn_if_sweep.py          analytical + numerical I–F characterisation
  ns_msn_compare.py           aLIF vs Thyristor vs MSN side-by-side
  ns_msn_v2_synapses.py       MSN core + Is1/Is2 synapses + Poisson input

Modular library
─────────────────────────────────────────────
  msn_lib.py                  MSNParams, make_msn, make_synapse + tuning guide

Network demonstrations (use msn_lib)
─────────────────────────────────────────────
  ns_msn_v3_bump.py           1 neuron + self-excit. bump test
  ns_msn_v4_network.py        20-neuron ring with local recurrent excitation
```

Each file's module docstring records its inheritance and what changed relative to its predecessor.

---

## 10. Limitations and planned extensions

1. **Slow firing rates.** At paper-faithful `C_m = 10` µF the maximum firing rate is ~ 8 Hz. To reach the 30–80 Hz range characteristic of biological neurons, `C_m` would have to drop ~ 5–10×, with a corresponding loss of strict paper faithfulness. This is a quantitative tuning choice, not a structural change.
2. **Linear closed-state `R_m`.** Replacing this with a thyristor-style `V_M = V_{\text{on}} + I_M R_m^{\text{lo}}` would close most of the spike-shape gap with the paper's Fig. 2 (see §6.4).
3. **No plasticity.** `make_synapse` exposes per-edge `w` already; adding STDP requires only trace variables in the `Synapses` model and a pre/post update rule. The 20-neuron ring is the natural testbed.
4. **No bursting.** Wu et al. §3 (the MSBN) adds a second compartment `R_s, C_s` in place of the ground reference at the bottom of `M`. The equations become
   $$
   C_m \dot V = I_{\text{in}} - (V - V_S)/R_m(s)
   \qquad
   C_s \dot V_S = (V - V_S)/R_m(s) - V_S/R_s
   $$
   with the threshold condition now on `V - V_S` rather than `V`. With `\tau_S = R_s C_s` short relative to `\tau_m`, this generates the four spiking modes (TS, FS, IB1, IB2) on the `(\tau_S, I_{\text{in}})` phase diagram. This is the next architectural extension when bursting is needed.

---

## Appendix A. Reproducibility

All scripts use `seed(42)` (where stochastic elements appear), `defaultclock.dt = 1` µs–`10` µs depending on the experiment, and `method='euler'` for the membrane integration. State recording uses `dt = 2` µs (single-spike work) to `2` ms (network heatmaps) to keep memory usage bounded.

Each figure in the repository is regenerated by running the corresponding `ns_msn_*.py` script directly.
