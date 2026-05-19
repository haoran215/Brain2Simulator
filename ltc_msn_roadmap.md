# LTC–MSN Roadmap
## Liquid Time-Constant Networks on Memristive Spiking Hardware

**Status:** Planning document — main branch contains MSN model only.  
**Branch:** `feature/ltc-coupling` (not yet created)  
**Last updated:** 2026-05

---

## 1. The LTC Model

Hasani et al., AAAI 2021 introduced Liquid Time-Constant Networks (LTCs), a
class of continuous-time recurrent neural networks whose hidden-state dynamics
are:

$$\frac{dx}{dt} = -\left[\frac{1}{\tau} + f(x, I, t, \theta)\right] x(t)
+ f(x, I, t, \theta)\, A$$

where $x(t) \in \mathbb{R}^N$ is the hidden state, $I(t)$ is the input,
$f(\cdot)$ is a learnable neural network (typically a tanh MLP), $A$ is a
bias vector, and $\tau$ is a base time constant.

Rearranging: $\dot x = -x/\tau + f(x,I)(A - x)$.  The term $f(x,I)(A-x)$
is a **conductance-based** synaptic drive with time-varying effective
time constant:

$$\tau_{\text{sys}} = \frac{\tau}{1 + \tau\, f(x, I, \theta)}$$

Key properties proven in the paper:
- **Bounded dynamics** (Theorem 1–2): $\tau_{\text{sys}}$ and $x$ remain in
  finite ranges even for unbounded input.
- **Universal approximation** (Theorem 3): LTCs can approximate any autonomous
  ODE system to arbitrary precision.
- **Superior expressivity** (Theorems 4–5): trajectory length grows faster
  than Neural ODEs and CT-RNNs.
- **Training:** BPTT (backpropagation through time) with Adam.  The paper
  shows this is more accurate than the adjoint method for stiff ODEs.

---

## 2. The MSN Model

The Memristive Spiking Neuron (Wu et al., PRApp 2025) implements an
analogue leaky-integrate-and-fire circuit:

$$C_m \dot V_m = I_0 + I_{\text{syn}}(t) - \frac{V_m}{R_m[s] + R_a}$$

$$\tau_s \dot I_{s1} = -I_{s1} + I_\delta H(V_{\text{pre}} - \Omega)$$

$$\tau_s \dot I_{s2} = -I_{s2} + I_{s1}$$

The memristor state $s \in \{0, 1\}$ switches based on hysteresis:

- **Close** ($s: 0 \to 1$): when $V_m \geq V_{\text{th}}$ — thyristor fires.
- **Reopen** ($s: 1 \to 0$): when $I_M = V_m/(R_m[s]+R_a) < I_{\text{hold}}$.

In the current code, $I_{\text{syn}} = I_{s2,\text{exc}} - I_{s2,\text{inh}}$
is injected as a **current source** (independent of $V_m$).  This is the
standard current-based LIF coupling.

The two-stage $I_{s1} \to I_{s2}$ cascade implements Rall's alpha-function
synapse, which provides biomimetic excitatory (AMPA/NMDA) and inhibitory
(GABAa/GABAb) synaptic waveforms.

---

## 3. The Comparison

| Property | LTC | MSN (current) |
|---|---|---|
| State variable | Continuous $x(t) \in \mathbb{R}$ | $V_m(t)$, switches at threshold |
| Spiking | No — rate-coded, continuous output | Yes — action potential via thyristor |
| Synaptic coupling | Conductance-based: $f(x,I)(A-x)$ | Current-based: $+I_{s2}$ (independent of $V_m$) |
| Time constant | Input-dependent $\tau_{\text{sys}}(t)$ | Fixed $\tau_{\text{open}} = C_m(R_{m,\text{hi}}+R_a)$ |
| Learning | BPTT — global, offline | STDP — local, online |
| Biological inspiration | Continuous neuron dynamics | Spiking, thyristor-as-channel |
| Hardware substrate | Digital chips (Loihi-2 in NCPs work) | Analogue memristive circuit |
| Expressivity bound | Theorem 5 (trajectory length) | Not yet characterised |
| Vanishing gradients | Yes (acknowledged in paper) | Avoided by design (no BPTT) |

The MSN operates in a **rate-coding regime** when $C_m$ is small and $I_0$
is above rheobase — in this regime the spike rate is approximately:

$$f \approx \frac{1}{\tau_{\text{open}} \ln\!\left(\frac{V_{m,\text{ss}}}{V_{m,\text{ss}} - V_{\text{th}}}\right) + t_{\text{spike}}}$$

which is a smooth function of $I_0$.  In this limit the MSN behaves as a
continuous-rate neuron and is the natural substrate for LTC dynamics.

---

## 4. The Link — Equation-Level Mapping

The single structural difference between a current-based MSN and an LTC is
the **driving force** $(A - V_m)$ multiplying the synaptic conductance.

**Current MSN equation:**

$$C_m \dot V_m = I_0 + I_{s2,\text{exc}} - I_{s2,\text{inh}} - \frac{V_m}{R_m[s]+R_a}$$

**After adding conductance-based coupling:**

$$C_m \dot V_m = I_0 + \frac{I_{s2,\text{exc}}(V_{A,\text{exc}} - V_m)}{V_{\text{scale}}}
- \frac{I_{s2,\text{inh}}(V_m - V_{A,\text{inh}})}{V_{\text{scale}}}
- \frac{V_m}{R_m[s]+R_a}$$

Regrouping into LTC form:

$$\dot V_m = -\frac{V_m}{\tau_{\text{sys}}} + \frac{1}{C_m}\!\left[I_0
+ \frac{I_{s2,\text{exc}}\,V_{A,\text{exc}} + I_{s2,\text{inh}}\,V_{A,\text{inh}}}{V_{\text{scale}}}\right]$$

where:

$$\frac{1}{\tau_{\text{sys}}} = \frac{1}{C_m(R_m[s]+R_a)}
+ \frac{I_{s2,\text{exc}} + I_{s2,\text{inh}}}{V_{\text{scale}}\, C_m}$$

This is **exactly** Hasani's Eq. 1 with the identifications:

| LTC symbol | MSN equivalent |
|---|---|
| $x(t)$ | $V_m(t)$ |
| $\tau$ | $C_m(R_m[s]+R_a)$ |
| $f(x,I)\cdot C_m$ | $(I_{s2,\text{exc}} + I_{s2,\text{inh}})/V_{\text{scale}}$ |
| $A_{\text{exc}}$ | $V_{A,\text{exc}}$ (excitatory reversal potential) |
| $A_{\text{inh}}$ | $-V_{A,\text{inh}}$ (inhibitory reversal potential) |
| $\tau_{\text{sys}}$ | $C_m(R_m[s]+R_a) / (1 + (I_{s2}(R_m[s]+R_a))/V_{\text{scale}})$ |

The "liquid" property holds: $\tau_{\text{sys}}$ shrinks when $I_{s2}$ is
large (high synaptic input → faster response) and expands toward the
bare RC constant when input is absent.

The physical implementation of the multiplier $(V_{A} - V_m)$ is a
**Gilbert-cell** or single AD633 analogue multiplier placed between the
synapse output current and the neuron's membrane node.  One multiplier
per synapse terminal.

---

## 5. Permeability — What Transfers, What Doesn't

### ✓ Transfers cleanly

| Feature | How |
|---|---|
| Liquid $\tau_{\text{sys}}$ | Conductance coupling, one multiplier per synapse |
| Bounded $V_m$ | Thyristor naturally clamps at $V_{\text{th}}$; conductance drive saturates as $V_m \to V_A$ |
| Per-edge trainable weight $w$ | Already in `msn_synapse.py` as `w : amp`; STDP ready |
| Biomimetic alpha-function synapse | Already implemented (Rall's function, two-stage cascade) |
| Rate coding | Already works with small $C_m$ in spiking regime |
| Hardware analogue implementation | Multiplier stage maps directly to Gilbert cell |

### ✗ Does not transfer directly

| Feature | Reason | Workaround |
|---|---|---|
| End-to-end differentiability | Spiking threshold is non-differentiable | Surrogate gradient (SG) or rate approximation for BPTT |
| Rich $f(\cdot)$ (MLP nonlinearity) | MSN $f$ is the fixed alpha-function shape | Use network depth (multiple layers) for expressivity |
| Continuous hidden state | MSN has discrete spike events | Rate-code approximation works in high-frequency regime |
| BPTT training | Spike reset breaks gradient flow | STDP (local) + eligibility traces (three-factor) |

---

## 6. Implementation Plan

### Branch: `feature/ltc-coupling`

Branch from current `main` after the hardware-calibration update.  Main
branch stays clean with the standard current-based MSN.

#### Step 1 — Conductance coupling in neuron equation (1 line change)

In `msn_neuron.py`, add an LTC equation variant:

```python
MSN_EQS_LTC = """
dVm/dt      = (I_0
               + Is2_exc * (V_A_exc - Vm) / V_scale
               - Is2_inh * (Vm - V_A_inh) / V_scale
               - Vm/(Rm_S + Ra)) / Cm              : volt
Rm_S        = (1 - s)*Rm_hi + s*Rm_lo              : ohm
...same remaining lines as MSN_EQS...
"""
```

And a `coupling` argument to `make_msn`:

```python
def make_msn(N, params=None, name='msn', coupling='current'):
    eqs = MSN_EQS_LTC if coupling == 'conductance' else MSN_EQS
    namespace = {...}   # add V_A_exc, V_A_inh, V_scale when conductance
    ...
```

Suggested defaults for `coupling='conductance'`:

```python
V_A_exc  = 5.0  # V   (reversal potential, > Vth always)
V_A_inh  = 0.0  # V   (shunting inhibition toward rest)
V_scale  = 5.0  # V   (multiplier reference; at Vm=0: gain=1)
```

With these defaults, at $V_m = 0$ (rest) the coupling is identical to the
current-based mode.  The "liquid" variation is ~30% across $V_m \in [0, V_{\text{th}}]$.

#### Step 2 — Synapse unchanged

`msn_synapse.py` requires no changes.  The $(V_A - V_m)$ driving force is
on the neuron side.  Per-edge `w` already supports STDP.

#### Step 3 — Verify $\tau_{\text{sys}}$ modulation

Quick verification script (add to `tests/`):

```python
# Run single LTC-MSN neuron with step input, measure effective τ_sys
# by fitting exponential recovery to Vm after a pulse.
# Check τ_sys_high_input < τ_sys_low_input.
```

#### Step 4 — Benchmark on two tasks

Both tasks should run on the identical circuit with `coupling='current'`
and `coupling='conductance'` and report accuracy + spike count.

1. **Temporal XOR** — two input streams, output = XOR of temporal pattern.
   Requires integration over time; LTC coupling expected to help.

2. **Mackey-Glass time series** — standard Hasani benchmark.
   Compare MSN-RC (random weights, linear readout) vs LTC-MSN-RC.

---

## 7. Potential Task Demonstrations

Listed by complexity:

| Task | Why interesting | Expected LTC benefit |
|---|---|---|
| Direction classification (existing) | Already implemented, spatial | Baseline; LTC may not help |
| Temporal XOR | Pure temporal, not spatial | Large — requires $\tau_{\text{sys}}$ adaptation |
| Sequential MNIST | Hasani benchmark, standard | Medium — long sequence integration |
| Mackey-Glass prediction | Standard CT-RNN benchmark | Large — chaotic time series |
| Navigation / heading integration | Hasani NCP demo | Large — continuous control |
| Human activity recognition | Dataset used in Hasani 2021 | Large — multivariate irregular series |

For a publication-quality comparison, run at least Temporal XOR +
Mackey-Glass + one control (direction classification).

---

## 8. LTC-MSN Online Learning vs Current STDP

### Current state (main branch)

The current model uses **STDP** (spike-timing-dependent plasticity) to
update synaptic weights:

- Rule: $\Delta w_{ij} \propto \text{STDP}(t_{\text{pre}}, t_{\text{post}})$
- Learning is **local** (depends only on pre- and post-neuron spike times)
- Weight $w$ changes, but $\tau_{\text{sys}}$ is constant
- Demonstrated in `ns_msn_rc_demo.py`
- Biological plausibility: high
- Computational expressivity: limited by static $\tau_{\text{sys}}$

### LTC-MSN model (feature branch)

The LTC coupling adds a **second adaptation mechanism** on top of STDP:

- **Fast (automatic):** $\tau_{\text{sys}}(t)$ modulates with instantaneous
  $I_{s2}$ — no learning rule needed, purely physical
- **Slow (learned):** STDP updates $w$ (synaptic weight), scaling the
  amplitude of $I_{s2}$ per synapse

These two mechanisms operate on different timescales:
- $\tau_{\text{sys}}$ modulation: sub-millisecond to millisecond (within a spike)
- STDP weight change: seconds to minutes (across trials)

This is analogous to the two-timescale plasticity observed in biology
(short-term synaptic dynamics + long-term potentiation/depression).

### Summary comparison

| Mechanism | Current (STDP only) | LTC-MSN (STDP + liquid $\tau$) |
|---|---|---|
| Adaptation timescale | Slow (trials) | Fast (spikes) + Slow (trials) |
| What changes | $w_{ij}$ | $w_{ij}$ + $\tau_{\text{sys}}(t)$ |
| Requires offline training | No | No |
| Differentiable | No | No (but SG possible) |
| Hardware implementation | Memristive $R_W$ | Memristive $R_W$ + Gilbert cell |
| Expressivity | Limited | Higher (LTC trajectory length bound) |
| Power cost | $C_m V_{\text{th}}^2 f$ | Same + multiplier quiescent current |

---

## 9. Branch and File Structure

```
main branch (clean MSN only)
├── msn_neuron.py          — hardware model, calibrated params
├── msn_synapse.py         — synapse factory, STDP-ready w
├── msn_variability.py     — per-neuron device scatter
├── configs/
│   ├── neuron_default.json
│   └── synapse_*.json
└── ns_msn_*.py            — demo scripts

feature/ltc-coupling branch (from main)
├── msn_neuron.py          — adds MSN_EQS_LTC + coupling arg to make_msn
├── ltc_params.py          — LTCParams dataclass (V_A_exc, V_A_inh, V_scale)
├── tests/
│   └── test_tau_sys.py    — verify τ_sys modulation
└── ns_ltc_*.py            — LTC demo scripts and benchmarks
```

---

*Document maintained by the hardware-simulation team.  Update when branch
is created and when benchmarks are run.*
