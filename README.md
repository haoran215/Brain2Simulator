# From abstract aLIF to a physically-faithful MSN simulation model

This document records the design choices, derivations, validation steps, and code organisation behind the Brian2 simulation of the Memristive Spiking Neuron (MSN) of Wu et al. 2023 [^1] in this repository. It is intended as the methodology section of a forthcoming write-up.

[^1]: J. Wu, K. Wang, O. Schneegans, P. Stoliar, M. Rozenberg, *Bursting dynamics in a spiking neuron with a memristive voltage-gated channel*, Neuromorph. Comput. Eng. **3**, 044008 (2023).

---

## 1. Motivation: what was wrong with the previous model

Previous repository state (files: [`ns_test.py`](ns_test.py), [`spike_Ra_sweep.py`](spike_Ra_sweep.py), [`modelcopare.py`](modelcopare.py)) implemented an "aLIF" abstraction in which the spike was an instantaneous Brian2 reset rule:

$$
C_m \frac{dV_m}{dt} = -\frac{V_m}{R_m^{\text{hi}} + R_a} + I_{\text{syn}} + I_0
$$

with `threshold='Vm > Vth'` triggering `reset='Vm = 0'` and a parameter `t_ref` providing the refractory period. $R_m^{\text{hi}}$ was held *fixed* at the open-state value.

The consequences:

1. **No spike shape.** `V_m(t)` is a sawtooth — exponential rise to threshold, instantaneous drop to zero. The externally measurable signal $V_{\text{out}} = V_m \cdot R_a/(R_m+R_a)$ is a single-sample spike, not a waveform.
2. **`t_ref` is unphysical.** The refractory period was treated as a free parameter and used (together with `C_m`) to fit user-specified $(I_{\min}, f_{\min})$ and $(I_{\max}, f_{\max})$ targets. There is no `t_ref` in the actual hardware.
3. **No depolarisation-block mechanism.** The model fired arbitrarily fast at high `I`. The real device latches at $I_{\text{in}} > I_{\text{hold}}$.
4. **Spike-shape sweeps were uninformative.** [`spike_Ra_sweep.py`](spike_Ra_sweep.py) varied `R_a` but produced no spike shape because the memristor never switched.

For network simulation of memristor hardware these limitations matter: synaptic events should be triggered by physically realisable spikes, the firing-rate ceiling should arise from a physical bound, and the spike width should set a physical timescale.

---

## 2. The physical reference (Wu et al. 2023)

Wu et al. introduce a **two-terminal memristor** `M` made of a thyristor `T` in parallel with a resistor `R` between its anode and gate. The macroscopic device exhibits a hysteretic two-state I–V characteristic (their Fig. 2, top-left inset):

- **Open state** `s = 0`: high resistance $R_m^{\text{hi}}$ (paper: ≈ 100 kΩ, dominated by the parallel resistor).
- **Closed state** `s = 1`: low resistance $R_m^{\text{lo}}$ (paper: tens-to-hundreds of Ω, thyristor in conduction).
- **Open → Closed** when the anode–cathode voltage exceeds a threshold $V_{\text{th}}$ set by the gate current $I_{\text{gt}}$.
- **Closed → Open** when the current through `R_a` falls below the holding current $I_{\text{hold}}$ (≈ 100 µA).

The single-compartment Memristive Spiking Neuron (MSN, paper §2, Fig. 1e) has:

- input current $I_{\text{in}}$ injected at the top node `V_m`,
- membrane capacitor `C_m` to ground,
- memristor `M` in series with a load resistor `R_a` (paper "Rload") between `V_m` and ground.

The externally measured spike is $V_{\text{out}} = V_m \cdot R_a / (R_m + R_a)$ (the voltage across `R_a`).

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
| Open → Closed | $V_m > V_{\text{th}}$ and `s = 0` | `s ← 1`; emit spike to downstream synapses |
| Closed → Open | $I_M < I_{\text{hold}}$ and `s = 1` | `s ← 0` |

where **`I_M = V_m / (R_m(s) + R_a)`** is the series current through the memristor + load resistor — the thyristor's anode current. $I_{\text{hold}}$ is the standard thyristor **holding current**: the minimum anode current required to maintain conduction. When $I_M$ drops below $I_{\text{hold}}$, the thyristor turns off.

**Why `I_hold` also equals `I_max`:** in the closed state ($R_m^{\text{lo}} \approx 0$), the steady-state current is $I_M^{ss} \approx I_{\text{in}}$. When $I_{\text{in}} > I_{\text{hold}}$, this steady state always exceeds $I_{\text{hold}}$, so the reopen event is never triggered and the neuron stays permanently latched (depolarisation block). Thus $I_{\text{max}} = I_{\text{hold}}$ is a consequence of the holding-current mechanism, not an independently set parameter.

### 3.1 Gate-current threshold

The thyristor closes when `V_m` exceeds the forward breakover voltage $V_{\text{th}}$, which is set by the gate current $I_{\text{gt}}$:

$$
V_{\text{th}} = I_{\text{gt}} \cdot R_m^{\text{hi}}
$$

($R_m^{\text{lo}}$ is negligible at the switch instant.) $V_{\text{th}}$ is stored as the primary parameter; $I_{\text{gt}}$ is the physically motivated way to derive it. Equivalently, the rheobase is approximately $I_{\text{gt}}$ because at threshold, the current flowing through the open-state circuit is $V_{\text{th}} / (R_m^{\text{hi}} + R_a) \approx I_{\text{gt}} \cdot R_m^{\text{hi}} / R_m^{\text{hi}} = I_{\text{gt}}$ when $R_a \ll R_m^{\text{hi}}$.

### 3.2 Spike discharge and turn-off

`V_m` discharges at the close event with a time constant approximately given by When `s` flips to 1, `R_m` drops from $R_m^{\text{hi}}$ to $R_m^{\text{lo}} \approx 0$, so the ODE becomes:

$$
C_m \frac{dV_m}{dt} \approx I_{\text{in}} - \frac{V_m}{R_a}
$$

`V_m` decays from $V_{\text{th}}$ toward the closed-state steady state $I_{\text{in}} \cdot R_a$ with the natural time constant

$$
\tau_{\text{close}} = C_m (R_m^{\text{lo}} + R_a) \approx C_m R_a
$$

The spike output $V_{\text{out}} = V_m \cdot R_a / (R_m^{\text{lo}} + R_a) \approx V_m$ rises to $V_{\text{th}}$ at the switch instant and then follows the same RC decay. The **spike peak amplitude** is:

$$
V_{\text{out}}^{\text{peak}} = V_{\text{th}} \cdot \frac{R_a}{R_m^{\text{lo}} + R_a}
$$

The device turns off when $I_M = V_m / R_a$ falls to $I_{\text{hold}}$, i.e. when:

$$
V_m = V_{\text{hold}} \equiv I_{\text{hold}} \cdot R_a
$$

(With $I_{\text{hold}} = 100\,\mu\text{A}$ and $R_a = 2\,\text{k}\Omega$: $V_{\text{hold}} = 0.2\,\text{V}$.) The **spike width** is the time for `V_m` to decay from $V_{\text{th}}$ to $V_{\text{hold}}$:

$$
t_{\text{spike}} = \tau_{\text{close}} \ln\!\left(\frac{V_{\text{th}} - I_{\text{in}} R_a}{V_{\text{hold}} - I_{\text{in}} R_a}\right)
$$

Both shape parameters ($V_{\text{out}}^{\text{peak}}$, $t_{\text{spike}}$) emerge from the circuit — no reset rule or manual $\tau_{\text{close}}$ is imposed.

### 3.3 Synaptic cascade — cascade lives on the synapse

Each pre-synaptic spike at neuron *j* contributes an instantaneous kick to a first-stage synaptic current $I_{s1}^{(j)}$, which feeds a second-stage current $I_{s2}^{(j)}$ via a passive cascade:

$$
\tau_{s1} \frac{dI_{s1}}{dt} = -I_{s1} + I_w \sum_{t_k} \delta(t - t_k)
$$

$$
\tau_{s2} \frac{dI_{s2}}{dt} = -I_{s2} + I_{s1}
$$

For $\tau_{s1} = \tau_{s2} = \tau_s$, the response of $I_{s2}$ to one pre-synaptic spike is the alpha function $(I_w/\tau_s) \cdot t \cdot e^{-t/\tau_s}$, peaking at $I_w/e$ at $t = \tau_s$.

**Where the ODE lives.** Architecturally, the cascade is integrated **on the Brian2 `Synapses` object**, not on the postsynaptic neuron. Each synapse type (E→I, I→E, mutual E↔E, …) owns its own `(τ_s1, τ_s2)` and contributes `I_{s2}` to the post neuron via a `(summed)` declaration. The neuron exposes named inlets (`I_exc`, `I_inh`, or any user-named inlet such as `I_inh_mutual`); each `Synapses` writes to exactly one inlet, and the neuron's Vm ODE uses the totals.

Biologically this is the receptor view: AMPA, NMDA, GABA-A, GABA-B each have their own kinetics; the postsynaptic neuron sees the sum. The practical payoff is that **multiple pathways with different time constants can converge on the same neuron** — fast E→I (~10–100 ms), slow I→E (~100–200 ms), and even slower mutual E↔E (~500 ms) can all coexist on a single E neuron without sharing one τ.

**Cascade settling time.** When firing begins at steady rate $f$, solving both ODEs with $\tau_{s1} = \tau_{s2} = \tau_s$ gives a closed-form build-up:

$$
I_{s2}(t) = A\!\left(1 - \left(1 + \frac{t}{\tau_s}\right)e^{-t/\tau_s}\right), \qquad A = I_w f \tau_s
$$

Settling milestones: 26% at $\tau_s$, 59% at $2\tau_s$, 80% at $3\tau_s$, 96% at $5\tau_s$, 99% at $7\tau_s$. With $\tau_s = 200$ ms, 96% convergence requires ~1 s — not 200 ms. An additional complication near rheobase: the firing rate $f$ is itself small when $I_0 + I_{s2}$ barely exceeds $I_{\min}$, making the early build-up sub-linear and extending the effective rise time beyond the constant-$f$ prediction (see §8.1).

The neuron sees $I_{\text{syn}} = I_{\text{exc}} - I_{\text{inh}}$, where each total is the sum over all inlets of the same kind.

### 3.4 Why this is not strictly LIF

Pure leaky integrate-and-fire is *defined* by the threshold-and-reset rule on `V_m`. In the MSN, `V_m` is continuous through the spike, the spike has a finite width, and "refractoriness" emerges from the discharge time constant rather than from an explicit `t_ref`. This places the MSN between LIF and Hodgkin–Huxley: a single voltage-gated channel (the memristor) replaces the abstraction of a magic reset rule. The macroscopic firing pattern is still LIF-like — type-1 excitability, integrate-then-fire — but the spike-generation mechanism is physical.

---

## 4. Implementation in Brian2

### 4.1 Why Brian2 (after considering scipy)

scipy's `solve_ivp` with event detection is the natural framework for hybrid continuous/discrete systems and gives sub-step accuracy on the switch instants. It was the initial recommendation for fitting a *single* spike shape. However:

- The end goal is **large-scale network simulation** with synapses and possibly plasticity. Brian2 is purpose-built for this.
- Brian2 supports **custom events** through `events={'name': '...'}` + `run_on_event()`. This provides exactly the mechanism needed for the close→open transition without porting back from scipy later.
- A 1 µs to 10 µs timestep is sufficient to resolve the spike shape ($\tau_{\text{close}} \approx 25$ ms) and the switch instants.

The model lives entirely in Brian2.

### 4.2 Code-generation backend and simulation cost

Brian2 probes for a working C++/Cython compiler at startup. Without one (e.g., WSL2 without `build-essential`), `x86_64-linux-gnu-g++` is absent and Brian2 emits a cascade of warnings (`Removing unsupported flag`, `Cannot use Cython`) before falling back to the NumPy backend — roughly 10× slower for large groups.

Two remedies:

1. **Install the compiler** (recommended for networks larger than a few dozen neurons):
   ```bash
   sudo apt-get install build-essential
   ```

2. **Opt in to NumPy explicitly** (prototyping, silences the warning chain):
   ```python
   prefs.codegen.target = 'numpy'
   ```

All scripts from `ns_msn_v3_bump.py` onward set option 2 explicitly at the top of the file.

**Timestep and simulation cost.** The MSN requires `dt = 1 µs` to resolve $\tau_{\text{close}} \approx 25$ ms; the aLIF and Thyristor models use 10–50 µs. For network runs, installing `build-essential` is strongly advised before scaling beyond a few dozen neurons.

### 4.3 NeuronGroup specification

Intrinsic params (`Cm`, `Ra`, `Rm_hi`, `Rm_lo`) are Brian2 namespace constants shared across the group. `Vth` and `I_hold` are promoted to per-neuron state variables so that `msn_variability.apply_variability(G)` can scatter device-specific values. The cascade ODEs are gone from the neuron; in their place are plain `amp` parameters used as named inlets written by Synapses objects:

```python
# Equations generated by _build_msn_eqs() for the default single-inlet case
eqs = """
dVm/dt = (I_0 + I_exc - (I_inh) - Vm/(Rm_S + Ra)) / Cm  : volt
Rm_S   = (1 - s)*Rm_hi + s*Rm_lo                         : ohm
I_M    = Vm / (Rm_S + Ra)                                 : amp
Vout   = Vm * Ra / (Rm_S + Ra)                            : volt
I_exc  : amp
I_inh  : amp
I_0    : amp
s      : 1
Vth    : volt
I_hold : amp
"""

G = NeuronGroup(
    N, eqs,
    threshold='Vm > Vth and s < 0.5',
    reset='s = 1',                            # do NOT reset Vm
    events={'reopen': 'I_M < I_hold and s > 0.5'},
    method='euler',
    namespace=dict(Cm=..., Ra=..., Rm_hi=..., Rm_lo=...),
)
G.run_on_event('reopen', 's = 0')
```

**`I_M = Vm / (Rm_S + Ra)`** is the series current through the memristor + load resistor — the thyristor's anode current. The reopen event fires when this drops below `I_hold`.

Brian2 allows only **one `Synapses` object** to write to a given `(inlet, target_group)` pair via `(summed)`. When multiple pathways of the same kind converge on one neuron with different kinetics (e.g. AMPA + NMDA, or GABA-A + GABA-B), declare additional named inlets at construction time. `make_msn` builds the equations dynamically so the `dVm/dt` expression sums all declared inlets:

```python
# AMPA + NMDA excitation, GABA-A + GABA-B inhibition
neurons = make_msn(N=100,
                   exc_inlets=('I_ampa', 'I_nmda'),
                   inh_inlets=('I_gaba_a', 'I_gaba_b'))
# dVm/dt = (I_0 + I_ampa + I_nmda - (I_gaba_a + I_gaba_b) - ...) / Cm
```

Each `Synapses` object writes to exactly one inlet via `target_var` in `SynapseParams`.

The Brian2 `threshold` event serves a dual role: it triggers the memristor close transition *and* fires the spike that downstream synapses listen to. This is physically correct — the moment the memristor closes is exactly when $V_{\text{out}}$ begins to rise, which is what a downstream synapse should see.

### 4.4 Synapse specification

Each Synapses object integrates its own cascade and writes the filtered current into a named inlet on the post neuron:

```python
syn = Synapses(source, target,
    model='''
        dIs1/dt = -Is1 / tau_s1                : amp (clock-driven)
        dIs2/dt = (-Is2 + Is1) / tau_s2        : amp (clock-driven)
        I_exc_post = Is2                       : amp (summed)
        w : amp
    ''',
    on_pre='Is1 += w',
    namespace={'tau_s1': 100*ms, 'tau_s2': 100*ms},
)
syn.connect(condition='i != j')
syn.w = weight * amp
```

`tau_s1`/`tau_s2` are per-synapse-group namespace constants — set by `SynapseParams.from_json(...)` for that synapse type. `w` is per-edge (Brian2 `model='w : amp'`) so individual edge weights are addressable for plasticity and heterogeneous networks. `Is1` and `Is2` are per-edge cascade state, recordable via a `StateMonitor` on the `Synapses` object.

---

## 5. Parameter choices and tuning principles

### 5.1 Hardware parameters

**Wu et al. 2023, Fig. 2 (paper reference values)**

| Symbol | Value | Description |
|---|---:|---|
| `C_m` | 10 µF | membrane capacitor |
| `R_a` | 2 kΩ | load resistor (paper "Rload") |
| $R_m^{\text{hi}}$ | 100 kΩ | open-state resistance |
| $R_m^{\text{lo}}$ | 500 Ω | closed-state resistance |
| $V_{\text{th}}$ | 1.5 V | thyristor close threshold |
| $I_{\text{hold}}$ | 100 µA | holding current (turn-off threshold) |

**Dec 2025 calibrated defaults in `MSNParams` (35 P0118MA thyristors)**

The code defaults in `msn_neuron.py` are calibrated against 35 P0118MA thyristors measured at `Ra = 2.2 kΩ`, `Rm = 680 kΩ`, `Cm = 100 nF`. The effective off-state resistance seen by the circuit is dominated by the thyristor's own anode–cathode impedance (~60 kΩ), not the external 680 kΩ gate resistor.

| Symbol | Code default | Description |
|---|---:|---|
| `Cm` | 100 nF | membrane capacitor |
| `Ra` | 2 200 Ω | load resistor |
| `Rm_hi` | 60 kΩ | effective open-state resistance |
| `Rm_lo` | 10 Ω | closed-state resistance (thyristor ≈ short) |
| `Vth` | 2.0 V | spike threshold |
| `I_hold` | 77 µA | median holding current (35-device dataset) |

### 5.2 Derived quantities (Dec 2025 calibrated defaults)

| Quantity | Formula | Value |
|---|---|---:|
| Rheobase | $I_{\min} = V_{\text{th}} / (R_m^{\text{hi}} + R_a)$ | ≈ 32 µA |
| Depol-block onset (`I_max`) | $I_{\max} = I_{\text{hold}}$ (see §3) | 77 µA |
| Hold voltage | $V_{\text{hold}} = I_{\text{hold}} \cdot (R_m^{\text{lo}} + R_a)$ | ≈ 0.17 V |
| Open-state τ | $\tau_{\text{open}} = C_m (R_m^{\text{hi}} + R_a)$ | ≈ 6.2 ms |
| Closed-state τ (spike width) | $\tau_{\text{close}} = C_m (R_m^{\text{lo}} + R_a)$ | ≈ 221 µs |
| Spike peak output | $V_{\text{out}}^{\text{peak}} \approx V_{\text{th}} \cdot R_a / R_a = V_{\text{th}}$ | ≈ 2.0 V |

### 5.3 Tonic bias `I_0` (per-neuron, set after construction)

| Regime | Behaviour |
|---|---|
| $I_0 \in (0, I_{\min})$ | silent on its own; needs synaptic input to fire |
| $I_0 \in (I_{\min}, I_{\max})$ | spontaneously firing |
| $I_0 > I_{\max}$ | latched closed → depolarisation block |

### 5.4 Synaptic weights and time constants

For $\tau_{s1} = \tau_{s2} = \tau_s$, the response of $I_{s2}$ to a single pre-synaptic spike is $(I_w/\tau_s) \cdot t \cdot e^{-t/\tau_s}$, peaking at $I_w/e \approx 0.37 I_w$ at $t = \tau_s$. For continuous Poisson input at rate $\lambda$ (with $\lambda \tau_s \gg 1$), the steady-state mean is $\langle I_{s2} \rangle \approx I_w \lambda \tau_s$.

$\tau_s$ should be **comparable to the target ISI**:
- $\tau_s \ll \text{ISI}$: spike-like blips with no integration (poor coupling to the slow MSN membrane).
- $\tau_s \sim \text{ISI}$: integration window matches the firing timescale.
- $\tau_s \gg \text{ISI}$: smooth low-pass — effectively a DC offset.

The library default is $\tau_s = 200$ ms; specific scripts (bump test, ring) override to 500 ms to ensure synaptic events outlive the membrane charging time.

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

Drives one MSN with constant $I_{\text{in}} = 92.4$ µA (paper's Fig. 2 caption). Verifies:

- `V_m` is continuous through the spike (no instantaneous drop).
- $V_{\text{out}}$ shows the peak and width set by §3.2.
- Memristor state `s(t)` toggles cleanly on the predicted events.
- The `(V_M, I_M)` orbit shows the same triangular topology as the paper's Fig. 2 main panel.

### 6.2 I–F characterisation ([`ns_msn_if_sweep.py`](ns_msn_if_sweep.py) → [`ns_msn_if_sweep.png`](ns_msn_if_sweep.png))

Sweeps drive currents from $I_{\min}$ to $I_{\text{hold}}$ in a single Brian2 NeuronGroup of N = 37 (vectorised parallel simulation). Compares numerical firing rates against the analytical formula:

$$
T(I_{\text{in}})
= \tau_{\text{open}} \ln\!\left(\frac{I_{\text{in}}(R_m^{\text{hi}}+R_a) - I_{\text{hold}} R_a}{I_{\text{in}}(R_m^{\text{hi}}+R_a) - V_{\text{th}}}\right)
- \tau_{\text{close}} \ln\!\left(\frac{(I_{\text{hold}} - I_{\text{in}}) R_a}{V_{\text{th}} - I_{\text{in}} R_a}\right)
$$

with $f = 1/T$, valid for $I_{\min} < I_{\text{in}} < I_{\text{hold}}$.

The first term is the charging time: `V_m` rises from $V_{\text{hold}} = I_{\text{hold}} R_a$ (where the previous spike ended) to $V_{\text{th}}$. The second term is the discharge time: `V_m` decays from $V_{\text{th}}$ to $V_{\text{hold}}$ through `R_a` while `s = 1`. Note the initial condition for the charging phase is now $V_{\text{hold}}$, not zero, because `V_m` is continuous and the previous spike ended at $V_m = V_{\text{hold}}$.

Results confirmed:
- **Type-1 excitability** at the left onset ($f \to 0$ as $I \to I_{\min}$).
- **Depolarisation-block cliff** at the right onset ($f$ drops abruptly to 0 at $I = I_{\text{hold}}$), which is the *physical* mechanism — not a parameter ceiling.
- Numerical points sit slightly above the analytical curve due to single-step overshoot at the close threshold; tightening `dt` reduces this.

### 6.3 Three-model comparison ([`ns_msn_compare.py`](ns_msn_compare.py) → [`ns_msn_compare.png`](ns_msn_compare.png))

Side-by-side simulation of three models at their own native parameter scales:

| Set | Topology | Spike mechanism | Refractory | Spike shape |
|---|---|---|---|---|
| **A — aLIF** | series `R_m + R_a` | threshold + reset (Vm → 0) | $t_{\text{ref}}$ parameter | none |
| **B — Thyristor** | parallel $g_a \parallel g_g$ | threshold + reset to $V_r = I_H/g_a \neq 0$ | `t_n` parameter | none |
| **D — MSN** | series `R_m + R_a` | hysteretic `R_m` state machine | emergent ($\tau_{\text{close}}$) | real |

Sets A and B are the previous abstractions from [`modelcopare.py`](modelcopare.py); set D is the new model. The figure shows that A and B have *no spike waveform* (instant jumps at the spike instant), while D has a continuous discharge waveform. Their I–F curves are also qualitatively different: A and B saturate at $1/t_{\text{ref}}$ and `1/t_n` parameter ceilings; D drops to zero at $I_{\text{hold}}$ (depol block).

### 6.4 Note on remaining quantitative mismatches with the paper

Two known discrepancies, both expected and tunable:

| Quantity | This model | Paper Fig. 2 | Cause |
|---|---|---|---|
| Spike width | $\tau_{\text{close}} \approx 25$ ms | ~1–3 ms | $R_m^{\text{lo}}$ fitted to 500 Ω and `R_a = 2` kΩ give $\tau_{\text{close}} = C_m(R_m^{\text{lo}}+R_a) \approx 25$ ms. The real thyristor on-state has a near-constant forward voltage $V_{\text{on}} \approx 0.65$ V plus small dynamic resistance. The two-state linear `R_m` abstraction gives an exponential decay rather than a constant-V plateau. |
| Vout peak | ≈ 1.2 V | ~150 mV | $V_{\text{out}}^{\text{peak}} = V_{\text{th}} R_a / (R_m^{\text{lo}} + R_a)$; with `R_a = 2` kΩ this is larger than the paper value. Tuning $R_m^{\text{lo}}$ and `R_a` together adjusts both width and peak. |

A future refinement is a *thyristor-style* closed state in which $V_M = V_{\text{on}} + I_M R_m^{\text{lo}}$, replacing the linear $R_m^{\text{lo}}$ with a clipped diode model. This is one extra subexpression in the equations and would close most of the gap.

---

## 7. Modular code organisation

After validation, the neuron and synapse construction were factored into a **split library** along the biological neuron/synapse boundary:

- [`msn_neuron.py`](msn_neuron.py) — `MSNParams`, `make_msn`. Equations are built dynamically via `_build_msn_eqs(exc_inlets, inh_inlets)`.
- [`msn_synapse.py`](msn_synapse.py) — `SynapseParams`, `make_synapse`. The cascade ODE (`Is1`, `Is2`) lives here.

Per-neuron JSON configs in [`configs/`](configs/) carry intrinsic parameters; per-synapse JSON configs carry kinetics + weight. Import directly from the library modules:

```python
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

# Single exc + inh inlet (default)
E = make_msn(N=10, params=MSNParams.from_json('configs/neuron_default.json'), name='E')
E.I_0 = 35e-6 * amp

# Each synapse type carries its own τ — different τ on the same target is fine
# because each Synapses object owns its own Is1/Is2 cascade
syn_E_to_I = make_synapse(E, I,
    params=SynapseParams.from_json('configs/synapse_default.json', key='exc'),
    connect=True, name='syn_E_to_I')

syn_I_to_E = make_synapse(I, E,
    params=SynapseParams.from_json('configs/synapse_default.json', key='inh'),
    connect=True, name='syn_I_to_E')
```

`MSNParams` carries the six intrinsic hardware parameters plus derived helpers (`operating_window()`, `time_constants()`, `summary()`); `SynapseParams` carries `(weight, kind, tau_s1, tau_s2, delay, target_var)`. Each has `from_json`, `to_json`, and `summary()` helpers.

**Multiple receptor types on one neuron** — declare named inlets at construction and assign a distinct `target_var` to each `SynapseParams`:

```python
# AMPA + NMDA excitation, GABA-A + GABA-B inhibition
neurons = make_msn(N=100,
                   exc_inlets=('I_ampa', 'I_nmda'),
                   inh_inlets=('I_gaba_a', 'I_gaba_b'))

make_synapse(pre, neurons,
    SynapseParams(kind='exc', weight=5e-6, tau_s1=2e-3,   tau_s2=5e-3,   target_var='I_ampa'),
    connect=..., name='ampa')
make_synapse(pre, neurons,
    SynapseParams(kind='exc', weight=3e-6, tau_s1=50e-3,  tau_s2=100e-3, target_var='I_nmda'),
    connect=..., name='nmda')
make_synapse(pre, neurons,
    SynapseParams(kind='inh', weight=8e-6, tau_s1=5e-3,   tau_s2=10e-3,  target_var='I_gaba_a'),
    connect=..., name='gaba_a')
make_synapse(pre, neurons,
    SynapseParams(kind='inh', weight=6e-6, tau_s1=100e-3, tau_s2=200e-3, target_var='I_gaba_b'),
    connect=..., name='gaba_b')
```

The rule: **one `Synapses` object per `(inlet_name, target_group)` pair**. Multiple outlet types from the same pre-synaptic neuron (different targets or different τ) are always free — each output connection is an independent `Synapses` with its own namespace.

---

## 8. Network-level demonstrations

### 8.1 Single neuron + self-loop bump ([`demo/ns_msn_v3_bump.py`](demo/ns_msn_v3_bump.py))

One MSN with $I_0 = 0.75 \cdot I_{\min}$ (subthreshold) and one self-excitatory edge (`τ_s = 200` ms). A 1 s step-current pulse triggers the first spikes; the self-loop then accumulates $I_{s2}$. After the pulse, whether activity is sustained or fades is governed by the criterion in §5.5.

### 8.2 20-neuron ring ([`demo/ns_msn_v4_network.py`](demo/ns_msn_v4_network.py))

Twenty MSN neurons in a 1-D periodic ring with local recurrent excitation: each neuron `i` projects to $i \pm 1$ and $i \pm 2$ (mod 20), giving 4 incoming edges per neuron (80 total). All 20 neurons receive subthreshold $I_0 = 0.85 \cdot I_{\min}$. A two-neuron cue at `i = 9, 10` is delivered at `t = 2.5` s.

Result: localised activity confined to a neighbourhood of the cue, fading over ~2 s as the recurrent current decays. This is the canonical ring-attractor / working-memory motif.

### 8.3 Two-neuron Winner-Take-All ([`demo/ns_msn_wta_demo.py`](demo/ns_msn_wta_demo.py))

Two MSN neurons connected by symmetric mutual inhibition. Three phases:
- **Phase 1 (0–3 s):** N1 driven above rheobase, N2 silent. N1 fires and builds $I_{\text{inh}}$ on N2.
- **Phase 2 (3–7 s):** N2 receives a stronger drive ($D2 = 2 \cdot I_{\min}$). N2 fires faster, rapidly suppresses N1.
- **Phase 3 (7–10 s):** N2 drive removed. N1 recovers as the inhibitory cascade decays.

Key tuning constraint: $I_w^{\text{inh}} < I_{\min} / (f_1 \cdot \tau_{s1})$ so that N2 can overcome N1's inhibition in Phase 2. With the Dec 2025 calibrated params, $I_w = 2$ µA and $\tau_s = 200$ ms work well.

### 8.4 Reservoir Computing ([`demo/ns_msn_rc_demo.py`](demo/ns_msn_rc_demo.py))

20 MSN neurons as a liquid-state reservoir for left/right Poisson-pattern classification. Two input streams (left, right) and recurrent connections each write to distinct named inlets (`I_exc_rec`, `I_exc_L`, `I_exc_R`) to satisfy the one-summed-writer rule. Ridge regression on spike counts achieves reliable train/test separation. A `USE_STDP` flag switches the recurrent weights between frozen-random and STDP-plastic; the STDP branch uses the cascade-on-synapse model with Apre/Apost traces.

---

## 9. File map and inheritance

```
Pre-existing (aLIF / Thyristor abstractions)
─────────────────────────────────────────────
  ns_test.py                  abstract aLIF + Poisson syn (no spike shape)
  spike_Ra_sweep.py           Ra sweep on the abstract aLIF
  modelcopare.py              aLIF vs Thyristor side-by-side

Paper-faithful MSN (validation)
─────────────────────────────────────────────
  ns_msn_v1.py                single trace, paper Fig. 2 reproduction
  demo/ns_msn_if_sweep.py     analytical + numerical I–F characterisation
  ns_msn_compare.py           aLIF vs Thyristor vs MSN side-by-side
  ns_msn_v2_synapses.py       MSN core + Is1/Is2 synapses + Poisson input

Canonical modular library (cascade on synapse)
─────────────────────────────────────────────
  msn_neuron.py               MSNParams, make_msn
                              make_msn(N, params, exc_inlets, inh_inlets, name)
                              Equations built dynamically — one inlet per
                              Synapses writer; multiple receptor types supported
  msn_synapse.py              SynapseParams, make_synapse
                              Cascade ODE (Is1, Is2) lives here; per-type τ
  msn_variability.py          apply_variability(G) — per-neuron Vth / I_hold
                              scatter from measured device distribution

Per-experiment configs (JSON, one file per neuron / synapse type)
─────────────────────────────────────────────
  configs/neuron_default.json     Dec 2025 calibrated MSNParams defaults
  configs/synapse_default.json    exc and inh SynapseParams defaults

Network demonstrations (all runnable from the demo/ directory)
─────────────────────────────────────────────
  demo/ns_msn_v3_bump.py      1 neuron + self-excit. bump test
  demo/ns_msn_v4_network.py   20-neuron ring with local recurrent exc
  demo/ns_msn_wta_demo.py     two-neuron mutual-inhibition WTA
  demo/ns_msn_rc_demo.py      reservoir-computing demo — left/right
                              Poisson classification, static or STDP weights
```

Each file's module docstring records its inheritance and what changed relative to its predecessor.

---

## 10. Limitations and planned extensions

1. **Slow firing rates.** At paper-faithful `C_m = 10` µF the maximum firing rate is on the order of single Hz. To reach the 30–80 Hz range characteristic of biological neurons, `C_m` would have to drop ~ 5–10×, with a corresponding loss of strict paper faithfulness.
2. **Linear closed-state `R_m`.** Replacing this with a thyristor-style $V_M = V_{\text{on}} + I_M R_m^{\text{lo}}$ would close most of the spike-shape gap with the paper's Fig. 2 (see §6.4).
3. **STDP plasticity — implemented in `ns_msn_rc_demo.py`.** The reservoir demo (`USE_STDP=True`) uses raw `Synapses` with Apre/Apost traces and the cascade-on-synapse model. `make_synapse` exposes per-edge `w` for heterogeneous and plastic weights. The long-term physical goal is to implement $I_w$ as the resistance of a non-volatile memristive device. This requires two distinct memristor operating regimes running simultaneously in the same circuit:

   | Role | Memristor type | Behaviour |
   |---|---|---|
   | Neuron ($R_m$) | Volatile / threshold-switching (TS) | Snaps to $R_m^{\text{lo}}$ at $V_{\text{th}}$; resets to $R_m^{\text{hi}}$ when $I_M < I_{\text{hold}}$ (no power needed to hold state) |
   | Synapse ($I_w$) | Non-volatile (filamentary, PCM, FeFET) | Resistance encodes the learned weight; persists without power until explicitly updated |

   The $I_{s1}/I_{s2}$ cascade already acts as an eligibility trace — its slow integration (~5$\tau_s \approx 1$ s settling, §3.3) provides the temporal coincidence window that a spike-timing-dependent rule needs.

4. **No bursting.** Wu et al. §3 (the MSBN) adds a second compartment $R_s, C_s$ in place of the ground reference at the bottom of `M`. The equations become
   $$
   C_m \dot V = I_{\text{in}} - (V - V_S)/R_m(s)
   \qquad
   C_s \dot V_S = (V - V_S)/R_m(s) - V_S/R_s
   $$
   with the threshold condition now on `V - V_S` rather than `V`. With $\tau_S = R_s C_s$ short relative to $\tau_m$, this generates the four spiking modes (TS, FS, IB1, IB2) on the $(\tau_S, I_{\text{in}})$ phase diagram.

5. **Parameter heterogeneity and code organisation — shipped.** Two layers of heterogeneity are supported: per-neuron intrinsic params and per-synapse-type kinetics.

---

## Appendix A. Reproducibility

All scripts use `seed(42)` (where stochastic elements appear), `defaultclock.dt = 1` µs–`10` µs depending on the experiment, and `method='euler'` for the membrane integration. State recording uses `dt = 2` µs (single-spike work) to `2` ms (network heatmaps) to keep memory usage bounded.

Each figure in the repository is regenerated by running the corresponding `ns_msn_*.py` script directly.
