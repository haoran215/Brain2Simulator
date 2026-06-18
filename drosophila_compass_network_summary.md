# Neuromorphic Simulation Notes: The Drosophila Compass Network (EPG/PEG/P-EN)
## Overview
This note provides a bottom-up structural and functional summary of the *Drosophila melanogaster* central complex compass network. It is formatted for direct insertion into simulation instructions for modeling ring attractor networks, CMOS-based memristive Spiking Neural Networks (SNNs), or computational neuroscience frameworks.

---

## 1. Neuropil Geography (The Network Nodes)
The network relies on three interconnected anatomical brain structures (neuropils). These are not individual cells, but geographical regions containing synaptic connections from multiple neuron classes:
* **Ellipsoid Body (E):** A doughnut-shaped (toroidal) structure. It acts as the physical space where the 360° heading direction of the fly is represented as a localized "bump" of neuronal activity.
* **Protocerebral Bridge (P):** A handlebar-shaped, segmented structure organized into functional columns (glomeruli). It acts as a routing and integration hub for position copies and self-motion (turning) inputs.
* **Gall (G):** A small, paired accessory structure acting as a primary output zone to translate high-level heading representation into motor commands.

---

## 2. Neuron Nomenclature & Structural Connectivity
Fly interneurons are named via their structural path: **`[Input Origin Location] -> [Intermediate/Output Location]`**. 
The compass dynamics emerge from the interaction of three distinct cell types:

### A. EPG Neurons (The Compass/Output Population)
* **Path:** $\text{E} \longrightarrow \text{P \& G}$
* **Anatomy:** Dendrites (inputs) are localized within specific "wedges" of the Ellipsoid Body (**E**). Axons project up to specific glomeruli in the Protocerebral Bridge (**P**) and terminate in the Gall (**G**).
* **Functional Role:** They act as the primary compass readout. They do *not* loop back to E themselves; they copy the local activity bump from E and transmit it up to P and outwards to G.

### B. PEG Neurons (The Baseline Recurrent/Stability Feedback)
* **Path:** $\text{P} \longrightarrow \text{E \& G}$
* **Anatomy:** Dendrites originate in **P**, and axons project back down into **E** and **G**.
* **Functional Role:** They provide direct topographic feedback ($\text{EPG} \rightarrow \text{P} \rightarrow \text{PEG} \rightarrow \text{E}$). This direct recurrent loop forms a classic **Ring Attractor**, sustaining the activity bump at its current position via feedback excitation when the agent is stationary or in the dark.

### C. P-EN Neurons (The Steering/Shift Feedback)
* **Path:** $\text{P} \longrightarrow \text{E \& N}$ (N = Noduli)
* **Anatomy:** Dendrites originate in **P**, where they also receive asymmetric self-motion / angular velocity inputs (vestibular-like turn signals). Axons project back down to **E**, but with a systematic **spatial offset (shift of one wedge)** to the left or right.
* **Functional Role:** When the agent rotates, asymmetric turn inputs activate either the left or right P-EN sub-population. Because their return paths to E are physically shifted, they excite the adjacent population of EPG neurons, causing the active "bump" to shift position along the ring.

---

## 3. Signal Flow & Ring Attractor Dynamics

### State 1: Static Maintenance (No Turning)
$$\text{Current Position Layer (E)} \xrightarrow{\text{EPG}} \text{Routing Layer (P)} \xrightarrow{\text{PEG}} \text{Current Position Layer (E)}$$
* *Result:* The bump remains stable via localized positive feedback loops.

### State 2: Dynamic Update (Clockwise/Counter-Clockwise Turning)
1. **Input:** $\text{Angular Velocity Signal} \longrightarrow \text{P-EN (at P)}$
2. **Integration:** $\text{EPG Position Copy (at P)} + \text{Turn Input (at P)} \longrightarrow \text{Asymmetric P-EN Activation}$
3. **Shifted Return:** $\text{P-EN} \xrightarrow{\text{Spatially Shifted Projections}} \text{Adjacent Wedge in E}$
4. **Update:** The EPG bump transitions to the neighboring population, updating the internal heading state.

---

## 4. Architectural Blueprint for SNN Implementation

To translate this biological topology into a neuromorphic or computational network architecture, initialize your layers and synaptic weight matrices using the following structural mapping:

```
                  [ Angular Velocity Inputs ]
                               │
                               ▼
  ┌────────────────────────────┴───────────────────────────┐
  │              Protocerebral Bridge (P Layer)            │
  └──────────────┬───────────────────────────▲─────────────┘
                 │                           │
  [PEG Feedback] │                           │ [EPG Projection]
                 │                           │ (Copy Position)
                 ▼                           │
  ┌──────────────────────────────────────────┴─────────────┐
  │               Ellipsoid Body (E Layer)                 │
  │               (Houses the Heading Bump)                │
  └──────────────▲─────────────────────────────────────────┘
                 │
  [P-EN Feedback]│
  (Spatially Shifted Topography)
```

### Layer Constraints
1. **E-Layer (Population of EPG Dendrites):** Arranged in a continuous ring topology representing 0–360 degrees.
2. **P-Layer (Population of PEG/P-EN Dendrites):** Arranged linearly/columnar to segment spatial coordinates.
3. **Synaptic Weight Matrices:**
   * $W_{\text{EPG}\to\text{P}}$: Purely topographic, mapping coordinate $x$ in E to coordinate $x$ in P.
   * $W_{\text{PEG}\to\text{E}}$: Purely topographic, returning localized reinforcement from $x$ in P back to $x$ in E.
   * $W_{\text{P-EN}\to\text{E}}$: Asymmetric shift matrix. Maps coordinate $x$ in P to coordinate $x \pm \Delta \theta$ in E, gated by the directional turning input vector.

---

## 5. Current & Timing Inventory of the 2-Neuron MSN Demo

This section documents the actual currents and synaptic time constants used in the
current 2-EB / 2-PB winner-take-all (WTA) flip-flop implementation
(`demo/ns_msn_compass_demo.py`), built on the MSN hardware neuron
(`msn_neuron.py`) and split-synapse library (`msn_synapse.py`).

### 5.1 Neuron operating window (sets the scale for every current)
From `msn_neuron.py`: $I_{gt} = V_{th}/(R_{m,hi}+R_a) = 2.0 / (60\,000 + 2\,200)$.

* **Rheobase** $I_{GT} \approx 32.2\ \mu\text{A}$ — below this a neuron is silent.
* **Depol-block ceiling** $I_{hold} = 100\ \mu\text{A}$ — total current above this latches
  the neuron permanently closed (silent).
* Every current below must keep a neuron's **total** input inside the ~32–100 µA window to spike.

### 5.2 Bias & external input currents (the "DC" drives)

| Current | Value | = µA | Target | Notes |
|---|---|---|---|---|
| `I0_EB`  | 0.90·I_GT | **28.9 µA** | EB_L, EB_R | subthreshold rest bias |
| `I0_GI`  | 0.80·I_GT | **25.7 µA** | EB_GI | lower — needs EB drive to fire |
| `I0_PB`  | 0.90·I_GT | **28.9 µA** | PB_l, PB_r | subthreshold rest bias |
| `SEED_I` | 1.8·I_GT  | **57.9 µA** | EB_R, first 80 ms | picks starting winner |
| `PULSE_I`| 0.16·I_GT | **+5.1 µA** | both PBs | sensory pulse, 60 ms, at t = 0.7 / 1.3 / 1.9 s |

### 5.3 Synaptic currents and time constants

Each synapse is an **alpha cascade** (`cascade='alpha'`, with `tau_s1 = tau_s2 = τ`).
For a single presynaptic spike the postsynaptic current is
$I_{s2}(t) = w\,(t/\tau)\,e^{-t/\tau}$, which **peaks at $t = \tau$** with peak value
$w/e = 0.368\,w$. So the weight `w` is the `Is1` injection, but the neuron actually
sees a peak of only ~37% of `w` per spike (sustained firing summates higher).

| Pathway | code var | `w` (peak inject) | peak `Is2` = w/e | `τ = τ_s1 = τ_s2` | time-to-peak |
|---|---|---|---|---|---|
| EB self-excitation (latch)         | `W_SELF`  | **11 µA** | 4.0 µA | **12 ms** | 12 ms |
| EB → EB_GI (drive inhibitor)       | `W_EBGI`  | **14 µA** | 5.2 µA | **8 ms**  | 8 ms  |
| EB_GI → EB (WTA inhibition)        | `W_GIEB`  | **9 µA**  | 3.3 µA | **8 ms**  | 8 ms  |
| EB → PB (position copy, same side) | `W_COPY`  | **9 µA**  | 3.3 µA | **12 ms** | 12 ms |
| PB → EB (cross-activation, opp.)   | `W_CROSS` | **10 µA** | 3.7 µA | **12 ms** | 12 ms |

**WTA-gain margin:** GI loop $W_{EBGI}\cdot W_{GIEB} = 14\cdot9 = 126$ vs cross loop
$W_{COPY}\cdot W_{CROSS} = 9\cdot10 = 90$. So 126 > 90 — the margin is *nominally*
satisfied, yet the pair still co-fires. The product rule is therefore
necessary-but-not-sufficient: the 0.90·I_GT bias sits so close to rheobase that even
the small loser-side leakage current crosses threshold. This is the key Stage-0 lever.

### 5.4 The time delay (the critical timing property)

**There is no explicit synaptic transmission delay anywhere in this network.**
`SynapseParams.delay` defaults to `0.0` and the demo never sets it, so `delay = 0 ms`
on all 10 synapses. The *only* timing in the loops is the **alpha-function
time-to-peak, which equals τ**. That gives these effective **loop latencies**
(spike → effect fully developed around the loop):

* **Self-latch:** ~**12 ms** (one synapse)
* **WTA inhibition loop** EB→GI→EB: 8 + 8 = ~**16 ms**
* **Flip / cross loop** EB→PB(copy) → PB→EB(cross): 12 + 12 = ~**24 ms**

So the flip path (~24 ms) is *slower* than the inhibition path (~16 ms). For a clean
flip the primed PB must cross-activate the opposite EB and let it win the WTA
**before** global inhibition collapses the whole pair — and currently the cross loop
is the slowest of the three, which is part of why the flips are mushy rather than crisp.

Two consequences for tuning:
1. The membrane's own $\tau_{open} \approx 6.2\ \text{ms}$ adds on top of each synaptic
   time-to-peak — the real per-stage latency is roughly $\tau_{syn} + \tau_{open}$.
2. To control flip timing (and later, turn speed when scaling up), the clean lever is
   to set `tau_s1 ≠ tau_s2` and/or add an explicit `delay` per pathway, rather than
   relying only on the equal-τ alpha peak. The model already supports both; they are
   just unused in the current demo.


