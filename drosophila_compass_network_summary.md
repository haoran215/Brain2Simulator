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
