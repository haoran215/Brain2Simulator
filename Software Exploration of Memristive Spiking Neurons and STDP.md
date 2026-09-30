# Research Objective

The current project already has:

1. A memristive spiking neuron (MSN) model.
2. A working STDP implementation.
3. Published work related to STDP.
4. A PCB implementation of the STDP mechanism.

Therefore, **do not treat STDP implementation itself as the main research problem**.

The current research question is:

> **What useful computations can emerge when the existing memristive spiking neuron dynamics are combined with local STDP?**

The software simulation should therefore be used as a **research exploration platform**.

The goal is to identify computational principles that are sufficiently interesting and sufficiently hardware-compatible to justify future hardware implementation.

The progression should be:

\[
\boxed{
\text{MSN}
\rightarrow
\text{STDP}
\rightarrow
\text{network dynamics}
\rightarrow
\text{computation}
\rightarrow
\text{hardware candidate}
}
\]

Do not start from MNIST and optimize classification accuracy.

---

# Phase 0 — Understand the existing system

Before implementing new experiments, inspect the existing repository and identify:

### MSN

- state variables;
- membrane dynamics;
- memristive state;
- nonlinearities;
- hysteresis;
- adaptation;
- refractory/reset mechanism;
- time constants;
- parameter ranges;
- device variability model.

### STDP

Identify the existing:

- pre-synaptic trace;
- post-synaptic trace;
- learning window;
- potentiation mechanism;
- depression mechanism;
- weight constraints;
- update timing;
- weight precision.

### Existing network

Identify whether the repository already contains:

- recurrent connections;
- WTA;
- inhibition;
- bump attractors;
- CPGs;
- head-direction networks;
- sensory encoding;
- robotics experiments.

Do not duplicate existing implementations.

The first deliverable should be a short map:

```text
MSN dynamics
      ↓
STDP mechanism
      ↓
existing network primitives
      ↓
existing behavioral/computational primitives
```

---

# Phase 1 — Characterize the MSN as a computational element

Do not immediately train anything.

Determine what computational properties the MSN itself provides.

Compare:

```text
LIF
vs
MSN
```

under identical inputs.

Measure:

- firing threshold;
- firing frequency;
- adaptation;
- response to pulse trains;
- response to burst inputs;
- response to irregular inputs;
- response to sustained input;
- response after previous stimulation;
- recovery dynamics;
- hysteresis;
- sensitivity to input timing;
- sensitivity to input history.

Most importantly, determine whether:

\[
V(t)
\]

can be described adequately by instantaneous input alone, or whether the memristive state introduces significant history dependence:

\[
V(t)=F(I(t),x(t))
\]

where \(x(t)\) is the internal device/neuron state.

If the latter is true, characterize the effective temporal memory of the neuron.

---

# Phase 2 — MSN + STDP as a temporal learning element

The first real learning experiment should NOT be MNIST.

Use controlled synthetic temporal patterns.

For example:

```text
Pattern A:
A → B → C → D

Pattern B:
A → C → B → D
```

or:

```text
Pattern A:
1 → 2 → 3

Pattern B:
1 → 3 → 2
```

Train the network using only local STDP.

Test whether the network can distinguish temporal patterns.

Compare:

\[
\text{LIF + STDP}
\]

against:

\[
\boxed{\text{MSN + STDP}}
\]

The purpose is to determine whether the MSN's internal state provides useful temporal processing.

---

# Phase 3 — Separate synaptic memory from neuronal memory

This is an important experiment.

There are potentially two sources of memory:

### Synaptic memory

\[
w_{ij}
\]

created by STDP.

### Neuronal/device memory

\[
x_i
\]

created by the internal MSN dynamics.

Study the four cases:

```text
A: LIF + fixed synapses

B: MSN + fixed synapses

C: LIF + STDP

D: MSN + STDP
```

This gives a 2 × 2 experiment:

| | Fixed synapses | STDP |
|---|---|---|
| LIF | A | C |
| MSN | B | D |

This experiment is extremely important because it tells us whether the useful behavior comes from:

\[
\text{neuron dynamics}
\]

or:

\[
\text{synaptic plasticity}
\]

or their interaction.

The key hypothesis to investigate is:

\[
\boxed{
\text{MSN memory}
\times
\text{synaptic memory}
}
\]

rather than treating them independently.

---

# Phase 4 — Reservoir-like computation

Investigate whether the MSN network can operate as a nonlinear temporal processing substrate.

Construct a recurrent MSN network:

\[
\mathbf{x}_{t+1}
=
F(\mathbf{x}_t,\mathbf{u}_t)
\]

where the MSN dynamics provide the internal state.

Test simple temporal tasks:

### Temporal XOR

Determine whether:

\[
y_t=x_t\oplus x_{t-\Delta}
\]

can be decoded from the network state.

### Delayed classification

Input:

```text
A
delay
B
```

and classify:

```text
AB
vs
BA
```

### Temporal sequence classification

Classify:

```text
A-B-C
A-C-B
B-A-C
B-C-A
```

The purpose is not to claim that the MSN is a reservoir computer.

The purpose is to determine whether its intrinsic dynamics produce useful nonlinear temporal representations.

---

# Phase 5 — STDP as self-organization

Once temporal computation works, investigate what STDP does to the network structure.

Start with a recurrent MSN network:

\[
W_{ij}
\]

and initially random connectivity.

Apply STDP.

Observe whether the network spontaneously develops:

- clusters;
- recurrent loops;
- winner-take-all structure;
- assemblies;
- sequential activation;
- oscillatory groups;
- attractor states.

Analyze:

\[
W(t)
\]

as a dynamical object.

Do not only measure classification accuracy.

Measure:

- weight distribution;
- connectivity distribution;
- clustering;
- reciprocity;
- sequence structure;
- population activity;
- attractor stability.

---

# Phase 6 — STDP + attractor formation

This phase connects directly to the existing bump/head-direction research.

Construct a recurrent MSN population arranged on a ring:

\[
\theta_i=\frac{2\pi i}{N}
\]

Initially use either:

1. random connectivity, or
2. weak local connectivity.

Then apply STDP while presenting structured temporal sequences.

Investigate whether STDP can produce or reinforce a localized activity bump:

\[
x_i(t)
\]

such that:

\[
x_i(t)\approx f(\theta_i-\theta_{HD})
\]

The key question is:

> Can local STDP and MSN dynamics generate or stabilize attractor-like representations?

Measure:

- bump width;
- bump amplitude;
- bump velocity;
- attractor stability;
- transition probability;
- noise tolerance;
- recovery after perturbation.

This should connect the current MSN model to the existing head-direction work.

---

# Phase 7 — Learn WTA rather than manually designing WTA

If possible, investigate whether STDP can produce competitive network structure.

Start with several MSN populations receiving overlapping inputs.

Use:

- excitatory recurrent connections;
- inhibitory interactions;
- local STDP.

Investigate whether the network develops specialization:

\[
\text{input cluster}
\rightarrow
\text{preferred neuron/population}
\]

This would move the system from:

```text
manually designed WTA
```

toward:

```text
self-organized WTA
```

Measure:

- selectivity;
- winner stability;
- competition;
- number of active neurons;
- response sparsity.

---

# Phase 8 — Sequence learning

This is potentially one of the most interesting directions.

Use STDP to learn temporal sequences:

\[
A\rightarrow B\rightarrow C\rightarrow D
\]

After learning, present:

\[
A
\]

and determine whether network activity evolves:

\[
A\rightarrow B\rightarrow C\rightarrow D
\]

without explicitly providing the later inputs.

This creates a potential sequence memory.

Investigate whether the MSN's intrinsic dynamics improve:

- sequence stability;
- timing robustness;
- tolerance to missing spikes;
- recovery from perturbation.

Compare:

\[
\text{LIF + STDP}
\]

and:

\[
\boxed{\text{MSN + STDP}}
\]

---

# Phase 9 — Continual learning

Only after the previous experiments work should continual learning be introduced.

Use sequential tasks:

```text
Task 1
↓
Task 2
↓
Task 3
```

Measure catastrophic forgetting:

\[
F_k=A_k^{before}-A_k^{after}
\]

Compare:

```text
LIF + STDP
MSN + STDP
```

Investigate whether the MSN's intrinsic memory provides any natural resistance to interference.

Do not assume that it will.

This must be experimentally demonstrated.

---

# Phase 10 — Hardware-aware constraints

Once interesting behavior has been found, progressively introduce hardware constraints.

## Weight precision

Test:

```text
FP32
8-bit
4-bit
3-bit
2-bit
```

## Device variability

Introduce:

\[
p_i=p_0(1+\epsilon_i)
\]

with:

\[
\epsilon_i\sim\mathcal N(0,\sigma^2)
\]

Test multiple \(\sigma\).

## Parameter mismatch

Perturb:

- threshold;
- time constants;
- gain;
- memristive state;
- leakage;
- synaptic efficacy.

## Limited dynamic range

Constrain:

\[
G_{min}\le G\le G_{max}
\]

## Update asymmetry

If the physical STDP implementation has asymmetric potentiation/depression behavior, reproduce that in simulation.

The objective is to discover:

> Which computational mechanisms survive realistic hardware imperfections?

---

# Phase 11 — Only now use standard datasets

MNIST should be treated as a benchmark, not the central research question.

Use:

\[
\text{MNIST}
\]

to demonstrate that the system can perform a standard classification task.

Then consider temporal/event-based datasets such as:

\[
\text{N-MNIST}
\]

or:

\[
\text{DVS Gesture}
\]

because they are more naturally aligned with:

\[
\text{spikes}
+
\text{temporal dynamics}
+
\text{STDP}
\]

For every dataset, maintain the same comparison:

\[
\boxed{
LIF + STDP
\quad vs \quad
MSN + STDP
}
\]

---

# Phase 12 — Identify the "killer application"

Do not assume beforehand that MNIST, continual learning, or temporal classification is the final application.

Instead, use the simulations to identify where the MSN provides a measurable advantage.

Possible outcomes include:

### A. Temporal memory

MSN maintains useful information through intrinsic dynamics.

### B. Robust sequence processing

MSN + STDP learns sequences under noise.

### C. Attractor formation

MSN networks naturally produce stable states.

### D. Hardware robustness

MSN computation remains functional under device mismatch.

### E. Low-precision operation

MSN networks maintain computation with very low synaptic precision.

### F. Continual learning

MSN + STDP shows different interference/forgetting behavior.

### G. Embodied computation

MSN dynamics simplify a closed-loop sensorimotor controller.

The simulation should determine which of these is actually supported.

---

# 13. Final conceptual framework

The entire project should be interpreted through three interacting forms of state:

### 1. Fast neuronal state

\[
V_i(t)
\]

### 2. Slow intrinsic/device state

\[
x_i(t)
\]

### 3. Synaptic state

\[
w_{ij}(t)
\]

Therefore the network can be viewed as:

\[
\boxed{
\begin{aligned}
\dot V_i &= F(V_i,x_i,I_i,W)\\
\dot x_i &= G(V_i,x_i)\\
\dot w_{ij} &= H(w_{ij},t_i^{pre},t_j^{post})
\end{aligned}
}
\]

This is the central mathematical object to investigate.

The research question becomes:

\[
\boxed{
\text{What computation emerges from the interaction of these three timescales?}
}
\]

This should guide the simulation work.

---

# 14. Priority order

Do NOT implement everything simultaneously.

Use this order:

```text
1. MSN characterization
        ↓
2. MSN + STDP temporal patterns
        ↓
3. 2×2 memory experiment
        ↓
4. recurrent MSN network
        ↓
5. sequence learning
        ↓
6. attractor/bump formation
        ↓
7. WTA/self-organization
        ↓
8. continual learning
        ↓
9. hardware constraints
        ↓
10. MNIST / N-MNIST / DVS
```

At every stage ask:

> What new computational property has been demonstrated that was not present in the previous stage?

If there is no new property, stop and redesign the experiment rather than adding complexity.

---

# 15. Criteria for a potentially publishable result

A strong result does NOT require:

\[
Accuracy_{MSN}>Accuracy_{LIF}
\]

on every dataset.

More interesting results could be:

\[
\boxed{
\text{MSN + STDP}
}
\]

showing:

- stronger temporal memory;
- more stable attractors;
- better sequence recall;
- robustness to noise;
- robustness to device mismatch;
- lower required synaptic precision;
- useful intrinsic memory;
- reduced need for externally designed network structure.

The key is to establish a causal relationship:

\[
\boxed{
\text{MSN dynamics}
\rightarrow
\text{network property}
\rightarrow
\text{computational advantage}
}
\]

rather than merely observing a difference in benchmark accuracy.

---

# 16. Final objective

The final goal of this software study is to identify one or two computational mechanisms worth implementing in hardware.

The workflow should therefore be:

\[
\boxed{
\text{Explore broadly in simulation}
}
\]

↓

\[
\boxed{
\text{Identify useful MSN + STDP behavior}
}
\]

↓

\[
\boxed{
\text{Test hardware constraints}
}
\]

↓

\[
\boxed{
\text{Select one mechanism}
}
\]

↓

\[
\boxed{
\text{Implement on neuromorphic hardware}
}
\]

The software stage is therefore not simply a demonstration platform.

It is a **design-space exploration tool for deciding what the hardware should actually implement.**