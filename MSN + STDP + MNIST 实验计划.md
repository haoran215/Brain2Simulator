# Experiment Plan: Memristive Spiking Neuron + STDP on MNIST

## 0. Context

I already have a Python/Brian2 codebase containing a **memristive spiking neuron (MSN) model** and related neuromorphic experiments.

The goal of this experiment is to investigate whether the MSN model can be used as the neuron model in a **local, spike-driven STDP learning system**, and to compare its behavior against a conventional LIF baseline.

This is not intended to be merely an MNIST classification demo.

The scientific questions are:

1. Can the MSN model support stable STDP-based learning?
2. Does the nonlinear/memristive neuron dynamics change the learning behavior compared with LIF?
3. How robust is the learning under hardware-oriented constraints such as limited weight precision and device variability?
4. Can the learned synaptic states be interpreted physically as memristive conductances?
5. Does the MSN provide any useful computational behavior beyond simply replacing LIF as a neuron model?

The implementation should remain as close as possible to the existing codebase and existing MSN equations.

---

# 1. First step: inspect the existing codebase

Before modifying anything:

### Identify

- the current MSN neuron equations;
- state variables;
- membrane voltage/state variable;
- memristive/device state variable;
- spike-generation condition;
- reset mechanism;
- refractory mechanism, if present;
- parameter definitions;
- existing simulation scripts;
- existing plotting/analysis utilities;
- existing neuron populations;
- existing synapse implementation;
- whether the current model is compatible with `Brian2.Synapses`.

Do NOT rewrite the MSN model unnecessarily.

The existing MSN model should become the neuron model used by the experiment.

First create a short technical summary of the existing model:

```text
MSN state variables:
- ...
- ...
- ...

Spike condition:
...

Reset:
...

Memristive state:
...

Relevant parameters:
...
```

If the existing implementation is not directly compatible with Brian2 `Synapses`, adapt it minimally.

---

# 2. Overall experimental architecture

The first architecture should be:

```text
MNIST image
      ↓
784 pixel intensities
      ↓
spike encoding
      ↓
784 input spike trains
      ↓
      STDP synapses
      ↓
10 MSN neurons
      ↓
winner-take-all / lateral inhibition
      ↓
digit prediction
```

Initial network:

```text
Input neurons: 784
Output neurons: 10
Connectivity: all-to-all
Synapses: 784 × 10 = 7840
Learning: local STDP
Neuron: MSN
```

Do not initially implement a complicated multi-layer network.

The purpose of the first experiment is to isolate:

```text
neuron dynamics + synaptic plasticity
```

---

# 3. MNIST spike encoding

MNIST is static spatial data, so it does not naturally contain temporal spike information.

Therefore, explicitly document the encoding method.

Start with a simple rate-based Poisson encoding:

```text
pixel intensity → Poisson firing rate
```

For example:

```text
I_pixel ∈ [0, 1]

r_pixel = r_max × I_pixel
```

Possible initial parameters:

```text
r_max = 100–200 Hz
presentation_time = 100–200 ms
```

Keep the encoding parameters configurable.

Do not optimize the encoder aggressively at the beginning.

The first goal is to establish a reproducible baseline.

---

# 4. STDP implementation

Implement a standard pair-based additive STDP rule first.

Use local pre- and post-synaptic traces:

\[
\frac{dA_{pre}}{dt}=-\frac{A_{pre}}{\tau_{pre}}
\]

\[
\frac{dA_{post}}{dt}=-\frac{A_{post}}{\tau_{post}}
\]

On a presynaptic spike:

\[
w \leftarrow w + A_{post}
\]

On a postsynaptic spike:

\[
w \leftarrow w + A_{pre}
\]

with the appropriate sign convention so that:

- causal pre → post firing produces potentiation;
- post → pre firing produces depression.

Use weight clipping:

\[
w_{min}\leq w\leq w_{max}
\]

The implementation should use Brian2's event-driven `Synapses` mechanism where possible.

Do not use backpropagation.

Do not use global error gradients.

The learning rule must remain local.

---

# 5. Establish a conventional LIF baseline

Before testing the MSN, implement the same network using a conventional LIF neuron.

Architecture:

```text
784 input neurons
        ↓
   STDP synapses
        ↓
10 LIF neurons
```

This is the control experiment.

The LIF and MSN experiments must use:

- identical MNIST subset;
- identical spike encoding;
- identical presentation time;
- identical STDP parameters initially;
- identical initialization strategy;
- identical classifier/readout strategy.

Only the neuron model should change.

This gives:

```text
LIF + STDP
vs.
MSN + STDP
```

This comparison is essential.

---

# 6. Experiment 1 — STDP sanity check

Before MNIST, test STDP using a very small synthetic network.

Example:

```text
2–10 input neurons
1 output neuron
```

Create controlled spike trains where temporal correlations are known.

Verify:

### Case A: pre before post

\[
t_{pre}<t_{post}
\]

Expected:

\[
\Delta w>0
\]

### Case B: post before pre

\[
t_{post}<t_{pre}
\]

Expected:

\[
\Delta w<0
\]

Plot:

```text
Δw vs Δt
```

The resulting curve should qualitatively resemble the intended STDP learning window.

Do this for both:

```text
LIF
MSN
```

This experiment must be completed before running MNIST.

---

# 7. Experiment 2 — Single-image learning

Use a very small MNIST subset first.

For example:

```text
100 images
```

Use only a few digit classes initially, e.g.

```text
0, 1, 2
```

Train the MSN network and visualize:

1. input spike raster;
2. output spike raster;
3. output firing rate;
4. synaptic weight evolution;
5. final weight matrix;
6. each output neuron's receptive field.

For each output neuron:

```text
784 synaptic weights
      ↓
reshape
28 × 28
      ↓
visualize as image
```

This is important because STDP should produce structured receptive fields rather than completely random weights.

---

# 8. Experiment 3 — 10-class MNIST

After the small experiment works:

```text
10 classes
```

Start with:

```text
N_train = 1,000
```

Then:

```text
N_train = 10,000
```

Finally:

```text
N_train = 60,000
```

Use the standard MNIST test set:

```text
N_test = 10,000
```

Do not immediately spend computational resources on the full dataset.

Each stage must be reproducible.

---

# 9. Classification/readout strategy

STDP itself is unsupervised.

Therefore explicitly separate:

### Learning

```text
MNIST → spike encoding → STDP → synaptic weights
```

from:

### Classification

```text
output neuron activity → digit label assignment
```

Use a standard post-training label assignment strategy.

For example:

1. Present training samples.
2. Record the output neuron with the strongest response for each sample.
3. Assign each output neuron the digit for which it responds most strongly.
4. Freeze the synaptic weights.
5. Evaluate on the test set.

Do not use backpropagation to assign or optimize the weights.

Document exactly how labels are assigned.

---

# 10. Primary comparison

The main experiment should be:

## Model A

```text
LIF + STDP
```

## Model B

```text
MSN + STDP
```

Keep all other parameters as equal as possible.

Compare:

### Classification

- test accuracy;
- per-class accuracy;
- confusion matrix.

### Neural dynamics

- average firing rate;
- spikes/image;
- output spike timing;
- number of active neurons;
- winner stability.

### Learning

- mean synaptic weight;
- weight distribution;
- weight evolution;
- STDP update magnitude;
- potentiation/depression ratio.

### Representation

- learned 28×28 receptive fields;
- similarity between neurons;
- class selectivity.

---

# 11. Experiment 4 — Weight quantization

This is particularly important for the neuromorphic hardware direction.

The first experiment can use floating-point weights.

Then progressively constrain the synaptic weights.

For example:

### Floating point

```text
w ∈ [0, 1]
```

### 8-bit

```text
256 levels
```

### 4-bit

```text
16 levels
```

### 3-bit

```text
8 levels
```

### 2-bit

```text
4 levels
```

For example:

\[
w\in
\left\{
0,
\frac{1}{7},
\frac{2}{7},
...
,1
\right\}
\]

for 3-bit normalized weights.

Implement quantization during/after STDP updates in a hardware-realistic way.

Compare:

```text
accuracy
spike count
weight distribution
learning stability
```

The goal is to determine how much synaptic precision the MSN network actually needs.

---

# 12. Experiment 5 — Device variability

Introduce device-level variability into the MSN parameters.

For example:

\[
p_i=p_0(1+\epsilon_i)
\]

where

\[
\epsilon_i\sim\mathcal{N}(0,\sigma^2)
\]

Possible parameters to perturb:

- threshold;
- leakage;
- memristive conductance;
- state-transition parameters;
- time constants;
- gain parameters.

Test several variability levels:

```text
σ = 0%
σ = 1%
σ = 5%
σ = 10%
σ = 20%
```

Run multiple random seeds for each condition.

Measure:

```text
accuracy
variance of accuracy
firing-rate distribution
weight distribution
```

This experiment is important because the scientific value of the MSN model is partly related to whether its behavior remains useful under non-ideal device conditions.

---

# 13. Experiment 6 — Memristor-constrained synaptic model

If appropriate for the existing codebase, map synaptic weight to a physical conductance:

\[
w \leftrightarrow G_{mem}
\]

and represent the synaptic state as:

\[
G_{min}\leq G_{mem}\leq G_{max}
\]

Then investigate whether STDP can be represented as:

\[
\Delta G=f(\Delta t)
\]

rather than treating weight as an abstract floating-point variable.

If the existing memristive device model has a physical update equation, use that equation instead of an arbitrary clipping rule.

The objective is:

```text
STDP algorithm
      ↓
synaptic weight
      ↓
memristive conductance
```

This should connect the algorithmic experiment to the eventual circuit/device implementation.

---

# 14. Important control experiment — remove MSN-specific dynamics

If the MSN contains nonlinear/hysteretic/device-dependent dynamics, determine whether the observed improvement comes from the neuron dynamics or simply from parameter differences.

Construct an appropriate simplified neuron model where possible.

Possible comparison:

```text
LIF
↓
MSN without special nonlinear mechanism
↓
full MSN
```

This allows us to distinguish:

```text
"MSN has more parameters"
```

from

```text
"MSN dynamics actually contribute to computation"
```

This distinction is scientifically important.

---

# 15. Metrics

Create a unified experiment table.

At minimum:

| Metric | LIF + STDP | MSN + STDP |
|---|---:|---:|
| Test accuracy | | |
| Train accuracy | | |
| Spikes/image | | |
| Output firing rate | | |
| Training time | | |
| Mean weight | | |
| Weight variance | | |
| Weight entropy | | |
| STDP updates/image | | |
| Robustness to variability | | |
| Accuracy at 8-bit | | |
| Accuracy at 4-bit | | |
| Accuracy at 3-bit | | |
| Accuracy at 2-bit | | |

For stochastic experiments, report:

\[
\text{mean}\pm\text{std}
\]

over multiple random seeds.

At least 3 seeds for preliminary experiments.

Prefer 5+ seeds for final comparisons if computationally feasible.

---

# 16. Required visualizations

Generate the following figures.

### Figure 1 — STDP learning window

\[
\Delta w(\Delta t)
\]

Compare LIF and MSN if meaningful.

---

### Figure 2 — Example spike raster

```text
input spikes
output spikes
```

for several MNIST samples.

---

### Figure 3 — Learned receptive fields

Show:

```text
10 output neurons
×
28 × 28 receptive field
```

for LIF and MSN.

---

### Figure 4 — Weight evolution

Plot:

\[
w(t)
\]

for representative synapses.

---

### Figure 5 — Weight distributions

Before and after training.

---

### Figure 6 — Accuracy vs training set size

```text
100
1k
10k
60k
```

---

### Figure 7 — Accuracy vs weight precision

```text
FP32
8-bit
4-bit
3-bit
2-bit
```

---

### Figure 8 — Accuracy vs device variability

```text
0%
1%
5%
10%
20%
```

---

# 17. Important scientific caveat about MNIST

MNIST is useful as a benchmark, but it is not an ideal dataset for demonstrating the fundamental advantage of STDP.

MNIST is fundamentally static spatial data.

STDP exploits temporal relationships:

\[
\Delta w=f(t_{post}-t_{pre})
\]

Therefore, a Poisson/rate encoding of MNIST introduces artificial temporal structure.

The MNIST experiment should therefore be interpreted as:

> Can the neuromorphic learning system perform a standard benchmark using local spike-based learning?

It should NOT be interpreted as definitive evidence that temporal STDP is advantageous for natural temporal computation.

---

# 18. Follow-up experiment — event-based dataset

After MNIST, consider an event-based dataset such as:

```text
N-MNIST
DVS Gesture
```

The architecture can then remain similar:

```text
event camera
    ↓
event stream
    ↓
MSN
    ↓
STDP
    ↓
classification
```

This is much more relevant to the hypothesis:

\[
\text{temporal dynamics}
+
\text{local plasticity}
\rightarrow
\text{useful computation}
\]

The event-based experiment should be treated as a second-stage experiment, not a prerequisite for getting the MNIST pipeline working.

---

# 19. Implementation requirements

Use:

```text
Python
Brian2
NumPy
Matplotlib
```

Use PyTorch only if genuinely necessary for dataset handling or comparison.

Do NOT replace the Brian2 implementation with a PyTorch ANN/SNN framework.

The experiment should remain event-driven and Brian2-based.

Organize the code approximately as:

```text
experiment/
│
├── models/
│   ├── lif.py
│   └── msn.py
│
├── learning/
│   └── stdp.py
│
├── encoding/
│   └── mnist_encoder.py
│
├── datasets/
│   └── mnist.py
│
├── experiments/
│   ├── stdp_sanity.py
│   ├── mnist_small.py
│   ├── mnist_full.py
│   ├── quantization.py
│   └── variability.py
│
├── analysis/
│   ├── receptive_fields.py
│   ├── weights.py
│   └── metrics.py
│
└── results/
```

Adapt this structure to the existing repository rather than forcing a new architecture if the repository already has an established structure.

---

# 20. Reproducibility requirements

Every experiment must specify:

```text
random seed
dataset split
number of samples
presentation time
encoding parameters
STDP parameters
initial weight distribution
neuron parameters
simulation timestep
Brian2 code generation target
```

Do not silently change parameters between experiments.

Store configuration parameters in a reproducible form.

---

# 21. Do not optimize for accuracy too early

The first priority is:

```text
correctness
↓
reproducibility
↓
understanding
↓
performance optimization
```

Do not immediately tune dozens of parameters to maximize MNIST accuracy.

If the MSN achieves lower accuracy than LIF, that is scientifically useful information.

The important question is:

> Why?

Potential explanations should be investigated through:

- firing dynamics;
- spike timing;
- STDP update statistics;
- weight saturation;
- neuron competition;
- membrane-state dynamics;
- device variability.

---

# 22. Final research question

The final experiment should allow us to answer:

> Does replacing a conventional LIF neuron with the memristive spiking neuron produce a measurable computational effect when the learning rule, input encoding, network architecture, and training procedure are held constant?

Then progressively ask:

\[
\boxed{
\text{Does the MSN provide useful robustness or computational behavior under hardware constraints?}
}
\]

Specifically:

```text
MSN
 ↓
STDP
 ↓
quantized weights
 ↓
device variability
 ↓
neuromorphic hardware constraints
```

The most interesting outcome is not necessarily higher MNIST accuracy.

A scientifically interesting result could instead be:

```text
similar accuracy
+
lower spike activity
+
greater robustness to quantization
+
greater tolerance to device variability
```

or some other measurable advantage.

---

# 23. Deliverables

At the end of the experiment, produce:

### Code

- reproducible LIF + STDP baseline;
- MSN + STDP implementation;
- MNIST encoder;
- training script;
- evaluation script;
- quantization experiment;
- device variability experiment.

### Data

Save:

- training accuracy;
- test accuracy;
- spike counts;
- synaptic weights;
- STDP statistics;
- parameter configurations;
- random seeds.

### Figures

At minimum:

1. STDP learning window
2. spike raster
3. receptive fields
4. weight evolution
5. weight distributions
6. accuracy vs dataset size
7. accuracy vs weight precision
8. accuracy vs device variability

### Report

The final report should contain:

```text
1. Objective
2. MSN model
3. STDP implementation
4. MNIST encoding
5. LIF baseline
6. MSN results
7. Quantization results
8. Device variability results
9. Analysis
10. Limitations
11. Conclusions
12. Next experiment: N-MNIST / DVS Gesture
```

Do not claim that the MSN is superior unless the experiments actually demonstrate a measurable advantage.

The primary objective is to establish a rigorous experimental bridge:

\[
\boxed{
\text{device model}
\rightarrow
\text{neuron dynamics}
\rightarrow
\text{STDP}
\rightarrow
\text{network learning}
\rightarrow
\text{hardware constraints}
}
\]

This bridge is more important than obtaining a high MNIST number.