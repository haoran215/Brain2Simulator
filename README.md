# Brain2simulator

A Brian2-based spiking neural network (SNN) simulator for memristor-coupled adaptive leaky integrate-and-fire (aLIF) neurons. Designed for hardware-aware neuromorphic computing research.

## Project structure

```
Brain2simulator/
├── modules/                    # Core simulator library
│   ├── neuron.py               # NeuronPopulation — aLIF model + monitors
│   ├── synapse.py              # SynapseConnection — Brian2 synapses
│   ├── network.py              # SNNNetwork — assembles from JSON config
│   ├── learning.py             # ReservoirReadout, build_stdp_synapses
│   └── regime.py               # RegimeDetector — rate vs temporal coding
├── config/
│   ├── direction_task.json     # 8-direction recognition task config
│   └── default_network.json   # Minimal template to start from
├── tasks/
│   └── direction_recognition.py  # End-to-end direction recognition task
└── run_demo.py                 # Entry point
```

## Installation

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install brian2 scikit-learn numpy matplotlib
```

## Quick start

```bash
python run_demo.py --task direction
```

This runs the 8-direction recognition task and saves a results plot to `direction_recognition.png`.

---

## How to configure the network

Everything is driven by a single JSON file. The structure has four required sections.

### 1. `simulation` — timestep and seed

```json
"simulation": {
  "dt_ms": 0.1,
  "seed": 42
}
```

| Key | Unit | Meaning |
|-----|------|---------|
| `dt_ms` | ms | Brian2 integration timestep. Use 0.05–0.1 ms for tau_m ~10 ms. |
| `seed` | — | RNG seed for reproducible connectivity. |

### 2. `neuron_defaults` — the aLIF model parameters

These apply to **all** aLIF populations unless overridden by `neuron_overrides`.

```json
"neuron_defaults": {
  "Ra_ohm":      2200,
  "Rm_hi_ohm": 100000,
  "Vthresh_V":     4.0,
  "I_min_uA":     40.0,
  "I_max_uA":    100.0,
  "f_min_Hz":     70.0,
  "f_max_Hz":    200.0,
  "I_0_uA":       10.0,
  "tau_s1_ms":    10.0,
  "tau_s2_ms":    10.0,
  "network_mode": "Is2",
  "Iw_exc_uA":    20.0,
  "Iw_inh_uA":    25.0
}
```

| Key | Unit | Meaning |
|-----|------|---------|
| `Ra_ohm` | Ω | Access resistance (series with memristor) |
| `Rm_hi_ohm` | Ω | Memristor resistance at reset (high state) |
| `Vthresh_V` | V | Spike threshold voltage |
| `I_min_uA` | µA | **Measured** drive current that produces `f_min_Hz` |
| `I_max_uA` | µA | **Measured** drive current that produces `f_max_Hz` |
| `f_min_Hz` | Hz | Minimum firing rate at `I_min` — used to **solve** Cm and t_ref |
| `f_max_Hz` | Hz | Maximum firing rate at `I_max` |
| `I_0_uA` | µA | Tonic bias current. 0 = silent without input; >0 = closer to threshold |
| `tau_s1_ms` | ms | Synaptic rise time constant (Is1, exponential kernel) |
| `tau_s2_ms` | ms | Synaptic decay time constant (Is2, alpha kernel) |
| `network_mode` | — | `"Is2"` (alpha fn drives Vm, NMDA-like) or `"Is1"` (exp drives Vm, AMPA-like) |
| `Iw_exc_uA` | µA | Default excitatory weight (can be overridden per connection) |
| `Iw_inh_uA` | µA | Default inhibitory weight |

**Important:** `Cm` (membrane capacitance) and `t_ref` (refractory period) are **solved automatically** from `I_min/f_min` and `I_max/f_max`. You never set them directly — you set the hardware operating points and the solver finds the biophysical parameters.

### 3. `populations` — define neuron groups

Each entry in the list creates one population.

#### Poisson input population

```json
{
  "id": "input",
  "n": 8,
  "type": "poisson",
  "rate_Hz": 20.0
}
```

| Field | Meaning |
|-------|---------|
| `id` | Unique name (used in connections and `set_poisson_rate()`) |
| `n` | Number of neurons |
| `type` | `"poisson"` — fires at independent Poisson-distributed spike times |
| `rate_Hz` | Default firing rate. Change per-trial with `snn.set_poisson_rate(id, hz)` |

#### aLIF reservoir population

```json
{
  "id": "reservoir",
  "n": 4,
  "type": "alif",
  "neuron_overrides": {
    "I_0_uA": 8.0
  }
}
```

| Field | Meaning |
|-------|---------|
| `id` | Unique name |
| `n` | Number of neurons |
| `type` | `"alif"` — memristor adaptive leaky integrate-and-fire |
| `neuron_overrides` | Any key from `neuron_defaults` to override for this population only |

### 4. `connections` — wire populations together

```json
{
  "id": "inp_to_res",
  "pre": "input",
  "post": "reservoir",
  "synapse_type": "exc",
  "connectivity": "all_to_all",
  "Iw_uA": 22.0
}
```

| Field | Values | Meaning |
|-------|--------|---------|
| `id` | string | Unique connection name |
| `pre` | population id | Pre-synaptic population |
| `post` | population id | Post-synaptic population |
| `synapse_type` | `"exc"` / `"inh"` | Excitatory injects into Is1_exc; inhibitory into Is1_inh |
| `connectivity` | see below | How pre and post are wired |
| `Iw_uA` | µA | Synaptic weight (spike current amplitude) |
| `allow_self` | bool | Whether a neuron can connect to itself (default `false`) |

**Connectivity options:**

| Value | Extra keys | Meaning |
|-------|-----------|---------|
| `"all_to_all"` | — | Every pre neuron connects to every post neuron |
| `"random"` | `p_connect` | Each pair connects with probability p |
| `"one_to_one"` | — | pre[i] → post[i] (requires equal population sizes) |
| `"fixed_in"` | `k_in` | Each post neuron receives exactly k_in random connections |

---

## Using SNNNetwork in Python

```python
from modules.network import SNNNetwork

snn = SNNNetwork('config/direction_task.json')
snn.summary()

# Change a Poisson population's rate
snn.set_poisson_rate('dir0', 100.0)   # Hz

# Advance simulation by 150 ms
snn.run(0.150)   # duration in seconds

# Read current simulation time
t_now = float(snn.net.t / second)

# Get spike data for a population (all spikes since t=0)
spike_i, spike_t = snn.get_spikes('reservoir')
# spike_i: neuron index array
# spike_t: spike time array (seconds)

# Access population metadata
n_res = snn.populations['reservoir'].n
```

---

## Tutorial: 8-Direction Recognition

This task demonstrates **rate-coding reservoir computing**: the network learns to distinguish 8 movement directions (0°, 45°, …, 315°) using fixed random reservoir weights and a trained linear readout.

### Step 1 — Understand the coding regime

The key parameter is `tau_s_ms` (synaptic time constant). It sets a crossover frequency:

```
f_cross = 1000 / tau_s_ms  Hz
```

- **Rate regime** (`f > f_cross`): synaptic pulses overlap → Is2 carries a smooth DC current → the mean current encodes firing rate
- **Temporal regime** (`f < f_cross`): pulses are resolved individually → spike timing matters → use STDP

With `tau_s_ms = 10 ms`, the crossover is at **100 Hz**. The direction task uses rates of 20–100 Hz, straddling both regimes but biased toward rate coding.

### Step 2 — Direction encoding

8 Poisson neurons, each tuned to one preferred direction θᵢ = 0°, 45°, …, 315°.

For a stimulus at angle θ, neuron i fires at:

```
r_i(θ) = r_base + r_mod × max(0, cos(θ − θᵢ))
```

- **Aligned neuron** (θ = θᵢ): fires at `r_base + r_mod` = 100 Hz
- **Opposite neuron** (θ = θᵢ + 180°): fires at baseline `r_base` = 20 Hz
- **Orthogonal neurons**: intermediate rates

This creates a smooth "population vector" that points in the stimulus direction.

### Step 3 — Reservoir transformation

The 8 input neurons project to 4 aLIF reservoir neurons via random fixed weights. Each direction produces a **unique pattern** of Is2 (mean synaptic current) across the 4 reservoir neurons because each neuron sees a different weighted sum of inputs.

The reservoir also has sparse recurrent excitation and inhibition, which amplifies and mixes the input patterns, creating richer representations.

### Step 4 — Readout and classification

After each 150 ms trial:

1. Count spikes of each reservoir neuron in the readout window (20–150 ms, skipping the 20 ms transient)
2. Stack spike counts into a feature vector: `x ∈ R^4`
3. After collecting all trials, train **multinomial logistic regression** on the feature matrix

```python
from modules.learning import ReservoirReadout

readout = ReservoirReadout(n_classes=8)
train_acc = readout.train(X_train, y_train)   # fit scaler + logistic reg
test_acc  = readout.score(X_test, y_test)
cm        = readout.confusion_matrix(X_test, y_test)
```

### Step 5 — Run it

```bash
python run_demo.py --task direction
```

Expected output:
```
Regime Analysis   tau_s = 10.0 ms   f_cross = 100 Hz
...
Building network from config/direction_task.json ...
Running 8 directions × 20 trials (160 total) ...
  Direction     0°  mean spikes/neuron = 8.3
  Direction    45°  mean spikes/neuron = 7.1
  ...
  Train accuracy : 98.3%
  Test  accuracy : 87.5%
```

The plot shows tuning curves (A), spike rasters (B), feature heatmap (C), PCA of reservoir states (D), confusion matrix (E), per-direction accuracy (F), and the regime diagram (G).

---

## Adding a new task

1. Create `config/my_task.json` — copy `config/default_network.json` and add your populations, connections, and a `"task"` section.
2. Create `tasks/my_task.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.network  import SNNNetwork
from modules.learning import ReservoirReadout

def run_my_task(config_path, plot_path):
    snn = SNNNetwork(config_path)

    X_all, y_all = [], []
    for label, stimulus in enumerate(my_stimuli):
        snn.set_poisson_rate('input', stimulus_rate)
        t0 = float(snn.net.t / second)
        snn.run(0.200)   # 200 ms trial
        spike_i, spike_t = snn.get_spikes('reservoir')
        features = ReservoirReadout.extract_spike_counts(
            spike_i, spike_t, n_neurons=snn.populations['reservoir'].n,
            t_start_s=t0, t_end_s=t0 + 0.200)
        X_all.append(features)
        y_all.append(label)

    readout = ReservoirReadout(n_classes=n_classes)
    readout.train(X_train, y_train)
    print(f"Test accuracy: {readout.score(X_test, y_test)*100:.1f}%")
```

3. Add your task to `run_demo.py`:

```python
elif task == 'my_task':
    from tasks.my_task import run_my_task
    run_my_task(os.path.join('config', 'my_task.json'), 'my_task_results.png')
```
