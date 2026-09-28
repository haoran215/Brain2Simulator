# MSN Simulator: Quick Walkthrough

A short, practical guide to building your own network with the MSN (Memristive Spiking Neuron) library.
For the physics and derivations, see [`README.md`](../README.md). This page covers only **how to use it**.

---

## 0. The whole picture in 7 steps

Every simulation, from 1 neuron to 1000, follows the same recipe:

```
1. Import            → brian2 + the 2 library modules
2. Settings          → backend, time step, random seed
3. Neuron parameters → load a JSON config (the hardware values)
4. Neurons           → make_msn(N, ...)          ← how many neurons
5. Drive             → I_0 (constant bias) and/or input spikes
6. Synapses          → make_synapse(pre, post, ...)  ← type + wiring
7. Monitor + run     → SpikeMonitor / StateMonitor, then run()
```

The two library functions do most of the work:

| Function | File | Builds |
|---|---|---|
| `make_msn(N, params, exc_inlets, inh_inlets, name)` | `msn_neuron.py` | a group of `N` MSN neurons |
| `make_synapse(source, target, params, connect, name)` | `msn_synapse.py` | all the connections from one group to another |

---

## 1. Import

Run scripts **from the repository root** so that `msn_neuron.py` and `configs/` can be found:

```bash
cd Brain2Simulator
uv run python my_experiment.py      # or: .venv/bin/python my_experiment.py
```

```python
import numpy as np
from brian2 import *                                  # NeuronGroup, run, ms, amp, Hz, ...
from msn_neuron      import MSNParams, make_msn
from msn_synapse     import SynapseParams, make_synapse
from msn_variability import apply_variability         # optional: device-to-device spread
```

> If your script lives in `demo/`, add the repository root to the path first (as the demos do):
> ```python
> import os, sys
> sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
> ```

## 2. Settings

```python
prefs.codegen.target = 'numpy'   # works without a C++ compiler; use 'cython' for speed on big networks
defaultclock.dt = 10*us          # 10–20 µs is fine for networks; 1 µs for fine spike-shape plots
seed(42)                         # reproducible Poisson input / random wiring
```

## 3. Neuron parameters

The neuron's hardware values live in a JSON file:

```python
p = MSNParams.from_json('configs/neuron_default.json')   # calibrated to 35 real devices
print(p.summary())
I_min, I_max = p.operating_window()                       # ≈ 32 µA, 77 µA
```

**`I_min` and `I_max` are the two numbers you need most.** They set every current you choose later:

| Bias current `I_0` | What the neuron does |
|---|---|
| `I_0 < I_min` | silent. It fires only if synapses push it over `I_min` |
| `I_min < I_0 < I_max` | fires on its own (the higher `I_0`, the faster it fires) |
| `I_0 > I_max` | locked on (depolarisation block): it stops spiking |

Available configs:

| File | Use |
|---|---|
| `configs/neuron_default.json` | standard choice (Dec 2025 hardware calibration) |
| `configs/neuron_hardware_fit.json` | fitted to one measured device (F–I curve + spike shape) |
| `configs/E_neuron.json`, `I_neuron.json` | slower paper-style values used by `two_E_one_I.py` |

To change one value: `p = MSNParams.from_json(...); p.Cm = 200e-9` (all values in SI units: F, Ω, V, A).

## 4. Neurons: how many, and their "inlets"

```python
E = make_msn(N=10, params=p, name='E')    # 10 neurons, one group
```

- **`N`** sets the number of neurons in the group. Neurons in a group are indexed `0 … N-1`.
- **`name`** must be unique within the script.
- Use **several groups** when neurons play different roles (E vs I, layer 1 vs layer 2). Use **one group** when they are the same kind of neuron. It is easier to connect within one group with an index rule.

### Inlets (read this, it is the one rule that trips people up)

Each neuron receives synaptic current through named **inlets**. By default a group has one excitatory inlet `I_exc` and one inhibitory inlet `I_inh`:

```
dVm/dt = (I_0 + I_exc − I_inh − Vm/(Rm+Ra)) / Cm
```

**Rule: each `make_synapse` call must write to its own inlet on the target group.**
So if a group receives **two or more** excitatory pathways (for example, external input *and* recurrent excitation), give it more inlets:

```python
E = make_msn(N=10, params=p,
             exc_inlets=('I_exc_in', 'I_exc_rec'),   # 2 excitatory pathways
             inh_inlets=('I_inh',),                  # 1 inhibitory pathway
             name='E')
```

and point each synapse to one of them with `target_var=` (see §6). If you break this rule, `run()` stops with:
`NotImplementedError: Multiple 'summed variables' target the variable 'I_exc' in group ...`

Quick check: **count the arrows coming *into* each group. Each arrow needs its own inlet.**

## 5. Drive: making neurons active

There are three ways to drive neurons. Mix them freely.

**(a) Constant bias:** set after `make_msn`, per neuron if you like:

```python
E.I_0 = 0.8 * I_min * amp                        # all neurons: just below threshold
E.I_0 = np.array([40, 0, 0, 0]) * 1e-6 * amp     # per-neuron (N=4): only neuron 0 fires on its own
```

**(b) Time-varying bias (pulse / step):** a `network_operation` runs every time step:

```python
@network_operation(when='start')
def pulse(t):
    E.I_0[0] = (1.5*I_min if 1*second <= t < 1.2*second else 0.8*I_min) * amp
```

**(c) Input spikes:** a Poisson source (or `SpikeGeneratorGroup` for exact times), connected with a synapse:

```python
inp  = PoissonGroup(10, rates=100*Hz, name='inp')
s_in = make_synapse(inp, E,
                    SynapseParams(kind='exc', weight=3e-6, tau_s1=20e-3, tau_s2=20e-3,
                                  target_var='I_exc_in'),
                    connect='i == j', name='in_to_E')     # input k → neuron k
```

**(d) Device variability (optional):** give each neuron its own threshold and holding current, as in real hardware:

```python
apply_variability(E, seed=1, scale=0.5)   # scale: 0 = identical devices, 1 = full measured spread
```

## 6. Synapses: type, strength, speed, wiring

```python
syn = make_synapse(source, target, params=SynapseParams(...), connect=..., name='...')
```

### 6.1 Choose the synapse type (`SynapseParams`)

| Field | Meaning | Typical values |
|---|---|---|
| `kind` | `'exc'` (adds current) or `'inh'` (subtracts current) | |
| `weight` | current kick per presynaptic spike, **always positive**, in A | 1–20 µA (`1e-6`–`20e-6`) |
| `tau_s1`, `tau_s2` | how fast the current rises and decays, in s | fast 5–20 ms · medium 100–200 ms · slow 500 ms |
| `cascade` | `'alpha'` (smooth rise then decay, the default) or `'exp'` (instant jump then decay, uses only `tau_s1`) | |
| `delay` | transmission delay, in s | 0 |
| `target_var` | which inlet on the target to write into | `None` → `I_exc` / `I_inh` |

Or load a ready-made type from `configs/`:

```python
exc = SynapseParams.from_json('configs/synapse_default.json', key='exc')   # 6 µA, 200 ms
inh = SynapseParams.from_json('configs/synapse_default.json', key='inh')   # 10 µA, 200 ms
e2i = SynapseParams.from_json('configs/syn_E_to_I.json')                   # single-type file: no key
```

"Inhibitory neuron" is not a neuron property. A neuron is inhibitory **because its outgoing synapses have `kind='inh'`**.

### 6.2 Choose the wiring (`connect=`)

`i` = index of the presynaptic neuron, `j` = index of the postsynaptic neuron.

| `connect=` | Wiring | Example use |
|---|---|---|
| `True` | all-to-all (includes self-loops if source is target) | E group → single I neuron |
| `'i != j'` | all-to-all, no self-loops | recurrent excitation |
| `'i == j'` | one-to-one (a **self-loop** if source is target) | input channel k → neuron k; self-excitation |
| `'i == j and i == 0'` | self-loop on neuron 0 only | pacemaker (`ns_msn_pacemaker.py`) |
| `'j == i + 1'` | chain 0→1→2→… | feed-forward chain |
| `'rand() < 0.2'` | random, 20% probability | reservoir |
| `'abs(i-j) == 1'` | nearest neighbours | ring / line (`ns_msn_v4_network.py`) |

The number of synapses created is `len(syn)`.

### 6.3 Change weights after creation (optional)

Every edge has its own weight `syn.w`:

```python
syn.w = np.random.gamma(2.0, 1.5e-6, size=len(syn)) * amp   # random heterogeneous weights
syn.w['i < 5'] = 8e-6 * amp                                   # only edges from neurons 0–4
```

> ⚠️ **Always store the result of `make_synapse` in a variable** (`s_EI = make_synapse(...)`).
> If you write `make_synapse(...)` on its own line, Brian2's `run()` silently leaves that synapse out of the simulation. (You'll see an `unused_brian_object` warning.)

## 7. Monitor, run, look

```python
spE = SpikeMonitor(E)                                        # spike times of every neuron
stE = StateMonitor(E, ['Vm', 'I_exc_in', 'I_inh'],           # any neuron variable or inlet
                   record=[0, 1], dt=1*ms)                   # which neurons, how often
stS = StateMonitor(s_in, ['Is2'], record=True, dt=1*ms)      # synaptic current per edge

run(2*second, report='text')

print('rates (Hz):', spE.count / (2*second))
plot(spE.t/second, spE.i, '|'); xlabel('t (s)'); ylabel('neuron'); show()
```

Useful neuron variables: `Vm` (membrane voltage), `Vout` (measured output spike), `s` (memristor state 0/1), `I_0`, and every inlet name you declared.

Keep `StateMonitor` `dt` at 1–2 ms for long runs. Recording every 10 µs step fills memory fast.

---

## 8. Complete minimal example (E–I network, copy and run)

Four excitatory neurons with Poisson input and recurrent excitation, plus one inhibitory neuron that feeds back to all of them.

```
 Poisson(4) ──exc──► E(4) ──exc──► I(1)
                      ▲ │  ◄──inh───┘
                      └─┘ exc (i != j)
```

```python
import numpy as np
from brian2 import *
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

prefs.codegen.target = 'numpy'
defaultclock.dt = 10*us
seed(42)

# --- neurons --------------------------------------------------------------
p = MSNParams.from_json('configs/neuron_default.json')
I_min, I_max = p.operating_window()

# E gets 2 excitatory pathways (input + recurrent) → 2 exc inlets
E = make_msn(N=4, params=p, exc_inlets=('I_exc_in', 'I_exc_rec'), name='E')
I = make_msn(N=1, params=p, name='I')                 # default inlets I_exc, I_inh
E.I_0 = 0.8 * I_min * amp                             # silent without input
I.I_0 = 0.5 * I_min * amp

# --- input ------------------------------------------------------------------
inp = PoissonGroup(4, rates=100*Hz, name='inp')

# --- synapses (4 arrows → 4 make_synapse calls) ----------------------------------
s_in = make_synapse(inp, E, SynapseParams(kind='exc', weight=3e-6, tau_s1=20e-3, tau_s2=20e-3,
                                          target_var='I_exc_in'),
                    connect='i == j', name='in_to_E')
s_EE = make_synapse(E, E, SynapseParams(kind='exc', weight=1e-6, tau_s1=50e-3, tau_s2=50e-3,
                                        target_var='I_exc_rec'),
                    connect='i != j', name='E_to_E')
s_EI = make_synapse(E, I, SynapseParams.from_json('configs/synapse_default.json', key='exc'),
                    connect=True, name='E_to_I')      # writes I.I_exc
s_IE = make_synapse(I, E, SynapseParams.from_json('configs/synapse_default.json', key='inh'),
                    connect=True, name='I_to_E')      # writes E.I_inh

# --- run ---------------------------------------------------------------------
spE, spI = SpikeMonitor(E), SpikeMonitor(I)
run(1*second, report='text')
print('E rates:', spE.count / second, ' I spikes:', spI.count[0])
```

Expected output: `E rates: [ 7.  7.  5. 10.] Hz  I spikes: 96`.

---

## 9. Recipes for common motifs

All recipes assume §1–3 are done (`p`, `I_min` defined). Each shows only the neurons + synapses; add a drive, monitors and `run()` as in §8.

### 9.1 Self-excitation (bump / pacemaker): `demo/ns_msn_v3_bump.py`, `demo/ns_msn_pacemaker.py`

```python
N0 = make_msn(N=1, params=p, name='N0')
N0.I_0 = 0.75 * I_min * amp                                    # subthreshold; kick it with a pulse (§5b)
s_self = make_synapse(N0, N0, SynapseParams(kind='exc', weight=10e-6, tau_s1=0.2, tau_s2=0.2),
                      connect='i == j', name='self_exc')
```
To get rhythmic bursting, add slow self-inhibition. That is a second pathway, so it needs its own inlet (here the default `I_inh`).

### 9.2 Feed-forward chain: `demo/ns_msn_v5_delay_chain.py`

```python
C = make_msn(N=5, params=p, name='chain')
C.I_0 = 0.8 * I_min * amp
C.I_0[0] = 1.2 * I_min * amp                                  # first neuron fires on its own
s_ff = make_synapse(C, C, SynapseParams(kind='exc', weight=8e-6, tau_s1=20e-3, tau_s2=20e-3),
                    connect='j == i + 1', name='ff')          # 0→1→2→3→4
```
At 6 µA the activity dies out by neuron 4. At 15 µA downstream neurons fire too fast and fall into depolarisation block. 8 µA propagates at about 90 Hz.

### 9.3 Winner-take-all (mutual inhibition): `demo/ns_msn_wta_demo.py`

```python
W = make_msn(N=2, params=p, name='wta')
W.I_0 = np.array([1.3, 1.6]) * I_min * amp                    # neuron 1 gets the stronger drive → wins
s_mut = make_synapse(W, W, SynapseParams(kind='inh', weight=2e-6, tau_s1=0.2, tau_s2=0.2),
                     connect='i != j', name='mutual_inh')
```
For N competitors, the same `'i != j'` line gives all-to-all mutual inhibition.

### 9.4 Half-centre oscillator / CPG (mutual + self inhibition): `demo/ns_msn_cpg_demo.py`

Two inhibitory pathways into each group → two inh inlets:

```python
A = make_msn(N=1, params=p, inh_inlets=('I_inh_mut', 'I_inh_self'), name='A')
B = make_msn(N=1, params=p, inh_inlets=('I_inh_mut', 'I_inh_self'), name='B')
A.I_0 = B.I_0 = 40e-6 * amp                                   # both above I_min
A.Vm  = 0.9 * volt                                            # break the symmetry, else they fire in lockstep
mut  = SynapseParams(kind='inh', weight=6e-6,   tau_s1=0.05, tau_s2=0.05, target_var='I_inh_mut')   # fast, strong
slf  = SynapseParams(kind='inh', weight=0.1e-6, tau_s1=3.0,  tau_s2=3.0,  target_var='I_inh_self')  # slow, weak
s_AB = make_synapse(A, B, mut, connect=True, name='A_to_B')
s_BA = make_synapse(B, A, mut, connect=True, name='B_to_A')
s_AA = make_synapse(A, A, slf, connect=True, name='A_to_A')   # adaptation → hands over control
s_BB = make_synapse(B, B, slf, connect=True, name='B_to_B')
```
Result: A and B take turns, each firing for about 3 s while the other is silent. The slow self-inhibition sets the period.

### 9.5 Ring / working memory: `demo/ns_msn_v4_network.py`

```python
N = 20
R = make_msn(N=N, params=p, name='ring')
R.I_0 = 0.85 * I_min * amp
ring = f'i != j and (abs(i-j) <= 2 or abs(i-j) >= {N-2})'    # neighbours ±1, ±2 with wrap-around
s_ring = make_synapse(R, R, SynapseParams(kind='exc', weight=2e-6, tau_s1=0.5, tau_s2=0.5),
                      connect=ring, name='ring_exc')
```
The ring is silent until you cue a few neurons (e.g. raise `R.I_0[9:11]` for 50 ms with a `network_operation`, §5b).

### 9.6 Reservoir (random recurrent network + inputs): `demo/ns_msn_rc_demo.py`, `ns_msn_rc_ei_demo.py`

```python
R = make_msn(N=20, params=p, exc_inlets=('I_exc_rec', 'I_exc_L', 'I_exc_R'), name='res')
R.I_0 = 0.7 * I_min * amp
apply_variability(R, seed=42, scale=0.5)
s_rec = make_synapse(R, R, SynapseParams(kind='exc', target_var='I_exc_rec'),
                     connect='i != j and rand() < 0.2', name='rec')
s_rec.w = np.random.gamma(2.0, 0.15e-6, size=len(s_rec)) * amp
L = PoissonGroup(10, rates=0*Hz, name='L')
Rr = PoissonGroup(10, rates=0*Hz, name='Rr')
s_L = make_synapse(L,  R, SynapseParams(kind='exc', weight=5e-6, target_var='I_exc_L'),
                   connect='j == i', name='in_L')             # L[k] → R[k]
s_R = make_synapse(Rr, R, SynapseParams(kind='exc', weight=5e-6, target_var='I_exc_R'),
                   connect='j == i + 10', name='in_R')        # Rr[k] → R[k+10]
# set L.rates / Rr.rates between run() calls to present different stimuli
```

---

## 10. Tuning cheat sheet

Most "nothing happens" and "everything is on" problems are fixed with these three rules:

1. **Place the bias relative to `I_min`.** `I_0 = 0.7–0.9 × I_min` gives a neuron that waits for input. `1.1–2 × I_min` gives a neuron that fires on its own. Never exceed `I_max`, where the neuron locks on.
2. **One spike can trigger the next neuron if** `weight > e × (I_min − I_0)` ≈ `2.7 × (I_min − I_0)`
   (with the alpha synapse, one spike delivers a peak current of `weight / e`).
3. **Sustained input from firing at rate `f`** adds about `weight × f × tau_s` to the target. If that exceeds `I_min − I_0`, recurrent excitation keeps itself going. If it's below, activity fades.

Other rules of thumb:

- More incoming edges per neuron → use a smaller weight per edge (e.g. 4 inputs → about ¼ of the weight).
- Choose `tau_s` close to the inter-spike interval you expect. Much shorter gives brief blips; much longer gives a steady DC offset.
- Slow synapses need time to settle: about 5 × `tau_s` (1 s for 200 ms). Run long enough.

## 11. Common mistakes

| Symptom | Cause / fix |
|---|---|
| `NotImplementedError: Multiple 'summed variables' target ...` | Two synapses write the same inlet → add inlets in `make_msn(..., exc_inlets=...)` and set `target_var` |
| A synapse seems to have no effect; `unused_brian_object` warning | You didn't assign `make_synapse(...)` to a variable |
| Error about a duplicate object name | Every `make_msn` / `make_synapse` needs a unique `name=` |
| `FileNotFoundError: configs/...` | Run from the repo root, or build the path with `os.path.join(repo_root, 'configs', ...)` |
| No spikes at all | `I_0` too low and weights too small; check with §10 rules 1–2 |
| Neuron fires, then goes silent while input is still high (`s` stays 1) | Total current > `I_max` (depolarisation block): lower `I_0` or excitatory weights |
| Very slow | `dt` too small, or recording every step; install `build-essential` and use `prefs.codegen.target = 'cython'` |
| Negative `weight` for inhibition | Don't: use `kind='inh'` with a positive weight |
