# Porting Guide — cascade-on-synapse refactor

This branch (`STDP-wip`) was written before the synaptic cascade was moved
from the neuron to the synapse. The core library
([`msn_neuron.py`](msn_neuron.py), [`msn_synapse.py`](msn_synapse.py)) is
now on the new architecture, but the scripts under
[`Classification-STDP/`](Classification-STDP/) and
[`Classification-supervised/`](Classification-supervised/) have not been
ported yet — each carries a `NEEDS PORT` banner.

This document is the recipe for porting them.

---

## 1. What changed and why

**Before.** The Is1 → Is2 synaptic cascade lived on the postsynaptic
neuron. `Is1_exc`, `Is2_exc`, `Is1_inh`, `Is2_inh` were state variables
on every `NeuronGroup`. `tau_s1` and `tau_s2` were namespace constants
on the neuron, so *all incoming synapses to a given neuron shared the
same kinetics*. Multiple `Synapses` objects could fan into a neuron with
`on_pre='Is1_exc_post += w'`, each adding to the same `Is1_exc`.

**After.** The cascade lives on the `Synapses` object. Each synapse type
carries its own `tau_s1`, `tau_s2` (per AMPA/NMDA/GABA-A receptor-style
kinetics). The neuron exposes summed *inlets* — by default `I_exc` and
`I_inh` — and each `Synapses` writes to one inlet via the Brian2
`(summed)` mechanism.

This lets you put **different time constants on different pathways
converging on the same neuron** (the whole point of the refactor) — for
example fast E→I excitation and slow mutual E↔E excitation onto the
same E neuron.

**The catch.** Brian2 allows only *one* `Synapses` object to write to a
given `(summed)` target. When two pathways of the same kind converge on
one neuron group, you must declare two distinct inlets at construction
and route each `Synapses` to its own.

---

## 2. Symbol map

| Old (cascade on neuron) | New (cascade on synapse) |
|---|---|
| `MSNParams(tau_s1=X, tau_s2=Y)` | `MSNParams()` + `SynapseParams(tau_s1=X, tau_s2=Y)` |
| `make_msn(N=N, params=params)` | `make_msn(params=params, N=N)` (or list of params) |
| neuron state var `Is1_exc` | per-edge synapse var `Is1` (on the `Synapses` object) |
| neuron state var `Is2_exc` | per-edge synapse var `Is2`, summed into neuron `I_exc` |
| neuron state var `Is2_inh` | per-edge `Is2` on inh `Synapses`, summed into `I_inh` |
| `on_pre='Is1_exc_post += w'` | use `make_synapse(...)` from `msn_synapse` |
| `StateMonitor(G, 'Is2_exc')` | `StateMonitor(G, 'I_exc')` (total) **or** `StateMonitor(syn, 'Is2')` (per-edge) |

---

## 3. The three porting patterns

### Pattern A — homogeneous synapses, one kind per target

This is the simplest case: one exc Synapses group and at most one inh
Synapses group target each neuron group. The default inlets (`I_exc`,
`I_inh`) are enough.

**Old:**
```python
from msn_neuron import MSNParams, make_msn
params = MSNParams(tau_s1=0.047, tau_s2=0.047)
G_E    = make_msn(N=100, params=params, name='E')
G_I    = make_msn(N=25,  params=params, name='I')
syn_e_i = Synapses(G_E, G_I, on_pre=f'Is1_exc_post += {w_e2i}*amp', name='syn_e_i')
syn_e_i.connect()
syn_i_e = Synapses(G_I, G_E, on_pre=f'Is1_inh_post += {w_i2e}*amp', name='syn_i_e')
syn_i_e.connect()
```

**New:**
```python
from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse
n_params = MSNParams()                                            # tau moved out
G_E = make_msn(params=n_params, N=100, name='E')
G_I = make_msn(params=n_params, N=25,  name='I')

p_e2i = SynapseParams(weight=w_e2i, kind='exc',
                      tau_s1=0.047, tau_s2=0.047)
p_i2e = SynapseParams(weight=w_i2e, kind='inh',
                      tau_s1=0.047, tau_s2=0.047)

syn_e_i = make_synapse(G_E, G_I, params=p_e2i, connect=True, name='syn_e_i')
syn_i_e = make_synapse(G_I, G_E, params=p_i2e, connect=True, name='syn_i_e')
```

### Pattern B — multiple Synapses of the same kind on one target

This breaks the `(summed)` rule. Solution: declare extra inlets on the
target via `make_msn(exc_inlets=..., inh_inlets=...)` and use
`SynapseParams.target_var` to route each Synapses to its own inlet.

**Old (input layer + recurrent E→I — both exc into G_E):**
```python
syn_in  = Synapses(input_layer, G_E, on_pre='Is1_exc_post += w', name='syn_in')
syn_e_e = Synapses(G_E,         G_E, on_pre=f'Is1_exc_post += {w_rec}*amp',
                   name='syn_e_e')
```

**New:**
```python
# Declare TWO exc inlets on G_E
G_E = make_msn(params=n_params, N=N_E, name='E',
               exc_inlets=('I_exc_input', 'I_exc_recur'))

p_in  = SynapseParams(weight=w_in,  kind='exc',
                      tau_s1=0.047, tau_s2=0.047,
                      target_var='I_exc_input')
p_rec = SynapseParams(weight=w_rec, kind='exc',
                      tau_s1=0.500, tau_s2=0.500,             # slow recurrent
                      target_var='I_exc_recur')

syn_in  = make_synapse(input_layer, G_E, params=p_in,  connect='i==j',
                       name='syn_in')
syn_e_e = make_synapse(G_E,         G_E, params=p_rec, connect='i!=j',
                       name='syn_e_e')
```

The Vm ODE on G_E automatically uses `I_exc = I_exc_input + I_exc_recur`.

### Pattern C — STDP / custom on_pre rules

The cascade equations must live in the `model` block of your custom
`Synapses`, alongside your STDP trace variables. The `on_pre` adds the
spike kick to **`Is1`** (on the synapse), not `Is1_exc_post` (no longer
exists). The summed assignment writes `Is2` to the target inlet.

**Old (STDP synapse from `train_stdp.py`):**
```python
syn = Synapses(
    source, G_E,
    model = '''
        w        : amp
        dApre/dt  = -Apre  / tau_pre  : 1 (event-driven)
        dApost/dt = -Apost / tau_post : 1 (event-driven)
    ''',
    on_pre  = '''
        Is1_exc_post += w * w_unit
        Apre  += 1
        w      = clip(w - Apost * lr_minus, 0*amp, w_max)
    ''',
    on_post = '''
        Apost += 1
        w      = clip(w + Apre  * lr_plus,  0*amp, w_max)
    ''',
)
```

**New:**
```python
# Make sure G_E declares an inlet for this pathway (e.g. 'I_exc_input').
# If only one exc pathway targets G_E, the default 'I_exc' is fine.

syn = Synapses(
    source, G_E,
    model = '''
        dIs1/dt   = -Is1 / tau_s1                : amp (clock-driven)
        dIs2/dt   = (-Is2 + Is1) / tau_s2        : amp (clock-driven)
        I_exc_input_post = Is2                   : amp (summed)
        w         : amp
        dApre/dt  = -Apre  / tau_pre   : 1 (event-driven)
        dApost/dt = -Apost / tau_post  : 1 (event-driven)
    ''',
    on_pre  = '''
        Is1   += w * w_unit
        Apre  += 1
        w      = clip(w - Apost * lr_minus, 0*amp, w_max)
    ''',
    on_post = '''
        Apost += 1
        w      = clip(w + Apre  * lr_plus,  0*amp, w_max)
    ''',
    namespace = {
        'tau_s1':   0.047*second,
        'tau_s2':   0.047*second,
        'tau_pre':  20*ms, 'tau_post': 20*ms,
        'lr_plus':  0.3e-6*amp, 'lr_minus': 0.3e-6*amp,
        'w_max':    15e-6*amp, 'w_unit': 1*amp,
    },
)
syn.connect(...)
syn.w = ...
syn.Is1 = 0*amp
syn.Is2 = 0*amp
```

Two changes from the old form:
1. The cascade ODE moves into the synapse's `model` block.
2. `on_pre` adds to `Is1` (the synapse's own state), not `Is1_exc_post`.
3. The `(summed)` line writes `Is2` to the target inlet on the post neuron.

If you have several STDP synapses converging on the same `G_E`, pick the
same `target_var` for all of them only if they should share kinetics;
otherwise give each its own inlet (Pattern B) and route to it.

---

## 4. Per-file porting plan for this branch

| File | Pattern | Notes |
|---|---|---|
| [`Classification-STDP/pavlov_demo.py`](Classification-STDP/pavlov_demo.py) | C | Single STDP synapse onto G_E, plus a US synapse. If both are exc into G_E, declare two exc inlets and route. |
| [`Classification-STDP/train_stdp.py`](Classification-STDP/train_stdp.py) | B + C | Input STDP synapse + recurrent E→I (exc) onto G_I; plus I→E (inh). Need two exc inlets on G_I if STDP plus E→I both hit it. |
| [`Classification-STDP/train_bsf.py`](Classification-STDP/train_bsf.py) | B | Similar to train_stdp but rate-coded (no STDP traces). Same multi-inlet treatment. |
| [`Classification-STDP/eval_stdp.py`](Classification-STDP/eval_stdp.py) | A or B | Mirrors `train_stdp.py` structure but without learning. |
| [`Classification-STDP/learning_curve.py`](Classification-STDP/learning_curve.py) | A or B | Wraps `train_stdp.py` / `eval_stdp.py`. |
| [`Classification-STDP/plot_schematic.py`](Classification-STDP/plot_schematic.py) | doc-only | Just changes `Is1_exc` labels in the figure annotations to `Is1`/`I_exc`. |
| [`Classification-supervised/eval_msn_brian2.py`](Classification-supervised/eval_msn_brian2.py) | B | 784 Poisson → 100 MSN with both exc (W>0) and inh (W<0) routed by sign. Pure Pattern A if only one exc and one inh group; Pattern B if you also have recurrent excitation on the MSN layer. |
| [`Classification-supervised/plot_raster.py`](Classification-supervised/plot_raster.py) | trivial | Only `MSNParams(tau_s1=..., tau_s2=...)` — drop those args, the constructor now ignores them. |

Each script's banner pointed here. Remove the banner once ported.

---

## 5. Recording / debugging

The old `StateMonitor(G, 'Is2_exc')` no longer works. Use one of:

- `StateMonitor(G, 'I_exc')` — the total summed exc current on the neuron.
- `StateMonitor(G, ['I_exc_input', 'I_exc_recur'])` — per-inlet contributions when multi-inlet.
- `StateMonitor(syn, 'Is2')` — per-edge cascade Is2 on a `Synapses` object.

---

## 6. Smoke test after porting

For each ported script:

```bash
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', 'Classification-STDP/train_stdp.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
"
```

A successful import means no symbol-level breakage. A successful short
run (set `--T 0.5` or similar) means the network builds and Brian2's
summed checks pass.

---

## 7. Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'Is1_exc_post'` in `on_pre` | reference to removed neuron state var | switch to `make_synapse` or move cascade into the synapse `model` (Pattern C) |
| `'I_exc' is set by more than one summed Synapses object` | two exc pathways into the same inlet | declare extra inlets via `make_msn(exc_inlets=...)` and route via `SynapseParams.target_var` |
| `MSNParams.__init__() got an unexpected keyword argument 'tau_s1'` | unlikely — `from_json` strips legacy keys, but constructor is strict | drop `tau_s1`/`tau_s2` from `MSNParams(...)` calls; put them on `SynapseParams` |
| `AttributeError: 'NeuronGroup' object has no attribute 'Is2_exc'` | recording removed state var | use `I_exc` (neuron total) or `syn.Is2` (per-edge) |
