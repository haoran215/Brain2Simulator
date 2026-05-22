# STDP in the MSN Reservoir — Failure Analysis and E-I Fix

**Date:** 2026-05-22
**Demos:** [`demo/ns_msn_rc_demo.py`](../demo/ns_msn_rc_demo.py) (original),
[`demo/ns_msn_rc_ei_demo.py`](../demo/ns_msn_rc_ei_demo.py) (E-I follow-up)
**Modules used:** [`msn_neuron.py`](../msn_neuron.py),
[`msn_synapse.py`](../msn_synapse.py),
[`msn_variability.py`](../msn_variability.py)

---

## 1. The question

The reservoir-computing demo classifies a *left* vs *right* Poisson stimulus
with 20 MSN neurons and a ridge-regression readout. The recurrent weights can
be either **frozen random** (standard RC) or **plastic via STDP**. The question
was simply: *does the network still work when STDP is turned on?*

The short answer turned out to be **no — for two separate reasons** — and
fixing it cleanly required adding the **inhibitory** half of the network that
was missing.

---

## 2. Problem 1 — STDP crashed before it ran

With `USE_STDP = True`, the original demo raised:

```
brian2.core.base.BrianObjectException: Error encountered with object named 'syn_rec'.
ZeroDivisionError: float division
```

### Root cause

The plastic branch builds the `Synapses` object by hand and **does not specify
an integration method**, so Brian2 auto-selected `'exact'`. The exact
integrator symbolically solves the two-stage synaptic cascade

```
dIs1/dt = -Is1 / tau_s1
dIs2/dt = (-Is2 + Is1) / tau_s2
```

whose closed-form solution contains a `1 / (tau_s2 - tau_s1)` term. In the
config both time constants are equal (`tau_s1 = tau_s2 = 0.2 s`), so that
denominator is **zero**.

The *static* branch never hit this because it goes through
`make_synapse()`, which passes `method='euler'`
([`msn_synapse.py:175`](../msn_synapse.py#L175)). Euler steps the ODE
numerically and never forms the `1/(tau_s2 - tau_s1)` term.

### Fix

Add `method='euler'` to the hand-written STDP `Synapses` object so it matches
the cascade integration that `make_synapse` already uses.

---

## 3. Problem 2 — STDP destroyed the classifier

Once it ran, STDP made things *worse*:

| Mode | Train acc | Test acc |
|------|-----------|----------|
| Static random reservoir | 100 % | **100 %** |
| Plastic (STDP), original params | 100 % | **50 %** (chance) |

### Root cause: runaway potentiation

The original STDP rule was symmetric and unbounded in practice:

```python
lr_plus  = 0.3e-6 * amp   # LTP step
lr_minus = 0.3e-6 * amp   # LTD step
w_max    = 15e-6  * amp   # ceiling
```

With co-active recurrent pairs, every spike pairing pushed weights up by
0.3 µA. Starting from a mean of **0.28 µA**, the weights ran up **~14×** to a
mean of **3.92 µA**, with many pinned near the 15 µA ceiling. Strong recurrent
excitation then coupled the *left* and *right* groups: both fired on every
trial, the feature clouds collapsed onto each other, and the readout was left
guessing — hence 50 %.

This is the classic instability of plain Hebbian learning: **LTP is
self-reinforcing and needs an opposing force** (bounded weights, homeostasis,
or inhibition) to stay useful.

---

## 4. Why was there *only* excitation? (the real fix)

The original reservoir modelled recurrent connections as **purely
excitatory** — which is both biologically wrong and the deeper cause of the
instability.

**MSNs (medium spiny neurons) are GABAergic projection neurons: they inhibit
each other.** A striatal-style network with only excitatory recurrence has no
biological basis and no stabilising feedback. Adding recurrent inhibition does
two things at once:

1. **Restores E/I balance** so excitatory STDP cannot run away — extra firing
   recruits extra inhibition, which caps the spiking that drives further LTP.
2. **Creates winner-take-all competition.** Wiring cross-group inhibition
   (L ⊣ R and R ⊣ L) means the stimulated side actively *suppresses* the other
   side, which **sharpens** the left/right contrast the readout depends on.

The E-I demo declares a dedicated inhibitory inlet on the neuron group and a
fast (50 ms) cross-group inhibitory pathway, both through the existing
factories:

```python
reservoir = make_msn(N=N, params=params,
                     exc_inlets=('I_exc_rec', 'I_exc_L', 'I_exc_R'),
                     inh_inlets=('I_inh_rec',))
...
syn_inh = make_synapse(
    reservoir, reservoir,
    SynapseParams(kind='inh', weight=1.0e-6,
                  tau_s1=50e-3, tau_s2=50e-3, target_var='I_inh_rec'),
    connect='(i < N_LEFT) != (j < N_LEFT) and rand() < 0.4')   # L⊣R, R⊣L
```

### STDP also given proper bounds

Alongside inhibition, the plastic excitation was tamed so potentiation
saturates gracefully instead of exploding:

| Parameter | Original | E-I demo | Effect |
|-----------|----------|----------|--------|
| `lr_plus` | 0.30 µA | 0.020 µA | 15× smaller LTP step |
| `lr_minus`| 0.30 µA | 0.025 µA | net **depression bias** (`> lr_plus`) |
| `w_max`   | 15 µA   | 1.0 µA  | hard ceiling near the input scale |

With these, the recurrent weights stay bounded — mean actually *drifts down*
from 0.336 µA → **0.199 µA** over training rather than blowing up.

---

## 5. Per-device variability

The E-I demo also runs the reservoir with hardware mismatch via
[`msn_variability.apply_variability`](../msn_variability.py), scattering each
neuron's `Vth` and `I_hold` around the calibrated means (35-device P0118MA
dataset). At `scale = 0.5`:

```
Vth    ∈ [1.18, 2.05] V       (per-neuron thresholds)
I_hold ∈ [70.3, 95.4] µA
```

This both enriches the reservoir (heterogeneous thresholds → more diverse
features) and confirms the classifier survives realistic device-to-device
variability.

---

## 6. Results

All runs: 20 neurons, 20 train + 10 test trials, ridge-regression readout.
Variability `scale = 0.5`. Figures saved alongside the demo.

| Configuration | Recurrent exc | Inhibition | Train | Test | Figure |
|---------------|---------------|------------|-------|------|--------|
| Original demo, static | random, frozen | none | 100 % | 100 % | `ns_msn_rc_demo_static.png` |
| Original demo, STDP (bug fixed) | plastic, **unbounded** | none | 100 % | **50 %** | `ns_msn_rc_demo_stdp.png` |
| E-I demo, static | random, frozen | **L⊣R** | 100 % | 100 % | `ns_msn_rc_ei_demo_static_ei.png` |
| E-I demo, STDP (bounded) | plastic, bounded | none | 100 % | 100 % | `ns_msn_rc_ei_demo_stdp_e.png` |
| **E-I demo, STDP + inhibition** | plastic, bounded | **L⊣R** | 100 % | **100 %** | `ns_msn_rc_ei_demo_stdp_ei.png` |

### Reading the results honestly

- **Bounded STDP is the numerical fix.** Once `lr` and `w_max` are sane, the
  plastic reservoir matches the static one (100 %) *even without inhibition* on
  this easy, linearly-separable task. Accuracy alone therefore does **not**
  prove inhibition is doing the work here — the weight distribution and raster
  do (bounded weights, clean L/R selectivity).
- **Inhibition is the biological fix and the safety margin.** It is what makes
  the network a plausible MSN circuit, and it is the principled guard against
  runaway potentiation that would return the moment the learning rate, run
  time, or stimulus statistics push the excitatory loop harder. On a harder
  task the cross-group competition it provides should also improve
  discrimination, not just preserve it.

---

## 7. Takeaways

1. **Always pin `method='euler'`** on hand-written `Synapses` that reuse the
   equal-`tau` cascade — the `'exact'` solver divides by `(tau_s2 - tau_s1)`.
2. **Plain Hebbian STDP needs an opposing force.** Bounded weights + a
   depression bias keep it stable; that is the minimum to avoid the 50 %
   collapse.
3. **Model the inhibition.** MSNs are GABAergic; an excitation-only recurrent
   network is neither biological nor self-stabilising. Cross-group inhibition
   gives E/I balance and winner-take-all competition for free.
4. **Variability is cheap insurance.** The classifier holds up under half-scale
   device mismatch, so the result is not an artifact of identical idealised
   neurons.

---

## 8. How to reproduce

```bash
# Original demo (the bug + the runaway), toggled by env var:
USE_STDP=False python demo/ns_msn_rc_demo.py      # static  → 100% test
USE_STDP=True  python demo/ns_msn_rc_demo.py      # STDP    → 50%  test

# E-I follow-up, toggled by USE_STDP / USE_INH / VAR_SCALE:
USE_STDP=True  USE_INH=True  python demo/ns_msn_rc_ei_demo.py   # main result → 100%
USE_STDP=True  USE_INH=False python demo/ns_msn_rc_ei_demo.py   # STDP, no inhibition
USE_STDP=False USE_INH=True  python demo/ns_msn_rc_ei_demo.py   # static + inhibition
```
