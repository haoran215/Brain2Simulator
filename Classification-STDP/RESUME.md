# Resume notes — STDP MSN classifier

Snapshot of where this work-in-progress sits at end of session 2026-05-08.
Read this before continuing tomorrow.

---

## What's on disk

```
Classification-STDP/
  methodology.md         design doc — architecture, encoding, STDP rule, plots
  train_stdp.py          Brian2 training: 784 Poisson → N E-MSN ↔ N I-MSN
                         with plastic input STDP + lateral WTA + homeostatic θ
                         + divisive L1 normalisation
  eval_stdp.py           post-hoc label-assign + test accuracy, frozen weights
  plot_weights.py        receptive-field static grid + animation
  plot_connectome.py     4-panel topology figure (input→E, E→I, I→E, labels)
  learning_curve.py      per-snapshot quick-eval, accuracy vs training-image
  compare_N.py           orchestrator for N=50/100/150 sweep + curve overlay
  weights_N100.npz       latest sanity run output (the third failed config —
                         keep as evidence, do not assume usable)
  RESUME.md              this file
```

All scripts are independently runnable. `compare_N.py` chains them.

---

## What works

- Pipeline scaffolding is complete and end-to-end runnable.
- Brian2 + cython codegen is verified on this machine (`gcc 13.3.0`,
  `Cython 3.1.3` already installed).
- Plot scripts (`plot_weights.py`, `plot_connectome.py`, `learning_curve.py`)
  produce sensible figures from any weights file.
- The MSN neuron, the lateral-inhibition wiring, the homeostatic θ, and
  the per-image presentation loop all integrate without bugs.

---

## What does NOT work — three failure modes hit, in order

| iter | STDP rule + extras | result | mean ⟨\|w\|⟩ | frac at w_max | other |
|---|---|---|---|---|---|
| 1 | symmetric pair-STDP (`w += η·apre·(w_max-w)^μ`), no `x_tar`, no norm | runaway potentiation | 1.000 | 100% by image 100 | θ → 95 µA |
| 2 | target-bias only (`w += η·(apre-x_tar)·(w_max-w)^μ`) | runaway potentiation again | 1.000 | 100% by image 200 | θ → 95 µA |
| 3 | full Diehl-Cook (`w += η·apre - η·x_tar·w^μ`) + L1 norm + `Cm = 0.05 µF` (200 Hz ceiling) | flat uniformity, zero specialisation | 0.099 | 0% | inter-neuron cosine similarity = **0.99** |

The third configuration is in `train_stdp.py` as the current default.

---

## Diagnosis of the third failure

```
weight range:     [0.066, 0.161]   ← compressed into 2.5x band
per-neuron std:   0.0203 (identical for every neuron)
inter-neuron cosine similarity: 0.99
```

**Symmetry was never broken.** All 100 E neurons started at
`uniform[0, 0.3]` and converged to the same flat receptive field —
they all average across all digits. Two coupled root causes:

1. **Init scheme too uniform.** STDP cannot break symmetry on its own
   when every neuron sees the same input statistics with the same prior.
2. **Lateral inhibition crushes uniformly instead of arbitrating.** With
   99 I→E synapses at `w_i2e = 30 µA` and τ_s1 = 200 ms cascade,
   equilibrium inhibition per E neuron ≈ 3 mA — that suppresses
   everyone equally rather than giving the temporary winner a
   competitive advantage. Firing rate stayed at 5 Hz despite the
   200 Hz ceiling.

---

## Open question raised at end of session

**Can we change the STDP rule itself?** Three candidates worth trying
(in order of fit to the memristive-synapse framing the user cares
about):

1. **Brader-Senn-Fusi 2007** ("stop-learning") — discrete weight
   levels, each STDP event is a probabilistic jump, voltage-gated.
   Maps directly onto memristor SET/RESET writes with finite states.
   Strongest fit for the device-physical narrative.

2. **Voltage-triggered STDP** (Clopath et al. 2010) — LTP requires
   post-spike + high recent membrane voltage; LTD scaled by a
   moving-average of post firing rate. The moving-average is built-in
   homeostasis: high-firing neurons auto-suppress their LTP, which
   directly addresses the specialisation failure we keep hitting.

3. **SoftHebb** (Moraitis et al. 2022) — Bayesian WTA, no
   normalisation needed. Strongest theoretical guarantees, weakest
   memristor mapping.

---

## Concrete next steps for tomorrow

In order of decreasing certainty that they help:

1. **Write a fast pure-numpy diagnostic** that simulates one (or a few)
   E neurons + 784 Poisson inputs + STDP + WTA, no Brian2 overhead.
   Runs in seconds. Use it to sweep init scheme, η, x_tar, w_unit,
   w_i2e, norm_target, and find a working combination *before* spending
   another 40 min on a Brian2 sanity. **Highest expected value of
   tomorrow's session.**
2. **Heterogeneous Gaussian init.** Replace `uniform[0, 0.3]` with
   `clip(N(0.15, 0.15), 0, w_max)` so neurons start with different
   random preference patterns — gives WTA something to arbitrate.
3. **Drop `w_i2e` ~6×** from 30 µA to 5 µA so lateral inhibition picks
   a winner instead of crushing the field.
4. **Bump `w_unit` 2-3×** from 1e-7 to 2.5e-7 so post-norm mean
   current lands inside the MSN spiking window instead of just above
   rheobase.
5. After 1–4 produce a single working sanity (N=100, ~1500 images,
   bimodal weights, digit-shaped receptive fields, accuracy > 30%),
   then run the full pipeline: `eval_stdp.py`, `learning_curve.py`,
   `plot_weights.py`, `plot_connectome.py`. Then `compare_N.py
   --Ns 50 100 150`.
6. Independent of everything above: try Brader-Senn-Fusi as a second
   training script (`train_bsf.py`) so the paper has both pair-STDP
   and a discrete-weight rule for comparison. The latter is the one
   that lands you in *Nature Electronics* territory if it works.

---

## Performance notes

- `--codegen cython` produces no measurable speedup over numpy on this
  machine (1948 s vs. 1946 s on the same workload). Inner integrator is
  not the bottleneck — per-image Python overhead is. Real fix: batch
  many image presentations into a single `net.run` driven by a
  `TimedArray` of Poisson rates instead of looping in Python. Not
  attempted yet; ~40 min of refactor.
- `Cm = 0.05 µF` makes `τ_close ≈ 0.03 ms`, smaller than the integrator
  `dt = 50 µs`. The simulation runs but is slightly subsampled at the
  spike-width transition. Set `--dt 10e-6` if reviewers ask.

---

## Don't repeat these mistakes

- **Don't trust `mean(W)` as a learning indicator** under L1
  normalisation — the L1 constraint forces `mean(W) = norm_target /
  784` regardless of distribution. Use `frac@max`, per-row std, and
  inter-neuron cosine similarity instead.
- **Don't tune `x_tar` upward to compensate for low post-rate** —
  with `μ = 0.2` the equilibrium goes as `(apre/x_tar)^5`, hyper-
  sensitive. Either change `μ` to ~1.0 or fix the rate via `Cm` /
  inhibition strength.
- **Don't blindly port Diehl-Cook constants.** They were tuned for
  LIF at 30–80 Hz post-rate. MSN at the paper-default `Cm = 1 µF`
  fires at ~3 Hz; the ratio of LTP to spike-driven decay is wrong
  by ~20×.
