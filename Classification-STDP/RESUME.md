# Resume notes — STDP / BSF MSN classifier

Snapshot of where this work sits at end of session 2026-05-10.
Read this before continuing tomorrow.

---

## What's on disk

```
Classification-STDP/
  methodology.md            design doc — architecture, encoding, plots
  train_stdp.py             Brian2 pair-STDP trainer (Diehl-Cook + L1 norm)
                            — kept for comparison, NOT the working rule
  train_bsf.py              Brian2 Brader-Senn-Fusi "stop-learning" trainer
                            — THE WORKING RULE, see "Why MNIST works" below
  pavlov_demo.py            10-pre × 10-post toy STDP sanity demo
                            (no MNIST) — confirms the rule wires up
  eval_stdp.py              post-hoc label-assign + test accuracy,
                            frozen weights (works on either trainer's npz)
  plot_weights.py           receptive-field static grid + animation
  plot_connectome.py        4-panel topology figure
  plot_pre_post.py          before/after pixel→neuron connection diagram
                            (K most-changed neurons)
  plot_schematic.py         architecture cartoon, no weights loaded
  learning_curve.py         per-snapshot quick-eval, accuracy vs image
  compare_N.py              orchestrator for N=50/100/150 sweep
  weights_bsf_N100.npz      working BSF run (N=100)
  weights_bsf_N20.npz       smaller BSF run for fast iteration
  weights_N100.npz          old failing pair-STDP run, kept as evidence
  pavlov_demo.png           figure produced by pavlov_demo.py
  schematic.png             figure produced by plot_schematic.py
  pre_post_bsf_N100*.png    figures produced by plot_pre_post.py
  connectome_bsf_N100.png   figure produced by plot_connectome.py
  RESUME.md                 this file
```

All scripts are independently runnable. `compare_N.py` chains them.

---

## Why MNIST classification works (BSF, not pair-STDP)

The pair-STDP path (`train_stdp.py`) hit three failure modes in a row,
all rooted in the same problem: **symmetry never broke** — every E
neuron converged to the same flat receptive field. See "Failure history"
below for the diagnosis.

`train_bsf.py` succeeds because it changes the rule, not the
hyperparameters. Three things together fix the symmetry-breaking
problem:

1. **Calcium-windowed gating ("stop-learning").** Each plastic synapse
   carries an internal `X ∈ [0, X_max]` and the **delivered current is
   binary** — `w_eff = w_jump · 𝟙[X > θ_X]`. Updates to X only happen
   inside a window on the post-neuron's calcium trace `C` (a leaky
   low-pass of post-spike history):
       LTP if  Vm_post > θ_V  AND  θ_lo_p ≤ C < θ_hi_p
       LTD if  Vm_post ≤ θ_V  AND  θ_lo_d ≤ C < θ_hi_d
       otherwise no update — *stop-learning*.
   Silent neurons (C below floor) and saturating neurons (C above
   ceiling) both freeze. This is the per-neuron homeostat that pair-STDP
   was missing — it stops the runaway potentiation that crushed iters
   1 and 2, and it stops the uniform-creep that flattened iter 3.

2. **Heterogeneous bistable init.** `X` starts as `clip(N(θ_X, 0.25))`
   so ~30% of synapses begin "ON" and ~70% "OFF" with a Gaussian spread
   around the readout threshold. WTA has something to arbitrate from
   step zero. Pair-STDP started uniform on `[0, 0.3]` and could not
   break that symmetry on its own.

3. **Lateral inhibition tuned for arbitration, not crushing.**
   `w_i2e = 5 µA` (down from the STDP path's 30 µA). Equilibrium
   inhibition is ~6× weaker, so the temporary winner gets a competitive
   advantage instead of every E neuron being suppressed equally.

The discrete-weight binary readout is also the right story for the
memristor framing — one bit per synapse, SET/RESET writes only, no
analog weight update needed at the device level. The continuous X is
just an internal state variable for the learning rule; the chip only
needs to store one bit per synapse and observe the local C and Vm.

`weights_bsf_N100.npz` is the current sanity run. Receptive fields are
digit-shaped (visible in `pre_post_bsf_N100_RFs.png`); inter-neuron
cosine similarity dropped from 0.99 (failing STDP) to a much lower
value with visible specialisation.

---

## Pavlov demo — what it is and why it exists

`pavlov_demo.py` is a **rule sanity check**, not part of the MNIST
pipeline. It runs on a tiny 10-pre × 10-post network with the *same
STDP code path* as `train_stdp.py`, so a passing demo proves the
plastic-synapse equations and Brian2 wiring are correct in isolation
from any data-pipeline or hyperparameter issue.

Setup (Pavlov's dog):
- pre 0..4 = "bell" (CS), pre 5..9 = "food" (US).
- bell→post weights start weak (`w = 0.05`, no salivation).
- food→post weights start strong (`w = 0.70`, unconditioned reflex).
- Each training trial: 50 ms bell alone, 300 ms bell + food paired,
  400 ms rest. CS precedes & overlaps US, so STDP grows bell→post.
- Every 5 trials, run a bell-only test and count post spikes.

Expected behaviour:
- Trial 0 test: bell-only post spikes ≈ 0 (CS doesn't drive post).
- Final test: bell-only post spikes high (CS now drives post on its
  own — the "salivation when bell rings" association has formed).
- ⟨w_bell⟩ rises monotonically toward `w_max`; ⟨w_food⟩ stays high.

The demo also pins down a small set of design choices used in the main
trainer:
- `Cm = 0.05 µF` lifts the MSN's max rate to ~200 Hz so STDP gets
  enough post-spikes per pairing window.
- `I_0` tonic bias sits *just below* `I_min ≈ 15 µA` so synaptic input
  only needs a few µA to evoke spikes — without this lift, 5 pre at
  50 Hz can't drive the MSN through the cascade in one trial.
- `tau_s = 100 ms` (vs 200 ms in MNIST) so the cascade reaches a
  useful fraction of `Is1_ss` within the sub-second pairing window.

Output: `pavlov_demo.png` with five panels (a) before/after weight
matrix, (b) ⟨w⟩ evolution split by bell vs food, (c) learning curve,
(d) early-trial raster, (e) late-trial raster.

---

## Failure history (kept for reference)

The pair-STDP path hit three failure modes; documented here so we don't
re-run them.

| iter | STDP rule + extras | result | mean ⟨\|w\|⟩ | frac at w_max | other |
|---|---|---|---|---|---|
| 1 | symmetric pair-STDP (`w += η·apre·(w_max-w)^μ`), no `x_tar`, no norm | runaway potentiation | 1.000 | 100% by image 100 | θ → 95 µA |
| 2 | target-bias only (`w += η·(apre-x_tar)·(w_max-w)^μ`) | runaway potentiation again | 1.000 | 100% by image 200 | θ → 95 µA |
| 3 | full Diehl-Cook (`w += η·apre - η·x_tar·w^μ`) + L1 norm + `Cm = 0.05 µF` | flat uniformity, zero specialisation | 0.099 | 0% | inter-neuron cosine sim = **0.99** |

Diagnosis of iter 3: weight range `[0.066, 0.161]` (compressed 2.5× band),
per-neuron std identical for every neuron, inter-neuron cosine 0.99 —
**symmetry was never broken**. Two coupled root causes:
1. Init too uniform — STDP cannot break symmetry on its own when every
   neuron sees the same input statistics with the same prior.
2. Lateral inhibition crushes uniformly. With 99 I→E synapses at
   `w_i2e = 30 µA` and τ_s1 = 200 ms cascade, equilibrium inhibition
   per E neuron ≈ 3 mA — that suppresses everyone equally rather than
   giving the temporary winner a competitive advantage.

Both issues are fixed in the BSF path (heterogeneous X init around
θ_X, `w_i2e` dropped to 5 µA, calcium-window homeostat per synapse).

---

## Open follow-ups

- Sweep N for BSF (50 / 100 / 150) with `compare_N.py`, plot accuracy
  vs N to support the discrete-weight argument.
- Voltage-triggered STDP (Clopath et al. 2010) as a third rule for
  comparison if reviewers ask for an analog-weight baseline that also
  has built-in homeostasis.
- The Python-loop overhead in train_bsf is still ~the bottleneck.
  Real fix: batch many image presentations into a single `net.run`
  driven by a `TimedArray` of Poisson rates instead of looping in
  Python. ~40 min of refactor, not blocking.

---

## Performance notes

- `--codegen cython` produces no measurable speedup over numpy on this
  machine — inner integrator is not the bottleneck, per-image Python
  overhead is.
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
- **Don't start synapses uniform when relying on WTA to specialise.**
  Symmetry has to be broken at init — the learning rule alone won't
  do it on identically-distributed input.