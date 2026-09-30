# Phase 1 — The MSN as a computational element

**Question.** With identical input currents, how does the MSN differ from a LIF
neuron? In particular, does the memristive state give the neuron a history
dependence, `V(t) = F(I(t), x(t))`, beyond what its RC membrane already provides?

**Short answer.** Below the holding current (`I_in < I_hold = 95 µA`), the MSN is
a LIF neuron with matched R, C and Vth, to within the input dependence of its
spike width. Its memristive state `s` lasts under 1 ms, and its history
dependence is set by τ_open = 6.3 ms (gone after about 30 ms). It shows no
adaptation and no hysteresis. **The only MSN-specific mechanism is the latch
above I_hold.** That latch has three measurable computational effects:
a non-monotonic (band-pass) F–I curve, an input-timed reset that makes the
neuron forget perturbations 2–4× faster, and higher spike-time reliability
when the input reaches the latch.

Code: [`experiments/phase1_characterize.py`](experiments/phase1_characterize.py) ·
models: [`models/neurons.py`](models/neurons.py) ·
numbers: [`results/phase1/phase1_results.json`](results/phase1/phase1_results.json)

---

## Setup

| | LIF-generic | LIF-matched | MSN |
|---|---|---|---|
| Membrane | R = 63.1 kΩ, C = 317 nF (τ = 20 ms) | R = 63.1 kΩ, C = 100 nF (τ = 6.31 ms) | Rm_hi+Ra = 63.1 kΩ, Cm = 100 nF (τ_open = 6.31 ms) |
| Threshold | 2.5 V | 2.5 V | 2.5 V |
| After a spike | reset 0 V, t_ref = 2 ms | reset to V_r = 0.214 V, t_ref = 0.80 ms | discharge through Rm_lo+Ra until I_M < I_hold |
| Above 95 µA | keeps firing | keeps firing | latches (s = 1) until I_in < I_hold |

- MSN parameters come from `configs/neuron_hardware_fit.json`.
- All three models have the same rheobase (39.6 µA), so every input current means the same thing for each model.
- LIF-matched uses the reopen voltage of the MSN as its reset, and the analytic MSN closed time at 65 µA as its t_ref.
- dt = 10 µs (forward Euler), Cython backend, seed 1. There's no device variability in this phase.

---

## Results

### P1 — F–I curve ([figure](results/phase1/P1_fi.png))

| | Rheobase | Max rate | Above 95 µA |
|---|---:|---:|---|
| LIF-generic | 39.65 µA | 123 Hz at 150 µA | keeps rising |
| LIF-matched | 39.65 µA | 389 Hz at 150 µA | keeps rising |
| MSN | 39.65 µA | 218 Hz at 90 µA | **silent from 96 µA** |

- MSN and LIF-matched coincide up to about 70 µA.
- Above that the MSN falls slightly below (211 vs 222 Hz at 85 µA). Its closed time grows with input, and LIF-matched's fixed t_ref does not.
- The MSN is the only model with a **non-monotonic F–I curve**: firing collapses to zero above I_hold.

### P2 — Sustained step ([figure](results/phase1/P2_step.png))

- There's no spike-frequency adaptation in any model: first-ISI rate equals steady-state rate within ±1.2%.
- First-spike latency is the same for MSN and LIF-matched (13.4 ms at 45 µA, 4.0 ms at 85 µA).
- At a 110 µA step the MSN fires **one** spike and then latches.

### P3 — Slow triangular ramp, 0 → 150 → 0 µA ([figure](results/phase1/P3_ramp.png))

- With a 4 s ramp, the MSN blocks at 95.0 µA on the way up and releases at 94.7 µA on the way down.
- Its onset/offset on the ramp (40.1 / 41.1 µA) is identical to LIF-matched's.
- So there's **no hysteresis** beyond the lag that every model shows on the fast 0.4 s ramp. The latch is a static function of the present input.

### P4 — Paired pulses: integration window ([figure](results/phase1/P4_pairs.png))

Setup: two 0.1 ms pulses on a 30 µA bias. The table gives the charge needed per pulse, divided by the single-pulse threshold Q₁.

| | Q₁ | Integration window (ratio 0.75) |
|---|---:|---:|
| LIF-generic | 194 nC | 21.9 ms |
| LIF-matched | 61.3 nC | **6.9 ms** |
| MSN | 61.3 nC | **6.9 ms** |

The MSN and LIF-matched curves are identical at every interval. Sensitivity to input timing is set by τ_open alone.

### P5 — Pulse trains and bursts ([figure](results/phase1/P5_trains.png))

- **Following fidelity:** trains of 1.5·Q₁ pulses at 10–1500 Hz. MSN and LIF-matched are identical: 1 spike per pulse up to 50 Hz, 1/2 at 100–300 Hz, 1/3 at 500–1000 Hz.
- **Burst detection:** trains of 0.35·Q₁ pulses. Both need 4 pulses at 1 ms spacing and 5 at 2 ms, and never fire at ≥ 5 ms spacing.

### P6 — Irregular input: frozen OU noise ([figure](results/phase1/P6_irregular.png))

Setup: σ = 20 µA frozen noise (τ = 2 ms), plus 4 µA of private noise per trial, 20 trials. Reliability is the Schreiber correlation with a 1 ms Gaussian kernel.

| μ | Model | Rate | CV | Reliability |
|---|---|---:|---:|---:|
| 45 µA | LIF-matched / MSN | 71.8 / 71.0 Hz | 0.59 / 0.58 | 0.688 / 0.687 |
| 65 µA | LIF-matched / MSN | 145.6 / 138.2 Hz | 0.39 / 0.37 | 0.719 / 0.729 |
| 85 µA | LIF-matched / MSN | 221.0 / 169.0 Hz | 0.24 / 0.31 | 0.821 / **0.885** |
| 45–85 µA | LIF-generic | 22–69 Hz | 0.36–0.16 | 0.45–0.54 |

The two models separate only when the input spends time above I_hold. That's about 31% of the time at μ = 85 µA (Gaussian estimate), where the MSN latches and releases on the input's own timing. The rates differ there, so the reliability gap should be read with that confound in mind.

### P7 — Response after previous stimulation ([figure](results/phase1/P7_history.png))

Setup: 200 ms of conditioning at 65 µA (firing) or 120 µA (block), then a gap at 30 µA, then a 50 µA probe. The conditioning length was jittered over 20 values so the phase of the last conditioning spike averages out.

| | History effect disappears (mean \|Δlatency\| < 2%) | Latency / control at 0.2 ms gap (after block) |
|---|---:|---:|
| LIF-generic | 100 ms | 1.75 |
| LIF-matched | 30 ms | 1.69 |
| MSN | 30 ms | **2.28** |

- The MSN's history lasts exactly as long as LIF-matched's.
- After a block, the MSN always restarts from the same low voltage. The effect is larger than in LIF-matched and doesn't depend on phase, but it decays with the same τ_open.

### P8 — Perturbation memory ([figure](results/phase1/P8_perturb.png))

Setup: 30 frozen-noise seeds. For each, one copy of the trajectory gets a 0.1·Q₁ kick at 300 ms. "Convergence" is the time until the kicked and un-kicked spike trains agree again (within 0.05 ms).

| Regime | Time above I_hold | LIF-matched median / p90 / max | MSN median / p90 / max |
|---|---:|---:|---:|
| Fluctuation-driven (μ = 35, σ = 15 µA) | 0% | 0 / 3 / 22 ms | 0 / 3 / 22 ms |
| Mean-driven (μ = 65, σ = 15 µA) | 2.5% | 71 / 280 / 561 ms | **36 / 113 / 150 ms** |
| Mean-driven, **never reaches I_hold** (μ = 55, σ = 7 µA) | 0% | 182 / 680 / 698 ms | 274 / 696 / 698 ms |

The third row is the control: without latch events the MSN keeps a perturbation
just as long as LIF-matched (most runs haven't converged by the end of the
simulation). The faster forgetting in row 2 therefore comes from the latch.
Brief excursions above I_hold reset the neuron's phase at a time set by the
input, not by its own history.

The long memory in the mean-driven rows is the **phase of a regular
oscillator**, the same in all three models. It isn't a device memory.

---

## Answer to the Phase 1 question

**Is `V(t) = F(I(t), x(t))`, with a significant internal state?** Only in the
trivial sense that every RC neuron has one:

- The MSN's `V(t)` depends on the input over the last ~τ_open (6.3 ms). The history effect is gone after about 30 ms, the same as LIF-matched.
- The memristive state `s` exists only during a spike (under 1 ms) or while the input holds it latched. In both cases it's set by the present input, not stored.
- The model has **no slow intrinsic state**: no adaptation, no hysteresis, no memory beyond the membrane.

**What the MSN adds over LIF-matched** comes entirely from the I_hold latch:

1. **Band-pass current tuning.** The neuron responds only inside 40–95 µA, and strong drive silences it.
2. **Input-timed reset.** Latch episodes erase phase history: 2× faster median forgetting and 4× shorter worst case (P8). The neuron also restarts from a fixed state after a block (P7).
3. **Higher spike-time reliability** under input that reaches the latch (P6), at a lower rate.

## Implications for the next phases

- **Phase 2/3 (temporal patterns, 2×2 memory experiment).** For patterns
  longer than about 30 ms, the MSN can't hold information that LIF-matched
  can't. Any such memory has to come from synapses (τ_s) or recurrence. The
  expected 2×2 outcome is B ≈ A unless the task drives neurons into the latch.
  That's worth running to confirm, but the design should include inputs that
  do reach I_hold.
- **MSN-specific hypotheses worth testing with STDP:**
  - *Latch as built-in stop-learning.* When potentiation pushes a neuron's total drive past I_hold, the neuron goes silent, and pair-STDP stops potentiating it. This bounds weights without normalisation, and it would address the runaway failure in `Classification-STDP/RESUME.md`.
  - *Latch as resynchronisation* for sequence timing robustness (Phase 8).
  - *Band-pass tuning* as a basis for competition or specialisation (Phase 7).
- **A slow intrinsic state** would need the MSBN second compartment (Rs, Cs;
  Wu et al. 2023 §3). Whether to add it depends on whether the hardware
  neuron has it.

## Caveats

- One parameter set (hardware fit). There's no device variability yet, and `apply_variability` changes Vth and I_hold, which moves the latch.
- Deterministic Euler integration at dt = 10 µs. Rates are resolved to about 2 Hz.
- LIF-matched uses a fixed t_ref (0.80 ms). The MSN's closed time varies with input (0.7–1.4 ms between 42 and 93 µA), which explains the small F–I difference between 70 and 95 µA.
- The P6 reliability comparison at μ = 85 µA is between unequal rates.
- P8 uses one kick size (0.1·Q₁) and one noise time constant (2 ms).

Reproduce: `uv run python experiment/experiments/phase1_characterize.py` (≈1 min on this machine).
