# Phase 0 — Map of the existing system

Plan: *Software Exploration of Memristive Spiking Neurons and STDP* (Phase 0).
Snapshot of the repository on 2026-09-30, branch `STDP-wip`, with the `MSN`
(`2057ffb`) and `HD_Stimu` branches also inspected.

```text
MSN dynamics ─────────► STDP mechanism ─────────► network primitives ─────────► computational primitives
2 state vars (Vm, s)    3 software rules          recurrent RC, E-I motif,       classification, WTA selection,
no slow state           (no PCB model in repo)    WTA, self-exc bump, CPG,       oscillation / bursting,
latch above I_hold                                pacemaker, delay chain,        transient persistence, delays,
                                                  2-state compass                heading flip
```

**Where the code lives.** `STDP-wip` has the library (`msn_neuron.py`,
`msn_synapse.py`, `msn_variability.py`), `Classification-STDP/` and
`Classification-supervised/`. The network demos (`demo/*.py`) and
`docs/STDP_EI_report.md` live only on `MSN`. The compass demo lives only on
`HD_Stimu`.

---

## 1. MSN dynamics

`msn_neuron.make_msn`, parameters from `configs/neuron_hardware_fit.json`
(Cm = 100 nF, Ra = 2.2 kΩ, Rm_hi = 60.9 kΩ, Rm_lo = 53 Ω, Vth = 2.50 V, I_hold = 95 µA).

| Item | Implementation |
|---|---|
| State variables | `Vm` (membrane voltage), `s ∈ {0,1}` (memristor open/closed). Per-neuron parameters `Vth`, `I_hold`. |
| Membrane | `Cm dVm/dt = I_0 + I_exc − I_inh − Vm/(Rm(s)+Ra)`, with `Rm(s) = (1−s)Rm_hi + s·Rm_lo` |
| Memristive state | `s`, a discrete event-driven switch, not an ODE |
| Spike | `Vm > Vth and s = 0` sets `s = 1` (the switch closes and Cm discharges through Rm_lo+Ra) |
| Reset / refractoriness | No voltage reset. The `reopen` event sets `s = 0` when `I_M < I_hold`. Refractoriness is the closed time (0.7–1.4 ms, rising with input). |
| Nonlinearities | Threshold switch; the latch (depolarisation block) for `I_in > I_hold` |
| Hysteresis | Only inside a spike (the switch closes at Vth and reopens at I_hold) |
| Adaptation | **None intrinsic.** Demos build it from slow self-inhibitory synapses (CPG, τ_s = 3 s). |
| Time constants | τ_open = Cm(Rm_hi+Ra) = 6.31 ms, τ_close = Cm(Rm_lo+Ra) = 225 µs |
| Operating window | I_min = 39.6 µA (rheobase) to I_max = I_hold = 95 µA; rate 0–217 Hz |
| Variability model | `msn_variability.apply_variability`: per-neuron `Vth`, `I_hold` from the 35-device distribution, clipped to the measured range |
| Variants | `MNIST_test/msn_bistable.py` (MSN_HH): a continuous bistable gate `x` replacing `s`. Same F–I, 3–31× slower to simulate. **Not implemented:** the MSBN second compartment (Rs, Cs) of Wu et al. 2023 §3, the only published variant with a slow intrinsic state and bursting (README on `MSN`, §limitations 4). |
| Timestep | dt = 10 µs is accurate to the ±2 Hz rate resolution (`experiment/results/fi_dt_check/`) |

## 2. STDP mechanisms in the repository

All three are Brian2 `Synapses` with event-driven updates at pre/post spikes. None models the PCB.

| | `ns_msn_rc_ei_demo.py` (`MSN`) | `pavlov_demo.py`, `train_stdp.py` | `train_bsf.py` |
|---|---|---|---|
| Rule | Pair-based additive | Diehl & Cook target-biased pair rule, plus L1 normalisation (`train_stdp`) | Brader–Senn–Fusi stop-learning |
| Pre trace | `Apre`, τ_pre = 20 ms, `+= 1` | `apre`, τ = 20 ms, `+= 1` | — (uses post `Vm` and calcium `C`) |
| Post trace | `Apost`, τ_post = 20 ms, `+= 1` | `apost`, τ = 20 ms, `+= 1` | calcium `C`, τ_C |
| Potentiation | on post: `w += lr_plus·Apre` | on post: `w += η_post(apre − x_tar·w^μ)` | on pre, if `Vm > θ_V` and C in the LTP window: `X` up |
| Depression | on pre: `w −= lr_minus·Apost` | on pre: `w −= η_pre·apost` | on pre, if `Vm ≤ θ_V` and C in the LTD window: `X` down |
| Constraints | clip `[0, 1 µA]` | clip `[0, w_max]` | `X ∈ [0, X_max]`, bistable drift |
| Weight units | Amps: the jump added to the synaptic cascade `Is1` | Dimensionless × `w_unit` | Binary: `w_eff = w_jump·𝟙[X > θ_X]` |
| Precision | float64 | float64 | **1 bit** delivered |
| Known status | Works in the E-I reservoir after capping `w_max` (`docs/STDP_EI_report.md`) | Runaway or flat receptive fields on MNIST (`Classification-STDP/RESUME.md`) | Working MNIST rule (N = 100) |

**Missing:** the PCB STDP window (Δw vs Δt), its weight levels, its
potentiation/depression asymmetry, and its update timing. Phase 10 needs these.

## 3. Existing network primitives

| Primitive | Where | What it is |
|---|---|---|
| Recurrent network | `demo/ns_msn_rc_demo.py`, `ns_msn_rc_ei_demo.py` (`MSN`) | 20-MSN reservoir, random recurrence, ridge readout. The E-I version has plastic E→E and variability. |
| Inhibition / E-I motif | `demo/two_E_one_I.py` (`MSN`) | 2E + 1I motif, slow mutual E vs global I |
| WTA | `demo/ns_msn_wta_demo.py` (`MSN`); `Classification-STDP` | 2-neuron mutual inhibition; N E + N I Diehl–Cook WTA |
| Bump | `demo/ns_msn_v3_bump.py` (`MSN`) | **Single** neuron with self-excitation: a transient burst that fades. Not a ring attractor. |
| CPG | `demo/ns_msn_cpg_demo.py` (`MSN`) | Half-centre oscillator: fast mutual inhibition, slow self-inhibition |
| Pacemaker | `demo/ns_msn_pacemaker.py` (`MSN`) | Self-excitation bursts terminated by the I_hold latch |
| Delay chain | `demo/ns_msn_v5_delay_chain.py` (`MSN`) | 4-neuron feed-forward chain of self-terminating bursts |
| Head direction | `demo/ns_msn_compass_demo.py` (`HD_Stimu`) | 2-state compass: EB_L/EB_R WTA plus global inhibitor and PB relays. **No N-neuron ring.** |
| Sensory encoding | MNIST scripts, RC demo | Poisson rate coding; constant-current coding (`MNIST_test`) |
| Robotics / closed loop | — | None |

## 4. Computational primitives demonstrated so far

Classification (reservoir, supervised MNIST, BSF unsupervised MNIST), selection
(WTA), rhythm generation (CPG, pacemaker), transient persistence (self-excitation
bump), timed propagation (delay chain) and state switching (compass flip).
**None of them relies on a slow state inside the neuron.** Every timescale
longer than ~10 ms comes from a synapse (τ_s from 5 ms to 3 s) or from recurrence.

## 5. Consequences for the plan

- Phase 6 (ring attractor) and Phase 7 (learned WTA) have starting points on
  `MSN`/`HD_Stimu` but no N-neuron ring. That part is new work.
- Phase 4 (reservoir) should extend `ns_msn_rc_ei_demo.py`, not start fresh.
- Phase 1 below tests directly whether the MSN has a neuronal memory that the
  2×2 experiment of Phase 3 could exploit.
