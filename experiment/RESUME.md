# Resume notes — MSN + STDP exploration

Paused 2026-09-30, branch `STDP-wip`. Read this first when continuing.

## Where we are

Governing plan: `Software Exploration of Memristive Spiking Neurons and STDP.md`.
It replaces the earlier `MSN + STDP + MNIST 实验计划.md`, and MNIST is no longer
the focus.

| Step | Status | Output |
|---|---|---|
| Bring `MSN` configs and `MNIST_test/` into this branch | done (**staged, not committed**) | `configs/neuron_hardware_fit.json` etc. |
| Choose the timestep | done: **dt = 10 µs** (rates within 2 Hz, spike width −1%) | `experiments/fi_dt_check.py`, `results/fi_dt_check/` |
| Three neuron models | done: `lif_generic`, `lif_matched`, `msn` | `models/neurons.py` |
| Phase 0: map of the system | done | `PHASE0_MAP.md` |
| Phase 1: MSN characterisation | done | `PHASE1_REPORT.md`, `experiments/phase1_characterize.py`, `results/phase1/` |
| Revised plan (MSBN + PCB STDP + SHD) | proposed, **waiting for the user's answers** | below |

`experiment/` and both plan `.md` files are untracked. Nothing has been committed.

## Phase 1 result in one paragraph

Below I_hold = 95 µA the MSN behaves like LIF-matched. Both have the same
rheobase (39.65 µA), integration window (6.9 ms) and burst response, and both
lose history after about 30 ms. There's no adaptation and no hysteresis, and the
state `s` exists for under 1.4 ms. The only MSN-specific mechanism is the
**I_hold latch**, which produces:
- a band-pass F–I curve;
- an input-timed reset that makes the neuron forget perturbations 2–4× faster (the control with no latch events shows no difference);
- higher spike-time reliability when the input reaches the latch.

## Decisions taken by the user

- Keep both LIF baselines. LIF-generic checks the pipeline; LIF-matched isolates the effect of the latch.
- Rs and Cs are highly tunable in hardware, so **add the MSBN second compartment** (Wu et al. 2023).
- The PCB STDP learning window comes from d'Hollande et al. 2026
  (`d’Hollande_2026_Neuromorph._Comput._Eng._6_034015.pdf`).
- Use **SHD** (Spiking Heidelberg Digits) as the temporal STDP benchmark, not MNIST.
  The user's words: "the key is to verify the SNN".

## Proposed next steps (not started)

- **A. MSBN model.** Equations:
  - `Cm dV/dt = I_in − g(s)(V − V_S)`
  - `Cs dV_S/dt = g(s)(V − V_S) − V_S/Rs`

  Rs sits where Ra is, so Cs = 0 must reproduce the MSN. Checks, in order:
  1. Reproduce the paper's TS/FS/IB1/IB2 modes.
  2. Run the Phase 1 battery over (Rs, Cs) to find where slow memory appears.

  Hypothesis: a slow τS (for example Cs = 33 µF with Rs = 3 kΩ) gives adaptation-like memory.
- **B. PCB STDP model.**
  - τp = 3.5 ms traces.
  - Integer pulse count n = n₀[1 − |Δτ|/(kτp)].
  - The sign comes from the SR flip-flop.
  - Digipot step position, with G = 1/(R_series + R_W), so ΔG ∝ n·δr·G².
  - The synaptic current lasts about 1 ms (the pre spike), not the 200 ms cascade.

  Check it by reproducing the paper's Figs. 7 (window), 8 (beating) and 10 (Pavlov).
- **C. SHD.** Pool the 700 channels to about 140, and treat time scale as one parameter. Readout is linear on time-binned hidden spikes. Comparisons, in order:
  1. Input spike counts.
  2. Input binned in time.
  3. Fixed random hidden layer.
  4. PCB-STDP hidden layer.

  Steps 3 and 4 run with LIF-matched, MSN and MSBN, over several seeds, plus a time-reversed SHD control. Start with a 1k-sample subset.
- **D.** Small synthetic temporal patterns as unit tests before SHD.

## Open questions for the user (asked, not yet answered)

1. SHD time scale: compress the data (recommended; equivalent to scaling the capacitors) or keep real time and scale Cm/Cs?
2. May I fetch `github.com/Integrative-Neuroscience/ActiveDendrite-STDP` to fit n₀ and k? What are the digipot step count (DS1804: 100?) and the series resistor value?
3. May I download SHD (Zenke lab, a few hundred MB, into git-ignored `data/`)?

## Environment notes

- There's no PDF tool on the machine. Text was extracted with `uv run --no-project --with pypdf ...`, which leaves the project's dependencies unchanged.
- Rerun Phase 1 with `uv run python experiment/experiments/phase1_characterize.py` (about 1 min).
