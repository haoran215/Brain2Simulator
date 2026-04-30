# Brain2Simulator (ns_config)

Simulation and analysis scripts for a **memristive spiking neuron–synapse model** implemented in **Brian2**.

This branch contains:
- a Brian2 network demo using the paper-style “memristor aLIF” neuron with a two-stage synaptic filter (Eqs. 9–12 referenced in code)
- scripts to characterise the single-spike waveform + synaptic impulse response
- a comparison script contrasting the abstract aLIF model against a thyristor-hardware-inspired model
- pre-generated figures (PNG) produced by the scripts

> Note: this is research code / a working sandbox. APIs and file names may change.

## Quick start

### 1) Create an environment and install deps

This repo is configured as a Python project via `pyproject.toml`.

- Python: **>= 3.12**
- Key deps: `brian2`, `numpy`, `matplotlib`

Using `uv` (recommended if you already use it):

```bash
uv sync
```

Or using `pip` (simple approach):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install brian2 numpy matplotlib
```

### 2) Run the main demo simulation

The most “end-to-end” demo in this branch is `ns_test.py`.

```bash
python ns_test.py
```

It will run a 10-second Brian2 simulation and save a figure to:
- `neuron_sim.png`

The produced plot includes:
- excitatory / inhibitory Poisson input rasters (rates modulated sinusoidally)
- output spike raster for two neurons
- membrane voltage traces `Vm(t)`
- synaptic current traces (second-stage currents) `Is2_exc(t)` and `Is2_inh(t)`

## What each script does

### `ns_test.py`
A Brian2 demo network:
- 2 neurons
- each neuron receives 1 excitatory + 1 inhibitory Poisson input channel
- Poisson rates vary sinusoidally over time
- synaptic input is modelled as an impulse “kick” to a first synaptic stage (`Is1_*`), filtered into a second stage (`Is2_*`)

Outputs:
- `neuron_sim.png`

### `Spikesynapsechar.py`
Characterisation / sanity-check script for:
- **single-spike shape** under tonic current injection (subthreshold charging + threshold + reset)
- **synaptic impulse response** to a single pre-synaptic spike
- analytical overlays for the expected exponential / alpha-function shapes
- an I–F curve sweep for the selected parameters

Outputs:
- `spike_synapse_characterisation.png`

### `modelcopare.py`
(typo in filename is preserved)

Compares three parameter sets / model variants:
- **SET A**: abstract aLIF memristor model (series topology)
- **SET B**: thyristor-hardware working fit (parallel conductance topology)
- **SET C**: thyristor hardware with actual component values

It prints a parameter comparison table, derives analytical I–F curves, runs representative Brian2 simulations, and saves:
- `model_comparison.png`

## Figures in the repo

This branch includes generated figures for convenience:
- `neuron_sim.png`, `neuron_sim_v2.png`, `neuron_sim_v3.png`
- `spike_synapse_characterisation.png`
- `model_comparison.png`

If you re-run the scripts, these images may be overwritten.

## Project structure

At the repo root (this branch):
- `pyproject.toml` — project metadata + dependencies
- `uv.lock` — locked dependency set for `uv`
- `*.py` — runnable analysis scripts
- `*.png` — generated plots
- `*.ipynb` — exploratory notebook (`memr_tutorial_good_fit_addVpre_inprogress.ipynb`)

## Notes / limitations

- The code is written as **scripts** rather than a packaged library.
- Some model details are described inline in docstrings and comments; the most complete explanations are in:
  - `ns_test.py` (network-level demo)
  - `Spikesynapsechar.py` (single-spike + synapse shape + analytical checks)
  - `modelcopare.py` (conceptual comparison against thyristor-inspired hardware model)

## License

MIT (see `LICENSE`).
