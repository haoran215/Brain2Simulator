# MSN MNIST classifier — methodology

End-to-end recipe for the rate-coded MNIST classifier built on the Wu et al.
2023 memristive spiking neuron (MSN). The pipeline trains a differentiable
*rate proxy* in PyTorch, transfers the resulting `100 × 784` weight matrix
to a Brian2 implementation of the actual MSN circuit, and reads the digit
out by majority spike count over a 10-neuron group.

```
              train_pytorch.py            eval_msn_brian2.py / plot_raster.py
            ┌──────────────────┐         ┌─────────────────────────────────┐
  MNIST ──▶ │ rate proxy SNN   │ ──W──▶ │ 784 Poisson → 100 MSN (Brian2)  │
            │ F_MAX·sigmoid(I) │         │ argmax over 10 group spike sums │
            └──────────────────┘         └─────────────────────────────────┘
```

Files in this directory:

| file | role |
|---|---|
| `train_pytorch.py` | trains the differentiable rate-proxy SNN, dumps `weights.npz` |
| `weights.npz`      | `W (100,784)` plus `F_MAX, N_HID, N_CLASSES, PER_CLASS, train_acc, test_acc` |
| `eval_msn_brian2.py` | runs the trained `W` through the real MSN simulator and reports accuracy |
| `plot_raster.py`   | visualises the spiking response — one MNIST digit per class |
| `eval_results.png` | confusion matrix + group-rate heatmap from the Brian2 evaluation |
| `raster_per_digit.png` | raster of the 100 output neurons for one image of each digit |

---

## 1. Network architecture

```
   784 input rates   ──[ W (100×784) ]──▶   100 MSN hidden rates
                                              │
                                              ▼   reshape to (10, 10)
                                            group-mean rate per class
                                              │
                                              ▼   argmax → digit
```

* **Inputs.** Each of the 784 pixel intensities `p ∈ [0, 1]` drives one
  Poisson source with rate `λ_max · p` Hz. Default `λ_max = 200 Hz`.
* **Hidden layer.** `N_HID = 100` MSN neurons partitioned into
  `N_CLASSES = 10` groups of `PER_CLASS = 10` (group `g` = neurons
  `[10g, 10g + 9]`). Each group encodes one digit.
* **Synapses.** A signed weight matrix `W ∈ ℝ^{100×784}` is split at the
  Brian2 boundary: `W > 0` connects pixel → `Is1_exc`, `W < 0` connects
  pixel → `Is1_inh` of the same neuron. Both excitatory and inhibitory
  channels feed into the same two-stage cascade
  `Is1 → Is2` with `τ_s1 = τ_s2 = 200 ms` (set on `MSNParams`).
* **Readout.** Spike counts over the presentation window are summed inside
  each group; the group with the most spikes wins. There is **no extra
  output layer** — the readout is just the partition.

There is no bias term: a per-neuron tonic current `I_0` would shift the MSN
operating point and complicate the I-F mapping, so all discriminative
information is carried by `W`.

---

## 2. Training (PyTorch surrogate, `train_pytorch.py`)

Brian2 simulation of 100 MSN neurons cannot be backprop'd through directly,
so training uses a **rate-domain surrogate** of one MSN:

```
rate(I) = F_MAX · sigmoid(I)            [Hz]      F_MAX ≈ 8 Hz
```

`F_MAX` is the saturation rate of the Wu et al. 2023 MSN at the onset of
depolarisation block. The sigmoid mimics the bounded, monotone shape of
the I-F curve in the spiking window without committing to any specific
slope.

Loss / optimiser:

* `cross_entropy(logits / TRAIN_TEMP, y)` with `TRAIN_TEMP = 0.3` — the
  group-mean rates are bounded in `[0, F_MAX]`, so a softmax temperature
  is needed to sharpen the readout enough for CE to drive learning.
* Adam, `lr = 3e-3`, batch 128, 10 epochs.
* No weight clipping — sign of `W` becomes the synapse polarity at
  transfer time.

Output: `weights.npz` with the dense `(100, 784)` weight matrix and
metadata. Train/test accuracies of the surrogate define the **transfer
ceiling** for the Brian2 evaluation.

```bash
uv run python Classification/train_pytorch.py --epochs 10
# → weights.npz, ~97% test accuracy on the rate surrogate
```

---

## 3. Brian2 evaluation (`eval_msn_brian2.py`)

The evaluation script rebuilds the network with real MSN dynamics and the
trained `W` already on the synapses:

1. **Inputs.** `PoissonGroup(784)` whose rates are reset per image to
   `image · λ_max · Hz`.
2. **Neurons.** `make_msn(100, params)` from `msn_neuron.py`. Tonic bias is
   zeroed (`G.I_0 = 0 A`) so that all drive comes through synapses.
3. **Synapses.** Two `Synapses` objects — one for `W > 0` weights routed
   into `Is1_exc`, one for `|W| < 0` routed into `Is1_inh`. Weights are
   converted to amperes via `--weight_scale` (default `5e-7 A` per unit
   `W`, chosen so the mean-field `<I_in>` lands inside the MSN spiking
   window `[I_min ≈ 15 µA, I_max = I_hold = 100 µA]`).
4. **Readout.** A `SpikeMonitor` collects output spikes; per image the
   network is `restore`d to its initial state, run for `T = 0.5 s`,
   spike counts are reshaped to `(10, 10)` and summed along the group
   axis, and the argmax is the prediction.

Key knobs (see the script docstring for the full list):

| flag | default | purpose |
|---|---|---|
| `--T`              | 0.5 s   | presentation time per image |
| `--lambda_max`     | 200 Hz  | peak Poisson rate at pixel intensity 1.0 |
| `--weight_scale`   | 5e-7 A  | amps per unit `W`; tunes operating point |
| `--tau_s`          | 200 ms  | sets both `τ_s1` and `τ_s2` of the synaptic cascade |
| `--n`              | 200     | number of test images |

Run:

```bash
uv run python Classification/eval_msn_brian2.py --n 200
```

The script prints a mean-field sanity check confirming `<I_in>` is inside
`[I_min, I_max]`, then loops through the test images, reporting a running
accuracy.

---

## 4. Raster visualisation (`plot_raster.py`)

`plot_raster.py` reuses `build_network` from `eval_msn_brian2.py` and runs
**one MNIST test image per class** (0 through 9). For each image it
records the full `(t, neuron_index)` spike train and produces
`raster_per_digit.png`:

* **Top two rows (10 panels).** One raster per input digit. Each spike is
  drawn at `(t, neuron_index)` and coloured by the *output group* the
  neuron belongs to (10 groups, `tab10` palette). The winning group's
  band is faintly shaded; the title shows `digit i → pred j` with a tick
  if correct.
* **Bottom-left.** A `(input digit) × (output group)` heatmap of total
  spike counts over `T`. The argmax of each row is outlined in cyan —
  this is what the readout reports.
* **Bottom-right.** The 10 input images themselves, so the predictions
  can be sanity-checked visually.

Run:

```bash
uv run python Classification/plot_raster.py
# → Classification/raster_per_digit.png
```

The default settings reproduce the figure currently in the directory; all
the same `--T / --lambda_max / --weight_scale / --tau_s / --seed` flags
from the evaluator are exposed so the raster can be regenerated under any
operating point.

---

## 5. Operating-point intuition

The MSN spikes only when its membrane current sits inside
`I_min ≤ I ≤ I_max = I_hold`. Below `I_min` the memristor never reaches
threshold; above `I_hold` the spike never reopens (depol-block). The
training-time sigmoid covers the monotone middle of this window, and the
`weight_scale` parameter exists precisely to keep the simulator there.
`MSNParams.summary()` prints both the current window and the open/close
time constants — eyeball it before changing `weight_scale` or `λ_max`.

The two-stage `Is1 → Is2` cascade with `τ_s = 200 ms` is what turns the
bursty Poisson input into a smooth quasi-DC current. Shortening `τ_s`
gives faster settling but noisier rate codes; lengthening it improves the
rate code at the cost of a longer minimum `T`.

---

## 6. Results

* **Surrogate ceiling (PyTorch rate proxy):** ~97% test accuracy.
* **Brian2 MSN classifier:** see `eval_results.png` for the confusion
  matrix and the per-class group-rate heatmap. The accuracy under the
  defaults above is within a few percentage points of the surrogate
  ceiling — the gap is dominated by Poisson finite-time noise (a longer
  `T` shrinks it) and the sigmoid/MSN I-F mismatch.
* **Raster sanity check (`raster_per_digit.png`):** at the default
  parameters all 10 sample digits classify correctly, with the correct
  group's row-band visibly denser than the others. Group spike counts
  for one representative run:

  ```
  digit 0 → pred 0   [37, 2, 10, 11, 13, 11, 0, 11, 10, 16]
  digit 1 → pred 1   [ 0,17,  9, 11,  0,  2, 2,  4,  3,  6]
  digit 2 → pred 2   [ 0, 9, 40,  1, 14, 11, 4,  9,  9,  0]
  digit 3 → pred 3   [ 1,14,  0, 25, 12, 16, 0, 18, 23, 12]
  digit 4 → pred 4   [ 0, 2,  0,  6, 17, 11, 3, 12,  7,  5]
  digit 5 → pred 5   [ 5, 4,  0, 15, 18, 48, 0, 10, 28, 10]
  digit 6 → pred 6   [ 0, 7, 12, 13, 11, 17,37, 10, 11,  0]
  digit 7 → pred 7   [ 3, 2, 13,  0, 10,  0, 0, 23,  3,  6]
  digit 8 → pred 8   [ 5, 5,  6, 14,  8, 20, 0,  0, 29,  5]
  digit 9 → pred 9   [ 5, 0,  7, 17,  4, 11, 0,  9, 13, 29]
  ```

---

## 7. Reproducing from scratch

```bash
# 1. train the surrogate; writes Classification/weights.npz
uv run python Classification/train_pytorch.py --epochs 10

# 2. measure Brian2 accuracy; writes Classification/eval_results.png
uv run python Classification/eval_msn_brian2.py --n 200

# 3. produce the per-digit raster; writes Classification/raster_per_digit.png
uv run python Classification/plot_raster.py
```

Each step is independent — only `weights.npz` is shared between them — so
the evaluator and raster scripts can be re-run under different
`--T / --lambda_max / --weight_scale / --tau_s` to explore the
rate-regime operating point without retraining.
