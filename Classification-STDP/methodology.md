# MSN MNIST classifier — unsupervised STDP

End-to-end recipe for an **unsupervised** rate-coded MNIST classifier built on
the Wu et al. 2023 memristive spiking neuron (MSN). Training uses pair-based
STDP only — no labels enter the weight updates. Labels are used **only after
training** to give each neuron a digit name; this is the standard
Diehl & Cook 2015 readout.

```
              train_stdp.py                       eval_stdp.py
            ┌──────────────────┐                ┌──────────────────────┐
  MNIST ──▶ │ 784 Poisson      │ ── STDP ──▶   │ freeze W, label each │
            │  → N E-MSN       │  on E-input   │ neuron by argmax     │
            │  ↔ N I-MSN (WTA) │  synapses     │ digit, then test     │
            │  + theta homeo   │               │ accuracy             │
            └──────────────────┘                └──────────────────────┘
```

Files in this directory:

| file | role |
|---|---|
| `train_stdp.py`     | trains plastic synapses with STDP, dumps `weights_N{N}.npz` plus snapshots |
| `eval_stdp.py`      | label-assignment + test accuracy for one trained N |
| `plot_weights.py`   | static 4-snapshot grid + animation MP4/GIF of receptive fields |
| `plot_connectome.py`| 4-panel topology figure (input→E, E→I, I→E, label assignments) |
| `learning_curve.py` | accuracy vs training-image count by re-evaluating each snapshot |
| `compare_N.py`      | orchestrates N=50/100/150 runs, aggregates accuracy and learning curves |

---

## 1. Network architecture (Diehl & Cook 2015 on MSN)

```
   784 Poisson (rate-coded pixels)
        │
        │   plastic STDP, w ∈ [0, w_max]
        ▼
   ┌──────────────┐
   │  N E-MSN     │  excitatory, plastic input
   │  homeo θ↑    │
   └──────┬───────┘
          │ 1:1 fixed strong (w_e2i)
          ▼
   ┌──────────────┐
   │  N I-MSN     │  inhibitory
   └──────┬───────┘
          │ all-to-all-except-self, fixed (w_i2e)
          └──▶ back to E (lateral inhibition / WTA)
```

* **Inputs.** 784 `PoissonGroup` sources at `λ_max · pixel` Hz.
  Default `λ_max = 63.75 Hz` (Diehl & Cook value); raised by 32 Hz on
  re-presentation if the E layer fires fewer than 5 spikes total.
* **Excitatory layer.** `N` MSN neurons (we sweep `N ∈ {50, 100, 150}`).
* **Inhibitory layer.** `N` MSN neurons paired 1:1 with E. The 1:1 connection
  is strong enough that one E spike reliably drives its paired I.
* **Lateral inhibition.** Each I neuron projects to **all E except its pair**,
  with a fixed inhibitory weight. The strongest E neuron suppresses the rest
  for the rest of the presentation — soft winner-take-all.
* **Adaptive threshold (homeostasis).** Each E neuron has a slow
  per-neuron `θ` that grows by `θ_plus` on each spike and decays with
  `τ_θ = 10^7 ms`. `θ` is fed in as an extra inhibitory current via a
  self-`Synapses(G_E, G_E, condition='i==j')` so we don't need to fork
  `msn_neuron.py`. This stops a single neuron from monopolising and forces
  population-wide specialisation.

---

## 2. Encoding

* **Pixels → rates.** Pixel `p ∈ [0, 1]` becomes Poisson rate `λ_max · p` Hz.
  No sign — STDP weights are non-negative.
* **Presentation.** Each image is shown for `T_present = 350 ms`, then the
  network runs `T_rest = 150 ms` with all input rates at zero so the
  synaptic cascade `Is1, Is2` and the STDP traces decay between images.
* **Re-presentation.** If E layer spikes fewer than 5 times during `T_present`,
  scale `λ_max` up by `+32 Hz` and re-present the same image. This
  guarantees every digit drives learning (Diehl & Cook trick).

---

## 3. STDP rule

Pair-based, target-biased, only on the 784 → E synapses (Diehl & Cook
2015, eq. 7). The target-bias `(apre - x_tar)` is what makes the network
specialise: silent inputs are *depressed* on every postsynaptic spike,
only inputs whose presynaptic trace is currently above `x_tar` get
potentiated.

```
dapre/dt  = -apre  / τ_pre        τ_pre  = 20 ms
dapost/dt = -apost / τ_post       τ_post = 20 ms

on pre :  apre  += 1
          w     -= η_pre * apost                          (post-trace LTD)

on post:  apost += 1
          w     += η_post * apre - η_post * x_tar * w^μ
                   ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
                   LTP gated by   unconditional decay
                   recent pre     on every post-spike
                   activity       — drives competitive
                                    specialisation
```

* `η_post = 1e-2`, `η_pre = 1e-4`, `x_tar = 0.4`, `μ = 0.2` — Diehl & Cook
  values, ported unchanged.
* The pre-spike LTD term has no `w^μ` factor so it does not stall at
  small `w`. The post-spike rule with `(apre - x_tar)` is the workhorse:
  the target bias provides competitive learning without per-neuron rate
  controllers.
* **Why STDP alone is not enough at MSN firing rates — divisive
  normalisation.** Even with the Diehl-Cook decay term, with the paper
  default `Cm = 1 µF` the MSN fires at ~3 Hz (~1 post-spike per
  presentation). At that rate the spike-driven decay (`η · x_tar · w^μ`)
  cannot keep up with LTP on active inputs, and the weights saturate to
  `w_max` within ~150 images. Two fixes go together:

  1. **Lift the firing rate.** We reduce `Cm` from 1 µF to 0.05 µF so
     `f_max = 1/(Cm·(Rm_hi+Ra)) ≈ 200 Hz`. This is a parameter choice,
     not a model change: the operating window `[I_min, I_max]`, the
     spike threshold, and the synaptic cascade are all unchanged. The
     8 Hz default ceiling is paper-set, not device-physical.
  2. **Divisive L1 normalisation.** Every `--norm_every` images
     (default 10), each E neuron's incoming weight vector is rescaled
     so its L1 sum equals `--norm_target` (default 78.0). This is what
     the canonical Diehl-Cook implementation actually does — STDP
     drives the relative weight order, normalisation enforces the
     sparse bimodal distribution.
* `w_max` is dimensionless. The kick delivered to `Is1_exc_post` per
  pre-spike is `w * w_unit` (with `w_unit` in amps); this decouples the
  learning algebra from the MSN's biological current scale.

E↔I and I→E synapses are **non-plastic** — they wire the WTA, nothing more.

---

## 4. Snapshots and weight evolution

`train_stdp.py --snapshot_every 500` dumps `W` every 500 training images to
`weights_N{N}.npz` under key `W_history` (shape `(n_snapshots, N, 784)`)
plus a `snapshot_steps` array. `plot_weights.py` consumes this:

* **Static figure (`weights_static_N{N}.png`).** Receptive-field grids at
  0% / 25% / 50% / 100% of training. Each grid tiles the `N` weight
  vectors as 28×28 patches. You can literally see Gaussian noise
  resolve into digit prototypes.
* **Animation (`weights_anim_N{N}.mp4` if ffmpeg available, else `.gif`).**
  Cycles through all snapshots at ~5 fps.

Plus a `training_trace_N{N}.png` with mean `|w|`, mean firing rate, mean θ
across training time — the standard "did learning actually happen" plots.

---

## 5. Readout (post-hoc label assignment)

After training:

1. **Re-present a labelled subset** (10 000 training images) with weights
   frozen and STDP off. Record per-image spike counts of all `N` E neurons.
2. For each E neuron, find which digit class produced the highest mean
   spike count over its presentations → `assigned_label[i]`.
3. **Test accuracy.** For each test image, sum the spike counts of all
   neurons assigned to digit `d` for `d ∈ {0..9}`; argmax gives the
   prediction.

This is the only step that uses labels. It does not change `W`.

---

## 6. Comparison: N = 50 vs 100 vs 150

`compare_N.py` runs the full pipeline for `N ∈ {50, 100, 150}`, sharing
hyperparameters, and writes `comparison.json` plus `comparison.png`:

* test accuracy vs N
* mean receptive-field "digit-likeness" (correlation of each receptive
  field with its assigned digit's mean image) vs N
* training wallclock vs N
* fraction of weights at `w_max` vs N

Expectation from Diehl & Cook scaling on this kind of dataset: ~75-80%
with N=50, ~85% with N=100, ~88% with N=150 — diminishing returns.
The MSN neuron may underperform the standard LIF baseline simply
because its saturation rate (~8 Hz) is much lower than LIF's (~150 Hz),
which costs spike statistics during the 350 ms window.

---

## 7. Operating-point notes

The MSN spikes only when its membrane current sits in
`I_min ≈ 15 µA ≤ I_in ≤ I_max = I_hold = 100 µA`. The plastic synapses
deliver `w * w_unit` amps per pre-spike. With default
`λ_max = 63.75 Hz`, `T_present = 350 ms`, mean active pixels ≈ 150,
saturating weights ≈ 1.0, target `<I_in> ≈ 50 µA`, the weight scale
works out to roughly `w_unit ≈ 1e-7 A`. The script prints a mean-field
sanity check at startup so you can confirm the operating point before
committing to a long training run.

The 200 ms `Is1 → Is2` cascade smooths the bursty input into a
quasi-DC current — it's also what makes 350 ms a reasonable
presentation time despite MSN's slow `τ_open ≈ 100 ms`.

---

## 8. Reproducing from scratch

```bash
# 1. train one size
uv run python Classification-STDP/train_stdp.py --N 100 --epochs 1

# 2. label-assign + test
uv run python Classification-STDP/eval_stdp.py --N 100

# 3. plots
uv run python Classification-STDP/plot_weights.py    --N 100
uv run python Classification-STDP/plot_connectome.py --N 100

# 4. learning curve (re-evaluates each snapshot in W_history)
uv run python Classification-STDP/learning_curve.py  --N 100

# 5. full N-sweep with combined learning-curve overlay (long-running)
uv run python Classification-STDP/compare_N.py --Ns 50 100 150
```

`--epochs 1` over the full 60 000 training images is enough for STDP to
converge; the receptive fields stabilise well before the second pass.
