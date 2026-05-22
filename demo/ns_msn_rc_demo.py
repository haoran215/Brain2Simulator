"""
ns_msn_rc_demo.py
=================

Reservoir Computing (RC) demo — left vs right pattern classification
with 20 MSN neurons.

Architecture
────────────
  [PoissonGroup L]  ──exc──► neurons  0– 9 ─┐
                                              │ recurrent weights W  →  linear readout
  [PoissonGroup R]  ──exc──► neurons 10–19 ─┘   y ∈ {left=0, right=1}

Left pattern  : neurons  0– 9 receive high-rate Poisson input.
Right pattern : neurons 10–19 receive high-rate Poisson input.
All neurons   : low-rate background noise (keeps Vm alive).

Key question: how do synaptic weights work in RC?
─────────────────────────────────────────────────
There are TWO sets of weights, with DIFFERENT roles:

  Reservoir weights W_rec  (this file, §3)
  ─────────────────────────────────────────
  • Start RANDOM — not a fixed scalar.
  • In standard RC they are FROZEN after init; only the readout is trained.
  • Optionally PLASTIC via STDP: the network self-organises without labels.
  • USE_STDP flag below switches between the two modes.
  • How to set: syn.w = np.random.gamma(...) * amp   (see §3)
  • How to make plastic: write a Synapses object with Apre/Apost traces (§3b)

  Readout weights W_out  (this file, §8)
  ───────────────────────────────────────
  • Trained OFFLINE with labelled examples — ridge regression on spike counts.
  • Never inside Brian2; pure numpy.
  • Formula: W_out = (X'X + λI)^{-1} X'y   (one matrix solve, no loops)
  • These are what the 'learning' IS in an RC system.

Trial structure (ms)
─────────────────────
  0 ──── 400 ms : stimulus on  (Poisson at stim_rate Hz)
  400 ── 800 ms : ITI          (background only; reservoir settles)
  Feature window: 200–400 ms  (avoid Is2 onset transient)
"""
#%%
import os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from brian2 import *

from msn_neuron  import MSNParams, make_msn
from msn_synapse import SynapseParams, make_synapse

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 0. Config                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
USE_STDP = os.environ.get('USE_STDP', 'True') == 'True'   # env-overridable; True → reservoir weights self-update via STDP (slower)

prefs.codegen.target = 'cython'  # use Cython for speed (pip install Cython if you don't have it)
start_scope()
defaultclock.dt = 10 * us

np.random.seed(55)

N           = 20
N_LEFT      = 10     # neurons 0–9   = "left" neurons
N_RIGHT     = 10     # neurons 10–19 = "right" neurons
n_train     = 20     # 10 left + 10 right
n_test      = 10     # 5  left + 5  right
stim_ms     = 500    # ms stimulus window per trial
iti_ms      = 500    # ms inter-trial interval (let Is2 decay to baseline)
trial_ms    = stim_ms + iti_ms
feat_start  = 200    # ms — feature window start (give Is2 time to build)
feat_end    = stim_ms
stim_rate   = 100.0  # Hz — Poisson rate for the stimulated side
bg_rate     =   5.0  # Hz — background noise (keep well below threshold)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. Load parameters                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
params = MSNParams.from_json(os.path.join(_REPO, 'configs/neuron_default.json'))
I_min, I_max = params.operating_window()

syn_exc_p = SynapseParams.from_json(os.path.join(_REPO, 'configs/synapse_default.json'), key='exc')
print(params.summary(), '\n')
print(f"I_min = {I_min*1e6:.2f} µA")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. Build reservoir (20 MSN neurons, all subthreshold)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Three named inlets — one per Synapses writer (Brian2 allows only one writer
# per (inlet, group) pair).  I_exc_rec: recurrent, I_exc_L / I_exc_R: inputs.
reservoir = make_msn(N=N, params=params,
                     exc_inlets=('I_exc_rec', 'I_exc_L', 'I_exc_R'),
                     name='reservoir')
# Tonic bias: subthreshold on its own, but crosses I_min when Poisson input
# arrives.  Rule: I_0 + <Is2_bg> < I_min  AND  I_0 + Is2(200ms) > I_min.
# I_0 = 0.7 × I_min, bg gives ~0.5 µA Is2, so total_bg ≈ 0.705 × I_min < I_min ✓
reservoir.I_0 = 0.7 * I_min * amp

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. Recurrent synaptic weights — two options                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if not USE_STDP:
    # ─── (a) STATIC random weights ────────────────────────────────────────────
    # Step 1: make_synapse creates the Synapses object with w as a per-edge
    #         state variable (model='w : amp').
    # Step 2: OVERRIDE w with a random array — this is the key answer to
    #         "how to define weights that are not a fixed value".
    #
    # Gamma distribution: all positive, right-skewed, mean = shape * scale.
    #   shape=2, scale=1.5 µA → mean ≈ 3 µA, most weights small, a few large.

    syn_rec = make_synapse(
        source  = reservoir,
        target  = reservoir,
        params  = SynapseParams(kind='exc', weight=syn_exc_p.weight,
                                tau_s1=syn_exc_p.tau_s1, tau_s2=syn_exc_p.tau_s2,
                                target_var='I_exc_rec'),
        connect = 'rand() < 0.15 and i != j',   # ~15% sparse, no self-loops
        name    = 'syn_rec',
    )
    n_syn = len(syn_rec.w)
    # Very weak weights so cross-group recurrent excitation stays sub-threshold.
    # Mean = shape × scale = 2 × 0.15 µA = 0.3 µA.  Even 3 incoming connections
    # at 5 Hz give only 0.3 × 3 × 5 × 0.2 = 0.9 µA extra drive — well below the
    # ~4.5 µA margin before the off-side crosses I_min.
    syn_rec.w = np.random.gamma(shape=2.0, scale=0.15e-6, size=n_syn) * amp
    print(f"Static RC:  {n_syn} recurrent synapses, "
          f"mean w = {np.mean(syn_rec.w/amp)*1e6:.3f} µA")

else:
    # ─── (b) PLASTIC weights — STDP ───────────────────────────────────────────
    # When you want the NETWORK to update its own weights, do NOT use
    # make_synapse (which only supports fixed w).  Write Synapses directly
    # with trace variables Apre / Apost.
    #
    # Rule (Bi & Poo 1998 STDP):
    #   post fires after  pre  → LTP: w += Apre  * lr_plus
    #   pre  fires after  post → LTD: w -= Apost * lr_minus
    #
    # Brian2 units: Apre, Apost are dimensionless (: 1).
    #               lr_plus, lr_minus must have units amp so that
    #               Apre * lr_plus has units amp and can be added to w.

    tau_pre   = 20  * ms
    tau_post  = 20  * ms
    lr_plus   = 0.3e-6 * amp    # LTP step (amps)
    lr_minus  = 0.3e-6 * amp    # LTD step (amps)
    w_max     = 15e-6  * amp

    syn_rec = Synapses(
        reservoir, reservoir,
        model = '''
            w                    : amp
            dIs1/dt = -Is1 / tau_s1                : amp (clock-driven)
            dIs2/dt = (-Is2 + Is1) / tau_s2        : amp (clock-driven)
            I_exc_rec_post = Is2                   : amp (summed)
            dApre/dt  = -Apre  / tau_pre           : 1 (event-driven)
            dApost/dt = -Apost / tau_post          : 1 (event-driven)
        ''',
        on_pre  = '''
            Is1   += w
            Apre  += 1
            w      = clip(w - Apost * lr_minus, 0*amp, w_max)
        ''',
        on_post = '''
            Apost += 1
            w      = clip(w + Apre  * lr_plus,  0*amp, w_max)
        ''',
        method = 'euler',   # NOT 'exact': the cascade's exact solver divides
                            # by (tau_s2 - tau_s1), which is 0 when they're equal.
        namespace = dict(
            tau_s1=syn_exc_p.tau_s1 * second,
            tau_s2=syn_exc_p.tau_s2 * second,
            tau_pre=tau_pre, tau_post=tau_post,
            lr_plus=lr_plus, lr_minus=lr_minus, w_max=w_max,
        ),
        name = 'syn_rec',
    )
    syn_rec.connect(condition='rand() < 0.15 and i != j')
    syn_rec.Is1 = 0 * amp
    syn_rec.Is2 = 0 * amp

    # Same random init as the static case — plasticity STARTS from random,
    # not from a fixed value.
    n_syn = len(syn_rec.w)
    syn_rec.w = np.random.gamma(shape=2.0, scale=0.15e-6, size=n_syn) * amp
    print(f"STDP RC:  {n_syn} recurrent synapses, "
          f"mean w_init = {np.mean(syn_rec.w/amp)*1e6:.3f} µA")

w_init = np.array(syn_rec.w / amp).copy()   # snapshot for later comparison

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. Input layer (PoissonGroups)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Two independent Poisson sources, one per side.
# Rates start at 0; set before each trial in the training loop.
input_L = PoissonGroup(N_LEFT,  rates=0*Hz, name='input_L')
input_R = PoissonGroup(N_RIGHT, rates=0*Hz, name='input_R')

# Input weight — needs to satisfy two constraints:
#   (1) ON side fires by feature window start (200 ms):
#       I_0 + Is2(200ms) > I_min
#       0.7·I_min + 0.264·(Iw·stim_rate·tau_s2) > I_min
#       Iw > 0.3·I_min / (0.264·stim_rate·tau_s2)
#          = 0.3×14.99e-6 / (0.264×100×0.2) = 0.854 µA  → use 2 µA ✓
#
#   (2) OFF side stays silent during stimulus:
#       I_0 + <Is2_bg_ss> < I_min
#       0.7·I_min + Iw·bg_rate·tau_s2 < I_min
#       Iw < 0.3·I_min / (bg_rate·tau_s2) = 0.3×14.99e-6/(5×0.2) = 4.5 µA ✓
# Each input synapse writes to its own named inlet — syn_in_L → I_exc_L,
# syn_in_R → I_exc_R — to satisfy Brian2's one-summed-writer-per-inlet rule.
inp_w   = 2e-6
inp_tau = syn_exc_p.tau_s2   # reuse exc cascade timescale for both inputs
I_0_val = 0.7 * I_min
print(f"\nInput Iw = {inp_w*1e6:.1f} µA,  "
      f"I_0 = {I_0_val*1e6:.2f} µA ({I_0_val/I_min*100:.0f}% of I_min)")
print(f"ON-side  at 200ms: {I_0_val*1e6 + 0.264*(inp_w*stim_rate*inp_tau)*1e6:.1f} µA "
      f"({'above' if I_0_val + 0.264*inp_w*stim_rate*inp_tau > I_min else 'BELOW'} I_min)")
print(f"OFF-side bg ss:    {I_0_val*1e6 + inp_w*bg_rate*inp_tau*1e6:.1f} µA "
      f"({'above' if I_0_val + inp_w*bg_rate*inp_tau > I_min else 'below'} I_min)\n")

# input_L neuron i  →  reservoir neuron i         (0 to N_LEFT-1)
# input_R neuron i  →  reservoir neuron i+N_LEFT   (N_LEFT to N-1)
syn_in_L = make_synapse(input_L, reservoir,
                        SynapseParams(kind='exc', weight=inp_w,
                                      tau_s1=syn_exc_p.tau_s1, tau_s2=syn_exc_p.tau_s2,
                                      target_var='I_exc_L'),
                        connect='j == i',
                        name='syn_in_L')
syn_in_R = make_synapse(input_R, reservoir,
                        SynapseParams(kind='exc', weight=inp_w,
                                      tau_s1=syn_exc_p.tau_s1, tau_s2=syn_exc_p.tau_s2,
                                      target_var='I_exc_R'),
                        connect='j == i + %d' % N_LEFT,
                        name='syn_in_R')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. Monitors                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sp_all = SpikeMonitor(reservoir)   # records (t, i) for every spike

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 6. Trial schedule                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Labels: 0=left, 1=right.  Interleaved so STDP sees alternating patterns.
labels_train = np.array([0, 1] * (n_train // 2))
labels_test  = np.array([0, 1] * (n_test  // 2))
all_labels   = np.concatenate([labels_train, labels_test])
n_total      = len(all_labels)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 7. Run all trials                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Continuous simulation: set rates, run, set rates to bg, run ITI.
# Time bookkeeping lets us extract per-trial spike counts afterwards.
t_windows = []   # (t_feat_start_ms, t_feat_end_ms) for each trial

print("─" * 60)
print(f"Running {n_total} trials × {trial_ms} ms  "
      f"= {n_total * trial_ms / 1000:.1f} s  "
      f"({n_total * trial_ms * 1000 // 10:,} steps) …")
print("─" * 60)

for k, label in enumerate(all_labels):
    t_trial_start = float(defaultclock.t / ms)

    # Set stimulus
    if label == 0:  # left
        input_L.rates = stim_rate * Hz
        input_R.rates = bg_rate   * Hz
    else:           # right
        input_L.rates = bg_rate   * Hz
        input_R.rates = stim_rate * Hz

    run(stim_ms * ms)   # stimulus window

    # ITI: silence stimulus, let Is2 / Vm decay
    input_L.rates = bg_rate * Hz
    input_R.rates = bg_rate * Hz
    run(iti_ms * ms)

    t_feat_s = t_trial_start + feat_start
    t_feat_e = t_trial_start + feat_end
    t_windows.append((t_feat_s, t_feat_e))

    if (k + 1) % 5 == 0:
        print(f"  Trial {k+1:2d}/{n_total}  label={'L' if label==0 else 'R'}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 8. Feature extraction — spike counts per trial per neuron               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
t_sp = np.array(sp_all.t / ms)
i_sp = np.array(sp_all.i)

X = np.zeros((n_total, N), dtype=float)
for k, (t0, t1) in enumerate(t_windows):
    mask = (t_sp >= t0) & (t_sp < t1)
    if mask.any():
        counts = np.bincount(i_sp[mask], minlength=N)
        X[k] = counts

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 9. Linear readout — ridge regression (no sklearn needed)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# y = +1 → right,  y = -1 → left
y_all = all_labels * 2.0 - 1.0    # {0,1} → {-1,+1}

X_train = X[:n_train]
y_train = y_all[:n_train]
X_test  = X[n_train:]
y_test  = y_all[n_train:]

# Ridge:  W_out = (X'X + λ * scale * I)^{-1} X'y
# Scale λ by the feature magnitude so the same λ works across experiments.
lam  = 1e-3
scale = np.max(X_train) ** 2 if X_train.max() > 0 else 1.0
A    = X_train.T @ X_train + lam * scale * np.eye(N)
b    = X_train.T @ y_train
w_out = np.linalg.solve(A, b)

y_pred_train = np.sign(X_train @ w_out)
y_pred_test  = np.sign(X_test  @ w_out)

acc_train = np.mean(y_pred_train == y_train)
acc_test  = np.mean(y_pred_test  == y_test)

print(f"\n{'─'*60}")
print(f"  Train accuracy : {acc_train*100:.0f}%  ({int(acc_train*n_train)}/{n_train})")
print(f"  Test  accuracy : {acc_test*100:.0f}%   ({int(acc_test*n_test)}/{n_test})")
print(f"{'─'*60}")

if acc_test < 0.7:
    print("  [NOTE] Low accuracy — neurons may not be firing enough. "
          "Try reducing I_min (lower Vth) or increasing inp_params.weight.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 10. Training diagram                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(15.8, 7.2))
fig.patch.set_facecolor('white')
gs_diag = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], hspace=0.24, wspace=0.18)

ax_a = fig.add_subplot(gs_diag[0, 0])
ax_b = fig.add_subplot(gs_diag[0, 1])
ax_c = fig.add_subplot(gs_diag[0, 2])
ax_d = fig.add_subplot(gs_diag[1, :])

for axp in (ax_a, ax_b, ax_c, ax_d):
    axp.set_axis_off()
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)

def box(axp, xy, w, h, text, fc='white', ec='0.2', text_color='0.1', fontsize=11, lw=1.4):
    patch = FancyBboxPatch(
    xy, w, h,
    boxstyle='round,pad=0.015,rounding_size=0.02',
    linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    axp.add_patch(patch)
    axp.text(xy[0] + w / 2, xy[1] + h / 2, text,
         ha='center', va='center', fontsize=fontsize,
         color=text_color, weight='bold')

def arrow(axp, p0, p1, color='0.25', text=None, text_pos=None, text_color='0.2', rad=0.0):
    style = '-|>'
    conn = f'arc3,rad={rad}' if rad else 'arc3'
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        connectionstyle=conn,
    )
    axp.add_patch(a)
    if text is not None and text_pos is not None:
        axp.text(
            text_pos[0],
            text_pos[1],
            text,
            ha='center',
            va='center',
            fontsize=9.2,
            color=text_color,
        )

# Panel A
box(ax_a, (0.04, 0.14), 0.26, 0.72, 'A\nInput signals', fc='#f7f7f7', ec='0.25', fontsize=12)
box(ax_a, (0.42, 0.64), 0.22, 0.12, 'Left trials', fc='#e8f1fb', ec='#2E86C1', text_color='#1f4e79', fontsize=9.2)
box(ax_a, (0.42, 0.43), 0.22, 0.12, 'Right trials', fc='#fdebe9', ec='#C0392B', text_color='#7f1d1d', fontsize=9.2)
box(ax_a, (0.42, 0.22), 0.22, 0.12, 'Background noise', fc='#f2f2f2', ec='0.55', text_color='0.25', fontsize=9.0)
arrow(ax_a, (0.30, 0.70), (0.42, 0.70), color='#2E86C1', text='Poisson input', text_pos=(0.36, 0.83), text_color='#2E86C1')
arrow(ax_a, (0.30, 0.49), (0.42, 0.49), color='#C0392B', text='Poisson input', text_pos=(0.36, 0.62), text_color='#C0392B')
arrow(ax_a, (0.30, 0.28), (0.42, 0.28), color='0.5', text='background', text_pos=(0.36, 0.12))
ax_a.text(0.5, 0.94, 'Input coding', ha='center', va='center', fontsize=11.5, weight='bold')

# Panel B
box(ax_b, (0.04, 0.14), 0.26, 0.72, 'B\nMSN reservoir', fc='#f7f7f7', ec='0.25', fontsize=12)
box(ax_b, (0.40, 0.66), 0.20, 0.11, '20 MSN neurons', fc='#eef8f5', ec='#16A085', text_color='#0b5345', fontsize=9.2)
box(ax_b, (0.40, 0.43), 0.20, 0.13, 'Random / STDP\nrecurrent weights', fc='#eef8f5', ec='#16A085', text_color='#0b5345', fontsize=8.8)
box(ax_b, (0.40, 0.20), 0.20, 0.13, 'Spike counts\nfeature vector X', fc='#eef8f5', ec='#16A085', text_color='#0b5345', fontsize=8.8)
arrow(ax_b, (0.30, 0.71), (0.40, 0.71), color='#16A085')
arrow(ax_b, (0.30, 0.49), (0.40, 0.49), color='#16A085')
arrow(ax_b, (0.30, 0.26), (0.40, 0.26), color='#16A085')
arrow(ax_b, (0.60, 0.53), (0.64, 0.53), color='#95a5a6', rad=0.55)
ax_b.text(0.5, 0.94, 'Reservoir dynamics', ha='center', va='center', fontsize=11.5, weight='bold')

# Panel C
box(ax_c, (0.04, 0.14), 0.26, 0.72, 'C\nReadout and output', fc='#f7f7f7', ec='0.25', fontsize=12)
box(ax_c, (0.40, 0.66), 0.20, 0.11, 'Ridge regression', fc='#fff4e6', ec='#D68910', text_color='#7a4d00', fontsize=9.2)
box(ax_c, (0.40, 0.43), 0.20, 0.13, 'W_out = (X^T X + λI)^-1\nX^T y', fc='#fff4e6', ec='#D68910', text_color='#7a4d00', fontsize=8.9)
box(ax_c, (0.40, 0.20), 0.20, 0.13, 'Left / Right\nclassification', fc='#fdecea', ec='#C0392B', text_color='#7f1d1d', fontsize=9.3)
arrow(ax_c, (0.30, 0.71), (0.40, 0.71), color='#D68910', text='offline fit', text_pos=(0.35, 0.84), text_color='#7a4d00')
arrow(ax_c, (0.30, 0.49), (0.40, 0.49), color='#D68910', text='predict', text_pos=(0.35, 0.62), text_color='#7a4d00')
ax_c.text(0.5, 0.94, 'Supervised decoding', ha='center', va='center', fontsize=11.5, weight='bold')

# Panel D: STDP mechanism + pipeline note
box(ax_d, (0.03, 0.20), 0.28, 0.54, 'If USE_STDP = True:\nBrian2 updates recurrent\nweights online', fc='#f7f7f7', ec='0.25', fontsize=10.0)
box(ax_d, (0.37, 0.20), 0.23, 0.54, 'on_pre\nApre += 1\nw -= Apost · lr_minus', fc='#eef8f5', ec='#16A085', text_color='#0b5345', fontsize=9.5)
box(ax_d, (0.65, 0.20), 0.23, 0.54, 'on_post\nApost += 1\nw += Apre · lr_plus', fc='#eef8f5', ec='#16A085', text_color='#0b5345', fontsize=9.5)
box(ax_d, (0.05, 0.06), 0.88, 0.08,
    'Default flow: collect reservoir activity first, then train the readout offline with ridge regression.  '
    'STDP is only for recurrent weights during the run.',
    fc='white', ec='0.85', text_color='0.2', fontsize=8.9, lw=1.0)
arrow(ax_d, (0.31, 0.48), (0.37, 0.48), color='#16A085', text='pre spike', text_pos=(0.34, 0.64), text_color='#0b5345')
arrow(ax_d, (0.60, 0.48), (0.65, 0.48), color='#16A085', text='post spike', text_pos=(0.625, 0.64), text_color='#0b5345')

diagram_path = 'demo/ns_msn_rc_training_diagram.png'
plt.savefig(diagram_path, dpi=200, bbox_inches='tight')
print(f"Training diagram saved → {diagram_path}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 11. Plot                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

# ─── Weight matrix ──────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
# Build dense N×N weight matrix from COO-like (pre_i, post_j, w_ij)
pre_idx  = np.array(syn_rec.i[:])
post_idx = np.array(syn_rec.j[:])
w_now    = np.array(syn_rec.w / amp)
W_mat = np.zeros((N, N))
W_mat[pre_idx, post_idx] = w_now * 1e6   # µA

im = ax.imshow(W_mat, cmap='hot', vmin=0,
               vmax=max(W_mat.max(), 1e-9), aspect='auto')
ax.axhline(N_LEFT - 0.5, color='cyan',  lw=1.5, ls='--')
ax.axvline(N_LEFT - 0.5, color='cyan',  lw=1.5, ls='--')
ax.set_xticks([N_LEFT//2, N_LEFT + N_RIGHT//2])
ax.set_xticklabels(['L neurons\n(0–9)', 'R neurons\n(10–19)'])
ax.set_yticks([N_LEFT//2, N_LEFT + N_RIGHT//2])
ax.set_yticklabels(['L', 'R'])
ax.set_title(f"Reservoir weight matrix W_rec (µA)\n"
             f"{'After STDP' if USE_STDP else 'Random init (frozen)'} — "
             f"{n_syn} synapses, mean={w_now.mean()*1e6:.2f} µA",
             fontweight='bold')
plt.colorbar(im, ax=ax, label='w (µA)')

# ─── Weight distribution (init vs final, only for STDP) ─────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.hist(w_init * 1e6, bins=20, alpha=0.6, color='steelblue', label='init')
if USE_STDP:
    ax.hist(w_now  * 1e6, bins=20, alpha=0.6, color='tomato',
            label='after STDP training')
ax.set_xlabel('w (µA)')
ax.set_ylabel('count')
ax.set_title('Recurrent weight distribution', fontweight='bold')
ax.legend(fontsize=9)

# ─── Spike raster (first 4 trials) ──────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
n_show  = min(4, n_total)
t_zoom  = n_show * trial_ms
mask_r  = t_sp < t_zoom
t_show  = t_sp[mask_r]
i_show  = i_sp[mask_r]

left_mask  = i_show < N_LEFT
right_mask = i_show >= N_LEFT
ax.scatter(t_show[left_mask],  i_show[left_mask],
           s=4, color='steelblue', alpha=0.8, label='Left neurons (0–9)')
ax.scatter(t_show[right_mask], i_show[right_mask],
           s=4, color='tomato',    alpha=0.8, label='Right neurons (10–19)')

# Shade feature windows
for k in range(n_show):
    t0_local = k * trial_ms + feat_start
    t1_local = k * trial_ms + feat_end
    color = 'steelblue' if all_labels[k] == 0 else 'tomato'
    ax.axvspan(t0_local, t1_local, alpha=0.12, color=color)
    ax.text((t0_local + t1_local) / 2, N - 0.5,
            'L' if all_labels[k] == 0 else 'R',
            ha='center', va='top', fontsize=9, fontweight='bold', color=color)

ax.set_xlim(0, t_zoom)
ax.set_ylim(-0.5, N - 0.5)
ax.axhline(N_LEFT - 0.5, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('t (ms)')
ax.set_ylabel('Neuron index')
ax.set_title(f'Spike raster — first {n_show} trials '
             f'(shaded = feature window 200–400 ms)', fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# ─── Feature scatter (first 2 PCs: left sum vs right sum) ───────────────────
ax = fig.add_subplot(gs[2, 0])
feat_L = X[:, :N_LEFT].sum(axis=1)
feat_R = X[:, N_LEFT:].sum(axis=1)

for split, label_val, color, marker in [
    (slice(None, n_train), y_train, 'k',       's'),
    (slice(n_train, None), y_test,  'dimgray',  'D'),
]:
    for cls, clr, mk, name in [(-1, 'steelblue', 'o', 'Left'),
                                 (+1, 'tomato',    'o', 'Right')]:
        sel = (y_all[split] == cls)
        ax.scatter(feat_L[split][sel], feat_R[split][sel],
                   color=clr, marker=marker, s=50, alpha=0.8,
                   label=f'{name} ({"train" if split.stop is None else "test"})' if color == 'k' else None)

# Decision boundary (w_out defines a hyperplane in N-dim; project onto L/R sums)
fL_range = np.array([feat_L.min() - 1, feat_L.max() + 1])
w_L = w_out[:N_LEFT].sum()
w_R = w_out[N_LEFT:].sum()
if abs(w_R) > 1e-9:
    fR_boundary = -w_L / w_R * fL_range
    ax.plot(fL_range, fR_boundary, 'k--', lw=1.5, label='decision boundary (projected)')

ax.set_xlabel('Spike count: left neurons (0–9)')
ax.set_ylabel('Spike count: right neurons (10–19)')
ax.set_title('Feature space — left vs right spike counts', fontweight='bold')
ax.legend(fontsize=8)

# ─── Accuracy bar ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
ax.bar(['Train', 'Test'], [acc_train * 100, acc_test * 100],
       color=['steelblue', 'tomato'], alpha=0.8, edgecolor='k')
ax.axhline(100, color='k',   ls='--', lw=1)
ax.axhline(50,  color='gray', ls=':',  lw=1, label='chance')
ax.set_ylim(0, 110)
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Readout accuracy\n'
             f'Train {acc_train*100:.0f}% · Test {acc_test*100:.0f}%',
             fontweight='bold')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

mode_str = 'STDP (plastic reservoir)' if USE_STDP else 'Static random reservoir'
fig.suptitle(
    f'MSN Reservoir Computing — Left / Right classification\n'
    f'{mode_str}  ·  {N} neurons  ·  {n_train} train + {n_test} test trials  ·  '
    f'ridge regression readout',
    fontsize=12, fontweight='bold', y=1.002)

out_path = 'demo/ns_msn_rc_demo_%s.png' % ('stdp' if USE_STDP else 'static')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Summary — weight initialisation and update patterns                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  RANDOM INIT (§3a)
#  -----------------
#  syn = make_synapse(...)          # creates Synapses with model='w : amp'
#  syn.w = np.random.gamma(2, 1.5e-6, len(syn.w)) * amp   # override here
# #  Alternative distributions:
#    np.random.uniform(0, 10e-6, ...)           # flat
#    np.random.lognormal(np.log(5e-6), 0.5, .) # log-normal
#    syn.w['i < 10'] = 8e-6 * amp              # subset assignment

# #  STDP PLASTICITY (§3b)
# #  ----------------------
# #  Use raw Synapses (NOT make_synapse) when you need state variables:
#    model  = 'w : amp / dApre/dt = ... / dApost/dt = ...'
#    on_pre  = 'Is1_exc_post += w; Apre += 1; w = clip(w - Apost*lr, ...)'
#    on_post = 'Apost += 1; w = clip(w + Apre*lr, ...)'
#  Weights self-update at every spike — no labels, no explicit training step.

# #  LINEAR READOUT (§9)
# #  --------------------
#  Readout weights W_out are NOT in Brian2.  Train once offline:
#    W_out = np.linalg.solve(X'X + λI, X'y)
#  X : (n_trials, N) spike-count matrix from the reservoir
#  y : (n_trials,)  labels {-1, +1}
#  Predict: y_hat = sign(X_new @ W_out)

# %%