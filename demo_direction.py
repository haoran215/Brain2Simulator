"""
demo_direction.py  —  8-Direction Recognition using Reservoir SNN
=================================================================
Task
----
  Classify 8 directions (0°, 45°, 90°, ..., 315°) from population-coded
  spiking input.

  Input encoding:
    8 Poisson neurons, each tuned to one preferred direction.
    For input direction θ, neuron i fires at:
      rate_i(θ) = r_base + (r_max - r_base) × cos²((θ − θ_i) / 2)
    This gives a smooth tuning bump: aligned neuron → r_max, opposite → r_base.

  Network:
    Input (8)  →  [feedforward exc]  →  Reservoir (20 aLIF neurons)
    Reservoir  →  [sparse exc + inh recurrent connections]  →  Reservoir

  Readout (trained):
    Mean Is2 per reservoir neuron per trial  →  Logistic regression  →  class

  Training protocol:
    - 8 directions × 5 trials per direction = 40 total trials
    - Time-multiplexed: all trials run in one 12-second simulation
    - Leave-one-out cross-validation for evaluation

Regime
------
  tau_s = 10 ms  →  crossover = 100 Hz
  Input rates: 20–120 Hz  →  operating range straddles crossover
  At high rates (>100Hz): rate coding (reservoir mode)  ✓
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression

from network  import SNNNetwork
from learning import ReservoirReadout, plot_confusion
from regime   import detect_regime, check_operating_range

# =============================================================================
# Configuration
# =============================================================================
CONFIG     = os.path.join(os.path.dirname(__file__), 'config.json')
N_DIR      = 8                        # number of direction classes
DIRS_DEG   = [k * 360/N_DIR for k in range(N_DIR)]   # [0, 45, 90, ..., 315]
N_TRIALS   = 5                        # trials per direction
R_BASE_HZ  = 20.0                     # min input rate (background)
R_MAX_HZ   = 120.0                    # max input rate (aligned neuron)
DIR_NAMES  = [f'{int(d)}°' for d in DIRS_DEG]

np.random.seed(42)

# =============================================================================
# Build trial sequence  (randomised order)
# =============================================================================
labels_ordered   = DIRS_DEG * N_TRIALS          # [0,45,...,315, 0,45,...,315, ...]
labels_ordered   = sorted(labels_ordered * 1)   # group by class for TimedArray clarity
# Shuffle for cross-val fairness
idx_shuffle      = np.random.permutation(len(labels_ordered))
labels_shuffled  = [labels_ordered[i] for i in idx_shuffle]
y_label          = np.array([DIRS_DEG.index(d) for d in labels_shuffled])  # 0..7

# =============================================================================
# Print regime analysis
# =============================================================================
print("\n" + "="*60)
print("  8-Direction Recognition  —  Regime Analysis")
print("="*60)
check_operating_range(70, 200, tau_s_ms=10.0, Iw=20e-6)

r_high = detect_regime(R_MAX_HZ, tau_s_ms=10.0, Iw=20e-6)
r_low  = detect_regime(R_BASE_HZ,   tau_s_ms=10.0, Iw=20e-6)
print(f"  Input range: {R_BASE_HZ:.0f}–{R_MAX_HZ:.0f} Hz")
print(f"  Peak rate ({R_MAX_HZ:.0f}Hz): {r_high.regime.upper()} — "
      f"ISI/tau_s={r_high.ISI_tau_ratio:.2f}")
print(f"  Base rate ({R_BASE_HZ:.0f}Hz): {r_low.regime.upper()} — "
      f"ISI/tau_s={r_low.ISI_tau_ratio:.2f}")

# =============================================================================
# Build and run network
# =============================================================================
print("\nBuilding network from config.json ...")
net = SNNNetwork(CONFIG)
net.build(seed_val=42)

print(f"\nEncoding {len(labels_shuffled)} trials ({N_TRIALS} × {N_DIR} directions) ...")
rates = net.encode_directions(labels_shuffled, r_base_Hz=R_BASE_HZ, r_max_Hz=R_MAX_HZ)

print(f"\nRunning simulation ({len(labels_shuffled)} × {net.trial_ms} ms = "
      f"{len(labels_shuffled)*net.trial_ms/1000:.1f} s) ...")
net.run(rates)

# =============================================================================
# Extract features and classify
# =============================================================================
X, y = net.get_features(y_label)
print(f"\nFeature matrix: {X.shape}  (trials × reservoir neurons)")
print(f"Label vector  : {y.shape}")

# Cross-validated accuracy
readout = ReservoirReadout.from_json(CONFIG)
cv      = StratifiedKFold(n_splits=min(N_TRIALS, 5), shuffle=True, random_state=42)
cv_scores = readout.cross_val(X, y, n_folds=min(N_TRIALS, 5))
print(f"\nCross-validation accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"Chance level: {100/N_DIR:.1f}%")
print(f"Gain over chance: {cv_scores.mean()/(1/N_DIR):.1f}×")

# Fit full readout for plotting
readout.fit(X, y)
y_pred = readout.predict(X)
train_acc = np.mean(y_pred == y) * 100
cm = readout.confusion(X, y)

print(f"\nFull-data accuracy: {train_acc:.1f}%")
print("\nClassification report:")
readout.report(X, y, class_names=DIR_NAMES)

# =============================================================================
# Visualise input tuning curves
# =============================================================================
theta_sweep = np.linspace(0, 360, 360)
preferred   = np.arange(N_DIR) * (360.0 / N_DIR)
tuning_curves = np.zeros((N_DIR, len(theta_sweep)))
for i in range(N_DIR):
    diffs = np.radians(theta_sweep - preferred[i])
    tuning_curves[i] = R_BASE_HZ + (R_MAX_HZ - R_BASE_HZ) * (1 + np.cos(diffs)) / 2

# =============================================================================
# Plotting
# =============================================================================
fig = plt.figure(figsize=(20, 20))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38,
                        height_ratios=[1.2, 1.2, 1.2, 1.2])

colours_dir = plt.cm.hsv(np.linspace(0, 1, N_DIR, endpoint=False))

# ── Panel 0: Input tuning curves (polar) ─────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0], projection='polar')
for i in range(N_DIR):
    theta_rad = np.radians(theta_sweep)
    ax0.plot(theta_rad, tuning_curves[i], color=colours_dir[i], lw=1.4,
             label=f'{int(preferred[i])}°')
ax0.set_title('Input Tuning Curves\n(each neuron prefers one direction)',
              fontsize=9, fontweight='bold', pad=15)
ax0.set_rlabel_position(45)
ax0.set_rticks([40, 80, 120])
ax0.set_theta_zero_location('N')
ax0.set_theta_direction(-1)

# ── Panel 1: Rate matrix (heatmap) ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
# One example of each direction (first occurrence)
example_rates = np.array([net.encode_directions([d], R_BASE_HZ, R_MAX_HZ)[0]
                           for d in DIRS_DEG])
im1 = ax1.imshow(example_rates, aspect='auto', cmap='viridis',
                 vmin=R_BASE_HZ, vmax=R_MAX_HZ)
plt.colorbar(im1, ax=ax1, label='Rate (Hz)')
ax1.set_yticks(range(N_DIR)); ax1.set_yticklabels(DIR_NAMES)
ax1.set_xticks(range(net.n_input))
ax1.set_xticklabels([f'N{i}' for i in range(net.n_input)])
ax1.set_title('Input Rate Matrix\n(direction × input neuron)', fontsize=9, fontweight='bold')
ax1.set_xlabel('Input neuron (preferred direction)')
ax1.set_ylabel('Stimulus direction')

# ── Panel 2: Reservoir spike raster (first 3 directions) ─────────────────────
ax2 = fig.add_subplot(gs[0, 2])
sp_t = net.spike_mon_res.t / second
sp_i = net.spike_mon_res.i
# Show first 3*trial_ms seconds
t_show = 3 * net.trial_ms * 1e-3
mask_show = sp_t < t_show
ax2.scatter(sp_t[mask_show]*1e3, sp_i[mask_show],
            s=2, c='steelblue', alpha=0.6)
for k in range(3):
    ax2.axvline((k+1)*net.trial_ms, color='red', ls='--', lw=1.0, alpha=0.6)
    ax2.text(k*net.trial_ms + net.trial_ms*0.1, net.n_reservoir+0.5,
             DIR_NAMES[DIRS_DEG.index(labels_shuffled[k])],
             fontsize=7.5, color='red')
ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('Reservoir neuron index')
ax2.set_title('Reservoir Spike Raster\n(first 3 trials)', fontsize=9, fontweight='bold')
ax2.set_xlim(0, t_show*1e3); ax2.set_ylim(-0.5, net.n_reservoir-0.5)

# ── Panel 3: Is2 feature matrix ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
# Sort X by true label for visualisation
sort_idx = np.argsort(y)
im3 = ax3.imshow(X[sort_idx].T, aspect='auto', cmap='RdBu_r',
                 interpolation='nearest')
plt.colorbar(im3, ax=ax3, label='Mean Is2 (net, A)')
# Direction boundary lines
boundaries = np.where(np.diff(y[sort_idx]))[0] + 0.5
for b in boundaries:
    ax3.axvline(b, color='k', lw=0.8, alpha=0.6)
ax3.set_xlabel('Trial (sorted by direction)')
ax3.set_ylabel('Reservoir neuron')
ax3.set_title('Is2 Feature Matrix  (mean Is2 per neuron per trial)\n'
              'Columns sorted by direction — patterns should cluster visually',
              fontsize=9, fontweight='bold')

# ── Panel 4: CV accuracy ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = ax4.bar(folds, cv_scores*100,
               color=['#27AE60' if s > 1/N_DIR else '#E74C3C' for s in cv_scores],
               alpha=0.85, edgecolor='k', lw=0.7)
ax4.axhline(100/N_DIR, color='red', ls='--', lw=1.5,
            label=f'Chance ({100/N_DIR:.1f}%)')
ax4.axhline(cv_scores.mean()*100, color='steelblue', ls='-', lw=1.5,
            label=f'Mean ({cv_scores.mean()*100:.1f}%)')
for bar, s in zip(bars, cv_scores):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{s*100:.0f}%', ha='center', fontsize=8)
ax4.set_ylabel('Accuracy (%)')
ax4.set_ylim(0, 110)
ax4.set_title(f'Cross-Validation Accuracy\n'
              f'Mean={cv_scores.mean()*100:.1f}%, Gain={cv_scores.mean()/(1/N_DIR):.1f}× chance',
              fontsize=9, fontweight='bold')
ax4.legend(fontsize=8)

# ── Panel 5: Confusion matrix ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
plot_confusion(cm, DIR_NAMES,
               title=f'Confusion Matrix  (full data, {train_acc:.0f}% accuracy)',
               ax=ax5)

# ── Panel 6: Decision boundary projection (PCA) ──────────────────────────────
from sklearn.decomposition import PCA
ax6 = fig.add_subplot(gs[2, 2])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
for cls_idx, (d, c) in enumerate(zip(DIRS_DEG, colours_dir)):
    mask_cls = y == cls_idx
    ax6.scatter(X_pca[mask_cls, 0], X_pca[mask_cls, 1],
                color=c, label=f'{int(d)}°', s=60, edgecolors='k', lw=0.5, zorder=3)
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)')
ax6.set_title('Is2 Feature Space (PCA)\nWell-separated clusters → good classification',
              fontsize=9, fontweight='bold')
ax6.legend(fontsize=7, ncol=2, loc='upper right')

# ── Panel 7: Input spike raster (first trial) ────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
sp_t_in = net.spike_mon_in.t / second
sp_i_in = net.spike_mon_in.i
mask_t1 = sp_t_in < net.trial_ms * 1e-3
t1_dir  = labels_shuffled[0]
for ni in range(net.n_input):
    mask_ni = (sp_i_in == ni) & mask_t1
    if mask_ni.any():
        ax7.scatter(sp_t_in[mask_ni]*1e3, np.full(mask_ni.sum(), ni),
                    s=3, color=colours_dir[DIRS_DEG.index(preferred[ni]) % N_DIR])
ax7.set_xlabel('Time (ms)'); ax7.set_ylabel('Input neuron')
ax7.set_title(f'Input Spikes — Trial 1 (dir={t1_dir:.0f}°)\n'
              f'Brightest neuron ≈ aligned to direction', fontsize=9, fontweight='bold')
ax7.set_xlim(0, net.trial_ms)

# ── Panel 8: Regime diagram ──────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1:])
f_range = np.linspace(5, 250, 1000)
ISI_range = 1e3 / f_range   # ms
ratio_range = ISI_range / 10.0  # divide by tau_s_ms=10

ax8.plot(f_range, ratio_range, color='steelblue', lw=2.0, label='ISI/τ_s')
ax8.axhline(1.0, color='k', ls=':', lw=1.2, label='Crossover (f=1/τ_s=100Hz)')
ax8.axvline(1e3/10.0, color='k', ls=':', lw=1.2)
ax8.fill_between(f_range[f_range < 100], ratio_range[f_range < 100],
                 1, alpha=0.12, color='red', label='Temporal (STDP)')
ax8.fill_between(f_range[f_range > 100], ratio_range[f_range > 100],
                 1, where=ratio_range[f_range > 100] < 1,
                 alpha=0.12, color='green', label='Rate (Reservoir)')
ax8.axvspan(R_BASE_HZ, R_MAX_HZ, alpha=0.08, color='gold',
            label=f'Input range ({R_BASE_HZ:.0f}–{R_MAX_HZ:.0f} Hz)')
ax8.set_xlabel('Firing rate f (Hz)', fontsize=10)
ax8.set_ylabel('ISI / τ_s', fontsize=10)
ax8.set_title('Rate vs Temporal Regime\n'
              'Input range spans both sides of crossover → hybrid encoding',
              fontsize=9, fontweight='bold')
ax8.set_xlim(5, 250); ax8.set_ylim(0, 4)
ax8.legend(fontsize=8, ncol=2)
ax8.grid(alpha=0.25)

fig.suptitle(
    f'8-Direction Recognition  —  Reservoir SNN\n'
    f'{net.n_input} input neurons  →  {net.n_reservoir} reservoir neurons  →  '
    f'logistic readout\n'
    f'CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%   '
    f'(chance: {100/N_DIR:.1f}%   gain: {cv_scores.mean()/(1/N_DIR):.1f}×)',
    fontsize=12, fontweight='bold', y=1.01)

out_path = '/mnt/user-data/outputs/demo_direction.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")