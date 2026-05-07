"""
Direction Recognition Task
===========================
Recognises 8 movement directions (0°, 45°, ..., 315°) using the
memristor aLIF reservoir SNN in the RATE coding regime.

Encoding
--------
  8 Poisson input neurons, each tuned to one preferred direction θ_i.
  For a stimulus at angle θ:
      r_i(θ) = r_base + r_mod · max(0, cos(θ − θ_i))

  Active inputs fire at up to r_base + r_mod ≈ 100 Hz (rate regime).
  Inactive inputs fire at r_base = 20 Hz (baseline noise).

Reservoir
---------
  4 recurrently connected aLIF neurons receive all 8 inputs.
  Each reservoir neuron sees a DIFFERENT weighted combination of inputs
  (because recurrent weights are random and fixed), so each direction
  produces a unique spatiotemporal pattern of reservoir activity.

Readout
-------
  Features: spike counts of 4 reservoir neurons over the readout window.
  Classifier: multinomial logistic regression (sklearn).
  Train/test split: 75%/25% of n_trials × n_directions trials.

Run
---
  python run_demo.py --task direction
  or directly:
  python tasks/direction_recognition.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.decomposition   import PCA

from modules.network  import SNNNetwork
from modules.learning import ReservoirReadout
from modules.regime   import RegimeDetector
from brian2           import second


# =============================================================================
# Helpers
# =============================================================================
def cosine_rate(theta_deg: float, pref_deg: float,
                r_base: float, r_mod: float) -> float:
    """Cosine tuning: r = r_base + r_mod * max(0, cos(theta - pref))."""
    diff = np.deg2rad(theta_deg - pref_deg)
    return r_base + r_mod * max(0.0, np.cos(diff))


def run_trial(snn, direction_deg, task_cfg, learn_cfg):
    """
    Present one direction stimulus, return reservoir spike count vector.

    Steps
    -----
    1. Set each dir_i Poisson group to cosine-tuned rate.
    2. Run for stim_duration_ms.
    3. Extract spike counts in readout window [offset, stim_dur].
    4. Run gap (reset-like: set all inputs to r_base, advance time).

    Returns
    -------
    features : (n_reservoir,)  spike counts
    """
    n_dir     = task_cfg['n_directions']
    stim_ms   = task_cfg['stim_duration_ms']
    gap_ms    = task_cfg['gap_duration_ms']
    r_base    = task_cfg['r_base_Hz']
    r_mod     = task_cfg['r_mod_Hz']
    offset_ms = learn_cfg.get('readout_offset_ms', 0.0)

    pref_angles = np.arange(n_dir) * (360.0 / n_dir)   # 0, 45, ..., 315

    # --- Set stimulus rates ------------------------------------------
    for i in range(n_dir):
        rate = cosine_rate(direction_deg, pref_angles[i], r_base, r_mod)
        snn.set_poisson_rate(f'dir{i}', rate)

    # Record time before stim
    t_stim_start = float(snn.net.t / second)

    # --- Run stimulus period -----------------------------------------
    snn.run(stim_ms / 1000.0)

    # --- Extract spike counts in readout window ----------------------
    spike_i, spike_t = snn.get_spikes('reservoir')
    t_win_start = t_stim_start + offset_ms / 1000.0
    t_win_end   = t_stim_start + stim_ms   / 1000.0

    res_pop = snn.populations['reservoir']
    n_res   = res_pop.n

    features = ReservoirReadout.extract_spike_counts(
        spike_i, spike_t, n_res, t_win_start, t_win_end)

    # --- Gap period (baseline input, stabilise) ----------------------
    for i in range(n_dir):
        snn.set_poisson_rate(f'dir{i}', r_base)
    snn.run(gap_ms / 1000.0)

    return features


# =============================================================================
# Main task
# =============================================================================
def run_direction_task(config_path: str, plot_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    task_cfg    = cfg['task']
    learn_cfg   = cfg['learning']
    n_dir       = task_cfg['n_directions']
    n_trials    = task_cfg['n_trials']
    pref_angles = np.arange(n_dir) * (360.0 / n_dir)
    dir_labels  = [f"{int(a)}°" for a in pref_angles]

    # Regime check
    tau_s_ms = cfg['neuron_defaults']['tau_s1_ms']
    regime   = RegimeDetector(tau_s_ms)
    regime.report(cfg['neuron_defaults']['f_min_Hz'],
                  cfg['neuron_defaults']['f_max_Hz'],
                  cfg['neuron_defaults']['Iw_exc_uA'])

    # Build network
    print(f"\nBuilding network from {config_path} ...")
    snn = SNNNetwork(config_path)
    snn.summary()

    # Collect trials
    print(f"\nRunning {n_dir} directions × {n_trials} trials "
          f"({n_dir * n_trials} total) ...")
    X_all, y_all = [], []
    raster_ex    = {}   # store spike snapshot per direction for plotting

    for d_idx, theta in enumerate(pref_angles):
        for trial in range(n_trials):
            feat = run_trial(snn, theta, task_cfg, learn_cfg)
            X_all.append(feat)
            y_all.append(d_idx)

        # Save spikes for raster plot (snapshot after all trials of this direction)
        spike_i, spike_t = snn.get_spikes('reservoir')
        raster_ex[d_idx] = (spike_i.copy(), spike_t.copy())

        print(f"  Direction {theta:5.0f}°  "
              f"mean spikes/neuron = {np.mean(X_all[-n_trials:]):.1f}")

    X_all = np.array(X_all, dtype=np.float32)   # (n_dir*n_trials, n_res)
    y_all = np.array(y_all, dtype=int)

    # Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all,
        random_state=cfg['simulation']['seed'])

    readout = ReservoirReadout(n_classes=n_dir)
    train_acc = readout.train(X_tr, y_tr)
    test_acc  = readout.score(X_te, y_te)
    cm        = readout.confusion_matrix(X_te, y_te)

    print(f"\n  Train accuracy : {train_acc*100:.1f}%")
    print(f"  Test  accuracy : {test_acc*100:.1f}%")

    # PCA on all features
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(readout.scaler.transform(X_all))

    # =========================================================================
    # Plotting
    # =========================================================================
    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42)

    n_res    = snn.populations['reservoir'].n
    dir_cmap = plt.cm.hsv(np.linspace(0, 1, n_dir, endpoint=False))

    # ── Panel A: Cosine tuning curves ────────────────────────────────
    ax_tune = fig.add_subplot(gs[0, 0], projection='polar')
    theta_fine = np.linspace(0, 2*np.pi, 360)
    r_base = task_cfg['r_base_Hz']
    r_mod  = task_cfg['r_mod_Hz']
    for i, pref in enumerate(pref_angles):
        r = np.array([cosine_rate(np.rad2deg(t), pref, r_base, r_mod)
                      for t in theta_fine])
        ax_tune.plot(theta_fine, r, color=dir_cmap[i], lw=1.5,
                     label=f'{int(pref)}°', alpha=0.8)
    ax_tune.set_title('A — Cosine Tuning\n(input rates per direction)',
                       fontsize=9, fontweight='bold', pad=12)
    ax_tune.set_ylim(0, r_base + r_mod * 1.1)

    # ── Panel B: Example spike raster (four directions) ───────────────
    ax_rast = fig.add_subplot(gs[0, 1:])
    stim_ms  = task_cfg['stim_duration_ms']
    gap_ms   = task_cfg['gap_duration_ms']
    trial_ms = stim_ms + gap_ms

    n_show    = 4
    show_dirs = [0, 2, 4, 6]   # 0°, 90°, 180°, 270°

    for plot_idx, d_idx in enumerate(show_dirs):
        spike_i, spike_t = raster_ex[d_idx]
        t_block_end = (d_idx * n_trials + n_trials) * trial_ms / 1000.0
        t_block_st  = t_block_end - 2 * trial_ms / 1000.0
        mask  = (spike_t >= t_block_st) & (spike_t < t_block_end)
        t_rel = (spike_t[mask] - t_block_st) * 1000   # ms
        t_off = plot_idx * (2 * trial_ms + 20)
        for ni in range(n_res):
            y_off = plot_idx * (n_res + 1)
            nm    = spike_i[mask] == ni
            if nm.any():
                ax_rast.vlines(t_rel[nm] + t_off, y_off + ni - 0.4,
                                y_off + ni + 0.4,
                                color=dir_cmap[d_idx], lw=0.9)
        ax_rast.text(t_off + trial_ms, plot_idx * (n_res+1) + n_res*0.5,
                     f"{int(pref_angles[d_idx])}°",
                     ha='center', fontsize=8, color=dir_cmap[d_idx],
                     fontweight='bold')
        for tt in [0, trial_ms]:
            ax_rast.axvline(t_off + tt, color='gray', ls=':', lw=0.8, alpha=0.5)

    ax_rast.set_title('B — Reservoir Spike Raster  (4 example directions, 2 trials each)',
                       fontsize=9, fontweight='bold')
    ax_rast.set_xlabel('Time  (ms)'); ax_rast.set_ylabel('Neuron')
    ax_rast.set_xlim(-5, n_show * (2*trial_ms + 20) + 5)

    # ── Panel C: Mean spike count heatmap ────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 0])
    mean_counts = np.zeros((n_dir, n_res))
    for d_idx in range(n_dir):
        mask = y_all == d_idx
        mean_counts[d_idx] = X_all[mask].mean(axis=0)
    im = ax_heat.imshow(mean_counts, aspect='auto', cmap='viridis', origin='upper')
    ax_heat.set_xticks(range(n_res))
    ax_heat.set_xticklabels([f'N{i}' for i in range(n_res)])
    ax_heat.set_yticks(range(n_dir))
    ax_heat.set_yticklabels(dir_labels, fontsize=8)
    ax_heat.set_title('C — Mean Spike Count\nper reservoir neuron per direction',
                       fontsize=9, fontweight='bold')
    ax_heat.set_xlabel('Reservoir Neuron')
    ax_heat.set_ylabel('Direction')
    plt.colorbar(im, ax=ax_heat, label='Spike count')

    # ── Panel D: PCA of reservoir states ─────────────────────────────
    ax_pca = fig.add_subplot(gs[1, 1])
    for d_idx in range(n_dir):
        mask = y_all == d_idx
        ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=dir_cmap[d_idx], s=40, alpha=0.75,
                        label=dir_labels[d_idx], edgecolors='none')
    ax_pca.set_title('D — PCA of Reservoir Features\n(2D projection of spike count space)',
                      fontsize=9, fontweight='bold')
    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    ax_pca.legend(fontsize=7, loc='best', ncol=2, markerscale=1.2)

    # ── Panel E: Confusion matrix ─────────────────────────────────────
    ax_cm = fig.add_subplot(gs[1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im_cm = ax_cm.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax_cm.set_xticks(range(n_dir)); ax_cm.set_xticklabels(dir_labels, fontsize=7)
    ax_cm.set_yticks(range(n_dir)); ax_cm.set_yticklabels(dir_labels, fontsize=7)
    ax_cm.set_xlabel('Predicted Direction')
    ax_cm.set_ylabel('True Direction')
    ax_cm.set_title(f'E — Confusion Matrix\nTest accuracy = {test_acc*100:.1f}%',
                     fontsize=9, fontweight='bold')
    for i in range(n_dir):
        for j in range(n_dir):
            ax_cm.text(j, i, f'{cm_norm[i,j]:.2f}',
                        ha='center', va='center', fontsize=6.5,
                        color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.colorbar(im_cm, ax=ax_cm, label='Fraction')

    # ── Panel F: Per-direction accuracy bar ──────────────────────────
    ax_bar = fig.add_subplot(gs[2, :2])
    per_dir_acc = cm.diagonal() / cm.sum(axis=1)
    bars = ax_bar.bar(dir_labels, per_dir_acc * 100,
                       color=dir_cmap, edgecolor='k', linewidth=0.7)
    ax_bar.axhline(100/n_dir, color='red', ls='--', lw=1.5,
                   label=f'Chance = {100/n_dir:.0f}%')
    ax_bar.axhline(test_acc*100, color='navy', ls='--', lw=1.5,
                   label=f'Mean test acc = {test_acc*100:.1f}%')
    for bar, acc in zip(bars, per_dir_acc):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f'{acc*100:.0f}%',
                    ha='center', va='bottom', fontsize=8)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_xlabel('Direction'); ax_bar.set_ylabel('Accuracy  (%)')
    ax_bar.set_title('F — Per-Direction Recognition Accuracy',
                      fontsize=9, fontweight='bold')
    ax_bar.legend(fontsize=8)

    # ── Panel G: Regime diagram ───────────────────────────────────────
    ax_reg = fig.add_subplot(gs[2, 2])
    f_range   = np.linspace(10, 220, 500)
    isi_ratio = 1000 / (f_range * tau_s_ms)
    ax_reg.plot(f_range, isi_ratio, color='steelblue', lw=2.0,
                label='ISI / τ_s')
    ax_reg.axhline(1.0, color='k', ls=':', lw=1.2)
    ax_reg.axvline(1000/tau_s_ms, color='purple', ls='--', lw=1.5,
                   label=f'Crossover = {1000/tau_s_ms:.0f} Hz')
    ax_reg.fill_between(f_range, isi_ratio, 1,
                          where=isi_ratio > 1, alpha=0.10, color='red',
                          label='Temporal zone')
    ax_reg.fill_between(f_range, isi_ratio, 1,
                          where=isi_ratio < 1, alpha=0.10, color='green',
                          label='Rate zone')
    ax_reg.axvline(r_base + r_mod, color='darkgreen', ls='-', lw=1.2,
                   label=f'r_max = {r_base+r_mod:.0f} Hz (active input)')
    ax_reg.axvline(r_base, color='gray', ls='-', lw=1.2,
                   label=f'r_base = {r_base:.0f} Hz (inactive input)')
    ax_reg.set_xlabel('Firing rate (Hz)'); ax_reg.set_ylabel('ISI / τ_s')
    ax_reg.set_title(f'G — Coding Regime\nτ_s = {tau_s_ms:.0f} ms',
                      fontsize=9, fontweight='bold')
    ax_reg.legend(fontsize=7, loc='upper right')
    ax_reg.set_xlim(10, 220); ax_reg.set_ylim(0, 4)
    ax_reg.grid(alpha=0.2)

    fig.suptitle(
        f'Direction Recognition  —  8 directions × {n_trials} trials  '
        f'|  {n_res} reservoir neurons  |  rate coding regime (Is2)\n'
        f'Train acc = {train_acc*100:.1f}%    Test acc = {test_acc*100:.1f}%    '
        f'Chance = {100/n_dir:.0f}%',
        fontsize=12, fontweight='bold', y=1.01)

    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\nPlot saved → {plot_path}")
    return test_acc


# =============================================================================
if __name__ == '__main__':
    cfg_path  = os.path.join(os.path.dirname(__file__),
                              '..', 'config', 'direction_task.json')
    plot_path = os.path.join(os.path.dirname(__file__),
                              '..', 'direction_recognition.png')
    run_direction_task(cfg_path, plot_path)
