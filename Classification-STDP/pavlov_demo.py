"""
pavlov_demo.py
==============
Toy Pavlov's-dog STDP demo — sanity check that the learning rule used in
train_stdp.py wires up correctly on a tiny network, with no MNIST.

Network
-------
    10 Poisson "pre"        →  10 MSN "post"
       0..4 = bell  (CS)        ("salivation")
       5..9 = food  (US)

All 100 pre→post synapses are plastic (same Diehl-Cook STDP rule used in
train_stdp.py). Bell→post weights start weak; food→post weights start
strong — the unconditioned bell-doesn't-do-anything / food-makes-you-
salivate reflex.

Protocol
--------
Each trial (500 ms):
      0 ─── 100 ─── 200 ─── 500 ms
      │ bell │ bell + food │  rest │
                │
                └─ CS precedes & overlaps US — STDP grows bell→post.

Every TEST_EVERY training trials we run a *test trial* (bell only) and
count post spikes. Initially this should be near zero; after pairing,
bell alone elicits salivation.

Output: pavlov_demo.png with three panels
  (a) weight matrix before vs after training
  (b) mean weight evolution (bell synapses vs food synapses)
  (c) learning curve — post spikes during bell-only test trials
"""

from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from brian2 import (
    Network, PoissonGroup, Synapses, SpikeMonitor,
    defaultclock, second, us, Hz, amp, seed,
)
from matplotlib.patches import Rectangle
from msn_neuron import MSNParams, make_msn


# ── network sizes ────────────────────────────────────────────────────────────
N_PRE     = 10
N_POST    = 10
N_BELL    = 5      # pre indices 0..4
# food = pre indices 5..9

# ── STDP (matches train_stdp.py) ─────────────────────────────────────────────
W_MAX     = 1.0
W_UNIT    = 3.0e-6     # A per unit w; bumped from train_stdp's 1e-7 because
                       # only 5 pre fire per stimulus here vs. ~150 in MNIST.
                       # Sized so 5 food spikes at 50 Hz with w=0.7 puts the
                       # MSN cascade Is2 into the [I_min, I_max] window.
ETA_POST  = 5e-4       # tuned for ~100 trials; larger values saturate by trial 4
ETA_PRE   = 1e-4
X_TAR     = 0.2
MU        = 0.5
TAU_PRE   = 20e-3
TAU_POST  = 20e-3

# ── MSN synaptic-cascade time constant ───────────────────────────────────────
# Project default is 200 ms, tuned for the 350 ms MNIST presentation. For
# the toy 10-pre demo we shorten to 100 ms so Is2 reaches a useful fraction
# of Is1_ss within a sub-second pairing window.
TAU_S     = 100e-3

# ── tonic bias on post ───────────────────────────────────────────────────────
# I_min ≈ Vth/(Rm_hi+Ra) ≈ 15 µA. Sit each post neuron a hair below it so
# synaptic input only needs to provide a few µA to evoke spikes — without
# this, 5 pre at 50 Hz can't drive the MSN through the cascade in one trial.
I_0_BIAS  = 10e-6      # A — sub-threshold so bell at w=0.05 stays silent

# ── initial weights ──────────────────────────────────────────────────────────
W_BELL_0  = 0.05       # CS → response, weak
W_FOOD_0  = 0.70       # US → response, strong (unconditioned reflex)

# ── stimulus ─────────────────────────────────────────────────────────────────
RATE_BELL = 50.0       # Hz, while bell is on
RATE_FOOD = 50.0       # Hz, while food is on

T_BELL_ALONE = 0.05    # s — bell ramps up first (CS precedes US)
T_PAIR       = 0.30    # s — bell + food together (driving post spikes)
T_REST       = 0.40    # s — quiet, traces decay (>3·tau_s)

T_TEST_BELL  = 0.30    # s — bell-only test pulse
T_TEST_REST  = 0.40    # s

# ── protocol ─────────────────────────────────────────────────────────────────
N_TRAIN     = 100
TEST_EVERY  = 5

DT          = 100 * us
SEED        = 0


# ── build ────────────────────────────────────────────────────────────────────
def build():
    seed(SEED)
    np.random.seed(SEED)
    defaultclock.dt = DT

    # Cm = 0.05 µF lifts MSN max rate to ~200 Hz so STDP gets enough post-
    # spikes per pairing window. tau_s now lives on the synapse.
    params = MSNParams(Cm=0.05e-6)

    P = PoissonGroup(N_PRE, rates=np.zeros(N_PRE) * Hz, name='pre')
    G = make_msn(params=params, N=N_POST, name='post')
    G.I_0 = I_0_BIAS * amp

    # STDP synapse with cascade in the model block.
    stdp_model = '''
        dIs1/dt   = -Is1 / tau_s1                : amp (clock-driven)
        dIs2/dt   = (-Is2 + Is1) / tau_s2        : amp (clock-driven)
        I_exc_post = Is2                         : amp (summed)
        w         : 1
        dapre/dt  = -apre /tau_pre  : 1 (event-driven)
        dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    on_pre = '''
        Is1 += w * w_unit
        apre += 1
        w = clip(w - eta_pre * apost, 0, w_max)
    '''
    on_post = '''
        apost += 1
        w = clip(w + eta_post * apre - eta_post * x_tar * w**mu, 0, w_max)
    '''
    syn = Synapses(
        P, G,
        model=stdp_model, on_pre=on_pre, on_post=on_post,
        method='euler',
        namespace=dict(
            tau_s1   = TAU_S    * second,
            tau_s2   = TAU_S    * second,
            tau_pre  = TAU_PRE  * second,
            tau_post = TAU_POST * second,
            w_unit   = W_UNIT   * amp,
            w_max    = W_MAX,
            eta_pre  = ETA_PRE,
            eta_post = ETA_POST,
            x_tar    = X_TAR,
            mu       = MU,
        ),
        name='syn',
    )
    syn.connect(True)
    syn.Is1 = 0 * amp
    syn.Is2 = 0 * amp

    syn_i = np.array(syn.i[:], dtype=np.int64)        # pre index
    syn_j = np.array(syn.j[:], dtype=np.int64)        # post index
    is_bell = syn_i < N_BELL
    w0 = np.where(is_bell, W_BELL_0, W_FOOD_0).astype(np.float64)
    syn.w[:] = w0

    sp_pre  = SpikeMonitor(P, name='sp_pre')
    sp_post = SpikeMonitor(G, name='sp_post')

    net = Network(P, G, syn, sp_pre, sp_post)
    return net, P, syn, syn_i, syn_j, sp_pre, sp_post


# ── run ──────────────────────────────────────────────────────────────────────
def main():
    net, P, syn, syn_i, syn_j, sp_pre, sp_post = build()

    rates_zero = np.zeros(N_PRE)
    rates_bell = rates_zero.copy(); rates_bell[:N_BELL] = RATE_BELL
    rates_food = rates_zero.copy(); rates_food[N_BELL:] = RATE_FOOD
    rates_both = rates_bell + rates_food

    def W_matrix() -> np.ndarray:
        W = np.zeros((N_POST, N_PRE), dtype=np.float64)
        W[syn_j, syn_i] = np.asarray(syn.w[:])
        return W

    def run_test_trial() -> tuple[int, float, float]:
        """Returns (post_spike_count, t_start_s, t_end_s) for the test."""
        t0 = float(defaultclock.t / second)
        c0 = sp_post.count[:].copy()
        P.rates = rates_bell * Hz
        net.run(T_TEST_BELL * second)
        P.rates = rates_zero * Hz
        net.run(T_TEST_REST * second)
        t1 = float(defaultclock.t / second)
        return int((sp_post.count[:] - c0).sum()), t0, t1

    def run_training_trial() -> None:
        P.rates = rates_bell * Hz
        net.run(T_BELL_ALONE * second)
        P.rates = rates_both * Hz
        net.run(T_PAIR * second)
        P.rates = rates_zero * Hz
        net.run(T_REST * second)

    print(f"[pavlov] N_train={N_TRAIN}  test_every={TEST_EVERY}")
    print(f"[pavlov] initial weights: bell={W_BELL_0}  food={W_FOOD_0}")
    print(f"[pavlov] STDP: η_post={ETA_POST} η_pre={ETA_PRE} x_tar={X_TAR} μ={MU}")

    W_history    = [W_matrix()]
    test_trials  = [0]
    test_windows = []   # list of (t_start_s, t_end_s)
    n0, t0, t1 = run_test_trial()
    test_counts  = [n0]
    test_windows.append((t0, t1))
    print(f"[trial 0/{N_TRAIN} | TEST] bell-only post spikes = {n0}")

    for t in range(1, N_TRAIN + 1):
        run_training_trial()
        W_history.append(W_matrix())
        if t % TEST_EVERY == 0:
            n, t0, t1 = run_test_trial()
            test_trials.append(t)
            test_counts.append(n)
            test_windows.append((t0, t1))
            W_now = W_history[-1]
            print(f"[trial {t}/{N_TRAIN} | TEST] bell-only post spikes = {n}  "
                  f"<w_bell>={W_now[:, :N_BELL].mean():.3f}  "
                  f"<w_food>={W_now[:, N_BELL:].mean():.3f}")

    W_history = np.stack(W_history, axis=0)   # (T+1, N_POST, N_PRE)

    # Slice spike monitors to extract early/late raster data
    pre_t  = np.asarray(sp_pre.t  / second)
    pre_i  = np.asarray(sp_pre.i)
    post_t = np.asarray(sp_post.t / second)
    post_i = np.asarray(sp_post.i)

    def slice_window(t_arr, i_arr, t0, t1):
        m = (t_arr >= t0) & (t_arr < t1)
        return t_arr[m] - t0, i_arr[m]

    # Early test = first one (trial 0). Late test = first one after weights
    # mostly saturated — pick the index whose mean bell weight first crosses 0.9.
    bell_mean_traj = W_history[:, :, :N_BELL].mean(axis=(1, 2))
    test_trial_idx_array = np.array(test_trials)
    bell_at_tests = bell_mean_traj[test_trial_idx_array]
    sat_test_idx = int(np.argmax(bell_at_tests > 0.9)) if (bell_at_tests > 0.9).any() else len(test_trials) - 1
    early_idx, late_idx = 0, sat_test_idx

    e_t0, e_t1 = test_windows[early_idx]
    l_t0, l_t1 = test_windows[late_idx]
    early_raster = (
        slice_window(pre_t,  pre_i,  e_t0, e_t1),
        slice_window(post_t, post_i, e_t0, e_t1),
        test_trials[early_idx],
        test_counts[early_idx],
    )
    late_raster = (
        slice_window(pre_t,  pre_i,  l_t0, l_t1),
        slice_window(post_t, post_i, l_t0, l_t1),
        test_trials[late_idx],
        test_counts[late_idx],
    )

    plot_results(
        W_history,
        np.array(test_trials),
        np.array(test_counts),
        early_raster,
        late_raster,
    )


# ── plotting ─────────────────────────────────────────────────────────────────
def _draw_raster(ax, pre_data, post_data, trial_idx, post_count, panel_letter):
    """One test-trial raster: bell pre below, post above. Bell-on shaded."""
    (pre_t, pre_i), (post_t, post_i) = pre_data, post_data

    # Shaded "bell on" band
    ax.add_patch(Rectangle(
        (0, -0.5), T_TEST_BELL, N_PRE + N_POST + 1,
        facecolor='gold', alpha=0.15, zorder=0,
    ))
    ax.text(T_TEST_BELL / 2, N_PRE + N_POST + 0.6, 'bell on',
            ha='center', va='bottom', fontsize=9, color='goldenrod')

    # Pre rows 0..9 at the bottom (only bell ever fires here, since test = bell only)
    pre_color = np.where(pre_i < N_BELL, 'tab:blue', 'tab:gray')
    ax.scatter(pre_t, pre_i, marker='|', s=80, c=pre_color, lw=1.2)
    # Post rows offset above pre with a gap
    ax.scatter(post_t, post_i + N_PRE + 1, marker='|', s=80,
               c='tab:red', lw=1.2)
    ax.axhline(N_PRE + 0.5, color='black', lw=0.6, alpha=0.4)

    ax.set_xlim(0, T_TEST_BELL + T_TEST_REST)
    ax.set_ylim(-0.7, N_PRE + N_POST + 0.7)
    ax.set_xlabel('time within test trial (s)')
    ax.set_yticks([N_BELL / 2 - 0.5, N_PRE / 2 + N_BELL / 2,
                   N_PRE + 1 + N_POST / 2 - 0.5])
    ax.set_yticklabels(['bell\n(pre 0-4)', 'food\n(pre 5-9)', 'post\n(MSN)'])
    ax.set_title(f'({panel_letter}) raster — trial {trial_idx} test  '
                 f'({post_count} post spikes)')
    ax.grid(True, axis='x', alpha=0.3)


def plot_results(W_hist: np.ndarray, test_trials: np.ndarray,
                 test_counts: np.ndarray,
                 early_raster, late_raster) -> None:
    W_before = W_hist[0]
    W_after  = W_hist[-1]

    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.45, wspace=1.1)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    # (a) weight matrices before / after
    ax = ax_a
    combined = np.full((N_POST, 2 * N_PRE + 1), np.nan)
    combined[:, :N_PRE] = W_before
    combined[:, N_PRE + 1:] = W_after
    im = ax.imshow(combined, cmap='magma', vmin=0, vmax=W_MAX, aspect='auto')
    ax.axvline(N_BELL - 0.5, color='cyan', lw=0.8, ls='--')
    ax.axvline(N_PRE + N_BELL + 0.5, color='cyan', lw=0.8, ls='--')
    ax.set_xticks([N_BELL/2 - 0.5, N_PRE/2 + N_BELL/2 - 0.5,
                   N_PRE + N_BELL/2 + 0.5,
                   N_PRE + N_PRE/2 + N_BELL/2 + 0.5])
    ax.set_xticklabels(['bell', 'food', 'bell', 'food'])
    ax.set_xlabel('  ←  before  →            ←  after  →')
    ax.set_ylabel('post neuron')
    ax.set_title('(a) weight matrix')
    plt.colorbar(im, ax=ax, label='w')

    # (b) mean weight evolution
    ax = ax_b
    bell_mean = W_hist[:, :, :N_BELL].mean(axis=(1, 2))
    food_mean = W_hist[:, :, N_BELL:].mean(axis=(1, 2))
    ax.plot(bell_mean, label='⟨w⟩  bell→post (CS)', lw=2)
    ax.plot(food_mean, label='⟨w⟩  food→post (US)', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('training trial')
    ax.set_ylabel('mean weight')
    ax.set_ylim(-0.02, W_MAX * 1.05)
    ax.set_title('(b) weight evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) learning curve
    ax = ax_c
    ax.plot(test_trials, test_counts, '-o', color='tab:red', lw=2)
    ax.set_xlabel('training trial')
    ax.set_ylabel('post spikes during bell-only test')
    ax.set_title('(c) learning curve  (Pavlov)')
    ax.grid(True, alpha=0.3)

    # (d, e) rasters
    e_pre, e_post, e_trial, e_count = early_raster
    l_pre, l_post, l_trial, l_count = late_raster
    _draw_raster(ax_d, e_pre, e_post, e_trial, e_count, 'd')
    _draw_raster(ax_e, l_pre, l_post, l_trial, l_count, 'e')

    fig.suptitle("Pavlov's-dog STDP demo — 10 pre × 10 post MSN neurons",
                 fontsize=13)

    out = pathlib.Path(__file__).resolve().parent / 'pavlov_demo.png'
    fig.savefig(out, dpi=150)
    print(f"\n[saved] {out}")
    print(f"[summary] bell test spikes: {test_counts[0]} → {test_counts[-1]}")
    print(f"[summary] mean bell weight: {bell_mean[0]:.3f} → {bell_mean[-1]:.3f}")
    print(f"[summary] mean food weight: {food_mean[0]:.3f} → {food_mean[-1]:.3f}")


if __name__ == '__main__':
    main()