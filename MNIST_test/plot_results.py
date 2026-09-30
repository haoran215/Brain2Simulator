"""
plot_results.py
===============
Figure for MNIST_test/README.md, drawn from results.json (bench_mnist.py).

    uv run python MNIST_test/plot_results.py   →  MNIST_test/mnist_results.png
"""

import os, json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(_HERE, 'results.json')))

MODELS = ['MSN (discrete, dt=1 µs)', 'MSN_HH (τx=1 µs, dt=0.1 µs)', 'MSN_HH (τx=10 µs, dt=1 µs)']
SHORT = ['MSN\n(dt = 1 µs)', 'MSN_HH  τx = 1 µs\n(dt = 0.1 µs)', 'MSN_HH  τx = 10 µs\n(dt = 1 µs)']
COLOR = ['#2a78d6', '#eb6834', '#1baf7a']           # fixed per model across panels
INK, MUTED, GRID, BASE = '#1f1f1e', '#6b6a64', '#e6e5df', '#b9b8b0'

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': BASE, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def hbars(ax, labels, values, colors, fmt, xlabel, log=False, notes=None, pad=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=colors, edgecolor='white', linewidth=2)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale('log')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    pad = pad or [0] * len(values)
    for yi, v, n, p in zip(y, values, notes or [''] * len(values), pad):
        ax.text(v * (1.12 if log else 1) + (0 if log else p + max(values) * 0.02), yi,
                fmt(v) + n, va='center', color=INK, fontsize=8.5)


fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))
(a, b), (c, d) = axs

# (a) F–I sanity ─────────────────────────────────────────────────────────────
S = R['sanity']; I = np.array(S['I_uA'])
a.plot(I, S['analytic'], '--', color=MUTED, lw=1.5, label='analytic MSN')
for m, s, col in zip(MODELS, SHORT, COLOR):
    a.plot(I, S[m], '-o', color=col, lw=2, ms=6, mec='white', mew=1.5,
           label=s.replace('\n', ' '))
a.set(xlabel='input current  I_in (µA)', ylabel='firing rate (Hz)',
      title='a   Firing rate vs input current (Brian2)')
a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
a.legend(frameon=False, fontsize=8, loc='lower right')

# (b) accuracy ───────────────────────────────────────────────────────────────
A = R['A']; n_te = R['config']['N_TEST']
acc_lab = ['raw pixels\n(linear baseline)', 'analytic MSN rates\n(T → ∞)'] + SHORT
acc = [A['pixels']['acc'], A['analytic']['acc']] + [A[m]['acc'] for m in MODELS]
acc_col = ['#cfcec7', '#a3a29a'] + COLOR
se = [100 * np.sqrt(p * (1 - p) / n_te) for p in acc]
hbars(b, acc_lab, [100 * v for v in acc], acc_col, lambda v: f'{v:.1f} %',
      'MNIST test accuracy (%)', pad=se)
b.errorbar([100 * v for v in acc], np.arange(len(acc))[::-1], xerr=se, fmt='none',
           ecolor=INK, elinewidth=1, capsize=3)
b.set_xlim(0, 100)
b.set_title(f'b   Accuracy — no significant difference (±1 SE, n = {n_te})')

# (c) wall time, current-driven ──────────────────────────────────────────────
wall = [A[m]['wall_s'] for m in MODELS]
hbars(c, SHORT, wall, COLOR, lambda v: f'{v:.1f} s',
      'wall-clock time, log scale (s)', log=True,
      notes=[f'   ({w / wall[0]:.1f}×)' if i else '   (1×)' for i, w in enumerate(wall)])
c.set_xlim(5, 5000)
c.set_title('c   Neurons only: 700 images × 100 neurons, 50 ms')

# (d) full SNN cost ──────────────────────────────────────────────────────────
B = R['B']
per = [B[m]['wall_per_sim_s'] for m in MODELS]
hbars(d, SHORT, per, COLOR, lambda v: f'{v:.0f} s',
      'compute time per simulated second, log scale (s)', log=True,
      notes=[f'   ≈ {B[m]["epoch_days"]:.0f} days / MNIST epoch' for m in MODELS])
d.set_xlim(20, 1e5)
d.set_title('d   Full SNN: 784 Poisson → 100, 78 400 synapses')

fig.suptitle('MSN vs MSN_HH on MNIST — same accuracy, MSN_HH up to 31× slower',
             x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5, w_pad=3)
out = os.path.join(_HERE, 'mnist_results.png')
fig.savefig(out, dpi=150, facecolor='white')
print('saved', out)
