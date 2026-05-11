"""
plot_schematic.py
=================
High-level architecture diagram of the unsupervised MSN classifier
(同 train_stdp.py / train_bsf.py — only the synapse rule differs):

    28×28 image  ──▶  784 Poisson  ──[plastic W]──▶  N E-MSN
                                                       │ 1:1 fixed E→I
                                                       ▼
                                                     N I-MSN
                                                       │ all-but-self I→E
                                                       └──▶  back to E (WTA)

Plus a side panel naming the four currents that flow into each E neuron:
plastic excitatory, lateral inhibitory, homeostatic θ tonic bias, and the
MSN's own internal cascade (Is1, Is2).

Pure schematic — does not load any trained weights.
Usage:
    uv run python Classification-STDP/plot_schematic.py
    uv run python Classification-STDP/plot_schematic.py --rule BSF
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


def draw_block(ax, xy, w, h, label, color, ec='k', subtext=None):
    box = FancyBboxPatch(xy, w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc=color, ec=ec, lw=1.4)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if subtext:
        ax.text(cx, cy + 0.07, label, ha='center', va='center',
                fontsize=12, fontweight='bold')
        ax.text(cx, cy - 0.10, subtext, ha='center', va='center',
                fontsize=9, style='italic', color='#333')
    else:
        ax.text(cx, cy, label, ha='center', va='center', fontsize=12,
                fontweight='bold')


def arrow(ax, p0, p1, color='k', lw=1.6, style='->', label=None,
          label_offset=(0, 0.05), curve=0.0, label_fontsize=9, label_color=None):
    arr = FancyArrowPatch(p0, p1,
                          arrowstyle=style,
                          mutation_scale=14,
                          color=color, lw=lw,
                          connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(arr)
    if label is not None:
        mx = (p0[0] + p1[0]) / 2 + label_offset[0]
        my = (p0[1] + p1[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=label_fontsize,
                color=label_color if label_color else color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', default='STDP/BSF',
                    help='label for the plastic synapse box')
    ap.add_argument('--fig',  default='Classification-STDP/schematic.png')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── (1) MNIST input image (mini 28×28 patch) ───────────────────────────
    rng = np.random.default_rng(0)
    digit = np.zeros((28, 28))
    yy, xx = np.ogrid[:28, :28]
    digit += np.exp(-((xx - 14) ** 2 + (yy - 12) ** 2) / 25) * 0.9
    digit += rng.uniform(0, 0.1, (28, 28))
    digit = np.clip(digit, 0, 1)
    ax.imshow(digit, cmap='gray_r', extent=(0.4, 2.2, 4.0, 5.8),
              aspect='auto', zorder=2)
    ax.add_patch(Rectangle((0.4, 4.0), 1.8, 1.8, fill=False, ec='k', lw=1.2))
    ax.text(1.3, 5.95, '28 × 28 image', ha='center', fontsize=10)
    ax.text(1.3, 3.8, 'pixel ∈ [0, 1]', ha='center', fontsize=9,
            style='italic', color='#555')

    # ── (2) Poisson encoder ─────────────────────────────────────────────────
    draw_block(ax, (3.0, 4.2), 1.6, 1.5,
               '784 Poisson',
               color='#FFE4B5',
               subtext='λ = λ_max · pixel\n(63.75 Hz · pixel)')
    arrow(ax, (2.25, 5.0), (3.0, 5.0), label='rate-code', label_offset=(0, 0.18))

    # ── (3) Plastic input synapses (W) ─────────────────────────────────────
    draw_block(ax, (5.1, 4.2), 1.7, 1.5,
               f'plastic W',
               color='#D7F0FF',
               subtext=f'{args.rule}\n(unsupervised)')
    arrow(ax, (4.62, 4.95), (5.1, 4.95), label=None)

    # ── (4) E-MSN layer ─────────────────────────────────────────────────────
    draw_block(ax, (7.3, 4.2), 1.8, 1.5,
               'N × E-MSN',
               color='#FFB3B3',
               subtext='excitatory\n+ homeo θ')
    arrow(ax, (6.83, 4.95), (7.3, 4.95), label='I_exc', label_offset=(0, 0.22),
          label_fontsize=8, label_color='#333')

    # ── (5) I-MSN layer ─────────────────────────────────────────────────────
    draw_block(ax, (10.0, 4.2), 1.8, 1.5,
               'N × I-MSN',
               color='#B3D9FF',
               subtext='inhibitory')
    # E → I 1:1
    arrow(ax, (9.13, 5.10), (10.0, 5.10),
          label='1 : 1 fixed\n(w_e2i)', label_offset=(0, 0.30),
          label_fontsize=8, label_color='#333')

    # ── (6) Lateral inhibition I → E (curved feedback under both blocks) ───
    arrow(ax, (10.9, 4.2), (8.2, 4.2),
          color='#205CC9', lw=2.0,
          curve=-0.45,
          label='I → E (all-but-self)\nw_i2e — lateral WTA',
          label_offset=(0, -0.55),
          label_fontsize=9, label_color='#205CC9')

    # ── (7) Homeostatic θ feedback loop on E ───────────────────────────────
    arrow(ax, (8.2, 5.7), (8.2, 6.45),
          color='#7B5BB6', lw=1.5, style='->')
    ax.text(8.2, 6.65, 'θ = θ · decay + θ₊·spike',
            ha='center', fontsize=8, color='#7B5BB6')
    arrow(ax, (8.4, 6.45), (8.4, 5.7),
          color='#7B5BB6', lw=1.5, style='->')
    ax.text(9.0, 6.05, '−θ as I_0\n(per-neuron tonic bias)',
            ha='left', fontsize=8, color='#7B5BB6', va='center')

    # ── (8) Side panel: synapse rule comparison ────────────────────────────
    panel = FancyBboxPatch((0.4, 0.5), 12.2, 2.6,
                           boxstyle="round,pad=0.04,rounding_size=0.1",
                           fc='#FAFAF2', ec='#888', lw=1.0)
    ax.add_patch(panel)
    ax.text(0.7, 2.85, 'Plastic input rule  (only the W block changes)',
            fontsize=11, fontweight='bold')

    # left col — STDP
    ax.text(0.7, 2.45, 'pair-STDP (Diehl & Cook)',
            fontsize=10, fontweight='bold', color='#1A6FB0')
    ax.text(0.7, 2.10,
            "  • on pre :  apre += 1;   w −= η·apost\n"
            "  • on post:  apost += 1;  w += η·apre − η·x_tar·w^μ\n"
            "  • continuous w ∈ [0, w_max] + L1 normalisation",
            fontsize=8.5, family='monospace', va='top')

    # right col — BSF
    ax.text(6.6, 2.45, 'Brader-Senn-Fusi (stop-learning)',
            fontsize=10, fontweight='bold', color='#B0461A')
    ax.text(6.6, 2.10,
            "  • binary readout:  w_eff = w_jump · [X > θ_X]\n"
            "  • on pre :  if Vm_post>θ_V & C∈[θ_lo_p,θ_hi_p):  X += a_LTP\n"
            "             if Vm_post≤θ_V & C∈[θ_lo_d,θ_hi_d):  X −= a_LTD\n"
            "  • on post:  C += J_C;     dC/dt = −C/τ_C  (decay)",
            fontsize=8.5, family='monospace', va='top')

    fig.suptitle(
        'Unsupervised MSN MNIST classifier — high-level architecture',
        y=0.98, fontsize=13, fontweight='bold')

    out = pathlib.Path(args.fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"[done] schematic → {out}")


if __name__ == '__main__':
    main()
