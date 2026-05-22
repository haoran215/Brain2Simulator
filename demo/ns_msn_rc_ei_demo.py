"""
ns_msn_rc_ei_demo.py
====================

Reservoir Computing (RC) demo — left vs right classification with an
EXCITATORY + INHIBITORY MSN reservoir, plastic (STDP) recurrent excitation,
and per-device variability.

This is the follow-up to ns_msn_rc_demo.py.  That earlier demo had two
problems documented in docs/STDP_EI_report.md:

  1. The STDP branch crashed (ZeroDivisionError) because the hand-written
     Synapses object let Brian2 pick the 'exact' integrator, whose closed
     form for the two-stage cascade divides by (tau_s2 - tau_s1) = 0.
  2. Once that was fixed, STDP DROPPED test accuracy to chance (50%):
     symmetric Hebbian potentiation with no opposing force ran the
     recurrent weights up ~14× until they saturated, coupling the L and R
     groups and destroying input selectivity.

What changed here
─────────────────
  • Recurrent INHIBITION added (§3b).  MSNs are GABAergic projection
    neurons — they inhibit each other.  Modelling recurrent connections
    as purely excitatory (as the old demo did) is biologically wrong AND
    numerically unstable.  Cross-group lateral inhibition (L ⊣ R, R ⊣ L)
    both (a) restores E/I balance so STDP can't run away, and (b) creates
    winner-take-all competition that SHARPENS left/right discrimination.
  • STDP given a hard ceiling (low w_max) and a smaller learning rate so
    potentiation saturates gracefully instead of exploding.
  • Per-device variability (msn_variability.apply_variability) scatters
    Vth and I_hold across neurons — richer reservoir, and a check that the
    classifier survives hardware mismatch.

Architecture
────────────
  [PoissonGroup L] ──exc──► neurons  0– 9 ─┐  recurrent exc (STDP)  ┐
  [PoissonGroup R] ──exc──► neurons 10–19 ─┘  recurrent inh (static)┘→ readout

Trial structure (ms): 0–500 stim · 500–1000 ITI · feature window 200–500 ms.
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
from brian2 import *

from msn_neuron      import MSNParams, make_msn
from msn_synapse     import SynapseParams, make_synapse
from msn_variability import apply_variability, device_summary

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 0. Config                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
USE_STDP   = os.environ.get('USE_STDP', 'True') == 'True'   # env-overridable
USE_INH    = os.environ.get('USE_INH',  'True') == 'True'   # recurrent inhibition
VAR_SCALE  = float(os.environ.get('VAR_SCALE', '0.5'))      # device-variability spread

prefs.codegen.target = 'numpy'
start_scope()
defaultclock.dt = 10 * us
np.random.seed(42)

N        = 20
N_LEFT   = 10
N_RIGHT  = 10
n_train  = 20
n_test   = 10
stim_ms  = 500
iti_ms   = 500
trial_ms = stim_ms + iti_ms
feat_start = 200
feat_end   = stim_ms
stim_rate  = 100.0
bg_rate    = 5.0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. Load parameters                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
params    = MSNParams.from_json(os.path.join(_REPO, 'configs/neuron_default.json'))
I_min, I_max = params.operating_window()
syn_exc_p = SynapseParams.from_json(os.path.join(_REPO, 'configs/synapse_default.json'), key='exc')
syn_inh_p = SynapseParams.from_json(os.path.join(_REPO, 'configs/synapse_default.json'), key='inh')
print(params.summary(), '\n')
print(f"I_min = {I_min*1e6:.2f} µA")
print(device_summary(), '\n')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. Build reservoir — exc inlets + a recurrent inh inlet                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# I_exc_rec : plastic recurrent excitation     (one summed writer)
# I_exc_L/R : feed-forward stimulus inlets      (one writer each)
# I_inh_rec : recurrent lateral inhibition      (one summed writer)
reservoir = make_msn(N=N, params=params,
                     exc_inlets=('I_exc_rec', 'I_exc_L', 'I_exc_R'),
                     inh_inlets=('I_inh_rec',),
                     name='reservoir')
reservoir.I_0 = 0.7 * I_min * amp

# Per-device variability: scatter Vth and I_hold around the calibrated means.
# scale=0.5 → half the measured hardware spread (keeps the L/R tuning intact
# while still giving the reservoir heterogeneous thresholds).
apply_variability(reservoir, seed=42, scale=VAR_SCALE)
print(f"Variability scale = {VAR_SCALE}:  "
      f"Vth ∈ [{np.min(reservoir.Vth/volt):.2f}, {np.max(reservoir.Vth/volt):.2f}] V,  "
      f"I_hold ∈ [{np.min(reservoir.I_hold/amp)*1e6:.1f}, {np.max(reservoir.I_hold/amp)*1e6:.1f}] µA")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3a. Recurrent EXCITATION — static or plastic (STDP)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if not USE_STDP:
    syn_exc = make_synapse(
        source = reservoir, target = reservoir,
        params = SynapseParams(kind='exc', weight=syn_exc_p.weight,
                               tau_s1=syn_exc_p.tau_s1, tau_s2=syn_exc_p.tau_s2,
                               target_var='I_exc_rec'),
        connect = 'rand() < 0.15 and i != j',
        name = 'syn_exc')
    n_exc = len(syn_exc.w)
    syn_exc.w = np.random.gamma(shape=2.0, scale=0.15e-6, size=n_exc) * amp
    print(f"Static exc: {n_exc} synapses, mean w = {np.mean(syn_exc.w/amp)*1e6:.3f} µA")
else:
    # ── Plastic recurrent excitation (Bi & Poo STDP) ────────────────────────
    # Stabilised vs the old demo:
    #   • w_max = 1 µA hard ceiling  (was 15 µA → runaway)
    #   • lr    = 0.02 µA per pairing (was 0.30 µA → 15× too large)
    #   • lr_minus > lr_plus → net depression bias keeps weights bounded.
    # method='euler' is REQUIRED: the 'exact' solver divides by
    # (tau_s2 - tau_s1) = 0 for this equal-tau cascade.
    tau_pre  = 20 * ms
    tau_post = 20 * ms
    lr_plus  = 0.02e-6 * amp
    lr_minus = 0.025e-6 * amp
    w_max    = 1.0e-6 * amp

    syn_exc = Synapses(
        reservoir, reservoir,
        model = '''
            w                                : amp
            dIs1/dt = -Is1 / tau_s1          : amp (clock-driven)
            dIs2/dt = (-Is2 + Is1) / tau_s2  : amp (clock-driven)
            I_exc_rec_post = Is2             : amp (summed)
            dApre/dt  = -Apre  / tau_pre     : 1 (event-driven)
            dApost/dt = -Apost / tau_post    : 1 (event-driven)
        ''',
        on_pre  = '''
            Is1  += w
            Apre += 1
            w     = clip(w - Apost * lr_minus, 0*amp, w_max)
        ''',
        on_post = '''
            Apost += 1
            w      = clip(w + Apre * lr_plus, 0*amp, w_max)
        ''',
        method = 'euler',
        namespace = dict(
            tau_s1=syn_exc_p.tau_s1 * second, tau_s2=syn_exc_p.tau_s2 * second,
            tau_pre=tau_pre, tau_post=tau_post,
            lr_plus=lr_plus, lr_minus=lr_minus, w_max=w_max),
        name = 'syn_exc')
    syn_exc.connect(condition='rand() < 0.15 and i != j')
    syn_exc.Is1 = 0 * amp
    syn_exc.Is2 = 0 * amp
    n_exc = len(syn_exc.w)
    syn_exc.w = np.random.gamma(shape=2.0, scale=0.15e-6, size=n_exc) * amp
    print(f"STDP exc:   {n_exc} synapses, mean w_init = {np.mean(syn_exc.w/amp)*1e6:.3f} µA")

w_init = np.array(syn_exc.w / amp).copy()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3b. Recurrent INHIBITION — static, cross-group lateral inhibition        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# MSNs are GABAergic: recurrent MSN→MSN connections are inhibitory.
# Wiring L⊣R and R⊣L makes the two groups COMPETE — when the stimulated
# side fires, it suppresses the other side, sharpening the L/R contrast and
# soaking up the excess excitation that STDP would otherwise let run away.
# Faster kinetics (tau=50 ms) than the 200 ms exc cascade → quick competition.
if USE_INH:
    syn_inh = make_synapse(
        source = reservoir, target = reservoir,
        params = SynapseParams(kind='inh', weight=1.0e-6,
                               tau_s1=50e-3, tau_s2=50e-3,
                               target_var='I_inh_rec'),
        connect = '(i < %d) != (j < %d) and rand() < 0.4' % (N_LEFT, N_LEFT),
        name = 'syn_inh')
    n_inh = len(syn_inh.w)
    print(f"Inhibition: {n_inh} cross-group synapses, w = {1.0:.2f} µA, tau = 50 ms")
else:
    n_inh = 0
    print("Inhibition: DISABLED")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. Input layer                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
input_L = PoissonGroup(N_LEFT,  rates=0*Hz, name='input_L')
input_R = PoissonGroup(N_RIGHT, rates=0*Hz, name='input_R')
inp_w   = 2e-6
syn_in_L = make_synapse(input_L, reservoir,
                        SynapseParams(kind='exc', weight=inp_w,
                                      tau_s1=syn_exc_p.tau_s1, tau_s2=syn_exc_p.tau_s2,
                                      target_var='I_exc_L'),
                        connect='j == i', name='syn_in_L')
syn_in_R = make_synapse(input_R, reservoir,
                        SynapseParams(kind='exc', weight=inp_w,
                                      tau_s1=syn_exc_p.tau_s1, tau_s2=syn_exc_p.tau_s2,
                                      target_var='I_exc_R'),
                        connect='j == i + %d' % N_LEFT, name='syn_in_R')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. Monitors                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sp_all = SpikeMonitor(reservoir)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 6–7. Trial schedule + run                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
labels_train = np.array([0, 1] * (n_train // 2))
labels_test  = np.array([0, 1] * (n_test  // 2))
all_labels   = np.concatenate([labels_train, labels_test])
n_total      = len(all_labels)
t_windows    = []

mode = (f"STDP exc + {'INH' if USE_INH else 'no inh'} + var{VAR_SCALE}"
        if USE_STDP else
        f"static exc + {'INH' if USE_INH else 'no inh'} + var{VAR_SCALE}")
print("─" * 60)
print(f"[{mode}]  Running {n_total} trials × {trial_ms} ms = "
      f"{n_total*trial_ms/1000:.1f} s …")
print("─" * 60)

for k, label in enumerate(all_labels):
    t0 = float(defaultclock.t / ms)
    if label == 0:
        input_L.rates, input_R.rates = stim_rate*Hz, bg_rate*Hz
    else:
        input_L.rates, input_R.rates = bg_rate*Hz, stim_rate*Hz
    run(stim_ms * ms)
    input_L.rates = input_R.rates = bg_rate * Hz
    run(iti_ms * ms)
    t_windows.append((t0 + feat_start, t0 + feat_end))
    if (k + 1) % 5 == 0:
        print(f"  Trial {k+1:2d}/{n_total}  label={'L' if label==0 else 'R'}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 8. Feature extraction                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
t_sp = np.array(sp_all.t / ms)
i_sp = np.array(sp_all.i)
X = np.zeros((n_total, N))
for k, (ta, tb) in enumerate(t_windows):
    mask = (t_sp >= ta) & (t_sp < tb)
    if mask.any():
        X[k] = np.bincount(i_sp[mask], minlength=N)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 9. Ridge-regression readout                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
y_all = all_labels * 2.0 - 1.0
X_train, y_train = X[:n_train], y_all[:n_train]
X_test,  y_test  = X[n_train:], y_all[n_train:]
lam   = 1e-3
scale = np.max(X_train) ** 2 if X_train.max() > 0 else 1.0
w_out = np.linalg.solve(X_train.T @ X_train + lam*scale*np.eye(N), X_train.T @ y_train)
acc_train = np.mean(np.sign(X_train @ w_out) == y_train)
acc_test  = np.mean(np.sign(X_test  @ w_out) == y_test)

print(f"\n{'─'*60}")
print(f"  Train accuracy : {acc_train*100:.0f}%  ({int(acc_train*n_train)}/{n_train})")
print(f"  Test  accuracy : {acc_test*100:.0f}%   ({int(acc_test*n_test)}/{n_test})")
print(f"{'─'*60}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 10. Plot                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)

# Exc weight matrix
ax = fig.add_subplot(gs[0, 0])
W_e = np.zeros((N, N))
W_e[np.array(syn_exc.i[:]), np.array(syn_exc.j[:])] = np.array(syn_exc.w/amp) * 1e6
im = ax.imshow(W_e, cmap='hot', vmin=0, vmax=max(W_e.max(), 1e-9), aspect='auto')
ax.axhline(N_LEFT-0.5, color='cyan', lw=1.2, ls='--'); ax.axvline(N_LEFT-0.5, color='cyan', lw=1.2, ls='--')
ax.set_xticks([N_LEFT//2, N_LEFT+N_RIGHT//2]); ax.set_xticklabels(['L', 'R'])
ax.set_yticks([N_LEFT//2, N_LEFT+N_RIGHT//2]); ax.set_yticklabels(['L', 'R'])
ax.set_title(f"Recurrent EXC W (µA)\n{'After STDP' if USE_STDP else 'Static'} — "
             f"mean {np.array(syn_exc.w/amp).mean()*1e6:.3f} µA", fontweight='bold')
plt.colorbar(im, ax=ax, label='w (µA)')

# Weight distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(w_init*1e6, bins=20, alpha=0.6, color='steelblue', label='init')
if USE_STDP:
    ax.hist(np.array(syn_exc.w/amp)*1e6, bins=20, alpha=0.6, color='tomato', label='after STDP')
ax.set_xlabel('w (µA)'); ax.set_ylabel('count')
ax.set_title('Recurrent exc weight distribution', fontweight='bold'); ax.legend(fontsize=9)

# Raster
ax = fig.add_subplot(gs[1, :])
n_show = min(4, n_total); t_zoom = n_show*trial_ms
m = t_sp < t_zoom
ts, isp = t_sp[m], i_sp[m]
lm, rm = isp < N_LEFT, isp >= N_LEFT
ax.scatter(ts[lm], isp[lm], s=4, color='steelblue', alpha=0.8, label='L neurons (0–9)')
ax.scatter(ts[rm], isp[rm], s=4, color='tomato', alpha=0.8, label='R neurons (10–19)')
for k in range(n_show):
    a, b = k*trial_ms+feat_start, k*trial_ms+feat_end
    c = 'steelblue' if all_labels[k]==0 else 'tomato'
    ax.axvspan(a, b, alpha=0.12, color=c)
    ax.text((a+b)/2, N-0.5, 'L' if all_labels[k]==0 else 'R', ha='center', va='top',
            fontsize=9, fontweight='bold', color=c)
ax.set_xlim(0, t_zoom); ax.set_ylim(-0.5, N-0.5)
ax.axhline(N_LEFT-0.5, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('t (ms)'); ax.set_ylabel('Neuron index')
ax.set_title(f'Spike raster — first {n_show} trials (shaded = feature window)', fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

# Feature space
ax = fig.add_subplot(gs[2, 0])
fL = X[:, :N_LEFT].sum(1); fR = X[:, N_LEFT:].sum(1)
for split, mk in [(slice(None, n_train), 's'), (slice(n_train, None), 'D')]:
    for cls, clr, nm in [(-1, 'steelblue', 'Left'), (+1, 'tomato', 'Right')]:
        sel = y_all[split] == cls
        ax.scatter(fL[split][sel], fR[split][sel], color=clr, marker=mk, s=50, alpha=0.8,
                   label=f'{nm} ({"train" if mk=="s" else "test"})')
fLr = np.array([fL.min()-1, fL.max()+1])
wL, wR = w_out[:N_LEFT].sum(), w_out[N_LEFT:].sum()
if abs(wR) > 1e-9:
    ax.plot(fLr, -wL/wR*fLr, 'k--', lw=1.5, label='decision boundary')
ax.set_xlabel('spike count: L neurons'); ax.set_ylabel('spike count: R neurons')
ax.set_title('Feature space — L vs R spike counts', fontweight='bold'); ax.legend(fontsize=8)

# Accuracy
ax = fig.add_subplot(gs[2, 1])
ax.bar(['Train', 'Test'], [acc_train*100, acc_test*100], color=['steelblue', 'tomato'],
       alpha=0.8, edgecolor='k')
ax.axhline(50, color='gray', ls=':', lw=1, label='chance')
ax.set_ylim(0, 110); ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Readout accuracy\nTrain {acc_train*100:.0f}% · Test {acc_test*100:.0f}%', fontweight='bold')
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

fig.suptitle(f'MSN E-I Reservoir — L/R classification  ·  {mode}  ·  '
             f'{n_exc} exc + {n_inh} inh synapses', fontsize=12, fontweight='bold', y=1.002)

tag = ('stdp' if USE_STDP else 'static') + ('_ei' if USE_INH else '_e')
out_path = f'demo/ns_msn_rc_ei_demo_{tag}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
