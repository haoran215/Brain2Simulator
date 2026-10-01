"""
ns_hd_4n_sweep.py
=================
Robustness sweeps of the turning mechanism of demo/ns_hd_4n.py, run in Brian2
(many short simulations in parallel).

Sweeps
──────
  grid      PB → EB shift weight × velocity-pulse amplitude.
            Protocol: seed EB1, one CW pulse, one CCW pulse.
  duration  velocity-pulse duration at the default parameters (one CW pulse).
  repeat    four CW pulses with a shrinking pause between them, and a CW pulse
            followed directly by a CCW pulse.
  train     6 s train of CW pulses, pulse width × pause; which neurons fire.
  block     velocity amplitude raised until GI goes into depolarisation block.

Run
───
    uv run python demo/ns_hd_4n_sweep.py --sweep grid | duration | repeat | train | block [--jobs 16]
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_DIR  = 4
SEED   = (0.2, 0.5, 0, 30e-6)   # t0, dur, EB index, amplitude — bump on EB1
T_VEL  = 2.0                    # s, first velocity pulse
SETTLE = 2.0                    # s, wait after the last pulse before reading the bump
BIN    = 0.05                   # s, bin of the winner timeline


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ One simulation (runs in a worker process)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_one(job):
    """job: dict(over={module global: value}, pb_pulses=[(t0, dur, side, amp)], T=…).

    Returns the job plus the spike times (s) of every neuron.
    """
    import ns_hd_4n as hd
    for name, val in job['over'].items():
        setattr(hd, name, val)
    res = hd.simulate([SEED], job['T'], pb_pulses=job['pb_pulses'])
    out = dict(job)
    for g in hd.GROUP_ORDER:
        out[g] = [hd.spikes_of(res['spk'][g], k) for k in range(res['groups'][g].N)]
    # total drive of GI (µA): tonic bias + summed EB → GI current
    gi = res['st']['GI']
    out['t'] = np.asarray(gi.t / hd.second)
    out['gi_drive'] = hd.I0_GI * 1e6 + np.asarray(gi.I_exc_eb[0] / hd.uA)
    return out


def run_jobs(jobs, n_proc):
    # one fresh process per simulation: Brian2 keeps global state between runs
    with mp.Pool(n_proc, maxtasksperchild=1) as pool:
        out = []
        for n, r in enumerate(pool.imap(run_one, jobs), 1):
            out.append(r)
            print(f"  {n}/{len(jobs)} done", flush=True)
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Analysis helpers                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def rate(ts, t_a, t_b):
    return np.sum((ts >= t_a) & (ts < t_b)) / (t_b - t_a)

def winner(res, t_a, t_b):
    """0-based index of the single active EB in [t_a, t_b); None if 0 or >1."""
    act = [k for k in range(N_DIR) if rate(res['EB'][k], t_a, t_b) > 5]
    return act[0] if len(act) == 1 else None

def timeline(res, t_a, t_b):
    """Most active EB (1-based, '.' if silent) in consecutive BIN-wide bins."""
    s = ''
    for a in np.arange(t_a, t_b - 1e-9, BIN):
        n = [np.sum((res['EB'][k] >= a) & (res['EB'][k] < a + BIN)) for k in range(N_DIR)]
        s += str(int(np.argmax(n)) + 1) if max(n) else '.'
    return s

def path(tl):
    """Collapse a timeline to the sequence of visited EBs, e.g. '1123' → '123'.

    A change counts only if the new EB leads for 3 bins in a row (150 ms), so
    the flicker while two EBs fire together during a switch is ignored.
    """
    out = ''
    for n in range(len(tl) - 2):
        c = tl[n]
        if c != '.' and tl[n:n + 3] == c * 3 and (not out or out[-1] != c):
            out += c
    return out

def steps_cw(p_):
    """Signed number of ring steps along a path ('1234' → +3, '14' → −1)."""
    tot = 0
    for a, b in zip(p_[:-1], p_[1:]):
        d = (int(b) - int(a)) % N_DIR
        tot += {1: 1, 3: -1, 2: 2}[d]
    return tot

def gap_ms(ts, t_a, t_b):
    w = ts[(ts >= t_a) & (ts <= t_b)]
    return np.max(np.diff(np.concatenate(([t_a], w, [t_b])))) * 1e3

def stray_pb(res, pb_pulses, allowed):
    """Spikes, during the velocity pulses, of PBs that hold no copy.

    allowed: per pulse, the set of 0-based indices that may fire (old and new EB).
    """
    n = 0
    for (t0, d, sd, _), ok in zip(pb_pulses, allowed):
        for m in range(N_DIR):
            if m not in ok:
                ts = res[f'PB_{sd}'][m]
                n += int(np.sum((ts >= t0) & (ts < t0 + d)))
    return n


def raster_grid(results, n_rows, n_cols, suptitle, fname):
    """One raster of all 13 neurons per run; each result carries its 'title'."""
    import ns_hd_4n as hd
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4 * n_rows),
                             sharey=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.35, wspace=0.05))
    labels = [hd.neuron_name(g_, k) for g_ in hd.GROUP_ORDER
              for k in range(len(results[0][g_]))]
    for r, ax in zip(results, axes.flat):
        for t0, d, *_ in r['pb_pulses']:
            ax.axvspan(t0, t0 + d, color='gray', alpha=0.15, lw=0)
        row = 0
        for g_ in hd.GROUP_ORDER:
            for k, ts in enumerate(r[g_]):
                c = hd.GI_COLOR if g_ == 'GI' else hd.EB_COLORS[k]
                ax.vlines(ts, row - 0.4, row + 0.4, color=c, lw=0.3)
                row += 1
        ax.set_ylim(row - 0.5, -0.5)
        ax.set_xlim(T_VEL - 0.5, r['T'])
        ax.set_yticks(range(row)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(r['title'], fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('time (s)')
    fig.suptitle(suptitle, fontweight='bold', y=0.91)
    out = os.path.join(os.path.dirname(__file__), fname)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"\nFigure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sweep 1 — shift weight × velocity amplitude                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GRID_W = np.round(np.arange(0.9, 2.01, 0.1), 2) * 1e-6      # PB → EB weight (A)
GRID_I = np.arange(15.0, 32.6, 2.5) * 1e-6                  # velocity pulse (A)
GRID_DUR, GRID_PERIOD = 0.5, 2.5

def sweep_grid(n_proc):
    import ns_hd_4n as hd
    pulses = lambda a: [(T_VEL, GRID_DUR, 'cw', a),
                        (T_VEL + GRID_PERIOD, GRID_DUR, 'ccw', a)]
    T = T_VEL + 2 * GRID_PERIOD
    jobs = [dict(over={'W_SHIFT': w}, pb_pulses=pulses(a), T=T, w=w, a=a)
            for w in GRID_W for a in GRID_I]
    print(f"Grid sweep: {len(GRID_W)} shift weights × {len(GRID_I)} velocity "
          f"amplitudes = {len(jobs)} runs of {T:.1f} s")
    results = run_jobs(jobs, n_proc)

    # code: 0 ok, 1 no move, 2 wrong move / bump lost, 3 moves by itself, 4 stray PB
    LABEL = {0: 'ok', 1: 'no move', 2: 'wrong / lost', 3: 'drifts', 4: 'other PB fires'}
    code = np.zeros((len(GRID_W), len(GRID_I)), int)
    print(f"\n{'W_shift':>7s} {'I_vel':>6s}  {'seed':4s} {'cw':4s} {'ccw':4s} "
          f"{'stray PB':>8s} {'gap GI':>7s}  result")
    print(f"{'(µA)':>7s} {'(µA)':>6s}  {'':14s} {'spikes':>8s} {'(ms)':>7s}")
    for r in results:
        t1, t2 = T_VEL, T_VEL + GRID_PERIOD
        w0 = winner(r, t1 - 1.0, t1)               # after seed, before any pulse
        w1 = winner(r, t2 - 1.0, t2)               # after CW pulse
        w2 = winner(r, T - 1.0, T)                 # after CCW pulse
        stray = stray_pb(r, r['pb_pulses'], [{0, 1}, {1, 0}])
        gap = gap_ms(r['GI'][0], t1, T)
        if w0 != 0:
            c = 3 if w0 is not None else 2
        elif w1 == 0 and w2 == 0:
            c = 1
        elif w1 == 1 and w2 == 0:
            c = 4 if stray else 0
        else:
            c = 2
        code[np.argmin(abs(GRID_W - r['w'])), np.argmin(abs(GRID_I - r['a']))] = c
        nm = lambda k: '—' if k is None else f'EB{k+1}'
        print(f"{r['w']*1e6:7.2f} {r['a']*1e6:6.1f}  {nm(w0):4s} {nm(w1):4s} "
              f"{nm(w2):4s} {stray:8d} {gap:7.1f}  {LABEL[c]}")

    print("\nMap (rows = W_shift µA, columns = I_vel µA):  "
          "o ok   - no move   x wrong/lost   d drifts   p other PB fires")
    print(' ' * 6 + ''.join(f'{a*1e6:6.1f}' for a in GRID_I))
    for i, w in enumerate(GRID_W):
        print(f'{w*1e6:6.2f}' + ''.join(f"{'o-xdp'[c]:>6s}" for c in code[i]))

    # ── Figure ─────────────────────────────────────────────────────────────
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ['#2ca02c', '#c7c7c7', '#d62728', '#ff7f0e', '#9467bd']
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(code, cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5,
              origin='lower', aspect='auto')
    ax.set_xticks(range(len(GRID_I)))
    ax.set_xticklabels([f'{a*1e6:.1f}' for a in GRID_I])
    ax.set_yticks(range(len(GRID_W)))
    ax.set_yticklabels([f'{w*1e6:.1f}' for w in GRID_W])
    ax.set_xlabel('velocity pulse amplitude (µA)')
    ax.set_ylabel('PB → EB shift weight (µA)')
    # mark the working point of ns_hd_4n.py
    ax.plot(np.interp(hd.TURN_I, GRID_I, range(len(GRID_I))),
            np.interp(1.4e-6, GRID_W, range(len(GRID_W))),
            marker='*', ms=18, color='k', ls='none')
    ax.legend(handles=[Patch(color=colors[c], label=LABEL[c]) for c in LABEL],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.set_title('Turning: one CW step then one CCW step (0.5 s pulses)\n'
                 '★ = working point', fontweight='bold')
    out = os.path.join(os.path.dirname(__file__), 'ns_hd_4n_sweep_grid.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sweep 2 — velocity-pulse duration                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DURATIONS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

def sweep_duration(n_proc):
    import ns_hd_4n as hd
    jobs = [dict(over={}, pb_pulses=[(T_VEL, d, 'cw', hd.TURN_I)],
                 T=T_VEL + d + SETTLE, d=d) for d in DURATIONS]
    print(f"Duration sweep: one {hd.TURN_I*1e6:.0f} µA CW pulse, "
          f"{len(jobs)} durations")
    results = run_jobs(jobs, n_proc)

    print(f"\n{'dur (s)':>7s} {'steps':>5s} {'final':5s} {'gap GI':>7s}  path   "
          f"timeline from pulse onset ({BIN*1e3:.0f} ms bins, | = pulse end)")
    steps = []
    for r in results:
        d, T = r['d'], r['T']
        tl = timeline(r, T_VEL, T)
        p_ = path(timeline(r, T_VEL - 0.3, T))
        w = winner(r, T - 1.0, T)
        n_end = int(round(d / BIN))
        steps.append(steps_cw(p_))
        print(f"{d:7.2f} {steps[-1]:5d} {'—' if w is None else 'EB%d' % (w+1):5s} "
              f"{gap_ms(r['GI'][0], T_VEL, T):7.1f}  {p_:6s} "
              f"{tl[:n_end]}|{tl[n_end:]}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(DURATIONS, steps, 'o-', color='C0')
    ax.set_xlabel('velocity pulse duration (s)')
    ax.set_ylabel('bump steps (CW)')
    ax.set_yticks(range(0, max(steps) + 2))
    ax.grid(alpha=0.3)
    ax.set_title(f'Bump displacement vs duration of one {hd.TURN_I*1e6:.0f} µA '
                 'CW velocity pulse', fontweight='bold')
    out = os.path.join(os.path.dirname(__file__), 'ns_hd_4n_sweep_duration.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sweep 3 — repetition (pause between pulses)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

REP_DUR    = 0.5
REP_PAUSES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0]   # s between pulses
REP_N      = 4

def sweep_repeat(n_proc):
    import ns_hd_4n as hd
    jobs = []
    for seq, tag in ((['cw'] * REP_N, 'cw x4'), (['cw', 'ccw'], 'cw, ccw')):
        for g in REP_PAUSES:
            pb = [(T_VEL + n * (REP_DUR + g), REP_DUR, sd, hd.TURN_I)
                  for n, sd in enumerate(seq)]
            jobs.append(dict(over={}, pb_pulses=pb, tag=tag, g=g,
                             T=pb[-1][0] + REP_DUR + SETTLE))
    print(f"Repetition sweep: {REP_DUR:.1f} s pulses of {hd.TURN_I*1e6:.0f} µA, "
          f"{len(jobs)} runs")
    results = run_jobs(jobs, n_proc)

    print(f"\n{'sequence':9s} {'pause':>6s} {'expect':>6s} {'steps':>5s} {'final':5s} "
          f"{'ok':3s} {'gap GI':>7s}  path")
    print(f"{'':9s} {'(s)':>6s} {'':24s} {'(ms)':>7s}")
    for r in results:
        T = r['T']
        p_ = path(timeline(r, T_VEL - 0.3, T))
        w = winner(r, T - 1.0, T)
        expect = REP_N if r['tag'] == 'cw x4' else 0
        want_path = '12341' if r['tag'] == 'cw x4' else '121'
        ok = p_ == want_path and w == 0
        print(f"{r['tag']:9s} {r['g']:6.2f} {expect:6d} {steps_cw(p_):5d} "
              f"{'—' if w is None else 'EB%d' % (w+1):5s} {'yes' if ok else 'NO ':3s} "
              f"{gap_ms(r['GI'][0], T_VEL, T):7.1f}  {p_}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sweep 4 — fast pulse train (pulse width × pause)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAIN_WIDTHS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]     # s
TRAIN_PAUSES = [0.02, 0.05, 0.1, 0.2, 0.3]          # s
TRAIN_LEN    = 6.0                                  # s of CW pulse train

def sweep_train(n_proc):
    import ns_hd_4n as hd
    jobs = []
    for wd in TRAIN_WIDTHS:
        for g in TRAIN_PAUSES:
            n = int(round(TRAIN_LEN / (wd + g)))
            pb = [(T_VEL + m * (wd + g), wd, 'cw', hd.TURN_I) for m in range(n)]
            jobs.append(dict(over={}, pb_pulses=pb, wd=wd, g=g, n=n,
                             t_end=pb[-1][0] + wd, T=pb[-1][0] + wd + SETTLE))
    print(f"Pulse-train sweep: CW pulses of {hd.TURN_I*1e6:.0f} µA for "
          f"{TRAIN_LEN:.0f} s, {len(jobs)} runs")
    results = run_jobs(jobs, n_proc)

    # rates are means over the whole train; 'co-EB' is the largest number of
    # EBs that fire in the same 50 ms bin; 'all-PB' is the share of the train
    # during which all four PB_cw fire in the same 50 ms bin
    print(f"\n{'width':>5s} {'pause':>5s} {'duty':>4s} {'n':>3s} | {'steps':>5s} "
          f"{'per pulse':>9s} {'final':5s} | {'EB1':>4s} {'EB2':>4s} {'EB3':>4s} "
          f"{'EB4':>4s} {'GI':>4s} | {'PBcw1-4':>19s} | {'PBccw1-4':>19s} | "
          f"{'co-EB':>5s} {'all-PB':>6s} {'gap GI':>6s}")
    print(f"{'(s)':>5s} {'(s)':>5s} {'(%)':>4s} {'':3s} | {'':21s} | "
          f"{'(Hz)':>24s} | {'(Hz)':>19s} | {'(Hz)':>19s} | {'':5s} {'(%)':>6s} "
          f"{'(ms)':>6s}")
    for r in results:
        a, b, T = T_VEL, r['t_end'], r['T']
        p_ = path(timeline(r, a - 0.3, T))
        w = winner(r, T - 1.0, T)
        f = lambda g_, k: rate(r[g_][k], a, b)
        bins = np.arange(a, b + 1e-9, BIN)
        cnt = lambda g_: np.array([np.histogram(r[g_][k], bins)[0]
                                   for k in range(N_DIR)])
        co_eb  = int(np.max(np.sum(cnt('EB') > 0, axis=0)))
        all_pb = np.mean(np.all(cnt('PB_cw') > 0, axis=0)) * 100
        r.update(path=p_, steps=steps_cw(p_))
        print(f"{r['wd']:5.2f} {r['g']:5.2f} {r['wd']/(r['wd']+r['g'])*100:4.0f} "
              f"{r['n']:3d} | {r['steps']:5d} {r['steps']/r['n']:9.2f} "
              f"{'—' if w is None else 'EB%d' % (w+1):5s} | "
              + ' '.join(f"{f('EB', k):4.0f}" for k in range(N_DIR))
              + f" {f('GI', 0):4.0f} | "
              + ' '.join(f"{f('PB_cw', k):4.0f}" for k in range(N_DIR)) + ' | '
              + ' '.join(f"{f('PB_ccw', k):4.0f}" for k in range(N_DIR))
              + f" | {co_eb:5d} {all_pb:6.0f} {gap_ms(r['GI'][0], a, b):6.1f}")

    for r in results:
        r['title'] = (f"width {r['wd']*1e3:.0f} ms, pause {r['g']*1e3:.0f} ms — "
                      f"{r['steps']} steps in {r['n']} pulses")
    raster_grid(results, len(TRAIN_WIDTHS), len(TRAIN_PAUSES),
                f'CW velocity pulse trains ({hd.TURN_I*1e6:.0f} µA, grey = pulse): '
                'spike raster of all 13 neurons', 'ns_hd_4n_sweep_train.png')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sweep 5 — velocity amplitude up to GI depolarisation block               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Expected failure: a strong velocity input makes the EBs very active, the
# summed EB → GI current exceeds I_hold, GI stops firing (depolarisation
# block) and, with the inhibition gone, the EBs and PBs all fire.
BLOCK_AMPS = np.array([20.0, 25.0, 27.5, 30.0, 32.5, 35.0, 40.0, 50.0]) * 1e-6
BLOCK_PROTOCOLS = [('one 0.5 s pulse', 0.5, 0.0, 1),          # label, width, pause, n
                   ('100 ms on / 20 ms off, 6 s', 0.1, 0.02, 50),
                   ('one 6 s pulse', 6.0, 0.0, 1)]

def sweep_block(n_proc):
    import ns_hd_4n as hd
    i_hold = hd.I_MAX * 1e6
    jobs = []
    for amp_ in BLOCK_AMPS:
        for label, wd, g, n in BLOCK_PROTOCOLS:
            pb = [(T_VEL + m * (wd + g), wd, 'cw', amp_) for m in range(n)]
            jobs.append(dict(over={}, pb_pulses=pb, a=amp_, label=label,
                             t_end=pb[-1][0] + wd, T=pb[-1][0] + wd + SETTLE))
    print(f"GI-block sweep: {len(BLOCK_AMPS)} velocity amplitudes × "
          f"{len(BLOCK_PROTOCOLS)} protocols = {len(jobs)} runs "
          f"(I_hold = {i_hold:.0f} µA)")
    results = run_jobs(jobs, n_proc)

    # during the input: peak GI drive, time GI drive is above I_hold, longest
    # GI silence, most EBs / PB_cw firing in the same 50 ms bin.
    # after: which EBs fire in the last second (one EB = bump recovered)
    print(f"\n{'I_vel':>5s} {'protocol':27s} | {'GI peak':>7s} {'>I_hold':>7s} "
          f"{'GI gap':>6s} {'GI rate':>7s} | {'co-EB':>5s} {'co-PB':>5s} "
          f"{'all EB':>6s} | {'steps':>5s} after")
    print(f"{'(µA)':>5s} {'':27s} | {'(µA)':>7s} {'(ms)':>7s} {'(ms)':>6s} "
          f"{'(Hz)':>7s} | {'':5s} {'':5s} {'(%)':>6s} |")
    for r in results:
        a, b, T = T_VEL, r['t_end'], r['T']
        m = (r['t'] >= a) & (r['t'] < b + 0.3)
        over = np.sum(r['gi_drive'][m] > i_hold) * (r['t'][1] - r['t'][0]) * 1e3
        bins = np.arange(a, b + 1e-9, BIN)
        on = lambda g_: np.array([np.histogram(r[g_][k], bins)[0]
                                  for k in range(N_DIR)]) > 0
        eb_on, pb_on = on('EB'), on('PB_cw')
        end = [k + 1 for k in range(N_DIR) if rate(r['EB'][k], T - 1.0, T) > 5]
        after = ('bump on EB%d' % end[0] if len(end) == 1 else
                 'no bump' if not end else 'EB' + ','.join(map(str, end)) + ' all firing')
        steps = steps_cw(path(timeline(r, a - 0.3, T)))
        r['title'] = (f"{r['a']*1e6:.1f} µA, {r['label']} — GI peak "
                      f"{r['gi_drive'][m].max():.0f} µA")
        print(f"{r['a']*1e6:5.1f} {r['label']:27s} | {r['gi_drive'][m].max():7.1f} "
              f"{over:7.0f} {gap_ms(r['GI'][0], a, b + 0.3):6.1f} "
              f"{rate(r['GI'][0], a, b):7.0f} | {int(eb_on.sum(0).max()):5d} "
              f"{int(pb_on.sum(0).max()):5d} {np.mean(eb_on.all(0))*100:6.0f} | "
              f"{steps:5d} {after}")

    raster_grid(results, len(BLOCK_AMPS), len(BLOCK_PROTOCOLS),
                'CW velocity input of increasing amplitude (grey = pulse): '
                'spike raster of all 13 neurons', 'ns_hd_4n_sweep_block.png')


SWEEPS = {'grid': sweep_grid, 'duration': sweep_duration, 'repeat': sweep_repeat,
          'train': sweep_train, 'block': sweep_block}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--sweep', choices=SWEEPS, required=True)
    ap.add_argument('--jobs', type=int, default=16, help='parallel processes')
    args = ap.parse_args()
    SWEEPS[args.sweep](args.jobs)
