# PAUSE — head-direction network (branch `HD_Stimu`)

Paused 2026-10-01. Nothing from this work is committed.

## Where things stand

The 4-direction head-direction ring of `Fig3.1_4Nturn_color.png` is built and working in
`demo/ns_hd_4n.py`: 4 EB + 1 GI + 4 PB_ccw + 4 PB_cw, 28 synapses.

| Stage | Command | Status |
|---|---|---|
| Connection check | `--stage connections` | done, wiring confirmed by you |
| EB bump | `--stage bump` | done, no firing gap after the pulse |
| WTA, all 12 ordered EB pairs | `--stage wta` | done, 13/13 pulses correct |
| EB → PB copy | shown in `--stage wta` | done, PBk fires only with EBk |
| Turning by PB velocity input | `--stage turn` | done, 8/8 one-step moves (4 CW + 4 CCW) |

Run with `uv run python demo/ns_hd_4n.py --stage <name>`. WTA takes about 4 minutes, turn about 3.

## Current parameters

Neuron: `MSNParams()` defaults except **Cm = 133 nF** (100 nF ∥ 33 nF). I_min = 32.2 µA, I_hold = 100 µA.

| Bias | Value |
|---|---|
| I_0 EB | 24.1 µA |
| I_0 GI | 6.1 µA |
| I_0 PB | 2.0 µA |

| Pathway | Type | Weight | τ |
|---|---|---|---|
| EBk → EBk (self) | exc | 2.394 µA | 150 ms |
| EBk → GI | exc | 14.06 µA | 30 ms |
| GI → EBk | inh | 1.503 µA | 100 ms |
| EBk → PBk_ccw / PBk_cw (copy) | exc | 5.34 µA | 100 ms |
| PBk_ccw → EB(k−1), PBk_cw → EB(k+1) (shift) | exc | 1.4 µA | 100 ms |

| Input | Value |
|---|---|
| WTA test pulse on one EB | 30 µA × 0.5 s, every 2.5 s |
| Velocity pulse on all PB_cw or all PB_ccw | 25 µA × 0.5 s, every 2.5 s |
| Bump seed (turn stage) | 35 µA × 0.5 s on EB1 |

## Latest results

**WTA (30 µA pulses)**

| Quantity | Fig2.1 / Fig2.2 | Simulation |
|---|---|---|
| Winner rate, steady / peak | 85 / 185 Hz | 86 / 182 Hz |
| GI rate, steady / peak | 100 / 185 Hz | 85 / 207 Hz |
| Self-excitation, steady / peak | 35 / 55 µA | 32.0 / 50.6 µA |
| EB → GI, steady / peak | 45 / 75 µA | 36.5 / 77.9 µA |
| GI → EB, steady / peak | 20 / 27 µA | 13.3 / 28.7 µA |
| Transition time | 500 ms | about 180 ms |
| Longest silence after a pulse | none | 15.5 ms winner, 16 ms GI |
| PB copy rate | — | 102–115 Hz, starts 140–210 ms after EBk |

**Turning (25 µA velocity pulses)**

| Quantity | Value |
|---|---|
| Correct one-step moves | 8 of 8 |
| Copying PB, before / during pulse | 99 Hz / 176–180 Hz |
| Shift current onto next EB, before / peak | 13.7 / 24.3 µA |
| New EB starts / old EB stops after pulse onset | about 180 ms / 425–457 ms |
| Non-copying PBs during pulse | silent (27 µA total, below 32.2 µA) |

## Decisions you made (keep these)

- Winner EB and GI must never stop firing when a pulse is removed.
- τ is the same within a pathway and different between pathways.
- I_0 may be high, even near or above I_min; PBs use a low I_0 with strong EB copy.
- Cm raised to 133 nF to bring f_max near 185 Hz. Lowering I_hold was tried and dropped.
- Velocity pulses go to the PB neurons, not to the EBs. Only the initial seed is an EB pulse.
- Copying PB should fire around 100 Hz, and 180–200 Hz during the velocity pulse.

## Open points

1. **WTA pulse band is narrow.** 28 and 31 µA work, 25 µA leaves a ~130 ms GI pause on
   opposite-side switches, 35 µA (the Fig2.1 value) fails because GI is pushed past I_hold.
2. **Velocity pulse target.** I send it to the whole PB_cw or PB_ccw population. Not yet
   confirmed by you that this is what you meant, rather than only PBk.
3. **Pulse length and repetition.** Only 0.5 s pulses every 2.5 s were tested. A longer pulse
   may move the bump more than one step; a faster repeat is untested.
4. **PB rate during the pulse** is 176–180 Hz, just under the 180–200 Hz target. The pulse is
   capped near 30 µA so non-copying PBs stay silent.
5. **Robustness not swept in Brian2.** Working windows for the PB → EB weight and the velocity
   amplitude come from a quick rate model only (roughly ±10–20%).
6. **Still off against the figures:** steady GI → EB and EB → GI currents are low, GI steady
   rate is 85 Hz, and the WTA transition is about 180 ms instead of 500 ms.
7. **Git:** `git status` shows six tracked demo files as deleted (`ns_msn_compass_demo`,
   `ns_msn_if_sweep`, `ns_msn_v1`, each `.py` and `.png`). I did not delete them.
   `git checkout -- demo/` restores them if that was not intended.

## Suggested next steps

- Confirm point 2, then test pulse duration and back-to-back pulses (point 3).
- Sweep the PB → EB weight and velocity amplitude in Brian2 to get real margins (point 5).
- Decide whether the 35 µA WTA pulse must work with PBs active (point 1).
- Commit `demo/ns_hd_4n.py` and the five figures once you are happy.

## Files

- `demo/ns_hd_4n.py` — network, all four stages
- `demo/ns_hd_4n_connections.png` — connectivity matrix
- `demo/ns_hd_4n_bump.png` — bump test
- `demo/ns_hd_4n_wta.png`, `demo/ns_hd_4n_wta_rates.png` — WTA overview and per-switch rates
- `demo/ns_hd_4n_turn.png` — turning test
- Reference figures: `Fig3.1_4Nturn_color.png`, `Fig2.1.png`, `Fig2.2.png`
