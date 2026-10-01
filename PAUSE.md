# PAUSE — head-direction network (branch `HD_Stimu`)

Updated 2026-10-02. `demo/ns_hd_4n.py` is committed up to the turning stage; the PB
depolarisation-block fix and `demo/ns_hd_4n_sweep.py` are not committed.

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
| I_0 PB | 5.5 µA |

| Pathway | Type | Weight | τ |
|---|---|---|---|
| EBk → EBk (self) | exc | 2.394 µA | 150 ms |
| EBk → GI | exc | 14.06 µA | 30 ms |
| GI → EBk | inh | 1.503 µA | 100 ms |
| EBk → PBk_ccw / PBk_cw (copy) | exc | 4.9 µA | 100 ms |
| PBk_ccw → EB(k−1), PBk_cw → EB(k+1) (shift) | exc | 1.4 µA | 100 ms |

| Input | Value |
|---|---|
| WTA test pulse on one EB | 30 µA × 0.5 s, every 2.5 s |
| Velocity pulse on all PB_cw or all PB_ccw | 25 µA × 0.5 s, every 2.5 s |
| Bump seed (bump and turn stages) | 30 µA × 0.5 s on EB1 |

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
| PB copy rate | — | about 100–112 Hz |
| Peak PB drive (I_0 + copy) | below I_hold = 100 µA | 93.3 µA |

**Turning (25 µA velocity pulses)**

| Quantity | Value |
|---|---|
| Correct one-step moves | 8 of 8 |
| Copying PB, before / during pulse | 99 Hz / 176–180 Hz |
| Shift current onto next EB, before / peak | 13.7 / 24.3 µA |
| New EB starts / old EB stops after pulse onset | about 180 ms / 425–457 ms |
| Non-copying PBs during pulse | silent (30.5 µA total, below 32.2 µA) |
| Peak PB drive | 89.7 µA (I_hold = 100 µA) |
| Longest pause of a copying PB | 25.9 ms |

**PB depolarisation block (fixed 2026-10-02)**

With the old values (copy 5.34 µA, PB I_0 2.0 µA, 35 µA seed) the copy current into PB1
peaked at 102 µA during the seed pulse, above I_hold, and PB1 went silent for 84 ms. WTA
pulses peaked near 97 µA. Now: copy 4.9 µA, PB I_0 5.5 µA, seed 30 µA. Steady PB drive is
unchanged (about 47 µA, 100 Hz). All stages rerun: bump ok, WTA 13/13, turn 8/8.

**Brian2 sweeps (`demo/ns_hd_4n_sweep.py`, run with the OLD PB values, not rerun since)**

| Sweep | Result |
|---|---|
| PB → EB weight, at 25 µA velocity | works 1.2–1.6 µA (nominal 1.4); 1.1 and below no move; 1.7 and above bump not held |
| Velocity amplitude, at 1.4 µA weight | works 15–30 µA; 32.5 µA fails (non-copying PBs fire) |
| Pulse duration, 25 µA | 0.2 s and below: no move; 0.3–1.0 s: one step; 1.25–1.5 s: two; 2 s: three; 3 s: four |
| GI pause at borderline durations | 116–175 ms at 0.2, 0.3 and 1.0 s |
| 4 × CW, pause between pulses | needs 0.2 s or more (0.3 s to avoid a 218 ms GI pause); 0.1 s or less gives 3 steps |
| CW then CCW | works at every pause, including none |

**Sweeps rerun with the NEW PB values (2026-10-02)**

| Sweep | Result |
|---|---|
| Single pulse width, 25 µA | 0.2 s and below: no move; 0.3–1.0 s: one step; 1.25–2.0 s: two; 3 s: three |
| GI pause at borderline widths | 123 ms at 0.2 s, 182 ms at 0.3 s |
| 6 s CW pulse train, duty 67% or more (pulses 50 ms or longer) | bump rotates continuously, about 1 step per second, whatever the pulse width |
| 6 s CW pulse train, short pulses (100 ms or less) at duty 50% or less | no move; only PB1_cw fires faster (the 100/100 ms case still rotates) |
| Train of 500 ms pulses, 300 ms pause | exactly one step per pulse (8 of 8) |
| Neurons firing together in a train | never more than 2 EBs (during a hand-over); never all PBs; GI never silent (80–140 Hz) |
| GI pause in trains with 200–300 ms pauses | 70–150 ms |

At 25 µA a fast pulse train does not make every neuron fire: non-copying PBs stay below
threshold by design.

**GI depolarisation-block failure (`--sweep block`, 2026-10-02)**

This is the failure you expected: strong PB input → all PBs fire → all EBs fire → summed
EB → GI current exceeds I_hold (100 µA) → GI stops → no inhibition → EBs and PBs all fire.

| Velocity amplitude | One 0.5 s pulse | 6 s train (100 ms on / 20 ms off) or one 6 s pulse |
|---|---|---|
| 20–25 µA | one step, GI drive peaks 61–63 µA | bump rotates, GI drive peaks 59–69 µA |
| 27.5 µA | one step (GI peak 71 µA), non-copying PB_cw fire during the pulse | FAILS: GI drive 295–355 µA, GI silent, all 4 EBs and 8 PBs fire |
| 30–32.5 µA | no step, bump returns to EB1 after a ~0.2 s silence of the whole network | FAILS, GI blocked about 1 s after onset |
| 35 µA | bump lost (no EB fires afterwards) | FAILS |
| 40–50 µA | FAILS (GI peak 219–279 µA) | FAILS |

The failure latches: after the input is removed the EBs keep each other and GI above
I_hold, GI stays blocked and the network does not recover. The trigger is the amplitude
(I_0 PB + pulse above I_min = 32.2 µA, i.e. pulse above 26.7 µA), not the repeat rate.

With PB I_0 now 5.5 µA, the upper velocity limit is expected to drop to about 26.5 µA
(I_0 + pulse must stay below 32.2 µA). Not yet checked by a rerun.

## Decisions you made (keep these)

- Winner EB and GI must never stop firing when a pulse is removed.
- A copying PB must never stop firing either: PB drive stays below I_hold.
- τ is the same within a pathway and different between pathways.
- I_0 may be high, even near or above I_min; PBs use a low I_0 with strong EB copy.
- Cm raised to 133 nF to bring f_max near 185 Hz. Lowering I_hold was tried and dropped.
- Velocity pulses go to the PB neurons, not to the EBs. Only the initial seed is an EB pulse.
- Copying PB should fire around 100 Hz, and 180–200 Hz during the velocity pulse.

## Open points

1. **WTA pulse band is narrow.** 28 and 31 µA work, 25 µA leaves a ~130 ms GI pause on
   opposite-side switches, 35 µA (the Fig2.1 value) fails because GI is pushed past I_hold.
2. **Velocity pulse target.** Confirmed by you 2026-10-02: the whole PB_cw or PB_ccw population.
3. **Pulse length and repetition.** Swept, see above. One step needs a 0.3–1.0 s pulse and
   at least 0.3 s between same-direction pulses.
4. **PB rate during the pulse** is 176–180 Hz, just under the 180–200 Hz target. The pulse is
   now capped near 26.5 µA so non-copying PBs stay silent (25 µA used, 1.7 µA margin).
5. **Robustness.** Swept in Brian2 with the old PB values (table above): weight −14% / +14%.
   The grid has not been rerun with the new PB values.
6. **Still off against the figures:** steady GI → EB and EB → GI currents are low, GI steady
   rate is 85 Hz, and the WTA transition is about 180 ms instead of 500 ms.
7. **Seed above 30 µA** would again push the PB toward block (35 µA seed gives 97.5 µA).

## Suggested next steps

- Decide whether the GI-block failure should be prevented (e.g. limit on velocity amplitude) or only documented.
- Rerun `--sweep grid` with the new PB values (about 30 minutes).
- Decide whether the 35 µA WTA pulse must work with PBs active (point 1).
- Commit `demo/ns_hd_4n.py`, `demo/ns_hd_4n_sweep.py` and the figures once you are happy.

## Files

- `demo/ns_hd_4n.py` — network, all four stages
- `demo/ns_hd_4n_connections.png` — connectivity matrix
- `demo/ns_hd_4n_bump.png` — bump test
- `demo/ns_hd_4n_wta.png`, `demo/ns_hd_4n_wta_rates.png` — WTA overview and per-switch rates
- `demo/ns_hd_4n_turn.png` — turning test
- `demo/ns_hd_4n_sweep.py` — sweeps (`--sweep grid | duration | repeat | train | block`)
- `demo/ns_hd_4n_sweep_grid.png`, `demo/ns_hd_4n_sweep_duration.png`, `demo/ns_hd_4n_sweep_train.png`, `demo/ns_hd_4n_sweep_block.png` — sweep figures
- Reference figures: `Fig3.1_4Nturn_color.png`, `Fig2.1.png`, `Fig2.2.png`
