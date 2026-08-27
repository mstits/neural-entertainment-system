# Research Record

This directory is the project's externally-reviewed research loop: data-rich
dossiers go out to deep-research consultation, concrete recipes with
pre-registered signposts come back, results (positive or negative) get
documented at the same evidentiary standard. Two threads:

## Thread 1 — Learned SMB under the honest protocol

The question: can a policy *learn* to clear SMB levels under the
research-standard evaluation (cold start, greedy, 25% sticky actions, start
jitter — Machado et al. 2018)?

| Doc | What it is |
|---|---|
| `FULL_DOSSIER_2026-07-22.md` | System architecture, six-month history, failure taxonomy — the base context document |
| `LEARNED_STICKY_DISTILLATION_BRIEF_2026-07-22.md` | Brief v1: the 1-2 wall, first consultation round |
| `LEARNED_STICKY_DISTILLATION_BRIEF_v2_2026-07-22.md` | Brief v2: the v1 recipe's diagnostic failure (greedy vs sampled) |
| `RESEARCH_PROMPT_2026-07-22.md`, `RESEARCH_PROMPT_V3_2026-07-23.md` | Driving prompts for the consultation rounds |
| `DOSSIER_V3_2026-07-23.md` | Five verified reward-shaping exploits closed; the imitation solution space eliminated with data; the gate-mirage result |
| `RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md` | **The outcome, AS RE-SCOPED 2026-08-27**: 1-1 learned (63–67%); the **CGSA-PPO recipe** failed its pre-registered signposts on three concurring seeds; measured robustness profile. The original "1-2 formally falsified for the policy class" generalization is **WITHDRAWN** — the same policy class at twice the width, from the same entrance, later cleared 38/100 (`CLAIMS.md` ADDENDUM N-1). The literature sentence is weakened to "we are aware of no published per-level 1-2 clear rate under Machado sticky-0.25, in either direction" — its cited source artifact does not exist in this tree. See `LEARNING_TRACK_AUDIT_2026-08-27.md` §9 |

## Thread 2 — Search in silently-looping castle mazes

The question: can Go-Explore search solve SMB's aliasing mazes (4-4, 7-4,
8-4) without game internals? **Resolved 2026-07-27: 4-4, 7-4, and 8-4 all
fell** (8-4 to the pipe-entry macro), completing the full 32-level
power-on-to-victory chain (commit db44fc7; see
`docs/receipts/full_run/receipts.json` and
`runs/live_show/smb_4_4_micro/chain_verify.json`) — an EXHIBITION-ledger
result (search, not a learned policy; see `../../CLAIMS.md`).

| Doc | What it is |
|---|---|
| `MAZE_DOSSIER_2026-07-24.md` | v1: five failed cell representations on 4-4, measured failure modes |
| `MAZE_PROMPT_2026-07-24.md` | v1 driving prompt |
| `MAZE_DOSSIER_V2_2026-07-24.md` | v2: the route-byte probe protocol executed; channel-leakage caught by its own causal filter; root hypothesis killed |
| `MAZE_DOSSIER_V3_2026-07-26.md` | v3: the 8-4 campaign (eight attempts, ~110M steps); the $0750 semantics reckoning; the water-exit wall, precisely characterized |
| `MAZE_PROMPT_V3_2026-07-26.md` | v3 driving prompt — led to the recipe that closed 8-4 and the full game |

## Thread 3 — Playing one non-SMB game well

The question: pick the single most tractable game outside SMB and take it as
far as it will go, against a bar fixed before compute.

Rygar was chosen on evidence, not preference: of Rygar / Contra / Kung Fu /
Zelda it was the **only one of the four whose declared progress signal passed
`scripts/progress_signal_gate.py` at HEAD**. The other three read SIGNAL
UNUSABLE, and Zelda is purity-blocked besides.

| Doc | What it is |
|---|---|
| `RYGAR_CAMPAIGN_2026-08-26.md` | The R1 campaign: verdict FAIL against the pre-registered bar; the wall moved 1,536 → 4,608 px of verified first-visit depth; the odometer-ratchet instrument defect; the fifth vacuous gate; the gate numbers for all four candidate games; the Kung Fu high-byte negative |
| `CONTRA_WALL_2026-08-27.md` | The gx-3072 wall: **held** (best gx 3072 vs prior 3072, no tape). What it physically is — a screen-locked lethal arena behind an exact hard camera stop, with the agent alive and in control inside it; the boundary-resident attack family falsified across eight arms; the prior deflated from "9 campaigns / 3,030 occurrences" to ~8 searches from 2 root states; and the clear hook shown unvalidatable until a stage boundary is reached |

Receipts: `../receipts/rygar/r1_tape_gx6242.json` (the replayable tape,
tracked rather than left under gitignored `runs/`), guarded by
`../../tests/test_rygar_r1_tape.py`, whose structural half runs with no ROM so
the receipt cannot fall to zero coverage on a fresh checkout. Room-fingerprint
decline: `../receipts/room_fp/rygar.md`.

Contra preserves **no** tape: no trajectory in that campaign exceeded 3072, so
there is nothing to guard. Its attack receipts live under gitignored
`runs/contra_wall/A1..A8/` and are indexed in the write-up.

**Ledger: EXHIBITION.** Search output. No policy was trained for this game and
no honest-protocol evaluation was run.

## Process rules (learned the hard way, kept on purpose)

1. Consult at decision forks with data-rich dossiers, never mid-execution.
2. Every dossier carries numbers, the refuted-approaches list, and our
   measured throughput economics.
3. Every accepted recipe carries pre-registered signposts and kill criteria
   before it gets compute.
4. Report back after every falsification — the loop compounds.
5. RAM interpretations are verified observationally (change-rate measured)
   before any search or reward logic keys on them.
6. Negative results are documented at the same standard as positives and are
   quotable with their data (see `CLAIMS.md` at the repo root).
