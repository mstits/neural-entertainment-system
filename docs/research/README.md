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
| `RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md` | **The outcome**: 1-1 learned (63–67%); 1-2 formally falsified for the policy class (three concurring seeds, pre-registered criteria); measured robustness profile; no published agent clears 1-2 under this protocol |

## Thread 2 — Search in silently-looping castle mazes

The question: can Go-Explore search solve SMB's aliasing mazes (4-4, 7-4,
8-4) without game internals? Score so far: 4-4 and 7-4 solved by the
consultation's heuristic-inversion recipe; 8-4 open.

| Doc | What it is |
|---|---|
| `MAZE_DOSSIER_2026-07-24.md` | v1: five failed cell representations on 4-4, measured failure modes |
| `MAZE_PROMPT_2026-07-24.md` | v1 driving prompt |
| `MAZE_DOSSIER_V2_2026-07-24.md` | v2: the route-byte probe protocol executed; channel-leakage caught by its own causal filter; root hypothesis killed |
| `MAZE_DOSSIER_V3_2026-07-26.md` | v3: the 8-4 campaign (eight attempts, ~110M steps); the $0750 semantics reckoning; the water-exit wall, precisely characterized |
| `MAZE_PROMPT_V3_2026-07-26.md` | v3 driving prompt (current open consult) |

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
