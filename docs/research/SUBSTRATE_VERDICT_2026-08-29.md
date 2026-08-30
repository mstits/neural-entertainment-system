# Substrate verdict — main run FAILED, control PARTIAL; the decomposition is the deliverable

**Date:** 2026-08-29. **Registration:**
`docs/proposals/SUBSTRATE_RUN_ADDENDUM_2026-08-29.md` (§1–§5 before any
compute; Addendum 2 after the main verdict, before any control compute)
over `scripts/eval_shared_substrate.py`'s in-file CONFIG.
**Receipts:** `docs/receipts/substrate/` (evals, manifests, composition,
collection, trim, training log, executor derivations); raw grids under
`runs/{shared_substrate,substrate_control_sep,substrate_pairs}/`.
**Adjudication:** every number below was derived independently twice —
executor and a second session that ran nothing, from raw per-seed eval
JSONs, verifier blind to the executor's read — with **exact agreement
both rounds** (4/4 per-leg counts each round; composition 4/4 SHAs;
trim 4/4 SHAs; head-routing hash-check 4/4; start-state audit clean).
**Ledger class: LEARNED-line negative result. Honest protocol
throughout** (cold entrance, greedy, sticky 0.25, jitter ±16, 50 eps ×
eval seeds {7,101}, per-episode RNG, 5 workers).

## The numbers

| level | RL specialist (like-for-like) | shared-trunk BC | separate-trunk BC | sharing effect |
|---|---|---|---|---|
| 1-1 | 43 | 31 | 11 | **+20** (transfer) |
| 1-2 | 43 | 0 | 0 | 0 (floor-bound — see caveat) |
| 1-3 | 21 | 0 | 14 | **−14** (interference) |
| 1-4 | 51 | 11 | 29 | **−18** (interference) |
| **sum** | **158** | **42** | **54** | net −12 |

The 158 threshold is the Addendum-1 like-for-like set, re-derived at
both verdict times from the four named receipts (1-2's 43 from the
`interference.jsonl` same-protocol control row, never the banked 38).

## Verdicts, per the registered rules

- **Main run: FAILED.** 42 vs 158; every level collapsed at the
  registered margin; not in the pre-declared 154..158 MIXED band; no
  VOID condition (8/8 per-seed results ok). Advisory exact binomial:
  P(X ≤ 42 | p₀ = 158/400) ≈ 1.5e-38.
- **Control: PARTIAL.** S_sep − S_shared = +12 > one collapse margin
  (10), so not METHOD-EXPLAINS; only 1-3 holds its like-for-like floor
  (14 ≥ 11), so not INTERFERENCE-DEMONSTRATED. Advisory:
  P(X ≥ 54 | p₀ = 42/400) = 0.034; P(X ≤ 54 | p₀ = 158/400) ≈ 2.2e-30.

## What is banked

1. **The decomposition — the deliverable.** Of the 116-clear gap
   between RL specialists (158) and the shared-trunk policy (42), the
   TRAINING METHOD costs 104 (158 → 54, separate trunks, identical
   data/exposure/epochs) and SHARING costs a net 12 (54 → 42). Anyone
   reading "shared trunks cause interference" off the main run alone
   would have been reading a BC-vs-RL gap with a small sharing term
   riding on top — which is why the main run's registered claim scope
   (interference-only) could not be delivered by its own design
   (MISTAKES `[confound]`, opened for exactly this).
2. **Sharing is real, level-specific, and BIDIRECTIONAL.** +20 positive
   transfer into 1-1 against −32 interference across 1-3/1-4. The
   aggregate net (−12) describes a system that does not exist; the
   per-level directions are the result. This reproduces the original
   pooled falsifier's signature (pooled BC CAPTURED 1-1 above its own
   specialist while destroying 1-2) at four levels.
3. **1-2 discriminates nothing here.** 0 clears in BOTH arms: offline
   BC cannot clear 1-2 at this exposure at all, so the level the
   original interference story was about is silent in this experiment
   — consistent with, and adding a colder-start replication to, the
   banked imitation-elimination result (clone-accuracy high, honest
   sticky eval 0.00).
4. **Offline BC at this exposure (37,834 pairs/level, 50 epochs,
   registered defaults) does not approach RL-trained specialists under
   the honest gate regardless of architecture** (42 and 54 vs 158).
   BC val-accuracy anti-correlated with eval at the extremes (1-3
   best-learned at 0.793 val → 0 clears) — the trainer's own
   proxy-metric warning, measured.

**Not banked, explicitly:** "interference confirmed" and "interference
refuted" — neither is supported. Interference is real, level-specific,
bidirectional, and an order of magnitude smaller than the method term
at this recipe.

## Named next step (NOT registered — operator's call)

The interference question at strength needs both arms trained by the
RL path (the only regime where either arm lives near 158). That is
expensive (four specialist-scale runs plus a shared-trunk RL run) and
is Matthew's call, not a thing to improvise. Nothing below that
answers it cleanly; nothing here licenses it.

## Cost

S1 collection ~4 min (600 eps), S2 training ~2 min, S3 eval ~8 min,
control (trim + 4 trainings + eval) ~12 min: **~26 minutes of machine
time** for the main result, the confound catch, and the decomposition.
