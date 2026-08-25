# THE WEDNESDAY PUSH — Day 3 continuation

Continuation of WEDNESDAY_PUSH_2026-08-24.md and
WEDNESDAY_PUSH_DAY2_2026-08-25.md. Written with v27 scored FAIL
(best-of-4 0.530), v28 (the pre-registered capacity experiment)
launched and running, RG-1 nearly fully adjudicated (1a/1b/1c/1d PASS;
1e edge-validity + 1a-stability pending a careful re-verification
pass), and two hardening waves + three onboarding waves banked.
Doctrine unchanged: purity line, honest eval, pre-registered gates,
receipts, VOID != FAIL.

## What's running right now (do not duplicate or contend with)

- v28 capacity training: 4 seeds x 250 iters, ~8h, local M4 compute.
  Nothing else heavy should launch concurrently until it completes.
- RG-1e/1a-stability careful re-verification: a Sonnet workflow
  re-deriving the edge-validity replay through the tested Solver
  machinery (a hand-rolled first attempt found a real methodology bug
  and then a suspicious 55% match rate that needs confirming or
  refuting properly).

## New lanes for Day 3

**Lane J — RG-1 final verdict + RG-2 Metroid.** Once the careful
re-verification lands, assemble the complete RG-1 verdict document
(all five criteria, PASS/FAIL per the pre-registered kill criteria in
runs/room_graph/PREREG.md) and bank it. Then run RG-2 (Metroid,
report-only per the design's own T6) — reconcile the scene-noise vs
fingerprint room-count question RG-0 flagged.

**Lane K — hardening wave 3.** Two waves have covered: odometer/scene
core, five solver mechanisms, assay scripts, trainer paths, room-graph,
ReDo, engine_driver.py, pyo3 bindings broadly, checkpoint_manager.py,
go_explore.py. Untouched: src/training/exploration_controller.py,
src/training/hazard_model.py + hazard_mask.py (the veto mechanism that
collapsed a working policy 31->0 months ago — worth a fresh look now
that the project understands failure classes better), scripts/train_
hazard.py, and the reward-function stack (src/utils/reward_functions.py).
Same pattern: find -> adversarially verify -> fix -> test.

**Lane L — League onboarding wave 4.** Mint start states for the next
batch of unminted ROMs (survey roms/ for titles with neither a
_start.state.bin nor a wave-2/3 mint attempt) and classify them with
the full odometer-era pipeline. Target 10-12 games.

**Lane M — capability/claims synthesis, Day 3.** Fold v27's FAIL
verdict (plus its two secondary findings — peak instability within a
single from-scratch run, training-telemetry unreliability), v28's
launch, hardening wave 2's P0, and onboarding wave 3's signed-axis fix
into CLAIMS.md and a dated capability report addendum.

## Sequencing

Lanes K, L, M can launch immediately and concurrently — all CPU-light
(reasoning, writing, brief tests, light smokes). Lane J's RG-2 half
needs the machine free of v28 contention for its own live compute
burst; hold it or run at reduced worker count until v28 finishes or a
natural gap appears. Every wave ends in a commit; nothing is claimed
without a receipt.
