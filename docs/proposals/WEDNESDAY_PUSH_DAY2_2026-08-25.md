# THE WEDNESDAY PUSH — Day 2 continuation

Continuation of docs/proposals/WEDNESDAY_PUSH_2026-08-24.md. Written
2026-08-25 with RG-1 (live Zelda gate) finishing its final run and v27
(fresh-recovery training, 4 seeds) complete and awaiting honest-gate
scoring. Doctrine unchanged: purity line, honest eval, pre-registered
gates, receipts for every claim, VOID != FAIL. Target: consume the
week's remaining Claude utilization on real advancement by Wednesday
morning.

## What Day 1 shipped (for orientation, not re-litigation)

Odometer core + scene detection; two generic solver death-detector
fixes; a GRU policy-class A/B (FAIL, mechanism untested — the stick-
detection probe then showed detection was never the wall); the
recovery assay (1-1 decomposed into a fatal window + a trainable
third; 1-2 closed with physics, ceiling ~0.53); three post-hoc
training families all FAIL on the consolidated 1-1 artifact (isolated-
optimum meta-finding); a hardening wave (16 confirmed bugs fixed, both
parity mysteries closed, suite green); League onboarding wave 1 (12
games classified) + wave 2 (11 new games minted with liveness
receipts); the room-graph engine (T1-T4, RG-0 falsifier 9/9, a second
audit pass that found and fixed two real bugs before any live compute
ran); v27 (fresh-run curriculum with the 27 mined recovery states
interleaved from the start, DR-mandated ReDo dormant-neuron recycling,
launched and complete).

## Immediate scoring (small, do first, not a workflow)

- v27 honest gate: 4 seeds x 2 checkpoints (peak-entrance + final) x 2
  eval seeds x 50 episodes, per docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md.
  PASS >= 0.80 / FAIL <= 0.767 / MARGINAL between. Run SEQUENTIALLY,
  never concurrent with any solver/training burst (the lesson from
  today's near-miss).
- RG-1 verdict: assemble against runs/room_graph/PREREG.md's exact
  §6 criteria (RG-1a validity, RG-1b routing lift, RG-1c integrity —
  note the control-harness gap PREREG.md flagged, RG-1d perf, RG-1e
  edge validity) from the four completed run directories.

## New lanes for Day 2

**Lane F — v27-successor design (contingent, prepare regardless).** If
v27 PASSES, the fresh-recovery-curriculum shape becomes the standard
recipe for every solver-taught level. Design (don't run) the recipe
generalized: for 1-2, 1-3, 1-4 — mine recovery states via the existing
assay + solver machinery, merge into a from-scratch curriculum ladder,
register the honest gate per level. If v27 FAILS or is MARGINAL, this
lane instead designs the v28 capacity experiment the registration
named as the fallback (parameter-budget hypothesis). Either way,
produce ready-to-launch configs and a pre-registration doc, gated on
reading the actual v27 verdict once scored.

**Lane G — hardening wave 2 (broader sweep).** Day 1's audit covered
the newest surfaces (odometer/scene, five solver mechanisms, assay
scripts, trainer paths, room-graph, ReDo). Untouched by any hostile
audit: scripts/engine_driver.py (the two-lane scheduler that is
supposed to run this whole system unattended), the pyo3 bindings in
nes_core/src/python.rs and pool.rs broadly (not just the odometer
additions), and src/training/checkpoint_manager.py +
src/training/go_explore.py. Same pattern: find -> adversarially
verify -> fix -> test.

**Lane H — RG-2 Metroid (report-only, per the room-graph design's own
T6).** Once RG-1 is scored and the machine is free, run the Metroid
side of the room-graph engine as a report-only exercise (no pre-
registered pass/fail — the design doc calls this a capability
demonstration, not a gate). Reconcile the scene-noise vs fingerprint
room-count question RG-0 already flagged.

**Lane I — onboarding wave 3.** Classify the 11 games minted in wave 2
(Zelda already spoken for by the room-graph lane; the other 10: Ice
Climber, 1942, Galaga, Arkanoid, Blaster Master, Bionic Commando,
Chip'n Dale, Batman, Paperboy, Tetris) with the full odometer-era
pipeline: gate, profile, smoke, classification.

## Sequencing

Score v27 + assemble RG-1 first (small, sequential, no new workflow
needed for the scoring itself — but writing up the RG-1 verdict doc
and reconciling it against every pre-registered criterion IS
workflow-sized and can run concurrently with the scoring). Lanes
F/G/I can launch immediately and concurrently (CPU-light: design,
audit-and-fix, and onboarding gates/light smokes). Lane H (RG-2) waits
for a free machine. Every wave ends in a commit; nothing is claimed
without a receipt.
