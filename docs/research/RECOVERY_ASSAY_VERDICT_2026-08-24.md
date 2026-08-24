# Recovery assay — VERDICT: the wall is BOTH, in measured proportions

The registered follow-up from the v26 adjudication, executed
2026-08-24. Receipts: runs/recovery_assay/{manifest.json, verdict.json,
states/, solve_ep*/}; collection via `eval_game --dump-stick-states`
(the verified harness — clear_rate 0.767 on the collection run itself,
reproducing the banked 0.76).

## Protocol as run

60 honest episodes of the banked 1-1 control (cold entrance, greedy,
sticky 0.25, jitter ±16). Every divergent stick (executed != chosen)
snapshotted post-stick: 1,592 states. 14 non-clear episodes; for each,
the LAST divergent stick before the episode end went to a 10-minute
8-worker Go-Explore adjudication: does ANY continuation clear 1-1
from exactly that state?

(Assay integrity: the first pass scored 0/14 twice — first from a
3-minute budget shown too short by a 10-minute manual probe that
solved, then from a stdout-grep success detector that could never fire;
ground truth is the solutions/ directory. Both defects fixed in
scripts/recovery_assay.py; the verdict below is filesystem-scored.)

## Result

| class | n | recovered |
|---|---|---|
| timeout episodes (alive at last stick) | 5 | 5 — sanity holds |
| true death-preceding sticks | 9 | **3 (33%)** |
| … of those landing 1–2 steps pre-death | 3 | **0** |
| … of those landing ≥8 steps pre-death | 6 | 3 |

## Reading

1. **The fatal window is real.** A stick that lands 1–2 steps before
   death IS the death — a mid-air commitment no continuation can undo
   (the solver, free to try every action sequence, finds nothing).
   Under p=0.25 these are irreducible: the honest ceiling on this
   level is strictly below 1.0 for ANY policy class. This is the
   mechanical-fate slice.
2. **The trainable slice is real too.** A third of true death sticks
   had a recovering continuation the policy failed to find — and the
   solver hands us the recovery trajectory as a receipt. These states
   plus their solutions are ready-made curriculum fuel
   (solver-as-teacher on post-stick states), the TRUE-AI-PLAYS
   pattern: every solved recovery = training data.
3. **Ceiling arithmetic (rough).** 60 episodes: 46 cleared, 5 timed
   out (length cap artifact), 6 mechanically-fatal-class deaths, 3
   trainable deaths. Perfect recovery training would take ~0.767 to
   ~0.83–0.85 on this level; the remaining gap to 1.0 is fate plus
   the timeout cap. The banked per-level rates (43/38/21/51) likely
   decompose the same way — measurable per level with this exact
   assay.

## Next (not run)

- Post-stick recovery distillation on 1-1: BC/DAgger the 3 recovery
  solutions (plus more sticks mined at scale — 1,592 states exist)
  into the control policy; re-run honest eval; gate = clear_rate
  above 0.80 with the same protocol.
- Repeat the assay on 1-2 (the 21% level): if its fatal-window share
  is much larger, the 1-2 BANKED verdict gets a mechanical
  explanation and stays closed with better evidence.

## Addendum: the 1-2 assay (same day)

Collection: 60 honest episodes of the banked consol2 artifact from
stage_03 (baseline reproduced 0.367; run scored 25/60), 1,337
snapshots. 16 of 35 death-sticks adjudicated at 10 solver-min each
(runs/recovery_assay_1_2/verdict.json):

| | 1-1 | 1-2 |
|---|---|---|
| death-sticks recovered | 3/9 (33%) | 3/16 (19%) |
| sticks ≤4 steps pre-death | 3/9 (33%) | **11/16 (69%)** |
| … recovered of those | 0/3 | 1/11 |
| implied honest ceiling | ~0.83–0.85 | **~0.53** |

**1-2's fatal window dominates.** Two-thirds of its sticky deaths are
decided within 4 steps — before any policy response is possible — and
even 45-step-gap states at the pole mostly have no recovering
continuation the solver can find. This is the mechanical explanation
for the paradigm-exhaustion verdict: hazard veto, options, imitation,
and recurrence all failed against a wall that is mostly PHYSICS at
p=0.25. The 1-2 BANKED verdict stands, now with receipts; its honest
ceiling under this protocol is ~0.5, and the banked ~0.37-0.40 sits
much closer to its ceiling than 1-1's 0.767 does to its ~0.85.

Routing rule going forward: run this assay BEFORE spending training
effort on any level's sticky rate — the recoverable share IS the
budget-worthiness signal.
