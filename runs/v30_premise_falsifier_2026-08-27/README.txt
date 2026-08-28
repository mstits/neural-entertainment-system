V30 PREMISE FALSIFIER — 2026-08-27
Cheap gate run BEFORE any v30 training spend. ~35 min of compute.

QUESTION: at the registered tau=0.25, does ReDo (a) fire, and (b) change
anything measurable about the network? And does the armed check void an
inert run instead of letting it report a number?

ARMS (all: configs/mario_1_1_v27_seed0.yaml verbatim, hidden 64 / trunk 32,
seed 0, --no-resume --no-supervise --strict-config, merged 785-rung ladder)
  pilot_tau0.25_h64.log        20 iters, redo_tau 0.25   (registered point)
  pilot_tau0.15_h64.log        20 iters, redo_tau 0.15   (below the sweep)
  pilot_tau0.25_h96.log        20 iters, redo_tau 0.25, tile_hidden_dim 96
                               (closes the width caveat the registration
                               named as an accepted risk)
  control_tau0.025_h64_VOIDED  30 iters requested, redo_tau 0.025 (v27/v28's
                               own point) — RAISED AT ITERATION 26

ANSWERS
  (a) FIRES. cum_recycled 353 over 20 iters, 19/20 iterations. Iter 1 is a
      BIT-EXACT reproduction of the banked sweep isolate_tau0.25.log:
      "fc2 5/32 recycled 5 cum 5 agree 0.8142 max_dlogit 0.288576".
  (b) CHANGES THE NETWORK, a lot. agree median 0.856 / min 0.702,
      max_dlogit to 0.802. The matched pair is bit-identical at iters 0-1
      (same mean_return to the decimal) and permanently diverged after the
      first recycle. Policy entropy holds 1.65-1.71 in the treatment while
      the control decays to 1.346 — a sustained ~0.31 nat gap.
  (c) ARMED CHECK WORKS, verified by running the inert case rather than
      reasoning about it. See the tail of control_tau0.025_h64_VOIDED.log.

BUT — THE REGISTERED DOSE IS THE REGISTRATION'S OWN FORBIDDEN DOSE.
tau 0.25 settles at 20 of 32 trunk units re-initialized EVERY iteration
from iter 5 on (62%). The registration's "RISK I REFUSE" clause defines
exactly this at tau 0.50 and rules it out. The 2-iteration sweep could not
see it: at orthogonal init the fc2 score minimum is 0.2848 and only 5 units
fall below 0.25; by iter 5 the minimum is 0.109 and 20 do.

"0.25 IS THE SMALLEST TAU THAT FIRES" IS FALSE. It rests on iters 0-1. In
the control the fc2 score minimum falls below 0.15 on 22/26 iterations and
below 0.10 on 10/26. tau 0.15 fires from iter 4 (176 units, 16/20 events).

WIDTH DOES NOT RESCUE IT. At tile_hidden_dim 96 the picture is the same:
19 of 32 trunk units per iteration (59%), cum 342, fc1 0/96 dormant. The
dormancy lives entirely in the fixed 32-unit trunk, so the finding is
width-invariant. Note the h96 arm's median agree is HIGHER (0.901) at the
same dose — more evidence that agreement does not measure the damage.

NO DORMANCY SUBSTRATE IN fc1 AT ALL: 0 of 64 (and 0 of 96) units at every
tau from 0.025 to 0.25 across all 86 measured iterations; fc1's score
minimum never drops below 0.309. The intervention is confined to a
32-unit bottleneck.

A FIXED TAU IS NOT A STABLE OPERATING POINT. Untreated control fc2 minimum:
0.285 (it0) -> 0.127 (it5) -> 0.101 (it15) -> 0.079 (it24), still falling.
The registration budgets 250 iterations on a threshold measured over 2.

REPRODUCE
  .venv/bin/python runs/v30_premise_falsifier_2026-08-27/analyze.py \
    runs/v30_premise_falsifier_2026-08-27/*.log
  .venv/bin/python scripts/redo_arm_gate.py \
    runs/v30_premise_falsifier_2026-08-27/pilot_tau0.25_h64.log --tau 0.25

ARMED-CHECK VERIFICATION (the anti-vacuity test for this workflow)
  armed_check_inert_case_VOIDED.log — a fast 8-env/128-step config at
  redo_tau 0.025 asked for 40 iterations. It raised at iteration 26, exit
  code 1, and printed no clear rate, no verdict, no number of any kind.
  Verified by RUNNING the inert case, not by reasoning about it.
  control_tau0.025_h64_VOIDED.log is the same abort on the full 60-env
  recipe (~10 min to the deadline, vs the 7h10m x 4 seeds that v27 and
  v28 each burned inert).

  scripts/redo_arm_gate.py adjudicates a finished run.log. Applied to all
  eight banked v27/v28 logs AT THEIR OWN registered tau of 0.025 it
  returns "VERDICT: VOID (redo never fired)", exit 2, on all eight — the
  gate reproduces the correct verdict on the historical failure. It is
  structurally incapable of printing PASS or FAIL for an unarmed seed.

  tests/test_redo_armed_gate.py — 22 tests. Every check ships with its
  revert-verified failure, executed, not asserted:
    - delete the cum_recycled==0 branch      -> 10 of 22 fail
    - delete the V6 over-dose ceiling        ->  3 of 22 fail
    - pool fc1+fc2 instead of worst-hit layer->  5 of 22 fail

  V6 (over-recycling ceiling) is NEW and was written because this pilot
  found the hole: the registered A4 identity abort voids below median
  agree 0.60, and the tau-0.25 arm sits at 0.856 (h96: 0.901) while
  re-initializing 62% of the trunk every iteration. Zeroing the outgoing
  actor/critic columns preserves the OUTPUT whatever the dose, so
  agreement is structurally insensitive to the damage. A4 alone would
  have certified a per-iteration partial network reset as surgical.

  Gate verdicts over every arm here, each at its own operating point:
    pilot_tau0.25_h64  VOID-OVERDOSE  (62.5% of the worst-hit layer)
    pilot_tau0.25_h96  VOID-OVERDOSE  (59.4%)
    pilot_tau0.15_h64  VOID-OVERDOSE  (37.5%)
    control_tau0.025   VOID (redo never fired)
