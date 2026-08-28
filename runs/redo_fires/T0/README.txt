T0 — v30 REGISTERED CAMPAIGN: NO-GO (2026-08-27)
=================================================

DECISION: do not spend the registered 12h/4-seed v30 budget. No training was
run under this task. This directory documents why the premise failed and
what a corrected registration would need, using the premise-falsifier pilot
already on disk at runs/v30_premise_falsifier_2026-08-27/ (raw logs,
analyze.py, README.txt) as the sole evidence base. Nothing here re-derives
that pilot; it adjudicates it against the v30 registration and closes the
loop the workflow brief asked for.

WHY THIS TASK EXISTS
---------------------
The registration for v30 set tau=0.25 as "the SMALLEST threshold on the
sweep that fires at all: the minimum dose," on the strength of a 2-iteration
sweep taken near orthogonal init (runs/v27_fresh_recovery/preflight/redo_forced/).
Two prior rounds (v27, v28) spent 7h10m x 4 seeds each on a treatment that
never armed (redo_tau 0.025, cum_recycled 0 on ~2000 checks). The lesson of
this whole workflow is that an unarmed run is VOID, not FAIL, and the fix
was to make arming a code-enforced gate rather than a registration promise.
The gate got built. This task is the step the workflow brief demanded
before spending the 12 hours it now guards: run the gate's own arms first,
cheaply, and read what they say before committing the budget.

WHAT THE PILOT MEASURED (see runs/v30_premise_falsifier_2026-08-27/ for
full logs; nothing below is re-run, only re-read against the registration)
------------------------------------------------------------------------
  (a) FIRES. tau=0.25, hidden 64: cum_recycled=353 over 20 iterations,
      19/20 firing. Iter-1 reproduces the banked 2-iteration sweep
      bit-exactly (fc2 5/32 recycled, agree 0.8142, max_dlogit 0.288576).
  (b) CHANGES THE NETWORK, substantially. Per-event agree median 0.856
      (min 0.702), max_dlogit to 0.802. Matched pair (tau 0.25 vs the
      banked tau 0.025) bit-identical through iters 0-1, permanently
      diverged after the first recycle. Entropy held 1.65-1.71 in
      treatment vs decaying to 1.346 in the untreated control.
  (c) THE ARMED CHECK WORKS, verified by running the inert case, not by
      reasoning about it. control_tau0.025_h64_VOIDED.log: the full
      60-env v27 recipe at v27's own tau=0.025 raised at iteration 26
      and the launcher aborted the whole sequence, no verdict/eval/
      clear_rate line anywhere in the log. Reproduced on an 8-env config
      (armed_check_inert_case_VOIDED.log, exit at iter 26).
  (d) BUT: at tau=0.25 the run settles at a median 20 of 32 trunk units
      (62.5%) re-initialized EVERY iteration from iter 5 onward. That is
      the exact regime the registration's own "RISK I REFUSE" clause
      describes at tau=0.50 and calls the DR's INCOMPATIBLE "network
      reset" family. The 2-iteration sweep could not see this: at
      orthogonal init the fc2 score minimum is 0.2848 (5 units below
      0.25); by iter 5 the minimum is 0.109 (20 units below 0.25).
  (e) "0.25 is the smallest tau that fires" is measurably false. The
      untreated control's own fc2 score minimum falls below 0.15 on
      22/26 iterations and below 0.10 on 10/26. tau=0.15 fires from
      iter 4 (176 units over 20 iters, 16/20 events, med 12/32 = 37.5%
      of trunk — still an overdose relative to the registration's own
      language, just a smaller one). The registered escalation ladder
      (0.25 -> 0.30, >=0.35 forbidden, no lower rung) points away from
      the region the data says is viable.
  (f) A4 (median greedy-argmax agree >= 0.60) CANNOT catch the overdose.
      Measured agree is 0.856 (h64) / 0.901 (h96) while 59-62% of the
      trunk is reset per iteration, both comfortably above the 0.60
      floor. recycle() zeroes the outgoing actor/critic columns by
      construction, so agreement is approximately output-preserving
      however much of the trunk was destroyed — agreement cannot see
      the dose. This is the eighth vacuous gate in this repo's history;
      the pilot wrote its revert-verified fix (a per-layer recycled-
      fraction ceiling, "V6") into scripts/redo_arm_gate.py, and applying
      it retroactively marks every treatment arm run in the pilot
      VOID-OVERDOSE (see gate verdicts below).
  (g) Width does not rescue it: tile_hidden_dim 96 gives 19/32 trunk
      units per iteration (59.4%), cum 342 — same picture. fc1 never
      goes dormant at any tau tested, 0/64 and 0/96 across 86 measured
      iterations (fc1 score minimum never below 0.309) — the entire
      effect lives in the 32-unit trunk, so "recycle dormant neurons in
      a 48k-parameter network" is in fact "periodically re-initialize
      most of a 32-unit bottleneck."
  (h) A fixed tau is not a stable operating point across the registered
      250-iteration budget. Untreated control fc2 score minimum: 0.285
      (it0) -> 0.127 (it5) -> 0.101 (it15) -> 0.079 (it24), still
      falling. The registration priced a 250-iteration campaign off a
      threshold measured over 2 iterations near init.

GATE VERDICTS (scripts/redo_arm_gate.py, run against
runs/v30_premise_falsifier_2026-08-27/*.log with the pilot's own V6 fix)
  pilot_tau0.25_h64        VOID-OVERDOSE  (62.5% of the worst-hit layer)
  pilot_tau0.25_h96        VOID-OVERDOSE  (59.4%)
  pilot_tau0.15_h64        VOID-OVERDOSE  (37.5%)
  control_tau0.025_h64     VOID (redo never fired)
No arm run under this pilot would have been eligible for a PASS/FAIL
verdict under the registration's own (corrected) gate. Launching the
registered 12h/4-seed campaign at tau=0.25 today would have produced four
more VOID runs, indistinguishable in kind from the v27/v28 failure this
whole workflow exists to stop repeating — just one gate-revision later.

STEPS / THROUGHPUT / WALL-CLOCK (measured, this pilot; no training run by
this task beyond re-reading these receipts)
  arm                    iters  wall-clock   env-steps/s (mean)  s/iter (mean)
  pilot_tau0.25_h64        20   17:53:19-18:05:34 (12m15s)   1712      36.7
  control_tau0.025_h64     25*  17:53:19-18:08:41 (15m22s)   1763      35.6
  pilot_tau0.15_h64        20   18:01:34-18:11:21 (9m47s)    2365      29.3
  pilot_tau0.25_h96        20   18:25:44-18:32:59 (7m15s)    2894      21.7
  armed_check_inert (8env) 25*  18:35:02-18:35:15 (13s)      2110       0.5
  * raised RuntimeError at iteration 26 (post-deadline check); no iter 25-29
    was logged as a completed training step, so 25 is the count of
    completed [redo]/[vanilla_ppo] iteration pairs, not a truncated 30.

  num_envs=60 (8 for the deadline-recheck arm); env-steps/iter ~61,440 for
  the 60-env arms (60 envs x 1024 rollout steps), ~1,018 for the 8-env arm.
  Approximate total env-steps moved by this pilot: 3 x 20 x 61,440
  (the three completed 60-env treatment/control-at-15 arms) + 25 x 61,440
  (control) + 25 x 1,018 (deadline recheck) ~= 5.25M env-steps.

  Wall-clock span, first arm start to last arm end: 17:53:19 -> 18:35:15
  = 41m56s (0.699h). Two arms (pilot_tau0.25_h64, control_tau0.025) started
  at the identical second and ran concurrently; summed serial-equivalent
  compute across all five arms is ~44m52s (0.748h). Device: CPU (Tile
  mode, torch intra-op threads capped to 1). No GPU/MPS involved. Both
  figures are consistent with the pilot's own self-report ("~35 min of
  compute, 4 arms"); the fifth arm here (armed_check_inert_case_VOIDED,
  13s) is the deadline re-verification and was not counted in that figure.

WHAT IT WOULD TAKE TO MAKE THE 12H CAMPAIGN WORTH RUNNING
-----------------------------------------------------------
The narrow hypothesis under test — does surgical dormant-unit recycling
under Sokar's definition buy anything on this stack — is still untested,
exactly as the workflow brief states. This pilot does not refute it; it
refutes the specific tau=0.25 fixed-threshold implementation of it. Three
changes, in order of cost, before any further compute is spent:

  1. RANK-BASED DOSE, NOT A FIXED TAU (no compute; ~1h of implementation
     against src/training/redo.py, which already computes the per-unit
     score needed). The control arm's own telemetry shows the tail
     drifting monotonically downward across training (0.285 -> 0.079 by
     iter 24, not yet converged), so any fixed threshold is either inert
     early or an overdose late. Replace "recycle score < tau" with
     "recycle the bottom k units of fc2, k small and fixed (the pilot's
     own data suggests k in 2-5, i.e. 6-16% of the 32-unit trunk, to
     land inside the dose the registration actually intended and never
     reached)." This is a stable operating point under drift by
     construction and needs no re-sweep to re-derive as training
     progresses.

  2. FIX A4 BEFORE using it as a safety valve (done, not yet merged into
     the campaign's config path). The pilot wrote and revert-verified a
     per-layer recycled-fraction ceiling in scripts/redo_arm_gate.py
     (V6); the registration's abort logic in src/training/trainer.py
     still only checks median agree, which this pilot shows cannot see
     an overdose by construction. Port V6's ceiling into the in-run A4
     abort (not just the post-hoc gate) so a mis-dosed live run halts
     at iter ~5-10 instead of iter 250.

  3. RE-SWEEP THE DOSE OVER THE REAL HORIZON, NOT 2 ITERATIONS (cheap:
     ~10 min, reuses the control arm already on disk — extend it to
     ~40-60 iterations at a few candidate k values). The registration's
     A1 Phase-M abort already budgets ~9 minutes for exactly this kind
     of check; it was simply run too short (20 iterations, still landed
     inside the regime that later proved unstable at iter 5+). A wider,
     longer sweep before arming removes the single root cause of this
     NO-GO: measuring a threshold near init and trusting it to hold for
     250 iterations of a monotonically drifting quantity.

  Total added cost before a re-registered v30 could responsibly launch:
  well under 2 hours, none of it the 12-hour campaign budget. That is the
  trade this document is banking: a ~1.5h correction now instead of a
  12-hour VOID-OVERDOSE campaign whose own safety gate cannot detect the
  failure mode that produced it.

WHAT THIS DOCUMENT DOES NOT CLAIM
-----------------------------------
  * No honest split-sample clear rate was measured (no honest protocol
    eval ran; nothing here is comparable to v27's 0.530 or v28's 0.670).
  * No checkpoint-selection ladder was walked; "peak_iter" below is the
    last iteration reached in the primary pilot arm (19), not a
    performance peak — no such peak exists to report.
  * This is not a verdict on the plasticity-loss hypothesis in either
    direction. It is a verdict on one implementation of the treatment
    (fixed tau=0.25) at the registered dose, and a scoped estimate of
    what a corrected implementation would cost to test properly.

RECEIPTS
  Raw logs, analyze.py, and the original pilot README:
    runs/v30_premise_falsifier_2026-08-27/
  Gate source (with the V6 over-dose ceiling this document relies on):
    scripts/redo_arm_gate.py
  Revert-verified gate tests:
    tests/test_redo_armed_gate.py
  This directory's machine-readable summary:
    runs/redo_fires/T0/verdict.json
