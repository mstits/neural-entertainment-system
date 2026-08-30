# V_adv on-policy re-execution — the sound run, and the retirement it earns

**Date:** 2026-08-28. **Registrations:**
`docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md` (parent, commit `35f6d60`,
inherited verbatim) + `docs/proposals/VADV_ONPOLICY_RERUN_ADDENDUM_2026-08-28.md`
(commit `2e7c663`, written before any collection compute). Predecessors:
offline VOID at R = 0.279 (`e09a220`), on-policy VOID on the collector
aliasing defect (`TWO_REGISTERED_TESTS_2026-08-27.md` Part I).

**Verdict: the arc reading is VOID under the strict registered A2 rule
(|A| = 0 of 26 — the positive control collapsed at every iterate) and
INDETERMINATE under the computed per-iterate signatures (26/26). Both
branches meet §11's operative condition, and neither is an operational
fault. THE §11 RETIREMENT RULE FIRES:**

> **V_adv is declared unable to adjudicate B5, and the question retires.**

**B5's standing verdict does not move — neither corroborated nor
re-opened.** No goalpost moved; no threshold was edited after the first
eta2 was read.

---

## 1. This time the instrument ran

Every precondition of the VOID's §1.9 was met before scoring:

* **The repair held on the artifact.** 26/26 banks collected in 0.864 h
  (3,224 episodes, ~1.6 M transitions, 0 purity violations), each bank
  chain-verified at write time by `assert_bank_wellformed` on the exact
  arrays handed to `np.savez`.
* **One instrument defect surfaced mid-job and was fixed as an operational
  fault, not absorbed.** The chain guard's first live run false-positived
  at iter 30: a `PC_SRC` rung-1013 episode (the one window spanning gx
  [0, 2873]) traversed the WALL band, the registered cross-population drop
  removed those interior rows, and the legitimate gap read as CHAIN BROKEN.
  Fifteen of 26 iterates carry drops, so most of the grid would have
  tripped. Fix (commit `5d0e016`): `row_step` is written into the bank so
  the gap is visible in the artifact itself, and the chain is asserted
  exactly across adjacent recorded steps — full strength against the
  aliasing class (which corrupts every adjacent pair), zero tolerance
  added. Revert-verified: 6 of 51 tests fail on the pre-fix guard,
  including the exact iter-30 false-positive reproduction. The full grid
  was recollected under the fixed guard.
* **Trajectory-identity diagnostic (addendum §3): 3,224 of 3,224 episodes
  bit-match the VOID run's outcomes** (source, steps, cleared/died/
  truncated, max_gx), and every penetration receipt is identical. The
  aliasing corrupted the recording, never the behaviour; this collection
  re-drove the same rollouts with the antecedents now recorded.
* **NC-b's acting range, re-derived critic-free BEFORE scoring**
  (`ncb_acting_range.json`): **MEASURABLE at 26/26 iterates, identical to
  PC_B5 at 0/26.** `NEG_gx_frozen` is a proper subset everywhere —
  1,686–2,034 qualifying rows in 21–33 cells against PC_B5's 2,966–3,209
  in 37–55. The addendum's INHERITED_CAP branch applies; the
  re-specification branch was never needed. The ninth vacuous gate's
  signature is absent from repaired data — and on sound data the control
  then *behaved*: **COLLAPSED at 25 of 26 iterates** (LIVE once, iter 240).
* **A1 reproduced a third time, bit-identical:** offline PC-1 eta2
  **0.6668875582236998**, LIVE, on `runs/interference/success_1_1.npz`.
* **Disclosure:** a single-iterate mechanics smoke of the committed
  pipeline ran on iter 10's bank before the full arc (sandboxed copy).
  No code or threshold changed afterward; the git record between
  `2e7c663` and the scoring run carries only the guard fix.

## 2. What the sound instrument read

Full table in `runs/vadv_onpolicy_rerun/arc_scored.json`; the shape is
uniform across the arc:

| region | verdict across 26 iterates | eta2 range | null q97.5 range |
|---|---|---|---|
| `WALL` (rung-893 rollouts, the arc's own training distribution) | **LIVE 26/26** | 0.043–0.109 | 0.022–0.034 |
| `PC_B5` (positive control; rungs this arc's critic never trained on) | **COLLAPSED 26/26** | 0.023–0.094 | 0.038–0.128 |
| `EARLY` (entrance, ~2 % of training mass; A6 coverage ok 26/26) | LIVE 7/26, COLLAPSED 19/26 | — | — |
| `NEG_gx_frozen` (NC-b) | COLLAPSED 25/26 | — | — |
| `INTERIOR` | **0 rows at 26/26** — §5.3 pre-declared branch | — | — |

* **A7 power = 1.0 at all 26 iterates** — every iterate admissible on
  power (the predecessor excluded two). Mechanistic note, disclosed: A7
  injects a calibrated effect into WALL's own rows, and WALL already
  carries a live effect, so the gate is saturated by the region's real
  signal — the registered **negative demonstration therefore FAILED TO
  FAIL** (8-cell down-sample, power 1.0 at calibrated effect 2.5e-9).
  On a region that is already LIVE the demonstration cannot fail; it
  certifies nothing about A7's failure mode on this data and is reported
  as exactly that. (Its failing case remains executed in the gate's unit
  tests and on the 2026-08-27 data, where it read 0.90.)
* **R is uninterpretable at every iterate**: with PC_B5's excess at or
  below zero, the registered ratio swings −89 to +437. The iter-250
  comparability row is published as registered — R = −8.35 against the
  predecessor's offline 0.279 — with the caveat the numbers force:
  comparability assumed a live denominator, and there is none.
* **R_early** (registered secondary, on-distribution reference; barred
  from the verdict): over the 7 EARLY-LIVE iterates, median **0.483**,
  range 0.280–0.636.

## 3. Adjudication — both roads end at the same place

**Strict registered path.** §7.1: *VOID(iterate) ⟺ any A1–A8 gate fails
at that iterate*; A2 requires PC-2 (`PC_B5`) LIVE. A2 fails at 26/26, so
|A| = 0 < 13 and the arc reading is **VOID — insufficient admissible arc
coverage**.

**Computed per-iterate path** (the driver evaluates signatures directly):
both signatures require `PC_B5` LIVE, so all 26 iterates read
**INDETERMINATE** — this time as a data outcome, not a pipeline identity:
the NC-b cap was armed and usable and was never what blocked a signature.

§11 fires on the INDETERMINATE branch directly, and on the VOID branch
because a collapsed positive control is not "a crash, an abort on
wall-clock, a purity-guard raise" — it is a property of the instrument on
its own data. One honesty note, stated rather than smoothed over: §11's
rationale sentence imagined "two computations with live controls" and this
run's positive control was not live. The operative condition is met
regardless, and the failure mode is *intrinsic to the design* — the
denominator band was inherited for comparability with 0.279, and no rerun
of this design can make a critic discriminate on rungs its run never
trained on. Fixing that would be a different denominator, a different
experiment, a new registration: exactly the "different instrument" §11
already routes to. Retiring now is the honest reading of a sound run.

## 4. The instrument finding — why V_adv cannot adjudicate B5

The LIVE/COLLAPSED verdict tracks the critic's training exposure,
monotonically: the rung-893 window (the only tau distribution this arc
trained on) reads LIVE 26/26; the entrance (~2 % of training mass) reads
LIVE 7/26; the PC rungs (zero mass — the cursor was parked at 893 before
iter 10) read LIVE 0/26. **V_adv on this stack measures where the critic
trained, not where the reward is flat.** Since CAPABILITY vs
MIS-SPECIFICATION is precisely a question about reward signal at states
the policy barely visits, the instrument is structurally confounded on
the exact contrast it was adopted to adjudicate.

Banked observation, no verdict weight: **the critic robustly
discriminates actions at the disputed gx-2674 wall state at every one of
26 iterates** (eta2 2–4× its own null q97.5, power 1.0). Read naively
that leans against "no action signal exists at the wall" — but WALL is
also the training distribution, which is the same confound, and the
machinery rightly refused to convert it into a signature without a live
positive control.

## 5. Disposition — §11, applied in its own words

1. **V_adv is declared unable to adjudicate B5, and the question
   retires.** Not "needs a further round"; not "a larger bank might."
   Three computations have now run: offline (VOID at R = 0.279, inside
   the indeterminate band), on-policy corrupt (VOID, instrument never
   ran), on-policy sound (this one — no signature declarable, positive
   control collapsed by training-exposure confound). That is an answer
   about the instrument, reported as one.
2. **B5's standing verdict does not move.** It remains what it has been
   since 2026-08-10: a never-retracted claim resting on `trailing 0/30,
   entrance 0/717` — evidence both live hypotheses predict. A retirement
   gives B5 nothing; this document may not be cited as corroboration.
3. **The rung-relative wavefront amendment stays DEFERRED, and its
   condition changes from "pending" to "closed by this route."** Anything
   that would revive it must be a different instrument with its own
   registration — named candidates, none authorised here: a
   counterfactual-restart assay at the entry state; an on-policy
   reward-decomposition read.
4. **The `[inert-treatment]` ledger entry is amended**: V_adv was built,
   controlled, run three times (once offline, once voided by a collector
   defect, once soundly to completion), and retired.
5. **The once-rule does not close B5.** The claim stays flagged
   under-instrumented in `CLAIMS.md`.

## 6. What survives, and is banked

* **The no-penetration measurement, now replicated on sound banks.**
  1,040 rung-893 episodes, ~1.04 M env-steps: gx never exceeded 2674,
  `min == max == 2674` at every iterate, `pen_rate = 0.0`. Across both
  collections that is 2,080 episodes without one exception. Per §5.3:
  INTERIOR is VOID, never COLLAPSED; a positive measurement; evidence
  for neither hypothesis. The failure localises to the gx-2674 state
  itself.
* **The rung-933 regression, replicated exactly from sound data**
  (`probe_933.json`): clears 8.3 % → 12.5 % → 16.7 % → 16.7 % → **0 %**
  across iters {10, 70, 130, 190, 250}; at iter 250 all 24 episodes
  deposit in WALL with max_gx 2674. A rung the curriculum demonstrably
  advanced through stopped clearing by the end of the arc. Diagnostic
  only, no verdict weight — and still no instrument pointed at it.
* **The NC-b exorcism receipt.** The control that was a byte-copy of the
  positive control on corrupt data is a proper subset on sound data and
  collapses 25/26 — the check that a threshold's acting range must be
  derived on the data it will see, executed and passed this time.
* **Collector determinism**: 3,224/3,224 episode outcomes identical
  across the pre-fix and post-fix collectors, seed-for-seed.

## 7. Costs

Collection 0.864 h + scoring ~1.0 h + gates/diagnostics ~0.1 h ≈ **2.0 h
compute**, inside the registered ≤ 3.0 h compute / 6.0 h lane ceilings.

## 8. Conformance to the NC-b pre-commitment (parallel registration)

A second session registered `docs/proposals/NCB_ACTING_RANGE_PRECOMMIT_2026-08-28.md`
(dfc202e → 44eb6de) in parallel with this execution; its own §4–§6 give
`2e7c663` precedence on the verdict path and scope its machinery to
receipts, bank soundness, and control-quality characterisation. Executed
here (`ncb_conformance.json`), with one ordering deviation disclosed: the
precommit wants these receipts before any tracking number, and they were
produced after the arc verdict was drafted. Nothing in them is
outcome-determinative — every gate passes and no cap branch changes:

* **Per-bank soundness (its §2): bitwise `alias_rate` = 0.0 at all 26
  banks** (the 2026-08-27 defect read 1.0), chain guard re-verified pass
  offline at all 26 → **26/26 survivors**, floor ≥ 20 met.
  VOID-THIN-BASIS and the §7 cause-split never engage (0 UNMEASURABLE).
* **Six-field NEG/PC identity anomalies (its §3(a)): 0 of 26.**
* **Control-quality characterisation (its §3(b), off the verdict path per
  its own §6):** NEG-vs-PC eta2 bootstrap CIs are disjoint at **0 %** of
  survivors (bar was ≥ 54 %), and the not-LIVE-where-PC-LIVE clause is
  vacuous (PC_B5 LIVE at 0 banks). Read for what it is: with the positive
  control itself collapsed everywhere, both regions sit near their nulls
  and NC-b never demonstrated it can *separate* from PC_B5 on on-policy
  data. That characterisation is one more reason no future rerun of this
  design could field the control — it informs the retirement, and caps
  nothing (there was no signature to cap).

## 9. Receipts

`runs/vadv_onpolicy_rerun/`: `collect_summary.json`, `collect_stdout.log`,
`iter_{00010..00260}.npz` + `_episodes.json` (row_step-carrying, chain-
verified), `ncb_acting_range.json`, `trajectory_identity_check.json`,
`A1/a1_reproduction.json`, `arc_scored.json(.jsonl)`, `arc_verdict.json`,
`a7_negative_demonstration.json`, `probe_933.json`, `extract_rows.json`,
`ncb_conformance.json`, `score_stdout.log`. Code: `scripts/collect_onpolicy_bank.py` +
`tests/test_collect_onpolicy_bank.py` (51, commit `5d0e016`);
`scripts/score_banked_iterates.py` unchanged (45 tests green).

---

## 10. Addendum (landing, same day) — the confound is also a power asymmetry

Added when both lanes were landed
(`docs/research/LANE_A_REPAIRED_AND_BOTTOMK_2026-08-28.md` §2). It does not
change the verdict — §11 fires on the strict A2 path and the computed-signature
path alike, and both are unaffected — but it changes what the retirement
forecloses, so it belongs beside §4 rather than only in the landing page.

§4 attributes the LIVE/COLLAPSED split to the critic's training exposure. That
reading is well supported and it is not the only mechanism present. Computed
over `arc_scored.json`: `WALL` carries a median **41,346 rows in 197 action
cells** against `PC_B5`'s **3,106 in 47** — a **13.3×** asymmetry the
registration does not control. A permutation null shrinks with n, so `WALL`'s
LIVE bar sits at a median q97.5 of **0.026** and `PC_B5`'s at **0.059**, on
effect sizes that broadly overlap (η² medians 0.0689 vs 0.0420; within 0.02 of
each other at 10 of 26 iterates). **`PC_B5`'s own η² would clear the LIVE bar
at 24 of 26 iterates if scored against `WALL`'s null.**

The asymmetry is structural and visible in the rollout protocol: 40 `WALL_SRC`
episodes per iterate, all truncated at a median 1,041 steps while pinned at
gx 2674, against 24 `PC_SRC` episodes that terminate in a median 190 steps
because they travel and clear.

**Consequence for §5's "different instrument".** A successor that only gave the
critic training exposure on the PC rungs would still score the positive control
against a null built from a seventh of the data. Any revival must **equalize
power between the contrasted regions** — equal rows, equal cells, or a null
pooled across both. That is a precondition neither named candidate currently
meets, and it is registered nowhere.

Scope: computed from the region row counts, cell counts, η² and null quantiles
as the scorer recorded them. The permutation nulls and η² were **not**
recomputed from the raw `.npz`.
