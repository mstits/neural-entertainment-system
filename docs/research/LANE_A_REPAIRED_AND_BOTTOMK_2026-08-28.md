# Landing: the repaired V_adv collector, and rank-based bottom-k ReDo

**Date:** 2026-08-28. Both lanes of the successor job licensed by
`docs/research/TWO_REGISTERED_TESTS_2026-08-27.md` (commit `7c73a73`) are
landed here. Neither produced a number that moves a standing verdict, and
both say so in their registration's own words.

| lane | registration | verdict | compute | what it moves |
|---|---|---|---|---|
| **A — `V_adv` on the repaired collector** | `VADV_ONPOLICY_PREREG_2026-08-27.md` (`9a9db2d`) + `VADV_ONPOLICY_RERUN_ADDENDUM_2026-08-28.md` (`77b8549`, pre-compute) | **VOID under strict A2 (\|A\| = 0/26) · INDETERMINATE under computed signatures (26/26) → §11 RETIREMENT FIRES** | **2.0 h** (≤ 3.0 h registered) | `V_adv` retires. **B5 does not move.** |
| **B — rank-based bottom-k ReDo (v32)** | `V32_REDO_BOTTOM_K_2026-08-28.md` (`e9cc5ed`, pre-compute) | **VOID-UNDERPOWERED** — no Θ. Ladder resolved: rung 1 `VOID-NO-TURNOVER`, rung 2 **GO**. 0 seeds, 0 ARMED, 0 scored. | **0.96 h** (13.0 h registered ceiling; 12.0 h unspent) | Nothing about plasticity. **The hypothesis remains untested.** |

Full write-ups, each separately citable and neither superseded by this
page: `docs/research/VADV_ONPOLICY_RERUN_2026-08-28.md` (Lane A) and
`docs/research/V32_PHASE_R_ADJUDICATION_2026-08-28.md` (Lane B). This
document lands them together, states each verdict beside the registration
that fixed its bar before compute, prices both VOIDs, and adds one finding
neither write-up carries: **§2, the power asymmetry that made Lane A's
positive control unable to pass.**

---

## 1. Lane A — the instrument ran soundly, and retired

The predecessor was VOID on a collector defect: a reused
`TileFeatureStacker._out` buffer made `state` bit-identical to `next_state`
on 100 % of rows in all 26 banks, so `Â = r + γV(s') − V(s)` collapsed to
`(γ−1)·V(s')` and carried zero action information by construction. Commit
`b2e806b` copies out of the buffer at both call sites and adds
`assert_bank_wellformed` on the arrays handed to `np.savez`. The
registration was re-executed, not re-designed.

**Verified physically before any gate was consulted**, because the last run
passed every gate while fatally aliased:

* **Exhaustive, all 26 banks, all rows** (not a spot check): `state !=
  next_state` on **1,628,514 of 1,628,514 rows** — zero identical rows
  anywhere. The defect that voided the predecessor, which held at 100 % of
  rows in every bank, is eliminated.
* **Chain integrity on every step-adjacent pair in the grid:
  1,625,285 of 1,625,285** — `next_state[i] == state[i+1]` bit-exact
  wherever `episode_id` matches and `row_step` increments by one. Exactly
  **5** intra-episode gaps exist across the whole grid — the registered
  cross-population drop, visible in the artifact itself, which is precisely
  what adding `row_step` was for.
* Episode-level motion is real: `PC_SRC` rollouts clear the level on **624
  of 624** episodes across the grid (`max_gx` 3266), `ENTR_SRC` spans
  `max_gx` 174 → 1460 over its 1,560, and `WALL_SRC` is pinned at 2674 over
  its 1,040 — the wall, not a frozen scene.
* The guard is live in **both** directions, exercised here rather than
  taken on report: `assert_bank_wellformed` raises `DEGENERATE` on a fully
  aliased bank, raises `CHAIN BROKEN` on a bank with a **single** aliased
  row in 200 (0.5 %), and does **not** raise on a bank carrying the
  registered cross-population drop as a `row_step` gap. Fires on the
  defect, silent on legitimate output — not a tenth vacuous gate, and not
  the false alarm its first version was.

**The hard precondition was met, and needed no re-specification.** NC-b's
acting range was re-derived critic-free *before* scoring: `NEG_gx_frozen`
is MEASURABLE at 26/26 and identical-to-`PC_B5` at **0/26** — a proper
subset everywhere (1,686–2,034 rows in 21–33 cells against `PC_B5`'s
2,966–3,209 in 37–55). The ninth vacuous gate's signature is absent from
repaired data. On sound data the control then behaved like a null:
COLLAPSED 25/26. `A1` reproduced a third time to exact float equality,
**0.6668875582236998**, LIVE.

**The reading.** `WALL` (the arc's own training distribution) LIVE 26/26 at
η² 0.043–0.109; `PC_B5`, the registered **positive control**, COLLAPSED
26/26. `EARLY` LIVE 7/26. `INTERIOR` 0 rows at 26/26 — the pre-declared
§5.3 branch: INTERIOR VOID, never COLLAPSED, a positive measurement and
evidence for neither hypothesis. Strict A2 therefore fails at every
iterate, |A| = 0 < 13, and the arc reading is VOID for insufficient
admissible coverage; the computed per-iterate signatures read INDETERMINATE
26/26 because both require `PC_B5` LIVE. Both branches meet §11's operative
condition, and a collapsed positive control is not an operational fault —
not a crash, not a wall-clock abort, not a purity-guard raise.

> **§11 fires: `V_adv` is declared unable to adjudicate B5, and the
> question retires.**

`R` — mean **12.38**, median 3.21, range −89.12 to +436.74 — is
uninterpretable at every iterate: the denominator (`PC_B5`'s excess over
its own null) is at or below zero everywhere. The registered iter-250
comparability row is published anyway with that caveat: **R = −8.35**
against the predecessor's offline **0.279**. `R_early` (registered
secondary, barred from the verdict): median **0.483** over the 7 EARLY-LIVE
iterates.

**Cost of the VOID: 2.0 h of compute** — collection 0.864 h (26/26 banks,
3,224 episodes, 0 purity violations, `aborted_on_wallclock: false`),
scoring ~1.0 h, gates and diagnostics ~0.1 h. Inside the registered ≤ 3.0 h
compute and 6.0 h lane ceilings. One operational fault was fixed rather
than absorbed mid-job: the new chain guard false-positived at iter 30 on a
legitimate mid-episode gap left by the registered cross-population drop;
`row_step` was written into the artifact so the chain is asserted only
across step-adjacent pairs (commit `6d700c5`, revert-verified 6/51), and
the full grid was recollected. No threshold moved.

**Disclosed and not smoothed over:** the registered A7 negative
demonstration **failed to fail** — it injects a calibrated effect into
`WALL`'s own rows, and `WALL` already carries a live signal, so the gate is
saturated by the region's real effect (power 1.0 at effect 2.5e-9). It
certifies nothing about A7's failure mode on this data. Since no signature
was declared, it cannot have manufactured one.

---

## 2. The additive finding — the positive control was never powered

*New here; not in the Lane A write-up. Computed directly from
`runs/vadv_onpolicy_rerun/arc_scored.json`.*

The Lane A document attributes the LIVE/COLLAPSED split to **critic
training exposure**, and that reading is well supported: the rung-893
window reads LIVE 26/26, the entrance (~2 % of training mass) LIVE 7/26,
the PC rungs (zero mass) LIVE 0/26. But it is not the only mechanism
present, and the better-evidenced one is simpler and worse.

**The two contrasted regions were never comparable on the registered
criterion, because they carry a ~13.3× sample-size asymmetry that nothing
in the registration controls.**

| | `WALL` | `PC_B5` |
|---|---|---|
| rows (median over 26 iterates) | **41,346** | **3,106** |
| action cells (median) | 197 | 47 |
| η² (median) | 0.0689 | 0.0420 |
| null q97.5 (median) | **0.0261** | **0.0588** |
| null q97.5 (range) | 0.0220–0.0337 | 0.0379–0.1276 |
| verdict | LIVE 26/26 | COLLAPSED 26/26 |

A permutation null shrinks with sample size. `WALL`'s bar sits at ~0.026;
`PC_B5`'s at ~0.059 — **more than twice as high, on effects that broadly
overlap.** The two regions' η² fall within 0.02 of each other at **10 of
26** iterates (iter 40: `WALL` 0.0439 vs `PC_B5` 0.0351). Decisively:

> **`PC_B5`'s own η² would clear the LIVE bar at 24 of 26 iterates if it
> were scored against `WALL`'s null.**

The asymmetry is structural, not incidental, and it is visible in the
rollout protocol itself. Per iterate the collector runs 40 `WALL_SRC`
episodes that are step-capped — median **1,041** steps, all 40 truncated,
`max_gx` pinned at 2674 — against 24 `PC_SRC` episodes that terminate in a
median **190** steps because they actually travel and clear. Roughly 5.5×
the rows per episode times 1.67× the episodes is the 13.3×.

**Consequence, and it strengthens the retirement rather than weakening it.**
A successor that merely gave the critic training exposure on the PC rungs
would **not** rescue this instrument: the positive control would still be
scored against a null built from a seventh of the data. Any redesign must
**equalize power between the contrasted regions** — equal rows, or equal
cells, or a null pooled across both — before it can claim a LIVE control.
That is a design requirement for the "different instrument" §11 routes to,
and it is registered nowhere yet.

Stated with its own scope: this is a computation over the published scored
arc — region row counts, cell counts, η², and null quantiles as the scorer
recorded them. The permutation nulls and η² were **not** recomputed from
the raw `.npz`; this checks the pipeline's inputs and internal consistency,
not its arithmetic.

---

## 3. What Lane A banks, and the one thing now claimed

**No-penetration, replicated on sound banks.** All **1,040** `WALL_SRC`
episodes across 26 iterates have `max_gx` exactly 2674 — `min == max ==
2674`, `pen_rate = 0.0` at every iterate, **1,084,678 env-steps**. `WALL_BAND` is
`(2674, 2872)`, so 2674 is the band's lower edge and the interior is empty
by physics. Across both collections that is **2,080 episodes without one
exception**. Per §5.3 this is INTERIOR VOID, a positive measurement, and
evidence for **neither** hypothesis. The failure localises at the gx-2674
state itself — the documented off-manifold-drift barrier, reproduced.

**The rung-933 clear-rate regression — CLAIMED (see `CLAIMS.md`).** The
diagnostic probe restarts 24 episodes per sampled iterate from rung 933, a
rung the curriculum demonstrably advanced through, and reads:

| iter | 10 | 70 | 130 | 190 | 250 |
|---|---|---|---|---|---|
| clear rate | 8.3 % | 12.5 % | 16.7 % | 16.7 % | **0 %** |
| deposits in `WALL` (of 24) | 22 | 21 | 20 | 20 | **24** |
| `max_gx` | 3266 | 3266 | 3266 | 3266 | **2674** |

It rises, plateaus, then goes to zero, and at iter 250 every one of the 24
episodes deposits at the wall. Measured on the corrupt collection and again
bit-identically on the repaired one — **twice, and unclaimed both times.**
It is claimed now for exactly what it is: a **measurement** under the
collector's restart protocol, not an honest-protocol result, carrying no
ledger status and no capability inference. Still no instrument is pointed
at it, and none is authorised here.

---

## 4. Lane B — the ladder is resolved, the campaign is not run

v31 banked a stopping statement with a live two-rung receipt: on a
Linear → LayerNorm → SiLU 32-unit trunk there is no fixed dormancy
threshold that is simultaneously firing, surgical and sustained.
Fixed-tau ReDo is forbidden. The licensed successor recycles the **bottom-k
by dormancy rank**, which caps the dose by construction — the drifting tail
changes *which* units are recycled, never *how many*. That successor was
registered in full before its own compute (`e9cc5ed`): k = 2, cadence 5,
`fc2` only, ARMED re-specified as B1–B4 because a rank rule fires by
construction and inheriting v31's F1/F2 would have installed a gate that
cannot fail; F3 distinctness retired and replaced by the threshold-free
turnover gate F3′; Θ bar 0.80 / 0.767 and the cross-fit split-sample
estimator inherited untouched.

**Both rungs of the registered ladder ran.**

| | rung 1 (k = 2, C = 5) | rung 2 (k = 4, C = 10) |
|---|---|---|
| R1 REACHED | PASS — 12 events / 12 checks, `cum_recycled` 24 = 2 × 12 | PASS — 12 / 12, `cum_recycled` 48 = 4 × 12 |
| R2 ARTIFACT-MATCH | PASS — **12/12 = 100 %** | PASS — **12/12 = 100 %** |
| R3 DOSE | PASS — exactly 0.0625 on all 12; `fc1` 0/64 | PASS — exactly 0.125 on all 12; `fc1` 0/64 |
| R4 TURNOVER | **FAIL — repeat_rate 11/11 = 1.00** | PASS — repeat_rate **0.909** |
| verdict | `VOID-NO-TURNOVER` → NO-GO-R4 | **GO** |

Rung 1's recycled sequence was
`[21,26] [5,26]×3 [5,9]×8` — every consecutive pair shared an index and the
set never turned over once. **The dose cap worked and the lesion happened
anyway.** The recovery curve is the finding, and no prior ReDo run in this
repository could make it, because every earlier trace ran at cadence 1 and
never left a recycled unit alone for a single iteration:

> **A re-initialized trunk unit does not climb out of the rank-bottom of
> the dormancy distribution within four free PPO updates. It sinks
> further.** 20 of 22 recycled units (90.9 %) were re-selected at the next
> cadenced check; median rank one check later is 1 of 32; the terminal pair
> falls from 0.086/0.094 at iter 20 to 0.050/0.037 at iter 45 while the
> layer median *rises* 0.140 → 0.155.

Rung 2 — the registered escalation, taken once, holding cumulative dose
exactly constant so the only thing that changed is the recovery window
(4 → 9 free PPO updates) — passes all four. The confound runs **against**
the result: larger k makes re-selection more likely under any null
(6.25 % → 12.5 %), and re-selection nonetheless fell 90.9 % → 75.0 %, with
distinct indices 4 → 14 and top-index share 45.8 % → 20.8 %.

**Reported with no verdict attached**, because the registration made R4
threshold-free on purpose and reinterpreting it after seeing the trace is
the exact error behind nine vacuous gates: rung 2's single clean break —
the one pair sharing no index, which is the entirety of what carries R4 —
falls at event 2 → 3, and from event 6 the set locks to `[8,23,26,30]` for
seven consecutive events. **A longer recovery window delays the lock-in; on
this evidence it does not prevent it.** A campaign at (k = 4, C = 10)
should expect a fixed four-unit lesion over roughly its back half.

**Cost of the VOID: 0.96 h** — two 12-check Phase R runs (60 iterations at
C = 5, 03:54:33 → 04:13:33; 120 at C = 10, 04:13:57 → 04:52:20) plus
adjudication, against the ~5.3 h of seed training and ~1.5 h of honest
ladder they gate. The 13.0 h ceiling stands
with 12.0 h unspent, the 0.80 / 0.767 bars are untouched, and the 0.05
winner's-curse budget is unspent. Throughput ran 18.9–19.1 s/iter against
the registered 25.4 ± 2.5 band — **outside it, flagged as §5 requires**, not
disqualifying, and bearing on no R1–R4 gate since none of them is a timing
statistic.

**Verified rather than taken on report:** both shipped gates re-run on
**copies** of the logs (the trainer holds a truncating `FileHandler` on
`run.log` — the Lane B post-mortem's own defect); a second implementation
sharing no code with the shipped gate, validated against the banked smoke
receipt and then agreeing field for field; and anti-vacuity executed by
hand — neutering B2 fails `test_artifact_mismatch_voids`, neutering B4
fails `test_zero_turnover_voids_at_repeat_rate_exactly_one`, neutering the
preflight mode-gating fails
`test_bottom_k_profile_rejects_a_threshold_mode_log_at_the_same_tau`; all
restored.

---

## 5. What Lane B does not license

**Seeds launched: 0. ARMED: 0. Scored: 0. Θ is not issued** — §5's
disposition for fewer than four ARMED-and-scored seeds is
**VOID-UNDERPOWERED**, with per-seed numbers banked individually, of which
there are none. Δ is NOT COMPUTED.

**The plasticity hypothesis REMAINS UNTESTED**, exactly where v31 left it.
The v27/v28 confound is **not** discharged; the §10.3 FAIL licence is not
triggered and cannot be, since it explicitly requires `repeat_rate < 1.00`,
which is precisely what rung 1 failed; and the 2026-08-25 DR ReDo
prescription may **not** be closed as EXECUTED-AND-NEGATIVE — it still has
not been executed. The standing prohibition is extended to v32: nothing in
v27, v28, v30, v31 **or v32** may be cited for or against plasticity loss.

Rung 1's VOID licenses exactly three things: the recovery measurement
quoted above, which belongs to no other run; that the bottom-k mechanism is
correctly implemented and reaches the trainer's hot path; and taking the §8
escalation. Rung 2's GO licenses exactly one: launching the 4-seed campaign
at (k = 4, C = 10).

**One numeral must be re-derived in writing before that launch, not
inherited.** B1's floor of "≥ 48 events of 50 cadenced checks" was written
for C = 5. At C = 10 a 250-iteration seed has **25** checks, so the floor
must be restated as **≥ 24 of 25** — 25 events × k = 4 = 100 units = 3.125
trunk turnovers, identical cumulative dose. That is arithmetic implied by
the registered rung, but it is a numeral, and numerals go in writing before
compute.

---

## 6. The joint ledger

* **Two VOIDs, priced: 3.0 h of compute total** (Lane A 2.0 h, Lane B
  0.96 h) against registered ceilings of 6.0 h and 13.0 h. Neither VOID
  enters any aggregate. No bar moved in either lane; no threshold was
  edited after a result was seen.
* **One retirement, on a sound run.** `V_adv` is gone from B5 by the
  registration's own §11, with all five consequences applied: B5 unchanged
  and still flagged under-instrumented; the rung-relative wavefront
  amendment stays DEFERRED with its condition moved "pending" → "closed by
  this route"; the `[inert-treatment]` ledger amended; the once-rule does
  not close B5. Revival requires a different instrument with its own
  registration — and, per §2 above, one that equalizes power between the
  regions it contrasts.
* **One claim added:** the rung-933 clear-rate regression, as a
  measurement, with no ledger status.
* **One ladder resolved and one operating point armed** — the v32 campaign
  is now mechanical to launch and has never been run.

## 7. What should happen next

The honest ordering, given that Lane A closed a road and Lane B opened one
it did not walk:

1. **Run the v32 campaign at (k = 4, C = 10)** — 4 seeds × 250 iterations
   sequentially, honest ladder, cross-fit Θ, with B1's floor restated to
   ≥ 24 of 25 in writing first. It is the only registered work in the
   program with a bar already fixed, an operating point already armed, and
   12.0 h of its own budget unspent. Until it runs, the plasticity
   hypothesis stays exactly where the DR left it in three campaigns'
   worth of prose.
2. **The rung-933 regression is the standing uninstrumented lead** — clears
   going 16.7 % → 0 % on a rung the curriculum advanced through, measured
   twice, with nothing pointed at it.

Not authorised here, and named only so nobody re-derives them as new: a
counterfactual-restart assay at the entry state, and an on-policy
reward-decomposition read. Either would need its own registration, and §2
sets a precondition neither currently meets.

## 8. Receipts

| path | what |
|---|---|
| `runs/vadv_onpolicy_rerun/collect_summary.json` | 26/26 banks, 0.864 h, per-iterate penetration + rung-933 probe |
| `runs/vadv_onpolicy_rerun/iter_{00010..00260}.npz` + `_episodes.json` | the banks themselves, `row_step`-carrying and chain-verified |
| `runs/vadv_onpolicy_rerun/ncb_acting_range.json` | the hard precondition, derived critic-free before scoring |
| `runs/vadv_onpolicy_rerun/arc_scored.json(.jsonl)`, `arc_verdict.json` | the 26-iterate reading and the R curve |
| `runs/vadv_onpolicy_rerun/a7_negative_demonstration.json` | the failed-to-fail disclosure |
| `runs/vadv_onpolicy_rerun/probe_933.json` | the regression claimed in §3 |
| `runs/vadv_onpolicy_rerun/{trajectory_identity_check,ncb_conformance}.json`, `A1/` | determinism, conformance, A1 |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/` | rung 1 log, adjudication, arm gate, independent verify, recovery curve |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r2/` | rung 2, the same set |
| `runs/v32_redo_bottom_k_2026-08-28/smoke/` | the pre-existing wiring receipt (`e9cc5ed`) |

Registrations: `docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md`,
`docs/proposals/VADV_ONPOLICY_RERUN_ADDENDUM_2026-08-28.md`,
`docs/proposals/NCB_ACTING_RANGE_PRECOMMIT_2026-08-28.md`,
`docs/proposals/V32_REDO_BOTTOM_K_2026-08-28.md`.
Code: `scripts/collect_onpolicy_bank.py` (+51 tests), `src/training/redo.py`
`redo_mode: bottom_k` (flag-gated, default off), `scripts/redo_arm_gate.py
--bottom-k`, `scripts/adjudicate_phase_r.py`.
