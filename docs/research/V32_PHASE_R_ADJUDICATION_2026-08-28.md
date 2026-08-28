# v32 Lane B — the Phase R ladder, both rungs. Rung 1 NO-GO, rung 2 GO, campaign not run.

Adjudicated against `docs/proposals/V32_REDO_BOTTOM_K_2026-08-28.md` (commit
`e9cc5ed`), whose every numeral was fixed before compute. Nothing here moves a
bar, and no bar was reopened.

---

## 0. Headline

**Both rungs of the registered ladder were run. Rung 1 NO-GO, rung 2 GO, and
the campaign that rung 2 licenses was not executed.** In one line each:

| | rung 1 (k=2, C=5) | rung 2 (k=4, C=10) |
|---|---|---|
| R1 / R2 / R3 | PASS / PASS / PASS | PASS / PASS / PASS |
| R4 turnover | **FAIL** — repeat_rate 1.00 | **PASS** — repeat_rate 0.909 |
| verdict | `VOID-NO-TURNOVER`, NO-GO-R4 | **GO** |

**Seeds launched: 0. ARMED: 0. Scored: 0. Θ is not issued. The plasticity
hypothesis REMAINS UNTESTED** — see §8. Everything below is preflight
measurement; none of it is a result about ReDo as a lever.

### 0.1 Rung 1

Phase R ran the full 60 iterations at exactly the registered operating point
(k = 2, C = 5, scope fc2). It passed R1, R2 and R3 and **failed R4**:

```
repeat_rate = 11/11 = 1.00
fc2 = [21,26] [5,26] [5,26] [5,26] [5,9] [5,9] [5,9] [5,9] [5,9] [5,9] [5,9] [5,9]
```

**Every one of the eleven consecutive event pairs shared at least one index.**
The recycled set never turned over even once. Per §11 this is
**VOID-NO-TURNOVER**; per §8 it is the ladder's one licensed trigger.

No training began at this rung. §4 A1 is unambiguous — *"No training begins
before Phase R returns GO."* The 0.80 / 0.767 bars stand untouched and the
0.05 winner's-curse budget is unspent.

And Phase R delivered the measurement it was written to make — the one no
prior ReDo run in this repository could make, because every prior run ran at
cadence 1 and never left a recycled unit alone for a single iteration:

> **A re-initialized trunk unit does not climb out of the rank-bottom within
> four free PPO updates. It sinks further.**

---

## 1. R1-R4, on the artifact

Adjudicated off a **copy** of the stdout log (A9), taken while the trainer
still held its own truncating `FileHandler` on `<checkpoint_dir>/run.log`.

| gate | requirement | measured | verdict |
|---|---|---|---|
| **R1 REACHED** | ≥ 10 recycle events of 12 cadenced checks; exactly one `mode=bottom_k` ENABLED line; no `[redo] disabled` | 12 events / 12 checks, `cum_recycled` 24 = 2 × 12, one ENABLED line | **PASS** |
| **R2 ARTIFACT-MATCH** | offline bottom-k recomputation reproduces logged indices on 100% of events | **12/12 = 100%**; min separation margin 0.00428 (reported, never gated) | **PASS** |
| **R3 DOSE** | `dose_fraction == 0.0625` every check; fc1 total 0; ceiling never trips | 0.0625 on all 12; fc1 0/64 on all 12; no overdose | **PASS** |
| **R4 TURNOVER** | `repeat_rate < 1.00` | **1.00 (11 of 11 pairs)** | **FAIL** |

`[redo] ENABLED tau=0.025 every_iters=5 scope=fc1,fc2 sample=4096
reset_moments=true mode=bottom_k k=2 recycle_scope=fc2` — exactly one line, in
the production log, from the real training loop.

Per §8 the split matters: R1/R2/R3 failing would be an implementation defect
and would mean STOP-and-re-register. **They passed.** The mechanism is wired,
reaches the hot path, fires by construction at the registered cadence, and the
units it recycles are provably the rank-bottom of the logged distribution. The
only thing that failed is the thing the registration named as the real
degenerate case under a rank rule.

---

## 2. The recovery curve — the finding

For every recycled unit, its score and rank at the next cadenced check
(rank 0 = lowest of 32; k = 2 means ranks 0-1 are the ones recycled):

- **20 of 22 recycled units with a next-check observation (90.9%) were
  re-selected five iterations later.** Median rank one check later: **1**.
  Only two ever escaped the bottom-2 (unit 21 → rank 3, unit 26 → rank 8),
  and the worst escape reached only rank 8 of 32.

The full index histogram over the 24 unit-events, reported with no verdict
attached (F3 distinctness is retired under a rank rule, §6.1): unit 5 × 11,
unit 9 × 8, unit 26 × 4, unit 21 × 1 — four distinct indices, top-index share
45.8%.

The score trajectories of the two units that formed the terminal lesion, with
the layer median for scale:

| iter | u5 | u9 | median |
|---|---|---|---|
| 20 | 0.0859 | 0.0936 | 0.1399 |
| 25 | 0.0470 | 0.0416 | 0.0991 |
| 30 | 0.0456 | 0.0479 | 0.1236 |
| 35 | 0.0427 | 0.0365 | 0.1439 |
| 40 | 0.0653 | 0.0595 | 0.1477 |
| 45 | 0.0503 | 0.0366 | 0.1548 |

From iter 20 the recycled pair sinks while the layer median **rises**. By iter
45 the gap is roughly fourfold.

Hand-checked at the final event (iter 55), independently of both gates: the
logged pair `[5, 9]` scored **0.0380 / 0.0406**, against a third-lowest unit
30 at **0.0708** — a gap of 0.0302, nearly a factor of two. The selection is
not a marginal tie being broken by sort order; the two units are separated
from the rest of the layer by a wide margin, and that margin is the lesion.

Re-initialization does not restore a unit to the middle of the distribution;
it pins it to the floor, and four free PPO updates do not lift it off.

This is mechanically the same object v31 measured at τ = 0.075 —
`fc2=[16]×7, [5,16]×24`, a permanent two-unit lesion — reproduced under a rule
that was chosen specifically because it caps the dose. **Capping the dose did
not prevent the lesion.**

Unit 5 appears in 11 of 12 events here and was also in v31's terminal set. That
recurrence is *noted, not claimed*: both runs are seed 0 on the same
architecture, so a shared index is at least as likely to be an artifact of a
shared initialization as a property of the unit. It is not offered as evidence
of anything.

The registration anticipated exactly this and wrote it down before compute
(§1.3, §6.2): *"The real degenerate case under a rank rule is not low
distinctness but ZERO TURNOVER."* F3'/R4 is that gate, and this is the first
time it has bitten on live data at a cadence where a repeat was **not**
structurally forced. Both banked v31 traces also read repeat_rate 1.00, but
both ran at cadence 1 where a repeat is guaranteed; the registration flagged
that C = 5 left the question "open in both directions." It is now closed in
one.

---

## 3. What was verified rather than taken on report

1. **Both shipped gates re-run on copied logs.** `redo_arm_gate.py --bottom-k`
   returns `VOID-NO-TURNOVER`, rc = 2. `adjudicate_phase_r.py` returns
   `NO-GO-R4`.
2. **An independent second implementation.** A verifier sharing no code with
   the shipped gate — own regexes, own bottom-k recomputation, own R1-R4
   arithmetic — was validated against the banked smoke receipt, which it
   reproduced exactly, then agreed with the shipped gate on the Phase R log
   field for field. A gate that only agrees with itself is not checked.
3. **Anti-vacuity executed, not asserted.** Each new gate was neutered by hand
   and the failure observed, then restored:
   - B2 artifact-match neutered → `test_artifact_mismatch_voids` fails (1/14).
   - B4 turnover neutered → `test_zero_turnover_voids_at_repeat_rate_exactly_one` fails (1/14).
   - Preflight mode-gating neutered → `test_bottom_k_profile_rejects_a_threshold_mode_log_at_the_same_tau` fails (1/12).

   All restored; 14/14 and 12/12 green; 174 passed across the
   redo/preflight/manifest subset.
4. **Config discipline.** All four seed configs are byte-identical to the
   Phase R config but for `name`. Against the v27 control, comments stripped,
   the only functional difference is the ReDo selection rule and its cadence —
   the registered single functional variable.
5. **Sequencing.** Lane A's collector finished at 02:13 and no collector was
   alive; Phase R launched 03:54. §5's precondition holds.

**One incidental correction of record.** The registration argues (§10.3) that
this architecture "measurably does not provide" an identity-preserving
intervention, citing median argmax-agreement of *"0.85-0.97, never 0.98"* at
any firing dose. At the rank-bottom k = 2 dose it reads **0.9860**. That is a
narrow factual update to the cited range, nothing more — it does not restore
any licence, because agreement was demoted (A5) precisely for being
structurally insensitive to dose, and v31 measured 0.97 while resetting 37.5%
of the trunk. A higher number from a smaller dose is what an insensitive
statistic looks like.

**One flag, recorded not excused:** throughput ran **18.9-19.1 s/iter**
against the registered band of **25.4 ± 2.5**. Outside the band. §5 says this
is *"flagged in the verdict"*, not disqualifying, and it is flagged here. The
likely cause is that Lane A's CPU contention had fully cleared. It bears on an
arm-vs-control timing comparison, not on R1-R4, none of which is a timing
statistic.

---

## 4. Disposition

Per §8, an R4-only failure has exactly one licensed successor: **escalate once
to (k = 4, C = 10) and re-run Phase R in full**, holding cumulative dose
exactly constant so the rung changes only the recovery window (4 → 9 free PPO
updates). That escalation was pre-committed in writing before rung 1's final
events landed, including its one genuine ambiguity:

> §7 defines Phase R as 60 iterations = 12 cadenced checks with R1 ≥ 10 of 12.
> At C = 10, 60 iterations yields only 6 checks and cannot satisfy that floor.
> **Reading adopted: "in full" = the same twelve cadenced checks, i.e. 120
> iterations at C = 10, R1's floor left at ≥ 10 of 12.** The rejected
> alternative — 60 iterations with R1 lowered to ≥ 5 — would weaken a
> registered floor after seeing rung-1 data, which is a moved goalpost.

Rung-2 config `configs/mario_1_1_v32_redo_bk_phase_r2.yaml` is a mechanical
clone of the Phase R config differing only in `name`, `redo_bottom_k: 4`, and
`redo_check_every_iters: 10`. Its preflight armed-pattern was confirmed to
mode-gate correctly on k = 4 / every_iters = 10.

---

## 5. What rung 1's VOID licenses — exactly and no wider

*(Scope note: this section adjudicates rung 1 alone, which is the VOID. Rung 2
and the campaign's actual standing are §7 and §8. Nothing in either widens the
list below.)*

**LICENSED, in writing:**

1. **The recovery measurement**, which is new and belongs to no other run:
   *"On a Linear → LayerNorm → SiLU 32-unit trunk, a re-initialized trunk unit
   does not climb out of the rank-bottom of the dormancy distribution within
   four free PPO updates; 20 of 22 recycled units (90.9%) were re-selected at
   the next cadenced check and their scores continued to fall while the layer
   median rose."*
2. **That the bottom-k mechanism is correctly implemented and live** — R1-R3
   passed on the production log, with the recycled indices certified against
   the logged score vectors by two independent implementations. The failure is
   about the architecture's response, not about wiring.
3. **The §8 escalation**, and nothing beyond taking it.

**NOT LICENSED — any of these in writing would be a fabrication:**

- Any statement about Θ, the 0.767 bar, or the fork. A VOID takes no branch of
  the fork and enters no aggregate. There is no Θ.
- **Any FAIL-class inference.** In particular the §10.3 FAIL licence — which
  includes the un-confounding of v27/v28 and closing the DR ReDo prescription
  as EXECUTED-AND-NEGATIVE — is **not** triggered. That licence explicitly
  requires `repeat_rate < 1.00`, which is precisely what failed here.
- Any claim that ReDo is or is not a lever of any size.
- **Any claim that plasticity loss was or was not the barrier.** Zero seeds,
  zero scored: the plasticity hypothesis **REMAINS UNTESTED**, exactly as it
  was after v31. The v27/v28 confound is **NOT** discharged.
- Hypothesis B (48k is a hard ceiling) confirmed or falsified.
- Any claim about other plasticity interventions (L2-init, CReLU, weight
  churn, layer-norm resets, periodic full resets), other widths, other levels,
  or other games.

---

## 6. Receipts

| path | what |
|---|---|
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/phase_r_stdout.log` | the 60-iteration production log at (k=2, C=5) |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/adjudication.json` | R1-R4 + the recovery curve, from the shipped adjudicator |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/arm_gate.txt` | B1-B4, `VOID-NO-TURNOVER`, rc 2 |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/independent_verify.json` | the second implementation's agreeing read |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/recovery_curve.json` | per-unit score and rank at each of the next four checks |
| `runs/v32_redo_bottom_k_2026-08-28/smoke/` | the pre-existing wiring receipt (commit `e9cc5ed`) |

---

## 7. Rung 2 — (k = 4, C = 10), the registered escalation. **GO.**

Launched after rung 1's process exited, sequenced by process completion and
never by polling. 120 iterations = the same twelve cadenced checks, per the
reading pre-committed in §4 above, which was written before any rung-2 data
existed.

| gate | requirement | measured | verdict |
|---|---|---|---|
| **R1 REACHED** | ≥ 10 events of 12 checks; one `mode=bottom_k k=4 every_iters=10` line | 12 events / 12 checks, `cum_recycled` 48 = 4 × 12 | **PASS** |
| **R2 ARTIFACT-MATCH** | 100% of events | **12/12 = 100%**; min separation margin 0.000165 | **PASS** |
| **R3 DOSE** | `dose_fraction == 0.125` every check; fc1 0; ceiling never trips | 0.125 on all 12; fc1 0/64 on all 12; no overdose | **PASS** |
| **R4 TURNOVER** | `repeat_rate < 1.00` | **0.9091 (10 of 11 pairs)** | **PASS** |

Both shipped gates return `ARMED` / `GO`; the independent implementation agrees
field for field.

### 7.1 Nine free updates do buy recovery where four do not

Holding cumulative dose exactly constant, the recovery window moved 4 → 9 free
PPO updates, and the recycled-unit behaviour changed materially:

| | rung 1 (k=2, C=5) | rung 2 (k=4, C=10) |
|---|---|---|
| re-selected at next check | **20 / 22 = 90.9%** | **33 / 44 = 75.0%** |
| median rank one check later | 1 | 2 |
| distinct fc2 indices | 4 | 14 |
| top-index share | 45.8% | 20.8% |

The two rungs are not a clean single-variable contrast — k moves with C by
construction — but **the confound runs against the result, not for it.** A
larger k makes re-selection *more* likely under any null (a unit sits in the
recycled band by chance 2/32 = 6.25% at k = 2, but 4/32 = 12.5% at k = 4), and
the observed rate nonetheless fell. Distinct indices more than tripled.

### 7.2 The finding the gate does not capture — reported, never gated

R4 passed, and the registered rule is exactly what it says: *"VOID at
`repeat_rate == 1.00` EXACTLY, nothing else gated."* The registration made it
threshold-free on purpose, because picking any fraction below 1.0 would be a
threshold nobody had checked against its acting range — the family of error
behind nine vacuous gates. **That rule is not reinterpreted here.** Rung 2 is a
GO.

But the trace is not uniform, and the campaign that inherits this GO should
have the whole of it:

```
[4,9,21,26] [3,4,18,21] [5,22,24,30] [22,23,24,30] [15,17,23,30]
[8,23,26,30] × 7
```

The single clean break — the one pair sharing no index, which is what carries
R4 — occurs at **event 2 → 3**, in the first third of the window. From event 6
(iter 50) onward the set locks to `[8, 23, 26, 30]` and never moves again for
**seven consecutive events**. In its second half rung 2 is doing what rung 1
did throughout: maintaining a fixed multi-unit lesion, not recycling.

So the honest reading of the ladder is narrower than "rung 2 works":

> **A longer recovery window delays the lock-in; on this evidence it does not
> prevent it.** Rung 1 locked by event 5 of 12 and never broke. Rung 2 turned
> over freely for five events, then locked by event 6 and never broke either.

This is reported with **no verdict attached**, in the same spirit as the
retired F3 distinctness count. It moves no bar, changes no gate, and does not
withhold the GO. It is the single most decision-relevant thing to carry into
the campaign, and a campaign run at (k = 4, C = 10) should expect the
treatment to be a fixed four-unit lesion over roughly its back half.

---

## 8. Where the campaign actually stands

**The v32 campaign was not executed.** Rung 2 returned GO at 04:52; the
registered campaign is 4 seeds × 250 iterations plus a 192-evaluation honest
ladder, which does not fit the remaining budget of this session.

- **Seeds launched: 0. ARMED: 0. Scored: 0.**
- **Θ is not issued.** Per §5, fewer than four ARMED and scored seeds is
  **VOID-UNDERPOWERED**, no Θ, per-seed numbers banked individually — and
  there are none to bank.
- The 0.80 / 0.767 bars stand untouched; the 0.05 winner's-curse budget is
  unspent; Δ is NOT COMPUTED.
- **The plasticity hypothesis REMAINS UNTESTED**, exactly as it stood after
  v31. The v27/v28 confound is **not** discharged, and the 2026-08-25 DR ReDo
  prescription may **not** be closed as EXECUTED-AND-NEGATIVE — it still has
  not been executed.

What changed is that it is now *executable*: the ladder is resolved, an
operating point has passed all four preflight gates, and the next action is
mechanical — launch seeds 0-3 at
`configs/mario_1_1_v32_redo_bk_phase_r2.yaml`'s numerals (k = 4, C = 10),
sequentially, stdout outside each checkpoint dir, then the honest ladder and
the cross-fit reducer.

**One numeral must be re-derived before that launch, not inherited:** B1's
floor of *"≥ 48 recycle events of the 50 cadenced checks"* was written for
C = 5. At C = 10 a 250-iteration seed has **25** cadenced checks, so the
registered budget arithmetic is 25 events × k = 4 = 100 units = 3.125 trunk
turnovers — identical cumulative dose, as §8 states — and B1's event floor
must be restated as ≥ 24 of 25. That is arithmetic implied by the registered
rung, not a new choice, but it must be written into the campaign's own
receipt before compute rather than discovered afterwards.

---

## 9. Suite integrity — a real regression, found and fixed

The full suite was run as §12 requires and returned **two** failures, not the
one known-environmental failure the baseline permits:

```
FAILED tests/test_night2_runner.py::test_dry_run_passes_live          (known-environmental, left alone)
FAILED tests/test_reward_dispatch_is_explicit.py::test_roster_dispatch_matches_the_frozen_pre_change_baseline
```

The second is a genuine regression: `configs/mario_1_1_v32_redo_bk_seed{1,2,3}
.yaml` and the two Phase R profiles declare `reward_id: mario` and were never
appended to the frozen reward-dispatch roster, so each tripped the "profile
gained a specialized reward it never had" assertion. The prior session shipped
seeds 1-3 without re-running the full suite; `phase_r2` compounded it.

Fixed as the test's own documented policy prescribes — an append of five rows
with a batch comment naming the registration that authorized them, and the
expected count moved 126+4+1 → 126+4+1+5. **The frozen rows were not
regenerated**, which the test correctly treats as unforgivable. Revert-verified:
deleting one appended row fails two tests; restoring passes 39.

Post-fix suite, re-run end to end: **6019 passed, 30 skipped, 3 xfailed**, and
the single known-environmental `test_night2_runner` failure left alone — the
registered baseline shape restored.
