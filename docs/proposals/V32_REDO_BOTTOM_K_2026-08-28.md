# v32 — ReDo at a rank-based bottom-k dose. Pre-registration.

Registered 2026-08-28, **before any v32 compute is spent.** Every numeral
below is fixed as of this commit. A change to any of them after compute
starts voids the run; this document is the goalpost.

---

## 0. The licence — why this experiment may exist at all

v31 ran and returned **VOID**, at a preflight, before its campaign spent
anything. Its two admissible dose rungs failed **in opposite directions
with no window between them**, and its registered §9 stopping rule fired
on its own terms:

> *"On a `Linear -> LayerNorm -> SiLU` 32-unit trunk there is no fixed
> dormancy threshold that is simultaneously firing, surgical, and
> sustained over a 250-iteration budget. Fixed-threshold ReDo is
> mis-specified for this architecture. The rank-based bottom-k dose
> (v30 §10.1) is the only remaining form of the intervention — a new
> experiment, with its own registration, not a rung of this one."*

Receipts, with the live two-rung trace behind that sentence
(`docs/research/TWO_REGISTERED_TESTS_2026-08-27.md` Part II,
`runs/v31_redo_surgical_2026-08-27/phase_m/PHASE_M_RESULT.json`):

* **tau = 0.10** fired at iter 16 exactly as predicted, then the dose
  climbed `1,1,1,1,5,6,6,6,8,11,12,12,12,12` to an equilibrium of
  **12/32 = 37.5%** — the same equilibrium v30 measured at tau 0.15 —
  and the in-run ceiling raised VOID-OVERDOSE at iter 29.
* **tau = 0.075** was surgical (max 2/32) and sustained (31 firing
  iterations, cum 55) and failed anyway on **two distinct indices in the
  whole run**: `fc2=[16]` x 7 then `fc2=[5,16]` x 24. A permanent
  two-unit lesion, not a recycle.

**FIXED-TAU ReDo IS FORBIDDEN. This registration runs no threshold arm,
in any rung, for any reason.** The tail drifts down across training and
swallows any fixed tau; that is banked and is not re-tested.

**The licensed successor, and the only one:** recycle the bottom-k units
by dormancy **RANK**, k fixed and cadenced. This **caps the dose by
construction** — the drifting tail changes WHICH units are bottom-k,
never HOW MANY are recycled. Under a rank rule the failure mode that
killed both of v31's rungs cannot occur.

**Inherited from v31 wholesale and NOT reopened:** the honest protocol,
the Theta bar (>= 0.80 PASS / <= 0.767 FAIL), the cross-fit split-sample
estimator, the 0.05 winner's-curse budget, the in-run dose ceiling
(§4.3), the arming deadline, `redo_arm_gate.py` verdict gating, the
damage abort, and the standing prohibition below.

**Standing prohibition, inherited and unchanged:** nothing in v27, v28,
v30 or v31 may be cited as evidence for or against the plasticity-loss
hypothesis. v27/v28 because the treatment was inert; v30 and v31 because
neither produced a scored number.

**Standing rule, inherited and binding:** *every threshold is checked
against its acting range on the data it will see.* Nine vacuous gates
have shipped in this repository. §6.3 applies this rule to this
registration's one new gate, on banked data, before compute.

---

## 1. The evidence this registration is built on

All banked, all measured, none of it new compute. Receipts:
`runs/v30_premise_falsifier_2026-08-27/` (86 dormancy checks, 4
trajectories, 3 taus) and
`runs/v31_redo_surgical_2026-08-27/phase_m/` (two 60-iteration Phase M
runs at exactly 0.10 and exactly 0.075).

### 1.1 Dormancy lives ONLY in the 32-unit trunk

`fc1` is **0 of 64 and 0 of 96 dormant at every tau from 0.025 to 0.25
across all 86 measured iterations**, and its score minimum never falls
below **0.3086**. Every ReDo event in every arm ever run here has been an
`fc2` event.

**Registered consequence, decided now:** **bottom-k is scoped to `fc2`
ONLY.** `fc1` receives zero recycles under this registration. A rank rule
applied to fc1 would recycle its two least-active units every cadence
forever, on a layer measured never to have a dormant unit — pure damage
with no mechanism behind it. This is a scope restriction derived from a
measurement made before this compute, not a rescue.

The dose denominator is therefore **32**, and one unit is 3.125%.

### 1.2 The genuine deep tail is 1-2 units of 32

`tau = 0.075` is the lowest threshold ever measured to fire on this
architecture. Its whole firing history, from
`phase_m_seed0_tau075.log`:

| iters | dormant fc2 | recycled indices |
|---|---|---|
| 0-28 | 0 / 32 | (never fired) |
| 29-35 | 1 / 32 | `[16]` |
| 36-59 | 2 / 32 | `[5, 16]` |

The untreated control agrees from the other side: at iters 20-25,
`p5 <= 0.10` on 5 of 6 iterations and `p10 <= 0.10` on 4 of 6, and `p5`
of 32 units is index ~1.6 while `p10` is index ~3.2 — **1 to 3 units in
the tail.**

### 1.3 A re-initialized unit lands at the BOTTOM, not the middle

This is the single most decision-relevant measurement in this document,
and it is what makes the cadence a registered numeral rather than a
default.

At tau = 0.075, unit 16's score immediately before its first recycle was
**0.0744**. On the next check it read **0.0442**; the check after,
**0.0277** — *deeper than it had ever been untreated*, against a trained
tail sitting at 0.075-0.093. Across iters 29-59 the run minimum sits in
**0.027-0.054**, never once climbing back to the pre-treatment tail.

The tau = 0.10 log says the same thing structurally. Its recycled index
set is **monotonically nested** across 14 consecutive firing events:

```
[9] -> [9] -> [9] -> [9] -> [2,9,12,22,24] -> [2,9,12,16,22,24] -> ...
    -> [1,2,7,9,10,12,16,17,22,23,24,27]  (iters 26-29, identical)
```

**No recycled index ever left the set.** Unit 9 was recycled on 14
consecutive checks.

Mechanically this is expected: `recycle()` zeroes the unit's outgoing
actor and critic columns, so a fresh unit contributes nothing to the
output and, until gradients rebuild that column, its post-SiLU magnitude
sits far under a layer mean set by trained units.

**Two consequences, both registered:**

1. **Cadence 1 is FORBIDDEN under a rank rule.** At C = 1 the k fresh
   units are re-selected on the very next check with probability ~1 — a
   guaranteed k-unit lesion, structurally, before any data. This is not a
   prediction; it is the measured behaviour of the recycle operator on
   this architecture.
2. **Every prior ReDo trace in this repository is confounded on
   recovery**, because every one ran at C = 1 and never left a recycled
   unit alone for a single iteration. **Whether a re-initialized trunk
   unit can climb back out has never been measured here.** v32 is the
   first design that can measure it, and §6.3 makes that measurement its
   one live gate.

### 1.4 The timescale of dormancy-score movement is ~5 iterations

Untreated fc2 score minimum (`phase_m_seed0_tau075.log`, iters 0-28 are
a clean untreated trajectory because tau 0.075 does not fire until 29):

| iter | 0 | 1 | 2 | 3 | 4 | 5 | 8 | 15 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| min | .2848 | .2248 | .1773 | .1572 | .1448 | .1272 | .1066 | .1014 | .0999 | .0794 | .0753 |

The whole init-to-plateau transient completes in **~5 iterations**
(0.2848 -> 0.1272); after that the minimum moves ~0.005-0.02 per
iteration. **Five iterations is one measured relaxation timescale of this
statistic on this architecture**, and that is where the cadence numeral
comes from.

---

## 2. The operating point

## **k = 2, cadence C = 5, scope fc2 only.** Registered.

Config, otherwise the v27 block verbatim: `redo_enabled: true`,
`redo_mode: bottom_k`, `redo_bottom_k: 2`, `redo_check_every_iters: 5`,
`redo_sample_batch: 4096`, `redo_reset_optimizer_moments: true`,
`tile_hidden_dim: 64`, `tile_trunk_dim: 32`, `num_envs: 60`, 250
iterations. Width is 64/32 because **v27 is the ReDo-off control** and a
single-variable delta requires identical width.

`redo_tau` is **not read on this path** and is pinned to its schema
default in every v32 config, with a comment saying so. A live tau numeral
on a rank-rule run would be exactly the kind of dead knob this repo has
shipped twenty of.

### 2.1 Why k = 2

* **It is the measured tail.** §1.2: the lowest firing threshold ever
  observed selected exactly 1 unit for 7 iterations and exactly 2 for the
  remaining 24; the untreated `p5`/`p10` straddle 1-3 units. k = 2 is the
  size of the thing the mechanism is for.
* **6.25% of the trunk per event — one quarter of the 0.25 ceiling**, and
  **constant**. This is the whole point of the design: v31's dose was an
  emergent property of a drifting distribution, and it drifted from 1 unit
  to 12. Here it is 2, at iter 0 and at iter 245, by construction.
* **k = 1 is rejected**: it cannot produce index turnover *within* an
  event, and it halves the turnover budget (50 units = 1.56 trunk
  turnovers over the run, under the inherited F2 floor of 2.0).
* **k >= 8 is FORBIDDEN by this registration, including as an
  escalation**: 8/32 = 0.25 *is* the dose ceiling, and the same
  measurement that forbade tau >= 0.15 (37.5% of the trunk per event,
  post-hoc-VOID under V6) forbids it here.

### 2.2 Why cadence C = 5

* **C = 1 is forbidden by measurement** (§1.3), not by preference.
* **C must buy the fresh unit free updates.** At C = 5 a recycled unit
  gets **4 full PPO updates** before it can be selected again — one
  measured relaxation timescale of the dormancy statistic on this
  architecture (§1.4).
* **Turnover budget clears the inherited floor with margin.** 250
  iterations / 5 = **50 cadenced checks x k = 2 = 100 recycled units =
  3.125 trunk turnovers**, against v31's F2 floor of `>= 64` units =
  2.0 turnovers. 1.5x margin.
* **It is cheaper.** The dormancy check (a 4096-row forward pass) runs on
  one iteration in five, so ReDo's compute overhead is a fifth of v31's.
* **The cadence is the EXISTING `redo_check_every_iters` key**, already
  live in `maybe_check_and_recycle`. No new cadence knob is declared; a
  new declared key is a new way to be inert.

### 2.3 The one honest cost, stated in advance

At C = 5 the first cadenced check is **iter 0**, where the network is
near init and the fc2 tail sits at 0.2848 — nothing is dormant. Bottom-k
will recycle 2 near-init units anyway, because a rank rule has no
abstention. Over 250 iterations the pre-crossing region (iters 0, 5, 10)
contributes **6 of 100 recycled units, 6%** — and re-sampling a near-init
weight row from the init distribution is close to a no-op.

**No warm-up iteration count is registered.** A warm-up would be a second
free parameter with no measurement behind it, and the cost it would buy
back is 6%. This paragraph exists so the property is recorded before
compute rather than discovered in an adjudication.

---

## 3. What ARMED means under bottom-k

A rank rule **fires by construction every cadence**. The v27-v31 arming
question ("did the treatment fire at all?") is therefore answered before
the run starts, and inheriting F1/F2 unchanged would install a gate that
cannot fail — the tenth vacuous gate, in a registration whose brief is
preventing the ninth.

**ARMED is re-specified. A seed is ARMED only if all four hold.** B1-B4
**replace** v31's F1/F2/F3 as the arming gate; where F1/F2's numerals
appear elsewhere in this document (§2.1, §2.2) they are cited as
**budget benchmarks** for the k and C justification, never as live
conditions. There is exactly one arming gate and it is this one.

### B1 — REACHED (the hot path, not the unit test)

The seed's `run.log` carries exactly one `[redo] ENABLED` line and it
renders, verbatim:

```
[redo] ENABLED tau=0.025 every_iters=5 scope=fc1,fc2 sample=4096 reset_moments=true mode=bottom_k k=2 recycle_scope=fc2
```

(one physical line; the wrap above is this document's, not the log's)

The log contains no `[redo] disabled`, and records **>= 48 recycle
events** out of the 50 cadenced checks in 250 iterations, with
`cum_recycled == 2 x events`.

Field notes, so a later gate cannot mis-grep them: `scope=fc1,fc2` is the
**module's** scope and is pinned by the inherited B6 evidence test;
`recycle_scope=fc2` is the **rule's effective** scope; and `tau=0.025` is
logged for provenance only — the rank rule never reads it (§2).

Two events of slack absorb `[redo] iter N: skipped (no gradient step)`
checks, which are legal and logged.

**This is the condition six previously-built signals failed.** A
mechanism firing in a unit test is not a mechanism firing in training;
B1 is read off the production log or the seed is not ARMED.

### B2 — GENUINELY THE BOTTOM (artifact-match)

On every recycle event the trainer logs the **full 32-value fc2 dormancy
score vector** beside the recycled indices. The gate **recomputes
bottom-k offline from the logged scores** and requires an exact
index-set match on **100% of events**.

This is Lane A's lesson applied before the fact rather than after it:

> *A guard that runs on the values in hand certifies the loop. Only a
> guard that runs on the bytes written certifies the file.*

B2 fails whenever selection and logging disagree — the exact class of
defect that voided Lane A's 2.13 h, where eight admissibility gates
passed on a bank whose rows were not transitions. It is not a
tautology: it compares two independently-produced artifacts.

**Reported alongside, never gated:** the separation margin
`min(non-recycled score) - max(recycled score)` per event, and its
distribution over the run.

### B3 — DOSE (structural; asserted anyway)

`dose_fraction = max(0/64, 2/32) = 0.0625` at **every** executed check,
and the trailing-10-check median never exceeds 0.25.

This is structural under a rank rule and cannot fail unless the
mechanism is not doing what its construction says — which is exactly why
it is asserted rather than assumed. The in-run ceiling
(`_REDO_DOSE_CEILING = 0.25`, `dose_ceiling_trips`) stays armed
unchanged, and V6 re-checks it post-hoc.

### B4 — RECOVERY (the live one)

See §6. `repeat_rate < 1.00`.

**Any of B1-B4 failing is VOID, not FAIL.** VOID enters no aggregate and
takes no branch of the fork.

---

## 4. Aborts

* **A1 — Phase R (§7, ~0.6 h).** R1-R4 at exactly (k=2, C=5). Any
  failure -> the §8 ladder or STOP. **No training begins before Phase R
  returns GO.**
* **A2 — arming deadline.** `_REDO_ARM_DEADLINE_ITERS = 40` stays armed
  unchanged. Under bottom-k the first fire is iter 0, so this is
  structurally satisfied; it is left in place because the case it now
  catches is "the hot path never executed", which is worth an abort.
* **A3 — in-run dose ceiling.** Unchanged (`M10(t) > 0.25` ->
  `RuntimeError`, seed VOID-OVERDOSE, whole sequence aborts).
  Structurally unreachable at k = 2; left armed for the same reason as
  A2.
* **A4 — damage abort (~45 min).** Inherited verbatim from v31 A5: after
  treatment seed 0 reaches iter 100, if cumulative entrance rate < 0.10
  **and** trailing rate < 0.10, cancel seeds 1-3. Verdict: *"bottom-k
  ReDo at k=2/C=5 is destructive on this stack."* v27's worst seed at
  iter 100 was 0.186 cumulative / 0.23 trailing, so the bar sits below
  half of that on both axes.
* **A5 — identity, DEMOTED, inherited.** Median greedy-argmax agree
  >= 0.60 is retained as gate condition V5 and reported; it is **not**
  the dose gate and never aborts on its own. v31 measured median agree
  **0.97 while 37.5% of the trunk was being reset every iteration** —
  the statistic is structurally insensitive to dose and that is now a
  live receipt, not an argument.
* **A6 — Delta cut at 10.5 h.** §5.
* **A7 — wall clock 13.0 h hard.** At the ceiling, stop and report what
  is banked.
* **A8 — protocol integrity.** `warp_rate != 0.0`, any `--strict-config`
  rejection, any honest-protocol deviation, or any config drift from the
  registered operating point -> VOID.
* **A9 — log integrity (new, from the Lane B post-mortem).** The trainer
  attaches its own **truncating** `FileHandler` to
  `<checkpoint_dir>/run.log`. Any orchestrator stdout redirect MUST go to
  a path outside that directory, and **any phase that reads a log another
  process writes MUST copy it first.** A phase adjudicated off an
  interleaved log is VOID. (v31's first "Phase M NO-GO" was a missing
  `mkdir`, timestamped in the same second as its own launch.)

---

## 5. Seeds, budget, sequencing

**Seeds 0, 1, 2, 3 — four, 250 iterations, `num_envs 60`, width 64/32**,
mirroring v27 exactly so the control is matched and Theta is the same
statistic as the banked best-of-4.

| phase | h |
|---|---|
| Phase R — recovery + dose measured at exactly (k=2, C=5), 60 iters | 0.60 |
| treatment training, 4 x 250 iters @ ~25 s/iter | 7.05 |
| treatment honest ladder, 24 ckpt x 2 es x 4 seeds = 192 evals @ ~28 s, 8 workers | 1.50 |
| v27 control backfill — the same 192-eval ladder, for Delta | 1.50 |
| gate runs, verdict, receipts, manifest | 0.30 |
| slack | 2.05 |
| **hard ceiling** | **13.0** |

**Ordering is registered so the wall clock cuts the least important thing
first:** Phase R -> seeds 0-3 -> treatment ladder (Theta) -> v27 backfill
(Delta) -> verdict. **At 10.5 h elapsed the v27 backfill is cut** and
Delta is reported as NOT COMPUTED. Theta, the gated statistic, is never
what gets cut.

**CPU sequencing with Lane A.** Lane A's on-policy collection runs
`--workers 2` and finishes first. **Lane B's Phase R starts only after
Lane A's collection process has exited** — sequenced by process
completion, never by polling. Training must log **25.4 +/- 2.5 s/iter**;
outside that band the arm-vs-control comparison is flagged in the
verdict.

**N is fixed at 4 and Theta requires 4 ARMED, scored seeds.** best-of-3
is a different statistic from the banked best-of-4 control and from the
bar; comparing them would be an estimator mismatch. Fewer than 4 ARMED
and scored inside the ceiling -> **VOID-UNDERPOWERED**: no Theta issued,
per-seed numbers banked individually as mechanism receipts.

### 5.1 What n = 4 cannot conclude — inherited, unchanged, restated

Across the 8 banked v27/v28 runs the per-seed honest rate is **bimodal**:
2 of 8 collapsed below 0.10 (v27 seed0 0.03, v28 seed2 0.09) and 6 sit in
0.46-0.67 (mean 0.527, sd ~0.08). **Max single seed ever observed:
0.67.** Therefore at n = 4:

* Powered only for a **>= +0.25** shift in the per-seed rate. A real but
  modest lever of +0.05 to +0.15 — the size the recovery assay's own
  ceiling analysis says is even available — is **invisible**.
* PASS requires a seed **0.13 above the best of all 8 prior seeds**, so
  absent a large effect FAIL is near-certain by construction.
* With a ~25% per-seed collapse rate, best-of-4 is a max statistic whose
  null already reaches ~0.50. **Any best-of-4 in 0.50-0.65 is
  indistinguishable from seed noise** and may never be reported as
  improvement over v27.
* n = 4 cannot separate "ReDo helped nothing" from "ReDo helped and hurt
  in equal measure", and cannot confirm Hypothesis B.

---

## 6. F3 under bottom-k — the decision, made here and not after

### 6.1 F3 as written is RETIRED

v31's F3 required **>= 6 distinct fc2 indices** recycled over the run and
**no single index above 60%** of recycled-unit-events. Inherited
unchanged it would be **wrong in the specific way this design is
right.**

Under a *threshold* rule, a fixed index set means the threshold is
selecting a permanent lesion: the treatment is not recycling, it is
holding two units down. Under a *rank* rule, taking whatever sits at the
bottom **is the definition of the mechanism**. If the same unit is
genuinely the least-active unit in the trunk after four free PPO updates
in which it could have recovered, then recycling it again is the
mechanism doing exactly its job on a unit that will not come back.
**Gating on distinctness would VOID bottom-k for working.**

So: **F3's distinctness threshold does not gate this campaign.** Distinct
index count, top-index share, and the full per-index histogram are
**REPORTED** on every seed, with no verdict attached.

### 6.2 What replaces it — F3', TURNOVER, and why that is the real failure

The degenerate case under a rank rule is not low distinctness. It is
**zero turnover**: the recycled set never changes because
re-initialization *itself* puts a unit at the bottom (§1.3 — fresh units
read 0.027-0.054 against a trained tail of 0.075-0.093). If that is the
whole story, bottom-k is a k-unit lesion by construction, and its FAIL
would be uninterpretable in exactly the way an overdose's is.

> **F3' — TURNOVER.** Let `repeat_rate` = the fraction of recycle events
> (after the first) whose index set shares **at least one** index with
> the immediately preceding event's index set.
>
> **VOID at `repeat_rate == 1.00` exactly.** Nothing else is gated.

**Threshold-free, deliberately.** No fraction below 1.0 is asserted,
because picking one would be a threshold nobody checked against its
acting range — the family of error that produced nine vacuous gates.
1.00 is the degenerate signature itself: it means the recycled set never
turned over even once in 50 events, so cadence C bought no recovery at
all. This is the same construction as Lane A's §1.8 non-degeneracy guard,
adopted for the same reason.

### 6.3 F3' checked against its acting range, on banked data, before compute

The standing rule requires the check to be evaluated on data it will
see. Both v31 Phase M logs are real ReDo traces from this exact trainer
on this exact architecture:

| banked log | events | `repeat_rate` | F3' |
|---|---|---|---|
| `phase_m_seed0_tau075.log` | 31 | **1.00** (unit 16 in 31/31, consecutive) | **VOIDs** |
| `phase_m_seed0_tau010.log` | 14 | **1.00** (index set monotonically nested) | **VOIDs** |

**The guard fires on every real ReDo trace this repository has ever
produced.** It is not decorative, and it is not tuned to pass this
campaign — as of registration time it is a gate that **100% of prior
evidence fails.**

Both those traces ran at cadence 1, where §1.3 says a repeat is
structurally forced. Whether cadence 5 changes it is the open question,
and it is open in both directions. That is what makes F3' a live branch
rather than a decoration, and it is why Phase R exists.

---

## 7. Phase R — the preflight, at exactly the registered operating point

`configs/mario_1_1_v32_redo_bk_seed0.yaml` verbatim, seed 0, 64/32,
`num_envs 60`, **60 iterations**, `--no-resume --no-supervise
--strict-config`, stdout redirected **outside** the checkpoint directory
(A9). ~0.6 h.

60 iterations gives **12 cadenced checks** — enough to see turnover or
its absence, and 60 is the same window v31's Phase M used, so the
throughput comparison is like-for-like.

**GO requires all four. All are evaluated at exactly (k=2, C=5).**

* **R1 — REACHED.** `>= 10` recycle events within the 12 cadenced checks
  of 60 iterations, from a log carrying exactly one
  `[redo] ENABLED ... mode=bottom_k k=2 every_iters=5` line. This is the
  cheap pre-check on B1.
* **R2 — ARTIFACT-MATCH.** Offline recomputation of bottom-k from the
  logged score vectors reproduces the logged recycled indices on
  **100%** of events. Cheap pre-check on B2. **Any mismatch is an
  implementation defect: STOP, fix, re-register** — it is not a dose
  question and no rung addresses it.
* **R3 — DOSE.** `dose_fraction == 0.0625` on every executed check; the
  in-run ceiling never trips. Cheap pre-check on B3.
* **R4 — TURNOVER.** `repeat_rate < 1.00` over the window. Cheap
  pre-check on B4, and the §8 ladder's only trigger.

**Also recorded from Phase R, whatever it decides**, as the measurement
that no prior run could make: the **recovery curve** — for each recycled
unit, its dormancy score at each of the next four checks, i.e. whether a
re-initialized trunk unit climbs out of the bottom-k when left alone.
This is banked as a mechanism finding in every branch, including STOP.

Phase R's checkpoint is **discarded**; the campaign runs all four seeds
from scratch with `--no-resume`.

---

## 8. The ladder, and the stopping rule

Two free numerals exist in this design, k and C. The ladder moves
**exactly one of them, once, in one direction**, and it is written here.

* **Rung 1 — (k = 2, C = 5).** The registered operating point.
* **If Phase R fails R4** (`repeat_rate == 1.00`: cadence 5 bought no
  recovery): escalate **once** to **(k = 4, C = 10)** and re-run Phase R
  in full at exactly that point.

  **Cumulative dose is held EXACTLY constant** — 250/10 = 25 events x
  k = 4 = **100 units = 3.125 trunk turnovers**, identical to rung 1 — so
  the rung changes the **recovery window** (4 -> 9 free PPO updates) and
  nothing else. Per-event dose 4/32 = **12.5%**, half the 0.25 ceiling.
  A second NO-GO is therefore about recovery, not about dose.
* **If Phase R fails R1, R2 or R3:** **STOP.** Those are implementation
  defects — the hot path did not run, the artifact does not match the
  loop, or the dose is not what construction says. No rung fixes code.
  Fix it, re-register, and start over.
* **If the second Phase R also NO-GOes: STOP**, launch nothing, and bank:

  > *"On a `Linear -> LayerNorm -> SiLU` 32-unit trunk, a re-initialized
  > trunk unit does not climb out of the rank-bottom of the dormancy
  > distribution within 9 free PPO updates. The recycled set cannot turn
  > over, so ReDo on this architecture is a k-unit lesion in every
  > admissible form — fixed-threshold (v31: mis-specified, no window
  > between over-dose and lesion) and rank-based (v32: dose capped, but
  > non-recovering). The intervention is RETIRED for this stack."*

  That statement, with its two-registration receipt, is the strongest
  result the no-training branch can produce, and it retires a
  prescription rather than leaving it open.
* **FORBIDDEN by this registration, including as escalations:** `C = 1`
  (§1.3, measured non-recovery); `k >= 8` (= the dose ceiling itself);
  any threshold arm; any k or C change after a seed is launched or after
  any Theta or Delta has been seen. A change after seeing a number is a
  moving goalpost and voids the campaign.

---

## 9. The bar and the estimator — inherited, not reopened

* **PASS**: Theta >= **0.80**
* **FAIL**: Theta <= **0.767**
* **MARGINAL**: 0.767 < Theta < 0.80 — reported as MARGINAL, never as
  PASS; authorizes no follow-on claim and no follow-on campaign.

**Theta = best-of-4 over seeds of the cross-fit split-sample honest clear
rate.** Candidate set = the 24 saved
`vanilla_ppo_iter_{00010..00240}.pt`. Per seed: `score_A` = honest rate
on eval seed 1 at the checkpoint argmax'd on eval seed 0; `score_B` =
honest rate on eval seed 0 at the checkpoint argmax'd on eval seed 1;
seed score = `(score_A + score_B)/2`. 100 scoring episodes per seed;
**every episode is used exactly once as a scoring episode and never for
the selection that chose its own checkpoint.** Ties -> later iter, both
directions reported.

This is the fix for the banked checkpoint-selection defect
(`project_checkpoint_selection_defect_2026-08-26`: the winner-selector
under-selected by 20-40 iterations on 4 of 4 seeds).

**Honest protocol, immutable:** cold entrance, greedy, sticky 0.25,
jitter +/-16, 50 eps x eval seeds {0,1} = 100 pooled, `--eval-rng
per-episode`, `max_steps 1500`, `rom_sha256` on every receipt.
`warp_rate` must be 0.0.

**Winner's-curse budget 0.05**, reported as `Theta_adj = Theta - 0.05`. A
PASS landing in `[0.80, 0.85)` is flagged "PASS within the measured
curse" and requires a confirmation re-score of the winning seed's
selected checkpoint on a **third eval seed (es2, 50 fresh episodes,
registered now)** before it may be called PASS in any external claim.

**Secondary — Delta, registered now so it cannot be a post-hoc rescue,
and it cannot produce a PASS.** `Delta = Theta - Theta_v27` under the
**identical** cross-fit reducer (`runs/v27_readjudication_2026-08-27/`).
`Delta >= +0.15` -> "bottom-k ReDo is a real lever on this stack", a
mechanism finding, never a gate PASS. `Delta <= +0.05` -> "not a lever."
In between -> indeterminate at n = 4; say so.

**The ReDo-off control is v27, unchanged, and no new control arm is
run.** ReDo was provably inert in v27 (0 recycles on every check of all
four seeds), which is what makes it a valid ReDo-off control. Residual
risks named and accepted: **not a paired test** (ReDo consumes numpy RNG
at every firing, so the streams diverge at the first recycle), and
machine conditions differ across days, mitigated by the throughput band
in §5.

**Protocol note, recorded so it is never used as a rescue.** Re-measured
under this registration's protocol the banked control reads **0.68**.
**0.767 does not move.** It is a numeral fixed pre-run; the 0.68 figure
is context and carries no verdict weight.

---

## 10. Interpretation — the fork, and exactly what a FAIL licenses

### 10.1 PASS (Theta >= 0.80)

Plasticity loss in Sokar's sense **was** the barrier at 48k parameters on
1-1, and a rank-capped dose is the form of the intervention that reaches
it. Hypothesis A (consolidation) confirmed; Hypothesis B (48k is a hard
ceiling) falsified. Bottom-k ReDo becomes a standing element of the
recipe. A PASS in `[0.80, 0.85)` requires the es2 confirmation re-score
before any external claim.

### 10.2 MARGINAL (0.767 < Theta < 0.80)

Reported as MARGINAL. Licenses nothing. In particular it does **not**
authorize a follow-on campaign at another (k, C) — that would be ladder
climbing after seeing the number, which is what this document exists to
prevent.

### 10.3 FAIL (Theta <= 0.767) — the licence, stated in advance

**A v32 FAIL licenses, in writing, exactly this and no more:**

1. *"With the dose capped by construction at k = 2 of 32 trunk units
   (6.25%) per event, cadenced every 5 iterations, sustained over >= 48
   events and >= 3.0 trunk turnovers, with the recycled units verified
   against the logged score vectors to be the rank-bottom of the
   dormancy distribution on 100% of events, and with the recycled set
   measured to turn over (`repeat_rate < 1.00`), recycling dormant units
   did not move the best-of-4 cross-fit honest clear rate above 0.767 on
   1-1 at 48k parameters."*
2. *"Dormant-neuron recycling is not a **large** lever (>= +0.25 in the
   per-seed honest rate) on this stack."* — the one inference n = 4 is
   powered for.
3. **The un-confounding of v27 and v28.** Their FAILs were
   uninterpretable on plasticity because the treatment was inert. A v32
   FAIL is the first FAIL from a firing, dose-capped, sustained,
   non-lesioning ReDo run, so that specific confound is **discharged**:
   v27's FAIL becomes a clean result about the merged fresh-run
   curriculum and v28's a clean result about parameter budget, each with
   the plasticity alternative tested rather than assumed away.
4. **Retirement of the prescription, in both its forms.** The 2026-08-25
   DR's ReDo prescription may be closed as **EXECUTED-AND-NEGATIVE** in
   the claims ledger and removed from the backlog. Jointly with v31's
   banked stopping statement (fixed-threshold: mis-specified) this
   retires ReDo from this program entirely: the threshold form has no
   admissible window, and the rank form was executed and did not clear
   the bar. **No previous result could do this**, and no successor
   intervention inherits a licence from it.

**A v32 FAIL does NOT license, and any of these in writing is a
fabrication:**

* *"Hypothesis B (the 48k budget is a hard ceiling) is CONFIRMED."* n = 4
  cannot confirm it, and the DR's inference assumes an
  identity-preserving intervention this architecture measurably does not
  provide (median agree 0.85-0.97, never 0.98, at any firing dose).
* *"Plasticity loss does not occur on this stack."* The statistic was
  measured and the treatment applied; neither establishes absence.
* Any claim about a **small** lever (+0.05 to +0.15) — the size the
  recovery assay's own ceiling analysis says is even available, and which
  n = 4 cannot see in either direction.
* Any claim about **other** plasticity interventions — L2-init
  regularization, CReLU, weight churn, layer-norm resets, periodic full
  resets. v32 tests ReDo at one rank dose on one architecture.
* Any claim about other widths, other levels, or other games.

### 10.4 VOID

VOID is not FAIL, takes no branch of the fork, and **enters no
best-of-N.** A VOID campaign licenses only the mechanism finding attached
to its specific void reason — §8's stopping statement for a second Phase
R NO-GO, and in every branch the §7 recovery curve, which is a
measurement no prior run could make.

---

## 11. VOID conditions, enumerated

| condition | verdict |
|---|---|
| `< 48` recycle events, or `cum_recycled != 2 x events` (B1) | VOID-NOT-REACHED |
| offline bottom-k recomputation disagrees with logged indices on any event (B2) | VOID-ARTIFACT-MISMATCH |
| `dose_fraction != 0.0625` on any check, or in-run `M10(t) > 0.25` (B3) | VOID-DOSE |
| `repeat_rate == 1.00` (B4 / F3') | VOID-NO-TURNOVER |
| `[redo] ENABLED` absent, mode/k/cadence != registered, or `[redo] disabled` present | VOID-NOT-ARMED / VOID-WRONG-POINT |
| median agree `< 0.60` | VOID-IDENTITY |
| fewer than 4 ARMED **and** scored seeds inside the wall clock | VOID-UNDERPOWERED (no Theta) |
| `warp_rate != 0.0`, config drift, protocol deviation | VOID |
| any phase adjudicated off a log written concurrently by another process (A9) | VOID |

---

## 12. Code required before compute, each with its executed failure

1. **Bottom-k rank selection** in `src/training/redo.py`, flag-gated
   (`redo_mode`), default `threshold` so every existing run is
   byte-identical. **SHIPPED with this registration, revert-verified**
   (§13).
2. **The fc2 score-vector log line** at INFO on every recycle event —
   the B2 artifact-match input. **SHIPPED**, pure logging, no RNG.
   Renders as `[redo] fc2 scores: [...]`, 32 values in unit order,
   measured BEFORE the recycle, on the line after the existing
   `[redo] recycled unit indices:` line whose prefix is unchanged so the
   inherited F3 parser keeps working.
3. **B2 artifact-match + F3' turnover conditions** in
   `scripts/redo_arm_gate.py`, parsing those lines. **OWED before Phase
   R**, with a synthetic mismatch trace that must VOID, a synthetic
   `repeat_rate == 1.00` trace that must VOID, a healthy trace that must
   ARM, and the executed deletion check with its failure count recorded.
4. **`scripts/adjudicate_phase_r.py`** — R1-R4 plus the recovery curve.
   **OWED before Phase R.**
4b. **`scripts/experiment_preflight.py`'s `redo` entry must gate the
   MODE, not just tau.** It currently builds its armed regex as
   `\[redo\] ENABLED tau=<profile tau>` and nothing else. Because the
   v32 configs pin `redo_tau: 0.025`, that regex **matches a v32 run and
   would equally match a threshold run at the same tau** — i.e. as of
   today it cannot tell the registered arm from the forbidden one. It
   must additionally require `mode=bottom_k k=<profile redo_bottom_k>`
   and `every_iters=<profile cadence>`. **OWED before Phase R**, with the
   executed failure: a synthetic threshold-mode log at tau 0.025 must
   fail the v32 preflight. Recording this here rather than discovering
   it later is the point — an armed-check that cannot distinguish the
   treatment from its forbidden sibling is a vacuous gate, and this
   would have been the tenth.
5. **`run_manifest.json`** gains `redo_mode`, `redo_bottom_k`,
   `redo_repeat_rate`, `redo_distinct_fc2_indices`. **OWED.**
6. **`configs/mario_1_1_v32_redo_bk_seed{0..3}.yaml`** — a clean
   single-functional-variable diff from the v27 seed configs. **seed0
   SHIPPED** (the whole diff below `name:` is the redo block and
   nothing else); seeds 1-3 **OWED**.
7. **A9 orchestration**: stdout redirected outside `<checkpoint_dir>`,
   and any log read by a later phase copied first. **OWED.**

Suite baseline that must hold after all of it:
`.venv/bin/pytest tests/ -q --timeout=120` -> ~5950 passed, 30 skipped,
3 xfailed, plus the one known-environmental failure
(`tests/test_night2_runner.py::test_dry_run_passes_live`), which is left
alone.

---

## 13. Anti-vacuity — executed, not asserted

Registered standard, unchanged: **every check ships with a
revert-verified failure.** Neutering bottom-k selection in
`src/training/redo.py` — making `select_units` fall back to the
threshold rule — must make the new tests in
`tests/test_redo_bottom_k.py` fail, and **the deletion must be executed
and the failure count recorded** in the commit message and in the
adjudication.

The mechanism must additionally be shown **reaching the trainer's
production path**: a short live run with the flag on, showing
`[redo] ENABLED ... mode=bottom_k` and real recycle events at the
registered cadence inside the actual training loop. Six signals in this
repository were once built and wired to nothing; a unit test is not a
receipt that the hot path runs.

### 13.1 Both, executed — the receipts

**Revert verification: 4 of 33 fail.** Neutering the `SELECT_BOTTOM_K`
branch in `check_and_recycle` so the rank rule falls through to the
threshold rule fails
`test_rank_rule_fires_when_no_unit_is_below_any_plausible_tau`,
`test_rank_rule_dose_is_exactly_k_at_the_drift_extreme`,
`test_rank_rule_never_touches_fc1`, and
`test_stats_carry_the_full_score_vector_and_it_reproduces_the_selection`.
Restoring returns 33/33.

**Production-path smoke: the hot path reaches it.**
`runs/v32_redo_bottom_k_2026-08-28/smoke/` — `train_game.py` on
`configs/mario_1_1_v32_redo_bk_seed0.yaml`, 11 iterations,
`--no-resume --no-supervise --strict-config --seed 0`, stdout redirected
outside the checkpoint directory per A9.

| iter | recycled fc2 | fc2 score min | B2 offline recomputation |
|---|---|---|---|
| 0 | `[9, 13]` | 0.2079 | matches |
| 5 | `[20, 25]` | 0.2639 | matches |
| 10 | `[9, 17]` | 0.2084 | matches |

All four ARMED conditions are exercised live: **B1** the ENABLED line
plus 3 events on 3 cadenced checks at exactly C = 5; **B2** offline
bottom-k recomputed from the logged score vectors reproduces the logged
indices on 3 of 3 events; **B3** dose 2/32 = 0.0625 constant, `fc1 0/64`
recycled at every check; **B4** `repeat_rate = 0/2 = 0.00`.

**The counterfactual on those same three activation batches is the
sharpest thing in this document.** At every admissible v31 tau — and at
the *forbidden* 0.15 — a threshold rule selects **zero** units on all
three checks. And at v30's tau 0.25 it selects **1, then 0, then 11**:
the dose swinging by an order of magnitude inside an 11-iteration
window, on the same trajectory where the rank rule took **2, 2, 2**.
That is v31's disease reproduced in miniature and the cap-by-construction
claim demonstrated beside it, on live trainer data.

**What the smoke is NOT.** It ran 11 iterations at `--num-envs 8` under
Lane A contention. It is a **wiring receipt only**. No timing claim, no
dose-drift claim, no turnover claim and no Theta claim may be derived
from it — in particular the `repeat_rate = 0.00` above is **not** a
Phase R result and does not discharge R4. §7 is still owed in full at
`num_envs 60` for 60 iterations.

---

## 14. Receipts this campaign must produce

| path | what |
|---|---|
| `runs/v32_redo_bottom_k_2026-08-28/smoke/` | the flag-on live smoke: `[redo] ENABLED ... mode=bottom_k` + recycle events in the real loop |
| `runs/v32_redo_bottom_k_2026-08-28/phase_r/` | the 60-iteration measurement at exactly (k=2, C=5), the R1-R4 adjudication, and the recovery curve |
| `runs/v32_redo_bottom_k_2026-08-28/arm_gate.json` | B1-B4 over all four seed logs |
| `runs/v32_redo_bottom_k_2026-08-28/theta.json` | the cross-fit reducer's output: per-seed `score_A`/`score_B`, Theta, `Theta_adj`, and the 192 eval receipts it consumed |
| `runs/v32_redo_bottom_k_2026-08-28/delta.json` | the v27 backfill under the identical estimator, or an explicit NOT COMPUTED with the reason |
| `checkpoints/mario_1_1_v32_redo_bk_seed{0..3}/` | run logs, 24 iterate checkpoints per seed, `run_manifest.json` |
| `docs/research/` | the adjudication document, written after the numbers, against these numerals |
