# Direction — 2026-08-28

**The recommendation of 2026-08-28 was OVERTURNED by the review it
commissioned.** Two of its four parts are reversed, two are modified, none
stands unchanged. The recommendation was written by the session that ran
v27, v28, v30, v31 and v32; it was reviewed adversarially; the review went
to the banked artifacts and found the recommendation's central mechanism
claim false and its proposed next purchase unfunded by the evidence cited
for it. That is the system working. This document is the corrected
direction, and it supersedes the recommendation entirely.

Every number below was recomputed from banked artifacts during the review.
**Zero training compute was spent producing this document.** Two receipts
were banked in the process, both pure offline aggregation over files that
already existed:

* `docs/receipts/v32_redo_bottom_k/campaign_recovery_curve.json` (working copy at `runs/v32_redo_bottom_k_2026-08-28/seeds/recovery_curve.json`; `runs/` is gitignored) — the
  full-campaign recovery curve that §7 of the v32 registration requires
  *"in every branch, including STOP"*, and which the campaign never
  computed.
* `docs/receipts/v32_redo_bottom_k/gain_vs_dormancy_score.json` — the
  measurement that overturns the stopping statement.

---

## 1. The decision

**Launch no training campaign. Spend the next unit of work on the
instrument, eval-only, and register the selection rule before anything
else is launched.**

Four actions, in order. Actions 1–3 are the decision; action 4 is the
condition on everything after it.

### Action 1 — Bank the corrected ReDo stopping statement, not the proposed one

The recommendation asked to bank:

> *"ReDo in both admissible forms cannot be operated stably on a
> `Linear -> LayerNorm -> SiLU` 32-unit trunk."*

**Do not bank that.** It attributes the failure to the architecture, and
the checkpoints on disk say the failure is the operator. Bank this
instead — narrower in scope, stronger in evidence, receipted at n = 384:

> **On this `Linear -> LayerNorm -> SiLU` 32-unit trunk the ReDo dormancy
> score is, to first order, a rank-readout of the learned LayerNorm gain.**
> Spearman(`norm2.weight`, fc2 dormancy score) = **+0.932 / +0.943 /
> +0.905 / +0.773** across the four v32 seeds; trained gains span
> **0.477 to 11.454** against layer means of 3.74–4.25. The standard
> recycle operator (`src/training/redo.py`) resets a recycled unit's gain
> to **exactly 1.0** and zeroes its **actor and critic** outgoing columns
> together with their Adam moments. With both head columns at zero the
> unit receives no gradient, so its gain stays pinned at 1.0 until those
> columns regrow from zero. **The operator therefore deposits its own
> recycled unit at the bottom of the statistic that selects it, and
> re-selects it.** Measured at the campaign's operating point (k = 4,
> C = 10; 4 seeds × 24 events × 4 units = **384 unit-observations**):
> **91.41 %** of recycled units are still rank-bottom-4 one check later,
> median next-check rank **2 of 32**, only **1.30 %** reach the upper half
> of the layer. Per seed: 86.5 / 91.7 / 90.6 / 96.9 %.
>
> **Escape is a threshold in cadence, not an impossibility.** Following
> each unit forward only while it is *not* re-recycled: at +1 check
> (9 free PPO updates) n = 384, median rank 2.0, 91.4 % still bottom-4;
> at **+2 checks (19 free updates) n = 28, median rank 12.0, 7.1 % still
> bottom-4**; and from +3 through +23 checks the median rank climbs 11 → 18
> with 0–8 % ever falling back. A unit that survives one extra window does
> not return.
>
> **RETIRED: the cadence-and-threshold search.** Every fixed threshold
> (v31: τ = 0.10 equilibrates at 12/32 and trips the dose ceiling;
> τ = 0.075 is a permanent two-unit lesion) and the rank rule at cadence 10
> are retired for this stack. What five registrations searched was the
> *delivery schedule* — τ → fixed τ → surgical τ → rank-based bottom-k —
> around an operator that all five held fixed and that no registered rung
> addresses.

Bank alongside it two corrections to the existing ledger:

* **`V30_REDO_ARMED_2026-08-27.md:48-50` is refuted in its stated reason.**
  It banked *"pre-activation LayerNorm re-normalizes to zero-mean /
  unit-variance across units, every forward pass, so no unit's activation
  magnitude can decay away relative to its layer."* The learned per-unit
  affine gain reintroduces exactly that decay — a 24× spread across units —
  and it is the dominant term in the score. **v30's VOID stands** (the DR's
  prescribed 0.025–0.1 τ range was genuinely unreachable); the mechanism
  reason banked with it is wrong and must be corrected in place.
* **The v32 §8 stopping text is right in its narrow clause and wrong in its
  generalization.** *"does not climb out of the rank-bottom … within 9 free
  PPO updates"* is **confirmed** (91.4 % at +1 check). *"The recycled set
  cannot turn over"* is **refuted** (median rank 12 at +2 checks, and it
  stays there). *"on this architecture"* is refuted by the gain finding.
  §8's antecedent never fired anyway (see §3, Position A).

### Action 2 — Do not spend eval compute scoring v32 seed 0

**Upheld from the recommendation, and this is the one part that survives
in substance.** Seed 0 is not in a different regime from the three VOIDs.
Verified from the receipts:

| seed | distinct fc2 | disjoint pairs | repeat_rate | longest identical run | next-check re-selection | verdict |
|---|---|---|---|---|---|---|
| 0 | 16 | **1 / 24** (iters 10→20) | 0.9583 | **19 events** | 86.5 % | ARMED |
| 1 | 9 | 0 / 24 | 1.0000 | 21 events | 91.7 % | VOID |
| 2 | 12 | 0 / 24 | 1.0000 | 18 events | 90.6 % | VOID |
| 3 | 7 | 0 / 24 | 1.0000 | 21 events | 96.9 % | VOID |

Seed 0's ARM rests on **exactly one disjoint pair**, at iters 10→20,
immediately after the registered init transient, followed by 19 consecutive
identical events on `fc2 = [8, 23, 26, 30]`. That is one Bernoulli event,
not a 25 % rate of a real phenomenon. Scoring it would measure a
rotating-then-frozen four-unit lesion, at n = 1, through a selector this
project has documented as under-selecting by 20–40 iterations with recorded
honest@peak low by +0.08…+0.21 on 4 of 4 runs tested. **No Θ exists, none is
inferred, and none will be manufactured from one seed.**

### Action 3 — Run the corrected peak ladder on the four archived v27 runs

**This is the next unit of work.** 192 evaluations over
`checkpoints/mario_1_1_v27_recovery_seed{0..3}/`, which hold complete
24-checkpoint grids (verified: 24 / 24 / 24 / 24 `.pt` files on disk). It
is the identical job F0 already ran on v28 and receipted at 5,333 s.

Why this and not a capacity campaign: `runs/v29_stability/f0_ladder/ladder.csv`
is 97 lines — 4 runs × 24 iterations, **all four of them v28**. There is no
v27 ladder. The **+0.14** headline that the entire capacity pivot rests on
(v27 0.530 → v28 0.670) is corrected on **one arm only**. Both v27 points
that were ever spot-corrected moved **up** (seed 0 0.040 → 0.120, seed 1
0.290 → 0.500), by amounts comparable to v28's own median correction, and
**v27 seed 2 — the 0.530 that *is* the best-of-4 anchor — has never been
re-scored.** The sign of the delta is unmeasured.

### Action 4 — Register the selection rule as standing, before any arm launches

**Split-sample selection is the standing gate rule for this line from now
on:** select the checkpoint on one eval seed, score it on the held-out
other, over the fixed 10-iteration grid. It is what F0 used, it is what
lands v28 on 0.670, and the winner's curse it leaves behind is **measured
at 0.05**. Argmax over `entrance_trailing_rate` is retired as a selector:
that series is ceiling-saturated (0.867–1.000, SE ≈ 0.09 on a 30-episode
window), so the argmax over ~25 draws is near-arbitrary.

Also register, in the same commit, that **the 0.767 FAIL bar is a
shared-stream, single-eval-seed, one-worker figure (46/60)** — never
measured under the two-seed per-episode protocol it gates. **The registered
threshold does not move** — a moved goalpost is a fabricated result and
this is not an exception. But any verdict template citing it must carry
that sentence, because v28's 0.670 against a same-protocol control of 0.680
means the FAIL is a correct *threshold call*, not a measured deficit.

### What is NOT decided here

The capacity hypothesis is **not refuted and not adopted**. It is
*unfunded*. Action 3 is the price of finding out whether it is worth
funding. And the recommendation's item 4 — "if plasticity is ever tested,
use a mechanism that does not fight LayerNorm (periodic full trunk resets,
or L2-init-style pull)" — is **retargeted**: deferral is upheld, but that
prescription aims at the wrong cause. Full trunk resets and L2-init pull
would both be *selected and scored by the identical gain-dominated
statistic* and would hit the identical trap. If plasticity is ever
revisited, the untested variable is the **operator**: do not reset
`norm2.weight` to 1.0 (reset to the layer median gain), or do not zero the
head columns. Those are the same trap seen from two ends — zero columns →
zero gradient → gain pinned at 1.0.

The direct measurement, from seed 2's final checkpoint, gain and mean
outgoing-column magnitude as a function of how many cadence windows a unit
went un-recycled:

| free windows | LN gain | mean \|actor col\| | mean \|critic col\| |
|---|---|---|---|
| 0 (recycled this check) | **1.000** exactly, 4/4 units | **0.000** exactly | **0.000** exactly |
| 1 | 1.10 – 1.36 | 0.09 – 0.13 | 0.03 – 0.32 |
| 2 | 1.78 | 0.14 | 0.69 |
| 20 – 23 | 0.73 – 5.80 | 0.34 – 1.72 | 0.02 – 0.75 |
| never recycled | 1.61 – 8.75 (mean ≈ 4.9) | 0.13 – 1.90 | **3.35 – 7.14** |

Across all four seeds, all sixteen units recycled at the final check read
gain exactly 1.000 and actor column exactly 0.000. The critic column is the
slowest thing in the network to regrow, and it is the one the operator
zeroes.

---

## 2. What it costs and what it can conclude

### Cost

| item | cost | compute class |
|---|---|---|
| Action 1 (bank statement + two ledger corrections) | minutes | none — already computed, receipts banked with this document |
| Action 2 (decline to score seed 0) | **negative** — saves ~48 evals | none |
| Action 3 (v27 corrected peak ladder) | **192 evals, ≈ 1.5 h wall clock** (F0's identical v28 job: 5,333 s, 192/192 ok) | **eval only, zero training** |
| Action 4 (register selection rule + bar caveat) | minutes | none |

**Total: ~1.5 hours, eval-only.** Against **~21 hours of training already
sunk** on the five ReDo campaigns (v27 7.14 h + v28 7.37 h = 14.5 h
receipted; v30 ≈ 0.70 h; v31 ≈ 0.56 h; v32 05:26:20 → 10:46:02 = 5.33 h),
and against the ~7–14 h a capacity arm would cost. Action 3 is **under
one-tenth** the cost of the thing it gates, and it can kill that thing
outright.

### What Action 3 can conclude

**It can conclude:** the **sign and size of the v27 → v28 gate delta under
a selection-unbiased estimator**. Specifically —

* If v27's split-sample best-of-4 lands **≥ 0.670**, the +0.14 is dead. The
  capacity pivot is refuted for the price of eval compute, and no capacity
  campaign is registered.
* If it lands **near the recorded 0.530**, capacity gets its **first
  honestly-measured data point** on this line, and a capacity campaign
  becomes registrable under §4's numerals.
* Either outcome also settles whether v27's per-seed field is homogeneous.
  v28's is: recorded per-seed [0.45, 0.23, 0.37, 0.67] gives
  χ²(3) = **41.45**, p = 5.2 × 10⁻⁹; split-sample-corrected
  [0.64, 0.50, 0.58, 0.67] gives χ²(3) = **7.02**, p = **0.071** —
  statistically indistinguishable from four seeds at one value. That
  heterogeneity *is* the "higher across-seed variance, upside > downside"
  reading, and it *is* the raw material of the verdict's seed-level
  coherence argument. Corrected, seed 2 — the anchor, "the seed that
  regressed most at the gate and on both mechanism reads" — reads 0.58
  against seed 0's 0.64 and seed 1's 0.50. It is not the worst seed by any
  margin the data can see. **The coherence argument does not survive.**

**It cannot conclude:** whether capacity is the lever. Nothing at n = 4
seeds × 100 episodes can. Monte-Carlo power of the honest gate at n₁ = 100
vs the fixed n₂ = 60 bar (8,000 Fisher trials, reproduced independently in
closed form):

| Δ vs bar | power |
|---|---|
| 0.05 | **0.105** |
| 0.10 | **0.337** |
| 0.15 | 0.714 |
| **0.165** | **0.812** |
| 0.22 | 0.995 |

80 % power arrives at **Δ ≈ 0.165**, not 0.22 — the gate is somewhat less
blind than one opposition position claimed, and I record that correction
against my own argument. It is still essentially blind to the 0.05–0.10
range that every other line of evidence says is the plausible size of a
"real, partial lever". **A gate that cannot see a partial lever cannot
confirm one, and a null from it is not evidence of absence.** That
symmetry is why Action 3 is scoped as *instrument repair*, not as a test of
capacity.

---

## 3. What was rejected, and why

Four opposition positions were argued. Each is named below with the
argument that defeated it, or the concession it won. **Three of the four
won concessions that changed the decision.** One was defeated on the text.

### Position A — "Retirement is wrong as written: no registered licence exists for it"

**Concession WON, on the licence. Defeated on the substance.**

The position is **correct on the text**, and the text matters. §10.3 item 4
grants *"Retirement of the prescription, in both its forms"* to exactly one
outcome — a v32 **FAIL** (Θ ≤ 0.767). There is no Θ. §10.4 closes the door
explicitly: *"VOID is not FAIL, takes no branch of the fork, and enters no
best-of-N. A VOID campaign licenses ONLY the mechanism finding attached to
its specific void reason."* The second route, §8's pre-written stopping
statement, fires *"If the second Phase R also NO-GOes"* — and the second
Phase R **GOed** (re-selection 90.9 % → 75.0 %, distinct 4 → 14), which is
why 5.3 h of compute was launched at all. **Neither antecedent occurred.**
Lifting a pre-written conclusion out of its registered antecedent is the
goalpost rule pointed the conservative way, and it is the same defect.

This is why **Action 1 re-grounds the statement on §7 and §10.4** — which
license the recovery curve as a mechanism receipt *in every branch,
including STOP* — rather than on §8 or §10.3. The finding is banked under
the licence that actually exists, and it is a different, better-evidenced
finding than the one §8 pre-wrote.

**Where the position was defeated:** its §5 argument. §5 reads *"Fewer than
4 ARMED and scored inside the ceiling → VOID-UNDERPOWERED: no Θ issued,
per-seed numbers banked individually as mechanism receipts."* The position
read that as a pre-registered *instruction to produce evals*. It is a
**VOID-condition definition** — it tells you what verdict class you are in,
not that you owe 48 evaluations. And it does not change the verdict either
way, because only one seed ARMED. The mechanism receipt §7 and §10.4
actually name in every branch **is the recovery curve** — which is free,
was never computed for the campaign (both pilots have a
`recovery_curve.json`; `seeds/` had none until this document banked one),
and which at 384 observations is 8× larger than both pilots combined.

**Its turnover-gate critique is upheld and acted on.** F3'-as-coded
operationalizes §6.2's *"the recycled set never changes"* as "no consecutive
pair of k = 4 sets is disjoint" — which at k = 4 of 32 demands all four
units swap simultaneously in one step. The gate separates **4.2 % of pairs
from 0.0 %** and calls one campaign admissible and three inadmissible.
Ranked by real churn the order is seed 0 (0.135) > seed 2 (0.094) > seed 1
(0.083) > seed 3 (0.031); the gate admitted the top of that gradient by a
single Bernoulli event and voided seed 2, the only seed showing late escape
(events 22–24, iters 220–240). **This is logged as a vacuous gate.**

**Its strongest single contribution, and it argued against itself:** the
position was assigned to defend ReDo, went to the banked data, computed the
recovery curve nobody had computed, found 91.4 % re-selection, and reported
that finding as the strongest evidence *against* its own position. That
finding is the reason the stopping statement can be re-grounded rather than
abandoned. **That is exactly what this workflow exists to produce.**

**One claim of its own that it correctly rejected, and I reject too:**
"seed 0 IS armed, 1-of-4 is a rate not a failure." It is one Bernoulli
event at the earliest post-transient check.

### Position B — "The capacity pivot is wrong: v28 did not measure capacity"

**Concession WON. This is the argument that changed the decision.**

It is not a philosophical objection to a null. It is a specific, checkable,
currently-unmeasured number with the measuring job already scoped, costed
and receipted. Verified in full:

* `ladder.csv` = 97 lines, 4 runs × 24 iters, **all v28**. No v27 ladder
  exists.
* Both v27 seeds ever spot-corrected moved **up**, by +0.08 and +0.21.
* **v27 seed 2 — the 0.530 anchor of the +0.14 — has never been corrected.**
* All four v27 24-checkpoint grids are **on disk**, at the same ~1.5 h eval
  cost F0 already paid for v28.
* The seed-level coherence claim dies under correction: χ² 41.45 → 7.02,
  and the anchor seed corrects to 0.58 against 0.64 and 0.50.
* The registered next dose is a **decrease**: Candidate 2 (hidden 64→80 +
  trunk 32→40) = **60,824 params**, against the **72,039** that just failed.
  "Pivot to capacity" is not currently a specified experiment.
* `rungs_per_100_iters = initial_tau / iters_to_entrance × 100` with
  `initial_tau = 744` in all eight runs — a deterministic monotone transform.
  The registered conjunction reads as two confirmations of ladder behaviour;
  it is algebraically one, and it is **null** (deltas −4, −3, +5, −1; paired
  p = 0.735; the 3/4 sign pattern has one-sided p = 0.3125).
* The strict/lenient combination rule that produced 3/4 rather than 1/4 was
  written **after** compute, and the verdict itself concedes it flips the
  row.

**Where it was corrected:** its "80 % power only against a 0.22 difference"
does not reproduce. Independent closed-form and 8,000-trial Fisher Monte
Carlo both land the 80 % point at **Δ ≈ 0.165**. The gate is less blind
than claimed. It is still blind to the range that matters, so the
conclusion stands and the number does not.

**Where it overreached, and I decline to follow it:** its Step 2 proposes
the 1-2 width-vs-recipe disambiguation (96k vs 200k, recipe held fixed) as
the substantive target. That is a *larger* capacity experiment with a live
positive attached and it is genuinely more interesting than a 1-1 width
step — but it is a **new campaign**, and the position's own weakest-point
section concedes it is arguing about sequencing and dose rather than about
the hypothesis. **Nothing is launched here.** It goes on the queue behind
Action 3, which is the thing that determines whether a capacity campaign of
any size is worth registering.

### Position C — "The learning track is not where the next unit of work belongs"

**Refused its own brief, and won two concessions doing so.**

The reviewer assigned this position read the artifacts and declined to
argue it, on the grounds that the learning track is the only track that can
produce a claimable result at all — solver output is EXHIBITION and by
`CLAIMS.md`'s two-ledger policy may never be described as learning. That
refusal is correct and is accepted.

**Concession 1 — the brief handed to this review quoted withdrawn numbers.**
It stated *"1-2 banked at 2/100"* and *"the policy class was falsified for
1-2 by a pre-registered, externally-reviewed protocol."* This repository's
own learning-track audit of 2026-08-27 marks the first **SUPERSEDED BY THIS
REPO'S OWN LEDGER** and the second **WITHDRAWN AS SCOPED**. `MISTAKES.md`
already carries an entry dated 2026-08-26 titled *"Briefed a campaign from
withdrawn numbers."* **The review workflow built to catch propagation was
itself briefed from the propagation, one day after the ledger superseded
it.** Any document that cites 1-2 must cite the ledger, not the brief.

**Concession 2 — the routing rule that was ignored.** The recovery assay of
2026-08-24 banked a rule stated as binding: *"run this assay before
spending training effort on any level's sticky rate."* It measured 1-1's
honest ceiling at ~0.83–0.85 and 1-2's at ~0.53. From a correctly-measured
control the entire maximum prize of the v27–v32 program plus a capacity
successor is roughly **+0.15 absolute, on one level, on an off-ledger
artifact**. That is what ~21 hours bought a search for. **This is the
single best argument in the dossier for why Action 3 must be eval-only.**

**Where its own strongest proposal was declined:** re-running 2-1. It is
genuinely the only unrun experiment whose outcome changes strategy — the
sole evidence about whether the pipeline crosses a *world* boundary, with
attempt 2 reading `clear_rate_strict` 0.0 at every probe over 96,675,840
env steps. It is on the queue. It is **not** this decision, because it is a
training campaign and this decision launches none, and because launching it
through the unrepaired selector reproduces the failure shape of the last
five. **Register it after Action 3 and Action 4, with the split-sample rule
in force.**

**Where it was defeated:** its chain arithmetic ("32 levels at 50 % needs
97.86 % per level") attacks a headline `CLAIMS.md` no longer makes — the
ledger already states *"per-level rates are the scoreboard"*. Its own
weakest-point section concedes this.

### Position D — the process auditor: "the apparatus is a net cost"

**Concession WON, and it binds regardless of target.** Verified directly:

* `scripts/go_explore_solve.py:2780` is still
  `if self.level_key(ram) > tuple(start_key):`, and **46 of 47 solve
  configs declare `level_key: []`**. `() > ()` is `False` — a compile-time
  constant, not a search outcome.
* **`make clear-lint` prints `NONE=37 REACHABLE=8` and exits 0.** It is
  wired into `make test`. It *reports* the defect rather than failing on it.
* `scripts/mistakes_tally.py` is not referenced anywhere in the `Makefile`.
  Its `stated` regex requires `\*\*(\d+)\*\*`, so the six sub-threshold rows
  (purity-leak 3, start-state 2, false-alarm 1, measurement 1, git 1,
  reward-exploit 1) are invisible to its drift check.
* `MISTAKES.md` stood at **54 entries** when this review opened, with
  `[vacuous-gate]` at 12, `[unverified-claim]` at 8, `[weak-eval]` at 7,
  and `[stale-artifact]`, `[process]`, `[inert-treatment]` at 6 each —
  **six categories at or past the 4-entry promotion threshold, with zero
  promotions made.** The file says so itself: *"Nothing has been promoted;
  the enforced ruleset is untouched."* This review adds three entries (§5),
  taking it to **57**, `[vacuous-gate]` to 13, `[weak-eval]` to 8 and
  `[process]` to 7. **Zero promotions, still.**

**The decision this compels:** a next campaign that repeats the last five's
failure shape is not worth launching whatever its target. See §5.

---

## 4. The pre-registration this decision owes

**Action 3 is not a campaign** — it is an eval-only re-scoring of archived
checkpoints under a rule that already exists and has already been executed
once on the adjacent arm. It does not owe a fresh registration; it owes the
numerals below written down before it runs, and those numerals are fixed as
of this commit.

### 4.1 Numerals fixed before Action 3 runs

| numeral | value |
|---|---|
| runs scored | `mario_1_1_v27_recovery_seed{0,1,2,3}` |
| checkpoint grid | every archived `vanilla_ppo_iter_*.pt`, 24 per run, **96 total** |
| evals | 2 eval seeds × 96 checkpoints = **192** |
| episodes per eval | **100**, canonical per-episode protocol, sticky 0.25, jitter uniform 0..16 |
| eval seeds | the same two F0 used on v28 — no new seed is introduced |
| selection rule | **split-sample**: argmax on eval seed A, report the held-out eval seed B score at that iteration; then the reciprocal; average the two |
| tie-break | later iteration wins, as F0 did — fixed here, before the numbers |
| statistic reported | v27 split-sample **best-of-4**, per-seed values, and χ²(3) homogeneity |
| comparison | against v28's split-sample best-of-4 = **0.670** (already banked) |
| training compute permitted | **zero** |

### 4.2 The fork, written before the numbers

* **v27 corrected best-of-4 ≥ 0.670** → the +0.14 is dead. **The capacity
  pivot is refuted on its own quantitative claim.** No capacity campaign is
  registered. Bank the refutation; the next work is Position C's 2-1 re-run
  or the 1-2 width-vs-recipe disambiguation, whichever the owner picks.
* **v27 corrected best-of-4 ≤ 0.570** (0.10 below v28's 0.670, the smallest
  gap this gate has 33 % power against) → capacity has its first
  honestly-measured data point. **A capacity campaign becomes registrable**,
  and must satisfy §4.3 before it launches.
* **(0.570, 0.670)** → **INDETERMINATE**, and it is reported as
  indeterminate. Not "trending", not "directional". The gate has under 33 %
  power in that band and a number inside it means the instrument cannot
  separate the arms. No capacity campaign is registered on an indeterminate.

**Nothing in §4.2 may be reinterpreted after the numbers exist.**

### 4.3 What a capacity campaign would owe, if §4.2 licenses one

This project has now had **five campaigns die at preflights and
adjudication defects**: v27 and v28 with ReDo inert (`redo_tau: 0.025`
against a mechanism whose own sweep first fires at 0.25; `recycled 0 cum 0`
on ~2,000 checks across all 8 runs, 14.5 h), v30 VOID at a preflight
(~42 min), v31 VOID at its own preflight (~34 min), v32 VOID-UNDERPOWERED
after 5.3 h on a turnover gate that separates 4.2 % from 0.0 %. **No future
campaign on this line launches without all seven of the following written
down before compute:**

1. **The construct-validity preflight (new, and it is the generalization of
   what went wrong five times).** A written demonstration, computed
   offline, that **the selection statistic is not a deterministic readout of
   a quantity the treatment itself sets.** On this trunk it is: the
   dormancy score correlates +0.77…+0.94 with the LayerNorm gain, and the
   recycle operator sets that gain to exactly 1.0. **That check costs
   minutes and would have killed v30, v31 and v32 before any of them
   registered.** No registration is admissible without it.
2. **The dose, and the proof the treatment is reachable at it.** The
   arithmetic that `redo_tau: 0.025` was unreachable existed before v27 and
   was not run. State the operating point and demonstrate, on a banked
   artifact, that the mechanism fires there.
3. **The exact bar, with its provenance sentence.** 0.767 is 46/60 at eval
   seed 0, shared-stream, one worker. It does not move. It is quoted with
   that sentence attached or it is not quoted.
4. **The power statement.** State the Δ the design can detect at 80 %, and
   state it against the Δ the evidence predicts. If the design cannot see
   the predicted effect, **say so in the registration** and either fix n or
   do not run. At n₁ = 100 / n₂ = 60 that number is **Δ ≈ 0.165**.
5. **The selection rule.** Split-sample, per §4.1. Named before compute,
   with the tie-break fixed.
6. **The combination rule for any multi-sub-metric read**, including the
   strict/lenient tie-break, **written before the parser is.** v28's was
   written at 22:07 on 2026-08-25, after the compute, and it flips the
   verdict row.
7. **A single-sentence statement of what a VOID licenses**, so that no
   later document lifts a pre-written conclusion out of an antecedent that
   never fired. §10.4 is the model; the failure was applying it.

---

## 5. The apparatus is a net cost, and that is part of the decision

The process auditor's finding is upheld and is **binding on the next
campaign, whatever its target**. Three repairs, all cheap, all before any
training arm launches:

1. **`make clear-lint` must exit non-zero.** It currently prints
   `NONE=37 REACHABLE=8` and returns 0 inside `make test`. A gate that
   reports a defect it was built to refuse is a vacuous gate, and
   `[vacuous-gate]` now stands at 13 entries.
2. **`scripts/mistakes_tally.py` must be wired into `make test`,** and its
   `stated` regex must drop the `\*\*` requirement so the six sub-threshold
   rows are visible to its drift check. The graduation table drifted on all
   six categories it listed while it was hand-maintained; a derived table
   nothing runs is hand-maintained again by another name.
3. **The promotion path must be exercised or retired.** Six categories sit
   at or past the 4-entry threshold with zero promotions across 57 entries.
   The four root causes that *did* ship enforcement shipped it as code
   (`purity-check`, `check_mechanism_receipt.py`, `redo_arm_gate.py`,
   `assert_bank_wellformed`) without waiting for a vote. **That is the
   evidence that prose promotion was never the operative mechanism.**
   Promote by shipping the check, and delete the "awaiting a call" column.

The first repair the next campaign owes is §4.3 item 1 — the
construct-validity preflight. It is the one that generalizes all five
failures, and it is the one this review found only because it went to the
checkpoints instead of the write-ups.

---

## 6. Receipts

| finding | file |
|---|---|
| campaign recovery curve, 384 observations, per-lag survival | `docs/receipts/v32_redo_bottom_k/campaign_recovery_curve.json` (working copy at `runs/v32_redo_bottom_k_2026-08-28/seeds/recovery_curve.json`; `runs/` is gitignored) |
| LN gain vs dormancy score; gain/column regrowth by free window | `docs/receipts/v32_redo_bottom_k/gain_vs_dormancy_score.json` |
| per-seed turnover, raw | `docs/receipts/v32_redo_bottom_k/seed{0..3}_armgate.json` |
| the recycle operator | `src/training/redo.py`, `recycle()` |
| v28 corrected peak ladder (the job Action 3 repeats on v27) | `runs/v29_stability/f0_ladder/ladder.csv`, `docs/research/F0_CORRECTED_PEAK_LADDER_2026-08-26.md` |
| v27 checkpoint grids, 24 each, on disk | `checkpoints/mario_1_1_v27_recovery_seed{0..3}/` |
| v32 registration §5, §6.2, §7, §8, §10.3, §10.4 | `docs/proposals/V32_REDO_BOTTOM_K_2026-08-28.md` |
| v30's refuted LayerNorm premise | `docs/proposals/V30_REDO_ARMED_2026-08-27.md:48-50` |
| the clear-predicate defect | `scripts/go_explore_solve.py:2780`; `make clear-lint` |

**Nothing in v27, v28, v30, v31 or v32 may be cited as evidence for or
against the plasticity-loss hypothesis.** That standing prohibition is
unchanged. What this document adds is that the five campaigns searched the
delivery schedule and never moved the operator — which is a finding about
*what was tested*, not about whether plasticity loss is real.
