# The Contra gx-3072 lock, campaign 2 — the two open branches, closed

**Date:** 2026-08-27
**Ledger: EXHIBITION, without exception.** Every number below is Go-Explore
search output or instrument measurement. No policy was trained for this game
and no honest-protocol evaluation was run. Nothing here may be described with
"the AI learned", "the AI plays", or "the AI beat" — see `CLAIMS.md`.
**Predecessor:** `docs/research/CONTRA_WALL_2026-08-27.md` (commit `1954037`),
which characterised the lock and falsified the boundary-resident attack family
from one shared root. This document does not re-derive any of it.
**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc`.
**ROM** `roms/Contra (USA).nes` sha256
`26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519`.
**Start state** `roms/Contra (USA)_start.state.bin` sha256
`b99f9be8e0266f6dbe8ac71bc591b0deec08e66e7925707d265965a4aab922c3`.

---

## 1. The answer, plainly

**Did the lock move? No.** Best verified gx is **3072**, against a prior of
**3072**. Six archives were scanned cell by cell for this write-up — the
ancestor plus all five archives this campaign produced — and **not one holds a
cell above gx bucket 192**. No tape exceeds 3072, so `docs/receipts/` gains no
Contra tape and no extraordinary claim is made.

**What is now closed that was not before.** The predecessor campaign named two
branches it had explicitly left untested. Both are now answered, and **both are
closed** — one of them by a third outcome its own decision rule did not
anticipate.

| Branch | Question it asked | Verdict |
|---|---|---|
| **Route A — approach** | Every prior attack reached the lock one way, from one lineage rooted in one savestate. Does a different approach arrive in a different, more attackable configuration? | **CLOSED — and the answer is a stronger negative than either outcome the brief anticipated.** Approach *does* change the arrival configuration, reproducibly and measurably. The configuration is **inert**. |
| **Route B — the objective** | Selection scored every one of the 1,182–1,331 distinct wall states identically, so the archive could tell them apart but could not prefer any of them. Fix the objective and the search concentrates. | **CLOSED as a diagnosis.** The defect was real, is now fixed, and **selection demonstrably discriminates inside the lock** — under offline shuffle control, and again live in the runs' own receipts. The camera still did not move. |

**And that second result is the one worth having.** The brief said so before the
work started, and the measurement bore it out: *if a shaped objective makes
selection genuinely prefer among the wall states and the camera still does not
move, that is evidence the lock is a game gate rather than a search artifact.*
It did, and it did not. See §6, including what that argument does **not**
establish.

---

## 2. Route A — approach changes the state, and the state changes nothing

Six approaches were run: four fresh from-power-on searches (`A_rush`,
`B_coverage`, `C_cautious`, `D_macro`), each with its own reset-booted mint,
action prior, archive-selection rule and seed, plus two derived life-count
variants; then a seventh, independently-designed falsifier (`A-fals`) built to
stress Route A's own stated limits. They were compared against two reference
populations from the prior lineage (`LEGACY_solve20` n=61, `PRIOR8_headwall`
n=8). **429 wall states** were collected in total.

### Approach really does determine the configuration

This is not a null result about the approach axis — the axis is live, and
strongly so.

- **`$00AA` is constant inside each arm and differs between arms** (`A_rush` 16,
  `B_coverage` 19, `C_cautious` 16, `D_macro` 0), while the entire prior lineage
  carries only `{0, 4}` across 61 verified legacy cells and `{0}` across the 8
  head-wall states. **180 fresh arrival states sit at a value present in zero of
  the 69 reference states all eight prior attacks were measured on.** (No claim
  is made about what any value means.)
- **Steps-to-arrive bands do not overlap:** B 772–816 < A 848–862 < D 941–1050 <
  C 1371–1520.
- **Irreversible capacity differs by 2×:** `B_coverage` reaches 255 agent-latched
  bytes (median 112) against `A_rush`'s 122 (median 57) — both above the
  characterisation's stated 20–74 per state.
- **Eight RAM bytes are constant within every arm and differ across arms.**
- Statistically: nearest-neighbour arm purity **0.920** against a 0.138 chance
  baseline; permutation statistic **+133.6** (within-arm 364.3, between-arm
  497.9), p = 0.0005. **Controls both directions:** a sham half/half split of a
  single arm scores **−3.2**, p = 0.86; an open-field-vs-wall positive control
  scores **+257.3**.
- Independence verified directly for this write-up: the six mint states restore
  to six distinct RAM images differing pairwise by **165–407 bytes**, with
  distinct file hashes, distinct seeds, distinct burst lengths, distinct priors
  and distinct selection rules. No shared seed, no shared ancestor savestate.

### And it buys nothing

- **The camera pair is `{(12, 0)}` across all 238 input-live states × 14 held
  masks, every arm, no exception** — while the identical probe reports 20–63
  camera pairs on open-field controls.
- **`px_max = 136` exactly** in all 8 populations under 14 masks × 32 steps.
  At the arrival instant the fresh approaches land *below* it (25–132); under
  held input they converge on 136 and stop.
- **A matched pilot gave the prior campaign's own single-lineage roots the
  LARGEST distinct-cell count of the four** (LEGACY 20,436 vs C 19,988,
  A 15,160, B 12,475), with everything else held fixed — action prior, burst
  length, resume rule, cell key, budget — and only the root population varying.
  Fresh configurations open *less* search surface, not more.
- **`beat_3072` is false across 2,417,912 worker-steps of fresh from-power-on
  search** and ~1.25 M alive steps resident at the lock.
- The independent 7th approach (`A-fals`) reproduces it cleanly: a genuinely
  fresh mint (entry frame 717, lives 2), three workers at burst lengths
  {80, 96, 112} and seeds {70001, 70002, 70003}, a prior alternating long
  right-run macros with long vertical-fire macros, and a UCB1-flavoured
  selection rule distinct in kind from all four Route A rules. 900 s,
  **588,600 worker-steps, 573,950 alive steps, `max_gx_alive` exactly 3072 on
  every worker**, 214 wall cells, nothing above.

### The by-product Route A should be remembered for

**45% of wall arrivals are input-DEAD** — 238 of 429 screen LIVE, i.e. roughly
five times the corpse rate the prior campaign's hand-picked 2-of-8 sample
implied. Per population: `C_cautious` 0.73, `B_coverage` 0.67,
`PRIOR8_headwall` 0.62, `LEGACY_solve20` 0.52, `D_macro` 0.50, `A_rush` 0.47.

The screen validated three ways on one code path: a **dead control** (900
forward steps past the final death) 0/3 LIVE at 2 input-dependent RAM bytes and
0 OAM; an **open-field control** 6/8 LIVE at median 442/120; and — the strongest,
because the labels predate the screen — it independently flags exactly
`head_wall_0` and `head_wall_1`, the same two the characterisation identified by
hand, plus `head_wall_3`, an unresponsive state the prior labelling did not
separate. LIVE-vs-DEAD separation is two orders of magnitude: 475–588
input-dependent bytes against 2–5.

**Never gate liveness on `lives == entry_lives` here.** It failed twice in this
campaign alone: holding flat past the final death, and reading a healthy 2
throughout the px-144 dead windows below.

### The one anomaly, chased down rather than published

`B_coverage` roots reach px 137/138/142/144 — above the supposedly universal
136 — replicated across four seeds, cold-restoring at progress exactly 3072 with
lives 2. **156 of 156 such witnesses screen input-DEAD** (140
`DEAD_UNRESPONSIVE`, 16 `DEAD_FATAL_WINDOW`; input-dependent RAM 0–4, OAM 0 —
the dead control's own signature), with median-alive counting down monotonically
across consecutive witnesses: one already-committed commitment, sampled a step
further along each time. **Corrected statement: `px_max = 136` holds for every
input-LIVE state in every approach; what is approach-dependent is which
already-committed dead windows a population can fall into.**

> **Route A's verdict.** A real, reproducible, statistically strong
> approach-dependent configuration space exists at the lock — and it is inert on
> every axis that matters. That is a third branch the original decision rule did
> not anticipate ("the states differ, so fan out" / "the states don't differ, so
> the family really is falsified"), and it is a stronger negative than either.
> **Do not fan out on approach.**

---

## 3. Route B — the defect was real, and it is fixed

### What the defect actually was (the premise, corrected)

The characterisation named it as "`max_gx_in_max_area` is frozen by construction
at 3072, so deeper carries no signal there." Traced end to end before any code
was written, the mechanism is sharper than that phrasing and **the brief's
premise needed correcting, not confirming**:

- Score is minted as `sect * 10000 + gx + score_bonus(ram)`. `sect` is 0 for
  every cell of the whole lineage; `score_bonus` is the typed-boss-HP term.
- **Score at the wall is not flat.** It takes exactly **14 values, 3072 through
  35072**, across the wall cells — verified again for this write-up on all six
  archives, where the identical 14-value spectrum appears in every one.
- But `best_score = 3072 + (16 − hp) * 2000` **where `hp` is itself a slot of the
  cell key**. Independently re-derived here: the identity holds for
  **100% of wall cells in all six archives**. So the score carries *zero*
  information beyond the key, and the one axis it does express is the typed-HP
  term the characterisation had already established is a transient multiplexing
  artifact — and `best_score` ratchets, so a cell that once caught a flicker
  keeps the elevated score forever.
- Consequence for the count arm: a 5.9× weight range across wall cells,
  expressed entirely on an artifact axis.
- **Consequence for the deep-frontier arm, which does 40% of the picking at
  default `--deep-bias 0.4`: it reads no score at all.** It sampled the
  2,079-cell top band **uniformly**.

That last point is why "just add a term to the score" would have produced a
**false abort**: a lexicographically safe merit in [0,1) moves the count-arm
weight by 3 × 10⁻⁵ and is invisible to the deep arm. Route B therefore had to be
two coupled edits under one flag — a merit comparator in the **domination test**,
and a merit read in **both selection arms**.

### Why enriching the key had already failed but enriching the score might not

The cell key controls **resolution** — what counts as a distinct state, hence how
a fixed budget is *partitioned*. The score controls **preference** — which
partition gets the budget. Attack 6 raised resolution and proved the extra axes
live (11,669 augmented cells against 1,331, 8.8×), but under an indifferent
preference more resolution can only **dilute**: every new cell landed in the same
uniform top band, so the per-cell budget fell ~8.8× at exactly the boundary that
needed concentration. Key enrichment is structurally a *spreading* operator. The
score is the only mechanism in this solver that can *concentrate*, and it had
never been changed.

### The four objectives

All four are lexicographic `(gx, then merit)` with merit constrained to [0,1),
so no merit can ever reorder two states of different gx. Merit lives in a
solver-side map, never in `Cell.best_score`, so `_sel_maxscore` — the normaliser
every non-lock cell's count-arm weight divides by — is untouched.

| | Objective | Merit | Live result |
|---|---|---|---|
| **B-O4** | **LEX-YIELD** | Laplace-smoothed generativity of the cell as a burst root, `(yields+1)/(bursts+2)`, read off bookkeeping `_assign()` already computed and threw away | 3 workers, 557 s (≈212 s armed), fresh from power-on, seed 44. 11,928 cells, 1,747 at the wall. `max_gx` 3072 every line. |
| **B-O1** | **LEX-SURVIVAL** | `1 − 1/(1 + n/64)`, n = consecutive alive-in-lock steps the lineage reached | 3 workers, 840 s (≈810 s armed at `--lock-pin-secs 30`), resumed `solve20`, seed 12345. 30,614 cells, 1,393 at the wall. `max_gx` 3072 on all 14 lines. |
| **B-O2** | **LEX-LATCH** | Control-differenced agent-latch count, with a **paired NOOP continuation from the cell's own blob on a private probe emulator** so a corpse cannot score | Branch `contra-lock-b-o2` (`ec21587`), not merged. 28,542 cells, 1,421 at the wall. `max_gx` 3072. |
| **B-O3** | **LEX-NOVELTY** | Count-based novelty of an entry-differenced RAM descriptor held **outside** the cell key, so cell cardinality stays A/B-comparable with all nine prior campaigns | 3 workers, 1,629 s (≈1,290 s armed), fresh from power-on, seed 83011. 21,219 cells, **3,393 at the wall** (2.5× solve20's). `max_gx` 3072 on all 27 lines. |

`solutions: 0` in every run, and it is **evidence of nothing** on this profile —
`level_key: []` makes `is_clear` reduce to `() > ()`, and the shipped 2-of-2
clear vote is a 1-of-1 `coord` vote at a measured null fire-rate of 1.00. It is
recorded and not interpreted.

**The torn-read filter cannot be hiding an advance.** `progress_glitches: 0` on
every line of every run, so the hampel screen never dropped a progress sample.
The approach buckets 187–191 hold contiguous raw gx 2996–3070 before 3072, so
3072 remains ordinary carry from 3071.

---

## 4. Did selection actually discriminate? Yes — three independent ways

This is the load-bearing question. A merit that changes nothing would mean the
defect was misdiagnosed and the branch aborts.

**(i) Offline, against the real banked archive, with a shuffle control.** Each
objective drove 20,000–30,000 real `select()` calls through the actual selection
code path over the real 1,331 wall cells of
`runs/play_one_well/contra/solve20/archive.pkl`, against two pre-registered
thresholds: total-variation distance from the arm-off pick distribution ≥ 0.20,
and top-merit-decile cells drawing ≥ 3× the bottom decile. Measured: TV 0.15–0.25
and decile ratios **2.0× to 8.4×** depending on objective and operating point.
**Paired shuffle control** — same merit *values*, reassigned across the same
keys, judged against the fixed real-merit ordering — collapses the decile ratio
to **0.79–1.0×** every time. A TV-only check would have missed this: shuffling
preserves the weight multiset, so `tv_shuffled` lands within a hair of
`tv_real` (0.232 vs 0.245 in B-O2's run). **The decile ratio against a fixed
ordering is the statistic that separates preference from re-weighted noise.**

**(ii) Distributionally, at the live operating point.** 200,000 real `select()`
calls on the 16,298-cell archive, count arm, `--lock-weight 4.0`: within the
lock, correlation between pick frequency and merit **0.688** with a 2.65×
top/bottom-decile ratio, and the lock's share of all picks rises **0.097 →
0.181**. Outside the lock the pick distribution moves **TV 0.0123 against a
0.0111 seed-to-seed noise floor** — indistinguishable from noise.

**(iii) Receipt-level, in the runs' own archives — the strongest line, and
re-derived independently for this landing** (`runs/contra_lock2/_LANDING/
dom_check.py`). B-O1 and B-O2 both resumed `solve20`, and under the *pre-change*
rule an equal-score cell can only ever ratchet toward **fewer** steps. Over the
16,298 cells each shares with the ancestor, their archives hold **1,176 and
1,118 equal-score replacements carrying MORE steps** — and **100% of them
(1,176/1,176 and 1,118/1,118) sit at gx bucket 192, with ZERO outside it.** The
new comparator provably ran live, and provably touched nothing outside the lock.

> **The statistic is only defined for a RESUMED archive**, and the receipt says
> so in its own docstring. B-O3 booted fresh, so a key it happens to share with
> `solve20` was reached independently and its step count was never compared
> against `solve20`'s by any domination test; its count is coincidence, not
> replacement. It is printed in the receipt only so nobody re-derives the number
> later and misreads it as a leak.

> **The abort criterion did not fire.** Selection can now prefer one wall state
> over another, proved under control. The named defect was real and is fixed.

### The inertness question, which is the one that could have voided everything

Six vacuous gates have shipped on this codebase, the most recent written by work
holding the previous five in its brief. So the guard was mutated rather than
inspected, in a throwaway worktree, by reverting it in source and watching the
tests fail:

- Forcing `in_lock_key` to always-True → **3 tests fail**.
- Dropping the `_in_lock` check in **both selection arms** — the exact "not inert
  outside the lock" defect → **4 tests fail**, including the leak probe.
- Removing `observe()`'s mode scoping → **1 test fails**.
- On B-O2's branch, the same mutation → **6 tests fail**.
- Forcing `lock_armed` to always-True fails only its own detector, **correctly**,
  because off-safety rests on the `lock_mode == "off"` string compare rather than
  the pin timer. That was disclosed by the work that found it rather than papered
  over, and it is the honest description of the contract.

Plus the byte-identity floor: with the arm off or the attributes absent, both
selection arms draw the identical RNG sequence and land on the identical cells
over 400–500 picks, and the observe-side burst loop produces identical picks,
identical record counts and identical final RNG state.

**One exactness caveat, stated because it is real:** an *armed* run at merit 0.0
everywhere is not byte-identical to off. Exact rejection sampling requires a
data-independent ceiling `Wmax = 1 + lock_weight`, so acceptance per attempt
drops below 1 and RNG consumption necessarily diverges the moment the arm is
armed — the same property the pre-existing ortho arm already has. What is
asserted, and tested, is that the **outcome distribution** stays uniform
(TV < 0.08).

---

## 5. What this campaign newly rules out

Cumulative with the predecessor, and stated only as far as the measurements go.

1. **Approach-family:** reaching the lock by a materially different route does
   not release it. Seven independent approaches (four fresh from-power-on
   searches with distinct mints/priors/selection rules/seeds, two derived
   life-count variants, one independently-designed 7th falsifier), 429 collected
   wall states, at least 2.4 M worker-steps of fresh search (Route A's own
   2,417,912, plus the 7th approach's 588,600), ~1.25 M alive steps resident
   at the lock. Camera `{(12,0)}` and `px_max` 136 in every input-live state of
   every arm.
2. **Approach-dependent configuration is live and inert.** Arms are separable at
   0.920 NN purity (chance 0.138, p = 0.0005, both controls behaving), and the
   separation predicts nothing about the camera. Fresh configurations also
   *shrink* the reachable search surface relative to the legacy lineage
   (20,436 → 12,475–19,988 cells under a matched pilot).
3. **Objective-indifference is not the blocker.** Four different lexicographic
   merits — generativity, survival, control-differenced latch progress, and
   descriptor novelty — each make selection measurably prefer among the wall
   states under shuffle control, and none moves the camera.
4. **The specific diagnosed defect is closed.** `best_score` at the wall being a
   deterministic function of a key slot, expressed on a transient artifact axis,
   with the 40%-of-picks deep arm reading no score at all: real, measured, fixed,
   and shown fixed live in the archives' own domination records.
5. **`solutions: 0` remains VOID on this profile** and enters no denominator.

---

## 6. Is the lock a game gate or a search artifact?

**Verdict: a game gate — strongly indicated, not proven.** The brief set this up
correctly and the evidence went the way it predicted.

**The argument.** A search artifact is a failure of the *searcher*: the states
that would escape exist and are reachable, but the algorithm cannot find,
represent, or prefer them. Three distinct forms of that hypothesis have now each
been tested and failed:

- *Cannot represent them* — falsified by the predecessor four ways (no cell
  collapse; invariant to 2× key resolution; invariant across four key
  compositions; the one run whose key genuinely collapsed walled **lower**, at
  2816).
- *Cannot reach them* — falsified by Route A. Seven approaches arrive in
  provably different configurations, and every one arrives at the same wall.
- *Cannot prefer them* — falsified by Route B, which is the branch that had never
  been run. Selection now discriminates among the 1,182–1,331 distinct wall
  states, on four different axes, and the camera does not move.

When all three failure modes of the searcher are excluded and the boundary is
still exactly one value — `$0064 = 12`, `$0065 = 0`, `px_max = 136`, reached by
ordinary carry, absorbing, not a timer, with the player alive and demonstrably
in control inside it — the remaining explanation is that **the game is not
offering the transition under the conditions the search can create.**

**What this does not establish, stated as plainly.**

- The runs were short: ~212 s, ~810 s, ~1,290 s of genuinely armed time. The
  hypothesis "the objective is the right lever but needs an order of magnitude
  more armed compute to concentrate on a long-surviving lineage" is **not
  excluded**, only unfunded.
- Four merits is not all merits. An axis none of {generativity, survival,
  control-differenced latch, descriptor novelty} expresses could still exist.
- "Game gate" here means *the transition is not offered to this search under
  these conditions*. It is **not** a claim about what the game contains, which
  would breach Tier-3 purity, and no such claim is made anywhere in this
  document.
- The predecessor's own residue stands untouched: a release gated on a specific
  sequence outside the searched latch space, a condition on an address outside
  the 2 KB CPU RAM window every probe read, or a condition tied to a specific
  entity a constant-hold protocol never targeted. None is cheap and none has a
  candidate.

---

## 7. Defects found and fixed in this landing

### 7.1 `--lock-objective latch` was a dead choice that read as armed — FIXED

On `main`, `latch` was a declared argparse choice with **no dispatch branch
anywhere in the solver**. It parsed, the progress line printed `lock_mode:
latch` with a non-zero `lock_cells`, and it changed not one draw — measured at
3,000 selections byte-identical to `off`. An operator would have read a null as
"the objective did not help" when in fact nothing ran. **That is the seventh
instance of the vacuity failure this codebase has shipped**, and it shipped
inside the very campaign whose brief holds the previous six.

Cause: B-O2's LEX-LATCH implementation lives only on the unmerged branch
`contra-lock-b-o2`, while the shared flag family that landed on `main` carried
its *name*.

Fixed by removing `latch` from the choices, so the CLI now rejects it outright,
and by adding **`tests/test_lock_objective_roster.py`** — a behavioural roster
guard, not a grep:

- **P1 (selection):** an armed mode's pick sequence must differ from `off`'s. A
  name with no branch takes the legacy line and consumes identical RNG.
- **P2 (observation):** an armed burst loop must leave a merit footprint `off`
  does not — `archive._merit` for the per-observation modes, `_lock_bursts` for
  the per-key one.
- **The non-vacuity direction, in code:**
  `test_the_roster_probe_reports_a_fabricated_name_inert` runs both probes
  against a name that deliberately does not exist and **requires them to come
  back inert**. If a fake name ever passes, the file is decoration and says so.
- **Mutation-verified before being trusted, three ways**, each run in a
  throwaway worktree and then restored:
  1. Re-adding `latch` to the choices tuple → the roster test fails with the
     exact diagnosis, `selection_differs: False, observation_differs: False`.
  2. A **half-wired** mode — `novelty` left in `select()` but its `observe()`
     branch disabled — → fails on the P2 half alone, which is the case a
     selection-only probe would have passed.
  3. Reverting the lock guard itself (`in_lock_key` forced True) → 3 tests fail
     across the two sibling inertness files, independently reproducing the
     adjudicated result rather than citing it.

If `contra-lock-b-o2` merges, the name returns **with** its dispatch and the same
roster test is what proves it live.

### 7.2 `lock_armed_secs` was not armed seconds — FIXED

The field was `round(now − _pin_time)`: time since the **frontier** last moved,
which begins accruing `--lock-pin-secs` *before* the objective steers anything.
Two reports read it as armed time and overstated their runs by 2.4–2.5×
(B-O3's "26.5 min armed" is ~21.5; B-O4's "512 s armed" is ~212). The ortho arm
already reports the same quantity under the honest name `pinned_secs`.

Fixed with a pure `lock_clocks(pin_time, now, pin_secs)` returning **two**
clocks — `lock_pinned_secs` (frontier) and `lock_armed_secs` (actually steering,
floored at 0) — cross-checked in test against `lock_armed()`, the predicate the
arms genuinely gate on, so the two must agree or the test fails. Mutation-
verified: restoring the old conflation fails two tests.

### 7.3 `--lock-weight`'s help text was wrong about the deep arm — FIXED

It claimed the deep arm "does not use this value at all". It does: that arm
accepts a candidate with probability `(1 + W·merit)/(1 + W)`, so `W` sets its
preference strength too. Corrected in place.

---

## 8. Corrections to the four attack reports

Carried here so nobody inherits them.

- **"Alive by construction because `observe()` resolves death first" is
  overstated.** Contra's `is_dead` is lives-only, and this campaign measured that
  byte holding flat through committed fatal windows; a screen of banked wall
  states finds roughly a third to a half input-DEAD. The alive claim stands on
  **tape replay and the differential screen**, not on code ordering.
- **B-O3's "matched off control" is not matched.** The armed run also carried
  `--gate-opener enumerate --gate-pin-secs -1` and the control did not. The gate
  never armed (`gate_armed: false` on all 28 lines, 0 injections, 0 candidates),
  so the material confound is small — but the pair is not matched and must not be
  cited as such.
- **The shipped `--lock-weight` default is 4.0**, not the 1.0 two reports state.
- **B-O2's resume failure was its own missing flag.** `solve20` *was* banked with
  `--kill-key` (`key_config.kk = 1`); the lineage-mismatch guard fired correctly
  on the run's own omission, not on the ancestor's.
- **B-O2's own disclosed calibration caveat stands:** `--lock-latch-ceiling 96`
  saturates within 1–2 minutes of arming, so most of its window ran with less
  merit headroom than its offline test demonstrated the mechanism has. Raise the
  ceiling (or shorten the hold window) before drawing further conclusions about
  that objective's live effectiveness.
- **The design's trace-inflation risk materialised, mildly.** Under LEX-SURVIVAL
  the wall's maximum `best_steps` rose 1,297 → **1,507** (+16%), exactly the
  ratchet inversion the design predicted when merit is ordered before step count.
  Measured on the other archives: B-O2 1,326, B-O3 1,276, B-O4 1,314. Worth
  watching, not yet worth a step ceiling.

---

## 9. REFUTED — A-fals's headline side-finding

`A-fals` reported that "every reached-gx-3072 claim in this whole body of work is
an archive/checkpoint-continuation result, not a single unbroken life." **Both
halves were tested and the generalisation does not hold.**

Its **own** concatenated tapes really do die: replayed from `mint7` they lose the
first life at step 73–116 and cap at gx 486–1266 (its own receipt records
`baseline_alive_at_wall: false`, `max_gx: 135`, terminal death at step 189).

The **solver's** tapes do not. **24 of 24 banked wall tapes** — six each from
B-O1/B-O2/B-O3/B-O4, all carrying `root_id: "entrance"` — replay end to end from
the declared start state on a fresh single-worker pool with the solver's own
recipe, and every one lands on gx **exactly 3072 with ZERO life losses** across
1,059–1,279 actions, ending alive with lives 2.

**Re-verified independently for this landing** with a harness written fresh
(`runs/contra_lock2/_LANDING/replay_check.py`, importing nothing from any attack
item): the three longest wall tapes from each of the four runs — twelve tapes,
1,254 to 1,507 actions — replayed from `roms/Contra (USA)_start.state.bin` on a
fresh `Pool(num_workers=1, frame_skip=4)` with the solver's own ordering
(`set_headless` → `set_skip_preprocess` → `reset_all` → `load_worker_state` →
rooting NOOP). Result: **12/12 reach gx exactly 3072, 0/12 exceed it, and 12/12
hold `lives` at 2 from first action to last** — `first_life_change_at: null`
everywhere. And it is not a property of the sampled twelve: **all 7,954 wall
traces across the four archives carry `root_id: "entrance"`** (B-O1 1,393,
B-O2 1,421, B-O3 3,393, B-O4 1,747, no other root id present), i.e. they are
full single-session tapes, not chained fragments.

**The defect is in A-fals's own tape bookkeeping, not in the campaign's
receipts.** Its recommendation to screen roots before use is unaffected and
stands: an independent 14-mask × 32-step differential screen of 30 restored wall
states gave **20/30 LIVE** at 124–368 input-dependent RAM bytes against a
same-lineage dead control at 2, with px capped at exactly 136 and gx never above
3072 under any mask.

---

## 10. No tape

**No trajectory in this campaign exceeded 3072**, so `docs/receipts/` gains no
Contra tape and no ROM-pinned receipt is owed. Verified for this write-up by
`runs/contra_lock2/_LANDING/land_verify.py`, which loads every archive this
campaign produced plus the ancestor and reports the top gx bucket and the count
of cells above it:

```
any_cell_above_3072: False
B-O1          cells= 30614  top=192  above=0  wall= 1393  steps_max=1507
B-O2          cells= 28542  top=192  above=0  wall= 1421  steps_max=1326
B-O3          cells= 21219  top=192  above=0  wall= 3393  steps_max=1276
B-O3 ctrl-off cells= 11474  top=192  above=0  wall= 1277  steps_max=1212
B-O4          cells= 11928  top=192  above=0  wall= 1747  steps_max=1314
solve20       cells= 16298  top=192  above=0  wall= 1331  steps_max=1297
```

The same script re-derives `best_score == 3072 + (16 − hp)·2000` for **100% of
wall cells in all six archives** and reports which `--lock-objective` choices have
a live dispatch — the check that caught §7.1.

### What was verified independently before landing

Written fresh for this write-up, importing nothing from any attack item's own
harness:

- `runs/contra_lock2/_LANDING/land_verify.py` → `land_verify.json` — the archive
  scan above, the score-identity re-derivation, and the dead-choice detector.
- `runs/contra_lock2/_LANDING/replay_check.py` → `replay_check.json` — the
  12-tape replay of §9: 12/12 reach 3072, 0/12 exceed it, 12/12 lose no life.
- `runs/contra_lock2/_LANDING/dom_check.py` → `dom_check.json` — the
  domination receipt of §4(iii), re-derived from the archives rather than
  taken from any report.
- The `latch` no-op, the two-clock defect and the `--lock-weight` help error were
  each reproduced, fixed, and then **re-broken on purpose** to watch the new
  tests fail before the fix was trusted.
- Run telemetry re-read from the raw `progress.jsonl` rather than from any
  report: `progress_glitches: 0` and `stasis_armed: true` on every line of every
  run, `max_gx_in_max_area: 3072` on **all 87** progress lines this campaign
  emitted (counted from `progress.jsonl` alone — a naive grep returns 174
  because each line is tee'd to a `.log` twin, which is precisely the 2×
  double-count the predecessor had to deflate out of the prior "3,030
  occurrences" figure), and `solutions: 0` throughout (recorded, not
  interpreted).

---

## 11. What to do next

1. **Do not fan out on approach, and do not enrich the cell key.** Route A is
   closed; key enrichment is a spreading operator and re-running it recreates
   attack 6's 8.8× dilution confound, which would make any future negative
   uninterpretable.
2. **If Contra is revisited on the release hypothesis, the one funded-but-untried
   variable is armed compute, not another merit.** The cheapest honest next
   experiment is a single long run of the objective with the best measured live
   merit dispersion (LEX-YIELD held 0.125–0.857 across 1,743 wall cells for its
   whole armed window) at an order of magnitude more armed time and
   `--lock-band > 0`, pre-registered as a one-shot falsifier with the abort
   stated in advance. Anything less is another 200-second sample of a
   distribution we already have five of.
3. **Fix the metric before, not after** (unchanged from the predecessor, and
   still not done): wire `current_level $0030` and `boss_defeated $003B` into
   `solve:` so a stage advance is representable in the key and visible in the
   headline, and split a monotone kill count out of `score_bonus`. Both are
   independent of the verdict and both must precede any post-wall run.
4. **Reconcile or retire branch `contra-lock-b-o2`.** LEX-LATCH is implemented,
   tested and unmerged, as a parallel and incompatible version of the same flag
   family based off `9bc5485`. The roster test now prevents its *name* shipping
   without it.
5. **Otherwise, re-shelve Contra with this receipt.** Two named open branches
   were closed for roughly two core-hours of search plus the landing. That is the
   correct outcome and it is recorded so nobody pays for it a third time.

---

## 12. Purity and ledger

**Purity (Tier 3), held throughout.** No disassembly, no RAM maps, no
walkthroughs, and no recall of anything about this title. The lock predicate
contains **no game constant**: `in_lock_key` is `key[0] == max_sect and
key[-5] == max_area and key[-1] >= topgx - band`, every term a property of this
run's own search state — no address, no bucket number, no `3072` — so it means
the same thing on a different game, a different core build, or a different
session's frontier, and refuses to mean anything on a run that has not reached a
frontier yet. It self-disarms the instant the frontier advances, because that is
what resets `_pin_time`, so a run that escaped the lock would revert to the exact
pre-change algorithm for the rest of its life.

**Ledger: EXHIBITION.** Solver and instrument measurement only. No
learned-capability claim is made or supported anywhere in this document.
Verdicts are PASS / FAIL / VOID: the mechanism PASSES (inertness and
non-vacuity both verified in code, by mutation, not asserted); the wall verdict
is **HELD**; `solutions: 0` is **VOID**.

---

## Receipts

Tracked, in-repo:

- `docs/research/CONTRA_LOCK2_2026-08-27.md` — this document
- `docs/research/CONTRA_WALL_2026-08-27.md` — the predecessor characterisation
- `CLAIMS.md` § *CONTRA LOCK ROUTE A+B 2026-08-27* — the ledger entry
- `scripts/go_explore_solve.py` — the `--lock-objective` family, `lock_clocks`,
  `in_lock_key`, `lock_armed`, the three merit functions and the three wiring
  sites
- `src/training/go_explore.py` — `GoExploreArchive.record(merit=...)`, the
  lexicographic comparator inserted strictly between score and steps
- `tests/test_lock_objective_roster.py` — the roster/clock guard (§7.1, §7.2)
- `tests/test_lock_objective_inertness.py` — B-O1's inertness + abort gate
- `tests/test_lock_objective_novelty.py` — B-O3's inertness + abort gate

Under gitignored `runs/`:

- `runs/contra_lock2/_LANDING/land_verify.py` / `land_verify.json` — the
  independent landing verification quoted in §10 (archive scan, score-identity
  re-derivation, dead-choice detector)
- `runs/contra_lock2/_LANDING/replay_check.py` / `replay_check.json` — the
  12-tape replay of §9
- `runs/contra_lock2/_LANDING/dom_check.py` / `dom_check.json` — the
  equal-score/more-steps domination receipt of §4(iii)
- `runs/contra_lock2/B-O1/run1{,.log}` — LEX-SURVIVAL
- `runs/contra_lock2/B-O2/` — LEX-LATCH (branch `contra-lock-b-o2`, `ec21587`)
- `runs/contra_lock2/B-O3/run1{,.log}`, `.../control_off/run1`, `analyze_run1.py`,
  `compare_final.py` — LEX-NOVELTY and its (unmatched, §8) off control
- `runs/contra_lock2/B-O4/run1`, `.../smoke` — LEX-YIELD
- `runs/contra_lock2/A-fals/` — the 7th approach: `mint7.state`,
  `mint7_manifest.json`, `mint_and_search.py`, `search_manifest.json`,
  `screen_wall_states.py`, `wall_screen_results.json`,
  `dead_control{.py,_result.json}`, `verify_and_phase_{test.py,result.json}`,
  `first_wall_tape_w{0,1,2}.json`
- `runs/play_one_well/contra/solve20/archive.pkl` — the ancestor archive

**Do not trust cross-build trace replay for this game.** The 2026-08-01
`traces.pkl` action traces do not replay on today's core — 16 of 16 wall traces
max out at raw progress 1807–1815. Every state used in this campaign was minted
fresh, and merit is deliberately never pickled, so a resumed archive starts every
lock cell at merit 0 rather than importing a number banked under another build.
