# Two registered tests, two VOIDs — the on-policy V_adv read and ReDo at the surgical dose

**Date:** registered 2026-08-27, executed 2026-08-27/28.
**Registrations, both written and committed before any compute:**
`docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md` (Lane A, commit `9a9db2d`)
and `docs/proposals/V31_REDO_SURGICAL_2026-08-27.md` (Lane B, commit `1011eff`).
**Verdicts:** **Lane A VOID. Lane B VOID.**
**Standing verdicts moved:** none. B5 is neither re-opened nor corroborated;
the plasticity-loss hypothesis remains untested.

---

## 0. The headline, including why each void happened and what it cost

Two experiments ran. Neither produced the result it was built to produce, and
the two voids are not the same kind of event.

**Lane A voided on a data-integrity defect discovered after the compute.** A
reused NumPy buffer meant every transition written to disk was `(s', a, s')` —
the antecedent state was never recorded, on 100 % of rows, in all 26 banks. The
estimator `Â = r + γV(s') − V(s)` therefore collapsed to `(γ−1)·V(s')`, a
function of one state carrying zero action information **by construction**.
Every η² in the results table is an artifact. **Cost: 2.13 h**, of which
**~1.27 h of scoring measured nothing.** The 0.86 h of rollout collection was
not wasted — it bought the penetration measurement in §1.7, which is real.

**Lane B voided on a preflight, before any campaign compute at all.** Both
admissible dose rungs failed Phase M in opposite directions with no window
between them: τ = 0.10 became a partial network reset (12 of 32 trunk units,
in-run ceiling raised at iter 29), τ = 0.075 recycled the *same two units*
forever (2 distinct indices, top index in 56.4 % of events) — a permanent
lesion, not a recycle. **Cost: 33.5 minutes of compute**, against the
**7.05 h of seed training and ~3 h of eval ladder** it prevented — a campaign
whose Θ would have been VOID on dose either way. **This is the system working,
and it is the second time in two days that a cheap preflight has saved a
double-digit-hour campaign** (`REDO_ACTUALLY_FIRES_2026-08-27.md` saved ~12 h).

**The asymmetry is the lesson.** Lane B's void was *purchased* by a check that
ran before the spend. Lane A's void was *discovered* by an adjudication that
ran after it, because no check in an eight-gate admissibility battery looked at
whether the rows on disk were transitions at all.

**Neither void is a FAIL.** Neither enters any aggregate, neither takes a
branch of its registered fork, and no numeral in either registration was
edited after compute began.

---
---

# PART I — LANE A: the on-policy V_adv read. VOID.

## 1.1 What was supposed to happen

The predecessor computation (`docs/research/VADV_B5_2026-08-27.md`, commits
`b9ed38e` + `789aefe`) returned **VOID at R = 0.279**, inside its pre-declared
indeterminate band `(0.20, 0.50)`. It named its own binding limitation: the
banks were go-explore expert-window states, **off-distribution** for a critic
the policy delivered to the wall 0/717 times, and all 612 qualifying `WALL`
rows were the **entry pixel** — the traversal in dispute was never observed.

Lane A's registration specified the repair: collect the bank from **B5's own
on-policy rollouts** so the wall states are on-distribution for the critic
scoring them; band coverage into the wall interior; score the whole iter-10-to-260
arc rather than one checkpoint; and pre-commit the null reference in writing.
All four were implemented. The repair to the *sampling* worked. The repair to
the *recording* was never made, because nobody knew it was needed.

## 1.2 The defect, proven on the receipts

`state` is bit-identical to `next_state` on **100 % of rows in every bank**:

```
iter 00010  rows (62366, 712)  identical_frac 1.0
iter 00050  rows (61654, 712)  identical_frac 1.0
iter 00150  rows (63497, 712)  identical_frac 1.0
iter 00250  rows (64337, 712)  identical_frac 1.0
iter 00260  rows (63998, 712)  identical_frac 1.0
```

**Root cause, proven rather than inferred.**
`src/emulation/frame_utils.py:190-204` — `TileFeatureStacker._flatten_oldest_to_newest()`
returns its own reused `self._out` buffer ("one allocation reused across
calls", an optimisation added for BC pretrain). `scripts/collect_onpolicy_bank.py`
held `lane.obs` as an alias of that buffer, then called
`s_new = stk.push(...)`, which mutates it **in place** and returns the same
object. So at the recording site `lane.obs is s_new` — and although
`append_transition_row` copies both arguments, both copies capture the
**successor**. The antecedent was overwritten before it was ever read.

Reproduced directly against the real stacker (five steps of the collector's own
loop pattern):

```
BEFORE (no .copy())    identical_frac=1.0
AFTER  (.copy())       identical_frac=0.0
```

**Physical-impossibility cross-check.** `PC_SRC` episodes clear the level —
at iter 250 the PC rows span `gx` 0 → 3266 — and yet `gx(s) == gx(s')` on
**100 % of 22,845 PC rows**. No real transition bank containing a cleared
episode can have that property.

## 1.3 Why that destroys the reading

Under the registered estimator `Â = r + γV(s') − V(s)`, substituting `s = s'`
gives `Â = r + (γ−1)·V(s')`. With `r` omitted (the registration's own declared
conservative omission) this is a **deterministic function of a single state**
that contains no action information whatsoever. `Var_a(Â)` within a cell can
then only pick up residual within-cell state heterogeneity sorted by action
label — the policy's own state-conditioning — which is not the quantity V_adv
is defined to measure.

The magnitudes corroborate. At iter 10, `WALL` raw = **3.93 × 10⁻⁹** against
`Var_batch[V]` = **332.73** — eleven orders of magnitude below the value
scale. Every η² in the published table (`WALL` 0.019-0.036, `PC_B5`
0.035-0.169) is an artifact of that residual, not a reading of advantage
variance.

## 1.4 The ninth vacuous gate, and it sat on the primary verdict path

`NC-b` (`NEG_gx_frozen`) is defined as *cells within `PC_B5` where no tried
action moves `gx`*. Because `gx(s) == gx(s')` everywhere, the frozen mask was
**universally true**, the `moving` exclusion set was empty, and
`NEG_gx_frozen` came back **bit-identical to `PC_B5` at 26 of 26 iterates** —
same `n_cells`, same `n_rows`, same `raw`, same `η²`, same null median, same
q97.5, every digit.

A negative control that is a byte-copy of the positive control is not a
control. It is the bug printing its own signature. **The run report read this
backwards**, citing the single iter-150 cap as evidence that "the safety
mechanism ... did real, falsifiable work here."

The structural consequence is worse than a mis-read diagnostic. `NEG == PC_B5`
implies `NEG LIVE ⟺ PC_B5 LIVE`; both registered signatures require `PC_B5`
LIVE; and the scoring driver forces INDETERMINATE whenever `NEG` is not
COLLAPSED and a signature would otherwise be declared. **Therefore no
signature — MIS-SPECIFICATION or CAPABILITY — was declarable at any iterate,
by construction, before a single checkpoint was loaded.** The reported
`frac_mis = frac_cap = 0.00` is an arithmetic identity of the pipeline, not a
reading of the data. The arc verdict "INDETERMINATE" is the pipeline restating
its own construction.

This is the **ninth** vacuous gate in this repository, and the second in two
days to be found inside a registration whose explicit brief was preventing the
previous ones.

## 1.5 Why eight admissibility gates all passed anyway

Every gate ran and every gate was correct about the thing it inspects. **None
of them inspects transition structure.**

| gate | what it checked | why it could not see the defect |
|---|---|---|
| A1 | offline PC-1 reproduction | computed on `runs/interference/success_1_1.npz`, which the collector never touched — η² = 0.6668875582236998 vs the predecessor's 0.6669, LIVE |
| A3 | zeroed-critic stub returns `raw == 0.0` | returns 0.0 regardless of what the successors are |
| A4/A5 | null non-degenerate, `Var_batch[V] > 0` | 332.73, computed on **states**, which are perfectly valid arrays |
| A6 | ≥ 20 cells, ≥ 400 rows | cell/row counts are unaffected by which state occupies which slot |
| A7 | injected-power gate | injects a synthetic effect (~5 × 10⁻⁵) into `WALL`'s own rows and detects it — a test of **detection**, not of data validity; its calibrated effect is four orders of magnitude above the entire real signal |
| A8 | checkpoint width == bank width | 712 == 712, true of a corrupt bank as much as a correct one |
| purity guard | `(world, level) == (0,1)` at `s` and `s'` | **ran on the true antecedents** (`lane.prev_ram` → `ram_new`) and passed with 0 violations across ~1.6 M transitions — it guarded transitions that were never the ones written to disk |

The last row is the sharpest. **Guard subject and artifact content diverged.**
The guard was correct, ran on the right values, and certified something other
than the file.

And the anti-vacuity tests did not catch it either, for a reason worth stating
plainly: all four revert-verified tests pass **distinct arrays** into
`append_transition_row` (e.g. `ns = np.ones(4)`), so none exercises the
aliasing. **The unit was correct and the integration was not. Nothing in the
job ever asserted on the artifact actually produced.**

## 1.6 Disposition — the retirement rule does NOT fire

The registration's §11 retires V_adv "if the arc rule returns INDETERMINATE, or
the reading is VOID for any reason other than a fixable operational fault
(crash, wall-clock abort, purity-guard raise)". Its stated rationale is that
**two computations "with live controls, real power and a pre-committed null"**
returned no signature.

That rationale is false here, on three counts:

1. **The controls were not live.** `NC-b` was a byte-copy of the positive
   control at 26/26.
2. **V_adv was never measured.** The estimator collapsed to a constant in the
   action dimension before any critic was consulted.
3. **The no-signature outcome was guaranteed by a data-construction defect
   before compute**, not produced by the instrument declining to discriminate.

Retiring an instrument on a run where the instrument never ran would **spend
the retirement on a null a bug manufactured** — precisely the class of error
this registration exists to prevent, and one level up from the §5.3
no-penetration rule that the same registration got right. This is a **fixable
operational fault in the collector**, which §11 excludes by name.

Therefore:

* **V_ADV IS NOT RETIRED.** The question of whether it can adjudicate B5 is
  still open, and one honest on-policy computation still owes an answer.
* **B5's standing verdict does not move**, in either direction. `R = 0.542` at
  the iter-250 comparability point **is not a reading**, cannot re-open
  anything, and **must not be tabled beside the predecessor's 0.279** — doing
  so would present an artifact of a broken bank as movement in the
  discriminator.
* **The rung-relative wavefront amendment stays DEFERRED and stays "pending".**
  It is **not** "closed by this route", because the route did not run.

## 1.7 What survives, and it is not nothing

Four things from this job are real and are banked.

**(1) The no-penetration measurement — the one thing the job earned.**
`lane.gx_trace` is computed from `s_new` directly, which the aliasing does not
corrupt. Across all 26 iterates, **1,040 rung-893 episodes and ~1.04 M
env-steps, `gx` never exceeded 2674** — `min == max == 2674` at every single
iterate, `pen_rate = 0.0` everywhere, zero exceptions.

Per the registration's pre-declared §5.3 reading, applied in full and without
amendment: `INTERIOR` is **VOID**, reported as VOID and **never as COLLAPSED**;
it is recorded as a **positive measurement**, not a gap; and it is evidence for
**neither hypothesis**, because CAPABILITY and MIS-SPECIFICATION both predict
it. Its value is localisation: *no rung-893 rollout at any of 26 iterates
exceeded gx 2676* places the failure **at the gx-2674 state itself** rather
than somewhere along a 198-px traversal, which is strictly stronger and more
specific than entrance 0/717. That the registration wrote this reading down
**before** seeing the data is why it can be banked now instead of argued about.

**(2) Instrument reproducibility.** A1 reproduced the predecessor's offline
η² = **0.6669** to ten significant figures (0.6668875582236998) on
`runs/interference/success_1_1.npz`, a file the collector never wrote.
`scripts/score_banked_iterates.py` is not implicated in this defect and its 45
tests stayed green throughout.

**(3) The adjacent-rung probe** (`runs/vadv_onpolicy/probe_933.json`),
diagnostic only and barred from R, is behavioural data read off `s_new` and is
therefore also uncorrupted. Rung-933 restarts cleared **8.3 % at iter 10,
12.5 % at iter 70, 16.7 % at iters 130 and 190 — and 0 % at iter 250**, where
all 24 episodes deposited in `WALL` with `max_gx` capped at 2674. A rung the
curriculum demonstrably advanced through earlier had stopped clearing by the
end of the arc. That is a regression signal worth its own instrument; it is
not a V_adv reading and carries no verdict weight here.

**(4) Process integrity held.** No goalpost was moved, no threshold was edited
after the first η² was read, the pre-committed permutation-median null was used
throughout with `eta2_null_analytic` emitted alongside and never entering R,
two iterates (40, 90) were **excluded on A7 power alone** (0.88, 0.82) rather
than silently scored, and the registered A7 negative demonstration did fail on
this run's own data (full-scale `WALL` power 0.98; the same gate at the same
calibrated effect down-sampled to 8 cells returns 0.90). The failure here is a
**data-integrity** defect, not a process-integrity one.

## 1.8 The fix, shipped with the verdict

Both call sites now copy out of the stacker's reused buffer, with the reason
written at the call site so the copy cannot be "simplified" away later. And
because a correct loop is not a correct artifact, a new **threshold-free**
guard runs on the arrays immediately before `np.savez`:

* **Chain.** Within one episode, the successor recorded at step `i` **is** the
  antecedent recorded at step `i+1`. Exact, no tolerance, true of any correct
  bank whether the scene moves or is frozen — and false at every step under the
  aliasing defect.
* **Non-degeneracy.** Not *every* row may satisfy `s == s'`. Individual frozen
  rows are legal (five identical frames inside a 4-frame stack), so **no
  fraction below 1.0 is asserted** — picking one would be a threshold nobody
  checked against its acting range, which is the family of error this document
  is already about. 1.0 is the defect's own signature and is impossible for a
  bank containing motion.

Revert-verified, executed not asserted: neutering `assert_bank_wellformed` to a
no-op fails **4 of 45** tests in `tests/test_collect_onpolicy_bank.py`,
including the exact 2026-08-27 artifact; restoring it returns 45/45. The
guard raises `[bank] DEGENERATE` on the real pre-fix loop output and passes on
the post-fix output.

## 1.9 Re-run conditions

A second on-policy reading is worth taking, and it is cheap — the checkpoints
and the protocol already exist. It requires, before any scoring:

1. The aliasing fix and the artifact guard above (**shipped**).
2. **Re-derive `NC-b`'s acting range on the repaired bank BEFORE scoring.** If
   `NEG_gx_frozen` is again a near-copy of `PC_B5` on *real* transitions, then
   NC-b is unusable on on-policy data and must be **re-specified in a written
   addendum** rather than left in place to cap the verdict. Checking a
   threshold against its acting range *on the data it will see* is the standing
   rule; on this run it was checked against data that did not exist.
3. A disclosed re-registration of what changed and what did not. The bands,
   the null reference, the R thresholds and the arc rule need not move and
   should not.

Only then is a second reading — and any retirement decision — meaningful.

---
---

# PART II — LANE B: v31, ReDo at the surgical dose. VOID.

## 2.1 What was supposed to happen

v30's premise-falsifier (`439b87f`, `d02987f`,
`docs/research/REDO_ACTUALLY_FIRES_2026-08-27.md`) established that ReDo fires
at τ = 0.25 but at **62 % of the 32-unit trunk re-initialised every iteration**
— the registration's own forbidden network-reset regime. It also located the
tail precisely: dormancy lives **only** in fc2 (fc1 was 0/64 and 0/96 across
all 86 measured iterations), and τ ≈ 0.10 looked reachable at a **surgical
~1-3 unit dose from ~iter 16**.

v31 registered exactly that: τ = 0.10, the **top of the Deep Research
prescription's own 0.025-0.1 range**, with the dose ceiling moved **in-run**
(a `RuntimeError` on a trailing-10-check median above 25 % of the worst-hit
layer), the arming deadline raised 25 → 40 to match τ = 0.10's measured first
crossing, and a new **F3 distinctness gate** (≥ 6 distinct fc2 indices, no
index above 60 % of recycled-unit-events) to catch the permanent-lesion case
that no gate caught before.

Nothing about that plan was wrong. It simply turned out that **no fixed τ on
this architecture is simultaneously firing, surgical and distributed.**

## 2.2 Phase G — the ceiling is live, not vacuous

The registration required a live run at τ = 0.25 with the in-run ceiling armed
to raise by iter ~9, *or the campaign does not start*. It did:

```
[redo] ENABLED tau=0.25 every_iters=1 scope=fc1,fc2 sample=4096 reset_moments=true
RuntimeError: [redo] VOID-OVERDOSE: trailing-10-check median dose of the
worst-hit layer exceeded 0.25 at iter 9 (tau=0.25).
```

**3 m 46 s.** Per Phase G's own terms this certifies a **check**, not the arm:
no Phase G result may be cited as evidence about the arm, about τ = 0.10, or
about the hypothesis.

## 2.3 Rung 1 — τ = 0.10 — NO-GO on M2 (VOID-OVERDOSE)

Run at exactly the registered operating point (`[redo] ENABLED tau=0.1`),
uncontended, 19.5-20.5 s/iter — **faster** than the registered 25.4 ± 2.5 s
band, so no contention contamination.

First firing at **iter 16**, exactly as the registration predicted. Then the
dose **climbed**:

```
units recycled, iters 16-29:  1, 1, 1, 1, 5, 6, 6, 6, 8, 11, 12, 12, 12, 12
```

The in-run ceiling raised `VOID-OVERDOSE` at **iter 29** (trailing-10-check
median 0.297 > 0.25). Equilibrium **12 of 32 = 37.5 %** — *identical to v30's
measured τ = 0.15 equilibrium*. **τ = 0.10 is not surgical on this
architecture.** Detected in **10 m 10 s**.

**The registered `frac ≈ 2.5·τ` law is REFUTED at its own point prediction.**
It predicted 7.6-8.0 of 32 units at τ = 0.10; the measured equilibrium was
12/32, with the last four checks all reading 12 and the trend still upward when
the ceiling fired. (The Phase M adjudicator prints
`eq_law_corroborated: True`; that field averages across the climb in a run
truncated by the abort and should not be cited. The law is reported as
refuted.)

**V5/A4 argument-agreement is confirmed structurally vacuous on live data:**
median `agree` **0.97** at τ = 0.10 while 37.5 % of the trunk was being reset
every iteration. The registration's decision to **demote** it from a dose gate
to a reported diagnostic was correct, and this is the receipt.

## 2.4 Rung 2 — τ = 0.075 — NO-GO on M5 (permanent lesion)

The registered ladder for an M2 failure is **de-escalate ONCE to τ = 0.075 and
re-run Phase M in full**. Exactly one rung, exactly one direction, using the
pre-minted config (a clean single-functional-variable diff: `redo_tau`
0.1 → 0.075).

τ = 0.075 ran all 60 iterations with no abort, and it looked good on three of
five criteria. It **fires** (from iter 29). It is **surgical** (max 2 of 32 =
6.25 %, far under the ceiling). It is **sustained** (31 firing iterations,
cum_recycled 55). And it failed anyway:

```
recycled unit indices: fc2=[16]      ×  7
recycled unit indices: fc2=[5, 16]   × 24
```

**Two distinct fc2 indices across the entire run.** Unit 16 appears in **100 %**
of firing events; top-index share **56.4 %** of recycled-unit-events. Against
M5's requirement of ≥ 4 distinct indices and no index above 60 %, this is a
**NO-GO** — and it is not a near miss on the share threshold, it is a
categorical failure on distinctness.

What that describes is a **permanent partial lesion of two trunk units**,
re-initialised every iteration and immediately re-dormant — self-sustaining,
not a recycle. **This is precisely the pathology F3/M5 was written to catch,
and this is the first time that gate has bitten on live data.** Detected in
**19 m 32 s**.

## 2.5 STOP — per the registration's own rule, not a judgement call

§9 states, in two independent places:

> If Phase M fails M5 (fixed index set) → **STOP**; a lesion is not a dose
> problem and no rung fixes it.

> If the second Phase M also NO-GOes, the campaign **STOPS**.

Both conditions are met. τ ≥ 0.15 and τ ≤ 0.05 are forbidden by this
registration **including as escalations**, each with the measurement that
forbids it. τ = 0.125 is not reachable: it is the M1/M4 escalation branch, not
the M2 branch, and taking a second rung after seeing these results would be a
moving goalpost.

**Zero treatment seeds were launched.** `armed = 0`, `scored = 0`, and
launching one now would violate the registration's own
*"NO TRAINING BEGINS BEFORE PHASE M RETURNS GO."*

**Θ IS NOT ISSUED.** N is fixed at 4 and Θ requires four ARMED and scored
seeds; with zero, the registered disposition is **VOID-UNDERPOWERED**, no Θ,
per-seed numbers banked individually as mechanism receipts. There are none to
bank. **The bar did not move: 0.80 and 0.767 stand untouched, and the 0.05
winner's-curse budget is unspent.**

The arm gate was re-run independently on both Phase M logs at their own τ,
with the registered thresholds. Both return `VOID-MINIMAL-DOSE`, rc = 2; the
τ = 0.075 log additionally VOIDs on F3. The gate is structurally incapable of
printing PASS or FAIL for an unarmed seed, as designed.

## 2.6 The fabricated NO-GO that nearly shipped

The first Phase M "result" was not a measurement. The orchestrator redirected
stdout with `> checkpoints/mario_1_1_v31_redo_seed0/run.log` but did not
`mkdir` that directory first (its `mkdir` came *after* the training call), so
the shell redirect failed, `train_game.py` never started, the adjudicator
crashed on a missing file, and the script wrote a **"Phase M NO-GO"** marker.
The marker and the launch are timestamped in the **same second** (22:27:49).

**Adjudicating a ladder rung off that would have been exactly the fabricated
result the shared discipline forbids** — a registered scientific verdict
derived from a missing directory.

Fixing it surfaced a **second latent defect** in the same path: the trainer
attaches its own `FileHandler` to `<checkpoint_dir>/run.log` and **truncates**
it (`mode="w"`), so a shell redirect to that same path produces two writers at
independent offsets and corrupts the very log Phase M is adjudicated from. The
interleaving was observed live, that run was killed, and stdout was redirected
to a separate path before the real Phase M was run.

Both Phase M logs carry exactly **one** `[redo] ENABLED` line, confirming no
restart contamination. (The τ = 0.10 log's closing
`[supervisor] restart budget exhausted` is the supervisor correctly giving up
on a run the registration wants dead.)

## 2.7 Anti-vacuity, executed not asserted

Both new gates were neutered in place and their suites re-run, then restored
and verified byte-clean:

| gate | neutered | restored |
|---|---|---|
| in-run dose ceiling (`dose_ceiling_trips`) | **4 of 11 fail** — incl. all three banked-pilot replays and the strict-inequality 8/32 boundary | 11/11 |
| F3 distinctness (`redo_arm_gate.py`) | **5 of 28 fail** — incl. single-index-lesion and the 60 %-share boundary | 28/28 |

The in-run ceiling matches the registration exactly: worst-hit layer via
`max()` (never pooled — the pooled denominator was v30's own vacuity), trailing
10 **checks** (not firing events), strict `>` so 8/32 survives and 9/32 trips,
inert before 10 checks. `_REDO_ARM_DEADLINE_ITERS = 40` and
`_REDO_DOSE_CEILING = 0.25` are hardcoded module constants, never config keys.

## 2.8 The plasticity hypothesis: REMAINS UNTESTED

Zero seeds launched, zero armed, zero scored, no Θ. **Neither Hypothesis A
(consolidation / plasticity loss was the barrier) nor Hypothesis B (48k is a
hard ceiling) received any evidence.**

Critically: **the v27/v28 plasticity confound is NOT discharged.** Discharging
it required a FAIL at a firing, surgical, sustained dose. No such run exists.
**The 2026-08-25 DR's ReDo prescription may NOT be closed as
EXECUTED-AND-NEGATIVE — it was never executed.**

## 2.9 What this VOID licenses, exactly and no wider

A VOID licenses only the mechanism finding attached to its specific void
reason. **LICENSED, in writing:**

1. **The registered §9 stopping statement, now carrying a live two-rung
   receipt:** *"On a Linear → LayerNorm → SiLU 32-unit trunk there is no fixed
   dormancy threshold that is simultaneously firing, surgical, and distributed
   over the training budget; fixed-threshold ReDo is mis-specified for this
   architecture."* The two admissible rungs fail in **opposite directions with
   no window between them** — τ = 0.10 resets 37.5 % of the trunk, τ = 0.075
   lesions the same 2 units forever.
2. **The registered `frac ≈ 2.5·τ` law is REFUTED** at its own point
   prediction (predicted 7.6-8.0 of 32 at τ = 0.10; measured 12/32 and still
   climbing).
3. **V5/A4 argmax-agreement is confirmed structurally vacuous on live data**
   (median 0.97 at 37.5 % trunk reset). The registration's demotion was
   correct.
4. **Phase G certifies the in-run dose ceiling is live, not vacuous** — and,
   per its own terms, says nothing about the arm, τ = 0.10, or the hypothesis.
5. **Exactly one successor is licensed:** the **rank-based bottom-k dose**
   (v30 §10.1), stable under the observed drift where no fixed threshold is —
   as a **NEW experiment with its own registration**, not a rung of this one.

**NOT LICENSED — any of these in writing would be a fabrication:** any
statement about Θ or the 0.767 bar; any claim that ReDo is or is not a lever of
any size; any claim that plasticity loss was or was not the barrier;
Hypothesis B confirmed or falsified; any claim about other plasticity
interventions (L2-init, CReLU, weight churn, layer-norm resets, periodic full
resets); and **any FAIL-class inference whatsoever**, since a VOID takes no
branch of the fork and enters no aggregate.

---
---

# PART III — what the two voids cost, and what they bought

| | Lane A | Lane B |
|---|---|---|
| verdict | **VOID** | **VOID** |
| registered budget | 6.0 h (≤ 3.0 h compute) | 13.0 h |
| spent | **2.13 h** | **0.56 h** (33.5 min compute) |
| void found | **after** the compute, by adjudication | **before** the campaign, by preflight |
| wasted | ~1.27 h of scoring | nothing — the preflight *is* the product |
| prevented | — | ~7.05 h seed training + ~3 h eval ladder |
| standing verdict moved | none (B5 unchanged, V_adv **not** retired) | none (plasticity untested) |
| goalposts moved | none | none |

**The one-sentence version.** Lane B's void is what a registration is *for*:
a 34-minute check refused a 10-hour campaign whose result would have been
uninterpretable, and produced a real, falsifiable, pre-registered mechanism
finding in the process. Lane A's void is what a registration cannot do on its
own: eight admissibility gates, a pre-committed null, a binding power gate and
four revert-verified anti-vacuity conditions all passed on a bank whose rows
were not transitions, because **every one of them tested the pipeline and none
tested the artifact.**

The generalisable rule, and the reason the §1.8 guard ships with this
document rather than after it:

> **A guard that runs on the values in hand certifies the loop. Only a guard
> that runs on the bytes written certifies the file. When those two differ, the
> gates are measuring a program that is not the one that produced your
> results.**

Every future bank collector in this repository should ship the chain invariant.
It costs one pass over the array and it is threshold-free.

## Next

1. **Re-run Lane A** on the repaired collector, with `NC-b`'s acting range
   re-derived on real transitions **before** scoring (§1.9). Cheap: the
   checkpoints and the protocol exist, and collection is 0.86 h. Until then
   V_adv is **not** retired and B5 stays flagged under-instrumented.
2. **Do not run another fixed-τ ReDo experiment.** §9's stopping statement
   is banked with a two-rung receipt. The only licensed successor is the
   rank-based bottom-k dose, and it needs its own registration and its own bar
   written before its own compute.
3. **The rung-933 regression** (§1.7, item 3) — clears falling 16.7 % → 0 %
   across the arc on a rung the curriculum had advanced through — is a real
   behavioural signal from uncorrupted data and currently has no instrument
   pointed at it.

## Receipts

**Lane A** — `runs/vadv_onpolicy/`: `collect_summary.json` (26/26 iterates,
0.864 h, `pen_rate = 0.0` and `gx_max = 2674` at every iterate, 0 purity
violations across ~1.6 M transitions); `iter_{00010..00260}.npz` + `_episodes.json`
sidecars (**all VOID — do not score**); `probe_933.json`; `A1/a1_reproduction.json`;
`arc_scored.json` / `.jsonl`; `arc_verdict.json`. Collector and its tests:
`scripts/collect_onpolicy_bank.py`, `tests/test_collect_onpolicy_bank.py` (45).
Scorer, not implicated: `scripts/score_banked_iterates.py` (45 tests green).

**Lane B** — `runs/v31_redo_surgical_2026-08-27/`: `phase_g/phase_g.log`;
`phase_m/PHASE_M_RESULT.json`, `phase_m_seed0_tau010.log`,
`phase_m_seed0_tau075.log`; `ladder_configs/` (pre-minted contingency rungs).
Status markers: `runs/redo_surgical/status/` (`02_phase_m`, `03_seeds_NOT_RUN`,
`99_campaign_stopped`). Machinery: `src/training/redo.py`,
`src/training/trainer.py`, `scripts/redo_arm_gate.py`,
`scripts/adjudicate_phase_m.py`, `scripts/score_cross_fit.py` (commits
`0906f7c`, `8606fb1`).
