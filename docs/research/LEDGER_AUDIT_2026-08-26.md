# Ledger audit, 2026-08-26

**How much of what this project believes it knows is actually measured?**

That is the only question this document answers. It was forced by a week
in which five separate gates were found to be incapable of returning
their own negative, and in which the fourth was written *by the agent
fixing the previous three, in the same session, having been briefed on
them.*

---

## The counts

### Claims

**61 load-bearing claims in `CLAIMS.md` were read and questioned.
39 warranted a ruling.**

| Ruling | Count |
|---|---|
| **STANDS** | **21** |
| **WEAKENED** | **15** |
| **WITHDRAWN** | **3** |
| *(examined, nothing to record)* | *22* |

Two things about that table matter more than the numbers in it.

**First: not one positive result was withdrawn.** Every withdrawal was
made against a receipt on disk or a re-run at HEAD, never against a
judgement. All three withdrawals are of *verdict language attached to
an instrument later shown incapable of the opposite verdict* — none is
of a measured positive. Withdrawing a sound result to look rigorous is
the same defect wearing the opposite mask, and the asymmetry held
everywhere it was tested.

**Second: the ledger is in better shape than the week's discoveries
predicted.** The four CONFIRMED clears are intact. All four
honest-protocol rates are intact and, if anything, understated. Every
Learned-ledger null is paired with a positive control from the same
harness in the same pass. That is the headline and it should not be
buried under the corrections: **the ledger came out mostly healthy.**

### Games — the number that actually matters

Claims are the wrong denominator for "what do we know". Games are.

| | Count |
|---|---|
| Games we have a **receipt** for | **42** |
| Games we **actually know something about** | **11** |

**~26%.** Three quarters of the corpus has a document and no knowledge.

*Receipt* = 40 distinct ROMs across the 45 shipped `solve:` profiles,
plus SMB and Lost Levels. Nearly every one carries a banked smoke, gate
receipt, or wave-doc verdict.

*Known* = an instrument **demonstrated capable of returning a positive
on that profile** returned a measurement about that game. Listed in
full, because a number like this is worthless without its members:

**Clear witnessed, predicate demonstrated firing (5)**

| Game | Witness |
|---|---|
| Super Mario Bros. | all 32 levels solved; TAS receipts; SMB-transfer clears Lost Levels 1-1 |
| Bubble Bobble | round 69→70 rendered in a visibly different layout; **115 banked solution files across 107 archives**, chain walking rounds 22→99 from independently-minted entrances; detector now fires 2/2 at +4/+18 |
| Castlevania | **14 banked solutions across the 5 named run trees** (cv_smoke 1, cv_chain_a 4, cv_chain_hw 3, cv_chain_hw2 3, cv_chain_poweron 3), plus more elsewhere; incl. a power-on 0→1→2 chain |
| Excitebike | `finale` hook fires at action 1143/1144; rendered FINISH line |
| Tetris-B | banked 4,329-action win; rendered LINES-000 / SUCCESS |

**Declared observable demonstrated moving under a positive-capable
instrument; no clear (6)**

| Game | Witness |
|---|---|
| Metroid | 87 distinct odometer values / 508 px over 111 live steps, re-run at HEAD; scene keys converted a frozen 272-cell smoke into 2,596 cells and **16 room transits**, with a deepest-cell diagnostic proving the counter registers real doors |
| Kirby | real doors proven by **0/7 nametable-hash overlap**; the same generic `area: 0x803` key tested and **refuted** with its own receipt (render blanking re-anchors the odometer; 2 spurious bumps) — a positive fact *and* a receipted negative |
| Bionic Commando | gate PASS at HEAD, 122 distinct / 483 px over the **full** 1200 steps, zero dropped |
| Castlevania III | room counter `$0057` advanced through **2 real transitions** during the smoke; frontier grew 1 → 173 cells, 162 still open |
| Rygar | gate PASS at HEAD, 116 distinct / 467 px; death-debounce A/B moved gx 1536→5360 at fixed budget |
| Punch-Out | blind nomination of addr 920, attack_agree 5/5 vs defense_agree 0/5; `max_gx_in_max_area=81` reproduces |

A companion sweep of the four League onboarding wave docs put this
number at **7**. The difference is scope, not disagreement: that sweep
surveyed the onboarding roster only, which excludes SMB entirely and
counts Tetris-B and Punch-Out as *negative verdicts that survive*
rather than as knowledge. Both numbers are defensible inside their own
frame. **11 of 42** is the whole-project figure.

The onboarding sweep's own result deserves restating here because it is
the sharpest measurement of the gap: **38 banked verdicts over 36
games; 16 survive re-derivation, 22 are vacuous.** 23 of those gates
were re-run at HEAD, and the 23 receipts were re-read for this audit
rather than relayed:

| Outcome | Count | Examples |
|---|---|---|
| **Flipped PASS → FAIL** | **9** | 1942 (600 distinct → 15 over 21 live steps), Batman TVG (318 → 5 over 9), blaster_master (126 → 15 over 22) |
| Already FAIL, still carrying `SOUND_ADVANCING` | 2 | darkwing_duck, journey_to_silius |
| Still PASS, but on the live portion only | 8 | gradius 74 of 1200 steps, ninja_gaiden_ii 55, metroid 111, castlevania 758 |
| **PASS with zero steps dropped** | **4** | excitebike, bionic_commando, mega_man_3, shatterhand |

The third row is not a criticism of the games — the D5 fix *should*
drop a post-death tail, and dropping it is the instrument working.
What it corrects is the banked description: a receipt reading "1200
steps, 74 distinct" described a measurement that actually covers 74
steps of live play. Two of these were reproduced independently here and
match to the digit (metroid 87 distinct / 111 steps; bionic_commando
122 / 1200).

Two causes account for nearly all of it, and both are the week's
pattern. **(a) The gate never watched the lives byte** — 28 of 45
profiles are contaminated by a post-death tail
(`runs/onboard_wave6_d5_sweep_v3.json`, re-counted here: 28), so
game-over animation was counted as progress. **(b) 40 banked gate
receipts carry `passed: true` on a zero-range odometer**
(`distinct=1, min=max=0`) across 13 profiles — the figure comes from
`1c153ab`'s own audit, and the override deleted the contradicting
instrument finding *before* the receipt was written, so the receipt
records no trace of what it suppressed.

---

## The three withdrawals

**1. Kung Fu — "a skill wall, not an instrument fault".**
Withdrawn against the repo's own commit `1c153ab`, landed hours before
the audit and never absorbed into the ledger. That commit deleted a
vacuous `passed = not instrument_findings` camera-static override and
names kungfu first among **13 profiles / 40 vacuous passes** carrying
the `distinct=1 / min=0 / max=0` signature.
`runs/odometer/gate_kungfu_left_2026-08-23.json` carries it verbatim,
including `instrument_findings: []` — the override deleted the
contradicting finding *before the receipt was written*. Under the
shipped gate the row reads `SIGNAL UNUSABLE — camera static`. The OAM
churn of 540 is a real positive and is kept.

The audit added one correction to the supporting clause: the
"~719,500 steps with `current_floor` never leaving 0" figure has no
receipt. What is on disk is stronger and different — a frontier pinned
at `cells: 32, max_gx_in_max_area: 244` across **both** Kung Fu runs,
~1.72M steps together (`kungfu_bootstrap` 514,239;
`live_show/kungfu` 1,206,456).

**2 & 3. Ninja Gaiden and Contra — odometer `SIGNAL-SOUND`.**
These two were ruled **STANDS** by the first sweep and that ruling was
wrong. It validated them against ONE vacuity signature — the
camera-static override it had just watched get fixed — and never
applied the second, which its own companion analysis had already
documented: `runs/onboard_wave6_d5_sweep_v3.json` finds **28 of 45
profiles contaminated by a post-death tail.**

Re-running the identical gate at HEAD, same profiles, same start
states:

| | Banked | HEAD | Steps dropped |
|---|---|---|---|
| Rygar | 117 distinct / 470 px, PASS | 116 / 467, **PASS** | 1062 (tail added ~nothing) |
| Ninja Gaiden | 126 / 1,384 px, PASS | **FAIL** — death byte reads 0 at start | 937 of 1200 |
| Contra | 162 / 635 px, PASS | **FAIL** — 19 distinct in 69 steps, too coarse | 1,131 of 1200 |

So ~94% of Contra's banked signal was game-over animation. Ninja
Gaiden's banked receipt recorded `lives_at_start: 255` — an already
underflowed byte — and **three later receipts already on disk**
(`gate_ninja_gaiden_2026-08-25{,_reverify,_1200}.json`) all read
`"passed": false`. None was cited.

Contra's second half is worse than a contaminated number. It was
reported as *"cross-validated against the RAM pair (162 vs 163
distinct)"* — the strongest-sounding evidence in the sentence — and
**the 163 has no receipt anywhere in the tree.**
`find runs -iname "*contra*" -name "*.json"` returns exactly two files:
`runs/ram_verify/contra.json`, which is not a signal gate and contains
no 163, and the odometer receipt being corrected.

What survives is narrower and worth keeping. Over Ninja Gaiden's 263
live steps the odometer reads **192 distinct / 748 px** — a live axis
with *better* resolution than the banked claim. The axis is real; the
verdict is not.

**The correction was applied at the source as well.** These four rows
originate in `docs/research/CAPABILITY_REPORT_2026-08-24.md` §1.1, and
that table is where the unreceipted "163" first appears — as a
left-hand "before" column entry, never as a measurement anyone banked.
Correcting `CLAIMS.md` alone would have left the stale table one grep
away, so the capability report now carries the same correction inline.

The correction was also applied at the source. These four rows
originate in `docs/research/CAPABILITY_REPORT_2026-08-24.md` §1.1, and
that is where the unreceipted "163" first appears — as a left-hand
"before" column entry, never as a measurement. Correcting `CLAIMS.md`
alone would have left the stale table one grep away, so the capability
report now carries the same correction inline.

---

## What the audit got wrong about itself

An audit that does not audit its own arithmetic has learned nothing
from the week it is auditing. Three corrections to the first sweep,
all found by re-derivation — and a fourth to this document:

**Its solutions census missed ~87% of the corpus.** It reported "38 run
dirs hold a non-empty `solutions/`" while correcting an enumeration.
The true count is **297 non-empty of 507** — the sweep had scanned
`runs/*/solutions` at depth 1. Re-classifying all 297 finds no strays,
so its family-level conclusion holds. It holds by luck, not by method,
and the method is the point.

**It replaced an undercount with another undercount.** Correcting
"three profiles with a non-empty `level_key`", it wrote "four
clear-capable profiles by three routes". `scripts/clear_reachability.py
--all` reports **NONE=37, REACHABLE=8** — eight profiles by **four**
routes: `level_key` (bubble_bobble, castlevania, kid_icarus), `finale`
(excitebike), `byte_change` (tetris_b), `score_jump` (ducktales),
`confluence` (contra, contra_blank). Quoting "four" forward would write
off `contra`, `contra_blank`, `ducktales` and `kid_icarus`'s
`solutions: 0` as compile-time constants when the repo's own guard says
their predicates are arithmetically fireable — **the exact mirror of
the Rygar error the sweep had just corrected.**

**Its self-reported tallies did not match its own rulings.** It
reported 18 STAND / 15 WEAKENED / 1 WITHDRAWN = 34. Counting the
rulings it actually issued gives 21 / 13 / 1 = 35. Small, and it is
the kind of small that this document is not allowed to shrug at.

**And this document reproduced the depth-1 error one more time before
catching it.** The first draft of the games table above credited
Bubble Bobble with "4 banked solutions" — a number inherited from the
same truncated census being corrected two paragraphs up. The real
figure is **115 solution files across 107 archives**. It was caught
only because every figure in that table was re-derived from the
filesystem before publishing, rather than copied from the sweep that
supplied it. The lesson is not that the sweep was careless; it is that
**a wrong number survives being quoted by someone who agrees with it.**
The only thing that stopped it the third time was re-running the
command.

---

## What was load-bearing enough to be worth checking, and held

Not everything needed the check. These did, because a public claim or a
research decision rests on them.

- **Castlevania's `0 solutions` is a real FAIL-class negative** and must
  never be swept up with Rygar's. `castlevania.yaml` declares
  `level_key: [0x0028]`; `clear_reachability` reports it REACHABLE; the
  predicate is demonstrated firing on that profile in **14 banked
  solutions across five independent run trees** (cv_smoke 1,
  cv_chain_a 4, cv_chain_hw 3, cv_chain_hw2 3, cv_chain_poweron 3),
  including a power-on chain walking 0→1→2. This is one of only a
  handful of genuine FAIL-class negatives in the repo. Preserving it
  was the single most important thing the audit had to not get wrong.
- **The four honest-protocol rates** (1-1 43%, 1-2 38%, 1-3 21%,
  1-4 51%) and the banked backward-1-1 control at 0.767. Each is a
  property of a specific sha256-pinned artifact, measured by a harness
  that returns positives on that very profile, with its predicate named.
  The checkpoint-selection defect runs *conservative* here: a rate
  measured on a fixed artifact can be understated by a bad selector,
  never inflated. **The ledger's most robust content.**
- **V28's FAIL verdict**, which closed a research line. It survives —
  but by a rescue the ledger never recorded. F0's 192-evaluation
  split-sample re-derivation lands on **0.670 exactly**. Now recorded
  (ADDENDUM V-1), along with the defect it rescues from.
- **The odometer certification, 5/5.** Three of its five checks are
  positive-form and a constant-zero odometer fails them immediately
  (`total_dx 551` with `max_regress 0`; a real restore differential
  185 → 368 → 185). Certification was on SMB; the per-game verdicts are
  separate probes, which is exactly why two of them fell.
- **ReDo's inertness** — the ledger's best entry, and the template for
  the rest of it. A null banked *together with* its positive control:
  the forced-recycle sweeps fire (9 recycles at tau 0.25, 6 then 13 at
  0.30). That is what makes it FAIL-class rather than VOID.
- **Every purity finding.** Re-verified against the configs directly,
  including a mechanical grep over the whole file (1,284 lines
  pre-audit) confirming
  `ZeldaReward` / `MetroidReward` appear only inside the quarantine
  blocks. Purity is fully intact; no positive is affected.

---

## The pattern underneath the 15 weakenings

**In every single case, the underlying receipt was MORE honest than
the ledger entry summarising it.**

| The receipt said | The ledger said |
|---|---|
| "as currently specified (§9 Task 2 scope), **Stage 1-2** does not yet achieve zero false confirmations" | "**the engine** … distinguishes signal from noise" |
| RG-0.5's literal wording is "unsatisfiable on the measured surface"; the test asserts something weaker | "including **all five** design-mandated assertions" |
| the self-derived nine are "**narrower than they should have been**" | "results stand" |
| F0: the per-seed v28 numbers are systematically low | quotes them as capability |

The failure mode is not dishonesty, and it is not sloppiness in the
work — the work generated the caveats. It is **dropping the receipt's
own scoping caveat when quoting its headline.** Which means the fix is
cheap, and it is what shipped: **eleven addenda annotated in place, no
entry deleted, three verdict clauses withdrawn, and one standing
rule.** Every original sentence is preserved verbatim; the corrections
sit beside them, per this file's existing ADDENDUM convention.

The worst instance found this week is new and was not in either
sweep's brief. The item-semantics engine, shipped 2026-08-25 to answer
the "a key can open a specific door" directive, reports Stage 3a as
run. Its receipt shows `total_leads: 0` on all four sequences with
`cap_hist_key_present_pregraft_run: false`, because line 616 reads
`cap_hist` off pre-graft archives that do not carry the key. **Zero
leads was a compile-time constant.** So Stage 3b — which the module
docstring calls "the ONLY thing in this engine allowed to turn a
Stage-3a lead into a claim", and which is the entire structural answer
to the rarity confound — **never executed at all.** Half the pipeline
has never run, and the ledger read as end-to-end validation.

---

## One new instance, found and closed during this audit

The clear-signal wire-up (`140218b`) banked Bubble Bobble at
`total_false_positive_crossings: 0`. That zero was scored on **the same
two tapes the `scene_cut` gate was calibrated on** — the gate is "the
smallest integer strictly above the null measured on those tapes", so
it *cannot* fire below threshold there. The zero was arithmetic, and it
was reported in the same field, with the same words, that an
out-of-sample zero would have used.

It is not thrown away: an in-sample zero is a real consistency check,
and a non-zero there would be a genuine defect. But it is not
specificity. `clear_detect.calibration_provenance()` now computes the
overlap mechanically from the banked calibration receipts and writes it
onto **every** receipt, empty or not, so its absence is visible rather
than ambiguous. Bubble Bobble reads `n_in_sample: 2`; SMB's five-trace
zero reads `n_in_sample: 0` — a genuinely out-of-sample zero, now
labelled as one. Mutating the overlap test to `if True` reddens
`test_the_disclosure_is_not_a_constant` by name.

How much this concedes is bounded, and worth saying so the correction
does not overshoot: **Tetris-B is also in-sample (`n_in_sample: 1`) and
still records one false positive.** The calibration bound constrains
the calibrated signal on the calibrated tape; it is not a blanket
guarantee of zero, and the instrument demonstrably still fires. The
Bubble Bobble zero is a weaker result than it read as, and a real one.

The out-of-sample evidence for Bubble Bobble does exist, and is worth
citing precisely because the receipt's zero does not carry it: the live
test replays a **different** tape (round 69, `chain_day2f`) and the
detector is silent across all 14 evaluations before the clear, then
fires 1 action after it.

---

## State of the clear detector

The campaign's own gate, stated without dressing:

- **Bubble Bobble fires.** 2/2 at +4 and +18 frames, reproduced at HEAD
  during this audit (2060 vs 2056; 2499 vs 2481).
- **Tetris-B does not.** It moved from *blind* to *measured-and-late* —
  true 8657, detected 8819, delta 162 against a 120-frame tolerance.
  Widening the tolerance would manufacture the hit. The row stays a
  real `NO_CLEAR`. Its in-tolerance `oam_quiesce` signal at +28 was
  refused 60 times by Rule 5, correctly: "the sprites went away" is
  what a death looks like.

**Gate (b) is 1 of 2, and that is the honest score.**

---

## The binding rule

The 2026-08-23 process audit already named this defect class — *an
assay with no positive control* — and built a preflight that enforces
it forward on **trainer arms only**. Nobody ran the rule backward over
the ledger. **Every finding of this week was reachable from that
sentence three days earlier.**

> An assay with no positive control applies to **discovery
> instruments, gates, detectors and clear predicates**, not only to
> trainer arms. **Every ledger entry citing an absence must name the
> positive that same instrument returned on that same profile — or say
> VOID.**

Instruments already meeting the rule, and the templates for the rest:
`scripts/clear_reachability.py` (poses the question verbatim in its
docstring; mutation-tested; refuses to certify a hook that cannot
fire), `tests/test_purity_quarantine_sweep.py` and
`tests/test_no_new_name_dispatch.py` (both refuse to certify an empty
scan), `scripts/odometer_cert.py`, the ReDo forced-recycle sweep, and
`scripts/anti_vacuity_scan.py` + `tests/test_anti_vacuity_gates.py`.

**The recommended next step is not another wave.** It is re-running all
38 onboarding gates under HEAD's D1/D5/D6 fixes and re-deriving every
classification from the output — because the instrument that produced
all 38 has since been demonstrated to pass on a constant (zero-range
camera), on a corpse (post-death tails, 28 of 45 profiles), and to
delete the finding that would have contradicted its own verdict before
writing the receipt. Secondary: the `SOUND_` prefix should be
mechanically refused wherever no gate on the **shipped** progress axis
passes. That alone catches six games.

---

## Verdict

**GOOD, with a systematic gap in one direction.** Nothing in this file
is fabricated, no positive was withdrawn, and no CONFIRMED clear,
honest-protocol rate or purity finding was disturbed. The corrections
are almost entirely *scope* corrections — entries claiming a little
more than their own receipts did.

What the audit cannot say is that the project knows what it has a
receipt for. **11 of 42.** The gap between those two numbers is not a
credibility problem; it is an accurate description of an early-stage
research corpus that has been more diligent about banking receipts than
about checking whether the instrument that produced them could ever
have said no. The receipts are real. Most of them are receipts for the
instrument, not for the game.
