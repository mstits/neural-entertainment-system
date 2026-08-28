# ReDo actually fires — the first execution of the Deep Research prescription

2026-08-27. Receipts: `runs/v30_premise_falsifier_2026-08-27/`,
`runs/redo_fires/T0/`. Registration under test:
`docs/proposals/V30_REDO_ARMED_2026-08-27.md`.
**Verdict: VOID. VOID is not FAIL.**

---

## 0. The finding, stated first

**The Deep Research round of 2026-08-25 mandated ReDo (Sokar et al. 2023) as
the fix for a diagnosed plasticity loss, and in both campaigns that were
supposed to test it — v27 and v28 — the treatment ran an order of magnitude
below its own firing threshold and never once fired. It recycled zero
neurons across all eight runs. The plasticity-loss hypothesis has therefore
been UNTESTED, not refuted, since the day the prescription was written. This
document reports the first time the intervention was actually executed.**

Three independent receipts establish the inertness, all of them banked before
this work started:

| receipt | what it says |
|---|---|
| Telemetry | `dormant fc1 0/N fc2 0/32 recycled 0 cum 0` on every one of ~2,000 per-iteration checks across **all 8 runs** (v27 and v28), zero skipped iterations, zero `[redo] disabled` lines |
| Config | `configs/mario_1_1_v27_seed{0..3}.yaml` and `mario_1_1_v28_seed{0..3}.yaml` all declare `redo_tau: 0.025` |
| Isolation sweep | `runs/v27_fresh_recovery/preflight/redo_forced/` — 0 recycles at τ 0.05/0.10/0.15/0.20; first firing at τ 0.25 |

Re-verified here by running the new gate against the banked logs directly:

```
$ .venv/bin/python scripts/redo_arm_gate.py checkpoints/mario_1_1_v2[78]_*/run.log
  ...
  checks=250 recycle_events=0 cum_recycled=0 first_recycle_iter=None
  enabled_taus=[0.025] saw_disabled=False
  - cum_recycled == 0 over 250 dormancy checks — ReDo never fired, so this run
    is VOID, not FAIL, and may not be cited for or against the plasticity-loss
    hypothesis
9/9 log(s) VOID — no verdict may be issued for them.   [exit 2]
```

Nine logs: the eight campaign runs plus the preflight. All VOID.

## 1. The correction — "ReDo mathematically guarantees" guaranteed nothing

`docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md:526` registered this as the
FAIL-branch inference, quoting the DR verbatim:

> **FAIL (<= 0.767)**: "Because ReDo mathematically guarantees that all 48k
> parameters were active and non-dormant when the agent encountered the
> recovery bands, the failure cannot be attributed to primacy bias, in-run
> plasticity loss, or warm-start interference ... the 48k capacity represents
> a hard, fundamental ceiling"

**That sentence is void as written, and it is corrected here.** ReDo
guaranteed nothing about the v27 or v28 parameter budget, because ReDo did
not run in v27 or v28. The mechanism was declared, instantiated, logged, and
inert. A guarantee that rests on an intervention that never fired is not a
weak guarantee — it is not a guarantee.

Two narrower statements do survive and replace it:

1. **The dormancy *statistic* was measured and read zero throughout.** Under
   the repo's Definition-1 implementation (`src/training/redo.py:67-82`,
   layer-mean normalization), no unit in either layer scored at or below
   0.025 at any point in either campaign. So plasticity was *measured-intact
   by that statistic*, not *actively maintained by the treatment*. AMENDMENT
   1's B2/B5 registered exactly this fallback in advance, and it earns its
   credit.
2. **A statistic reading zero because the threshold is unreachable is a much
   weaker fact than the same statistic reading zero because the network is
   healthy.** §5 below shows the fc2 dormancy-score minimum settles at
   **0.079–0.127** once training is past ~iter 5 — an order of magnitude
   above 0.025 and never approaching it. The v27/v28 zeros are consistent
   with a perfectly healthy trunk *and* with a badly degraded one; the
   threshold could not distinguish them.

`CLAIMS.md` already carries the first half of this correction (ADDENDUM at the
v28 secondary finding). This document supplies the second half: the reason the
telemetry read zero is architectural, and it removes most of the reassurance
the zeros were carrying.

## 2. What does NOT change — v27 and v28 are not retracted

**Neither verdict is retracted. Both stand as measured.**

* **v27 FRESH-RECOVERY: FAIL, best-of-4 0.530** (re-adjudicated 0.500 under
  the corrected selector, `runs/v27_readjudication_2026-08-27/`).
* **v28 CAPACITY: FAIL, best-of-4 0.670** against the same 0.767 bar.

Every episode was run, every receipt is real, both reproduce. The numbers are
not in question and no headline claim moves.

**What changes is the interpretation.** Each campaign registered two variables
and delivered one:

| campaign | registered variables | variables that actually executed |
|---|---|---|
| v27 | merged 785-rung ladder **+ ReDo** | merged 785-rung ladder |
| v28 | capacity 48k→72k **+ ReDo** | capacity 48k→72k |

So:

* **v27 FAIL is a real result about the merged fresh-run curriculum.** It is
  not evidence about plasticity loss.
* **v28 FAIL is a real result about parameter budget.** It is not evidence
  about plasticity loss.
* **Neither may be cited, in either direction, on the plasticity-loss
  hypothesis.** That prohibition is now enforced in code
  (`scripts/redo_arm_gate.py`), not in prose.

One consequence worth naming: the *inertness that voids v27 as a ReDo test is
exactly what makes v27 a valid ReDo-off control* for any future armed arm —
identical ladder, recipe, seeds, iteration count and `num_envs`, with a
treatment that provably did nothing. That is the one piece of value the defect
returns.

## 3. The first real test — what ran

Five arms, ~42 minutes of wall clock, all on
`configs/mario_1_1_v27_seed0.yaml` verbatim with only `redo_tau` (and, in one
arm, `tile_hidden_dim`) changed. `--no-resume --no-supervise --strict-config`,
CPU, `num_envs 60`, 20 iterations each (26 for the control, which raised).

| arm | τ | width | cum_recycled | events | first fire | median agree | median trunk frac/event |
|---|---|---|---|---|---|---|---|
| `pilot_tau0.25_h64` | 0.25 | 64/32 | **353** | 19/20 | iter 1 | 0.8564 | **0.625** |
| `pilot_tau0.25_h96` | 0.25 | 96/32 | 342 | 19/20 | iter 1 | 0.9011 | 0.594 |
| `pilot_tau0.15_h64` | 0.15 | 64/32 | 176 | 16/20 | iter 4 | 0.9501 | 0.375 |
| `control_tau0.025_h64` | 0.025 | 64/32 | **0** | 0/26 | never | — | — |
| `armed_check_inert (8env)` | 0.025 | 64/32 | 0 | 0/26 | never | — | — |

**The narrow premise holds. ReDo fires, and it changes the network
substantially.**

* Iteration 1 of the τ=0.25 arm reproduces the banked 2-iteration sweep
  **bit-exactly**: `fc2 5/32 recycled 5 cum 5 agree 0.8142 max_dlogit
  0.288576`.
* The matched pair (τ=0.25 vs τ=0.025, same seed, same code — ReDo draws its
  sample batch *before* testing dormancy, so RNG consumption is identical) is
  **bit-identical at iterations 0 and 1** (`mean_return 676.5` then `660.3`,
  `entropy 1.7808` then `1.7513`, to the digit in both logs) and permanently
  diverges after the first recycle.
* Policy entropy holds **1.65–1.71** in the treatment while the control decays
  to **1.3463** by iter 19 — a sustained ~0.31 nat gap, which is the
  plasticity signature the DR predicted.

This is not the "armed but does nothing" case. The treatment is loud.

**The premise that fails is the DOSE.**

## 4. VOID — two independent grounds

### Ground 1: there is no number to judge

The registered bar is **Θ ≥ 0.80 PASS / Θ ≤ 0.767 FAIL**, where **Θ =
best-of-4 over seeds of the cross-fit split-sample honest clear rate** —
`score_A` = honest rate on eval seed 1 at the checkpoint argmax'd on eval seed
0, `score_B` the mirror, seed score = their mean, 100 scoring episodes per
seed, every episode used exactly once as a scoring episode and never for the
selection that chose its own checkpoint.

**That estimator was never applied, because there is nothing to apply it to.**
Grepped across all five arm logs: **0** occurrences of `clear_rate`, **0** of
`honest`, **0** eval-seed lines. No v30 checkpoints exist. No candidate ladder
was walked. Θ does not exist for the treatment arm or for any control.

Issuing PASS or FAIL against a bar with no measurement would be the exact
fabrication this registration forbids. **The bar does not move and was not
moved: 0.80 and 0.767 stand untouched for a future arm.** One good consequence
— no maximum was read off any table, so the registered 0.05 winner's-curse
budget remains unspent and intact.

`estimator_correct = false` here means **never executed**, not *executed
wrongly*. The cross-fit machinery, the 24-checkpoint candidate set, the
100-pooled-episode protocol and the es2 confirmation rule all stand unused and
unmodified.

### Ground 2: every treatment arm is VOID on dose, independently

From **iteration 5 onward**, τ=0.25 re-initializes a **median 20 of 32 trunk
units — 62.5% — every single iteration.** At width 96 it is 19/32 (59.4%). At
τ=0.15 it is 12/32 (37.5%).

The registration's own **"RISK I REFUSE"** clause describes this regime
verbatim, at τ=0.50:

> at 0.50 it takes 18–23 of 32 **every iteration, repeating the same index
> set** … that is not recycling dormant units, it is a per-iteration partial
> reset of two-thirds of the trunk, which is precisely the "network reset"
> family the DR ruled **INCOMPATIBLE** in its own §4. A FAIL under that dose
> would be uninterpretable.

**The registered operating point lands in the registered forbidden regime.**
20 of 32 is inside 18–23 of 32. The registration named the failure mode
correctly and then mis-measured where it starts.

Why the sweep missed it: it ran **two iterations, near orthogonal init**. At
iter 0 the fc2 score minimum is **0.2848** and only 5 of 32 units fall below
0.25. By iter 5 the minimum is **0.109** and 20 do. A threshold calibrated at
initialization becomes an ever-larger dose as training proceeds.

The gate confirms it, and voids all three treatment arms:

```
$ .venv/bin/python scripts/redo_arm_gate.py .../pilot_tau0.25_h64.log
  median_recycled_frac=0.625 max_recycled_frac=0.625 trunk_dim=32
  - median 62.5% of the worst-hit layer re-initialized per firing event ...
    over the 25% ceiling — this is a per-iteration partial network reset, not a
    surgical recycle ...
1/1 log(s) VOID — no verdict may be issued for them.   [exit 2]
```

## 5. "0.25 is the smallest τ that fires" is false — and the DR's own range is partly reachable

The registration's §1 concluded that the DR's prescribed 0.025–0.1 range "is
not merely unlucky on this stack; it is **unreachable**," and its §2 registered
0.25 as "the **smallest** threshold on the sweep that fires at all: the minimum
dose." Both claims rest on iterations 0 and 1. Measured over 20–26 iterations,
**both are wrong**, and the second correction is the interesting one.

The new per-layer tail telemetry (`min / p5 / p10` of the dormancy score,
logging only — see §7) on the **untreated** control:

| iter | fc2 min | fc2 p5 | fc2 p10 |
|---|---|---|---|
| 0 | 0.2848 | — | — |
| 5 | 0.1272 | 0.1325 | 0.1347 |
| 10 | 0.1215 | 0.1397 | 0.1507 |
| 15 | 0.1014 | 0.1174 | 0.1252 |
| 20 | 0.0999 | 0.1035 | 0.1070 |
| 24 | **0.0794** | 0.0822 | 0.0845 |
| 25 | 0.0811 | 0.0874 | 0.0898 |

Reading it directly:

* **τ = 0.15 fires.** Verified by running it: cum 176 over 20 iterations,
  16/20 events, first at iter 4. The control's own minimum falls below 0.15 on
  **22 of 26** iterations. The registered sweep declared 0.15 inert on two
  iterations of evidence.
* **τ = 0.10 — the TOP of the DR's own prescribed range — is reachable, and at
  a surgical dose.** The control's minimum falls below 0.10 on **10 of 26**
  iterations, all of them ≥ iter 16, while p5 stays at or above 0.10 until
  iter 21. On this trajectory τ=0.10 would recycle **nothing before ~iter 16
  and roughly 1–3 units (3–10% of the trunk) thereafter** — comfortably inside
  a 25% ceiling. The telemetry records only min/p5/p10, so the exact count at
  that dose is **not derivable** from these receipts and must be measured, not
  inferred.
* **τ = 0.025 is genuinely unreachable.** `0/26` control iterations reach even
  0.05; the run minimum is 0.0794. v27 and v28 could not have fired.

So the honest restatement of the registration's §1: **the *bottom* of the DR's
range is architecturally unreachable on a `Linear → LayerNorm → SiLU` trunk;
the *top* of it is not. It was unreachable at the two-iteration measurement
horizon the repo evaluated it on.** The registered escalation ladder
(0.25 → 0.30, ≥ 0.35 forbidden, **no lower rung**) therefore points away from
the only region the data says is viable.

**And no fixed τ is a stable operating point over the registered 250-iteration
budget.** The control's fc2 minimum drifts monotonically downward —
0.285 → 0.127 → 0.101 → 0.079 — and is *still falling* at iter 24 where the
measurement stops. The registration budgeted 250 iterations on a threshold
measured over 2.

## 6. The eighth vacuous gate — found live inside this registration, and closed

The registered safety valve **A4** voids the arm if median greedy-argmax
agreement over the first 50 recycle events falls below 0.60.

**A4 cannot detect the overdose it exists to catch.** Measured agreement:

| arm | trunk fraction recycled per event | median agree | A4 verdict |
|---|---|---|---|
| τ=0.25, h64 | **62.5%** | 0.8564 | PASSES |
| τ=0.25, h96 | **59.4%** | 0.9011 | PASSES |
| τ=0.15, h64 | **37.5%** | 0.9501 | PASSES |

The reason is architectural, not statistical: `recycle()` **zeroes the outgoing
actor and critic weight columns by construction**, so the network's output is
approximately preserved *however many hidden units were re-initialized*.
Agreement is structurally insensitive to the dose. A4 is a check that cannot
fail on the failure mode it names.

That makes it **the eighth vacuous gate in this repo — and it was written into
a registration whose entire brief was the previous seven.** It is worth stating
plainly: the discipline did not fail for lack of attention. It failed because
the check measured the wrong quantity, and the wrong quantity was the one the
mechanism was designed to hold constant.

**Closure, with the revert-verified failure shipping alongside it.**
`scripts/redo_arm_gate.py` gains **V6**: the median recycled fraction *of the
worst-hit layer* per firing event must be ≤ 0.25. Applied to the arms above it
VOIDs all three, with the reason printed.

Anti-vacuity receipts, executed rather than asserted (`tests/test_redo_armed_gate.py`,
22 tests):

* delete the `cum_recycled == 0` branch → **10/22 fail**
* delete the V6 over-dose ceiling → **3/22 fail**
* pool fc1+fc2 instead of taking the worst-hit layer → **5/22 fail**

That last one is the vacuity the gate's own author introduced and then caught:
V6's first version pooled both layers and reported 20/96 = 21% (**passing**)
for an event that reset 20 of 32 trunk units = 62%. **fc1 never goes dormant**
(§8), so a pooled denominator is permanent ballast that can never contribute
to the numerator. The fix is `max` over layers, not a sum.

## 7. The armed check works — verified by running the inert case, twice

Three places, because one place is how a vacuous gate gets written.

**(1) In-run deadline.** `src/training/trainer.py`, module constant
`_REDO_ARM_DEADLINE_ITERS = 25` — **hardcoded, deliberately not a config key**
(20 declared keys in the flagship recipe never executed; a new declared key is
a new way to be inert). Placed *after* the whole ReDo hook rather than inside
the `_rd is not None` branch, so an off-cadence or skipped iteration cannot buy
past it.

Live proof, not a unit test — the full 60-env v27 recipe at v27's own τ=0.025:

```
18:08:41 [ERROR] train_game: [supervisor] trainer crashed (attempt 1/0):
RuntimeError: [redo] VOID: armed at tau=0.025 but cum_recycled==0 after 26
iterations. ... This run is VOID, not FAIL. Do not issue a verdict.
18:08:41 [ERROR] train_game: [supervisor] restart budget exhausted — aborting
```

The log contains **zero** verdict, eval or `clear_rate` lines. Reproduced on a
fast 8-env config (`armed_check_inert_case_VOIDED.log`, same abort at iter 26).
Cost of a mis-armed arm: **~10.6 minutes**, against the **7 h 10 m per arm**
that v27 and v28 each burned inert.

**(2) At verdict time.** `scripts/redo_arm_gate.py` — conditions V1 (τ equals
the registered operating point), V2 (no `[redo] disabled`), V3
(`cum_recycled > 0`), V4 (≥10 events and ≥20 units, closing the "technically
fired once" loophole), V5 (median agree ≥ 0.60), V6 (median trunk fraction ≤
0.25). Exit 0 ARMED, exit 2 VOID. **The words PASS and FAIL do not appear on
any output path.** Applied to the historical v27 seed-0 log it prints
`VERDICT: VOID (redo never fired)` and exits 2 — the new gate reproduces the
correct verdict on the failure that motivated it.

**(3) In the receipt.** `run_manifest.json` gains `redo_tau`,
`redo_cum_recycled`, `redo_recycle_events`, `redo_first_recycle_iter`,
`redo_median_agree`, so a later reader cannot mistake an inert run for a
tested one.

**Standing rule registered by this work:** *every preflight condition is
evaluated at exactly the registered operating point; a preflight that passes at
any other τ voids the arm it certified.* This is the structural fix for the V7
defect, where the arming pilot ran at τ=0.50 — **20× the registered operating
point** — while the campaign ran at 0.025. V6's τ check enforces it: the τ=0.15
arm is voided partly *for being at the wrong τ*, exactly as intended.

**Telemetry added** (`src/training/redo.py`, `dormancy_scores` /
`score_tail`): the `[redo] iter` line now carries `min/p5/p10` of the per-unit
dormancy score per layer. Pure logging — no RNG consumption, no behavior
change, `redo_enabled: false` byte-identity unaffected. Every finding in §5 is
readable straight off the log because of it, and its absence is precisely why
the v27/v28 telemetry could establish "ReDo never fired" but not "how close did
it get."

## 8. Mechanism findings, banked at ~42 minutes of compute

1. **There is no dormancy substrate in fc1 at all.** 0 of 64 and 0 of 96 units
   dormant at every τ from 0.025 to 0.25, across all **86 measured
   iterations**; fc1's score minimum never falls below **0.309**. All ReDo
   activity is confined to the 32-unit trunk. "Recycle dormant neurons in a
   48k-parameter network" is, on this architecture, **"periodically
   re-initialize part of a 32-unit bottleneck."** Width does not change this —
   the h96 arm behaves the same, because the dormancy lives in the trunk both
   widths share.
2. **ReDo's identity-preservation guarantee does not hold under pre-activation
   LayerNorm.** At τ=0.25 a firing event changes the greedy argmax on ~14% of
   sampled states (agree 0.856 median, 0.702 min) with `max_dlogit` to 0.802.
   The DR's V7-era criterion of `agree ≥ 0.98` is unachievable at any firing τ
   on this stack. Banked as a finding, not worked around.
3. **The dormancy tail drifts monotonically downward through training** and had
   not converged at iter 24. Any fixed-τ registration is therefore
   mis-specified over a 250-iteration budget by construction.
4. **Directional only, n=1, explicitly not a verdict:** at iter 19 the control
   had advanced **14** ladder rungs (τ=224/784) against the treatment's **12**
   (τ=304/784), with `mean_return` 692.2 vs 405.8 and 10 clears vs 3. No hint
   of a large positive lever. This is one seed, 20 iterations, no honest eval,
   and it licenses no claim in either direction.

## 9. What the registered n could and could not have concluded

Recorded here so it cannot be re-derived later as a post-hoc rescue, and
because the binding caveat is stranger than usual.

**The binding one: n_effective = 0 scored seeds.** No campaign launched, no
ladder walked, no honest episode run. A VOID carries strictly less inferential
content than even an underpowered FAIL. **The reported treatment and control
figures of 0 mean NOT MEASURED, not "measured a zero clear rate."** Reading
those zeros as performance numbers would be a fabrication.

**The caveat that would have bound had the campaign run**, preserved from the
registration §10: across the 8 banked runs the per-seed honest rate is
**bimodal** — 2 of 8 collapsed below 0.10 (v27 seed0 0.03, v28 seed2 0.09) and
6 sit in 0.46–0.67 (mean 0.527, sd ≈ 0.08). Max single seed ever observed:
**0.67**. Therefore at n=4:

* Powered only for a **≥ +0.25** shift in the per-seed rate. A real but modest
  lever of +0.05 to +0.15 — the size the recovery assay's own ceiling analysis
  (0.767 → 0.83–0.85) says is even *available* — would be **invisible**. A FAIL
  would rule out "ReDo is a big lever"; it could never rule out "ReDo is a
  small lever."
* PASS requires a seed **0.13 above the best of all 8 prior seeds**, so absent a
  large effect FAIL is near-certain by construction.
* With a ~25% per-seed collapse rate, best-of-4 is a max statistic whose null
  already reaches ~0.50. **Any best-of-4 in 0.50–0.65 is indistinguishable from
  seed noise** and may never be reported as "improvement over v27's 0.500."
* n=4 cannot separate "ReDo helped nothing" from "ReDo helped and hurt in equal
  measure."
* **n=4 cannot confirm Hypothesis B (capacity)** regardless of what the DR's
  §6.2 asserts, because that inference assumes an identity-preserving
  intervention this architecture measurably does not provide (§8.2).

**The DR's §6 fork is not entered.** A VOID takes neither branch.

## 10. What a corrected registration needs

All cheap; none of it the 12-hour budget.

1. **Replace the fixed τ with a rank-based dose.** Recycle the bottom-k units
   of fc2 per check, k ≈ 2–5 (6–16% of the trunk), with τ *derived* from k.
   This is stable under the monotonic drift of §5 where no fixed threshold is,
   and `src/training/redo.py` already computes the per-unit score the rule
   needs. ~1 h.
2. **Port the V6 over-dose ceiling into the in-run abort.** Today only the
   post-hoc gate carries it, so a live mis-dosed run would still burn to iter
   250. It belongs beside `_REDO_ARM_DEADLINE_ITERS` in
   `src/training/trainer.py`, firing at iter ~5–10.
3. **Re-sweep the dose over 40–60 iterations, never 2, before arming
   anything.** Half the telemetry needed is already on disk.
4. **Test τ ≈ 0.10 explicitly** — the top of the DR's own prescribed range,
   which §5 shows is reachable at a surgical dose after ~iter 16 and which the
   registered ladder forbids reaching. Executing the prescription as written
   may be closer than the registration concluded.

Estimated added cost: **under 2 hours.** It converts a near-certain
uninterpretable FAIL into a test that can actually speak to the hypothesis.

## 11. Standing prohibition

**Nothing in v27, v28, or v30 may be cited as evidence for or against the
plasticity-loss hypothesis.** v27 and v28 because the treatment was inert;
v30 because it produced no scored number and every treatment arm is VOID on
dose. The hypothesis stands exactly where the DR left it on 2026-08-25:
diagnosed, prescribed, and untested.

## 12. Receipts

| path | what |
|---|---|
| `runs/v30_premise_falsifier_2026-08-27/` | 5 run logs, `analyze.py`, `analysis.txt`, 4 configs — the primary evidence |
| `runs/redo_fires/T0/{README.txt,verdict.json}` | the T0 adjudication against the registration (commit `439b87f`) |
| `docs/proposals/V30_REDO_ARMED_2026-08-27.md` | the registration this document adjudicates |
| `scripts/redo_arm_gate.py` | the verdict-time gate, V1–V6 |
| `tests/test_redo_armed_gate.py` | 22 tests, three revert-verified failures |
| `src/training/trainer.py` | `_REDO_ARM_DEADLINE_ITERS`, the in-run VOID |
| `src/training/redo.py` | `dormancy_scores` / `score_tail`, the tail telemetry |
| `checkpoints/mario_1_1_v2[78]_*/run.log` | the 8 inert campaign logs, VOIDed by the gate |

Suite at the time of writing: **5829 passed, 30 skipped, 3 xfailed**, plus the
one known-environmental failure (`tests/test_night2_runner.py::test_dry_run_passes_live`),
which is left alone. Baseline 5807 + 22 new gate tests = 5829 exactly.
