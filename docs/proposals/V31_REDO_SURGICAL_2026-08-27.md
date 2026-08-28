# v31 — ReDo at the surgical dose. Pre-registration.

Registered 2026-08-27, **before any v31 compute is spent.** Every numeral
below is fixed as of this commit. A change to any of them after compute
starts voids the run; this document is the goalpost.

---

## 0. What this is, and what it is not

**This is a NEW experiment. It is not a rung of v30's escalation ladder.**

`docs/proposals/V30_REDO_ARMED_2026-08-27.md` registered ReDo at
tau = 0.25 with exactly one escalation step (0.25 -> 0.30) and the
explicit clause **"tau >= 0.35 is forbidden by this registration"**, plus
a stopping rule if 0.30 also voided. That ladder was walked and stopped:
the T0 premise-falsifier (`439b87f`, `d02987f`, adjudicated in
`docs/research/REDO_ACTUALLY_FIRES_2026-08-27.md`) found the registered
operating point lands **inside v30's own forbidden "network reset"
regime** — 20 of 32 trunk units re-initialized every iteration from
iter 5 — and returned **VOID on dose** for every treatment arm. The
NO-GO was honoured; no campaign was launched; ~12 h was saved.

v30's ladder pointed **upward**, away from the only region its own
telemetry says is viable. Climbing further up it after seeing that
result would be a moving goalpost. Registering a **new** experiment at a
**lower** tau, with its own ladder, its own aborts, and its own stopping
rule, written before any compute, is not. This document does that, and
says so out loud so a later reader cannot mistake it for goalpost drift.

**Inherited verbatim from v30 and NOT reopened:** the Theta bar (0.80 /
0.767), the cross-fit split-sample estimator, the honest protocol, the
0.05 winner's-curse budget, the `[0.80, 0.85)` confirmation rule, the
verdict-time gate `scripts/redo_arm_gate.py` conditions V1/V2/V3/V5/V6,
and the standing rule below.

**Standing rule, inherited and binding:** *every preflight condition is
evaluated at exactly the registered operating point; a preflight that
passes at any other tau is VOID.* (§5.1 registers the one deliberate
exception and explains why it is not an exception at all.)

**Standing prohibition, inherited and unchanged:** nothing in v27, v28
or v30 may be cited as evidence for or against the plasticity-loss
hypothesis. v27/v28 because the treatment was inert; v30 because it
produced no scored number and every treatment arm is VOID on dose.

---

## 1. The evidence this registration is built on

All of it is banked, all of it measured, none of it new compute.
Receipts: `runs/v30_premise_falsifier_2026-08-27/` (5 logs, 86 measured
dormancy checks across 4 trajectories at 3 taus).

### 1.1 The dormancy lives in a 32-unit trunk, and only there

`fc1` is **0 of 64 and 0 of 96 dormant at every tau from 0.025 to 0.25
across all 86 measured iterations**; its score minimum never falls below
**0.3086**. Every ReDo event in every arm was an `fc2` event. On this
architecture "recycle dormant neurons in a 48k-parameter network" means
**"periodically re-initialize part of a 32-unit bottleneck."** Width does
not change it: the `tile_hidden_dim: 96` arm behaves identically, because
the dormancy is in the trunk both widths share.

Consequence for this registration: **the dose is a fraction of 32.** One
unit is 3.125%. The whole experiment lives inside a range of about ten
integers, which is why the dose must be registered as an abort and not
as a hope.

### 1.2 The untreated tail, and where tau = 0.10 sits in it

Untreated control (`control_tau0.025_h64_VOIDED.log`, 26 checks, fc2
score min / p5 / p10):

| iter | min | p5 | p10 |
|---|---|---|---|
| 0 | 0.2848 | 0.3080 | 0.3224 |
| 5 | 0.1272 | 0.1325 | 0.1347 |
| 15 | 0.1014 | 0.1174 | 0.1252 |
| 20 | 0.0999 | 0.1035 | 0.1070 |
| 24 | **0.0794** | 0.0822 | 0.0845 |
| 25 | 0.0811 | 0.0874 | 0.0898 |

* The minimum crosses 0.10 at **iter 16** and is below it on **10 of 26**
  iterations, all of them >= 16.
* By iters 20-25, `p5 <= 0.10` on **5 of 6** iterations and
  `p10 <= 0.10` on **4 of 6** — so by iter ~24 the untreated network
  already has roughly **4-5 units** beneath 0.10.
* `tau = 0.025` is genuinely unreachable: **0 of 26** iterations reach
  even 0.05. v27 and v28 could not have fired.
* The tail **drifts monotonically down and had not converged** at the
  measurement horizon.

### 1.3 The equilibrium law — the single most decision-relevant number here

Under treatment the dormant count does **not** run away. It climbs for a
few iterations after the first fire and then sits at a fixed point:

| arm | tau | units/event, med | frac of trunk | last-10 firing events |
|---|---|---|---|---|
| `pilot_tau0.15_h64` | 0.15 | 12 / 32 | **0.375** | 0.375 (flat) |
| `pilot_tau0.25_h64` | 0.25 | 20 / 32 | **0.625** | 0.625 (flat) |
| `pilot_tau0.25_h96` | 0.25 | 19 / 32 | **0.594** | 0.594 (flat) |

Two h64 points fall on a line through the origin to four decimal places:

> **equilibrium recycled fraction ~= 2.5 x tau** (h96 gives 2.375 x tau).

**Registered prediction, stated before the measurement:** at
**tau = 0.10** this law predicts an equilibrium dose of
**0.238-0.250 of the trunk, i.e. 7.6-8.0 of 32 units.**

That is **exactly on the 0.25 ceiling.** This registration does not
pretend otherwise. The strict inequality in the ceiling (`> 0.25` trips,
`= 0.25` does not) puts 8/32 on the admissible side and 9/32 on the void
side — a one-unit margin. **Phase M (§5.2) exists precisely because this
is a knife edge**, it costs 0.6 h against 7 h of training, and the
de-escalation rung (tau = 0.075, predicted 6/32) is pre-registered in
§9 for the branch where the knife edge falls the wrong way.

The prediction is itself falsifiable and is registered as a secondary
finding either way: if Phase M measures an equilibrium fraction inside
[0.20, 0.30] the two-point law is corroborated; outside it, the law is
refuted and recorded as refuted.

### 1.4 The eighth vacuous gate, and why it is not the dose gate

v30's A4 aborted on median greedy-argmax agreement < 0.60. Measured:

| arm | trunk fraction per event | median agree | A4 |
|---|---|---|---|
| tau 0.25 h64 | **62.5%** | 0.8564 | PASSES |
| tau 0.25 h96 | **59.4%** | 0.9011 | PASSES |
| tau 0.15 h64 | **37.5%** | 0.9501 | PASSES |

`recycle()` zeroes the outgoing actor **and** critic columns by
construction, so the network's output is approximately preserved
*however many* hidden units were re-initialized. Agreement is
structurally insensitive to the dose. **A4 could not fail on the failure
mode it named.**

**In this registration the agreement condition is retained as
`redo_arm_gate.py` V5 (median agree >= 0.60) and is explicitly DEMOTED:
it is a reported diagnostic, never the dose gate.** The acting dose gate
is §3. Writing that sentence down is the fix; leaving A4 in place as if
it protected the dose is how a ninth vacuous gate gets written.

---

## 2. tau — the operating point

## **tau = 0.10.** Registered.

Otherwise the v27 block verbatim: `redo_enabled: true`,
`redo_check_every_iters: 1`, `redo_sample_batch: 4096`,
`redo_reset_optimizer_moments: true`, `tile_hidden_dim: 64`,
`tile_trunk_dim: 32`, `num_envs: 60`, 250 iterations. Width is 64/32
because **v27 is the ReDo-off control** and a single-variable Delta
requires identical width. (v30 §2 asserted the arm runs at 96/32; that
was an error — `configs/mario_1_1_v27_seed0.yaml` declares no
`tile_hidden_dim` and takes the schema default 64. v28, not v27, is the
96-wide recipe. Corrected here.)

**Justification.**

1. **0.10 is the top of the Deep Research prescription's own range
   (0.025-0.1).** v30 concluded that range was "unreachable" on a
   `Linear -> LayerNorm -> SiLU` trunk. That conclusion rested on two
   iterations at near-orthogonal init and is **false as stated**: the
   *bottom* of the range is unreachable (0/26 iterations reach 0.05); the
   *top* is not (min crosses 0.10 at iter 16 and stays below). v31 is the
   first execution of the prescription **inside its own prescribed
   range**.
2. **It is the lowest tau with direct evidence of firing.** The untreated
   control's min crosses 0.10 at iter 16 and is below on every subsequent
   iteration measured. The rung below (0.075) has **no** untreated
   crossing in 26 iterations (run-min 0.0794) and therefore carries a
   real VOID-NEVER-FIRED risk; three campaigns have already died of not
   firing, and one has died of over-firing. 0.10 is the point where both
   risks are smallest and both are pre-registered with a cheap detector.
3. **It is inside the surgical band on two independent trajectories.**
   The untreated control at iters 20-25 has ~1-5 units below 0.10
   (`p5`/`p10` straddle it). The tau-0.15 *treated* trajectory's `p10`
   sits at 0.068-0.103 from iter 6, i.e. ~3-4 units below 0.10. Both
   readings land in 3-13% of the trunk.
4. **The honest counterweight, stated in advance:** the equilibrium law
   of §1.3 extrapolates to 8/32 = 25.0%, right on the ceiling, and the
   two supporting readings in (3) come from an *untreated* trajectory and
   from a trajectory treated at a *different* tau. **The dose at exactly
   0.10 has never been measured.** Under the standing rule it must be
   measured at exactly 0.10 before training, which is Phase M, and the
   campaign may not start until it is.

**Forbidden by this registration, with the measurement that forbids it:**

* **tau >= 0.15** — measured 37.5% of the trunk per event at 0.15,
  post-hoc-VOID under V6 and in-run-VOID under §3. Not admissible at any
  point in this campaign, including as an escalation.
* **tau <= 0.05** — 0 of 86 measured iterations across all four
  trajectories reach 0.05. Provably inert; a run there is
  VOID-NEVER-FIRED by construction.
* **Admissible band: tau in (0.05, 0.15), and only the three rungs
  0.075 / 0.10 / 0.125 may be used** (§9).

---

## 3. The in-run dose ceiling — the abort this registration exists for

v30's over-dose ceiling (V6) is **post-hoc only**. A live mis-dosed run
still burns to iter 250 and is voided afterwards. The tail drifts
**down** across training, so a fixed tau that is surgical at iter 30 can
be a partial network reset at iter 200. That risk is the reason this
experiment could waste 7 hours, so the ceiling moves into the run.

### 3.1 The condition

At every dormancy check from the 10th check onward, let

```
frac_t = max( dormant_fc1 / hidden_dim , dormant_fc2 / trunk_dim )
M10(t) = median( frac_{t-9} .. frac_t )          # trailing 10 CHECKS
```

**If `M10(t) > 0.25`, raise `RuntimeError` immediately.** The run is
**VOID-OVERDOSE**, and — like the arming deadline — the launcher aborts
the **whole 4-seed sequence**.

* **Worst-hit layer, never pooled.** `fc1` never goes dormant (§1.1), so
  a pooled denominator is permanent ballast that can never contribute to
  the numerator: it reports 20/96 = 21% (passing) for an event that reset
  20 of 32 trunk units = 62%. This is the vacuity v30 caught in V6's own
  first draft; it is not reintroduced here.
* **Trailing median over CHECKS, not over firing events.** A median over
  firing events only is blind to how often the treatment fires; a median
  over the trailing 10 checks tracks the dose the network is actually
  taking.
* **0.25 exactly, inherited from V6.** The in-run abort and the
  verdict-time gate use the *same numeral* so they cannot disagree.
  Choosing a tighter in-run number would abort runs the verdict-time gate
  would have admitted; choosing a looser one would let V6-VOID runs burn
  to completion. V6 remains authoritative at verdict time; §3 is an
  early-abort, not a substitute for it.
* **Strict `>`.** 8/32 = 0.25 survives, 9/32 = 0.28125 aborts. Given
  §1.3's prediction of 7.6-8.0 units this is a one-unit margin, and it is
  registered as such rather than discovered later.

### 3.2 What happens when it trips

A trip is **not a FAIL and not a null result.** It is a positive
mechanism finding, and it is registered now so it cannot be written as a
rescue afterwards:

> *"On a `Linear -> LayerNorm -> SiLU` trunk, tau = 0.10 is surgical
> early and becomes a partial network reset by iteration K. No fixed
> dormancy threshold is a stable operating point over a 250-iteration
> budget on this architecture."*

with K reported. That statement **corroborates v30 §5's drift claim with
a live receipt** and is the strongest evidence this campaign can produce
short of a scored Theta. It licenses exactly one successor: the
**rank-based bottom-k dose** (recycle the bottom k units of fc2 per
check, k fixed, tau derived from k — v30 §10.1), which is stable under
monotone drift where no fixed threshold is. That successor is a **new
experiment with its own registration**, not a rung of this one, and this
campaign does not run it.

### 3.3 Anti-vacuity — the ceiling ships with its failure, executed twice

**Replayed against banked logs** (`tests/`, new cases; the per-iteration
counts are already on disk):

| banked log | tau | `M10 > 0.25` first trips at |
|---|---|---|
| `pilot_tau0.25_h64.log` | 0.25 | **iter 9** (M10 = 0.6094) |
| `pilot_tau0.25_h96.log` | 0.25 | **iter 9** (M10 = 0.5781) |
| `pilot_tau0.15_h64.log` | 0.15 | **iter 10** (M10 = 0.2969) |
| `control_tau0.025_h64_VOIDED.log` | 0.025 | **never** (26 checks) |
| `armed_check_inert_case_VOIDED.log` | 0.025 | **never** (26 checks) |

Three real positives and two real negatives on data the gate will
actually see, plus a synthetic surgical trace at 2-4 of 32 that must
never trip. **Deleting the ceiling must make these tests fail, and the
deletion must be executed and the failure count recorded** — the same
standard v30 applied to V6 (`3 of 22 fail`) and to the pooled-denominator
vacuity (`5 of 22 fail`).

**Executed live — Phase G, §5.1.** A real training run at tau 0.25 with
the ceiling armed must raise at iter ~9. Reasoning about the check is not
a receipt; running the failure is.

### 3.4 Cost of a trip

Firing is expected to start ~iter 16 and to equilibrate within ~10
iterations, so a dose breach is detected by ~iter 35: **~15 minutes**,
against 1 h 47 m per seed and 7 h 10 m per arm.

---

## 4. The arming floor — re-derived for a surgical dose

v30's floor was **>= 10 recycle events and >= 20 cumulative units** (gate
condition V4). That floor was written for a regime that fires **20 units
per iteration**; inherited unchanged into a regime that fires **2-4 units
per iteration**, it would be cleared in the first three firing
iterations of a 250-iteration run and would certify a treatment that was
active for 1% of training.

**The question, answered now, before any data:** *if ReDo at the surgical
dose fires 3 units in total across a full run, is that ARMED or VOID?*

> ## **VOID-MINIMAL-DOSE. Decided in advance.**
>
> Three units is **0.09 trunk turnovers** across 250 iterations. A
> treatment that touches the network nine-hundredths of one trunk's worth
> over seven hours cannot be the difference between an 0.50 and an 0.80
> honest clear rate, and a FAIL under it would carry exactly the
> non-informativeness that voided v27 and v28. The floor for a surgical
> dose must be **raised, not lowered**: when the per-event count is small,
> the **number of events** has to carry the sustainedness that the
> per-event count used to carry.

### 4.1 The registered floor — all four conditions, at exactly tau = 0.10

A seed is **ARMED** only if all of:

* **F1 — sustained.** `>= 40` firing iterations (checks with
  `recycled >= 1`) out of 250. That is 16% of training. Below it the
  treatment was a perturbation, not a regime.
* **F2 — turnover.** `cum_recycled >= 64 = 2.0 x trunk_dim`. Expressed as
  trunk turnover rather than a raw count, so the numeral is scale-free in
  trunk width. Two full trunk-equivalents is the smallest budget under
  which "the treatment re-initialized the trunk" is a true sentence.
* **F3 — distinctness.** `>= 6` distinct `fc2` indices recycled over the
  run, and **no single index accounting for more than 60% of all
  recycled-unit-events.** A treatment that resets the *same* two or three
  units every iteration forever is not a recycle; it is a permanent
  partial lesion of a 32-unit trunk, and its FAIL would be uninterpretable
  in exactly the way an overdose's is. v30 observed the repeating-index
  pathology at tau 0.50 (`fc2 = [1,2,4,5,7,9,13,16,...]` identical at
  iters 0-3) and no gate catches it today. **This is a new condition and
  it must ship revert-verified** (§10).
* **F4 — plus the inherited gate conditions** V1 (`[redo] ENABLED
  tau=0.1` and nothing else), V2 (no `[redo] disabled`), V3
  (`cum_recycled > 0`), V5 (median agree >= 0.60, demoted per §1.4), V6
  (median dose of the worst-hit layer <= 0.25).

Invocation, fixed now:

```
scripts/redo_arm_gate.py <seed run.log> --tau 0.10 \
    --min-events 40 --min-units 64 --max-frac 0.25 --min-agree 0.60
```

### 4.2 Is the floor live, or is it a ninth vacuous gate?

Checked against the range it will actually act on, which is the standard
this repo now requires:

* **Expected**: first fire ~iter 16, then firing on most checks -> ~230
  events and 460-900 cumulative units. F1/F2 clear by 5-10x. So the floor
  does **not** void the expected run.
* **Reachable failure**: a seed whose tail crosses 0.10 late — after
  ~iter 210 — produces < 40 events and is correctly VOID-MINIMAL-DOSE.
  Seed-to-seed variation in the crossing iteration is real (only one
  trajectory has ever been measured), so this is a live branch, not a
  decoration.
* **F3's failure is live too**: at a 2-4 unit dose with a self-sustaining
  dormant pool, a fixed index set is exactly the pathology to expect.
* **Phase M pre-checks the projection** (M4, §5.2) so a floor failure is
  caught in 0.6 h rather than after 7.

### 4.3 The arming deadline is raised to 40 iterations

`src/training/trainer.py::_REDO_ARM_DEADLINE_ITERS` is currently **25**,
chosen for a tau that fires at **iter 1** — 24 iterations of margin. At
tau = 0.10 the measured first crossing is **iter 16**, leaving **9**.
That is not enough margin against seed-to-seed variation in the crossing
iteration, and it would void legitimate seeds.

**Registered: `_REDO_ARM_DEADLINE_ITERS = 40` for this campaign**, which
restores the same 24-iteration margin. It remains a **hardcoded module
constant, not a config key** — 20 declared config keys in the flagship
recipe never executed, and a new declared key is a new way to be inert.
Cost of a mis-armed arm rises from ~10.6 min to **~17 min**, still two
orders of magnitude below the 7 h 10 m that v27 and v28 each burned
inert. The existing revert-verified deadline test is re-pointed at 40 and
must still fail when the deadline is deleted.

---

## 5. Preflights — both of them, and the standing rule

### 5.1 Phase G — the ceiling's own failure, executed live (~0.15 h)

Run `configs/mario_1_1_v27_seed0.yaml` at **tau = 0.25** for at most 12
iterations with the §3 ceiling armed. **It must raise `RuntimeError`
VOID-OVERDOSE at iter ~9.** If it does not raise, the ceiling is vacuous
and **the campaign does not start.**

**This is not a preflight condition on the arm, and it does not violate
the standing rule.** The standing rule governs conditions the *arm* must
satisfy — those are evaluated at exactly 0.10, in Phase M. Phase G
certifies that a *check* can fire, using the one dose measured to fire
it. Registered explicitly: **no result from Phase G may be cited as
evidence about the arm, about tau 0.10, or about the plasticity-loss
hypothesis.** It is an anti-vacuity receipt for a gate and nothing else.

### 5.2 Phase M — the dose, measured at exactly tau = 0.10 (~0.6 h)

`configs/mario_1_1_v27_seed0.yaml` verbatim at **tau = 0.10**, seed 0,
64/32, `num_envs 60`, **60 iterations**, `--no-resume --no-supervise
--strict-config`, with the recycled-index log line promoted to INFO
(§10). 60 rather than 40 iterations because the drift test (M3) needs two
widely-separated windows, and 20 extra iterations cost ~11 min against
7 h of training.

**GO requires all five.** Any failure is a NO-GO and the ladder in §9
applies. All are evaluated at exactly tau = 0.10.

* **M1 — fires.** `>= 1` recycle event within 60 iterations. Otherwise
  tau 0.10 is unreachable on a treated trajectory.
* **M2 — dose under the ceiling.** `M10(t) <= 0.25` at every check, i.e.
  the §3 in-run ceiling never trips during Phase M.
* **M3 — equilibrium, not drift.** median dose over checks 56-60 minus
  median dose over checks 26-30 must be `<= +2 units` (`+0.0625` of the
  trunk). The equilibrium law of §1.3 predicts ~0; the untreated
  control's *unrecycled* tail predicts a rise. If the dose is still
  climbing by more than 2 units over 30 iterations, it will breach the
  ceiling before iter 250 and there is no point spending 7 h to discover
  that.
* **M4 — arming projection.** `>= 8` firing iterations within the 60. At
  the expected crossing (~iter 16) this window contains ~44; 8 is a floor
  with genuine room to fail, and it is the cheap pre-check on F1.
* **M5 — distinctness.** `>= 4` distinct fc2 indices recycled in the
  window and no single index above 60% of recycled-unit-events — the
  cheap pre-check on F3.

**Phase M measures seed 0 only.** Seeds 1-3 are protected by the in-run
ceiling (§3) and the arming deadline (§4.3), not by Phase M. Phase M's
checkpoint is **discarded**; the campaign runs all four seeds from
scratch with `--no-resume`. The 0.6 h is the price of the measurement,
paid rather than saved by a resume path that has bitten this repo before.

**Also recorded from Phase M, whatever it decides:** the measured
equilibrium fraction at tau = 0.10, adjudicating §1.3's registered
prediction (corroborated if inside [0.20, 0.30], refuted otherwise).

---

## 6. The bar and the estimator — inherited, not reopened

* **PASS**: Theta >= **0.80**
* **FAIL**: Theta <= **0.767**
* **MARGINAL**: 0.767 < Theta < 0.80 — reported as MARGINAL, never as
  PASS, and it authorizes no follow-on claim and no follow-on campaign.

**Theta = best-of-4 over seeds of the cross-fit split-sample honest clear
rate.** Candidate set = the 24 saved
`vanilla_ppo_iter_{00010..00240}.pt`. Per seed:

* `score_A` = honest rate on eval seed 1, at the checkpoint argmax'd on
  eval seed 0;
* `score_B` = honest rate on eval seed 0, at the checkpoint argmax'd on
  eval seed 1;
* seed score = `(score_A + score_B) / 2`.

100 scoring episodes per seed; **every episode is used exactly once as a
scoring episode and never for the selection that chose its own
checkpoint.** Ties -> later iter, both directions reported.

**Honest protocol, immutable:** cold entrance, greedy, sticky 0.25,
jitter +/-16, 50 eps x eval seeds {0,1} = 100 pooled, `--eval-rng
per-episode`, `max_steps 1500`, `rom_sha256` on every receipt.
`warp_rate` must be 0.0.

**Winner's-curse budget 0.05**, reported as `Theta_adj = Theta - 0.05`. A
PASS landing in `[0.80, 0.85)` is flagged **"PASS within the measured
curse"** and requires a confirmation re-score of the winning seed's
selected checkpoint on a **third eval seed (es2, 50 fresh episodes,
registered now)** before it may be called PASS in any external claim.

**N is fixed at 4 and Theta requires 4 ARMED, scored seeds.** best-of-3
is a different statistic from the banked best-of-4 controls and from the
bar; comparing them would be an estimator mismatch. If fewer than 4 seeds
are ARMED and scored within the wall clock, the campaign is
**VOID-UNDERPOWERED**: no Theta is issued, and the per-seed numbers are
banked as mechanism receipts only, reported individually with full
receipts. Reporting fewer seeds honestly is exactly this; computing a
best-of-3 against a best-of-4 bar is not.

**Secondary — Delta, registered now so it cannot be a post-hoc rescue,
and it cannot produce a PASS.** `Delta = Theta - Theta_v27`, where
`Theta_v27` is v27's four banked seeds re-scored under the **identical**
cross-fit estimator (§8 pays for this; v27 has only 9 sparse receipts
today and no 24-point ladder — `runs/v27_readjudication_2026-08-27/`).

* `Delta >= +0.15` -> "ReDo at a surgical dose is a real lever on this
  stack" — a mechanism finding, reported as such, never a gate PASS.
* `Delta <= +0.05` -> "ReDo at a surgical dose is not a lever."
* in between -> indeterminate at n = 4; say so.

**The ReDo-off control is v27, unchanged, and no new control arm is
run.** ReDo was provably inert in v27 (0 recycles on every check of all
four seeds), so v27 *is* the control: identical merged 785-rung ladder,
recipe, width 64/32, seeds 0-3, 250 iterations, `num_envs 60`. The
inertness that voids v27 as a ReDo test is what makes it a valid ReDo-off
control. Residual risks accepted and named: **not a paired test** (ReDo
consumes numpy RNG at every firing, so the streams diverge at the first
recycle — four independent draws against four independent draws), and
**machine conditions differ across days**, mitigated by the throughput
band in §8 and flagged in the verdict if breached.

**Protocol note, recorded so it is never used as a rescue.** The 0.767
bar was measured at n=60, eval seed 0 only, 1 worker. Re-measured under
this registration's protocol the banked control reads **0.68**
(`runs/v27_readjudication_2026-08-27/readjudication.json`). **0.767 does
not move.** It is a numeral fixed pre-run; moving it would be a
fabricated result. The 0.68 figure is context and carries no verdict
weight.

---

## 7. Interpretation — and exactly what a FAIL licenses

### 7.1 PASS (Theta >= 0.80)

Plasticity loss in Sokar's sense **was** the barrier at 48k parameters on
1-1. Hypothesis A (consolidation) confirmed; Hypothesis B (the 48k budget
is a hard ceiling) falsified. ReDo at a surgical dose becomes a standing
element of the recipe. A PASS in `[0.80, 0.85)` requires the es2
confirmation re-score before any external claim.

### 7.2 MARGINAL (0.767 < Theta < 0.80)

Reported as MARGINAL. Licenses nothing. In particular it does **not**
authorize a follow-on campaign at another tau — that would be ladder
climbing after seeing the number, which is the thing this document
exists to prevent.

### 7.3 FAIL (Theta <= 0.767) — the licence, stated in advance

v30 ruled that a FAIL **at a damaging dose** licensed exactly one
sentence: *"plasticity loss in Sokar's sense is not the large lever
here."* That narrowness was a consequence of the confound: at 62% of the
trunk per iteration, "ReDo damaged a healthy network" and "plasticity was
not the barrier" are inseparable.

**At a surgical dose the confound is bounded by measurement, so the
licence is wider. Exactly this wide, and no wider:**

A v31 FAIL licenses, in writing:

1. *"At tau = 0.10 — inside the Deep Research prescription's own
   0.025-0.1 range — with the dose measured surgical (median <= 25% of
   the worst-hit layer per firing event, projected 6-25%, ceiling-armed
   in-run) and the treatment sustained (>= 40 firing iterations, >= 2.0
   trunk turnovers, >= 6 distinct units), recycling dormant units did not
   move the best-of-4 cross-fit honest clear rate above 0.767 on 1-1 at
   48k parameters."*
2. *"Dormant-neuron recycling is therefore not a **large** lever
   (>= +0.25 in the per-seed honest rate) on this stack."* — the one
   inference n = 4 is powered for.
3. **The un-confounding of v27 and v28.** Their FAILs were
   uninterpretable on plasticity because the treatment was inert. After a
   v31 FAIL at a firing, surgical, sustained dose, that specific
   confound is discharged: the treatment has now been executed properly
   and did not clear the bar. v27's FAIL becomes a clean result about the
   merged fresh-run curriculum and v28's a clean result about parameter
   budget, **each with the plasticity alternative tested rather than
   assumed away.** This is the widening: v30's FAIL licensed a sentence
   about the lever; v31's additionally licenses closing the confound.
4. **Retirement of the prescription.** The 2026-08-25 DR's ReDo
   prescription may be closed as **EXECUTED-AND-NEGATIVE** in the claims
   ledger and removed from the backlog. No previous FAIL could do this.

A v31 FAIL does **NOT** license, and any of these in writing is a
fabrication:

* *"Hypothesis B (the 48k budget is a hard ceiling) is CONFIRMED."* The
  DR's §6.2 inference assumes an identity-preserving intervention that
  this architecture measurably does not provide — median agree 0.85-0.97,
  never the DR's 0.98, at any firing tau (v30 §8.2). n = 4 also cannot
  confirm it.
* *"Plasticity loss does not occur on this stack."* The statistic was
  measured and the treatment applied; neither establishes absence.
* Any claim about a **small** lever (+0.05 to +0.15) — which is the size
  the recovery assay's own ceiling analysis (0.767 -> 0.83-0.85) says is
  even available, and which n = 4 cannot see in either direction.
* Any claim about **other** plasticity interventions — L2-init
  regularization, CReLU, weight churn, layer-norm resets, periodic
  full resets. v31 tests ReDo at one dose on one architecture.

### 7.4 VOID

VOID is not FAIL, takes no branch of the fork, and **enters no
best-of-N.** A VOID campaign licenses only the mechanism finding attached
to its specific void reason (§3.2 for the dose ceiling; §1.3's prediction
adjudication from Phase M in every case).

---

## 8. Seeds, budget, sequencing

**Seeds 0, 1, 2, 3 — four, 250 iterations, `num_envs 60`, width 64/32**,
mirroring v27 exactly so the control is matched and Theta is the same
statistic as the banked best-of-4. Not fewer (§6).

| phase | h |
|---|---|
| Phase G — ceiling anti-vacuity, live at tau 0.25, <= 12 iters | 0.15 |
| Phase M — dose measured at exactly tau 0.10, 60 iters | 0.60 |
| treatment training, 4 x 250 iters @ 25.4 s/iter | 7.05 |
| treatment honest ladder, 24 ckpt x 2 es x 4 seeds = 192 evals @ ~28 s, 8 workers | 1.50 |
| v27 control backfill — the same 192-eval ladder, for Delta | 1.50 |
| gate runs, verdict, receipts, manifest | 0.30 |
| slack | 1.90 |
| **hard ceiling** | **13.0** |

**Ordering is registered, and it is chosen so the wall clock cuts the
least important thing first:** Phase G -> Phase M -> seeds 0-3 ->
treatment ladder (Theta) -> v27 backfill (Delta) -> verdict. **At 10.5 h
elapsed, the v27 backfill is cut** and Delta is reported as NOT COMPUTED.
Theta, the gated statistic, is never the thing that gets cut.

**CPU sequencing with Lane A.** Lane A's on-policy rollout collection
runs `--workers 2` and finishes first. **Lane B starts Phase G only after
Lane A's collection process has exited** — sequenced by process
completion, never by polling. Training must log **25.4 +/- 2.5 s/iter**;
outside that band the arm-vs-control comparison is flagged in the
verdict. (The v30 pilot logged a median 34.0 s/iter under contention;
that is the number this band exists to exclude.)

---

## 9. The ladder, and the stopping rule

v30's ladder (0.25 -> 0.30, stop) is **retired**, not extended: it points
away from the viable region. This is v31's own ladder, fixed now, one
rung in each direction, with a written stop.

* **Rung 1 — tau = 0.10.** The registered operating point.
* **If Phase M fails M1 or M4** (does not fire, or fires too rarely to
  project past F1): escalate **once** to **tau = 0.125** and re-run
  Phase M in full at exactly 0.125. 0.125 is the largest rung strictly
  below the measured-overdose point of 0.15.
* **If Phase M fails M2 or M3** (dose at or over the ceiling, or still
  climbing): de-escalate **once** to **tau = 0.075** and re-run Phase M
  in full at exactly 0.075. Predicted equilibrium 6/32 = 0.1875 under
  §1.3's law.
* **If Phase M fails M5** (a fixed index set): **stop.** A lesion is not
  a dose problem and no rung fixes it.
* **Exactly one rung is taken, in exactly one direction.** If the second
  Phase M also NO-GOes, **the campaign STOPS** and banks:

  > *"On a `Linear -> LayerNorm -> SiLU` trunk there is no fixed dormancy
  > threshold that is simultaneously firing and surgical over a
  > 250-iteration budget. Fixed-threshold ReDo is mis-specified for this
  > architecture. The rank-based bottom-k dose (v30 §10.1) is the only
  > remaining form of the intervention — a new experiment, with its own
  > registration, not a rung of this one."*

* **A rung may only be taken on a Phase M NO-GO, before training.** Once
  seeds are launched, no tau change is admissible for any reason. A tau
  change after seeing a Theta, a Delta, or a partial ladder is a moving
  goalpost and voids the campaign.
* **tau >= 0.15 and tau <= 0.05 are forbidden by this registration**
  (§2), including as escalations.

---

## 10. Aborts

* **A1 — Phase G (~0.15 h).** The §3 ceiling must raise at tau 0.25 by
  iter ~9. If it does not, the ceiling is vacuous; **the campaign does
  not start.**
* **A2 — Phase M (~0.6 h).** M1-M5 at exactly tau 0.10. Any failure ->
  §9 ladder or STOP. **No training begins before Phase M returns GO.**
* **A3 — arming deadline (~17 min).** `_REDO_ARM_DEADLINE_ITERS = 40`;
  `cum_recycled == 0` at iter 40 -> `RuntimeError`, seed VOID, whole
  4-seed sequence aborts.
* **A4 — in-run dose ceiling (~15 min after equilibrium).** §3.1;
  `M10(t) > 0.25` -> `RuntimeError`, seed VOID-OVERDOSE, whole sequence
  aborts, finding banked per §3.2.
* **A5 — damage abort (~45 min).** After treatment seed 0 reaches iter
  100: if cumulative entrance rate < **0.10** *and* trailing rate <
  **0.10**, cancel seeds 1-3. v27's worst seed at iter 100 was 0.186
  cumulative / 0.23 trailing, so the bar sits below half of that on
  **both** axes and fires only on catastrophic degradation, never on seed
  noise. Verdict: *"ReDo at tau 0.10 is destructive on this stack."*
* **A6 — identity, DEMOTED.** Median greedy-argmax agree >= 0.60 is
  retained as gate condition V5 and reported, but it is **not** the dose
  gate and never aborts on its own. §1.4 is the reason; writing this down
  is the fix for the eighth vacuous gate.
* **A7 — Delta cut at 10.5 h.** §8.
* **A8 — wall clock 13.0 h hard.** At the ceiling, stop and report what
  is banked.
* **A9 — protocol integrity.** `warp_rate != 0.0`, any `--strict-config`
  rejection, any honest-protocol deviation, or any config drift from the
  registered operating point -> VOID.

---

## 11. VOID conditions, enumerated

**VOID is not FAIL. VOID enters no best-of-N and takes no branch of the
fork.**

| condition | verdict |
|---|---|
| `cum_recycled == 0` at iter 40 | VOID-NEVER-FIRED |
| `< 40` firing iterations, or `cum < 64` units | VOID-MINIMAL-DOSE |
| `< 6` distinct fc2 indices, or one index `> 60%` of unit-events | VOID-MINIMAL-DOSE (F3) |
| `[redo] ENABLED tau` != 0.10 (or the taken rung), or `[redo] disabled` present | VOID-WRONG-TAU / VOID-NOT-ARMED |
| in-run `M10(t) > 0.25`, or post-hoc V6 median dose `> 0.25` | VOID-OVERDOSE |
| median agree `< 0.60` | VOID-IDENTITY |
| fewer than 4 ARMED **and** scored seeds inside the wall clock | VOID-UNDERPOWERED (no Theta; per-seed banked as mechanism receipts) |
| `warp_rate != 0.0`, config drift, protocol deviation | VOID |

---

## 12. Code required before any compute, each with its executed failure

Nothing here is a research question; all of it is small, and none of it
may be skipped, because every one of these is a place a vacuous gate has
been written before.

1. **In-run dose ceiling** in `src/training/trainer.py`, beside the
   arming deadline: trailing-10-check median of the worst-hit layer's
   recycled fraction, `> 0.25` -> `RuntimeError`. Ships with the §3.3
   replay tests (3 positives, 2 negatives, 1 synthetic negative) **and**
   the executed deletion check with its failure count recorded.
2. **`_REDO_ARM_DEADLINE_ITERS = 40`** (§4.3), with the existing
   revert-verified deadline test re-pointed and still failing on
   deletion.
3. **Promote the recycled-index log line to INFO**
   (`src/training/trainer.py`, currently `log.debug`). Pure logging: no
   RNG consumption, no behavior change, `redo_enabled: false`
   byte-identity unaffected (`tests/test_redo_mechanism.py`).
4. **Gate condition F3 (distinctness)** in `scripts/redo_arm_gate.py`,
   parsing those index lines. Ships with a synthetic single-index trace
   that must VOID and a healthy multi-index trace that must ARM, plus the
   executed deletion check.
5. **Cross-fit reducer as code, not arithmetic in a document** — the
   estimator of §6 applied to the per-`(checkpoint, eval_seed)` receipts
   `scripts/eval_game.py` writes, with tests. v27's readjudication was
   done by hand; Theta and Delta must both come out of the same committed
   reducer or they are not the same estimator.
6. **`run_manifest.json`** carries `redo_tau`, `redo_cum_recycled`,
   `redo_recycle_events`, `redo_first_recycle_iter`, `redo_median_agree`,
   and (new) `redo_median_dose_frac`, `redo_distinct_fc2_indices`.

Suite baseline that must hold after all of it:
`.venv/bin/pytest tests/ -q --timeout=120` -> ~5850 passed, 30 skipped,
3 xfailed, plus the one known-environmental failure
(`tests/test_night2_runner.py::test_dry_run_passes_live`), which is left
alone.

---

## 13. What n = 4 cannot conclude — inherited, unchanged, restated

Recorded here so it cannot be re-derived later as a post-hoc rescue.

Across the 8 banked v27/v28 runs the per-seed honest rate is
**bimodal**: 2 of 8 collapsed below 0.10 (v27 seed0 0.03, v28 seed2 0.09)
and 6 sit in 0.46-0.67 (mean 0.527, sd ~0.08). **Max single seed ever
observed: 0.67.** Therefore at n = 4:

* Powered only for a **>= +0.25** shift in the per-seed rate. A real but
  modest lever of +0.05 to +0.15 — the size the recovery assay's ceiling
  analysis says is even available — is **invisible**.
* PASS requires a seed **0.13 above the best of all 8 prior seeds**, so
  absent a large effect FAIL is near-certain by construction. That is
  what the gate is for, and why Delta exists to carry the informative
  signal.
* With a ~25% per-seed collapse rate, best-of-4 is a max statistic whose
  null already reaches ~0.50. **Any best-of-4 in 0.50-0.65 is
  indistinguishable from seed noise** and may never be reported as
  "improvement over v27's 0.500."
* n = 4 cannot separate "ReDo helped nothing" from "ReDo helped and hurt
  in equal measure."
* n = 4 cannot confirm Hypothesis B, regardless of the DR's §6.2 — see
  §7.3.

---

## 14. Receipts this campaign must produce

| path | what |
|---|---|
| `runs/v31_redo_surgical_2026-08-27/phase_g/` | the ceiling raising at tau 0.25, live |
| `runs/v31_redo_surgical_2026-08-27/phase_m/` | the 60-iteration dose measurement at exactly tau 0.10, plus the M1-M5 adjudication and the §1.3 prediction verdict |
| `runs/v31_redo_surgical_2026-08-27/arm_gate.json` | `redo_arm_gate.py --tau 0.10 --min-events 40 --min-units 64` over all four seed logs |
| `runs/v31_redo_surgical_2026-08-27/theta.json` | the cross-fit reducer's output: per-seed `score_A`/`score_B`, Theta, `Theta_adj`, and the 192 eval receipts it consumed |
| `runs/v31_redo_surgical_2026-08-27/delta.json` | the v27 backfill under the identical estimator, or an explicit NOT COMPUTED with the reason |
| `checkpoints/mario_1_1_v31_redo_seed{0..3}/` | run logs, 24 iterate checkpoints per seed, `run_manifest.json` |
| `docs/research/` | the adjudication document, written after the numbers, against these numerals |
