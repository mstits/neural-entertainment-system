# DR NEVER-EXECUTED AUDIT — 2026-08-27

One question: **how much of what this project paid deep research for was never
actually done?**

Answer, up front: **about a third of individual prescriptions, and two thirds of
those were disclosed backlog or reasoned substitutions rather than gaps anyone
was hiding from.** Ten rounds mined, 51 prescriptions extracted, 17 un-executed,
**4 of which put a standing verdict at risk.** Two rounds came back fully clean.
The process mostly works. ReDo was not typical — but it was not unique either,
and the three cases nearest it are named below.

---

## 1. The known instance, and what it cost

`responses/20260825T052330Z_v27_isolated_optimum.md` §5 issued "Decision B: The
Mandated ReDo Intervention" as a formal amendment to the v27 registration. It was
implemented, wired, preflight-verified as armed, and **it never fired once.**

Re-measured for this audit, from the receipts:

| fact | measured |
|---|---|
| configured `redo_tau` | **0.025** in all 8 seed configs (`configs/mario_1_1_v2{7,8}_seed{0..3}.yaml`) |
| telemetry across v27 + v28 | **2,000 per-iteration lines, every one `recycled 0 cum 0`** |
| isolation sweep, τ ≤ 0.20 | 0 recycles |
| isolation sweep, τ = 0.25 | first fire (0, then 5) |
| isolation sweep, τ = 0.30 / 0.35 | 13/2 · 15/8 |
| sweep last write | `isolate_tau0.35.log` — 2026-08-25 **00:19:42** |
| first training run opens | `train_seed0.log` — 2026-08-25 **00:22:02** |
| gap | **140 seconds** |
| wall-clock burned | **14.5 h** (v27 7.14 h + v28 7.37 h, 8 runs) |

**The mechanism cannot fire below τ ≈ 0.25. Both campaigns ran an order of
magnitude below that.** The evidence was on disk 140 seconds before the budget
started and was never read.

So v27 FAIL (0.530) and v28 FAIL (0.670) were single-variable arms described as
two-variable. **Both numbers survive** — neither depended on ReDo acting — but
the framing did not. Logged already at `MISTAKES.md` 2026-08-27
`[inert-treatment]`; repeated here only as the calibration bar for everything
below.

A second fact worth carrying: **the repo already knew.** B6's offline instrument
sweep on 2026-08-11 recorded *"dormancy 0.000 everywhere — plasticity clean, ReDo
not indicated"* (`docs/research/B5_PREREG_2026-08-08.md:437`), from a different
DR round that had explicitly gated ReDo on measuring dormancy first. That gate
was honored, returned a decline, and the decline was not consulted two weeks
later.

---

## 2. The count

Ten rounds mined. Corpus is 33 rounds; the 23 not mined here were excluded by the
inventory as lower-priority, superseded, or already adjudicated with a banked
verdict file — see §7 for what that leaves open.

| round | prescriptions | un-executed | clean? |
|---|---|---|---|
| `v27_isolated_optimum` (template) | 1 | 1 | no |
| `v22_commitment_options_ppo` | 3 | 3 | no |
| `v15_d3_success_detection` | 2 | 2 | no |
| `v20_rate_ceiling` | 6 | 2 | no |
| `audit7_peak_and_pivot` | 6 | 2 | no |
| `audit6_rate_building` | 5 | 2 | no |
| `v25_sticky_wall` | 5 | 2 | no |
| `v15_d1_backward_curricula` | 15 | 3 | no |
| `audit5_v12r_online_paradigm` | 5 | 0 | **yes** |
| `v26_gru_learning_failure` | 3 | 0 | **yes** |
| **total** | **51** | **17** | **2 / 10** |

By discriminant class:

| class | what it looks like | count |
|---|---|---|
| 1 — never implemented | no code was written | **13** |
| 2 — implemented, wired to nothing | code exists, no production path reaches it | **1** |
| 3 — implemented, wired, never ran | reaches production, no campaign enabled it | **1** |
| 4 — ran outside its firing range | passes every arming check, measures nothing | **2** |
| — executed and tested | produced a verdict | 23 |
| — executed and superseded / rejected with reasoning | correctly declined, in writing | 11 |

**34 of 51 (67%) landed or were properly declined.** Class 4 — the one that hides
— accounts for **2 of 51**.

---

## 3. Is the corpus clean? Mostly, yes — with three real holes

This must not be buried, because it is the more important finding: **the process
mostly works.**

- **`audit5_v12r_online_paradigm` is the corpus's positive control.** All four
  prescribed mechanisms (KL-anchor from the A7 prior, bottleneck-anchored
  backward curriculum, SIL, phase-gated kernel adversary) exist in production
  code, are wired into `configs/mario_1_2_online_v2.yaml`, and are independently
  certified **FIRED** by `scripts/check_mechanism_receipt.py` — `kl_anchor` 109
  non-zero observations, `sil` peak 3596, `backward` held at the terminal rung.
  The DR's own kill criteria are implemented at its own thresholds (KL > 0.15
  sustained 2M steps; SIL-starvation at 20M steps, hard-coded at
  `scripts/run_online_campaign.py:135`). The one mechanism not in the final
  banked recipe — the un-anchored adversary — was actually run in attempt 7,
  observed to degrade the policy, and consciously descoped with the reasoning
  written down in `runs/online_1_2_attempt_ledger.md`.
- **`v26_gru_learning_failure` is clean.** Its sole prescription, the `v27_PROBE`
  supervised stick-detection diagnostic, was built as `scripts/stick_probe.py`
  the same evening, its gate PASSED, a real loader bug in the PASS-supporting
  numbers was caught and **disclosed rather than buried** (CLAIMS.md marks them
  SUPERSEDED FIGURES), and the expensive pathway it gated was declined with
  documented reasoning.
- Most of the 13 class-1 items are **openly logged backlog, not concealment.**
  SWA is recorded as "still queued, never run" in three separate internal audits
  across eight days. Learned interruption is recorded as "not scheduled yet" in
  two. That is disclosure, and it is the opposite of the ReDo pattern.
- Several "gaps" are **reasoned bypasses that outperformed the prescription.**
  audit6's Q3 round-2 consolidation recipe was replaced wholesale by "more of
  consolidation-1", and the substitute banked 2.0% → 38.0%, above the round's own
  25–30% prediction.

What is *not* clean is a narrower and sharper thing: **the prose layer credits
mechanics the code did not run.** Three clusters — v22's options registration
(§4.1–4.3), the missing `V_adv` instrument (§4.4), and the confluence detector's
null-rate gate (§4.5) — each verified in code, plus one receipt-integrity near
miss (§5).

---

## 4. The findings that touch a standing verdict

### 4.1 v22 semi-MDP GAE — implemented, unit-tested, imported by nothing but its test

`src/training/smdp_gae.py` exists and implements the exact prescribed formulas.
`tests/test_smdp_gae.py` covers it and passes. **The only file in the repo that
imports it is that test.**

```
$ grep -rn "smdp_gae" src/ scripts/ tests/
src/training/smdp_gae.py:58:def smdp_gae(...)
tests/test_smdp_gae.py:12,14,54,55,64,74      # the test, and nothing else
```

`ppo_updater.py:153` calls `batched_gae` with an identical argument list whether
`commitment_options` is on or off. This is discriminant class 2, and it hides
*better* than class 1 because the file, the docstring and the green tests all
read as evidence the mechanic shipped.

`docs/proposals/OPTIONS_PREREG_2026-08-22.md` still lists it as **adopted**.
`docs/research/ZELDA_VISION_AGENT_AUDIT_2026-08-25.md:166` names
"`commitment_policy.py` + `smdp_gae.py` — FAILED its gate", crediting an orphaned
module with a failure it was never present for.

There *is* a partial defense, and it should be recorded because it means this was
not a silent drop: integration commit `cd00fdf` argues the substitution in its
message — *"Per-step GAE with the dense value stream then IS the correct
estimator read at decision rows."* But that reasoning never left the commit
message, and it is conditioned on a dense value stream that §4.2 shows does not
exist in training.

**The two estimators are not equivalent at λ<1.** Expanding the per-step sum,
each held-state critic value enters the decision's advantage with coefficient
`γ(1−λ)(γλ)^(i−1)` — identically zero in the semi-MDP form, non-zero only for
k ≥ 2. The deviation biases exactly the quantity the experiment adjudicated:
duration preference. The run failed **by duration overcommitment**, k=4 chosen in
93.6% of states.

### 4.2 v22 dense critic — the compounding half

Addendum item 4 asked the critic to be trained as a dense auxiliary task on every
env step, decoupled from the actor's decision cadence. It is not.
`ppo_updater.py:374` builds one permutation over `valid_indices` and drives the
entire K-epoch minibatch loop from it; there is no second index set and no
value-only pass. Under `commitment_options`, `trainer.py` sets
`valid_buf[t,i] = not _commit_held_buf[t,i]`, so **held rows never enter a
value-loss gradient** — while their rollout-time critic outputs are injected into
every k ≥ 2 advantage as read-only bootstrap targets.

So held-state critic values are simultaneously (a) never regressed and (b)
weighted into exactly the advantages that decide duration. A duration-dependent
bias sourced from the least-fitted values in the network is not neutral in an
experiment whose verdict is "the policy overcommitted."

### 4.3 v22 §6 eval-argmax overcommitment mitigation — claimed present, absent in code

The registration's stand-in reads: *"Eval-argmax overcommitment is mitigated by
the KL-anchor already standing in the campaign machinery."* All three prescribed
forms are absent: `ppo_clip_eps: 0.2` is a single global
(`configs/mario_1_2_options.yaml:70`) threaded as one scalar with no
duration-conditional branch; `CommitmentPolicy.pair_actor` is one flat 18-way
`nn.Linear` (`commitment_policy.py:76`), an integer decomposition of a single
softmax, not a factorized `(a,k)` head; no KL term on the duration marginal
exists.

**Verdict at risk:** the standing **OPTIONS MECHANISM FAIL** (control 8/100 vs
treatment 0/100, `runs/options/verdict.json`). The number stands — the treatment
fired loudly, and `OPTIONS_NEGATIVE_2026-08-23.md` is honestly scoped to what
ran. What does not stand is the **scope**: this was a test of *fixed-duration,
open-loop, unprotected commitment options under a per-step advantage estimator
and an untrained held-state critic*, not a test of temporal abstraction. A
downstream line (v23 Castlevania options) currently inherits that FAIL.

### 4.4 V_adv — the free discriminator for a never-retracted capability wall

`v15_d1`'s ADOPT bullet named four offline instruments. Three were built into
`scripts/score_banked_iterates.py` (top-two logit margin, dormant fraction,
effective rank); a fifth — parameter drift — was substituted in, which is why
nobody noticed the fourth went missing. **Advantage variance
`V_adv = E_s[Var_a(Â)]` was never implemented** (repo-wide grep: only an
unrelated local in `kernel_adversary.py:209`).

The banked verdict file `plans/v15_d1_backward_curricula_verdicts.md:381` adopts
it explicitly — *"ADOPT: log the missing instruments (no new training)"* — with
V_adv named first and an amendment attached (calibrate against B4's healthy
iterations). That is an adoption with a correction, not a rejection.

**Verdict at risk:** `docs/research/B5_PREREG_2026-08-08.md:414`, **"THIS IS A
REAL CAPABILITY WALL at gx ~2674–2872"** (RUN 3 FINAL VERDICT, 2026-08-10), never
retracted anywhere in the tree.

The defense offered elsewhere is that 0/717 entrance successes over 249
iterations is strong evidence independent of V_adv. **It is not.** 0/717 is the
symptom, and both hypotheses predict it. The same verdicts file, line 13, states
the round's one real contribution as exactly this: *"the advance gate is
reachability-based while the reward is progress-based, so a progress-flat
bottleneck is a mis-specification, not a capability wall."* V_adv was the adopted
instrument for separating "the agent cannot" from "the reward gives no gradient
here." A zero success count cannot separate them; a near-zero advantage variance
at the stall rung would have.

Weighing the other way honestly: the wall is scoped to rung-893 restarts, which
place the agent at gx 2674 *without momentum*, and later entrance runs pass that
x *with* momentum — so subsequent success does not by itself falsify the verdict.
The verdict is not refuted. It is **under-instrumented**, by an instrument that
costs hours and no compute.

### 4.5 The confluence detector's null-rate gate — armed and mute

`v15_d3` prescribed replacing the hand-built `tally` window matcher with a sparse
change-point detector on per-byte surprisal. Never implemented (zero hits for
OCD / e-detector / changepoint / surprisal). That alone would be an ordinary
class-1 backlog item — the banked verdict file sequenced it at slot 5 of 6, and
slots 1–3 genuinely shipped.

**The finding is the mitigation, not the absence.** The interim guard that *was*
built — `MAX_NULL_RATE = 0.05` plus `DEGENERATE` eligibility
(`scripts/clear_reachability.py:397,741`) — reads its measured rates from
`solve.null_rates`:

```
$ grep -rln null_rates configs/
(0 files)
```

**Ten confluence profiles. Zero declare a null rate.** The gate is implemented,
wired, armed, emits an eligibility table into every receipt, and **sits at an
operating point where it cannot act.** `clear_detect.py:3008` admits it in
writing: *"no profile carries a measured null today and a DEAD signal already
votes 0 by never firing."*

Meanwhile `tally`'s null fire rate is **measured at 1.00 on four games** — Contra
58/58 (`CONTRA_WALL_2026-08-27.md:415`), Castlevania 22/22, 28/28, 43/43, Bubble
Bobble 30/30. This is ReDo mirrored: ReDo carried no bits by never firing;
`tally` carries no bits by always firing. Both passed their arming checks.

**Verdict at risk:** Contra's clear-detector nulls — **19 `solutions/`
directories across 18 solver runs, all empty** (`CONTRA_WALL_2026-08-27.md:459`),
over **2,417,912 worker-steps** of from-power-on search
(`CONTRA_ROUTE_A_2026-08-27.md:363`) — are **VOID, not misses**, which the repo
self-caught today at `CONTRA_WALL_2026-08-27.md:434`. Fed its measured null back
in, `clear_quorum(contra.yaml, null_rates={'tally': 1.0})` returns
**UNREACHABLE, ceiling 1.0, required 2.0** — "the shipped 2-of-2 is a 1-of-1
`coord` vote wearing a corroborator's clothes." The measurement exists; no
profile carries it. Forward-live: every
confluence profile's `clear_quorum` arming receipt ("FIREABLE, ceiling 2.0,
required 2.0, zero slack") is vacuous. **Not** at risk: banked CONFIRMED clears —
CLAIMS.md ADDENDUM P-2 records the confluence detector scored 0 on the witnessed
Bubble Bobble and Tetris-B clears, so no CONFIRMED status runs through it.

---

## 5. A receipt-integrity finding: an experiment that never trained has a PASS on disk

v20's Rank-1 falsifier — one shared trunk + 4 per-level heads on 1-1..1-4,
compare the SUM of honest clears to the isolated baseline of 153/400 — is
**class 3: built, wired, never run.**

- `runs/shared_substrate/manifest.json` → `"status": "pending"`
- its own `default_checkpoint_dir`, `checkpoints/shared_substrate_v1`, **does not
  exist**
- CLAIMS.md has **zero** mentions of `shared_substrate`

That much is honestly disclosed once — `PROCESS_AUDIT_2026-08-23.md:122`,
"the trunk-plus-heads experiment remains distinct and unscheduled." But two other
surfaces disagree, and neither carries that caveat:

1. **`runs/shared_substrate/eval_shared_substrate.jsonl` holds 860 rows. 688
   carry `/fake/dir/` fixture paths — and 172 do not.** Those 172 are repeated
   verdict records reading `{"verdict": "SUPERSEDES", "aggregate": {"baseline_sum":
   153, "shared_sum": 200, "delta": 47, "beats_baseline": true}}`. They come from
   the test suite: `tests/test_eval_shared_substrate.py:501` uses 200 as its
   "generous win" fixture, several tests redirect `receipt_log` to `tmp_path`, and
   at least one path falls through to the module-level default at
   `scripts/eval_shared_substrate.py:121` — which points at the **real** receipt
   log. They accumulate on every `pytest` run: 11 rows on 08-18, 48 on 08-26, **62
   today**.
2. **`docs/proposals/README.md` §10, committed 2026-08-25 — two days *after* the
   audit above — marks the parent round `COMPLETED/ACTIONED` and states the
   shared-substrate ranking "shipped as commit `f757506`."** A harness shipping is
   not a hypothesis being tested.

An auditor who checks the two places one normally checks — the status index and
the receipts — concludes this experiment ran and won. Only the manifest's
`status` field and one research doc dissent.

This is the single highest-ranked recommendation across five convergent DR rounds
(v16, v18, v19, v20, and the synthesis), and the question it gates — *is
per-level specialization the right unit for the remaining 28 levels?* — is still
fully open.

---

## 6. Ranked backlog — cheapest first, verdict-bearing first

| # | item | cost | what it buys |
|---|---|---|---|
| **1** | **Compute `V_adv` in `scripts/score_banked_iterates.py`** over the banked B4 and v4 iterates, beside the three instruments already there | **hours; offline, no emulator, no training** | **Turns a standing FAIL back into an open question.** Separates "REAL CAPABILITY WALL at gx 2674–2872" from the reward mis-specification its own adopted discriminator was meant to rule out. Cheapest item in the corpus that touches a verdict. If it returns a mis-specification signature it also supplies the written addendum the rung-relative wavefront amendment has been waiting on. |
| **2** | **Purge the 172 fabricated `SUPERSEDES` rows; repoint the test default at `tmp_path`** | **minutes** | Stops a never-trained experiment asserting a PASS in its own canonical receipt, and stops the file growing on every test run. Pair with a one-line correction to `docs/proposals/README.md` §10. |
| **3** | **Write a measured `solve.null_rates` block into each of the 10 confluence profiles** | **hours** (the null measurement already exists) | Un-inerts the `DEGENERATE` gate that is already built and armed. Makes every `clear_quorum` arming receipt mean something. Would have caught the Contra VOID before 2.4M worker-steps. |
| **4** | **SWA over the banked 1-2 checkpoints around the 12/30 peak** | **4 h, zero rollouts** (the project's own estimate) | The cheapest unexecuted *capability* item in the corpus. Ranked #1 by audit7 and #3 by v20 a day apart, adopted into the build order by `RESEARCH_SYNTHESIS_2026-08-17.md:130`, queued unrejected since 08-17. Directly addresses the 08-26 checkpoint-selection defect the project caught statistically instead. |
| **5** | **Correct `OPTIONS_PREREG` and `ZELDA_VISION_AGENT_AUDIT:166`** to record what actually ran | **minutes** | Stops an orphaned module being credited with a failure it was not present for, and re-scopes the options FAIL from "temporal abstraction fails" to "fixed-duration unprotected options fail." |
| **6** | **Wire `smdp_gae` + a dense critic pass, then re-register the options A/B** | **1–2 days code + a full two-arm 1-2 campaign** | Makes the options question answerable. The no-rescue clause bars retuning, so this needs a *new* registration. Unblocks the v23 Castlevania options line, which currently inherits a FAIL from an experiment that did not test the prescribed mechanism. |
| **7** | **Schedule the shared-substrate run** (gated on 1-3/1-4 trajectory collection) | one overnight run + the collection dependency | Answers the top-ranked recommendation of five convergent rounds: is per-level specialization the right unit for 28 remaining levels? |
| **8** | **Learned interruption for commitment options** | new experiment + registration | Addresses the measured failure mode (k=4 in 93.6% of states) directly. Should not be built on the uncorrected estimator of items 5–6 — sequence after them. |

Items 1–3 total well under a day and no emulator time. Items deliberately *not*
listed: audit6's round-2 consolidation recipe (its substitute already beat the
round's own target — do not re-run), the anneal-to-zero LR trigger (DR ranked it
below SWA and named its own downside), and the three `v15_d1` items that were
explicitly declined in writing (permanent rung mixture, EMA spike, rung-relative
wavefront).

---

## 7. Method, and what this audit did not cover

Traced to **code**, not to documents — grep for the identifier, confirm a
production path reaches it, read the configured value. That discipline is the
point: the repo's prose layer is exactly what failed here, and today's learning
audit independently found it "consistently describes a stricter, more canonical,
better-instrumented experiment than the one the code ran."

Before reporting non-execution, the rejection was hunted in `plans/`, `CLAIMS.md`,
`docs/proposals/` and `docs/research/`. **Three candidate findings were dropped on
that check** and are recorded here so they are not re-reported: `v15_d1`'s
permanent rung mixture (`"SPIKE (not adopt)"`, verdicts file line 403), its EMA
policy-weights spike (demoted for carrying zero citations, line 424), and the
rung-relative wavefront amendment (self-deferred, *"Not this quarter"*, line 525,
conditioned on a written addendum re-opening B5 that never came).

**Coverage limit.** 10 of 33 rounds. The 23 unmined divide into three groups:
already adjudicated with a banked verdict file in `plans/` (v11, v12, v13, v14,
v15_d2, v6, and the PR-MDP/SHAPO pre-registrations); executed and closed
(`audit1`, whose rehabilitate-or-close protocol ran same-day and returned CLOSE;
`v4`/`v7`, the banked adversarial line); and **not yet checked** — `v21`, `v24`,
`v18`, `v17`, `v19`, `v16`, `audit0`, `audit2`, `audit3`, `audit4`, and the older
infrastructure PDFs v5/v8/v9/v10. Of those, `v24_scroll_odometer` and
`v21_hazard_decision_rule` bear on live machinery and carry named sub-prescriptions
with zero grep hits (`nametable_hash`, `oam_fusion`, the viability-kernel
relabeling). They are the natural next batch.

No emulator ran and no training was touched for this audit; it was read and grep
only.
