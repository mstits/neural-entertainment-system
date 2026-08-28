# v32 Lane B — Phase R adjudicated. NO-GO on R4, and the recovery question is answered.

Adjudicated against `docs/proposals/V32_REDO_BOTTOM_K_2026-08-28.md` (commit
`e9cc5ed`), whose every numeral was fixed before compute. Nothing here moves a
bar, and no bar was reopened.

---

## 0. Headline

Phase R ran the full 60 iterations at exactly the registered operating point
(k = 2, C = 5, scope fc2). It passed R1, R2 and R3 and **failed R4**:

```
repeat_rate = 11/11 = 1.00
fc2 = [21,26] [5,26] [5,26] [5,26] [5,9] [5,9] [5,9] [5,9] [5,9] [5,9] [5,9] [5,9]
```

**Every one of the eleven consecutive event pairs shared at least one index.**
The recycled set never turned over even once. Per §11 this is
**VOID-NO-TURNOVER**; per §8 it is the ladder's one licensed trigger.

**No training began.** §4 A1 is unambiguous — *"No training begins before Phase
R returns GO."* Zero seeds launched, zero ARMED, zero scored. **Θ is not
issued.** The 0.80 / 0.767 bars stand untouched and the 0.05 winner's-curse
budget is unspent.

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
45 the gap is roughly fourfold. Re-initialization does not restore a unit to
the middle of the distribution; it pins it to the floor, and four free PPO
updates do not lift it off.

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

## 5. What this VOID licenses — exactly and no wider

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
