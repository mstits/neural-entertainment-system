# V_adv ON-POLICY PRE-REGISTRATION — the adjudicating version the VOID named

**Registered 2026-08-27, before any V_adv number, any critic value, any `eta2`
and any verdict statistic was computed from any checkpoint in this job.**
Coverage counts, tape geometry and wall-clock costs were measured first and are
disclosed in full in §3 — the same order `VADV_PREREG_2026-08-27.md` used, and
for the same reason: a threshold must be checked against its acting range *on
the data it will see*, and that check is a coverage measurement, not an effect
size.

**Predecessor:** `docs/proposals/VADV_PREREG_2026-08-27.md` (registration,
commit `b9ed38e`) and `docs/research/VADV_B5_2026-08-27.md` (verdict, commit
`789aefe`). That computation returned **VOID at R = 0.279**, inside its own
pre-declared indeterminate band `(0.20, 0.50)`.

**This is a new experiment with a new registration, not an amendment.** It
exists because the VOID named four binding limitations and four repairs. All
four repairs are below. Everything the VOID did *not* name as broken —
thresholds, band edges, the R statistic, the A1–A8 gate set, the LIVE /
COLLAPSED rules, gamma, the cell rule — is **inherited verbatim** so the two
computations are directly comparable. Every deviation is listed in §2 with its
reason.

---

## 1. What is inherited verbatim (do not re-derive; copy)

From `VADV_PREREG_2026-08-27.md` §2, §5, §6:

* **The statistic.** `Â(s,a) = γ·V_θ(s')·(1 − done) − V_θ(s)` from the banked
  checkpoint's **critic head** over **real banked successors**; `γ = 0.99`.
  `r(s,a)` is omitted, and the omission's declared bias direction is unchanged
  (§9.2).
* **The primary reading.** `eta2 = SS_between / (SS_between + SS_within)` with
  the state cell as blocking factor. `raw` and `raw_norm` are cross-checks and
  are never compared across checkpoints.
* **The cell key.** Loose key = last stacked frame, indices `[534, 711)`,
  animation-phase byte dropped. Qualification: ≥ 2 distinct actions each with
  ≥ 2 rows, ≥ 6 qualifying rows total. Exact-key sensitivity check registered
  as secondary.
* **Band edges.** `WALL = [2674, 2872)`, `PC_B5 = [2872, 3267)`,
  `APPROACH = [2400, 2674)`, `EARLY = [160, 1600)`. gx decoded from the
  observation's own `out[175] / out[176]` contract.
* **LIVE(region)** ⟺ `eta2 > null_q(0.975)` **and** `eta2 / median(null) ≥ 1.5`.
  **COLLAPSED(region)** ⟺ `eta2 ≤ null_q(0.975)` **and** `hi(region) < lo(PC_B5)`.
  Anything else INDETERMINATE.
* **`R = eta2_excess(WALL) / eta2_excess(PC_B5)`**, each excess taken over that
  region's own null.
* **The bands of R and their meanings** — §7 below, unchanged in every digit.
* **A1–A8**, unchanged in requirement — §8 below, with the acting-range check
  each one now needs on *this* data.
* **The binding stop rule.** *If A1 and A3 do not both hold, the metric does not
  discriminate on this data, the job stops there, and that is the whole report.*

`n_perm = 1000`, `n_boot = 2000`, `seed = 20260827`, loose cell key — the
predecessor's final settings, held fixed across every iterate so the arc is one
measurement and not twenty-six.

---

## 2. The four repairs, each named by the VOID

| # | VOID §  | limitation | repair registered here |
|---|---|---|---|
| 1 | §5.1 | **binding.** The bank was go-explore expert-window states, off-distribution for a critic whose policy reached the wall 0/717 times. Cannot rank (a) no reward gradient, (b) critic under-fit from non-visitation, (c) capability failure causing (b). | The bank is collected from **B5 run 3's own rollouts** — each iterate's own policy, sampled, under the training-time sticky and the training-time restart draw (§4). Wall states become on-distribution for the critic being scored. |
| 2 | §5.2 | All 612 qualifying `WALL` rows decoded to gx = 2674 exactly; the traversal in dispute was never observed. | An `INTERIOR` region, a registered minimum coverage, a **penetration receipt**, and — because §3.1 shows penetration may be structurally impossible — a **pre-declared no-penetration reading** (§5.3). |
| 3 | §7 next-step 3 | One checkpoint (iter 250) was scored; R may move across the collapse. | The **whole 26-point iter 10–260 grid** is scored at identical settings, with a pre-committed arc-aggregation rule (§7.2) that cannot be gamed after the fact. |
| 4 | §5.3 | `R` moved 0.279 → 0.111 — across a decision boundary — depending on an unstated choice of null reference. | The null reference is **pre-committed in writing here, before any number** (§6). |

Nothing else moves.

---

## 3. Measured before this registration was written (full disclosure)

Everything in this section is a **coverage, geometry or cost measurement**. No
critic was loaded to produce any `eta2`, `raw`, null or `R` in this section, and
none exists anywhere in this job at the time of writing.

### 3.1 The band's geometry — the fact that reframes repair 2

Read off the project's own minted tape index
(`checkpoints/backward_states/1-2/index.json`, 1094 entries, provenance
`solver_tape`) and off the pooled offline bank. No disassembly, no RAM map
beyond the encoder's existing `out[175]/out[176]` contract.

**The solver's own clearing route does not pass through gx 2675–2871.**

    entries  682–919 : gx pinned at 2674 for ~240 consecutive action-steps
    entries  920–971 : gx falls to 2595, climbs back to 2656
    entries  972–977 : gx reads 0 (area transition)
    entries  978+    : gx re-appears at 2872 and climbs to 3175

So the 198-pixel `WALL` band is not 198 pixels of terrain the tape crosses. Its
**reachable content on the tape is the single value gx = 2674** (plus 2870/2871,
which occur only at entries 1011–1013, i.e. *after* the transition). The
predecessor's §5.2 phrasing — "the band is 198 px wide; only its entry pixel was
measured" — is therefore **too generous to itself**: the entry pixel is very
nearly the only thing there *is* on the solver's route.

In the pooled offline bank (110,805 transitions) the interior is reachable but
almost never reached: **gx = 2674 carries 899 rows; gx = 2676 carries 49; every
value from 2680 to 2870 carries exactly one row each — 29 rows, contiguous
indices 52752–52780 of `runs/iq_1_2`, a single trajectory running right and
accelerating.** One traversal in 110,805 transitions. A single pass can never
qualify a cell (qualification needs ≥ 2 actions × ≥ 2 rows), which is the
arithmetic reason the predecessor's `WALL` reduced to its entry pixel. It was
not a sampling accident and more offline data would not have fixed it.

**Consequence, registered:** `INTERIOR` coverage cannot be *designed*; it can
only be *earned* by the policy penetrating. §5.3 pre-declares what happens when
it is not earned, and that disposition is written here, before the run.

### 3.2 What the curriculum did, from the checkpoints themselves

Every one of the 26 iterates carries
`backward_curriculum = {tau: 893, n_entries: 1094, advances: 5, ...}` with
`entrance_successes: 0` and `entrance_attempts` rising 27 → 444. So:

* The cursor was **already parked at rung 893 before iter 10** (run 3 resumed
  run 2's cursor). **Rung 893's 41-entry restart window is the only tau
  distribution this arc ever trained on**, alongside the ~2 % entrance mass.
* The five rungs the curriculum **advanced through** are
  `tau ∈ {1093, 1053, 1013, 973, 933}` (`tau_init −1`, `advance_actions 40`).
  Each cleared the level at ≥ 20 % over ≥ 30 attempts — `advance_threshold 0.2`,
  `min_attempts 30`, and a tau success is `_bwd_env_clear`, i.e. **the level was
  cleared**.
* Their 41-entry windows in gx: `1093 → [3005, 3175]`, `1053 → [2871, 3005]`,
  `1013 → [0, 2873]`, `973 → [0, 2656]`, `933 → [2612, 2674]`.
  Only the first three place mass in `PC_B5`; the last two lie below it.

### 3.3 On-policy coverage pilot — iter 250, W = 2, sampled + sticky 0.25

| rollout source | eps | rows | outcome | region | rows | qual rows | qual cells |
|---|---|---|---|---|---|---|---|
| rung 893 window | 12 | 12,388 | 0 clear / 0 death / 12 truncate | `WALL` | 12,381 | **11,970** | **140** |
| rung 893 window | 12 | — | — | `INTERIOR` | **0** | 0 | 0 |
| rungs {1093,1053,1013} | 24 | 4,756 | **24 clear** / 0 death | `PC_B5` | 4,752 | **3,075** | **43** |
| true entrance | 12 | 3,182 | 0 clear / **12 death** | `EARLY` | 1,876 | 581 | **7** |

Also measured: over 12 rung-893 episodes (≈ 12,000 steps) gx took **three
distinct values — 2670, 2672, 2674 — and never exceeded 2674.** Max gx per
entrance episode: 174, 182, 454, 654, 664, 666, 672, 674, 674, 862, 894, 898.

What this fixes in the gate settings, before the run:

* `WALL` at 140 cells / 11,970 rows clears A6 (20 / 400) by 7× and 30×. The
  predecessor's `WALL` was 28 / 612. **A6 is comfortably inside its acting range
  on the on-policy bank** — which is exactly the check the eight shipped vacuous
  gates existed for.
* `PC_B5` needs ≥ 24 episodes to clear A6's cell floor (43 cells at 24 eps; the
  earlier 12-episode pilot gave 19, one short of the gate). **Registered
  episode counts below are set from this measurement, and the gate is not
  moved to fit them.**
* `EARLY` at 12 entrance episodes gives 7 cells and would **fail** A6.
  60 entrance episodes are registered; if `EARLY` still misses A6 it is
  **VOID as pre-declared**, exactly as `APPROACH` was in the predecessor.
* **`INTERIOR` returned zero rows.** The no-penetration branch of §5.3 is the
  *expected* branch, is written before the run, and yields **no evidence for
  either hypothesis**.

### 3.4 Cost, measured

Emulation at `--workers 2`: **~610 env-steps/s** with the net forward in-loop.
Scoring, measured on synthetic arrays of the sizes above: `n_perm = 1000` costs
≈ 12 s and `n_boot = 2000` ≈ 21 s at 24,000 rows / 300 cells; ≈ 2 s / 3 s at
3,100 rows / 43 cells. A7 at 50 trials × 200 permutations ≈ 65 s per iterate.
Budget in §10 is built from these numbers, not from a guess.

### 3.5 What MIS-SPECIFICATION asserts, in this run's own code

Stated so the hypothesis under test is concrete, **not** so it is pre-judged.
`nes_core/src/rewards.rs:1531` pays forward progress on a **high-water mark** —
`new_progress = max(0, x − max_x_visited)`, so backtracking earns 0 (it is not
penalised, it is unrewarded). `configs/mario_1_2_backward.yaml` sets
`checkpoint_scale: 0.0` (dense checkpoints off), `air_bonus 0`, `survival_bonus
0`, `score_delta 0`, `time_penalty −0.01`, `death_penalty −15`,
`completion_bonus +50`. Therefore from the moment an episode's `max_x_visited`
reaches 2674 until gx exceeds it again, the only reward terms that can fire are
the per-step time penalty, death, and the clear. **Whether the critic still
discriminates actions in that window is precisely what this instrument
measures, and the answer is not assumed here.**

---

## 4. Rollout protocol — the bank is the policy's own behaviour

New script `scripts/collect_onpolicy_bank.py`, tests in
`tests/test_collect_onpolicy_bank.py`. It reuses the proven driving pattern of
`scripts/gen_iq_transitions.py` (`nes_core.Pool`, `load_worker_state`,
`step_all`, `TileFeatureStacker`, `resolve_encoder`).

**Checkpoints.** Exactly the 26 files
`checkpoints/mario_1_2_backward/prevrun_20260809_222300/vanilla_ppo_iter_{00010..00260 step 10}.pt`
— B5 run 3, the arc that produced the standing verdict. All 26 verified
`fc1.weight` shape `(128, 712)`, `fc2 (32,128)`, `actor (6,32)`. The
`vanilla_ppo_iter_00000.pt` in the parent directory belongs to a different
(2026‑08‑14) run and is **excluded**; runs are never mixed.

**One bank per iterate.** Iterate *k*'s bank is collected with iterate *k*'s own
policy and scores only iterate *k*'s own critic. Banks are **never pooled for a
scored reading**; pooling is permitted only for the §5.2 interior *coverage*
receipt, which is a count and carries no `eta2`.

**Policy mode: SAMPLED from the policy's own softmax, temperature 1.0.** Three
reasons, registered:
1. It is the behaviour that generated the critic's own training data
   (`vanilla_ppo`, sticky 0.25 in-training), so it is what "on-distribution"
   means for this critic. Greedy is a different distribution.
2. `Var_a(Â)` requires **≥ 2 distinct actions in a cell**. A greedy rollout emits
   one action per state; under the registered cell rule **no cell could ever
   qualify** and the instrument would return VOID everywhere by construction.
   That would have been the ninth vacuous gate. Sampled is what keeps A6 inside
   its acting range.
3. The honest-protocol greedy eval is a *different measurement* with a different
   purpose and is not part of this job.

**Noise.** `sticky_action_prob = 0.25`, roll gated on `step > 0` within each
episode (`sticky_episode_boundary_reset: true`, the config's own setting and the
honest harness's own rule). No start-jitter: training restarts are tape states,
and jitter belongs to the honest eval, not to a distribution-matching bank.

**Start states, per region — a hard partition, registered.**

| designated rung set | draws | populates |
|---|---|---|
| **WALL_SRC** — tau 893 | uniform over tape entries [853, 893] (the 41-entry window, `window_frames 160 / frames_per_step 4`) | `WALL`, `INTERIOR` |
| **PC_SRC** — tau ∈ {1093, 1053, 1013} | uniform over rung, then uniform over that rung's 41-entry window | `PC_B5` |
| **ENTR_SRC** — the true entrance | `runs/live_show/smb_4_4_micro/entrance_after_1-1.state` | `EARLY` (secondary) |

**Cross-population is dropped, not merged.** A `PC_SRC` row landing in `WALL`, or
a `WALL_SRC` row landing in `PC_B5`, is discarded. Reason registered in advance:
rungs 933/973 and the PC rungs produce *clearing* trajectories, and letting them
deposit rows in `WALL` would raise `WALL`'s action-discrimination for a reason
that has nothing to do with the critic at rung 893 — an artifact that would push
`R` toward CAPABILITY, i.e. toward corroborating B5. The partition removes it.

**Episode cap** = the run's own per-rung budget,
`min(rollout_steps 1536, 600 + 2.0·(1094 − r))` for an episode restarted at tape
entry `r` — 1002 steps at rung 893, ~602–682 at the PC rungs, 1536 at the
entrance (the formula's 2788 is swallowed by the global cap, as in training).

**Episode counts per iterate** (set from §3.3, then held fixed):
**40 WALL_SRC**, **24 PC_SRC**, **60 ENTR_SRC**.

**Termination and `done`.** A row is recorded for every step until the episode
ends. `done = 1` on **death** (`RAM[$000E] ∈ {6, 11}` or the pool's own done
flag) **or on the level clear**, detected as a change in `(RAM[$075F],
RAM[$075C])` from the episode's own start value — the config's declared
`ram_mapping.world / .level`, no new address. **No row after that transition is
recorded.** A cap-truncated episode ends with `done = 0` and its rows are kept
(partial-episode bootstrapping, the convention `scripts/smodice_data.py` already
uses).

**LEVEL-IDENTITY PURITY GUARD (hard, revert-verified).** A row is admitted only
if `(world, level) == (0, 1)` at **both** `s` and `s'`. Any violation raises
`RuntimeError` and VOIDs the job. This is not hypothetical: the §3.3 pilot's
*first* version, without the guard, recorded 1,420 rows at gx 0 and hundreds at
gx 404/416/418/548 — **1-3 states aliasing into 1-2's gx bands after a clear.**
The shipped test asserts those rows are dropped and **must fail when the guard is
reverted**; a check that has never been seen to fail is not a check.

**Determinism.** Collector RNG seed `20260827 + iter`. Bank written to
`runs/vadv_onpolicy/iter_{NNNNN}.npz` with schema
`state / action / next_state / done / truncated / src_rung / episode_id`,
plus a sidecar JSON of per-episode outcomes and max gx.

---

## 5. Regions, and the interior rule

### 5.1 Regions scored

| region | band | source | status |
|---|---|---|---|
| `WALL` | `[2674, 2872)` | WALL_SRC | **PRIMARY — numerator of R** |
| `PC_B5` | `[2872, 3267)` | PC_SRC | **PRIMARY — denominator of R, positive control** |
| `INTERIOR` | `(2674, 2872)` i.e. `[2676, 2872)` | WALL_SRC | registered secondary — §5.2/§5.3 |
| `EARLY` | `[160, 1600)` | ENTR_SRC | registered secondary; VOID if A6 fails |
| `NEG_gx_frozen` | cells within `PC_B5` where no tried action moves gx | PC_SRC | NC-b, disposition inherited (§8) |

`APPROACH` is **not scored**: §3.2 shows the WALL_SRC window never reaches it
and the partition forbids populating it from other rungs. Dropping an
unpopulated region is not a goalpost move; scoring it from a foreign rung set
would be.

### 5.2 Interior coverage rule — the minimum, stated as a number

`INTERIOR` is scored **only** if, at that iterate, it independently satisfies
A6: **≥ 20 qualifying cells and ≥ 400 qualifying rows** — the same floor every
other region must clear, not a relaxed one.

Two receipts are emitted per iterate whether or not the gate is met, and pooled
across the grid for the summary:

* `pen_rate` — fraction of that iterate's 40 WALL_SRC episodes whose max gx
  exceeds **2676**;
* `gx_max` — the maximum gx over all WALL_SRC rows;
* `n_interior_rows`, `n_interior_qual_rows`, `n_interior_qual_cells`.

Pooled interior coverage across all 26 iterates is reported as a single
penetration receipt. **Pooling is for the count only** — a pooled `eta2` would
mix 26 critics and is forbidden.

### 5.3 NO-PENETRATION READING — pre-declared, because §3.3 says this is the likely branch

If `INTERIOR` fails its coverage gate — including the expected case
`pen_rate = 0` at every iterate:

1. **`INTERIOR` is VOID, and VOID is reported.** It is **never** COLLAPSED.
   Reading absent data as "the critic cannot tell actions apart there" would
   manufacture the mis-specification signature out of an empty region — the
   identical defect A3 caught in the predecessor, one level up. This is the
   single most important line in this document.
2. **It is recorded as a positive measurement, not as a gap.** "No rung-893
   rollout at any of 26 iterates, across ~1,040 episodes and ~1.0 M env-steps,
   exceeded gx 2676" is a strictly stronger and more localised statement than
   `entrance 0/717`: it places the failure **at the gx-2674 state itself**, not
   somewhere along a 198-pixel traversal.
3. **It is evidence for neither hypothesis.** CAPABILITY and MIS-SPECIFICATION
   both predict it. Registered now so it cannot be spent later.
4. **The primary verdict is unaffected in its machinery.** `R` is computed on
   `WALL` and `PC_B5` exactly as registered. `WALL` will be dominated by the
   entry state — as it was in the predecessor — but now **on-distribution for
   the critic scoring it**, which is repair 1 and the whole point.
5. **It caps what V_adv can ever say.** A critic-based offline instrument cannot
   score states the policy never occupies. With no penetration, V_adv can speak
   to the **entry decision at gx 2674** and to nothing beyond it. If the primary
   `R` is also indeterminate, §11 fires and the question retires; the traversal
   is then explicitly **not** something a further V_adv iteration could reach.

If `0 < n_interior_qual_rows` but the gate is missed, the disposition is
identical (VOID, counts reported). If the gate **is** met, `INTERIOR` is scored
and reported with its own LIVE/COLLAPSED verdict — as a **registered secondary
that cannot move the primary verdict**, because its band was not part of the
predecessor's `R` and adding it to `R` now would be a goalpost move.

### 5.4 ADJACENT-RUNG PROBE — registered secondary, diagnostic only

At the five iterates **{10, 70, 130, 190, 250}**, 24 episodes from **rung 933**
(window entries [893, 933], gx 2612–2674) — a rung the curriculum demonstrably
advanced through. Reported: clear rate, max gx, and whether those trajectories
deposit rows in `WALL`. This is the cheapest measurement that can say whether
the failure sits *at* the entry state or *before* it. **It is barred from
entering `R`, from populating any scored region, and from moving any verdict.**
Its rows are written to a separate file and are not part of any scored bank.

---

## 6. NULL REFERENCE — pre-committed, in writing, before any number

**The null reference is the median of the NC-c within-cell action-label
permutation distribution, `n_perm = 1000`, `seed = 20260827`, computed
separately for each region on that region's own rows.** `R`'s numerator and
denominator are each `eta2_obs − median(null)` for that region.

This matches the predecessor's registered reference exactly, and the choice is
made **before** any number here, for three reasons:

1. It preserves comparability with `R = 0.279`. Changing the reference would
   make the two computations incommensurable and would silently rewrite the
   predecessor's verdict.
2. The permutation null is the correct null *for this statistic on this data*:
   it destroys the action↔successor pairing while preserving the cell structure,
   the successor-value distribution, the critic and the critic's scale. The
   analytic `df_b/(df_b+df_w)` null assumes a balanced Gaussian design this data
   does not have.
3. **Discipline.** On the predecessor's data the analytic null gave `R = 0.111`
   — inside MIS-SPECIFICATION, the more interesting conclusion. Adopting it now
   would be choosing a reference *because* of where it lands. It is not adopted.

`eta2_null_analytic` continues to be **emitted on every reading** and reported
in the results table, so the gap between observed and its own null stays visible
on the face of the output. **It never enters `R` and never moves a verdict.** If
the two references again disagree across a decision boundary, that disagreement
is reported as a disclosed sensitivity — it is not a second chance at a verdict.

---

## 7. R bands and the arc rule

### 7.1 Per-iterate bands — unchanged in every digit

* **MIS-SPECIFICATION signature** ⟺ `PC_B5` LIVE **and** `WALL` COLLAPSED
  **and** `R ≤ 0.20`.
* **CAPABILITY signature** ⟺ `PC_B5` LIVE **and** `WALL` LIVE **and**
  `R ≥ 0.50`.
* **INDETERMINATE** ⟺ anything else, including `R ∈ (0.20, 0.50)`.
* **VOID(iterate)** ⟺ any A1–A8 gate fails at that iterate. A VOID iterate
  **enters no aggregate** — it is neither a signature nor a non-signature.

### 7.2 Arc aggregation — pre-committed, so the grid cannot be mined

Let **A** = the set of iterates on the 26-point grid passing **all** of A1–A8.

* If `|A| < 13` (half the grid) → the whole reading is **VOID — insufficient
  admissible arc coverage.** No `R` is reported as a verdict.
* **MIS-SPECIFICATION** ⟺ **≥ 80 % of A** carry the MIS-SPECIFICATION
  signature.
  → B5's verdict is **re-opened by written addendum**, and the rung-relative
  wavefront amendment deferred since 2026-08-11 becomes **live**.
* **CAPABILITY** ⟺ **≥ 80 % of A** carry the CAPABILITY signature.
  → B5's verdict is **corroborated** by an instrument that could have
  contradicted it.
* **Otherwise INDETERMINATE** → §11 retirement fires.

Why 80 % and not "the last checkpoint" or "the mean R": a discriminator whose
reading flips across an arc is not adjudicating anything, and the honest report
of that is INDETERMINATE. Reporting a mean `R` would let one extreme iterate
carry the arc; reporting one chosen iterate is the defect repair 3 exists to
fix. **The full 26-point `R` curve is published regardless of the verdict**, so
a reader can see the trajectory the rule summarises.

**Iter 250 is the designated comparability point**: its `R` is reported beside
the predecessor's `0.279` in the same table. It has **no special weight in the
rule** — it is one of 26.

---

## 8. Admissibility A1–A8, each checked against its acting range on THIS data

| gate | requirement (inherited) | acting-range check on the on-policy bank |
|---|---|---|
| **A1** | PC-1 LIVE on the 1-1 arc. **Stop-the-job gate.** | Unchanged and **offline**: B4 / v4 iterates × `runs/interference/success_1_1.npz`, gx [160, 3100]. Re-running it reproduces the predecessor's η² = 0.6669 as a free instrument-reproducibility check. Not rolled out — A1 asks whether the metric shows LIVE where the agent demonstrably learned, and 1-1 is that place. |
| **A2** | PC-2 (`PC_B5`) LIVE. | Pilot: 43 cells / 3,075 qualifying rows at 24 eps, 24/24 clears. In range. |
| **A3** | NC-a zeroed critic returns `raw == 0.0` with `degenerate_critic`, verdict VOID. **Stop-the-job gate.** | Unchanged; run per job, not per iterate. |
| **A4** | NC-c null non-degenerate (`q975 > q025`, `n_valid > 0`). | Checked per region per iterate. |
| **A5** | `Var_batch[V_θ] > 0` on the pooled scored bands. | Checked per iterate. |
| **A6** | ≥ 20 qualifying cells **and** ≥ 400 qualifying rows in `WALL` **and** `PC_B5`. | **Measured before registering:** `WALL` 140 / 11,970 at 12 eps (registered at 40); `PC_B5` 43 / 3,075 at 24 eps. Both inside range with margin. `EARLY` and `INTERIOR` fall under the same floor and are VOID if they miss it. |
| **A7** | **Binding power gate.** Inject a `PC_B5`-sized effect into `WALL`'s own rows; detection must be ≥ 0.95. | 50 trials, `n_perm = 200`, at the effect size reproducing that iterate's own `PC_B5` η². **Plus a registered negative demonstration on this run's own data**: the same gate run on a `WALL` region down-sampled to 8 cells must return power < 0.95. A power gate that has never been seen to fail on the data in front of it is not a gate. |
| **A8** | Checkpoint input width == bank obs width (712); never `strict=False`. | **Pre-verified: all 26 iterates load `fc1.weight (128, 712)`.** |

**NC-b (`NEG_gx_frozen`) disposition, inherited unchanged:** it is a
*conservative* null (a return-predicting critic may legitimately stay live in a
one-step-flat cell). **If NC-b does not collapse, the "reward provably flat"
null is VOID, only NC-a and the permutation null survive, and this reading is
capped at INDETERMINATE by itself** — which is what happened in the predecessor
(η² 0.175 vs its q97.5 0.048) and may well happen again. Registered before the
run, applied whatever it says.

**Anti-vacuity, shipped with the mechanism.** The 45 existing tests in
`tests/test_score_banked_iterates.py` are unchanged and must stay green. The new
collector ships with tests that must **fail on revert** for: the
level-identity purity guard (§4), the cross-population partition (§4), the
`done`-on-clear rule, and the sticky `step > 0` gate. Each is revert-verified in
an isolated scratch tree before the run, and the revert-verification is
reproduced by a second hand at publication — the predecessor's standard, kept.

---

## 9. Registered risks and declared bias directions

1. **`PC_B5` is off-distribution for this arc's critic.** §3.2: the cursor was
   parked at rung 893 before iter 10, so run 3 never trained on the PC rungs.
   The on-policy repair therefore lands on the **numerator** (`WALL`), and the
   denominator is now the off-distribution side — an inversion of the
   predecessor's §5.1, disclosed here, not discovered later. **Declared bias
   direction:** an under-fit control depresses `eta2_excess(PC_B5)` and
   therefore **inflates `R`**, i.e. biases toward **CAPABILITY / corroborating
   B5**, away from the more interesting conclusion. This is conservative in the
   same direction as risk 2 and is not a thumb on the scale toward re-opening.
   *Mitigation registered, not improvised:* `EARLY` from `ENTR_SRC` is an
   **on-distribution** secondary reference (entrance episodes are ~2 % of the
   run's own mass, 717 of them), and `R_early = eta2_excess(WALL) /
   eta2_excess(EARLY)` is reported beside `R`. **`R_early` is secondary and
   cannot move the verdict** — swapping the primary control would break
   comparability with 0.279, which is the whole reason the bands were inherited.
2. **`r(s,a)` omitted from `Â`** — inherited. A progress-based `r` contributes
   more action-variance where progress is achievable than at a progress-flat
   wall, so omitting it makes `R` larger: conservative against re-opening B5.
3. **The interior may be unmeasurable in principle** (§3.1). Handled by §5.3,
   pre-declared.
4. **Forced start states are not a stationary on-policy distribution.** The
   actions are the policy's own; the *starts* are forced, exactly as the
   training loop forced them. This is "the policy's own behaviour from the
   rungs the run itself used", which is what repair 1 asks for, and it is not
   claimed to be more.
5. **Loose cell key merges non-identical states** — pushed into `SS_within` by
   construction; exact-key sensitivity check registered as secondary.
6. **A live V_adv does not prove capability**, only that the critic
   discriminates actions there. A CAPABILITY reading is corroboration of B5, not
   an independent proof, and must be written that way.
7. **Sampled rollouts are not the honest protocol** and no clear rate from this
   job is a learned result. Everything here is instrument data. Solver output is
   exhibition; only honest-protocol results are learned.

---

## 10. Budget and abort conditions

**Hard wall-clock ceiling for Lane A: 6.0 hours**, of which **≤ 3.0 hours of
compute**. Built from §3.4, not guessed:

| stage | estimate |
|---|---|
| rollouts, 26 iterates × (40 + 24 + 60 eps) at ~610 steps/s, `--workers 2` | ~45 min |
| scoring, 26 × (WALL + PC_B5 + EARLY + INTERIOR) at n_perm 1000 / n_boot 2000 | ~20 min |
| A7, 26 × 50 trials × 200 perms, + the 8-cell negative demonstration | ~30 min |
| A1 / A3 / A5 controls, adjacent-rung probe, receipts | ~15 min |
| collector + tests + revert-verification + write-up | remainder |

**CPU discipline:** rollout collection runs at `--workers 2` and completes
before Lane B training takes the machine. Sequence within the lane; never poll.

**Abort conditions — each stops the job and reports what is banked:**

* **A1 or A3 fails** → stop. The metric does not discriminate on this data and
  that is the whole report. (Inherited binding rule.)
* **Purity guard raises** (any row with `(world, level) ≠ (0, 1)`) → the whole
  job is **VOID**. No partial bank is scored.
* **Rollout collection exceeds 3.0 h** → stop collecting; score only iterates
  whose bank is complete. If `|A| < 13` → **VOID — insufficient admissible arc
  coverage.**
* **Total lane exceeds 6.0 h** → stop and report VOID with what is banked.
  Fewer iterates honestly reported beats a grid overrun.
* **`WALL` misses A6 at more than 30 % of the grid** → the on-policy bank does
  not support the reading → **VOID**. There is **no silent fallback to the
  offline bank**; that would re-introduce the confound this job exists to
  remove.
* **Any checkpoint fails A8** → that iterate is VOID; never `strict=False`.
* **Any threshold in this document is edited after the first `eta2` is read** →
  the job is VOID and is reported as a process failure. A moved goalpost is a
  fabricated result here.

---

## 11. RETIREMENT RULE — what happens if this version is also indeterminate

**If the §7.2 arc rule returns INDETERMINATE, or the reading is VOID for any
reason other than a fixable operational fault (a crash, an abort on wall-clock,
a purity-guard raise), then:**

1. **V_adv is declared unable to adjudicate B5, and the question retires.**
   Written plainly, in those words, in the verdict document. Not "needs a
   further round"; not "a larger bank might". Two computations of the
   discriminator B5 itself adopted — one offline with an off-distribution wall,
   one on-policy with the wall on-distribution, both with live controls, real
   power and a pre-committed null — will have returned no signature. That is an
   answer about the instrument, and it is reported as one.
2. **B5's standing verdict does not move.** It is **neither corroborated nor
   re-opened.** It remains what it has been since 2026-08-10: a never-retracted
   claim resting on `trailing 0/30, entrance 0/717`, evidence **both** live
   hypotheses predict. A retirement gives B5 nothing — a VOID is not a
   corroboration, and this document may not be cited as one.
3. **The deferred rung-relative wavefront amendment STAYS DEFERRED — and its
   condition changes from "pending" to "closed by this route."** It was deferred
   on 2026-08-11 explicitly conditioned on *a written addendum re-opening B5*.
   With V_adv retired, that addendum will not come from V_adv. The amendment is
   therefore **not live, and is not to be revived by a further V_adv round.**
   Anything that would revive it must be a **different instrument with its own
   registration** — one that can observe the disputed behaviour rather than
   score a critic on states the policy does not occupy. Named candidates, none
   authorised here: a counterfactual-restart assay at the entry state, or an
   on-policy reward-decomposition read. Each needs its own bar written before
   its own compute.
4. **The `[inert-treatment]` ledger entry is amended** to record that V_adv was
   built, controlled, run twice, and retired — which is a materially different
   receipt from an instrument that was adopted and never written. That
   distinction is the entire lesson of the predecessor's §6, and closing this
   loop is what stops the ninth instance.
5. **The once-rule does not close B5 on this.** Retiring the instrument is not
   closing the question; it is recording that this instrument cannot answer it.
   The claim stays flagged as under-instrumented in `CLAIMS.md`.

---

## 12. What is on disk when this is done

| path | what |
|---|---|
| `docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md` | this file, committed before any number |
| `scripts/collect_onpolicy_bank.py` | the rollout collector |
| `tests/test_collect_onpolicy_bank.py` | purity guard, partition, done-on-clear, sticky gate — all revert-verified |
| `runs/vadv_onpolicy/iter_*.npz` | 26 per-iterate banks + per-episode sidecars |
| `runs/vadv_onpolicy/arc.json` | the 26-point `R` curve, all region readings, all A-gates |
| `runs/vadv_onpolicy/penetration.json` | `pen_rate`, `gx_max`, interior counts, pooled |
| `runs/vadv_onpolicy/probe_933.json` | the adjacent-rung diagnostic |
| `docs/research/VADV_ONPOLICY_2026-08-27.md` | the verdict document, written after |

`runs/` is gitignored; the JSON receipts are local artifacts.
