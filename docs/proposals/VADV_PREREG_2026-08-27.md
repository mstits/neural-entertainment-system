# V_adv PRE-REGISTRATION — the missing B5 discriminator

**Registered 2026-08-27, before any V_adv number was computed from any
checkpoint.** Data-coverage counts were measured first and are disclosed in
full below (§7); no effect size, no critic value, and no verdict statistic was
looked at before this file was written.

---

## 1. The standing verdict this instrument is pointed at

`docs/research/B5_PREREG_2026-08-08.md:414`, RUN 3 FINAL VERDICT (2026-08-10),
never retracted:

> THIS IS A REAL CAPABILITY WALL at gx ~2674-2872

Its evidence is `trailing 0/30, entrance 0/717` over 249 iterations at rung 893
(gx 2674, area 2) — confirmed in the banked checkpoint itself:
`checkpoints/mario_1_2_backward/prevrun_20260809_222300/vanilla_ppo_iter_00250.pt`
carries `backward_curriculum = {tau: 893, n_entries: 1094, window: [],
advances: 5, entrance_attempts: 423, entrance_successes: 0}`.

Both live hypotheses predict a zero success count:

* **CAPABILITY** — the policy cannot execute the traversal.
* **MIS-SPECIFICATION** — the reward gives no gradient there, so nothing could
  have learned it. Named confound (`plans/v15_d1_backward_curricula_verdicts.md:13`):
  a reachability-based advance gate paired with a progress-based reward.

`v15_d1` ADOPTED `V_adv = E_s[Var_a(Â)]` as the discriminator. It was never
implemented (parameter drift was substituted in as the fifth instrument). This
document registers its reading before it is run.

## 2. What V_adv is here, exactly

### 2.1 Where Â comes from

**The critic, one-step, over banked real successors. Not the actor.**

    Â(s, a) = γ · V_θ(s') · (1 − done)  −  V_θ(s)

`V_θ` is the scalar critic head of the banked checkpoint; `s'` is the **real
banked successor** of `(s, a)` from an offline transition bank; `γ = 0.99`
(`configs/mario_1_2_backward.yaml:181`). No emulator, no training, no rollouts.

Three rejected alternatives, and why:

* **Rollout-derived Â** — needs the emulator. Forbidden by the job's terms.
* **Policy-implied soft advantage** (`Â = centred logits`) — a pure function of
  the actor head. It is a monotone re-reading of `top_two_margin`, which is
  already instrument #2, and B6 measured margins *rising 14×* through the
  entropy collapse (`B5_PREREG_2026-08-08.md`, B6 FINAL VERDICT). An
  actor-sourced V_adv would therefore report LIVE at the wall for the wrong
  reason — confident-wrong noop sharpening — and would be the fifth vacuous
  instrument this job exists to avoid. **Explicitly excluded.**
* **Q-head advantage** — `TilePolicyNetwork` has `critic: Linear(trunk, 1)`.
  There is no per-action value head. `Var_a` cannot be read at a single state
  without successors; this is why the successor bank is load-bearing.

The immediate reward `r(s,a)` is **omitted** from Â and the omission is
declared. Direction of the resulting bias is registered here in advance: a
progress-based `r` would contribute *more* action-variance in a region where
progress is achievable than at a progress-flat wall, so omitting it makes the
wall/control ratio **larger** — i.e. biased toward reading LIVE, i.e. biased
toward **corroborating** B5's standing verdict, not toward re-opening it. The
omission is conservative against the conclusion that would be more interesting.

### 2.2 The statistic

Rows are grouped into **state cells**. Within a cell, `V_θ(s)` is (near)
constant across `a`, so `Var_a(Â) = γ² · Var_a(V_θ(s'_a))` — the quantity
"does the critic value this state's action-successors differently".

Two readings are computed on every region; the second is primary.

1. **`raw`** — the literal `E_s[Var_a(Â)]`: mean over cells of the population
   variance of the per-action *mean* advantage. In critic units. **Never
   compared across checkpoints.**
2. **`eta2`** — the pooled between-action share of advantage variance, with the
   state cell as a blocking factor:

       SS_between = Σ_cells Σ_a  n_{c,a} · (mean_{c,a}Â − mean_c Â)²
       SS_within  = Σ_cells Σ_a  Σ_i (Â_i − mean_{c,a}Â)²
       eta2       = SS_between / (SS_between + SS_within)

   Dimensionless, in `[0, 1]`, pooled so `n` is rows not cells.

### 2.3 Normalisation — and why an unnormalised V_adv would be vacuous

A raw `E_s[Var_a(Â)]` scales as `c²` under any critic rescale `V → cV + b`, so
it would collapse or inflate purely from critic scale drift across the training
arc. Three defences, all registered:

* `eta2` is a **ratio of two variances of the same `V_θ`** — exactly invariant
  under `V → cV + b`. Primary reading.
* `raw_norm = raw / Var_batch[V_θ(s)]`, with `Var_batch` computed once per
  checkpoint over the **pooled union of all scored regions** so every region
  shares one denominator. Same invariance class. Reported as a cross-check.
* **The verdict statistic is a within-checkpoint ratio between two regions
  scored by the same critic on the same bank**, so any residual scale term
  cancels identically.

**Degeneracy guard:** if `Var_batch[V_θ] == 0` (a constant critic), `raw_norm`
is `0/0`. It is reported as `None` with `degenerate_critic: true`, **not as
0.0**. A dead critic must read VOID, never COLLAPSED.

### 2.4 Cell construction — registered before running

* **Key**: the *last* stacked frame of the 712-dim v2 tile observation,
  indices `[534, 711)` — tiles, velocities, on_ground, powerup, lives, sub-x,
  gx page, gx fine — with the animation-phase byte (index 711) dropped.
  History frames are dropped from the key only; **Â is always computed from the
  full 712-dim observation the network actually consumes.**
* **Qualification**: a cell is scored iff ≥ 2 distinct actions each have ≥ 2
  rows and the qualifying rows total ≥ 6. Rows for actions below the per-action
  minimum are dropped from the cell.
* **Sensitivity check** (registered, secondary): exact 712-byte state key.

Loose grouping merges states that are not identical. That inflates the raw
between-action term with state heterogeneity — which is precisely why `eta2`
(which pushes heterogeneity into `SS_within`) is primary and `raw` is not.

### 2.5 Region labels come from the observation, not from a RAM map

`SMBTileObservationV2` already exposes `out[175] = global_x >> 8` and
`out[176] = (global_x & 0xFF) >> 1` (`src/emulation/tile_observations/smb.py`).
Absolute gx is therefore decodable from the banked observation itself:

    gx = obs[534 + 175] · 256 + obs[534 + 176] · 2

(2-pixel quantisation from the encoder's own `>> 1`.) No new address is read,
no disassembly is consulted; this is the encoder's existing contract.

Area disambiguation: in 1-2, area 1 tops out at gx 151
(`checkpoints/ge_1_2_rungs_gx/ladder.txt`), so **any gx ≥ 160 is unambiguously
area 2**. Every band below starts at or above 160 for that reason.

## 3. State sampling — which states, and why they span both regions

**Primary bank (1-2, true `next_state`):** `runs/smodice_1_2/transitions.npz`
(40,785 rows) ∪ `runs/iq_1_2/transitions.npz` (70,020 rows) = 110,805
transitions with `state / action / next_state / done`, sourced from the
project's own go-explore 1-2 solutions replayed under the eval's own sticky-0.25
drift (`scripts/smodice_data.py`). Stratified ~50/50 progressed vs failed
windows, so the action axis is not survivorship-filtered.

**Bands (gx, area 2 of 1-2):**

| label | band | what it is |
|---|---|---|
| `WALL` | `[2674, 2872)` | rung 893. `trailing 0/30`, `entrance 0/717`, 249 iterations. **The test region.** |
| `PC_B5` | `[2872, 3266]` | the rungs B5's own curriculum **advanced through** at ≥ 20 % trailing (tau walked 1093 → 893, "5 advances through the flag zone"). **Positive control.** |
| `APPROACH` | `[2400, 2674)` | states leading into the wall. Registered secondary. |
| `EARLY` | `[160, 1600]` | early 1-2. Registered secondary **only** — see the warning below. |

**Why `PC_B5` and not early 1-2, for the B5 arc.** B5 is a *backward*
curriculum pinned at rung 893 (gx 2674). Its on-policy support is gx ≥ 2674.
Early 1-2 is **off-distribution for B5's critic**, so a flat critic there would
be an artifact of never having been trained there, not evidence about reward
gradient. `PC_B5` is adjacent to the wall, on-distribution, learned by *this
run*, and scored by *this run's* critic. That is the only positive control for
which "learned vs not learned" is the sole difference between the two bands.

**Secondary bank (1-1, instrument validation):**
`runs/interference/success_1_1.npz` — 80,327 observations across 189 *successful*
1-1 trajectories with `traj_len` boundaries, so consecutive rows **within a
trajectory** are true `(s, a, s')` transitions (80,138 of them; splices across
episode boundaries are excluded by construction, not by a filter).

**Registered coverage escalation, to be used only if §6's power gate fails:**
`runs/funnel_1_2_dr10/funnel_1_2_merged.npz` (458,580 rows) and
`runs/dars_1_2/wall_*.npz` (18k–50k rows) are `obs/act` only with no trajectory
boundaries; transitions would be consecutive rows filtered by
`|Δgx| ≤ 24` to drop episode splices. If used, they are reported as a
**separate secondary bank with its own verdict line**, never merged into the
primary reading.

## 4. Controls

### 4.1 POSITIVE control — must be LIVE or nothing else is reported

* **PC-1 (instrument validation).** B4 / v4 1-1-backward iterates
  (`checkpoints/mario_1_1_backward/vanilla_ppo_iter_*.pt`,
  `checkpoints/mario_1_1_backward_v4/vanilla_ppo_iter_*.pt`) scored on the 1-1
  success-trajectory transitions, gx `[160, 3100]`. 1-1 is learnable at 0.76
  honest; if V_adv is not LIVE on a checkpoint arc that demonstrably learned
  1-1, **the metric is broken and proves nothing at the wall.**
* **PC-2 (within-B5 reference).** B5 run-3 iterates scored on `PC_B5`
  (gx `[2872, 3266]`), the band its own curriculum advanced through.

### 4.2 NEGATIVE controls — must COLLAPSE

* **NC-a — zeroed critic (mechanism check).** Set `critic.weight := 0` on a real
  checkpoint. `V_θ` is then provably constant, so the reward-carrying signal is
  provably absent. Requirement: `raw == 0.0` exactly, `eta2` undefined, and the
  `degenerate_critic` flag set. This also proves the degeneracy guard fires
  rather than silently emitting a comfortable 0.0.
* **NC-b — a region where the reward provably carries no one-step gradient.**
  Cells in which **every** qualifying transition has `gx(s') == gx(s)` — no
  action tried at that state moves progress at all, so a progress-based reward
  is provably flat across the tried actions. Measured availability: 469 such
  cells / 25,828 rows in the primary bank. Drawn **from the same gx band as the
  positive control** so region is held fixed and reward-flatness is the only
  difference.
  *Registered caveat and its disposition:* `V_θ` predicts discounted return, not
  one-step reward, so an NC-b cell can legitimately stay live if one action
  leads toward eventual progress. NC-b is therefore a **conservative** null. **If
  NC-b does not collapse, the "reward provably flat" null is VOID, only NC-a and
  the permutation null survive, and the wall reading may then be reported as at
  best INDETERMINATE — never as a verdict.** Registered now, not after seeing it.
* **NC-c — within-cell action-label permutation (defines the collapsed line).**
  Permute `a_i` within each cell, recompute `eta2`, 1,000 draws, seed 20260827.
  This destroys the action↔successor pairing while preserving every marginal:
  the cell structure, the successor-value distribution, the critic, and the
  critic's scale. It is the exact reference distribution for "this region shows
  no action effect".

## 5. Thresholds — the numbers, written down now

Let `null_q(p)` be the p-quantile of NC-c's permutation distribution for that
region, and `[lo, hi]` the 95 % bootstrap CI of observed `eta2` (2,000
resamples over cells, seed 20260827).

* **LIVE(region)** ⟺ `eta2_obs > null_q(0.975)` **and**
  `eta2_obs / median(null) ≥ 1.5`.
* **COLLAPSED(region)** ⟺ `eta2_obs ≤ null_q(0.975)` **and**
  `hi(region) < lo(positive control)`.
* Anything else: **INDETERMINATE**.

**Verdict statistic** (both excesses over their own permutation medians, so
each region is measured against its own null):

    R = eta2_excess(WALL) / eta2_excess(PC_B5)

* **MIS-SPECIFICATION signature** ⟺ `PC_B5` LIVE **and** `WALL` COLLAPSED
  **and** `R ≤ 0.20`.
  → B5's verdict is re-opened with a written addendum; the rung-relative
  wavefront amendment deferred since 2026-08-11 becomes live again.
* **CAPABILITY signature** ⟺ `PC_B5` LIVE **and** `WALL` LIVE **and**
  `R ≥ 0.50`.
  → B5's verdict is corroborated by an instrument that could have contradicted
  it, which is worth strictly more than the original assertion.
* **VOID** ⟺ any admissibility gate in §6 fails, **or** `PC_B5` and `WALL` are
  not separable, **or** `R` falls in `(0.20, 0.50)`.
  VOID is a real outcome and is reported as itself. **It is not FAIL, and it is
  not licence to re-read the numbers a different way.**

**Acting-range check (the ReDo lesson — τ 0.025 against a ≥ 0.25 firing
threshold, never fired in v27 or v28).** Every threshold above is a **ratio
against a reference measured on the same data**, not an absolute constant:

* `eta2 ∈ [0, 1]` by construction.
* Its H0 expectation is **not** 0 — it is `df_between / (df_between + df_within)`,
  which for a 2-action cell with 3 rows is `1/2`. A naive absolute threshold
  would have been catastrophically outside its acting range. This is exactly why
  the cell-qualification rule (§2.4) and the permutation null (NC-c) exist, and
  why `eta2_null_analytic` is emitted on every reading so the gap between the
  observed value and its own null is visible on the face of the output.
* `R = 1.0` means "the wall discriminates actions exactly as much as the region
  the run demonstrably learned". `0.20` and `0.50` straddle that. They cannot be
  out of range the way an absolute τ can.

## 6. Admissibility — checked and reported BEFORE any wall number is read

| gate | requirement |
|---|---|
| **A1** | PC-1 LIVE on the 1-1 arc. If the metric cannot show a live signal where the agent demonstrably learned, **stop and report that**; nothing at the wall is reported. |
| **A2** | PC-2 (`PC_B5`) LIVE. |
| **A3** | NC-a returns `raw == 0.0` with `degenerate_critic: true`. |
| **A4** | NC-c's null is non-degenerate (`null_q(0.975) > null_q(0.025)`). |
| **A5** | Critic non-degenerate on the pooled batch: `Var_batch[V_θ] > 0`. |
| **A6** | Coverage: ≥ 20 qualifying cells **and** ≥ 400 qualifying rows in each of `WALL` and `PC_B5`. |
| **A7** | **Power gate, binding.** Inject a synthetic action effect of the size measured in `PC_B5` into `WALL`'s own rows and re-run the test, 200 injections. Detection must be ≥ 0.95. If `WALL`'s actual `n` could not detect a `PC_B5`-sized effect, "collapsed at the wall" is unfalsifiable there → **VOID**, escalate coverage (§3), do not report. |
| **A8** | Checkpoint input width equals the bank's observation width (712). Mismatch → VOID for that checkpoint, never a `strict=False` partial load. |

**If A1 and A3 do not both hold — the positive control live and the negative
control collapsed — the metric does not discriminate on this data, the job stops
there, and that is the whole report.**

## 7. Coverage measured before registration (full disclosure)

Measured on the pooled primary bank under the §2.4 cell rule, **before** any
threshold above was written and before any critic was loaded:

| band | qualifying cells | qualifying rows |
|---|---|---|
| `WALL [2674, 2872)` | **28** | **612** |
| `PC_B5 [2872, 3266]` | 392 | 24,391 |
| `APPROACH [2400, 2674)` | 13 | 375 |
| `EARLY [160, 1600]` | 209 | 11,762 |
| NC-b gx-frozen (all bands) | 469 | 25,828 |
| 1-1 secondary bank (PC-1) | 1,503 | 22,336 |

`WALL` is thin: 28 cells / 612 rows, against an A6 gate of 20 / 400. The gate
was set from what a pooled permutation test needs, and the measurement is
disclosed here rather than the gate being moved to fit it. `APPROACH` (13 cells)
**fails A6 and is pre-declared VOID** on the primary bank.

Thin `n` costs **power**, not validity — a permutation null is exact at small
`n`. Low power biases toward failing to detect LIVE, i.e. toward reading
COLLAPSED, i.e. toward the *more interesting* conclusion. **A7 exists precisely
to stop that**, and it is binding.

## 8. Anti-vacuity test shipped with the mechanism

`tests/test_score_banked_iterates.py`:

* identical advantage across actions ⇒ `eta2 == 0`, verdict `COLLAPSED`;
* separated per-action advantage ⇒ `eta2` high, verdict `LIVE`;
* the permutation null on a live batch sits **below** the observed value;
* `eta2_null_analytic` is emitted so a small-cell reading cannot be mistaken for
  a signal;
* the degeneracy guard returns `None` + `degenerate_critic`, never `0.0`;
* both verdicts are checked to **flip** when the mechanism is reverted (V_adv
  stubbed to a constant), so the assertions cannot pass vacuously.

## 9. Known risks, registered

1. **Off-distribution critic reads.** The banks are go-explore expert-window
   states, not B5's on-policy rollouts. Mitigation: `PC_B5` and `WALL` are
   adjacent bands from the same bank and the same collection process, and the
   verdict is their ratio.
2. **`r` omitted from Â** (§2.1) — conservative against re-opening B5.
3. **Thin `WALL` coverage** (§7) — A7 is the guard.
4. **Loose cell key merges non-identical states** — pushed into `SS_within` by
   construction; exact-key sensitivity check registered.
5. **NC-b is a one-step null against a return-predicting critic** — disposition
   pre-registered in §4.2.
6. **2-pixel gx quantisation** from the encoder's `>> 1`; band edges are far
   from any 2-pixel ambiguity.
7. **A live V_adv does not prove capability**, only that the critic
   discriminates actions there. The CAPABILITY reading is corroboration of B5,
   not an independent proof of it, and must be written that way.

---

# APPENDIX — RESULTS (post-registration; everything above was fixed first)

Run 2026-08-27, `scripts/score_banked_iterates.py`, primary bank
(`runs/smodice_1_2` ∪ `runs/iq_1_2`, 110,805 transitions), checkpoint
`checkpoints/mario_1_2_backward/prevrun_20260809_222300/vanilla_ppo_iter_00250.pt`
(B5 RUN 3, the arc that produced the standing verdict).
`n_perm = 300`, `n_boot = 300`, `gamma = 0.99`, loose cell key.

## Admissibility

| gate | result |
|---|---|
| **A1** PC-1, 1-1 arc (`mario_1_1_backward_v4` iter 150 × 1-1 success transitions) | **PASS — LIVE.** η² = **0.667**, permutation null median 0.114 / q97.5 0.189, bootstrap CI [0.528, 0.794], 1,425 cells / 19,003 rows. The metric shows a strong live signal exactly where the agent demonstrably learned. |
| **A3** NC-a zeroed critic | **PASS — VOID.** `degenerate_critic: true`, `raw = 0.0`, `raw_norm = None`. |
| **A5** critic non-degenerate | PASS. `Var_batch[V] = 8979.1`. |
| **A6** coverage | PASS for `WALL` (28 cells / 612 rows) and `PC_B5` (392 / 24,391). **`APPROACH` FAILS (13 cells) and is VOID as pre-declared.** |
| **A7** power gate on `WALL` | **PASS.** Injected effects at 0.25 / 0.5 / 1.0 / 2.0 × sd(Â) — induced η² 0.17 / 0.29 / 0.56 / 0.83 — all detected at power **1.0** (30 trials each). The wall band is thin but **not** underpowered; a `PC_B5`-sized effect there would have been seen. |
| **A8** width match | PASS (712 = 712). |

**A3 caught a live defect in this instrument.** The zeroed critic first reported
`COLLAPSED`, not `VOID` — the arithmetic carve-off in `classify_vadv` read a
uniformly-zero advantage as a confident absence of action effect, manufacturing
the mis-specification signature out of a provably broken checkpoint. Fixed
(`score_vadv` now overrides to VOID with `void_reason: degenerate_critic`) and
pinned by a test. A separate float-dust defect was found the same way: summing
squared deviations of a constant leaves ~1e-31 residue, and the ratio of two
dust terms gave a variance-free batch **η² = 0.889, a confident LIVE**. Both
fixes are revert-verified.

## Readings

| region | cells | rows | η² | null med | null q97.5 | boot 95 % CI | verdict |
|---|---|---|---|---|---|---|---|
| `PC_B5` [2872, 3267) | 392 | 24,391 | **0.307** | 0.079 | 0.106 | [0.210, 0.423] | **LIVE** |
| `WALL` [2674, 2872) | 28 | 612 | **0.133** | 0.070 | 0.121 | [0.071, 0.363] | **LIVE** |
| `EARLY` [160, 1600) | 209 | 11,762 | 0.312 | 0.086 | 0.147 | [0.163, 0.507] | LIVE (secondary) |
| `APPROACH` [2400, 2674) | 13 | 375 | 0.120 | 0.064 | 0.098 | — | **VOID (A6)** |
| `NC-b` gx-frozen | 129 | 17,151 | 0.175 | 0.033 | 0.048 | [0.089, 0.303] | **LIVE — the null did not collapse** |

**Verdict statistic:**

    R = eta2_excess(WALL) / eta2_excess(PC_B5)
      = (0.133 - 0.070) / (0.307 - 0.079)
      = 0.0627 / 0.2281
      = 0.275

## VERDICT: **VOID**

`R = 0.275` falls inside `(0.20, 0.50)` — the band pre-declared as too
ambiguous to move a verdict in either direction. Independently, `WALL`'s
bootstrap CI [0.071, 0.363] overlaps `PC_B5`'s [0.210, 0.423], so the COLLAPSED
condition (`hi(WALL) < lo(PC_B5)`) is not met either. And **NC-b did not
collapse** (η² 0.175 vs its null q97.5 0.048), which under §4.2's
pre-registered disposition voids the "reward provably flat" null and caps this
reading at INDETERMINATE by itself.

Neither registered signature fires:

* **not** MIS-SPECIFICATION — that needed `WALL` COLLAPSED and `R ≤ 0.20`.
* **not** CAPABILITY — that needed `R ≥ 0.50`.

**B5's standing verdict is neither corroborated nor re-opened by this
instrument on this data.** It remains what it was before this job: a
never-retracted claim resting on evidence that both hypotheses predict.

What *can* be said without moving a goalpost:

1. **The critic is not flat at the wall.** `WALL` is LIVE against its own
   permutation null, and A7 shows that is not an artifact of thin data. The
   strong form of the mis-specification story — "the critic cannot tell the
   actions apart at gx 2674" — is **not** what the data shows.
2. **But the action signal there is markedly weaker** than in the band B5's own
   curriculum advanced through: 27.5 % of the excess. That is a real gradient
   difference in the direction the mis-specification hypothesis predicts, at a
   magnitude the registration declared insufficient to act on.
3. The instrument is **not vacuous**: it reads 0.667 on a demonstrably-learned
   arc, VOID on a provably dead critic, and it separated four regions of the
   same level scored by the same critic.

**VOID is the outcome, and VOID is not FAIL.** The registration is not re-read
to find a reading it supports.

## Named next steps (not run; each would need its own registration)

* Score the **whole B5 iterate arc** (iters 10–260), not one checkpoint —
  `R` may move across the collapse, and the single-checkpoint reading cannot
  see that.
* Escalate `WALL` coverage with the registered secondary banks (§3) to tighten
  the bootstrap CI, which is what the ambiguity is made of.
* The `PC_B5`-vs-`WALL` contrast assumes the expert-window bank represents both
  bands equally. A bank collected from B5's **own** rollouts at both rungs would
  remove risk 1 entirely.
