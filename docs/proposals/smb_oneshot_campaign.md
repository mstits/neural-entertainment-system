# CAMPAIGN SPEC — "SMB One-Shot": sequential cold clear of World 1 (1-1→1-4)

**Judge-synthesizer verdict.** Adopt **Angle A (curriculum-maximalist) as the
spine**, graft **four specific pieces from Angle B**. Angle A wins the two axes
the brief elevates above all others — **cold-eval-metric integrity** and
**wall-breaking power** — because (a) it *found and fixed* the real bug that makes
the DoD unmeasurable today, and (b) it composes both historically-proven
wall-breaks (June dense-ladder for 1-2, July 1-4 backward-state ladder + Go-Explore)
instead of discarding one. Angle B contributes a cleaner warp guard, a cell-key
bug fix, a sharper fail-fast tripwire, and confirms the winner-rekey.

---

## 0. Scoring (out of 10 per axis)

| Axis | Angle A | Angle B | Winner / graft |
|---|---:|---:|---|
| **Cold-eval-metric integrity** | **10** | 4 | **A decisive.** A found `eval_game.py` breaks on `cleared_any` (latches at the 1-1 flagpole, rewards.rs:1480-1513 / eval:233-238) and reconstructs the sequential predicate from RAM via `won` semantics. B subprocesses `eval_game.py` and parses `clear_rate` — the *latched 1-1 rate* — keying the winner on the scarred metric. |
| **Wall-breaking power (1-2 & 1-4)** | **9** | 6 | **A decisive.** A keeps the dense per-level ladder (`checkpoint_scale=1`, the June 1-2 recipe) AND pre-seeds the July 1-4 backward states. B sets `checkpoint_scale=0`, deleting the exact mechanism that broke the 1-2 wall, betting the archive replaces it (proven on 1-4, NOT on 1-2). |
| **Forgetting-resistance** | **9** | 8 | A: fixed 25% floor on every cleared level-entry + cold regression alarm. B: depth-banded uniform return sampling. Both strong; **A's guaranteed per-level floor** is more explicit. Adopt A. |
| **Warp-zone handling** | 8 | 8 | Tie on metric-immunity (both use `won` semantics). **Graft B's `smb_sequential_cell → None` admission predicate** — strictly cleaner than A's hand-wavy "reward nudge" layer 3, and required as the GE-burst cell key. |
| **Minimality in contended lane** | 8 | 8 | Both concentrate in `_run_vanilla_ppo`. A is *additive over proven machinery*; B *deletes* the advance gate (riskier than it looks). Adopt A's additive shape; defer GE-burst out of the first edit. |
| **Fail-fast milestone quality** | 9 | 8 | A: 5 phases + per-phase iter budgets + 2×-budget halt. **Graft B's "no NEW cold area byte in 150 iters → stall" tripwire** as a second, sharper gate (whichever fires first). |

**Grafts from B into the A spine:** (1) `smb_sequential_cell()` warp-admission
predicate (Q1/Q5), (2) the world/area-in-cell-key bug fix, (3) the
150-iter-no-new-cold-area fail-fast tripwire, (4) re-key `save_winner` on the cold
sequential rate (A concurs). **Rejected from B:** deleting the dense ladder
(`checkpoint_scale=0`); making the archive the sole curriculum; cyclic consolidation
as the *primary* (kept only as an abort-fallback, §Q4).

**One synthesis improvement over BOTH designs:** the sequential predicate is
implemented **exactly once**, in `eval_game.py --sequential` (a `SequentialTracker`
helper). The in-loop probe *subprocesses that same flag* — so the number that gates
training is byte-identical to the number that scores the DoD. A had two predicate
sites (`cold_eval.py` + `eval_game.py --sequential`); B had one wired to the wrong
metric. This spec has one, wired to the right metric.

---

## 1. Chosen architecture (one paragraph)

A **hand-ordered sub-stage ladder** (world, area, x-bucket) generalizes the proven
area-byte curriculum: 3 buckets each for 1-1/1-2/1-3, a **dense 8-bucket ladder for
1-4** pre-seeded from the existing `stage_1_4_bw_x*` blobs so we warm-start from the
Bowser bridge on iter 0 instead of waiting to discover it. The existing 50%/5-iter
rolling advance gate is reused verbatim, measuring `region_of(ram)` (sub-stage order)
instead of the scalar packed byte. A **three-way warm-start partition** (Frontier 50 /
Retention 25 on cleared level-entries / Spread 25) holds early levels. A **cold-eval
probe** — subprocessing `eval_game.py --sequential` (greedy argmax from the 1-1 start,
sequential predicate from `won` RAM semantics) every 25 iters — **is the primary
metric, the regression alarm, and the winner-selector** (`best_cold.pt`). Warps are
blocked by construction (off-ladder + `smb_sequential_cell→None` admission) and
metric-immune (`won` semantics), reported only as a "beyond" stat. Entropy
consolidation (0.01→0.002) is **gated on the cold probe proving the frontier reached
1-4, and is reversible**. Pixel-CNN is the deliverable vehicle; a zero-new-code tile
scout front-runs ladder validation and generates encoder-agnostic seed states.
Go-Explore survives only as a **deferred, bounded unstick subroutine** invoked on a
stall.

---

## 2. Definitive answers to the 9 hard questions

### Q1 — Sub-stage design
Replace the scalar `smb_curriculum_anchors[stage]` (packed `world<<4|area`) with an
**ordered `SubStage(order, world, area, x_lo, x_hi, seed_blob, reward_profile)` list**
(the sequential World-1 path). **Anchor tuple = `(world, area, x_bucket)`**, identity
from `smb_sequential_cell(ram)` (world `$075F` unbucketed, area `$0760` unbucketed, x
`$006D<<8|$0086` bucketed). **Buckets/level:** 3 for 1-1/1-2/1-3 (entry / mid-chokepoint
/ exit), **8 for 1-4** (~256 x-units each, matching the pre-existing backward states).
**Gate thresholds:** the proven **50% over a 5-iter rolling window**, uniform across
sub-stages, applied to `region_of(ram) ≥ k+1` for current sub-stage `k`. **Seeding:
disk-first, live-fill.** At build, glob `stage_0N.state`, `depth_0_<area>_<x>.state.bin`
(29,738 available), `stage_1_4_bw_x*.state`; bind each sub-stage to its nearest-x blob.
Sub-stages with no disk seed fall back to the existing mid-rollout live capture
(`smb_pending_capture`, trainer 4868-4879), extended to fire on **x-bucket** crossings.
`region_of()` credits only sequential-path states, so a warp cannot raise the region.
*(Angle A spine; `smb_sequential_cell` cell key from Angle B.)*

### Q2 — Forgetting (retention + regression alarm)
**Warm-start partition** of the 24-env pool, recomputed each iter from frontier order `F`:
- **Frontier 50%** — sub-stages `F` and `F-1`.
- **Retention 25%** — the **level-entry** sub-stage of every already-cleared level
  (1-1 cold, 1-2 entry, 1-3 entry), sampled uniformly. **Fixed floor: ~6 envs ×
  1024 steps on each cleared level every iter, regardless of frontier depth.**
- **Spread 25%** — uniform over all buckets below the frontier.

**Regression alarm = the cold probe.** Maintain a per-level cold high-water mark. If
`cold_furthest_seq` for an already-cleared *early* level regresses for **2 consecutive
probes**: (a) log `FORGETTING`, (b) raise Retention → 40% for 100 iters, (c) if it
persists, **roll back to `best_cold.pt` + reset the optimizer** (reuse anti-collapse
path, trainer 5745-5751). **Winner selection: `best_cold.pt` keyed on
`cold_seq_clear_rate`** (re-key `save_winner`, 5967) — training telemetry never selects
the deliverable. *(Angle A; winner-rekey concurred by B.)*

### Q3 — Go-Explore × curriculum (precedence)
**ONE driver: the sub-stage curriculum.** Go-Explore is a **deferred, bounded unstick
subroutine**, never concurrent. The code's mutual exclusion (`go_explore_on = enabled
and not smb_curriculum_active`, 4612) is preserved: the burst is a **direct
`GoExploreArchive` call**, not the `go_explore_on` branch, so it never flips the
curriculum flag. **Trigger:** current sub-stage's gate hasn't fired for 60 iters AND
the cold probe already *reaches* that sub-stage stochastically (block is "find the next
frontier state," not "can't play this one"). **Action:** ≤30-iter archive burst seeded
from the stalled sub-stage, cell = `smb_sequential_cell` (world/area IN the key — B's
fix). **Harvest:** write the archive's furthest state as sub-stage `F+1`'s `seed_blob`,
resume curriculum. **This lane is DEFERRED** — built only if a stall actually occurs
(the 1-4 backward ladder is already seeded, so the burst is expected 0–1 times). Keeps
the first contended edit clean. *(Angle A precedence; B's cell fix.)*

### Q4 — Consolidation
**Primary: gated, reversible terminal decay** (proven June recipe). **Trigger (BOTH
required):** (1) a cold probe shows `cold_furthest_seq` reached **area 4 (1-4)**, AND
(2) the frontier is the 1-4 tail (order ≥ 15) with ≥1 stochastic WIN (world→2, `won`)
in the last window. **Schedule:** linear `entropy_coef` 0.01→0.002 over 80 iters +
`rnd_intrinsic_coef` 0.1→0.0, **freezing the warm-start distribution** (hold Frontier
on the 1-4 tail + Retention on 1-1..1-3). **Abort/rearm (reversibility is the
anti-premature-commit safeguard):** if a probe during consolidation shows early-level
forgetting OR greedy 1-4 clear fails to rise for 40 iters → restore `entropy_coef=0.01`
and resume frontier training. **Abort-fallback:** if terminal consolidation aborts
twice, switch to a short explore(0.03)↔consolidate(0.003) oscillation with probes in
the low phase (**Angle B's cyclic idea, demoted to fallback**). Never consolidate on a
clock. *(Angle A primary; Angle B cyclic as fallback.)*

### Q5 — Warp zones
**Block on the sequential track; report as "beyond."** Justification: the DoD is
explicitly *sequential* World-1; embracing the warp optimizes an easier objective and
starves 1-3/1-4. Three grounded layers:
1. **Off-ladder by construction.** Sub-stages ordered by sequential `order`; a warp
   lands in world-2 area-0, matching no sub-stage → cannot advance the frontier and
   cannot be selected as a warm-start seed.
2. **Admission predicate (B's graft, replaces A's reward nudge).** `smb_sequential_cell`
   returns `None` for world≠1 or non-allowlisted area (warp-room area byte excluded) →
   warp states never seed the ladder, never enter any archive burst, never get captured
   live. A ~3-line guard at the capture/seed sites.
3. **Metric warp-immunity.** The cold predicate credits a clear only on `won`
   (world→2 with prev `$075C==3`, the shipped F52 guard, rewards.rs:1322). A 1-2 warp
   (world rise from `$075C==1`) scores `seq_clear=0, warp_taken=1`, reported under
   `cold_furthest_any` ("beyond"), never DoD. Observed continuously via `cold_warp_rate`;
   if it climbs, escalate to zeroing the 1-2 warp-approach checkpoint (one-line
   reward-table edit). *(Angle A layers 1+3; Angle B admission predicate for layer 2.)*

### Q6 — Encoder
**Primary: pixel-CNN (`nature_dqn`).** The only historically **cold-capable** SMB
clears (1-2 June, 1-4 stochastic July) were on pixels; 1-4's firebar/lava/Bowser
representation on the 13×13 tile grid is unproven for a *greedy* castle clear, and **no
cross-encoder transfer exists** (committing to tile then switching = full retrain — not
costed because it is rejected). Cost accepted: ~25–40 s/iter (pixel) vs 5.5–12 s (tile);
the dense seeded ladder offsets it by collapsing each rung's exploration horizon.
**Secondary: tile scout (zero new code, YAML-only).** The ladder, `smb_sequential_cell`,
probe, and predicate are all encoder-agnostic. `smb_oneshot_tiles.yaml` (`encoder:
smb_tiles`) runs ~4× faster to (1) validate ladder mechanics before the pixel run burns
compute, and (2) generate encoder-agnostic `depth_*.state.bin` seeds the pixel ladder
consumes. Its *policy* is discarded; DoD is claimed on pixel. *(Both designs concur.)*

### Q7 — Reward config
**ONE global platform-shaping profile** (`smb_1_4_dense` weights) — the per-level dense
checkpoint ladder auto-switches by area byte inside a single episode via
`checkpoints_for()` (rewards.rs:1154), so only scalar weights are global:
`forward_progress 3.0`, `completion_bonus 2000`, `time_penalty -0.05`,
`death_penalty 0.0`, `air_bonus 1.5`, `jump_clear_bonus 50 (min_dx 16)`,
`survival_bonus 0.0` (camping is a platform local-optimum), `rnd_intrinsic 0.1`,
**`checkpoint_scale 1.0` (dense ladders ON — the June wall-break; this is where the
spec KEEPS A over B).** **1-4 needs NO bespoke reward stage:** it is linear x-progress,
`LEVEL_1_4` runs dense checkpoints to the axe at x2560 (rewards.rs:1116-1147); its
difficulty is handled by **sub-stage density (8 seeded buckets)**, not new shaping.
`entropy_coef`/`rnd_coef` change only via the §Q4 consolidation decay — one global
schedule, not per-stage config, keeping everything in one lane. *(Angle A.)*

### Q8 — Budget + milestones
Pixel primary, 24 envs, ~30 s/iter ⇒ ~120 iters/hr. Probe every **25 iters** (8
greedy eps for the curve; full 24 greedy+stochastic at each GO gate). See §5 table.
**Two fail-fast tripwires, whichever fires first:** (a) A's **2× per-phase iter
budget** without cold advance → HALT; (b) B's **no NEW cold area byte in 150 iters**
→ stall (escalate: GE-burst / richer seeding, or abort the arm). Total ~1,850 iters
≈ 1.5–2 days pixel; tile scout front-runs A–C in ~½ day. *(A budget + B tripwire.)*

### Q9 — New code (minimal, one contended lane)
See §3 work-list. Net-new ≈ 300 LOC, ~200 in isolated/unit-tested modules; the
contended `_run_vanilla_ppo` edit is ~120 lines of thin glue routing through those
modules. Rust untouched (sequential predicate reconstructed in Python from RAM). The
GE-burst hook (Lane 5) is deferred out of the first edit. **One implementation lane
owns `_run_vanilla_ppo`; no other work may edit that block concurrently.**

---

## 3. Implementation work-list (per lane: files, size, tests)

| Lane | Change | Files | Size | Tests |
|---|---|---|---:|---|
| **1** *(first, isolated)* | **Sequential predicate + `eval_game.py --sequential`.** `SequentialTracker` (update(ram); `.seq_clear`, `.furthest_seq`, `.furthest_any`, `.warp_taken`) + `smb_sequential_cell(ram, x_bucket=8)→tuple\|None` (world/area unbucketed, warp-admission). `--sequential`: don't break on `cleared_any`; run to death/timeout/WIN; emit `seq_clear_rate`, `furthest_seq`, `furthest_any`, `warp_rate`. | **new** `src/training/smb_sequential.py` (~70); **edit** `scripts/eval_game.py` (~40) | ~110 | Unit: synthetic RAM traces — full 1-1→1-4 castle clear ⇒ `seq_clear`; 1-2 warp ⇒ `seq_clear=0, warp_taken=1, furthest_any=world2`; region monotonic, no rise on warp. Golden: replay a recorded 1-1-clear trace if present. |
| **2** *(isolated)* | **Sub-stage ladder builder + `region_of`.** `SubStage` dataclass; `build_ladder(seed_glob)` binds disk seeds by nearest-x; `region_of(ram)→order` via `smb_sequential_cell`. | **new** `src/training/smb_substage_ladder.py` | ~140 | Unit: build from a temp dir of fake `.state` files ⇒ correct order + nearest-x binding; `region_of` correct on synthetic RAM; warp RAM does NOT raise region. |
| **3** *(isolated)* | **Cold-eval probe wrapper.** `probe(net, cfg, episodes, device)→dict` writes net→temp ckpt, subprocess `eval_game.py --sequential --checkpoint <tmp> --episodes N`, parse JSON → `{cold_seq_clear_rate, cold_furthest_seq, cold_furthest_any, cold_warp_rate}`. | **new** `src/training/cold_probe.py` | ~50 | Unit: run against a known checkpoint; assert keys + ranges; mock subprocess for the parse path. |
| **4** *(THE contended lane — single owner)* | **`_run_vanilla_ppo` rewire.** (a) build ladder + `region_of` advance (reuse 50%/5 gate); (b) three-way retention warm-start; (c) periodic `cold_probe`, emit `cold_*` metrics, per-level high-water, `best_cold.pt` (re-key `save_winner` 5967 on `cold_seq_clear_rate`), forgetting alarm + rollback; (d) gated reversible consolidation; (e) warp guard at capture/seed sites (skip `smb_sequential_cell==None`). | **edit** `src/training/trainer.py` (~4197-5967) | ~120 | Smoke: short run, tiny env count ⇒ ladder builds, advance fires on synthetic, `cold_*` in `metrics.jsonl`, `best_cold.pt` written, forgetting alarm triggers on injected regression. `make parity` unaffected. |
| **5** *(DEFERRED — build only on stall)* | **Go-Explore unstick burst.** Direct `GoExploreArchive` call seeded from stalled sub-stage, cell = `smb_sequential_cell`, ≤30 iters, harvest furthest → sub-stage seed. | **edit** `src/training/trainer.py` (guarded) | ~40 | Unit on the harvest hook; only wired if Phase D stalls. |
| **6** *(data)* | **Configs.** `smb_oneshot.yaml` (pixel, `smb_curriculum:true`, `substage_ladder:true`, platform reward `checkpoint_scale:1`, `cold_eval`/`consolidate`/`go_explore_fallback` knobs) + `smb_oneshot_tiles.yaml` (encoder swap only). | **new** `configs/` | 2 files | Config-schema load test. |

**Lane order:** 1 → 2 → 3 → (4 depends on 1+2+3) → 6 → 5 (deferred). Lanes 1–3 are
independently testable with ZERO trainer edits; Lane 4 is the only contended edit.

---

## 4. Campaign profile config outline (`configs/smb_oneshot.yaml`)

```yaml
game: super_mario_bros
encoder: nature_dqn            # pixel-CNN, the deliverable vehicle
training_mode: vanilla_ppo
num_envs: 24                   # machine sweet spot (60 for tile scout)
start_state_path: <1-1 cold boot>   # DoD reference start — MUST be set

smb_curriculum: true
substage_ladder: true          # NEW: ordered (world,area,x) sub-stage list
  seed_globs:                  # disk-first binding
    - checkpoints/super_mario_bros/smb_curriculum/stage_0*.state
    - checkpoints/super_mario_bros/smb_curriculum/stage_1_4_bw_x*.state
    - checkpoints/auto_curriculum/depth_0_*.state.bin
  buckets: {1_1: 3, 1_2: 3, 1_3: 3, 1_4: 8}
advance: {pct: 0.50, window: 5}          # reuse proven gate, on region_of()
warm_start: {frontier: 0.50, retention: 0.25, spread: 0.25}

reward:                        # ONE global platform-shaping profile
  forward_progress: 3.0
  completion_bonus: 2000.0
  time_penalty: -0.05
  death_penalty: 0.0
  air_bonus: 1.5
  jump_clear_bonus: 50.0
  jump_clear_min_dx: 16
  survival_bonus: 0.0
  checkpoint_scale: 1.0        # dense per-level ladders ON (June wall-break)
  rnd_intrinsic_coef: 0.1

cold_eval:                     # THE primary metric + winner selector
  every: 25
  curve_episodes: 8            # cheap curve
  gate_episodes: 24            # full at each GO gate
  greedy: true                 # argmax; stochastic also reported
  winner: best_cold.pt         # save_winner re-keyed on cold_seq_clear_rate
  forgetting: {probes: 2, retention_bump: 0.40, bump_iters: 100}

consolidate:                   # gated, reversible, never a clock
  trigger: cold_probe_reached_1_4
  entropy: {from: 0.01, to: 0.002, iters: 80}
  rnd: {from: 0.1, to: 0.0}
  abort_if: {early_forgetting: true, no_gain_iters: 40}
  fallback: cyclic             # Angle B oscillation if terminal aborts 2×

go_explore_fallback:           # DEFERRED subroutine, not the go_explore_on branch
  stall_patience: 60
  burst_iters: 30
  cell: smb_sequential         # world/area IN key (Angle B fix)

warp: {mode: block, report_beyond: true}   # off-ladder + admission + won-metric
```

`smb_oneshot_tiles.yaml`: identical, `encoder: smb_tiles`, `num_envs: 60`. No other diff.

---

## 5. Milestone / GO-NO-GO table (pixel, 24 envs, ~30 s/iter)

| Phase | Sub-stages | Cold-probe GO gate | Iter budget (wall-clock) | NO-GO tripwire (whichever first) |
|---|---|---|---|---|
| **A. 1-1** | 0–2 | greedy clears 1-1 in ≥50% of probes | 0–200 (~1.7 h) | iter 250: cold 1-1 clear < 30% · OR no new cold area in 150 iters |
| **B. 1-2 low road** | 3–6 | cold `furthest_seq ≥ 1-3` (past warp) | 200–600 (~3.3 h) | iter 700: cold never reaches 1-3 · OR `cold_warp_rate` rising + seq flat |
| **C. 1-3** | 7–9 | cold `furthest_seq ≥ 1-4` (enters castle) | 600–1000 (~3.3 h) | iter 1100: cold never reaches 1-4 |
| **D. 1-4** | 10–17 | cold reaches x≥2300 bridge; ≥1 stochastic WIN (`won`) | 1000–1700 (~5.8 h) | iter 1900: no stochastic WIN AND GE-burst (Lane 5) failed |
| **E. Consolidate** | freeze | **greedy cold `seq_clear ≥ 1/24` = DoD** (stretch ≥ 6/24) | 1700–1850 (~1.3 h) | iter 1950: greedy cold `seq_clear` stays 0 |

**Total ≈ 1,850 iters ≈ 1.5–2 days pixel.** Tile scout front-runs A–C in ~½ day to
de-risk the ladder before the pixel run reaches those phases. **Global halt rule:** any
phase exceeding 2× its budget without cold advance → HALT and inspect (warp diversion /
mis-bucketed seed / reward mis-fire) rather than burn overnight compute — the exact
discipline the 7/24-scored-0 episode demands.

Precedent anchors: 1-1 in 100–300 iters (literature); 1-2 solved with the dense ladder
(June); 1-4 stochastic 7/24 with Go-Explore + tuned hyperparams (July).

---

## 6. FIRST IMPLEMENTATION LANE — cold-handable spec (Lane 1)

**Goal.** Make the DoD *measurable*. Today `eval_game.py` breaks the episode on
`episode_success()` = `cleared_any`, a latch set on the **first 1-1 flagpole**
(rewards.rs:1480-1513; eval:233-238) — so it reports a 1-1 clear as "clear" and can
never observe sequential 1-4. This lane adds a sequential predicate and a `--sequential`
mode. **Zero trainer edits; fully unit-testable.**

**Deliverable 1 — `src/training/smb_sequential.py` (~70 LOC).**
- `class SequentialTracker`:
  - RAM addrs: `WORLD=0x075F`, `AREA=0x0760`, `DISPLAY=0x075C`, `X_HI=0x006D`,
    `X_LO=0x0086`. Allowlist of world-1 sequential area bytes: `{0,1,2,3,4}`
    (1-1 main / 1-2 pipe-intro+underground / 1-3 / 1-4 — verify exact bytes against
    `checkpoints_for()` in rewards.rs before finalizing; keep as a config constant).
  - `.update(ram)`: track `prev_world`, `prev_display`. Set `seq_clear=True` on
    `world > prev_world AND prev_display == 3` (the shipped F52 castle guard — a real
    1→2 castle clear). Set `warp_taken=True` on `world > prev_world AND prev_display != 3`
    (warp). Track `furthest_seq` = deepest `(world,level)` on the world-1 chain;
    `furthest_any` = deepest incl. warps.
  - Properties: `seq_clear: bool`, `furthest_seq: (world,level)`, `furthest_any`,
    `warp_taken: bool`.
- `def smb_sequential_cell(ram, x_bucket=8) -> tuple|None`: returns
  `(world, area, x//x_bucket)` with world/area **unbucketed**; returns `None` if
  `world != 1` OR `area not in allowlist` (warp/off-chain admission guard). Shared later
  by the ladder `region_of` and the GE-burst cell key.

**Deliverable 2 — `scripts/eval_game.py --sequential` (~40 LOC edit).**
- New flag `--sequential` (default off; preserves current behavior for other games).
- When set, in the per-episode loop (currently 214-238): instantiate a
  `SequentialTracker`; call `.update(ram)` each step; **do NOT break on
  `episode_success()`**; break only on real Mario death (`rew_done` / pool done) or
  `tracker.seq_clear` (world-1 castle beaten) or `max_steps`.
- Result JSON gains: `seq_clear_rate` (episodes with `tracker.seq_clear` / N),
  `furthest_seq_level`, `furthest_any_level`, `warp_rate`. Keep existing keys for
  back-compat. Append to `eval.jsonl` as today.

**Acceptance tests (`tests/` — TDD, write first):**
1. Synthetic RAM sequence 1-1→(flag)→1-2→1-3→1-4→(world byte 1→2 with prev
   `$075C==3`) ⇒ `seq_clear True`, `furthest_seq=(1,4)`, `warp_taken False`.
2. Synthetic 1-1→1-2→(world byte 1→2 with prev `$075C==1`, i.e. warp) ⇒
   `seq_clear False`, `warp_taken True`, `furthest_any` world≥2, `furthest_seq=(1,2)`.
3. `smb_sequential_cell`: world-2 RAM ⇒ `None`; 1-3 x=800 ⇒ `(1,3,100)`;
   two different levels at same x_page ⇒ distinct cells (the Angle-B collision the
   old `smb_1_4_go_explore.yaml` cell had).
4. `eval_game.py --sequential` on an existing 1-1-clearing checkpoint ⇒ episode does
   NOT stop at the 1-1 flag (length exceeds the 1-1 flagpole step); `seq_clear_rate`
   reflects genuine world-1 completions only.

**Definition of done for Lane 1:** `python scripts/eval_game.py --game mario
--sequential --episodes 24 --checkpoint <best current>` runs greedy from the 1-1 start,
plays *through* the 1-1 flag, and reports `seq_clear_rate`/`furthest_seq`/`warp_rate`.
This is the number every later phase gates on and the number the DoD is scored by.

---

## 7. Single biggest risk + mitigation

**Risk: the contended `_run_vanilla_ppo` edit (Lane 4) regresses the proven curriculum
machinery** — five behaviors (ladder advance, retention split, probe, consolidation,
warp guard) land in one ~120-line edit of a block that already carries the shipped
1-2/stage-0→2 curriculum. A subtle break there silently degrades a *proven* pipeline.
**Mitigation:** (1) All logic lives in the **isolated, unit-tested Lanes 1–3 modules**;
Lane 4 is thin glue calling them. (2) **Single-owner lane discipline** — no other work
edits that block concurrently. (3) **Defer Lane 5 (GE-burst) out of the first edit** —
it's built only if Phase D actually stalls, keeping the initial edit minimal. (4) The
**tile scout runs the identical Lane-4 code ~4× faster**, so a curriculum-mechanics
regression surfaces in hours on tile before the pixel run commits 1.5 days. (5) `make
parity` must stay green after the edit (Rust untouched, but the smoke test asserts it).

---

## Status — 2026-07-16

- **1-2 link welded.** Self-imitation BC on the 1-2 specialist's own successful
  trajectories produced `checkpoints/mario_1_2_underground_consolidate/robust_1_2_handoff.pt`;
  chain-verified (clears 1-2 cold from the 1-1 exit handoff state, hands off a valid
  1-3 entry state).
- **Die-respawn eval artifact discovered.** Cold evals that started a specialist
  mid-level could die, respawn at the level start, and still count the episode's
  eventual clear — inflating clear rates. Old specialist clear-rate claims measured
  under that harness are invalid; all weld gates now use single-life cold evals.
- **Seed binding hardened.** Curriculum seed states are now content-verified
  (area/level bytes checked against the state's claimed level) and viability-filtered
  (dead/glitched snapshots rejected) before entering the pool.
- **1-3 / 1-4 welds running.** `consolidate_level` weld runs are live for 1-3
  (`configs/smb_weld_1_3.yaml`, entry `checkpoints/handoffs/handoff_1-3.state`) and
  1-4 (`configs/smb_weld_1_4.yaml`, entry `stage_1_4.state`).
- **Artifacts promoted into the repo:** composite World-1 manifest
  (`configs/composite_world1.yaml`), weld profiles (`configs/smb_weld_1_3.yaml`,
  `configs/smb_weld_1_4.yaml`), and chain handoff states
  (`checkpoints/handoffs/handoff_1-2.state`, `handoff_1_2.state` back-compat copy,
  `handoff_1-3.state`).

## 2026-07-16 — WORLD 1 ONE-SHOT: ACHIEVED

`eval_composite --manifest configs/composite_world1.yaml` reports
**seq_clear_rate 1.0** (seeds 0 and 1, single life, warp-guarded): cold boot
→ 1-1 (536 steps) → 1-2 (767) → 1-3 (452) → 1-4 castle clear (534), 2,289
steps total. Method: level-keyed composite of four specialists, each link
welded to the exact handoff frame captured from the live chain
(`--capture-handoffs`) via self-imitation — collect the policy's own
stochastic clears from the fixed entry (hybrid rollouts: stochastic bridge,
argmax finish), behavior-clone into the same net (solo-demo fallback when
multi-demo labels conflict), greedy-verify inline. 1-4 required backward
induction across 16 rungs (harvested survival-verified seeds gx108-1800 +
curriculum states). Winning artifacts archived in
`runs/world1_oneshot_20260716/`.
