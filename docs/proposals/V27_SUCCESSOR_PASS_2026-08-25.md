# v27 successor — PASS branch: generalizing fresh-recovery-curriculum to 1-2, 1-3, 1-4

Written 2026-08-25, Lane F of `docs/proposals/WEDNESDAY_PUSH_DAY2_2026-08-25.md`:
*"If v27 PASSES, the fresh-recovery-curriculum shape becomes the standard
recipe for every solver-taught level. Design (don't run) the recipe
generalized... register the honest gate per level."* Lane F says
**prepare regardless** of the verdict, so this document is written now.

**Status check, stated plainly.** As of this writing v27's 4x250 training
is complete (`runs/v27_fresh_recovery/train_seed{0..3}.log`) but its
honest-gate eval did not produce scoreable output — every file under
`runs/v27_fresh_recovery/gate/*.json` (16 files, all 135 bytes) contains
only a `nes_core::Pool` start-state log line, no episode data. No
clear_rate exists yet for v27 on either checkpoint/seed/eval-seed
combination. `docs/proposals/WEDNESDAY_PUSH_DAY2_2026-08-25.md` itself
says the gate is "complete and awaiting honest-gate scoring." **This
document is therefore prepared ahead of a verdict that does not exist
yet**, exactly as Lane F asks. Nothing below should be read as "v27
passed" — it is the design that activates *if* it does. Whoever reads
the eventual v27 gate result should re-open `runs/v27_fresh_recovery/gate/`
and either fix and re-run the eval capture or treat the training-only
receipts as insufficient for a verdict before spending anything below.

Everything here is design + precisely-scoped config diffs against real,
already-read files. Nothing was run for this document beyond read-only
inspection (`peek_u16_consistent` RAM reads on existing savestates,
`load_index`/`gx_report` on existing ladders, `git`/`grep`/`python -c`
introspection) — no training, no solver bursts.

---

## 0. What generalizes unchanged (recap only — full spec in the parent doc)

`docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md` (+ Amendment 1 +
Addendum 2) is the mechanism source. Three pieces travel to every level
as-is, because they are properties of the architecture and the merge
procedure, not of 1-1's geometry:

1. **ReDo** (`src/training/redo.py`, wired into `src/training/trainer.py`
   at `redo_enabled`/`redo_tau`/etc.). Confirmed by reading the
   implementation for this document (not assumed): `maybe_check_and_recycle`
   reads `net.fc1.weight.shape`, `net.fc2.weight.shape` and reports
   `hidden_dim`/`trunk_dim` dynamically — it is genuinely width-agnostic,
   and `trainer.py:4902-4916` gates it on `_is_tile_mode and not
   self._recurrent` with `hasattr(net, "fc1"/"norm1"/"fc2"/"norm2"/"actor"/
   "critic")`, printing `[redo] configured but UNSUPPORTED here` and
   forcing `redo_on=False` (a VOID-triggering line, not a silent no-op) on
   any net that doesn't match. Every backward-curriculum config below uses
   the non-recurrent feedforward `TilePolicyNetwork`, so ReDo applies
   unmodified. Only the **numeric τ** needs a fresh per-level sweep (V7,
   §5) — the mechanism, scope (fc1+fc2, heads excluded), dormancy
   statistic (layer-mean, not layer-max), reset rule (Kaiming incoming,
   zero outgoing, LN affine reset, zeroed Adam moment slices) and cadence
   (every gradient-taking iteration) are architecture properties.
2. **The merge procedure's shape** (gx-nearest insertion into one ladder,
   ±2-rung shift on a dip, sentinel `frame: 900000+i`, one `states_dir`
   / one `TauScheduler` / one draw site — no second restart pool). The
   *numbers* the procedure runs on (ladder size, gx span, kept-fraction
   floor) do not generalize and are computed per level below.
3. **The honest-gate protocol shape**: cold entrance from the level's own
   registered start state, greedy, sticky 0.25, jitter ±16, 50 episodes
   x 2 eval seeds, per-seed artifact selection (peak trailing-entrance
   checkpoint + final checkpoint), best-of-4-seeds pooled. The *numeric
   thresholds* are level-specific (§§2-4).

Everything else — assay sample size, mining budget, iteration budget,
gx tolerances, the PASS/FAIL bar itself — is 1-1-specific in the parent
doc and is re-derived per level below, per the task's own instruction
not to copy 1-1's numbers.

---

## 1. Cross-cutting fixes required before ANY of 1-2/1-3/1-4 can launch

These are shared prerequisites, done once, not per level.

### 1.1 `scripts/merge_recovery_ladder.py` hardcodes 1-1's outcome, not its procedure

Read in full for this document. Three literals block reuse:

```python
GX_SPAN = (40, 3266)     # tape gx_first .. tape gx_last (self-check b)
VOID_MIN_KEPT = 24       # V1: < 24 kept recovery states = treatment too thin
...
    if n_ladder != 758 or n_recovery != 27:
        sys.exit(f"ABORT (V1 count): expected 758 ladder + 27 recovery, "
                 f"got {n_ladder} + {n_recovery}")
```

`n_ladder`/`n_recovery` are already computed dynamically from
`load_index()` two lines above the abort — the `!= 758 or != 27` check is
a redundant assertion of *1-1's own outcome*, not a structural
requirement. The needed patch, in order:

1. **Delete the `!= 758 or != 27` abort.** Nothing downstream needs it —
   the written-artifact self-checks (a)-(e) already re-validate structure
   generically against whatever `n_ladder`/`kept` turn out to be.
2. **Replace `VOID_MIN_KEPT = 24` with a fraction**, not an absolute:
   `VOID_MIN_KEEP_FRAC = 24 / 27` (0.8889 — 1-1's own measured ratio),
   applied as `void_min_kept = math.ceil(VOID_MIN_KEEP_FRAC * n_recovery)`,
   exposed as an overridable `--min-keep-frac` flag. Log the resolved
   number per level (e.g. "1-2: 33 recovery candidates -> VOID floor 30").
3. **Replace the hardcoded `GX_SPAN` tuple** with a per-run derived span:
   `(ladder_entries[0].gx - MARGIN, ladder_entries[-1].gx + MARGIN)` with
   `MARGIN` a small constant (1-1's own recovery states range up to gx
   3266 against a ladder ending at gx 3140 — a 126px overshoot near the
   flag; round up to `MARGIN = 200` as a documented, not re-derived,
   choice). This is a sanity bound against a measurement bug, not an
   exact-match requirement, so a generous margin is correct.
4. **Add area-aware nearest-rung matching** (new — needed for any
   multi-segment ladder, §2.3 below). The current `nearest = min(range
   (n_ladder), key=lambda j: (abs(ladder_entries[j].gx - gx_r), j))`
   searches the *entire* ladder by raw gx distance, ignoring `.area`.
   That is silently wrong whenever a level's gx coordinate re-bases more
   than once (1-2 does, confirmed below) — a recovery state in segment 2
   can have the same raw gx as an unrelated rung in segment 1. Patch:
   restrict the candidate set to `ladder_entries[j].area ==
   m["area"]` before ranking by gx distance; if no same-area candidate
   exists within tolerance, drop the state and record why (same
   accounting the script already does for a gx-dip drop).

None of this changes 1-1's own already-written `1-1-v27` artifact —
the patch only affects future invocations.

### 1.2 `scripts/mine_recovery_tapes.py` — no hardcoded level, but a silent order-bias at scale

`select_candidates` iterates manifest records in episode order and caps
with `candidates[:max_states]` (default 36). For 1-1, 9 true-death
episodes x 4 per-episode = exactly 36 candidates, so the cap never
truncated anything. **1-2 has 33 true-death episodes** (measured from
`runs/recovery_assay_1_2/manifest.json`: 60 episodes, 25 cleared, 35
non-clear-with-sticks, 2 of those timeouts at length >= 1500, 33 true
deaths) — 33 x 4 = 132 raw candidates, and the 36-item default would
silently keep only the *first* 9 episodes' worth (manifest order, not a
random or representative sample) and drop the rest without comment. No
code change is required — `--max-states` is already a CLI flag — but the
**default must not be used for 1-2**; pass `--max-states 132` explicitly
(§2.2). 1-3/1-4 need their own count once their assays run (§3, §4).

### 1.3 Lineage integrity — measured, not assumed, and it fails for 1-2 today

Every backward-curriculum artifact for a level (ladder root state,
recovery-assay start state, honest-eval cold-entrance state) must be one
lineage — this is not new doctrine, it is already stated in both
`mario_1_1_backward.yaml`'s and `mario_1_2_backward.yaml`'s own headers
("a mismatched pair silently trains on a two-lineage mixture"). Checked
for all three target levels by reading `index.json` `root_state` next to
each config's own `start_state_path`:

| level | ladder root_state | banked-control lineage (assay ran here) | match? |
|---|---|---|---|
| 1-2 | `runs/live_show/smb_4_4_micro/entrance_after_1-1.state` (`checkpoints/backward_states/1-2`) | `checkpoints/.../smb_curriculum/stage_03.state` (`mario_1_2_consol2.yaml`, and `runs/recovery_assay_1_2` ran from here) | **NO** |
| 1-3 | `checkpoints/.../smb_curriculum/stage_05.state` (`checkpoints/backward_states/1-3_online`) | same file, `mario_1_3_online_v1.yaml` | yes (same path, byte-identical by construction) |
| 1-4 | `checkpoints/.../smb_curriculum/stage_10.state` (`checkpoints/backward_states/1-4_online`) | same file, `mario_1_4_online_v1.yaml` | yes |

1-2's mismatch is not a hunch — `mario_1_2_backward.yaml`'s own header
already flags it qualitatively ("the 0/400 CGSA-era honest baseline was
measured on a DIFFERENT 1-2 entrance and is not directly comparable").
For this document the two states were RAM-diffed directly
(`nes_core.Pool.peek_u16_consistent` over addresses `0x0000-0x07FF`,
both states loaded via `load_worker_state`): **208 of 2048 zero-page
bytes differ**, including SMB's enemy-slot table (addresses `0x45-0x76`,
all zero in `entrance_after_1-1.state`, populated in `stage_03.state`).
These are not the same machine state saved twice — replaying 1-2's
existing solver tape (rooted at `entrance_after_1-1.state`) against
`stage_03.state` is not guaranteed to reproduce the same trajectory
(SMB's enemy spawn/behavior timing depends on exactly this RAM region),
so the existing `checkpoints/backward_states/1-2` ladder cannot be
reused in-place for a recovery experiment whose recovery states and
honest-eval entrance both live in the `stage_03` lineage. **Stage -1 of
the 1-2 design (§2.1) re-mints the ladder from `stage_03.state` before
anything else.** 1-3 and 1-4 need no such step — their online-campaign
config, their minted ladder, and their assay all already point at the
identical file path.

### 1.4 A caveat that does not block launch, but must ride every verdict

1-1's v27 compared a fresh run of **the same recipe family** (bare
four-term reward, gx-ladder restarts, default 64/32 `TilePolicyNetwork`)
against a banked control from **that same family** (0.767, itself a
plain backward-curriculum artifact). For 1-2/1-3/1-4, the currently
banked numbers (38.0% / 21.0% / 51.0%) all come from the
**online-campaign / consol2 family** — wavefront PBRS + KL-anchored warm
start + self-imitation + entrance-pinned restarts, at a *larger* net
(`tile_hidden_dim: 256, tile_trunk_dim: 64`, ~200k params) — not from
the plain backward-curriculum family this design generalizes (which
uses the CGSA-borrowed `tile_hidden_dim: 128, tile_trunk_dim: 32`,
~96.1k params: `fc1` 712->128 = 91,392 incl. bias, `norm1` 256, `fc2`
128->32 = 4,128 incl. bias, `norm2` 64, `actor` 198, `critic` 33).
Continuing from the consolidated online/consol2 checkpoints was
considered and rejected: that is exactly the "post-hoc gradient onto a
consolidated artifact" family that FAILED three ways on 1-1
(`RECOVERY_DISTILL_1_1_2026-08-24.md`) and is why v27 tests a **fresh**
run at all. So the generalized design below is, correctly, also fresh —
but that means a PASS or FAIL against 38%/21%/51% conflates two
variables (recipe family, and net capacity) that 1-1's comparison did
not. **Register this now, not after seeing a number**: a FAIL here does
not by itself indict recovery-in-curriculum — it could mean the smaller
from-scratch net is intrinsically weaker on this level than the
consolidated-and-larger incumbent, independent of recovery states or
ReDo. A clean single-variable follow-up (fresh run at matched 256/64
capacity) is out of scope for this design and is the natural next
registration if a verdict here comes back ambiguous.

---

## 2. Level 1-2

### 2.1 Current state

- Banked control: `checkpoints/_preserved/consol2_40pct_strict_iter01120.pt`,
  **38.0%** (38/100 strict flagpole predicate: seed 7 24/50, seed 101
  14/50; median max-gx 2095; 40/100 past the bottleneck), lineage
  `stage_03.state`, net 256/64.
- Existing plain-backward ladder `checkpoints/backward_states/1-2`:
  1094 entries (of 1215 minted, `n_actions=1215`), **3 segments**
  (`areas: [1, 2]`, `segment_starts: [0, 116, 972]` — 1-2 re-bases gx
  once at the pipe into the underground at step ~116 and once more
  inside area 2 at step ~972, per `gx_report`'s own comment: "1-2 step
  971: area 2 throughout, gx 2656 -> 0"), gx range 0..3175/3266, tape
  `runs/live_show/smb_4_4_micro/lvl_1-2/solutions/sol_000.actions.npy`.
  **Wrong lineage for this experiment** (§1.3) — do not merge into this
  ladder as-is.
- `configs/mario_1_2_backward.yaml` (the plain backward-curriculum
  recipe, read in full): `tile_hidden_dim: 128, tile_trunk_dim: 32`,
  `rollout_steps: 1536`, `entropy_guard: {floor: 0.08, boost: 3.0}`,
  `backward_curriculum: {states_dir: checkpoints/backward_states/1-2,
  window_frames: 160, advance_threshold: 0.2, advance_actions: 40,
  min_attempts: 30, rung_step_budget: {base: 600, per_entry: 2.0}}`,
  `start_state_path: runs/live_show/.../entrance_after_1-1.state`. Its
  own registered gate (GATE B5, header comment) has never been passed:
  the most advanced logged run (`runs/mario_1_2_backward_seed0_run3.log`,
  268 iterations) shows tau **parked at rung 893/1093 (gx 2674) from at
  least iter 221 through iter 268** with trailing rate 0/0 and 16,034
  truncated attempts accumulated — a hard stall, not slow progress. gx
  2674 is the documented "1-2 x~2674 off-manifold-drift barrier"
  (`configs/mario_1_4_online_v1.yaml`'s own header names it this
  explicitly, and 40/100 of the banked control's honest episodes cross
  it, 60/100 do not). **This is the same lineage bug as §1.3** — this
  run trained against `entrance_after_1-1.state`, a different state than
  the one the banked 38.0% control and the recovery assay use.

### 2.2 Assay (DONE — but partial)

`runs/recovery_assay_1_2/{manifest.json,verdict.json}` (real numbers,
read directly, not summarized secondhand): 60 episodes of the banked
consol2 control from `stage_03.state`, sticky 0.25, greedy, collection
reproduced clear_rate 25/60 (0.417, consistent with the banked 0.380 at
n=100). Non-clear: 35 episodes with sticks (33 true-death, 2 timeout at
the 1500-step cap). **Only 16 of the 35 were adjudicated** (10 solver-min
x 8 workers each) — `3/16 recovered (18.8%)`; of the 11/16 sticks landing
<=4 steps pre-death, only 1 recovered — the fatal window dominates.
Implied honest ceiling **~0.53** (`docs/research/RECOVERY_ASSAY_VERDICT_2026-08-24.md`,
Addendum). **19 of the 35 true-death episodes were never adjudicated.**
Recommended before finalizing the PASS bar (not strictly mandatory, cost
is small): adjudicate the remaining 19 —

```
.venv/bin/python scripts/recovery_assay.py adjudicate \
  --dir runs/recovery_assay_1_2 --sample 35 --minutes 10 --workers 8
```

(re-running `adjudicate` with `--sample 35` against the existing manifest
re-scores everything, including the 16 already done — cheap relative to
mining, ~35x10/8 ~ 44 min). Until this runs, treat the 0.53 ceiling and
the PASS bar derived from it (below) as provisional on a 46%-sampled
population — exactly the "spot-check a null before trusting the
aggregate ratio" lesson the assay-integrity history already teaches.

### 2.3 Stage -1 (1-2 only) — re-root the ladder before mining or merging

Required by §1.3's measured 208/2048-byte RAM mismatch:

1. Solve 1-2 fresh from the `stage_03.state` root (a new Go-Explore
   solve — real solver time, not free; size against whatever timing
   `runs/ge_1_2_solve/` or equivalent banked receipts show for the
   original solve, else budget a fresh multi-hour run):
   ```
   .venv/bin/python scripts/go_explore_solve.py \
     --profile configs/mario_1_2_consol2.yaml \
     --root-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state \
     --out runs/ge_1_2_solve_stage03 --workers 8 --want-solutions 1
   ```
2. Mint a new ladder from that tape:
   ```
   .venv/bin/python scripts/mint_backward_states.py --level 1-2 \
     --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state \
     --actions runs/ge_1_2_solve_stage03/solutions/sol_000.actions.npy \
     --out checkpoints/backward_states/1-2-stage03 \
     --profile configs/mario_1_2_consol2.yaml
   ```
3. Every step below (mining's `--profile`, the merge's `--ladder`, the
   treatment config's `states_dir`) uses `checkpoints/backward_states/1-2-stage03`,
   never the old `checkpoints/backward_states/1-2`. Also write
   `configs/mario_1_2_backward_stage03.yaml`: a byte-for-byte clone of
   `configs/mario_1_2_backward.yaml` with only `start_state_path` ->
   `stage_03.state` and `backward_curriculum.states_dir` ->
   `checkpoints/backward_states/1-2-stage03` changed (plus a corrected
   description — the original's description text is fine here, it
   already describes the mechanism, not the entrance). This is the
   level's own "banked recipe" the treatment configs diff against,
   analogous to how v27 diffed against `mario_1_1_backward.yaml` itself.

### 2.4 Mining budget

33 true-death episodes (all from the manifest, not just the 16 the
assay adjudicated — mining draws independently from the full manifest)
x `--per-episode 4` = 132 raw candidates. Use the real count, not the
36 default (§1.2):

```
.venv/bin/python scripts/mine_recovery_tapes.py \
  --assay-dir runs/recovery_assay_1_2 \
  --out runs/recovery_distill_1_2/fuel \
  --profile configs/mario_1_2_backward_stage03.yaml \
  --per-episode 4 --minutes 10 --workers 8 \
  --max-states 132 --timeout-len 1500
```

Solver wall-time estimate: 132 x 10 min / 8 workers ~ 165 min (~2.75 hr),
versus 1-1's 36 x 10 / 8 ~ 45 min. VOID threshold is the script's own,
unmodified: **< 15 verified tapes kept** (`mine_recovery_tapes.py`'s
registered floor, level-independent).

### 2.5 Merge

After the §1.1 patch (count-literal removed, fraction-based
`VOID_MIN_KEPT`, area-aware nearest-rung matching — **mandatory here**,
1-2's ladder is 3 segments):

```
.venv/bin/python scripts/merge_recovery_ladder.py \
  --ladder checkpoints/backward_states/1-2-stage03 \
  --recovery checkpoints/backward_states/1-2-recovery \
  --out checkpoints/backward_states/1-2-v27succ \
  --profile configs/mario_1_2_v27succ_seed0.yaml \
  --tapes runs/recovery_distill_1_2/fuel/tapes \
  --min-keep-frac 0.8889
```

VOID floor: `ceil(0.8889 x kept-candidate-count)` — resolved once
mining reports how many of the 132 candidates actually verified.

### 2.6 Config diff

Base (post Stage -1): `configs/mario_1_2_backward_stage03.yaml`.
Treatment: `configs/mario_1_2_v27succ_seed{0,1,2,3}.yaml`, diffing
exactly as v27 diffed 1-1's:

```diff
-name: Mario 1-2 backward
+name: Mario 1-2 v27-successor recovery seed{N}
 backward_curriculum:
-    states_dir: "checkpoints/backward_states/1-2-stage03"
+    states_dir: "checkpoints/backward_states/1-2-v27succ"
+reinforce:
+  redo_enabled: true
+  redo_tau: 0.025            # PENDING level-specific V7 tau-sweep, §5
+  redo_check_every_iters: 1
+  redo_sample_batch: 4096
+  redo_reset_optimizer_moments: true
```

Everything else — `tile_hidden_dim: 128`, `tile_trunk_dim: 32`,
`rollout_steps: 1536`, `entropy_coef: 0.01` + `entropy_guard` block,
`sticky_action_prob: 0.25` + `sticky_episode_boundary_reset: true`,
`rung_step_budget: {base: 600, per_entry: 2.0}`, `window_frames: 160`,
`advance_threshold: 0.2`, `advance_actions: 40`, `min_attempts: 30`,
`num_envs: 60`, `wavefront_reward.enabled: false`, `go_explore.enabled:
false` — carried verbatim, exactly the "banked recipe, config-only"
principle. No `kl_anchor_checkpoint` (fresh run, nothing to leash to).

### 2.7 Iteration budget (scaled to 1-2's own ladder, not 1-1's 250)

Formula: `iters = round(250 x rungs_level / 785)` using 1-1's own merged
785-rung ladder as the reference rate (0.3185 iters/rung) — 1-1's actual
spent budget, not an invented constant. 1-2's pre-merge ladder is 1094
rungs (merge adds a small recovery uplift, ~3.6% for 1-1's 27/758; not
material to the order of magnitude):

**iters/seed = round(250 x 1094 / 758) = 360.**

(Using 785 in the denominator instead of 758 changes this by <2%; 360
either way after rounding.) At the measured ~24-25 s/iter for this
recipe shape (`rollout_steps=1536, num_envs=60` — read directly off
`runs/mario_1_2_backward_seed0_run3.log`'s own throughput lines), 360
iters ~ 150 min/seed, ~10.0 hr for 4 seeds run sequentially on one
machine — larger than 1-1's ~7 hr, appropriately, since 1-2's tape is
~35% longer.

### 2.8 Seeds

0, 1, 2, 3 — carried convention, matching the banked best-of-4 selection
this gate compares against.

### 2.9 Pre-registered honest gate (against 1-2's OWN banked control)

```
.venv/bin/python scripts/eval_game.py --game mario \
  --profile configs/mario_1_2_v27succ_seedN.yaml --checkpoint <ckpt> \
  --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state \
  --episodes 50 --max-steps 2400 --sticky-prob 0.25 --start-jitter 16 \
  --eval-seed {0,1} --action-select {greedy,sampled}
```

(`--max-steps 2400` matches this profile's own `max_episode_steps`, not
1-1's 1500 — 1-2's tape is 1215 actions, and 1500 would leave only 285
steps of margin versus 1-1's 597; `--action-select sampled`, not
`sample` — `eval_game.py:97` defines `_ACTION_SELECT_MODES = ("greedy",
"sampled")` exactly, and `sample` is not a valid choice, a small
correction to the wording v27's own doc used.)

Artifact selection: per seed, peak trailing-entrance-rate checkpoint +
final checkpoint; best pooled greedy across 4 seeds is the number.

- **PASS**: best-of-4 pooled >= **0.50** (ceiling ~0.53 minus a 0.03
  margin, the same margin 1-1's 0.83-lower-bound -> 0.80 bar used), with
  `warp_rate == 0.0` (1-2 has a real warp-zone pipe to World 4 in its
  underground section — this check is load-bearing here, not cosmetic).
- **FAIL**: best-of-4 pooled <= **0.38** (the banked control, exact
  figure, not rounded).
- **MARGINAL**: (0.38, 0.50).
- Caveat carried from §2.2: this bar is provisional pending the 19
  un-adjudicated assay episodes; if a full-population adjudication moves
  the ceiling materially, re-register the PASS number before scoring,
  don't rescale after seeing a result.

---

## 3. Level 1-3

### 3.1 Current state

- Banked control: `checkpoints/_preserved/one_three_FINAL_consol2_iter00690.pt`,
  **21.0%** (21/100: 10/50 + 11/50, chain-advance 26/100, median max-gx
  1800.5 of ~2515), lineage `stage_05.state`, net 256/64 (per
  `mario_1_3_online_v1.yaml`, structurally identical to 1-2's consol2
  net).
- Existing ladder `checkpoints/backward_states/1-3_online`: **540
  entries, single segment** (`areas: [3]`, monotone, `decreases: 3` /
  `max_drop: 2` px — a very clean tape), gx 0..2514, tape
  `runs/ge_1_3_solve/solutions/sol_000.actions.npy` (540 actions),
  `root_state: stage_05.state` — **matches the banked control's own
  lineage exactly** (same file, confirmed by path string, not merely by
  "same stage number"). No §1.3-style Stage -1 needed for 1-3.
- `configs/mario_1_3_backward.yaml` exists (read in full) but has two
  real bugs, both must be fixed before use as this level's base recipe:
  1. `backward_curriculum.states_dir: checkpoints/backward_states/1-3` —
     **this directory does not exist** (`ls checkpoints/backward_states/`
     shows only `1-3_online`). This config would fail at load time.
     Fix: `states_dir: checkpoints/backward_states/1-3_online`.
  2. `description:` is a **verbatim copy-paste of 1-2's description**
     ("Reverse start-state curriculum on SMB **1-2**... 1-2 entrance").
     Cosmetic, but must be corrected for the record — a stale
     description is exactly the kind of drift this project's own
     doctrine (name your lineage, name your entrance) exists to prevent.
  No `runs/mario_1_3_backward*.log` exists anywhere in the repo — **this
  config has apparently never been launched**, likely because of bug
  (1). Unlike 1-1 (which has a real 0.767 backward-only control) and
  1-2 (which stalled visibly at a documented wall), **1-3 has zero prior
  evidence the plain backward-curriculum mechanism even engages its
  ladder.**

### 3.2 Consequence: an extra sanity signpost before spending the full budget

Because there is no 1-3 backward-curriculum precedent at all, recommend
a short, cheap dry run of the FIXED base config (no recovery merge, no
ReDo — the corrected `mario_1_3_backward.yaml` verbatim) for ~150
iterations before committing the full seed x iteration budget to the
treatment. Read the S1/S2/S3-style signposts already registered in
`mario_1_2_backward.yaml`'s header (rung advance by iter ~1/3 of budget;
gx crossed some fraction of the tape by ~2/3; entrance column non-zero
by ~90%), scaled to 1-3's own 180-iter budget (§3.5) rather than 1-2's
1000/600/200 marks. This is not a new gate — it is the minimum diligence
1-1 and 1-2 both got before their numbers were trusted, applied here
because 1-3 skipped it.

### 3.3 Assay (NOT RUN)

```
.venv/bin/python scripts/recovery_assay.py collect --dir runs/recovery_assay_1_3 \
  --checkpoint checkpoints/_preserved/one_three_FINAL_consol2_iter00690.pt \
  --profile configs/mario_1_3_online_v1.yaml \
  --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state \
  --episodes 60 --sticky 0.25 --max-steps 1500 --jitter 16 --seed 0

.venv/bin/python scripts/recovery_assay.py adjudicate \
  --dir runs/recovery_assay_1_3 --sample <N_noclear> --minutes 10 --workers 8
```

`N_noclear` should be the FULL non-clear-with-sticks population from the
collection, not a 16-of-35-style partial sample (§2.2's lesson, applied
before the fact instead of after): at a 21% banked clear rate, expect
roughly 45-49 of 60 episodes non-clear; adjudicating all of them at 10
solver-min x 8 workers costs ~49 x 10 / 8 ~ 61 min — cheap, no reason to
sample a subset here the way 1-1/1-2's assays did.

### 3.4 Mining budget — formula only, pending the assay

`--max-states = 4 x (true-death episode count measured above)`, no
default-36 truncation (§1.2). Cannot be given a number before the assay
runs; do not reuse 1-1's 36 or 1-2's 132 as a placeholder.

```
.venv/bin/python scripts/mine_recovery_tapes.py \
  --assay-dir runs/recovery_assay_1_3 --out runs/recovery_distill_1_3/fuel \
  --profile configs/mario_1_3_backward.yaml \
  --per-episode 4 --minutes 10 --workers 8 \
  --max-states <4 x true_death_count> --timeout-len 1500
```

### 3.5 Merge, config, iteration budget

Single segment (`areas: [3]`) — the §1.1 area-aware fix is harmless but
not load-bearing here (nearest-by-gx alone is already correct on a
single-segment tape).

```
.venv/bin/python scripts/merge_recovery_ladder.py \
  --ladder checkpoints/backward_states/1-3_online \
  --recovery checkpoints/backward_states/1-3-recovery \
  --out checkpoints/backward_states/1-3-v27succ \
  --profile configs/mario_1_3_v27succ_seed0.yaml \
  --tapes runs/recovery_distill_1_3/fuel/tapes \
  --min-keep-frac 0.8889
```

Config diff, base = `configs/mario_1_3_backward.yaml` **after fixing
both bugs in §3.1**:

```diff
-name: Mario 1-3 backward
+name: Mario 1-3 v27-successor recovery seed{N}
-description: 'Reverse start-state curriculum on SMB 1-2 over the banked solver tape, ...'
+description: >
+  Reverse start-state curriculum on SMB 1-3 over the banked solver tape,
+  recovery states from the 1-3 assay merged in from iteration 0, ReDo
+  dormant-neuron recycling. Honest gate: cold greedy sticky-0.25 +
+  jitter-16 from the 1-3 entrance (stage_05.state), 50 episodes, 2 seeds.
 backward_curriculum:
-    states_dir: checkpoints/backward_states/1-3
+    states_dir: checkpoints/backward_states/1-3-v27succ
+reinforce:
+  redo_enabled: true
+  redo_tau: 0.025            # PENDING level-specific V7 tau-sweep, §5
+  redo_check_every_iters: 1
+  redo_sample_batch: 4096
+  redo_reset_optimizer_moments: true
```

Everything else (`tile_hidden_dim: 128, tile_trunk_dim: 32,
rollout_steps: 1536, entropy_guard, rung_step_budget: {base: 600,
per_entry: 2.0}, sticky 0.25 + boundary reset`) carried verbatim — note
`rollout_steps: 1536` here has ~2.8x slack over 1-3's 540-action tape
(vs. 1-2's tighter ~1.26x over its 1215-action tape); this is safe but
was clearly copied from the 1-2 template rather than re-derived for
1-3's shorter tape. Not a blocker, flagged as exactly the kind of
un-rederived number this document's whole exercise is about — a future
pass could safely lower it, this one does not, to keep the diff minimal
and auditable.

**Iteration budget**: `round(250 x 540 / 758) = 178` -> **180
iters/seed**. At ~24-25 s/iter (same rollout_steps/num_envs shape as
1-2): ~75 min/seed, ~5.0 hr for 4 seeds sequential.

**Seeds**: 0, 1, 2, 3.

### 3.6 Pre-registered honest gate

```
.venv/bin/python scripts/eval_game.py --game mario \
  --profile configs/mario_1_3_v27succ_seedN.yaml --checkpoint <ckpt> \
  --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state \
  --episodes 50 --max-steps 2400 --sticky-prob 0.25 --start-jitter 16 \
  --eval-seed {0,1} --action-select {greedy,sampled}
```

- **PASS**: best-of-4 pooled >= `ceiling_lower_bound - 0.03` (formula
  from 1-1's own derivation: PASS sat exactly at the assay's ceiling
  lower bound minus 0.03 — 0.83 - 0.03 = 0.80). **Numeric value filled
  in only after §3.3's assay produces a ceiling estimate** — do not
  borrow 1-1's 0.80 or 1-2's 0.50.
- **FAIL**: best-of-4 pooled <= **0.21** (the banked control, known now).
- **MARGINAL**: between.
- `warp_rate == 0.0` (or the equivalent illegitimate-shortcut check for
  1-3 — verify against the adapter before wiring; 1-3 is not known to
  have a warp zone, but the check should still run as standard hygiene).

---

## 4. Level 1-4

### 4.1 Current state

- Banked control: `checkpoints/_preserved/one_four_MEASURED60_iter00960.pt`,
  **51.0%** (24/50 + 27/50, chain predicate identical, median max-gx
  2428 of 2431 — the median episode reaches the axe), lineage
  `stage_10.state`, net 256/64. Per CLAIMS.md's own text: *"its reverse
  walk exhausted its budget without reaching the entrance — so the rate
  below was earned WITHOUT the backward curriculum completing."* Same
  non-engaged-ladder pattern as 1-2, for a different underlying reason
  (budget exhaustion rather than a hard wall) — the online campaign's
  51% comes from wavefront PBRS + KL-anchor + SIL, not from the ladder
  reaching tau 0.
- **1-4's "clear" is a world increment, not a flagpole touch** — 1-4 is
  a castle level; CLAIMS.md calls this out explicitly as "a different
  branch of the strict predicate... a test that the pipeline crosses
  level TYPES rather than level instances." `checkpoints/backward_states/
  1-4_online`'s own `tail` field confirms this mechanically:
  `{"end_wd": [1, 0], "end_area": 0, "end_gx": 2431}` — `end_wd` moves
  from world 0 to world 1, i.e. the tape's own recorded "clear" is the
  world-index change, not a `warp_rate`/flagpole flag. **Before wiring
  this level's honest gate, verify the exact `eval_game.py`/SMB-adapter
  field name that scores this (a `world_advance`-style flag or
  equivalent) — do not assume the flagpole `warp_rate`/`clear_rate`
  semantics used for 1-1/1-2/1-3 carry over unexamined.** This is
  registered as VOID condition V8 (§5).
- Existing ladder `checkpoints/backward_states/1-4_online`: **490
  entries, single segment** (`areas: [4]`, `decreases: 2`, `max_drop: 3`
  px), gx 45..2431, tape `runs/ge_1_4_solve/solutions/sol_000.actions.npy`
  (490 actions), `root_state: stage_10.state` — matches the banked
  control's lineage exactly, same as 1-3. No Stage -1 needed.
- **No `mario_1_4_backward.yaml` exists** — 1-4 has never had a plain
  backward-curriculum config at all, unlike 1-2/1-3. Constructed below
  by the same substitution pattern already used to derive
  `mario_1_3_backward.yaml` from `mario_1_2_backward.yaml` (swap
  states_dir, start_state_path, description; every hyperparameter
  carried verbatim from the sibling family) — this is not invented YAML,
  it is the existing, already-verified template applied one more time.

### 4.2 New base config: `configs/mario_1_4_backward.yaml`

Full draft (clone of `configs/mario_1_3_backward.yaml`'s already-fixed
form, §3.1, with only the level-identifying fields changed):

```yaml
name: Mario 1-4 backward
rom_path: roms/Super Mario Bros. (World).nes
rom_hashes:
- 811b027eaf99c2def7b933c5208636de
frame_skip: 4
start_state_path: checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_10.state
description: >
  Reverse start-state curriculum on SMB 1-4 (the castle level) over the
  banked solver tape, CGSA's PPO block with the CGSA curriculum,
  Go-Explore restarts and the wavefront shaping removed. Honest gate:
  cold greedy sticky-0.25 + jitter-16 from the 1-4 entrance (stage_10.state),
  50 episodes, 2 seeds. Clear predicate is the WORLD-INCREMENT flag, not
  a flagpole warp_rate check — verify the exact field name before use
  (see V8, docs/proposals/V27_SUCCESSOR_PASS_2026-08-25.md).
ram_mapping:
  x_position_page: 109
  x_position_low: 134
  y_position: 206
  lives: 1882
  score_start: 2014
  score_end: 2019
  world: 1887
  level: 1884
  player_state: 14
  float_state: 29
reward_weights:
  forward_progress: 1.0
  time_penalty: -0.01
  death_penalty: -15.0
  completion_bonus: 50.0
  score_delta: 0.0
  air_bonus: 0.0
  survival_bonus: 0.0
  jump_clear_bonus: 0.0
  checkpoint_scale: 0.0
action_space:
- []
- - right
- - right
  - A
- - right
  - B
- - right
  - A
  - B
- - A
reinforce:
  enabled: true
  trainer_mode: vanilla_ppo
  encoder: smb_tiles_pos
  recurrent: false
  tile_hidden_dim: 128
  tile_trunk_dim: 32
  preserve_elite_diversity: true
  freeze_pre_ppo_elite: true
  episodes_per_genome: 2
  symlog_rewards: true
  value_loss: huber
  value_coef: 0.5
  gae_lambda: 0.95
  gamma: 0.99
  bc_epochs: 0
  warmup_gens_ga_only: 10
  rollout_steps: 1536
  ppo_minibatch_size: 256
  steps: 3
  lr: 0.0003
  ppo_clip_eps: 0.2
  async_pipeline: false
  device: cpu
  panic_isolation: false
  entropy_coef: 0.01
  entropy_floor: 0.02
  entropy_coef_max: 0.05
  rnd_intrinsic_coef: 0.0
  rnd_loss_coef: 0.0
  demo_anchor_enabled: false
  bc_replay_enabled: false
  preprocess_f16: false
  smb_curriculum: false
  num_envs: 60
  sticky_action_prob: 0.25
  sticky_episode_boundary_reset: true
  go_explore:
    enabled: false
  wavefront_reward:
    enabled: false
  backward_curriculum:
    states_dir: checkpoints/backward_states/1-4_online
    tau_init: -1
    window_frames: 160
    advance_threshold: 0.2
    advance_actions: 40
    min_attempts: 30
    entrance_weight: 1.0
    count_truncations: true
    rung_step_budget:
      base: 600
      per_entry: 2.0
    pin_entrance: false
    entropy_guard:
      floor: 0.08
      boost: 3.0
ga_params:
  mutation_rate: 0.8
  mutation_std: 0.001
  adaptive_mutation_scale: false
  elite_fraction: 0.4
  stale_gens_before_restart: 50
  restart_fraction: 0.5
max_episode_steps: 2400
```

This config has never been run — treat any first pilot on it the same
way §3.2 treats 1-3: a cheap dry run before spending the full budget,
since (like 1-3) there is zero prior evidence the plain ladder mechanism
engages 1-4's geometry at all.

### 4.3 Assay (NOT RUN)

```
.venv/bin/python scripts/recovery_assay.py collect --dir runs/recovery_assay_1_4 \
  --checkpoint checkpoints/_preserved/one_four_MEASURED60_iter00960.pt \
  --profile configs/mario_1_4_online_v1.yaml \
  --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_10.state \
  --episodes 60 --sticky 0.25 --max-steps 1500 --jitter 16 --seed 0

.venv/bin/python scripts/recovery_assay.py adjudicate \
  --dir runs/recovery_assay_1_4 --sample <N_noclear> --minutes 10 --workers 8
```

At a 51% banked clear rate expect ~49 non-clear episodes; adjudicate all
of them (~49 x 10 / 8 ~ 61 min), same reasoning as §3.3. **1-4 is a
castle level with different hazard geometry (lava/fire-bar pits, not
platform gaps)** — `recovery_assay.py`'s outcome predicate reads
`0x075A` (lives), `0x075F`/`0x0760` (world/area), `0x00B5` (pit-fall
flag), all engine-wide SMB addresses valid on any level, but the
`0x00B5 >= 2` pit-death check's specific *behavior* (what triggers it,
what false-positives look like) was only empirically validated on 1-1
("false positives at x~314" per the assay script's own commentary
history). Run a small manual sanity check (a handful of known-death
replays on 1-4, confirm the predicate tags them correctly) before
trusting the collection's death/clear split at scale — this is the same
"probe a sample by hand before trusting the aggregate" discipline as the
assay-integrity lessons (§6).

### 4.4 Mining, merge, config, iteration budget

Mining budget: formula only, `--max-states = 4 x true_death_count`,
pending the assay (same as 1-3, §3.4).

Merge: single segment (`areas: [4]`), area-aware fix harmless-but-inert
here, same command shape as 1-3:

```
.venv/bin/python scripts/merge_recovery_ladder.py \
  --ladder checkpoints/backward_states/1-4_online \
  --recovery checkpoints/backward_states/1-4-recovery \
  --out checkpoints/backward_states/1-4-v27succ \
  --profile configs/mario_1_4_v27succ_seed0.yaml \
  --tapes runs/recovery_distill_1_4/fuel/tapes \
  --min-keep-frac 0.8889
```

Config diff, base = `configs/mario_1_4_backward.yaml` (§4.2):

```diff
-name: Mario 1-4 backward
+name: Mario 1-4 v27-successor recovery seed{N}
 backward_curriculum:
-    states_dir: checkpoints/backward_states/1-4_online
+    states_dir: checkpoints/backward_states/1-4-v27succ
+reinforce:
+  redo_enabled: true
+  redo_tau: 0.025            # PENDING level-specific V7 tau-sweep, §5
+  redo_check_every_iters: 1
+  redo_sample_batch: 4096
+  redo_reset_optimizer_moments: true
```

**Iteration budget**: `round(250 x 490 / 758) = 161.6` -> **160
iters/seed**. At ~24-25 s/iter: ~67 min/seed, ~4.4 hr for 4 seeds
sequential — the smallest of the three, matching 1-4's shortest tape.

**Seeds**: 0, 1, 2, 3.

### 4.5 Pre-registered honest gate

```
.venv/bin/python scripts/eval_game.py --game mario \
  --profile configs/mario_1_4_v27succ_seedN.yaml --checkpoint <ckpt> \
  --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_10.state \
  --episodes 50 --max-steps 2400 --sticky-prob 0.25 --start-jitter 16 \
  --eval-seed {0,1} --action-select {greedy,sampled}
```

- **PASS**: best-of-4 pooled >= `ceiling_lower_bound - 0.03`, numeric
  value filled in only after §4.3's assay runs.
- **FAIL**: best-of-4 pooled <= **0.51** (banked control).
- **MARGINAL**: between.
- Clear predicate: the level's own world-increment flag (V8, §5) — NOT
  `warp_rate`, which is a flagpole-family concept that may not even
  apply to a castle level's adapter output. Verify and name the actual
  field before the first eval command is run for real.

---

## 5. Consolidated VOID conditions (registered, v27-style)

A VOID means "the experiment never happened" — no PASS/FAIL/MARGINAL
verdict may be read. Numbered to track v27's own V1-V7 where the same
check generalizes; new checks get new numbers.

- **V0 — assay-first gate (new; formalizes the standing CLAIMS.md
  routing rule "run this assay before spending training effort on any
  level's sticky rate").** No PASS/FAIL numeric bar may be finalized,
  and no seed x iteration training budget may be spent, for a level
  whose recovery assay has not adjudicated its full non-clear-episode
  population (or a documented, deliberate subset with its own
  sampling-error caveat attached). **1-3, 1-4: unmet — assay not yet
  run.** **1-2: partially met — 16/35 adjudicated; the 0.50 PASS bar in
  §2.9 is provisional until the remaining 19 are scored (§2.2).**
- **V1 — merge count/kept gate (generalized).** Abort, or VOID after the
  fact, if the merge script's self-checks (a)-(e) fail, or if kept
  recovery states fall below `ceil(0.8889 x candidate-count)` — the
  level's own resolved number (§1.1), never 1-1's literal 24.
- **V1b — segment/area mismatch (new; mandatory for 1-2's 3-segment
  ladder, inert-but-checked for 1-3/1-4's single-segment ladders).** A
  recovery state whose gx-nearest ladder rung has a *different* area
  byte is not a valid match. The patched merge script must restrict
  candidates to same-area rungs (§1.1 item 4); any recovery state with
  no same-area candidate within tolerance is DROPPED and counted against
  V1's kept-fraction floor, never silently matched across a segment
  boundary.
- **V1c — lineage-identity gate (new; mandatory for 1-2 only, §1.3).**
  The ladder's `root_state` and the level's recovery-assay start-state
  must be RAM-identical (a full zero-page peek comparison, not just
  gx/area) before any merge. **1-2 is known to FAIL this today** (208 of
  2048 bytes differ) and must complete Stage -1 (§2.3: re-solve + re-mint
  from `stage_03.state`) before V1c can pass. 1-3 and 1-4 pass trivially
  (identical file path throughout their whole pipeline).
- **V2 — preflight, frozen-actor class (generalized).** Per-level
  2-iter pilot via `scripts/experiment_preflight.py`; VOID if no
  non-critic parameter moved, if any freeze sentinel outlasts the
  level's full iteration budget, if the log lacks `[backward] ENABLED:
  <n> states ... from checkpoints/backward_states/<level>-v27succ`, or
  if it contains `[backward] configured but INERT` / `[backward]
  disabled`. Extended per Amendment 1: VOID also if the pilot log lacks
  `[redo] ENABLED tau=0.025` or any treatment-run log contains `[redo]
  disabled`.
- **V3 — recovery-pool liveness (generalized).** A throwaway copy of the
  seed-0 profile with `tau_init` pinned into a merged-index window that
  covers >= 1 recovery entry (per that level's own merge manifest): 2
  iters must show >= 1 `[backward] iter ... frame 9000xx` line and zero
  `[backward] env N restart at tau T failed` lines.
- **V4 — wrong-ladder guard, per seed (generalized).** Any real run
  whose `[backward] ENABLED` line names a `states_dir` other than that
  level's own `<level>-v27succ`, or a count other than that level's own
  merge manifest total, is VOID for that seed — this exact class voided
  2-1 attempt 1 already; it is a per-level, not a 1-1-only, risk.
- **V5 — actor/ladder liveness mid-run, per seed (generalized).** A seed
  whose tau has advanced 0 rungs by ~60% of that level's own iteration
  budget (not 1-1's literal iter 150) is VOID-machinery for that seed.
  If all seeds void this way, the experiment is VOID, not FAIL.
  - **V5b (new, 1-2-specific).** If tau re-stalls inside the documented
    gx 2600-2700 off-manifold-drift band (§2.1) for more than 40% of
    the run's iteration budget with trailing rate 0, AND no merged
    recovery rung sits in or past that band with a nonzero trailing
    rate, that seed is VOID — a re-encounter of a known, pre-existing
    non-engagement wall, not new evidence about the recovery treatment.
    If a recovery rung IS in that band and IS being earned (nonzero
    trailing rate), the same stall pattern instead counts as ordinary
    ladder progress and is not VOID.
- **V6 — eval lineage (generalized).** Any gate eval not run with that
  level's own registered start-state (1-2: `stage_03.state` — NOT
  `entrance_after_1-1.state`; 1-3: `stage_05.state`; 1-4: `stage_10.state`)
  and the registered flags is discarded, not reported.
- **V7 — ReDo forced-recycle preflight (generalized; carries Addendum
  2's REVISED criteria, not the original flat-0.98 floor that already
  failed once on 1-1).** Must be **re-run per level**, not assumed from
  1-1's own tau-sweep table — a different level's trained net (even
  same architecture/width) can have a different dormancy distribution.
  A throwaway `redo_tau: 0.5`-pinned pilot must show: (b-i) finite
  (non-NaN/Inf) max_dlogit at every recycle event; (b-ii) agreement
  non-increasing in units-recycled-per-event with no chaotic blow-up;
  (b-iii) at the registered tau=0.025 on an actually-trained net,
  per-event recycle counts are expected single-digit — real-run events
  exceeding 15 simultaneously-recycled units PAUSE the run for manual
  review rather than continuing blind (Addendum 2's own soft-VOID
  trigger, carried forward unchanged since it is a property of the net
  and the mechanism, not of 1-1's geometry).
- **V8 — castle-predicate verification (new, 1-4-specific, §4.1).**
  Before any 1-4 gate eval is scored, the exact `eval_game.py`/adapter
  field that scores a legitimate world-increment clear must be
  identified and named in the eventual verdict writeup. Any 1-4 eval run
  under an unverified or assumed-flagpole predicate is discarded, not
  reported, exactly as V6 discards a wrong-lineage eval.

---

## 6. Sequencing recommendation

Not a gate, a practical order, cheapest-and-most-informative first:

1. **1-3 and 1-4 assays** (§3.3, §4.3) — cheapest (no mining/merge cost
   yet), and per CLAIMS.md's own binding routing rule these should run
   *before* any training effort regardless of this document. Do this
   even if v27's own verdict is still unscored — assay running is
   measurement, not a treatment spend.
2. **1-2's remaining 19 assay adjudications** (§2.2) — cheap (~44 min),
   tightens the 1-2 PASS bar before it is spent against.
3. **1-2 Stage -1** (§2.3) — the one item on this list that is a real,
   multi-hour compute spend (a fresh solve) independent of the v27
   verdict question; the most expensive single prerequisite here.
4. Mining for whichever of 1-2/1-3/1-4 has a completed assay (§2.4, §3.4,
   §4.4).
5. Merge (§2.5, §3.5, §4.4) — seconds, after the §1.1 script patch lands.
6. V2/V3/V7 preflights per level — minutes each, before any seed budget.
7. The 4x{360, 180, 160}-iteration training runs themselves — gated on
   the v27 verdict actually landing PASS (per Lane F's own framing), and
   on every VOID condition above clearing first.

Every step through (6) is measurement, patching, or a solver run that
answers a question this project already commits to answering regardless
of v27's outcome (the assay-first doctrine). Only step (7) — the actual
training spend — should wait on reading v27's real verdict once its
gate-eval capture is fixed and re-run (see the Status check at the top
of this document).

---

## 7. Non-gate mechanism reads (report with any verdict, per level)

Same three reads v27 registers for 1-1, applied per level:

1. Recovery-band rung rates: trailing-window rate while tau's window
   covers >= 1 recovery entry (`frame >= 900000`) vs. adjacent windows
   with none.
2. Assay re-run on the winning artifact: did the trainable-slice
   fraction shrink relative to that level's own pre-training measurement
   (1-2: 18.8%; 1-3/1-4: TBD)?
3. Ladder telemetry vs. that level's own pre-merge ladder: iters-to-
   entrance, rungs/100 iters — does the merged ladder slow the walk or
   stall it, and specifically for 1-2, does it ever cross gx 2674?
4. Dormancy/recycle telemetry (Amendment 1's B5 read #4): per-layer
   dormant fraction, cumulative recycles, and the identity-preservation
   series (agreement, max|delta logit|) — reported per level, since
   §1.1/§1.4's architecture note means 1-2/1-3/1-4's 128/32 net may have
   a materially different dormancy profile than 1-1's 64/32 net even at
   the same tau.

## Receipts layout (per level, mirroring v27's own)

- Stage -1 (1-2 only): `runs/ge_1_2_solve_stage03/`,
  `checkpoints/backward_states/1-2-stage03/index.json`
- Assay: `runs/recovery_assay_1_{2,3,4}/{manifest,verdict}.json`
- Mining: `runs/recovery_distill_1_{2,3,4}/fuel/{mining.json,tapes/}`
- Merge manifest: `checkpoints/backward_states/1-{2,3,4}-v27succ/index.json`
  (`recovery_map`)
- Preflights: `runs/v27succ_1_{2,3,4}/preflight/`
- Training: `checkpoints/mario_1_{2,3,4}_v27succ_seed{0..3}/`,
  `runs/mario_1_{2,3,4}_v27succ_seed{0..3}.log`
- Gate evals: `runs/v27succ_1_{2,3,4}/eval_seed{N}_{ckpt}_{es}.json`
- Verdict: appended to this file, per house style — one section per
  level, not one section for all three, since they gate independently.
