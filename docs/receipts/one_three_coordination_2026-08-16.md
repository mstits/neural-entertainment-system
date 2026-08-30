# 1-3 coordination audit + campaign-package runbook

Audit date: 2026-08-15 (token-bound; the 24h mission soak owns the
emulator — `pgrep -f soak_harness` -> pid 25861 at audit time). Nothing
in this document was produced by running the emulator: every fact below
is a file listing, a JSON/YAML read, or a code read.

---

## 1. What exists on disk for 1-3

### 1.1 Solver artifacts — BANKED, REUSABLE (`runs/ge_1_3_solve/`)

| file | size | mtime | content |
|---|---|---|---|
| `solutions/sol_000..003.{json,actions.npy}` | 4.1-4.4 KB each | 2026-07-21 23:29-23:30 | **4 verified clears** |
| `archive.pkl` | 146 MB | 2026-07-21 | Go-Explore cell archive |
| `traces.pkl` | 1.6 MB | 2026-07-21 | search traces |
| `roots.json` / `progress.jsonl` / `archive.stats.json` | small | 2026-07-21 | provenance + progress |

All four solutions share one root and one start/clear pair:

```
root_state : checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state
start_wd   : [0, 2]   ("1-3")      clear_wd : [0, 3]   ("1-4")
provenance : search   root_id  : entrance
actions    : sol_000 540 | sol_001 530 | sol_002 519 | sol_003 498
```

`runs/ge_1_3_solve.log` tail: `4 solutions`, archive `max_area 3`,
`max_gx_in_max_area 2515`, 319,344 search steps in 128 s. The solve is
a 2-minute problem — **1-3 is already solved**; the campaign is a
*learning* problem, not a search problem.

Reading of `max_area 3`: the area byte never leaves 3 for the whole
solve, so **the 1-3 tape is single-segment** — no odometer re-base
(unlike 1-2, which re-bases twice). That materially simplifies rung
selection (§3.3).

### 1.2 The parallel lane's config — EXISTS (`configs/mario_1_3_backward.yaml`)

Untracked, mtime 2026-08-14 18:08. Byte-identical to
`/Users/stits/Documents/nes-league-integration/configs/mario_1_3_backward.yaml`
(mtime 20:22 — the league copy is the later write of the same bytes).

What it declares:

* `start_state_path: .../smb_curriculum/stage_05.state` — **the same
  root the solver used**, i.e. the true 1-3 entrance. Confirms stage_05
  as the canonical 1-3 entrance state.
* `reinforce.backward_curriculum.states_dir: checkpoints/backward_states/1-3`
* backward machinery only: `go_explore.enabled false`,
  `wavefront_reward.enabled false`, no KL anchor, no SIL, no adversary.
  Net is h128/trunk32 (**not** the h256/trunk64 shape the 1-2 online
  stack uses).
* `tau_init -1`, `window_frames 160`, `advance_threshold 0.2`,
  `advance_actions 40`, `min_attempts 30`, `entrance_weight 1.0`,
  `count_truncations true`, `rung_step_budget {base 600, per_entry 2.0}`,
  `entropy_guard {floor 0.08, boost 3.0}`.
* Its `description` still says *"Reverse start-state curriculum on SMB
  **1-2**"* — a copy-paste residue from the 1-2 backward profile. Not
  our file to fix; noted so nobody quotes it as evidence of a 1-2 run.

### 1.3 The lane's minted states — **DO NOT EXIST YET**

```
$ ls -la checkpoints/backward_states/
1-1/   (906 entries, 2026-08-08)
1-2/   (1218 entries, 2026-08-08)
        <- no 1-3
```

`checkpoints/backward_states/1-3/` is **absent**. The frontier note of
"540 minted states" matches `sol_000`'s 540 actions exactly, i.e. it is
the *planned* mint size (one state per tape step of the longest
solution), not a mint that has landed. The parallel lane's config
therefore points at a path that does not exist; that run cannot have
started (the trainer's backward loader would raise on the missing
index).

**Coordination consequence:** their ladder is a dependency we cannot
assume. §3.3 registers both branches.

### 1.4 1-3 checkpoints/eval results on disk — ALL PRE-DATE THIS LANE

| path | mtime | what it is |
|---|---|---|
| `checkpoints/mario_1_3/best_1-3.pt` | 2026-07-16 | world-1 one-shot era |
| `checkpoints/mario_1_3/robust_1_3_handoff.pt` | 2026-07-16 | chain handoff policy |
| `checkpoints/mario_1_3/eval.jsonl` | 2026-07-16 | `clear_rate 1.0`, 10 eps, **sequential chain eval, not the honest protocol** |
| `checkpoints/mario_1_3/consolidate_1-3.json` | 2026-07-16 | `best_tgt_rate 0.0`, `target_rate 0.0`, iter 395 |
| `checkpoints/mario_1_3/{metrics,run.log,prevrun_*}` | 2026-07-27 | last touched by the show lane |
| `checkpoints/winners/smb_1-3_greedy_clear.pt` | 2026-06-25 | pre-07-16; distrust per the world-1 correction |
| `checkpoints/handoffs/handoff_1-3.state`, `runs/*/handoff_1-3.state`, `runs/ge_chain_final_a/lvl_02_1-3/`, `runs/live_show/smb_4_4_micro/lvl_1-3/` | Jul | solver-chain seam states |
| `/Users/stits/Documents/nes-league-integration/scripts/diag_1_3_death_point.py` | league lane | one-rollout death-point diagnostic (uses `stage_02.state` — a *different, older* 1-3 warm-start convention; do not mix with stage_05) |

**Nothing newer than 2026-07-27 exists for 1-3.** No file on disk
suggests the parallel lane's backward run has produced a checkpoint,
a metrics stream, or an eval. As of this audit their 1-3 lane has
authored exactly one artifact: the config in §1.2.

### 1.5 Ours to reuse from the proven 1-2 stack

| artifact | role for 1-3 |
|---|---|
| `configs/mario_1_2_online_v2.yaml` | structural template (cloned, not edited) |
| `scripts/run_online_campaign.py` | the controller — now `--campaign-config`-parameterized (§4) |
| `scripts/select_restart_states.py` | dense ladder -> 6 rungs (+ new `--auto-targets`) |
| `scripts/mint_backward_states.py` | tape -> dense ladder |
| `src/utils/wavefront_reward.py` `_cli()` | **this is the dmap builder** (§3.2) |
| `scripts/replay_to_demos.py` + `scripts/bc_distill.py` | solver tape -> npz -> BC anchor (§3.1) |
| `checkpoints/_preserved/consol2_40pct_strict_iter01120.pt` | 1-2 @ 38% — **not** a 1-3 anchor; kept as the cross-level transfer arm (deferred, see §6) |
| `checkpoints/_preserved/backward_1_1_seed3_iter140.pt` | the 1-1 winner (2.7 MB, 2026-08-13 18:42) — same note |

---

## 2. Reuse-vs-build ledger (the whole point of this audit)

| thing | source | our action |
|---|---|---|
| 1-3 entrance state | `stage_05.state` (their config == solver root) | **REUSE verbatim** |
| verified 1-3 clears | `runs/ge_1_3_solve/solutions/*` | **REUSE read-only** (dmap + BC demos + ladder all derive from these) |
| 1-3 ram_mapping / rom / action_space | `configs/mario_1_3_backward.yaml` | **REUSE verbatim** (identical to 1-2's mapping; copied so our profile is self-contained) |
| dense backward ladder | `checkpoints/backward_states/1-3/` | **THEIRS IF IT LANDS**; else we mint to a lane-private dir (§3.3). Never written by us. |
| 6-rung restart set | — | **BUILD FRESH** at `checkpoints/online_1_3/restart_states/` (our dir; mirrors `checkpoints/online_1_2/restart_states/`) |
| wavefront dmap | — | **BUILD FRESH** `checkpoints/wavefront/mario_1_3_dmap.pkl` |
| KL anchor | — | **BUILD FRESH** `checkpoints/bc_1_3/anchor_h256/vanilla_ppo_iter_00000.pt` (BC h256 on 1-3 solver demos; 1-2's A7 is a 1-2 policy and is NOT transferable as an anchor) |
| campaign controller | `scripts/run_online_campaign.py` | **PARAMETERIZE** (no fork) |
| campaign profile | — | **BUILD FRESH** `configs/mario_1_3_online_v1.yaml` |
| campaign thresholds/phases | — | **BUILD FRESH** `configs/campaign_1_3.yaml` |
| their backward machinery config | `configs/mario_1_3_backward.yaml` | **READ ONLY, NEVER EDITED** — different net shape (h128/trunk32), different mechanism set; the two lanes are an intentional A/B (backward-only vs the four-mechanism online stack) |

**Duplication we deliberately avoid:** we do not re-run their solve, do
not re-mint their ladder into their directory, and do not write
anything under `checkpoints/backward_states/1-3/`. Our restart set is a
*selection* (6 files copied out) into our own directory, exactly as the
1-2 campaign did.

---

## 3. In-window steps, in order (emulator required — QUEUED, NOT RUN)

Nothing below has been executed. Every command is written against paths
that exist today except the outputs it creates. Total wall-clock
estimate: **~55-70 min of emulator time** before the campaign launch.

Preconditions for the whole block: soak finished (`pgrep -f soak_harness`
returns nothing) and `source .venv/bin/activate`.

### W1. Mint the dense backward ladder — ~6-10 min

Branch A — the parallel lane's ladder landed
(`checkpoints/backward_states/1-3/index.json` exists): **skip W1**, use
their dir read-only as the `--ladder` input in W2.

Branch B — still absent (state at audit time):

```
python scripts/mint_backward_states.py \
    --level 1-3 \
    --run runs/ge_1_3_solve \
    --profile configs/mario_1_3_online_v1.yaml \
    --out checkpoints/backward_states/1-3_online
```

Lane-private output dir (`..._online` suffix) so the two lanes can never
collide on one path. `mint_backward_states.py` aborts unless the replay
reproduces the banked clear on this machine lineage — that abort is the
verification, do not bypass it.

### W2. Select the 6 restart rungs — ~10 s

```
python scripts/select_restart_states.py \
    --ladder <W1 output or the lane's checkpoints/backward_states/1-3> \
    --out checkpoints/online_1_3/restart_states \
    --auto-targets 6
```

`--auto-targets` (new, tested) is the *pre-registered* rule for a level
whose gx range is not known in advance: it reads the ladder's own
deepest segment and lays 6 evenly spaced rungs at
`floor_to_50(max_gx * k / 6)` for k = 5..1, plus 0 (the true entrance).
No hand-picked coordinates, no level internals — the numbers come from
our own solver tape. Expected shape for 1-3 (single segment, tape max
gx ~2500-2900): rungs near 2100 / 1700 / 1250 / 850 / 400 / 0.

Compatibility check (done by reading, not running): the lane's ladder
and ours are the same format by construction —
`mint_backward_states.py` writes `index.json` via
`backward_curriculum.write_index`, and `select_restart_states.py` reads
it via `backward_curriculum.load_index`. One writer, one reader, one
schema. **No adapter is needed** as long as their mint used
`mint_backward_states.py`; if their dir turns out to hold raw `.state`
files with no `index.json`, fall back to Branch B (re-mint ours) rather
than writing an adapter — minting is 10 minutes and keeps provenance
intact.

### W3. Build the 1-3 wavefront dmap — ~3-5 min

The dmap builder is `src/utils/wavefront_reward.py`'s `_cli()` (this is
how `checkpoints/wavefront/mario_1_2_dmap.pkl` was built: replay every
banked solution from the root and take the min steps-to-goal per
`(area, x//16, y//32)` cell):

```
python -m src.utils.wavefront_reward \
    --solutions runs/ge_1_3_solve/solutions/sol_000.actions.npy \
                runs/ge_1_3_solve/solutions/sol_001.actions.npy \
                runs/ge_1_3_solve/solutions/sol_002.actions.npy \
                runs/ge_1_3_solve/solutions/sol_003.actions.npy \
    --root-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state \
    --profile configs/mario_1_3_online_v1.yaml \
    --rom "roms/Super Mario Bros. (World).nes" \
    --out checkpoints/wavefront/mario_1_3_dmap.pkl
```

All four solutions share the root, which is the builder's precondition.
Acceptance: the printed `[wavefront] N cells; dist range [0, D]` line
with `D` within a few steps of 540 (the longest tape). A `D` far above
540 means a solution from a different root leaked in — stop and fix.

### W4. Mint the BC demo npz from the solver tapes — ~2-4 min

```
for i in 000 001 002 003; do
  python scripts/replay_to_demos.py \
      --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state \
      --actions runs/ge_1_3_solve/solutions/sol_$i.actions.npy \
      --profile configs/mario_1_3_online_v1.yaml \
      --out checkpoints/bc_1_3/demos/demos_1_3_sol_$i.npz \
      --root-id entrance
done
```

The profile must name the encoder the anchor net will be trained on —
`configs/mario_1_3_online_v1.yaml` sets `encoder: smb_tiles_pos`
(712-wide), same as the 1-2 stack. Acceptance: ~2087 pairs total
(540+530+519+498) and 712-wide rows.

### W5. Train the BC anchor (h256/trunk64) — ~20-30 min

```
python scripts/bc_distill.py \
    --demos "checkpoints/bc_1_3/demos/demos_1_3_sol_*.npz" \
    --profile configs/mario_1_3_online_v1.yaml \
    --out checkpoints/bc_1_3/anchor_h256 \
    --hidden-dim 256 --trunk-dim 64 \
    --epochs 120 --lr 1e-3 --seed 0
```

Shape contract (load-bearing — `kl_anchor.load_actor_into` raises on any
mismatch): `fc1.weight = (256, 712)`, `fc2.weight = (64, 256)`, matching
`reinforce.tile_hidden_dim/tile_trunk_dim` in the profile.
Acceptance: train argmax accuracy reported; **this is a fit number, not
a capability claim** — Dossier v3 already eliminated naive BC as a
policy (clone acc 1.0 -> 0.00 honest). The anchor exists to give the
critic something to warm up against and to bound early drift, exactly
as A7 did for 1-2.

### W6. Measure the anchor's honest median and re-register the
competence floor — ~10 min

```
python scripts/eval_game.py --game mario_1_3_online_v1 \
    --profile configs/mario_1_3_online_v1.yaml \
    --rom "roms/Super Mario Bros. (World).nes" \
    --checkpoint checkpoints/bc_1_3/anchor_h256/vanilla_ppo_iter_00000.pt \
    --episodes 30 --max-steps 3000 --sequential --level-clear \
    --start-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state \
    --eval-seed 20260816 --sticky-prob 0.25 --start-jitter 16 \
    --eval-workers 5 --eval-rng per-episode
```

`configs/campaign_1_3.yaml` ships `kill_probe_median_floor: 100.0` as a
**provisional** value. The registered rule: after W6, set the floor to
`round(0.8 * measured median max-gx)` and record the measurement line in
this receipt before launch. (1-2's 150.0 floor was derived the same way
from A7's ~187 median; shipping 1-3's floor un-measured would be the
same mistake as attempt-3's un-calibrated KL threshold.)

### W7. Dry-run the assembly — ~30 s

```
python scripts/run_online_campaign.py \
    --campaign-config configs/campaign_1_3.yaml --dry-run
```

Validates: base + all six phase profiles against the strict schema, the
BC anchor shape-infers into the configured net, the 6 restart states
load and match their manifest sha256s, the wavefront dmap exists and
unpickles, and the probe command assembles against real paths. All six
checks must print PASS.

### W8. Launch — long

```
python scripts/run_online_campaign.py --campaign-config configs/campaign_1_3.yaml
```

Nothing else needs a flag: the campaign yaml carries the profile, the
run dir, the log, the restart dir, the probe game name, the thresholds
and the six phases.

---

## 4. What was built token-bound today (no emulator)

1. **`configs/mario_1_3_online_v1.yaml`** — the 1-3 base profile.
   Structural clone of `mario_1_2_online_v2.yaml` (same four mechanisms,
   same batch shape 1536x60, same PBRS-invariant gammas 0.99/0.99, same
   action space), retargeted: stage_05 entrance, 1-3 ram_mapping/rom
   from the lane's backward config, `mario_1_3_dmap.pkl`, the BC anchor,
   `checkpoints/online_1_3/restart_states`.
2. **`configs/campaign_1_3.yaml`** — the controller's CONFIG+PHASES
   overrides for 1-3, including the audit-7 **rung-progress kill**.
3. **`scripts/run_online_campaign.py`** — parameterized, not forked:
   `--campaign-config`, `campaign_name`/`probe_game` lifted into CONFIG,
   the rung-progress kill, and a dmap dry-run check. **1-2 defaults are
   byte-preserved**: with no `--campaign-config`, CONFIG and PHASES are
   exactly what they were, and `tests/test_online_campaign.py` still
   pins them.
4. **`scripts/select_restart_states.py`** — additive `--auto-targets N`
   + `auto_targets()`; default behavior unchanged.
5. **Tests** — `tests/test_campaign_1_3_package.py` (see §5).

### The audit-7 rung-progress kill

Registered in `configs/campaign_1_3.yaml` as
`kill_rung_halfway_steps: 25_000_000`. Semantics: during the
`restart_at_entrance`-gated phase (the SPDL reverse walk), the
controller reads the persisted `backward_curriculum` cursor from the
latest checkpoint at each probe cadence. If, after 25M **phase** env
steps, `tau` has not walked back past the ladder midpoint
(`tau > (n_entries - 1) // 2`), the campaign aborts with a
`KILL rung-progress` line in `campaign.jsonl`.

Why this kill exists: 1-2's attempt ledger burned three attempts on
phases that consumed budget while the *actual* curriculum cursor sat
still; a budget-abort at 60M reports "no advance" only after the whole
window is gone. Halfway-by-25M is the earliest point at which "the
reverse walk is not walking" is distinguishable from slow progress
given 60M of budget. It is deliberately a *rung* criterion, not a reward
or probe criterion — the reverse walk's own state variable is the thing
being falsified.

---

## 5. Tests (real pytest output in the handoff message)

`tests/test_campaign_1_3_package.py`:

* **schema** — `configs/mario_1_3_online_v1.yaml` passes
  `check_profile` clean; entrance is stage_05; gammas equal at 0.99;
  net dims are h256/trunk64 (the anchor contract); the ROM hash matches
  the 1-2 profile's.
* **campaign-config merge** — `configs/campaign_1_3.yaml` merges over
  the controller's CONFIG/PHASES; unknown config keys, unknown gates,
  non-sequential phase indices and wrong-typed values all raise.
* **1-2 goldens** — the merge is pure: after loading the 1-3 doc into a
  copy, the module's own CONFIG/PHASES are untouched; every pinned 1-2
  threshold and the six 1-2 phase names/gates/budgets are re-asserted.
* **rung-progress kill** — fires only past the step threshold, only
  when tau is still above the midpoint, never on a missing/short ladder,
  and clears the moment tau crosses.
* **auto-targets adapter** — 6 evenly spaced descending targets ending
  at 0, computed off the deepest segment, on a synthetic single-segment
  (1-3-shaped) ladder and re-checked on the 1-2-shaped three-segment
  ladder; the produced targets then feed `select_entries` and resolve
  within tolerance.

---

## 5b. Token-bound verification actually run (2026-08-15)

Emulator untouched throughout (`pgrep -f soak_harness` before and after:
pid 25861 alive).

**Tests**

```
pytest tests/test_campaign_1_3_package.py           35 passed
pytest tests/test_online_campaign.py                 (1-2 goldens) pass
pytest tests/test_select_restart_states.py           (selector) pass
pytest tests/test_campaign_1_3_package.py tests/test_online_campaign.py \
       tests/test_select_restart_states.py          67 passed in 0.21s
pytest tests/test_config_schema.py tests/test_backward_curriculum.py \
       tests/test_wave_monotone.py tests/test_hardening4b_catalog.py
                                                    164 passed in 4.17s
pytest tests/ -k "campaign or profile or config or restart or curriculum \
       or eval_rng or provenance"                   755 passed, 5 skipped
```

Teeth checked by mutation (both restored immediately):
`tile_trunk_dim 64 -> 32` in the 1-3 profile fails
`test_1_3_net_dims_are_the_bc_anchor_contract`;
`kill_rung_halfway_steps 25M -> 90M` in the campaign doc fails
`test_campaign_1_3_arms_the_rung_progress_kill_inside_its_phase_budget`.

**Dry runs**

`python scripts/run_online_campaign.py --dry-run` (1-2, no flag) —
`ALL CHECKS PASSED`, including the new `wavefront dmap loads — 310
cells, dist range [0, 742]` check and the unchanged
`restart states ... 6 rungs, gx [0, 497, 1000, 1499, 2000, 2502]`.
The 1-2 campaign is unaffected by the parameterization.

`python scripts/run_online_campaign.py --campaign-config
configs/campaign_1_3.yaml --dry-run` — all seven schema checks PASS
(base + six phases); exactly four FAILs, and they are precisely the
four in-window artifacts:

```
FAIL KL anchor shape-infers ...    -> W5 output missing
FAIL wavefront dmap loads ...      -> W3 output missing
FAIL restart states load ...       -> W2 output missing
FAIL probe command assembles ...   -> missing checkpoints/bc_1_3/anchor_h256/...
```

This is the acceptance signature for the package: **the only things the
1-3 assembly is missing are the things the emulator has to make.** After
W2/W3/W5 the same command must print `ALL CHECKS PASSED` before launch.

**Auto-target rule, validated against the one human-chosen answer**

Run read-only against the banked 1-2 dense ladder, output into the
session scratchpad (nothing written into the repo):

```
$ python scripts/select_restart_states.py \
    --ladder checkpoints/backward_states/1-2 --out <scratch> --auto-targets 6
[select] auto targets (6 rungs): [2600, 2100, 1550, 1050, 500, 0]
  x=   0 -> gx    0 (step   0, area 1)
  x= 500 -> gx  498 (step 271, area 2)
  x=1050 -> gx 1051 (step 371, area 2)
  x=1550 -> gx 1550 (step 435, area 2)
  x=2100 -> gx 2100 (step 579, area 2)
  x=2600 -> gx 2599 (step 937, area 2)
```

The rule reproduces the hand-chosen 1-2 ladder
(2500/2000/1500/1000/500/0) to within 150 px on every rung, and every
rung resolves inside the tolerance. That is the only external check this
rule can get, and it passes — which is why it is safe to use on 1-3,
where nobody can eyeball the answer. Pinned as
`test_auto_targets_agree_with_the_hand_chosen_1_2_ladder`.

Caveat recorded so nobody mis-reads a future comparison: the banked
`checkpoints/online_1_2/restart_states` was selected from a *different*
mint (a scratchpad ladder, per its `index.json` `actions` path), so
re-running the default selector against `checkpoints/backward_states/1-2`
yields different tape steps for the same targets. That is a difference of
source ladder, not of selection logic — `split_segments`,
`pick_bottleneck_segment` and `select_entries` are untouched by today's
change (additive `auto_targets` + an `--auto-targets` branch + a manifest
`purpose` string that now names the level and the actual targets instead
of hard-coding "1-2 / x~2674").

---

## 6. Deferred, deliberately

* **Cross-level transfer arm.** `consol2_40pct_strict_iter01120.pt`
  (1-2 @ 38%) and `backward_1_1_seed3_iter140.pt` (the 1-1 winner) are
  the obvious "does 1-2 competence transfer to 1-3?" arm. Not in v1:
  the anchor's job here is to bound early drift, and an anchor trained
  on a *different level's* manifold would make the KL telemetry
  uninterpretable. Queue it as arm B after v1 has a baseline.
* **Bottleneck telemetry.** 1-2's `bottleneck_x 2674` was a measured
  barrier. 1-3 has no measured barrier; `campaign_1_3.yaml` sets
  `bottleneck_survival_x 2000` purely as a deep-progress telemetry
  threshold (it feeds `probe_summary` only — no gate, no kill reads it).
  Do not quote it as a 1-3 wall.
* **Their lane's result.** If the backward-only lane produces a 1-3
  honest number before our campaign lands, that is the A arm of a real
  A/B and belongs in the comparison, not in our config.

## Post-repair corrections (final verification pass, same day)

The repair phase superseded several paragraphs above; per receipt
discipline they stay, with corrections here:
- Rung kill: registered at 65M cumulative (not "halfway-by-25M"), and
  the 1-2 ledger records no attempt lost to a stalled cursor — the kill
  is a provisional bound, not a calibrated one.
- auto_targets: the step-space gap WAS selection logic (scaling off the
  wrong segment); fixed — now returns (2200,1750,1300,850,400,0) scaled
  off the 2674 segment. The paragraph claiming otherwise is superseded.
- Ladder counts: actual on-disk is 903/1215 .state files (not 906/1218);
  the load-bearing inference (backward_states/1-3 absent; 540 = planned
  mint size) reconfirmed.
- W3 dmap acceptance: expect D_start ~490-515 (min steps-to-goal per
  cell sits below tape length; 1-2's dmap reads [0,742]); the guard is
  against D far ABOVE 540 only.
- W7: the retargeted dry-run emits 12 checks and correctly fails 5
  pending in-window artifacts (provenance check added post-repair).
- Falsifier wall-clock: ~3h (1000 emulator episodes after the
  power-driven raise to 4x100 gate legs + positive controls), not 2h.
- Scheduling: the "no-overnight" concern cited in review is obsolete —
  the compute policy was lifted 2026-08-14; the 1-3 campaign is cleared
  to run overnight after the falsifier.

## Interference falsifier result (2026-08-16, runs/interference/)

Pre-registered 2-level joint-BC experiment, all four legs 100 episodes,
canonical honest protocol (cold, greedy, sticky-0.25, jitter-16,
per-episode RNG), strict episode_success predicate:

| level | specialist control | joint net | leg decision |
|---|---|---|---|
| 1-1 | 43/100 | **52/100** | holds (p_above 2.1e-11) |
| 1-2 | 42/100 | 4/100 | fails (p_below 1.3e-06) |

VERDICT: partial_interference. A single 200k-param net (200,071;
h256/trunk64) trained 50 epochs CE on 122,490 pair-balanced success
transitions (61,245/level, collected sampled T=1.0 under the honest
noise profile: 1-1 189/300 strict, 1-2 97/300 strict) does not merely
degrade — it CAPTURES one level and loses the other, and the captured
leg exceeded its own specialist (median max-gx 3266 = flag).

Consequences banked:
1. Naive pooled distillation into this architecture is FALSIFIED as a
   generalist path. Next falsifiable step per the research round: add a
   Level-ID token to the observation (cheapest), or scale capacity, or
   interference-aware training (per-level heads / EWC-class). Same bar.
2. 1-2's banked 38/100 (2026-08-15, sequential shared-stream) is
   REPLICATED cross-protocol at 42/100 here (per-episode, 5 workers).
3. 1-1's honest strict rate MEASURES 43/100 here. The 0.76 figure that
   circulated in prose traces to a differently-configured evaluation
   (checkpoints/mario_1_1_consolidate_exp/eval.jsonl, 0.467 over 120 eps,
   shared-stream, no --sequential/--level-clear). Under one protocol
   1-1 and 1-2 are comparable (~0.43 vs ~0.42), not 2:1 apart. The
   frontier map should be restated in measured, protocol-named terms.

## W4 anomaly: sol_001 quarantined (2026-08-16)

`replay_to_demos.py` reported `final gx 0, ps 0` for
`runs/ge_1_3_solve/solutions/sol_001.actions.npy` while sol_000/002/003
all reproduced their banked clear (`final gx 2514, ps 5`). The tape's
own json claims a clear (`start_wd [0,2] -> clear_wd [0,3]`, 530 steps),
so the tape and the replay disagree — a provenance conflict, not a
judgement call. Per the project's replay-verification discipline the
demo was quarantined to `checkpoints/bc_1_3/demos_quarantine/` and the
BC anchor trains on the three verified tapes (1,557 pairs) rather than
on a tape whose clear does not reproduce on this machine lineage.
Root-causing sol_001 (stale lineage? hw-flag drift? a genuinely
non-reproducing archive cell?) is queued as a separate investigation;
it does not block the campaign, which needs only a warm-start anchor.

## Replay-verification failures across banked tapes (2026-08-16/17)

Two of eight banked solver tapes failed replay during BC-demo minting:
- 1-3 `sol_001` (json claims clear; replay ended gx 0)
- 1-4 `sol_003` (json claims clear; replay ended gx 0)
Both quarantined; their levels' anchors trained on the verified tapes
only. HYPOTHESIS worth testing before trusting any banked archive:
the core changed under them. Commit e9f3164 landed the DMC-DMA stall
propagation fix plus an APU abs-store tick-index correction — both
alter intra-frame cycle interleaving, which is exactly what a
frame-perfect tape depends on. Tapes recorded on the pre-fix core would
diverge only where they cross a DMC-active moment, which fits the
partial (2/8) failure rate. The migration gate recorded in the campaign
memory called for a replay-integrity sweep of banked solutions before
adopting the fixed core; these two failures may be that sweep arriving
unannounced. QUEUED: run every banked SMB solution tape through
replay_to_demos (or the mint gate) on the current core, count
divergences, and compare against a pre-fix build before any conclusion.

### Correction, same session: the DMC hypothesis is REFUTED

The loaded extension was built 2026-08-14 17:02; the DMC/APU fix landed
at 19:06 — the running core PREDATES the fix, so it cannot explain the
divergences. Metadata is identical between failing and passing tapes
(same root_state, same start_wd/clear_wd, same `provenance: search`),
and tape length shows no clean pattern (1-4's shortest fails, 1-3's
shortest passes). The residual explanation is the simplest one: these
tapes were banked without an end-to-end replay-from-root verification.
`mint_backward_states.py` verifies only the single tape it consumes, so
a partially-unverifiable solution set can sit in a run directory
indefinitely. Standing finding: ~2 of 8 banked SMB solution tapes do
not reproduce from their own recorded root on the current core. QUEUED
(unchanged in priority, corrected in rationale): sweep every banked
solution tape through the replay gate and quarantine what fails, so
downstream consumers inherit only verified provenance.
