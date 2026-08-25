# v27 — Fresh run with recovery states in the curriculum FROM THE START

Registered 2026-08-24, before any v27 training ran. Sequel #2 from
docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md, whose meta-finding
this experiment is designed to split:

> The consolidated 48k artifact is an ISOLATED OPTIMUM. Every gradient
> that touches it — imitation or on-policy, leashed or not — makes it
> worse. The trainable slice the assay measured exists in the world
> (the solver found the recoveries), but no post-hoc mechanism tested
> can transfer it into this artifact.

## The registered question

Is untrainability a property of **consolidation itself** (post-hoc
improvement impossible; recovery states must be in the curriculum from
the start of a fresh run), or of the **parameter budget**? v27 tests
the consolidation half with this machinery, config-only, exactly as
the FAIL verdict specified. It does NOT touch capacity — that is v28's
variable if v27 fails, and the DR v27 consultation (below) is asked to
adjudicate that ordering before launch.

Premise carried forward (measured, not assumed): 1/3 of the control's
true death-preceding sticks have a solver-verified recovering
continuation (3/9, receipts runs/recovery_assay/); perfect recovery
training moves honest 1-1 from 0.767 toward the measured ceiling
~0.83–0.85. The 27 mined post-stick states in
checkpoints/backward_states/1-1-recovery are the entire treatment.

## Mechanism decision: index interleave, not a second restart pool

The 27 recovery states merge into the banked restart ladder
(checkpoints/backward_states/1-1: the 903-action solver tape, 758
minted rungs) **by index interleave at their gx-mapped tape
positions**, producing one merged 785-entry ladder that the existing
`backward_curriculum` machinery consumes unchanged. The alternative —
a second restart pool drawn with a mixing weight alongside the ladder
— is rejected. Three reasons, in order of force:

1. **Config-and-data only.** The trainer supports exactly one
   `states_dir`, one `TauScheduler`, one draw site
   (src/training/trainer.py backward block; src/training/
   backward_curriculum.py). A second pool means new trainer machinery:
   a second draw branch, new provenance codes, a new scoring rule.
   The registered question says "config-only with this machinery" for
   a reason — every new mechanism bolted onto a pre-registered run
   this month re-opened the void class that preflight exists for
   (2-1 trained on another level's ladder; Phase 3's NaN-frozen actor;
   the options arm's 1e12 sentinel).
2. **Scoring integrity is the treatment.** Interleaved recovery
   entries are ordinary rungs: their outcomes land in the trailing
   window under the tau-guard, so the advance gate cannot walk the
   cursor past a recovery band until the policy clears **from those
   states too** at >= 0.2 over >= 30 attempts. Recovery competence
   becomes a precondition for ladder progress, scheduled at exactly
   the difficulty the ladder has earned. A second pool's episodes
   either contaminate the rung rate with a different distribution's
   outcomes (the exact failure `TauScheduler.record`'s tau-guard
   exists to prevent) or go unscored — an invisible, unmeasurable
   treatment either way.
3. **Variant B is the receipt against an always-on recovery
   distribution.** A uniform recovery-state restart pool (mixing
   weight 1.0, in effect) accelerated erosion on the consolidated
   artifact: recovery-pool clears 0.49 -> 0.30, entrance rate 0.31 ->
   0.06 across 60 iters. A mixing-weight pool is diluted Variant B,
   applied at every iter regardless of what the cursor has earned.
   The fresh net changes the starting point, not the scheduling
   argument.

Named counterpoint, registered now: interleave exposure is
**transient** — each recovery band is drawn only while the 40-entry
window covers it. Two properties of the recipe mitigate: training runs
sticky-0.25, so post-stick predicaments keep arising on-policy at
every later rung (the interleave seeds the recover-branch gradient
when it is learnable; sticky rollouts maintain the exposure), and
`entrance_weight: 1.0` keeps honest full-level episodes in the mix all
run. If v27 FAILS **and** the mechanism read (below) shows recovery
bands were earned and then forgotten, a persistent-mixing arm becomes
the registered follow-up — with its own registration, not this one.

## Phase 0 — mint the merged ladder (before any training)

`scripts/merge_recovery_ladder.py` (to be written; contract below)
builds `checkpoints/backward_states/1-1-v27`:

    .venv/bin/python scripts/merge_recovery_ladder.py \
      --ladder checkpoints/backward_states/1-1 \
      --recovery checkpoints/backward_states/1-1-recovery \
      --out checkpoints/backward_states/1-1-v27

Contract:

1. Load both indexes (758 + 27 entries). The recovery index's `gx: 0`
   values are placeholders — for each recovery state, boot a 1-worker
   pool from the v27 profile (same ROM md5, frame_skip 4),
   `load_worker_state(blob)`, and read gx = x_position_page*256 +
   x_position_low via the profile's own `ram_mapping` (the observable
   the reward already uses — nothing new crosses the purity line).
   Record the area byte alongside.
2. Map each recovery state to the ladder entry nearest by gx (1-1 is
   a single segment; ladder gx spans 40..3140). Insertion `step` = the
   nearest entry's step (duplicate steps are legal; `load_index`'s
   sort is stable). If the merged sequence dips > 16 px at the
   insertion point (`gx_report` tolerance), shift the insertion by up
   to +-2 rungs; if it still dips, DROP that state and record why.
3. Write the merged index: 785 entries sorted by step; recovery blobs
   byte-copied as `r_000.state`..`r_026.state`; recovery entries carry
   `frame: 900000 + i` — `entry.frame` is telemetry-only (the window
   draw counts entries, `go_explore.start_window`), so the marker is
   inert to training and makes a recovery rung *loudly identifiable*
   in the `[backward] iter ... (step S frame F gx G)` log line. Meta:
   `every_frames: 4`, `stride_steps: 1` (preserves advance_entries =
   40), provenance of both sources, and a `recovery_map` list of
   {file, source_tape, measured_gx, mapped_step, sha256} — this is the
   merge manifest.
4. Self-check, abort on failure: entry count 758 + kept; every kept
   recovery gx inside [40, 3266]; `gx_report(merged)` monotone at
   tolerance 16 / reset_max 256; every blob loads in the emulator
   without error; sha256 of every recovery blob matches its source in
   runs/recovery_distill/fuel/tapes/*.start.state.

The recovery ACTION tapes are never read. Start states only — the
tape-as-teacher family is adjudicated dead (Dossier v3; Variant A).

## Exact config diff

Four configs written and schema-validated:
`configs/mario_1_1_v27_seed{0,1,2,3}.yaml`. Machine-verified diff vs
the banked recipe `configs/mario_1_1_backward.yaml` — three keys, two
of them inert:

```diff
-name: Mario 1-1 backward
+name: Mario 1-1 v27 recovery seed{N}     # per-seed checkpoint dir slug
-description: >  (control-experiment text)
+description: >  (v27 registration text)
 reinforce:
   backward_curriculum:
-    states_dir: "checkpoints/backward_states/1-1"
+    states_dir: "checkpoints/backward_states/1-1-v27"
```

Everything else is the banked recipe VERBATIM: rollout_steps 1024,
lr 3e-4, entropy_coef 0.005 (floor off), sticky 0.25 +
sticky_episode_boundary_reset, num_envs 60, window_frames 160,
advance_threshold 0.2, advance_actions 40, min_attempts 30,
entrance_weight 1.0, truncation_is_failure false, bc_epochs 30 +
bc_replay (self-imitation of the run's own elites, not solver tapes).
**No `kl_anchor_checkpoint`** — a fresh run has nothing to leash to,
and the leash is an adjudicated FAIL besides. The four seed files are
byte-identical modulo the seed number (verified), so the checkpoint
dirs `checkpoints/mario_1_1_v27_recovery_seed{N}/` cannot collide and
auto-resume cannot cross seeds.

## Seeds, budget, launch

- **Seeds 0, 1, 2, 3** — mirroring the banked best-of-4 (which this
  gate compares against) and the v26 A/B. Train-seed via `--seed N`;
  the seed is recorded in the run manifest.
- **250 iters per seed**, the banked budget: ~61,440 samples/iter,
  ~15.4M env steps/seed, ~61M total. Sequential lanes on the M4 (one
  60-env pool saturates the P-cores), same overnight envelope as the
  two prior 4-seed campaigns.
- Launch, per seed:

```
.venv/bin/python scripts/train_game.py --game mario \
  --profile configs/mario_1_1_v27_seedN.yaml \
  --iters 250 --seed N --no-resume --strict-config \
  2>&1 | tee runs/mario_1_1_v27_seedN.log
```

## Pre-registered gate

Honest protocol, identical to the recovery-distill registration: cold
entrance from `runs/live_show/smb_4_4_micro/entrance_start.state`
(the tape's own root — lineage is part of the protocol), greedy,
sticky 0.25, jitter +-16, 50 episodes x eval seeds {0, 1}, max-steps
1500. Sampled-mode is measured and reported alongside; the gate reads
greedy. Per artifact:

```
.venv/bin/python scripts/eval_game.py --game mario \
  --profile configs/mario_1_1_v27_seedN.yaml --checkpoint <ckpt> \
  --start-state runs/live_show/smb_4_4_micro/entrance_start.state \
  --episodes 50 --max-steps 1500 --sticky-prob 0.25 --start-jitter 16 \
  --eval-seed {0,1} --action-select {greedy,sample}
```

Artifact selection, fixed now: per seed, evaluate exactly TWO
checkpoints under the full protocol — the checkpoint with the peak
trailing entrance rate in the `[backward]` telemetry (ties -> later
iter) and the final checkpoint; the seed's number is the better
pooled greedy of the two. The experiment's number is the
**best-of-4** — the same selection freedom the banked 0.76 had.

- **PASS**: best-of-4 pooled >= **0.80** (the assay's trainable-slice
  prediction), with warp_rate 0.0 (legitimate flagpole). Verdict:
  the isolated optimum is a property of post-hoc consolidation;
  recovery-in-curriculum-from-the-start transfers the slice, and
  becomes the standard recipe shape for every solver-taught level.
- **FAIL**: best-of-4 pooled <= **0.767** (the banked control's
  measured number). Verdict: from-the-start inclusion adds nothing at
  48k; the parameter-budget hypothesis takes the floor and v28 is a
  capacity experiment.
- **MARGINAL**: (0.767, 0.80) — report per-seed spread, do not
  relaunch or extend without the DR v27 answer folded in.

A PASS and a FAIL are both publishable outcomes of the registered
fork; neither is a reason to rerun with tweaks.

## Mechanism reads (registered, non-gate)

Reported with the verdict regardless of outcome:

1. **Recovery-band rung rates**: from each run's `[backward]` lines,
   the trailing-window rate while tau's window covers >= 1 recovery
   entry (identifiable by marker frames >= 900000 at the tau entry and
   by the merge manifest's mapped steps) vs adjacent windows with
   none. Answers: were recovery rungs *harder*, and were they earned?
2. **Assay re-run on the winning artifact**:
   `scripts/recovery_assay.py`, 60 episodes — did the
   trainable-slice fraction (3/9 on the control) shrink? This is the
   direct measurement of "recovery competence transferred", separable
   from the headline rate.
3. **Ladder telemetry vs banked**: iters-to-entrance, rungs/100
   iters, entrance-rate trajectory — does the merged ladder slow the
   walk (27 extra rungs, denser back half) or stall it?

## VOID conditions

A VOID is "the experiment never happened"; no verdict may be read.

- **V1 — merge gate.** Phase 0 aborts (count/gx-span/monotonicity/
  sha256/load failure), or > 3 of the 27 recovery states were dropped
  by the +-2-rung monotonicity rule (< 24 kept = treatment too thin —
  the distill registration's "fuel below threshold" clause, adapted).
  No merge manifest in the states dir, no launch.
- **V2 — preflight (the frozen-actor class).** Before the 4-seed
  spend: a 2-iter pilot at seed 0, then
  `scripts/experiment_preflight.py --before <it0> --after <it2>
  --log <pilot.log> --profile configs/mario_1_1_v27_seed0.yaml
  --iters 250`. VOID if A fails (no non-critic parameter moved), or C
  fails (any freeze sentinel outlasting 250 iters), or the log lacks
  the line `[backward] ENABLED: 785 states ... from
  checkpoints/backward_states/1-1-v27`, or it contains `[backward]
  configured but INERT` / `[backward] disabled`.
- **V3 — recovery-pool liveness (the trained-on-the-wrong-ladder
  class).** Same pilot, second check, using a THROWAWAY copy of the
  seed-0 profile with `tau_init` pinned to a merged index position
  whose window contains recovery entries (from the merge manifest): 2
  iters must show >= 1 `[backward] iter ... frame 9000xx` tau line
  (the marker proves the merged pool is the one being drawn) and ZERO
  `[backward] env N restart at tau T failed` lines (a corrupt
  recovery blob silently falls back to the entrance — a silent
  treatment amputation). Real runs then launch with `tau_init: -1`
  from the registered, unmodified configs.
- **V4 — wrong-ladder guard, per seed.** Any real run whose
  `[backward] ENABLED` line names a states_dir other than
  `checkpoints/backward_states/1-1-v27`, or a count other than the
  manifest's 758 + kept: that seed is void (this exact class voided
  2-1 attempt 1).
- **V5 — actor/ladder liveness mid-run, per seed.** S1 of the banked
  registration: a seed whose tau has advanced 0 rungs by iter 150 is
  VOID-machinery for that seed (the ladder never engaged; that is not
  evidence about recovery states). If all four seeds void this way
  the experiment is VOID, not FAIL — FAIL requires walked ladders.
- **V6 — eval lineage.** Any gate eval not run with
  `--start-state runs/live_show/smb_4_4_micro/entrance_start.state`
  and the registered flags is discarded, not reported.

## DR v27 consultation

Prompt written:
`~/Documents/research-consult/prompts/v27_isolated_optimum.md`. It
carries the full isolated-optimum autopsy (all three post-hoc
families with signatures and numbers, plus the earlier
degradation-under-any-gradient results) and asks **one decision**:

> (A) launch v27 as registered — the fresh-run interleave at 48k is
> the correct single-variable falsifier of the consolidation
> hypothesis, capacity waits for v28 — or (B) name exactly ONE change
> the autopsy plus literature (warm-start gap, plasticity loss,
> primacy bias, dormant neurons) already forces before the budget is
> spent, with what each v27 outcome would then mean for the fork.

Send via `scripts/consult_deep_research.py` (the live Interactions-API
bridge). Sequencing: phase 0 and the V2/V3 pilot may run immediately
(they spend minutes, not the budget); the 4x250 launch waits for the
DR answer. If DR answers (B), the change gets its own registration
amending this one — this document does not mutate silently.

## Receipts layout

- Merge manifest: `checkpoints/backward_states/1-1-v27/index.json`
  (`recovery_map`)
- Pilot + preflight: `runs/v27_fresh_recovery/preflight/`
- Training: `checkpoints/mario_1_1_v27_recovery_seed{0..3}/`,
  `runs/mario_1_1_v27_seed{0..3}.log`
- Gate evals: `runs/v27_fresh_recovery/eval_seed{N}_{ckpt}_{es}.json`
- Verdict: appended to THIS file, per house style.
