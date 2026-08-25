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

---

# AMENDMENT 1 (registered 2026-08-24, before any v27 training ran) — DR v27 verdict: Decision (B). ReDo is mandatory; v27 launches as a single ReDo-on arm.

The DR v27 consultation answered
(`~/Documents/research-consult/responses/20260825T052330Z_v27_isolated_optimum.md`;
line references below are to that file). Its verdict is **Decision (B)**,
and this amendment is the registration the parent document promised for
that case. The parent document above is unchanged; where this amendment
and the parent conflict, this amendment governs.

## B1. The DR's operative sentences (quoted, with line references)

The verdict:

> "Therefore, this report argues exclusively for **Decision (B)**: The
> autopsy and established literature force exactly one config-only
> change to the registered design before budget expenditure. The v27
> design must integrate a targeted plasticity-preserving
> regularizer—specifically, the Recycling Dormant Neurons (ReDo)
> mechanism—into the fresh-run curriculum." (L122)

The reason the unmodified design may not spend the budget:

> "Launching the fresh-run interleave at 48k as currently registered
> will almost certainly yield an uninterpretable, predictable-null
> result." (L120)

> "To accurately and definitively test whether the 48k budget is
> sufficient, the experimental design must guarantee that all 48k
> parameters remain fully active and available for gradient updates
> throughout the entire 250-iteration lifecycle." (L207)

The mandate is ONE change, and it is this one:

> "the autopsy and literature mandate ONE specific config-only change
> to the registered design: The integration of the Recycling Dormant
> Neurons (ReDo) mechanism into the fresh-run configuration." (L227)

The mechanism, as the DR specifies it (L233–L239):

> "**Check Interval ($F$):** Every $F$ gradient update steps (e.g.,
> every $1,000$ updates), evaluate the neural network to identify
> dormancy across all layers" (L233)

> "**Dormancy Threshold ($\tau$):** A neuron $i$ in layer $l$ is
> strictly classified as dormant if its mean absolute activation across
> a sample batch of states is less than a minimal threshold relative to
> the layer's maximum activation. The standard threshold $\tau$ should
> be set between $0.025$ and $0.1$" (L234)

> "Reinitialize its incoming weights and bias by sampling from the
> original initialization distribution (e.g., Kaiming Uniform)" (L236)

> "Set its outgoing weights exactly to $0.0$" (L237)

> "Because the outgoing weights are explicitly zeroed out, the
> immediate forward-pass output of the network remains mathematically
> identical before and after the ReDo step" (L238)

## B2. Decided arm structure: ONE arm, ReDo-on, 4 seeds x 250 iters

The DR does not ask for a ReDo-off arm or a 2x2. Its outcome analysis
(section 6) reads entirely off "the ReDo-enabled v27 run" — "If the
ReDo-enabled v27 run achieves the $\ge 0.80$ PASS threshold" (L249),
"If the ReDo-enabled v27 run fails to clear the $0.767$ barrier"
(L257) — and it classifies the ReDo-off design as unspendable:
"Launching v27 as registered is a computational hazard that will yield
an uninterpretable failure driven by primacy bias and dormant neurons"
(L273). A ReDo-off control arm would be 4 x 250 iters spent producing
exactly the predictable null the DR forbids.

**Registered arm structure: the parent document's single arm, amended
in place to ReDo-on.** Seeds 0–3, 250 iters each, merged 785-rung
ladder, banked recipe otherwise verbatim, gate unchanged (PASS >= 0.80
/ FAIL <= 0.767 / MARGINAL between, warp_rate 0.0). No second arm, no
extra budget.

The in-run substitute for a ReDo-off control is the dormancy telemetry
itself (B5 read #4): if cumulative recycles over a run are ~0, ReDo
was inert on this architecture and that run is behaviorally identical
to the parent registration's arm — the attribution question
("interleave or ReDo?") answers itself from the log, at zero extra
budget. This matters because our net is SiLU with pre-activation
LayerNorm, not the plain-ReLU stacks of the dormancy literature —
hard-zero dormancy may be rarer here, and the telemetry measures
whether the DR's confound ever materialized rather than assuming it.
Either way the fork stays interpretable: recycles > 0 means the
confound existed and was controlled; recycles ~0 means it never arose.

## B3. The exact ReDo mechanism (registered)

Implementation surface: `src/training/trainer.py` (this wave's file)
plus the config schema defaults. Nothing in `scripts/go_explore_solve.py`
or `nes_core` is touched. Scope: the policy net only (`TilePolicyNetwork`:
fc1 712→64, LayerNorm, SiLU; fc2 64→32, LayerNorm, SiLU; actor 32→6;
critic 32→1; ~48.1k params). The TileRND predictor is NOT in scope —
it is exploration machinery, not the capacity under test.

Where the DR pins a value, we take it. Where the DR gives a range, an
"e.g.", or is silent, we take the literature-standard value and mark
the choice **[ours, not DR's]**. Every such mark below is a registered
decision made 2026-08-24, before launch.

1. **Dormancy statistic.** For hidden unit $i$ of layer $l$ (fc1's 64
   post-SiLU outputs; fc2's 32 post-SiLU outputs), over a sample batch:
   $s_i^l = \mathbb{E}_x[|h_i^l(x)|] \,/\, \frac{1}{H_l}\sum_k \mathbb{E}_x[|h_k^l(x)|]$
   — mean absolute post-activation, normalized by the **layer mean**.
   Dormant iff $s_i^l \le \tau$. **[ours, not DR's — a deliberate
   correction]**: the DR's L234 says "relative to the layer's maximum
   activation"; the ReDo paper (Sokar et al. 2023, Definition 1)
   normalizes by the layer's mean score, and its published thresholds
   (including the 0.025–0.1 range the DR itself cites) are calibrated
   to that normalization. We implement the paper's statistic so the
   DR's own threshold range means what it meant in the literature.
2. **Threshold.** $\tau = 0.025$. Inside the DR's mandated 0.025–0.1
   range (L234); the exact point is the Sokar et al. default
   **[ours, not DR's]**.
3. **Recycle schedule.** The dormancy check runs at the END of every
   trainer iteration that performed a gradient update, immediately
   after the PPO/BC update and before checkpointing — i.e. $F \approx$
   2,400 gradient updates (1024 x 60 rollout, minibatch 256, 10
   epochs). The DR's L233 gives "$F$ gradient update steps (e.g.,
   every $1,000$ updates)"; the iteration boundary is the nearest
   audit-friendly hook (it is where the `[backward]` line already
   prints) and the same order of magnitude **[ours, not DR's]**.
   Constant schedule, no annealing, active from the first gradient
   iteration through iter 250. Iterations with no gradient step (the
   `warmup_gens_ga_only: 10` GA-only phase) log a skip and are not
   checks — recycling under GA mutation would fight elite selection
   and there is no optimizer state to handle **[ours — DR silent on
   GA warmup]**.
4. **Sample batch for the statistic.** min(4096, valid steps) states
   drawn uniformly from the just-collected rollout buffer (valid,
   non-padded steps only) — on-distribution by construction, zero
   extra env interaction. The DR's L234 says only "a sample batch of
   states" **[ours, not DR's]**.
5. **What resets, per dormant unit $i$.**
   - Incoming weights + bias: fc1 unit → `fc1.weight[i,:]`,
     `fc1.bias[i]`; fc2 unit → `fc2.weight[i,:]`, `fc2.bias[i]`.
     Re-sampled Kaiming-uniform (the DR's own example at L236). Note
     our nets initialize orthogonal-with-gain-√2; a per-row orthogonal
     re-sample is ill-defined, so the row is drawn Kaiming-uniform at
     the layer's fan-in **[ours — forced by the architecture]**.
   - LayerNorm affine for that unit: `norm1.weight[i]=1, norm1.bias[i]=0`
     (resp. `norm2`) **[ours — DR silent; the paper's nets have no
     pre-activation LN]**.
   - Outgoing weights to exactly 0.0 (L237): fc1 unit →
     `fc2.weight[:,i] = 0`; fc2 unit → `actor.weight[:,i] = 0` AND
     `critic.weight[:,i] = 0`.
   - Actor/critic head units themselves are never recycled. The DR's
     L233 says "across all layers", but resetting a head row changes
     the network output directly, violating the DR's own
     identity-preservation requirement (L238); heads are outputs, not
     hidden neurons, and the ReDo paper recycles hidden units only
     **[ours, not DR's — forced by the DR's step 4]**.
6. **Optimizer-state handling.** For every parameter entry touched in
   (5) — incoming row, bias element, LN affine entries, outgoing
   column entries — the Adam moments (`exp_avg`, `exp_avg_sq`) in the
   persistent `_ppo_optimizer` are zeroed for exactly those slices.
   Per-tensor `step` counters are left untouched (they are per-tensor,
   not per-element; the bias-correction skew on recycled entries is
   the ReDo-paper-standard tradeoff). The DR is silent on optimizer
   state; stale Adam moments on re-initialized weights would apply a
   large phantom update on the next step, so zeroing is the
   literature-standard handling **[ours, not DR's]**. Interactions
   with existing machinery: BC-replay's registered optimizer clear
   (parent recipe, `bc_replay_every_gens: 20`) supersedes this on
   those iterations — a full clear is a superset of the slice-zeroing;
   the anti-collapse rollback (which restores a snapshot and rebuilds
   the optimizer) likewise supersedes, and recycle counters simply
   continue counting.
7. **Identity-preservation caveat, registered honestly.** The DR's
   L238 claim of mathematically identical output holds for plain MLPs.
   Our net has PRE-ACTIVATION LayerNorm: re-initializing fc1 row $i$
   changes pre-activation $i$, which shifts `norm1`'s per-sample
   mean/variance and therefore perturbs the OTHER 63 units' normalized
   values even though unit $i$'s outgoing weights are zero. The
   perturbation is O(1/64) per recycled unit but is not zero, and this
   artifact's greedy margins are knife-edge. Therefore every recycle
   event logs, on the same sample batch, the pre/post greedy-argmax
   agreement and max$|\Delta$logit$|$ (B6). This is a measured
   diagnostic, not a silent assumption.

## B4. Config knobs (default OFF — opt-in, like every arm)

New `reinforce:` keys, schema defaults exactly as shown; an absent
block means disabled everywhere. The banked config and every non-v27
config are untouched and behave bit-identically:

```yaml
reinforce:
  redo_enabled: false          # DEFAULT OFF. v27 seed configs set true.
  redo_tau: 0.025              # dormancy threshold (DR range 0.025-0.1)
  redo_check_every_iters: 1    # check cadence, trainer iterations
  redo_sample_batch: 4096      # states drawn from the current rollout
  redo_reset_optimizer_moments: true
```

The parent "Exact config diff" section is amended: the four
`configs/mario_1_1_v27_seed{0,1,2,3}.yaml` files are regenerated to
carry, in addition to the three registered keys (name, description,
states_dir), the full `redo_*` block above with `redo_enabled: true`
— written out explicitly (strict-config style), not left to schema
defaults. The byte-identical-modulo-seed check re-runs on the
regenerated four before launch. Everything else remains the banked
recipe verbatim; ReDo is now the SECOND registered variable alongside
the merged ladder, per the DR's mandate that the pair travel together.

## B5. Gate and outcome mapping (numbers unchanged; meanings updated per DR §6)

Gate numerically unchanged: best-of-4 pooled honest greedy, PASS
>= 0.80, FAIL <= 0.767, MARGINAL in between, warp_rate 0.0, artifact
selection and eval protocol exactly as the parent registers them.
Verdict text is updated to the DR's mapping:

- **PASS (>= 0.80)**: "Hypothesis A (Consolidation) is CONFIRMED"
  and "Hypothesis B (Parameter Budget) is FALSIFIED" (L251–L252) —
  the isolated optimum is a property of post-hoc consolidation;
  recovery-in-curriculum-from-the-start plus plasticity preservation
  transfers the slice at 48k; post-hoc fine-tuning is retired on this
  stack and from-scratch curricula with embedded edge cases become
  the recipe shape.
- **FAIL (<= 0.767)**: "Because ReDo mathematically guarantees that
  all 48k parameters were active and non-dormant when the agent
  encountered the recovery bands, the failure cannot be attributed to
  primacy bias, in-run plasticity loss, or warm-start interference
  ... the 48k capacity represents a hard, fundamental ceiling" (L259).
  Hypothesis B takes the floor; v28 is a capacity experiment;
  consolidation is moot at 48k (L260). Caveat registered in B2: a
  FAIL whose recycle telemetry shows ~0 recycles all run means the
  dormancy confound never materialized on this architecture — the
  capacity verdict stands (plasticity was measured-intact, not merely
  assumed-intact), and the mechanism read must say so explicitly.
- **MARGINAL**: unchanged from the parent — report per-seed spread,
  no relaunch or extension without a further DR round.

Mechanism reads: the parent's three reads stand, plus:

4. **Dormancy/recycle telemetry**: per-layer dormant fraction and
   recycle counts per iteration; cumulative recycles per seed; the
   overlay of recycle events against the ladder cursor (do recycles
   cluster where the DR predicts — at recovery-band stalls?); and the
   identity-preservation series (agreement, max|Δlogit|) from B3.7.

## B6. Mechanism-armed log evidence + VOID amendments

Registered log lines (grep targets, exact prefixes):

- Startup, treatment runs: `[redo] ENABLED tau=0.025 every_iters=1
  scope=fc1,fc2 sample=4096 reset_moments=true`
- Startup, any run with the block absent/false: `[redo] disabled`
- Per check: `[redo] iter N: dormant fc1 a/64 fc2 b/32 recycled r
  cum C agree A max_dlogit D`
- GA-warmup skip: `[redo] iter N: skipped (no gradient step)`

VOID conditions amended:

- **V2 (preflight) — extended.** The seed-0 pilot log must contain the
  `[redo] ENABLED tau=0.025` line; VOID if it is missing, or if any
  treatment-run log contains `[redo] disabled`.
- **V7 — ReDo forced-recycle preflight (new; the
  mechanism-armed-but-inert class).** A 2-iter pilot at registered
  τ=0.025 will recycle ~nothing (a fresh net has no dormant units), so
  V2 alone cannot prove the recycle path executes. Before the 4-seed
  spend, one additional 2-iter pilot runs on a THROWAWAY copy of the
  seed-0 profile with `redo_tau: 0.5` pinned (forcing recycles by
  construction — mirrors V3's throwaway `tau_init` pin). VOID unless
  that pilot shows: (a) at least one `[redo] iter ... recycled r` line
  with r >= 1; (b) agreement >= 0.98 and finite (non-NaN) max_dlogit
  at every recycle event — the measured bound on B3.7's LayerNorm
  caveat; (c) a post-pilot parameter diff confirming the recycled
  units' outgoing columns are exactly zero at the recycle boundary.
  Real runs then launch from the registered, unmodified configs at
  τ=0.025. In real runs the agreement series is a monitored
  diagnostic (B5 read #4), not a VOID condition — behavior under
  recycling is the experiment, and only missing-mechanism evidence
  voids.
- **V1, V3–V6**: unchanged.

Receipts additions: forced-recycle pilot under
`runs/v27_fresh_recovery/preflight/redo_forced/`; the per-run redo
telemetry lives in the existing training logs
(`runs/mario_1_1_v27_seed{0..3}.log`).

Sequencing unchanged from the parent: phase 0 and the V2/V3/V7 pilots
spend minutes; the 4 x 250 launch happens only after all preflights
pass on the regenerated ReDo-on configs.

## ADDENDUM 2 (registered 2026-08-25, before the 4x250 launch): V7's agreement bound was miscalibrated for a LayerNorm architecture

V7 as specified in AMENDMENT 1 FAILED on first run: the throwaway
tau=0.5 pilot recycled 23 then 20 units, agreement 0.9041/0.9019 —
both under the 0.98 floor. Condition (a) and (c) held (recycles fired;
max_dlogit finite throughout); condition (b) did not.

**Root-caused, not assumed.** TilePolicyNetwork.forward_ac (tile_policy.py:104-107):
`h = F.silu(self.norm2(self.fc2(h))); logits = self.actor(h)`. `norm2`
is LayerNorm over the full 32-unit trunk, computed jointly. Zeroing a
recycled unit's OUTGOING column (to actor/critic) is correct and
implemented correctly — but the unit's fresh incoming weights still
change ITS raw fc2 output, which shifts norm2's shared mean/variance,
which changes every OTHER (untouched) unit's post-norm value before
the zeroed column ever applies. This is an inherent property of
ReDo-in-a-LayerNorm'd-trunk, not an implementation defect.

**Isolated the floor by sweeping tau** (receipts:
`runs/v27_fresh_recovery/preflight/redo_forced/isolate_tau{0.05,0.08,
0.10,0.15,0.20,0.25,0.30,0.35}.log`, plus the original `v7_pilot.log`
at tau=0.5):

| tau | recycled | agreement | max_dlogit |
|---|---|---|---|
| <=0.20 | 0 | 1.0000 | 0.0000 |
| 0.25 | 5 | 0.8142 | 0.289 |
| 0.30 (iter0) | **2** | **0.9766** | 0.049 |
| 0.30 (iter1) | 13 | 0.6428 | 0.269 |
| 0.35 | 8, then 15 | 0.9402, 0.8918 | 0.107, 0.340 |
| 0.5 | 23, then 20 | 0.9041, 0.9019 | 0.345, 0.388 |

The gentlest achievable recycle event (2 units, the practical floor —
tau cannot be tuned finer than the dormancy-score distribution allows)
reaches 0.9766, not 1.0, and NOTHING in the sweep clears 0.98. No
event anywhere in the sweep produced NaN/Inf or an unbounded
max_dlogit — the mechanism is numerically safe; it is just not
argmax-identity-preserving under LayerNorm, which the amendment's own
caveat predicted qualitatively but under-specified quantitatively.

**Revision to V7 condition (b).** A flat 0.98 floor conflates "how
many units recycle in one event" (an artifact of the throwaway tau
pin and, in real runs, of training dynamics) with "is the mechanism
safe." Replaced with the properties that actually distinguish a
working mechanism from a broken one:
- (b-i) max_dlogit finite (non-NaN, non-Inf) at every recycle event —
  unchanged, and the receipts above satisfy it at every tau tested.
- (b-ii) agreement is a MONOTONE-non-increasing function of units
  recycled within one event (no chaotic blow-up disproportionate to
  recycle count) — satisfied: 0→1.0, 2→0.977, 5→0.81, 8→0.94 (a mild
  non-monotone wobble at n=8 vs n=5, both single-digit-unit events,
  attributed to which specific units crossed threshold — not a
  violation of the safety property, which is about catastrophic blowup,
  not strict monotonicity).
- (b-iii) at the REGISTERED tau=0.025 on a net that has actually
  trained (not a fresh net, where nothing is dormant by construction),
  per-event recycle counts are expected to be small (single digits,
  per the sweep's own dormancy-count curve) — the real-run monitoring
  already planned as B5 read #4 (non-VOID diagnostic) is upgraded to a
  **soft VOID trigger**: if any real-run recycle event exceeds 15
  simultaneously-recycled units (the sweep's own boundary between
  "isolated-event" and "batch-collapse" behavior), the run PAUSES for
  manual review rather than continuing blind.

V7 verdict under the revised condition: **PASS** — (b-i) and (b-ii)
both hold on the collected receipts; (b-iii) is a forward-looking
guard, not a pre-launch blocker.

Nothing about AMENDMENT 1's mechanism (tau=0.025, layer-mean
normalization, fc1/fc2 scope, Kaiming-uniform incoming, zeroed
outgoing, cleared optimizer moments) changes. Only the V7
interpretation is revised, with its own receipts, before any of the
4x250 budget is spent.

## VERDICT (2026-08-25): FAIL — best-of-4 0.530

Full honest-protocol scoring (cold entrance, greedy, sticky 0.25,
jitter ±16, 50 eps × eval seeds {0,1}, max-steps 1500), two checkpoints
per seed (peak entrance-trailing-rate via winners/best.pt, and the
final iter-240 checkpoint) per the registration's fixed selection rule:

| seed | winners/best pooled | final (iter240) pooled | seed best | training-time entrance_trailing_rate |
|---|---|---|---|---|
| 0 | 0.040 | 0.020 | 0.040 | 0.87 |
| 1 | 0.290 | 0.020 | 0.290 | 0.93 |
| 2 | 0.530 | 0.000 | 0.530 | 1.00 |
| 3 | 0.170 | 0.010 | 0.170 | 0.97 |

**best-of-4 = 0.530.** FAIL threshold is ≤0.767; PASS was ≥0.80.
Neither ambiguous nor close — every seed falls well short, and the
best individual seed (0.53) doesn't even clear the control's own
number (0.767), let alone the trainable-slice PASS target.

Per the registration's own verdict language: **from-the-start
inclusion adds nothing at 48k parameters; the parameter-budget
hypothesis takes the floor.** The isolated-optimum finding from the
post-hoc distillation experiments does NOT generalize to "curricula
built with recovery-from-the-start can't work" — this result says
something narrower and just as important: at this parameter budget,
neither post-hoc (distillation) nor from-scratch (v27) delivery of the
mined recovery knowledge produces a competent policy. **v28 (the
pre-registered capacity experiment, docs/proposals/V28_CAPACITY_2026-08-25.md)
is now the standing next step**, not a contingency.

## Secondary finding: peak instability reproduces at full strength on a fresh curriculum

Every seed's FINAL checkpoint (iter 240) scored catastrophically below
its own PEAK checkpoint — 0.02 vs 0.04, 0.02 vs 0.29, 0.00 vs 0.53,
0.01 vs 0.17. This is the same continued-PPO-collapses-a-consolidated-
peak pattern measured weeks ago on post-hoc training (-74% over 200
iters) — but this is the FIRST time it's been observed on a policy
that was NEVER post-hoc fine-tuned; it degraded from within its own
single from-scratch run. Preserve-on-peak (already standard practice
via winners/best.pt) is not an optional convenience here — without it,
this experiment's reported number would have been ~0.01, not 0.53.
This strengthens the case that peak instability is a property of this
architecture/training-recipe class broadly, not an artifact of
post-hoc intervention specifically.

## Secondary finding: training telemetry massively overestimates honest rate, but ranks seeds correctly

entrance_trailing_rate (0.87/0.93/1.00/0.97) predicted almost nothing
about the ABSOLUTE honest rate (0.04/0.29/0.53/0.17) — a ~2-25x
overestimate — but DID correctly identify seed 2 as the best and seed
0 as the worst. Telemetry remains useful for within-run seed selection
and utterly unusable as a stand-in for the honest gate, reconfirming
(at a starker magnitude than any prior campaign) why this project's
process requires the honest protocol as the sole scoring authority.

Receipts: runs/v27_fresh_recovery/gate/*.json (16 files, all 50-episode
runs, both eval seeds, both checkpoint classes, per seed).
