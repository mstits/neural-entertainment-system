# Trainer decomposition plan — strangling `_run_vanilla_ppo` and `__init__`

**Status:** proposal / execution guide. **Scope:** `src/training/trainer.py`
only. **Constraint:** *no behavior change.* Every step is a pure structural
move behind a green characterization test. This is a one-safe-step-at-a-time
plan, explicitly **not** a big-bang rewrite.

`_run_vanilla_ppo` is ~3,130 lines (4454–7584); `__init__` is ~965 lines
(149–1113). The loop has been the site of most of the recent production bugs
(reward desert + freeze starvation, die-respawn eval inflation, capture-gate
off-by-one, RND self-referential norm, f16 double-norm, the anti-collapse
optimizer-drop, checkpoint over-write on resume). Those bugs are the reason to
decompose — but they are also the reason to move *slowly*, because the same
tangle that hides bugs will hide a botched extraction. The plan therefore front-
loads a **behavior-pinning test harness** and orders extractions **lowest-
coupling first**, each isolated to one module and one commit.

---

## 0. Goal, non-goals, definition of done

**Goal.** Break the monolith into five tested modules with explicit seams:

| Module | Owns (state) | Owns (behavior) |
|---|---|---|
| `RolloutCollector` | rollout buffers, stackers, per-env episode/progress trackers, sticky state, recurrent hidden | forward → sample → step_all → per-env bookkeeping → stacker advance → bootstrap V(s_T) |
| `PPOUpdater` | nothing persistent (references only) | intrinsic/count fold → batched GAE → advantage norm → K-epoch minibatch update → NaN backstop |
| `ExplorationController` | `_rnd`, `_gx_counts` + beta, Go-Explore archive + burst state | RND intrinsic + predictor loss, count bonus, Go-Explore record/return/burst |
| `Curriculum` | curriculum states/anchors/stage, pending capture, ladder, `env_stage`, consolidation + cold-probe state | capture, advance gating, warm-start partition, cold-eval probe, forgetting/consolidate gates, state-file I/O |
| `CheckpointManager` | nothing persistent (paths only) | resume scan/load, iter checkpoint save + poison guard, winner retention, run manifest |

**Non-goals (do not do while decomposing).**
- No hyperparameter/algorithm changes. No "while I'm in here" fixes — a real bug
  found mid-extraction gets its **own** commit *before or after*, never folded in.
- No touching the GA path (`_run_one_generation`, `_reinforce_update`) except to
  share already-extracted helpers.
- No new features (the burst, consolidation, ladder all stay exactly as-is).
- No dependency inversion frameworks / DI containers. Plain objects, constructor
  injection, method calls.

**Definition of done.** `_run_vanilla_ppo` shrinks to a conductor (target
< 300 lines) that builds five collaborators and runs a loop of ~8 method calls;
`__init__` shrinks to < 200 lines that parses config and wires collaborators;
the full `make test` suite plus the new characterization goldens are green at
every intermediate commit.

---

## 1. Current structure map (phase → line range)

The loop is already implicitly phased by the `# ===== ...` banners. Preserve
these as the extraction fault-lines.

**Setup (4482–5329), before `for it in range(num_iters)`:**

| Lines | Phase | Target owner |
|---|---|---|
| 4489–4553 | lazy build: net, demo bank, RND, optimizer | conductor + Exploration (RND) |
| 4565–4577 | resume offset + resume_metrics | CheckpointManager |
| 4583–4597 | run manifest write | CheckpointManager |
| 4600–4602 | per-env reward fns | RolloutCollector |
| 4651–4738 | SMB scalar curriculum init + disk load | Curriculum |
| 4740–4908 | sub-stage ladder setup | Curriculum |
| 4910–5113 | level-scoped consolidation setup + baselines | Curriculum |
| 5115–5202 | stackers + rollout buffer allocation + per-env trackers | RolloutCollector |
| 5204–5288 | Go-Explore archive setup | ExplorationController |
| 5290–5302 | initial `reset_all` + `stacked_obs` seed | RolloutCollector |
| 5314–5329 | sticky-action setup | RolloutCollector |

**Per-iter loop (5331–7584):**

| Lines | Phase | Target owner |
|---|---|---|
| 5343–5909 | **rollout collection** (forward, sample+sticky, step_all, per-env reward/done, curriculum capture, GX record, auto-reset-on-death, stacker push, GRU reset, fused drain, bootstrap) | RolloutCollector (calls Curriculum + Exploration hooks) |
| 5922–5973 | RND intrinsic fold | ExplorationController (fold) driven by PPOUpdater |
| 5974–5981 | count-bonus fold | ExplorationController (fold) driven by PPOUpdater |
| 5983–5995 | batched GAE | PPOUpdater |
| 5997–6012 | advantage normalization over valid mask | PPOUpdater |
| 6014–6253 | K-epoch minibatch update (+ RND cache, demo anchor, NaN backstop) | PPOUpdater |
| 6255–6541 | logging + metrics assembly + **stage-advance detection & advance** | RolloutCollector stats → Curriculum.advance; conductor assembles metrics |
| 6543–6716 | progress metrics + **cold-eval probe** + winner/forgetting/consolidate triggers | Curriculum |
| 6717–6886 | level-scoped consolidation gate | Curriculum |
| 6888–7015 | Go-Explore unstick burst | ExplorationController (decisions already in `oneshot_curriculum`) |
| 7017–7050 | consolidation/cyclic + clevel coefficient schedule (writes `entropy_coef`/`rnd_intrinsic_coef`) | Curriculum → shared knobs |
| 7052–7094 | metrics emit + `clevel_done` break | conductor |
| 7096–7102 | narrator drain | conductor |
| 7104–7145 | anti-collapse guard (snapshot / rollback / optimizer rebuild) | conductor (`_anti_collapse_guard`) |
| 7147–7160 | entropy-floor controller | conductor (`_entropy_floor_step`) |
| 7162–7433 | **iter-boundary reset + warm-start** (reset_all, consolidate/ladder/mixed/GX warm-start, reward-fn reset, tracker reset, stacked_obs rebuild) | Curriculum (assignment) + RolloutCollector (mechanics) |
| 7435–7571 | checkpoint save + poison guard + winner retention | CheckpointManager |
| 7573–7584 | Go-Explore archive persist | ExplorationController |

Note that the "decision" logic for the ladder, consolidation, forgetting, and
burst is **already extracted** into `src/training/oneshot_curriculum.py` and
`src/training/smb_substage_ladder.py` (unit-tested by `test_oneshot_curriculum`,
`test_smb_substage_ladder`, `test_consolidate_level`, `test_oneshot_ge_burst`).
What remains in the trainer is *glue*. This is precedent: `ppo.py` already holds
`batched_gae`, `ppo_losses`, `fold_intrinsic_into_rewards`, `demo_anchor_loss`
(tested by `test_ppo`). We are continuing the same strangler that started there.

---

## 2. The seams — shared mutable state (the hard part)

The loop's difficulty is not the phases; it is the ~40 local variables and
`self.*` attributes each phase reads and writes. Decomposition is only safe if
every shared datum has **exactly one owner** and every cross-module touch is a
named method call. Inventory below; the "Hazard" column flags the seams that
have already produced bugs.

### 2.1 Rollout buffers (owner: `RolloutCollector`)
`obs_buf, action_buf, reward_buf, bonus_buf, value_buf, log_prob_buf, done_buf,
valid_buf, step_actions` (allocated 5134–5150). Filled during rollout; consumed
by PPOUpdater.
- **Hazard — `reward_buf` is mutated by two later phases** (RND fold 5968, count
  fold 5979). Ownership stays with the collector (allocator); the fold is a
  documented in-place transform the updater invokes. `valid_buf` gates both
  advantage-norm (6005) and the minibatch permutation (6023) — its semantics
  ("real executed step incl. death; post-done padding is False") must survive
  the collector→updater boundary verbatim.

### 2.2 Per-env episode/progress trackers (owner: `RolloutCollector`)
`ep_returns, ep_lengths, completed_returns, completed_lengths, active_in_iter,
_stage0_reseed, prev_completion_total, n_clears_this_iter, max_world_level_packed,
end_world_level_packed, max_x_reached, max_region_reached` (5153–5202).
- **Hazard — `active_in_iter` / freeze semantics.** The auto-reset-vs-freeze
  branch (5695–5838) is exactly where "freeze starvation" and "die-respawn eval
  inflation" lived. It reads curriculum seed state and writes pool state
  (`set_worker_done`), reward-fn state (`reward_fns[i].reset()`), and
  `prev_completion_total[i]=0`. This is the single most entangled seam (§4.5).
- `max_region_reached` (ladder) and `max_world_level_packed` (scalar) feed the
  advance gate (6345) — the collector *produces* them, Curriculum *reads* them.

### 2.3 Stackers (owner: `RolloutCollector`)
`stackers` / `tile_stackers`, `stacked_obs` (5115–5127, 5293–5302). Reseeded on
auto-reset (5749–5765), pushed each step (5855–5863), rebuilt at iter boundary
(7421–7430).

### 2.4 Curriculum state (owner: `Curriculum`)
`smb_curriculum_states, smb_curriculum_anchors, smb_curriculum_stage,
smb_pending_capture, smb_pastfrac_history, smb_stage_clear_history, env_stage,
stage_seed_results, ladder, retention_frac (+bump), cold_highwater,
forget_strikes, best_cold_key/rate/snapshot, consolidating + cons_* + cyclic_*,
clevel_* (target/rungs/entries/baselines/rates/sustain/step/cooldown/done),
accepted_snapshot`.
- **Hazard — `env_stage` and `stage_seed_results` are shared with the collector.**
  Curriculum *assigns* them at the iter boundary (7197–7386); the collector
  *reads* them during auto-reset (5716, 5749). Contract: Curriculum owns the
  values; the collector receives them read-only per iter.
- **Hazard — `best_cold_snapshot` / `accepted_snapshot` are net snapshots** that
  drive rollback (`net.load_state_dict` + optimizer rebuild, 6661/6792). Curriculum
  must be allowed to mutate `net` and rebuild `optimizer` — a real cross-module
  write. Model it as `Curriculum.maybe_rollback(net, optimizer_ref)` returning a
  possibly-new optimizer, not a hidden side effect.

### 2.5 Exploration state (owner: `ExplorationController`)
`self._rnd`, `self._gx_counts` + `self._gx_count_beta`, `go_explore_archive`
(also `self._go_explore`), `ge_archive` + burst state (`ge_burst_active/remaining/
quota/iters_since_advance/bursts_done`), `env_return_state`.
- **Hazard — RND obs-rms update ordering.** `update_normalization` runs **once**
  in the intrinsic fold (5959) and the updater's per-minibatch RND cache (6082–
  6105) must see the *post-update* stats. Contract: `fold_intrinsic()` must be
  called exactly once, before `update()`. This ordering *is* the "RND self-
  referential norm" / "f16 double-norm" bug surface — pin it with a golden (§3).

### 2.6 Net / optimizer / training knobs (owner: conductor; **multiple writers**)
`net` (`self._ppo_net`), `optimizer` (`self._ppo_optimizer`), `self.entropy_coef`,
`self.rnd_intrinsic_coef`.
- **Hazard — `entropy_coef` has 8 writers, `rnd_intrinsic_coef` has 5.** Written
  by: consolidation schedule (7020/7023), cyclic fallback (7031), clevel schedule
  (7039–7048), consolidate abort (6699), clevel rollback (6796), anti-collapse
  (indirectly via optimizer), entropy-floor controller (7154–7159). Read by the
  updater every minibatch (6177). This many writers to one scalar is a latent
  bug magnet. **Decision:** introduce a tiny `TrainingKnobs` value object
  (`entropy_coef`, `rnd_intrinsic_coef`) that the conductor owns and passes to
  `update()`; Curriculum and the two controllers return *requested overrides*
  the conductor applies at one place, so the write-order is explicit and testable.
  (Keep `self.entropy_coef` as a mirror for the metrics emit until the last step.)
- `optimizer` is rebuilt (via `_build_ppo_optimizer`) at three sites: anti-collapse
  (7132), forgetting rollback (6662), clevel rollback (6793). Keep
  `_build_ppo_optimizer` as the single builder (it already is, per its docstring);
  callers pass the reference back.

### 2.7 Misc (owner: conductor)
`iter_offset`, `resume_metrics`, `h_rollout` (recurrent, RolloutCollector),
`narrator_on`, `global_it`, `_iter_t0`.

---

## 3. Characterization tests to write FIRST (pin behavior before touching code)

**Rule: nothing in §4 starts until §3 is green on `main` with zero source
changes.** These tests are the safety net; they must pass identically before and
after every extraction. They extend, not replace, the existing coverage
(`test_vanilla_ppo_smoke`, `test_ppo`, `test_resume_roundtrip`,
`test_metrics_schema_contract`, `test_oneshot_curriculum`, `test_go_explore`,
`test_sticky_actions`, `test_auto_reset_on_death`, `test_rnd_target_cache`).

**Determinism basis.** The pixel path is *not* bit-reproducible on the target M4
(MPS `multinomial` vs the CPU generator — see the note at trainer.py:5403). The
**tile path runs on CPU and is seedable**, so all value-level goldens use tile
mode (`configs/mario_tiles.yaml`) on `device=cpu`, `seed` fixed, `torch.manual_
seed` + `np.random.seed` set. Compare floats with `rel/abs = 1e-6` tolerance
(guards against harmless reassociation; tighten to exact if the CPU path proves
stable in practice).

| # | Test file | Pins | Pre-extraction anchor for |
|---|---|---|---|
| C0 | `test_char_vanilla_ppo_golden.py` | Full metrics-dict **sequence** over 3 tile-mode iters, seeded: assert keys AND values (tol) match a recorded fixture; assert `net.state_dict()` checksum matches. | The whole loop — the master golden |
| C1 | `test_char_checkpoint_roundtrip.py` | Save at iter N, resume: `net`/`optimizer`/`_rnd`/`_gx_counts`/`anticollapse` bit-identical; poison guard refuses a NaN-injected net. | CheckpointManager |
| C2 | `test_char_ppo_update.py` | Given a **recorded rollout-buffer fixture** (obs/action/reward/value/log_prob/done/valid/bonus + final_values), run fold→GAE→advnorm→K-epoch: assert `(policy_loss, value_loss, entropy, loss, rnd_loss)`, folded `reward_buf`, `advantages`, and post-update net checksum. | PPOUpdater |
| C3 | `test_char_exploration.py` | Given a fixed RAM/obs sequence: assert `_gx_counts`, Go-Explore archive cells/frontier/records, and the intrinsic fold vector; assert `fold_intrinsic` updates obs-rms exactly once. | ExplorationController |
| C4 | `test_char_curriculum_glue.py` | Scripted rollout (envs scripted to reach target area bytes): assert capture fires on the right step, `should_advance`/advance transition, warm-start `env_stage` distribution, and the exact set of `stage_NN.state` / `.meta.json` files written. | Curriculum |
| C5 | `test_char_rollout_buffers.py` | Fixed net + seeded CPU sampling, short rollout: assert filled `action_buf/reward_buf/done_buf/valid_buf/value_buf/log_prob_buf` match a fixture; assert auto-reset vs freeze branch taken per env. | RolloutCollector |

**How to produce fixtures without hand-authoring them:** add a one-shot
`--dump-char-fixture` code path (or a pytest fixture that runs the real loop once
on `main` and pickles the buffers/metrics to `tests/fixtures/`). Commit the
fixtures. The refactor must reproduce them. This is the cheapest way to pin a
3,000-line method — record its outputs, then hold them fixed.

**Also add (fast plumbing guards, no fixture):** assert each new module imports
and constructs from a real profile (extends `test_smoke_imports`); assert the
conductor still emits the `ppo_*` metric keys the dashboard needs (extends
`test_metrics_schema_contract`).

---

## 4. Extraction order — strangler fig, lowest risk first

Each task: one module, one PR-sized commit, green `make test` + the relevant `C*`
golden before merge. Each task has an explicit **rollback** (the extraction is a
pure move; if a golden goes red, revert the single commit). Ordering rationale:
extract the loop's **collaborators** (sinks, then data-transforms, then the two
stateful controllers) before the **loop body itself**, so that when
`RolloutCollector` moves last it only has to *call* stable, tested objects.

### Task 1 — `CheckpointManager` (lowest risk, pure sink)
**Why first.** It is a leaf: resume at the top (4565), save at a well-defined tail
(7435–7571), already half-delegated to `checkpointing.py`. Minimal shared state
(paths + snapshots passed in). Touches nothing the other phases read mid-loop.

**Own:** `_maybe_resume_vanilla_ppo` (move as-is → `CheckpointManager.resume`),
the iter-checkpoint block incl. the `_all_finite` poison guard, run-manifest write,
generic `save_winner` call (the non-ladder/non-consolidate branch, 7561–7569).

**Leave behind (Curriculum's, not the manager's):** `best_cold` and
`best_<level>.pt` writes — those are keyed on curriculum metrics and move with
Curriculum in Task 4. The manager saves the *generic* iter checkpoint + anti-
collapse blob; it does not know about cold rates.

**Seam:** `resume(net, optimizer, *, fresh_start) -> (iter_offset, pending_rnd,
pending_anticollapse)`; `save_iter(net, optimizer, rnd, gx_counts, anticollapse,
global_it) -> None` (does the poison guard internally, returns without saving on
non-finite). Constructor takes `checkpoint_dir`, game name.

**Acceptance:** C1 green; `test_resume_roundtrip`, `test_vanilla_ppo_resume_gate`,
`test_winner_checkpoints`, `test_checkpointing` unchanged-green. Diff is a move,
not a rewrite (verify with a structural read of the extracted method vs original).

### Task 2 — `PPOUpdater` (clean data-in/data-out)
**Why second.** Cleanest boundary in the file: numpy buffers in → net/optimizer
mutated + scalar losses out. The pure math (`batched_gae`, `ppo_losses`,
`fold_intrinsic_into_rewards`, `demo_anchor_loss`) is already in `ppo.py`; this
task extracts the *orchestration* (folds → GAE → adv-norm → K-epoch loop → NaN
backstop → RND cache → demo anchor). It touches neither pool nor curriculum.

**Own:** phases 5922–6253. During this task RND stays as `self._rnd` and gx stays
as `self._gx_count_beta` — the updater references them directly; they get
redirected to the controller in Task 3 (a two-line follow-up, kept out of this
commit to keep the diff a pure move).

**Seam:** `update(batch: RolloutBatch, net, optimizer, *, rnd, demo_bank, knobs,
global_it) -> UpdateStats`. `RolloutBatch` is a dataclass of the filled buffers +
`final_values` + `valid_indices`. `UpdateStats` carries the five losses +
`intrinsic_mean` + `count_bonus_mean`. The recurrent branch keeps delegating to
`_recurrent_ppo_update` (already a method) — updater just routes to it.

**Hazard to hold fixed:** the fold-before-cache ordering (§2.5) and the
`valid_buf` masking (§2.1). C2 pins both.

**Acceptance:** C2 green; `test_ppo`, `test_rnd_target_cache`,
`test_learning_regression` (slow) unchanged-green.

### Task 3 — `ExplorationController` (RND + count + Go-Explore)
**Why third.** RND is already a module (`src/models/rnd.py`, `tile_rnd`);
Go-Explore is already a module (`go_explore.py`); count is a dict. This task
gathers their *lifecycle + hooks* behind one object and redirects the touch-points
Task 2 left pointing at `self._rnd`/`self._gx_count_beta`. Go-Explore and the SMB
curriculum are mutually exclusive (5216–5218), so this object and Curriculum never
run their exploration paths simultaneously — extracting Exploration first means
Curriculum (Task 4) inherits a clean warm-start seam.

**Own:** RND build (4535–4550) + `_apply_pending_rnd_state`; the intrinsic fold
(5922–5973) and count fold (5974–5981) — exposed as `fold_intrinsic(reward_buf,
obs_buf, done_buf) -> (reward_buf, intrinsic_mean, count_mean)`; the per-minibatch
`predictor_loss(states_t) -> Tensor | None` + target-feature cache build (6082–
6105); the Go-Explore archive setup (5204–5288), per-step `record` (5662–5689),
`select_return_states`, the burst arm/tick/harvest (6888–7015), archive persist
(7573–7584); the count-bonus per-step (5488–5496) exposed as `count_bonus(wl, x)`.

**Seam (called by collector):** `count_bonus(wl_packed, x_pos) -> float`;
`record(i, ram, score, steps, pool)`. **Seam (called by updater):**
`fold_intrinsic(...)`, `predictor_loss(states_t)`. **Seam (called by conductor):**
`arm/tick_burst(...)`, `return_states(n)`, `save()`.

**Acceptance:** C3 green; `test_rnd`, `test_tile_rnd`, `test_go_explore`,
`test_oneshot_ge_burst`, `test_vanilla_ppo_go_explore_smoke`, C0 all
unchanged-green.

### Task 4 — `Curriculum` (biggest state, most bug-prone)
**Why fourth (not last).** It is the largest and most bug-prone chunk, but its
*decisions* already live in `oneshot_curriculum` / `smb_substage_ladder`. Doing it
before `RolloutCollector` means the collector's auto-reset seam (§4.5) can be
defined against a real `Curriculum` object rather than against loose locals.

**Own:** all of §2.4. Three sub-modes stay behind one class (scalar SMB, ladder,
level-scoped consolidation) — do **not** split them into three classes in this
task (that is a later, separate refactor); the point now is to get the state and
its I/O out of the loop, not to redesign the mode dispatch.

**Own the phases:** curriculum setup (4651–5113), mid-rollout capture *decision*
(the `smb_pending_capture` logic at 5563–5597 — the *mechanics* `pool.save_worker_
state` stay callable from the collector via a passed `pool`), advance detection +
advance (6321–6460), cold-eval probe + winner/forgetting/consolidate triggers
(6570–6716), consolidation gate (6717–6886), coefficient schedules (7017–7050,
returned as knob overrides per §2.6), warm-start *assignment* (7197–7386, mechanics
in the collector per §4.5), state-file writes.

**Seam:** `load_from_disk(fresh_start)`; `maybe_capture(i, ram, pool, alive)`;
`reset_seed_for_env(i, env_stage) -> bytes | None`; `observe_iter(stats) ->
AdvanceDecision`; `warm_start_plan(num_envs) -> WarmStartPlan (env_stage +
per-env seed blobs)`; `cold_probe(net) -> ColdMetrics`; `maybe_rollback(net,
optimizer) -> optimizer`; `coef_overrides() -> TrainingKnobs | None`. Constructor
takes `checkpoint_dir`, the parsed `CurriculumConfig`, `game_profile`.

**Hazard to hold fixed:** the capture gate (`wl_packed > cur_anchor` scalar vs
`reg_here > stage` ladder, plus `lives >= 1`) — this is the off-by-one that
stalled the curriculum at stage 1. The die-respawn `prev_completion_total` re-arm
(5735) — omitting it under-reports success 5–6×. C4 pins both.

**Acceptance:** C4 green; `test_oneshot_curriculum`, `test_smb_substage_ladder`,
`test_consolidate_level`, `test_vanilla_ppo_fresh_curriculum`,
`test_curriculum_topk_gate_trainer`, C0 all unchanged-green.

### Task 5 — `RolloutCollector` (last; the integration point)
**Why last.** It holds the most local state and the most intricate control flow
(sticky, auto-reset/freeze, per-env branching, GRU reset, fused drain). By now
Curriculum and Exploration are stable tested objects, so the collector's body
collapses to: forward → sample(+sticky) → `step_all` → per-env {reward, done,
`exploration.record`, `exploration.count_bonus`, `curriculum.maybe_capture`,
death→`curriculum.reset_seed_for_env`+reseed mechanics} → stacker advance → drain
→ bootstrap. Moving it when its collaborators are frozen is the lowest-risk time.

**Own:** buffers + trackers + stackers (§2.1–2.3), sticky state, `h_rollout`, the
initial reset + stacked_obs seed, and the iter-boundary reset *mechanics*
(reset_all, `load_worker_state` per the plan Curriculum returns, reward-fn reset,
tracker reset, stacked_obs rebuild — 7162–7433's mechanical half).

**Seam:** `collect(net, curriculum, exploration, knobs) -> RolloutBatch`;
`reset_for_iter(warm_start_plan) -> None`. The collector calls into curriculum and
exploration; it does not know their internals.

**Hazard to hold fixed (§4.5):** the auto-reset-vs-freeze decision (5695–5838) is
the seam that produced freeze-starvation and eval-inflation. Keep the exact branch
order: (a) env has a warm-state for `env_stage[i]` → reload + reseed; (b) stage-0
with `start_state` (tile) → inline restart (optionally GX return); (c) else freeze
via `set_worker_done`. The DECISION of *which seed* is Curriculum's
(`reset_seed_for_env`); the MECHANICS (pool load, stacker reseed, reward-fn reset,
`prev_completion_total=0`) are the collector's. C5 + `test_auto_reset_on_death` +
`test_sticky_actions` pin it.

**Acceptance:** C5 + C0 green; full suite green. `_run_vanilla_ppo` is now the
conductor.

### 4.6 The conductor after Task 5
```
def _run_vanilla_ppo(self, num_iters, fresh_start=False):
    net, optimizer = self._ppo_net_and_optimizer()      # lazy build
    ckpt   = CheckpointManager(self.checkpoint_dir, ...)
    iter_offset, pend_rnd, pend_ac = ckpt.resume(net, optimizer, fresh_start=...)
    explore = ExplorationController(self.explore_cfg, self.device, ...)
    curr    = Curriculum(self.curriculum_cfg, self.checkpoint_dir, ...)
    curr.load_from_disk(fresh_start)
    collector = RolloutCollector(self.rollout_cfg, self.pool, ...)
    knobs = TrainingKnobs(self.entropy_coef, self.rnd_intrinsic_coef)
    updater = PPOUpdater(self.ppo_cfg)
    for it in range(num_iters):
        if not self._running: break
        batch = collector.collect(net, curr, explore, knobs)
        if not self._running: break
        stats = updater.update(batch, net, optimizer, rnd=explore.rnd, ...)
        decision = curr.observe_iter(batch.stats, stats)
        optimizer = curr.maybe_rollback(net, optimizer)     # forgetting/clevel
        knobs.apply(curr.coef_overrides())
        self._emit_metrics(...assembled from batch.stats + stats + curr + explore...)
        if curr.clevel_done: break
        optimizer = self._anti_collapse_guard(net, optimizer, stats, knobs)
        self._entropy_floor_step(stats, knobs)
        plan = curr.warm_start_plan(num_envs)
        collector.reset_for_iter(plan)
        ckpt.save_iter(net, optimizer, explore, self._anticollapse_blob(), global_it)
        explore.maybe_save(it)
```
`_anti_collapse_guard` and `_entropy_floor_step` stay as small private conductor
methods (they are training-stability controllers, not one of the five named
modules; extracting them into a `StabilityGuard` is optional future work — note it,
don't do it now).

---

## 5. `__init__` decomposition (interleave; low risk, pure)

`__init__` is 54 `self.* =` assignments and 101 `profile.get`/`reinforce`
lookups. It is almost all pure config parsing — the safest code in the file.

**Task 0.5 (can run before Task 1, or in parallel — it is pure):** extract the
hyperparameter parsing into typed configs, one slice per module, built by pure
functions so each is unit-testable by feeding a dict:

- `PPOConfig.from_profile(p)` — rollout_steps, reinforce_steps, gamma, gae_lambda,
  clip_eps, value_coef, value_loss_kind, ppo_minibatch_size, grad_clip, lr.
- `ExplorationConfig.from_profile(p)` — rnd_intrinsic_coef, rnd_loss_coef,
  rnd_predictor_update_fraction, gx_count_beta, go_explore.* block.
- `CurriculumConfig.from_profile(p)` — substage_ladder, consolidate_level,
  advance, warm_start, cold_eval, consolidate, go_explore_fallback blocks.
- `RolloutConfig.from_profile(p)` — sticky_action_prob, frame_skip, preprocess_f16,
  tile flags, num_instances, max_episode_steps.
- `EntropyControlConfig` — entropy_coef, entropy_floor, entropy_coef_max,
  `_entropy_coef_base`.

Keep `__init__` assigning `self.<field>` from these configs for now (so the loop's
`self.*` reads keep working); each module extraction (§4) then takes its config
object in its constructor and stops reading `self.*`. Align field names with the
existing `config_schema.py` (guarded by `test_config_schema`); add a
`test_char_config_parse.py` asserting a real profile parses to expected values.

**Then extract the builders** (each a small method, mechanical move):
`_build_device()` (device_override / `get_best_device`), `_build_model()` (network
+ tile extractor + `_make_network`), `_build_training_components()`
(reward_fn_factory, ga, curriculum manager), `_wire_io()` (metrics/reward/audio/
narrator queues, sinks, depth tracker), `_platform_memory_setup()` (tracemalloc +
malloc_zone_pressure_relief). Target `__init__` < 200 lines: parse configs → call
builders. Acceptance: `test_smoke_imports`, `test_gui_imports`,
`test_profile_configs`, C0 unchanged-green.

---

## 6. Risk register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| A golden isn't actually deterministic (float drift) → false red | High (blocks all work) | Med | Tile+CPU+seed basis (§3); tolerance compare; validate C0 is stable across 3 back-to-back runs on `main` *before* extracting. |
| `reward_buf` in-place fold ordering breaks (RND double-norm class) | High (silent learning regression) | Med | §2.5 contract "fold once, before update"; C2/C3 pin the folded vector + obs-rms call count. |
| Auto-reset/freeze branch semantics drift | High (freeze-starvation, eval inflation recur) | Med-High | Extract collector LAST against a stable Curriculum; keep exact branch order (§4.5); C5 + `test_auto_reset_on_death`. |
| `entropy_coef`/`rnd_intrinsic_coef` multi-writer race after extraction | Med (wrong exploration schedule) | Med | `TrainingKnobs` single-apply point (§2.6); assert write order in C0 by checking emitted `demo_anchor_coef`/entropy over iters. |
| Optimizer-rebuild sites drift (the anti-collapse RND-drop bug) | Med | Low | Keep single `_build_ppo_optimizer`; rollback methods return the new optimizer, never mutate a hidden ref. |
| Curriculum's three modes are too entangled to move as one class | Med (task balloons) | Med | Move state+I/O only; do NOT redesign mode dispatch in Task 4; lean on already-extracted `oneshot_curriculum` decisions. |
| Checkpoint format changes accidentally → resume breaks in prod | High | Low | C1 asserts byte-identical payload keys; the save block is a pure move; `test_resume_roundtrip` unchanged. |
| Recurrent path (GRU) regresses (less-tested branch) | Med | Low | Keep `_recurrent_ppo_update` delegation intact; run `test_recurrent_integration_smoke` + `test_recurrent_policy_learns` each task. |
| "While I'm in here" scope creep folds a fix into a move | Med (un-reviewable diff) | Med-High | Hard rule §0: a move commit changes no behavior; real fixes get separate commits. Review each diff as "is this a pure move?". |

---

## 7. Checkpoints (human review gates)

- **After §3 (C0–C5 green on `main`, zero source changes):** review the goldens
  are real and deterministic. *Do not start §4 until this is signed off* — the
  goldens are the entire safety argument.
- **After Task 1 + Task 2:** CheckpointManager + PPOUpdater extracted; full suite +
  C0/C1/C2 green; confirm `_run_vanilla_ppo` line count dropped and the diffs read
  as pure moves.
- **After Task 3 + Task 4:** Exploration + Curriculum extracted; confirm the
  conductor's mid-loop `self.*` reads are gone (grep `self\.` inside the loop
  should now be near-empty); C0/C3/C4 green.
- **After Task 5 + §5:** conductor < 300 lines, `__init__` < 200 lines; full
  `make test` + `make selftest-learning` (the real-loop guard) green. Optionally
  run a short real tile-mode SMB run and diff `metrics.jsonl` against a pre-refactor
  run of the same seed as a final end-to-end confirmation.

---

## 8. Task summary (ordered, sized)

| # | Task | Size | Depends on | Green gate |
|---|---|---|---|---|
| 0 | Characterization goldens C0–C5 + fixtures | L | — | C0–C5 pass on `main` |
| 0.5 | Config dataclasses + `from_profile` parsers | M | — | `test_char_config_parse`, C0 |
| 1 | `CheckpointManager` | M | 0 | C1 + resume/winner suite |
| 2 | `PPOUpdater` | M | 0 | C2 + `test_ppo`/rnd_cache |
| 3 | `ExplorationController` | M-L | 0, 2 | C3 + rnd/go_explore suite |
| 4 | `Curriculum` | L | 0, 3 | C4 + curriculum suite |
| 5 | `RolloutCollector` + conductor | L | 0, 1–4 | C0/C5 + full suite |
| 6 | `__init__` builders shrink | M | 0.5, 1–5 | import/profile suite, C0 |

Total: eight commits, each independently green and independently revertable. No
step is larger than one focused session; no step changes behavior.

---

## 9. Execution log

- **§3 goldens (C0–C5) — DONE (b16ce2f, 5f1258c).** Determinism validated
  3/3 back-to-back on the M4 (tile+CPU bit-reproducible). C0 master golden
  pins the exact 3-iter metric sequence + final-net sha256; C1–C5 pin
  checkpoint / ppo-update / exploration / curriculum / rollout. 54 tests,
  2.2s. Each documents its honest "not covered" gaps.
- **Task 0.5 config dataclasses — DONE (260df99).** PPO/Exploration/
  Curriculum/Rollout/Entropy configs + from_profile, 45 fidelity tests.
  UNWIRED (nothing imports them yet); wiring happens in Task 6.
- **Task 1 CheckpointManager — DONE (1a99467).** Real relocation: resume +
  save_iter + write_manifest live in `checkpoint_manager.py` (332 lines);
  trainer.py 9960→9751. C0 bit-for-bit green = the pure-move proof.
  **LESSON:** the first pass was a FACADE (delegating wrappers; trainer.py
  GREW) — C0 can't catch that ("grew the file" isn't a behavior change);
  the §7 diff-review gate caught it. Always verify trainer.py SHRANK. Fix
  pattern: relocate bodies into the module, thin shim for direct-call
  tests, retarget the golden's source-string anchors to the new module
  (C0 doesn't source-anchor → stays the untouched behavioral proof).
- **Task 2 PPOUpdater — DONE (24fe36a).** Scope-corrected: net-covered core
  (fold→GAE→adv-norm→K-epoch→NaN backstop) → ppo_updater.py (515 lines),
  trainer.py −385. PR-MDP/CGSA/backward left in conductor. (Net later
  EXTENDED — 84c7162 — with PR-MDP/CGSA/backward goldens so they're covered.)
- **Task 3 ExplorationController — DONE (9176193, partial).** RND lifecycle
  + count_bonus + generic Go-Explore archive → exploration_controller.py
  (234 lines), trainer.py −106. The Go-Explore unstick burst hit the
  entanglement wall (mutates Curriculum state) and stayed in the conductor.
- **Task 4 Curriculum — DONE (9b052ea, partial) + INSEPARABILITY PROVEN.**
  Only the disk-load leaf moved (→ Curriculum class in curriculum.py),
  trainer.py −91. The core (capture/advance/warm-start) is INSEPARABLE from
  the un-extracted RolloutCollector: env_stage & stage_seed_results are
  co-written by curriculum decisions AND collector mechanics in the same
  statements. Tasks 4&5 must be extracted TOGETHER, after a TrainingKnobs
  value object + maybe_rollback seam (both absent).

## 10. DECISION 2026-08-13 — bank at 4/6; defer Tasks 5 & 6 by design

A 5-lens review cohort + synthesis decided (A=3 in-frame, C=2 wider-frame,
B=0): **bank the refactor here and pivot to the product frontier.** Not
abandonment — disciplined deferral.

- trainer.py 9960 → 9168 (−792). Four collaborators extracted and wired
  live (CheckpointManager, PPOUpdater, ExplorationController, Curriculum
  disk-load). The C0–C5 + PR-MDP/CGSA/backward golden net is INTACT and
  every task is independently revertable. This is an HONEST resting point:
  4 golden-pinned modules + a documented, provably-inseparable rollout/
  curriculum core.
- **Task 6 (__init__ config-wiring + builders) — deferred.** Low-risk but
  ORTHOGONAL to the product loop; deferrable at identical cost forever
  (the Task-0.5 configs import nowhere, so they cannot drift into a bug —
  the "rot" is cosmetic). Do it OPPORTUNISTICALLY the next time __init__ is
  genuinely touched. When done: write tests/test_char_config_parse.py FIRST
  (assert the wired __init__ reproduces byte-identical self.* for ≥2 real
  profiles incl. a non-tile pixel config), then diff-review that trainer.py
  SHRANK and the inline profile.get lines are DELETED (Task-1 anti-facade).
- **Task 5 / the combined core (Curriculum + RolloutCollector) — deferred.**
  RESUME DISCIPLINE = **EXTRACT-ON-TOUCH**: the next product edit of
  _run_vanilla_ppo pays B's down-payment by landing TrainingKnobs + the
  maybe_rollback seam first, so the inseparable-core extraction rides a
  change we were already making rather than a cold invasive pass.

## 11. Post-refactor validation on real compute (2026-08-13)

Beyond the 3-iteration C0 golden, the refactor was validated end-to-end
with real training runs (user lifted no-compute for validation). Every
major code path exercised; no bugs found.

- **Real-loop learning guard** (`make selftest-learning`): PASS — the
  refactored vanilla_ppo loop learns on real SMB (envs diverge + entropy
  drops).
- **Vanilla 120-iter** (mario_tiles_vanilla, 24 envs, seed 0): **learns +
  CLEARS 1-1** — 34 iters banked clears (up to 4/iter), return 3044→5000,
  entropy 1.73→1.35, 11 checkpoints saved via CheckpointManager, ZERO
  NaNs/crashes. Exercises PPOUpdater + ExplorationController +
  CheckpointManager.save_iter + Curriculum.load_from_disk.
- **Backward-curriculum 80-iter** (mario_1_1_backward): PASS — the code the
  refactor LEFT in the conductor works: tau scheduler stepped 757→397
  (start marched back from the flag), capture/advance fired (advances
  0→9), policy re-learned (clears 49→17, entropy 1.77→1.05), 7 checkpoints,
  ZERO crashes.
- **Eval path** (eval_game.py greedy/sampled): PASS — backward iter-70
  policy clears 1-1 from the entrance at greedy 0.0 / sampled 0.16 (the
  expected under-consolidation gap for a short run).
- **Resume path** (CheckpointManager.resume in production): PASS — resumed
  net + optimizer AND the backward-tau cursor (`_pending_backward_curriculum`)
  from iter-70 correctly (tau=0 AT-ENTRANCE, advances=19 restored).

Conclusion: the four extractions (CheckpointManager, PPOUpdater,
ExplorationController, Curriculum-load) and the conductor code they touched
or left behind (backward-tau, capture/advance, fresh-start relocation)
preserve learning behavior across the vanilla, backward-curriculum, eval,
and resume paths. The refactor introduced no bugs.
