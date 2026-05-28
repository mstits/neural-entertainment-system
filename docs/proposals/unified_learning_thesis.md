# Unified Learning Thesis — beating six NES games with one framework

Authored 2026-05-16. Source-of-truth for the project's learning-side
work going forward. Every training-stack commit should reference a
phase / milestone defined here, or it's noise.

The question answered here: **given that the emulator stack is
largely done and the training stack is a fragmented pile of
techniques accumulated over months of reactive engineering, what's
the minimum coherent program that ships a single training framework
which, configured per game, beats all six NES games already
profiled in this repo — and does so within realistic Apple-Silicon
compute (weekend-per-game training time)?**

---

## 1. Thesis statement

A single training framework — `vanilla_ppo` evolved with intrinsic
exploration, recurrence, level conditioning, and (for open-worlds)
a hierarchical subgoal layer — beats:

| Game | Class | Definition of "beaten" |
|---|---|---|
| Super Mario Bros. | Linear platformer | Cleared world 8-4 (Bowser) end-to-end |
| Contra | Linear action | Cleared stage 8 (alien lair) end-to-end |
| Mega Man | Stage-select platformer | Beat all 6 robot masters + Wily's castle |
| Castlevania | Linear platformer | Cleared block 18 (Dracula) end-to-end |
| The Legend of Zelda | Open world | All 8 dungeons + Ganon defeated |
| Metroid | Metroidvania | Mother Brain defeated |

Each game runs from cold-boot (or a defined start state) through to
its end-credit / victory state. No human-in-the-loop. Configuration
per game lives entirely in `configs/<game>.yaml` + the Rust reward
function. The algorithm itself stays uniform.

**Compute budget**: roughly a weekend of M4 Max wall-clock per game
at 60 parallel envs × rollout_steps=1024 × frame_skip=4. ~500-1000
PPO iterations per game depending on level count and difficulty.

**Project artifact**: working codebase. No paper, no broadcast, no
GitHub Actions, no hosted demo. README + per-game demo command +
final highlight reels per game.

---

## 2. Game class analysis

### Class A — linear progression
Four games: SMB, Contra, Mega Man, Castlevania.

Common structure: discrete sequence of stages, each with a defined
"clear" event (touch flag, reach door, defeat boss). Per-stage
obstacle vocabulary differs enough that a single MLP trained on
stage N typically can't pass stage N+1 without adaptation. Forward
progress correlates with goal at most stages but breaks down at
"stop and time the platform" moments (SMB 1-3 lifts, Contra base
turrets, Castlevania medusa heads).

Literature solutions that apply directly:
- **PPO** (Schulman 2017) — base algorithm
- **RND** (Burda 2018) — intrinsic reward solves "policy converges
  on safe-failing strategy because beyond-cliff is unrewarded
  unknown territory" exactly as observed in our SMB 1-3 plateau
- **Curriculum learning** with save-state anchors (Vinyals 2019
  AlphaStar; already shipped here as `smb_curriculum/stage_NN.state`)
- **Auto-reset on death within rollout** (standard PPO baseline
  practice from OpenAI baselines + SB3; shipped 2026-05-16)
- **Recurrent policy** (Hausknecht 2015) — LSTM cell on top of MLP
  trunk gives long-horizon credit assignment + level identity
  carries across timesteps
- **Level conditioning** — one-hot of current curriculum stage as
  additional policy input. Trivial. Standard multi-task RL pattern.

Mega Man is the odd one out in this class: it has a STAGE SELECT
screen where the agent must choose which of 6 robot masters to
attack. That's a meta-decision on top of the per-stage policy. Two
options:
- Treat each robot master's stage as a curriculum stage, force
  ordering (canonical optimal: Cut → Elec → Ice → Fire → Bomb → Guts).
  Simplifies to standard linear-game treatment.
- Train a stage-select meta-policy that learns weapon trade-offs.
  Open research. Defer until linear baseline works.

### Class B — open world
Two games: Zelda, Metroid.

Common structure: persistent world map, item-gated progress,
subgoals (dungeons, items, bosses), no per-level "clear" event.
Forward progress is meaningless as a single scalar — progress is the
union of (map cells discovered) × (items acquired) × (bosses
defeated) × (dungeons cleared).

Literature solutions:
- **Hierarchical RL** — high-level policy picks subgoals (`reach
  dungeon 1`, `acquire boomerang`, `defeat Ganon`), low-level
  policies execute those subgoals as primitive skills (`move to
  tile (x,y)`, `kill enemy in current screen`, `fire arrow`).
  Reference architectures: FeUdal Networks (Vezhnevets 2017),
  Option-Critic (Bacon 2017), HIRO (Nachum 2018).
- **Intrinsic motivation** — RND or count-based exploration drives
  discovery of new map cells in absence of extrinsic reward.
- **Memory** — LSTM/GRU/Transformer required for "remember which
  dungeon entries you've found", "remember you have the candle".
- **Hindsight Experience Replay** (Andrychowicz 2017) — turns
  failed trajectories into successful ones for sub-goals you
  accidentally hit.
- **World models** (Hafner DreamerV3 2023) — latent imagination
  reduces sample cost for sparse-reward navigation.

Subgoal definitions per game live in `configs/<game>.yaml` and the
reward function. The framework doesn't hardcode "go to dungeon 3";
it provides the subgoal layer and the game profile specifies what
subgoals exist.

---

## 3. Unified architecture

### Components every game uses

```
┌─ observation ─┐    ┌─ encoder ─────┐    ┌─ backbone ─┐    ┌─ heads ────┐
│ frames / RAM  │ -> │ tile-decode   │ -> │ MLP        │ -> │ actor      │
│ stack         │    │   or pixel CNN│    │ (+ LSTM)   │    │ critic     │
│               │    │ + level 1-hot │    │            │    │ (+ rnd-v)  │
└───────────────┘    └───────────────┘    └────────────┘    └────────────┘
                                                   |
                                  optional ┌───────┴───────┐
                                           │ world model    │
                                           │ (DreamerV3)    │
                                           └────────────────┘
```

- **Encoder**: per-game observation encoder. SMB uses
  `smb_tiles` (13×13 RAM-decoded tile grid). Other games use either
  game-specific RAM decoders (Contra has known sprite-table
  layouts) or pixel CNNs (Nature DQN / IMPALA, already in
  `src/models/policy_network.py`).
- **Backbone**: shared MLP trunk; LSTM cell optional via
  `network.recurrent: true` in profile. LSTM hidden state is reset
  on episode boundary (done=true) and on curriculum stage advance.
- **Heads**: actor (logits over action_space), critic (V), optional
  intrinsic-value head when RND is enabled.
- **World model (opt-in)**: DreamerV3-style latent dynamics model.
  Trained alongside the actor-critic from the rollout buffer.
  Provides imagined rollouts for sample-efficient updates. Default
  off; enable for hard-exploration games.

### Class-A extension — level conditioning + auto-reset

For linear games, the framework adds:
- One-hot of `curriculum_stage` (max 32 to cover SMB) prepended to
  the observation feature vector. Network learns level-specific
  action distributions without inferring level from pixels.
- Auto-reset on death within rollout (shipped 2026-05-16). Each env
  attempts the current stage 5-10 times per rollout instead of
  freezing after first death.
- Per-stage save-state warm-start (shipped 2026-05-14). All envs
  start each iter at the current stage's anchor state. Curriculum
  advances on first detection of post-anchor area-byte.

### Class-B extension — subgoal hierarchy

For open-world games, the framework adds a **meta-policy** layer:
- Subgoal vocabulary defined in profile: `["go_to_dungeon_1",
  "acquire_sword", "find_boomerang", ...]`. Encoded as integers.
- Meta-policy: input = current observation + flags for "subgoals
  completed so far"; output = next subgoal to pursue. Trained with
  PPO on a longer timescale than the primitive policy.
- Primitive policy: input = observation + current subgoal one-hot;
  output = action. Trained on intrinsic + subgoal-completion reward.
- Subgoal-completion detection lives in the Rust reward function
  (e.g., `ram[$0648..$0656]` for Zelda's inventory bitfield).

This is structurally Option-Critic / FeUdal-style. Implementation
details (single-network vs two-network, gradient flow between
levels) are an open design question to be resolved in Phase 3.

### Optimization — PPO + RND

Total loss: `L = L_policy + value_coef * L_value + intrinsic_coef *
L_rnd - entropy_coef * H`.

PPO hyperparameters per game profile. RND coefficient defaults to
0 (off); enabled per profile for hard-exploration games. Entropy
coefficient stays at canonical 0.01 by default; per profile may
override for known-stuck levels.

---

## 4. Per-game profile spec

Every game lives in `configs/<game>.yaml`. Schema:

```yaml
game: super_mario_bros           # logical name
class: linear_platformer          # or open_world, stage_select
rom: roms/Super Mario Bros. (World).nes
expected_md5: 8e3630186e35d477231bf8fd50e54cdd

emulation:
  frame_skip: 4
  start_state_path: roms/Super Mario Bros. (World)_start.state.bin
  num_envs: 60

action_space:
  - []                             # NOOP
  - ["right"]
  - ["right", "A"]
  - ["right", "B"]
  - ["right", "A", "B"]
  - ["A"]
  - ["left"]

observation:
  encoder: smb_tiles               # or pixel_dqn, pixel_impala, ram_zelda, etc.
  stack_size: 4

reward:
  type: mario                      # dispatches to MarioReward in nes_core::rewards
  weights:
    forward_progress: 1.0
    completion_bonus: 50.0
    death_penalty: -15.0
    time_penalty: 0.0
    score_delta: 0.025
    checkpoint_scale: 0.0          # canonical: no shaping
    # all other shaping signals 0

learning:
  algorithm: vanilla_ppo            # only supported value for now
  network:
    backbone: mlp
    hidden_dims: [128, 64]
    recurrent: true                 # LSTM on the trunk
    level_conditioned: true         # one-hot of curriculum_stage
  exploration:
    rnd_enabled: true
    rnd_coef: 0.1
    entropy_coef: 0.01
  curriculum:
    enabled: true
    advance_gate: byte_gt_anchor   # first env past current anchor
    state_dir: checkpoints/smb_curriculum
  ppo:
    lr: 3.0e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    value_coef: 0.5
    grad_clip: 0.5
    rollout_steps: 1024
    ppo_epochs: 10
    minibatch_size: 256

# Class-B only:
# subgoals:
#   - id: acquire_sword
#     detector: { type: ram_flag, addr: 0x0657, mask: 0x01 }
#   - id: enter_dungeon_1
#     detector: { type: map_tile, world: overworld, tile: [7, 7] }
#   - ...

success_criteria:
  type: stage_completion           # or boss_defeated, inventory_complete
  target: world_8_castle_clear     # game profile defines the predicate
```

Per-game extensions go in the profile, never in the trainer code.
The trainer reads the profile and dispatches. Adding a 7th NES
game = a new profile + a new Rust reward function + maybe a new
observation encoder. No trainer changes.

---

## 5. Infrastructure gap analysis

### Present and working
- `nes_core` Rust emulator (99.9% library compat, byte-exact)
- Per-game Rust reward functions (6 games)
- `smb_tiles` observation encoder
- Vanilla PPO trainer (`_run_vanilla_ppo` in `src/training/trainer.py`)
- Curriculum save-state system + sidecar anchor JSON
- Auto-reset on death within rollout
- Pool API: `save_worker_state`, `load_worker_state`, `step_all`,
  `reset_all`, `set_worker_done`
- GUI dashboard with metrics, env grid, replay infrastructure
- Mesen oracle parity harness
- Pixel encoders: Nature DQN, IMPALA (in `src/models/policy_network.py`)
- Auto-resume of vanilla_ppo from highest-iter checkpoint

### Scaffolded but not integrated
- **RND module** — exists in `src/training/`, not wired into the
  vanilla_ppo loss. Needs: intrinsic-value head on the network,
  RND-prediction loss term, intrinsic-reward addition to GAE input,
  `rnd_coef` knob in profile. **Estimated**: 1-2 days.
- **DreamerV3** — scaffold exists with Gaussian latents. Real
  DreamerV3 uses categorical latents. Needs: rewrite latent
  distribution, train world model from rollout buffer, hook
  imagined rollouts into actor update. **Estimated**: 1-2 weeks.
- **`TileRecurrentPolicyNetwork`** — already exists in
  `src/models/tile_policy.py`. Needs: wire `network.recurrent: true`
  through the trainer's PPO update (LSTM hidden state must persist
  per-env across rollout steps and reset on done). **Estimated**:
  2-3 days.

### Missing entirely
- **Level conditioning input** — straightforward add: one-hot vec
  appended to encoded obs. ~50 LOC. **Estimated**: half a day.
- **Per-game checkpoint dirs** — currently all checkpoints share
  `checkpoints/`. Multi-game training needs
  `checkpoints/<game_name>/`. **Estimated**: 1 day refactor.
- **Per-game observation encoders for Contra/MM/Castlevania/Zelda/
  Metroid** — Contra and Castlevania likely work with pixel
  encoders; Zelda needs RAM-projection (overworld map, inventory).
  **Estimated**: 1-2 weeks total.
- **Hierarchical RL infrastructure** — meta-policy, subgoal
  vocabulary, subgoal-completion detectors. Greenfield. **Estimated**:
  2-3 weeks.
- **Cross-game training launcher** — `make train GAME=zelda`. Per-
  game eval suite (automated game-completion checker). **Estimated**:
  3-4 days.
- **Multi-game eval scoreboard** — a single dashboard view that
  shows progress on all 6 games. **Estimated**: 2-3 days.

---

## 6. Phased plan

### Phase 0 — Foundation (~1 week)
- Per-game checkpoint directory restructure
- Profile loader pattern (extends existing)
- `make train GAME=<name>` single-command training launcher
- Per-game eval suite (each game has a Python script that loads a
  checkpoint, runs N episodes, reports clear rate + furthest stage
  reached)
- Multi-game eval scoreboard view (existing GUI dashboard
  extension)
- **Acceptance**: from a clean shell, `make train GAME=mario` and
  `make train GAME=contra` both launch successfully and write to
  separate checkpoint dirs. `make eval GAME=mario` reports a clear
  rate for the latest checkpoint.

### Phase 1 — Linear-game unified baseline (~1-2 weeks)
- Wire RND into vanilla_ppo's loss (intrinsic value head + RND
  prediction loss + intrinsic reward in GAE)
- Wire LSTM into the policy network (per-env hidden state, reset
  on done)
- Wire level-conditioning input (one-hot curriculum_stage)
- Finish DreamerV3 (categorical latents) — defer if Phase 1 SMB
  result is good without it
- **Acceptance**: SMB cleared world 1 entirely (1-1 through 1-4)
  within ~weekend of training, using the unified baseline. Specifically:
  curriculum advances through all four stages of world 1, including
  the 1-4 castle Bowser fight (or warp to world 2 via 1-2 pipe).

### Phase 2 — Apply baseline to other linear games (~1 week + training)
- Per-game observation encoders for Contra, MM, Castlevania
- Per-game curriculum anchor schemas (stage byte addresses
  differ — Contra uses `$0064`, MM uses scene IDs, etc.)
- Per-game success criteria
- **Acceptance**: Contra cleared through stage 4 (jungle base),
  Mega Man cleared at least 2 robot masters from cold start,
  Castlevania cleared blocks 1-2 (Frankenstein boss).

### Phase 3 — Hierarchical RL infrastructure (~2-3 weeks)
- Subgoal vocabulary spec in profile YAML
- Subgoal-completion detector framework in Rust (extends reward fn
  with `subgoal_satisfied(ram, subgoal_id) -> bool`)
- Meta-policy network (separate from primitive)
- Two-tier training loop (primitive trains on subgoal completion,
  meta trains on game-progress reward)
- **Acceptance**: Zelda's first dungeon cleared from cold start.
  Specifically: agent acquires wooden sword from cave, finds
  dungeon 1 entrance, clears all rooms, defeats Aquamentus,
  picks up triforce shard 1.

### Phase 4 — Open-world game completion (~1-2 weeks + training)
- Memory module (LSTM or transformer for map-history)
- Multi-objective reward calibration for Zelda + Metroid
- Subgoal hierarchy tuning (number of subgoals, ordering, optional
  vs required)
- **Acceptance**: Zelda beaten (Ganon defeated, end credits seen),
  Metroid beaten (Mother Brain defeated).

### Phase 5 — Final demonstration (~1 week)
- Six clean-checkpoint training runs (one per game) producing
  end-to-end game completions
- Per-game highlight-reel rendering (pull best replay from each
  training run, render to MP4)
- README rewrite (replaces the current README) — top-level demo
  command, results, screenshots, per-game training time
- **Acceptance**: `make demo GAME=<any>` plays back a recorded
  completion of that game. README clearly shows all six games
  beaten.

Total: ~7-10 weeks of focused engineering + ~6 weekends of training
compute. Some phases parallelize (Phase 2 game-specific work can
overlap with Phase 3 hierarchical infrastructure if labor permits).

---

## 7. Anti-goals

Explicitly **not** doing, even if it would help on a particular
metric:

- **Reward shaping for SMB.** Canonical recipe stays as configured
  (`checkpoint_scale: 0.0`). If PPO can't solve a level with the
  canonical signal + RND + LSTM + level conditioning, the right
  answer is more capacity or hierarchical structure, not more
  reward signals. Per-game profile may have light shaping if the
  game's structure demands it (Contra's score is naturally a
  per-frame signal), but never as a substitute for missing
  algorithmic capacity.
- **Distributed training.** Single M4 Max. No multi-machine setup,
  no cloud, no GPU clusters.
- **GitHub Actions / CI.** Local-first. `make test`, `make
  selftest`, `make check` are the development loop.
- **Broadcast / streaming layer.** GUI exists for the developer to
  watch training. Recording for highlight reels is done via OBS
  externally.
- **Web service / hosted demo.** Local executable + working
  checkpoints. Nothing more.
- **AI attribution.** No commits, PRs, code comments, docs,
  README, or any other artifact attributing this work to AI, LLM,
  Claude, Anthropic, etc.
- **A generic-purpose RL library.** This is a specific project
  with a specific thesis (beat 6 NES games). Code is organized for
  that. If others want a general-purpose RL library, they should
  use SB3 or RLlib.
- **Multi-game-at-once training.** Each game trains independently.
  No multi-task cross-game distillation. The shared element is
  the framework + algorithm, not a single shared policy.
- **GA mode as a path to thesis completion.** GA stays in the
  codebase for historical / experimental reasons but the thesis
  is delivered via vanilla_ppo + extensions. If a particular level
  proves genuinely intractable for the unified baseline, GA may be
  evaluated as a fallback per-game — but only after Phases 1-4 are
  shipped and shown to not work for that level.

---

## 8. Decision log

- **Why one framework instead of per-game-optimal**: ease of
  iteration. A bug fix in PPO+RND benefits all 6 games. A
  per-game RL stack would 6x the maintenance surface.
- **Why no reward shaping**: canonical SMB-PPO recipe rejected
  shaping; we're staying compatible. The literature has clearer
  solutions for sparse-reward hard-exploration (RND, curiosity,
  hierarchy) that don't require per-level hand-tuning.
- **Why hierarchical RL only for Class B**: linear games can
  plausibly be solved with single-policy + good exploration +
  curriculum. Open-worlds structurally need the hierarchy because
  there's no single forward axis.
- **Why DreamerV3 is opt-in not default**: world model adds
  significant code + compute. Linear-game baseline should clear
  Class A games without it. Reserve DreamerV3 for if it's needed
  for Class B (open-worlds) or specific hard linear levels.
- **Why level conditioning instead of bigger network**: cheaper.
  If 14k MLP + level one-hot can specialize per level, we don't
  need 100k. Empirical test pending in Phase 1.

---

## 9. Open questions

These need resolution by Phase 3 at the latest:

1. **Hierarchical training stability**: how do meta-policy and
   primitive policy interact gradient-wise? Option-critic does
   joint gradient flow; FeUdal does goal-conditioned-only flow.
   Choice affects implementation complexity significantly.
2. **Subgoal granularity**: in Zelda, is "enter dungeon 3" one
   subgoal, or is it "find dungeon 3 entrance" + "navigate to
   dungeon 3 entrance" + "enter"? Granularity trades primitive-
   policy difficulty against meta-policy decision cadence.
3. **Memory architecture for Zelda map**: LSTM hidden state is
   not enough to remember 128 overworld tiles. May need explicit
   map tensor with learned attention. Defer until Phase 3.
4. **Mega Man stage select**: train it as a fixed-ordering
   curriculum (canonical optimal route) or as a meta-decision
   alongside the linear-game framework? Meta-decision is more
   honest to the game but more research.
