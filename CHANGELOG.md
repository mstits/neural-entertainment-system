# Changelog — Neural Entertainment System (NES)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

In this changelog, **NES** refers to this project (Neural Entertainment
System). The original Nintendo Entertainment System hardware is named in full.

## [Unreleased]

### Performance (2026-07-14 integration pass)

Measure-first campaign over the emulator core, trainer, and PGO
pipeline: a 6-lane profiling sweep, then 5 implementation lanes, each
adversarially reviewed, with a whole-tree composition review before
merge. End-to-end on fresh-PGO wheels (same machine, same day,
quiet): **Zelda 16-worker pool_step 16.23 → 14.17 ms (−12.7%),
869 → 991 samples/s (+14.0%)**; single-env NES fps: Punch-Out +12.0%
(roughly half of it from the PGO-workload fix below), Contra +13.1%,
Gradius +11.5%, SMB +10.1%, Zelda +10.9%; 60-env tile mode
~11.2–12.3k env-steps/s (no regression, modest gain). Per-lane shares
come from lane-level A/Bs — the end-to-end delta also includes PGO
whole-program relayout between the two profiles, so it is reported as
the integrated campaign, not summed per-lane credits. Fidelity
unchanged: 748 pytest + 146 parity tapes + learning guard + 262 cargo
tests + ASM/interpreter lockstep soaks all green.

- **PPU idle-HBlank early-return under skip_render** (all mappers,
  default on): visible-scanline dots 258–320 that provably do no
  observable work collapse to a counter increment. ~5% single-env.
  The BG-pixel-pipeline elision variant ships byte-exact but OFF
  (`ppu_skip_bg` feature): it regresses MMC2/MMC3 via hot-loop code
  layout and stays parked pending PGO re-evaluation.
- **MMC1 flat CHR window** served to the PPU via a cached pointer
  (~4.6% on MMC1 games; CHR-RAM writes and bank switches rebuild the
  window in place). The matching PRG flat-window read path is ~0%
  under the production ASM CPU (which already fetches through that
  window) and pays only on interpreter builds — kept for that case.
- **APU muted-worker channel-timer skip**: pulse/triangle/noise
  timers, envelopes, and sweeps pause on silent training workers;
  DMC, frame counter, length counters, and $4015/IRQ semantics stay
  fully live (byte-identical over 11.5M-checkpoint lockstep).
- **CPU bulk=1 fixed-overhead trim** (ASM builds): the NMI-predict
  query is skipped where its result is provably discarded; budget
  lookup cached; opcode-table install hoisted. ~4% single-env on
  bulk=1 mappers (Zelda).
- **Action sampling moved off MPS**: `torch.multinomial` on MPS
  decomposes into ~10 serial kernels (~0.83 ms per collection step —
  more than the CNN forward itself); sampling now runs on the
  CPU-moved logits, replacing (not adding to) the existing per-step
  device sync. Note: seeded runs draw from the CPU generator now, so
  they no longer bit-reproduce pre-change trajectories. Fused Adam on
  cpu-device optimizers (tile update ~2% faster).
- **PGO stage-2 workload now covers MMC2** (Punch-Out added to
  Zelda+Contra): the old profile compiled the MMC2 CHR-latch path
  cold, costing Punch-Out ~6% — recovered with the other games
  neutral-to-positive and the Zelda training denominator unregressed.
- Debunked this campaign (measured, do not re-chase): further ASM
  opcode porting (coverage is already 98–99% of cycles with zero
  interpreter fallbacks), bulk-cycle raises at 16 workers,
  torch.compile/fp16/channels_last/MLX/ANE for the collection
  forward, allocator and codegen-flag changes, GIL-side frame-build
  copies (already zero-copy). The next structural headline is
  event-driven PPU catch-up (>9% of pool_step ceiling, high risk).

### Fixed (2026-07-12 validation pass)

Post-perf-day validation of the spectator/audio recipe across Zelda,
Contra, and SMB surfaced seven issues (full evidence in
`reports/validation_2026-07-12.md`); all fixed same day. Gates: 735
fast tests + 146 parity tapes + the real-loop learning guard.

- **`zelda_gui_tuned.yaml` is now a complete, bootable profile** (it
  lacked `action_space`/`name`/`frame_skip` and had never booted) with
  `frame_skip: 4` for 30 fps spectator tiles at 2× and its own
  checkpoint subtree. A new config-lint test asserts every GUI-offered
  profile meets the Trainer boot contract.
- **Trainer construction crashes now surface in the GUI** instead of
  dying silently in the training thread ("Starting training…" forever);
  the audio-mixer button reports when no trainer is running.
- **Done workers no longer black out spectator tiles**: the adapter
  serves each done worker's last live frame (Rust still skips all
  emulation + frame copies for them). Re-validated live: 60/60 Contra
  tiles (was 3/60).
- **The GUI Resume checkbox now controls vanilla_ppo auto-resume**
  (it silently resumed regardless; `train_game.py --no-resume` shares
  the same gate), and resuming runs announce the source checkpoint.
- **Mario/Contra level clears now record as curriculum successes even
  when the agent dies later in the episode** (durable clear latch; the
  old predicate lost every 1-1 clear to the level-change re-arm +
  death negation, pinning success_rate at 0.00 and freezing curriculum
  advancement in tile/GA mode). Validated by a 150-gen from-scratch
  tile run whose depth tracker showed repeated clears into 1-2/1-3.

### Performance (2026-07-11 optimization pass)

Measure-first pass over the training hot path, the Rust core, and the
spectator grid. Every emulation-touching change gated on the 146-tape
parity suite + Mesen lockstep (33/33); fidelity unchanged on default
settings.

- **Tile-mode PPO update ~1.9× faster** (real 60-env A/B): torch's
  default intra-op thread pool thrashed against the emulator's rayon
  workers on cpu-device runs; now capped to 1 thread with a temporary
  raise for the once-per-iter RND pass. Wall time -16%/iter.
- **Rollout device-syncs cut 3× → 1×/step**: values/log-probs
  accumulate on-device and drain in one fused transfer after the
  rollout (bit-identical buffers); only the action transfer the
  emulator needs remains per-step.
- **Punch-Out (MMC2) and Gradius (CNROM) now run on the ASM CPU**
  (+12-13% and +21% respectively, single-env): both mappers gained
  flat ASM PRG windows. Wiring Gradius byte-exact also surfaced and
  fixed two pre-existing engine-wide ASM MMIO-timing misalignments
  ($4014 OAM-DMA arming on STY/STX; indexed/indirect MMIO store and
  early-read commit cycles) — an across-the-board fidelity win.
- **Tile mode stops shipping a dead 7-14 KB observation buffer** per
  worker per step (~430 MB of zeroing + 61k numpy objects per
  iteration at 60 envs, all discarded); audio drain FFI now only
  polls workers whose audio is on.
- **Batched PPU Replace mode finally wired** (shipped in April with
  zero callers): per-game `reinforce.batched_render` knob, enabled
  for Contra/Zelda/Metroid/Mario headless runs (+10-27% single-env
  benched); `preprocess_f16` propagated to all 16 pixel configs;
  opt-in `reinforce.asm_bulk_cycles` budget for MMC1/UxROM (default
  unchanged; per-game lockstep+Mesen+parity rungs required to raise).
- **Spectator grid at 32+ tiles**: nearest-neighbor final scaling +
  alternating-parity repaint budget (~15 fps/tile, imperceptible at
  those cell sizes), and zero paint work while minimized — headroom
  for 64-tile stream layouts.
- New guard tests: batched-render RAM-identity over 240 SMB frames,
  adapter sentinel/audio-gating suite (15 tests), grid paint-budget
  tests, Punch-Out + Gradius ASM lockstep soaks, Tecmo Bowl parity
  tape (second CNROM title).
- **Spectator scale for the live grid**: realtime pacing moved out of
  the emulator worker threads (the old design capped simultaneously
  paced workers at ~12 — one parked thread each; now one pool-level
  sleep, GIL released, capacity bounded only by the emulation budget),
  a 0.25-16× pace multiplier (`reinforce.pace_multiplier`), audio
  production decoupled from pacing (`set_worker_audio`) so the mixer's
  all-mode finally sums every game's audio at once (1/√n normalized,
  pitch-up above 1× via the existing resampler), and paced workers now
  render only the displayed final sub-frame (3-4× less PPU work each,
  verified bit-identical emulation). Finished/dead spectator workers
  no longer throttle the pool.

Product-hardening + cross-game reward program. The tool went from "trains SMB"
to installable, crash-safe, and reward-complete across 16 games, with every
emulator-fidelity and win-predicate change validated against a ground-truth
oracle (Mesen) or reached live on the emulator, and the whole change set put
through an adversarial regression review.

### Added

- **Bespoke reward functions with real win predicates for 16 games** (was ~6):
  added Tetris, Bubble Bobble, Punch-Out, Kung Fu, Gradius, Excitebike,
  Ghosts'n Goblins, DuckTales, Kid Icarus, Double Dragon. Every RAM address is
  validated live on the emulator; `episode_success()` is a genuine win
  (stage/match/floor/round clear or game beaten), never a cumulative-reward
  proxy. Win RAM is **verified-live** for SMB, Punch-Out, and Kung Fu.
- **Go-Explore** (first-return-then-explore) wired into `vanilla_ppo` (opt-in,
  mutually exclusive with the SMB curriculum) — the lever that cracked the
  SMB 1-4 Bowser fight (policy crosses 1-4 → world 2 reliably from mid-1-4).
- **Live win-predicate test** that drives the real game to a real win and
  asserts `episode_success()` fires — the structural fix for self-referential
  tests that let win bugs hide behind synthetic RAM.
- Winner retention, seeded + ROM-MD5 run manifests, crash-safe `catch_unwind`
  on state load, dead-worker revival, and per-game `configs/*.yaml`.

### Fixed

- **Emulator fidelity (Mesen-validated):** MMC5 PRG/CHR banking, MMC1 SUROM
  512 KB PRG-A18, MMC3 A12 scanline-IRQ, PPU forced-blank backdrop + OAM mask,
  PPU greyscale/color-emphasis, OAM-DMA bus routing, cartridge NES 2.0/archaic
  robustness (crafted headers no longer abort the process), ASM ADC/SBC
  page-cross cycle cost. Parity stays byte-exact (146 tapes); Dragon Warrior
  III/IV migrated to the Mesen-lockstep oracle where nes-py is the inaccurate party.
- **Win-predicate bugs** found by adversarial review + win-verification: Zelda
  (declared victory 3 dungeons early), Metroid (read a garbage byte), Castlevania
  (dead boss address, then a stage-index off-by-one), SMB castle-clear (credited
  warp pipes), MegaMan (any single boss), GenericReward (`total > 0` always true;
  frame-counter score exploit), Bubble Bobble (HUD-tile score exploit), curriculum
  (under-sampling regressed strong agents), demo GIF crash on winner checkpoints.

### Changed

- **README rewritten for a new user.** Adds a crisp "What this is", a
  requirements section, and an end-to-end **Train a game** walkthrough
  (`roms/` → `scripts/capture_start_state.py` → `make train GAME=<name>` →
  `make eval GAME=<name>` → `make scoreboard`, plus the GUI for watching live),
  using only commands that exist in the `Makefile` and `scripts/`.
- **Relabelled the default trainer as `vanilla_ppo`** throughout the README and
  architecture diagram; the GA-based modes (`ga_ppo`, `pure_ppo`) are now
  described as legacy, matching what `scripts/train_game.py` actually launches.
- **Replaced the contradictory SMB status** with a single honest "What actually
  works today" section: world-1 training produces greedy clears of 1-1/1-2/1-3
  via the save-state curriculum; 1-4 and full autonomous 8-4 remain unsolved;
  Contra learns but does not yet clear; the other games are scaffolded. Notes
  that no pre-trained checkpoints ship (they are gitignored).
- Removed the fresh-clone install step that ran a smoke script hard-wired to a
  gitignored ROM; the documented post-install check is now `make test`, which
  needs no ROM.

## [0.2.0] — pre-release

Major training-stack expansion. Three independent but complementary
shipments: an upgraded pixel-CNN training pipeline, a DreamerV3 world-model
trainer scaffold, and a per-game tile-based optimization path for SMB. The
default behavior of every existing profile is preserved; new capabilities
are opt-in via profile YAML or the new GUI dropdowns.

### Added (2026-05-14 / 2026-05-15)

**SMB whole-pool save-state curriculum (`src/training/trainer.py`)**
- `_run_vanilla_ppo` now auto-detects when the pool reaches a new SMB
  area-byte and snapshots the worker state via `pool.save_worker_state`.
- All envs warm-start from the captured state every iter via
  `pool.load_worker_state`. Stage advances are persisted to
  `checkpoints/smb_curriculum/stage_NN.state` + sidecar
  `stage_NN.meta.json` (anchor byte) so restarts resume mid-curriculum.
- Per-iter telemetry: `max-W-L: 1-1=N 1-2=M ...  |  end-W-L: ...
  | curriculum stage=N (anchor area=N, N/N envs past stage)`.
- `rollout_steps` raised from 512 to 1024 so the natural 1-1 → cutscene
  → 1-2 transit (≈400 RL steps) fits within a single rollout.

**Vanilla PPO auto-resume on startup**
- Scans `checkpoint_dir` for the highest-numbered
  `vanilla_ppo_iter_*.pt` and loads `net.state_dict` + Adam state on
  startup. Previously only the GA-mode `gen_*.pt` checkpoints were
  honored by `trainer.run(resume_from=...)`.

### Fixed (2026-05-14)

**`nes_core::Pool::load_worker_state` PPU/cycle re-sync bug**
- After `apply_state` restored the NES state, `frame_cycle_target` was
  left at its pre-load value (typically ~30k from cold-boot), while the
  loaded state's `nes.cycles` was orders of magnitude larger. The next
  `advance_one_frame` computed `target = stale + 29781` < `nes.cycles`,
  so BOTH `while cycles < target` loops were immediately false, the NES
  never ticked, and the screen showed the previous frame indefinitely.
- Fix: `w.frame_cycle_target = None` in `load_worker_state` (matches
  what `reset()` already does). Save-state curriculum was structurally
  blocked on this — RAM restored correctly but the game couldn't
  advance from the loaded state.

### Added

**Training dashboard (`src/gui/training_dashboard.py`)**
- Single-pane observer view replacing the older `MetricsWindow` as the
  primary "watch learning happen" surface.
- Plot grid: best/avg fitness, reward signal stack, PPO learning
  telemetry (loss/policy/value/entropy), depth + curriculum success
  (dual-axis), and a world-model losses panel that lights up in
  Dreamer mode.
- Side pane: replay-buffer fill bar, depth records list, recent
  highlight clips, and a live world-model reconstruction strip
  (orig vs. decoded frames, refreshed every 10 train steps).
- Reads `checkpoints/metrics.jsonl` directly — no trainer-side queue.

**Pixel-CNN training improvements (`src/models/policy_network.py`,
`src/training/trainer.py`)**
- Orthogonal weight init with PPO-standard gains (√2 for relu trunks,
  0.01 for actor head, 1.0 for critic).
- Optional `LayerNorm` on the trunk FC, enabled by default. Stabilizes
  the value head when reward magnitudes vary across orders of
  magnitude.
- `IMPALA` ResNet encoder option (`reinforce.encoder: impala`) — 3
  stages × 2 residual blocks. ~3.4M params, better representation per
  param than Nature-DQN at higher per-step cost.
- `symlog_rewards` PPO option — compresses returns via
  `sign(r)·log(1+|r|)` so the value head's MSE stays in PPO-friendly
  numerical range when per-episode reward is in the thousands.
- DrQ random-shift augmentation (`reinforce.drq_aug`) on PPO update
  observations only — pads to 88×88 with replicate edges, random-
  crops back to 84×84.
- RND (Random Network Distillation) intrinsic-motivation auxiliary
  loss with running-mean/std normalization on both observations and
  the per-state error magnitude (`src/models/rnd.py`).
- Backwards-compat state_dict migration: legacy Nature-DQN
  checkpoints (top-level `conv1.weight`, etc.) are auto-rewritten
  to the new `encoder.conv1.weight` layout on load.
- GA-only warmup gens (`reinforce.warmup_gens_ga_only`) — skip PPO
  for the first N generations so behavior-cloned weights spread
  through the population before PPO drift starts.
- Optional `preserve_elite_diversity` flag — skip the post-PPO clone-
  overwrite that historically collapsed the elite pool to a single
  policy. When True, PPO updates only `best`; the other elites keep
  their own structurally-distinct weights.

**DreamerV3 scaffold (`src/training/dreamer.py` and friends)**
- Self-contained world-model trainer alongside the existing PPO+GA
  trainer. Selected per-profile via `training_mode: dreamer` or
  per-launch via the GUI dropdown.
- Encoder + RSSM + decoder + reward head + continue head
  (`src/models/world_model.py`). Categorical 32×32 stochastic latent
  with straight-through gradients (V3-proper); Gaussian latent
  available behind a flag for ablation.
- Actor with straight-through categorical sampler + critic with
  Polyak-EMA target network + λ-returns
  (`src/models/dreamer_ac.py`).
- Sequential replay buffer with bulk-add and ring wrap-around
  (`src/training/replay_buffer.py`).
- Atomic checkpointing every N train steps with auto-resume.
- World-model reconstruction visualization dumped to
  `<checkpoint_dir>/dreamer/reconstruction.npy` for the dashboard.

**SMB Tile Mode — per-game optimization path**
- Generic tile-observation framework (`src/emulation/tile_observations/`)
  with a `TileObservation` Protocol + factory. Adding a new game's
  encoder is one new file under that directory.
- SMB-specific tile decoder (`src/emulation/tile_observations/smb.py`)
  reading `$0500-$069F` (level metatiles) + 5 enemy slots + Mario
  state into a 13×13 grid + 6 scalars (175 features total).
- Small actor-critic MLP (`src/models/tile_policy.py`) with the same
  forward surface as `PolicyNetwork`. ~14k params at default widths.
- Dense reward checkpoints in `MarioReward`
  (`nes_core/src/rewards.rs`). Seven RAM-readable bonuses fire once
  each per episode at hand-picked x-positions through 1-1
  (50/100/150/250/400/600/1000 at x = 350/720/1100/1640/2100/2700/2900).
  Multiplier is `reward_weights.checkpoint_scale`; default 0.0
  preserves prior behavior.
- New profile `configs/mario_tiles.yaml` opting into the full tile
  recipe: encoder swap, dense rewards, elite-diversity preservation,
  tighter trajectory cap, NEAT-paper-inspired GA tuning.

**GUI improvements (`src/gui/`)**
- Trainer-mode dropdown (`(profile)` / `GA + PPO` / `DreamerV3`)
  next to the profile picker. Selection persists across sessions.
- Multi-pick BC demo file picker — ⌘-click multiple `.fm2` /
  `.state.bin` files; the trainer's BC pipeline replays each from a
  cold-boot env and concatenates the (state, action, reward) tuples.
- "Open Dashboard…" button to reopen the unified dashboard if the
  user closes it during a live run.
- BC seed cache key now includes the encoder name + frame_skip;
  swapping architectures auto-invalidates stale caches. Stale caches
  are renamed to `*.pt.stale` rather than crashing the run.
- Session geometry persistence: main window remembers its size and
  position across launches.

**Tests (`tests/`)**
- `test_replay_buffer.py` — sequential replay buffer wrap-around,
  bulk-add, sample shape invariants, deterministic seeding.
- `test_world_model.py` — encoder/decoder shapes, latent kind
  switching, KL non-negativity, full loss backprop.
- `test_rnd.py` — Welford-style running stats, frozen-target gradient
  isolation, normalization stability, state_dict roundtrip.
- `test_dreamer_ac.py` — actor/critic shapes, straight-through
  gradient flow, target Polyak update math, λ-returns edge cases.
- `test_dreamer_trainer.py` — train-step + checkpoint roundtrip +
  pruning + latent-kind propagation.
- `test_smb_tile_extractor.py` — SMB RAM decoding (factory, output
  shape/dtype, tile decoding, enemy markers, scalars).
- `test_tile_policy.py` — tile network shape, init, save/load,
  gradient flow.
- `test_mario_checkpoint_reward.py` — dense reward checkpoint
  behavior (fire-once-per-episode, multi-checkpoint accumulation,
  scale-zero disable, reset re-arms).

Total: 28 new tests; full suite is now 309 tests.

### Changed

- `_reinforce_update` now returns `(loss, stats_dict)` so the trainer
  can emit per-component PPO metrics (`ppo_policy_loss`,
  `ppo_value_loss`, `ppo_entropy`, optional `rnd_loss` /
  `rnd_intrinsic_avg`) per generation. Dashboard reads these to chart
  PPO health over time.
- `MarioReward::new` signature gains a `checkpoint_scale` parameter.
- `Trainer.__init__` reads `reinforce.encoder` / `preserve_elite_diversity`
  / `warmup_gens_ga_only` / `bc_epochs` / `symlog_rewards` /
  `drq_aug` / `rnd_intrinsic_coef` / `rnd_loss_coef` from the profile
  YAML.

### Fixed

- BC seed cache no longer crashes the run when the cached architecture
  doesn't match the current encoder. The shape-mismatch is caught,
  the cache file renamed to `*.pt.stale`, and BC pretraining is
  re-run cleanly.
- BC dataset off-by-one alignment in `build_dataset`: each emitted
  pair was `(state_AFTER_chunk, action_DURING_chunk)`, which is one
  frame_skip chunk ahead of what the runtime policy sees at decision
  time. The network was being trained to predict actions for states
  one chunk in the future. Fixed by maintaining a `last_state`
  buffer; emitted pairs are now `(state_BEFORE_chunk,
  action_DURING_chunk)`. Required adding `.get()` methods to
  `FrameStacker` and `TileFeatureStacker` so the chunk-start state
  can be sampled without mutating the ring buffer.
- Persistent PPO Adam optimizer's accumulated `m`/`v` moments are
  now cleared whenever BC replay injects a fresh policy into the
  population (`_run_bc_replay` sets `self._ppo_optimizer = None`).
  Without this, the optimizer's stale momentum from the pre-BC
  policy's gradients pulls the freshly-injected BC weights back
  toward the pre-BC policy on the very next PPO update, undoing
  the anchor.

### Vanilla PPO trainer mode

Third trainer mode alongside `ga_ppo` (default) and `pure_ppo`. Matches
the literature recipe that empirically clears SMB 1-1 — yumouwei's
super-mario-bros-reinforcement-learning and uvipen's
super-mario-bros-PPO-pytorch both use this shape: **single policy
network, N parallel envs as rollout collectors, batched GAE,
K-epoch PPO update**. No GA. No population-of-policies. No BC
injection.

Why a third mode rather than tuning the existing ones: a 2-day
investigation of GA-PPO hybrid + BC anchor on canonical and
mario_tiles produced ~1 lucky-walk clear per 80-100 gens but
never a *committed-policy* clear. Empirical diagnosis: GA-style
data mixing (30 workers running 30 different policies, results
folded into PPO's gradient) violates PPO's stable-policy-across-
updates assumption. The literature recipes don't have this problem
because they don't have a GA on top.

Activated by setting `reinforce.trainer_mode: vanilla_ppo` in a
profile. `run()` dispatches to `_run_vanilla_ppo()` BEFORE entering
the GA loop. The new method:

- Reuses the existing parallel-env pool (`RustPool`) — same 30
  workers, just as N rollout collectors for ONE shared policy
- Per iteration: fresh `reset_all()` → rollout for `rollout_steps`
  per env → bootstrap V(s_T) → GAE-λ backward sweep with done-mask
  → global advantage normalization across the (env × step) batch
  → K-epoch minibatched PPO update
- Persistent `_ppo_net` + `_ppo_optimizer` (same machinery the
  GA-mode PPO update uses; reused so the Adam-state fix applies)
- Checkpoint every 10 iters as `vanilla_ppo_iter_NNNNN.pt`

New YAML knobs: `rollout_steps` (default 512), `ppo_minibatch_size`
(default 256), `gae_lambda` (default 0.95).

New profile `mario_vanilla_ppo.yaml`: yumouwei's canonical
hyperparameters (lr=3e-4, γ=0.9, λ=0.95, clip=0.2, K=10,
value_coef=0.5, entropy_coef=0.01) pinned via tests so they
don't drift.

### BC anchor pipeline

End-to-end overhaul of the BC replay → injection → anchor path. All
four mechanisms now fire correctly in cascade when a clear is
captured (verified empirically: capture → trigger → train → reset
fires in <1 s).

- **Immediate trigger on new capture.** Previously BC replay only
  fired on `gen % bc_replay_every_gens == 0` (default every 20
  gens). A clear captured at gen 5 waited 15 gens before being
  anchored — during which PPO could drift the population away from
  the clear-finding policy. The trainer now tracks
  `_last_bc_buffer_size` across gens and fires BC immediately when
  the buffer grows, AND on the modulo schedule. Emits
  `BC replay: triggered immediately (buffer grew N -> M this gen)`.
- **Single-trajectory training.** New `bc_replay_train_window`
  knob (default 3, canonical uses 1) caps how many of the most
  recent buffered trajectories actually feed BC. Aggregating
  across multiple trajectories from different source genomes
  produced near-uniform action labels (e.g. 14-18% per action
  across 6 actions vs uniform=16.67%) because different policies
  jumped at slightly different frames over the same states. BC
  cross-entropy on contradictory labels splits the difference into
  a low-confidence near-uniform distribution. `train_window=1`
  gives the network a single coherent set of (state, action)
  labels per state.
- **Source `genome_id` tagging.** Each captured trajectory now
  carries the `genome_id` it came from (5-tuple instead of 4-tuple).
  Diagnostics-only for now — BC selection is by recency — but
  makes the buffer's provenance inspectable post-hoc. Cache format
  bumped to v2 with backward-compat for v1 caches (loaded with
  sentinel `genome_id=-1`).
- **`bc_demo_path` YAML fallback.** Profiles can now self-declare
  their BC demo via `reinforce.bc_demo_path`. Explicit ctor arg
  (GUI file-picker, CLI `--bc-demo`) still wins; falls through to
  YAML when no path is passed.

## [0.1.0] — pre-release

First public release. Ships the emulator core, training infrastructure, GUI,
and the validation harnesses that gate them.

### Added

**Rust NES core (`nes_core/`)**
- 6502 CPU interpreter + AArch64 assembly fast path (`cpu_asm.s`,
  ~4,300 lines) covering 151 official + ~30 stable illegal opcodes.
  99.97% hit rate on real ROMs; falls back to interpreter for the rest.
- PPU with NEON-SIMD batched render path (sprite eval, background
  fetch, palette expansion).
- APU with all 5 channels (2 pulse, triangle, noise, DMC) +
  mapper-extension audio (MMC5, VRC6).
- 36 mappers: NROM, MMC1/3/5, MMC2/4, AxROM, UxROM, CNROM, AOROM, all
  Konami VRC variants (2/4/6/7), Sunsoft FME-7, Namco N163, plus
  multi-cart and discrete-logic mappers. 793/794 ROMs in the test
  library boot — the single load-failure (`Yoshi (USA).nes`) is a
  truncated dump.
- Versioned save state (`NCST\x01` magic + bincode body).
- `enum_dispatch`-based mapper trait — zero virtual-call overhead in
  the hot path.
- Rayon-based zero-IPC worker pool. N parallel NES instances stepped
  in a single PyO3 call with no pickling, shared memory, or
  subprocesses.
- macOS-native audio via cpal — 5-channel pan matrix, per-channel
  43,653 → 44,100 Hz resampler.
- PGO build pipeline (`scripts/pgo_build.sh`) — ~+81% throughput on M4
  Max via 3-stage instrument → profile → rebuild.

**Training infrastructure (`src/`)**
- PPO + GAE on top of a genetic algorithm with optional behavior
  cloning warm-start.
- PyTorch MPS policy networks (Nature-DQN CNN); Core ML export for
  ANE inference at replay time.
- Behavior cloning pipeline with per-frame action tape format and
  weighted imitation by demo reward.
- Reward functions for Mario, Zelda, Contra, Mega Man, Castlevania,
  Metroid, plus a generic motion+score-hunt fallback for any ROM.
- Mario reward uses `visited_x_max` gating — backtracking is no-op,
  not a penalty (lets the agent retreat to time jumps).
- Defensive policy sampler (`_safe_sample_from_logits`) that recovers
  from NaN/Inf logits without killing the training process.
- Depth-driven curriculum manager.

**GUI (`src/gui/`)**
- PyQt6 main window with ROM picker, training start/stop, BC demo
  picker, mixer & metrics windows.
- Live N-instance frame grid (one tile per worker).
- Interactive Play window with two save modes:
  - **Save State** — NCST snapshot (use with `--start-state`)
  - **Save BC Tape** — raw per-frame action mask (use with `--bc-demo`)

**Validation harnesses (`tests/`)**
- `nestest` byte-exact CPU validation: 8,991 instructions vs the
  Nintendulator golden trace (PC + opcode + asm + A/X/Y/P/SP + CYC).
- ASM-vs-Rust differential fuzz harness (`asm_diff_fuzz`). Recent
  240M-instruction soak: 0 divergences.
- Mesen oracle harness (`tests/parity/test_mesen_lockstep.py`) — 31
  ROMs replayed against `Mesen --testRunner` with byte-exact CPU/RAM
  diff at every step.
- Parity gate (`make parity`) — 146 tests in ~110s.
- Library-wide playability sweep (`scripts/playability_sweep.py`) and
  RAM-divergence sweep (`scripts/parity_sweep.py`).

### Performance

Measured on an M4 Max MacBook Pro vs nes-py (LaiNES C++) on Contra.
Numbers vary with chip, OS, and background load.

- `fs=1` single-env, full render: 0.72–0.75× nes-py.
- `fs=4` single-env: 1.23–1.25× nes-py.
- `fs=4` 12-parallel (training workload): up to 3.72× nes-py.
- `fs=16` 12-parallel: up to 3.58× nes-py.
- 200–320× realtime per worker, 12–19k aggregate fps at N=12.

### Known Limitations

- **SMB level 1-1 convergence not demonstrated**. The training
  infrastructure runs, BC warm-start works, reward shape is sound, but
  reaching the flag with a 16- or 64-genome population in a few hundred
  generations requires longer demos and/or larger populations than the
  defaults. This release ships the *infrastructure* for that work.
- **Audio sign-off** done on built-in MacBook speakers and headphones;
  USB DAC sign-off pending.
- **Metal PPU offload** experimented with (`nes_core/src/metal_render.rs`
  v1) but not used — Metal dispatch overhead dwarfs the per-frame
  compute on this workload size. v2 (batched-across-workers) is open
  research.

### Acknowledgements

`nes_core/` was originally forked from
[RustedNES](https://github.com/PhilipK/RustedNES) (Jason Hansen, 2018,
MIT/Apache-2.0). Most of it has been rewritten — AArch64 ASM CPU, NEON
PPU, rayon pool, save state, audio mixer, half the mappers — but the
mapper trait surface and several mapper implementations carry forward.
Reference comparisons against [LaiNES](https://github.com/AndreaOrru/LaiNES)
and [Mesen](https://www.mesen.ca/) drove most of the cycle-accuracy
fixes. See README *Acknowledgements* for full credits.
