# Changelog — Neural Entertainment System (NES)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

In this changelog, **NES** refers to this project (Neural Entertainment
System). The original Nintendo Entertainment System hardware is named in full.

## [0.2.0] — pre-release

Major training-stack expansion. Three independent but complementary
shipments: an upgraded pixel-CNN training pipeline, a DreamerV3 world-model
trainer scaffold, and a per-game tile-based optimization path for SMB. The
default behavior of every existing profile is preserved; new capabilities
are opt-in via profile YAML or the new GUI dropdowns.

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
