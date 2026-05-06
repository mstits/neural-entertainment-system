# Architecture

A deeper dive into how the training lab fits together. For the short version,
see the top-level `README.md`. For the roadmap to public release, see
`docs/proposals/unified_rust_plan_v3.md`.

## Overview

This repository pairs a hand-written Rust NES emulator (`nes_core`) with a
PyTorch + PyQt6 training framework. The emulator runs N instances in-process
under a rayon worker pool; the trainer runs PPO + GAE on top of a genetic
algorithm, with optional behavior cloning warm starts. Everything per-step
lives on the Rust side behind a single PyO3 call.

## Language boundary

Python owns what is slow-moving or user-facing:

- PyTorch MPS inference and training (policy network, PPO updates, GA
  operators on `state_dict`s).
- PyQt6 GUI (ROM picker, training grid, metrics, reward tuning, replay).
- Checkpoint I/O, metrics JSONL, configuration YAML parsing.
- Narrator caption templates and depth memo persistence (`*.jsonl`).

Rust owns what runs every step:

- 6502 CPU, PPU, APU, mappers, mirroring, DMA.
- Reward functions (byte-exact ports of the original Python rewards).
- Frame preprocess (XRGB → gray → 84×84 → fp16) via NEON SIMD.
- Worker pool (rayon par_iter, zero IPC).
- Audio mixer (cpal stereo, 5-channel pan matrix, per-channel resampler).
- Depth tracker and narrator hot logic (RAM-delta detection).
- Versioned save state (`NCST\x01` magic, bincode body).

The split is deliberate: anything in the inner loop is Rust, anything in
the outer loop or the UI is Python. The PyO3 boundary is crossed once per
trainer step via `pool.step_all(actions)`.

## Emulator core — module layout

`nes_core/src/` is a single crate with the following module responsibilities.

### CPU
- `cpu.rs` — pure-Rust 6502 interpreter. Per-cycle dispatch via
  `Instruction.cycles: &[CycleFn]`. Passes blargg's CPU test suite. Runs on
  every target; the authoritative reference for the ASM path.
- `cpu_asm.rs` + `cpu_asm.s` — AArch64 assembly core gated on
  `#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]`. 151 official
  opcodes + ~30 stable illegals (LAX, SAX, DCP, ISC, SLO, RLA, SRE, RRA +
  NOP variants). Unknown opcodes fall back cleanly to the Rust core.

### PPU
- `ppu.rs` — PPU state machine and per-pixel renderer. Supports a
  `skip_render` fast path used on non-display frames (e.g. 15 of 16 when
  `frame_skip=16`).
- `ppu_neon.rs` — batched NEON renderer. Currently gated behind
  `BatchedRenderMode` (Off by default) pending the v3 invalidation-tracking
  fix. When Replace mode lands, mid-scanline PPU state changes will fall
  back to per-pixel for the dirty scanline only.
- `metal_render.rs` — Metal compute shim behind the `metal` feature.
  Measured 10x slower than CPU for per-frame palette expansion because of
  kernel-launch overhead; kept as a stub for future batched use.
- `oam_dma.rs` — OAM DMA state machine.

### APU and audio
- `apu.rs` — 5 channels (2 pulse, triangle, noise, DMC). NTSC ~43653 Hz
  native rate. Exposes `generate_sample_channels -> [f32; 5]` so the audio
  layer can pan channels independently.
- `audio.rs` — cpal Core Audio stream, 5-channel stereo pan matrix,
  per-channel linear resampler (43653 → 44100 Hz), worker-to-callback
  ring buffers. Solo/mute/master volume state owned here.

### Memory, bus, input
- `memory.rs` — 2 KB CPU RAM, open-bus logic, PPU register mirroring.
- `system_bus.rs` — the CPU's view of the world. Routes reads and writes
  to RAM, mappers, PPU registers, APU registers, and the controller port.
- `input.rs` — controller state (8-bit shift register per port).

### Cartridges and mappers
- `cartridge.rs` — iNES 1.0 + NES 2.0 parser, lenient on reserved bytes,
  MD5 hash of the payload for profile dirty-detect. NES 2.0 byte 10 is
  parsed as two nibbles (low = volatile PRG-RAM shift, high = battery
  PRG-RAM shift); the parser sums both regions. Reading only the low
  nibble used to silently drop battery-only carts' SRAM (Zelda
  `flags10=0x70`), which crashed the game on boot via open-bus reads.
- `mapper.rs` — `MapperEnum` dispatch via `enum_dispatch`. Trait
  includes a `set_cpu_cycle(u64)` hook (default no-op) called from
  `Nes::tick` once per CPU cycle. Mappers that need cycle-precise
  quirks (currently MMC1's RMW consecutive-write filter) read it.
- `mapper/mapper0.rs` … `mapper/mapper85.rs` — per-mapper state and bank
  switching. MMC5 audio split into `mmc5_audio.rs`. VRC2/VRC4 share
  `vrc24.rs`; VRC6 + VRC6 audio split into `vrc6.rs` + `vrc6_audio.rs`.
  MMC1 (`mapper1.rs`) implements the consecutive-write filter:
  `INC $FFFF` and other RMW opcodes do dummy-write + real-write on
  back-to-back CPU cycles; real MMC1 hardware ignores the second
  write, and we now match that.

### Pool, FFI, preprocess
- `pool.rs` — rayon worker pool. Holds `Vec<UnsafeCell<Worker>>` behind a
  `Sync` wrapper so `par_iter_mut` can run N workers concurrently with
  per-index exclusivity. `catch_unwind` per worker isolates panics;
  `pool.set_panic_isolation(False)` turns isolation off for production
  hot paths where a panic should fail loud.
- `python.rs` — PyO3 bindings. Zero-copy in: `PyReadonlyArray1<'py, i32>`
  for actions. Zero-copy out: numpy f16 arrays for observations.
- `preprocess.rs` — NEON XRGB → gray → 84×84 area resize → f16. Baked
  `/255.0` into the fp16 conversion so the trainer can skip the
  CPU→MPS normalize kernel.
- `sink.rs` + `sink/` — frame + audio sink abstractions so the GUI can hand
  in a callback and receive frames without copying through Python.

### Training-side helpers (still in Rust)
- `rewards.rs` — per-game reward functions (Mario, Zelda, Contra, Mega Man,
  Castlevania, Metroid). Dispatch via `build_reward_function(profile)`.
- `narrator.rs` — RAM-delta event detection (first dungeon entry,
  triforce pickup, depth record, etc.).
- `depth_tracker.rs` — per-genome deepest-RAM-key record for
  curriculum auto-promotion.

## Data flow — one `pool.step_all` call

```mermaid
sequenceDiagram
    participant Trainer as Python Trainer
    participant PyO3 as PyO3 binding
    participant Pool as Rust Pool (rayon)
    participant Worker as Worker i
    participant Torch as PyTorch MPS

    Trainer->>Trainer: stack frames -> obs (numpy)
    Trainer->>Torch: forward(obs) -> logits
    Torch-->>Trainer: sampled actions (i32 array)
    Trainer->>PyO3: pool.step_all(actions)
    PyO3->>Pool: release GIL, par_iter_mut workers
    par workers in parallel
        Pool->>Worker: step(action_i, frame_skip)
        Worker->>Worker: CPU -> PPU -> APU tick loop
        Worker->>Worker: preprocess frame -> f16 84x84
        Worker->>Worker: reward fn dispatch
        Worker->>Worker: narrator + depth update
    end
    Pool-->>PyO3: frames[N], f16_obs[N], ram[N], rewards[N], done[N]
    PyO3->>PyO3: reacquire GIL, wrap as numpy views
    PyO3-->>Trainer: tuple of numpy arrays
    Trainer->>Trainer: append to trajectory, loop
```

Per the post-PGO hot-path baseline on 16 workers at `fs=16`:
- `pool.step_all` is ~84% of wall-time.
- PyTorch MPS forward is ~10%.
- Everything else (reward dispatch, narrator, depth, numpy wrap) is under 1%.

## Hot path — `Nes::step`

```mermaid
flowchart TB
    Start([Nes::step action]) --> Input[input.update controller shifts]
    Input --> CPU[cpu.tick]
    CPU -->|reads / writes| Bus{system_bus dispatch}
    Bus -->|0x0000-0x1FFF| RAM[RAM 2 KB]
    Bus -->|0x2000-0x3FFF| PPUreg[PPU registers]
    Bus -->|0x4000-0x401F| APUreg[APU + input regs]
    Bus -->|0x4020-0xFFFF| Mapper[mapper dispatch]
    CPU --> PPUtick[ppu.tick x3 per CPU cycle]
    PPUtick -->|per-pixel or ppu_neon| FrameBuf[frame buffer XRGB]
    CPU --> APUtick[apu.tick]
    APUtick --> Samples[5-channel samples]
    Mapper -.cycle hook.-> CycleHook[set_cpu_cycle u64]
    CycleHook -.MMC1 RMW filter.-> Mapper
    FrameBuf --> FrameSink[frame sink + NEON preprocess]
    Samples --> AudioSink[audio sink: resample + pan]
    FrameSink --> Done([step done])
    AudioSink --> Done
```

The `set_cpu_cycle` dotted edge is the per-CPU-cycle hook used by
mappers that need to detect back-to-back register writes (currently
only MMC1, for the RMW dummy-write filter).

## Mapper tree

```mermaid
graph TD
    Root[Supported mappers<br/>36 total / 99.9% of US library]
    Root --> Discrete[Discrete logic]
    Root --> MMC[MMC family]
    Root --> Konami[Konami VRC]
    Root --> Nintendo[Namco / Sunsoft / Tengen]
    Root --> Multicart[Pirate / Multicart]

    Discrete --> NROM[NROM 0]
    Discrete --> UxROM[UxROM 2]
    Discrete --> CNROM[CNROM 3]
    Discrete --> AxROM[AxROM 7]
    Discrete --> Colordreams[Colordreams 11 / 66]
    Discrete --> CPROM[CPROM 13]
    Discrete --> BNROM[BNROM 34]
    Discrete --> Camerica[Camerica 71]
    Discrete --> Nina03[Nina-03 79]

    MMC --> MMC1[MMC1 / SxROM 1]
    MMC --> MMC3[MMC3 / TxROM 4]
    MMC --> MMC5[MMC5 / ExROM 5 + audio]
    MMC --> MMC2[MMC2 / PxROM 9]
    MMC --> MMC4[MMC4 10]
    MMC --> TxSROM[TxSROM 118]
    MMC --> TQROM[TQROM 119]
    MMC --> NWC[NWC 1990 105]

    Konami --> VRC2a[VRC2a 22]
    Konami --> VRC2b[VRC2b 23]
    Konami --> VRC4[VRC4 21 / 25]
    Konami --> VRC6[VRC6 24 / 26 + audio]
    Konami --> VRC7[VRC7 85]

    Nintendo --> N163[Namco N163 19]
    Nintendo --> S4[Sunsoft-4 68]
    Nintendo --> FME7[FME-7 69]
    Nintendo --> RAMBO[Tengen RAMBO-1 64]

    Multicart --> Caltron[Caltron 6-in-1 41]
    Multicart --> Nes37[NES-ZZ 37]
    Multicart --> Nes47[NES-QJ 47]
    Multicart --> Nina06[NINA-06 / HES 113]
    Multicart --> Action52[Action 52 228]
    Multicart --> Quattro[Camerica Quattro 232]
    Multicart --> Maxi15[Maxi 15 234]
```

Live compatibility matrix in `reports/full_library.md`. Programmatic access:
`nes_core.supported_mappers()` from Python.

## Build pipeline

The Rust crate builds via `maturin` into an abi3 Python wheel. Typical
commands:

```bash
# Install everything (venv, deps, wheel, PGO)
bash scripts/install_macos.sh

# Rebuild the wheel only
cd nes_core && maturin develop --release --features "python,asm_cpu"

# Apply an existing PGO profile
bash scripts/pgo_build.sh apply

# Full PGO: instrument -> record profile -> rebuild
bash scripts/pgo_build.sh full
```

Feature flags:
- `python` — PyO3 module + cpal audio. Set by maturin.
- `asm_cpu` — AArch64 ASM 6502 core. Only effective on aarch64.
- `simd` — NEON/SSE palette and audio paths.
- `metal` — Metal compute PPU shim (off by default).
- `pgo` — build marker for the PGO wrapper script.
- `ppu_neon_stats` — per-frame counters for the batched PPU renderer.

Release profile settings in `nes_core/Cargo.toml`:
- `opt-level = 3`
- `lto = "fat"`
- `codegen-units = 1`
- `panic = "unwind"` — load-bearing. Under `panic = "abort"`, any
  `panic!()` in a worker takes down the Python process via `SIGABRT`.
  Unwind lets `catch_unwind` in `pool.rs` isolate the bad worker.
- `strip = "symbols"`, `overflow-checks = false`.

PGO on the hot path lifts throughput ~81% (710 → 1289 worker-steps/sec on
M4 Max) and shaves 47% off 1-generation wall-time.

## Memory and safety

`unsafe` surface is three call sites, all documented in `nes_core/SECURITY.md`:

1. `preprocess.rs` — NEON intrinsics. Bounded by `chunks_exact(48)` so every
   `vld3q_u8` reads exactly 48 bytes; tail runs the scalar path.
2. `pool.rs::drain_audio` — one `slice::from_raw_parts` to reinterpret
   `Vec<i16>` as bytes for `PyByteArray::new_bound`, which copies before the
   `Vec` drops.
3. `cpu_asm.rs` + `cpu_asm.s` — AArch64 ASM 6502 core. ROM is read-only via a
   raw pointer; RAM access is masked to `address & 0x07FF`; MMIO round-trips
   through the same Rust `SystemBus` the interpreter uses.

Panic isolation: every worker step runs inside `catch_unwind`. A panic marks
the worker dead; it returns zero-filled frames and RAM forever after so the
trainer can skip it without stalling. `pool.set_panic_isolation(False)` turns
this off for production runs where a panic should fail loud rather than be
papered over.

Save states carry a `NCST\x01` magic prefix; version bumps refuse to load old
blobs with a clear `PyValueError` rather than silently corrupting trainer
state.

## Training side

The training stack is two-tier: a **universal pixel-CNN path** that works on
any NES game with no per-game code, and **per-game optimized paths** (currently
only SMB tile mode) that swap in a smaller architecture + dense reward shaping
for games we've invested in. The encoder choice is per-profile:

```mermaid
flowchart LR
    Profile[Profile YAML<br/>configs/&lt;game&gt;.yaml] -->|reinforce.encoder| Dispatch{Encoder?}
    Profile -->|training_mode| ModeDispatch{Trainer?}
    ModeDispatch -->|ga_ppo default| Trainer[Trainer<br/>GA + PPO + GAE]
    ModeDispatch -->|dreamer| Dreamer[DreamerTrainer<br/>world model + actor/critic]
    Dispatch -->|nature_dqn / impala| CNN[PolicyNetwork<br/>1.7M params<br/>4×84×84 stacked frames]
    Dispatch -->|smb_tiles| Tile[TilePolicyNetwork<br/>~14k params<br/>175 RAM-decoded features]
    CNN -->|forward| Trainer
    Tile -->|forward| Trainer
    Trainer -->|metrics.jsonl| Dashboard[TrainingDashboardWindow<br/>fitness · WM losses · replay · recon]
    Dreamer -->|metrics.jsonl| Dashboard
```

The two trainer classes (PPO+GA and Dreamer) coexist; profiles pick one via
`training_mode`. The two encoder paths coexist; profiles pick one via
`reinforce.encoder`. Both selections are independent — Dreamer can run on
pixels, PPO+GA can run on tiles, and the GUI dropdown can override either at
launch time.

### Policy networks
- **`src/models/policy_network.py`** — Nature-DQN CNN
  (Conv(32,8×8,/4) → Conv(64,4×4,/2) → Conv(64,3×3,/1) → FC(512) → actor + critic).
  ~1.7M params at 8 actions. Used by every profile that doesn't opt into a
  smaller encoder. Optional `encoder=impala` swaps the conv stack for a
  3-stage IMPALA ResNet (~3.4M params, better representation per param).
  Orthogonal init with PPO-standard gains, optional `LayerNorm` on the trunk.
- **`src/models/tile_policy.py`** — small actor-critic MLP
  (Linear(175,64) → SiLU → LayerNorm → Linear(64,32) → SiLU → actor + critic).
  ~14k params. Used in tile mode (currently SMB only). Same `forward_ac()`
  surface as `PolicyNetwork` so the trainer dispatches transparently.
- **`src/models/world_model.py`** — DreamerV3 stack: encoder, RSSM
  (deterministic GRU + categorical 32×32 stochastic latent), decoder, reward
  head, continue head. ~9M params total. KL with per-sample free-nats
  regularization.
- **`src/models/dreamer_ac.py`** — Dreamer actor (straight-through categorical
  sampler) + critic with Polyak EMA target net. Operates on world-model
  latents, not raw obs.
- **`src/models/rnd.py`** — Random Network Distillation predictor + frozen
  target. Optional intrinsic-motivation bonus for sparse-reward games.

### Tile observations (`src/emulation/tile_observations/`)
RAM-decoded semantic observations as an alternative to raw pixels. Generic
`TileObservation` Protocol; per-game implementations decode that game's RAM
layout into a fixed-size feature vector. Currently:
- **`smb.py`** — SMB-specific 13×13 tile grid centered on Mario plus 6
  scalars (velocity, on-ground flag, powerup, lives, sub-tile X). Reads
  `$0500-$069F` (level metatiles) and the 5 enemy slots. 175 features total.

Adding a new game's tile encoder is one new file under
`tile_observations/<game>.py` and a one-line entry in
`get_extractor()`. No trainer changes.

### PPO + GAE (`src/training/trainer.py`)
Clipped objective, value head shared with the policy trunk, GAE-λ advantage
estimation over trajectories collected per generation. Runs on the elite
genome after each GA generation. Bulk numpy arrays are pre-allocated once
per generation to avoid per-step alloc churn. Supports symlog reward
transform, DrQ random-shift augmentation (pixel mode only), RND auxiliary
loss, GA-only warmup gens, and `preserve_elite_diversity` mode that skips
the post-PPO clone-overwrite so the elite pool keeps structurally-distinct
weights.

### DreamerV3 trainer (`src/training/dreamer.py`)
Self-contained alternative to PPO+GA. Selected via `training_mode: dreamer`
in the profile (or the GUI dropdown). Coordinates data collection through
the same Rust pool, world-model training on sequence batches from a replay
buffer (`src/training/replay_buffer.py`), and actor/critic training on
imagined latent rollouts. Atomic checkpointing every N train steps, auto-
resume on next launch. Reconstruction visualization is dumped every 10
steps to `<checkpoint_dir>/dreamer/reconstruction.npy` for the dashboard.

### Genetic algorithm (`src/training/genetic_algorithm.py`)
Tournament selection, uniform crossover, Gaussian mutation on PyTorch
`state_dict`s. Runs once per generation boundary. Genome names
(`genome_names.py`) persist across mutation and crossover so stream
viewers can follow specific lineages. Stale-best restart fires when the
all-time best hasn't improved in `stale_gens_before_restart` gens (set to
0 to disable, useful when the elite needs uninterrupted PPO time).

### Curriculum (`src/training/curriculum.py`)
Stage progression driven by success rate. Stage transitions can swap the
start state so stage 2 drops agents into Dungeon 1, stage 3 into Dungeon 2,
etc. Depth memo (`checkpoints/depth_memo.jsonl`) tracks new-record RAM keys
for auto-promotion.

### Behavior cloning (`src/training/behavior_cloning.py`)
Supervised warm start from a recorded play file. Writes `.state.bin` binary
snapshots (same `NCST\x01` format) so the trainer can hand the run off
byte-exact. Used as an optional stage 0 before the GA starts exploring.
Multi-demo aware: pass a list (or a directory) of `.state.bin`/`.fm2` files
and the pipeline replays each from a cold-boot env, concatenating the
(state, action, reward) tuples for richer state coverage. BC seed cache
key includes encoder name + frame_skip so swapping architectures auto-
invalidates the cache.

### Reward functions (`src/utils/reward_functions/__init__.py`)
Thin Python dispatcher (~40 LOC) that calls `nes_core.build_reward_function`
with the game profile. All six games implemented in Rust (`rewards.rs`).
Byte-exact parity with the old Python implementations was verified
pre-deletion. Mario's reward function ships with **dense progress
checkpoints** — RAM-readable bonuses fire once each at hand-picked
x-positions through 1-1 (past first pipe, past first pit, ..., near flag).
Enable per-profile via `reward_weights.checkpoint_scale: 1.0`; set to 0 to
disable.

### Training dashboard (`src/gui/training_dashboard.py`)
Single-pane observer window. Reads `checkpoints/metrics.jsonl` directly
(no trainer-side queue). Panels: best/avg fitness, reward signal stack,
PPO learning telemetry (loss/policy/value/entropy), depth + curriculum
success, world-model losses (Dreamer mode only), recent depth records,
recent highlight clips, replay-buffer fill, and a live world-model
reconstruction strip showing original vs. decoded frames. Replaces the
older standalone `MetricsWindow` as the primary "watch learning happen"
surface.

## Audio

`nes_core::audio::AudioMixer` owns a single cpal Core Audio stream,
stereo (`channels: 2`), 44100 Hz. Every worker pushes PCM into a per-instance
ring; the cpal callback drains all rings, applies the 5-channel pan matrix,
and sums into the stereo output.

- Resampler: per-channel linear interpolation from the APU's native
  43653 Hz up to 44100 Hz.
- Pan matrix (`PAN_MATRIX[5]`): pulse 1 slight-left, pulse 2 slight-right,
  triangle center, noise center, DMC slight-left.
- Solo mode: `AudioMixer::solo(worker_id)` mutes every other worker's
  contribution with a 5 ms fade; the GUI's audio-mixer window drives this.
- Worker ring cap: ~150 ms of samples (~6.5 KB per worker) so fast-forward
  emulation cannot grow audio buffers unboundedly.

The Python side (`src/audio/ram_music.py`) is a thin façade — 147 LOC down
from 807 pre-migration. No chiptune synth or NSF playback; real APU audio
only.

## GUI

`src/gui/main.py` is the app controller. On start it constructs the
`Pool`, spawns the in-process trainer thread, and opens the selected
windows.

- `main_window.py` — ROM picker, start state, profile, reward overrides,
  instance/population/episode knobs, live metrics readout. Trainer
  dropdown selects `(profile)` / `GA + PPO` / `DreamerV3` and overrides
  the profile's `training_mode` at launch. Multi-pick BC demo picker
  (⌘-click) feeds a colon-joined list to the trainer's BC pipeline.
- `training_dashboard.py` — unified observer view (replaces the older
  `MetricsWindow` as the primary surface). Five plot panels +
  side-pane with replay-fill bar, depth records list, highlight clips
  list, and a live world-model reconstruction strip.
- `emulator_grid.py` — NxN live frame grid. Pulls from the pool's frame
  callback at 30 fps.
- `metrics_window.py` — legacy three-panel metrics view, still
  importable but no longer auto-opened on Start.
- `reward_tuning_window.py` — live sliders for reward weights. Changes
  apply at the next episode boundary.
- `audio_mixer_window.py` — per-instance mute/solo + master volume.
  Maps directly to `nes_core.AudioMixer` modes.
- `play_window.py` — keyboard-driven single-emulator session; writes
  `NCST\x01` binary snapshots via `env.save_state()`.
- `replay_window.py` — loads a checkpoint, drives one emulator with the
  best genome's policy. Prefers Core ML / ANE inference when a paired
  `.mlpackage` exists (~0.08 ms/fwd) else PyTorch MPS (~0.65 ms/fwd).
- `highlight_recorder.py` — per-worker ring buffer; flushes the last 4
  seconds to `highlights/*.mp4` when a banner event fires.

## Validation harnesses

Five layers of test, each catching a different class of bug. They form
a pyramid: tighter, narrower tests at the top; broader, looser tests at
the base.

```mermaid
graph TD
    A["nestest CPU validation<br/>nes_core/tests/nestest_validation.rs<br/>8991 instructions byte-exact vs Nintendulator log<br/>(PC + opcode + asm + A/X/Y/P/SP + CYC)"]
    B["Per-opcode cycle audit<br/>nes_core/tests/opcode_cycle_audit.rs<br/>56 opcodes verified against LaiNES OPERATION_CYCLES"]
    C["Byte-exact ROM fleet<br/>tests/parity/test_lockstep_baseline.py<br/>5 baseline games at 600 idle frames<br/>tests/parity/test_long_soak_byte_exact.py<br/>10 games at 2400 idle frames"]
    D["Library-wide bucket guard<br/>tests/parity/test_library_buckets.py<br/>~440 ROMs gated by sweep-derived ceilings<br/>(byte_exact, tight, moderate, loose, wide)"]
    E["Library-wide playability sweep<br/>scripts/playability_sweep.py<br/>~794 ROMs cold-boot + Start tap<br/>(advances / noisy / frozen / crashed)"]

    A --> B
    B --> C
    C --> D
    D --> E

    classDef tight fill:#cfe,stroke:#393
    classDef broad fill:#fec,stroke:#963
    class A,B,C tight
    class D,E broad
```

Layer responsibilities:

- **nestest** is the CPU spec gate. Any opcode change re-runs this
  first. The test is in Rust so it runs in seconds without Python.
- **opcode cycle audit** is the cycle-accuracy gate for the official
  6502 timing tables. Catches off-by-one cycle bugs that nestest
  doesn't exercise.
- **byte-exact fleet** lockstep N idle frames against `nes-py` for the
  five known-good baseline games. The strictest end-to-end test we
  have on the integration of CPU + PPU + APU + mappers.
- **library buckets** stops a ROM regressing by more than one bucket
  vs the committed `parity_sweep.json` baseline. Catches changes that
  silently break a class of mappers without landing on a baseline ROM.
- **playability sweep** is the loosest gate: cold-boot a ROM, idle to
  title, multi-tap Start, observe RAM change. Catches games that
  *boot* (parity_sweep tests that) but *don't progress* (this catches
  that). The "crashed" bucket — PC ends in $FFE0-$FFFF with frozen
  zero-page — was the specific signal that exposed the Bill & Ted's
  MMC1 RMW bug and the `roms/zelda.nes` PRG-RAM nibble bug.

When a "ROM looks broken" report arrives, run them in order: nestest,
then opcode audit, then byte-exact, then bucket, then playability. The
first one that fires names the bug-class layer.

## Cross-references

- `README.md` — top-level overview.
- `nes_core/SECURITY.md` — full `unsafe` audit, FFI error mapping, save
  state format, ASM fuzz results.
- `nes_core/KNOWN_ISSUES.md` — per-mapper quirks and open bugs.
- `docs/proposals/unified_rust_plan_v3.md` — the authoritative roadmap to
  public release.
- `docs/proposals/archive/hot_path_baseline.md` — measured bottleneck percentages.
- `docs/proposals/pgo_results.md` — PGO measurement writeup.
- `reports/full_library.md` — ROM compatibility matrix.
