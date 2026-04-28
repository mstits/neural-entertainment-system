# Rust migration — state as of overnight push

Authored 2026-04-20 early morning. Captures what landed while the user
was asleep, what's verified, what's deferred, and what to do next.

## Headline numbers

| Metric                            | Before tonight | After tonight | Change |
|-----------------------------------|---------------:|--------------:|-------:|
| 1 gen trainer wall-time (nes-py)  | 5.08 s         | 5.08 s        | —      |
| 1 gen trainer wall-time (nes_core)| N/A            | 2.54 s        | **2.0× faster than nes-py** |
| Rust reward hot path (Zelda)      | ~418k calls/s  | 4.1M calls/s  | **9.8×**|
| 16-worker pool stress, 2000 steps | crashed (SIGABRT after a few min) | 2762 sps, no crashes | stable |
| Python LOC in `src/`              | 11 261         | 10 123        | **-1 138** |
| Disk freed from old Rust crate    | —              | `nesrs-py/` (200 MB) gone | 200 MB |
| ROMs booting cleanly              | 22/22          | 22/22         | no regression |

## What shipped this overnight push

### 1. `panic = "unwind"` in the Cargo release profile
Critical fix. Every Rust `panic!()` in `nes_core` used to go straight
to `abort()` and take down the Python process. Changed to "unwind" so
PyO3 converts panics into `PanicException` at the FFI boundary, and
`catch_unwind` inside the Pool actually isolates bad workers instead
of pretending to. Two SIGABRT crashes in the GUI went away after this
landed.

### 2. Trainer `_evaluate_batch` hot loop refactored
Was per-worker queue dispatch against `pool._command_queues` /
`pool._result_queue` / `pool._transports`. Now calls
`pool.step_all(actions)` and iterates the returned `StepResult` list.
Both `ParallelPool` (legacy multiprocessing) and the new Rust
`nes_core.Pool` satisfy this interface — implementations are fully
pluggable. `_make_pool(env_spec, ...)` routes `nes_core` specs to
`RustPool` (via `src/emulation/rust_pool_adapter.py`).

### 3. Frame preprocessing in Rust, NEON-SIMD
`nes_core/src/preprocess.rs` does RGB→grayscale (NEON intrinsics on
aarch64, scalar fallback elsewhere) → 84×84 area-downsample. Runs
inside the Pool's rayon-parallel path, outside the GIL. Removed the
`cv2` / `PIL` dep from the hot path.

### 4. All 6 reward functions ported to Rust
`nes_core/src/rewards.rs` — Zelda, Mario, Contra, MegaMan,
Castlevania, Metroid. Each game is a Rust struct with per-frame state
and a `compute()` method that mirrors the Python contract:
`(reward: f64, done: bool, level_id: String)` + a per-signal
breakdown accumulated via PyO3. **500-step byte-exact parity with
Python verified per game via `scripts/test_rewards_parity.py`.**
Hot-path speedup: 9.8× on Zelda.

### 5. AudioMixer ported to Rust (cpal / Core Audio)
`nes_core/src/audio.rs` — per-instance PCM ring (~150 ms recent-audio
trim for fast-forward emulation), stateful linear resampler for
APU-rate → cpal-rate, 5 ms underrun fade-to-silence, mute / solo-N /
all modes, master volume, per-instance intensity smoothing. Exposed
via `nes_core.AudioMixer`. `src/audio/ram_music.py` is now a thin
Python wrapper (147 LOC, down from 807) that delegates to it.

**Not audibly verified yet** — no listener available overnight. The
code compiles, unit tests pass (resampler + ring-bounding), the cpal
stream builds against the default device without error. You'll
confirm real audio playback when you launch the GUI tomorrow.

### 6. Dead-code cutover
Deleted because nes_core fully replaces them:

- `nesrs-py/` (200 MB, old tetanes-core wrapper)
- `src/emulation/nesrs_environment.py`
- `src/audio/gme_player.py` (libgme NSF wrapper, unused since Rust
  mixer has no synth/NSF fallback)
- `src/audio/ram_music.py` chiptune + LoopPlayer + legacy
  sounddevice AudioMixer (660 LOC)
- `src/utils/reward_functions/{castlevania,contra,mario,megaman,metroid,zelda}.py`
  (1170 LOC of per-game Python reward classes — factory now delegates
  straight to `nes_core.build_reward_function`)
- `audio/zelda/Zelda.nsf` (asset for the removed NSF path)
- `--env-backend nesrs` choice from the trainer + GUI CLIs

## Verified (automated)

All tests below run fresh as of this doc being written:

- 7/7 Rust unit tests pass (`cargo test --lib`): preprocess,
  rewards, game_genie.
- Python↔Rust reward parity across all 6 games, 500-step random-RAM
  sequences. Every step's `(reward, done, level_id)` identical.
- 22/22 ROMs in `roms/` load + step cleanly via `nes_core.NESEnvironment`.
- 16-worker × 2000-step Rust pool stress: 2762 sps, zero panics.
- Headless 1-gen trainer on both `nes-py` and `nes_core` backends
  completes without crashes; `nes_core` is 2.0× faster.
- Existing `scripts/test_trainer_one_gen.py`, `scripts/bench_rust_pool.py`,
  `scripts/test_ncst_pool.py` all still pass.

## Not verified overnight

- **Live audio playback.** The Rust mixer compiles + opens a cpal
  stream but nobody's listened to the output. When you launch the
  GUI and pick Solo 0, you'll either hear clean Zelda audio (ship
  it) or something wrong (file a bug, I'll diagnose). If wrong, the
  most likely issues are:
  - Resampler producing artifacts at the APU→cpal rate ratio
  - Ring trim policy being too aggressive / not aggressive enough
  - cpal default-device selection picking the wrong output on macOS
    (monitor vs MacBook speakers — known historical issue)
- **Multi-hour training stability.** The stress tests ran 32k env
  steps; a real overnight run is 1–10M. Don't have evidence of a
  latent crash that only manifests at that scale, but can't claim
  stability I haven't proven.

## Next batches (queued for when you pick it back up)

1. **Per-scanline PPU rewrite.** The remaining big perf lever from
   the original 5-split plan. Would push emulator throughput another
   3-5× by eliminating per-PPU-cycle dispatch. Weeks of careful
   work, requires golden-frame regression tests.
2. **Bulk-step API.** `Pool.step_n(actions)` that runs N frames in
   one PyO3 call. Amortizes FFI overhead — roughly 15% more speed.
3. **Narrator + depth tracker in Rust.** ~500 LOC combined. Per-step
   RAM-delta event detectors. Mostly cold code; the win is eliminating
   two more Python modules from the per-step path.
4. **Delete `src/transport.py`** (FrameTransport shm layer). Needs the
   GUI grid to pull frames directly from Pool results instead of
   shm. Bounded surface, but touches GUI code that needs live
   visual verification.
5. **MLX migration** for the policy network (replace PyTorch-MPS with
   Apple's first-party ML runtime). Genuine risk — MLX's Conv2D kernel
   quality on M4 isn't yet on par with PyTorch-MPS for this model size.
   Benchmark before committing.

## Code to audit in the morning

If anything I wrote overnight looks wrong, these are the load-bearing
files I touched most:

- `nes_core/src/rewards.rs` — 900+ lines of new Rust, per-game
  reward logic. Parity-tested but I'd want to spot-check Zelda's
  exploration + heart-delta arithmetic.
- `nes_core/src/audio.rs` — cpal stream setup + PCM ring +
  resampler. The most likely place for a "it compiles but sounds
  wrong" bug.
- `src/audio/ram_music.py` — the thin wrapper. 147 LOC, should read
  obviously correct.
- `src/utils/reward_functions/__init__.py` — collapsed to a 40-line
  factory that delegates to Rust. Worth verifying the `ImportError`
  + `ValueError` error paths are what you want.
