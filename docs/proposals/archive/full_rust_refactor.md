# Full Rust Refactor: Single Purpose-Built NES Core

## Project description

Replace both current NES emulator backends (`nes-py` wrapping LaiNES C++, and
`nesrs` wrapping tetanes-core Rust) with one new purpose-built Rust NES core,
designed from day one for reinforcement-learning training throughput AND
faithful APU audio AND clean save-state. The Python training pipeline
(GA + PPO + BC, GUI, narrator, highlights, depth tracker) stays in Python.
Only the emulator layer changes, but completely.

This is the "rip the bandaid" path. We accept 7-12 weeks of solo effort to
collapse two backends into one, eliminate the audio/speed tradeoff, gain
memory safety, and end the architectural drift caused by maintaining two
incompatible emulator wrappers with different capability surfaces.

## Why now

Current state has produced a sustained set of pain points:

- **nes-py is fast (~363 steps/s @ 4 workers) but has no APU.** It was built
  for RL training, not faithful emulation. We hacked around the missing audio
  with a libgme NSF playback path driven by reading a known RAM byte
  (`$0605` for Zelda) — fragile, no SFX, mapping-dependent, "awful" per the
  user when actually heard.
- **nesrs (tetanes-core) is faithful but 2.5× slower (~143 steps/s).** Built
  hoping it would be faster than nes-py; benchmarks proved otherwise (see
  `docs/rust_nes_core.md` for the measured numbers). Its only unique
  strengths today are real APU audio and a `save_state` RPC.
- **Two backends doubles the maintenance surface.** The pool, trainer,
  environments, header sanitizer, audio mixer, CLI flags, and tests all
  branch on backend. Recent example: tetanes rejects ROMs with non-zero
  reserved iNES bytes 7-15, requiring a Python-side header sanitizer that
  nes-py doesn't need.
- **The user's goal is unattended Switch streaming with real game audio
  AND watchable training progress.** Today this requires picking either
  speed (silent) or audio (slow). One core that does both removes the
  forced choice.
- **C++ memory-safety risk.** LaiNES is single-author, ~9 years old,
  unaudited. A new Rust core eliminates a class of memory-safety bugs and
  undefined behavior on malformed ROMs.

## Goals

### Performance

- **Target: ≥5× current nes-py throughput** (≥1800 steps/s @ 4 workers;
  stretch ≥3000 steps/s). Hard ceiling on emulator-only changes is ~2×
  due to Amdahl (ML inference is 35% of the loop), so the 5×+ target
  requires the skip-render trick + async pipeline below.
- **Skip PPU rendering on non-observed frames.** With `frame_skip=4`,
  only 25% of frames need a rendered output. The other 75% can run
  CPU + APU + memory only, skipping the entire PPU rendering path
  (which is ~half of emulator wall time). Single biggest win, gated
  by a `step_no_render(action)` API alongside `step()`.
- **Async inference pipeline** (Python-side, paired with this refactor):
  compute action_{t+1} on the GPU in parallel with the worker stepping
  action_t. Already stubbed via `game_profile["async_pipeline"]`.
  Doubles the effective loop because the GPU forward pass and the
  emulator step run concurrently rather than serially.
- **Achieve target speed by deliberate non-cycle-accurate design:**
  per-scanline PPU rendering (not per-pixel), straightforward 6502
  interpreter, NEON SIMD palette conversion on Apple Silicon (with
  portable fallback), release-LTO + `-C target-cpu=native` builds,
  Profile-Guided Optimization on a representative training run.
- **Bulk-step API**: `step_n(actions: &[u8]) -> (frames, audio, dones)`
  so one PyO3 call covers N frames. Eliminates per-frame FFI overhead.
  The non-observed frames inside the batch use the no-render fast path
  automatically.
- **Zero-copy frame export** into numpy via the PyO3 buffer protocol —
  current nesrs allocates per call.
- **APU gating**: skip APU mixing entirely when the AudioMixer reports
  no active subscriber. Channel state machines still tick so re-enabling
  produces continuous audio.
- **rayon for the genuine parallelism inside the core** (e.g. parallel
  scanline rendering when more than 4 workers are stepping at once).
  Per-instance the core stays single-threaded; rayon only kicks in
  for embarrassingly-parallel batched work.
- **macOS Accelerate framework** for any audio resampling / FFT-style
  work. Apple's vDSP routines beat naive Rust loops by 5-10× on
  short signal blocks.

### Audio

- **Real APU output published every step**, not driven from RAM byte
  guesswork. Same code path as nesrs already exposes (`get_audio()` →
  `transport.publish(audio=...)` → `AudioMixer.push_audio()`), but now at
  training speed not 0.39× of it.
- **Stereo support** (current nesrs is mono).
- **Configurable sample rate** (default 44100, retain).

### Visual

- **256×240 RGB frame export** matching the existing transport layout.
- **Optional sub-resolution rendering** for headless tiles (e.g. publish
  84×84 directly when no GUI subscriber, eliminating one preprocess
  step). Coordinates with the "skip video on headless tiles" Python win.
- **Deterministic palette** (configurable: NES classic, modern,
  user-supplied .pal file).

### Save-state / integrity

- **In-memory save_state/load_state byte API.** No tempfile round-trip
  (current nesrs hits the filesystem twice per snapshot — kills hot
  paths).
- **Versioned save-state format** so we can evolve without silently
  corrupting old snapshots. Header includes core version + ROM CRC.
- **Header sanitizer built into ROM load** — accept iNES 1.0 ROMs with
  garbage in reserved bytes 7-15, the same way LaiNES does. No external
  Python sanitizer needed.

### Security / correctness

- **Memory-safe Rust everywhere.** No raw C++ pointer arithmetic in the
  6502/PPU/APU.
- **No panics across the FFI boundary.** All errors surface as Python
  exceptions with structured info.
- **Reject malformed ROMs cleanly** with descriptive errors, not
  silent corruption.
- **Test against nestest, blargg's PPU/APU/CPU tests, and a regression
  suite of "boots correctly + ~1k steps don't diverge" for the games
  this repo cares about (Zelda, Mario, Castlevania, Contra, Metroid,
  MegaMan series).**

### Codebase reduction

When the new core ships, delete:

- `nesrs-py/` (entire crate — replaced).
- `src/emulation/nesrs_environment.py` adapter (replaced by new wrapper).
- The header sanitizer in `_sanitize_ines_header` (built into core).
- `src/audio/ram_music.py` NSF/synth fallback paths (real audio works now).
- `src/audio/gme_player.py` libgme wrapper (no longer needed).
- `audio/<game>/*.nsf` files (no longer used).
- `--env-backend` CLI flag from `src/gui/main.py` and `src/training/trainer.py`.
- The dual env_spec branch in `parallel_pool.py`.

Estimated lines deleted: ~2000-2500.

## Non-goals

- **Cycle-accurate emulation.** We're explicitly trading accuracy for speed.
  Games that require cycle-accurate timing (e.g. Battletoads' raster
  effects, some MMC5 demos) are out of scope.
- **Mapper completeness.** Top 10 mappers (NROM, MMC1, UxROM, CNROM, MMC3,
  MMC5-basic, AxROM, ColorDreams, GxROM, MMC2) covers ~95% of the games
  in `roms/`. Other mappers refuse to load with a clear error message.
- **VS System / PlayChoice-10 / NES 2.0 special ROMs.** Out of scope.
- **Replacing PyTorch / MPS for the ML path.** Native Rust ML on Apple
  Silicon is not competitive with PyTorch-MPS today (`burn-wgpu` 2-5×
  slower on Conv2D, `candle` no MPS). The ML stack stays in Python.
- **Backwards compatibility with existing `.state.bin` files.** Old
  snapshots from nesrs and the curriculum auto-promotion directory
  (`checkpoints/auto_curriculum/`) will break. We accept this; current
  snapshots are early-experimental data, not durable assets.

## Constraints

- **Apple Silicon (macOS arm64) is the primary build target.** Linux
  x86_64 is a secondary nice-to-have; Windows is out of scope for this
  refactor.
- **Python 3.11 venv** (`.venv/bin/python`); bindings via PyO3 0.22 +
  maturin + abi3-py39 for cross-version wheels.
- **Existing trainer surface stays unchanged.** The Python NESEnvironment
  class shape (`reset`, `step`, `get_audio`, `sample_rate`, `save_state`,
  `load_state`, `FRAME_WIDTH`, `FRAME_HEIGHT`) is the contract — the new
  Rust core must be a drop-in for both nes-py and nesrs at that
  interface. Trainer / pool / GUI code does not change.
- **No new Python dependencies for end users.** Wheel ships with the
  compiled `.so`; only PyO3 build-time deps are new.
- **Tests must pass under `pytest tests/` end-to-end** including
  `test_parallel_pool.py`, `test_emulator_env.py`, the curriculum
  promotion tests, and the daily-recap script.

## Existing work to absorb

The following infrastructure stays and will be exercised by the new core:

- `src/emulation/parallel_pool.py` — multiprocessing pool with command /
  result queues, FrameTransport-based shm publish, save_state RPC,
  demo-worker slot 0. The new core plugs into the existing `env_spec`
  resolution with no pool changes.
- `src/transport.py` — version-2 SHM header with audio slot. New core
  publishes through this unchanged.
- `src/audio/ram_music.py` `AudioMixer.push_audio()` PCM path — keep this
  half (the half that works), delete the synth/NSF fallback half.
- `src/training/trainer.py` audio drainer thread + reopen-on-mode-change
  fixes (this session) — preserved.
- `src/training/depth_tracker.py`, narrator, highlight recorder, daily
  recap — all unchanged.

## Risks

| Risk                                      | Mitigation                                                                 |
|-------------------------------------------|----------------------------------------------------------------------------|
| New core has timing/sprite/scrolling bugs | Test against blargg's reference ROMs + golden-frame regression on Zelda overworld scroll, dungeon transitions. |
| Mapper outside top 10 needed mid-build    | Mapper trait is pluggable; can add MMC2/4 etc. as discovered. Worst case: training ROM swaps to a supported one. |
| 2× speedup target not met                 | Baseline measure at end of CPU+PPU phase; if <1.5×, evaluate whether to ship anyway (still simpler than two backends) or extend optimization. |
| `.state.bin` migration breaks workflows   | Acceptable per goals; wipe `checkpoints/auto_curriculum/` at cutover, regenerate from new runs. |
| 7-12 week solo effort overruns            | Phase gates (CPU+PPU mvp, then mappers, then APU, then save_state, then integration); can ship partial wins. |
| ROM-incompatibility regressions on games we don't actively use | Out of scope per non-goals; document supported mapper list. |

## Phases (ballpark)

The ordering matters: the trainer can't run without CPU+PPU+at least
one mapper, but APU and save_state can land later.

1. **CPU (6502) + memory bus + iNES loader.** 1-2 weeks.
2. **PPU renderer (background, sprites, priority, sprite-0 hit).** 2-3 weeks.
3. **Top 10 mappers (NROM through MMC3 and the MMC5 basic case).** 1-2 weeks.
4. **PyO3 wrapper at the existing NESEnvironment shape.** 0.5 weeks.
5. **APU (5 channels: 2 pulse, triangle, noise, DMC).** 1-2 weeks.
6. **Save-state byte API + versioned format.** 0.5 weeks.
7. **Bulk-step API + zero-copy frame export + SIMD palette.** 1 week.
8. **Test harness: blargg + golden-frame regression + game smoke tests.** 1-2 weeks.
9. **Integration / cutover: replace nes-py and nesrs in the pool, delete dead code, validate end-to-end training run on Zelda.** 0.5-1 week.

**Total realistic: 9-13 weeks solo. Stretch goal: 7 weeks if everything
goes well.**

## Success criteria

The refactor is done when:

1. A fresh `pytest tests/` passes end-to-end with the new core as the
   only emulator backend.
2. A Zelda training run reaches gen 100 without worker crashes.
3. Measured throughput on the 4-worker bench script ≥2× current nes-py
   (≥720 steps/s).
4. Real APU audio is audible from the GUI on Solo-0 mode without any
   NSF / synth fallback in the code path.
5. The auto-curriculum pipeline writes and loads `.state.bin` snapshots
   without filesystem round-trips.
6. The codebase has one emulator backend, one set of tests, one
   environment adapter. `nesrs-py/`, `gme_player.py`, the NSF assets,
   and the `--env-backend` flag are gone from the tree.
7. `docs/rust_nes_core.md` is rewritten to describe the new core, not
   the old tetanes-vs-nes-py comparison.
