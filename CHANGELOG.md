# Changelog — Neural Entertainment System (NES)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

In this changelog, **NES** refers to this project (Neural Entertainment
System). The original Nintendo Entertainment System hardware is named in full.

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
