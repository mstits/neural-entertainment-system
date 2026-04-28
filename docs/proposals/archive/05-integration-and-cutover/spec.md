# Split 05 — Integration and Cutover

## Context

Source proposal: [`../full_rust_refactor.md`](../full_rust_refactor.md)
Interview decisions: [`../deep_project_interview.md`](../deep_project_interview.md)
Manifest: [`../project-manifest.md`](../project-manifest.md)

The big-bang integration. Everything that needs the new core to be
feature-complete lands here: save-state byte API, perf pass (bulk-step,
zero-copy frames, SIMD palette), the PyO3 wrapper at the existing
`NESEnvironment` shape, and the cutover that deletes nes-py, nesrs,
gme_player, NSF assets, the `--env-backend` flag, and the iNES
header sanitizer.

This is the longest and most cross-cutting split because "done" means
"pytest tests/ green with the new core as the only backend, Zelda
training run reaches gen 100, real APU audio audible."

## Key decisions inherited from interview / proposal

- **Hard cutover.** Both old backends are deleted in this split, not
  parked behind a flag. No transition period.
- **No save-state migrator.** Old `.state.bin` files in
  `checkpoints/auto_curriculum/` are wiped at cutover; they're
  early-experimental data, not durable assets.
- **PyO3 0.22 + maturin + abi3-py39** for the wrapper, matching
  current nesrs build setup.
- **Drop-in NESEnvironment shape:** `reset`, `step`, `get_audio`,
  `sample_rate`, `save_state`, `load_state`, `FRAME_WIDTH`,
  `FRAME_HEIGHT`. Trainer / pool / GUI Python code does not change.
- **Codebase reduction target:** ~2000-2500 LOC net deletion.

## Deliverables

### Core perf pass (in `nes_core/`)

1. **Bulk-step API:** `step_n(actions: &[u8]) -> (frames, audio, dones)`
   so one PyO3 call covers N frames. Frame buffers reused across the
   batch; audio is a single concatenated buffer. Internally calls
   `step_no_render` for non-observed frames (gated by frame_skip from
   the wrapper) and `step` only for the final observed frame in each
   batch — this is the single biggest perf win the new core enables.
2. **Zero-copy frame export** via PyO3's buffer protocol — return a
   numpy array sharing the underlying Rust buffer instead of cloning
   per call.
3. **NEON SIMD palette → RGB conversion** on Apple Silicon (with a
   portable fallback for other arches). 4×-8× win on the palette pass.
4. **macOS Accelerate (vDSP) hooks** for any audio resampling or FFT
   work in the APU's downsample path. Apple's vDSP routines beat naive
   loops by 5-10× on short signal blocks.
5. **APU gating wired to the AudioMixer subscriber count.**
6. **Profile-Guided Optimization build profile** for release wheels:
   instrumented build → run a representative training session →
   re-build with PGO data. ~10-20% additional speedup on hot paths.

### Python-side companion: async inference pipeline

7. **Wire `game_profile["async_pipeline"]`** so the trainer computes
   action_{t+1} on MPS while the worker steps action_t. Currently
   stubbed in `Trainer.__init__` (`self.async_pipeline = bool(
   game_profile.get("async_pipeline", False))`). Implementing this
   in `_evaluate_batch` overlaps the GPU forward pass with the CPU
   emulator step, ~1.5-2× combined-loop speedup on top of everything
   the core delivers.

### Save-state API

5. **`save_state(&self) -> Vec<u8>`** producing a versioned, in-memory
   snapshot. Header includes `magic` + `version` + ROM CRC.
6. **`load_state(&mut self, bytes: &[u8]) -> Result<()>`** verifying
   header, version, and ROM CRC; rejects mismatched ROM with a
   descriptive error.
7. **No tempfile round-trip** — the bytes API is direct in/out, unlike
   the current nesrs path that hits the filesystem twice per snapshot.

### PyO3 wrapper (in `nes_core/python/`)

8. **`NESEnvironment` Python class** matching the existing shape used
   by `src/emulation/nes_environment.py` and
   `src/emulation/nesrs_environment.py`. Drop-in replacement; no
   trainer code changes.
9. **Maturin build hooks** producing a wheel installable into `.venv/`.

### Cutover (in `src/`)

10. **Replace `env_spec` defaults** in `src/training/trainer.py` and
    `src/emulation/parallel_pool.py` to point at the new
    `nes_core.NESEnvironment`. Single backend.
11. **Delete:**
    - `nesrs-py/` (entire crate)
    - `src/emulation/nes_environment.py` (the nes-py wrapper) and
      remove the `nes-py` PyPI dep from `requirements.txt`
    - `src/emulation/nesrs_environment.py`
    - `_sanitize_ines_header` + its callers
    - `src/audio/gme_player.py`
    - `audio/<game>/*.nsf` files
    - The synth + NSF fallback paths in `src/audio/ram_music.py`
      (keep the PCM ring + AudioMixer; delete `_ChiptuneGenerator`,
      `LoopPlayer.set_song` NSF/synth branches, `_load_nsf_track_map`,
      `gme_player` imports). Mixer becomes PCM-only.
    - `--env-backend` CLI flag from `src/gui/main.py` and
      `src/training/trainer.py`
    - The dual `env_spec` branch in `parallel_pool.py`

### Validation

12. **`pytest tests/` passes end-to-end** with the new core as the
    only backend. Update `test_parallel_pool.py` and other env-spec
    references.
13. **Zelda training smoke run reaches gen 100** without worker
    crashes. Logged in `checkpoints/metrics.jsonl`.
14. **4-worker bench shows ≥720 steps/s** (≥2× current nes-py).
15. **GUI Solo-0 mode produces audible game audio** end-to-end with
    no NSF / synth in the code path.
16. **`docs/rust_nes_core.md` rewritten** to describe the new core
    instead of the deleted tetanes-vs-nes-py comparison.

## Dependencies

- **Provided by 01:** core crate scaffolding, CPU, bus, iNES loader.
- **Provided by 02:** working PPU + frame buffer accessor.
- **Provided by 03:** all top-10 mappers booting target ROMs.
- **Provided by 04:** APU + `drain_audio` accessor.

This split cannot start until 01-04 are substantially complete.

## Provides to other splits

None — this is the terminal split. Project complete when this lands.

## Risks for this split

- **Existing test references both backends.** Some tests
  (`test_parallel_pool.py`) explicitly take an `env_spec` argument;
  cutover means updating them. Others may import from the deleted
  modules. Plan a grep pass over `tests/` early in this split.
- **The `.state.bin` wipe surprises someone.** Document loudly in
  the cutover commit message; mention in `docs/rust_nes_core.md`'s
  rewrite.
- **Perf target miss.** If bulk-step + zero-copy + SIMD don't get to
  ≥720 steps/s, evaluate whether to ship anyway (we still gain audio
  + integrity + a single backend) or extend the perf pass.
- **Audio routing through `AudioMixer.push_audio` might surface a
  bug** that was masked by the synth fallback always-having-audio.
  Test the PCM-only path explicitly before deleting the synth code.
- **Apple-Silicon-only NEON intrinsics** make the build harder on
  other arches. Use `#[cfg(target_arch = ...)]` + portable fallback;
  CI on x86_64 should still build.

## Acceptance criteria

1. The codebase has exactly one emulator backend. `nesrs-py/`,
   `gme_player.py`, NSF assets, and `--env-backend` are gone from
   the tree (verified via `git log` + `find`).
2. `pytest tests/` passes. `pytest tests/test_parallel_pool.py` uses
   the new `nes_core.NESEnvironment` via the standard env_spec.
3. A fresh Zelda run launched via the GUI reaches gen 100 without
   worker crashes. `checkpoints/metrics.jsonl` shows monotonically-
   advancing generation rows.
4. `python scripts/bench_backends.py` (rewrite to remove the dual-
   backend A/B; just measure throughput on the new core) reports
   ≥720 steps/s @ 4 workers.
5. With the GUI's audio mixer set to Solo-0, real ROM audio is
   audible (verified by the user pressing Start, hearing Zelda's
   title theme).
6. `docs/rust_nes_core.md` has been rewritten to describe the new
   core's architecture, supported mappers, perf characteristics,
   and the deleted-on-cutover surface.
7. Net `git diff --stat` shows ≥2000 LOC removed in `src/` Python +
   the deleted `nesrs-py/` crate, after accounting for the new Rust
   core's additions in `nes_core/`.
