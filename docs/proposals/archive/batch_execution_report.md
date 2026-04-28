# 11-Batch Rust Migration — Execution Report

Authored 2026-04-20. Covers the session that executed the 11-batch
plan in `final_rust_plan.md`. What landed, what deferred, and why.

## Headline

| Metric                            | Before session | After session | Δ                 |
|-----------------------------------|---------------:|--------------:|-------------------|
| `src/` Python LOC                 | 10,123         | **8,692**     | **−1,431 LOC**    |
| `nes_core/src/` Rust LOC          | 11,819         | **12,866**    | +1,047 (narrator + depth_tracker added; disassembler −649) |
| Legacy modules deleted (py)       | 0              | **5**         | nes_environment, demo_worker, 3 legacy tests |
| Rust hot-path modules added       | 0              | **2**         | depth_tracker, narrator |
| Rust dead code removed            | 0              | **1**         | disassembler.rs (649 LOC, forked from upstream, zero callers) |
| Rust unit tests                   | 7              | **15**        | +8 (depth_tracker 4, narrator 4) |
| Cargo build warnings              | 95             | **0**         | edition-2024 unsafe-op-in-unsafe-fn lint allowed crate-wide; stale unsafe blocks dropped; unused imports removed |
| `nes-py` on the emulator path     | fallback       | **removed**   | single Rust backend |
| `cv2` / `PIL` in the emulator     | present        | **removed**   | NumPy-only helper in src |
| Save-state on-wire format         | naked bincode  | **versioned** | `NCST\x01` prefix rejects mismatched versions |

## Batch-by-batch

### ✅ Batch 1 — Retire `nes-py` backend entirely

- Deleted `src/emulation/nes_environment.py` (543 LOC, nes-py wrapper).
- Extracted `FrameStacker`, `preprocess_frame`, `stack_frames`, NES
  button constants into `src/emulation/frame_utils.py` (134 LOC,
  pure Python, no emulator deps).
- Removed `nes-py>=8.2.0` from `requirements.txt`.
- Deleted `scripts/bench_backends.py` (A/B benchmark with nothing to
  A/B).
- Deleted legacy tests that covered the nes-py path:
  `test_nes_environment.py`, `test_nesrs_adapter.py`,
  `test_reset_stability.py`, `test_emulator_smoke.py`,
  `test_reward_functions.py` (targeted deleted Python reward classes).
- Reduced `--env-backend` CLI choices to `("nes_core",)` in trainer +
  GUI entry points.
- Updated every caller to import from `frame_utils` (trainer,
  behavior_cloning, replay_window, play_window, parallel_pool,
  demo_worker, diagnose, bench, tests).
- **Exit verified**: `grep -r "nes[_-]py" src/ tests/` returns only
  dev-tooling mentions; 1-gen trainer smoke passes in 2.54 s.

### ✅ Batch 2 (partial) — Delete demo_worker; trim ParallelPool

**Scope reduced from original plan.** The plan called for deleting
`ParallelPool` + `FrameTransport` entirely. That requires moving the
trainer into a thread of the GUI process (or otherwise removing
GUI↔trainer IPC), which is a bigger architectural change than
warranted in a single session. What landed:

- Deleted `src/emulation/demo_worker.py` (327 LOC) — the
  multiprocessing demo-streaming path. Not wired to nes_core.Pool,
  so it was effectively dead code on the production flow.
- Stripped demo-worker wiring from `parallel_pool.py` (−~100 LOC),
  `rust_pool_adapter.py`, `trainer.py` (`_bc_online_update`,
  `_drain_demo_pairs`, `_demo_buffer`, `--demo-worker` CLI),
  `gui/main.py`, `gui/main_window.py` (demo-worker checkbox + picker
  + recents entries), `gui/emulator_grid.py` (demo-tile indexing).
- Deleted `tests/test_demo_worker.py`.

**Kept for now (not removed)**:
- `src/emulation/parallel_pool.py` — still used by `FakeNES`-backed
  tests (`test_parallel_pool`, `test_trainer`, `test_soak`,
  `test_backtest`). Production flow never enters this path
  (`_make_pool` routes `nes_core:NESEnvironment` to `RustPool`).
- `src/transport.py` + `FrameTransport` — GUI↔trainer IPC still
  relies on shm for live frame display.

**Remaining work (future session):** move trainer from subprocess to
a thread in the GUI process; then FrameTransport and ParallelPool
can be deleted entirely. Tracked as a follow-up.

### ✅ Batch 3 — Narrator + depth_tracker to Rust

- New `nes_core/src/depth_tracker.rs` (190 LOC) — per-game RAM
  readers (Zelda / Mario / Generic), lexicographic max-key
  tracker. 4 unit tests pass.
- New `nes_core/src/narrator.rs` (276 LOC) — breakdown-delta event
  detector, per-(worker, kind) rate limiter, first-ever dedup,
  combo-kill tracker with configurable threshold. 4 unit tests
  pass.
- Both exposed via PyO3 as `nes_core.DepthTracker` and
  `nes_core.Narrator`.
- `src/training/depth_tracker.py`: 210 → **97 LOC** (−113).
  Thin wrapper delegating to Rust; Python handles JSONL memo file
  I/O only.
- `src/training/narrator.py`: 288 → **165 LOC** (−123).
  Thin wrapper delegating to Rust; Python handles caption
  templates + GUI queue push.
- All 4 pre-existing depth-tracker tests pass against the Rust-backed
  wrapper. Narrator smoke-tested end-to-end.

### ✅ Batch 7 — Narrator subprocess → thread

**No-op / already done.** Narrator has always run inline in the
trainer's thread, not as a subprocess. The `multiprocessing` import
in `narrator.py` was only for the `mp.Queue` type that the GUI
supplies as its output channel. Marked complete without code changes.

### ❌ Batch 4 — GA core ops to Rust (DEFERRED)

**Rationale**: GA state is PyTorch `state_dict` (dict of torch
tensors). Porting `crossover` / `mutate` / `tournament_select` to
Rust would require:

- Converting every tensor to flat `f32` buffers on every op call,
  then back to tensors for the next generation.
- Losing PyTorch's MPS acceleration on the actual tensor ops.
- Adding parity tests for RNG determinism across Python `torch.rand`
  and Rust `rand_chacha`.

GA runs once per generation (16 genomes × a handful of ops), not a
perf hot path. The round-trip cost likely makes it slower than the
current MPS path.

Keeping the port as a future option if MPS ever goes away (e.g. for
an MLX migration), at which point an all-CPU GA in Rust might win.

### ❌ Batch 5 — Bulk-step API `Pool.step_n` (DEFERRED)

**Rationale**: The existing `Pool::step_all` already uses the
skip-render fast path internally (`python.rs::step` passes
`tick_skip_render=true` to every frame except the last of the
`frame_skip` window). A true `step_n` that runs N `step_all` calls
in a single PyO3 hop would shave FFI overhead (~15-20% estimated)
but requires:

- Coordinating APU sample continuity across the N internal frames.
- Deciding how done flags + final-frame capture work when N > 1.
- Updating every caller (trainer, bench scripts, tests) to feed a
  `Vec<Vec<u8>>`.

Not worth the API-surface churn in this session. The
already-implemented skip-render does most of the per-frame work
reduction.

### ❌ Batch 6 — Per-scanline PPU rewrite (DEFERRED)

**Rationale**: Per the plan this is "3-5 days of careful work" with
golden-frame regression tests on multiple games. Requires dedicated
effort and visual verification — not a session-boundary batch. Left
as the biggest remaining perf lever.

### ✅ Batch 8 — Security / integrity hardening pass

- Audited every `unsafe` block in `nes_core` (23 total: 22 in
  preprocess.rs NEON intrinsics, 1 in pool.rs audio drain).
  Documented the safety argument for each in
  `nes_core/SECURITY.md`.
- Added **versioned save-state format**: `NCST\x01` magic prefix on
  every `save_state()` + `save_worker_state()` output. Load path
  rejects unknown version bytes with a structured `PyValueError`
  instead of silently deserialising into the wrong layout.
- Confirmed `panic = "unwind"` is still set (required for
  `catch_unwind` in `pool.rs`; was the root cause of the
  SIGABRT crashes the overnight push eliminated).
- Audited the FFI boundary — every PyO3 function returns
  `PyResult<_>` with structured `PyValueError` /
  `PyIndexError` / `PyRuntimeError` on malformed input. No silent
  corruption paths.
- `SECURITY.md` documents the audit, the two `unsafe` sites, the
  save-state versioning policy, and what's **not** audited (mapper
  code is forked from upstream RustedNES; rayon + cpal are trusted
  crates).

### ❌ Batch 9 — Live audio verification (DEFERRED — user action)

Needs user to launch the GUI, pick Solo-0, and listen to real Zelda
audio. cpal stream opens clean; unit tests pass; real-device
verification still blocked on user availability. Flagged for morning
review.

### ❌ Batch 10 — MLX migration (DEFERRED — stretch goal)

Needs a benchmark of MLX vs PyTorch-MPS on the specific 84×84×4 → 6
policy shape before committing. PyTorch-MPS is the proven path; no
reason to touch it without empirical evidence of a ≥1.2× win.

### ✅ Batch 11 — Documentation + test harness sweep

- `nes_core/SECURITY.md` added (Batch 8).
- This report (`docs/proposals/batch_execution_report.md`) captures
  what ran.
- `docs/proposals/final_rust_plan.md` remains authoritative for the
  overall target; this report supplements it with what actually
  executed and what deferred.
- `requirements.txt` updated to remove `nes-py`.
- `scripts/install_macos.sh` updated to build the Rust wheel via
  maturin + verify `nes_core` import instead of `nes_py`.

## Tests

All passing as of session end:

- **Rust unit tests**: 15/15 (depth_tracker 4, narrator 4, rewards 5,
  preprocess 2).
- **Python tests** (excluding legacy-env tests deleted in Batch 1):
  54/54 across test_depth_tracker, test_transport,
  test_genetic_algorithm, test_behavior_cloning, test_curriculum,
  test_genome_names, test_policy_network, test_properties,
  test_timing, test_coreml_export.
- **End-to-end**: 1-gen headless trainer completes in 2.54 s on M4,
  same as pre-session baseline. No regressions.

## What's still in Python (and why)

| Module                          | LOC  | Reason |
|---------------------------------|-----:|--------|
| training/trainer.py             | 1810 | Orchestrator — holds PyTorch refs, manages checkpoints |
| gui/main_window.py              | 631  | PyQt6 window |
| gui/main.py                     | 919  | PyQt6 wiring + IPC with trainer subprocess |
| emulation/parallel_pool.py      | 614  | Test-only pool for FakeNES-backed tests |
| gui/emulator_grid.py            | 461  | PyQt6 grid rendering |
| transport.py                    | 395  | GUI↔trainer shm IPC — retires when trainer moves to thread |
| training/behavior_cloning.py    | 388  | PyTorch BC loop |
| gui/replay_window.py            | 319  | PyQt6 |
| gui/play_window.py              | 300  | PyQt6 |
| training/genetic_algorithm.py   | 302  | PyTorch state_dict ops — not worth porting (see Batch 4) |
| training/narrator.py            | 165  | Thin Rust wrapper + caption templates |
| training/depth_tracker.py       | 97   | Thin Rust wrapper + JSONL memo I/O |
| audio/ram_music.py              | 147  | Thin `nes_core.AudioMixer` wrapper |
| utils/reward_functions/         | 48   | 40-line dispatcher to `nes_core.build_reward_function` |
| Other                           | ~1100 | Trainer internals, curriculum, BC, models, utils |

Python outer loop remains intentional per the plan — PyQt6 and
PyTorch-MPS are the only reasons. Every per-step hot path is now
in Rust.

## Next session priorities

**Revised 2026-04-20 post-measurement** — `scripts/bench_hot_path.py`,
`bench_emulator_phases.py`, `bench_async_pipeline.py`, and
`bench_worker_scaling.py` captured the full per-layer profile on
M4 Max. Findings in `docs/proposals/hot_path_baseline.md`.

**Key correction**: Batch 6 (per-scanline PPU rewrite) is
**downgraded** — PPU pixel rendering is only 15.9% of emulator time
at production `frame_skip=16`, not the plan's assumed 50%+. Ceiling
is ~14% trainer wall-time for weeks of work.

**Free wins already landed this session:**
- **PGO (+81% hot-path throughput, -47% trainer wall-time)** — biggest
  win by a wide margin. `scripts/pgo_build.sh`. Runtime code
  unchanged; pure compiler-side perf pass. Details in
  `docs/proposals/pgo_results.md`.
- Default worker count bumped 16 → 20 (measured **+9.6%** on M4 Max).
- Zero cargo warnings (95 → 0).
- Versioned save-state format rejects mismatched versions.
- 649 LOC dead Rust (`disassembler.rs`) deleted.

**Prototyped, bench-validated, ready to integrate next session:**
- MPS-native async forward (kick `net()` before `step_all`, materialise
  after): measured **+1.8%**. Needs `_evaluate_batch` refactor.

**Prototyped, REJECTED:**
- `threading.Thread`-based async pipeline: -4.2% on M4 Max.
  Thread-spawn + PyTorch Python orchestration defeats the overlap.

**Real priorities for next session (by measurement):**
1. **Verify audio live** (Batch 9) — user launches GUI solo-0.
2. **Integrate MPS-native async** into `_evaluate_batch` (+1.8%,
   already bench-validated).
3. **CPU bulk-stepping** — the 84% bottleneck. Architectural,
   weeks of work. Biggest absolute upside. Start with a sampling
   profile of nes_core::Nes::step to find the worst per-cycle hot
   spot.
4. **MLX vs MPS Conv2D bench** (Batch 10) — gated on a measured
   ≥1.2× win. Even 2× on MPS only nets 5% trainer wall-time.
5. **Trainer-as-thread refactor** → finish deleting
   `parallel_pool.py` + `transport.py`.
6. **Per-scanline PPU** (Batch 6) — deferred until after (2-4).
