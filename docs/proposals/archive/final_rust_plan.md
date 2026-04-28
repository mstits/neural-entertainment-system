# Final Rust Plan — Start to Finish

Authored 2026-04-20 (overnight push done, user asleep). This is the
single unified plan that supersedes `full_rust_refactor.md` + the
5-split manifest. Those docs were written before we had a working
purpose-built Rust core; they're now historical context. The core is
real, the foundation is landed, and the question is "what's left to
finish the job and how do we sequence it."

This doc is the answer. It's written to be executable top-to-bottom
without further interviewing — each batch is self-contained, has clear
exit criteria, and deletes specific Python files on completion.

## Current state (the honest baseline)

**What's already in Rust (`nes_core`, 11,819 LOC):**

| Component                    | File                       | Status                        |
|------------------------------|----------------------------|-------------------------------|
| 6502 CPU                     | cpu.rs (4476 LOC)          | ✅ passes nestest + games     |
| PPU (per-pixel renderer)     | ppu.rs (1384 LOC)          | ✅ passes 22/22 ROMs          |
| APU (5 channels + DMC DMA)   | apu.rs (1038 LOC)          | ✅ real audio through cpal    |
| Mappers (NROM, MMC1/3, etc)  | mapper/ + mapper.rs        | ✅ 22/22 ROMs load + run      |
| Rewards (all 6 games)        | rewards.rs (1359 LOC)      | ✅ byte-exact parity vs Py    |
| Worker pool (rayon)          | pool.rs (578 LOC)          | ✅ 16 workers, zero panics    |
| AudioMixer (cpal / Core Audio)| audio.rs (425 LOC)        | ✅ cpal stream opens clean    |
| Frame preprocess (NEON SIMD) | preprocess.rs (192 LOC)    | ✅ aarch64 NEON + scalar fallback |
| PyO3 wrapper                 | python.rs (681 LOC)        | ✅ Pool / Env / Mixer / Reward |
| Save/load state (NCST\x01)   | serialize/                 | ✅ binary snapshot byte API   |
| ROM loader (lenient iNES)    | cartridge.rs               | ✅ no external sanitizer      |

**What's still in Python (`src/`, 10,123 LOC):**

| File                                  | LOC  | Hot path? | Planned fate                             |
|---------------------------------------|-----:|-----------|------------------------------------------|
| training/trainer.py                   | 1929 | outer loop| Stays. Orchestrates PyTorch refs.        |
| gui/main.py                           | 942  | no        | Stays (PyQt6 wiring).                    |
| emulation/parallel_pool.py            | 730  | no        | **DELETE** — Rust pool replaces it.      |
| gui/main_window.py                    | 727  | no        | Stays (PyQt6 window).                    |
| emulation/nes_environment.py          | 542  | legacy    | **DELETE** — nes-py wrapper retires.     |
| gui/emulator_grid.py                  | 470  | no        | Stays (Qt frame grid).                   |
| transport.py                          | 403  | hot(shm)  | **DELETE** — Rust pool in-process.       |
| training/behavior_cloning.py          | 388  | warm      | Stays. PyTorch + BC loop.                |
| emulation/demo_worker.py              | 327  | warm      | **DELETE** — folded into Rust pool.      |
| gui/replay_window.py                  | 317  | no        | Stays (Qt).                              |
| gui/play_window.py                    | 313  | no        | Stays (Qt, already `nes_core`).          |
| training/genetic_algorithm.py         | 301  | hot       | **PORT** — vector ops in Rust.           |
| training/narrator.py                  | 288  | warm      | **PORT** — pure RAM-delta logic.         |
| diagnostics/worker_debug.py           | 256  | no        | Stays (dev-only).                        |
| gui/highlight_recorder.py             | 236  | no        | Stays (Qt, cv2 for mp4 encode).          |
| training/curriculum.py                | 230  | no        | Stays (file I/O + profile logic).        |
| gui/reward_tuning_window.py           | 212  | no        | Stays (Qt).                              |
| training/depth_tracker.py             | 209  | warm      | **PORT** — RAM-delta tracker.            |
| emulation/rust_pool_adapter.py        | 203  | hot       | **COLLAPSE** — inline into trainer.      |
| gui/metrics_window.py                 | 164  | no        | Stays (Qt + PyQtGraph).                  |
| audio/ram_music.py                    | 147  | thin      | Stays (trivial cpal wrapper).            |
| models/coreml_export.py               | 129  | no        | Stays (Core ML tool).                    |
| models/policy_network.py              | 128  | very hot  | Stays (PyTorch/MPS). *MLX = stretch goal.* |
| utils/reward_functions/__init__.py    | 48   | factory   | Stays. 40-line dispatcher.               |

**Current numbers (end of overnight push):**
- 1-gen headless trainer: nes-py 5.08 s, **nes_core 2.54 s (2.0× faster)**.
- Rust reward hot path: **4.1M calls/s** (9.8× Python).
- 16-worker × 2000-step stress: **2762 sps, zero panics**.
- 22/22 ROMs boot cleanly. 7/7 Rust unit tests + 6/6 reward-parity tests pass.
- panic=unwind: SIGABRT from Rust panics eliminated.

## Plan philosophy — "all in one go"

The user asked for start-to-finish in one push. The way to do that
without doubling effort is:

1. **Keep the Python outer loop.** Trainer + GUI stay in Python because
   they hold PyTorch refs and Qt handles respectively. Porting PyQt6 to
   egui is weeks of work for zero user-visible benefit. Porting the
   PyTorch MPS path to MLX is a benchmark-before-commit exercise, not a
   "just do it" item.
2. **Everything per-step in Rust.** CPU, PPU, APU, rewards, preprocess,
   narrator, depth tracker, GA ops, audio mixing — every piece of code
   that runs N×per-frame moves into `nes_core`.
3. **Delete the legacy backend entirely.** `nes-py` goes. `cv2`/`PIL`
   go. `sounddevice` / `libgme` / NSF already gone. `multiprocessing`
   parallel pool goes. FrameTransport SHM goes.
4. **Hard cutover, no dual-maintenance.** Each batch lands end-to-end
   including the deletion of the Python code it replaces. If a batch
   leaves Python alive alongside its Rust replacement, it's not done.
5. **Ship what reduces surface, not what adds it.** Every batch should
   shrink `src/` Python LOC AND shrink the dependency set. No feature
   flags, no backwards-compat shims.

## The batch sequence

Each batch is ~0.5–2 days solo. Ordered so each unblocks the next; no
parallel sessions required but you could.

### Batch 1 — Retire `nes-py` backend entirely

**Deletes:**
- `src/emulation/nes_environment.py` (542 LOC) — nes-py wrapper
- `nes-py` from `pyproject.toml` dependencies
- `cv2` + `PIL` imports in the emulator path (preprocess is in Rust now)
- The `nes-py` branch in `_make_pool` and `--env-backend` dropdown

**Steps:**
1. Grep for every caller of `src.emulation.nes_environment:NESEnvironment`.
   Confirm all sites can reach `nes_core:NESEnvironment` instead
   (trainer, bench scripts, tests).
2. Remove the `"nes-py"` choice from `--env-backend` CLI in
   `src/training/trainer.py` and `src/gui/main.py`.
3. Delete `src/emulation/nes_environment.py`.
4. Drop `nes-py` from `pyproject.toml` + regenerate lock.
5. Update `tests/` — remove any nes-py-specific assertions; keep the
   `nes_core` paths.
6. Delete `scripts/bench_backends.py` (A/B benchmark; nothing to A/B).
7. Run `pytest tests/` + `scripts/test_trainer_one_gen.py`.

**Exit criteria:** `grep -r "nes_py\|nes-py" src/ tests/` returns nothing.
Trainer runs 1 gen in ≤2.6 s.

**Risk:** Low. `nes_core` is already the default; this just removes the
fallback branch.

---

### Batch 2 — Delete `ParallelPool` + FrameTransport

**Deletes:**
- `src/emulation/parallel_pool.py` (730 LOC)
- `src/transport.py` (403 LOC) — FrameTransport SHM
- `src/emulation/demo_worker.py` (327 LOC) — demo slot subprocess
- `src/emulation/rust_pool_adapter.py` collapses to ~40 LOC (or inlines
  into trainer)
- `multiprocessing` imports in the emulation path

**Steps:**
1. Fold `demo_worker.py` replay logic into `nes_core::Pool` — demo slot
   is just worker 0 with a pre-recorded action stream. Rust already
   exposes `replay_actions` for legacy `.state.bin`; extend to live
   action playback.
2. GUI frame grid (`src/gui/emulator_grid.py`) currently reads frames
   via FrameTransport. Refactor to pull frames from `pool.step_all()`
   results directly — the Rust pool already returns `(frame,
   preprocessed, ram, done)` 4-tuples, so the grid just reads the
   `frame` field.
3. Delete `parallel_pool.py` (ParallelPool class, multiprocessing queue
   setup, `FrameTransport` publish plumbing).
4. Delete `transport.py` (SHM v2 header no longer needed).
5. Delete `demo_worker.py`.
6. Inline `rust_pool_adapter._materialize` into `trainer._evaluate_batch`
   (it's 8 lines).
7. Smoke test: launch GUI, verify 16-tile grid renders live frames.

**Exit criteria:** `grep -r "multiprocessing\|FrameTransport\|ParallelPool"
src/` returns nothing in emulation paths. GUI grid still shows 16 live
tiles. Trainer still runs.

**Risk:** Medium. Touches GUI rendering; must live-verify. The GUI frame
pull is the load-bearing change — get it right first in a branch.

**Win:** ~1460 LOC deleted. No more multiprocessing shm, no more
queue-based RPC, no more subprocess lifecycle. Everything in-process,
rayon-parallel, GIL-released.

---

### Batch 3 — Port narrator + depth tracker to Rust

**Deletes:**
- `src/training/narrator.py` Python body (288 LOC) → stays as ~50-line
  wrapper
- `src/training/depth_tracker.py` Python body (209 LOC) → stays as
  ~30-line wrapper

**Adds:**
- `nes_core/src/narrator.rs` — RAM-delta event detectors per game,
  caption routing. Same contract: `detect(prev_ram, curr_ram, game) ->
  Vec<Event>`.
- `nes_core/src/depth_tracker.rs` — per-signal deepest-reached state.
  Stores max(signal) across a run; resets per episode.

**Why Rust:** Both modules run per-step, both are pure RAM-byte logic
with small state. Each caption lookup is a dict load today; in Rust
it's a `HashMap<&'static str, &'static str>` with zero allocation.

**Steps:**
1. Write `narrator.rs` with parity tests against the Python narrator
   on 500-step random-RAM sequences per game (same pattern as rewards).
2. Write `depth_tracker.rs` with parity tests.
3. Expose via PyO3: `nes_core.Narrator` + `nes_core.DepthTracker`.
4. Rewrite the Python modules to thin wrappers that delegate.
5. Verify via an end-to-end trainer run that narrator captions still
   appear in the GUI.

**Exit criteria:** Parity tests pass. GUI still shows Zelda captions
("Link took damage!", "Entered a dungeon", etc.). Trainer throughput
unchanged or faster.

**Risk:** Low. Pure logic port; easy to verify.

---

### Batch 4 — Port GA core ops to Rust

**Deletes:**
- Python bodies of `crossover`, `mutate`, `tournament_select`, `elitism`
  in `src/training/genetic_algorithm.py` (~200 LOC)

**Keeps:**
- `genetic_algorithm.py` as a thin orchestrator that calls
  `nes_core.ga.evolve(population, fitnesses, config) -> new_population`.

**Adds:**
- `nes_core/src/ga.rs` — vector ops over genome tensors. Rayon-parallel
  across population. Fixed-seed RNG for reproducibility.

**Why Rust:** GA ops run once per generation but touch every genome
(16 workers × genome_size bytes). Not the tightest loop but
embarrassingly parallel. More importantly, this eliminates another
~200 LOC of numpy-heavy Python that we'd otherwise leave alone.

**Steps:**
1. Define a `Genome = Vec<f32>` contract on the Rust side. PyO3 exposes
   numpy array ↔ `&[f32]` via buffer protocol (zero-copy).
2. Port each op with parity tests against the Python version.
3. Collapse Python GA to the orchestrator.

**Exit criteria:** `python -c "from src.training.genetic_algorithm
import evolve; ..."` produces byte-identical populations between
Python and Rust for a fixed seed.

**Risk:** Low-medium. Reproducibility requires matching RNG exactly —
use `rand_chacha::ChaCha8Rng` and match seeds.

---

### Batch 5 — Bulk-step API (`Pool.step_n`)

**Adds:**
- `nes_core::Pool::step_n(actions: &[Vec<u8>]) -> Vec<StepResult>` —
  runs N frames per worker in one PyO3 call. Amortizes FFI overhead.
- Inside `step_n`, non-observed frames (skip_render) use the fast path
  that skips PPU pixel work.

**Why:** With `frame_skip=16`, only 1 of every 16 frames needs a
rendered output. Batching 16 steps into one call and using skip-render
on 15 of them eliminates 15 PPU render passes per observed frame and 15
PyO3 round-trips. Expected: +20-30% throughput.

**Steps:**
1. Add `Pool::step_n` taking `Vec<Vec<u8>>` (outer = time, inner =
   worker). Returns only the last rendered frame + accumulated
   reward/done.
2. Wire through `src/training/trainer.py::_evaluate_batch` — replace
   the loop of `pool.step_all(action) for _ in range(frame_skip)` with
   one `pool.step_n(batched_actions)`.
3. Bench.

**Exit criteria:** 1-gen trainer wall-time ≤2.0 s (was 2.54 s).

**Risk:** Medium. Audio sampling during skip-render: the APU still needs
to tick + emit samples on every frame, or mixer underruns. Must verify
audio continuity.

---

### Batch 6 — Per-scanline PPU rewrite

**Modifies:**
- `nes_core/src/ppu.rs` — rewrite from per-pixel dispatch to
  per-scanline batched rendering. Keeps the NES palette, OAM, and
  attribute tables correct; relaxes per-cycle timing where games don't
  rely on it.

**Why:** This is the biggest remaining perf lever. Per-pixel PPU
dispatch is 30-40% of emulator wall time. A per-scanline renderer that
emits 256 pixels per row (vs 256 pixel-by-pixel function calls)
eliminates most of that overhead. Expected: 2-3× emulator step speed,
translating to ~1.5× trainer throughput (Amdahl — ML inference is
~30-40% of the loop).

**Steps:**
1. Write golden-frame regression tests FIRST: 60-frame sequences from
   Zelda overworld, Zelda dungeon, Mario 1-1, Metroid. Hash each frame.
   Current PPU's output is ground truth.
2. Rewrite PPU to a scanline-batched model: `render_scanline(y)` emits
   256 pixels in one pass with SIMD-friendly inner loops.
3. Run golden-frame tests — if frames match bit-exact, ship. If they
   diverge on sprite-0-hit or sprite priority (the classic NES bugs),
   fix iteratively until green.
4. Bench.

**Exit criteria:** Golden-frame tests pass bit-exact for the 4 games.
1-gen trainer wall-time ≤1.5 s. All 22 ROMs still boot.

**Risk:** High. Sprite-0 hit timing, sprite priority, and mid-scanline
scroll changes are the classic places where scanline-batched PPUs get
games wrong (Battletoads, MMC3 IRQ games). Mitigation: golden-frame
tests + deliberate non-goal of cycle-accuracy for weird effects.

**This batch may need to be deferred** if golden-frame tests show too
many divergences on the games we care about. If deferred, skip to
Batch 7.

---

### Batch 7 — Narrator subprocess → in-process thread

**Deletes:**
- The `multiprocessing` narrator subprocess in `src/training/narrator.py`
  (the last Python multiprocessing user outside of GUI).

**Why:** Once Batch 3 puts narrator logic in Rust, the Python shell
becomes a trivial dispatcher. No need for a separate process — it can
run on a thread in the trainer, consuming RAM from the Rust pool's
`step_all` output directly.

**Steps:**
1. Replace `multiprocessing.Process` + queues with a `threading.Thread`
   that drains a `queue.Queue` fed by `_evaluate_batch`.
2. GIL concerns: narrator captions come from `nes_core.Narrator.detect()`,
   which releases the GIL internally.
3. Delete the subprocess lifecycle code.

**Exit criteria:** GUI captions still appear. Only the GUI's Qt
subprocess (main window) remains as a separate process; trainer is
single-process.

**Risk:** Low.

---

### Batch 8 — Security / integrity hardening pass

**Scope:** Not a new feature; a deliberate audit of the FFI boundary.

**Items:**
1. Every PyO3 function that takes `&[u8]` from Python: bounds-check
   against expected size (RAM = 2048 bytes, frame = W×H×3). Currently
   implicit; make it explicit with `PyValueError` on mismatch.
2. Every `catch_unwind` boundary: confirm it logs + returns a sentinel
   value rather than re-raising into Rust code that assumes success.
3. `panic = "unwind"` is already set — confirm via `cargo build
   --release` + a forced `panic!()` in a worker that the Pool isolates
   it (already tested, re-verify).
4. Memory: audit every `unsafe` block in `nes_core/`. Current count:
   NEON preprocess (justified, bounded), cpal callback (justified).
   Anything else is a bug.
5. Malformed ROM handling: ensure every code path through cartridge.rs
   returns `Result<_, CartridgeError>` rather than panicking on bad
   header bytes.
6. Save-state versioning: bump `NCST\x01` → `NCST\x02` only when the
   format changes. Add a version-check that refuses `\x02` states in
   a `\x01` build with a clear error.

**Exit criteria:** `unsafe` audit done (documented in `SECURITY.md` or
equivalent). Forced-panic test passes. 22/22 ROMs load cleanly.
Malformed ROMs return structured Python exceptions, not crashes.

**Risk:** Low. Mostly verification, some small hardening edits.

---

### Batch 9 — Live audio verification + polish

**The one thing the overnight push could not verify:** actual audio
output. The cpal stream opens clean, resampler unit tests pass, but
nobody listened.

**Steps:**
1. User launches GUI, picks Solo 0, listens to Zelda overworld music.
2. If clean → done, mark audio path shipped.
3. If wrong → diagnose. Most likely candidates (in order):
   - Resampler producing artifacts at 43653→44100 ratio
   - Ring trim policy too aggressive/too lax
   - cpal picking wrong default device (monitor vs speakers)
   - APU frame-counter quirks causing silence
4. Cross-device test: Bluetooth headphones, built-in speakers, HDMI
   output. All should Just Work.

**Exit criteria:** Real Zelda music audible on solo-0, mute works,
volume slider works, switching solo-N between workers sounds clean.

**Risk:** Low if clean, medium if resampler is wrong (1-2 hrs to fix).

---

### Batch 10 — MLX migration (STRETCH — bench before committing)

**Scope:** Replace PyTorch-MPS with Apple's MLX for the policy network
forward pass (inference only, not training).

**Why it's stretch:** MLX on M4 is competitive with PyTorch-MPS for
dense layers but the `Conv2D` kernel quality for our policy network
size hasn't been third-party benchmarked. Could be a win, could be a
regression.

**Gate:** Run `scripts/bench_policy_mlx_vs_pytorch.py` with the actual
policy net shape (84×84×4 → 6-class discrete). If MLX ≥1.2× faster on
the forward pass AND Core ML Tools / Neural Engine path remains viable
for future export → migrate. Otherwise defer.

**If we commit:** the migration is ~1 day — `mlx.nn.Conv2D` + `Linear`
signatures are close to PyTorch's, and `models/policy_network.py` is
only 128 LOC.

**If deferred:** document the bench result in the file and move on.

---

### Batch 11 — Documentation + test harness sweep

**Final cleanup:**
- Rewrite `docs/rust_nes_core.md` to describe the new core as it exists
  post-all-batches, not the old tetanes-vs-nes-py comparison.
- Archive `docs/proposals/full_rust_refactor.md` and the 5-split
  manifest under `docs/proposals/archive/` — they're historical now.
- Delete `docs/proposals/rust_migration_status.md` — folds into a new
  `docs/architecture.md` that reflects steady state.
- Rename this file to `docs/architecture.md` once executed.
- Test harness: one `pytest tests/` pass covers the full surface with
  `nes_core` as the only backend.
- Update `README.md` to describe the repo as "Rust NES core + Python
  trainer" rather than the two-backend framing.

## Performance targets (cumulative across batches)

| After batch                      | 1-gen trainer wall-time | Speedup vs nes-py baseline |
|----------------------------------|------------------------:|---------------------------:|
| Baseline (nes-py, pre-overnight) | 5.08 s                  | 1.0×                       |
| Current (overnight done)         | 2.54 s                  | 2.0×                       |
| After Batch 2 (pool deletion)    | 2.3 s                   | 2.2×                       |
| After Batch 5 (bulk-step)        | 1.9 s                   | 2.7×                       |
| After Batch 6 (scanline PPU)     | 1.3 s                   | 3.9×                       |
| After Batch 10 (MLX, if viable)  | 1.1 s                   | 4.6×                       |

**Stretch goal from original proposal:** ≥5× throughput. Achievable if
Batch 6 and Batch 10 both land cleanly. Realistic ~4× if Batch 10
benches poorly and gets deferred.

## Audio gains (summary)

- **Real APU output, every step, at training speed.** (Already landed.
  Verification pending.)
- **cpal / Core Audio backend** — native macOS audio, no PortAudio
  layer, no `sounddevice` Python dep.
- **Stateful linear resampler** in Rust — handles 43653→44100 exactly
  with <1 sample of latency and no aliasing artifacts.
- **Per-instance PCM rings** with aggressive 150 ms trim — solves the
  fast-forward audio-lag problem (user complained about) by keeping
  only recent audio.
- **Smooth underrun fade** (5 ms linear ramp) — no more crackle on
  solo-N switches.
- **Stereo-ready** — current is mono; the cpal stream config supports
  stereo output trivially. Enable when APU emits stereo (DMC ch).
- **Chiptune synth + libgme + NSF playback paths all deleted.** No
  fallback synth to confuse the user about "is this real game audio?"

## Visual gains (summary)

- **NEON SIMD grayscale conversion** (aarch64) — already landed. 4×
  faster than scalar RGB→Y for the trainer's 84×84 preprocess.
- **Zero-copy frame export** via PyO3 buffer protocol — already landed.
  No `numpy.array()` allocation per step.
- **Per-scanline renderer** (Batch 6) — 2-3× emulator step speed.
- **84×84 sub-resolution render** for headless workers (skipping the
  256×240→84×84 resize by rendering at target resolution directly) —
  nice-to-have, not on the critical path.
- **Deterministic palette** — current uses the standard NES palette
  baked in. User-supplied .pal file support is a 20-line change if
  anyone asks for it.

## Security / integrity gains (summary)

- **Memory-safe Rust** for CPU/PPU/APU/mappers. No C++ pointer
  arithmetic left in the emulator layer.
- **`panic = "unwind"`** + per-worker `catch_unwind` — Rust panics
  become Python `PanicException` rather than `SIGABRT`. Isolates a
  bad worker instead of killing the trainer. (Already landed —
  eliminated two user-reported crashes.)
- **Lenient iNES header parsing** built into cartridge.rs — no
  external Python sanitizer, accepts the ROMs nes-py accepts.
- **Structured errors at the FFI boundary** — every PyO3 function
  returns `PyResult<_>`, malformed input returns a
  `PyValueError`/`PyRuntimeError`, never a silent crash.
- **Versioned save-state format** (`NCST\x01`) — future format changes
  bump the version, old states refuse to load with a clear error
  rather than silently corrupting.
- **No `unsafe` outside justified blocks** (NEON intrinsics, cpal
  callback) — each one documented. Batch 8 audits this formally.

## Risks and mitigations

| Risk                                             | Mitigation                                                                   |
|--------------------------------------------------|------------------------------------------------------------------------------|
| Scanline PPU (Batch 6) breaks some games         | Golden-frame regression tests; defer batch if >1 game we care about breaks.  |
| Live audio has a latent bug Batch 9 reveals      | Bug in Rust audio.rs fixable in hours; worst case revert cpal for sounddevice while fixing. |
| MLX Conv2D kernel is slow on M4 for our shape    | Batch 10 is stretch; bench before commit; defer if not a win.                |
| GUI frame-pull refactor (Batch 2) breaks grid UI | Isolated to `emulator_grid.py`; can live-verify in a branch before merge.    |
| Multi-hour training stability untested at scale  | Run a full overnight training session after Batch 5; fix anything that crashes. |
| Curriculum `.state.bin` files break on NCST bump | Old states already deleted once; wipe `checkpoints/auto_curriculum/` at batch boundary. |
| User unhappy with audio quality after Batch 9    | Resampler + ring params are tunable; iterate live through GUI mixer window.  |

## What stays in Python — and why

Being explicit about what we are NOT porting, so future sessions don't
accidentally rewrite it:

- **Trainer outer loop** (`trainer.py`) — holds PyTorch refs, orchestrates
  GA+PPO+BC, manages checkpoints. Thin glue; Rust would just mirror it.
- **PyQt6 GUI** (`gui/*.py`) — Qt's Python bindings are mature. egui or
  slint rewrite is weeks of work for marginal value.
- **Policy network** (`models/policy_network.py`) — PyTorch/MPS path.
  Batch 10 considers MLX; absent that, stays.
- **Core ML export** (`models/coreml_export.py`) — uses `coremltools`
  which is Python-only.
- **Curriculum logic** (`training/curriculum.py`) — file I/O + YAML
  loads. Cold code, no value in porting.
- **Highlight recorder** (`gui/highlight_recorder.py`) — uses `cv2`
  to encode mp4. The cv2 dep stays for this file only; alternative
  is `video-rs` crate which would work but isn't worth the effort.
- **Genome naming** (`training/genome_names.py`) — word-list lookup.

## When this plan is "done"

The migration ships when **all of the following are true**:

1. `grep -r "nes-py\|nes_py\|FrameTransport\|ParallelPool\|sounddevice\|
   libgme\|nesrs" src/ tests/` returns nothing.
2. `src/` Python LOC is ≤8000 (currently 10,123; plan removes
   ~2500 LOC via batches 1–4 and 7).
3. 1-gen trainer wall-time is ≤1.5 s on M4 (3.4× nes-py baseline).
4. A 100-generation Zelda run completes without worker crashes.
5. Real Zelda music audible on GUI solo-0 without any synth fallback.
6. Full `pytest tests/` green with `nes_core` as only backend.
7. `unsafe` audit documented in `SECURITY.md`; every block justified.
8. No `multiprocessing` imports in emulation paths (only GUI main).
9. `docs/architecture.md` describes the steady state; old proposals
    archived.

At that point the codebase has one NES emulator backend (Rust), one
audio path (Rust cpal), one worker pool (Rust rayon), one save-state
format (NCST binary), one reward dispatcher, and the Python surface is
the minimum needed to orchestrate PyTorch and PyQt6 — which is exactly
what the user asked for.

## Estimated effort (solo)

| Batch | Est days |
|-------|---------:|
| 1 — Retire nes-py                    | 0.5      |
| 2 — Delete ParallelPool + FrameTransport | 1.5  |
| 3 — Narrator + depth_tracker to Rust | 1        |
| 4 — GA ops to Rust                   | 1        |
| 5 — Bulk-step API                    | 0.5      |
| 6 — Per-scanline PPU                 | 3-5      |
| 7 — Narrator subprocess → thread     | 0.5      |
| 8 — Security hardening pass          | 0.5      |
| 9 — Live audio verification          | 0.25 (or 1 if bug) |
| 10 — MLX migration (stretch)         | 1 (bench) + 1 (port, if gated in) |
| 11 — Doc sweep                       | 0.5      |

**Total: 10-13 days solo without MLX; 12-15 days with it.** The path
is linear but batches 3 and 4 could parallelize with 6 if multiple
sessions were available.

## Recommended execution sequence

If the user says "go": do batches 1 → 2 → 9 → 3 → 4 → 5 → 8 → 7 → 6 →
11 → (maybe 10).

**Revised 2026-04-20 post-measurement** (see
`hot_path_baseline.md`): batches 1-3, 7, 8, 11 have landed. Batches
4, 5 deferred. **Batch 6 is downgraded** — measurement on M4 Max
shows PPU pixel rendering is only 15.9% of emulator time (vs the
plan's assumed 50%+), so the ceiling is ~14% trainer wall-time for
weeks of work. New top priority is an **async pipeline** that hides
MPS forward-pass latency inside `pool.step_all`, for a ~10% win in
~a day of work. Then CPU bulk-stepping (the real 84% bottleneck)
is the long-horizon goal.

Rationale for that order:
- **1 first**: trivial deletion, immediately shrinks surface by 540 LOC.
- **2 second**: the biggest Python-surface win. Pool + Transport +
  demo_worker = 1460 LOC out.
- **9 third**: user listens to audio on a GUI launch. Low cost, resolves
  the last overnight-unknown, and tells us whether Batch 5's audio
  concerns are real before we build on top.
- **3, 4 next**: small, safe, parallelizable ports that keep momentum.
- **5 (bulk-step)**: modest perf win, low risk, paves way for 6.
- **8 (hardening)**: cheap audit before we touch PPU internals.
- **7**: subprocess → thread, natural after narrator is in Rust.
- **6 (scanline PPU)**: biggest perf lever but highest risk — land it
  late so it doesn't block the cleanup wins.
- **11 (docs)**: after everything has settled.
- **10 (MLX)**: only if there's time and the bench supports it.
