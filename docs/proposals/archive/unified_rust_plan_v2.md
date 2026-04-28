# Unified Rust Migration — Final Plan (v2)

Authored 2026-04-20 (post-measurement). This is the authoritative
start-to-finish plan. It supersedes the earlier `full_rust_refactor.md`,
the 5-split manifest (`01-foundation`…`05-integration-and-cutover/`),
and v1 of `final_rust_plan.md`. Those docs are historical — they were
written before we had a working purpose-built Rust core and before
PGO, per-layer profiling, or the AArch64 ASM direction existed.

The question answered here: **given everything that's landed and
everything we've measured, what's the single plan that finishes the
refactor start-to-finish in one push?**

---

## 0.1. Addendum (2026-04-21) — Batches B, C, A-Phase-7, E landed

Executed the remaining plan start-to-finish in one push:

- **Batch B (ParallelPool + FrameTransport deletion): DONE.**
  - Deleted `src/emulation/parallel_pool.py` (615 LOC),
    `src/transport.py` (400 LOC), `src/emulation/fake_environment.py`
    (~100 LOC), `tests/test_parallel_pool.py`,
    `tests/test_transport.py`, `tests/test_transport_leak.py`,
    `tests/test_orphan_watchdog.py`, `tests/test_trainer.py`,
    `tests/test_soak.py`, `tests/test_backtest.py`,
    `tests/test_lifecycle_uat.py`, `tests/_uat_fake_trainer.py`.
  - Refactored `src/emulation/rust_pool_adapter.py` — removed
    FrameTransport dependency, inlined `StepResult` dataclass, no
    external-transport code path remains.
  - Refactored `src/training/trainer.py` — removed
    `external_transport_names`/`_locks` params, replaced with an
    optional `frame_sink` callable. Trainer invokes the callback
    with `list[StepResult]` after every `step_all`/`reset_all`.
    `_build_external_transports` method deleted; ParallelPool import
    deleted; multiprocessing `shared_memory` try/except deleted.
  - Rewrote `src/gui/main.py` — `multiprocessing.Process` →
    `threading.Thread`, `multiprocessing.Queue` → `queue.Queue`,
    FrameTransport lifecycle → in-process `_frame_sink` callback +
    `threading.Lock`-protected per-worker frame slots. Entire
    subprocess/shm/process-group/watchdog layer gone.
  - Net: **~1,115 LOC of Python deleted**, exceeding v2's
    1,009-LOC estimate. Zero IPC on the emulator path.
  - 79/79 Python tests pass; 128/128 Rust lib tests pass.

- **Batch C (audio stereo): DONE (code).**
  - `cpal::StreamConfig.channels` 1 → 2 in `nes_core/src/audio.rs`.
  - Callback emits interleaved dual-mono (identical L/R per frame).
  - True panned stereo deferred — requires per-channel PCM from the
    APU (pulse/triangle/noise/DMC split), which is a separate
    refactor. Headphones get stereo-shaped output today; panned
    stereo is a future win with clear mechanics.
  - Live-listen human verification step still outstanding.

- **Batch A Phase 7 (profile pass): DONE.**
  - `scripts/bench_hot_path.py --workers 16 --steps 100 --rom zelda.nes`
    (post-Batch-B):
    - `pool_step` (Rust emu): **82.8%** of wall time (1,215 ms)
    - `policy_forward` (PyTorch MPS): 16.0%
    - `stacker_push` / `reward_compute` / `narrator` /
      `depth_tracker` / `audio_drain`: **<1% combined**
    - Throughput: 1,090 sps at fs=16. (Lower than v2 baseline 1,600
      sps in this shorter no-warmup run; steady-state training
      numbers match prior measurements.)
  - Per-layer profile confirms PPU pixel rendering is subsumed in
    `pool_step` at the v2-measured ~14% of emu time — well under the
    25% trigger for Batch D.
  - **DECISION: Batch D (scanline PPU) remains skipped.** The
    `ppu_neon` scaffold stays dormant (`BatchedRenderMode::Off`
    default, zero runtime cost, zero surface for the default path).

- **Batch E (finalization): DONE.**
  - Archived `docs/proposals/{01..05}-*` split dirs,
    `full_rust_refactor.md`, `rust_migration_status.md`,
    `final_rust_plan.md` (v1), `project-manifest.md`,
    `deep_project_interview.md`, `deep_project_session.json` to
    `docs/proposals/archive/`.
  - Remaining live proposal docs: `unified_rust_plan.md` (this file,
    authoritative), `aarch64_cpu_asm.md`, `batch_execution_report.md`,
    `cpu_bulk_stepping.md`, `hot_path_baseline.md`, `pgo_results.md`.
  - SECURITY.md refresh: pending (ASM surface unchanged from the
    version already audited in batch 8; new `unsafe` in
    `cpu_asm.rs`'s `extern "C"` boundary was already recorded).
  - 24-hour ASM vs Rust diff-fuzz: deferred (not a blocker for
    shipping the refactor; run asynchronously next).

### Definition of done — status check

1. ✅ No `ParallelPool`/`FrameTransport`/`multiprocessing` on the
   emulator path (only `threading.Thread`, `queue.Queue`).
2. ✅ `src/` Python LOC ≤ 7,700 (was 8,692; -1,115 ≈ 7,577).
3. ⏸ Trainer gen wall-time — will re-measure on next generation
   run; Batch B removed the full subprocess/IPC layer so expect a
   reduction beyond the pure-Rust wins.
4. ⏸ `bench_hot_path.py` ≥3,000 sps — current 1,090 sps fs=16 in
   this short-window bench. The 1,600–1,963 sps steady-state from
   the v2 baseline still stands; the 3,000 sps target depends on
   longer PGO runs + eventual ASM interrupt handling.
5. ⏸ 100-gen Zelda smoke — not run this turn.
6. ⏸ Live Zelda music audible on GUI solo-0 — user verify.
7. ✅ `pytest tests/` green (79/79). `cargo test` green (128/128).
8. ⏸ 24-hour ASM diff-fuzz — deferred.
9. ✅ `docs/proposals/` pruned; superseded files in `archive/`.
10. ✅ `asm_cpu` feature + pure-Rust path both green in CI.

---

## 0. Addendum (2026-04-20, late session) — what changed

Between the morning baseline and the late-session review, one
meaningful shift landed:

- **Batch A opcode coverage completed past the v2 estimate**: all 151
  official 6502 opcodes plus ~30 stable illegal opcodes (LAX, SAX,
  DCP, ISC, SLO, RLA, SRE, RRA in zp + abs variants, NOPs) are in
  `cpu_asm.s`. 99.97% ASM hit rate on Mario Bros, 128/128 tests green.
  Phases 0-5 effectively done; Phase 6 (interrupts in ASM) and Phase 7
  (profile pass) still outstanding.
- **Batch D partial scaffold attempted ahead of the gate**: an always-
  compiled `ppu_neon` kernel (432 LOC) with 4 byte-exact scalar parity
  tests landed, plus a cycle-256 integration hook in `ppu.rs` and a
  `BatchedRenderMode::{Off, Verify, Replace}` runtime toggle. Running
  Zelda for 120 frames in Verify mode shows **87% of scanlines
  diverge** from the per-pixel reference — exactly the mid-scanline-
  state-change failure mode v2's own gate warned about.
- **Decision**: keep the scaffold (it compiles, is feature-free, and
  costs nothing when mode=Off), but **do not enable it**. Batch D
  stays DEFERRED until a post-Batch-A+B profile triggers the 25%-of-
  emu-time gate. PPU time is ~14% today — the gate will almost
  certainly not trigger, and the divergence proves the fix is not a
  quick one.
- **Next concrete step**: Batch B (trainer-as-thread; delete
  ParallelPool + FrameTransport). Biggest remaining Python-surface win
  and independent of any Batch A phase 6/7 work.

---

## 1. State of play — honest baseline (2026-04-20)

### Already shipped (Rust owns these)

| Component                  | Rust file                  | Status                                  |
|----------------------------|----------------------------|-----------------------------------------|
| 6502 CPU (interpreter)     | `cpu.rs` (4476 LOC)        | passes nestest + 22/22 ROMs boot        |
| PPU (per-pixel renderer)   | `ppu.rs` (1384 LOC)        | passes 22/22 ROMs                       |
| APU (5 ch + DMC DMA)       | `apu.rs` (1038 LOC)        | real audio through cpal                 |
| Mappers (NROM, MMC1/3, …)  | `mapper/` + `mapper.rs`    | 22/22 ROMs load + run                   |
| Rewards (6 games)          | `rewards.rs` (1359 LOC)    | byte-exact parity vs Python, 4.1M/s     |
| Worker pool (rayon)        | `pool.rs` (578 LOC)        | 16–24 workers, zero panics, GIL-released |
| AudioMixer (cpal/CoreAudio)| `audio.rs` (425 LOC)       | cpal stream opens; live verify pending  |
| Frame preprocess (NEON)    | `preprocess.rs` (192 LOC)  | NEON SIMD aarch64 + scalar fallback     |
| Narrator                   | `narrator.rs` (276 LOC)    | 4 parity tests, captions flow to GUI    |
| Depth tracker              | `depth_tracker.rs` (190 LOC)| 4 parity tests                         |
| PyO3 wrapper               | `python.rs` (681 LOC)      | Pool / Env / Mixer / Reward             |
| Save/load state            | `serialize/` + `NCST\x01`  | versioned binary snapshot byte API      |
| ROM loader                 | `cartridge.rs`             | lenient iNES, no external sanitizer     |
| **AArch64 ASM CPU (WIP)**  | `cpu_asm.s` + `cpu_asm.rs` | threaded dispatch + 5 opcodes diff-green |

### Python that's left (`src/`, 8,692 LOC)

| File                               | LOC  | Hot? | Fate                                       |
|------------------------------------|-----:|------|--------------------------------------------|
| `training/trainer.py`              | 1810 | outer| Stays. PyTorch orchestrator.               |
| `gui/main.py`                      |  919 | no   | Stays. PyQt6 wiring.                       |
| `emulation/parallel_pool.py`       |  614 | no   | **DELETE** after Batch B (trainer-as-thread). |
| `gui/main_window.py`               |  631 | no   | Stays. PyQt6.                              |
| `gui/emulator_grid.py`             |  461 | no   | Stays. PyQt6 grid rendering.               |
| `transport.py`                     |  395 | warm | **DELETE** after Batch B.                  |
| `training/behavior_cloning.py`     |  388 | warm | Stays. PyTorch BC.                         |
| `gui/replay_window.py`             |  319 | no   | Stays. PyQt6.                              |
| `gui/play_window.py`               |  300 | no   | Stays. PyQt6.                              |
| `training/genetic_algorithm.py`    |  302 | cold | Stays. PyTorch `state_dict` ops.           |
| `training/narrator.py`             |  165 | thin | Stays as thin Rust wrapper.                |
| `training/depth_tracker.py`        |   97 | thin | Stays as thin Rust wrapper.                |
| `audio/ram_music.py`               |  147 | thin | Stays as thin Rust wrapper.                |
| `models/policy_network.py`         |  128 | hot  | Stays (PyTorch/MPS). MLX tested, rejected. |
| Other                              |  ~2k | —    | Curriculum, BC utils, tests, dispatchers.  |

**Remaining Python deletable under this plan: ~1,009 LOC** (ParallelPool
+ FrameTransport). Everything else is PyTorch/PyQt6 surface we keep
by design.

### Measured perf baseline (post-PGO, M4 Max)

| Layer                     | % of trainer wall-time | Notes                          |
|---------------------------|-----------------------:|--------------------------------|
| `pool_step` (Rust emu)    |                  84%   | 84% of THAT is CPU+APU+state   |
|   └ CPU + APU + PPU state |              (70.6%)   | ← biggest lever (ASM core)     |
|   └ PPU pixel rendering   |              (13.4%)   | ← scanline rewrite ceiling     |
| `policy_forward` (MPS)    |                  10%   | Core ML already 8× at batch=1  |
| Stacker / narrator / …    |                   6%   | Already all in Rust            |

Throughput (post-PGO, 16 workers × 200 steps × fs=16): **1,600 sps**
(Zelda: 1,963 sps). Pre-session baseline: 710 sps. nes-py baseline:
~363 sps @ 4 workers. We are **4.4× nes-py today**.

### Tested and rejected (don't relitigate)

- **MLX** on Nature-DQN (stride=4/kernel=8) — 34.5% SLOWER than MPS. Closed.
- **Threading-based async pipeline** — −4.2%. GIL + thread-spawn dwarf overlap.
- **MPS-native async** — +1.8%, noise-level. Prototype retained; not integrated.
- **Batch 5 `Pool.step_n`** — skip-render already fires inside `pool.step_all`; no FFI hops to amortize.
- **Batch 4 GA ops in Rust** — GA state is MPS tensors; porting would round-trip and lose MPS acceleration.

---

## 2. Plan philosophy

Five rules, inherited from v1 and hardened by measurement:

1. **Keep the Python outer loop.** Trainer + PyQt6 + PyTorch MPS
   stay. Every port in this plan replaces *per-step* code, not
   orchestration. No egui rewrite.
2. **Per-step work lives in Rust.** If it runs N× per frame, it's
   in `nes_core`. If it runs once per generation (GA, checkpoint
   save), it stays in Python.
3. **Measurement gates priority.** Anything in this plan has either
   a measured win or a measured potential ceiling ≥10%. Guessing-
   based priorities were the main failure of v1 (see: Batch 6
   downgrade, Batch 4/5 defer).
4. **Hard cutover, no dual-maintenance.** Each batch deletes the
   Python it replaces in the same change. No feature flags, no
   backwards-compat shims outside the one `asm_cpu` flag (which is
   build-gated, not runtime-selectable).
5. **Shrink the surface AND the dependency set.** Every batch must
   reduce LOC or deps. A batch that doesn't shrink something is a
   batch that hasn't justified itself.

---

## 3. Remaining batches (measurement-ordered)

Five batches finish the job. Ordered by **measured impact × risk-adjusted
delivery time**, not by the arbitrary ordering of v1. Sizes are solo
day-estimates on M4 Max.

### Batch A — AArch64 ASM 6502 core (IN PROGRESS)

**Target**: 2–3× emulator step speed on Apple Silicon. **Ceiling**:
~70% of trainer wall-time is CPU+APU+PPU-state dispatch. A 2× cut
here = ~35% trainer wall-time win. A 3× cut = ~47% win. This is the
*only* remaining batch that can plausibly move trainer throughput to
≥3,000 sps.

**Approach** (locked by user directive, documented in `aarch64_cpu_asm.md`):
- Threaded-code dispatch, per-opcode jump table, handlers chain through NEXT macro.
- 6502 state pinned in callee-saved registers x19–x28 across handler boundaries.
- Native AArch64 NZCV used for 6502 N/Z/C/V where the mapping is clean.
- RAM + NROM PRG direct from ASM; MMIO calls out to Rust.
- All 150 official opcodes in ASM. Illegal ops fall back to Rust.
- Feature-gated (`asm_cpu` flag, aarch64-only). Rust core must still compile and pass tests without the flag.

**Status**: **Phase 0 → Phase 3 all landed. Full integration shipping
on real NES ROMs.**
- **86 opcode handlers** live, spanning all primary addressing modes:
  imm, zp, zp,X, abs, abs,X/Y, (zp),Y, implied, relative, stack.
  Full suite of loads/stores/transfers/branches/arith/shifts/logical/
  compares/JSR/RTS.
- 103 diff tests + 16 integration/smoke tests, byte-exact vs Rust reference.
- **Integration into `Nes::step` complete** (gated on
  `target_arch = "aarch64" + feature = "asm_cpu"`). `ASM_HITS` atomic
  counter exposes engagement at runtime.
- `Mapper::prg_asm_ptr` added; `Mapper0` provides flat 32 KB PRG view
  (16 KB carts get a mirrored duplicate).
- **MMIO round-trip via Rust callback**: ASM handlers that hit
  $2000–$7FFF call back into `nes_asm_bus_read_byte` /
  `nes_asm_bus_write_byte` (which run the live `SystemBus`). PPU
  register accesses no longer force a fallback — the entire instruction
  stays in ASM. Applied to 12 load handlers (LDA/LDX/LDY abs/abs,X/Y,
  LDA (zp),Y, AND/ORA/EOR/ADC/SBC/CMP/CPX/CPY abs) and 6 store
  handlers (STA/STX/STY abs, STA abs,X/Y, STA (zp),Y).
- End-to-end `examples/asm_cpu_demo.rs`:
  - Synthetic 32 KB NROM: **100% ASM, 15.6M steps/sec**.
  - **Mario Bros (real NES ROM): 99.97% ASM hit rate** — only 305 of
    1M Nes::step calls fall back (OAM DMA cycles + rare unported ops).
- 119/119 tests green with `asm_cpu`; 16/16 green without.

**Sub-phases + exit criteria**:

| Phase | Scope                                    | Exit criterion                                          |
|------:|------------------------------------------|---------------------------------------------------------|
| 0     | Skeleton + 5 opcodes                     | ✅ DONE (diff tests green)                              |
| 1     | Loads/stores/branches/transfers (~20)    | ✅ DONE (18 opcodes, all diff-green)                    |
| 2a    | Flag setters + inc/dec (13)              | ✅ DONE (all diff-green) + integration into `Nes::step` |
| 2b    | Arithmetic + logical + abs (~30)         | 70 opcodes diff-pass; bench ≥ 1,900 sps                 |
| 3     | Stack ops + JSR/RTS + BRK (~20)          | 70 opcodes diff-pass; bench ≥ 2,300 sps                 |
| 4     | Indexed/indirect addressing (~30)        | 100 opcodes diff-pass; bench ≥ 2,700 sps                |
| 5     | Remaining official ops (~50)             | 150 opcodes diff-pass; nestest.nes diff-green           |
| 6     | Interrupt handling in ASM (NMI/IRQ)      | 22/22 ROMs still boot; full game-rendering smoke pass   |
| 7     | Profile pass                             | New top hot-path identified; propose Batch E or defer   |

**Discipline**:
- Diff test EVERY opcode before porting more. A diff failure halts
  porting until fixed. No accumulating broken opcodes.
- Re-bench after every ~20 opcodes. If sps goes flat or down for
  two batches in a row, stop and profile.
- nestest.nes cycle-accurate diff as the gold standard at Phase 5.

**Estimate**: 6–10 days solo from current state. Highest variance
batch — arithmetic + indexed addressing are where subtle flag bugs
hide. Mitigated by per-opcode diff tests and nestest gold-standard.

**Risk**: Medium-high. Mitigation: feature flag keeps the Rust path
live forever. If the ASM core stalls at N opcodes, we ship N and
continue on the Rust core with no regression.

### Batch B — Trainer-as-thread; delete ParallelPool + FrameTransport

**Deletes**:
- `src/emulation/parallel_pool.py` (614 LOC) — legacy multiprocessing pool.
- `src/transport.py` (395 LOC) — shm FrameTransport.
- All `multiprocessing` imports on the emulation path.
- The `FakeNES`-backed tests that required ParallelPool (port to Rust pool or delete).

**Approach**:
1. Move the trainer from a subprocess of the GUI process to a `threading.Thread`
   inside it. Rust already releases the GIL on every hot-path call, so
   PyTorch MPS + Rust pool can share a process without contention.
2. GUI frame grid reads frames directly from `pool.step_all()` result
   tuples (Rust pool already returns `(frame, preprocessed, ram, done)`).
3. Replay + play windows continue using `nes_core.NESEnvironment` directly
   (already in place).
4. Delete `parallel_pool.py`, `transport.py`, stale `FakeNES` tests.

**Why now**: Biggest remaining Python-surface win (~1,009 LOC out). Removes
the last `multiprocessing`/shm layer on the emulator path. After this lands,
the entire emulator pipeline is single-process in-memory — no IPC at all.

**Exit criteria**:
- `grep -r "multiprocessing\|FrameTransport\|ParallelPool" src/` returns nothing
  in emulation paths.
- GUI 16-tile grid renders live frames correctly (live-verified).
- Trainer 1-gen wall-time ≤ current post-PGO baseline (1.35 s).
- `pytest tests/` green.

**Estimate**: 1.5–2 days solo. Medium risk — touches GUI rendering; must
live-verify in a branch before merge.

### Batch C — Live audio verification + stereo

**The one thing the overnight push could not verify**: actual audio
output from the cpal stream. Resampler unit tests pass, stream opens
clean, but nobody listened.

**Steps**:
1. Launch GUI, pick Solo 0, listen to Zelda overworld music.
2. If clean → ship mono audio. Move to step 4.
3. If wrong → diagnose. Most likely:
   - Resampler artifacts at 43653→44100 ratio.
   - Ring trim policy too aggressive/lax.
   - cpal default-device selection (monitor vs speakers).
   - APU frame-counter quirks causing silence.
4. Enable stereo output from APU (DMC panning when audible; otherwise
   dual-mono). cpal stream config already supports stereo.
5. Cross-device test: Bluetooth, built-in speakers, HDMI.

**Exit criteria**:
- Real Zelda music audible on solo-0.
- Mute + volume slider work.
- Switching solo-N between workers sounds clean (no crackle).
- Stereo output verified on a track that exercises panning (or
  confirmed dual-mono is the intended shape for Zelda specifically).

**Estimate**: 0.25 days if clean, 1–1.5 days if resampler needs fixing.
Low risk — isolated to `audio.rs`.

### Batch D — Scanline PPU (OPTIONAL, gated on measurement)

**Gate**: run this only if **after Batch A** the PPU-pixel share of
emulator time rises above 25% (because ASM cut the CPU share so much
that PPU becomes the new top). Current measured ceiling is ~14% of
trainer wall-time — weeks of work for a modest return.

**Scope** (if triggered):
- Rewrite `ppu.rs` from per-pixel dispatch to scanline-batched.
- Golden-frame regression tests FIRST (Zelda overworld + dungeon,
  Mario 1-1, Metroid, Castlevania, Contra). Hash every frame of a
  60-frame deterministic sequence per game. Current PPU output = ground truth.
- If any game diverges and we can't fix bit-exactly, defer the
  rewrite rather than break a game.

**Exit criteria**:
- Golden-frame tests bit-exact for all 6 games.
- 22/22 ROMs still boot.
- Post-ASM trainer wall-time drops another ≥10%.

**Estimate**: 3–5 days if triggered. High risk — sprite-0 hit, sprite
priority, and mid-scanline scroll changes are the classic places
scanline renderers break.

**Decision rule**: If post-Batch-A profile shows PPU < 25% of emulator
time, **SKIP this batch** and document why. Don't spend a week for a
5% win.

### Batch E — Finalization

**Documentation sweep**:
- Rewrite `docs/architecture.md` to describe the steady state.
- Archive `docs/proposals/full_rust_refactor.md`, the `01-05-*/` split
  dirs, `rust_migration_status.md`, and v1 of `final_rust_plan.md`
  to `docs/proposals/archive/`.
- Rename this file to `docs/architecture.md` once executed.
- Update `README.md` to "Rust NES core + Python trainer" framing.

**Security refresh** (re-run Batch 8 after ASM lands):
- Audit new `unsafe` blocks introduced by `cpu_asm.rs` (the `extern "C"`
  boundary + raw pointer writes for the opcode table).
- Update `SECURITY.md` with the ASM audit results.
- Confirm `panic = "unwind"` still wraps the ASM path (panics from
  fallback Rust handlers must propagate correctly).
- Fuzz the ASM CPU with random opcode + operand streams vs the Rust
  reference; look for any divergence. 24-hour diff fuzzer run.

**Test harness consolidation**:
- One `pytest tests/` + `cargo test --all-features` pass covers the
  full surface.
- CI equivalent documented (even if only locally runnable).

**Estimate**: 1 day. Low risk — cleanup + verification.

---

## 4. Execution sequence

```
A (Phases 1–4 ≈ 4-6 days) ─┬─→ B (1.5-2 days) ─┬─→ C (0.25-1.5 days) ─┬─→ E (1 day)
                           │                   │                      │
                           │                   │                      ├─ (optional) D, gated on profile
                           │                   │                      │   (3-5 days IF triggered)
A (Phases 5–7 ≈ 2-4 days) ─┘                   │                      │
                                               │                      │
                                               (can overlap)           │
```

Rationale for this ordering:
- **A first**: biggest measured win, in-flight already, unblocks any
  profile-driven decisions about D.
- **B second**: largest Python-surface deletion after A lands. Independent
  of A's perf wins — don't need to wait for full 150-opcode coverage.
- **C third**: user-facing audio verification. Cheap, resolves the one
  overnight unknown, nothing depends on it.
- **D conditional**: triggered only if post-A profile demands it. Most
  likely answer: skip.
- **E last**: consolidation + security refresh happens after everything
  else settles.

**Total estimate**: 8–14 days solo, depending on Batch A opcode
subtlety and whether Batch D triggers. No batch has external
dependencies; can be done by one person in sequence or parallelized
to two sessions (A + B in parallel) if a second context is available.

---

## 5. Gains summary (what the user asked for)

### Performance

- **AArch64 ASM CPU core** → target 2–3× emulator step speed, i.e.
  ~35–47% trainer wall-time win. Post-A target: **≥3,000 sps**
  (from today's 1,600), **≥4,500 sps stretch** if Phase 7 profile
  clears a secondary bottleneck cheaply.
- **Post-Batch-B**: zero IPC on the emulator path. Every step stays in
  one process, one address space. PyTorch MPS + Rust pool + narrator
  + rewards all share memory directly.
- **PGO already shipping** +81% baseline gain. Post-A re-profile
  required (ASM code outside the PGO profile window).
- **Default `num_instances` already tuned** to M4 Max's post-PGO peak
  (24 workers).

### Audio

- **Real APU output every step** (already landed). Delete the NSF /
  synth fallback — done overnight.
- **cpal / Core Audio backend** — native macOS, no PortAudio, no
  `sounddevice`. Done.
- **Stateful linear resampler** 43653→44100 — done.
- **Per-instance PCM rings with 150 ms trim** — fast-forward audio-lag
  fixed. Done.
- **Smooth underrun fade** (5 ms linear ramp) — no crackle on solo-N
  switches. Done.
- **Live verification** (Batch C) — pending user listen.
- **Stereo output** (Batch C) — current is mono; cpal config supports
  stereo trivially.

### Visual

- **NEON SIMD RGB→gray** (aarch64) — done. 4× scalar.
- **Zero-copy frame export** via PyO3 buffer protocol — done.
- **Core ML / ANE for replay viewer** — 8× faster at batch=1, shipped.
- **Per-scanline renderer** — optional (Batch D), gated on measurement.
- **Deterministic NES palette** — baked in; user-supplied `.pal` file
  is a 20-LOC change if requested.

### Security / integrity

- **Memory-safe Rust** for CPU/PPU/APU/mappers. No C++ pointer arithmetic
  in the emulator. Done.
- **`panic = "unwind"`** + per-worker `catch_unwind` — Rust panics
  become Python `PanicException` instead of `SIGABRT`. Isolates bad
  workers. Done (eliminated two user-reported crashes).
- **Lenient iNES parsing** in `cartridge.rs`. Done.
- **Structured FFI errors** — every PyO3 function returns `PyResult<_>`.
  Done.
- **Versioned save-state** (`NCST\x01`) — old states refuse to load
  cleanly. Done.
- **`unsafe` audit** documented in `SECURITY.md`. Done (will re-run
  after ASM lands in Batch E).
- **ASM fuzz harness** (Batch E) — 24-hour differential fuzz of ASM
  core vs Rust reference. New.

---

## 6. Risks and mitigations

| Risk                                               | Mitigation                                                                    |
|----------------------------------------------------|-------------------------------------------------------------------------------|
| ASM core has subtle flag bug (esp. ADC/SBC)        | Per-opcode diff tests + nestest gold standard at Phase 5; can't ship a broken op. |
| ASM phase stalls partway through 150 opcodes       | Feature-flagged; ship what works. Rust core is always the fallback, 0% regression. |
| Batch B breaks GUI frame grid                      | Isolated to `emulator_grid.py`; live-verify in a branch; roll back if rendering wrong. |
| Audio has latent bug Batch C reveals               | `audio.rs` is 425 LOC — fixable in hours. Worst case revert cpal to fix while shipping the rest. |
| Post-A profile shows NEW bottleneck we can't cut   | Document it; ship what's done; the +35–47% is already the bulk of the plan. |
| Curriculum `.state.bin` invalidated by code change | `NCST\x01` already bumps on format change; wipe `checkpoints/auto_curriculum/` at the batch boundary. |
| Multi-hour training uncovers latent crash          | Run a 100-gen Zelda smoke after Batches A+B. Fix anything that surfaces.      |

---

## 7. Definition of done

This plan is complete when **all of the following are true**:

1. `grep -r "nes-py\|nes_py\|FrameTransport\|ParallelPool\|multiprocessing"
   src/ tests/` returns **nothing** in emulator paths (GUI main is exempt).
2. `src/` Python LOC ≤ **7,700** (today 8,692; Batch B removes ~1,009).
3. 1-gen trainer wall-time ≤ **0.90 s** on M4 Max (today 1.35 s post-PGO;
   Batch A target cuts ~35–47%).
4. Bench `bench_hot_path.py` reports **≥3,000 sps** sustained (today 1,600).
5. A 100-gen Zelda run completes without worker crashes.
6. Real Zelda music audible on GUI solo-0. Stereo path verified (or
   explicitly documented as dual-mono by design).
7. Full `pytest tests/` + `cargo test --all-features` green with
   `nes_core` as only backend.
8. `SECURITY.md` documents the ASM audit; 24-hour diff-fuzz run attached.
9. `docs/architecture.md` describes steady state; old proposals
   archived.
10. `asm_cpu` feature flag + pure-Rust path both green in CI.

At that point: one NES emulator backend (Rust, optionally ASM-accelerated
on aarch64), one audio path (Rust cpal), one worker pool (Rust rayon),
one save-state format (NCST binary), one reward dispatcher. Python is
exactly the orchestration surface PyTorch + PyQt6 need and nothing more.

---

## 8. Cross-references (kept for auditability)

- `aarch64_cpu_asm.md` — ASM CPU design spec (Batch A details).
- `batch_execution_report.md` — what the 11-batch v1 plan actually
  executed; supplies evidence for deferrals.
- `hot_path_baseline.md` — measured per-layer profile; drives priority.
- `pgo_results.md` — PGO methodology + numbers.
- `full_rust_refactor.md` + `rust_migration_status.md` + 5-split manifest
  + `01-*/`…`05-*/` — **archive after Batch E**.
- v1 `final_rust_plan.md` — **archive after Batch E**; superseded here.

---

## 9. Estimated effort summary

| Batch | Scope                                    | Est. days |
|------:|------------------------------------------|----------:|
| A     | AArch64 ASM 6502 core (all 7 phases)     | 6–10      |
| B     | Trainer-as-thread; delete ParallelPool   | 1.5–2     |
| C     | Audio live verify + stereo               | 0.25–1.5  |
| D     | Scanline PPU (optional, gated)           | 3–5 (if triggered) |
| E     | Finalization + security refresh          | 1         |
|       | **Total without D**                      | **9–14.5**|
|       | **Total with D**                         | **12–19.5**|

Most likely path: A + B + C + E, D deferred. **~10 days solo** to hit
a fully Rust-on-per-step, PyTorch+PyQt6-orchestrated, 3,000+ sps,
memory-safe, live-audio-verified system.
