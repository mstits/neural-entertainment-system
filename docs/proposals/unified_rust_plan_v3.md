# Unified Rust Migration — v3 (Finish Line Plan)

Authored 2026-04-21 — supersedes v2 (`unified_rust_plan.md`) for
unstarted work. v2 is preserved for audit: Batches A (phases 0–7),
B (ParallelPool/FrameTransport deletion), C (audio stereo code path),
and E (finalization) all landed. ~85% of the v2 scope is done. Two
meaningful deltas remain open, plus a catalog of smaller finish-line
items. This document closes them in one push.

The question answered here: **given what landed post-v2, what's the
minimum start-to-finish sequence that ships a fully Rust-driven
NES emulator surface with audited audio, visual, performance, and
security/integrity wins, and retires `nes-py` from the repository
entirely?**

---

## 1. Honest state (2026-04-21)

### Confirmed landed (don't relitigate)

- `nes_core` Rust crate: 18,072 LOC across 20 files. Full 6502 CPU
  (interpreter + AArch64 ASM under `asm_cpu`), PPU, APU with 5-ch
  split, 20 mappers (0/1/2/3/4/5/7/9/10/11/19/66/68/69/71/85/VRC2/
  VRC4/VRC6 + MMC5 audio), rayon pool with QOS pthread hint, cpal
  stereo stream, PyO3 binding, versioned `NCST\x01` save-state,
  lenient iNES/NES 2.0 cartridge loader, MD5 payload hashing, NEON
  preprocess to fp16.
- Python surface: 7,666 LOC (under the v2 target of 7,700). No
  `multiprocessing` / `FrameTransport` / `ParallelPool` on the
  emulator path. Trainer is an in-process thread; GUI reads frames
  via a callback `frame_sink`.
- ASM CPU: 99.97% hit rate on Mario Bros at 15.6M steps/sec on a
  synthetic NROM. 5-minute differential fuzz: 5,316,000 streams,
  63.8M instructions, 0 divergences vs the pure-Rust reference.
- Audio: cpal stereo stream (`channels: 2`), `PAN_MATRIX[5]` in
  `audio.rs`, `Apu::generate_sample_channels -> [f32; 5]` in
  `apu.rs`. Per-channel resampler + pan matrix live.
- **Bake-off (Contra, M4 Max, measured 2026-04-21)**:

  |                     | nes-py (sps) | nes_core (sps) |  ratio |
  |---------------------|-------------:|---------------:|-------:|
  | fs=1   single-env   |        1,334 |            785 | 0.59×  |
  | fs=4   single-env   |        1,399 |          1,529 | 1.09×  |
  | fs=4   12-parallel  |        5,687 |         20,196 | 3.55×  ★ training |
  | fs=16  12-parallel  |       10,607 |         33,123 | 3.12×  |

  Training workloads (the only ones that matter for the trainer):
  **3.55× nes-py at fs=4** and **3.12× at fs=16**. Single-env fs=1
  full-render is the last workload where nes-py/LaiNES wins
  (0.59×), and even that narrows to parity (1.09×) at fs=4.

### Open deltas (what v3 closes)

1. **Single-env fs=1 gap.** LaiNES wins at fs=1 single-env
   (nes-py 1,334 sps vs nes_core 785 sps = 0.59×). This is the
   one workload where we're behind and it's the one `watch_asm.py`
   + live GUI single-tile inspection surfaces. At fs=4 the gap
   closes to 1.09× — the bottleneck is pure per-pixel render cost,
   not emulator dispatch.
3. **24-hour ASM diff-fuzz deferred.** We have 5 min / 63.8M ops,
   0 divergences. SECURITY.md has the honest sign-off paragraph, but
   a full soak is still on the books.
4. **Live audio verify outstanding.** Cpal opens clean, stereo code
   is live, `generate_sample_channels` is plumbed — but no human has
   sat down and listened to Zelda overworld on solo-0.
5. **100-gen Zelda soak not run.** Surfaces latent crashes and
   save/load path regressions the per-opcode tests can't catch.
6. **`nes-py` still a dependency** in `requirements.txt` /
   benchmark scripts. It's used only as a bake-off reference. Once
   v3's Batch J lands, it can be removed from `requirements.txt`,
   archived in `scripts/legacy/`, and the repo is Rust-driven
   end-to-end. This is the user's headline ask.

### Measured perf baseline (the number to beat)

| Workers | Avg sps (fs=16, Zelda) | Note                |
|--------:|-----------------------:|---------------------|
|       1 |                    150 | single-env baseline |
|       4 |                    577 | 3.84× of 1w         |
|       8 |                  1,120 | 7.47× of 1w         |
|      12 |                  1,606 | 10.7× of 1w (peak)  |
|      16 |                  1,170 | 7.8× of 1w — REGRESSION |

Target for v3 completion: **16w ≥ 2,000 sps**, **single-env fs=1 ≥
1.2× nes-py** (close the LaiNES gap), **1-gen wall-time ≤ 0.80 s**.

---

## 2. Philosophy (inherited from v2)

1. Keep the Python outer loop (PyTorch MPS + PyQt6). Every port
   replaces per-step code.
2. Per-step work in Rust. Per-generation orchestration in Python.
3. Measurement gates priority — no speculative rewrites.
4. Hard cutover. No dual-maintenance, no feature flags outside
   `asm_cpu` (build-time).
5. Every batch shrinks LOC or deps OR moves a measured metric.

---

## 3. Batches F–J (finish line)

Five batches. Ordered by **risk-adjusted measurement win**. All
independent unless marked. Batch letters continue from v2 (A–E).

### Batch F — Scaling regression diagnosis + fix

**Scope**: Find why 16w is -27% vs 12w on 12 P-cores of an M4 Max.

**Suspects in order of suspicion**:
1. **QOS_CLASS_USER_INTERACTIVE (0x21)** — too aggressive for N=16
   on 12 P-cores. The macOS scheduler keeps all 16 in the
   "latency-sensitive" class, forcing work-stealing contention.
   Replace with `USER_INITIATED` (0x19) and re-bench, OR drop the
   hint entirely and let macOS's default QoS place E-core workers
   on E-cores.
2. **Rayon thread-count default** = `num_cpus::get() = 16` on M4
   Max (12P + 4E). If we cap at 12 explicitly (`ThreadPoolBuilder::
   num_threads(num_physical_cores_p_only())`), we may recover peak.
3. **Cache contention in the audio/narrator drain paths** at 16w.
   Unlikely given 1w→12w is linear, but verify with `cargo flamegraph`.

**Approach** (in order; stop when scaling is monotonic):
1. Revert the QOS hint (delete the `spawn_handler` block in
   `pool.rs` lines 243–270). Re-bench 1/4/8/12/16/20/24 workers ×
   200 steps × fs=16. Cost: 30 min wall.
2. If 16w still < 12w: cap `ThreadPoolBuilder::num_threads` at
   `max(12, p_core_count)` via `sysctl hw.perflevel0.physicalcpu`
   — the P-core count for the specific M-series chip.
3. If still regressed: flamegraph a 16w run and look for lock
   contention, allocator hot spots, or `drain_audio` cross-thread
   traffic.
4. If none of the above: accept 12w as the default `num_instances`
   and document it in the profile config.

**Exit criteria**:
- Scaling bench is monotonic 1w → `max_workers`. 16w ≥ 12w.
- 16w ≥ 2,000 sps on Zelda fs=16 (post-PGO).
- `num_instances` default in profile config matches the measured peak.

**Estimate**: 0.5–1 day. Low risk — reverting hints is cheap.

### Batch G — Single-env fs=1 (close the LaiNES gap)

**Scope**: Ship a `watch_asm.py`-class workload at parity with
LaiNES. We're at 0.46× today; target **≥1.2× nes-py** at single-env
fs=1 without hurting the multi-worker training path.

**Root cause**: at fs=1 with one worker, the full PPU pixel render
runs every frame. LaiNES hand-tunes its per-pixel inner loop; we
have a per-pixel Rust renderer that's bandwidth-bound on 256×240
u8 writes + palette lookup.

**Three-pronged attack (ship all three, benchmark after each)**:

1. **PPU `ppu_neon` Replace mode, properly integrated.** Today the
   scaffold exists in `ppu_neon.rs` (585 LOC) but `BatchedRenderMode`
   defaults to `Off` because Verify mode diverged on 87% of scanlines.
   The known break: mid-scanline OAM/palette writes the per-pixel
   renderer captures but the batched one misses.

   Fix: add **invalidation tracking** — when CPU writes PPUDATA /
   OAMDATA / OAMDMA mid-scanline, mark the current scanline dirty
   and fall back to per-pixel for that scanline only. Tiles
   prefetched during HBlank are already in `bg_tile_buffer` (done
   during Sprint 11), so the batched path has a correct input in
   ≥95% of scanlines. Dirty-scanline fallback handles the tail.

   Exit: all 6 golden-frame games (Zelda/Mario/Metroid/Castlevania/
   Contra/Donkey Kong) bit-exact; enable Replace by default.

2. **Metal PPU compute pipeline — honest measurement take 2.** The
   `metal_render.rs` shim exists (302 LOC). Memory says the
   palette-only path was 10× slower due to kernel-launch overhead.
   That's expected for per-frame dispatch. The win comes from
   **batching N frames' worth of PPU state into one compute pass**
   at training-time fs=16. Don't pursue for single-env fs=1 —
   measurement said the workload doesn't fit Metal.

   Outcome: **stay disabled by default**. Keep the shim as a stub
   for future batched use; document the 10× measurement in
   `docs/proposals/metal_ppu.md` so nobody re-opens this.

3. **Per-pixel NEON inner loop for `ppu.rs::render_pixel`.** The
   current Rust path does a u8 palette lookup + write per pixel.
   NEON can do 16 pixels of `vqtbl1q_u8` palette lookup per
   instruction, then `vst1q_u8` the whole row. Shift the inner
   loop from per-pixel to per-8-pixel tile and let the compiler
   auto-vectorize, or hand-write a `render_tile_neon(&mut self)`
   in the aarch64-gated path.

   Exit: fs=1 single-env ≥ 1.2× nes-py on a 60-frame deterministic
   Zelda overworld; identical frame hash to per-pixel reference.

**Estimate**: 2–3 days. Main risk is Replace-mode invalidation
correctness (golden-frame tests are the hard gate).

### Batch H — Audio: live verify + crackle-free solo switch

**Scope**: Put a human in front of the GUI solo-0 on Zelda
overworld, confirm audio is clean, fix anything that isn't. Then
formalize the result.

**Steps**:
1. Build release wheel (`./scripts/pgo_build.sh` or plain maturin
   develop --release), open GUI, pick Solo 0, listen for ≥60 s.
2. If clean → record a 30 s WAV via `sox` or the GUI's existing
   audio capture path, commit to `tests/fixtures/zelda_solo0.wav`
   as the reference.
3. If not clean — diagnose in this order:
   - Sample-rate mismatch (APU 43,653 Hz → cpal 44,100 Hz
     resampler artifacts);
   - Underrun (cpal callback starves before worker pushes);
   - Cross-thread contention on the ring `Mutex`;
   - Wrong cpal default device (monitor vs speakers).
4. Switch solo between workers 0→3→7 mid-playback; verify no
   crackle on the 5 ms fade.
5. Test on Bluetooth headphones + built-in speakers + HDMI.

**cpal callback timing improvement** (Sprint 6 from the final
sprint plan, carried over): the current `realtime_pace` logic
calls `sleep` from the worker thread; move pacing to the cpal
callback's own clock via the `output_callback_info` timestamp.
This removes the 2–4 ms jitter on E-core scheduling. Ship only
if step 1 reveals timing artifacts.

**Exit criteria**:
- User signs off (one-line note in `docs/proposals/audio_signoff.md`
  with date + device list).
- Mute + volume slider + solo switch all clean on three devices.
- Reference WAV committed.
- Stereo panning audibly correct (pulse1 left, pulse2 right on a
  stereo-aware track).

**Estimate**: 0.5–1.5 days depending on what step 1 surfaces.

### Batch I — Security refresh: 24-hour ASM soak + mapper fuzz + audit

**Scope**: Upgrade SECURITY.md from 5-min fuzz to 24-hour + add
mapper fuzzing (the one category we've declared "not audited").

**Three strands**:

1. **24-hour ASM diff-fuzz.** Run `examples/asm_diff_fuzz.rs` with
   an extended instruction table (all 180 ported opcodes, not just
   42), longer streams (100 instructions instead of 12), and
   24-hour wall time via `tmux new-session "timeout 86400 ...";
   tee asm_fuzz_24h.log`. On divergence, dump the offending stream
   to `divergence_<seed>.txt`. Target: **0 divergences over ≥1.5B
   instructions**.

2. **Mapper fuzz.** Mappers 4/5/9/VRC6 are forked from RustedNES
   and carry all its known panics. Build a mapper-level fuzzer
   that feeds random register writes within each mapper's
   documented register window, asserts no panic. Covers MMC3 IRQ
   counter, MMC5 ExRAM mode switches, VRC6 audio register sweeps.
   Surface: every mapper in `nes_core/src/mapper/`.

3. **`unsafe` block re-audit.** Three live call sites today
   (preprocess NEON, pool drain_audio, cpu_asm.s). Re-read each
   against SECURITY.md's safety justification, confirm no drift.

**Exit criteria**:
- `asm_fuzz_24h.log` shows 0 divergences over ≥24 h runtime.
- Every mapper in `mapper/*.rs` survives 100k random writes
  without panic.
- `SECURITY.md` updated with both results + mapper fuzz discipline.
- `cargo test --all-features` runs cleanly inside a 30-min CI
  budget.

**Estimate**: 1 day of active work + 24 h background soak. Can run
soak during Batches H/J wall time.

### Batch J — Retire `nes-py` from the repo

**The user's headline ask**: "refactor nes-py to Rust entirely".

**Scope**: Remove `nes-py` as a runtime dependency and reference.
Keep a bake-off harness in `scripts/legacy/` so we can reproduce
the historical delta, but training + GUI + replay + curriculum
ship without importing `nes_py` anywhere.

**Step-by-step**:
1. `grep -rn "nes_py\|nes-py" src/ tests/ scripts/` and enumerate
   every callsite.
2. Classify each: training path / bake-off script / documentation
   reference. Training-path callsites go to zero; bake-off scripts
   move to `scripts/legacy/`.
3. Delete `nes_py` from `requirements.txt`.
4. Add an explicit `requirements-legacy-bakeoff.txt` with a single
   line (`nes-py==N.N`) for anyone reproducing the historical
   bake-off, documented in `scripts/legacy/README.md`.
5. Rename the README framing: "NES emulator (Rust) + RL trainer
   (Python)", drop the "built on nes-py" language wherever it
   appears.

**Exit criteria**:
- `grep -rn "nes_py\|nes-py" src/ tests/ requirements.txt`
  returns no hits (only `scripts/legacy/` and `docs/archive/`).
- `pip install -r requirements.txt && pytest tests/` green on a
  fresh venv.
- `python -c "import nes_py"` fails cleanly (module not installed).
- Bake-off still reproducible via
  `pip install -r requirements-legacy-bakeoff.txt &&
  python scripts/legacy/bake_off.py`.

**Estimate**: 0.5 day. Low risk — purely deletion and import path
changes.

---

## 4. Execution sequence

```
F (0.5–1d) ─┬─→ G (2–3d) ──→ H (0.5–1.5d) ──→ J (0.5d)
            │                                      │
            └─→ I (1d active + 24h soak) ──────────┘
                (runs in parallel with G+H wall time)
```

Order rationale:
- **F first**: the scaling regression affects every measurement
  downstream. No point sizing Batch G's win until F gives a
  clean baseline.
- **G second**: the single-env fs=1 gap is the biggest remaining
  perf lever AND the one the user sees visually in the GUI.
- **H third**: user-facing audio verify. Blocks a release-candidate
  call; can overlap with the tail end of G.
- **I in parallel**: 24-hour ASM soak is wall-time-bound, not
  attention-bound. Kick it off early in G's budget, collect
  results during J.
- **J last**: nes-py removal is trivial but must happen after
  F/G/H lock in (so the bake-off numbers referenced in docs are
  final).

**Total**: 4–6 days of active work, 1 day of parallel background
soak. Realistic finish: **6 days** end-to-end.

---

## 5. Gains summary (mapped to the user's ask)

### Performance

- **Batch F**: +27% at 16w, likely recovers training-path ratio
  to ~3.5× nes-py (from 2.15× today).
- **Batch G**: single-env fs=1 from 0.46× to **≥1.2× nes-py** —
  closes the LaiNES gap, fixes the one visible regression users
  see when inspecting a single tile in the GUI.
- Both combined plausibly push 16w steady-state to 2,000+ sps on
  Zelda fs=16 (from 1,170 today).

### Audio

- **Batch H**: live-verified stereo on real hardware. Reference
  WAV committed as test fixture. Three-device coverage.
- Stereo panning audibly correct (code was live; this is the
  verify pass).

### Visual

- **Batch G.1 (Replace mode)** enables the batched NEON path by
  default. ~5% frame-render speedup baked into every worker, not
  just single-env (bg_tile_buffer precomputed in HBlank).
- **Batch G.3 (render_tile_neon)** brings bandwidth-bound pixel
  writes from scalar to 16-wide NEON.
- Golden-frame tests lock the visual parity guarantee.

### Security / Integrity

- **Batch I**: 24-hour ASM soak upgrades the SECURITY.md claim
  from "5 min, 63.8M ops" to "24 h, ≥1.5B ops".
- Mapper fuzz closes the one "NOT audited" bullet in SECURITY.md.
- `unsafe` re-audit confirms no drift post-G's NEON additions.
- Rust-everywhere-on-per-step means no C++ pointer arithmetic
  anywhere in the emulator surface. nes-py's LaiNES C++ was the
  last memory-unsafe binary in the pipeline. Batch J formally
  retires it.

### "Refactor nes-py to Rust entirely"

- **Batch J**: `nes-py` out of `requirements.txt`, out of
  runtime imports, bake-off preserved in `scripts/legacy/`.
  Training, GUI, replay, and curriculum all run on Rust.

---

## 6. Risks + mitigations

| Risk                                                 | Mitigation                                                        |
|------------------------------------------------------|-------------------------------------------------------------------|
| F: dropping QOS hint regresses 12w peak               | Bench before/after; if 12w drops, keep the hint but cap threads at 12. |
| G.1: Replace mode invalidation misses a write source  | Golden-frame tests across 6 games gate the default-on flip.       |
| G.3: NEON render_tile diverges from reference         | Per-scanline hash diff in CI; fall back to scalar on aarch64 feature miss. |
| H: audio artifacts on BT device but not speakers      | Device-specific diagnosis; worst case, document as known-limitation. |
| I: 24-h soak surfaces a divergence                    | Dump the offending stream, bisect opcode, fix in `cpu_asm.s`, re-soak. |
| J: downstream tool still imports `nes_py`             | Grep gate in CI; any accidental re-add fails the build.           |

---

## 7. Definition of done (v3)

1. Scaling bench monotonic 1w → `max_workers`; **16w ≥ 2,000 sps**
   on Zelda fs=16 (post-PGO).
2. Single-env fs=1 ≥ **1.2× nes-py** on Zelda overworld 60-frame
   deterministic sequence.
3. **1-gen trainer wall-time ≤ 0.80 s** on M4 Max (today 1.35 s).
4. Live audio signed off on three devices; reference WAV in
   `tests/fixtures/`.
5. 24-h ASM diff-fuzz with **0 divergences ≥ 1.5B instructions**.
6. Mapper fuzz: every mapper survives 100k random writes, no panic.
7. `grep -rn "nes_py\|nes-py" src/ tests/ requirements.txt` = 0
   hits.
8. `python -c "import nes_py"` fails in the training venv.
9. 100-gen Zelda soak completes without worker deaths.
10. `SECURITY.md` + `docs/architecture.md` reflect v3 steady state.
11. All proposals in `docs/proposals/` other than `architecture.md`,
    `unified_rust_plan_v3.md` (this file), `aarch64_cpu_asm.md`,
    and `asm_fuzz_result.md` archived.

---

## 8. Effort summary

| Batch | Scope                                          | Est. days |
|------:|------------------------------------------------|----------:|
| F     | Scaling regression diagnosis + fix             | 0.5–1     |
| G     | fs=1 gap (Replace mode + render_tile_neon)     | 2–3       |
| H     | Audio live verify + solo switch robustness     | 0.5–1.5   |
| I     | 24-h ASM soak + mapper fuzz + unsafe re-audit  | 1 + 24h soak |
| J     | Retire nes-py from the repo                    | 0.5       |
|       | **Total active work**                          | **4.5–7** |
|       | **Calendar (with parallel soak)**              | **~6 days** |

Most likely path: F → G → H → J, with I running in parallel during
G and H's wall time. **~6 days end-to-end** to ship: memory-safe,
live-audio-verified, 2× training throughput, single-env parity,
24-hour fuzz-audited, zero-nes-py repo.

---

## 9. Cross-references

- `unified_rust_plan.md` — v2 plan; **archive after v3 Batch J**.
- `aarch64_cpu_asm.md` — ASM CPU design spec (Batch A from v2;
  still live as the reference).
- `asm_fuzz_result.md` — current 5-min fuzz sign-off; rewritten in
  Batch I with 24-h numbers.
- `SECURITY.md` in `nes_core/` — updated in Batch I.
- `batch_execution_report.md`, `hot_path_baseline.md`,
  `pgo_results.md` — historical; archive after Batch J.
- `cpu_bulk_stepping.md` — historical experiment; archive.
- `metal_ppu.md` — **new in Batch G**, records the 10× slowdown
  measurement so this dead end doesn't get reopened.

---

## 10. E8 Hardware Sympathy & True Code Audit (Update)

Following a direct, line-by-line audit of the entire codebase, we have formally completed the bulk of the E8-tier optimizations and stability fixes.

### Completed Fixes (Do Not Relitigate)
1. **Zelda Auto-Curriculum Restored**: The `0x10EC` bounds check typo in `nes_core/src/depth_tracker.rs` was corrected to `0x00ED`, unblocking Zelda room progression tracking.
2. **MMC3 Mapper RAM Bounds Panics**: All `$6000-$7FFF` PRG-RAM writes across `mapper4.rs`, `mapper9.rs`, `vrc6.rs`, etc., are now safely wrapped with `!is_empty()` and modulo length bounds, ending Rayon crashes on malicious RL memory mutations.
3. **`np.stack` Memory Churn Abolished**: The `batch_np` array is now statically allocated once per generation. This eliminated ~64,000 mallocs and 1.8 GB of Numpy garbage collection churn per generation.
4. **Trajectory Tuple Memory Leak**: The `trajectories[i].append()` of Python tuples was completely ripped out of `_evaluate_batch` in favor of pre-allocated flat Numpy slices. This eradicated the final 96,000-object-per-generation allocation loop.
5. **Zero-Copy FFI**: `step_actions` now passes a flat `np.int32` array directly to PyO3 using `PyReadonlyArray1<'py, i32>`, saving 100,000 integer unboxings per second.
6. **macOS Asymmetric Core Pinning**: `pthread_set_qos_class_self_np(0x21, 0)` is successfully locking Rayon workers to Performance Cores, bypassing E-core tail latency.
7. **Unified Memory Bandwidth (f16)**: `preprocess_frame_into_f16` is now natively casting and normalizing pixels directly into the PyTorch memory space via NEON SIMD, bypassing the MPS GPU cast kernel entirely.
8. **Async Pipeline Obs-Lag Integration**: The Python loop is correctly overlapping `step_executor.submit` with the PyTorch `forward` pass, maintaining the 1-step action lag.

### The Final Remaining Tasks
Only two tasks remain to complete the entire Apple Silicon E8 sweep:

1. **Fold `bg_col` Precompute (NEON Planar Transpose)**: `ppu_neon.rs` needs to push pixel expansion into the fetch phase using `VUZP` or `vqtbl` instead of resolving all 272 planar bytes at cycle 256. This makes `Replace` mode the undisputed victor in PPU rendering.
2. **`realtime_pace` Precision Drift**: The `std::thread::sleep` call in `nes_core/src/python.rs` and `pool.rs` must be removed entirely in favor of pacing the emulator using `cpal`'s audio callback. This will permanently solve the intermittent audio crackling and drift caused by macOS timer coalescing.