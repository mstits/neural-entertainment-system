# `nes_core` security / integrity notes

Summary of the Batch 8 hardening pass (2026-04-20), refreshed after
the AArch64 ASM 6502 core + Batch B (IPC removal) landed (2026-04-21),
and refreshed again (2026-08-25) after the `unsafe` inventory below
was found to be stale against the crate's actual `src/` tree. Three
sites had grown to eleven files. Every `unsafe` block + every FFI
boundary audited; this file documents what was checked, what was
changed, and what's still trusted.

## Batch B integrity change (2026-04-21)

`src/emulation/parallel_pool.py` (multiprocessing-based worker pool)
and `src/transport.py` (shared-memory FrameTransport) are **deleted**.
The NES emulator pool is now 100% in-process via `nes_core.Pool`
(rayon). Security implications:

- **No more SHM leaks.** The resource-tracker had chronic false-
  positive leaks on SIGKILL'd trainer runs (64 leaked SharedMemory
  blocks reported per forced-shutdown). That entire class of bug is
  impossible now: no POSIX SHM is ever created on the emulator path.
- **No more process-group SIGKILL requirement.** The GUI's
  `killpg(pgid, SIGKILL)` fallback on stop, the worker-side
  `_parent_death_watchdog` orphan cleanup, and the "10 second grace
  window" beachball are all gone. Stop is now a thread-join inside
  the GUI process.
- **Reduced attack surface.** The shared-memory layer could be
  reattached by anything with the `/dev/shm/<name>` path (local
  privilege required); with that gone, the only cross-process
  boundary left is PyQt6 ↔ user input, which was never trust-
  sensitive.

## `unsafe` surface

**162 lines match `unsafe` across 11 files** (`grep -rn unsafe src | wc -l`
/ `grep -rln unsafe src | wc -l`, re-counted 2026-09-03). This section
previously said "three call sites in the entire crate," which was
accurate in 2026-04-21 but stopped being maintained as the NEON
render kernels (`ppu_neon.rs`, `metal_render.rs`), the in-process
worker pool (`pool.rs`), and four `unsafe impl Send`/`Sync` markers
landed. None of that went unreviewed at landing time. The growth is
real feature work, most of it already carries inline `// SAFETY:`
comments; this document just didn't track it. Re-run the count
yourself:

```
grep -rn unsafe nes_core/src
grep -c unsafe nes_core/src/<file>.rs   # per-file
```

One structural note before the per-file list: `src/lib.rs` carries a
crate-wide `#![allow(unsafe_op_in_unsafe_fn)]`. Rust 2024 edition
normally requires every unsafe operation inside an `unsafe fn` body
to *also* sit in its own nested `unsafe {}` block; PyO3 0.22's macro
expansion trips that lint 74+ times per build, so it's silenced
crate-wide until the 0.23 upgrade. Practical effect: `unsafe fn`
bodies elsewhere in the crate (`preprocess.rs`, `pool.rs`'s NEON
helpers, `ppu_neon.rs`) call raw intrinsics straight in the fn body
without a redundant inner `unsafe {}`, a lint/readability
difference, not a soundness one, since the fn's own `unsafe` still
gates every call inside it.

Files below are ordered by grep hit count, highest first. Per-file
counts include comments that mention the word "unsafe" in prose (a
few, mostly in `preprocess.rs`), not only live `unsafe` keywords,
called out inline wherever that inflates the number.

### 1. `src/pool.rs` - worker-pool concurrency + NEON pixel unpack (82)

The in-process, rayon-parallel worker pool that replaced the old
multiprocessing `ParallelPool`. The `CORRECTED` `step_all_native` ordering comment (`e9dafa8`, 2026-08-28) that previously held one of these matching lines was superseded by the `ParallelSectionGuard` widening fix (DO-25) and no longer exists in that form. Its unsafe surface has three shapes:

- **4 `unsafe impl` markers** (`Sync for WorkerCell`, `Send`/`Sync`
  for `Pool`). `WorkerCell` is an `UnsafeCell<Worker>` newtype;
  sharing `&WorkerCell` across threads is sound only because the
  *only* way to mint a `&mut Worker` from it is the private
  `worker_mut` helper below, and every caller of that helper is
  proven (by inline comment) to hold unique access. `Pool` is
  `Send`/`Sync` for the same reason at the container level: rayon's
  `par_iter().enumerate()` in `step_all`/`reset_all`/`collect_results`
  hands each task a unique index/`&WorkerCell`, and every other `Pool`
  method runs sequentially on the Python trainer thread (serialized
  by the GIL, never overlapping an in-flight `step_all`). Both impls
  carry a multi-paragraph `// SAFETY:` comment above them in the
  source spelling this out. Reused here, not re-derived.
- **~50 call sites** dereferencing a `WorkerCell`: `unsafe fn
  worker_mut(cell) -> &mut Worker` (the sole exclusive-access
  chokepoint, itself carrying a `// SAFETY:` doc comment naming every
  sequential caller) plus its ~40 call sites across `step_all`,
  `reset_all`, and the sequential per-worker setters (`save_worker_state`,
  `set_worker_pace`, `load_worker_state`, `set_batched_render_mode`,
  `load_start_state`, `drain_audio`, …); and ~8 read-only
  `unsafe { &*cell.0.get() }` peeks (`peek_max_x_per_worker`,
  `get_odometer_scene_per_worker`, `get_odometer_per_worker`,
  `odo_debug`, `peek_nametables`, `peek_oam`) which need no
  exclusivity, only the same "sequential with step_all/reset_all"
  contract, each with its own one-line `// SAFETY:` comment. It's a
  large count but one audited contract, not 50 independent designs.
- **`drain_audio`'s `slice::from_raw_parts` on `Vec<i16>`**, unchanged
  from the prior audit: `samples` owns the allocation, `byte_len =
  samples.len() * 2` matches `i16`'s layout exactly, and
  `PyByteArray::new_bound` copies the bytes into Python-owned memory
  before `samples` drops at end-of-fn. No dangling pointer escapes.
- **`xrgb_to_gray_neon` / `xrgb_to_rgb_neon`**, the same NEON
  deinterleave-and-store pattern as `preprocess.rs` below (`vld4q_u8`
  reads exactly 64 bytes per 16-pixel iteration from a slice-derived
  pointer, scalar tail handles the `< 16`-pixel remainder), fused
  straight from XRGB8888 into RGB/grayscale to skip an intermediate
  184 KB buffer per worker per step.
- **6 `unsafe { worker_mut(...) }` call sites in `pool_coverage_tests`**
  (added 2026-09-03 with the pool coverage tests). Test-only, inside
  `#[cfg(test)]`, single-threaded: each takes a `&mut Worker` from a
  pool the test itself owns and no other thread can observe, which is
  the same uniqueness proof every production caller of `worker_mut`
  carries. They add no production surface.

Verdict: one audited concurrency contract (`worker_mut` + its
callers) accounts for the bulk of the count; the NEON and
`from_raw_parts` pieces are bounded the same way the rest of this
document already treats NEON/FFI-copy code. Not a risk.

### 2. `src/cpu_asm.rs` — AArch64 ASM 6502 core (28)

Gated on `#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]`.
Default builds (no flag) skip the entire module and use the pure-Rust
CPU, unaffected by this surface. Of the 28 lines, 6 are inside
`#[cfg(test)]` (`mod tests`, starting line 814): unit tests that call
`nes_cpu_run_block` directly or dump opcode-table addresses for
debugging; they don't ship in any built artifact. The remaining 22
are production:

- Two `unsafe extern "C" { ... }` blocks importing the ASM entry point
  (`nes_cpu_run_block`, with its own `# Safety` doc comment on pointer
  validity/sizing) and the opcode-handler symbol table.
- `install_opcode_table_once` populates the 256-slot dispatch table
  from those symbols exactly once (`std::sync::Once`), then it's
  read-only for the process lifetime: no dynamic writes thereafter.
- `nes_asm_bus_read_byte` / `nes_asm_bus_write_byte`: the `extern
  "C"` MMIO trap callbacks the ASM core calls into. They reconstruct
  `&mut SystemBus` from the raw `bus_ptr` the caller (Rust's
  `try_step_asm`) handed the ASM side one call earlier, and run the
  same `SystemBus::read_byte`/`write_byte` the pure-Rust CPU uses,
  with no bypass. Thread-local `ASM_TICK`/`ASM_STALL_EXTRA` cells (set
  just before entering ASM, cleared just after) let the callback tick
  the *live* PPU/APU sinks for the in-flight `Nes::step` call without
  trait-object dispatch.
- PRG ROM is handed to ASM as a raw `*const u8` via
  `Mapper::prg_asm_ptr`. All 6502 PC reads stay within the mapper's
  returned slice bounds (branches into RAM / MMIO take the MMIO
  round-trip path above). ROM is never written.
- RAM (`[u8; 2048]`) accessed via `*mut u8`; bounds are enforced by
  the ASM masking `address & 0x07FF` before any load/store. 6502
  can't address past 0x07FF in RAM by construction of the instruction
  set, so the mask is exhaustive.
- Opcode coverage: 151 official + ~30 stable illegal (LAX, SAX,
  DCP, ISC, SLO, RLA, SRE, RRA in zp+abs + NOP variants). 99.97%
  ASM hit rate on Mario Bros. Unknown/unported opcodes fall back to
  Rust cleanly (no UB).
- 103 per-opcode diff tests + 16 integration/smoke tests exercise
  every addressing mode and flag update.

`src/nes.rs`'s `tick_impl` (see entry 7 below) is this file's other
half (same feature gate, same call), so read the two together.

Verdict: the ASM surface is bounded by the mask-and-mapper contract
and is behind a feature flag. Panics from the fallback-Rust path
propagate normally through `panic = "unwind"` (verified).

**Differential fuzz results** (2026-04-21 first soak, 2026-04-24
overnight extended soak):

Randomized 6502 instruction streams through both the ASM core and the
pure-Rust reference, asserting byte-exact A/X/Y/SP/P/PC + 2 KB RAM
FNV-1a hash after every step. Generator covers a 42-opcode table
spanning every addressing mode the ASM implements. Streams are
seeded for reproducibility. See `examples/asm_diff_fuzz.rs`.

| run                           | iters       | instr/stream | total instructions | wall      | cases/s  | divergences |
|------------------------------|------------:|-------------:|-------------------:|----------:|---------:|------------:|
| 2026-04-21 first soak        | 100 000 000 | 12           |     1 200 000 000  | 91 min    | 18 320   | 0           |
| 2026-04-24 overnight (post-PGO + post-MMC1-RMW + post-NES2.0-fix) | 100 000 000 | 16 | 1 600 000 000 | 95 min | 17 581 | 0 |

The 2026-04-24 run was specifically scheduled to gate the MMC1
consecutive-write filter and APU frame-counter inhibit init flips
landed in `59458f4`: code paths the fuzz exercises millions of
times. **Zero divergences after 1.6 billion instructions on the
post-fix binary** is the regression-soak confirmation that those
fixes did not perturb opcode-level CPU correctness.

Combined: 2.8 billion 6502 instructions diff'd across both runs,
0 divergences. The per-opcode diff-test suite gates every new ASM
opcode at landing time; the soak is the belt-and-braces composed
-state-machine check.

Re-run command (no code changes needed):
```
cargo run --release --features asm_cpu --example asm_diff_fuzz \
    -- <iterations> <instrs_per_stream> <seed>
```

Full log of the original soak: `docs/proposals/asm_fuzz_result.md`.

### 3. `src/sink/video_sink.rs` — NEON palette-lookup unpack (23)

`Xrgb8888VideoSink::write_frame`'s hot path: turn the PPU's per-pixel
6-bit palette indices into XRGB8888 u32s by table lookup, 16 pixels
per NEON iteration (`xrgb8888_write_neon`). Almost all 23 lines are
individual `unsafe { *palette.add(idx) }` scalar gathers inside the
vectorized loop body (NEON has no native indexed-gather-from-memory
op on this target, so the 16 lookups are done as 16 raw pointer
reads) plus their scalar-tail twins, one audited pattern repeated,
not 23 separate designs. Bounds are covered by the function's own
doc comment: `palette` must point at a run of ≥64 valid `u32`
entries (one 64-color emphasis slice of `XRGB8888_EMPHASIS_PALETTE`),
and every `frame_buffer` byte is a 6-bit index (0..=63) by
construction of the PPU's palette RAM, so every `.add(idx)` lands
inside that run. `vld1q_u8` reads exactly 16 bytes per full iteration
from a slice-derived pointer; the scalar tail covers the `< 16`-pixel
remainder. Only reached via `#[cfg(target_arch = "aarch64")]`; other
targets take the plain-Rust indexing loop right above it.

Verdict: same "chunking bounds the SIMD read, scalar tail covers the
remainder" shape this document already applies to `preprocess.rs`.
Not a risk.

### 4. `src/ppu.rs` — CHR static-pointer cache + in-register NEON (6)

- `unsafe impl Send for Ppu` / `unsafe impl Sync for Ppu`, with an
  inline `// SAFETY:` comment: `chr_cache_ptr` (see next bullet)
  aliases memory owned by the same `Nes` that owns this `Ppu`, and
  Pool workers each own a private `Nes` that rayon ships across
  threads as a unit; the pointer never crosses a `Ppu` instance
  boundary.
- `read_chr_byte`'s `unsafe { *self.chr_cache_ptr.add(address) }`:
  a raw-pointer fast path for mappers that guarantee no runtime CHR
  banking (`mapper.chr_static_ptr()`), refreshed at scanline
  boundaries and reset to null on any state load (`chr_cache_ptr =
  null` at deserialize, forcing the safe `mapper.chr_read_byte`
  dispatch until the next refresh). `address` is a `u16` PPU address
  already masked to the $0000-$1FFF CHR range by the caller.
- Three small in-register NEON helpers gated on `target_feature =
  "neon"` (`shift_background_registers`'s `vshl_n_u16` over a 4-lane
  `[u16; 4]`, plus two 8-lane sprite-counter/pattern-shift ops in the
  per-cycle sprite pipeline), no pointer arithmetic, just vector ops
  over already-in-bounds fixed-size arrays loaded whole.

Verdict: the CHR pointer is scoped and reset defensively; the NEON
bits carry no bounds risk (fixed-size array, no indexing). Not a
risk.

### 5. `src/metal_render.rs` — Metal compute FFI, feature-gated + unused (6)

Gated on `#[cfg(feature = "metal")]`, which is **not** in the crate's
`default` feature set and is not enabled by anything else in `src/`
(`grep -rn 'feature = "metal"'` outside this file returns nothing);
it only compiles when a build explicitly opts in. The file's own doc
comment says why it isn't wired up: measured 10× *slower* than the
CPU path per-frame (Metal dispatch overhead dominates a kernel this
small), kept only to prove the device/pipeline/buffer plumbing ahead
of a planned fused v2 kernel.

The 6 unsafe sites are Objective-C/Metal FFI through `objc2`/
`objc2-metal`: retaining the system default `MTLDevice`
(`Retained::retain`), building a `NonNull` around a Rust palette
slice to hand to `newBufferWithBytes_length_options`, and the compute
encoder calls (`setComputePipelineState`, `setBuffer_offset_atIndex`,
`setBytes_length_atIndex`, `waitUntilCompleted`) that `objc2-metal`'s
safe wrapper types still require `unsafe` for per Apple's underlying
API contract. Buffers are `MTLResourceStorageModeShared` (CPU-visible
unified memory) sized via `ensure_buffers` before every dispatch, and
the two `copy_nonoverlapping` calls copy exactly `n` elements where
`n = src.len().min(dst.len())`'s caller-side `debug_assert_eq!`
enforces `src.len() == dst.len()`.

Verdict: standard FFI-into-a-vendor-framework shape, correctly scoped
behind a non-default feature that nothing in this crate turns on.
Not a risk in the default build; worth a second look before anyone
flips the feature on for production.

### 6. `src/preprocess.rs` — NEON SIMD intrinsics (5, 2 of them real)

Carried over from the prior audit. Of the 5 grep hits, 3 are comment
prose that happens to contain the word "unsafe" (a comment explaining
*why* the fn body doesn't need nested `unsafe {}`, per the crate-wide
lint allow above); the actual code is one `unsafe { rgb_to_grayscale_neon(...) }`
call site plus the `unsafe fn rgb_to_grayscale_neon` it calls.

- Only called from `preprocess_rgb_to_gray_84`, guarded by
  `#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]`.
  Non-aarch64 (or NEON-less) builds take the scalar fallback, never
  reach this function.
- Inner loop iterates in 16-pixel (48 RGB byte) chunks via
  `vec_chunks = n_pixels / 16`, so every `vld3q_u8` reads exactly 48
  bytes starting at a guaranteed-in-bounds offset. No partial chunks
  enter the vector path.
- The tail after the SIMD chunks (`scalar_start..n_pixels`) runs the
  scalar path on the remainder (always < 16 pixels).
- Output buffer is pre-sized by the caller (`resize_area` allocates
  the correct `h*w` u8 vector before calling; `rgb_to_grayscale_into`
  additionally `debug_assert_eq!`s both buffer lengths).

Verdict: unsafe is load-bearing (performance) and soundness is tied
to the chunk-count arithmetic. Not a risk.

### 7. `src/nes.rs` — Send/Sync marker + ASM MMIO tick glue (5)

- `unsafe impl Send for Nes` / `unsafe impl Sync for Nes`, with an
  inline `// SAFETY:` comment: `cached_prg_asm_ptr` aliases memory
  owned by the same `Nes` (the mapper's `prg_asm_window` `Vec`), and
  each Pool worker owns a private `Nes` that rayon ships across
  threads as a unit, same shape as the `Ppu` impl above.
- The remaining 3 sites are `tick_impl`, the monomorphized `extern
  "C"` callback installed into `cpu_asm`'s thread-local `ASM_TICK`
  before every ASM-core call. It reconstructs `&mut SystemBus` and
  `&mut SinkCtx<V, A>` from the raw pointers `try_step_asm` handed
  it one frame earlier on the same call stack, so the sinks it ticks
  are the exact ones `Nes::step`'s caller already owns mutably for
  this call. All 3 sites live inside
  `#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]`, the
  same gate as `cpu_asm.rs` (entry 2), so default builds never
  compile them.

Verdict: same aliasing shape already accepted for `Ppu`/`Pool`; the
callback pointers are same-call-stack, same-thread by construction.
Not a risk.

### 8. `src/ppu_neon.rs` — batched scanline renderer, bench-only (3)

The module's own header comment: **"Status: Bench-only. NOT wired
into the live PPU tick yet."** It's a from-scratch NEON
reimplementation of `ppu.rs::render_pixel`'s per-cycle compositing
loop, built and correctness-gated (`tests::neon_matches_scalar_byte_exact`
asserts byte-for-byte parity against the scalar reference) so a
future per-scanline integration can drop it in without diverging,
but nothing in `src/` calls `render_scanline_neon` outside this
file's own tests/benches yet.

- Dispatch is `#[cfg(target_arch = "aarch64")]` → NEON, else the
  portable `render_scanline_scalar`, the one real unsafe *call site*.
- `render_scanline_neon` itself (`unsafe fn`, `#[target_feature(enable
  = "neon")]`) is called only from that guarded call site, with
  asserted preconditions above it (`tiles.len() >= 33`) and a
  `copy_nonoverlapping` sized `tile_count * 8 ≤ 272` by construction
  (`tile_count = tiles.len().min(34)`), documented inline.

Verdict: unused in production today, so its risk is "wrong bytes in
a future integration," not a live exposure, but it's real unsafe
code and should get the same scrutiny as the rest of this list once
it's wired up. Flagged again under "What's NOT audited" below.

### 9. `src/lib.rs` — crate-wide lint allow (2)

Both hits are the `#![allow(unsafe_op_in_unsafe_fn)]` attribute and
its explanatory comment at the top of the file, not a call site.
Covered in the intro above; listed here only so the file-by-file
total adds up to 162.

### 10. `src/python.rs` — NEON frame repack for PyO3 (1)

`frame_to_numpy`'s `#[cfg(all(target_arch = "aarch64", target_feature
= "neon"))]` block: the same XRGB8888 → RGB8 deinterleave as
`pool.rs`'s `xrgb_to_rgb_neon` (`vld4q_u8`/`vst3q_u8`, 16 pixels per
64-byte stride), building the `(H, W, 3)` array PyO3 hands back to
Python. The loop bound is `while i + 16 <= n`; the defensive scalar
tail after it is dead in practice (`FRAME_PIXELS = 240 * 256 = 61440`
is exactly divisible by 16) but kept for any future non-16-aligned
buffer size, per its own comment.

Verdict: same NEON-unpack shape as entries 1 and 3. Not a risk.

### 11. `src/mapper/mapper1.rs` — test-only raw-slice assertion (1)

The single site is inside `#[cfg(test)] mod prg_window_tests` (test
module starts line 562, the unsafe site is line 717); it never
ships in a built artifact. `assert_chr_window_matches` calls
`std::slice::from_raw_parts(p, 0x2000)` on the pointer
`chr_static_ptr()` returns, to snapshot the CHR window for comparison
against the per-read bank-math oracle in the same test. `p` is backed
by the same 8 KB `chr_window` buffer the mapper allocates and keeps
alive for the duration of the test.

Verdict: test-only, not a production risk.

---

**Summary:** 162 `unsafe`-matching lines across 11 files, up from the
"three call sites" this section claimed as of 2026-04-21. Of the 162,
roughly a dozen are comments rather than code (`preprocess.rs` ×3,
`lib.rs` ×2, plus a few explanatory lines folded into the counts
above), 14 are test-only (`cpu_asm.rs` ×6 in `mod tests`,
`pool.rs`'s `pool_coverage_tests` ×6, `mapper1.rs`'s single site), and one file's entire 6-site surface
(`metal_render.rs`) sits behind a non-default feature nothing else
enables. The rest is real, audited, mostly already commented in
place: one worker-pool concurrency contract (`pool.rs`), one ASM FFI
boundary split across two files (`cpu_asm.rs` + `nes.rs`), and four
NEON pixel-unpack kernels (`pool.rs`, `sink/video_sink.rs`,
`preprocess.rs`, `python.rs`) that share the same "chunked SIMD read,
scalar tail, pre-sized output" shape this document has applied since
the 2026-04-20 pass.

## FFI boundary: error handling

Every PyO3 function in the `nes_core` module either returns
`PyResult<_>` or is infallible by construction. Error mapping:

- **Malformed ROM / iNES parse** → `PyRuntimeError` with the
  underlying error message.
- **Bounds mismatch** (RAM range out of 0x0000..0x0800, wrong worker
  id) → `PyValueError` / `PyIndexError`.
- **Save-state version mismatch** → `PyValueError` with the
  actual version byte seen.
- **Save-state body corruption** → `PyValueError` with the bincode
  error.
- **Panics inside `Pool::step_all`** → caught by `catch_unwind`
  per-worker. The offending worker is marked `dead` and returns
  zero-filled frames/RAM forever after. Other workers are
  unaffected.

The `panic = "unwind"` setting in `Cargo.toml` release profile is
**load-bearing**. Under `panic = "abort"` the `catch_unwind` harness
is ineffective and any worker panic takes down the whole Python
process via `SIGABRT`. Verified via a forced `panic!()` in a worker;
the rest of the pool continues cleanly.

## Versioned save-state format

Blobs returned by `NESEnvironment::save_state` and
`Pool::save_worker_state` are prefixed with `b"NCST\x01"` (5 bytes).
Version byte bumps when the `nes::State` layout changes so old blobs
refuse to load with a clear error:

- `NCST\x01` body → loaded normally.
- `NCST\x02...` or higher → `PyValueError("unsupported NCST version
  byte ...")`. Prevents silent corruption on layout change.
- No magic prefix → **behaviour differs by loader, and the previous
  wording here was wrong.** Corrected 2026-08-25 after an attempted
  "migrate every naked blob, then flip to strict refusal" change was
  investigated and abandoned:

  - `Pool::load_start_state` treats a non-magic file as a **legacy
    action-replay tape**, not as a bincode body: the bytes are replayed
    frame-by-frame as NES controller input on worker 0, and the
    resulting state is snapshotted and broadcast to the other workers.
    This path is live and load-bearing; it is documented nowhere else
    in this file, which is the defect this entry fixes.
  - `Pool::load_worker_state` and `python.rs::load_state` pass a
    non-magic blob to `decode_state` as a bincode body. Verified
    2026-08-25: every naked `.state.bin` currently on disk is **already
    refused** by these two loaders (`invalid u8 while decoding bool`),
    so the permissive fallback protects zero real files today.

  All 8 remaining non-magic `.state.bin` files under `roms/` were
  measured and are **button tapes, not savestates**: 12 distinct byte
  values, all NES button bitmasks (0/1/2/4/8/16/32/64/128 and sums),
  versus 256 distinct values plus an `NCST\x01` prefix on a real state.
  They are consumed by ~15 callers including `tests/parity/
  test_zelda_input_replay.py`, which is inside the `make parity`
  baseline.

  Consequence: "migrate every blob to NCST, then flip to strict
  refusal" is **unsatisfiable as stated**: prefixing a tape with
  `NCST\x01` injects five bogus input frames (`N`=0x4E, `C`=0x43,
  `S`=0x53, `T`=0x54, `\x01`) that desynchronise every replay, and
  routes the file into the bincode branch where it fails. The correct
  close, if this is ever taken up, is: leave `load_start_state`'s
  replay branch permissive **by design**, flip only the other two
  loaders (which already refuse these files in practice), and rename
  the tapes to `*.tape.bin` so the format is not implied by the
  extension. That rename touches ~15 call sites and is not done here.

Callers who care about durability (curriculum auto-promotion, play
recordings) should write the prefixed output directly, which
`env.save_state()` + `pool.save_worker_state()` both now do.

## Malformed-input acceptance

The cartridge loader is lenient by design: iNES 1.0 headers with
non-zero reserved bytes 7-15 (NES 2.0 extensions) are accepted.
Malformed magic bytes or truncated headers fail cleanly with a
`PyRuntimeError`, not a panic. Not a security boundary in practice
(trust model: "the user picks the ROM"), but kept structured so a
hostile `.nes` file in the `roms/` directory can't DoS the whole
training run.

## ROM and state file trust boundary

Files in `roms/`, `checkpoints/`, and `runs/` are loaded without hash verification or signature validation: the trust model assumes the user directly controls what is placed on disk. Provenance checks (e.g., `provenance_check.py`) apply only at specific gates (demo-bank allowlisting, autonomous planner ingestion), not at every file load. Parser defenses (malformed ROM rejection, save-state version/corruption detection) guard against accidental data damage and truncation DoS, but do not verify ROM identity or state integrity. For a single-user local tool, this is an acceptable posture; it would require hardening (known-good ROM library with signed digests, or key-derived state attestation) only if the project grows a second contributor or migrates to shared-machine deployment.

## Allocations and memory

- `Vec::with_capacity` is used at size-known allocation sites so
  realloc doesn't surprise on hot paths.
- No arbitrary string parsing from user input: ROM paths are
  `PathBuf`, action bitmasks are `u8`, RAM addresses are `u16`
  bounds-checked against `0..=0x0800`.
- The APU's PCM ring buffer has a hard cap (~150 ms of 43653 Hz
  samples per worker, ~6.5 KB); fast-forward emulation cannot grow
  audio buffers unboundedly.

## What's NOT audited

- **Mapper implementations** are forked from RustedNES. Top-10 mappers
  are exercised by the 22-ROM smoke test; obscure mappers may still
  `panic!` on unexpected register writes. The panic hits the
  `catch_unwind` harness and marks the worker dead, but the root
  cause is upstream.
- **Rayon internals**: trusted crate, no audit.
- **cpal**: trusted crate for Core Audio. Buffer management follows
  its documented contract (single thread owns the callback closure;
  rings use `Mutex` for cross-thread push from the worker pool).
- **objc2 / objc2-metal / objc2-foundation** (`src/metal_render.rs`,
  `metal` feature): trusted crates wrapping Apple's Metal framework;
  no independent audit of the wrapper's own internal `unsafe`. The
  module they back is feature-gated off by default and not called
  from anywhere else in the crate; see unsafe-surface entry 5.
- **`src/ppu_neon.rs`'s batched renderer**: correctness-tested against
  the scalar reference in isolation (unsafe-surface entry 8), but not
  yet exercised by the mapper-class smoke suite because it isn't
  wired into the live PPU tick. Re-audit its bounds assumptions
  (`tiles.len() >= 33`, the 272-byte `bg_col` copy) before the
  per-scanline integration lands, not after.
