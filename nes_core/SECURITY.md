# `nes_core` security / integrity notes

Summary of the Batch 8 hardening pass (2026-04-20), refreshed after
the AArch64 ASM 6502 core + Batch B (IPC removal) landed (2026-04-21).
Every `unsafe` block + every FFI boundary audited; this file documents
what was checked, what was changed, and what's still trusted.

## Batch B integrity change (2026-04-21)

`src/emulation/parallel_pool.py` (multiprocessing-based worker pool)
and `src/transport.py` (shared-memory FrameTransport) are **deleted**.
The NES emulator pool is now 100% in-process via `nes_core.Pool`
(rayon). Security implications:

- **No more SHM leaks.** The resource-tracker had chronic false-
  positive leaks on SIGKILL'd trainer runs (64 leaked SharedMemory
  blocks reported per forced-shutdown). That entire class of bug is
  impossible now — no POSIX SHM is ever created on the emulator path.
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

Three call sites in the entire crate (was two; ASM core added one).

### 1. `src/preprocess.rs` — NEON SIMD intrinsics (~22 blocks)

The whole `rgb_to_grayscale_neon` path is `unsafe fn` because ARM
intrinsics (`vld3q_u8`, `vmulq_u16`, etc.) have the `unsafe`
declaration. Safety is bounded by the dispatch function:

- Only called from `preprocess_rgb_to_gray_84`, guarded by
  `#[cfg(target_arch = "aarch64")]`. Non-aarch64 takes the scalar
  fallback, never reaches NEON code.
- Inner loop iterates `chunks_exact(48)` over the RGB input, so every
  `vld3q_u8` reads exactly 48 bytes (16 pixels × 3 channels) starting
  at a guaranteed-in-bounds offset. No partial chunks.
- The tail after the SIMD chunks runs the scalar path on the remainder
  (always < 48 bytes).
- Output buffer is pre-sized by the caller (`resize_area` allocates
  the correct `h*w` u8 vector before calling).

Verdict: unsafe is load-bearing (performance) and soundness is tied
to `chunks_exact`'s contract. Not a risk.

### 2. `src/pool.rs::drain_audio` — `slice::from_raw_parts` on `Vec<i16>`

```rust
let samples: Vec<i16> = std::mem::take(&mut w.audio);
let byte_len = samples.len() * 2;
let bytes_ptr = samples.as_ptr() as *const u8;
let slice = unsafe { std::slice::from_raw_parts(bytes_ptr, byte_len) };
Ok(pyo3::types::PyByteArray::new_bound(py, slice))
```

Safety:
- `samples` owns the allocation; pointer is valid for the scope of
  the function (`samples` drops at end-of-fn, after `new_bound`
  copies its bytes into a Python-owned `PyByteArray`).
- `byte_len = samples.len() * 2` exactly matches the layout of
  `Vec<i16>` — `i16` has size 2, alignment 2; reading as `u8` is
  always valid per the `i16` → `u8` layout-compatibility rule.
- `new_bound` copies into a Python buffer immediately; no dangling
  pointer escapes.

Verdict: minimal `unsafe` with explicit safety comment already
present. Kept as-is.

### 3. `src/cpu_asm.rs` + `src/cpu_asm.s` — AArch64 ASM 6502 core

Gated on `#[cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]`.
Default builds (no flag) skip the entire module and use the pure-Rust
CPU — unaffected by this surface.

- `extern "C"` handler table (one pointer per opcode, 256 × 8 bytes)
  populated once at init. No dynamic writes thereafter. Handlers are
  referenced by static function pointer, not by runtime-computed
  offset.
- PRG ROM is handed to ASM as a raw `*const u8` via
  `Mapper::prg_asm_ptr`. All 6502 PC reads stay within the mapper's
  returned slice bounds (branches into RAM / MMIO take the MMIO
  round-trip path, which calls back into Rust). ROM is never written.
- RAM (`[u8; 2048]`) accessed via `*mut u8`; bounds are enforced by
  the ASM masking `address & 0x07FF` before any load/store. 6502
  can't address past 0x07FF in RAM by construction of the instruction
  set, so the mask is exhaustive.
- MMIO addresses ($2000–$7FFF) trap through
  `nes_asm_bus_read_byte` / `nes_asm_bus_write_byte`, which run the
  live Rust `SystemBus` — the same code path the pure-Rust CPU uses.
  No new bypass.
- Opcode coverage: 151 official + ~30 stable illegal (LAX, SAX,
  DCP, ISC, SLO, RLA, SRE, RRA in zp+abs + NOP variants). 99.97%
  ASM hit rate on Mario Bros. Unknown/unported opcodes fall back to
  Rust cleanly (no UB).
- 103 per-opcode diff tests + 16 integration/smoke tests exercise
  every addressing mode and flag update.

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
landed in `59458f4` — code paths the fuzz exercises millions of
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

## FFI boundary — error handling

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
process via `SIGABRT`. Verified via a forced `panic!()` in a worker
— the rest of the pool continues cleanly.

## Versioned save-state format

Blobs returned by `NESEnvironment::save_state` and
`Pool::save_worker_state` are prefixed with `b"NCST\x01"` (5 bytes).
Version byte bumps when the `nes::State` layout changes so old blobs
refuse to load with a clear error:

- `NCST\x01` body → loaded normally.
- `NCST\x02...` or higher → `PyValueError("unsupported NCST version
  byte ...")`. Prevents silent corruption on layout change.
- No magic prefix → accepted as a legacy naked-bincode body (kept for
  backwards-compat with older `.state.bin` files). Will be flipped to
  strict refusal once every on-disk blob is migrated.

Callers who care about durability (curriculum auto-promotion, play
recordings) should write the prefixed output directly — which
`env.save_state()` + `pool.save_worker_state()` both now do.

## Malformed-input acceptance

The cartridge loader is lenient by design: iNES 1.0 headers with
non-zero reserved bytes 7-15 (NES 2.0 extensions) are accepted.
Malformed magic bytes or truncated headers fail cleanly with a
`PyRuntimeError`, not a panic. Not a security boundary in practice
(trust model: "the user picks the ROM"), but kept structured so a
hostile `.nes` file in the `roms/` directory can't DoS the whole
training run.

## Allocations and memory

- `Vec::with_capacity` is used at size-known allocation sites so
  realloc doesn't surprise on hot paths.
- No arbitrary string parsing from user input — ROM paths are
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
- **Rayon internals** — trusted crate, no audit.
- **cpal** — trusted crate for Core Audio. Buffer management follows
  its documented contract (single thread owns the callback closure;
  rings use `Mutex` for cross-thread push from the worker pool).
