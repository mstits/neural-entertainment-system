# Split 01 — Foundation: Fork + CPU + Bus + iNES Loader

## Context

Source proposal: [`../full_rust_refactor.md`](../full_rust_refactor.md)
Interview decisions: [`../deep_project_interview.md`](../deep_project_interview.md)
Manifest: [`../project-manifest.md`](../project-manifest.md)

This is the foundation split. It establishes the new Rust crate and
brings up CPU + memory bus + iNES ROM loading. Every other split
(02 PPU, 03 mappers, 04 APU) depends on this landing first.

## Key decisions inherited from interview

- **Fork target chosen: [`jasonrhansen/RustedNES`](https://github.com/jasonrhansen/RustedNES)**
  (`rustednes-core` crate). Selected from a survey of 10+ candidates
  (sprocketnes, pinky, nestur, lochnes, rusticnes, Corrosion, etc.).
  Selection criteria: dual MIT+Apache licence, pure-Rust core (no C
  deps), recently active (last commit Feb 2026), well-modularized
  (one-file-per-mapper layout), serde-based snapshot already wired
  (half-built for our save_state work), 9.5k LOC total with mappers
  0/1/2/3/4/7/9 already implemented (7 of our top 10).
- **Aggressively rewrite for our needs** rather than vendor as-is.
  Specifically: PPU rewrite for per-scanline + skip-render fast path
  (the upstream PPU is per-cycle); APU keep mostly intact; expand
  mappers to fill the missing 5/11/66; replace serde snapshots with
  a versioned binary format that's stable across upstream churn.
- **Hard cutover** at the end of split 05. This split must not touch
  anything in `src/` Python — it only creates the new Rust crate.
- **New crate location:** `nes_core/` at the repo root, parallel to
  `src/`. The PyO3 wrapper directory under it (`nes_core/python/`) is
  set up here too but only the Rust library lands in this split — the
  Python binding is split 05's job.

## Deliverables

1. **Forked source assessed and rewritten.** Pick the source crate,
   document the choice and license, port the 6502 CPU + memory bus to
   our crate layout. Code reformatted to our conventions; no
   dead-on-arrival files from the original.
2. **iNES 1.0 ROM loader** that accepts ROMs with garbage in reserved
   bytes 7-15 (matches LaiNES behavior; unblocks the Zelda ROM with
   no external sanitizer). Rejects malformed ROMs with descriptive
   errors, never panics.
3. **Memory bus** with PRG/CHR/RAM regions wired up. Mapper trait
   defined as a placeholder (concrete implementations land in split
   03) so the bus can route reads/writes through it.
4. **Build pipeline:** `cargo build --release` succeeds on macOS arm64
   with LTO + `-C target-cpu=native` configured. CI hook (or local
   `cargo test`) runs blargg's CPU test suite.
5. **Placeholder shapes** for `PpuStub`, `ApuStub` so the bus can
   compile; real implementations land in splits 02/04.
6. **Test ROM fetch script:** `scripts/fetch_test_roms.sh` downloads
   blargg's public-domain CPU/PPU/APU test ROMs to a gitignored
   directory.

## Dependencies

- **External:** Rust toolchain (stable), the chosen forked source
  crate's repo, blargg's test ROM URLs.
- **Other splits:** none — this is the root.

## Provides to other splits

- `Bus` type with read/write methods routed through a `Mapper` trait
  (split 03 fills in concrete mappers).
- `Cpu` type with `step()` advancing one instruction and reporting
  cycles consumed.
- `RomCart` parsed from iNES bytes with PRG/CHR/mapper-id/mirroring
  metadata exposed.
- Build/test infrastructure that splits 02-05 inherit unchanged.

## Risks for this split

- **Forked source license incompatible** with this repo's terms —
  must check and document in the spec before rewriting begins.
- **Forked source quality lower than expected** — if the chosen crate
  is buggy or untested, rewriting takes longer than estimated. Have
  a fallback plan (next-best fork, or accept the from-scratch cost).
- **Coding-standard drift** — splits 2-4 will be hard to start if the
  CPU/bus code keeps churning. Lock the public surface at the end of
  this split.

## Acceptance criteria

1. `cargo test` in `nes_core/` passes; blargg's `cpu_dummy_reads.nes`,
   `instr_test-v5/all_instrs.nes`, `cpu_timing_test6.nes` execute
   without panicking and report PASS via the standard $6000 results
   protocol.
2. `RomCart::from_bytes(...)` parses `roms/zelda.nes` (the dirty-header
   ROM) successfully without an external sanitizer.
3. `cargo build --release` produces a binary with LTO enabled
   (verified via `cargo metadata` / Cargo.toml inspection).
4. The crate compiles into the `Bus` ↔ `Mapper` ↔ `Cpu` topology
   with `PpuStub`/`ApuStub` placeholders so splits 02-04 can begin.
