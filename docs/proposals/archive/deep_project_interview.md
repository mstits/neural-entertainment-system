# Deep-Project Interview — Full Rust Refactor

Source requirements: `docs/proposals/full_rust_refactor.md`

## Decisions captured

### 1. Cutover strategy: **Hard cutover**

Delete `nes-py` and `nesrs-py` the day the new core boots Zelda end-to-end.
No parallel-backend transition period. Forces ourselves to ship a complete
replacement and avoids re-introducing the dual-maintenance split that
this refactor exists to delete.

**Implication for splits:** integration is a single sharp cutover phase,
not a gradual deprecation. The "delete dead code" work concentrates in
one final integration split.

### 2. Mapper coverage: **Top 10 hard-coded**

Implement NROM, MMC1, UxROM, CNROM, MMC3, MMC5-basic, AxROM, ColorDreams,
GxROM, MMC2 in the initial mapper phase. ROMs with mappers outside this
set refuse to load with a clear, descriptive error.

**Implication for splits:** mapper work is a single bounded phase with
known scope, not an open-ended "add as needed" stream. The mapper trait
design lands in the same split.

### 3. Phasing: **Big bang**

Build the new core to feature-parity in isolation, then cut over in one
operation. No incremental alphas, no triple-backend mid-flight (nes-py +
nesrs + partial new core).

**Implication for splits:** the new core builds in its own dedicated
sub-tree (e.g. `nes_core/`) without touching the trainer until the
integration split. The Python trainer / pool / GUI stay unchanged for
the entire build phase. Splits are shaped around the new core's
internal architecture, not around the Python integration surface.

### 4. Starting point: **Fork a minimal existing Rust NES core**

Start from a small single-author Rust core (e.g. starrhorne/nes),
aggressively rewrite for our needs. Saves the 6502 CPU + iNES loader
work. We still own PPU/APU/mapper rewrites.

**Implication for splits:** the first split is "fork + assess + rewrite
CPU/bus to our coding standards" rather than "implement 6502 from
spec." The PPU and APU splits don't depend on a CPU split because the
forked core ships a working CPU.

## Decisions deferred to defaults

- **Crate location:** new crate at `nes_core/` (PyO3 wrapper at
  `nes_core/python/`), parallel to the existing `src/` Python tree.
  `nesrs-py/` directory deleted at cutover.
- **Save-state migration:** no migrator. Existing `.state.bin` files in
  `checkpoints/auto_curriculum/` are wiped at cutover per the
  proposal's non-goals section. They're early-experimental data, not
  durable assets.
- **Test ROMs:** reference blargg's public-domain test suite via
  download script (`scripts/fetch_test_roms.sh`); do not commit ROM
  bytes to the repo.
- **Effort commitment:** the user invoked `/deep-project` to plan the
  9-13 week solo refactor. Splits assume this commitment.

## Goals re-stated for split shaping

- Performance: ≥2× nes-py throughput, ≥720 steps/s @ 4 workers.
- Audio: real APU at training speed, no NSF/synth fallback in the
  shipped code.
- Save-state: in-memory bytes API, versioned format.
- Memory safety: no `unsafe` outside the PyO3 boundary unless explicitly
  documented.
- Codebase reduction: ~2000-2500 LOC net deletion at cutover.

## Constraints re-stated

- Apple Silicon macOS arm64 primary target.
- Drop-in replacement at the existing `NESEnvironment` Python class
  shape — trainer / pool / GUI code does not change.
- PyO3 0.22 + maturin + abi3-py39.
- All existing tests must pass with the new core as the only backend.
