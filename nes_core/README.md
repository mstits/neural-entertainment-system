# nes_core

Purpose-built NES emulator core for RL training. Replaces both `nes-py`
(C++ LaiNES, no APU, fast) and `nesrs` (tetanes-core wrapper, has APU
but 2.5× slower) with one core: real APU audio at training speed.

See `../docs/proposals/full_rust_refactor.md` for the design and
`KNOWN_ISSUES.md` for the in-progress punch list.

## Build

Pure Rust:
```
cargo build --release
cargo test --release
cargo run --release --example boot_zelda -- ../roms/zelda.nes
```

Python wheel (via maturin, in the project's `.venv`):
```
../.venv/bin/maturin develop --release
```

After `maturin develop`, the Python module is importable as
`import nes_core`. The exposed class is `NESEnvironment`, drop-in for
`src.emulation.nes_environment.NESEnvironment`.

> **Maturin install gotcha.** `maturin develop --release` does NOT
> reliably overwrite `.venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so`
> on every build — it sometimes leaves a stale binary behind. After a
> code change, copy the freshly built dylib into place explicitly:
>
> ```
> cp nes_core/target/release/libnes_core.dylib \
>    .venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so
> codesign --force --sign - .venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so
> ```
>
> The `codesign` step is required on macOS 26+ (Sequoia/Tahoe), which
> kills loads of unsigned dylibs with `EXC_BAD_ACCESS (Code Signature
> Invalid)`.

## Diagnostic APIs

`NESEnvironment` exposes a few read-only debug methods used by the
parity / playability harnesses and by the GUI's pre-trap auto-capture:

- `cpu_state() -> (pc, a, x, y, sp, flags, nmi_pended)` — full 6502
  register dump. `flags` uses the canonical `NV.BDIZC` byte layout.
- `get_ram(addr) -> u8` and `get_ram_range(start, end) -> np.ndarray` —
  peek 2 KB CPU RAM without side effects.
- `get_frame() -> np.ndarray` — render the current PPU frame buffer
  to an RGB `(240, 256, 3)` array, no step.
- `save_state() -> bytes` and `load_state(blob)` — `NCST\x01`-prefixed
  bincode snapshot of CPU + RAM + PPU + APU + mapper + cartridge
  state.

## Validation tests

```
cargo test --release --test nestest_validation        # 8991-instruction CPU spec gate
cargo test --release --test opcode_cycle_audit        # 56 opcodes vs LaiNES timing
cargo test --release --test nes20_prg_ram_sizing      # NES 2.0 byte 10 nibble parsing
cargo test --release --test zelda_real_rom            # NES 2.0 zelda.nes integration
cargo test --release --test skip_render_parity        # frame-skip vs full-render
```

The full pyramid (Rust + Python) is documented in
`../docs/ARCHITECTURE.md#validation-harnesses`.
