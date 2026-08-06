# Contributing — Neural Entertainment System (NES)

Thanks for your interest. This project is a Rust Nintendo Entertainment
System emulator paired with a Python RL training framework — the recursion
in the name (a NES that emulates a NES) is intentional. Contributions are welcome — bug reports, mapper
support, perf improvements, new reward functions, alternative trainer
algorithms, anything.

## Setup

Targets macOS on Apple Silicon. Other targets compile but aren't the primary
focus.

```bash
git clone <your-fork-url>
cd macos-emulation-and-training
bash scripts/install_macos.sh  # creates .venv, builds the Rust wheel
source .venv/bin/activate
```

## Build

The Rust wheel is built via `maturin`:

```bash
make build              # release, asm_cpu enabled
make build-pgo          # release + PGO instrumentation pass (~3 min)
make build-pgo-apply    # reapply existing PGO profile (~15 s)
```

After every Rust edit you need `make build` (or one of the PGO variants) for
the change to reach Python.

## Test

The whole tree is gated by tests. Two levels:

```bash
make test    # pytest, incl. slow real-emulator guards — Python trainer + utils
make parity  # nes_core vs reference oracle, ~110 s — the fidelity gate

# Rust-side
cd nes_core && cargo test --all-features
```

`make parity` is the gate that catches CPU / PPU / mapper timing
regressions. **Don't merge anything that breaks it.**

For PR-shaped changes:

```bash
make test && make parity   # everything green before you push
```

The CPU is also continuously validated by `nes_core/examples/asm_diff_fuzz.rs`
(differential fuzz of the AArch64 ASM core vs the pure-Rust reference). For
big CPU changes, run a long soak:

```bash
ITERATIONS=10000000 INSTRS=16 ./scripts/asm_diff_fuzz_soak.sh
```

## Code style

- **Rust:** rustfmt defaults (`cargo fmt`). Clippy on the hot path is a
  guideline, not a rule — some intentional unsafety + manual SIMD lives in
  `cpu_asm.s`, `ppu_neon.rs`, and the rayon pool. New unsafe code should
  have a comment explaining the invariant it relies on.
- **Python:** ruff defaults; type hints encouraged but not required.
- **Comments:** prefer to document *why* something is the way it is, not
  *what* it does. The codebase has a lot of "this looks weird because
  $REASON" comments — those are useful. Tutorial-grade comments aren't.

## Where to file issues

Bug reports go in GitHub Issues. Useful context to include:

- **Rust-side bug:** the offending ROM (or a minimal repro), the failure
  signature (panic / wrong output / timing divergence), and the closest
  matching `tests/parity/` test if any.
- **Trainer bug:** the trainer log line where it diverges, the
  `metrics.jsonl` snippet, the genome / generation it happened on if known.
- **Build / install:** OS version, Python version, `cargo --version`, what
  `bash scripts/install_macos.sh` printed at the failing step.

Please grep `nes_core/KNOWN_ISSUES.md` before filing — known-open issues
already have diagnosis notes.

## Adding a mapper

1. Implement the trait in `nes_core/src/mapper/mapperN.rs` (look at MMC1 /
   MMC3 / VRC6 for examples spanning the complexity range).
2. Wire it into `MapperEnum` in `nes_core/src/mapper.rs` (the
   `from_cartridge` match + the `State` enum).
3. Add at least one ROM to `tests/parity/mesen_tapes/` and a regression test
   that asserts byte-exact RAM after N idle frames.
4. Run `make parity`.
5. Run `python scripts/playability_sweep.py` if you have a wide ROM library
   to sanity-check the mapper boots its games.

## Adding a reward function

1. Implement in `nes_core/src/rewards.rs` (look at `MarioReward` —
   `visited_x_max` is a useful pattern for any "reach the end" task).
2. Add a unit test under `#[cfg(test)] mod tests`.
3. Wire into the `Reward` enum + `build_reward` factory.
4. Add a `configs/<game>.yaml` profile with the RAM map and weights.

## Pull requests

Small focused PRs are easier to review and bisect. If your change spans
multiple concerns (mapper + reward + GUI), prefer to split into separate
commits — even within one PR — so the change story reads cleanly in
`git log`.

The README's perf claims are measured numbers, not aspirations. If your
change moves them, please re-measure with `make bench-hot` and note the
delta in the PR.

## License

By contributing you agree that your contributions are licensed under the
existing MIT (Python side) / dual MIT-or-Apache-2.0 (Rust side) terms.
