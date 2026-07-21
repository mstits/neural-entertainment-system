# Crate-hygiene eviction list — `nes_core/src`

Audit date: 2026-07-20. Scope: `nes_core/src` only. **Report only — no code changed.**

## Method + ground truth

The shipped build is `python + asm_cpu` (pinned in `nes_core/pyproject.toml`
`[tool.maturin] features = ["python", "asm_cpu", "pyo3/extension-module"]`; the
PGO path in `scripts/pgo_build.sh` also goes through `maturin develop`, so it
inherits the same feature set). `make build` passes the same string explicitly.

Authoritative dead-code signal — the compiler, against the shipped feature set:

```
cd nes_core && cargo check --features "python,asm_cpu" --message-format=short 2>&1 \
  | grep -iE "never used|never read|never constructed|unused"
```

That run is **clean except two lints** (`unused import: Memory` at `nes.rs:7`,
`unused_mut` at `pool.rs:1349`). So every struct/fn/const in the crate is
reachable in the shipped build. The real "dead scaffolding" is therefore
**feature-gated code never enabled, runtime flags never set, and semantically
no-op reward plumbing** — none of which the compiler can see. Each item below
carries the exact command/test that proves it.

> Ranking = confidence × safety. Highest-confidence, lowest-blast-radius first.

---

## A. Definitely dead — safe to remove

### A1. `simd` Cargo feature — gates nothing, never enabled
- **Where:** `nes_core/Cargo.toml:106` (`simd = []`)
- **Why dead:** declared feature that gates zero code and is passed to zero
  builds. The NEON paths use `#[cfg(target_feature = "neon")]`
  (`preprocess.rs:42/61`), not this cargo feature.
- **Proof:**
  ```
  grep -rn 'feature = "simd"' nes_core/src   # -> no matches
  grep -rn simd nes_core/src Makefile scripts pyproject.toml nes_core/pyproject.toml  # -> no matches
  ```
  Removing the line cannot change any build because nothing references it.

### A2. `pgo` Cargo feature — gates nothing, PGO is driven by RUSTFLAGS
- **Where:** `nes_core/Cargo.toml:109` (`pgo = []`)
- **Why dead:** the Cargo.toml comment claims "the wheel build script consults
  it," but PGO is toggled entirely via `RUSTFLAGS=-Cprofile-generate/-Cprofile-use`
  in `scripts/pgo_build.sh:99-100,210-211`. The feature is never passed to cargo
  and gates no `cfg`.
- **Proof:**
  ```
  grep -rn 'feature = "pgo"' nes_core/src            # -> no matches
  grep -rn -- '--features pgo\|features.*pgo' scripts Makefile nes_core/pyproject.toml  # -> no matches
  grep -n 'profile-generate\|profile-use' scripts/pgo_build.sh  # -> RUSTFLAGS, not a feature
  ```

### A3. Unused import `Memory` in `nes.rs`
- **Where:** `nes_core/src/nes.rs:7` (`use crate::memory::{Memory, Ram};`)
- **Why dead:** `Memory` (trait) is not referenced in `nes.rs`; only `Ram` is.
- **Proof:** the shipped-feature `cargo check` emits
  `warning: unused import: Memory --> src/nes.rs:7:21`. Removing an
  rustc-flagged unused import is definitionally safe (change to
  `use crate::memory::Ram;`).

### A4. Needless `mut` on the worker closure in `pool.rs`
- **Where:** `nes_core/src/pool.rs:1349` (`let mut work = || { ... }`)
- **Why dead:** the binding is never mutated.
- **Proof:** shipped-feature `cargo check` emits
  `warning: variable does not need to be mutable --> src/pool.rs:1349:21`.
  Lint-only cleanup; drop the `mut`.

### A5. Metroid `new_item` reward weight — plumbed but never read
- **Where:** field `nes_core/src/rewards.rs:2094` (`_new_item_bonus: f64`),
  stored at `rewards.rs:2144`, fed from `build_reward` at `rewards.rs:4501`
  (`w(weights, "new_item", 30.0)`).
- **Why dead:** the field is underscore-prefixed (compiler-silenced) and never
  read anywhere in `MetroidReward::compute`; the author's own comment says
  "item flags live in WRAM (unreachable)". A `new_item` weight in a Metroid
  profile silently does nothing. (Contrast Zelda's `new_item` at
  `rewards.rs:4443`, which **is** applied at `rewards.rs:744` — do not touch
  that one.)
- **Proof:**
  ```
  grep -n "_new_item_bonus" nes_core/src/rewards.rs   # stored at 2144, never read
  # 2094 shows the "unreachable" comment; 4501 shows "// unused: item flags are WRAM-only"
  ```
  Removal = drop the field, its `new(...)` parameter, and the `build_reward`
  line (3 mechanical sites, all inside the private `rewards` module).

---

## B. Looks dead — verify before removing

### B1. `metal` feature + `metal_render.rs` (302 lines) + `objc2*` deps
- **Where:** feature `Cargo.toml:132`; deps `Cargo.toml:52` (`objc2`),
  `:53` (`objc2-metal`), `:67` (`objc2-foundation`); module
  `nes_core/src/metal_render.rs` (whole file, `#![cfg(feature = "metal")]` at
  line 34); declared only at `lib.rs:39-40`.
- **Why it looks dead:** the `metal` feature is enabled by **no** build
  (`make build`, `pyproject`, `pgo_build.sh` all omit it), so the module never
  compiles into a shipped artifact. It is **not wired into the render path** —
  `metal_render` is referenced only by `pub mod metal_render` in `lib.rs`; there
  is no call site in `ppu.rs` / `nes.rs` / `sink/`. Its own header says the v1
  palette kernel is "**10× slower** than the CPU path... Don't use v1 in
  production" (`metal_render.rs:25-32`).
- **Verify first because:** this is deliberate future scaffolding (v2 fused
  tile-decode kernel) — see memory note "Metal PPU is future work." Removing it
  also drops three optional deps. Confirm with the owner that the Metal PPU bet
  is abandoned before evicting; if kept, it should carry a "not wired / not
  shipped" banner.
- **Proof:**
  ```
  grep -rn "metal_render\|feature = \"metal\"" nes_core/src   # only lib.rs + the module itself
  grep -rn -- "metal" Makefile scripts nes_core/pyproject.toml # no --features metal anywhere
  cargo check --features "python,asm_cpu,metal"   # compiles, but nothing calls it
  ```

### B2. `disable_asm_cpu` runtime flag — never set; disable-arm never fires
- **Where:** field `nes_core/src/nes.rs:39` (default `false`, `:103`); guard
  `nes.rs:244` (`if asm_handler_cycles > 0 && !self.disable_asm_cpu`); Python
  setter `nes_core/src/pool.rs:1047` (`fn set_disable_asm_cpu`).
- **Why it looks dead:** nothing in `src/`, `scripts/`, `configs/`, or `tests/`
  ever calls `set_disable_asm_cpu`, so the flag is always `false` and the
  interpreter-only fallback arm (ASM disabled) is never taken in any run. It is
  the only Pool setter with zero external callers.
- **Verify first because:** it is a **public PyO3 API** and an intentional perf
  escape hatch (the setter doc: "asm_cpu wins ~15% on single-env headless
  cold-boot ... off by default"). It may be called from notebooks/experiments
  outside this repo. Confirm no external caller before removing the field +
  setter + guard.
- **Proof:**
  ```
  grep -rn "disable_asm_cpu" src scripts configs tests   # no matches (Python/config side)
  # setter-caller census: set_disable_asm_cpu is the only 0-caller Pool setter
  ```

### B3. Diagnostic / A-B-only features never in any shipped config
- **Where (all `Cargo.toml`):** `asm_hit_counter:126`, `ppu_neon_stats:153`,
  `mmc1_prg_bankmath_ab:140`, `mmc1_chr_dispatch_ab:147`.
- **Why they look dead:** none is enabled by any build or config; they never
  fire in shipped runs.
- **Verify first / likely KEEP:** unlike A1/A2 these **do gate real
  instrumentation** — `ppu_neon_stats` guards 17 cfg sites in `ppu.rs`,
  `asm_hit_counter` an `AtomicU64` counter in `nes.rs:345`/`cpu_asm.rs`, and the
  `mmc1_*_ab` pair the A/B baseline paths in `mapper/mapper1.rs`. They are
  purpose-built measurement harnesses documented as re-verification tools.
  Flagged here only because the task asked for "flags that never fire in shipped
  configs"; **do not evict on hygiene grounds alone** — that would delete a
  working A/B rig. Decision is the owner's, not a mechanical cleanup.
- **Proof:**
  ```
  for f in asm_hit_counter ppu_neon_stats mmc1_prg_bankmath_ab mmc1_chr_dispatch_ab; do
    grep -rn -- "--features.*$f" Makefile scripts nes_core/pyproject.toml; done   # nothing
  grep -rc 'feature = "ppu_neon_stats"' nes_core/src/ppu.rs   # 17 real cfg sites -> not empty decls
  ```
- **Note:** `ppu_batch_stats` (`Cargo.toml:166`) is **NOT** in this list — it is
  live via `make ppu-batch-profile` (`Makefile:106`,
  `examples/ppu_batch_profile.rs`). Leave it.

---

## C. Ruled OUT — looks dead by reputation, is actually LIVE (do NOT remove)

Documented so a future pass doesn't "clean these up" and break the emulator.

- **`cpu_asm.rs` / the whole `asm_cpu` core** — LIVE. It is a default shipped
  feature and `disable_asm_cpu` defaults `false`, so the ASM fast path runs on
  every eligible instruction. The residual NMI/PPU-batching bug that prompted
  the old "ASM CPU disabled" memory note was **closed 2026-04-26** (see the
  `[tool.maturin]` changelog block in `pyproject.toml`; byte-exact vs the
  interpreter over 200k instructions). The memory note is stale.
- **`nes_asm_op_unimpl` fallback (`cpu_asm.rs:300`, opcode `0x02` KIL at
  `cpu_asm.rs:2472`)** — LIVE, intentional. The ASM core is a partial port;
  unported opcodes/MMIO return to the Rust interpreter. This is the architecture,
  not dead code — the fallback executes constantly. There are **no** disabled or
  commented-out opcode registrations in `install_opcode_table`.
- **`ppu_neon.rs` batched renderer / `BatchedRenderMode::Replace`** — LIVE.
  Enabled by `configs/smb_oneshot_tiles.yaml:29` and
  `configs/smb_consolidate_level_tiles.yaml:67` (`batched_render: replace`),
  wired via `pool.rs set_batched_render_mode` and `trainer.py:2207`.
- **All 16 per-game `*Reward` structs + `GenericReward`** (`rewards.rs`) — LIVE
  product surface, reachable through `build_reward` (dispatched by ROM name).
  cargo flags them "never constructed" only in a `--no-default-features` build
  because `build_reward` is `python`-gated; they are all constructed in the
  shipped build. Not dead — this is multi-game capability.
- **`narrator.rs`, `game_genie.rs` (Cheat), `depth_tracker.rs`,
  `smb_tile_extract.rs`** — LIVE under `python` (used from `python.rs` / `nes.rs`);
  they appear in the no-`python` dead-code report only because their consumers
  are feature-gated.
- **`preprocess.rs:47 rgb_to_grayscale_scalar` (`#[allow(dead_code)]`)** — KEEP.
  It is the non-NEON portability fallback (`#[cfg(not(... neon))]` at line 42);
  dead on M-series only, required for other targets.
- **`apu.rs:294 generate_sample_channels`, `audio.rs` legacy entry points** —
  live under `python` (called by the mixer); the `#[allow(dead_code)]` covers
  the no-`python` build.

---

## Suggested order of operations

1. Land A1–A4 immediately (zero behavior change; `cargo check` proves each).
2. Land A5 with a one-line changelog note (drops a no-op reward key).
3. Take B1 and B2 to the owner as explicit keep/kill decisions (public API /
   future-bet scaffolding).
4. Leave B3 as-is unless the owner declares the A/B rigs retired.

Regression gate for any removal: `make build` + `make selftest` + `make parity`
must stay green (the parity harness catches PPU/reward regressions the compiler
cannot).
