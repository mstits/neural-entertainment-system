# PGO (Profile-Guided Optimization) — measured win

**Date**: 2026-04-20
**Hardware**: M4 Max MacBook Pro, 128 GB
**Workload**: `scripts/bench_hot_path.py --workers 16 --steps 200 --frame-skip 16`

## Headline

| Metric                | Pre-PGO | Post-PGO | Δ          |
|-----------------------|--------:|---------:|-----------:|
| `pool_step` per step  | 20.11 ms| 10.47 ms | **−48%**   |
| `policy_forward`      | 2.28 ms | 1.83 ms  | −20%       |
| Total per step        | 22.53 ms| 12.42 ms | **−45%**   |
| **Worker-steps/sec**  | **710** | **1289** | **+81%**   |

And on the real-world 1-gen trainer (`scripts/test_trainer_one_gen.py`,
4 workers, stub reward, 120 step budget):

| Config           | Mean wall-time | Min wall-time |
|------------------|---------------:|--------------:|
| Pre-PGO baseline | 2.54 s         | —             |
| Post-PGO         | **1.35 s**     | **1.28 s**    |

**Net: 47% faster headless training on M4 Max, pure build-system change.
Runtime code is identical.**

## Why it works

PGO instruments a build, runs a representative workload to profile
branch frequencies + inline hotness, merges the profile, and rebuilds
with the profile feeding rustc's optimizer. On emulator-heavy code:

- The 6502 instruction decoder has ~150 branches; PGO tells rustc
  which are hot so they become the fast path in generated code.
- PPU tick's `skip_render` branch is now a cold hint (most calls take
  the slow path because we use `frame_skip=16` where 15 of 16 frames
  take the skip path — wait, that's OPPOSITE — skip_render is the
  HOT path for 15/16 frames). PGO correctly identifies it.
- Mapper `enum_dispatch` branches resolve to the mapper in use for
  the profiled ROM (Zelda = MMC1).
- Small functions in the cycle path get unconditionally inlined
  even past the default threshold.

## How to apply

```bash
# First time (or after any nes_core change):
bash scripts/pgo_build.sh full

# Subsequent rebuilds (reusing cached profile data):
bash scripts/pgo_build.sh apply
```

The `full` mode instruments, runs a workload (~3 min), merges
profdata, rebuilds. The `apply` mode skips the instrumented build
and just applies the cached profdata (~15 sec).

**PGO is NOT automatic on `maturin develop --release`.** Any rebuild
without the `pgo_build.sh` wrapper drops back to the non-PGO wheel.
The script is idempotent — run it after any `nes_core/src/*` change
that warrants a fresh perf pass.

## Workload caveat

The profile data was collected on `scripts/bench_hot_path.py` with
the **Zelda** ROM. Training on a different game (Mario, Contra,
Metroid) exercises different mappers + reward code — the wins may
be smaller. For game-specific runs, re-generate with:

```bash
# Edit scripts/pgo_build.sh to point at the workload that matches
# your training profile, or pass `train` as the workload arg:
bash scripts/pgo_build.sh full train
```

## Storage

- `nes_core/pgo/raw/` — raw `.profraw` dumps (output of instrumented
  run, ~several MB per run, ~30 files). Gitignored.
- `nes_core/pgo/nes_core.profdata` — merged, used by `-Cprofile-use`.
  10 MB, also gitignored. Regenerate via `scripts/pgo_build.sh`.

## What PGO does NOT help with

- **PyTorch MPS path** — the `policy_forward` 20% win is incidental
  (PGO'd the pyo3 glue and numpy calls, not the Metal kernels).
- **Python code** — no effect.
- **Anything that doesn't run during the instrumented workload** —
  e.g. game-specific reward functions on non-Zelda games.
