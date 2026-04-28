<!-- SPLIT_MANIFEST
01-foundation-fork-and-cpu
02-ppu-renderer
03-mappers
04-apu
05-integration-and-cutover
END_MANIFEST -->

# Project Manifest — Full Rust Refactor

Decomposes the "rip the bandaid" replacement of `nes-py` + `nesrs` with a
single purpose-built Rust NES core. Source requirements:
[`full_rust_refactor.md`](./full_rust_refactor.md). Interview transcript:
[`deep_project_interview.md`](./deep_project_interview.md).

## Why these splits

The work has clear technical phase boundaries that map to independent
planning units:

- **Foundation** is everything every other split needs (CPU, bus, ROM
  loader, build pipeline). Has to land first.
- **PPU**, **mappers**, and **APU** are independent of each other once the
  foundation exists. They could be picked up by separate engineers /
  sessions in parallel without stepping on each other.
- **Integration and cutover** absorbs the remaining work that requires a
  feature-complete core: save-state byte API, perf pass (bulk-step,
  zero-copy, SIMD), PyO3 wrapper at the existing `NESEnvironment` shape,
  replacing both backends in the pool, and the dead-code deletion that
  defines "done."

The user committed to a **big bang cutover**, so all of 1-4 land in the
new crate without touching the trainer; only split 5 modifies Python
code in `src/`.

## Splits

| #  | Name                          | Estimate | Depends on | Risk |
|----|-------------------------------|---------:|------------|------|
| 01 | foundation-fork-and-cpu       | 1-2 wk   | —          | Low — forking a known-working core, tests are blargg's CPU suite |
| 02 | ppu-renderer                  | 2-3 wk   | 01         | High — sprite priority, sprite-0 hit, scrolling timing are the classic NES bugs |
| 03 | mappers                       | 1-2 wk   | 01         | Medium — top 10 are well-documented, MMC3 IRQ timing is the trap |
| 04 | apu                           | 1-2 wk   | 01         | Medium — frame counter quirks + DMC DMA stalls are easy to get wrong |
| 05 | integration-and-cutover       | 2-3 wk   | 01, 02, 03, 04 | Medium — touches every Python file that branches on backend; cutover invalidates all existing `.state.bin` files |

**Total midpoint estimate: ~10 weeks solo. Critical path: 01 → 02 → 05.**

## Dependency graph

```
                   ┌─→ 02-ppu-renderer ─┐
01-foundation ─────┼─→ 03-mappers ──────┼─→ 05-integration-and-cutover
                   └─→ 04-apu ──────────┘
```

Splits 02, 03, 04 all unblock once 01 lands. They have zero coupling
between each other — different files, different test suites, different
debugging surfaces. A team of 2-3 could parallelize the middle layer.
Solo, the critical path is 01 → 02 → 05 because PPU is the longest of
the three middle splits.

## Execution order

1. **01-foundation-fork-and-cpu** — must land first, blocks everything.
2. **02, 03, 04** — any order; parallelize if multiple sessions/engineers.
   Solo, recommend 02 first (longest, highest risk; finish it while
   focus is fresh).
3. **05-integration-and-cutover** — last; deletes nes-py + nesrs +
   gme_player + NSF assets + the `--env-backend` flag, validates a
   full Zelda gen-100 training run.

## Cross-cutting concerns

- **Coding standards**: forked CPU code must be reformatted to our
  module layout in split 01 before any other split touches the
  codebase. Splits 2-4 should not have to chase a moving target.
- **PyO3 wrapper shape**: the existing `NESEnvironment` Python class
  contract (`reset`, `step`, `get_audio`, `sample_rate`, `save_state`,
  `load_state`, `FRAME_WIDTH`, `FRAME_HEIGHT`) is the integration
  contract. Splits 1-4 should design their public Rust APIs with this
  shape in mind so split 05's wrapper is thin glue, not a translation
  layer.
- **Testing rigor**: each split publishes its own test suite. Split 05
  is the integrator and runs everything end-to-end including the
  Python `pytest tests/` pass.
- **Memory safety**: no `unsafe` in any split outside the PyO3 boundary
  unless explicitly justified in a comment.

## /deep-plan commands

Once this manifest is approved and directories are created, plan each
split with:

```
/deep-plan @docs/proposals/01-foundation-fork-and-cpu/spec.md
/deep-plan @docs/proposals/02-ppu-renderer/spec.md
/deep-plan @docs/proposals/03-mappers/spec.md
/deep-plan @docs/proposals/04-apu/spec.md
/deep-plan @docs/proposals/05-integration-and-cutover/spec.md
```

Recommended order: plan 01 first and execute it; once 01 is in
review/merged, plan 02-04 in parallel; plan 05 only after 02-04 are
substantially complete to keep the integration spec accurate.
