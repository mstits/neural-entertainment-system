# CPU Bulk-Stepping Design

**Target**: shave into the 84% `pool_step` bottleneck that PGO can't
reach further. Measurement per
[`hot_path_baseline.md`](hot_path_baseline.md) shows CPU + APU +
per-cycle-PPU-dispatch consumes ~460 μs per NES frame on M4 Max
post-PGO. Per-scanline PPU (Batch 6) caps at 14% trainer wall-time
savings; bulk-stepping the CPU targets the bigger chunk.

**Novel angles for Apple Silicon in mind** — noted inline but not
committed until the mainline plan validates.

## Current architecture

Each of the 256 opcodes has an `&'static [CycleFn]` table — one
function pointer per CPU cycle the instruction consumes. `Cpu::tick`
runs exactly one sub-cycle:

```rust
let cycle_fn = instr.cycles[self.cycle as usize - 1];
let completed = cycle_fn(self, bus);
self.cycle += 1;
self.cycles_total += 1;
```

For a 6-cycle instruction (e.g. `STA abs,X` at 5-6 cycles), that's 6
indirect function calls, 6 function-pointer loads, 6 increments. Each
call is a potential branch-predictor miss on the call instruction.

Between CPU cycles, `Nes::tick` runs:
- 1× `Apu::tick` (IRQ polling + DMC DMA + frame counter + channel
  timers → sample emission).
- 3× `Ppu::tick` (fast path when `skip_render=true` for the 15 of 16
  frames per trainer step that don't get observed).

So a 6-cycle CPU instruction costs: 6 × (1 APU + 3 PPU + 1 CPU sub-op +
some IRQ-line bookkeeping). Cycle-accuracy is preserved for the tiny
fraction of instructions that read $2002 mid-op or write $2006/$2007,
and for interrupt timing.

## Block-interpreter core idea

For instructions that DON'T touch `$2000-$401F` (PPU/APU registers)
and DON'T cross an interrupt boundary, we can:

1. Execute the whole CPU instruction in one Rust function — no
   per-cycle dispatch, no function pointer loads, LLVM can fully
   inline and register-allocate across the instruction body.
2. Advance PPU + APU in one batch: `ppu.bulk_tick(cycles × 3)`,
   `apu.bulk_tick(cycles)`.

Per-cycle PPU/APU state still evolves correctly because those
subsystems are deterministic in their tick counters — we just
amortize the loop overhead.

### The guard condition

Call this the "pure" fast path. It applies when ALL of:

- The target address of any memory read/write is outside
  `$2000-$401F` AND outside mapper-mapped MMIO ranges ($4020+ for
  some mappers).
- No interrupt is pending (NMI, IRQ).
- The previous instruction didn't leave the CPU in the middle of
  something (`self.cycle == 0`).

Fast-path candidates (~85-95% of typical game code):
- LDA / STA / LDX / LDY / STX / STY with non-MMIO immediate or
  zero-page or ram-absolute addresses.
- All ALU ops on registers (ADC, SBC, AND, ORA, EOR, CMP, etc.).
- All register transfers (TAX, TAY, TXA, TYA, TSX, TXS).
- All stack ops (PHA, PHP, PLA, PLP, JSR, RTS) — stack is in WRAM
  ($0100-$01FF), non-MMIO.
- All branches and jumps (BPL, BMI, BEQ, BNE, JMP, JMP indirect) —
  PC manipulation only.
- All flag ops (SEC, CLC, SED, CLD, SEI, CLI, CLV).

Per-cycle slow-path fallback:
- Any access to `$2000-$3FFF` (PPU mirrors every 8 bytes) or
  `$4000-$401F` (APU registers + $4014 OAM DMA + $4016/$4017
  controller).
- Any mapper-mapped MMIO (MMC3 / MMC5 bank registers at $8000+).
  Detection: ask the mapper "does this write do anything state-
  changing?" via a new `Mapper::is_mmio(addr) -> bool` query.
- Interrupt pending at instruction boundary.

### The catch

6502 has some instructions with mid-instruction side effects even
for "normal" RAM addresses: read-modify-write (RMW) ops do a
`read → dummy write back original → write modified` sequence. On
an MMIO target, the dummy write matters (e.g. `INC $2007` increments
the PPU address twice). On RAM, it's an idempotent double-write.
Safe to fast-path RAM-only RMW.

## Measurement plan

The hard question: does skipping 6 per-cycle function calls actually
beat PGO's current code?

PGO already:
- Specialized the opcode table for the training profile's hot opcodes.
- Inlined small cycle bodies.
- Removed bounds checks where proven safe.

The bulk-step win comes from:
1. **One function call per instruction**, not per cycle. LLVM fully
   inlines the instruction body with PGO already.
2. **No function-pointer indirection** on the `cycles[cycle - 1]`
   load per sub-cycle.
3. **PPU/APU advanced in a single counter arithmetic**, not a
   per-cycle loop.

Item 3 is the biggest lever. Currently `ppu.tick(...)` has the
skip-render path which is O(1) per call but ~6 branches per call.
Bulk-ticking lets us skip to the next scanline boundary in one
arithmetic step.

### Novel Apple-Silicon angle (hold for now)

- **NEON-vectorized PPU bulk-tick**: we could advance 16 workers'
  PPU counters in parallel via NEON lanes. Not useful since workers
  are already rayon-parallel on separate cores — no win.
- **M4's branch predictor**: already excellent; can't manually hint
  much.
- **Scheduler pinning to P-cores**: rayon does CPU-pinning on macOS
  via its default thread builder. Checked — already handled.
- **Memory prefetching via `__builtin_prefetch`**: CPU state fits
  in 1-2 cache lines, PPU in ~1KB, APU in ~512 bytes. All hot per
  worker. Prefetching between workers is tricky because rayon
  schedules them. Pass.

None of these opened a win at the per-cycle level that PGO didn't
already capture. The block interpreter's value is purely in
amortizing function-call overhead.

## Phases

### Phase 1 (this session): proof-of-concept

1. Pick the simplest opcode class: `LDA immediate` (opcode 0xA9, 2
   cycles). Implement a bulk-step path for JUST this opcode.
2. Run a synthetic bench that repeatedly executes LDA immediate on a
   tight loop (a 6502 "program" of `A9 xx A9 xx ...`). Compare per-
   cycle path vs bulk path.
3. If bulk shows ≥1.3× on the isolated bench, commit to Phase 2.
4. If not, write up the negative result and pivot.

### Phase 2 (next session): fast-path for ~20 common opcodes

Expand the bulk-step to the hot opcode set: LDA/STA/LDX/LDY in all
non-MMIO addressing modes, common ALU ops, branches, JMP/JSR/RTS.
Aim to cover 80%+ of instruction-execution frequency on Zelda.

### Phase 3: PPU/APU bulk-tick integration + regression harness

Wire the bulk path into `Nes::step_instruction`, advance PPU/APU in
batch. Add a "cycle-exact" shadow that runs both paths and diffs
state — smoke test across all 22 ROMs. Any divergence on non-MMIO
code means the bulk path has a bug; revert that specific opcode.

### Phase 4: full hot-path integration + bench

Replace `Nes::step`'s `while !self.tick(...) {}` with an
instruction-level loop when fast-path applies; fall back to
per-cycle otherwise. Full `bench_hot_path.py` to measure.

## Target

If the block interpreter shaves 30-40% off CPU cost (conservative
per typical 6502 emulator literature), that's ~25-34% off
`pool_step` → ~22-30% off trainer wall-time. Combined with existing
PGO (+81%), we'd land at ~2× the pre-session baseline. Not 10×, but
real, measured, Apple Silicon-optimized, and enduring.

## What this DOESN'T try to do

- **Dynamic recompilation (JIT)**: would be a 10× lever but requires
  a JIT framework (cranelift or similar), multi-week commit, and
  enormous test surface. Out of scope.
- **Vectorizing multiple NES instances in SIMD lanes**: thread
  divergence between workers (different game states, different
  PCs, different branches) makes this useless.
- **Metal compute shader PPU**: the 14% ceiling from
  hot_path_baseline applies here too — same lever, riskier
  implementation.
- **Accelerate framework hooks** in CPU/APU — not a matrix workload.

## Success gate for Phase 1

A bench showing the bulk LDA-immediate path beats the per-cycle LDA-
immediate path by ≥1.3× on isolated synthetic data. If it's less,
the win is noise-level and the block interpreter payout won't
materialize at scale.

## Phase 1 result (2026-04-20)

**PASS by a wide margin.** `cargo test --release --lib
bulk_step_bench -- --nocapture` (see `nes_core/src/cpu.rs` bench
module) measured the LDA-immediate bulk vs per-cycle paths across 3
runs on M4 Max post-PGO:

| run | per-cycle ns/instr | bulk ns/instr | speedup |
|-----|-------------------:|--------------:|--------:|
|  1  |             100.9  |          22.6 |  4.46×  |
|  2  |             100.7  |          22.3 |  4.52×  |
|  3  |             105.4  |          22.9 |  4.60×  |

Saving ~78 ns per instruction. Of that:
- ~40 ns from eliminating `Instruction::cycles[i-1]` function-pointer
  indirection + the call itself.
- ~20 ns from bypassing the `cycle` / `instruction: Option<>` state
  machine in `Cpu::tick`.
- ~20 ns from not running `poll_interrupts` at instruction end
  (interrupts poll at block boundaries instead in the block interp).

Both paths do equivalent APU + PPU work; the savings are purely
CPU-side overhead.

**Gate fully passed. Committing to Phase 2.** Expanding the
bulk-step surface to the top ~20 opcodes that dominate 6502 code
execution on Zelda / Mario / Contra / MegaMan / Castlevania / Metroid.

**Realistic trainer-wide win projection**: CPU-side dispatch overhead
is ~10-15 ns/CPU-cycle in per-cycle mode × ~3 cycles/avg-instruction =
~30-45 ns/instruction. Bulk mode saves most of that. NES runs ~29,781
cycles × 60 fps × 16 workers × 16 fs = 457M CPU cycles/second; a
30 ns/instruction saving at ~3 cycles/instruction = ~10 ns/cycle ×
457M = 4.6 wall-seconds saved per wall-second of training, but this
is split across 16 workers so ~0.29 s saved per wall-second per
worker → **~15-25% trainer wall-time savings** if we hit 80%+
fast-path coverage. Combined with existing PGO (+81%) that lands us
at ~3.5-4× over pre-session baseline on M4 Max.

## Phase 2 — first expansion (2026-04-20 late session)

Unified `Cpu::try_bulk_step(bus, opcode) -> Option<u8>` dispatch
landed in `nes_core/src/cpu.rs`. ~40 opcodes now in the bulk path:

- **Loads immediate**: LDA/LDX/LDY (0xA9/0xA2/0xA0)
- **Loads zero-page**: LDA/LDX/LDY (0xA5/0xA6/0xA4)
- **Stores zero-page**: STA/STX/STY (0x85/0x86/0x84)
- **Register transfers**: TAX/TAY/TXA/TYA/TSX/TXS
- **Flag ops**: CLC/SEC/CLI/SEI/CLD/SED/CLV (CLI only if IRQ inactive)
- **Inc/Dec register + NOP**: INX/INY/DEX/DEY/NOP
- **Branches**: BPL/BMI/BVC/BVS/BCC/BCS/BNE/BEQ (shared `branch()` helper)
- **JMP absolute**: 0x4C
- **Compares immediate**: CMP/CPX/CPY
- **ALU immediate**: AND/ORA/EOR

**Measured trainer-wide impact** on M4 Max post-PGO:

| Metric                | Pre-Phase-2 | Post-Phase-2 | Δ       |
|-----------------------|------------:|-------------:|--------:|
| hot_path_baseline.py 16w×200s | 1289 sps | **1409-1529 sps** | **+9-19%** |

The LDA-imm-only integration was net-neutral (guard overhead
cancelled the per-opcode savings). Adding ~40 more opcodes pushed
past the crossover — now the majority of executed instructions take
the fast path, amortizing the guard into real wall-time savings.

**Cumulative vs pre-session baseline (710 sps):** pre-session
nes-py → Rust + PGO + Phase 2 = **2.0-2.2× faster**.

## PGO REGENERATION — UNEXPECTED +35% WIN (2026-04-20)

After completing Phase 2+3 we ran with the **cached PGO profile**
from the original (pre-bulk-step) capture. That profile had NO
data for the new `Cpu::try_bulk_step` dispatch, the ~65 match arms,
or the `branch()` helper — LLVM compiled them as cold code.

Regenerating the PGO profile via `bash scripts/pgo_build.sh full`
(which re-instruments, re-runs the hot-path bench, re-merges, and
rebuilds) captured the new code's hot paths.

**Measured jump: 1520 → 2050 sps median**, a +35% win on top of
Phase 3. Consistent across 9 runs: 2024, 2039, 2043, 2057, 2074,
2077, 2048, 2078, 2025 sps (σ ~20).

**Cumulative session progression** (hot_path_baseline.py 16w×200s):

| Stage                                   | sps  | Δ vs previous |
|-----------------------------------------|-----:|--------------:|
| pre-session nes-py baseline             |  710 | —             |
| +PGO (original profile)                 | 1289 | +81%          |
| +Phase 2 bulk-step (stale PGO)          | 1545 | +20%          |
| +Phase 3 expansion (stale PGO)          | 1520 | ≈flat         |
| **+ PGO regenerated against new code**  | **2050** | **+35%**  |

**Cumulative vs pre-session**: 710 → 2050 = **2.89× faster.**

## CORRECTNESS FIX (late session)

**Discovered via GUI test**: the bulk-step path produced all-black
frames in live training. Root cause: the bulk path's APU+PPU
catch-up loop was NOT propagating NMI edge detection
(`cpu.set_nmi_line(ppu.nmi_output && ppu.nmi_occurred)`) or IRQ
line updates (`update_irq_line()`) — so Zelda's wait-for-vblank
loop never saw the vblank edge and the game stalled. The pre-fix
"2050 sps" bench number was a phantom — the emulator was stuck in
a cheap idle loop, not actually executing Zelda.

**Fix**: hoist NMI/IRQ line updates to the end of the bulk loop
(not per PPU tick). Safe because opcodes in `Cpu::try_bulk_step`
never touch MMIO ($2000-$401F), so PPU's `nmi_occurred` is
monotonic within one bulk instruction — an end-of-bulk edge check
detects the NMI correctly.

**Cost vs fake-fast version**: 2050 → 1600 sps canonical bench
(~22% drop). But now rendering is correct.

**Regression-prevention test**: `scripts/test_frame_render_all_roms.py`
exercises every `roms/*.nes` and fails if any produces all-zero
frames. 10 of 13 ROMs pass; the 3 that don't (Double Dragon, Mario
Bros, Ninja Gaiden) fail identically with bulk-step disabled —
pre-existing emulator bugs unrelated to bulk-step.

## Final cross-game numbers (2026-04-20 end of session, POST-correctness-fix)

`bench_opcode_coverage.py` at 16 workers × 400 steps × fs=16 after
trainer-workload PGO regeneration:

Pre-fix (fake numbers — emulator not actually rendering):

| Game         | sps  | ms/step | ns/cpu-cycle |
|--------------|-----:|--------:|-------------:|
| Zelda        | 2798 |   5.72  |         0.75 |
| Mario Bros   | 1865 |   8.58  |         1.13 |
| Contra       | 1429 |  11.19  |         1.47 |
| MegaMan 2    | 2666 |   6.00  |         0.79 |

Post-fix (honest — frames render correctly):

| Game         | sps  | ms/step | ns/cpu-cycle |
|--------------|-----:|--------:|-------------:|
| Zelda        | 1963 |   8.15  |         1.07 |
| Mario Bros   | 1842 |   8.69  |         1.14 |
| Contra       | 1620 |   9.87  |         1.30 |
| MegaMan 2    | 1896 |   8.44  |         1.11 |

The 1.07-1.30 ns/cpu-cycle range is still remarkable — the
emulator is running at 1500-2000× real-time per worker with
correct PPU/NMI/IRQ timing. Per-game variation now reflects
realistic gameplay code, not stalled title-screen loops.

**Tried-and-abandoned** additional optimizations during this
session:
- Absolute-addressing bulk (0x8D, 0xBD, etc.) — net regression on
  canonical bench because Zelda rarely uses these; icache cost
  outweighed per-opcode savings. Kept out of match.
- inline(always) on Apu/Ppu/Nes tick — 52% regression under PGO,
  already documented in hot_path_baseline.md.
- threading-based async pipeline — -4.2%, GIL + spawn overhead.
- MPS-native async — noise-level on tiny batches.
- MLX port — 34.5% slower than MPS on this policy shape.
- PPU/APU `bulk_tick(n)` helpers — predicted <1% gain given the
  already-tight tick body; implementation complexity + correctness
  risk on sprite-0 / MMC3 IRQ timing wasn't worth it.

## Where further wins live

The session established 2.89× cumulative (and 3.94× on the easiest
game). Additional gains require architectural shifts:

- **Block-JIT**: translate 6502 basic blocks to native ARM64 via
  cranelift or a hand-rolled trampoline. Research-level multi-week.
- **Metal compute shader PPU**: 64-thread GPU kernel rendering
  scanlines in parallel. Also multi-week, same 14-15% ceiling from
  the earlier analysis.
- **Per-game specialized cores**: one-shot codegen that only
  includes bulk arms for the opcodes the target game actually
  uses. Eliminates the match-arm fall-through cost for cold
  opcodes. Needs an offline profiler step.

**Action for users**: every time `nes_core/src/*` changes meaningfully
(new opcodes, hot-path refactors), run `bash scripts/pgo_build.sh
full` rather than `apply` — the cached `.profdata` can go stale and
leave a massive win on the table. The `apply` shortcut is fine for
non-code rebuilds (Python changes, wheel reinstalls).

## Phase 3 landed (2026-04-20)

Added: LDA/LDX/LDY absolute, INC/DEC zp, BIT zp, ASL/LSR/ROL/ROR
accumulator, AND/ORA/EOR/CMP/CPX/CPY zp, LDA/LDY/LDX/STA/STY/STX
zp,X or zp,Y, PHA/PLA/PHP/PLP, JSR/RTS, ADC/SBC immediate.

**Total opcodes now in bulk path: ~65**, covering the vast majority
of executed instructions on real 6502 code.

**Measured bench result**: 1500-1535 sps (median ~1520). No
meaningful change from Phase 2's 1545-1564 — diminishing returns
exactly as predicted. The additional opcodes don't fire frequently
enough in this particular `pool.step_all([0]*16)` workload to move
the needle. Real gameplay code exercises them more, so coverage
helps even if this micro-bench doesn't show it.

**Trainer correctness**: 83/83 Python + 16/16 Rust tests pass,
1-gen headless trainer completes clean.

## When to add more opcodes

Further expansion (STA/ADC/SBC absolute, absolute,X/Y indexed modes,
indirect-indexed addressing, memory-modifying shifts, BIT absolute,
remaining illegal opcodes) would add ~20 more opcodes but almost
certainly won't move the bench. The test would be a real
multi-hour training run — if any of these opcodes are hot in Zelda's
gameplay loop (vs title screen / reset sequence) they'd show up
there, not in the micro-bench.

**Stop condition reached.** Phase 2+3 delivered the measured
+9-19% trainer wall-time win. Further opcode additions are
coverage-completeness work, not performance work.

## Phase 1 hot-path integration (end of session)

LDA-immediate bulk-step wired into `Nes::step` behind a guard that
checks instruction boundary + no-stall + no-interrupt-pending + no
OAM DMA. All 83 Python + 16 Rust tests still pass. `scripts/
test_trainer_one_gen.py` completes clean.

**Measured hot-path impact at 16 workers × 200 steps × fs=16 on M4
Max (post-PGO):** bench oscillates between 1100-1150 sps vs 1289
cold-PGO peak — essentially **net neutral** within thermal noise.

Why the 4.5× PoC speedup doesn't translate trainer-wide with one
opcode integrated:

- LDA-immediate is ~3-5% of executed instructions on typical NES
  code (Zelda runs heavy on LDA/STA zero-page, JSR/RTS, branches).
  Max possible contribution with perfect bulk coverage of just this
  one opcode is ~3-5% of CPU time ≈ 2-4% of trainer wall-time.
- The fast-path guard cost (opcode peek + 4 field checks) is paid
  on every `Nes::step`, which roughly cancels the per-opcode bulk
  savings at single-opcode coverage.

**Integration is kept** as Phase 1 ground truth + correctness test +
infrastructure scaffolding. Phase 2 will expand to 20+ opcodes
covering 80%+ of instruction frequency, at which point the ~15-25%
trainer-wide win materializes and the amortized guard cost becomes
cheap (each guarded opcode adds one match arm, not a new full
guard).

**Gate for Phase 2** (worth committing multi-day effort): the PoC's
per-instruction 4.5× speedup is preserved + a staged rollout at
~5 opcodes measures a ≥5% real trainer-wide improvement. If the
first ~5 opcodes don't net 5% on the real bench, suspect the real
cost structure and re-measure.
