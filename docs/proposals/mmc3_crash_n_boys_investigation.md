# Crash 'n' the Boys (MMC3) cold-boot crash — partial investigation

**Status:** OPEN. Diagnosis below; fix not yet identified.

## Symptom

`Crash 'n' the Boys - Street Challenge (USA).nes` (mapper 4 / MMC3,
US PRG 32 KB CHR 8 KB) is the only ROM left in the playability
sweep's `crashed` bucket after the 2026-04-23 MMC1 RMW fix landed.
PC enters `$FFE0-$FFFF` within ~20 frames of cold boot and stays
there indefinitely. Zero-page RAM does NOT change.

## Confirmed root cause: MMC3 IRQ delivery

Empirical proof: temporarily forcing `Mapper4::irq_pending() -> false`
(no IRQs delivered) makes the game advance normally — PC reaches
`$F5FE` (game code), zero-page changes, Start press progresses past
title. So the crash IS our MMC3 IRQ behaviour.

But the obvious "IRQ fires too often" hypothesis isn't the bug:

- IRQ counter latch: `0xAF` (175 scanlines), set by the game during
  init. Fires roughly once per 175 scanlines = once every 0.73
  frames. NOT one-fire-per-scanline.
- Game IRQ handler at `$FB9B`: starts with `PHA TXA PHA TYA PHA STA
  $E000 STA $E001 ...`. `STA $E000` is the canonical MMC3 IRQ ack —
  acknowledges + disables — and `STA $E001` re-enables.
- `prg_write_byte` traces of the boot show `$E001` write happens
  three times during init, then `$E000` once, then `$E001` again.
  After that — NOTHING. No further `$E000` writes despite the IRQ
  firing and (per disassembly) the handler containing the ack.

So either the handler isn't running, or our `prg_write_byte` trace
isn't seeing all writes. ASM CPU's MMIO writes do go through
`prg_write_byte` (proven by the working MMC1 RMW fix), so the trace
SHOULD catch it.

## What's known to be working

- nestest: 8991 instructions byte-exact, including cycle counter.
  CPU is correct.
- MMC1 RMW filter via `Mapper::set_cpu_cycle` hook: doesn't affect
  MMC3 (default no-op impl).
- IRQ counter / latch / reload semantics in `Mapper4` look correct
  per NESdev wiki spec.
- IRQ acknowledgement via `$E000` write IS implemented (clears
  `irq_active` and `irq_enabled`).

## What's NOT yet checked

1. **Is the IRQ entry sequence completing?** Maybe the CPU pushes
   PC + P, fetches the vector, jumps to `$FFF7`, but the JMP
   indirect at `$FFF7` doesn't complete because `irq_line_low` is
   re-asserted before the JMP finishes. (Shouldn't happen — `I=1`
   should block — but worth verifying via per-cycle PC trace.)
2. **`I` flag state during the supposed handler.** If `I=1` (set
   by IRQ entry) is somehow being cleared between the JMP and the
   first PHA at `$FB9B`, that breaks the contract.
3. **IRQ counter ticking too fast.** `on_scanline_tick` is called
   from PPU. Is it called once per scanline, or could it be called
   multiple times due to PPU bulk-step?
4. **MMC3 Revision A vs B/C semantics.** NESdev notes a subtle
   difference in when the counter clocks. We may be implementing
   the wrong revision for this game.
5. **`irq_active` is set when counter reaches 0 AND irq_enabled.**
   If `irq_enabled` is somehow oscillating (e.g. some `$E000` write
   we don't see), behaviour gets weird.

## Diagnostic script for next session

```python
import nes_core
ROM = "roms/Crash 'n' the Boys - Street Challenge (USA).nes"
env = nes_core.NESEnvironment(rom_path=ROM, frame_skip=1)
env.reset()
# Sample PC every frame for 100 frames
for f in range(100):
    env.step(0)
    pc = env.cpu_state()[0]
    print(f"f={f}: PC=0x{pc:04X}")
# Result: ~93% of samples show PC=$FFF7 (the IRQ vector trampoline).
# Remaining 7% are early-boot setup PCs.
```

## Concrete next steps

1. **Per-cycle PC trace through one IRQ firing.** Add a hook in
   `Cpu::tick` to log PC every cycle when in the trap region. Confirm
   whether the JMP at `$FFF7` ever fully executes and reaches
   `$FB9B`.
2. **Verify `I` flag is set on IRQ entry.** Add a `eprintln!` in
   `Cpu::tick`'s IRQ-entry path showing `I` before/after.
3. **Compare IRQ counter ticks per frame.** Add a counter,
   `eprintln!` per-frame total. Should be 240 (one per visible
   scanline) — if it's higher, we're double-clocking somewhere.
4. **Audit `prg_write_byte` ASM-CPU path.** Verify MMC3 register
   writes from the ASM CPU are routed through the same path our
   `eprintln!` traces. If not, that explains why we don't see the
   handler's `$E000` write.

## Why this matters

Crash 'n' the Boys is the last `crashed` ROM (post Bill & Ted's fix).
Closing it would put the playability sweep at:

| bucket      | v3  | hypothetical post-fix |
|-------------|----:|----------------------:|
| advances    | 432 | 433                   |
| noisy       | 250 | 250                   |
| frozen      | 110 | 110                   |
| crashed     |   1 | 0                     |
| load_failed |   1 | 1                     |

A complete `crashed=0` is a clean signal — every supported ROM
either advances or is heuristic-noise (frozen / noisy).

## Cross-references

- `nes_core/src/mapper/mapper4.rs` — MMC3 implementation
- `nes_core/KNOWN_ISSUES.md` — symptom-level entry
- `playability_sweep_v3.json` — the bucket assignment data
- NESdev wiki: <https://www.nesdev.org/wiki/MMC3>
