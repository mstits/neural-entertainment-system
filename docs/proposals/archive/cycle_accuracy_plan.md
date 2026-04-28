# Cycle-accuracy parity plan — nes_core → nes-py/LaiNES byte-exact RAM

## Problem

`tests/parity/test_lockstep_baseline.py` — established 2026-04-23 — compares nes_core vs nes-py CPU RAM byte-by-byte after N cold-boot idle frames. Current gap:

| ROM | bytes diverged at 600 idle frames |
|-----|-----------------------------------|
| Mega Man | 11 |
| Legend of Zelda | 40 |
| Contra | 163 |
| Metroid | 187 |
| Super Mario Bros. | 198 |

All five games diverge from nes-py in the first 1-5 frames after cold-boot reset. Divergence stabilizes by ~frame 60 for most games and stays bounded — it does not grow unbounded. Games still play most of the time, but specific cycle-sensitive sequences fail (e.g. Zelda cave sword-pickup cutscene on Zelda overworld).

`tests/parity/lockstep.py` is the diagnostic tool: run both emulators with the same button sequence and it returns the first frame + specific addresses where RAM diverges.

## Root causes (ranked by suspected leverage)

Each row is one systematic class of cycle-accuracy gap. Fixing any one should drop RAM divergence on multiple games.

### 1. $2002 PPUSTATUS vblank race — HIGH leverage
When the CPU reads `$2002` at the exact cycle the PPU sets the vblank flag (scanline 241 cycle 1), real hardware suppresses the vblank flag read AND the NMI for that frame. `nes_core::Ppu::read_ppu_status` at `nes_core/src/ppu.rs:424` does not check scanline/cycle — it returns whatever the current `nmi_occurred` says and clears it.

Fix: in `read_ppu_status`, check `(self.scanline, self.scanline_cycle())`. If within the race window (scanline 241, cycles 0-2 approximately), handle the suppression as described on the NESdev wiki "PPU frame timing" page.

Evidence this is the bug: Zelda's cave cutscene depends on precise `$2002` polling during a wait loop. A one-cycle misalignment in the read could cause the "press A to pickup sword" branch to miss the event-fire flag.

### 2. NMI edge detection timing — HIGH leverage
`nes_core::Nes::tick` at `nes.rs:349-353` calls `ppu.tick` then `cpu.set_nmi_line` in the same inner loop. This means NMI edge detection happens between PPU ticks rather than at the end of a CPU cycle. Real hardware latches NMI on the rising edge of φ2 of the second half of each CPU cycle.

Fix: measure whether moving the `set_nmi_line` call to after all 3 PPU ticks of a CPU cycle (instead of after each) changes the RAM divergence count.

### 3. Per-opcode cycle counts — MEDIUM leverage
Every 6502 opcode has a documented cycle count, including:
- Page-cross penalty on `abs,X`, `abs,Y`, `ind,Y` (+1 cycle when effective address crosses a page)
- Branch-taken penalty (+1 cycle on taken branch, +1 more on page-cross)
- JSR/RTS/RTI exact cycle breakdown

Fix: cross-reference `nes_core/src/cpu.rs`'s opcode table against the canonical reference (Synertek hardware manual or the NESdev 6502 reference) and flag any mismatches. Nestest ROM's log would surface these immediately but isn't currently in the repo.

### 4. RESET 7-cycle pulse — LOW leverage (verified 2026-04-23)
Tested in this session: consuming 7 CPU cycles + 21 PPU ticks during `Nes::reset` has **zero measurable effect** on RAM divergence. Either nes-py/LaiNES also skips this, or the effect is masked by the frame-1 CPU init code that runs thousands of instructions. Skip this fix until higher-leverage items are done.

### 5. APU frame counter + DMC DMA stall — MEDIUM leverage
APU generates IRQs on a 240Hz frame counter. DMC DMA steals 4 CPU cycles on each sample fetch. If either is cycle-misaligned with nes-py, games that use APU IRQs (Battletoads, some Mega Man levels) will diverge heavily. Zelda does use DMC samples for Link's "heart" sound but not time-critically.

### 6. MMC1 consecutive-write detection — Zelda-specific
Zelda uses MMC1 (mapper 1). MMC1 has a 5-bit shift register: consecutive writes to `$8000-$FFFF` within 2 CPU cycles of each other are ignored for the second write (the RMW-instruction protection). If `nes_core/src/mapper/mmc1.rs` doesn't implement this, Zelda's bank-swap routines run with slightly-different bank alignment.

## Proposed workflow

For each fix:
1. Pick a root cause from the list above.
2. Apply the minimum fix to `nes_core/src/{cpu,ppu,apu,mapper}.rs`.
3. `make build && cp nes_core/target/release/libnes_core.dylib .venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so`.
4. Run `pytest tests/parity/test_lockstep_baseline.py -q`.
5. Each improvement fails the test with "improved! was X now Y"; edit the ceiling, commit with the fix.
6. Run `make parity` + `make test` + `cargo test --release` to confirm no regressions.

The baseline test's fail-on-improvement semantics force every fix to be measured and committed together. Over several sessions, the ceilings should descend monotonically to 0 (byte-exact parity) or settle at the minimum achievable given non-cycle-accurate constraints.

## Open question: can we actually reach 0?

nes-py/LaiNES aims for cycle-accuracy but is itself not byte-exact with real hardware for all edge cases. Final convergence number may be some small non-zero floor (<10 bytes/game) representing genuine non-determinism (RAM init, open-bus behavior on unmapped reads, DMC DMA jitter under mapper IRQ contention). That's acceptable for RL training and correct game play — the games just need to behave correctly, not match nes-py pixel-for-pixel forever.

Deliverable of this plan: all 5 baseline games at ≤5 bytes RAM divergence per 600 idle frames, and the Zelda cave sword-pickup cutscene completing correctly in the GUI.
