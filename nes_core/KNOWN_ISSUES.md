# Known issues

History-of-bugs format: open issues at top, closed issues kept below
with the commit hash + date so the next investigator can grep instead
of re-discovering.

**No open issues.** All 794 tested ROMs in the local library boot.
The single ROM previously failing `Cartridge::load` (`Yoshi (USA).nes`)
is a truncated dump, not an emulator bug. See the CLOSED entry below.

## CLOSED

### Crash 'n' the Boys (MMC3) cold-boot trap — IRQ entry I-flag missing

Commit `dbfad69` (2026-04-27). Per 6502 spec, both BRK and IRQ entry
must set the I flag AFTER pushing P, masking further interrupts during
the handler. `brk_push_status` did. `interrupt_push_status` did not.

Without I-set on IRQ entry, mappers that hold the IRQ line asserted
(MMC3 scanline IRQ, MMC5, FME-7) re-entered the handler on every
subsequent instruction-boundary poll. Each re-entry pushed 3 more
bytes onto the stack. Stack underflow → garbage RTI → PC parked in
the IRQ vector region ($FFE0-$FFFF) at $FFF7.

Cold-boot trap fired in 19 frames, no input. Found by walking
`step_one_instruction()` from cold reset until PC entered $FFE0+ and
stayed there for 100+ instructions; the disassembly showed the IRQ
handler being re-entered every 601 cycles without ever RTIing.

Fix: one-line `self.flags.i = true` added to `interrupt_push_status`.
146 parity tests still pass; 8,991-instruction nestest CPU validation
still byte-exact vs the Nintendulator golden trace.

### Zelda cave-stuck — cycle-locked advance_one_frame

Commit `72b2e9c` (2026-04-24), follow-ups `1fe2a7d` + `28ad5e2`.
Symptom: Link entered the sword cave but couldn't pick up the sword
or exit through the south doorway. Diagnosed via byte-by-byte CPU
RAM diff vs Mesen as a per-frame cycle-count drift: nes_core's
"step CPU instructions until frame_written" loop ran ±5 cycles per
frame depending on which instruction crossed the boundary, while
LaiNES/nes-py run a strict 29,781-cycle frame. The accumulated drift
moved the cave-entry transition to the wrong PPU scanline, breaking
the nametable update.

Fix: `advance_one_frame` now runs an exact 29,781 CPU cycles per
frame, matching the structural step boundary of the reference
emulators. User confirmed sword pickup + cave exit work in the GUI;
554 parity tests pass.

### MMC1 RMW consecutive-write filter — Bill & Ted's now boots

Commit `59458f4` (2026-04-23). NESdev wiki: MMC1 ignores the second of
two writes that hit the shift register on consecutive CPU cycles
(RMW dummy + real write of `INC/DEC/ASL/SLO/...`). nes_core had no
filter, so `INC $FFFF` in Bill & Ted's reset code corrupted the
shift-register protocol → bank state wrong → indirect JMP to garbage
→ BRK → IRQ trap loop at $FFF1.

Fix: added `Mapper::set_cpu_cycle(u64)` trait hook (default no-op)
called from `Nes::tick` once per CPU cycle. Mapper1 stores
`last_register_write_cycle` and silently drops a register write whose
cycle is `last + 1`.

See `memory/project_mmc1_rmw_consecutive_write_fix.md`.

### NES 2.0 byte-10 PRG-RAM nibble — `roms/zelda.nes` (and other NES 2.0 carts) now boot

Commit `6185b2e` (2026-04-23). NES 2.0 header byte 10 packs two shift
counts: low nibble = volatile PRG-RAM, high nibble = battery PRG-RAM.
The parser was reading only the low nibble. For a Zelda dump with
`flags10 = 0x70`, that produced 0 bytes of PRG-RAM. The game's
title-screen scratch writes to $6000-$60FF then reads garbage
(open bus), eventually indirect-JMPing into non-code → BRK → trap.

Fix: sum both nibbles per spec.

See `memory/project_nes20_prg_ram_nibble_bug.md`.

### Yoshi (USA) cartridge fails to load

`Yoshi (USA).nes` is a truncated dump (the only ROM in the local
library that fails `Cartridge::load` with `UnexpectedEof`). Not an
emulator bug. The file itself is incomplete. Replace from a clean
source. Tracked as `ours_panic` (1) in `parity_sweep.json`.

### MMC1 CHR-RAM out-of-bounds on 256 KB PRG ROMs (CLOSED in split 03)

Original symptom: `index out of bounds: the len is 8192 but the
index is 8192` panic at `src/mapper/mapper1.rs:184` when booting
`MEGAMAN2.NES` (MMC1, 256 KB PRG, CHR-RAM). Fix: `% chr.len()`
modulo in CHR-bank-select. Mega Man 2-6 all boot.

### Mario Bros (NROM) all-zero frames (CLOSED in split 02)

PPU register init bug; rendering enabled by default. Fixed alongside
the broader PPU rewrite.

## Performance baseline (post split-04, pre-PGO — superseded)

The numbers below are the original RustedNES vendor's pre-skip-PPU
baseline. Current production numbers are in
`docs/proposals/archive/hot_path_baseline.md` and the README.

| frame_skip | nes_core | nes-py    | ratio |
|------------|---------:|----------:|------:|
| 1          | 612 fps  | 1196 fps  | 0.51× |
| 4          | 701 fps  | 1458 fps  | 0.48× |
| 8          | 700 fps  | 1477 fps  | 0.47× |

After the skip-PPU + PGO + ASM CPU work, nes_core leads `nes-py` at all
worker counts of 4+ and ties at 1 worker (see README's perf table).

### Working ROMs (confirmed at 697-1518 fps standalone Rust, split-01 era)

- `zelda.nes`        — mapper 1 — 729 fps, 51558/61440 nonzero, audio peak 0.208
- `CONTRA.NES`       — mapper 2 — 805 fps, 726/61440 nonzero, audio peak 0.208
- `Metroid (USA) 01` — mapper 1 — 697 fps, 12366/61440 nonzero, audio peak 0.374
- `CASTLEVA.NES`     — mapper 2 — 708 fps, 20430/61440 nonzero, audio peak 0.208
- `MEGAMAN2.NES`     — mapper 1 — 676 fps, 1889/61440 nonzero (after MMC1 fix)
- `MEGAMAN3/5/6`     — mapper 1 — 720-790 fps each (after MMC1 fix)
