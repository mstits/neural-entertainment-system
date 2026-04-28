# AArch64 Threaded-Code 6502 CPU — Design

Direction set by user 2026-04-20: abandon incremental Rust
optimizations on the bulk-step path in favor of a pure-AArch64-ASM
6502 core with threaded-code dispatch. Target: eliminate the Rust
dispatch overhead entirely, replace it with direct handler-chaining
through a per-opcode jump table, and pin all 6502 architectural
state in AArch64 registers across handler boundaries.

## Scope (locked)

- **AArch64 (Apple Silicon) only.** No portability hedging.
- **All 150 official 6502 opcodes** in ASM. Unofficial/illegal opcodes
  fall back to Rust unless nestest coverage shows training ROMs need
  them.
- **Inline memory access hot path** in ASM: direct RAM access, direct
  PRG ROM access for NROM (and any mapper whose mapping is static).
  MMIO and mapper-mediated reads/writes call out to Rust.
- **Register-pinned 6502 state** across handler chain. Documented at
  the top of the `.s` file.
- **Native AArch64 NZCV** used for 6502 N/Z/C/V where the mapping
  is clean (i.e. LDA/CMP/ADC/SBC/shifts). Pack/unpack to/from a P
  byte only at exit / at PHP/PLP.
- **Cycle counter pinned in a register**, not memory. Decrement toward
  target on each opcode's documented cycle count.

## Out of scope (explicit NOs)

- No PPU work beyond what the current bulk-step already does (the
  video path is fine; the CPU is the ceiling).
- No general Rust cleanup or "while we're here" refactors.
- No cross-platform considerations. x86_64 / Linux / Windows all not
  targets.
- No CNN / training pipeline changes.

## Register allocation (draft)

AArch64 has x0-x30, with x0-x7 for arg/return, x8 for indirect return,
x9-x15 caller-saved (scratch), x16-x17 IP (intra-procedure-call), x18
platform, x19-x28 callee-saved, x29 FP, x30 LR, sp/xzr.

For a long-running threaded-dispatch core we want the 6502 state in
*callee-saved* registers (x19-x28) so the state survives any Rust
call-outs we make without having to spill/restore.

**Proposed mapping:**

| Reg  | Contents                        | Notes                          |
|------|---------------------------------|--------------------------------|
| x19  | 6502 PC (u16 in low 16 bits)    | Incremented per opcode fetch   |
| w20  | 6502 A (u8 in low 8 bits)       |                                |
| w21  | 6502 X                          |                                |
| w22  | 6502 Y                          |                                |
| w23  | 6502 SP (u8, stack base $0100)  |                                |
| w24  | 6502 P packed (N,V,B,E,D,I,Z,C) | Only valid at entry/exit; inside the core, live flags are held in AArch64 NZCV + separate I/D/B/E bools in x24 high bits |
| x25  | cycles_remaining (signed i64)   | Decremented per opcode; exits when ≤ 0 |
| x26  | ptr to SystemBus (struct &mut)  | For MMIO call-outs + opcode fetch fast path |
| x27  | ptr to RAM (2 KB)               | Direct access to $0000-$07FF (mirrored 0-$1FFF) |
| x28  | ptr to PRG ROM base             | Valid only for NROM / static mapping; nullptr → fallback |

PC is kept in x19 as a full 64-bit so `add x19, x19, #1` is one
instruction — mask to u16 only when writing back or when indexing
memory.

NZCV mapping to 6502 flags:
- **6502 N** ↔ AArch64 N (same sign-bit semantics on byte results if
  we set flags on the byte move)
- **6502 Z** ↔ AArch64 Z (same)
- **6502 C** ↔ AArch64 C for ADC/SBC (CAVEAT: 6502 SBC-carry semantics
  are inverse of typical two's-complement borrow; we may need an
  explicit `eor` before/after)
- **6502 V** ↔ AArch64 V (same)

6502 I, D, B, E: kept as separate bits in the upper half of x24 (not
in native flags; toggled explicitly by SEI/CLI/SED/CLD/BRK).

## Threaded dispatch

Every opcode handler ends with the same 4-instruction "next" tail:

```asm
.macro NEXT cycles
    subs    x25, x25, #\cycles        // decrement cycle budget
    b.le    exit_cycles_done          // out of budget → return
    ldrb    w0, [x27, x19]            // fetch opcode (RAM only; slow
                                      //   path rewrites x0 for PRG)
    add     x19, x19, #1              // advance PC past opcode
    adr     x1, opcode_table
    ldr     x2, [x1, x0, lsl #3]      // 8-byte function pointer
    br      x2
.endm
```

The dispatch table is an array of 256 function-pointer-sized entries
(AArch64 pointers are 8 bytes). Each entry is the address of that
opcode's handler (or a fallback-to-Rust stub for unimplemented/illegal).

## Memory access hot path

Every read/write classifies the address in ASM before calling out:

```asm
// Fast-path read: x0 = address (u16)
read_byte:
    cmp     x0, #0x2000
    b.lt    .read_ram                 // $0000-$1FFF: RAM (mirrored)
    cmp     x0, #0x8000
    b.ge    .read_prg                 // $8000-$FFFF: PRG ROM
    b       .read_slow                // MMIO or PRG-RAM → Rust

.read_ram:
    and     x0, x0, #0x07FF           // mirror mask
    ldrb    w0, [x27, x0]
    ret

.read_prg:
    cbz     x28, .read_slow           // x28 null → non-NROM, call Rust
    and     x0, x0, #0x7FFF           // offset into 32 KB PRG
    ldrb    w0, [x28, x0]
    ret

.read_slow:
    // Save state to Cpu struct, call Rust bus_read, restore, return
    // (see macro spill_cpu / reload_cpu in the .s file)
```

Mapper dispatch cost before: one `MapperEnum::prg_read_byte` call per
CPU read, which resolves to a match on the enum variant, then the
concrete impl. After: one compare + one load for NROM (the Zelda case
we profile against). MMC1/UxROM etc. fall back to Rust.

## Exit protocol

`nes_cpu_run_block(cpu: *mut Cpu, bus: *mut SystemBus, cycles_target: i64) -> u32`

Exit code (u32 return):
- 0: cycle budget reached (normal pool-step boundary)
- 1: interrupt fetch pending (NMI or unmasked IRQ; let Rust handle)
- 2: MMIO read needed (address in `cpu.mmio_pending_addr`)
- 3: MMIO write needed (addr + value in cpu fields)
- 4: Unimplemented opcode encountered (opcode in `cpu.unimpl_opcode`)

For MMIO, the Rust side handles the read/write via the existing
`SystemBus::read_byte` / `write_byte`, updates the corresponding
6502 register, then re-enters `nes_cpu_run_block` for the next
instruction.

For interrupts, Rust handles the push-state + jump-to-vector via the
existing `Cpu::poll_interrupts` path, then re-enters.

## Differential testing harness

Every batch of opcodes ported gets a diff test before I move on.
Two layers:

1. **Per-opcode synthetic**: construct a minimal Cpu + RAM with the
   opcode + operand, run one instruction through both the Rust
   reference and the ASM path, diff the resulting CpuState. Ships
   in `nes_core/tests/asm_diff_per_opcode.rs`.

2. **nestest.nes cycle-accurate**: load the standard nestest ROM,
   run through both paths with identical input, compare the
   per-cycle output log (PC + A + X + Y + P + SP + cycles_total)
   at every instruction boundary. First divergence = regression.
   Gold-standard for correctness. ROM fetched lazily into
   `tests/data/nestest.nes` (instructions in test module — not
   auto-downloaded in CI).

Rule: a diff failure halts porting until fixed. No accumulating
broken opcodes.

## Feature flag

`nes_core/Cargo.toml`:
```toml
[features]
asm_cpu = []   # enables AArch64 assembly 6502 core
```

Gate the ASM inclusion and the ASM-routing in `Cpu::tick` /
`Nes::step` on `#[cfg(all(feature = "asm_cpu", target_arch = "aarch64"))]`.
On any other platform or with the flag off, behavior is the current
Rust core — tests MUST pass in both configurations.

## Phased rollout

**Phase 0 (this session)**: infrastructure.
- Feature flag wired.
- `global_asm!` including `cpu_asm.s` on aarch64+flag.
- FFI stub: `extern "C" fn nes_cpu_run_block(...)` declared.
- Minimal `cpu_asm.s` with dispatch skeleton + 5 opcodes (LDA #imm,
  STA zp, JMP abs, BNE rel, NOP).
- Per-opcode diff tests for all 5.
- Smoke test: can execute a short 6502 program end-to-end via ASM
  path and get byte-identical state to Rust path.

**Phase 1**: core arithmetic + load/store.
- 20 more opcodes: LDA/LDX/LDY in imm/zp/zp,X/abs. STA/STX/STY same.
  ADC/SBC/AND/ORA/EOR/CMP imm + zp. INC/DEC zp. Branches.
- After porting 20, re-bench. Report aggregate FPS.

**Phase 2**: full official opcode set.
- Stack ops (PHA/PLA/PHP/PLP, JSR/RTS).
- Indexed-indirect + indirect-indexed addressing.
- Shifts/rotates memory-destination.
- BRK.
- Bench after batch.

**Phase 3**: nestest harness.
- Fetch nestest.nes.
- Run ASM core through it; diff against the official nestest.log.
- Fix any divergences.

**Phase 4**: profile the new top-10 hot paths. If CPU dispatch is
no longer #1, report what is + propose next target.

## Measurement schedule

- After every batch of ~20 opcodes ported: `bench_hot_path.py` at
  16 workers × 200 steps × fs=16. Report total sps + delta vs prior
  batch.
- If sps goes flat or down for two batches in a row: stop, profile,
  investigate.

Numbers to beat (current session's honest baselines on M4 Max
post-PGO):
- Canonical bench: 1600 sps
- Zelda per-game: 1963 sps
- Pre-session baseline: 710 sps

Target (user's stated ambition — "blood from a stone"): sustained
improvement over 1600 sps across every opcode batch, with
correctness preserved.
