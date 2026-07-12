// ============================================================================
// AArch64 threaded-code 6502 CPU core for nes_core
// ============================================================================
//
// Design: docs/proposals/aarch64_cpu_asm.md
//
// Entry point:
//   u32 nes_cpu_run_block(
//       cpu: *mut AsmCpuState,        // x0 — 6502 arch state + mmio shim
//       ram: *mut u8,                 // x1 — internal 2 KB RAM base
//       prg: *const u8,               // x2 — PRG-ROM base (nullptr for non-NROM)
//       cycles_target: i64,           // x3 — run until ≤ 0
//   );
//
// Return (in w0):
//   0 — cycles budget reached (normal exit)
//   1 — unimplemented opcode (opcode in AsmCpuState.unimpl_opcode)
//   2 — mmio or mapper read required (addr in AsmCpuState.mmio_pending_addr)
//   3 — mmio or mapper write required (addr+value in AsmCpuState.mmio_*)
//
// Register allocation (pinned across the dispatch loop):
//   x19 — 6502 PC (u16 in low 16 bits)
//   w20 — 6502 A (u8)
//   w21 — 6502 X (u8)
//   w22 — 6502 Y (u8)
//   w23 — 6502 SP (u8; stack base is $0100)
//   x24 — packed P flags + I/D/B/E bits (live flag state kept in
//         native NZCV where clean; x24 holds only I/D/B/E + the
//         last-written C for ops that don't use AArch64 carry)
//           bit 0 = 6502 C (as explicit bool, in case PHP/PLP needs it)
//           bit 2 = I (interrupt inhibit)
//           bit 3 = D (decimal — no-op on NES)
//           bit 4 = B (break)
//           bit 5 = E (expansion; always 1 when pushed)
//         N, Z, V live in native NZCV; repacked on exit.
//   x25 — cycles_remaining (i64, decremented per-opcode; ≤0 ⇒ exit)
//   x26 — ptr to AsmCpuState (for spills + MMIO return-path fields)
//   x27 — ptr to RAM (2 KB base; mirror mask 0x07FF applied on index)
//   x28 — ptr to PRG-ROM base (nullptr when non-NROM / non-static)
//
// Caller-saved scratch: x0-x17.
//
// Exit tail protocol: each handler ends with the `NEXT` macro, which
// fetches the next opcode and jumps to its handler via the per-opcode
// dispatch table.
// ============================================================================

.section __TEXT,__text
.globl  _nes_cpu_run_block
.p2align 2

// ----- P-flag bit positions (match pack_p_flags in Rust) ----------------
.equ P_BIT_C, 0
.equ P_BIT_Z, 1
.equ P_BIT_I, 2
.equ P_BIT_D, 3
.equ P_BIT_B, 4
.equ P_BIT_E, 5
.equ P_BIT_V, 6
.equ P_BIT_N, 7

// ----- AsmCpuState struct offsets (keep in sync with cpu_asm.rs) -------
.equ CPU_PC,           0   // u16
.equ CPU_A,            2   // u8
.equ CPU_X,            3   // u8
.equ CPU_Y,            4   // u8
.equ CPU_SP,           5   // u8
.equ CPU_P,            6   // u8 (packed on entry/exit)
.equ CPU_CYCLES,       8   // u64 (absolute cycle counter)
.equ CPU_UNIMPL_OP,   16   // u8 — filled on exit-code 1
.equ CPU_MMIO_ADDR,   18   // u16
.equ CPU_MMIO_VALUE,  20   // u8
.equ CPU_BUS_PTR,     24   // u64 — raw SystemBus ptr for callbacks
.equ CPU_INITIAL_BUDGET, 32 // i64 — saved on entry (x3 is caller-saved)
.equ CPU_CYCLES_TICKED,  40 // u32 — running count of CPU cycles the MMIO
                            //       callback has already ticked PPU/APU for
.equ CPU_PENDING_EXIT,   44 // u8  — set by callback on NMI/IRQ rising edge;
                            //       NEXT polls + exits to Lexit_cycles so
                            //       the Rust slow path services the IRQ at
                            //       the next instruction boundary

// ----- Exit codes ------------------------------------------------------
.equ EXIT_CYCLES,      0
.equ EXIT_UNIMPL,      1
.equ EXIT_MMIO_READ,   2
.equ EXIT_MMIO_WRITE,  3

// ============================================================================
// Dispatch-tail macro — every handler ends with this (variable cycle cost).
// NZCV must be in the desired 6502-mapped state BEFORE the macro runs.
// ============================================================================

.macro NEXT cycles
    subs    x25, x25, #\cycles
    b.le    Lexit_cycles
    // Interrupt-pending check: MMIO callback sets pending_exit=1 when
    // an NMI rising edge or IRQ fires inside the batch. Bail to the
    // cycles-exit path so Rust's poll_interrupts services it at the
    // next instruction boundary (matches real 6502 NMI/IRQ delivery).
    ldrb    w17, [x26, #CPU_PENDING_EXIT]
    cbnz    w17, Lexit_cycles
    // Fetch next opcode. PC is in low 16 bits of x19; opcode space is
    // PRG ROM almost always; check RAM first for test harnesses.
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    1f                          // RAM-space code
    // PRG ROM: require x28 non-null (NROM)
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrb    w0, [x28, x0]
    b       2f
1:  and     x0, x0, #0x07FF
    ldrb    w0, [x27, x0]
2:  add     x19, x19, #1                // PC advances past opcode
    and     x19, x19, #0xFFFF
    // Global symbol reference — Mach-O only emits PAGE/PAGEOFF
    // relocations on exported symbols, not L-local aliases.
    adrp    x1, _nes_asm_opcode_table@PAGE
    add     x1, x1, _nes_asm_opcode_table@PAGEOFF
    ldr     x2, [x1, x0, lsl #3]
    br      x2
.endm

// ============================================================================
// FETCH_PC_BYTE \dest — load the byte at the current PC into \dest
// (zero-extended to 32 bits) and advance PC by 1. Follows the same
// RAM-or-PRG classification as NEXT. `\@` scopes local labels to the
// macro invocation so multiple uses in one handler don't collide.
// ============================================================================

.macro FETCH_PC_BYTE dest
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    100f
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrb    \dest, [x28, x0]
    b       101f
100: and    x0, x0, #0x07FF
    ldrb    \dest, [x27, x0]
101: add    x19, x19, #1
    and     x19, x19, #0xFFFF
.endm

// ============================================================================
// SET_NZ \val — clear and rewrite N (bit 7) + Z (bit 1) bits in w24
// from an 8-bit result value in \val. Trashes w1, w2. Kept as a
// macro rather than a helper function so we avoid the bl/ret cost in
// every load/ALU/transfer handler.
// ============================================================================

.macro SET_NZ val
    // 2026-04-26 fix: use w16/w17 as scratch (caller-saved, NOT used
    // by any opcode handler for live values). Original used w1/w2
    // which CMP handlers had ALREADY loaded memory + sub-result into,
    // so SET_NZ silently corrupted the subsequent `cmp w3, w1` carry
    // computation. Manifested as Mario falling through floor — CMP
    // $9504,Y at SMB PC=$94EE produced wrong carry, BCS took wrong
    // branch, ground-tile-buffer write at $94F7 went to wrong target.
    mov     w16, #0x82                  // N|Z mask
    bic     w24, w24, w16
    and     w17, \val, #0xFF
    cmp     w17, wzr
    cset    w16, eq
    orr     w24, w24, w16, lsl #P_BIT_Z
    tst     w17, #0x80
    cset    w16, ne
    orr     w24, w24, w16, lsl #P_BIT_N
.endm

// ============================================================================
// ADC_CORE \mop — A = A + mop + C, update N/Z/C/V. Used by both
// ADC (mop = memory byte) and SBC (mop = memory byte XOR 0xFF, which
// gives A + ~M + C = A - M - (1-C) per 6502 SBC spec).
//
// Reads w20 (A) and w24 (P byte). Writes w20 + N/Z/V/C bits of w24.
// Clobbers w2, w3, w4, w5, w6, w10, w11. The V temp lives on w11 (NOT
// w7) so the indexed ADC/SBC callers' page-cross flag in w7 survives the
// macro — otherwise abs,X / abs,Y / (zp),Y charged the extra cycle on
// signed overflow instead of a real page cross.
// ============================================================================
.macro ADC_CORE mop
    and     w2, w20, #0xFF              // A (clean 8-bit)
    ubfx    w3, w24, #P_BIT_C, #1       // old C (0/1)
    add     w4, w2, \mop
    add     w4, w4, w3                  // A + M + C (9-bit result)
    lsr     w5, w4, #8
    and     w5, w5, #1                  // new C = bit 8
    eor     w6, w2, \mop                // A ^ M
    eor     w11, w2, w4                 // A ^ result  (V temp on w11, NOT w7)
    bic     w11, w11, w6                // (A ^ result) & ~(A ^ M)
    tst     w11, #0x80
    cset    w11, ne                     // V = 1 if bit 7 set
    and     w20, w4, #0xFF              // A = result
    mov     w10, #0xC3                  // N|V|Z|C mask
    bic     w24, w24, w10
    tst     w20, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cmp     w20, #0
    cset    w10, eq
    orr     w24, w24, w10, lsl #P_BIT_Z
    orr     w24, w24, w11, lsl #P_BIT_V
    orr     w24, w24, w5, lsl #P_BIT_C
.endm

// ============================================================================
// Entry point
// ============================================================================
_nes_cpu_run_block:
    // Save callee-saved registers we'll clobber (x19-x28).
    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     x29, x30, [sp, #80]
    mov     x29, sp

    // Load state from AsmCpuState.
    mov     x26, x0                     // x26 = cpu state ptr
    mov     x27, x1                     // x27 = ram
    mov     x28, x2                     // x28 = prg (may be 0)
    mov     x25, x3                     // x25 = cycles budget (live)
    str     x3, [x26, #CPU_INITIAL_BUDGET] // save so exit path can compute
                                           // cycles_consumed after bl's have
                                           // clobbered x3

    ldrh    w19, [x26, #CPU_PC]
    ldrb    w20, [x26, #CPU_A]
    ldrb    w21, [x26, #CPU_X]
    ldrb    w22, [x26, #CPU_Y]
    ldrb    w23, [x26, #CPU_SP]
    ldrb    w24, [x26, #CPU_P]
    // Unpack NZCV from P into native flags.
    bl      Lunpack_p_to_nzcv

    // Kick the dispatch loop by fetching the first opcode.
    NEXT    0

// ============================================================================
// Exit paths — repack native NZCV into P, store state, return code.
// ============================================================================
Lexit_cycles:
    mov     w0, #EXIT_CYCLES
    b       Lexit_common

Lexit_unimpl_fetch:
    // Fetch failed — PRG is null AND PC not in RAM. Treat as
    // unimplemented-opcode fetch; Rust falls back to per-cycle core.
    mov     w0, #EXIT_UNIMPL
    strb    wzr, [x26, #CPU_UNIMPL_OP]
    b       Lexit_common

Lexit_unimpl_opcode:
    // w10 = unimplemented opcode byte (set by dispatch table fallback)
    strb    w10, [x26, #CPU_UNIMPL_OP]
    mov     w0, #EXIT_UNIMPL
    // Rewind PC past the fetched opcode byte so the Rust fallback
    // re-executes the full instruction. NEXT advanced PC by 1 during
    // the opcode fetch before dispatch to Lunimpl.
    sub     x19, x19, #1
    and     x19, x19, #0xFFFF
    b       Lexit_common

Lexit_mmio_read:
    // w10 = addr, caller will fill the value + resume at next opcode
    strh    w10, [x26, #CPU_MMIO_ADDR]
    mov     w0, #EXIT_MMIO_READ
    b       Lexit_common

Lexit_mmio_write:
    // w10 = addr, w11 = value
    strh    w10, [x26, #CPU_MMIO_ADDR]
    strb    w11, [x26, #CPU_MMIO_VALUE]
    mov     w0, #EXIT_MMIO_WRITE
    b       Lexit_common

Lexit_common:
    // Repack native flags into the 6502 P byte.
    bl      Lpack_nzcv_to_p
    strh    w19, [x26, #CPU_PC]
    strb    w20, [x26, #CPU_A]
    strb    w21, [x26, #CPU_X]
    strb    w22, [x26, #CPU_Y]
    strb    w23, [x26, #CPU_SP]
    strb    w24, [x26, #CPU_P]

    // Convert cycles-remaining back into cycles-consumed for the
    // caller's accounting: cycles_consumed = initial_target - x25.
    // The caller tracks initial_target itself; we just expose x25.
    // (Actually cycles_total on the Cpu struct can be updated by
    // caller via returned delta; see cpu_asm.rs.)
    ldr     x1, [x26, #CPU_CYCLES]
    ldr     x3, [x26, #CPU_INITIAL_BUDGET] // reload — x3 clobbered by bl's
    sub     x2, x3, x25                    // cycles consumed
    add     x1, x1, x2
    str     x1, [x26, #CPU_CYCLES]

    ldp     x29, x30, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96
    ret

// ============================================================================
// Flag packing helpers.
// ============================================================================
//
// Lunpack_p_to_nzcv: reads w24 (packed P byte) and sets native NZCV +
// leaves I/D/B/E bits in x24 high byte at positions I=bit10, D=bit11,
// B=bit12, E=bit13 (arbitrary layout — we only need to pack back out
// correctly). C also kept at bit 0 for PHP/PLP consistency.
//
// For the Phase 0 skeleton we keep this simple and just work with the
// packed P byte in w24 directly. Performance polish comes after the
// architecture is verified.
Lunpack_p_to_nzcv:
    // w24 already has the packed P byte. We use helper bit extracts
    // per-instruction rather than maintaining NZCV across the
    // dispatch loop. This is SLOWER than the designed-in native
    // NZCV tracking but gets the skeleton up quickly for diff
    // testing — Phase 1 replaces with proper NZCV pinning.
    ret

Lpack_nzcv_to_p:
    // w24 already holds the packed P byte in the Phase 0 skeleton.
    ret

// ============================================================================
// Opcode handlers.
//
// Phase 0: only 5 handlers implemented — LDA-imm, STA-zp, JMP-abs,
// BNE, NOP. All other opcodes land in Lunimpl which exits with
// EXIT_UNIMPL so the Rust caller falls back to the per-cycle core.
// ============================================================================

// ----- NOP (0xEA, 2 cycles) -----

// ============================================================================
// HOT opcode handlers — frequency-ordered for I-cache locality.
// Order derived from empirical NES gameplay opcode histograms.
// ============================================================================

// ----- NOP (0xEA, 2 cycles) -----
_op_nop:
    NEXT    2

// ----- LDA #imm (0xA9, 2 cycles) -----

_op_lda_imm:
    // Fetch operand byte at PC
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    1f
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrb    w20, [x28, x0]
    b       2f
1:  and     x0, x0, #0x07FF
    ldrb    w20, [x27, x0]
2:  add     x19, x19, #1
    and     x19, x19, #0xFFFF
    // Update N + Z bits in w24 (P) from w20 (A).
    // Clear N (bit 7) and Z (bit 1) in w24. 0x82 isn't a valid
    // BIC logical-immediate on AArch64 — load mask into a scratch
    // reg first.
    mov     w2, #0x82
    bic     w24, w24, w2
    cmp     w20, wzr
    cset    w1, eq
    orr     w24, w24, w1, lsl #P_BIT_Z
    tst     w20, #0x80
    cset    w1, ne
    orr     w24, w24, w1, lsl #P_BIT_N
    NEXT    2

// ----- STA zp (0x85, 3 cycles) -----

_op_lda_zp:
    FETCH_PC_BYTE w0
    ldrb    w20, [x27, x0]      // zero-page always RAM
    SET_NZ  w20
    NEXT    3

// ----- LDX zp (0xA6, 3 cycles) -----

_op_lda_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w20, [x27, x0]
    SET_NZ  w20
    NEXT    4

// ----- STA $nn,X (0x95, 4 cycles) -----

_op_lda_abs:
    FETCH_PC_BYTE w2                    // low addr byte (FETCH clobbers w0/x0)
    FETCH_PC_BYTE w3                    // high addr byte
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Llda_abs_ram
    cmp     w0, #0x8000
    b.lo    Llda_abs_mmio               // $2000-$7FFF → MMIO callback
    cbz     x28, Llda_abs_mmio          // non-NROM PRG → MMIO callback
    and     w0, w0, #0x7FFF
    ldrb    w20, [x28, x0]
    b       Llda_abs_done
Llda_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w20, [x27, x0]
Llda_abs_done:
    SET_NZ  w20
    NEXT    4
Llda_abs_mmio:
    mov     w9, w0
    // Read at the same intra-instruction position as the Rust core:
    // `cycle_zero_early_commit` runs the MMIO read during the CPU's
    // dispatch tick, which `Nes::tick` executes AFTER that cycle's
    // APU tick + 3 PPU dots. Pre-charge 1 cycle so Lmmio_read's
    // cumulative tick includes the dispatch cycle — without it the
    // ASM read lands 3 PPU dots early and a $2002 poll straddling a
    // flag transition (vblank / sprite-0) diverges from the slow
    // core (found via Gradius asm-vs-slow lockstep at the attract-
    // mode sprite-0 wait).
    sub     x25, x25, #1
    bl      Lmmio_read
    and     w20, w0, #0xFF
    SET_NZ  w20
    NEXT    3

// ----- LDX abs (0xAE, 4 cycles) — RAM/PRG direct, MMIO via callback -----

_op_lda_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8              // base
    add     w4, w0, w21                     // +X (unmasked to detect cross)
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w0, #0xFF00
    cmp     w5, w6
    cset    w7, ne                          // w7 = extra cycle
    cmp     w4, #0x2000
    b.lo    Llda_abs_x_ram
    cmp     w4, #0x8000
    b.lo    Llda_abs_x_mmio
    cbz     x28, Llda_abs_x_mmio
    and     w4, w4, #0x7FFF
    ldrb    w20, [x28, x4]
    b       Llda_abs_x_done
Llda_abs_x_ram:
    and     w4, w4, #0x07FF
    ldrb    w20, [x27, x4]
Llda_abs_x_done:
    SET_NZ  w20
    cbnz    w7, Llda_abs_x_cross
    NEXT    4
Llda_abs_x_cross:
    NEXT    5
Llda_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w20, w0, #0xFF
    SET_NZ  w20
    cbnz    w7, Llda_abs_x_cross
    NEXT    4

// ----- LDA abs,Y (0xB9, 4/5 cycles) -----

_op_lda_abs_y:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w4, w0, w22
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w0, #0xFF00
    cmp     w5, w6
    cset    w7, ne
    cmp     w4, #0x2000
    b.lo    Llda_abs_y_ram
    cmp     w4, #0x8000
    b.lo    Llda_abs_y_mmio
    cbz     x28, Llda_abs_y_mmio
    and     w4, w4, #0x7FFF
    ldrb    w20, [x28, x4]
    b       Llda_abs_y_done
Llda_abs_y_ram:
    and     w4, w4, #0x07FF
    ldrb    w20, [x27, x4]
Llda_abs_y_done:
    SET_NZ  w20
    cbnz    w7, Llda_abs_y_cross
    NEXT    4
Llda_abs_y_cross:
    NEXT    5
Llda_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w20, w0, #0xFF
    SET_NZ  w20
    cbnz    w7, Llda_abs_y_cross
    NEXT    4

// ----- LDX abs,Y (0xBE, 4/5 cycles) -----

_op_lda_ind_y:
    FETCH_PC_BYTE w0
    add     w4, w0, #1
    and     w4, w4, #0xFF
    ldrb    w2, [x27, x0]
    ldrb    w3, [x27, x4]
    orr     w8, w2, w3, lsl #8              // base (w8 preserved)
    add     w4, w8, w22                     // effective
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w8, #0xFF00
    cmp     w5, w6
    cset    w7, ne
    cmp     w4, #0x2000
    b.lo    Llda_ind_y_ram
    cmp     w4, #0x8000
    b.lo    Llda_ind_y_mmio
    cbz     x28, Llda_ind_y_mmio
    and     w4, w4, #0x7FFF
    ldrb    w20, [x28, x4]
    b       Llda_ind_y_done
Llda_ind_y_ram:
    and     w4, w4, #0x07FF
    ldrb    w20, [x27, x4]
Llda_ind_y_done:
    SET_NZ  w20
    cbnz    w7, Llda_ind_y_cross
    NEXT    5
Llda_ind_y_cross:
    NEXT    6
Llda_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w20, w0, #0xFF
    SET_NZ  w20
    cbnz    w7, Llda_ind_y_cross
    NEXT    5

// ----- AND abs (0x2D, 4 cycles) — RAM/PRG direct, MMIO via callback -----

_op_ldx_imm:
    FETCH_PC_BYTE w21
    SET_NZ  w21
    NEXT    2

// ----- LDY #imm (0xA0, 2 cycles) -----

_op_ldx_zp:
    FETCH_PC_BYTE w0
    ldrb    w21, [x27, x0]
    SET_NZ  w21
    NEXT    3

// ----- LDY zp (0xA4, 3 cycles) -----

_op_ldx_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lldx_abs_ram
    cmp     w0, #0x8000
    b.lo    Lldx_abs_mmio
    cbz     x28, Lldx_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w21, [x28, x0]
    b       Lldx_abs_done
Lldx_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w21, [x27, x0]
Lldx_abs_done:
    SET_NZ  w21
    NEXT    4
Lldx_abs_mmio:
    mov     w9, w0
    // Dispatch-tick read alignment — see Llda_abs_mmio rationale.
    sub     x25, x25, #1
    bl      Lmmio_read
    and     w21, w0, #0xFF
    SET_NZ  w21
    NEXT    3

// ----- LDY abs (0xAC, 4 cycles) — RAM/PRG direct, MMIO via callback -----

_op_ldy_imm:
    FETCH_PC_BYTE w22
    SET_NZ  w22
    NEXT    2

// ----- LDA zp (0xA5, 3 cycles) -----

_op_ldy_zp:
    FETCH_PC_BYTE w0
    ldrb    w22, [x27, x0]
    SET_NZ  w22
    NEXT    3

// ----- STX zp (0x86, 3 cycles) -----

_op_ldy_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lldy_abs_ram
    cmp     w0, #0x8000
    b.lo    Lldy_abs_mmio
    cbz     x28, Lldy_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w22, [x28, x0]
    b       Lldy_abs_done
Lldy_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w22, [x27, x0]
Lldy_abs_done:
    SET_NZ  w22
    NEXT    4
Lldy_abs_mmio:
    mov     w9, w0
    // Dispatch-tick read alignment — see Llda_abs_mmio rationale.
    sub     x25, x25, #1
    bl      Lmmio_read
    and     w22, w0, #0xFF
    SET_NZ  w22
    NEXT    3

// ----- AND zp (0x25, 3 cycles) -----

_op_sta_zp:
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    1f
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrb    w0, [x28, x0]
    b       2f
1:  and     x0, x0, #0x07FF
    ldrb    w0, [x27, x0]
2:  add     x19, x19, #1
    and     x19, x19, #0xFFFF
    // w0 = zero-page address. Store A there (always RAM, no MMIO check).
    strb    w20, [x27, x0]
    NEXT    3

// ----- JMP abs (0x4C, 3 cycles) -----

_op_sta_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    strb    w20, [x27, x0]
    NEXT    4

// ----- STA abs,X (0x9D, 5 cycles) — always 5 (no page-cross discount) -----

_op_sta_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lsta_abs_mmio
    and     w0, w0, #0x07FF
    strb    w20, [x27, x0]
    NEXT    4
Lsta_abs_mmio:
    mov     w9, w0
    mov     w10, w20
    mov     w3, #4
    bl      Lmmio_write
    // Force end-of-block exit: any MMIO write could have changed
    // PPU state, triggered OAM DMA, or switched banks (MMC1+). Rust
    // re-queries the mapper on the next Nes::step, picking up fresh
    // x28 if the window moved.
    NEXT    4

// ----- STX abs (0x8E, 4 cycles) -----

_op_stx_zp:
    FETCH_PC_BYTE w0
    strb    w21, [x27, x0]
    NEXT    3

// ----- STY zp (0x84, 3 cycles) -----

_op_stx_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lstx_abs_mmio
    and     w0, w0, #0x07FF
    strb    w21, [x27, x0]
    NEXT    4
Lstx_abs_mmio:
    mov     w9, w0
    mov     w10, w21
    mov     w3, #4
    bl      Lmmio_write
    NEXT    4

// ----- STY abs (0x8C, 4 cycles) -----

_op_sty_zp:
    FETCH_PC_BYTE w0
    strb    w22, [x27, x0]
    NEXT    3

// ----- TAX (0xAA, 2 cycles) — X = A, set N/Z -----

_op_sty_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lsty_abs_mmio
    and     w0, w0, #0x07FF
    strb    w22, [x27, x0]
    NEXT    4
Lsty_abs_mmio:
    mov     w9, w0
    mov     w10, w22
    mov     w3, #4
    bl      Lmmio_write
    NEXT    4

// ----- JSR abs (0x20, 6 cycles) — push return_addr-1 (hi, lo), jump -----
// On entry PC is at operand_low. We fetch 2 operand bytes (advances PC
// by 2), compute new_pc from operands, then compute return_addr as
// the address of the LAST operand byte fetched (which is
// original_opcode_addr + 2). That's what 6502 pushes for RTS to
// resume at return_addr+1 = original_opcode_addr + 3, the next op.

_op_beq:
    FETCH_PC_BYTE w10
    sxtb    w10, w10                    // sign-extend signed-8 → 32
    tst     w24, #(1 << P_BIT_Z)
    b.eq    Lbeq_not_taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbeq_same_page
    NEXT    4
Lbeq_same_page:
    NEXT    3
Lbeq_not_taken:
    NEXT    2

// ----- AND #imm (0x29, 2 cycles) -----

_op_bne:
    // Fetch offset byte
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    1f
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrsb   w10, [x28, x0]
    b       2f
1:  and     x0, x0, #0x07FF
    ldrsb   w10, [x27, x0]
2:  add     x19, x19, #1
    and     x19, x19, #0xFFFF
    // Z bit (position P_BIT_Z in w24) determines whether to branch.
    tst     w24, #(1 << P_BIT_Z)
    b.ne    Lbne_not_taken
    // Taken — add signed offset to PC. Page cross check adds 1 cycle.
    // ldrsb wrote w10 sign-extended to 32 bits; `sxtw` source must
    // be a w-register, not x-register.
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbne_same_page
    NEXT    4
Lbne_same_page:
    NEXT    3
Lbne_not_taken:
    NEXT    2

// ============================================================================
// Phase 1 handlers — immediate/zero-page loads, stores, branches,
// register transfers, immediate ALU/compare. Each goes through the
// FETCH_PC_BYTE + SET_NZ macros + NEXT tail.
// ============================================================================

// ----- LDX #imm (0xA2, 2 cycles) -----

// ----- BPL rel (0x10, 2/3/4 cycles) — branch if N=0 -----
_op_bpl:
    FETCH_PC_BYTE w10
    sxtb    w10, w10
    tst     w24, #(1 << P_BIT_N)
    b.ne    Lbpl_not_taken              // N=1 → not taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbpl_same_page
    NEXT    4
Lbpl_same_page:
    NEXT    3
Lbpl_not_taken:
    NEXT    2

// ----- BMI rel (0x30) — branch if N=1 -----

_op_bmi:
    FETCH_PC_BYTE w10
    sxtb    w10, w10
    tst     w24, #(1 << P_BIT_N)
    b.eq    Lbmi_not_taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbmi_same_page
    NEXT    4
Lbmi_same_page:
    NEXT    3
Lbmi_not_taken:
    NEXT    2

// ----- BVC rel (0x50) — branch if V=0 -----

_op_bcc:
    FETCH_PC_BYTE w10
    sxtb    w10, w10
    tst     w24, #(1 << P_BIT_C)
    b.ne    Lbcc_not_taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbcc_same_page
    NEXT    4
Lbcc_same_page:
    NEXT    3
Lbcc_not_taken:
    NEXT    2

// ----- BCS rel (0xB0) — branch if C=1 -----

_op_bcs:
    FETCH_PC_BYTE w10
    sxtb    w10, w10
    tst     w24, #(1 << P_BIT_C)
    b.eq    Lbcs_not_taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbcs_same_page
    NEXT    4
Lbcs_same_page:
    NEXT    3
Lbcs_not_taken:
    NEXT    2

// ----- LDA abs (0xAD, 4 cycles) — RAM/PRG direct, MMIO via callback -----

_op_bvc:
    FETCH_PC_BYTE w10
    sxtb    w10, w10
    tst     w24, #(1 << P_BIT_V)
    b.ne    Lbvc_not_taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbvc_same_page
    NEXT    4
Lbvc_same_page:
    NEXT    3
Lbvc_not_taken:
    NEXT    2

// ----- BVS rel (0x70) — branch if V=1 -----

_op_bvs:
    FETCH_PC_BYTE w10
    sxtb    w10, w10
    tst     w24, #(1 << P_BIT_V)
    b.eq    Lbvs_not_taken
    mov     w11, w19
    add     x19, x19, w10, sxtw
    and     x19, x19, #0xFFFF
    eor     w12, w11, w19
    tst     w12, #0xFF00
    b.eq    Lbvs_same_page
    NEXT    4
Lbvs_same_page:
    NEXT    3
Lbvs_not_taken:
    NEXT    2

// ----- BCC rel (0x90) — branch if C=0 -----

_op_cmp_imm:
    FETCH_PC_BYTE w0
    and     w3, w20, #0xFF
    and     w4, w0, #0xFF
    sub     w1, w3, w4                  // A - imm (32-bit, unmasked)
    SET_NZ  w1
    // C = (A >= imm). Clear C then OR in if true.
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs                      // unsigned >= (HS = CS)
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    2

// ----- CPX #imm (0xE0, 2 cycles) -----

_op_cmp_zp:
    FETCH_PC_BYTE w0
    ldrb    w4, [x27, x0]
    and     w3, w20, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    3

// ----- BIT zp (0x24, 3 cycles) -----
// Z = (A & mem) == 0; N = mem bit 7; V = mem bit 6. A unchanged.

_op_cpx_imm:
    FETCH_PC_BYTE w0
    and     w3, w21, #0xFF
    and     w4, w0, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    2

// ----- CPY #imm (0xC0, 2 cycles) -----

_op_cpx_zp:
    FETCH_PC_BYTE w0
    ldrb    w4, [x27, x0]
    and     w3, w21, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    3

// ----- CPY zp (0xC4, 3 cy) -----

_op_cpy_imm:
    FETCH_PC_BYTE w0
    and     w3, w22, #0xFF
    and     w4, w0, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    2

// ============================================================================
// Phase 2a handlers — flag setters + register inc/dec + memory inc/dec zp.
// All trivial bit manipulation on w24 or register ops with SET_NZ.
// ============================================================================

// ----- CLC (0x18, 2 cycles) — clear C -----

_op_cpy_zp:
    FETCH_PC_BYTE w0
    ldrb    w4, [x27, x0]
    and     w3, w22, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    3

// ----- ASL zp,X (0x16, 6 cy) -----

_op_pha:
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF              // safety mask
    strb    w20, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    NEXT    3

// ----- PLA (0x68, 4 cycles) — pull A from stack, set N/Z -----

_op_pla:
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w20, [x27, x0]
    SET_NZ  w20
    NEXT    4

// ----- TXS (0x9A, 2 cycles) — SP = X, no flags -----

_op_php:
    orr     w1, w24, #(1 << P_BIT_B)
    orr     w1, w1, #(1 << P_BIT_E)
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    strb    w1, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    NEXT    3

// ============================================================================
// Phase 2l — indexed-indirect (zp,X) addressing variants.
// ptr = (zp + X) & 0xFF. base = read16 from ptr (zp wrap on high byte).
// ============================================================================

.macro IND_X_COMPUTE_ADDR
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF           // ptr low
    add     w3, w0, #1
    and     w3, w3, #0xFF           // ptr high (wrapped)
    ldrb    w4, [x27, x0]           // base low
    ldrb    w5, [x27, x3]           // base high
    orr     w0, w4, w5, lsl #8      // effective addr in w0
.endm

// ----- LDA (zp,X) (0xA1, 6 cy) -----

_op_plp:
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w1, [x27, x0]
    mov     w2, #(1 << P_BIT_B)
    bic     w1, w1, w2
    orr     w1, w1, #(1 << P_BIT_E)
    mov     w24, w1
    NEXT    4

// ----- STA abs (0x8D, 4 cycles) — RAM direct, $2000+ via callback -----

_op_tax:
    mov     w21, w20
    SET_NZ  w21
    NEXT    2

// ----- TAY (0xA8, 2 cycles) — Y = A, set N/Z -----

_op_tay:
    mov     w22, w20
    SET_NZ  w22
    NEXT    2

// ----- TXA (0x8A, 2 cycles) — A = X, set N/Z -----

_op_txa:
    mov     w20, w21
    SET_NZ  w20
    NEXT    2

// ----- TYA (0x98, 2 cycles) — A = Y, set N/Z -----

_op_tya:
    mov     w20, w22
    SET_NZ  w20
    NEXT    2

// ----- BEQ rel (0xF0, 2/3/4 cycles) — branch if Z set -----

_op_tsx:
    mov     w21, w23
    SET_NZ  w21
    NEXT    2

// ----- PHP (0x08, 3 cycles) — push P with B,E bits set -----

_op_txs:
    mov     w23, w21
    and     w23, w23, #0xFF
    NEXT    2

// ----- TSX (0xBA, 2 cycles) — X = SP, set N/Z -----

_op_adc_imm:
    FETCH_PC_BYTE w1
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    2

// ----- SBC #imm (0xE9, 2 cycles) — A = A - M - (1-C); implemented via ADC(~M) -----

_op_adc_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    ADC_CORE w1
    NEXT    3

// ----- SBC zp (0xE5, 3 cycles) -----

_op_sbc_imm:
    FETCH_PC_BYTE w1
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    2

// ----- ADC zp (0x65, 3 cycles) -----

_op_sbc_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    3

// ----- CMP abs (0xCD, 4 cycles) -----

_op_and_imm:
    FETCH_PC_BYTE w0
    and     w20, w20, w0
    SET_NZ  w20
    NEXT    2

// ----- ORA #imm (0x09, 2 cycles) -----

_op_and_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    and     w20, w20, w1
    SET_NZ  w20
    NEXT    3

// ----- ORA zp (0x05, 3 cycles) -----

_op_ora_imm:
    FETCH_PC_BYTE w0
    orr     w20, w20, w0
    SET_NZ  w20
    NEXT    2

// ----- EOR #imm (0x49, 2 cycles) -----

_op_ora_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    orr     w20, w20, w1
    SET_NZ  w20
    NEXT    3

// ----- EOR zp (0x45, 3 cycles) -----

_op_eor_imm:
    FETCH_PC_BYTE w0
    eor     w20, w20, w0
    SET_NZ  w20
    NEXT    2

// ----- CMP #imm (0xC9, 2 cycles) -----
// Sets N/Z on (A - imm) and C = (A >= imm).

_op_eor_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    eor     w20, w20, w1
    SET_NZ  w20
    NEXT    3

// ----- CMP zp (0xC5, 3 cycles) -----

_op_inx:
    add     w21, w21, #1
    and     w21, w21, #0xFF
    SET_NZ  w21
    NEXT    2

// ----- INY (0xC8, 2 cycles) — Y++, N/Z -----

_op_iny:
    add     w22, w22, #1
    and     w22, w22, #0xFF
    SET_NZ  w22
    NEXT    2

// ----- DEX (0xCA, 2 cycles) — X--, N/Z -----

_op_dex:
    sub     w21, w21, #1
    and     w21, w21, #0xFF
    SET_NZ  w21
    NEXT    2

// ----- DEY (0x88, 2 cycles) — Y--, N/Z -----

_op_dey:
    sub     w22, w22, #1
    and     w22, w22, #0xFF
    SET_NZ  w22
    NEXT    2

// ----- INC zp (0xE6, 5 cycles) — mem[zp]++, N/Z -----

_op_inc_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    add     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    5

// ----- DEC zp (0xC6, 5 cycles) — mem[zp]--, N/Z -----

_op_dec_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    sub     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    5

// ============================================================================
// Phase 2b handlers — more branches, absolute loads, zero-page logical,
// BIT zp, and PHA/PLA stack ops.
// ============================================================================

_op_jmp_abs:
    // Fetch low byte of target
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    1f
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrb    w10, [x28, x0]
    b       2f
1:  and     x0, x0, #0x07FF
    ldrb    w10, [x27, x0]
2:  add     x19, x19, #1
    and     x19, x19, #0xFFFF
    // Fetch high byte
    and     x0, x19, #0xFFFF
    cmp     x0, #0x2000
    b.lt    3f
    cbz     x28, Lexit_unimpl_fetch
    and     x0, x0, #0x7FFF
    ldrb    w11, [x28, x0]
    b       4f
3:  and     x0, x0, #0x07FF
    ldrb    w11, [x27, x0]
4:  // Compose target = (hi << 8) | lo
    orr     w19, w10, w11, lsl #8
    NEXT    3

// ----- BNE rel (0xD0, 2/3/4 cycles) -----

_op_jsr_abs:
    FETCH_PC_BYTE w2                        // low byte target
    // PC is now at operand_high byte. This is return_addr - 0 ... but
    // 6502 pushes PC-1 AFTER reading BOTH operand bytes. We still
    // need to fetch operand_high, then push PC as-is (which is
    // opcode+2 at that moment). 6502's RTS adds 1 to re-resume at
    // opcode+3. So: push PC BEFORE fetching operand_high.
    mov     x10, x19                        // save PC (= opcode+2)
    // Push high byte of x10, then low byte.
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    lsr     w1, w10, #8
    strb    w1, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    strb    w10, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    // Now fetch operand_high at PC (which is still opcode+2).
    FETCH_PC_BYTE w3
    orr     w19, w2, w3, lsl #8
    and     x19, x19, #0xFFFF
    NEXT    6

// ----- ADC #imm (0x69, 2 cycles) -----

_op_rts:
    // Pull low
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w2, [x27, x0]
    // Pull high
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w3, [x27, x0]
    orr     w19, w2, w3, lsl #8
    add     x19, x19, #1
    and     x19, x19, #0xFFFF
    NEXT    6

// ============================================================================
// Phase 2l+: remaining 16 official opcodes that were falling to the
// pure-Rust path — zp,X + abs,X addressing modes for ALU / shift /
// compare, plus the two missing CPx zp variants. These complete
// coverage of the official 6502 instruction set in ASM.
// ============================================================================

// ----- ORA zp,X (0x15, 4 cy) -----

_op_clc:
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    NEXT    2

// ----- SEC (0x38, 2 cycles) — set C -----

_op_sec:
    orr     w24, w24, #(1 << P_BIT_C)
    NEXT    2

// ----- CLI (0x58, 2 cycles) — clear I -----

_op_cli:
    mov     w2, #(1 << P_BIT_I)
    bic     w24, w24, w2
    NEXT    2

// ----- SEI (0x78, 2 cycles) — set I -----

_op_sei:
    orr     w24, w24, #(1 << P_BIT_I)
    NEXT    2

// ----- CLD (0xD8, 2 cycles) — clear D -----

_op_cld:
    mov     w2, #(1 << P_BIT_D)
    bic     w24, w24, w2
    NEXT    2

// ----- SED (0xF8, 2 cycles) — set D -----

_op_sed:
    orr     w24, w24, #(1 << P_BIT_D)
    NEXT    2

// ----- CLV (0xB8, 2 cycles) — clear V -----

_op_clv:
    mov     w2, #(1 << P_BIT_V)
    bic     w24, w24, w2
    NEXT    2

// ----- INX (0xE8, 2 cycles) — X++, N/Z -----

_op_bit_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]              // mem value
    // N = mem bit 7, V = mem bit 6 → clear those two bits in w24, then OR mem's
    mov     w2, #0xC0                   // N|V mask (bits 6+7)
    bic     w24, w24, w2
    and     w3, w1, #0xC0               // keep bits 6,7 of mem
    orr     w24, w24, w3
    // Z = (A & mem) == 0
    mov     w2, #(1 << P_BIT_Z)
    bic     w24, w24, w2
    and     w4, w20, w1
    cbnz    w4, Lbit_zp_done
    orr     w24, w24, #(1 << P_BIT_Z)
Lbit_zp_done:
    NEXT    3

// ----- PHA (0x48, 3 cycles) — push A onto stack -----

_op_bit_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lbit_abs_ram
    cmp     w0, #0x8000
    b.lo    Lbit_abs_mmio
    cbz     x28, Lbit_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Lbit_abs_do
Lbit_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Lbit_abs_do:
    mov     w2, #0xC0
    bic     w24, w24, w2
    and     w3, w1, #0xC0
    orr     w24, w24, w3
    mov     w2, #(1 << P_BIT_Z)
    bic     w24, w24, w2
    and     w4, w20, w1
    cbnz    w4, Lbit_abs_done
    orr     w24, w24, #(1 << P_BIT_Z)
Lbit_abs_done:
    NEXT    4
Lbit_abs_mmio:
    mov     w9, w0
    // Dispatch-tick read alignment — see Llda_abs_mmio rationale.
    // x25 is restored after the call because this path rejoins
    // Lbit_abs_do, whose shared tail still charges NEXT 4.
    sub     x25, x25, #1
    bl      Lmmio_read
    add     x25, x25, #1
    mov     w1, w0
    b       Lbit_abs_do

// ----- RTI (0x40, 6 cycles) — pull P, pull PC (lo, hi) -----

_op_asl_a:
    and     w1, w20, #0xFF
    lsl     w2, w1, #1
    lsr     w3, w2, #8
    and     w3, w3, #1                      // new C = old bit 7
    and     w20, w2, #0xFF
    // Clear N|Z|C then write
    mov     w10, #0x83                      // N|Z|C mask
    bic     w24, w24, w10
    tst     w20, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w20, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    2

// ----- LSR A (0x4A, 2 cycles) — A>>1, C=old bit 0, N always 0 -----

_op_lsr_a:
    and     w1, w20, #0xFF
    and     w3, w1, #1                      // new C = old bit 0
    lsr     w20, w1, #1
    mov     w10, #0x83
    bic     w24, w24, w10
    // N=0 always (result has bit 7 = 0)
    cbnz    w20, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    2

// ----- ROL A (0x2A, 2 cycles) — A<<1 | old_C; C=old bit 7 -----

_op_rol_a:
    and     w1, w20, #0xFF
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w2, w1, #1
    orr     w2, w2, w4
    lsr     w3, w2, #8
    and     w3, w3, #1                      // new C = old bit 7 (in bit 8 of w2)
    and     w20, w2, #0xFF
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w20, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w20, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    2

// ----- ROR A (0x6A, 2 cycles) — old_C → bit 7, A >>= 1, C=old bit 0 -----

_op_ror_a:
    and     w1, w20, #0xFF
    ubfx    w4, w24, #P_BIT_C, #1
    and     w3, w1, #1                      // new C = old bit 0
    lsr     w2, w1, #1
    orr     w20, w2, w4, lsl #7             // bit 7 ← old_C
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w20, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w20, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    2

// ============================================================================
// Phase 2f handlers — indexed loads (abs,X/Y) with page-cross cycle
// penalty, indirect-indexed LDA, abs logical/arith + MMIO bail, and
// zero-page RMW shifts.
// ============================================================================

// ----- LDA abs,X (0xBD, 4/5 cycles) -----

// ============================================================================
// COLD opcode handlers — abs-indexed arithmetic, illegal/unofficial ops,
// BRK/RTI, and other rarely-executed paths.
// ============================================================================

_op_lda_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Llda_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Llda_ind_x_mmio
    cbz     x28, Llda_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w20, [x28, x0]
    b       Llda_ind_x_done
Llda_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w20, [x27, x0]
Llda_ind_x_done:
    SET_NZ  w20
    NEXT    6
Llda_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    and     w20, w0, #0xFF
    SET_NZ  w20
    NEXT    6

// ----- STA (zp,X) (0x81, 6 cy) -----

_op_sta_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.hs    Lsta_ind_x_mmio
    and     w0, w0, #0x07FF
    strb    w20, [x27, x0]
    NEXT    6
Lsta_ind_x_mmio:
    mov     w9, w0
    mov     w10, w20
    // Last-cycle commit — see Lsta_abs_x_mmio rationale.
    sub     x25, x25, #6
    bl      Lmmio_write
    NEXT    0

// ----- ORA (zp,X) (0x01, 6 cy) -----

_op_ora_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Lora_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Lora_ind_x_mmio
    cbz     x28, Lora_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Lora_ind_x_do
Lora_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Lora_ind_x_do:
    orr     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    6
Lora_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    mov     w1, w0
    b       Lora_ind_x_do

// ----- AND (zp,X) (0x21, 6 cy) -----

_op_and_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Land_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Land_ind_x_mmio
    cbz     x28, Land_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Land_ind_x_do
Land_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Land_ind_x_do:
    and     w20, w20, w1
    SET_NZ  w20
    NEXT    6
Land_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    mov     w1, w0
    b       Land_ind_x_do

// ----- EOR (zp,X) (0x41, 6 cy) -----

_op_eor_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Leor_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Leor_ind_x_mmio
    cbz     x28, Leor_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Leor_ind_x_do
Leor_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Leor_ind_x_do:
    eor     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    6
Leor_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    mov     w1, w0
    b       Leor_ind_x_do

// ----- ADC (zp,X) (0x61, 6 cy) -----

_op_adc_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Ladc_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Ladc_ind_x_mmio
    cbz     x28, Ladc_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Ladc_ind_x_do
Ladc_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Ladc_ind_x_do:
    ADC_CORE w1
    NEXT    6
Ladc_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    and     w1, w0, #0xFF
    b       Ladc_ind_x_do

// ----- SBC (zp,X) (0xE1, 6 cy) -----

_op_sbc_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Lsbc_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Lsbc_ind_x_mmio
    cbz     x28, Lsbc_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Lsbc_ind_x_do
Lsbc_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Lsbc_ind_x_do:
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    6
Lsbc_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    eor     w1, w0, #0xFF
    and     w1, w1, #0xFF
    b       Lsbc_ind_x_do

// ----- CMP (zp,X) (0xC1, 6 cy) -----

_op_cmp_ind_x:
    IND_X_COMPUTE_ADDR
    cmp     w0, #0x2000
    b.lo    Lcmp_ind_x_ram
    cmp     w0, #0x8000
    b.lo    Lcmp_ind_x_mmio
    cbz     x28, Lcmp_ind_x_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Lcmp_ind_x_do
Lcmp_ind_x_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Lcmp_ind_x_do:
    and     w3, w20, #0xFF
    sub     w2, w3, w1
    SET_NZ  w2
    mov     w10, #(1 << P_BIT_C)
    bic     w24, w24, w10
    cmp     w3, w1
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    6
Lcmp_ind_x_mmio:
    mov     w9, w0
    mov     w2, #6
    bl      Lmmio_read
    mov     w1, w0
    b       Lcmp_ind_x_do

// ----- PLP (0x28, 4 cycles) — pull P (clear B, force E) -----

_op_cmp_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lcmp_abs_ram
    cmp     w0, #0x8000
    b.lo    Lcmp_abs_mmio
    cbz     x28, Lcmp_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w4, [x28, x0]
    b       Lcmp_abs_do
Lcmp_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w4, [x27, x0]
Lcmp_abs_do:
    and     w3, w20, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    4
Lcmp_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w4, w0, #0xFF
    b       Lcmp_abs_do

// ----- CPX abs (0xEC, 4 cycles) -----

_op_cpx_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lcpx_abs_ram
    cmp     w0, #0x8000
    b.lo    Lcpx_abs_mmio
    cbz     x28, Lcpx_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w4, [x28, x0]
    b       Lcpx_abs_do
Lcpx_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w4, [x27, x0]
Lcpx_abs_do:
    and     w3, w21, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    4
Lcpx_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w4, w0, #0xFF
    b       Lcpx_abs_do

// ----- CPY abs (0xCC, 4 cycles) -----

_op_cpy_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lcpy_abs_ram
    cmp     w0, #0x8000
    b.lo    Lcpy_abs_mmio
    cbz     x28, Lcpy_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w4, [x28, x0]
    b       Lcpy_abs_do
Lcpy_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w4, [x27, x0]
Lcpy_abs_do:
    and     w3, w22, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    4
Lcpy_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w4, w0, #0xFF
    b       Lcpy_abs_do

// ----- LDA $nn,X (0xB5, 4 cycles) — zp + X, wrap at $FF, RAM only -----

_op_sta_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Lsta_abs_x_mmio
    and     w0, w0, #0x07FF
    strb    w20, [x27, x0]
    NEXT    5
Lsta_abs_x_mmio:
    mov     w9, w0
    mov     w10, w20
    // Commit at the instruction's LAST cycle like the Rust core
    // (sta_write_byte_abs is the final entry in 0x9D's cycles array;
    // the non-indexed abs stores early-commit at cycle 0 in BOTH
    // engines, but indexed stores commit late). Pre-charging the full
    // 5 instruction cycles makes Lmmio_write's cumulative tick bring
    // APU/PPU to the write cycle before the bus write lands — without
    // this, APU register writes via STA $400x,X land 5 cycles early
    // and pulse phase diverges from the slow core (found via Gradius
    // asm-vs-slow lockstep; its sound engine uses indexed stores).
    sub     x25, x25, #5
    bl      Lmmio_write
    NEXT    0

// ----- STA abs,Y (0x99, 5 cycles) -----

_op_sta_abs_y:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w22
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Lsta_abs_y_mmio
    and     w0, w0, #0x07FF
    strb    w20, [x27, x0]
    NEXT    5
Lsta_abs_y_mmio:
    mov     w9, w0
    mov     w10, w20
    // Last-cycle commit — see Lsta_abs_x_mmio rationale.
    sub     x25, x25, #5
    bl      Lmmio_write
    NEXT    0

// ----- STA (zp),Y (0x91, 6 cycles) — indirect indexed Y -----

_op_sta_ind_y:
    FETCH_PC_BYTE w0                        // zp operand
    add     w4, w0, #1
    and     w4, w4, #0xFF                   // zp-wrapped next
    ldrb    w2, [x27, x0]                   // base low
    ldrb    w3, [x27, x4]                   // base high
    orr     w1, w2, w3, lsl #8
    add     w1, w1, w22
    and     w1, w1, #0xFFFF
    cmp     w1, #0x2000
    b.hs    Lsta_ind_y_mmio
    and     w1, w1, #0x07FF
    strb    w20, [x27, x1]
    NEXT    6
Lsta_ind_y_mmio:
    mov     w9, w1
    mov     w10, w20
    // Last-cycle commit — see Lsta_abs_x_mmio rationale.
    sub     x25, x25, #6
    bl      Lmmio_write
    NEXT    0

// ----- ASL A (0x0A, 2 cycles) — A<<1, C=old bit 7, N/Z from result -----

_op_ldx_abs_y:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w4, w0, w22
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w0, #0xFF00
    cmp     w5, w6
    cset    w7, ne
    cmp     w4, #0x2000
    b.lo    Lldx_abs_y_ram
    cmp     w4, #0x8000
    b.lo    Lldx_abs_y_mmio
    cbz     x28, Lldx_abs_y_mmio
    and     w4, w4, #0x7FFF
    ldrb    w21, [x28, x4]
    b       Lldx_abs_y_done
Lldx_abs_y_ram:
    and     w4, w4, #0x07FF
    ldrb    w21, [x27, x4]
Lldx_abs_y_done:
    SET_NZ  w21
    cbnz    w7, Lldx_abs_y_cross
    NEXT    4
Lldx_abs_y_cross:
    NEXT    5
Lldx_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w21, w0, #0xFF
    SET_NZ  w21
    cbnz    w7, Lldx_abs_y_cross
    NEXT    4

// ----- LDY abs,X (0xBC, 4/5 cycles) -----

_op_ldy_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w4, w0, w21
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w0, #0xFF00
    cmp     w5, w6
    cset    w7, ne
    cmp     w4, #0x2000
    b.lo    Lldy_abs_x_ram
    cmp     w4, #0x8000
    b.lo    Lldy_abs_x_mmio
    cbz     x28, Lldy_abs_x_mmio
    and     w4, w4, #0x7FFF
    ldrb    w22, [x28, x4]
    b       Lldy_abs_x_done
Lldy_abs_x_ram:
    and     w4, w4, #0x07FF
    ldrb    w22, [x27, x4]
Lldy_abs_x_done:
    SET_NZ  w22
    cbnz    w7, Lldy_abs_x_cross
    NEXT    4
Lldy_abs_x_cross:
    NEXT    5
Lldy_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w22, w0, #0xFF
    SET_NZ  w22
    cbnz    w7, Lldy_abs_x_cross
    NEXT    4

// ----- LDA (zp),Y (0xB1, 5/6 cycles) -----
// base16 = read16(zp_operand, zp wraps); effective = base + Y.

_op_and_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Land_abs_ram
    cmp     w0, #0x8000
    b.lo    Land_abs_mmio
    cbz     x28, Land_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Land_abs_do
Land_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Land_abs_do:
    and     w20, w20, w1
    SET_NZ  w20
    NEXT    4
Land_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    4

// ----- ORA abs (0x0D, 4 cycles) -----

_op_ora_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lora_abs_ram
    cmp     w0, #0x8000
    b.lo    Lora_abs_mmio
    cbz     x28, Lora_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Lora_abs_do
Lora_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Lora_abs_do:
    orr     w20, w20, w1
    SET_NZ  w20
    NEXT    4
Lora_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    orr     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    4

// ----- EOR abs (0x4D, 4 cycles) -----

_op_eor_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Leor_abs_ram
    cmp     w0, #0x8000
    b.lo    Leor_abs_mmio
    cbz     x28, Leor_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Leor_abs_do
Leor_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Leor_abs_do:
    eor     w20, w20, w1
    SET_NZ  w20
    NEXT    4
Leor_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    eor     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    4

// ----- ADC abs (0x6D, 4 cycles) -----

_op_adc_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Ladc_abs_ram
    cmp     w0, #0x8000
    b.lo    Ladc_abs_mmio
    cbz     x28, Ladc_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Ladc_abs_do
Ladc_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Ladc_abs_do:
    ADC_CORE w1
    NEXT    4
Ladc_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w1, w0, #0xFF
    ADC_CORE w1
    NEXT    4

// ----- SBC abs (0xED, 4 cycles) -----

_op_sbc_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Lsbc_abs_ram
    cmp     w0, #0x8000
    b.lo    Lsbc_abs_mmio
    cbz     x28, Lsbc_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w1, [x28, x0]
    b       Lsbc_abs_do
Lsbc_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
Lsbc_abs_do:
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    4
Lsbc_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    eor     w1, w0, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    4

// ----- ASL zp (0x06, 5 cycles) -----

_op_asl_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    lsl     w2, w1, #1
    lsr     w3, w2, #8
    and     w3, w3, #1                      // new C
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    5

// ----- LSR zp (0x46, 5 cycles) -----

_op_lsr_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    and     w3, w1, #1                      // new C
    lsr     w2, w1, #1
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    5

// ----- ROL zp (0x26, 5 cycles) -----

_op_rol_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w2, w1, #1
    orr     w2, w2, w4
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    5

// ----- ROR zp (0x66, 5 cycles) -----

_op_ror_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    and     w3, w1, #1
    lsr     w2, w1, #1
    orr     w2, w2, w4, lsl #7
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    5

// ============================================================================
// Phase 2g handlers — JMP (abs), INC/DEC abs (RAM or PRG fast path,
// MMIO bails), INC/DEC abs,X.
// ============================================================================

// ----- JMP (ind) (0x6C, 5 cycles) — classic 6502 $xxFF page-wrap bug -----
// PCL fetched from (ind), PCH fetched from (ind & 0xFF00) | ((ind+1) & 0xFF).

_op_jmp_ind:
    FETCH_PC_BYTE w2                    // ind low
    FETCH_PC_BYTE w3                    // ind high
    orr     w0, w2, w3, lsl #8          // pointer addr
    // PCL address = pointer
    // PCH address = (pointer & 0xFF00) | ((pointer+1) & 0xFF)  — 6502 bug
    mov     w8, w0                      // save pointer
    add     w1, w0, #1
    // ptr_hi = (ptr & FF00) | (ptr+1 & 00FF)
    and     w4, w8, #0xFF00
    and     w5, w1, #0x00FF
    orr     w4, w4, w5
    // Read PCL from ptr (w0) — classify
    cmp     w0, #0x2000
    b.lo    Ljmp_ind_pcl_ram
    cmp     w0, #0x8000
    b.lo    Ljmp_ind_bail
    cbz     x28, Ljmp_ind_bail
    and     w0, w0, #0x7FFF
    ldrb    w2, [x28, x0]
    b       Ljmp_ind_pcl_done
Ljmp_ind_pcl_ram:
    and     w0, w0, #0x07FF
    ldrb    w2, [x27, x0]
Ljmp_ind_pcl_done:
    // Read PCH from w4
    cmp     w4, #0x2000
    b.lo    Ljmp_ind_pch_ram
    cmp     w4, #0x8000
    b.lo    Ljmp_ind_bail
    cbz     x28, Ljmp_ind_bail
    and     w4, w4, #0x7FFF
    ldrb    w3, [x28, x4]
    b       Ljmp_ind_pch_done
Ljmp_ind_pch_ram:
    and     w4, w4, #0x07FF
    ldrb    w3, [x27, x4]
Ljmp_ind_pch_done:
    orr     w19, w2, w3, lsl #8
    and     x19, x19, #0xFFFF
    NEXT    5
Ljmp_ind_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- INC abs (0xEE, 6 cycles) — RAM direct, MMIO bails -----

_op_inc_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Linc_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    add     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    6
Linc_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- DEC abs (0xCE, 6 cycles) -----

_op_dec_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Ldec_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    sub     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    6
Ldec_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- INC abs,X (0xFE, 7 cycles) -----

_op_inc_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Linc_abs_x_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    add     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    7
Linc_abs_x_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- DEC abs,X (0xDE, 7 cycles) -----

_op_dec_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Ldec_abs_x_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    sub     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    7
Ldec_abs_x_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Additional addressing modes: zp,Y (LDX/STX), ind,X (LDA/STA),
// abs,X/Y (AND/ORA/EOR/ADC/SBC/CMP — reading via RAM/PRG/MMIO callback).
// ============================================================================

// ----- LDX zp,Y (0xB6, 4 cycles) — zp + Y, wrap $FF -----

_op_ldx_zpy:
    FETCH_PC_BYTE w0
    add     w0, w0, w22
    and     w0, w0, #0xFF
    ldrb    w21, [x27, x0]
    SET_NZ  w21
    NEXT    4

// ----- STX zp,Y (0x96, 4 cycles) -----

_op_stx_zpy:
    FETCH_PC_BYTE w0
    add     w0, w0, w22
    and     w0, w0, #0xFF
    strb    w21, [x27, x0]
    NEXT    4

// ----- LDY zp,X (0xB4, 4 cycles) -----

_op_ldy_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w22, [x27, x0]
    SET_NZ  w22
    NEXT    4

// ----- STY zp,X (0x94, 4 cycles) -----

_op_sty_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    strb    w22, [x27, x0]
    NEXT    4

// ----- BRK (0x00, 7 cycles) — software interrupt / IRQ via $FFFE -----
// BRK is a 1-byte opcode but semantically it consumes one extra byte
// (the one after the opcode is considered the "signature"). CPU
// advances PC past that byte before pushing the return address.

_op_brk:
    // PC was advanced by 1 during opcode fetch. Advance once more
    // to skip the BRK signature byte.
    add     x19, x19, #1
    and     x19, x19, #0xFFFF
    // Push PCH, then PCL
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    lsr     w1, w19, #8
    strb    w1, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    strb    w19, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    // Push P with B flag (bit 4) and E flag (bit 5) set.
    orr     w1, w24, #(1 << P_BIT_B)
    orr     w1, w1, #(1 << P_BIT_E)
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    strb    w1, [x27, x0]
    sub     w23, w23, #1
    and     w23, w23, #0xFF
    // Set I flag (interrupt disable)
    orr     w24, w24, #(1 << P_BIT_I)
    // Load PC from IRQ/BRK vector at $FFFE/$FFFF (PRG-mapped on NROM).
    cbz     x28, Lbrk_bail
    movz    w0, #0xFFFE, lsl #0
    and     w0, w0, #0x7FFF
    ldrb    w2, [x28, x0]
    add     w0, w0, #1
    and     w0, w0, #0x7FFF
    ldrb    w3, [x28, x0]
    orr     w19, w2, w3, lsl #8
    NEXT    7
Lbrk_bail:
    // Without a PRG pointer we can't read the vector; bail to Rust.
    // PC is at [fetched+1], rewind to opcode so Rust re-executes.
    sub     x19, x19, #2
    and     x19, x19, #0xFFFF
    mov     x10, #0
    b       Lasm_bail_rewind

// ============================================================================
// Phase 2i — indirect-indexed Y non-store variants (AND/ORA/EOR/ADC/SBC/CMP).
// Pattern: base16 = read16(zp_operand, zp wraps); effective = base + Y;
// RAM direct / PRG via x28 / MMIO via callback.
// ============================================================================

.macro IND_Y_FETCH_VALUE
    FETCH_PC_BYTE w0                        // zp operand
    add     w4, w0, #1
    and     w4, w4, #0xFF
    ldrb    w2, [x27, x0]
    ldrb    w3, [x27, x4]
    orr     w8, w2, w3, lsl #8              // base (w8 preserved)
    add     w4, w8, w22                     // effective
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w8, #0xFF00
    cmp     w5, w6
    cset    w7, ne                          // w7 = extra cycle
.endm

// Shared read from effective addr in w4 → w1, classifying RAM/PRG/MMIO.
// Clobbers x0. Jumps to \bail_lbl on MMIO requiring callback.
.macro IND_Y_LOAD_MEM rd, bail_lbl
    cmp     w4, #0x2000
    b.lo    100f
    cmp     w4, #0x8000
    b.lo    \bail_lbl
    cbz     x28, \bail_lbl
    and     w1, w4, #0x7FFF
    ldrb    \rd, [x28, x1]
    b       101f
100: and    w1, w4, #0x07FF
    ldrb    \rd, [x27, w1, uxtw]
101:
.endm

// ----- ORA (zp),Y (0x11, 5/6 cycles) -----

_op_ora_ind_y:
    IND_Y_FETCH_VALUE
    IND_Y_LOAD_MEM w1, Lora_ind_y_mmio
    orr     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Lora_ind_y_cross
    NEXT    5
Lora_ind_y_cross:
    NEXT    6
Lora_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #5
    bl      Lmmio_read
    ldr     w7, [sp], #16
    orr     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Lora_ind_y_cross
    NEXT    5

// ----- AND (zp),Y (0x31) -----

_op_and_ind_y:
    IND_Y_FETCH_VALUE
    IND_Y_LOAD_MEM w1, Land_ind_y_mmio
    and     w20, w20, w1
    SET_NZ  w20
    cbnz    w7, Land_ind_y_cross
    NEXT    5
Land_ind_y_cross:
    NEXT    6
Land_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #5
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Land_ind_y_cross
    NEXT    5

// ----- EOR (zp),Y (0x51) -----

_op_eor_ind_y:
    IND_Y_FETCH_VALUE
    IND_Y_LOAD_MEM w1, Leor_ind_y_mmio
    eor     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Leor_ind_y_cross
    NEXT    5
Leor_ind_y_cross:
    NEXT    6
Leor_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #5
    bl      Lmmio_read
    ldr     w7, [sp], #16
    eor     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Leor_ind_y_cross
    NEXT    5

// ----- ADC (zp),Y (0x71) -----

_op_adc_ind_y:
    IND_Y_FETCH_VALUE
    IND_Y_LOAD_MEM w1, Ladc_ind_y_mmio
    ADC_CORE w1
    cbnz    w7, Ladc_ind_y_cross
    NEXT    5
Ladc_ind_y_cross:
    NEXT    6
Ladc_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #5
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w1, w0, #0xFF
    ADC_CORE w1
    cbnz    w7, Ladc_ind_y_cross
    NEXT    5

// ----- SBC (zp),Y (0xF1) -----

_op_sbc_ind_y:
    IND_Y_FETCH_VALUE
    IND_Y_LOAD_MEM w1, Lsbc_ind_y_mmio
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    cbnz    w7, Lsbc_ind_y_cross
    NEXT    5
Lsbc_ind_y_cross:
    NEXT    6
Lsbc_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #5
    bl      Lmmio_read
    ldr     w7, [sp], #16
    eor     w1, w0, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    cbnz    w7, Lsbc_ind_y_cross
    NEXT    5

// ----- CMP (zp),Y (0xD1) -----

_op_cmp_ind_y:
    IND_Y_FETCH_VALUE
    IND_Y_LOAD_MEM w1, Lcmp_ind_y_mmio
Lcmp_ind_y_do:
    and     w3, w20, #0xFF
    sub     w2, w3, w1
    SET_NZ  w2
    mov     w10, #(1 << P_BIT_C)
    bic     w24, w24, w10
    cmp     w3, w1
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    cbnz    w7, Lcmp_ind_y_cross
    NEXT    5
Lcmp_ind_y_cross:
    NEXT    6
Lcmp_ind_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #5
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w1, w0, #0xFF
    b       Lcmp_ind_y_do

// ============================================================================
// Phase 2j — abs,X/Y non-store variants (AND/ORA/EOR/ADC/SBC/CMP).
// ============================================================================

.macro ABS_X_FETCH_VALUE
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w4, w0, w21
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w0, #0xFF00
    cmp     w5, w6
    cset    w7, ne
.endm

.macro ABS_Y_FETCH_VALUE
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w4, w0, w22
    and     w4, w4, #0xFFFF
    and     w5, w4, #0xFF00
    and     w6, w0, #0xFF00
    cmp     w5, w6
    cset    w7, ne
.endm

.macro ABS_X_LOAD_MEM rd, bail_lbl
    cmp     w4, #0x2000
    b.lo    200f
    cmp     w4, #0x8000
    b.lo    \bail_lbl
    cbz     x28, \bail_lbl
    and     w1, w4, #0x7FFF
    ldrb    \rd, [x28, w1, uxtw]
    b       201f
200: and    w1, w4, #0x07FF
    ldrb    \rd, [x27, w1, uxtw]
201:
.endm

// ----- AND abs,X (0x3D, 4/5) -----

_op_and_abs_x:
    ABS_X_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Land_abs_x_mmio
    and     w20, w20, w1
    SET_NZ  w20
    cbnz    w7, Land_abs_x_cross
    NEXT    4
Land_abs_x_cross:
    NEXT    5
Land_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Land_abs_x_cross
    NEXT    4

// ----- ORA abs,X (0x1D) -----

_op_ora_abs_x:
    ABS_X_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Lora_abs_x_mmio
    orr     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Lora_abs_x_cross
    NEXT    4
Lora_abs_x_cross:
    NEXT    5
Lora_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    orr     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Lora_abs_x_cross
    NEXT    4

// ----- EOR abs,X (0x5D) -----

_op_eor_abs_x:
    ABS_X_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Leor_abs_x_mmio
    eor     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Leor_abs_x_cross
    NEXT    4
Leor_abs_x_cross:
    NEXT    5
Leor_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    eor     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Leor_abs_x_cross
    NEXT    4

// ----- ADC abs,X (0x7D) -----

_op_adc_abs_x:
    ABS_X_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Ladc_abs_x_mmio
    ADC_CORE w1
    cbnz    w7, Ladc_abs_x_cross
    NEXT    4
Ladc_abs_x_cross:
    NEXT    5
Ladc_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w1, w0, #0xFF
    ADC_CORE w1
    cbnz    w7, Ladc_abs_x_cross
    NEXT    4

// ----- SBC abs,X (0xFD) -----

_op_sbc_abs_x:
    ABS_X_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Lsbc_abs_x_mmio
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    cbnz    w7, Lsbc_abs_x_cross
    NEXT    4
Lsbc_abs_x_cross:
    NEXT    5
Lsbc_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    eor     w1, w0, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    cbnz    w7, Lsbc_abs_x_cross
    NEXT    4

// ----- CMP abs,X (0xDD) -----

_op_cmp_abs_x:
    ABS_X_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Lcmp_abs_x_mmio
Lcmp_abs_x_do:
    and     w3, w20, #0xFF
    sub     w2, w3, w1
    SET_NZ  w2
    mov     w10, #(1 << P_BIT_C)
    bic     w24, w24, w10
    cmp     w3, w1
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    cbnz    w7, Lcmp_abs_x_cross
    NEXT    4
Lcmp_abs_x_cross:
    NEXT    5
Lcmp_abs_x_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w1, w0, #0xFF
    b       Lcmp_abs_x_do

// ----- AND abs,Y (0x39) -----

_op_and_abs_y:
    ABS_Y_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Land_abs_y_mmio
    and     w20, w20, w1
    SET_NZ  w20
    cbnz    w7, Land_abs_y_cross
    NEXT    4
Land_abs_y_cross:
    NEXT    5
Land_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Land_abs_y_cross
    NEXT    4

// ----- ORA abs,Y (0x19) -----

_op_ora_abs_y:
    ABS_Y_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Lora_abs_y_mmio
    orr     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Lora_abs_y_cross
    NEXT    4
Lora_abs_y_cross:
    NEXT    5
Lora_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    orr     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Lora_abs_y_cross
    NEXT    4

// ----- EOR abs,Y (0x59) -----

_op_eor_abs_y:
    ABS_Y_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Leor_abs_y_mmio
    eor     w20, w20, w1
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Leor_abs_y_cross
    NEXT    4
Leor_abs_y_cross:
    NEXT    5
Leor_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    eor     w20, w20, w0
    and     w20, w20, #0xFF
    SET_NZ  w20
    cbnz    w7, Leor_abs_y_cross
    NEXT    4

// ----- ADC abs,Y (0x79) -----

_op_adc_abs_y:
    ABS_Y_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Ladc_abs_y_mmio
    ADC_CORE w1
    cbnz    w7, Ladc_abs_y_cross
    NEXT    4
Ladc_abs_y_cross:
    NEXT    5
Ladc_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w1, w0, #0xFF
    ADC_CORE w1
    cbnz    w7, Ladc_abs_y_cross
    NEXT    4

// ----- SBC abs,Y (0xF9) -----

_op_sbc_abs_y:
    ABS_Y_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Lsbc_abs_y_mmio
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    cbnz    w7, Lsbc_abs_y_cross
    NEXT    4
Lsbc_abs_y_cross:
    NEXT    5
Lsbc_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    eor     w1, w0, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    cbnz    w7, Lsbc_abs_y_cross
    NEXT    4

// ----- CMP abs,Y (0xD9) -----

_op_cmp_abs_y:
    ABS_Y_FETCH_VALUE
    ABS_X_LOAD_MEM w1, Lcmp_abs_y_mmio
Lcmp_abs_y_do:
    and     w3, w20, #0xFF
    sub     w2, w3, w1
    SET_NZ  w2
    mov     w10, #(1 << P_BIT_C)
    bic     w24, w24, w10
    cmp     w3, w1
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    cbnz    w7, Lcmp_abs_y_cross
    NEXT    4
Lcmp_abs_y_cross:
    NEXT    5
Lcmp_abs_y_mmio:
    str     w7, [sp, #-16]!
    mov     w9, w4
    add     w2, w7, #4
    bl      Lmmio_read
    ldr     w7, [sp], #16
    and     w1, w0, #0xFF
    b       Lcmp_abs_y_do

// ============================================================================
// Phase 2k — memory shifts (ASL/LSR/ROL/ROR abs, zp,X, abs,X).
// 6502 shifts on memory are read-modify-write; for RAM we do inline
// RMW, for PRG range we bail (games don't RMW ROM; if they do we fall
// back to Rust).
// ============================================================================

// ----- ASL abs (0x0E, 6 cy) -----

_op_asl_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lasl_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    lsl     w2, w1, #1
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6
Lasl_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- LSR abs (0x4E, 6 cy) -----

_op_lsr_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Llsr_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    and     w3, w1, #1
    lsr     w2, w1, #1
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6
Llsr_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- ROL abs (0x2E, 6 cy) -----

_op_rol_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lrol_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w2, w1, #1
    orr     w2, w2, w4
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6
Lrol_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- ROR abs (0x6E, 6 cy) -----

_op_ror_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lror_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    and     w3, w1, #1
    lsr     w2, w1, #1
    orr     w2, w2, w4, lsl #7
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6
Lror_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- INC zp,X (0xF6, 6 cy) -----

_op_inc_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    add     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    6

// ----- DEC zp,X (0xD6, 6 cy) -----

_op_dec_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    sub     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    SET_NZ  w1
    NEXT    6

// ----- BIT abs (0x2C, 4 cy) — Z=(A&mem)==0, N=mem bit 7, V=mem bit 6 -----

_op_rti:
    // Pop P (ignore bit 4 B, keep bit 5 E always set)
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w1, [x27, x0]
    // Clear B, force E
    mov     w2, #(1 << P_BIT_B)
    bic     w1, w1, w2
    orr     w1, w1, #(1 << P_BIT_E)
    mov     w24, w1
    // Pop PCL
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w2, [x27, x0]
    // Pop PCH
    add     w23, w23, #1
    and     w23, w23, #0xFF
    add     x0, x23, #0x100
    and     x0, x0, #0x1FF
    ldrb    w3, [x27, x0]
    orr     w19, w2, w3, lsl #8
    NEXT    6

// ----- RTS (0x60, 6 cycles) — pull PC (lo, hi), add 1 -----

_op_ora_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    orr     w20, w20, w1
    SET_NZ  w20
    NEXT    4

// ----- AND zp,X (0x35, 4 cy) -----

_op_and_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    and     w20, w20, w1
    SET_NZ  w20
    NEXT    4

// ----- EOR zp,X (0x55, 4 cy) -----

_op_eor_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    eor     w20, w20, w1
    SET_NZ  w20
    NEXT    4

// ----- ADC zp,X (0x75, 4 cy) -----

_op_adc_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    ADC_CORE w1
    NEXT    4

// ----- CMP zp,X (0xD5, 4 cy) -----

_op_cmp_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w4, [x27, x0]
    and     w3, w20, #0xFF
    sub     w1, w3, w4
    SET_NZ  w1
    mov     w2, #(1 << P_BIT_C)
    bic     w24, w24, w2
    cmp     w3, w4
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    4

// ----- SBC zp,X (0xF5, 4 cy) -----

_op_sbc_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    4

// ----- CPX zp (0xE4, 3 cy) -----

_op_asl_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    lsl     w2, w1, #1
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6

// ----- LSR zp,X (0x56, 6 cy) -----

_op_lsr_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    and     w3, w1, #1
    lsr     w2, w1, #1
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6

// ----- ROL zp,X (0x36, 6 cy) -----

_op_rol_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w2, w1, #1
    orr     w2, w2, w4
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6

// ----- ROR zp,X (0x76, 6 cy) -----

_op_ror_zpx:
    FETCH_PC_BYTE w0
    add     w0, w0, w21
    and     w0, w0, #0xFF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    and     w3, w1, #1
    lsr     w2, w1, #1
    orr     w2, w2, w4, lsl #7
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    6

// ----- ASL abs,X (0x1E, 7 cy) — RAM only; MMIO bail -----

_op_asl_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Lasl_abs_x_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    lsl     w2, w1, #1
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    7
Lasl_abs_x_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- LSR abs,X (0x5E, 7 cy) -----

_op_lsr_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Llsr_abs_x_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    and     w3, w1, #1
    lsr     w2, w1, #1
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    7
Llsr_abs_x_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- ROL abs,X (0x3E, 7 cy) -----

_op_rol_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Lrol_abs_x_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w2, w1, #1
    orr     w2, w2, w4
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    7
Lrol_abs_x_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ----- ROR abs,X (0x7E, 7 cy) -----

_op_ror_abs_x:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w21
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.hs    Lror_abs_x_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    and     w3, w1, #1
    lsr     w2, w1, #1
    orr     w2, w2, w4, lsl #7
    strb    w2, [x27, x0]
    mov     w10, #0x83
    bic     w24, w24, w10
    tst     w2, #0x80
    cset    w10, ne
    orr     w24, w24, w10, lsl #P_BIT_N
    cbnz    w2, 1f
    orr     w24, w24, #(1 << P_BIT_Z)
1:  orr     w24, w24, w3, lsl #P_BIT_C
    NEXT    7
Lror_abs_x_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3a: NOP variants (stable illegals). Games that use buggy
// assemblers or copy-protected-binary streams occasionally land on
// these — real 6502 ignores the operand-read side effects except
// for the extra cycles. We match by doing the operand fetch (which
// has the same cycle cost as a read) + no register updates.
// ============================================================================

// Implicit NOP (0x1A 0x3A 0x5A 0x7A 0xDA 0xFA) — 2 cycles, no operand.

_op_nop_impl2:
    NEXT    2

// NOP imm (0x80 0x82 0x89 0xC2 0xE2) — 2 cycles, skip 1 operand byte.

_op_nop_imm:
    FETCH_PC_BYTE w0
    NEXT    2

// NOP zp (0x04 0x44 0x64) — 3 cycles, skip 1 operand byte, dummy read.

_op_nop_zp:
    FETCH_PC_BYTE w0
    // Real 6502 reads [zp]; the read has no visible effect on RAM.
    // Skip the load — no writes or register updates happen here.
    NEXT    3

// NOP zp,X (0x14 0x34 0x54 0x74 0xD4 0xF4) — 4 cycles.

_op_nop_zpx:
    FETCH_PC_BYTE w0
    NEXT    4

// NOP abs (0x0C) — 4 cycles, skip 2 operand bytes.

_op_nop_abs:
    FETCH_PC_BYTE w0
    FETCH_PC_BYTE w0
    NEXT    4

// NOP abs,X (0x1C 0x3C 0x5C 0x7C 0xDC 0xFC) — 4 or 5 cycles. Take the
// 4-cycle lower bound — mixing per-execution page-cross cycles here
// requires tracking the cross, which isn't worth the complexity for
// an instruction whose only observable side effect is the skip.

_op_nop_abs_x:
    FETCH_PC_BYTE w0
    FETCH_PC_BYTE w0
    NEXT    4

// ============================================================================
// Phase 3b: LAX — LDA + LDX combined. Sets A = X = mem, updates N/Z.
// Used by a few commercial titles (Battletoads etc.). All addressing
// modes except imm (which is unstable on some CPUs) route to this.
// ============================================================================

// LAX zp (0xA7, 3 cy)

_op_lax_zp:
    FETCH_PC_BYTE w0
    ldrb    w20, [x27, x0]
    mov     w21, w20
    SET_NZ  w20
    NEXT    3

// LAX zp,Y (0xB7, 4 cy)

_op_lax_zpy:
    FETCH_PC_BYTE w0
    add     w0, w0, w22
    and     w0, w0, #0xFF
    ldrb    w20, [x27, x0]
    mov     w21, w20
    SET_NZ  w20
    NEXT    4

// LAX abs (0xAF, 4 cy) — RAM/PRG fast path; MMIO via callback.

_op_lax_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.lo    Llax_abs_ram
    cmp     w0, #0x8000
    b.lo    Llax_abs_mmio
    cbz     x28, Llax_abs_mmio
    and     w0, w0, #0x7FFF
    ldrb    w20, [x28, x0]
    b       Llax_abs_done
Llax_abs_ram:
    and     w0, w0, #0x07FF
    ldrb    w20, [x27, x0]
Llax_abs_done:
    mov     w21, w20
    SET_NZ  w20
    NEXT    4
Llax_abs_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w20, w0, #0xFF
    mov     w21, w20
    SET_NZ  w20
    NEXT    4

// LAX abs,Y (0xBF, 4-5 cy; we use 4 plus ignore page-cross penalty)

_op_lax_abs_y:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    add     w0, w0, w22
    and     w0, w0, #0xFFFF
    cmp     w0, #0x2000
    b.lo    Llax_aby_ram
    cmp     w0, #0x8000
    b.lo    Llax_aby_mmio
    cbz     x28, Llax_aby_mmio
    and     w0, w0, #0x7FFF
    ldrb    w20, [x28, x0]
    b       Llax_aby_done
Llax_aby_ram:
    and     w0, w0, #0x07FF
    ldrb    w20, [x27, x0]
Llax_aby_done:
    mov     w21, w20
    SET_NZ  w20
    NEXT    4
Llax_aby_mmio:
    mov     w9, w0
    mov     w2, #4
    bl      Lmmio_read
    and     w20, w0, #0xFF
    mov     w21, w20
    SET_NZ  w20
    NEXT    4

// ============================================================================
// Phase 3c: SAX — store (A & X). Does not update any flags.
// ============================================================================

_op_sax_zp:
    FETCH_PC_BYTE w0
    and     w1, w20, w21
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    NEXT    3

_op_sax_zpy:
    FETCH_PC_BYTE w0
    add     w0, w0, w22
    and     w0, w0, #0xFF
    and     w1, w20, w21
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    NEXT    4

_op_sax_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lsax_abs_bail
    and     w0, w0, #0x07FF
    and     w1, w20, w21
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    NEXT    4
Lsax_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3d: DCP — DEC then CMP. RMW, 5-7 cycles by addressing mode.
// Only zp + abs variants implemented in ASM; indexed variants bail.
// ============================================================================

// DCP zp (0xC7, 5 cy)

_op_dcp_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    sub     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    // CMP A, w1
    and     w3, w20, #0xFF
    sub     w2, w3, w1
    SET_NZ  w2
    mov     w4, #(1 << P_BIT_C)
    bic     w24, w24, w4
    cmp     w3, w1
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    5

// DCP abs (0xCF, 6 cy)

_op_dcp_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Ldcp_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    sub     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    and     w3, w20, #0xFF
    sub     w2, w3, w1
    SET_NZ  w2
    mov     w4, #(1 << P_BIT_C)
    bic     w24, w24, w4
    cmp     w3, w1
    cset    w5, hs
    orr     w24, w24, w5, lsl #P_BIT_C
    NEXT    6
Ldcp_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3e: ISC (ISB) — INC then SBC. RMW, 5-7 cycles.
// zp + abs only; indexed bails.
// ============================================================================

// ISC zp (0xE7, 5 cy)

_op_isc_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    add     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    5

// ISC abs (0xEF, 6 cy)

_op_isc_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lisc_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    add     w1, w1, #1
    and     w1, w1, #0xFF
    strb    w1, [x27, x0]
    eor     w1, w1, #0xFF
    and     w1, w1, #0xFF
    ADC_CORE w1
    NEXT    6
Lisc_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3f: SLO — ASL then ORA. RMW, 5-7 cycles. zp + abs only.
// ============================================================================

// SLO zp (0x07, 5 cy)

_op_slo_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    lsl     w2, w1, #1
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    // C from shift
    mov     w4, #(1 << P_BIT_C)
    bic     w24, w24, w4
    orr     w24, w24, w3, lsl #P_BIT_C
    // A |= shifted
    orr     w20, w20, w2
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    5

// SLO abs (0x0F, 6 cy)

_op_slo_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lslo_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    lsl     w10, w1, #1
    lsr     w11, w10, #8
    and     w11, w11, #1
    and     w10, w10, #0xFF
    strb    w10, [x27, x0]
    mov     w4, #(1 << P_BIT_C)
    bic     w24, w24, w4
    orr     w24, w24, w11, lsl #P_BIT_C
    orr     w20, w20, w10
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    6
Lslo_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3g: RLA — ROL then AND. RMW, 5-7 cycles. zp + abs only.
// ============================================================================

// RLA zp (0x27, 5 cy)

_op_rla_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w2, w1, #1
    orr     w2, w2, w4
    lsr     w3, w2, #8
    and     w3, w3, #1
    and     w2, w2, #0xFF
    strb    w2, [x27, x0]
    mov     w5, #(1 << P_BIT_C)
    bic     w24, w24, w5
    orr     w24, w24, w3, lsl #P_BIT_C
    and     w20, w20, w2
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    5

// RLA abs (0x2F, 6 cy)

_op_rla_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lrla_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    lsl     w10, w1, #1
    orr     w10, w10, w4
    lsr     w11, w10, #8
    and     w11, w11, #1
    and     w10, w10, #0xFF
    strb    w10, [x27, x0]
    mov     w5, #(1 << P_BIT_C)
    bic     w24, w24, w5
    orr     w24, w24, w11, lsl #P_BIT_C
    and     w20, w20, w10
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    6
Lrla_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3h: SRE — LSR then EOR. RMW, 5-7 cycles. zp + abs only.
// ============================================================================

// SRE zp (0x47, 5 cy)

_op_sre_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    and     w3, w1, #1
    lsr     w2, w1, #1
    strb    w2, [x27, x0]
    mov     w4, #(1 << P_BIT_C)
    bic     w24, w24, w4
    orr     w24, w24, w3, lsl #P_BIT_C
    eor     w20, w20, w2
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    5

// SRE abs (0x4F, 6 cy)

_op_sre_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lsre_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    and     w11, w1, #1
    lsr     w10, w1, #1
    strb    w10, [x27, x0]
    mov     w4, #(1 << P_BIT_C)
    bic     w24, w24, w4
    orr     w24, w24, w11, lsl #P_BIT_C
    eor     w20, w20, w10
    and     w20, w20, #0xFF
    SET_NZ  w20
    NEXT    6
Lsre_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Phase 3i: RRA — ROR then ADC. RMW, 5-7 cycles. zp + abs only.
// ============================================================================

// RRA zp (0x67, 5 cy)

_op_rra_zp:
    FETCH_PC_BYTE w0
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    and     w3, w1, #1
    lsr     w2, w1, #1
    orr     w2, w2, w4, lsl #7
    strb    w2, [x27, x0]
    mov     w5, #(1 << P_BIT_C)
    bic     w24, w24, w5
    orr     w24, w24, w3, lsl #P_BIT_C
    ADC_CORE w2
    NEXT    5

// RRA abs (0x6F, 6 cy)

_op_rra_abs:
    FETCH_PC_BYTE w2
    FETCH_PC_BYTE w3
    orr     w0, w2, w3, lsl #8
    cmp     w0, #0x2000
    b.hs    Lrra_abs_bail
    and     w0, w0, #0x07FF
    ldrb    w1, [x27, x0]
    ubfx    w4, w24, #P_BIT_C, #1
    and     w11, w1, #1
    lsr     w10, w1, #1
    orr     w10, w10, w4, lsl #7
    strb    w10, [x27, x0]
    mov     w5, #(1 << P_BIT_C)
    bic     w24, w24, w5
    orr     w24, w24, w11, lsl #P_BIT_C
    ADC_CORE w10
    NEXT    6
Lrra_abs_bail:
    mov     x10, #3
    b       Lasm_bail_rewind

// ============================================================================
// Shared helpers — MMIO bridges + bail-rewind trampoline.
// ============================================================================

// Shared bail helper. On entry x10 holds the PC rewind distance (bytes
// already fetched for this instruction — opcode + operands). Jumps to
// the UNIMPL exit so Rust fallback re-executes the instruction.
Lasm_bail_rewind:
    sub     x19, x19, x10
    and     x19, x19, #0xFFFF
    mov     w10, wzr                    // opcode byte unused by caller
    b       Lexit_unimpl_opcode

// ============================================================================
// Lmmio_read — Rust nes_asm_bus_read_byte(bus_ptr, addr, cycles_consumed, cpu).
//
// 2026-04-26 fix: pass `cycles_consumed_before_this_instr` in w2
// (computed here from initial_budget - x25 — the live remainder, not
// yet decremented for the current instruction). The callback uses
// this to tick PPU for ALL CPU cycles consumed since the last MMIO
// callback (including preceding non-MMIO instructions in the same
// ASM batch), so $2002 / $2007 reads see a correctly-positioned PPU.
//
// Without this, SMB's wait-for-vblank polling loop never saw vblank
// inside an ASM batch and Mario fell through the floor (see
// `tests/asm_vs_slow_smb.rs` lockstep test). Caller sites still have
// a vestigial `mov w2, #N` setting the instruction's base cycles —
// harmless, overwritten here.
//
// In:  w9 = 16-bit address; x25 = live cycle remainder
// Out: w0 = value
// Preserves: x19-x28; NZCV + x0-x17 clobbered.
// ============================================================================
Lmmio_read:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    ldr     x0, [x26, #CPU_BUS_PTR]
    cbz     x0, Lmmio_read_nullptr
    and     w1, w9, #0xFFFF
    // Compute cycles consumed since ASM entry: initial_budget - x25.
    ldr     x4, [x26, #CPU_INITIAL_BUDGET]
    sub     x2, x4, x25
    mov     x3, x26                     // cpu_state_ptr
    bl      _nes_asm_bus_read_byte
    ldp     x29, x30, [sp], #16
    ret
Lmmio_read_nullptr:
    mov     w0, wzr
    ldp     x29, x30, [sp], #16
    ret

// ============================================================================
// Lmmio_write — Rust nes_asm_bus_write_byte(bus_ptr, addr, val, cyc, cpu).
// Same cumulative-cycles fix as Lmmio_read above: pass
// (initial_budget - x25), not per-instruction cycles.
// In:  w9 = addr, w10 = value, x25 = live cycle remainder.
// Preserves: x19-x28.
// ============================================================================
Lmmio_write:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    ldr     x0, [x26, #CPU_BUS_PTR]
    cbz     x0, Lmmio_write_nullptr
    and     w1, w9, #0xFFFF
    and     w2, w10, #0xFF
    // Compute cycles consumed since ASM entry. Overwrites caller's
    // per-call `mov w3, #N` instr_cycles slot.
    ldr     x4, [x26, #CPU_INITIAL_BUDGET]
    sub     x3, x4, x25
    mov     x4, x26                     // cpu_state_ptr
    bl      _nes_asm_bus_write_byte
    ldp     x29, x30, [sp], #16
    ret
Lmmio_write_nullptr:
    ldp     x29, x30, [sp], #16
    ret

// ----- Unimplemented-opcode trampoline -----
// On entry x0 holds the opcode byte. Store + exit.
Lunimpl:
    // x0 was the opcode byte loaded by NEXT.
    mov     x10, x0
    b       Lexit_unimpl_opcode

// ============================================================================
// Per-opcode dispatch table — 256 entries × 8 bytes = 2048 bytes.
//
// Layout: the label `_nes_asm_opcode_table` IS the table itself (2048
// bytes starting at that address), NOT a pointer to it. Rust's
// `static mut nes_asm_opcode_table: [usize; 256]` maps directly onto
// this 2048-byte region — so `install_opcode_table` writes to the
// right place.
//
// `Lopcode_table` is a local alias for the same address, used by the
// NEXT macro's dispatch sequence.
//
// Entries default to `Lunimpl`; Rust patches in handler addresses
// for the opcodes we've implemented via `install_opcode_table`.
// ============================================================================
.section __DATA,__data
.p2align 3
.globl _nes_asm_opcode_table
_nes_asm_opcode_table:
Lopcode_table:
.rept 256
    .quad Lunimpl
.endr

// Handler-address exports. Each is a single .quad pointer-sized
// value that Rust reads as `extern "C" static xxx: usize` — a
// single pointer. Used by `install_opcode_table` to populate the
// dispatch table.
.globl _nes_asm_op_nop
_nes_asm_op_nop: .quad _op_nop
.globl _nes_asm_op_lda_imm
_nes_asm_op_lda_imm: .quad _op_lda_imm
.globl _nes_asm_op_sta_zp
_nes_asm_op_sta_zp: .quad _op_sta_zp
.globl _nes_asm_op_jmp_abs
_nes_asm_op_jmp_abs: .quad _op_jmp_abs
.globl _nes_asm_op_bne
_nes_asm_op_bne: .quad _op_bne
.globl _nes_asm_op_unimpl
_nes_asm_op_unimpl: .quad Lunimpl

// Phase 1 handler exports.
.globl _nes_asm_op_ldx_imm
_nes_asm_op_ldx_imm: .quad _op_ldx_imm
.globl _nes_asm_op_ldy_imm
_nes_asm_op_ldy_imm: .quad _op_ldy_imm
.globl _nes_asm_op_lda_zp
_nes_asm_op_lda_zp: .quad _op_lda_zp
.globl _nes_asm_op_ldx_zp
_nes_asm_op_ldx_zp: .quad _op_ldx_zp
.globl _nes_asm_op_ldy_zp
_nes_asm_op_ldy_zp: .quad _op_ldy_zp
.globl _nes_asm_op_stx_zp
_nes_asm_op_stx_zp: .quad _op_stx_zp
.globl _nes_asm_op_sty_zp
_nes_asm_op_sty_zp: .quad _op_sty_zp
.globl _nes_asm_op_tax
_nes_asm_op_tax: .quad _op_tax
.globl _nes_asm_op_tay
_nes_asm_op_tay: .quad _op_tay
.globl _nes_asm_op_txa
_nes_asm_op_txa: .quad _op_txa
.globl _nes_asm_op_tya
_nes_asm_op_tya: .quad _op_tya
.globl _nes_asm_op_beq
_nes_asm_op_beq: .quad _op_beq
.globl _nes_asm_op_and_imm
_nes_asm_op_and_imm: .quad _op_and_imm
.globl _nes_asm_op_ora_imm
_nes_asm_op_ora_imm: .quad _op_ora_imm
.globl _nes_asm_op_eor_imm
_nes_asm_op_eor_imm: .quad _op_eor_imm
.globl _nes_asm_op_cmp_imm
_nes_asm_op_cmp_imm: .quad _op_cmp_imm
.globl _nes_asm_op_cpx_imm
_nes_asm_op_cpx_imm: .quad _op_cpx_imm
.globl _nes_asm_op_cpy_imm
_nes_asm_op_cpy_imm: .quad _op_cpy_imm

// Phase 2a handler exports.
.globl _nes_asm_op_clc
_nes_asm_op_clc: .quad _op_clc
.globl _nes_asm_op_sec
_nes_asm_op_sec: .quad _op_sec
.globl _nes_asm_op_cli
_nes_asm_op_cli: .quad _op_cli
.globl _nes_asm_op_sei
_nes_asm_op_sei: .quad _op_sei
.globl _nes_asm_op_cld
_nes_asm_op_cld: .quad _op_cld
.globl _nes_asm_op_sed
_nes_asm_op_sed: .quad _op_sed
.globl _nes_asm_op_clv
_nes_asm_op_clv: .quad _op_clv
.globl _nes_asm_op_inx
_nes_asm_op_inx: .quad _op_inx
.globl _nes_asm_op_iny
_nes_asm_op_iny: .quad _op_iny
.globl _nes_asm_op_dex
_nes_asm_op_dex: .quad _op_dex
.globl _nes_asm_op_dey
_nes_asm_op_dey: .quad _op_dey
.globl _nes_asm_op_inc_zp
_nes_asm_op_inc_zp: .quad _op_inc_zp
.globl _nes_asm_op_dec_zp
_nes_asm_op_dec_zp: .quad _op_dec_zp

// Phase 2b handler exports.
.globl _nes_asm_op_bpl
_nes_asm_op_bpl: .quad _op_bpl
.globl _nes_asm_op_bmi
_nes_asm_op_bmi: .quad _op_bmi
.globl _nes_asm_op_bvc
_nes_asm_op_bvc: .quad _op_bvc
.globl _nes_asm_op_bvs
_nes_asm_op_bvs: .quad _op_bvs
.globl _nes_asm_op_bcc
_nes_asm_op_bcc: .quad _op_bcc
.globl _nes_asm_op_bcs
_nes_asm_op_bcs: .quad _op_bcs
.globl _nes_asm_op_lda_abs
_nes_asm_op_lda_abs: .quad _op_lda_abs
.globl _nes_asm_op_ldx_abs
_nes_asm_op_ldx_abs: .quad _op_ldx_abs
.globl _nes_asm_op_ldy_abs
_nes_asm_op_ldy_abs: .quad _op_ldy_abs
.globl _nes_asm_op_and_zp
_nes_asm_op_and_zp: .quad _op_and_zp
.globl _nes_asm_op_ora_zp
_nes_asm_op_ora_zp: .quad _op_ora_zp
.globl _nes_asm_op_eor_zp
_nes_asm_op_eor_zp: .quad _op_eor_zp
.globl _nes_asm_op_cmp_zp
_nes_asm_op_cmp_zp: .quad _op_cmp_zp
.globl _nes_asm_op_bit_zp
_nes_asm_op_bit_zp: .quad _op_bit_zp
.globl _nes_asm_op_pha
_nes_asm_op_pha: .quad _op_pha
.globl _nes_asm_op_pla
_nes_asm_op_pla: .quad _op_pla
.globl _nes_asm_op_txs
_nes_asm_op_txs: .quad _op_txs
.globl _nes_asm_op_tsx
_nes_asm_op_tsx: .quad _op_tsx
.globl _nes_asm_op_php
_nes_asm_op_php: .quad _op_php
.globl _nes_asm_op_plp
_nes_asm_op_plp: .quad _op_plp

// Phase 2l — indexed-indirect (zp,X).
.globl _nes_asm_op_lda_ind_x
_nes_asm_op_lda_ind_x: .quad _op_lda_ind_x
.globl _nes_asm_op_sta_ind_x
_nes_asm_op_sta_ind_x: .quad _op_sta_ind_x
.globl _nes_asm_op_ora_ind_x
_nes_asm_op_ora_ind_x: .quad _op_ora_ind_x
.globl _nes_asm_op_and_ind_x
_nes_asm_op_and_ind_x: .quad _op_and_ind_x
.globl _nes_asm_op_eor_ind_x
_nes_asm_op_eor_ind_x: .quad _op_eor_ind_x
.globl _nes_asm_op_adc_ind_x
_nes_asm_op_adc_ind_x: .quad _op_adc_ind_x
.globl _nes_asm_op_sbc_ind_x
_nes_asm_op_sbc_ind_x: .quad _op_sbc_ind_x
.globl _nes_asm_op_cmp_ind_x
_nes_asm_op_cmp_ind_x: .quad _op_cmp_ind_x

// Phase 2c handler exports.
.globl _nes_asm_op_sta_abs
_nes_asm_op_sta_abs: .quad _op_sta_abs
.globl _nes_asm_op_stx_abs
_nes_asm_op_stx_abs: .quad _op_stx_abs
.globl _nes_asm_op_sty_abs
_nes_asm_op_sty_abs: .quad _op_sty_abs
.globl _nes_asm_op_jsr_abs
_nes_asm_op_jsr_abs: .quad _op_jsr_abs
.globl _nes_asm_op_rts
_nes_asm_op_rts: .quad _op_rts

// Phase 2d handler exports.
.globl _nes_asm_op_adc_imm
_nes_asm_op_adc_imm: .quad _op_adc_imm
.globl _nes_asm_op_sbc_imm
_nes_asm_op_sbc_imm: .quad _op_sbc_imm
.globl _nes_asm_op_adc_zp
_nes_asm_op_adc_zp: .quad _op_adc_zp
.globl _nes_asm_op_sbc_zp
_nes_asm_op_sbc_zp: .quad _op_sbc_zp
.globl _nes_asm_op_cmp_abs
_nes_asm_op_cmp_abs: .quad _op_cmp_abs
.globl _nes_asm_op_cpx_abs
_nes_asm_op_cpx_abs: .quad _op_cpx_abs
.globl _nes_asm_op_cpy_abs
_nes_asm_op_cpy_abs: .quad _op_cpy_abs
.globl _nes_asm_op_lda_zpx
_nes_asm_op_lda_zpx: .quad _op_lda_zpx
.globl _nes_asm_op_sta_zpx
_nes_asm_op_sta_zpx: .quad _op_sta_zpx

// Phase 2e handler exports.
.globl _nes_asm_op_sta_abs_x
_nes_asm_op_sta_abs_x: .quad _op_sta_abs_x
.globl _nes_asm_op_sta_abs_y
_nes_asm_op_sta_abs_y: .quad _op_sta_abs_y
.globl _nes_asm_op_sta_ind_y
_nes_asm_op_sta_ind_y: .quad _op_sta_ind_y
.globl _nes_asm_op_asl_a
_nes_asm_op_asl_a: .quad _op_asl_a
.globl _nes_asm_op_lsr_a
_nes_asm_op_lsr_a: .quad _op_lsr_a
.globl _nes_asm_op_rol_a
_nes_asm_op_rol_a: .quad _op_rol_a
.globl _nes_asm_op_ror_a
_nes_asm_op_ror_a: .quad _op_ror_a

// Phase 2f handler exports.
.globl _nes_asm_op_lda_abs_x
_nes_asm_op_lda_abs_x: .quad _op_lda_abs_x
.globl _nes_asm_op_lda_abs_y
_nes_asm_op_lda_abs_y: .quad _op_lda_abs_y
.globl _nes_asm_op_ldx_abs_y
_nes_asm_op_ldx_abs_y: .quad _op_ldx_abs_y
.globl _nes_asm_op_ldy_abs_x
_nes_asm_op_ldy_abs_x: .quad _op_ldy_abs_x
.globl _nes_asm_op_lda_ind_y
_nes_asm_op_lda_ind_y: .quad _op_lda_ind_y
.globl _nes_asm_op_and_abs
_nes_asm_op_and_abs: .quad _op_and_abs
.globl _nes_asm_op_ora_abs
_nes_asm_op_ora_abs: .quad _op_ora_abs
.globl _nes_asm_op_eor_abs
_nes_asm_op_eor_abs: .quad _op_eor_abs
.globl _nes_asm_op_adc_abs
_nes_asm_op_adc_abs: .quad _op_adc_abs
.globl _nes_asm_op_sbc_abs
_nes_asm_op_sbc_abs: .quad _op_sbc_abs
.globl _nes_asm_op_asl_zp
_nes_asm_op_asl_zp: .quad _op_asl_zp
.globl _nes_asm_op_lsr_zp
_nes_asm_op_lsr_zp: .quad _op_lsr_zp
.globl _nes_asm_op_rol_zp
_nes_asm_op_rol_zp: .quad _op_rol_zp
.globl _nes_asm_op_ror_zp
_nes_asm_op_ror_zp: .quad _op_ror_zp

// Phase 2g handler exports.
.globl _nes_asm_op_jmp_ind
_nes_asm_op_jmp_ind: .quad _op_jmp_ind
.globl _nes_asm_op_inc_abs
_nes_asm_op_inc_abs: .quad _op_inc_abs
.globl _nes_asm_op_dec_abs
_nes_asm_op_dec_abs: .quad _op_dec_abs
.globl _nes_asm_op_inc_abs_x
_nes_asm_op_inc_abs_x: .quad _op_inc_abs_x
.globl _nes_asm_op_dec_abs_x
_nes_asm_op_dec_abs_x: .quad _op_dec_abs_x

// Phase 2h — zp indexed Y/X variants.
.globl _nes_asm_op_ldx_zpy
_nes_asm_op_ldx_zpy: .quad _op_ldx_zpy
.globl _nes_asm_op_stx_zpy
_nes_asm_op_stx_zpy: .quad _op_stx_zpy
.globl _nes_asm_op_ldy_zpx
_nes_asm_op_ldy_zpx: .quad _op_ldy_zpx
.globl _nes_asm_op_sty_zpx
_nes_asm_op_sty_zpx: .quad _op_sty_zpx
.globl _nes_asm_op_brk
_nes_asm_op_brk: .quad _op_brk
.globl _nes_asm_op_rti
_nes_asm_op_rti: .quad _op_rti

// Phase 2i/2j exports.
.globl _nes_asm_op_ora_ind_y
_nes_asm_op_ora_ind_y: .quad _op_ora_ind_y
.globl _nes_asm_op_and_ind_y
_nes_asm_op_and_ind_y: .quad _op_and_ind_y
.globl _nes_asm_op_eor_ind_y
_nes_asm_op_eor_ind_y: .quad _op_eor_ind_y
.globl _nes_asm_op_adc_ind_y
_nes_asm_op_adc_ind_y: .quad _op_adc_ind_y
.globl _nes_asm_op_sbc_ind_y
_nes_asm_op_sbc_ind_y: .quad _op_sbc_ind_y
.globl _nes_asm_op_cmp_ind_y
_nes_asm_op_cmp_ind_y: .quad _op_cmp_ind_y
.globl _nes_asm_op_and_abs_x
_nes_asm_op_and_abs_x: .quad _op_and_abs_x
.globl _nes_asm_op_ora_abs_x
_nes_asm_op_ora_abs_x: .quad _op_ora_abs_x
.globl _nes_asm_op_eor_abs_x
_nes_asm_op_eor_abs_x: .quad _op_eor_abs_x
.globl _nes_asm_op_adc_abs_x
_nes_asm_op_adc_abs_x: .quad _op_adc_abs_x
.globl _nes_asm_op_sbc_abs_x
_nes_asm_op_sbc_abs_x: .quad _op_sbc_abs_x
.globl _nes_asm_op_cmp_abs_x
_nes_asm_op_cmp_abs_x: .quad _op_cmp_abs_x
.globl _nes_asm_op_and_abs_y
_nes_asm_op_and_abs_y: .quad _op_and_abs_y
.globl _nes_asm_op_ora_abs_y
_nes_asm_op_ora_abs_y: .quad _op_ora_abs_y
.globl _nes_asm_op_eor_abs_y
_nes_asm_op_eor_abs_y: .quad _op_eor_abs_y
.globl _nes_asm_op_adc_abs_y
_nes_asm_op_adc_abs_y: .quad _op_adc_abs_y
.globl _nes_asm_op_sbc_abs_y
_nes_asm_op_sbc_abs_y: .quad _op_sbc_abs_y
.globl _nes_asm_op_cmp_abs_y
_nes_asm_op_cmp_abs_y: .quad _op_cmp_abs_y
.globl _nes_asm_op_bit_abs
_nes_asm_op_bit_abs: .quad _op_bit_abs

// Phase 2k — memory shifts + indexed inc/dec.
.globl _nes_asm_op_asl_abs
_nes_asm_op_asl_abs: .quad _op_asl_abs
.globl _nes_asm_op_lsr_abs
_nes_asm_op_lsr_abs: .quad _op_lsr_abs
.globl _nes_asm_op_rol_abs
_nes_asm_op_rol_abs: .quad _op_rol_abs
.globl _nes_asm_op_ror_abs
_nes_asm_op_ror_abs: .quad _op_ror_abs
.globl _nes_asm_op_inc_zpx
_nes_asm_op_inc_zpx: .quad _op_inc_zpx
.globl _nes_asm_op_dec_zpx
_nes_asm_op_dec_zpx: .quad _op_dec_zpx

// Phase 2m handler exports — remaining 16 official 6502 opcodes.
.globl _nes_asm_op_ora_zpx
_nes_asm_op_ora_zpx: .quad _op_ora_zpx
.globl _nes_asm_op_and_zpx
_nes_asm_op_and_zpx: .quad _op_and_zpx
.globl _nes_asm_op_eor_zpx
_nes_asm_op_eor_zpx: .quad _op_eor_zpx
.globl _nes_asm_op_adc_zpx
_nes_asm_op_adc_zpx: .quad _op_adc_zpx
.globl _nes_asm_op_cmp_zpx
_nes_asm_op_cmp_zpx: .quad _op_cmp_zpx
.globl _nes_asm_op_sbc_zpx
_nes_asm_op_sbc_zpx: .quad _op_sbc_zpx
.globl _nes_asm_op_cpx_zp
_nes_asm_op_cpx_zp: .quad _op_cpx_zp
.globl _nes_asm_op_cpy_zp
_nes_asm_op_cpy_zp: .quad _op_cpy_zp
.globl _nes_asm_op_asl_zpx
_nes_asm_op_asl_zpx: .quad _op_asl_zpx
.globl _nes_asm_op_lsr_zpx
_nes_asm_op_lsr_zpx: .quad _op_lsr_zpx
.globl _nes_asm_op_rol_zpx
_nes_asm_op_rol_zpx: .quad _op_rol_zpx
.globl _nes_asm_op_ror_zpx
_nes_asm_op_ror_zpx: .quad _op_ror_zpx
.globl _nes_asm_op_asl_abs_x
_nes_asm_op_asl_abs_x: .quad _op_asl_abs_x
.globl _nes_asm_op_lsr_abs_x
_nes_asm_op_lsr_abs_x: .quad _op_lsr_abs_x
.globl _nes_asm_op_rol_abs_x
_nes_asm_op_rol_abs_x: .quad _op_rol_abs_x
.globl _nes_asm_op_ror_abs_x
_nes_asm_op_ror_abs_x: .quad _op_ror_abs_x

// Phase 3 handler exports (stable illegals).
.globl _nes_asm_op_nop_impl2
_nes_asm_op_nop_impl2: .quad _op_nop_impl2
.globl _nes_asm_op_nop_imm
_nes_asm_op_nop_imm: .quad _op_nop_imm
.globl _nes_asm_op_nop_zp
_nes_asm_op_nop_zp: .quad _op_nop_zp
.globl _nes_asm_op_nop_zpx
_nes_asm_op_nop_zpx: .quad _op_nop_zpx
.globl _nes_asm_op_nop_abs
_nes_asm_op_nop_abs: .quad _op_nop_abs
.globl _nes_asm_op_nop_abs_x
_nes_asm_op_nop_abs_x: .quad _op_nop_abs_x
.globl _nes_asm_op_lax_zp
_nes_asm_op_lax_zp: .quad _op_lax_zp
.globl _nes_asm_op_lax_zpy
_nes_asm_op_lax_zpy: .quad _op_lax_zpy
.globl _nes_asm_op_lax_abs
_nes_asm_op_lax_abs: .quad _op_lax_abs
.globl _nes_asm_op_lax_abs_y
_nes_asm_op_lax_abs_y: .quad _op_lax_abs_y
.globl _nes_asm_op_sax_zp
_nes_asm_op_sax_zp: .quad _op_sax_zp
.globl _nes_asm_op_sax_zpy
_nes_asm_op_sax_zpy: .quad _op_sax_zpy
.globl _nes_asm_op_sax_abs
_nes_asm_op_sax_abs: .quad _op_sax_abs
.globl _nes_asm_op_dcp_zp
_nes_asm_op_dcp_zp: .quad _op_dcp_zp
.globl _nes_asm_op_dcp_abs
_nes_asm_op_dcp_abs: .quad _op_dcp_abs
.globl _nes_asm_op_isc_zp
_nes_asm_op_isc_zp: .quad _op_isc_zp
.globl _nes_asm_op_isc_abs
_nes_asm_op_isc_abs: .quad _op_isc_abs
.globl _nes_asm_op_slo_zp
_nes_asm_op_slo_zp: .quad _op_slo_zp
.globl _nes_asm_op_slo_abs
_nes_asm_op_slo_abs: .quad _op_slo_abs
.globl _nes_asm_op_rla_zp
_nes_asm_op_rla_zp: .quad _op_rla_zp
.globl _nes_asm_op_rla_abs
_nes_asm_op_rla_abs: .quad _op_rla_abs
.globl _nes_asm_op_sre_zp
_nes_asm_op_sre_zp: .quad _op_sre_zp
.globl _nes_asm_op_sre_abs
_nes_asm_op_sre_abs: .quad _op_sre_abs
.globl _nes_asm_op_rra_zp
_nes_asm_op_rra_zp: .quad _op_rra_zp
.globl _nes_asm_op_rra_abs
_nes_asm_op_rra_abs: .quad _op_rra_abs
