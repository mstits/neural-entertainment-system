//! AArch64 assembly 6502 CPU core — Phase 0 skeleton.
//!
//! See `docs/proposals/aarch64_cpu_asm.md` for the design and the
//! register allocation + dispatch protocol. The ASM lives in
//! `cpu_asm.s`; this module exposes the FFI entry point + owns the
//! shared `AsmCpuState` struct that the ASM reads/writes.
//!
//! Only compiled on `target_arch = "aarch64"` + `feature = "asm_cpu"`.

#![cfg(all(target_arch = "aarch64", feature = "asm_cpu"))]

use core::arch::global_asm;

// Link the assembly unit into the crate. The `.s` file uses Mach-O
// section directives + underscore-prefixed globals so this works on
// Apple Silicon macOS specifically.
global_asm!(include_str!("cpu_asm.s"));

/// State shared between the ASM core and Rust. Layout MUST match the
/// offsets declared at the top of `cpu_asm.s` (CPU_* constants).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AsmCpuState {
    pub pc: u16,             // offset  0
    pub a: u8,               // offset  2
    pub x: u8,               // offset  3
    pub y: u8,               // offset  4
    pub sp: u8,              // offset  5
    pub p: u8,               // offset  6 — packed N V - - D I Z C
    pub _pad_7: u8,          // offset  7 — pad so cycles aligns at 8
    pub cycles: u64,         // offset  8 — absolute cycle counter
    pub unimpl_opcode: u8,   // offset 16 — filled on EXIT_UNIMPL
    pub _pad_17: u8,         // offset 17
    pub mmio_addr: u16,      // offset 18 — filled on EXIT_MMIO_*
    pub mmio_value: u8,      // offset 20 — filled on EXIT_MMIO_WRITE
    pub _pad_21_23: [u8; 3], // offset 21..24 — pad to 8-byte multiple
    pub bus_ptr: u64,        // offset 24 — raw SystemBus ptr for MMIO callbacks
    pub initial_budget: i64, // offset 32 — stashed by entry prologue; exit uses
                             //             it to compute cycles_consumed since
                             //             x3 is caller-saved and clobbered by
                             //             any MMIO callback (bl _nes_asm_bus_*).
    pub cycles_ticked_in_callback: u32, // offset 40 — running count of CPU
                                        //             cycles the MMIO callback
                                        //             has already ticked PPU/APU
                                        //             for. Nes::step subtracts
                                        //             this from its catch-up loop.
    pub pending_exit: u8,    // offset 44 — set by MMIO callback when an
                             //             interrupt (NMI rising edge or IRQ)
                             //             fires inside an ASM batch. NEXT
                             //             macro polls this and exits to
                             //             Lexit_cycles so the Rust slow path
                             //             services the interrupt at the next
                             //             instruction boundary, matching real
                             //             6502 NMI/IRQ delivery timing.
    pub _pad_45_47: [u8; 3], // offset 45..48 — pad to 8-byte alignment
}

/// Exit reason returned by `nes_cpu_run_block`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AsmExit {
    Cycles = 0,
    Unimpl = 1,
    MmioRead = 2,
    MmioWrite = 3,
}

unsafe extern "C" {
    /// Run the ASM 6502 core until the cycle budget is exhausted, an
    /// unimplemented opcode is hit, or an MMIO access is required.
    ///
    /// # Safety
    /// - `cpu` must point to an initialized `AsmCpuState`.
    /// - `ram` must point to at least 2048 writable bytes (NES WRAM).
    /// - `prg` is nullable. When non-null, it must point to at least
    ///   32768 bytes of readable memory (NROM PRG ROM).
    pub fn nes_cpu_run_block(
        cpu: *mut AsmCpuState,
        ram: *mut u8,
        prg: *const u8,
        cycles_target: i64,
    ) -> u32;
}

// ============================================================================
// Opcode-table installer. The ASM file declares a default 256-entry
// table populated with `Lunimpl` via `.rept 256 / .quad Lunimpl /
// .endr`. At startup we overwrite the slots for the opcodes we do
// have handlers for.
// ============================================================================

unsafe extern "C" {
    static nes_asm_op_nop: usize;
    static nes_asm_op_lda_imm: usize;
    static nes_asm_op_sta_zp: usize;
    static nes_asm_op_jmp_abs: usize;
    static nes_asm_op_bne: usize;
    static nes_asm_op_unimpl: usize;
    // Phase 1.
    static nes_asm_op_ldx_imm: usize;
    static nes_asm_op_ldy_imm: usize;
    static nes_asm_op_lda_zp: usize;
    static nes_asm_op_ldx_zp: usize;
    static nes_asm_op_ldy_zp: usize;
    static nes_asm_op_stx_zp: usize;
    static nes_asm_op_sty_zp: usize;
    static nes_asm_op_tax: usize;
    static nes_asm_op_tay: usize;
    static nes_asm_op_txa: usize;
    static nes_asm_op_tya: usize;
    static nes_asm_op_beq: usize;
    static nes_asm_op_and_imm: usize;
    static nes_asm_op_ora_imm: usize;
    static nes_asm_op_eor_imm: usize;
    static nes_asm_op_cmp_imm: usize;
    static nes_asm_op_cpx_imm: usize;
    static nes_asm_op_cpy_imm: usize;
    // Phase 2a.
    static nes_asm_op_clc: usize;
    static nes_asm_op_sec: usize;
    static nes_asm_op_cli: usize;
    static nes_asm_op_sei: usize;
    static nes_asm_op_cld: usize;
    static nes_asm_op_sed: usize;
    static nes_asm_op_clv: usize;
    static nes_asm_op_inx: usize;
    static nes_asm_op_iny: usize;
    static nes_asm_op_dex: usize;
    static nes_asm_op_dey: usize;
    static nes_asm_op_inc_zp: usize;
    static nes_asm_op_dec_zp: usize;
    // Phase 2b.
    static nes_asm_op_bpl: usize;
    static nes_asm_op_bmi: usize;
    static nes_asm_op_bvc: usize;
    static nes_asm_op_bvs: usize;
    static nes_asm_op_bcc: usize;
    static nes_asm_op_bcs: usize;
    static nes_asm_op_lda_abs: usize;
    static nes_asm_op_ldx_abs: usize;
    static nes_asm_op_ldy_abs: usize;
    static nes_asm_op_and_zp: usize;
    static nes_asm_op_ora_zp: usize;
    static nes_asm_op_eor_zp: usize;
    static nes_asm_op_cmp_zp: usize;
    static nes_asm_op_bit_zp: usize;
    static nes_asm_op_pha: usize;
    static nes_asm_op_pla: usize;
    // Phase 2c.
    static nes_asm_op_sta_abs: usize;
    static nes_asm_op_stx_abs: usize;
    static nes_asm_op_sty_abs: usize;
    static nes_asm_op_jsr_abs: usize;
    static nes_asm_op_rts: usize;
    // Phase 2d.
    static nes_asm_op_adc_imm: usize;
    static nes_asm_op_sbc_imm: usize;
    static nes_asm_op_adc_zp: usize;
    static nes_asm_op_sbc_zp: usize;
    static nes_asm_op_cmp_abs: usize;
    static nes_asm_op_cpx_abs: usize;
    static nes_asm_op_cpy_abs: usize;
    static nes_asm_op_lda_zpx: usize;
    static nes_asm_op_sta_zpx: usize;
    // Phase 2e.
    static nes_asm_op_sta_abs_x: usize;
    static nes_asm_op_sta_abs_y: usize;
    static nes_asm_op_sta_ind_y: usize;
    static nes_asm_op_asl_a: usize;
    static nes_asm_op_lsr_a: usize;
    static nes_asm_op_rol_a: usize;
    static nes_asm_op_ror_a: usize;
    // Phase 2f.
    static nes_asm_op_lda_abs_x: usize;
    static nes_asm_op_lda_abs_y: usize;
    static nes_asm_op_ldx_abs_y: usize;
    static nes_asm_op_ldy_abs_x: usize;
    static nes_asm_op_lda_ind_y: usize;
    static nes_asm_op_and_abs: usize;
    static nes_asm_op_ora_abs: usize;
    static nes_asm_op_eor_abs: usize;
    static nes_asm_op_adc_abs: usize;
    static nes_asm_op_sbc_abs: usize;
    static nes_asm_op_asl_zp: usize;
    static nes_asm_op_lsr_zp: usize;
    static nes_asm_op_rol_zp: usize;
    static nes_asm_op_ror_zp: usize;
    // Phase 2g.
    static nes_asm_op_jmp_ind: usize;
    static nes_asm_op_inc_abs: usize;
    static nes_asm_op_dec_abs: usize;
    static nes_asm_op_inc_abs_x: usize;
    static nes_asm_op_dec_abs_x: usize;
    // Phase 2h.
    static nes_asm_op_ldx_zpy: usize;
    static nes_asm_op_stx_zpy: usize;
    static nes_asm_op_ldy_zpx: usize;
    static nes_asm_op_sty_zpx: usize;
    static nes_asm_op_brk: usize;
    static nes_asm_op_rti: usize;
    // Phase 2i/2j.
    static nes_asm_op_ora_ind_y: usize;
    static nes_asm_op_and_ind_y: usize;
    static nes_asm_op_eor_ind_y: usize;
    static nes_asm_op_adc_ind_y: usize;
    static nes_asm_op_sbc_ind_y: usize;
    static nes_asm_op_cmp_ind_y: usize;
    static nes_asm_op_and_abs_x: usize;
    static nes_asm_op_ora_abs_x: usize;
    static nes_asm_op_eor_abs_x: usize;
    static nes_asm_op_adc_abs_x: usize;
    static nes_asm_op_sbc_abs_x: usize;
    static nes_asm_op_cmp_abs_x: usize;
    static nes_asm_op_and_abs_y: usize;
    static nes_asm_op_ora_abs_y: usize;
    static nes_asm_op_eor_abs_y: usize;
    static nes_asm_op_adc_abs_y: usize;
    static nes_asm_op_sbc_abs_y: usize;
    static nes_asm_op_cmp_abs_y: usize;
    static nes_asm_op_bit_abs: usize;
    static nes_asm_op_asl_abs: usize;
    static nes_asm_op_lsr_abs: usize;
    static nes_asm_op_rol_abs: usize;
    static nes_asm_op_ror_abs: usize;
    static nes_asm_op_inc_zpx: usize;
    static nes_asm_op_dec_zpx: usize;
    static nes_asm_op_txs: usize;
    static nes_asm_op_tsx: usize;
    static nes_asm_op_php: usize;
    static nes_asm_op_plp: usize;
    static nes_asm_op_lda_ind_x: usize;
    static nes_asm_op_sta_ind_x: usize;
    static nes_asm_op_ora_ind_x: usize;
    static nes_asm_op_and_ind_x: usize;
    static nes_asm_op_eor_ind_x: usize;
    static nes_asm_op_adc_ind_x: usize;
    static nes_asm_op_sbc_ind_x: usize;
    static nes_asm_op_cmp_ind_x: usize;
    // Phase 2m — official zp,X + abs,X + CPx zp handlers.
    static nes_asm_op_ora_zpx: usize;
    static nes_asm_op_and_zpx: usize;
    static nes_asm_op_eor_zpx: usize;
    static nes_asm_op_adc_zpx: usize;
    static nes_asm_op_cmp_zpx: usize;
    static nes_asm_op_sbc_zpx: usize;
    static nes_asm_op_cpx_zp: usize;
    static nes_asm_op_cpy_zp: usize;
    static nes_asm_op_asl_zpx: usize;
    static nes_asm_op_lsr_zpx: usize;
    static nes_asm_op_rol_zpx: usize;
    static nes_asm_op_ror_zpx: usize;
    static nes_asm_op_asl_abs_x: usize;
    static nes_asm_op_lsr_abs_x: usize;
    static nes_asm_op_rol_abs_x: usize;
    static nes_asm_op_ror_abs_x: usize;
    // Phase 3 — stable illegal opcodes.
    static nes_asm_op_nop_impl2: usize;
    static nes_asm_op_nop_imm: usize;
    static nes_asm_op_nop_zp: usize;
    static nes_asm_op_nop_zpx: usize;
    static nes_asm_op_nop_abs: usize;
    static nes_asm_op_nop_abs_x: usize;
    static nes_asm_op_lax_zp: usize;
    static nes_asm_op_lax_zpy: usize;
    static nes_asm_op_lax_abs: usize;
    static nes_asm_op_lax_abs_y: usize;
    static nes_asm_op_sax_zp: usize;
    static nes_asm_op_sax_zpy: usize;
    static nes_asm_op_sax_abs: usize;
    static nes_asm_op_dcp_zp: usize;
    static nes_asm_op_dcp_abs: usize;
    static nes_asm_op_isc_zp: usize;
    static nes_asm_op_isc_abs: usize;
    static nes_asm_op_slo_zp: usize;
    static nes_asm_op_slo_abs: usize;
    static nes_asm_op_rla_zp: usize;
    static nes_asm_op_rla_abs: usize;
    static nes_asm_op_sre_zp: usize;
    static nes_asm_op_sre_abs: usize;
    static nes_asm_op_rra_zp: usize;
    static nes_asm_op_rra_abs: usize;
    static mut nes_asm_opcode_table: [usize; 256];
}

/// Install the Phase 0 opcode handlers into the dispatch table.
/// Idempotent. Call once per process before any `nes_cpu_run_block`
/// invocation.
///
/// Uses raw-pointer writes to avoid the Rust-2024
/// `static_mut_refs` lint — the ASM side treats the table as
/// `.quad` data that the Rust side patches at runtime. Idempotent +
/// single-threaded install is fine for this use (PoC harness);
/// production use would gate with a `OnceLock`.
pub fn install_opcode_table() {
    unsafe {
        let base = &raw mut nes_asm_opcode_table as *mut usize;
        // Reset the whole table to Lunimpl first — cheap, and makes
        // the install order self-evident.
        for i in 0..256 {
            base.add(i).write(nes_asm_op_unimpl);
        }
        base.add(0xEA).write(nes_asm_op_nop);
        base.add(0xA9).write(nes_asm_op_lda_imm);
        base.add(0x85).write(nes_asm_op_sta_zp);
        base.add(0x4C).write(nes_asm_op_jmp_abs);
        base.add(0xD0).write(nes_asm_op_bne);
        // Phase 1.
        base.add(0xA2).write(nes_asm_op_ldx_imm);
        base.add(0xA0).write(nes_asm_op_ldy_imm);
        base.add(0xA5).write(nes_asm_op_lda_zp);
        base.add(0xA6).write(nes_asm_op_ldx_zp);
        base.add(0xA4).write(nes_asm_op_ldy_zp);
        base.add(0x86).write(nes_asm_op_stx_zp);
        base.add(0x84).write(nes_asm_op_sty_zp);
        base.add(0xAA).write(nes_asm_op_tax);
        base.add(0xA8).write(nes_asm_op_tay);
        base.add(0x8A).write(nes_asm_op_txa);
        base.add(0x98).write(nes_asm_op_tya);
        base.add(0xF0).write(nes_asm_op_beq);
        base.add(0x29).write(nes_asm_op_and_imm);
        base.add(0x09).write(nes_asm_op_ora_imm);
        base.add(0x49).write(nes_asm_op_eor_imm);
        base.add(0xC9).write(nes_asm_op_cmp_imm);
        base.add(0xE0).write(nes_asm_op_cpx_imm);
        base.add(0xC0).write(nes_asm_op_cpy_imm);
        // Phase 2a.
        base.add(0x18).write(nes_asm_op_clc);
        base.add(0x38).write(nes_asm_op_sec);
        base.add(0x58).write(nes_asm_op_cli);
        base.add(0x78).write(nes_asm_op_sei);
        base.add(0xD8).write(nes_asm_op_cld);
        base.add(0xF8).write(nes_asm_op_sed);
        base.add(0xB8).write(nes_asm_op_clv);
        base.add(0xE8).write(nes_asm_op_inx);
        base.add(0xC8).write(nes_asm_op_iny);
        base.add(0xCA).write(nes_asm_op_dex);
        base.add(0x88).write(nes_asm_op_dey);
        base.add(0xE6).write(nes_asm_op_inc_zp);
        base.add(0xC6).write(nes_asm_op_dec_zp);
        // Phase 2b.
        base.add(0x10).write(nes_asm_op_bpl);
        base.add(0x30).write(nes_asm_op_bmi);
        base.add(0x50).write(nes_asm_op_bvc);
        base.add(0x70).write(nes_asm_op_bvs);
        base.add(0x90).write(nes_asm_op_bcc);
        base.add(0xB0).write(nes_asm_op_bcs);
        base.add(0xAD).write(nes_asm_op_lda_abs);
        base.add(0xAE).write(nes_asm_op_ldx_abs);
        base.add(0xAC).write(nes_asm_op_ldy_abs);
        base.add(0x25).write(nes_asm_op_and_zp);
        base.add(0x05).write(nes_asm_op_ora_zp);
        base.add(0x45).write(nes_asm_op_eor_zp);
        base.add(0xC5).write(nes_asm_op_cmp_zp);
        base.add(0x24).write(nes_asm_op_bit_zp);
        base.add(0x48).write(nes_asm_op_pha);
        base.add(0x68).write(nes_asm_op_pla);
        // Phase 2c.
        base.add(0x8D).write(nes_asm_op_sta_abs);
        base.add(0x8E).write(nes_asm_op_stx_abs);
        base.add(0x8C).write(nes_asm_op_sty_abs);
        base.add(0x20).write(nes_asm_op_jsr_abs);
        base.add(0x60).write(nes_asm_op_rts);
        // Phase 2d.
        base.add(0x69).write(nes_asm_op_adc_imm);
        base.add(0xE9).write(nes_asm_op_sbc_imm);
        base.add(0x65).write(nes_asm_op_adc_zp);
        base.add(0xE5).write(nes_asm_op_sbc_zp);
        base.add(0xCD).write(nes_asm_op_cmp_abs);
        base.add(0xEC).write(nes_asm_op_cpx_abs);
        base.add(0xCC).write(nes_asm_op_cpy_abs);
        base.add(0xB5).write(nes_asm_op_lda_zpx);
        base.add(0x95).write(nes_asm_op_sta_zpx);
        // Phase 2e.
        base.add(0x9D).write(nes_asm_op_sta_abs_x);
        base.add(0x99).write(nes_asm_op_sta_abs_y);
        base.add(0x91).write(nes_asm_op_sta_ind_y);
        base.add(0x0A).write(nes_asm_op_asl_a);
        base.add(0x4A).write(nes_asm_op_lsr_a);
        base.add(0x2A).write(nes_asm_op_rol_a);
        base.add(0x6A).write(nes_asm_op_ror_a);
        // Phase 2f.
        base.add(0xBD).write(nes_asm_op_lda_abs_x);
        base.add(0xB9).write(nes_asm_op_lda_abs_y);
        base.add(0xBE).write(nes_asm_op_ldx_abs_y);
        base.add(0xBC).write(nes_asm_op_ldy_abs_x);
        base.add(0xB1).write(nes_asm_op_lda_ind_y);
        base.add(0x2D).write(nes_asm_op_and_abs);
        base.add(0x0D).write(nes_asm_op_ora_abs);
        base.add(0x4D).write(nes_asm_op_eor_abs);
        base.add(0x6D).write(nes_asm_op_adc_abs);
        base.add(0xED).write(nes_asm_op_sbc_abs);
        base.add(0x06).write(nes_asm_op_asl_zp);
        base.add(0x46).write(nes_asm_op_lsr_zp);
        base.add(0x26).write(nes_asm_op_rol_zp);
        base.add(0x66).write(nes_asm_op_ror_zp);
        // Phase 2g.
        base.add(0x6C).write(nes_asm_op_jmp_ind);
        base.add(0xEE).write(nes_asm_op_inc_abs);
        base.add(0xCE).write(nes_asm_op_dec_abs);
        base.add(0xFE).write(nes_asm_op_inc_abs_x);
        base.add(0xDE).write(nes_asm_op_dec_abs_x);
        // Phase 2h.
        base.add(0xB6).write(nes_asm_op_ldx_zpy);
        base.add(0x96).write(nes_asm_op_stx_zpy);
        base.add(0xB4).write(nes_asm_op_ldy_zpx);
        base.add(0x94).write(nes_asm_op_sty_zpx);
        base.add(0x00).write(nes_asm_op_brk);
        base.add(0x40).write(nes_asm_op_rti);
        // Phase 2i — (zp),Y non-store variants.
        base.add(0x11).write(nes_asm_op_ora_ind_y);
        base.add(0x31).write(nes_asm_op_and_ind_y);
        base.add(0x51).write(nes_asm_op_eor_ind_y);
        base.add(0x71).write(nes_asm_op_adc_ind_y);
        base.add(0xF1).write(nes_asm_op_sbc_ind_y);
        base.add(0xD1).write(nes_asm_op_cmp_ind_y);
        // Phase 2j — abs,X non-store variants.
        base.add(0x3D).write(nes_asm_op_and_abs_x);
        base.add(0x1D).write(nes_asm_op_ora_abs_x);
        base.add(0x5D).write(nes_asm_op_eor_abs_x);
        base.add(0x7D).write(nes_asm_op_adc_abs_x);
        base.add(0xFD).write(nes_asm_op_sbc_abs_x);
        base.add(0xDD).write(nes_asm_op_cmp_abs_x);
        // Phase 2j — abs,Y non-store variants.
        base.add(0x39).write(nes_asm_op_and_abs_y);
        base.add(0x19).write(nes_asm_op_ora_abs_y);
        base.add(0x59).write(nes_asm_op_eor_abs_y);
        base.add(0x79).write(nes_asm_op_adc_abs_y);
        base.add(0xF9).write(nes_asm_op_sbc_abs_y);
        base.add(0xD9).write(nes_asm_op_cmp_abs_y);
        base.add(0x2C).write(nes_asm_op_bit_abs);
        // Phase 2k.
        base.add(0x0E).write(nes_asm_op_asl_abs);
        base.add(0x4E).write(nes_asm_op_lsr_abs);
        base.add(0x2E).write(nes_asm_op_rol_abs);
        base.add(0x6E).write(nes_asm_op_ror_abs);
        base.add(0xF6).write(nes_asm_op_inc_zpx);
        base.add(0xD6).write(nes_asm_op_dec_zpx);
        base.add(0x9A).write(nes_asm_op_txs);
        base.add(0xBA).write(nes_asm_op_tsx);
        base.add(0x08).write(nes_asm_op_php);
        base.add(0x28).write(nes_asm_op_plp);
        // Phase 2l — (zp,X).
        base.add(0xA1).write(nes_asm_op_lda_ind_x);
        base.add(0x81).write(nes_asm_op_sta_ind_x);
        base.add(0x01).write(nes_asm_op_ora_ind_x);
        base.add(0x21).write(nes_asm_op_and_ind_x);
        base.add(0x41).write(nes_asm_op_eor_ind_x);
        base.add(0x61).write(nes_asm_op_adc_ind_x);
        base.add(0xE1).write(nes_asm_op_sbc_ind_x);
        base.add(0xC1).write(nes_asm_op_cmp_ind_x);
        // Phase 2m — remaining 16 official opcodes (zp,X + abs,X
        // ALU/shift/compare + CPx zp). Closes the gap on ASM
        // coverage of the official 6502 instruction set.
        base.add(0x15).write(nes_asm_op_ora_zpx);
        base.add(0x35).write(nes_asm_op_and_zpx);
        base.add(0x55).write(nes_asm_op_eor_zpx);
        base.add(0x75).write(nes_asm_op_adc_zpx);
        base.add(0xD5).write(nes_asm_op_cmp_zpx);
        base.add(0xF5).write(nes_asm_op_sbc_zpx);
        base.add(0xE4).write(nes_asm_op_cpx_zp);
        base.add(0xC4).write(nes_asm_op_cpy_zp);
        base.add(0x16).write(nes_asm_op_asl_zpx);
        base.add(0x56).write(nes_asm_op_lsr_zpx);
        base.add(0x36).write(nes_asm_op_rol_zpx);
        base.add(0x76).write(nes_asm_op_ror_zpx);
        base.add(0x1E).write(nes_asm_op_asl_abs_x);
        base.add(0x5E).write(nes_asm_op_lsr_abs_x);
        base.add(0x3E).write(nes_asm_op_rol_abs_x);
        base.add(0x7E).write(nes_asm_op_ror_abs_x);
        // Phase 3 — stable illegal opcodes.
        // Implicit NOPs (6 slots, 2 cy each).
        base.add(0x1A).write(nes_asm_op_nop_impl2);
        base.add(0x3A).write(nes_asm_op_nop_impl2);
        base.add(0x5A).write(nes_asm_op_nop_impl2);
        base.add(0x7A).write(nes_asm_op_nop_impl2);
        base.add(0xDA).write(nes_asm_op_nop_impl2);
        base.add(0xFA).write(nes_asm_op_nop_impl2);
        // NOP imm (5 slots, 2 cy).
        base.add(0x80).write(nes_asm_op_nop_imm);
        base.add(0x82).write(nes_asm_op_nop_imm);
        base.add(0x89).write(nes_asm_op_nop_imm);
        base.add(0xC2).write(nes_asm_op_nop_imm);
        base.add(0xE2).write(nes_asm_op_nop_imm);
        // NOP zp (3 slots, 3 cy).
        base.add(0x04).write(nes_asm_op_nop_zp);
        base.add(0x44).write(nes_asm_op_nop_zp);
        base.add(0x64).write(nes_asm_op_nop_zp);
        // NOP zp,X (6 slots, 4 cy).
        base.add(0x14).write(nes_asm_op_nop_zpx);
        base.add(0x34).write(nes_asm_op_nop_zpx);
        base.add(0x54).write(nes_asm_op_nop_zpx);
        base.add(0x74).write(nes_asm_op_nop_zpx);
        base.add(0xD4).write(nes_asm_op_nop_zpx);
        base.add(0xF4).write(nes_asm_op_nop_zpx);
        // NOP abs (1 slot) + NOP abs,X (6 slots).
        base.add(0x0C).write(nes_asm_op_nop_abs);
        base.add(0x1C).write(nes_asm_op_nop_abs_x);
        base.add(0x3C).write(nes_asm_op_nop_abs_x);
        base.add(0x5C).write(nes_asm_op_nop_abs_x);
        base.add(0x7C).write(nes_asm_op_nop_abs_x);
        base.add(0xDC).write(nes_asm_op_nop_abs_x);
        base.add(0xFC).write(nes_asm_op_nop_abs_x);
        // SBC-alt (0xEB) — reuses sbc_imm handler directly.
        base.add(0xEB).write(nes_asm_op_sbc_imm);
        // LAX — load A and X from memory, set N/Z.
        base.add(0xA7).write(nes_asm_op_lax_zp);
        base.add(0xB7).write(nes_asm_op_lax_zpy);
        base.add(0xAF).write(nes_asm_op_lax_abs);
        base.add(0xBF).write(nes_asm_op_lax_abs_y);
        // SAX — store (A & X), no flags.
        base.add(0x87).write(nes_asm_op_sax_zp);
        base.add(0x97).write(nes_asm_op_sax_zpy);
        base.add(0x8F).write(nes_asm_op_sax_abs);
        // DCP — DEC + CMP.
        base.add(0xC7).write(nes_asm_op_dcp_zp);
        base.add(0xCF).write(nes_asm_op_dcp_abs);
        // ISC — INC + SBC.
        base.add(0xE7).write(nes_asm_op_isc_zp);
        base.add(0xEF).write(nes_asm_op_isc_abs);
        // SLO — ASL + ORA.
        base.add(0x07).write(nes_asm_op_slo_zp);
        base.add(0x0F).write(nes_asm_op_slo_abs);
        // RLA — ROL + AND.
        base.add(0x27).write(nes_asm_op_rla_zp);
        base.add(0x2F).write(nes_asm_op_rla_abs);
        // SRE — LSR + EOR.
        base.add(0x47).write(nes_asm_op_sre_zp);
        base.add(0x4F).write(nes_asm_op_sre_abs);
        // RRA — ROR + ADC.
        base.add(0x67).write(nes_asm_op_rra_zp);
        base.add(0x6F).write(nes_asm_op_rra_abs);
    }
}

use std::cell::Cell;
use std::sync::Once;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
static INSTALL_ONCE: Once = Once::new();

/// Thread-local function-pointer + context that the ASM's MMIO
/// callbacks use to tick APU/PPU with the CURRENT real sinks
/// belonging to the in-flight `Nes::step` caller. Populated by
/// `Nes::step` before entering ASM, cleared after.
///
/// The tick_fn is monomorphized per (V: VideoSink, A: AudioSink) at
/// the Nes::step call site, so no trait-object dispatch is involved.
/// The ctx pointer aliases the sinks that Nes::step already owns
/// mutably for this call — safe because MMIO callbacks run on the
/// same thread while `try_step_asm` is on the stack.
pub type AsmTickFn = unsafe extern "C" fn(
    bus_ptr: *mut core::ffi::c_void,
    sink_ctx_ptr: *mut core::ffi::c_void,
    cpu_cycles: u32,
);

thread_local! {
    static ASM_TICK: Cell<Option<(*mut core::ffi::c_void, AsmTickFn)>> =
        const { Cell::new(None) };
}

#[inline]
pub fn set_asm_tick(ctx: *mut core::ffi::c_void, f: AsmTickFn) {
    ASM_TICK.with(|t| t.set(Some((ctx, f))));
}

#[inline]
pub fn clear_asm_tick() {
    ASM_TICK.with(|t| t.set(None));
}

#[inline]
unsafe fn invoke_asm_tick(bus_ptr: *mut core::ffi::c_void, cycles: u32) {
    ASM_TICK.with(|t| {
        if let Some((ctx, f)) = t.get() {
            unsafe { f(bus_ptr, ctx, cycles) };
        }
    });
}

/// Counter of instructions that went through the ASM fast path.
/// Useful to verify integration is engaged on real ROMs.
pub static ASM_HITS: AtomicU64 = AtomicU64::new(0);

/// Per-opcode base cycle count for opcodes that have an ASM handler
/// installed. Zero means "no ASM handler — use Rust fallback". Populated
/// alongside the dispatch table in `install_opcode_table`. Used by
/// Nes::step to pre-tick PPU/APU for the instruction's cycles with
/// the real sinks before entering ASM, so MMIO callbacks observe a
/// correctly-positioned PPU without needing to tick themselves.
pub static ASM_OPCODE_CYCLES: [AtomicU8; 256] = {
    const Z: AtomicU8 = AtomicU8::new(0);
    [Z; 256]
};

#[inline]
pub fn asm_opcode_cycles(opcode: u8) -> u8 {
    ASM_OPCODE_CYCLES[opcode as usize].load(Ordering::Relaxed)
}

fn populate_asm_opcode_cycles() {
    // Derive from Cpu::OPCODES (each Instruction has a cycles slice
    // whose len is the base cycle count). Only mark opcodes that have
    // an installed ASM handler (i.e., the dispatch table slot is not
    // the Lunimpl sentinel).
    unsafe {
        let base = &raw const nes_asm_opcode_table as *const usize;
        let unimpl_addr = nes_asm_op_unimpl;
        for i in 0..256 {
            let handler = *base.add(i);
            if handler == unimpl_addr {
                continue;
            }
            if let Some(ref inst) = crate::cpu::OPCODES[i] {
                ASM_OPCODE_CYCLES[i].store(inst.base_cycles() as u8, Ordering::Relaxed);
            }
        }
    }
}

/// MMIO read callback invoked from ASM handlers when an absolute-mode
/// address lands in $2000-$7FFF.
///
/// **2026-04-26 structural fix**: ticks APU/PPU for ALL CPU cycles
/// consumed since the last MMIO callback (not just the current
/// instruction). The ASM CPU only ticks PPU via this callback —
/// without cumulative ticking, non-MMIO instructions in the same
/// batch (BPL, etc.) accumulate cycles that never reach PPU until
/// end-of-batch, causing $2002 polling loops to miss vblank inside
/// the batch. Repro: `tests/asm_vs_slow_smb.rs` lockstep harness.
///
/// After ticking, polls `ppu.nmi_output && ppu.nmi_occurred` — if
/// NMI is now pending, sets `pending_exit=1`. The ASM NEXT macro
/// reads `pending_exit` after every cycle decrement and bails to
/// the cycles-exit path so Rust's `poll_interrupts` services NMI
/// at the next instruction boundary (matches real 6502 NMI delivery
/// timing).
///
/// ABI: x0 = bus_ptr, w1 = addr, w2 = cycles_consumed_before_this_instr,
/// x3 = cpu_state_ptr. Returns u8 in w0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nes_asm_bus_read_byte(
    bus_ptr: *mut core::ffi::c_void,
    addr: u16,
    cycles_consumed_before_instr: u32,
    cpu_state_ptr: *mut AsmCpuState,
) -> u8 {
    let already_ticked = unsafe { (*cpu_state_ptr).cycles_ticked_in_callback };
    if cycles_consumed_before_instr > already_ticked {
        let to_tick = cycles_consumed_before_instr - already_ticked;
        unsafe { invoke_asm_tick(bus_ptr, to_tick) };
        unsafe {
            (*cpu_state_ptr).cycles_ticked_in_callback = cycles_consumed_before_instr;
        }
    }
    let bus = unsafe { &mut *(bus_ptr as *mut crate::system_bus::SystemBus) };
    let val = bus.read_byte(addr);
    // After the read, the PPU has been ticked through this instruction's
    // run-up to the MMIO read point. Bail to Rust if either:
    //   * NMI line now asserted (PPU entered vblank during ticks) — so
    //     Rust services the IRQ at the next instruction boundary.
    //   * CPU stall pending (OAM DMA / DMC DMA) — Rust's per-cycle
    //     `handle_oam_dma` runs the 513-514 cycle stall correctly.
    if bus.ppu_nmi_pending() || bus.cpu_stall_pending() {
        unsafe { (*cpu_state_ptr).pending_exit = 1; }
    }
    val
}

/// MMIO write callback. Same cumulative-cycles + NMI-exit semantics as
/// `nes_asm_bus_read_byte`.
///
/// ABI: x0 = bus_ptr, w1 = addr, w2 = value,
/// w3 = cycles_consumed_before_this_instr, x4 = cpu_state_ptr.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nes_asm_bus_write_byte(
    bus_ptr: *mut core::ffi::c_void,
    addr: u16,
    val: u8,
    cycles_consumed_before_instr: u32,
    cpu_state_ptr: *mut AsmCpuState,
) {
    let already_ticked = unsafe { (*cpu_state_ptr).cycles_ticked_in_callback };
    if cycles_consumed_before_instr > already_ticked {
        let to_tick = cycles_consumed_before_instr - already_ticked;
        unsafe { invoke_asm_tick(bus_ptr, to_tick) };
        unsafe {
            (*cpu_state_ptr).cycles_ticked_in_callback = cycles_consumed_before_instr;
        }
    }
    let bus = unsafe { &mut *(bus_ptr as *mut crate::system_bus::SystemBus) };
    bus.write_byte(addr, val);
    // STA $4014 activates OAM DMA (513-514 cycle CPU stall). The ASM
    // CPU has no notion of the stall — it would just keep dispatching
    // instructions, advancing PC past where the stall should hold it.
    // Bail so Rust's `Nes::tick` slow path takes over and runs the DMA
    // cycle-by-cycle (`handle_oam_dma`) before resuming normal CPU.
    if bus.ppu_nmi_pending() || bus.cpu_stall_pending() {
        unsafe { (*cpu_state_ptr).pending_exit = 1; }
    }
}

/// Install the dispatch table on first call; idempotent + thread-safe.
#[inline]
pub fn install_opcode_table_once() {
    INSTALL_ONCE.call_once(|| {
        install_opcode_table();
        populate_asm_opcode_cycles();
    });
}

/// Run ONE 6502 instruction through the ASM core from the current
/// Rust CPU state. Returns `Some(cycles)` if the ASM path handled the
/// opcode, `None` on unimplemented / MMIO (caller falls back to Rust).
///
/// `ram` is a pointer into the NES's 2 KB internal RAM (must be
/// writable). `prg` is a pointer to a flat 32 KB PRG ROM region or
/// `None` for banked mappers (ASM will bail on any PRG fetch).
///
/// On success the Rust Cpu state is updated to reflect the ASM-run
/// instruction (pc, a, x, y, sp, flags, cycles_total). On failure,
/// PC is rewound past the fetched opcode so the Rust fallback path
/// re-fetches and executes normally.
pub struct AsmStepResult {
    pub cycles_consumed: u32,
    pub cycles_ticked_in_callback: u32,
}

pub fn try_step_asm(
    cpu: &mut crate::cpu::Cpu,
    ram: *mut u8,
    prg: Option<*const u8>,
    bus: Option<*mut core::ffi::c_void>,
    cycles_budget: i64,
) -> Option<AsmStepResult> {
    install_opcode_table_once();

    let flags_byte: u8 = cpu.flags.into();
    let pc_on_entry = cpu.regs.pc;
    let mut asm = AsmCpuState {
        pc: pc_on_entry,
        a: cpu.regs.a,
        x: cpu.regs.x,
        y: cpu.regs.y,
        sp: cpu.regs.sp,
        p: flags_byte,
        bus_ptr: bus.map(|p| p as u64).unwrap_or(0),
        ..Default::default()
    };

    let prg_ptr = prg.unwrap_or(core::ptr::null());
    let exit = unsafe {
        nes_cpu_run_block(&mut asm as *mut _, ram, prg_ptr, cycles_budget)
    };

    let cycles_consumed = asm.cycles as u32;
    let ticked = asm.cycles_ticked_in_callback;

    // Successful block exit: write back full state.
    if exit == AsmExit::Cycles as u32 {
        cpu.regs.pc = asm.pc;
        cpu.regs.a = asm.a;
        cpu.regs.x = asm.x;
        cpu.regs.y = asm.y;
        cpu.regs.sp = asm.sp;
        cpu.flags = asm.p.into();
        return Some(AsmStepResult {
            cycles_consumed,
            cycles_ticked_in_callback: ticked,
        });
    }

    // Unimpl with prior instructions completed: commit the partial
    // state (PC rewound to the unimpl opcode by ASM), caller ticks
    // PPU/APU for consumed cycles, next Nes::step falls back to Rust
    // for the unimpl instruction.
    if exit == AsmExit::Unimpl as u32 && cycles_consumed > 0 {
        cpu.regs.pc = asm.pc;
        cpu.regs.a = asm.a;
        cpu.regs.x = asm.x;
        cpu.regs.y = asm.y;
        cpu.regs.sp = asm.sp;
        cpu.flags = asm.p.into();
        return Some(AsmStepResult {
            cycles_consumed,
            cycles_ticked_in_callback: ticked,
        });
    }

    // No instruction completed (first op was unimpl or MMIO-out-of-band).
    // Leave cpu untouched; Rust fallback re-executes from cpu.regs.pc.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Map a 6502 address $pc to PRG offset. For our 32 KB synthetic
    /// PRG, PC $8000 ↔ prg[0]; PC $C000 ↔ prg[0x4000]. The ASM
    /// `and x0, #0x7FFF` does the same.
    fn put_at(prg: &mut [u8], pc: u16, bytes: &[u8]) {
        let offset = (pc as usize) & 0x7FFF;
        prg[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    // ========================================================================
    // Differential test harness — run identical initial state + program
    // through the ASM path AND the Rust reference, compare final state.
    //
    // Per the design doc: every opcode ported to ASM must pass a diff
    // test before adding more opcodes. If a diff fails, stop and fix.
    // ========================================================================

    use crate::cartridge::Cartridge;
    use crate::nes::Nes;
    use crate::sink::{AudioSink, VideoSink};

    /// Initial CPU state we set up on both paths before running.
    #[derive(Clone, Debug)]
    struct DiffInit {
        pc: u16,
        a: u8,
        x: u8,
        y: u8,
        sp: u8,
        p: u8,
        ram: Vec<u8>,
        prg_bytes: Vec<(u16, u8)>,
    }

    impl Default for DiffInit {
        fn default() -> Self {
            Self {
                pc: 0xC000,
                a: 0,
                x: 0,
                y: 0,
                sp: 0xFD,
                p: 0x24,
                ram: vec![0u8; 2048],
                prg_bytes: Vec::new(),
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct DiffSnapshot {
        pc: u16,
        a: u8,
        x: u8,
        y: u8,
        sp: u8,
        p: u8,
        ram_after: Vec<u8>,
    }

    fn build_ines_rom(init: &DiffInit) -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2);
        rom.push(0);
        rom.push(0);
        rom.push(0);
        rom.extend_from_slice(&[0u8; 8]);
        let mut prg = vec![0u8; 32 * 1024];
        let reset_off = prg.len() - 4;
        prg[reset_off] = (init.pc & 0xFF) as u8;
        prg[reset_off + 1] = (init.pc >> 8) as u8;
        for (addr, byte) in &init.prg_bytes {
            let off = (*addr as usize) & 0x7FFF;
            prg[off] = *byte;
        }
        rom.extend(prg);
        rom
    }

    struct DiscardSinks;
    impl VideoSink for DiscardSinks {
        fn write_frame(&mut self, _: &[u8]) {}
        fn frame_written(&self) -> bool { false }
        fn pixel_size(&self) -> usize { 4 }
    }
    impl AudioSink for DiscardSinks {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize { 0 }
    }

    fn run_rust(init: &DiffInit, n_instructions: usize) -> DiffSnapshot {
        let rom = build_ines_rom(init);
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
        let mut nes = Nes::new(cart);
        nes.reset();
        // Reset's implicit step-until-frame walked the CPU to an
        // unknown state — overwrite with our known init for a clean
        // diff-test baseline. State omits `instruction` (non-pub)
        // which apply_state reconstructs from (opcode, cycle,
        // active_interrupt).
        let mut state = nes.cpu_state_for_diff_test();
        state.regs.pc = init.pc;
        state.regs.a = init.a;
        state.regs.x = init.x;
        state.regs.y = init.y;
        state.regs.sp = init.sp;
        state.flags = init.p.into();
        state.cycle = 0;
        state.opcode = 0;
        state.stall_cycles = 0;
        state.nmi_pended = false;
        state.nmi_line_low = false;
        state.irq_line_low = false;
        state.active_interrupt = None;
        nes.cpu_apply_state_for_diff_test(&state);
        nes.ram_mut_for_diff_test().copy_from_slice(&init.ram);

        let mut vs = DiscardSinks;
        let mut aus = DiscardSinks;
        // Pure per-cycle reference — bypasses Nes::step (which now
        // routes through the ASM fast path). Each iteration ticks
        // until exactly one CPU instruction completes.
        for _ in 0..n_instructions {
            while !nes.tick(&mut vs, &mut aus) {}
        }

        let st = nes.cpu_state_for_diff_test();
        let p_byte: u8 = st.flags.into();
        DiffSnapshot {
            pc: st.regs.pc,
            a: st.regs.a,
            x: st.regs.x,
            y: st.regs.y,
            sp: st.regs.sp,
            p: p_byte,
            ram_after: nes.ram_for_diff_test().to_vec(),
        }
    }

    /// Run ASM with an explicit cycle budget. The threaded-dispatch
    /// loop runs until either (a) the budget goes to 0 on the NEXT
    /// inside a handler (exits cleanly after that instruction's
    /// completion), (b) an unimpl opcode is hit, or (c) MMIO.
    ///
    /// For diff-testing, the caller chooses a budget equal to the
    /// sum of the intended instructions' cycle counts so the ASM
    /// exits at the same boundary as `run_rust(n)`.
    fn run_asm(init: &DiffInit, cycles_budget: i64) -> DiffSnapshot {
        install_opcode_table();
        let mut prg = vec![0u8; 32 * 1024];
        for (addr, byte) in &init.prg_bytes {
            let off = (*addr as usize) & 0x7FFF;
            prg[off] = *byte;
        }
        let mut ram = init.ram.clone();
        let mut cpu = AsmCpuState {
            pc: init.pc,
            a: init.a,
            x: init.x,
            y: init.y,
            sp: init.sp,
            p: init.p,
            ..Default::default()
        };
        let _exit = unsafe {
            nes_cpu_run_block(
                &mut cpu as *mut _,
                ram.as_mut_ptr(),
                prg.as_ptr(),
                cycles_budget,
            )
        };
        DiffSnapshot {
            pc: cpu.pc,
            a: cpu.a,
            x: cpu.x,
            y: cpu.y,
            sp: cpu.sp,
            p: cpu.p,
            ram_after: ram,
        }
    }

    fn diff_check(init: &DiffInit, n: usize, cycles: i64, p_mask: u8, label: &str) {
        let r = run_rust(init, n);
        let a = run_asm(init, cycles);
        assert_eq!(r.pc, a.pc, "{label}: PC rust={:#06X} asm={:#06X}", r.pc, a.pc);
        assert_eq!(r.a, a.a, "{label}: A rust={:#04X} asm={:#04X}", r.a, a.a);
        assert_eq!(r.x, a.x, "{label}: X rust={:#04X} asm={:#04X}", r.x, a.x);
        assert_eq!(r.y, a.y, "{label}: Y rust={:#04X} asm={:#04X}", r.y, a.y);
        assert_eq!(r.sp, a.sp, "{label}: SP rust={:#04X} asm={:#04X}", r.sp, a.sp);
        assert_eq!(
            r.p & p_mask, a.p & p_mask,
            "{label}: P mask={:#04X} rust={:#04X} asm={:#04X}",
            p_mask, r.p, a.p,
        );
        assert_eq!(r.ram_after, a.ram_after, "{label}: RAM diverged");
    }

    const P_MASK_PHASE0: u8 = 0b1000_0010; // N + Z

    #[test]
    fn diff_lda_imm() {
        let init = DiffInit {
            prg_bytes: vec![(0xC000, 0xA9), (0xC001, 0x42)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "LDA #$42");
    }

    #[test]
    fn diff_sta_zp() {
        let init = DiffInit {
            a: 0x99,
            prg_bytes: vec![(0xC000, 0x85), (0xC001, 0x10)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "STA $10");
    }

    #[test]
    fn diff_jmp_abs() {
        let init = DiffInit {
            prg_bytes: vec![
                (0xC000, 0x4C), (0xC001, 0x34), (0xC002, 0xC1),
                (0xC134, 0xEA),
            ],
            ..Default::default()
        };
        // JMP(3) + NOP(2) = 5 cycles.
        diff_check(&init, 2, 5, P_MASK_PHASE0, "JMP $C134 + NOP");
    }

    #[test]
    fn diff_bne_not_taken() {
        let init = DiffInit {
            p: 0x26,
            prg_bytes: vec![(0xC000, 0xD0), (0xC001, 0x10), (0xC002, 0xEA)],
            ..Default::default()
        };
        // BNE-not-taken(2) + NOP(2) = 4 cycles.
        diff_check(&init, 2, 4, P_MASK_PHASE0, "BNE not taken + NOP");
    }

    #[test]
    fn diff_bne_taken_same_page() {
        let init = DiffInit {
            p: 0x24,
            prg_bytes: vec![
                (0xC000, 0xD0), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        // BNE-taken-same-page(3) + NOP(2) = 5 cycles.
        diff_check(&init, 2, 5, P_MASK_PHASE0, "BNE taken same page + NOP");
    }

    #[test]
    fn diff_nop() {
        let init = DiffInit {
            prg_bytes: vec![(0xC000, 0xEA)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "NOP");
    }

    // ========================================================================
    // Phase 1 diff tests — 18 new opcodes. P masks:
    //   NZ     : loads, transfers, logical, stores (stores don't touch flags)
    //   NZC    : compares
    //   none   : branches (Z-dependent behavior checked via PC + cycle budget)
    // ========================================================================

    const P_MASK_NZC: u8 = 0b1000_0011; // N + Z + C

    #[test]
    fn diff_ldx_imm() {
        let init = DiffInit {
            prg_bytes: vec![(0xC000, 0xA2), (0xC001, 0x80)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "LDX #$80");
    }

    #[test]
    fn diff_ldy_imm_zero() {
        let init = DiffInit {
            y: 0x55,
            prg_bytes: vec![(0xC000, 0xA0), (0xC001, 0x00)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "LDY #$00 (sets Z)");
    }

    #[test]
    fn diff_lda_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x40] = 0xBE;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xA5), (0xC001, 0x40)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "LDA $40");
    }

    #[test]
    fn diff_ldx_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x12] = 0x7F;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xA6), (0xC001, 0x12)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "LDX $12");
    }

    #[test]
    fn diff_ldy_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x20] = 0x81;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xA4), (0xC001, 0x20)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "LDY $20");
    }

    #[test]
    fn diff_stx_zp() {
        let init = DiffInit {
            x: 0x5A,
            prg_bytes: vec![(0xC000, 0x86), (0xC001, 0x30)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "STX $30");
    }

    #[test]
    fn diff_sty_zp() {
        let init = DiffInit {
            y: 0xA5,
            prg_bytes: vec![(0xC000, 0x84), (0xC001, 0x31)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "STY $31");
    }

    #[test]
    fn diff_tax() {
        let init = DiffInit {
            a: 0x90,
            prg_bytes: vec![(0xC000, 0xAA)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "TAX");
    }

    #[test]
    fn diff_tay() {
        let init = DiffInit {
            a: 0x00,
            y: 0x55,
            prg_bytes: vec![(0xC000, 0xA8)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "TAY (sets Z)");
    }

    #[test]
    fn diff_txa() {
        let init = DiffInit {
            x: 0x7F,
            prg_bytes: vec![(0xC000, 0x8A)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "TXA");
    }

    #[test]
    fn diff_tya() {
        let init = DiffInit {
            y: 0xC0,
            prg_bytes: vec![(0xC000, 0x98)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_PHASE0, "TYA");
    }

    #[test]
    fn diff_beq_not_taken() {
        let init = DiffInit {
            p: 0x24, // Z=0
            prg_bytes: vec![(0xC000, 0xF0), (0xC001, 0x10), (0xC002, 0xEA)],
            ..Default::default()
        };
        // BEQ not taken (2) + NOP (2) = 4
        diff_check(&init, 2, 4, P_MASK_PHASE0, "BEQ not taken + NOP");
    }

    #[test]
    fn diff_beq_taken_same_page() {
        let init = DiffInit {
            p: 0x26, // Z=1
            prg_bytes: vec![
                (0xC000, 0xF0), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        // BEQ taken same page (3) + NOP (2) = 5
        diff_check(&init, 2, 5, P_MASK_PHASE0, "BEQ taken same page + NOP");
    }

    #[test]
    fn diff_and_imm() {
        let init = DiffInit {
            a: 0xF0,
            prg_bytes: vec![(0xC000, 0x29), (0xC001, 0x0F)],
            ..Default::default()
        };
        // A = 0xF0 & 0x0F = 0x00, Z=1
        diff_check(&init, 1, 2, P_MASK_PHASE0, "AND #$0F");
    }

    #[test]
    fn diff_ora_imm() {
        let init = DiffInit {
            a: 0x0F,
            prg_bytes: vec![(0xC000, 0x09), (0xC001, 0x80)],
            ..Default::default()
        };
        // A = 0x0F | 0x80 = 0x8F, N=1
        diff_check(&init, 1, 2, P_MASK_PHASE0, "ORA #$80");
    }

    #[test]
    fn diff_eor_imm() {
        let init = DiffInit {
            a: 0xAA,
            prg_bytes: vec![(0xC000, 0x49), (0xC001, 0xFF)],
            ..Default::default()
        };
        // A = 0xAA ^ 0xFF = 0x55
        diff_check(&init, 1, 2, P_MASK_PHASE0, "EOR #$FF");
    }

    #[test]
    fn diff_cmp_imm_equal() {
        let init = DiffInit {
            a: 0x42,
            prg_bytes: vec![(0xC000, 0xC9), (0xC001, 0x42)],
            ..Default::default()
        };
        // A == imm → Z=1, C=1, N=0
        diff_check(&init, 1, 2, P_MASK_NZC, "CMP #$42 equal");
    }

    #[test]
    fn diff_cmp_imm_greater() {
        let init = DiffInit {
            a: 0x80,
            prg_bytes: vec![(0xC000, 0xC9), (0xC001, 0x01)],
            ..Default::default()
        };
        // A > imm → Z=0, C=1, N=(0x80-0x01=0x7F)>>7=0
        diff_check(&init, 1, 2, P_MASK_NZC, "CMP #$01 greater");
    }

    #[test]
    fn diff_cmp_imm_less() {
        let init = DiffInit {
            a: 0x10,
            prg_bytes: vec![(0xC000, 0xC9), (0xC001, 0x80)],
            ..Default::default()
        };
        // A < imm → Z=0, C=0, N=(0x10-0x80=0x90)>>7=1
        diff_check(&init, 1, 2, P_MASK_NZC, "CMP #$80 less");
    }

    #[test]
    fn diff_cpx_imm() {
        let init = DiffInit {
            x: 0x05,
            prg_bytes: vec![(0xC000, 0xE0), (0xC001, 0x03)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_NZC, "CPX #$03");
    }

    #[test]
    fn diff_cpy_imm() {
        let init = DiffInit {
            y: 0x10,
            prg_bytes: vec![(0xC000, 0xC0), (0xC001, 0x20)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_NZC, "CPY #$20");
    }

    // ========================================================================
    // Phase 2a diff tests — flag setters + inc/dec.
    // ========================================================================

    const P_MASK_C: u8 = 0b0000_0001;
    const P_MASK_I: u8 = 0b0000_0100;
    const P_MASK_D: u8 = 0b0000_1000;
    const P_MASK_V: u8 = 0b0100_0000;

    #[test]
    fn diff_clc() {
        let init = DiffInit {
            p: 0x25, // C=1 initially
            prg_bytes: vec![(0xC000, 0x18)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_C, "CLC");
    }

    #[test]
    fn diff_sec() {
        let init = DiffInit {
            p: 0x24, // C=0 initially
            prg_bytes: vec![(0xC000, 0x38)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_C, "SEC");
    }

    #[test]
    fn diff_cli() {
        let init = DiffInit {
            p: 0x24, // I=1 initially
            prg_bytes: vec![(0xC000, 0x58)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_I, "CLI");
    }

    #[test]
    fn diff_sei() {
        let init = DiffInit {
            p: 0x20, // I=0 initially
            prg_bytes: vec![(0xC000, 0x78)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_I, "SEI");
    }

    #[test]
    fn diff_cld() {
        let init = DiffInit {
            p: 0x2C, // D=1 initially
            prg_bytes: vec![(0xC000, 0xD8)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_D, "CLD");
    }

    #[test]
    fn diff_sed() {
        let init = DiffInit {
            p: 0x24, // D=0
            prg_bytes: vec![(0xC000, 0xF8)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_D, "SED");
    }

    #[test]
    fn diff_clv() {
        let init = DiffInit {
            p: 0x64, // V=1 initially
            prg_bytes: vec![(0xC000, 0xB8)],
            ..Default::default()
        };
        diff_check(&init, 1, 2, P_MASK_V, "CLV");
    }

    #[test]
    fn diff_inx_wrap() {
        let init = DiffInit {
            x: 0xFF,
            prg_bytes: vec![(0xC000, 0xE8)],
            ..Default::default()
        };
        // X = 0x00, Z=1
        diff_check(&init, 1, 2, P_MASK_PHASE0, "INX wrap");
    }

    #[test]
    fn diff_iny() {
        let init = DiffInit {
            y: 0x7F,
            prg_bytes: vec![(0xC000, 0xC8)],
            ..Default::default()
        };
        // Y = 0x80, N=1
        diff_check(&init, 1, 2, P_MASK_PHASE0, "INY");
    }

    #[test]
    fn diff_dex() {
        let init = DiffInit {
            x: 0x01,
            prg_bytes: vec![(0xC000, 0xCA)],
            ..Default::default()
        };
        // X = 0x00, Z=1
        diff_check(&init, 1, 2, P_MASK_PHASE0, "DEX");
    }

    #[test]
    fn diff_dey_wrap() {
        let init = DiffInit {
            y: 0x00,
            prg_bytes: vec![(0xC000, 0x88)],
            ..Default::default()
        };
        // Y = 0xFF, N=1
        diff_check(&init, 1, 2, P_MASK_PHASE0, "DEY wrap");
    }

    #[test]
    fn diff_inc_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x50] = 0x7F;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xE6), (0xC001, 0x50)],
            ..Default::default()
        };
        // mem[$50] 0x7F → 0x80, N=1
        diff_check(&init, 1, 5, P_MASK_PHASE0, "INC $50");
    }

    #[test]
    fn diff_dec_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x60] = 0x01;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xC6), (0xC001, 0x60)],
            ..Default::default()
        };
        // mem[$60] 0x01 → 0x00, Z=1
        diff_check(&init, 1, 5, P_MASK_PHASE0, "DEC $60");
    }

    // ========================================================================
    // Phase 2b diff tests — 6 branches, 3 abs loads, 4 zp logical, BIT zp, PHA/PLA
    // ========================================================================

    #[test]
    fn diff_bpl_taken() {
        let init = DiffInit {
            p: 0x24, // N=0 → taken
            prg_bytes: vec![
                (0xC000, 0x10), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 5, P_MASK_PHASE0, "BPL taken + NOP");
    }

    #[test]
    fn diff_bmi_taken() {
        let init = DiffInit {
            p: 0xA4, // N=1 → taken
            prg_bytes: vec![
                (0xC000, 0x30), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 5, P_MASK_PHASE0, "BMI taken + NOP");
    }

    #[test]
    fn diff_bvc_taken() {
        let init = DiffInit {
            p: 0x24, // V=0 → taken
            prg_bytes: vec![
                (0xC000, 0x50), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 5, P_MASK_V, "BVC taken + NOP");
    }

    #[test]
    fn diff_bvs_taken() {
        let init = DiffInit {
            p: 0x64, // V=1 → taken
            prg_bytes: vec![
                (0xC000, 0x70), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 5, P_MASK_V, "BVS taken + NOP");
    }

    #[test]
    fn diff_bcc_taken() {
        let init = DiffInit {
            p: 0x24, // C=0 → taken
            prg_bytes: vec![
                (0xC000, 0x90), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 5, P_MASK_C, "BCC taken + NOP");
    }

    #[test]
    fn diff_bcs_taken() {
        let init = DiffInit {
            p: 0x25, // C=1 → taken
            prg_bytes: vec![
                (0xC000, 0xB0), (0xC001, 0x10),
                (0xC012, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 5, P_MASK_C, "BCS taken + NOP");
    }

    #[test]
    fn diff_lda_abs_ram() {
        let mut ram = vec![0u8; 2048];
        ram[0x0200] = 0x73;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xAD), (0xC001, 0x00), (0xC002, 0x02)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDA $0200 (RAM)");
    }

    #[test]
    fn diff_lda_abs_prg() {
        // Read from PRG ROM itself — put a byte at $CF34 via prg_bytes.
        let init = DiffInit {
            prg_bytes: vec![
                (0xC000, 0xAD), (0xC001, 0x34), (0xC002, 0xCF),
                (0xCF34, 0x81),
            ],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDA $CF34 (PRG)");
    }

    #[test]
    fn diff_ldx_abs_ram() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0xEE;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xAE), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDX $0100 (RAM)");
    }

    #[test]
    fn diff_ldy_abs_ram() {
        let mut ram = vec![0u8; 2048];
        ram[0x00AB] = 0x55;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xAC), (0xC001, 0xAB), (0xC002, 0x00)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDY $00AB (RAM)");
    }

    #[test]
    fn diff_and_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x40] = 0x0F;
        let init = DiffInit {
            a: 0xF0,
            ram,
            prg_bytes: vec![(0xC000, 0x25), (0xC001, 0x40)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "AND $40");
    }

    #[test]
    fn diff_ora_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x50] = 0x80;
        let init = DiffInit {
            a: 0x0F,
            ram,
            prg_bytes: vec![(0xC000, 0x05), (0xC001, 0x50)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "ORA $50");
    }

    #[test]
    fn diff_eor_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x60] = 0xFF;
        let init = DiffInit {
            a: 0xAA,
            ram,
            prg_bytes: vec![(0xC000, 0x45), (0xC001, 0x60)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_PHASE0, "EOR $60");
    }

    #[test]
    fn diff_cmp_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x70] = 0x42;
        let init = DiffInit {
            a: 0x42,
            ram,
            prg_bytes: vec![(0xC000, 0xC5), (0xC001, 0x70)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_NZC, "CMP $70 equal");
    }

    const P_MASK_NZV: u8 = 0b1100_0010;

    #[test]
    fn diff_bit_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x30] = 0xC0; // N=1 V=1 from mem, A & mem = 0 → Z=1
        let init = DiffInit {
            a: 0x0F,
            ram,
            prg_bytes: vec![(0xC000, 0x24), (0xC001, 0x30)],
            ..Default::default()
        };
        diff_check(&init, 1, 3, P_MASK_NZV, "BIT $30");
    }

    #[test]
    fn diff_sta_abs_ram() {
        let init = DiffInit {
            a: 0x99,
            prg_bytes: vec![(0xC000, 0x8D), (0xC001, 0x50), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "STA $0150");
    }

    #[test]
    fn diff_stx_abs_ram() {
        let init = DiffInit {
            x: 0x7E,
            prg_bytes: vec![(0xC000, 0x8E), (0xC001, 0x42), (0xC002, 0x00)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "STX $0042");
    }

    #[test]
    fn diff_sty_abs_ram() {
        let init = DiffInit {
            y: 0x33,
            prg_bytes: vec![(0xC000, 0x8C), (0xC001, 0x80), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "STY $0180");
    }

    const P_MASK_NZVC: u8 = 0b1100_0011; // N + Z + V + C

    #[test]
    fn diff_adc_imm_basic() {
        let init = DiffInit {
            a: 0x50,
            p: 0x24, // C=0
            prg_bytes: vec![(0xC000, 0x69), (0xC001, 0x10)],
            ..Default::default()
        };
        // 0x50 + 0x10 + 0 = 0x60, no overflow, no carry
        diff_check(&init, 1, 2, P_MASK_NZVC, "ADC #$10 + 0x50 no carry");
    }

    #[test]
    fn diff_adc_imm_carry_in() {
        let init = DiffInit {
            a: 0x10,
            p: 0x25, // C=1
            prg_bytes: vec![(0xC000, 0x69), (0xC001, 0x20)],
            ..Default::default()
        };
        // 0x10 + 0x20 + 1 = 0x31
        diff_check(&init, 1, 2, P_MASK_NZVC, "ADC #$20 + 0x10 C=1");
    }

    #[test]
    fn diff_adc_imm_overflow() {
        let init = DiffInit {
            a: 0x50,
            p: 0x24, // C=0
            prg_bytes: vec![(0xC000, 0x69), (0xC001, 0x50)],
            ..Default::default()
        };
        // 0x50 + 0x50 = 0xA0, V=1 (signed overflow: pos+pos=neg)
        diff_check(&init, 1, 2, P_MASK_NZVC, "ADC #$50 + 0x50 overflow");
    }

    #[test]
    fn diff_adc_imm_carry_out() {
        let init = DiffInit {
            a: 0xFF,
            p: 0x24, // C=0
            prg_bytes: vec![(0xC000, 0x69), (0xC001, 0x02)],
            ..Default::default()
        };
        // 0xFF + 0x02 = 0x101, A=0x01, C=1
        diff_check(&init, 1, 2, P_MASK_NZVC, "ADC #$02 + 0xFF carry out");
    }

    #[test]
    fn diff_sbc_imm_basic() {
        let init = DiffInit {
            a: 0x50,
            p: 0x25, // C=1 (no borrow)
            prg_bytes: vec![(0xC000, 0xE9), (0xC001, 0x10)],
            ..Default::default()
        };
        // 0x50 - 0x10 - 0 = 0x40; C=1 (no borrow)
        diff_check(&init, 1, 2, P_MASK_NZVC, "SBC #$10 from 0x50");
    }

    #[test]
    fn diff_sbc_imm_borrow() {
        let init = DiffInit {
            a: 0x50,
            p: 0x24, // C=0 (borrow pending)
            prg_bytes: vec![(0xC000, 0xE9), (0xC001, 0x10)],
            ..Default::default()
        };
        // 0x50 - 0x10 - 1 = 0x3F
        diff_check(&init, 1, 2, P_MASK_NZVC, "SBC #$10 from 0x50 with borrow");
    }

    #[test]
    fn diff_sbc_imm_underflow() {
        let init = DiffInit {
            a: 0x10,
            p: 0x25, // C=1
            prg_bytes: vec![(0xC000, 0xE9), (0xC001, 0x20)],
            ..Default::default()
        };
        // 0x10 - 0x20 = 0xF0, C=0 (borrow occurred)
        diff_check(&init, 1, 2, P_MASK_NZVC, "SBC #$20 from 0x10 underflow");
    }

    #[test]
    fn diff_adc_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x30] = 0x25;
        let init = DiffInit {
            a: 0x10,
            ram,
            p: 0x25, // C=1
            prg_bytes: vec![(0xC000, 0x65), (0xC001, 0x30)],
            ..Default::default()
        };
        // 0x10 + 0x25 + 1 = 0x36
        diff_check(&init, 1, 3, P_MASK_NZVC, "ADC $30");
    }

    #[test]
    fn diff_sbc_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x40] = 0x05;
        let init = DiffInit {
            a: 0x10,
            ram,
            p: 0x25, // C=1
            prg_bytes: vec![(0xC000, 0xE5), (0xC001, 0x40)],
            ..Default::default()
        };
        // 0x10 - 0x05 - 0 = 0x0B
        diff_check(&init, 1, 3, P_MASK_NZVC, "SBC $40");
    }

    #[test]
    fn diff_cmp_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x01FF] = 0x42;
        let init = DiffInit {
            a: 0x50,
            ram,
            prg_bytes: vec![(0xC000, 0xCD), (0xC001, 0xFF), (0xC002, 0x01)],
            ..Default::default()
        };
        // A=0x50 cmp 0x42 → A>M, C=1, Z=0, N=0
        diff_check(&init, 1, 4, P_MASK_NZC, "CMP $01FF");
    }

    #[test]
    fn diff_cpx_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0x30;
        let init = DiffInit {
            x: 0x30,
            ram,
            prg_bytes: vec![(0xC000, 0xEC), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        // X=M → Z=1, C=1, N=0
        diff_check(&init, 1, 4, P_MASK_NZC, "CPX $0100 equal");
    }

    #[test]
    fn diff_cpy_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0050] = 0x20;
        let init = DiffInit {
            y: 0x10,
            ram,
            prg_bytes: vec![(0xC000, 0xCC), (0xC001, 0x50), (0xC002, 0x00)],
            ..Default::default()
        };
        // Y=0x10 cmp 0x20 → Y<M, C=0, Z=0, N=(0x10-0x20=0xF0)→1
        diff_check(&init, 1, 4, P_MASK_NZC, "CPY $0050 less");
    }

    #[test]
    fn diff_lda_zpx_no_wrap() {
        let mut ram = vec![0u8; 2048];
        ram[0x50] = 0x77;
        let init = DiffInit {
            x: 0x10,
            ram,
            prg_bytes: vec![(0xC000, 0xB5), (0xC001, 0x40)],
            ..Default::default()
        };
        // zp=0x40 + X=0x10 = 0x50
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDA $40,X");
    }

    #[test]
    fn diff_lda_zpx_wraps() {
        let mut ram = vec![0u8; 2048];
        ram[0x02] = 0x99;
        let init = DiffInit {
            x: 0x04,
            ram,
            prg_bytes: vec![(0xC000, 0xB5), (0xC001, 0xFE)],
            ..Default::default()
        };
        // zp=0xFE + X=0x04 = 0x102 → wraps to 0x02
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDA $FE,X wraps");
    }

    #[test]
    fn diff_sta_abs_x() {
        let init = DiffInit {
            a: 0x42,
            x: 0x10,
            prg_bytes: vec![(0xC000, 0x9D), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        // base=$0100 + X=$10 = $0110 (RAM)
        diff_check(&init, 1, 5, P_MASK_PHASE0, "STA $0100,X");
    }

    #[test]
    fn diff_sta_abs_y() {
        let init = DiffInit {
            a: 0x99,
            y: 0x05,
            prg_bytes: vec![(0xC000, 0x99), (0xC001, 0x20), (0xC002, 0x00)],
            ..Default::default()
        };
        // base=$0020 + Y=$05 = $0025 (RAM)
        diff_check(&init, 1, 5, P_MASK_PHASE0, "STA $0020,Y");
    }

    #[test]
    fn diff_sta_ind_y() {
        let mut ram = vec![0u8; 2048];
        ram[0x40] = 0x00; // low byte of base
        ram[0x41] = 0x01; // high byte of base → base=$0100
        let init = DiffInit {
            a: 0x55,
            y: 0x08,
            ram,
            prg_bytes: vec![(0xC000, 0x91), (0xC001, 0x40)],
            ..Default::default()
        };
        // effective = $0100 + Y=$08 = $0108
        diff_check(&init, 1, 6, P_MASK_PHASE0, "STA ($40),Y");
    }

    #[test]
    fn diff_lda_abs_x_no_cross() {
        let mut ram = vec![0u8; 2048];
        ram[0x0150] = 0x88;
        let init = DiffInit {
            x: 0x50,
            ram,
            prg_bytes: vec![(0xC000, 0xBD), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        // base=$0100 + X=$50 = $0150, same page = 4 cycles
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDA $0100,X no cross");
    }

    #[test]
    fn diff_lda_abs_x_pagecross() {
        let mut ram = vec![0u8; 2048];
        ram[0x0300] = 0x77;
        let init = DiffInit {
            x: 0x10,
            ram,
            prg_bytes: vec![(0xC000, 0xBD), (0xC001, 0xF0), (0xC002, 0x02)],
            ..Default::default()
        };
        // base=$02F0 + X=$10 = $0300, cross = 5 cycles
        diff_check(&init, 1, 5, P_MASK_PHASE0, "LDA $02F0,X with cross");
    }

    #[test]
    fn diff_lda_abs_y_no_cross() {
        let mut ram = vec![0u8; 2048];
        ram[0x0030] = 0x22;
        let init = DiffInit {
            y: 0x30,
            ram,
            prg_bytes: vec![(0xC000, 0xB9), (0xC001, 0x00), (0xC002, 0x00)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDA $0000,Y no cross");
    }

    #[test]
    fn diff_ldx_abs_y() {
        let mut ram = vec![0u8; 2048];
        ram[0x0040] = 0x66;
        let init = DiffInit {
            y: 0x40,
            ram,
            prg_bytes: vec![(0xC000, 0xBE), (0xC001, 0x00), (0xC002, 0x00)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDX $0000,Y");
    }

    #[test]
    fn diff_ldy_abs_x() {
        let mut ram = vec![0u8; 2048];
        ram[0x0200] = 0x44;
        let init = DiffInit {
            x: 0x00,
            ram,
            prg_bytes: vec![(0xC000, 0xBC), (0xC001, 0x00), (0xC002, 0x02)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "LDY $0200,X");
    }

    #[test]
    fn diff_lda_ind_y() {
        let mut ram = vec![0u8; 2048];
        ram[0x10] = 0x00;
        ram[0x11] = 0x02;                     // pointer → $0200
        ram[0x0208] = 0xCC;
        let init = DiffInit {
            y: 0x08,
            ram,
            prg_bytes: vec![(0xC000, 0xB1), (0xC001, 0x10)],
            ..Default::default()
        };
        // base=$0200 + Y=$08 = $0208 (same page, 5 cycles)
        diff_check(&init, 1, 5, P_MASK_PHASE0, "LDA ($10),Y");
    }

    #[test]
    fn diff_and_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0x0F;
        let init = DiffInit {
            a: 0xAA,
            ram,
            prg_bytes: vec![(0xC000, 0x2D), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "AND $0100");
    }

    #[test]
    fn diff_ora_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0x80;
        let init = DiffInit {
            a: 0x01,
            ram,
            prg_bytes: vec![(0xC000, 0x0D), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "ORA $0100");
    }

    #[test]
    fn diff_eor_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0xFF;
        let init = DiffInit {
            a: 0x55,
            ram,
            prg_bytes: vec![(0xC000, 0x4D), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_PHASE0, "EOR $0100");
    }

    #[test]
    fn diff_adc_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0x25;
        let init = DiffInit {
            a: 0x10,
            ram,
            p: 0x24, // C=0
            prg_bytes: vec![(0xC000, 0x6D), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        // 0x10 + 0x25 = 0x35
        diff_check(&init, 1, 4, P_MASK_NZVC, "ADC $0100");
    }

    #[test]
    fn diff_sbc_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0100] = 0x05;
        let init = DiffInit {
            a: 0x10,
            ram,
            p: 0x25, // C=1
            prg_bytes: vec![(0xC000, 0xED), (0xC001, 0x00), (0xC002, 0x01)],
            ..Default::default()
        };
        diff_check(&init, 1, 4, P_MASK_NZVC, "SBC $0100");
    }

    #[test]
    fn diff_asl_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x40] = 0x81;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0x06), (0xC001, 0x40)],
            ..Default::default()
        };
        // 0x81 << 1 = 0x02, C=1, N=0, Z=0
        diff_check(&init, 1, 5, P_MASK_NZC, "ASL $40");
    }

    #[test]
    fn diff_lsr_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x50] = 0x03;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0x46), (0xC001, 0x50)],
            ..Default::default()
        };
        // 0x03 >> 1 = 0x01, C=1, N=0
        diff_check(&init, 1, 5, P_MASK_NZC, "LSR $50");
    }

    #[test]
    fn diff_rol_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x60] = 0x80;
        let init = DiffInit {
            ram,
            p: 0x24, // C=0
            prg_bytes: vec![(0xC000, 0x26), (0xC001, 0x60)],
            ..Default::default()
        };
        // 0x80 << 1 | 0 = 0x00, C=1, Z=1
        diff_check(&init, 1, 5, P_MASK_NZC, "ROL $60 with C=0");
    }

    #[test]
    fn diff_ror_zp() {
        let mut ram = vec![0u8; 2048];
        ram[0x70] = 0x02;
        let init = DiffInit {
            ram,
            p: 0x25, // C=1 in
            prg_bytes: vec![(0xC000, 0x66), (0xC001, 0x70)],
            ..Default::default()
        };
        // (0x02 >> 1) | (1 << 7) = 0x81, C=0
        diff_check(&init, 1, 5, P_MASK_NZC, "ROR $70 with C=1 in");
    }

    #[test]
    fn diff_jmp_ind_normal() {
        let mut ram = vec![0u8; 2048];
        ram[0x10] = 0x34;
        ram[0x11] = 0xC1;
        let init = DiffInit {
            ram,
            prg_bytes: vec![
                (0xC000, 0x6C), (0xC001, 0x10), (0xC002, 0x00),
                (0xC134, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 7, P_MASK_PHASE0, "JMP ($0010) + NOP");
    }

    #[test]
    fn diff_jmp_ind_page_bug() {
        let mut ram = vec![0u8; 2048];
        ram[0xFF] = 0x44;
        ram[0x00] = 0xC2;   // 6502 bug: $00FF+1 wraps to $0000, not $0100
        let init = DiffInit {
            ram,
            prg_bytes: vec![
                (0xC000, 0x6C), (0xC001, 0xFF), (0xC002, 0x00),
                (0xC244, 0xEA),
            ],
            ..Default::default()
        };
        diff_check(&init, 2, 7, P_MASK_PHASE0, "JMP ($00FF) page-bug + NOP");
    }

    #[test]
    fn diff_inc_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0200] = 0x7F;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xEE), (0xC001, 0x00), (0xC002, 0x02)],
            ..Default::default()
        };
        diff_check(&init, 1, 6, P_MASK_PHASE0, "INC $0200");
    }

    #[test]
    fn diff_dec_abs() {
        let mut ram = vec![0u8; 2048];
        ram[0x0300] = 0x01;
        let init = DiffInit {
            ram,
            prg_bytes: vec![(0xC000, 0xCE), (0xC001, 0x00), (0xC002, 0x03)],
            ..Default::default()
        };
        diff_check(&init, 1, 6, P_MASK_PHASE0, "DEC $0300");
    }

    #[test]
    fn diff_inc_abs_x() {
        let mut ram = vec![0u8; 2048];
        ram[0x0250] = 0x10;
        let init = DiffInit {
            x: 0x50,
            ram,
            prg_bytes: vec![(0xC000, 0xFE), (0xC001, 0x00), (0xC002, 0x02)],
            ..Default::default()
        };
        diff_check(&init, 1, 7, P_MASK_PHASE0, "INC $0200,X");
    }

    #[test]
    fn diff_dec_abs_x() {
        let mut ram = vec![0u8; 2048];
        ram[0x0260] = 0x80;
        let init = DiffInit {
            x: 0x60,
            ram,
            prg_bytes: vec![(0xC000, 0xDE), (0xC001, 0x00), (0xC002, 0x02)],
            ..Default::default()
        };
        diff_check(&init, 1, 7, P_MASK_PHASE0, "DEC $0200,X");
    }

    #[test]
    fn diff_asl_a() {
        let init = DiffInit {
            a: 0x81,
            prg_bytes: vec![(0xC000, 0x0A)],
            ..Default::default()
        };
        // 0x81 << 1 = 0x02, C=1 (old bit 7), N=0, Z=0
        diff_check(&init, 1, 2, P_MASK_NZC, "ASL A");
    }

    #[test]
    fn diff_lsr_a() {
        let init = DiffInit {
            a: 0x03,
            prg_bytes: vec![(0xC000, 0x4A)],
            ..Default::default()
        };
        // 0x03 >> 1 = 0x01, C=1 (old bit 0), N=0
        diff_check(&init, 1, 2, P_MASK_NZC, "LSR A");
    }

    #[test]
    fn diff_rol_a() {
        let init = DiffInit {
            a: 0x81,
            p: 0x24, // C=0 in
            prg_bytes: vec![(0xC000, 0x2A)],
            ..Default::default()
        };
        // (0x81 << 1) | 0 = 0x02, C=1 (old bit 7)
        diff_check(&init, 1, 2, P_MASK_NZC, "ROL A C=0");
    }

    #[test]
    fn diff_ror_a_with_carry() {
        let init = DiffInit {
            a: 0x02,
            p: 0x25, // C=1 in
            prg_bytes: vec![(0xC000, 0x6A)],
            ..Default::default()
        };
        // (0x02 >> 1) | (1 << 7) = 0x81, C=0 (old bit 0)
        diff_check(&init, 1, 2, P_MASK_NZC, "ROR A C=1 in");
    }

    #[test]
    fn diff_sta_zpx() {
        let init = DiffInit {
            a: 0xAB,
            x: 0x05,
            prg_bytes: vec![(0xC000, 0x95), (0xC001, 0x20)],
            ..Default::default()
        };
        // zp=0x20 + X=0x05 = 0x25
        diff_check(&init, 1, 4, P_MASK_PHASE0, "STA $20,X");
    }

    #[test]
    fn diff_jsr_rts_roundtrip() {
        let init = DiffInit {
            a: 0x11,
            sp: 0xFD,
            prg_bytes: vec![
                (0xC000, 0x20), (0xC001, 0x10), (0xC002, 0xC1), // JSR $C110
                (0xC003, 0xEA),                                  // NOP after RTS returns
                (0xC110, 0xA9), (0xC111, 0x55),                  // subroutine: LDA #$55
                (0xC112, 0x60),                                  // RTS
            ],
            ..Default::default()
        };
        // JSR(6) + LDA_imm(2) + RTS(6) + NOP(2) = 16 cycles
        diff_check(&init, 4, 16, P_MASK_PHASE0, "JSR $C110 / LDA #$55 / RTS / NOP");
    }

    #[test]
    fn diff_pha_pla_roundtrip() {
        let init = DiffInit {
            a: 0x42,
            sp: 0xFD,
            prg_bytes: vec![
                (0xC000, 0x48),       // PHA (A=0x42, push)
                (0xC001, 0xA9),       // LDA #$00
                (0xC002, 0x00),
                (0xC003, 0x68),       // PLA (restores A=0x42)
            ],
            ..Default::default()
        };
        // PHA(3) + LDA_imm(2) + PLA(4) = 9 cycles
        diff_check(&init, 3, 9, P_MASK_PHASE0, "PHA / LDA #0 / PLA");
    }

    /// Diagnostic: print addresses the Rust side sees vs what ASM
    /// would read.
    #[test]
    fn debug_addresses() {
        install_opcode_table();
        unsafe {
            let table_addr = &raw const nes_asm_opcode_table as *const [usize; 256] as usize;
            let op_nop_addr = &raw const nes_asm_op_nop as *const usize as usize;
            let op_nop_value = nes_asm_op_nop;
            let op_lda_value = nes_asm_op_lda_imm;
            eprintln!("table base addr: {:#x}", table_addr);
            eprintln!("op_nop pointer addr: {:#x}", op_nop_addr);
            eprintln!("op_nop target (handler addr): {:#x}", op_nop_value);
            eprintln!("op_lda_imm target: {:#x}", op_lda_value);
            let table_slot_ea = ((&raw const nes_asm_opcode_table) as *const usize).add(0xEA);
            eprintln!("table[0xEA] addr: {:#x} contains: {:#x}",
                      table_slot_ea as usize, *table_slot_ea);
            let table_slot_a9 = ((&raw const nes_asm_opcode_table) as *const usize).add(0xA9);
            eprintln!("table[0xA9] addr: {:#x} contains: {:#x}",
                      table_slot_a9 as usize, *table_slot_a9);
        }
    }

    /// Smoke test: build a 32 KB PRG image with LDA #$42; STA $05;
    /// NOP; JMP $C000 (tight loop), run the ASM core for 100 cycles
    /// starting from $C000, and verify the final state.
    #[test]
    fn skeleton_smoke() {
        install_opcode_table();

        let mut ram = vec![0u8; 2048];
        let mut prg = vec![0u8; 32 * 1024];
        // Program at PC $C000:
        // A9 42       LDA #$42
        // 85 05       STA $05
        // EA          NOP
        // 4C 00 C0    JMP $C000
        put_at(&mut prg, 0xC000, &[0xA9, 0x42, 0x85, 0x05, 0xEA, 0x4C, 0x00, 0xC0]);

        let mut cpu = AsmCpuState::default();
        cpu.pc = 0xC000;
        cpu.sp = 0xFD;
        cpu.p = 0x24; // I=1, E=1 — reset state

        let exit = unsafe {
            nes_cpu_run_block(
                &mut cpu as *mut _,
                ram.as_mut_ptr(),
                prg.as_ptr(),
                100,
            )
        };
        assert_eq!(exit, AsmExit::Cycles as u32, "expected cycles exit");
        assert_eq!(cpu.a, 0x42, "A should be 0x42 after LDA");
        assert_eq!(ram[0x05], 0x42, "$05 should hold the stored value");
        // Program loops JMP $C000, so PC should be back at $C000 or
        // partway through.
        assert!(
            cpu.pc == 0xC000 || cpu.pc == 0xC002 || cpu.pc == 0xC004 || cpu.pc == 0xC005,
            "PC in an unexpected position: {:#06X}",
            cpu.pc
        );
    }

    #[test]
    fn bne_taken_and_not_taken() {
        install_opcode_table();
        let mut ram = vec![0u8; 2048];
        let mut prg = vec![0u8; 32 * 1024];

        // At PC $C000:
        // A9 00       LDA #$00        (sets Z=1)
        // D0 02       BNE +2          (NOT taken; Z=1)
        // A9 37       LDA #$37        (A becomes $37, Z=0)
        // D0 FE       BNE -2          (taken; infinite loop on $37 →
        //                              budget exhaustion)
        put_at(&mut prg, 0xC000, &[0xA9, 0x00, 0xD0, 0x02, 0xA9, 0x37, 0xD0, 0xFE]);

        let mut cpu = AsmCpuState::default();
        cpu.pc = 0xC000;
        cpu.sp = 0xFD;
        cpu.p = 0x24;

        let exit = unsafe {
            nes_cpu_run_block(
                &mut cpu as *mut _,
                ram.as_mut_ptr(),
                prg.as_ptr(),
                50,
            )
        };
        assert_eq!(exit, AsmExit::Cycles as u32);
        assert_eq!(cpu.a, 0x37);
    }

    #[test]
    fn jmp_abs_lands_correctly() {
        install_opcode_table();
        let mut ram = vec![0u8; 2048];
        let mut prg = vec![0u8; 32 * 1024];

        // 4C 34 C1     JMP $C134 at $C000
        // NOP sled at $C134..$C140
        put_at(&mut prg, 0xC000, &[0x4C, 0x34, 0xC1]);
        put_at(&mut prg, 0xC134, &[0xEA; 12]);

        let mut cpu = AsmCpuState::default();
        cpu.pc = 0xC000;
        cpu.sp = 0xFD;
        cpu.p = 0x24;

        let exit = unsafe {
            nes_cpu_run_block(
                &mut cpu as *mut _,
                ram.as_mut_ptr(),
                prg.as_ptr(),
                20,
            )
        };
        assert_eq!(exit, AsmExit::Cycles as u32);
        assert!(cpu.pc >= 0xC134, "PC should have jumped to ≥ 0xC134, got {:#06X}", cpu.pc);
    }

    /// End-to-end integration: synthetic 32 KB NROM ROM whose reset
    /// vector points at a loop using only ASM-ported opcodes. After
    /// stepping N CPU instructions through Nes::step, the ASM_HITS
    /// counter must be > 0 — proves the integration bridge runs
    /// inside the real emulator path, not just the test harness.
    #[cfg(feature = "asm_hit_counter")]
    #[test]
    fn integration_asm_hits_on_real_nes_step() {
        use std::sync::atomic::Ordering;

        let before = ASM_HITS.load(Ordering::Relaxed);

        // Program at $C000:
        //   A9 05       LDA #$05
        //   85 10       STA $10
        //   E6 10       INC $10
        //   A5 10       LDA $10
        //   C9 0A       CMP #$0A
        //   D0 F8       BNE -8 (→ back to INC)
        //   EA          NOP (loop marker)
        //   4C 00 C0    JMP $C000 (infinite loop)
        let mut init = DiffInit::default();
        init.prg_bytes = vec![
            (0xC000, 0xA9), (0xC001, 0x05),
            (0xC002, 0x85), (0xC003, 0x10),
            (0xC004, 0xE6), (0xC005, 0x10),
            (0xC006, 0xA5), (0xC007, 0x10),
            (0xC008, 0xC9), (0xC009, 0x0A),
            (0xC00A, 0xD0), (0xC00B, 0xF8),
            (0xC00C, 0xEA),
            (0xC00D, 0x4C), (0xC00E, 0x00), (0xC00F, 0xC0),
        ];
        let rom = build_ines_rom(&init);
        let cart = Cartridge::load(&mut std::io::Cursor::new(rom)).unwrap();
        let mut nes = Nes::new(cart);
        nes.reset();

        // After reset, PC is at the reset vector ($C000 per our ROM).
        let mut vs = DiscardSinks;
        let mut aus = DiscardSinks;
        // Run ~500 CPU instructions' worth of Nes::step calls.
        for _ in 0..500 {
            nes.step(&mut vs, &mut aus);
        }

        let after = ASM_HITS.load(Ordering::Relaxed);
        let delta = after - before;
        assert!(
            delta > 50,
            "ASM fast path should have fired many times, got {} hits",
            delta,
        );
        eprintln!(
            "integration_asm_hits_on_real_nes_step: {} instructions ran through ASM",
            delta,
        );
    }

    /// Unimpl exit — verify the ASM core bails correctly on any
    /// opcode we haven't ported yet.
    #[test]
    fn unimpl_exits_cleanly() {
        install_opcode_table();
        let mut ram = vec![0u8; 2048];
        let mut prg = vec![0u8; 32 * 1024];

        // 0x02 is KIL/JAM — illegal opcode, not ported. Place at PC
        // $C000 which maps to prg[0x4000] for a 32 KB NROM.
        prg[0x4000] = 0x02;

        let mut cpu = AsmCpuState::default();
        cpu.pc = 0xC000;
        cpu.sp = 0xFD;
        cpu.p = 0x24;

        let exit = unsafe {
            nes_cpu_run_block(
                &mut cpu as *mut _,
                ram.as_mut_ptr(),
                prg.as_ptr(),
                100,
            )
        };
        assert_eq!(exit, AsmExit::Unimpl as u32);
        assert_eq!(cpu.unimpl_opcode, 0x02);
    }
}
