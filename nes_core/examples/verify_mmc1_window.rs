//! Verify that MMC1's cached prg_asm_window matches prg_peek_byte for
//! every address after fresh load.

use nes_core::cartridge::Cartridge;
use nes_core::mapper::Mapper;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../roms/zelda.ines1.nes".into());
    let bytes = std::fs::read(&path).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    println!("ROM: {} — mapper {}, prg_rom_num_banks {}",
             path, cart.mapper, cart.prg_rom_num_banks);

    let mapper = nes_core::mapper::MapperEnum::from_cartridge(cart);

    let window_ptr = mapper.prg_asm_ptr();
    match window_ptr {
        None => { println!("mapper.prg_asm_ptr() returned None"); return; }
        Some(ptr) => {
            let window = unsafe { std::slice::from_raw_parts(ptr, 32 * 1024) };
            // For each addr in $8000..$FFFF, compare window[addr & 0x7FFF] vs prg_peek_byte(addr)
            let mut mismatches = 0usize;
            let mut first_mismatch = None;
            for addr in 0x8000u32..=0xFFFFu32 {
                let a = addr as u16;
                let window_byte = window[(a as usize) & 0x7FFF];
                let peek = mapper.prg_peek_byte(a);
                if window_byte != peek {
                    mismatches += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((a, window_byte, peek));
                    }
                }
            }
            if mismatches == 0 {
                println!("✓ Window matches prg_peek_byte across ALL $8000-$FFFF (32768 addresses)");
            } else {
                println!("✗ {} mismatches", mismatches);
                if let Some((addr, w, p)) = first_mismatch {
                    println!("  first mismatch at ${:04X}: window={:02X} peek={:02X}", addr, w, p);
                }
            }

            // Also show reset vector
            let reset_lo = window[0x7FFC];
            let reset_hi = window[0x7FFD];
            let reset_pc = (reset_lo as u16) | ((reset_hi as u16) << 8);
            let peek_lo = mapper.prg_peek_byte(0xFFFC);
            let peek_hi = mapper.prg_peek_byte(0xFFFD);
            let peek_pc = (peek_lo as u16) | ((peek_hi as u16) << 8);
            println!("Reset vector from window: ${:04X}", reset_pc);
            println!("Reset vector from peek:   ${:04X}", peek_pc);

            // Dump 16 bytes starting at reset PC
            print!("Bytes at ${:04X}: ", reset_pc);
            for i in 0..24u16 {
                let b = mapper.prg_peek_byte(reset_pc.wrapping_add(i));
                print!("{:02X} ", b);
            }
            println!();
        }
    }
}
