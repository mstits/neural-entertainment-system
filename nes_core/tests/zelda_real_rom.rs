//! Verify the actual `roms/zelda.nes` ROM file (the user's NES 2.0
//! dump) gets 8 KB battery PRG-RAM allocated. This is the integration
//! test that the live-GUI-crash fix actually applies to the real ROM
//! the user is loading, not just synthetic test headers.

use nes_core::cartridge::Cartridge;
use std::fs::File;
use std::io::BufReader;

#[test]
fn user_zelda_nes_allocates_8kb_prg_ram() {
    let path = "../roms/zelda.nes";
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            // ROM is git-ignored; skip if not present locally.
            eprintln!("skipping: {} not present", path);
            return;
        }
    };
    let mut r = BufReader::new(f);
    let cart = Cartridge::load(&mut r).expect("zelda.nes should load");

    assert!(cart.is_nes20, "zelda.nes should be detected as NES 2.0");
    assert_eq!(
        cart.prg_ram.len(),
        8192,
        "user's zelda.nes (NES 2.0, flags10=0x70) must allocate 8 KB battery PRG-RAM",
    );
    assert!(
        cart.is_battery_backed,
        "battery flag should be set on this Zelda dump",
    );
}
