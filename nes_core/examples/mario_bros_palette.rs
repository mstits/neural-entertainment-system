//! Look at the actual palette indices Mario Bros writes to its frame
//! buffer + the PPU mask register state. If PPUMASK is 0 (rendering
//! disabled), all pixels are clear_pixel = 0x0F (palette index for
//! "black backdrop"), which renders as solid black through Xrgb8888.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;
struct A;
impl AudioSink for A { fn write_sample(&mut self,_:f32){} fn samples_written(&self)->usize{0} }

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../roms/Mario Bros.nes".into());
    let bytes = std::fs::read(&path).unwrap();
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).unwrap();
    let mut nes = Nes::new(cart);
    let mut buf = vec![0u32; FRAME_PIXELS];
    let mut a = A;
    for _ in 0..600 {
        let mut v = Xrgb8888VideoSink::new(&mut buf);
        while !v.frame_written() { nes.step(&mut v, &mut a); }
    }
    let unique: std::collections::HashSet<u32> = buf.iter().copied().collect();
    println!("after 600 frames: unique pixel values: {:?}", unique);
    // Read back the actual palette indices via PPU memory. The PPU
    // exposes peek_byte for palette ram at $3F00..$3F1F.
    let mut nes = Nes::new(Cartridge::load(&mut std::io::Cursor::new(std::fs::read(&path).unwrap())).unwrap());
    for _ in 0..600 {
        let mut v = Xrgb8888VideoSink::new(&mut buf);
        while !v.frame_written() { nes.step(&mut v, &mut a); }
    }
    print!("palette ram: ");
    for addr in 0x3F00..0x3F20 {
        print!("{:02x} ", nes.ppu.peek_byte(addr));
    }
    println!();
}
