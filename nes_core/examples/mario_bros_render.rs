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
    for milestone in [1, 60, 180, 600, 1800, 3000] {
        let mut nes2 = Nes::new(Cartridge::load(&mut std::io::Cursor::new(std::fs::read(&path).unwrap())).unwrap());
        let mut buf = vec![0u32; FRAME_PIXELS];
        for _ in 0..milestone {
            let mut v = Xrgb8888VideoSink::new(&mut buf);
            while !v.frame_written() { nes2.step(&mut v, &mut a); }
        }
        let nz = buf.iter().filter(|&&p| p!=0).count();
        let unique: std::collections::HashSet<u32> = buf.iter().copied().collect();
        println!("after {milestone:>5} frames: nonzero={nz}/{FRAME_PIXELS}, unique colors={}", unique.len());
    }
}
