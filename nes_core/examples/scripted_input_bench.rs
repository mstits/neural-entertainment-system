//! Play a ROM with a scripted input sequence, capture final-frame hash.
//! Use to verify ASM vs pure-Rust byte-exact under interactive input.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu;
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;
const BTN_START: u8 = 0x10;

struct Null;
impl AudioSink for Null {
    fn write_sample(&mut self, _: f32) {}
    fn samples_written(&self) -> usize { 0 }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../roms/zelda.ines1.nes".into());
    let frames: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(600);

    let bytes = std::fs::read(&path).expect("rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("ines");
    let mut nes = Nes::new(cart);

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = Null;

    for i in 0..frames {
        let pad = nes.game_pad_1();
        use nes_core::input::Button;
        // Press Start on frame 120 for 10 frames; release after.
        let start_pressed = (120..130).contains(&i);
        pad.set_button_pressed(Button::Start, start_pressed);

        let mut video = Xrgb8888VideoSink::new(&mut video_buf);
        while !video.frame_written() {
            nes.step(&mut video, &mut audio);
        }
    }

    // FNV-1a of final frame
    let mut h: u64 = 0xcbf29ce484222325;
    for p in &video_buf {
        h ^= *p as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let nonzero = video_buf.iter().filter(|&&p| p != 0).count();
    println!("{} frames | {} nonzero / {} | FNV-1a {:016x}", frames, nonzero, FRAME_PIXELS, h);
}
