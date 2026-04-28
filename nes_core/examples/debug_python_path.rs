//! Reproduce the exact loop python.rs uses to find the regression.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu::{SCREEN_HEIGHT, SCREEN_WIDTH};
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = SCREEN_WIDTH * SCREEN_HEIGHT;

struct CapAudio {
    n: usize,
}
impl AudioSink for CapAudio {
    fn write_sample(&mut self, _: f32) {
        self.n += 1;
    }
    fn samples_written(&self) -> usize {
        self.n
    }
}

fn advance_one_frame(nes: &mut Nes, video_buf: &mut Vec<u32>, audio: &mut CapAudio) {
    let mut video = Xrgb8888VideoSink::new(video_buf);
    while !video.frame_written() {
        nes.step(&mut video, audio);
    }
}

fn main() {
    let bytes = std::fs::read("../roms/zelda.nes").unwrap();
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).unwrap();
    let mut nes = Nes::new(cart);
    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = CapAudio { n: 0 };

    let frame_skip = 4;
    nes.reset();
    audio.n = 0;

    // Sample nonzero count at multiple frame milestones to see if
    // Zelda's intro animation transitions through black screens.
    for milestone in [60, 120, 180, 300, 500, 800] {
        let mut nes = Nes::new(
            Cartridge::load(
                &mut std::io::Cursor::new(std::fs::read("../roms/zelda.nes").unwrap()),
            )
            .unwrap(),
        );
        let mut video_buf = vec![0u32; FRAME_PIXELS];
        let mut audio = CapAudio { n: 0 };
        nes.reset();
        for _ in 0..milestone {
            advance_one_frame(&mut nes, &mut video_buf, &mut audio);
        }
        let nz = video_buf.iter().filter(|&&p| p != 0).count();
        println!("after {milestone:>4} frames (no skip): nonzero = {nz}");
    }

    let nonzero = video_buf.iter().filter(|&&p| p != 0).count();
    println!("after 200 steps fs=4 with skip-render: nonzero u32 pixels = {nonzero} / {FRAME_PIXELS}");
}
