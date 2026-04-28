//! End-to-end smoke: load zelda.nes, boot the NES, run ~60 frames
//! (~1 second of game time at 60Hz), confirm CPU+PPU+APU produced
//! the expected output and report rough throughput.

use nes_core::cartridge::Cartridge;
use nes_core::nes::Nes;
use nes_core::ppu;
use nes_core::sink::{AudioSink, VideoSink, Xrgb8888VideoSink};

const FRAME_PIXELS: usize = ppu::SCREEN_WIDTH * ppu::SCREEN_HEIGHT;

/// Captures raw audio samples produced during stepping; counts bytes
/// and tracks peak amplitude so we can prove the APU is alive.
struct CountingAudio {
    samples: Vec<f32>,
    peak: f32,
}
impl AudioSink for CountingAudio {
    fn write_sample(&mut self, s: f32) {
        if s.abs() > self.peak {
            self.peak = s.abs();
        }
        self.samples.push(s);
    }
    fn samples_written(&self) -> usize {
        self.samples.len()
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../roms/zelda.nes".to_string());
    let bytes = std::fs::read(&path).expect("read rom");
    let cart = Cartridge::load(&mut std::io::Cursor::new(bytes)).expect("parse iNES");
    println!("Loaded {}: mapper={} mirroring={:?}", path, cart.mapper, cart.mirroring);

    let mut nes = Nes::new(cart);

    let mut video_buf = vec![0u32; FRAME_PIXELS];
    let mut audio = CountingAudio {
        samples: Vec::new(),
        peak: 0.0,
    };
    let target_frames = 60usize;
    let mut frames_done = 0usize;

    let start = std::time::Instant::now();
    while frames_done < target_frames {
        // Sink lifetime must be shorter than the step call. New sink per frame.
        let mut video = Xrgb8888VideoSink::new(&mut video_buf);
        // Step the NES one CPU instruction at a time until a frame lands.
        while !video.frame_written() {
            nes.step(&mut video, &mut audio);
        }
        frames_done += 1;
    }
    let dur = start.elapsed();

    let nonzero_pixels = video_buf.iter().filter(|&&p| p != 0).count();
    println!(
        "{} frames in {:.3}s = {:.1} fps; cpu_cycles={}; audio_samples={} (peak {:.3}); nonzero pixels in last frame: {}/{}",
        frames_done,
        dur.as_secs_f64(),
        frames_done as f64 / dur.as_secs_f64(),
        nes.cycles,
        audio.samples_written(),
        audio.peak,
        nonzero_pixels,
        FRAME_PIXELS,
    );
}
