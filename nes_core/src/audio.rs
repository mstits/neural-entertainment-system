//! Rust-native AudioMixer, replacing `src/audio/ram_music.py`.
//!
//! Consumes per-instance PCM pushed from the trainer hot loop and
//! plays it through a Core Audio output stream (via `cpal`). Modes
//! mirror the Python mixer — `mute`, `solo-<id>`, `all` — and the
//! public surface is a drop-in for the Python class so the trainer +
//! GUI audio-mixer window talk to the Rust backend unchanged.
//!
//! What this mixer DOES have (ported faithfully):
//!   * Per-instance PCM ring with tight "keep only recent audio"
//!     trimming (~150 ms max buffer). Under fast-forward emulation,
//!     older chunks get dropped so the listener hears current-gamestate
//!     audio instead of a 2 s backlog.
//!   * Linear resampler from source rate (typically 43,653 Hz, the
//!     NES APU's native rate) to the output device rate. Stateful —
//!     maintains the fractional source-position carry between pushes
//!     so successive calls concatenate seamlessly.
//!   * Smooth fade-to-silence on underrun (5 ms linear ramp) — no
//!     audible click when the producer falls behind.
//!   * Master volume + per-instance intensity smoothing for the
//!     fitness-momentum audio cue.
//!
//! What this mixer DOES NOT have (vs the legacy Python):
//!   * Chiptune synth fallback.
//!   * libgme NSF playback.
//!
//! Both were hacks to give nes-py (no APU) something audible. nes_core
//! has a real APU so synth/NSF paths are unused — dropping them cuts
//! complexity + the libgme + pynoise deps.
//!
//! Stereo output: the ring buffer stores interleaved L/R `(f32, f32)`
//! pairs. The legacy `push_audio_i16` entry point still takes mono and
//! fills the ring with dual-mono pairs for backward compatibility with
//! callers that haven't plumbed per-channel APU data through yet. A
//! new `push_audio_channels_i16` entry point accepts 5 per-channel
//! streams and applies the standard NES-stereo pan matrix
//! (pulse1→L, pulse2→R, triangle→center, noise→center-left,
//! dmc→center-right) so true panned stereo reaches the cpal output
//! when the producer side is wired up.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Configurable output sample rate. 44100 matches the mixer's historical
/// Python default. Any rate the output device supports is fine.
const OUTPUT_RATE: u32 = 44_100;
/// Ring-buffer cap per instance, in output-rate samples (~150 ms).
const RING_CAP_SAMPLES: usize = (OUTPUT_RATE as usize * 150) / 1000;
/// Underrun fade length: 5 ms of linear ramp from last-sample to zero.
const UNDERRUN_FADE_SAMPLES: usize = (OUTPUT_RATE as usize * 5) / 1000;

/// Playback mode — one of `Mute`, `Solo(i)`, or `All`.
#[derive(Clone, Copy)]
enum Mode {
    Mute,
    Solo(usize),
    All,
}

struct InstanceState {
    // Stereo PCM ring — each entry is one (L, R) frame at OUTPUT_RATE.
    ring: VecDeque<(f32, f32)>,
    has_pcm: bool,
    // Last L/R frame played, for the underrun fade start value.
    last_sample: (f32, f32),
    // Remaining samples in the current underrun fade (0 when not in fade).
    fade_remaining: usize,
    // Linear-resampler carry — last mono source sample of the previous
    // push, kept per stream. `mono_carry` is used by `push_audio_i16`;
    // `channel_carries` is used by `push_audio_channels_i16` (one slot
    // per APU channel).
    mono_carry: Option<f32>,
    #[allow(dead_code)]
    channel_carries: [Option<f32>; 5],
    // Smoothed per-instance gain (fitness-momentum driven).
    intensity_target: f32,
    intensity_smoothed: f32,
}

impl InstanceState {
    fn new() -> Self {
        Self {
            ring: VecDeque::new(),
            has_pcm: false,
            last_sample: (0.0, 0.0),
            fade_remaining: 0,
            mono_carry: None,
            channel_carries: [None; 5],
            intensity_target: 1.0,
            intensity_smoothed: 1.0,
        }
    }
}

/// NES-stereo pan matrix. One (L_gain, R_gain) per APU channel in the
/// order `[pulse1, pulse2, triangle, noise, dmc]`. Matches the Famitracker
/// / Mesen-S hard-pan convention most NES-stereo mods adopt:
///   pulse1 full-left, pulse2 full-right, triangle center (with a
///   small gain cut for balance), noise center-left, DMC center-right.
#[allow(dead_code)]
const PAN_MATRIX: [(f32, f32); 5] = [
    (1.0, 0.0), // pulse1
    (0.0, 1.0), // pulse2
    (0.7, 0.7), // triangle
    (0.8, 0.5), // noise
    (0.5, 0.8), // dmc
];

struct SharedState {
    instances: Vec<InstanceState>,
    mode: Mode,
    volume: f32,
}

/// Commands sent from any thread to the dedicated stream-control
/// worker thread. cpal::Stream is `!Send` (AudioUnit / run-loop
/// affinity), so the Stream is pinned to a single thread that owns
/// it. Setter methods (`set_mode` etc.) become thread-safe because
/// they only enqueue a command; the worker does the !Send work.
enum StreamCmd {
    Start(mpsc::Sender<Result<(), String>>),
    Stop,
    Shutdown,
}

fn stream_control_thread(
    rx: mpsc::Receiver<StreamCmd>,
    state: Arc<Mutex<SharedState>>,
) {
    let mut stream: Option<cpal::Stream> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            StreamCmd::Stop => {
                if let Some(s) = stream.take() {
                    let _ = s.pause();
                    drop(s);
                }
            }
            StreamCmd::Start(reply) => {
                // Always tear down + rebuild so the stream picks up
                // the CURRENT default output device — users plug in
                // headphones between mode toggles.
                if let Some(s) = stream.take() {
                    let _ = s.pause();
                    drop(s);
                }
                let result = build_output_stream(&state);
                match result {
                    Ok(s) => {
                        stream = Some(s);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                }
            }
            StreamCmd::Shutdown => {
                if let Some(s) = stream.take() {
                    let _ = s.pause();
                    drop(s);
                }
                break;
            }
        }
    }
}

fn build_output_stream(state: &Arc<Mutex<SharedState>>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default audio output device".to_string())?;
    // Stereo output. The ring holds L/R pairs; each pair was either
    // produced with the standard NES-stereo pan matrix (when the
    // producer supplies per-channel APU data via
    // `push_audio_channels_i16`) or as dual-mono from the mono
    // `push_audio_i16` path. The callback drains pairs straight into
    // the cpal interleaved-stereo buffer.
    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(OUTPUT_RATE),
        buffer_size: cpal::BufferSize::Default,
    };
    let state = Arc::clone(state);
    let stream = device
        .build_output_stream(
            &config,
            move |out: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                fill_callback(&state, out);
            },
            move |err| {
                eprintln!("[nes_core::AudioMixer] stream error: {err}");
            },
            None,
        )
        .map_err(|e| format!("build_output_stream: {e}"))?;
    stream.play().map_err(|e| format!("stream.play: {e}"))?;
    Ok(stream)
}

/// Rust audio mixer. Owns the output stream + one PCM ring per
/// training instance. Public methods mirror the Python class so the
/// PyO3 wrapper in `python.rs` can forward calls 1:1.
///
/// Internally, cpal::Stream lives on a dedicated worker thread and
/// is driven by mpsc commands, so this struct is `Send + Sync` and
/// can be touched from any thread the Python side hands it to
/// (the trainer's audio-drainer thread, for instance). Previously
/// the pyclass was marked `unsendable`, which caused a panic when
/// the drainer thread called `set_mode`.
pub struct AudioMixer {
    state: Arc<Mutex<SharedState>>,
    cmd_tx: mpsc::Sender<StreamCmd>,
    control_thread: Option<JoinHandle<()>>,
    num_instances: usize,
}

impl AudioMixer {
    pub fn new(num_instances: usize) -> Self {
        let state = Arc::new(Mutex::new(SharedState {
            instances: (0..num_instances).map(|_| InstanceState::new()).collect(),
            mode: Mode::Mute,
            volume: 0.4,
        }));
        let (cmd_tx, cmd_rx) = mpsc::channel::<StreamCmd>();
        let thread_state = Arc::clone(&state);
        let control_thread = thread::Builder::new()
            .name("nes-core-audio-stream".into())
            .spawn(move || stream_control_thread(cmd_rx, thread_state))
            .expect("spawn audio-stream thread");
        Self {
            state,
            cmd_tx,
            control_thread: Some(control_thread),
            num_instances,
        }
    }

    pub fn set_mode(&self, mode_str: &str) -> Result<(), String> {
        let mode = if mode_str == "mute" {
            Mode::Mute
        } else if mode_str == "all" {
            Mode::All
        } else if let Some(suffix) = mode_str.strip_prefix("solo-") {
            let idx: usize = suffix.parse().map_err(|_| format!("bad solo index: {mode_str}"))?;
            Mode::Solo(idx)
        } else {
            return Err(format!("unknown mode: {mode_str}"));
        };
        {
            let mut s = self.state.lock().unwrap();
            s.mode = mode;
        }
        match mode {
            Mode::Mute => {
                self.cmd_tx
                    .send(StreamCmd::Stop)
                    .map_err(|e| format!("audio worker gone: {e}"))?;
            }
            _ => {
                let (reply_tx, reply_rx) = mpsc::channel();
                self.cmd_tx
                    .send(StreamCmd::Start(reply_tx))
                    .map_err(|e| format!("audio worker gone: {e}"))?;
                match reply_rx.recv_timeout(Duration::from_secs(3)) {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(e),
                    Err(_) => return Err("audio stream build timeout".into()),
                }
            }
        }
        Ok(())
    }

    pub fn set_volume(&self, v: f32) {
        let mut s = self.state.lock().unwrap();
        s.volume = v.clamp(0.0, 1.0);
    }

    pub fn set_instance_intensity(&self, instance_id: usize, intensity: f32) {
        if instance_id >= self.num_instances {
            return;
        }
        let mut s = self.state.lock().unwrap();
        s.instances[instance_id].intensity_target = intensity.clamp(0.1, 1.8);
    }

    pub fn num_instances(&self) -> usize {
        self.num_instances
    }

    /// Clear a voice's buffered PCM and resampler carry state. Worker
    /// voice indices are REUSED across level transitions (a new Solver
    /// spins up a fresh Pool on the same 0..workers slots); without this,
    /// the first push of the new level's audio linearly interpolates from
    /// `mono_carry` — the last sample of the PREVIOUS level's unrelated
    /// waveform — producing an audible splice artifact (a brief
    /// deep/buzzy transient, live-show report 2026-08-06). Leaves
    /// `intensity_target`/`intensity_smoothed` untouched — those track
    /// the current mix level (set by `set_instance_intensity`), not
    /// stream content.
    pub fn reset_instance(&self, instance_id: usize) {
        if instance_id >= self.num_instances {
            return;
        }
        let mut s = self.state.lock().unwrap();
        let inst = &mut s.instances[instance_id];
        inst.ring.clear();
        inst.has_pcm = false;
        inst.last_sample = (0.0, 0.0);
        inst.fade_remaining = 0;
        inst.mono_carry = None;
        inst.channel_carries = [None; 5];
    }

    /// Current mode as a string compatible with the GUI + trainer
    /// callers: `"mute"`, `"solo-<n>"`, or `"all"`.
    pub fn mode_str(&self) -> String {
        let s = self.state.lock().unwrap();
        match s.mode {
            Mode::Mute => "mute".into(),
            Mode::All => "all".into(),
            Mode::Solo(i) => format!("solo-{i}"),
        }
    }

    /// Push PCM samples from the emulator for one training instance.
    /// Expects a flat int16 mono slice; resamples from `src_rate` to
    /// the output rate if needed, then stores as dual-mono `(L, R)`
    /// pairs — the legacy path gets identical L/R. Trims the ring to
    /// the most recent ~150 ms so fast-forward emulation can't queue
    /// multi-second backlogs.
    pub fn push_audio_i16(&self, instance_id: usize, src: &[i16], src_rate: u32) {
        if src.is_empty() || src_rate == 0 || instance_id >= self.num_instances {
            return;
        }
        // Convert to float in [-1, 1].
        let src_f32: Vec<f32> = src.iter().map(|&x| x as f32 / 32768.0).collect();
        let mut s = self.state.lock().unwrap();
        let inst = &mut s.instances[instance_id];
        let resampled = resample_linear(src_f32, src_rate, OUTPUT_RATE, &mut inst.mono_carry);
        if resampled.is_empty() {
            return;
        }
        for sample in resampled {
            inst.ring.push_back((sample, sample));
        }
        // Trim oldest to stay below the cap.
        while inst.ring.len() > RING_CAP_SAMPLES && inst.ring.len() > 1 {
            inst.ring.pop_front();
        }
        inst.has_pcm = true;
    }

    /// Push per-APU-channel PCM for true panned stereo output. Accepts
    /// five same-length int16 mono slices in the order
    /// `[pulse1, pulse2, triangle, noise, dmc]` — matching the return
    /// order of `Apu::generate_sample_channels`. Resamples each channel
    /// independently (channel carries are tracked per-stream so
    /// fractional boundaries don't bleed across channels), then
    /// applies the pan matrix and stores the resulting `(L, R)` pairs.
    ///
    /// The five slices must be equal-length; mismatched lengths are
    /// ignored to avoid producing L/R-mismatched frames. `src_rate` is
    /// the shared rate of all five channels (typically
    /// `apu::SAMPLE_RATE`, ~43,653 Hz).
    #[allow(dead_code)]
    pub fn push_audio_channels_i16(
        &self,
        instance_id: usize,
        channels: &[&[i16]; 5],
        src_rate: u32,
    ) {
        if src_rate == 0 || instance_id >= self.num_instances {
            return;
        }
        let len = channels[0].len();
        if len == 0 {
            return;
        }
        // Mismatched lengths would produce L/R-skewed frames; refuse.
        if !channels.iter().all(|c| c.len() == len) {
            return;
        }
        let mut s = self.state.lock().unwrap();
        let inst = &mut s.instances[instance_id];

        // Resample each channel with its own carry.
        let mut resampled: [Vec<f32>; 5] = Default::default();
        let mut min_n = usize::MAX;
        for (ch_idx, src) in channels.iter().enumerate() {
            let src_f32: Vec<f32> = src.iter().map(|&x| x as f32 / 32768.0).collect();
            let out = resample_linear(
                src_f32,
                src_rate,
                OUTPUT_RATE,
                &mut inst.channel_carries[ch_idx],
            );
            min_n = min_n.min(out.len());
            resampled[ch_idx] = out;
        }
        if min_n == 0 {
            return;
        }
        // Walk the shortest-common prefix across the five channels and
        // sum per-frame with the pan matrix so L and R get independent
        // weights. Any over-shoot in a longer channel's resampled
        // buffer is dropped; its carry still advanced so the next push
        // picks up where it left off.
        for i in 0..min_n {
            let mut l = 0.0f32;
            let mut r = 0.0f32;
            for ch_idx in 0..5 {
                let v = resampled[ch_idx][i];
                let (lg, rg) = PAN_MATRIX[ch_idx];
                l += v * lg;
                r += v * rg;
            }
            inst.ring.push_back((l, r));
        }
        while inst.ring.len() > RING_CAP_SAMPLES && inst.ring.len() > 1 {
            inst.ring.pop_front();
        }
        inst.has_pcm = true;
    }

    pub fn stop_stream(&self) {
        let _ = self.cmd_tx.send(StreamCmd::Stop);
    }
}

impl Drop for AudioMixer {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(StreamCmd::Shutdown);
        if let Some(h) = self.control_thread.take() {
            let _ = h.join();
        }
    }
}

/// Cpal audio callback — fills the device buffer by pulling PCM from
/// the current mode's instance(s). On underrun, fades to silence
/// instead of clicking.
fn fill_callback(state: &Arc<Mutex<SharedState>>, out: &mut [f32]) {
    let mut s = state.lock().unwrap();
    // `out` is interleaved stereo (L, R, L, R, ...). Actual audio
    // frames = out.len() / 2.
    let frames = out.len() / 2;

    if matches!(s.mode, Mode::Mute) {
        for sample in out.iter_mut() {
            *sample = 0.0;
        }
        return;
    }

    // Smooth per-instance intensity toward target (one step per
    // callback, ~23 ms at 1024-sample buffer × 44 kHz).
    let alpha = 0.2f32;
    for inst in &mut s.instances {
        inst.intensity_smoothed += alpha * (inst.intensity_target - inst.intensity_smoothed);
    }

    let mode = s.mode;
    let volume = s.volume;

    // Accumulate L/R frames into a stereo mix buffer.
    let mut mix = vec![(0f32, 0f32); frames];
    match mode {
        Mode::Solo(idx) => {
            if idx < s.instances.len() {
                let gain = s.instances[idx].intensity_smoothed;
                drain_instance(&mut s.instances[idx], &mut mix, gain);
            }
        }
        Mode::All => {
            let n = s.instances.len() as f32;
            let norm = 1.0 / n.sqrt().max(1.0);
            let mut tmp = vec![(0f32, 0f32); frames];
            for inst in s.instances.iter_mut() {
                let gain = inst.intensity_smoothed;
                for t in tmp.iter_mut() {
                    *t = (0.0, 0.0);
                }
                drain_instance(inst, &mut tmp, gain);
                for (m, t) in mix.iter_mut().zip(tmp.iter()) {
                    m.0 += t.0;
                    m.1 += t.1;
                }
            }
            for m in mix.iter_mut() {
                m.0 *= norm;
                m.1 *= norm;
            }
        }
        Mode::Mute => unreachable!(),
    }

    // Interleave L then R into the cpal buffer, applying master volume.
    for (i, m) in mix.iter().enumerate() {
        let l = (m.0 * volume).clamp(-1.0, 1.0);
        let r = (m.1 * volume).clamp(-1.0, 1.0);
        out[i * 2] = l;
        out[i * 2 + 1] = r;
    }
}

/// Drain up to `out.len()` stereo frames for one instance. Uses the
/// ring first, falls to a fade-from-last-sample when it underruns.
/// When the instance has never produced PCM, returns silence.
fn drain_instance(inst: &mut InstanceState, out: &mut [(f32, f32)], gain: f32) {
    let frames = out.len();
    let mut written = 0usize;
    while written < frames {
        if let Some(sample) = inst.ring.pop_front() {
            out[written] = (sample.0 * gain, sample.1 * gain);
            inst.last_sample = sample;
            written += 1;
        } else {
            break;
        }
    }

    if written == frames {
        inst.fade_remaining = 0;
        return;
    }

    // Underrun region — fade from last sample toward silence.
    let fade_total = UNDERRUN_FADE_SAMPLES;
    if written > 0 || inst.fade_remaining == 0 {
        inst.fade_remaining = fade_total;
    }
    let underrun_n = frames - written;
    let fade_n = inst.fade_remaining.min(underrun_n);
    if fade_n > 0 {
        let last_l = inst.last_sample.0 * gain;
        let last_r = inst.last_sample.1 * gain;
        let start_ratio = inst.fade_remaining as f32 / fade_total as f32;
        let end_ratio =
            ((inst.fade_remaining as isize - fade_n as isize).max(0) as f32) / fade_total as f32;
        for i in 0..fade_n {
            let t = i as f32 / fade_n as f32;
            let r = start_ratio + (end_ratio - start_ratio) * t;
            out[written + i] = (last_l * r, last_r * r);
        }
        inst.fade_remaining = inst.fade_remaining.saturating_sub(fade_n);
    }
    // Tail of underrun past the fade stays zero (already zeroed).
    // `has_pcm` remains true — the ring will refill and we resume.
    if !inst.has_pcm {
        // Never-produced instance: everything is silence.
        for s in &mut out[written..] {
            *s = (0.0, 0.0);
        }
    }
}

/// Linear-interpolation resampler from `src_rate` to `dst_rate`.
/// Stateful via `carry` — the caller passes the same `Option<f32>`
/// slot across successive pushes so the fractional source boundary is
/// preserved and output is gap-free between calls. Matches the
/// semantics of `AudioMixer._resample_linear` in Python.
fn resample_linear(
    src: Vec<f32>,
    src_rate: u32,
    dst_rate: u32,
    carry: &mut Option<f32>,
) -> Vec<f32> {
    if src_rate == dst_rate {
        return src;
    }
    let ratio = src_rate as f32 / dst_rate as f32;
    let mut input: Vec<f32> = Vec::with_capacity(src.len() + 1);
    if let Some(c) = carry.take() {
        input.push(c);
    }
    input.extend(src);
    let n_src = input.len();
    if n_src < 2 {
        *carry = input.last().copied();
        return Vec::new();
    }
    let max_out_pos = (n_src - 1) as f32 / ratio;
    let n_out = max_out_pos.floor() as usize;
    if n_out == 0 {
        *carry = input.last().copied();
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f32 * ratio;
        let idx0 = pos as usize;
        let frac = pos - idx0 as f32;
        let idx1 = (idx0 + 1).min(n_src - 1);
        let s = input[idx0] * (1.0 - frac) + input[idx1] * frac;
        out.push(s);
    }
    *carry = Some(input[n_src - 1]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity() {
        let src = vec![0.0, 1.0, 2.0, 3.0];
        let mut carry = None;
        let out = resample_linear(src.clone(), 44100, 44100, &mut carry);
        assert_eq!(out, src);
    }

    #[test]
    fn resample_upsample_2x() {
        let src: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut carry = None;
        let out = resample_linear(src, 22050, 44100, &mut carry);
        // Should be roughly twice the length
        assert!(out.len() >= 15, "expected roughly 2x samples, got {}", out.len());
    }

    #[test]
    fn mixer_push_and_ring_bound() {
        let m = AudioMixer::new(4);
        // Push enough audio to exceed the ring cap.
        let samples = vec![100i16; OUTPUT_RATE as usize]; // 1 second of audio
        m.push_audio_i16(0, &samples, 44100);
        let s = m.state.lock().unwrap();
        // Ring should have been trimmed to <= cap.
        assert!(
            s.instances[0].ring.len() <= RING_CAP_SAMPLES + 1,
            "ring grew to {}", s.instances[0].ring.len()
        );
        assert!(s.instances[0].has_pcm);
        // Mono-only push produces dual-mono frames (L == R).
        let (l, r) = s.instances[0].ring[0];
        assert_eq!(l, r);
    }

    #[test]
    fn mixer_channels_push_pans_lr_differently() {
        let m = AudioMixer::new(2);
        // pulse1 full-left, pulse2 full-right. Feed only pulse1 →
        // L should be non-zero, R should be exactly zero.
        let len = 1024usize;
        let p1 = vec![10_000i16; len];
        let zero = vec![0i16; len];
        let channels: [&[i16]; 5] = [&p1, &zero, &zero, &zero, &zero];
        m.push_audio_channels_i16(0, &channels, 44100);
        let s = m.state.lock().unwrap();
        assert!(!s.instances[0].ring.is_empty());
        let (l, r) = s.instances[0].ring[0];
        assert!(l > 0.0, "pulse1 should drive L: L={l}");
        assert_eq!(r, 0.0, "pulse1 should NOT reach R: R={r}");

        // Now flip: pulse2-only should drive R and leave L silent.
        let m2 = AudioMixer::new(1);
        let channels2: [&[i16]; 5] = [&zero, &p1, &zero, &zero, &zero];
        m2.push_audio_channels_i16(0, &channels2, 44100);
        let s2 = m2.state.lock().unwrap();
        let (l2, r2) = s2.instances[0].ring[0];
        assert_eq!(l2, 0.0, "pulse2 should NOT reach L: L={l2}");
        assert!(r2 > 0.0, "pulse2 should drive R: R={r2}");
    }
}
