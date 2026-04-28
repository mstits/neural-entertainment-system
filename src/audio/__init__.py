"""Audio subsystem — thin Python wrapper around `nes_core.AudioMixer`.

The Rust NES core emits real APU samples every step; `ram_music.AudioMixer`
is the small façade the trainer + GUI use to push them at cpal (Core
Audio on macOS). See `nes_core/src/audio.rs` for the live implementation.
"""
