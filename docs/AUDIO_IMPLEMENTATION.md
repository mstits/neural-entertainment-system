# Audio Implementation Guide

> **⚠ HISTORICAL — PRE-RUST-MIGRATION.**
>
> This document describes the old nes-py + chiptune synth + NSF
> fallback pipeline that was **deleted** during the Rust migration
> (see `proposals/batch_execution_report.md`).
>
> **Current state (2026-04-20)**:
> - Real APU audio is emitted every step by `nes_core::apu`.
> - `nes_core::audio::AudioMixer` owns a cpal / Core Audio output
>   stream with per-instance PCM rings, a stateful linear resampler
>   (43,653 Hz APU → 44,100 Hz device), and underrun-fade.
> - Python `src/audio/ram_music.py` is a 147-line thin wrapper
>   around `nes_core.AudioMixer` — trainer/GUI API unchanged.
>
> Nothing below reflects live code. Kept for historical context.

## The problem, corrected

NES's underlying emulator (**nes-py**, by Kautenja) **does not
emulate audio at all**. An earlier version of this doc assumed nes-py
wrapped LaiNES — it does not. nes-py has its own minimal NES
implementation covering only CPU, PPU, cartridge, mappers, and the main
bus. There is no APU in the C++ source, no sample buffer anywhere, and
no path to "just expose a Sound() getter" because there's nothing
producing samples.

Verified by reading the whole C++ tree at
`third_party/nes-py/nes_py/nes/{include,src}/` — zero occurrences of
`apu`, `audio`, `sound`, `channel`, or `mixer` in any file.

## What full emulator audio would actually cost

Implementing the NES APU from scratch on top of nes-py:

- 2 pulse (square-wave) channels with envelope, sweep, length-counter
- 1 triangle channel with linear counter
- 1 noise channel with shift register
- 1 DMC (delta-modulation) channel for sampled audio
- Frame sequencer (4-step and 5-step modes)
- Nonlinear mixer
- Integration with main bus + cycle timing
- Downsample from 1.79 MHz CPU clock to 48 kHz output

This is genuinely multi-week work for someone who hasn't written an APU
before. Risk of subtle timing bugs is high. Not an overnight item.

## Realistic alternatives, in rising effort order

### 1. RAM-based music detection (what we ship now)

Most NES games write the "current song to play" to a known RAM address.
Our training loop already has live access to RAM. We can:

1. Watch that RAM byte each frame.
2. When it changes, play a pre-rendered loop of the corresponding song.
3. Crossfade between songs.

You get **music changing in sync with the game** ("overworld theme",
"dungeon theme", "boss theme", "death jingle"). You don't get sound
effects (sword swings, enemy hits, etc.).

Implementation: `src/audio/ram_music.py` + `src/gui/audio_mixer_window.py`.
Zero C++ work, zero nes-py changes. Pure Python + `sounddevice`.

See the active code for the shipping implementation.

### 2. Swap the emulator entirely

Replace nes-py with Nestopia-based or Mesen-based Python bindings. Both
have real APU. But:

- Nestopia bindings (py-mednafen etc.) are unmaintained.
- Mesen is C#/.NET, no Python binding.
- You'd be rewriting every nes-py call site in the project.
- Lose the FakeEnv test harness and pgroup setup that's been validated.

Probably 2-3 days of work for a uncertain payoff.

### 3. Implement the APU on top of nes-py

Fork nes-py, add APU source files modeled after one of:

- LaiNES APU code (GPL — licence compatibility needed)
- Mesen's APU (MIT — but in C#)
- blargg's nes_emu APU (LGPL — cleanest reference)

All require wiring into nes-py's bus and cycle-counting. Multi-week.

## Song-ID addresses by game

Populated as we add ROMs. For Zelda (NES):

- `$00FC` — current sound-effect trigger byte (transient)
- `$0605` — currently-playing song ID (persistent; used by the
  game's music driver)

For Mario (NES):

- `$00FB` — sound queue for music
- `$00F7` — in-game song ID

Other games — add as you dig into their RAM maps. NesDev wiki has
most of them.
