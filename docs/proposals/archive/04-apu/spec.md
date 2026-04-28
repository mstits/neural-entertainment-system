# Split 04 — APU (5 Channels)

## Context

Source proposal: [`../full_rust_refactor.md`](../full_rust_refactor.md)
Interview decisions: [`../deep_project_interview.md`](../deep_project_interview.md)
Manifest: [`../project-manifest.md`](../project-manifest.md)

Implements the NES Audio Processing Unit. This is the split that
finally delivers real game audio at training speed — the central
goal that motivated the whole refactor.

## Key decisions inherited from interview / proposal

- **All 5 channels.** Pulse 1, Pulse 2, Triangle, Noise, DMC.
- **Stereo output** (proposal goal). NES native is mono; we mix to
  stereo with a configurable per-channel pan (default: all center).
- **Sample rate configurable** (default 44100 Hz to match the
  existing AudioMixer + transport layout).
- **APU gating.** When the AudioMixer reports no active subscriber,
  the APU mixing path can be skipped entirely as a perf win. The
  underlying register state still ticks correctly so a later switch
  to "subscribed" doesn't restart from silence.

## Deliverables

1. **Frame counter** in 4-step and 5-step modes with the documented
   IRQ behavior. This is the timer that drives envelope, length
   counter, and sweep updates across all channels.
2. **Pulse 1 + Pulse 2** with envelope, sweep, length counter, duty
   cycle. The two pulse channels share an identical implementation
   parameterized by sweep negation behavior.
3. **Triangle** with linear counter + length counter.
4. **Noise** with LFSR shift register (15-bit) and length counter.
5. **DMC (Delta Modulation Channel).** Sample playback from PRG ROM,
   with the documented CPU stall (+1 to +4 cycles) on each DMA fetch.
   This affects CPU timing — the bus must surface the stall.
6. **Mixer** producing int16 stereo PCM at the configured sample
   rate. Use the standard non-linear mixing formulas from nesdev
   wiki (sums-then-LUT, not arithmetic mean).
7. **Audio buffer accessor:** `pub fn drain_audio(&mut self) -> &[i16]`
   returning samples produced since the last drain. Matches the
   existing `nesrs.get_audio()` shape so split 05's PyO3 wrapper
   can route directly into `AudioMixer.push_audio()`.

## Dependencies

- **Provided by 01-foundation-fork-and-cpu:**
  - `Bus` access for DMC sample fetches and the stall surface.
  - CPU cycle pacing (APU advances per CPU cycle).
- **None of splits 02 or 03 are blocking** — APU is functionally
  independent of PPU and (mostly) of mappers. DMC fetches use
  PRG ROM addressing which any mapper supports.

## Provides to other splits

- `Apu` type with `step(cpu_cycles: u32)` advancing internal state.
- `drain_audio()` for the integration split's PyO3 layer to publish
  through the existing transport audio slot.
- A working audio path that lets the AudioMixer's `push_audio()`
  route real ROM audio — replacing the synth/NSF fallback that
  exists today.

## Risks for this split

- **Frame counter quirks.** The 5-step mode does NOT generate IRQs
  but length counters tick on a different cadence. Easy to confuse
  with 4-step.
- **DMC DMA stall.** Adding 1-4 cycles to the CPU on DMC fetches
  changes Mario's controller-read timing in a way that some test
  ROMs detect. Mitigation: blargg's `dmc_dma_during_read4` test.
- **Mixing math.** The non-linear mixing produces audibly different
  output than naive sum-and-clip; the synth-fallback we've been
  shipping does naive mixing. Use the LUT formulas.
- **Sample-rate resampling.** APU output is at 1.789773 MHz / N for
  some N depending on the channel period; downsampling to 44100 Hz
  cleanly without aliasing requires a low-pass filter. nesdev wiki
  documents the standard approach; copy it.

## Acceptance criteria

1. blargg's `apu_test`, `apu_mixer`, `dmc_basics`,
   `dmc_dma_during_read4` test ROMs pass.
2. `drain_audio()` on Zelda's title screen produces samples whose
   spectrum (FFT peak) matches the NSF playback of the title theme
   to within 5% on the dominant frequency.
3. APU gating: with `gate_disabled=true`, `drain_audio()` returns
   empty and the channel state machines still advance correctly so
   re-enabling produces continuous audio (no gap or restart).
4. DMC fetches correctly stall the CPU; a test that runs `nestest`-
   style instruction count comparison with vs without DMC active
   shows the documented cycle delta.
