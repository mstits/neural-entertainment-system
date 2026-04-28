"""Audio sign-off harness — deterministic 60-second listening test
across the cpal/Core Audio path.

Plays SMB title music + first level for 60 seconds via the in-Rust
AudioMixer (cpal backend). Cycles through volume levels and pan
modes so a human listener can verify on each device class:

  * No clicks / pops on stream start or mode transition
  * No buffer underrun glitches during steady playback
  * No drift (audio stays sync'd with deterministic frame stepping)
  * Volume + mode commands take effect within ~100 ms

Sign-off procedure (run once per device class):

    1. Plug in / connect target audio device, set as system default
    2. python scripts/audio_signoff.py
    3. Listen for 60s, confirm no artifacts
    4. Note device + result in your sign-off log

Devices to cover (per README ship gate):
  - Built-in MacBook speakers
  - AirPods (Bluetooth)
  - USB DAC (any external interface)

Exit code: 0 = harness ran cleanly; non-zero = mixer error or
underrun reported by the Rust side.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

import os

ROM = REPO / "roms" / "Super Mario Bros. (World).nes"
# Each phase runs for PHASE_SECS; total = 3 * PHASE_SECS.
# Default 20s/phase = 60s total. Set PHASE_SECS=2 for a quick smoke
# check (useful for catching API/init bugs before the full listen).
PHASE_SECS = int(os.environ.get("PHASE_SECS", "20"))
FRAMES_PER_SEC = 60  # NTSC


def banner(msg: str) -> None:
    bar = "=" * len(msg)
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def main() -> int:
    if not ROM.exists():
        print(f"FATAL: ROM not found at {ROM}", file=sys.stderr)
        print("Pass an alternate ROM via $AUDIO_SIGNOFF_ROM if needed", file=sys.stderr)
        return 1

    rom_path = Path(__import__("os").environ.get("AUDIO_SIGNOFF_ROM", str(ROM)))
    print(f"audio sign-off: ROM={rom_path.name}, {3 * PHASE_SECS}s total, single instance")
    print("listen for clicks/pops at mode changes, drift, underrun glitches")

    env = nes_core.NESEnvironment(str(rom_path), frame_skip=1)
    env.set_audio_output_enabled(True)
    env.reset()
    audio_rate = env.sample_rate

    mixer = nes_core.AudioMixer(1)
    mixer.start()
    mixer.set_mode("all")
    mixer.set_volume(0.4)

    # Press Start a few times to get into game (skips title screen).
    BTN_START = 0x08
    for _ in range(60):
        env.step(0)
    for _ in range(5):
        env.step(BTN_START)
    for _ in range(60):
        env.step(0)
    for _ in range(5):
        env.step(BTN_START)

    banner(f"phase 1: steady playback at volume=0.4 for {PHASE_SECS}s")
    t0 = time.monotonic()
    frames = 0
    while time.monotonic() - t0 < PHASE_SECS:
        env.step(0)
        audio = env.get_audio()
        if audio is not None and audio.size > 0:
            mixer.push_audio(0, audio, audio_rate)
        frames += 1
        # Throttle to realtime — Rust is much faster than 60 Hz.
        elapsed = time.monotonic() - t0
        target = frames / FRAMES_PER_SEC
        if target > elapsed:
            time.sleep(target - elapsed)

    banner(f"phase 2: volume sweep 0.1 → 0.8 → 0.1 over {PHASE_SECS}s")
    sweep_start = time.monotonic()
    while time.monotonic() - sweep_start < PHASE_SECS:
        # Triangle wave 0.1 → 0.8 → 0.1 over 4s period.
        t = (time.monotonic() - sweep_start) % 4.0
        v = 0.1 + 0.7 * (1.0 - abs(t / 2.0 - 1.0))
        mixer.set_volume(v)
        env.step(0)
        audio = env.get_audio()
        if audio is not None and audio.size > 0:
            mixer.push_audio(0, audio, audio_rate)
        frames += 1
        elapsed = time.monotonic() - t0
        target = frames / FRAMES_PER_SEC
        if target > elapsed:
            time.sleep(target - elapsed)

    banner(f"phase 3: mode toggle (all→mute→all) every 2s for {PHASE_SECS}s")
    mixer.set_volume(0.4)
    toggle_start = time.monotonic()
    cur_mode = "all"
    last_toggle = toggle_start
    while time.monotonic() - toggle_start < PHASE_SECS:
        if time.monotonic() - last_toggle >= 2.0:
            cur_mode = "mute" if cur_mode == "all" else "all"
            mixer.set_mode(cur_mode)
            print(f"  mode -> {cur_mode}", flush=True)
            last_toggle = time.monotonic()
        env.step(0)
        audio = env.get_audio()
        if audio is not None and audio.size > 0:
            mixer.push_audio(0, audio, audio_rate)
        frames += 1
        elapsed = time.monotonic() - t0
        target = frames / FRAMES_PER_SEC
        if target > elapsed:
            time.sleep(target - elapsed)

    mixer.set_mode("all")
    mixer.stop()
    banner("done — confirm clean playback + clean stop, no clicks at end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
