"""
Live audio for N parallel training instances.

Thin Python façade over `nes_core.AudioMixer` (the Rust mixer, cpal
backend). The trainer + GUI mixer window call this class exactly as
they did the previous Python implementation; all the actual mixing,
resampling, ring buffering, and stream management lives in Rust now.

What used to be here (deleted because unreachable once nes_core
became the default backend):

  * `_ChiptuneGenerator` — procedural pentatonic melodies keyed on
    the game's "current song" RAM byte. Legacy fallback for envs
    without an APU. nes_core has a real APU so this path has no callers.
  * `_FileLoop` — pre-rendered .ogg/.wav loader. Same story as above.
  * `LoopPlayer` — multi-source audio source wrapper (NSF > file >
    synth). Drops out with its sources.
  * `AudioMixer` (Python) — sounddevice-based mixer with per-instance
    PCM rings, linear resampler, underrun fade. All now in Rust's
    `nes_core::audio::AudioMixer`, driven by `cpal` / Core Audio.

See `nes_core/src/audio.rs` for the live implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)


# Kept as a thin lookup for callers that still import it from this
# module (e.g. legacy tests). The Rust mixer doesn't use song-byte
# RAM reads anymore — chiptune + NSF are gone — so this is a
# compatibility shim that returns the historical address. If a caller
# doesn't know about a game, they get None and skip song updates.
_SONG_ADDRESSES = {
    "zelda": 0x0605,
    "mario": 0x00F7,
    "contra": 0x00F0,
    "megaman": 0x00F0,
    "castlevania": 0x00F0,
    "metroid": 0x0050,
}


def song_address_for(game_name: str) -> "int | None":
    lower = (game_name or "").lower()
    for frag, addr in _SONG_ADDRESSES.items():
        if frag in lower:
            return addr
    return None


class AudioMixer:
    """Live audio mixer — delegates to `nes_core.AudioMixer`.

    Surface matches the historical Python API so callers (trainer,
    GUI mixer window, tests) don't change:

        mixer = AudioMixer(num_instances=16, game="zelda")
        mixer.start()
        mixer.set_mode("solo-0")        # or "mute", "all"
        mixer.set_volume(0.4)           # [0.0, 1.0]
        mixer.set_instance_intensity(3, 1.5)
        mixer.push_audio(3, np.int16 samples, audio_rate=43653)
        mixer.push_ram(3, 2kb_ram_bytes)  # no-op (legacy)
        mixer.stop()

    `mixer.mode` and `mixer.num` are readable attributes for the
    trainer's fast-path gate (`if mixer.mode != "mute": ...`).
    """

    def __init__(
        self,
        num_instances: int,
        game: str,
        audio_root: "Path | str" = "audio",
        sample_rate: int = 44100,
    ) -> None:
        import nes_core

        self._inner = nes_core.AudioMixer(num_instances)
        self.num = num_instances
        self.game = game
        self.sample_rate = sample_rate
        # Cached mode string for the trainer's `mixer.mode != "mute"`
        # gate — avoids a PyO3 roundtrip per step. Kept in sync by
        # set_mode().
        self._mode: str = "mute"
        self.volume: float = 0.4
        # Some callers (narrator path) look up song_addr on the mixer.
        self.song_addr = song_address_for(game)

    def start(self) -> None:
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()

    def set_mode(self, mode: str) -> None:
        self._inner.set_mode(mode)
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, float(v)))
        self._inner.set_volume(self.volume)

    def set_instance_intensity(self, instance_id: int, intensity: float) -> None:
        self._inner.set_instance_intensity(instance_id, float(intensity))

    def push_audio(
        self,
        instance_id: int,
        audio,
        audio_rate: int,
    ) -> None:
        if audio is None or audio_rate <= 0:
            return
        if not isinstance(audio, np.ndarray):
            return
        # Downmix stereo → mono if caller passed a (N, 2) buffer.
        if audio.ndim == 2 and audio.shape[1] == 2:
            audio = audio.astype(np.int32).mean(axis=1).astype(np.int16)
        elif audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = np.clip(audio * 32767.0, -32767.0, 32767.0).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.int16)
        if audio.size == 0:
            return
        self._inner.push_audio(instance_id, audio, int(audio_rate))

    def push_ram(self, instance_id: int, ram_snapshot: bytes) -> None:
        # Legacy API — fed the chiptune/NSF synth in the old Python
        # mixer. nes_core's APU emits real game audio via push_audio()
        # so there's nothing to do here.
        return
