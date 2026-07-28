"""In-process NES worker pool backed by `nes_core.Pool`.

`nes_core.Pool` runs N NES instances in-process via rayon and returns
frames/RAM in a single PyO3 call per step. No subprocess fork, no
shared memory — the trainer thread and the GUI thread share the same
address space and see results directly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import nes_core

# APU sample production rate at 1.0× pacing (NTSC CPU clock / 41).
# StepResult.audio_rate scales this by the active pace multiplier so
# the mixer's resampler pitch-shifts instead of overflowing its ring.
BASE_AUDIO_RATE = 43653

# Shared observation served in skip_preprocess (tile) mode. The Rust
# pool ships a 0-length sentinel instead of a fresh 7-14 KB zero buffer
# per worker per step (tile policies read `ram_snapshot`, never
# `preprocessed`); every StepResult in skip mode aliases this single
# read-only array so the (84, 84) shape invariant holds for any stray
# consumer. Marked non-writeable so an accidental in-place write fails
# loud instead of corrupting every other slot sharing the buffer.
_PP_SKIP_SINGLETON = np.zeros((84, 84), dtype=np.uint8)
_PP_SKIP_SINGLETON.setflags(write=False)

# Shared empty audio buffer for workers whose audio output is off (the
# default: Pool workers boot with audio disabled; set_worker_pace and
# set_worker_audio enable it). Skipping their per-step drain_audio
# saves an FFI call + bytearray copy + np.frombuffer per worker per
# step. Read-only: it is aliased into every muted worker's StepResult.
_AUDIO_EMPTY = np.zeros(0, dtype=np.int16)
_AUDIO_EMPTY.setflags(write=False)

# Full RGB frame shape (SCREEN_HEIGHT, SCREEN_WIDTH, 3). The Rust
# done-worker short-circuit skips emulation and ships a 0-length /
# (1, 1, 3) sentinel instead — anything that is not this shape is a
# sentinel the adapter must not surface to a spectator grid as-is.
_FRAME_SHAPE = (240, 256, 3)


@dataclass
class StepResult:
    worker_id: int
    frame: np.ndarray
    ram_snapshot: bytes
    done: bool
    preprocessed: np.ndarray
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    audio_rate: int = 0


class RustPool:
    """In-process NES pool backed by `nes_core.Pool`."""

    def __init__(
        self,
        rom_path: str,
        num_workers: int,
        frame_skip: int = 4,
        env_spec: str = "nes_core:NESEnvironment",
        start_state_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not env_spec.startswith("nes_core"):
            raise ValueError(
                f"RustPool only supports nes_core env_spec; got {env_spec!r}"
            )
        # A configured start state MUST exist. nes_core.Pool would raise
        # a RuntimeError at start() anyway (fs read failure), but that
        # fires deep inside the training thread; checking at
        # construction makes a stale path a clear STARTUP error naming
        # the file, and guarantees no caller can ever fall back to a
        # silent cold boot (which trains the title-screen demo / the
        # wrong level). None keeps its meaning: deliberate cold boot.
        if start_state_path and not Path(start_state_path).is_file():
            raise FileNotFoundError(
                f"start_state_path={start_state_path!r} does not exist or is "
                "not a regular file. Refusing to build the pool: proceeding "
                "would cold-boot and silently train the wrong content. Fix "
                "the path or pass start_state_path=None to cold-boot "
                "deliberately."
            )
        self.rom_path = rom_path
        self.num_workers = num_workers
        self.frame_skip = frame_skip
        self.start_state_path = start_state_path
        self.env_spec = env_spec
        self._inner: Optional[nes_core.Pool] = None
        self._seed = seed
        # Workers whose audio output is currently enabled. Only these
        # get a per-step drain_audio call in _materialize; everyone
        # else's APU sample generation is off in Rust (Worker::new
        # disables it), so their sample buffer stays empty and there is
        # nothing to drain. Fed by BOTH set_worker_pace (whose Rust
        # side welds audio to pacing on a pace *transition*) and
        # set_worker_audio (audio only). Between the two, the last call
        # wins, mirroring the Rust semantics. Per-worker and dynamic:
        # any subset of workers — including all of them — can be
        # enabled simultaneously.
        self._audio_workers: set[int] = set()
        # Mirrors each worker's realtime_pace flag so the adapter
        # reproduces the Rust transition-only pace↔audio weld exactly
        # (a redundant set_worker_pace call never clobbers a later
        # set_worker_audio override).
        self._paced: set[int] = set()
        # Per-worker (monotonic-time, n_samples) of the last drain plus
        # an EMA of measured production rate — feeds
        # _effective_audio_rate so audio_rate tracks reality.
        self._audio_rate_track: dict[int, tuple[float, int]] = {}
        self._audio_rate_ema: dict[int, float] = {}
        # Worker ids whose audio just turned off: their ring is drained
        # ONCE more on the next step (shipping any tail samples instead
        # of letting them sit and burst out on a later re-enable).
        self._audio_flush: set[int] = set()
        # Active pool pacing speed. Stamped onto StepResult.audio_rate
        # (BASE_AUDIO_RATE × multiplier) so the mixer's resampler
        # consumes exactly as fast as paced workers produce. Only ever
        # set by set_pace_multiplier AFTER the inner call landed — an
        # old .so that ignores the multiplier keeps rate 1.0×.
        self._pace_multiplier: float = 1.0
        # Last full-resolution frame shipped per worker, keyed by index.
        # A done worker short-circuits emulation in Rust and returns a
        # (1, 1, 3) / 0-length sentinel; _materialize hands that worker
        # its cached previous frame (zero-copy reference reuse) so a
        # spectator tile keeps its last live image instead of going
        # black. Populated only when full frames arrive (non-headless),
        # cleared on start / shutdown / reset_all.
        self._last_frames: dict[int, np.ndarray] = {}

    def _clear_audio_tracking(self) -> None:
        """Reset all audio/pacing bookkeeping to a fresh pool's state
        (audio off everywhere, nobody paced, multiplier 1.0)."""
        self._audio_workers.clear()
        self._paced.clear()
        self._audio_flush.clear()
        self._pace_multiplier = 1.0
        self._audio_rate_track.clear()
        self._audio_rate_ema.clear()

    def start(self) -> None:
        if self._inner is not None:
            return
        # A fresh inner pool boots with audio off on every worker, no
        # pacing, and multiplier 1.0 — drop any state cached from a
        # previous inner (shutdown -> start respawn) so drain gating
        # and audio_rate stamping match the new pool's reality.
        self._clear_audio_tracking()
        self._last_frames.clear()
        self._inner = nes_core.Pool(
            rom_path=self.rom_path,
            num_workers=self.num_workers,
            frame_skip=self.frame_skip,
            start_state_path=self.start_state_path,
        )

    def shutdown(self) -> None:
        self._inner = None
        self._clear_audio_tracking()
        self._last_frames.clear()

    @property
    def num_dead(self) -> int:
        """Workers currently marked dead (panicked, not yet revived).
        reset_all revives them each iter, so a persistently-nonzero value
        flags a worker that keeps re-panicking."""
        if self._inner is None:
            return 0
        try:
            return int(self._inner.num_dead)
        except Exception:
            return 0

    @property
    def ga_worker_offset(self) -> int:
        return 0

    @property
    def ga_worker_count(self) -> int:
        return self.num_workers

    def reset_all(self) -> list[StepResult]:
        if self._inner is None:
            raise RuntimeError("RustPool.start() must be called first")
        # New generation/rollout: drop cached spectator frames so a
        # done worker from the previous rollout can't leak its last
        # image forward. reset ships full frames that repopulate the
        # cache in the _materialize call below.
        self._last_frames.clear()
        return self._materialize(self._inner.reset_all())

    def step_all(self, actions: Iterable[int]) -> list[StepResult]:
        if self._inner is None:
            raise RuntimeError("RustPool.start() must be called first")
        # Zero-copy fast path: pre-built numpy uint8 array flows
        # through to Rust as a `PyReadonlyArray1<u8>` — no PyLong
        # unbox, no Rust Vec alloc. Training hot path always passes
        # this shape. Fallback preserves legacy list/iterable callers.
        if isinstance(actions, np.ndarray) and actions.dtype == np.uint8:
            if actions.shape != (self.num_workers,):
                raise ValueError(
                    f"actions length {actions.shape[0]} != num_workers "
                    f"{self.num_workers}"
                )
            payload = actions if actions.flags["C_CONTIGUOUS"] else np.ascontiguousarray(actions)
            return self._materialize(self._inner.step_all(payload))
        actions = list(actions)
        if len(actions) != self.num_workers:
            raise ValueError(
                f"actions length {len(actions)} != num_workers {self.num_workers}"
            )
        return self._materialize(
            self._inner.step_all([int(a) & 0xFF for a in actions])
        )

    def _materialize(self, raw: list) -> list[StepResult]:
        results: list[StepResult] = []
        audio_workers = self._audio_workers
        for i, (frame, pp, ram_bytes, done) in enumerate(raw):
            # Spectator frame reuse. A done worker short-circuits its
            # step in Rust and ships a (1, 1, 3) / 0-length sentinel
            # rather than re-rendering the ~180 KB framebuffer. Cache
            # each worker's last full frame and hand a sentinel-shipping
            # worker that cached reference back (zero-copy) so a
            # spectator tile keeps its last live image instead of going
            # black until the next reset_all. Headless training never
            # ships full frames, so the cache stays empty and this is a
            # no-op on the hot path.
            if frame.shape == _FRAME_SHAPE:
                self._last_frames[i] = frame
            else:
                cached = self._last_frames.get(i)
                if cached is not None:
                    frame = cached
            if pp.size == 0:
                # skip_preprocess (tile mode) sentinel: Rust ships a
                # 0-length array instead of a dead 7-14 KB zero buffer
                # per worker per step. Serve the shared zeros singleton
                # so StepResult.preprocessed keeps its (84, 84) shape
                # invariant for any stray consumer.
                pp = _PP_SKIP_SINGLETON
            elif pp.dtype == np.uint8 and pp.shape == (84, 168):
                # Zero-copy reinterpret from the Rust fallback uint8 array
                pp = pp.view(np.float16).reshape((84, 84))
            elif pp.shape != (84, 84) or pp.dtype not in (np.float16, np.uint8):
                # Fail loud at the adapter boundary on malformed slots.
                # A worker crash returning a weird shape would otherwise
                # propagate into the CNN as an opaque "expected (4, 84,
                # 84), got (X, Y, Z)" deep in the forward pass.
                # Accept either dtype on the (84,84) shape — uint8 is
                # the legacy/non-preprocess-f16 path; float16 is the
                # preprocess-f16 path. Both are valid live formats.
                raise ValueError(
                    f"worker {i}: unexpected preprocessed shape/dtype "
                    f"{pp.shape}/{pp.dtype}; expected (84,84)/{{float16,uint8}} "
                    f"or (84,168)/uint8 fallback"
                )
            # Only audio-enabled workers produce samples (Rust gates
            # APU sample output on set_worker_pace / set_worker_audio);
            # everyone else's buffer is structurally empty, so skip the
            # FFI drain + copy for them. Freshly-disabled workers get
            # one final flush so the ring's tail ships instead of
            # sitting until a later re-enable.
            if i in audio_workers:
                audio = self.drain_audio(i)
            elif i in self._audio_flush:
                audio = self.drain_audio(i)
                self._audio_flush.discard(i)
            else:
                audio = _AUDIO_EMPTY
            results.append(StepResult(
                worker_id=i,
                frame=frame,
                ram_snapshot=bytes(ram_bytes),
                done=bool(done),
                preprocessed=pp,
                audio=audio,
                audio_rate=(
                    self._effective_audio_rate(i, audio.size)
                    if audio.size > 0 else 0
                ),
            ))
        return results

    def _effective_audio_rate(self, worker_id: int, n_samples: int) -> int:
        """Rate the mixer must consume this worker's audio at so playback
        keeps up with PRODUCTION, measured, not assumed.

        The old stamp (BASE × pace-multiplier) was only correct for paced
        workers. An unpaced training/search worker runs 50-200× realtime,
        so stamping 1× flooded the mixer ring 200:1 — near-total sample
        drops that presented as silence/garble in the GUI's "all" mode.
        Measuring samples/wall-second per worker makes every regime
        coherent: paced workers measure ≈ BASE (unchanged pitch), unpaced
        ones play as machine-speed fast-forward — all samples heard, no
        overflow. EMA-smoothed so per-step jitter doesn't warble the
        resampler; clamped to [BASE/2, 64×BASE]."""
        now = time.monotonic()
        prev = self._audio_rate_track.get(worker_id)
        self._audio_rate_track[worker_id] = (now, n_samples)
        if prev is None:
            return int(BASE_AUDIO_RATE * self._pace_multiplier)
        dt = now - prev[0]
        if dt <= 1e-4:
            return int(BASE_AUDIO_RATE * self._pace_multiplier)
        inst = n_samples / dt
        ema_prev = self._audio_rate_ema.get(worker_id, inst)
        ema = 0.7 * ema_prev + 0.3 * inst
        self._audio_rate_ema[worker_id] = ema
        return int(min(max(ema, BASE_AUDIO_RATE * 0.5),
                       BASE_AUDIO_RATE * 64))

    def set_worker_done(self, worker_id: int, done: bool) -> None:
        """Mark a worker as episode-done so subsequent step_all calls
        skip its NES emulation. Cleared automatically on the next
        reset_all. Trainer should call this once a genome's episode
        terminates so we don't burn frame_skip × remaining-steps NES
        cycles on dead workers.
        """
        if self._inner is None:
            return
        try:
            self._inner.set_worker_done(int(worker_id), bool(done))
        except AttributeError:
            # Older nes_core build without the method — silently
            # no-op so Python doesn't crash if someone runs against
            # a stale install.
            pass

    def save_worker_state(
        self, worker_id: int, timeout: float = 5.0,
    ) -> Optional[bytes]:
        if self._inner is None:
            return None
        blob = self._inner.save_worker_state(worker_id)
        return bytes(blob) if blob is not None else None

    def load_worker_state(self, worker_id: int, data: bytes) -> None:
        if self._inner is None:
            return
        self._inner.load_worker_state(worker_id, bytes(data))

    def _note_audio(self, worker_id: int, on: bool) -> None:
        """Record a worker's audio-enable state for drain gating."""
        if on:
            self._audio_workers.add(worker_id)
            self._audio_flush.discard(worker_id)
        elif worker_id in self._audio_workers:
            self._audio_workers.discard(worker_id)
            # Ring may still hold samples produced before the flip —
            # drain it once more on the next step, then stop.
            self._audio_flush.add(worker_id)

    def set_worker_pace(self, worker_id: int, on: bool) -> None:
        """Toggle realtime pacing for one worker. Pacing is enforced at
        pool level (one sleep per step_all), so any number of workers
        may be paced. Back-compat side effect preserved from the Rust
        side: a pace *transition* also flips the worker's audio output
        to match. set_worker_audio afterwards overrides (last call
        wins)."""
        if self._inner is None:
            return
        wid = int(worker_id)
        on = bool(on)
        self._inner.set_worker_pace(wid, on)
        if not (0 <= wid < self.num_workers):
            # Rust silently ignores out-of-range ids; mirror that so
            # the audio/pace bookkeeping stays 1:1 with actual pool
            # state.
            return
        # Mirror the Rust weld exactly: audio flips only when the pace
        # state actually changes (Rust early-outs on no-op calls, so a
        # redundant set_worker_pace never clobbers a later
        # set_worker_audio override).
        if (wid in self._paced) != on:
            if on:
                self._paced.add(wid)
            else:
                self._paced.discard(wid)
            self._note_audio(wid, on)

    def set_worker_audio(self, worker_id: int, on: bool) -> None:
        """Enable/disable APU sample output for one worker WITHOUT
        touching its pacing — lets the mixer's `all` mode hear many
        workers while pool-level pacing stays under separate control.
        Emulation semantics (RAM, NMI/IRQ, sprite-0, DMC/frame-counter
        IRQs) are bit-identical either way. Old .so => silent no-op
        (drain gating is left untouched so we never drain a ring the
        binary can't fill)."""
        if self._inner is None:
            return
        setter = getattr(self._inner, "set_worker_audio", None)
        if setter is None:
            return
        wid = int(worker_id)
        on = bool(on)
        setter(wid, on)
        if not (0 <= wid < self.num_workers):
            # Rust silently ignores out-of-range ids; mirror that.
            return
        self._note_audio(wid, on)

    def set_pace_multiplier(self, mult: float) -> None:
        """Set the pool pacing speed multiplier (1.0 = realtime). The
        clamp mirrors Rust ([0.25, 16.0]; non-finite resets to 1.0) so
        the audio_rate stamped on StepResult always matches the speed
        the pool actually paces at. Old .so => silent no-op, including
        the stored multiplier (the binary would still pace at 1.0×, so
        scaling audio_rate would pitch-shift wrongly)."""
        if self._inner is None:
            return
        setter = getattr(self._inner, "set_pace_multiplier", None)
        if setter is None:
            return
        m = float(mult)
        if not math.isfinite(m):
            m = 1.0
        else:
            m = min(16.0, max(0.25, m))
        setter(m)
        self._pace_multiplier = m

    def set_spectator_subframe_skip(self, on: bool) -> None:
        """Toggle final-frame-only rendering for paced spectator
        workers (Rust default ON). Emulation-neutral — only framebuffer
        writes on non-final sub-frames are elided, and only the final
        sub-frame ever ships to Python. Old .so => silent no-op."""
        if self._inner is None:
            return
        setter = getattr(self._inner, "set_spectator_subframe_skip", None)
        if setter is None:
            return
        setter(bool(on))

    def peek_max_x_per_worker(self) -> Optional[list[int]]:
        """Peak SMB world-x position seen during the most recent
        step_all call, per worker. Returns None if the underlying
        nes_core build doesn't expose the method (older binary).

        Used by the trainer to override the reward function's x
        view, capturing transient mid-frame_skip peaks that the
        final-frame-only RAM read would miss.
        """
        if self._inner is None:
            return None
        try:
            return list(self._inner.peek_max_x_per_worker())
        except AttributeError:
            return None

    def drain_audio(self, worker_id: int) -> np.ndarray:
        if self._inner is None:
            return np.zeros(0, dtype=np.int16)
        ba = self._inner.drain_audio(worker_id)
        return np.frombuffer(bytes(ba), dtype=np.int16)
