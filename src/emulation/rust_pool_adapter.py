"""In-process NES worker pool backed by `nes_core.Pool`.

`nes_core.Pool` runs N NES instances in-process via rayon and returns
frames/RAM in a single PyO3 call per step. No subprocess fork, no
shared memory — the trainer thread and the GUI thread share the same
address space and see results directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

import nes_core

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
# default: Pool workers boot with audio disabled; only set_worker_pace
# enables it). Skipping their per-step drain_audio saves an FFI call +
# bytearray copy + np.frombuffer per worker per step.
_AUDIO_EMPTY = np.zeros(0, dtype=np.int16)
_AUDIO_EMPTY.setflags(write=False)


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
        self.rom_path = rom_path
        self.num_workers = num_workers
        self.frame_skip = frame_skip
        self.start_state_path = start_state_path
        self.env_spec = env_spec
        self._inner: Optional[nes_core.Pool] = None
        self._seed = seed
        # Workers whose audio output is currently enabled (paced via
        # set_worker_pace). Only these get a per-step drain_audio call
        # in _materialize; everyone else's APU sample generation is off
        # in Rust (Worker::new disables it; only set_worker_pace flips
        # it), so their sample buffer stays empty and there is nothing
        # to drain. Per-worker and dynamic: any subset of workers —
        # including all of them — can be paced on simultaneously.
        self._audio_workers: set[int] = set()

    def start(self) -> None:
        if self._inner is not None:
            return
        self._inner = nes_core.Pool(
            rom_path=self.rom_path,
            num_workers=self.num_workers,
            frame_skip=self.frame_skip,
            start_state_path=self.start_state_path,
        )
        # Fresh Pool workers boot with audio off; drop any stale
        # bookkeeping from a previous start/shutdown cycle.
        self._audio_workers.clear()

    def shutdown(self) -> None:
        self._inner = None
        self._audio_workers.clear()

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
            # Only paced workers produce audio (Rust gates APU sample
            # generation on set_worker_pace); everyone else's buffer is
            # structurally empty, so skip the FFI drain + copy for them.
            if i in audio_workers:
                audio = self.drain_audio(i)
            else:
                audio = _AUDIO_EMPTY
            results.append(StepResult(
                worker_id=i,
                frame=frame,
                ram_snapshot=bytes(ram_bytes),
                done=bool(done),
                preprocessed=pp,
                audio=audio,
                audio_rate=43653 if audio.size > 0 else 0,
            ))
        return results

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

    def set_worker_pace(self, worker_id: int, on: bool) -> None:
        if self._inner is None:
            return
        wid = int(worker_id)
        self._inner.set_worker_pace(wid, bool(on))
        if not (0 <= wid < self.num_workers):
            # Rust silently ignores out-of-range ids; mirror that so
            # the audio bookkeeping stays 1:1 with actual pool state.
            return
        if on:
            self._audio_workers.add(wid)
        elif wid in self._audio_workers:
            # Pace-off just disabled sample generation in Rust. Drain
            # one final time to flush residue accumulated since the
            # last per-step drain, then stop draining this worker.
            self.drain_audio(wid)
            self._audio_workers.discard(wid)

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
