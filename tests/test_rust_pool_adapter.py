"""RustPool adapter boundary tests — _materialize + audio-drain gating.

Exercises the adapter against mocked raw step tuples, so no rebuilt
nes_core binary is required. Guards two hot-path invariants:

1. skip_preprocess sentinel: Rust ships a 0-length preprocessed array
   in tile mode; the adapter must substitute the shared zeros((84, 84))
   singleton (shape invariant preserved, no per-slot allocation) while
   the f16 / uint8 pixel paths stay byte-identical.

2. Audio-drain gating: only workers whose audio was enabled via
   set_worker_pace get a per-step drain_audio FFI call. Unpaced
   workers' APU sample generation is off in Rust, so their buffers
   are structurally empty and draining them is pure per-step waste.
   The mechanism must stay per-worker and dynamic — any subset of
   workers (including all) can be paced on simultaneously.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emulation.rust_pool_adapter import (
    _AUDIO_EMPTY,
    _PP_SKIP_SINGLETON,
    RustPool,
)


class FakeInner:
    """Stands in for nes_core.Pool: records drain/pace calls and serves
    canned audio bytes per worker."""

    def __init__(self) -> None:
        self.drain_calls: list[int] = []
        self.pace_calls: list[tuple[int, bool]] = []
        self.audio_bytes: dict[int, bytes] = {}

    def drain_audio(self, worker_id: int) -> bytearray:
        self.drain_calls.append(worker_id)
        return bytearray(self.audio_bytes.pop(worker_id, b""))

    def set_worker_pace(self, worker_id: int, on: bool) -> None:
        self.pace_calls.append((worker_id, on))


def _pool(num_workers: int = 3) -> tuple[RustPool, FakeInner]:
    p = RustPool(rom_path="unused.nes", num_workers=num_workers)
    inner = FakeInner()
    p._inner = inner
    return p, inner


def _frame() -> np.ndarray:
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _slot(pp: np.ndarray, ram: bytes = b"\x00" * 8, done: bool = False):
    return (_frame(), pp, ram, done)


# ---------------------------------------------------------------------------
# Part 1 — preprocessed handling
# ---------------------------------------------------------------------------


def test_empty_pp_substitutes_shared_singleton() -> None:
    """A 0-length preprocessed slot (skip_preprocess sentinel) must be
    replaced by the module-level zeros((84, 84)) singleton — same object
    in every slot, correct shape/dtype, all-zero."""
    p, _ = _pool(num_workers=2)
    raw = [_slot(np.zeros((0, 0), dtype=np.uint8)) for _ in range(2)]
    results = p._materialize(raw)
    assert len(results) == 2
    for r in results:
        assert r.preprocessed is _PP_SKIP_SINGLETON
        assert r.preprocessed.shape == (84, 84)
        assert r.preprocessed.dtype == np.uint8
        assert not r.preprocessed.any()


def test_pp_singleton_is_readonly() -> None:
    """A stray consumer writing into the shared singleton would corrupt
    every other slot aliasing it — the write must fail loud."""
    with pytest.raises(ValueError):
        _PP_SKIP_SINGLETON[0, 0] = 1


def test_uint8_pp_passes_through_byte_identical() -> None:
    """The (84, 84) uint8 pixel path must be untouched: same object,
    same bytes."""
    p, _ = _pool(num_workers=1)
    pp = np.arange(84 * 84, dtype=np.uint32).astype(np.uint8).reshape(84, 84)
    results = p._materialize([_slot(pp)])
    assert results[0].preprocessed is pp


def test_f16_fallback_pp_reinterprets_bytes() -> None:
    """The (84, 168) uint8 fallback must still view to float16 (84, 84)
    with byte-identical content."""
    p, _ = _pool(num_workers=1)
    src = np.random.default_rng(7).random((84, 84)).astype(np.float16)
    pp = src.view(np.uint8).reshape(84, 168)
    results = p._materialize([_slot(pp)])
    out = results[0].preprocessed
    assert out.shape == (84, 84)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out, src)


def test_malformed_nonempty_pp_fails_loud() -> None:
    """Non-empty malformed slots must still raise at the adapter
    boundary (worker-crash garbage must not reach the CNN)."""
    p, _ = _pool(num_workers=1)
    bad_shapes = [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((84, 84), dtype=np.float32),
        np.zeros((84, 84, 1), dtype=np.uint8),
    ]
    for bad in bad_shapes:
        with pytest.raises(ValueError, match="unexpected preprocessed"):
            p._materialize([_slot(bad)])


def test_full_zero_pp_from_old_binary_still_accepted() -> None:
    """Back-compat: a main-build binary without the sentinel returns a
    full (84, 84) zero buffer in skip mode — must pass through, not
    raise."""
    p, _ = _pool(num_workers=1)
    pp = np.zeros((84, 84), dtype=np.uint8)
    results = p._materialize([_slot(pp)])
    assert results[0].preprocessed is pp


# ---------------------------------------------------------------------------
# Part 2 — audio-drain gating
# ---------------------------------------------------------------------------


def test_unpaced_workers_never_drained() -> None:
    """Default state (no worker paced): zero drain_audio FFI calls and
    the shared empty-audio array is served."""
    p, inner = _pool(num_workers=3)
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(3)]
    results = p._materialize(raw)
    assert inner.drain_calls == []
    for r in results:
        assert r.audio is _AUDIO_EMPTY
        assert r.audio.size == 0
        assert r.audio_rate == 0


def test_paced_worker_is_drained_with_samples() -> None:
    """A paced worker gets a real per-step drain: samples decoded as
    int16 and audio_rate set."""
    p, inner = _pool(num_workers=3)
    p.set_worker_pace(1, True)
    assert (1, True) in inner.pace_calls
    inner.audio_bytes[1] = np.array([3, -7, 42], dtype=np.int16).tobytes()
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(3)]
    results = p._materialize(raw)
    assert inner.drain_calls == [1]
    np.testing.assert_array_equal(
        results[1].audio, np.array([3, -7, 42], dtype=np.int16)
    )
    assert results[1].audio_rate == 43653
    assert results[0].audio is _AUDIO_EMPTY
    assert results[2].audio is _AUDIO_EMPTY


def test_pace_off_flushes_residue_then_stops_draining() -> None:
    """Toggling pace off must drain once (flush residue accumulated
    since the last step) and then stop the per-step drain."""
    p, inner = _pool(num_workers=2)
    p.set_worker_pace(0, True)
    inner.audio_bytes[0] = np.array([1, 2], dtype=np.int16).tobytes()
    p.set_worker_pace(0, False)
    # The flush happened at toggle-off...
    assert inner.drain_calls == [0]
    assert inner.audio_bytes == {}
    # ...and subsequent steps no longer drain worker 0.
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(2)]
    results = p._materialize(raw)
    assert inner.drain_calls == [0]
    assert results[0].audio is _AUDIO_EMPTY


def test_all_workers_paced_simultaneously() -> None:
    """The mechanism is per-worker and dynamic: pacing every worker on
    drains every worker, every step."""
    n = 4
    p, inner = _pool(num_workers=n)
    for i in range(n):
        p.set_worker_pace(i, True)
    for i in range(n):
        inner.audio_bytes[i] = np.array([i], dtype=np.int16).tobytes()
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(n)]
    results = p._materialize(raw)
    assert inner.drain_calls == list(range(n))
    for i, r in enumerate(results):
        np.testing.assert_array_equal(r.audio, np.array([i], dtype=np.int16))
        assert r.audio_rate == 43653


def test_repeated_pace_on_is_idempotent() -> None:
    """Re-pacing an already-paced worker (mixer re-applies solo mode on
    every queue drain) must not duplicate drains."""
    p, inner = _pool(num_workers=2)
    p.set_worker_pace(1, True)
    p.set_worker_pace(1, True)
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(2)]
    p._materialize(raw)
    assert inner.drain_calls == [1]


def test_pace_off_on_never_paced_worker_is_noop() -> None:
    """_apply_pace_for_mode unpaces every non-soloed worker each mode
    change; workers that were never paced must not be flush-drained."""
    p, inner = _pool(num_workers=3)
    p.set_worker_pace(2, False)
    assert inner.drain_calls == []
    assert (2, False) in inner.pace_calls


def test_out_of_range_pace_does_not_pollute_tracking() -> None:
    """Rust silently ignores out-of-range worker ids; the adapter's
    bookkeeping must mirror that so _materialize never indexes a
    phantom worker."""
    p, inner = _pool(num_workers=2)
    p.set_worker_pace(99, True)
    assert p._audio_workers == set()


def test_shutdown_clears_audio_tracking() -> None:
    """A start/shutdown cycle boots fresh workers with audio off — the
    paced-worker set must not leak across pool lifetimes."""
    p, _ = _pool(num_workers=2)
    p.set_worker_pace(0, True)
    assert p._audio_workers == {0}
    p.shutdown()
    assert p._audio_workers == set()


def test_audio_empty_singleton_is_readonly() -> None:
    with pytest.raises(ValueError):
        _AUDIO_EMPTY.resize(4)
