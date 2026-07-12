"""RustPool adapter boundary tests — _materialize + spectator audio/pacing.

Exercises the adapter against mocked raw step tuples, so no rebuilt
nes_core binary is required. Guards the hot-path invariants:

1. skip_preprocess sentinel: Rust ships a 0-length preprocessed array
   in tile mode; the adapter must substitute the shared zeros((84, 84))
   singleton (shape invariant preserved, no per-slot allocation) while
   the f16 / uint8 pixel paths stay byte-identical.

2. Audio-drain gating: only workers whose audio output is on get a
   per-step drain_audio FFI call, fed by BOTH set_worker_pace (Rust
   welds audio to pacing on a pace *transition*) and set_worker_audio
   (audio only, last call wins), with a single tail-flush drain on the
   step after audio turns off. The mechanism must stay per-worker and
   dynamic — any subset of workers (including all) can be enabled
   simultaneously.

3. set_pace_multiplier: mirrors the Rust clamp ([0.25, 16.0],
   non-finite resets to 1.0) and stamps
   StepResult.audio_rate = BASE_AUDIO_RATE * m.

4. Old .so (methods missing) => silent no-op everywhere.

The inner nes_core.Pool is replaced by a Python fake, so these tests
verify the ADAPTER contract. What only a live run can verify: that the
rebuilt .so actually paces at the multiplier, that Mode::All mixes many
workers audibly, and that the GUI grid stays smooth — see the manual
checklist in the integration notes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emulation.rust_pool_adapter import (
    _AUDIO_EMPTY,
    _PP_SKIP_SINGLETON,
    BASE_AUDIO_RATE,
    RustPool,
)


class FakeInner:
    """Stands in for nes_core.Pool: records drain/pace/audio/knob calls
    and serves canned int16 audio bytes per worker."""

    def __init__(self, num_workers: int = 3) -> None:
        self.num_workers = num_workers
        self.drain_calls: list[int] = []
        self.pace_calls: list[tuple[int, bool]] = []
        self.audio_calls: list[tuple[int, bool]] = []
        self.mult_calls: list[float] = []
        self.skip_calls: list[bool] = []
        self.audio_bytes: dict[int, bytes] = {}

    def fill_ring(self, worker_id: int, samples: list[int]) -> None:
        """Append int16 samples to a worker's pending audio ring."""
        self.audio_bytes[worker_id] = (
            self.audio_bytes.get(worker_id, b"")
            + np.asarray(samples, dtype=np.int16).tobytes()
        )

    def step_all(self, actions):
        return [
            (
                np.zeros((240, 256, 3), dtype=np.uint8),
                np.zeros((84, 84), dtype=np.uint8),
                bytes(2048),
                False,
            )
            for _ in range(self.num_workers)
        ]

    def drain_audio(self, worker_id: int) -> bytearray:
        self.drain_calls.append(worker_id)
        return bytearray(self.audio_bytes.pop(worker_id, b""))

    def set_worker_pace(self, worker_id: int, on: bool) -> None:
        self.pace_calls.append((worker_id, on))

    def set_worker_audio(self, worker_id: int, on: bool) -> None:
        self.audio_calls.append((worker_id, on))

    def set_pace_multiplier(self, mult: float) -> None:
        self.mult_calls.append(mult)

    def set_spectator_subframe_skip(self, on: bool) -> None:
        self.skip_calls.append(on)


class OldInner(FakeInner):
    """Simulates a stale .so predating the spectator audio API."""

    def __getattribute__(self, name):
        if name in (
            "set_worker_audio",
            "set_pace_multiplier",
            "set_spectator_subframe_skip",
        ):
            raise AttributeError(name)
        return super().__getattribute__(name)


def _pool(num_workers: int = 3, inner: FakeInner | None = None) -> tuple[RustPool, FakeInner]:
    p = RustPool(rom_path="unused.nes", num_workers=num_workers)
    if inner is None:
        inner = FakeInner(num_workers)
    p._inner = inner
    return p, inner


def _frame() -> np.ndarray:
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _slot(pp: np.ndarray, ram: bytes = b"\x00" * 8, done: bool = False):
    return (_frame(), pp, ram, done)


def _step(pool: RustPool):
    return pool.step_all([0] * pool.num_workers)


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
    assert results[1].audio_rate == BASE_AUDIO_RATE
    assert results[0].audio is _AUDIO_EMPTY
    assert results[2].audio is _AUDIO_EMPTY


def test_pace_off_flushes_residue_then_stops_draining() -> None:
    """Toggling pace off must drain the residue accumulated since the
    last step ONCE — on the next step, so the tail SHIPS in that
    StepResult instead of being discarded at toggle time — and then
    stop the per-step drain."""
    p, inner = _pool(num_workers=2)
    p.set_worker_pace(0, True)
    inner.audio_bytes[0] = np.array([1, 2], dtype=np.int16).tobytes()
    p.set_worker_pace(0, False)
    # No drain at toggle time — the flush is deferred to the next step
    # so the tail samples reach the mixer instead of vanishing.
    assert inner.drain_calls == []
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(2)]
    results = p._materialize(raw)
    assert inner.drain_calls == [0]
    assert results[0].audio.tolist() == [1, 2]
    assert results[0].audio_rate == BASE_AUDIO_RATE
    # ...and subsequent steps no longer drain worker 0.
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
        assert r.audio_rate == BASE_AUDIO_RATE


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
    raw = [_slot(np.zeros((84, 84), dtype=np.uint8)) for _ in range(3)]
    p._materialize(raw)
    assert inner.drain_calls == []
    assert (2, False) in inner.pace_calls


def test_out_of_range_pace_does_not_pollute_tracking() -> None:
    """Rust silently ignores out-of-range worker ids; the adapter's
    bookkeeping must mirror that so _materialize never indexes a
    phantom worker."""
    p, inner = _pool(num_workers=2)
    p.set_worker_pace(99, True)
    assert p._audio_workers == set()
    assert p._paced == set()
    p.set_worker_audio(99, True)
    assert p._audio_workers == set()


def test_shutdown_clears_audio_tracking() -> None:
    """A start/shutdown cycle boots fresh workers with audio off — the
    audio/pace bookkeeping must not leak across pool lifetimes."""
    p, _ = _pool(num_workers=2)
    p.set_worker_pace(0, True)
    p.set_worker_audio(1, True)
    p.set_pace_multiplier(2.0)
    assert p._audio_workers == {0, 1}
    assert p._paced == {0}
    assert p._pace_multiplier == 2.0
    p.set_worker_audio(1, False)  # pending tail flush
    assert p._audio_flush == {1}
    p.shutdown()
    assert p._audio_workers == set()
    assert p._paced == set()
    assert p._audio_flush == set()
    assert p._pace_multiplier == 1.0


def test_audio_empty_singleton_is_readonly() -> None:
    with pytest.raises(ValueError):
        _AUDIO_EMPTY.resize(4)


# ---------------------------------------------------------------------------
# Part 3 — set_worker_audio (audio without pacing)
# ---------------------------------------------------------------------------


def test_set_worker_audio_forwards_and_gates_drain() -> None:
    p, inner = _pool(num_workers=3)
    inner.fill_ring(1, [10, -10, 20])
    p.set_worker_audio(1, True)
    assert inner.audio_calls == [(1, True)]
    assert inner.pace_calls == [], "set_worker_audio must not touch pacing"
    results = _step(p)
    assert inner.drain_calls == [1], "only the enabled worker is drained"
    assert results[1].audio.tolist() == [10, -10, 20]
    assert results[1].audio_rate == BASE_AUDIO_RATE
    assert results[0].audio.size == 0 and results[2].audio.size == 0


def test_audio_off_flushes_ring_once() -> None:
    p, inner = _pool(num_workers=3)
    p.set_worker_audio(0, True)
    _step(p)
    inner.fill_ring(0, [5, 6])  # tail produced before the flip lands
    p.set_worker_audio(0, False)
    results = _step(p)
    # Flush-once: the tail ships instead of sitting in the ring.
    assert results[0].audio.tolist() == [5, 6]
    inner.drain_calls.clear()
    _step(p)
    assert inner.drain_calls == [], "after the flush the ring is not drained"


def test_pace_weld_fires_on_transition_only() -> None:
    """Mirrors Rust: set_worker_pace early-outs when the pace state is
    unchanged, so a redundant pace call must NOT clobber a later
    set_worker_audio override (last call wins)."""
    p, inner = _pool(num_workers=3)
    p.set_worker_pace(0, True)   # welds audio ON
    p.set_worker_audio(0, False)  # last call: audio OFF, still paced
    _step(p)  # consumes the flush-once drain
    inner.drain_calls.clear()
    p.set_worker_pace(0, True)   # no transition -> no weld
    _step(p)
    assert inner.drain_calls == [], (
        "redundant set_worker_pace(0, True) must not re-enable audio "
        "after set_worker_audio(0, False)"
    )


def test_audio_reenable_cancels_pending_flush() -> None:
    p, inner = _pool(num_workers=3)
    p.set_worker_audio(2, True)
    p.set_worker_audio(2, False)  # pending flush
    p.set_worker_audio(2, True)   # back on before any step
    inner.fill_ring(2, [1])
    results = _step(p)
    assert results[2].audio.tolist() == [1]
    inner.drain_calls.clear()
    inner.fill_ring(2, [2])
    results = _step(p)
    assert inner.drain_calls == [2], "worker must still be treated as enabled"
    assert results[2].audio.tolist() == [2]


# ---------------------------------------------------------------------------
# Part 4 — pace multiplier
# ---------------------------------------------------------------------------


def test_set_pace_multiplier_forwards_and_stamps_audio_rate() -> None:
    p, inner = _pool(num_workers=3)
    p.set_pace_multiplier(4.0)
    assert inner.mult_calls == [4.0]
    p.set_worker_audio(0, True)
    inner.fill_ring(0, [3])
    results = _step(p)
    assert results[0].audio_rate == int(BASE_AUDIO_RATE * 4.0)
    # Silent workers keep rate 0 regardless of the multiplier.
    assert results[1].audio_rate == 0


@pytest.mark.parametrize(
    "raw, applied",
    [
        (100.0, 16.0),   # clamp high
        (0.01, 0.25),    # clamp low
        (float("nan"), 1.0),   # non-finite resets
        (float("inf"), 1.0),
        (2.0, 2.0),      # in-range passthrough
    ],
)
def test_set_pace_multiplier_clamp_mirrors_rust(raw, applied) -> None:
    p, inner = _pool(num_workers=3)
    p.set_pace_multiplier(raw)
    assert inner.mult_calls == [applied]
    p.set_worker_audio(0, True)
    inner.fill_ring(0, [1])
    results = _step(p)
    assert results[0].audio_rate == int(BASE_AUDIO_RATE * applied)


# ---------------------------------------------------------------------------
# Part 5 — spectator sub-frame skip
# ---------------------------------------------------------------------------


def test_set_spectator_subframe_skip_forwards() -> None:
    p, inner = _pool(num_workers=3)
    p.set_spectator_subframe_skip(False)
    p.set_spectator_subframe_skip(True)
    assert inner.skip_calls == [False, True]


# ---------------------------------------------------------------------------
# Part 6 — back-compat: old .so / not-started pool => silent no-op
# ---------------------------------------------------------------------------


def test_old_so_new_methods_are_silent_noops() -> None:
    p, inner = _pool(num_workers=3, inner=OldInner(3))
    p.set_worker_audio(0, True)
    p.set_pace_multiplier(4.0)
    p.set_spectator_subframe_skip(False)
    inner.fill_ring(0, [1])
    results = _step(p)
    # set_worker_audio no-opped: gating untouched, ring never drained.
    assert inner.drain_calls == []
    assert results[0].audio.size == 0
    # set_pace_multiplier no-opped: an old binary paces at 1.0x, so the
    # stored multiplier must stay 1.0 (no wrong pitch-shift).
    p.set_worker_pace(0, True)  # weld still works on old .so
    inner.fill_ring(0, [1])
    results = _step(p)
    assert results[0].audio_rate == BASE_AUDIO_RATE


def test_unstarted_pool_noops() -> None:
    p = RustPool(rom_path="unused.nes", num_workers=3)
    p.set_worker_pace(0, True)
    p.set_worker_audio(0, True)
    p.set_pace_multiplier(2.0)
    p.set_spectator_subframe_skip(False)
    assert p._audio_workers == set()
    assert p._pace_multiplier == 1.0


# ---------------------------------------------------------------------------
# Part 7 — trainer wiring: mixer-mode -> pace/audio mapping, pace knob
# ---------------------------------------------------------------------------


class _TrainerStub:
    """Bare attribute bag for calling Trainer methods unbound."""

    def __init__(self, pool, num_instances=3):
        self.pool = pool
        self.num_instances = num_instances
        # _apply_pool_knobs surface
        self.preprocess_f16 = False
        self.panic_isolation = True
        self.batched_render = "off"
        self.asm_bulk_cycles = 1
        self._is_tile_mode = False
        self._frame_sink = None
        self.pace_multiplier = 1.0
        self._active_pace_multiplier = 1.0


def _trainer_cls():
    from src.training.trainer import Trainer

    return Trainer


def test_mode_all_enables_audio_everywhere_without_pacing() -> None:
    p, inner = _pool(num_workers=3)
    stub = _TrainerStub(p)
    _trainer_cls()._apply_pace_for_mode(stub, "all")
    assert inner.audio_calls == [(0, True), (1, True), (2, True)]
    assert inner.pace_calls == [], "'all' must not touch pacing"


def test_mode_mute_unpaces_and_silences_everyone() -> None:
    p, inner = _pool(num_workers=3)
    # Simulate a prior 'all' mode: audio on everywhere, nobody paced.
    for w in range(3):
        p.set_worker_audio(w, True)
    inner.audio_calls.clear()
    stub = _TrainerStub(p)
    _trainer_cls()._apply_pace_for_mode(stub, "mute")
    assert inner.pace_calls == [(0, False), (1, False), (2, False)]
    # The pace weld can't silence never-paced workers (transition-only),
    # so mute must pin audio off explicitly.
    assert inner.audio_calls == [(0, False), (1, False), (2, False)]
    assert p._audio_workers == set()


def test_mode_solo_paces_and_hears_only_the_soloed_worker() -> None:
    p, inner = _pool(num_workers=3)
    # Simulate a prior 'all' mode.
    for w in range(3):
        p.set_worker_audio(w, True)
    inner.audio_calls.clear()
    stub = _TrainerStub(p)
    _trainer_cls()._apply_pace_for_mode(stub, "solo-1")
    assert inner.pace_calls == [(0, False), (1, True), (2, False)]
    assert inner.audio_calls == [(0, False), (1, True), (2, False)]
    assert p._audio_workers == {1}


def test_mode_all_legacy_backend_falls_back_to_unpace() -> None:
    class LegacyPool:
        def __init__(self):
            self.pace_calls = []

        def set_worker_pace(self, worker_id, on):
            self.pace_calls.append((worker_id, on))

    legacy = LegacyPool()
    stub = _TrainerStub(legacy)
    _trainer_cls()._apply_pace_for_mode(stub, "all")
    assert legacy.pace_calls == [(0, False), (1, False), (2, False)]


def test_pool_knobs_apply_pace_multiplier_only_with_frame_sink() -> None:
    p, inner = _pool(num_workers=3)
    stub = _TrainerStub(p)
    stub.pace_multiplier = 2.0
    stub._frame_sink = None  # headless: must never pace
    _trainer_cls()._apply_pool_knobs(stub)
    assert inner.mult_calls == []
    assert stub._active_pace_multiplier == 1.0

    stub._frame_sink = object()
    _trainer_cls()._apply_pool_knobs(stub)
    assert inner.mult_calls == [2.0]
    assert stub._active_pace_multiplier == 2.0
    # The adapter now stamps scaled audio_rate on audible workers.
    p.set_worker_audio(0, True)
    inner.fill_ring(0, [1])
    assert _step(p)[0].audio_rate == int(BASE_AUDIO_RATE * 2.0)


def test_pool_knobs_multiplier_default_off() -> None:
    p, inner = _pool(num_workers=3)
    stub = _TrainerStub(p)
    stub.pace_multiplier = 1.0  # default = off
    stub._frame_sink = object()
    _trainer_cls()._apply_pool_knobs(stub)
    assert inner.mult_calls == [], "1.0 means the knob is not pushed"
