"""`progress: {atomic: true}` — routes the two-byte progress sample
through the Rust torn-read-guarded `Pool.peek_u16_consistent` binding
instead of composing it from a step snapshot. Opt-in, default off
everywhere; raises a clear error rather than silently falling back to
the torn read the flag exists to avoid when the installed nes_core
build predates the binding.

See scripts/go_explore_solve.py: GenericGame.progress_atomic_read and
the "ATOMIC PEEK" splice in Solver.observe(). Mirrors the fixture
style of tests/test_go_explore_solve_repairs.py's torn-read section.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from scripts.go_explore_solve import GenericGame, Solver

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"


def _cv_profile(**progress):
    p = {"lo": 0x0040, "hi": 0x0041}
    p.update(progress)
    return {"solve": {"rom": "roms/x.nes", "progress": p, "y": 0x003F,
                      "level_key": [0x0028], "lives": 0x002A}}


def _cv_game(**progress) -> GenericGame:
    return GenericGame(_cv_profile(**progress))


# --- profile surface ---------------------------------------------------

def test_atomic_is_off_by_default():
    assert _cv_game().progress_atomic is False


def test_atomic_true_is_read_off_the_profile():
    assert _cv_game(atomic=True).progress_atomic is True


def test_atomic_needs_a_two_byte_progress_read():
    prof = {"solve": {"rom": "roms/x.nes",
                      "progress": {"lo": 0x0040, "atomic": True},
                      "y": 0x3F, "level_key": [0x28], "lives": 0x2A}}
    with pytest.raises(SystemExit, match="two-byte progress read"):
        GenericGame(prof)


# --- GenericGame.progress_atomic_read -----------------------------------

def test_progress_atomic_read_delegates_to_the_pool_binding():
    calls = []

    class _Pool:
        def peek_u16_consistent(self, wid, lo, hi):
            calls.append((wid, lo, hi))
            return (999, True)

    g = _cv_game(atomic=True)
    assert g.progress_atomic_read(_Pool(), 3) == (999, True)
    assert calls == [(3, 0x0040, 0x0041)]


def test_progress_atomic_read_passes_through_an_inconsistent_verdict():
    class _Pool:
        def peek_u16_consistent(self, wid, lo, hi):
            return (0x02FF, False)

    g = _cv_game(atomic=True)
    assert g.progress_atomic_read(_Pool(), 0) == (0x02FF, False)


def test_progress_atomic_read_refuses_a_pool_without_the_binding():
    g = _cv_game(atomic=True)
    with pytest.raises(SystemExit, match="peek_u16_consistent"):
        g.progress_atomic_read(SimpleNamespace(), 0)


# --- end to end through observe() ---------------------------------------

class _KeyArchive:
    def __init__(self) -> None:
        self.cells: dict = {}

    def __len__(self) -> int:
        return len(self.cells)

    def record(self, ram, state, score, steps, key=None) -> bool:
        new = key not in self.cells
        self.cells[key] = SimpleNamespace(key=key, best_score=score,
                                          best_steps=steps, state=state)
        return new


def _observe_solver(pool, atomic=True, smooth="off"):
    game = _cv_game(atomic=atomic, smooth=smooth) if smooth != "off" \
        else _cv_game(atomic=atomic)
    f = SimpleNamespace(
        game=game, archive=_KeyArchive(), traces={}, roots={}, pool=pool,
        start_wd=(0,), start_lives=5,
        max_area=0, max_gx_in_area={}, max_sect=0, _pin_time=0.0,
        ortho_mode="off", gate_mode="off", door_weight=0.0,
        time_bins=False, kill_key=False, _recorded_new=False,
        progress_smooth=smooth, progress_jump=128,
        _prog_glitches=0, _c_local=set())
    f.observe = MethodType(Solver.observe, f)
    return f


def _ram(v: int, lives: int = 5) -> bytearray:
    ram = bytearray(2048)
    ram[0x002A] = lives
    ram[0x0041], ram[0x0040] = v >> 8, v & 0xFF
    return ram


def test_atomic_read_wins_over_a_torn_raw_snapshot():
    # The raw bytes in `ram` are the gx-767 tear itself; the atomic peek
    # says the true, adjudicated value is 751. The recorded cell must
    # reflect 751, never 767.
    pool = SimpleNamespace(peek_u16_consistent=lambda wid, lo, hi: (751, True),
                           save_worker_state=lambda wid: b"blob")
    f = _observe_solver(pool)
    status = f.observe(0, _ram(0x02FF), [], 0, "entrance", ctx={})
    assert status == "live"
    assert f.max_gx_in_area[0] == 751
    assert 767 not in {k[-1] for k in f.archive.cells}


def test_an_inconsistent_verdict_is_dropped_like_a_transition_frame():
    pool = SimpleNamespace(
        peek_u16_consistent=lambda wid, lo, hi: (0x02FF, False),
        save_worker_state=lambda wid: b"blob")
    f = _observe_solver(pool)
    status = f.observe(0, _ram(0x02FF), [], 0, "entrance", ctx={})
    assert status == "live"
    assert f.max_gx_in_area == {}
    assert len(f.archive) == 0


def test_atomic_peek_is_called_with_the_worker_id_and_the_profiles_addrs():
    calls = []

    def _peek(wid, lo, hi):
        calls.append((wid, lo, hi))
        return (500, True)

    pool = SimpleNamespace(peek_u16_consistent=_peek,
                           save_worker_state=lambda wid: b"blob")
    f = _observe_solver(pool)
    f.observe(7, _ram(500), [], 0, "entrance", ctx={})
    assert calls == [(7, 0x0040, 0x0041)]


def test_atomic_supersedes_smoothing_the_pair_is_never_re_sampled():
    # Even with a smooth mode configured, the atomic-consistent value is
    # already adjudicated: observe() must not also run it through
    # progress_pair / progress_glitch.
    pool = SimpleNamespace(peek_u16_consistent=lambda wid, lo, hi: (751, True),
                           save_worker_state=lambda wid: b"blob")
    f = _observe_solver(pool, atomic=True, smooth="median3")
    calls = []
    real = f.game.progress_pair
    f.game.progress_pair = lambda ram: (calls.append(1), real(ram))[1]
    f.observe(0, _ram(751), [], 0, "entrance", ctx={})
    assert calls == []
    assert f._prog_glitches == 0


def test_atomic_off_never_touches_the_pool_binding():
    def _boom(wid, lo, hi):
        raise AssertionError("peek_u16_consistent must not be called")

    pool = SimpleNamespace(peek_u16_consistent=_boom,
                           save_worker_state=lambda wid: b"blob")
    f = _observe_solver(pool, atomic=False)
    status = f.observe(0, _ram(751), [], 0, "entrance", ctx={})
    assert status == "live"
    assert f.max_gx_in_area[0] == 751


def test_atomic_true_without_the_binding_raises_a_clear_error():
    pool = SimpleNamespace(save_worker_state=lambda wid: b"blob")
    f = _observe_solver(pool)
    with pytest.raises(SystemExit, match="peek_u16_consistent"):
        f.observe(0, _ram(751), [], 0, "entrance", ctx={})


# --- real binding smoke test (no mocks) ----------------------------------

@pytest.mark.skipif(not _SMB_ROM.exists(), reason="SMB ROM not available")
def test_the_real_pool_binding_exists_and_is_callable():
    # Proves the rebuilt nes_core actually exports peek_u16_consistent
    # (the hasattr checks above are only meaningful if this is true) and
    # that GenericGame.progress_atomic_read drives it correctly against
    # a live worker, not a stub.
    from nes_core import Pool

    pool = Pool(rom_path=str(_SMB_ROM), num_workers=1, frame_skip=4)
    assert hasattr(pool, "peek_u16_consistent")
    g = _cv_game(atomic=True, lo=0x0086, hi=0x006D)
    value, consistent = g.progress_atomic_read(pool, 0)
    assert isinstance(value, int) and 0 <= value <= 0xFFFF
    assert isinstance(consistent, bool)
    with pytest.raises(IndexError):
        pool.peek_u16_consistent(1, 0x0086, 0x006D)
    with pytest.raises(ValueError):
        pool.peek_u16_consistent(0, 0x0800, 0x0086)
