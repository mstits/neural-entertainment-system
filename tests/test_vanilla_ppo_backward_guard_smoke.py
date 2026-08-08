"""End-to-end wiring smoke for the backward-curriculum entropy guard.

The decision logic is unit-tested in tests/test_backward_entropy_guard.py.
What those pure tests cannot see is the half that broke in B4 v1: whether
the guard's multiplier actually reaches `entropy_coef` in the live loop,
whether it COMPOUNDS across iters (the failure mode of applying a
multiplier to persistent state), and whether an absent config block leaves
the loop exactly as it was.

Same synthetic-tape rig as tests/test_vanilla_ppo_backward_smoke.py: a
dozen states harvested by stepping the SMB entrance forward with no-ops,
4 envs, 3 iters. The tape only has to be a valid state ladder.
"""

from __future__ import annotations

import logging
import queue as _queue
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_1_1_backward.yaml"

pytestmark = pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / backward profile not present.",
)

FRAME_SKIP = 4
N_STATES = 12
ITERS = 3
# The base coefficient the 1-1 backward profile trains at. Asserted rather
# than read so a profile edit that changes it surfaces here instead of
# quietly weakening the compounding check below.
BASE_COEF = 0.005
COEF_MAX = 0.05
BOOST = 3.0
# The value mario_1_2_backward.yaml — the ONLY profile that ships the
# guard — sets for the adaptive entropy-floor controller.
SHIPPED_1_2_ENTROPY_FLOOR = 0.02


def _observed_entropy(lines: list[str]) -> float:
    """Pull the guard's trailing entropy out of the `[backward]` telemetry.

    Used only to prove the floor-controller branch a test intends to
    exercise is the one that actually fired, so an ordering assertion can
    never pass vacuously because the live entropy sat between the
    controller's two bands.
    """
    marks = [m.split("ent~", 1)[1].split()[0]
             for m in lines if m.startswith("[backward] iter") and "ent~" in m]
    assert marks, [m for m in lines if m.startswith("[backward] iter")]
    return float(marks[-1])


def _mint_synthetic_tape(out_dir: Path, start_state: Path) -> None:
    """A dozen states, one per no-op step from the entrance."""
    from nes_core import Pool

    from src.training.backward_curriculum import (
        StateEntry, state_filename, write_index,
    )

    pool = Pool(rom_path=str(_SMB_ROM), num_workers=1, frame_skip=FRAME_SKIP)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.load_worker_state(0, start_state.read_bytes())
    buf = np.zeros(1, dtype=np.uint8)
    ram = pool.step_all(buf)[0][2]
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    try:
        for step in range(N_STATES):
            name = state_filename(step)
            (out_dir / name).write_bytes(pool.save_worker_state(0))
            entries.append(StateEntry(
                step=step, frame=step * FRAME_SKIP,
                gx=(int(ram[0x006D]) << 8) | int(ram[0x0086]),
                area=int(ram[0x0760]), file=name,
            ))
            ram = pool.step_all(buf)[0][2]
    finally:
        pool.shutdown()
    write_index(out_dir, entries, {
        "level": "synthetic", "frame_skip": FRAME_SKIP, "every_frames": 4,
        "stride_steps": 1, "n_actions": N_STATES, "reached_clear": True,
    })


def _tiny_profile(
    states_dir: Path, guard: dict | None, entropy_floor: float | None = None
) -> dict:
    profile = yaml.safe_load(_PROFILE.read_text())
    rl = profile["reinforce"]
    assert float(rl["entropy_coef"]) == BASE_COEF, rl["entropy_coef"]
    assert float(rl["entropy_coef_max"]) == COEF_MAX, rl["entropy_coef_max"]
    # The 1-1 profile ships the adaptive floor controller OFF, which is
    # what lets the exact-product assertions below stay simple. The
    # profile that actually SHIPS the guard (mario_1_2_backward.yaml)
    # ships it ON, so `entropy_floor` overrides that here — see
    # test_guard_survives_an_active_entropy_floor_controller.
    assert float(rl.get("entropy_floor", 0.0)) == 0.0, rl.get("entropy_floor")
    if entropy_floor is not None:
        rl["entropy_floor"] = entropy_floor
    rl["rollout_steps"] = 48
    rl["steps"] = 2
    rl["ppo_minibatch_size"] = 16
    rl["bc_epochs"] = 0
    rl["bc_replay_enabled"] = False
    rl["rnd_intrinsic_coef"] = 0.0
    rl["rnd_loss_coef"] = 0.0
    rl["num_envs"] = 4
    block = dict(
        rl["backward_curriculum"],
        states_dir=str(states_dir),
        advance_actions=4, min_attempts=4, window_frames=16,
        truncation_is_failure=True,
    )
    if guard is None:
        block.pop("entropy_guard", None)
    else:
        block["entropy_guard"] = guard
    rl["backward_curriculum"] = block
    return profile


def _run(profile: dict, tmp: str, caplog):
    """Run the tiny loop; return (trainer, log lines)."""
    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    trainer = Trainer(
        rom_path=str(_SMB_ROM),
        game_profile=profile,
        num_instances=4,
        population_size=4,
        checkpoint_dir=tmp,
        start_state_path=str(ROOT / profile["start_state_path"]),
        env_spec="nes_core",
        max_episode_steps=12,
        metrics_queue=metrics_q,
        seed=7,
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        trainer.run(num_generations=ITERS, resume_from=None)
    return trainer, [r.getMessage() for r in caplog.records]


def _with_tape(fn):
    with tempfile.TemporaryDirectory(prefix="bwd_guard_") as tmp:
        states = Path(tmp) / "tape"
        base = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / base["start_state_path"])
        return fn(states, tmp)


def test_guard_boosts_entropy_coef_and_does_not_compound(caplog) -> None:
    """Armed on every iter, the coefficient is base x boost — ONCE.

    Applying a multiplier to persistent state every iter would give
    base * boost**ITERS; the loop strips the previous iter's boost before
    any other controller reads the coefficient, so the boost stays the
    outermost factor no matter how long the guard holds.
    """
    def _go(states, tmp):
        # floor above any achievable entropy => armed from the first iter
        # the sample floor allows; trailing/min_samples 1 so a 3-iter
        # smoke can reach the armed state at all.
        return _run(_tiny_profile(states, {
            "floor": 10.0, "boost": BOOST, "trailing": 1, "min_samples": 1,
        }), tmp, caplog)

    trainer, lines = _with_tape(_go)

    assert any(m.startswith("[backward-guard] configured") for m in lines), \
        [m for m in lines if "backward" in m]
    armed = [m for m in lines if m.startswith("[backward-guard] ARMED")]
    assert len(armed) == 1, armed          # one transition, not one per iter
    assert "x3.00" in armed[0], armed[0]
    assert not [m for m in lines if "[backward-guard] disarmed" in m]

    assert trainer.entropy_coef == pytest.approx(BASE_COEF * BOOST), (
        f"expected the boost applied exactly once "
        f"({BASE_COEF} x {BOOST}); compounding over {ITERS} iters would "
        f"give {BASE_COEF * BOOST ** ITERS}"
    )

    # The per-iter telemetry carries the armed state + the coefficient in
    # force, which is what the registered kill criterion is read off.
    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert len(iters) == ITERS, iters
    assert "guard ARMED" in iters[-1], iters[-1]
    assert "armed " in iters[-1], iters[-1]


def _expected_unboosted_after(iters: int, *, rising: bool) -> float:
    """Replay the adaptive entropy-floor controller ALONE for `iters` iters.

    Only the controller's own arithmetic is replicated here (trainer.py,
    `if self.entropy_floor > 0.0:`) — deliberately NOT the guard's
    strip/re-apply ordering, which is the thing the tests below exist to
    pin and so must not be duplicated into the expectation. If the
    controller's formula ever changes, these numbers move and the tests
    must be re-derived; that is the intended signal.
    """
    coef = BASE_COEF
    for _ in range(iters):
        coef = (min(coef * 1.5 + 1e-4, COEF_MAX) if rising
                else max(coef * 0.9, BASE_COEF))
    return coef


def test_guard_survives_an_active_entropy_floor_controller(caplog) -> None:
    """ORDERING: the boost is stripped BEFORE the floor controller reads it.

    This is the one property the shipped B5 configuration depends on that
    the smokes around it cannot see: they all pin
    mario_1_1_backward.yaml, where `entropy_floor` is 0.0 and the
    adaptive controller never runs at all. The only profile that ships
    the guard, mario_1_2_backward.yaml, runs it at 0.02.

    Inverting the two blocks is catastrophic rather than subtle. The
    controller RESCALES `entropy_coef`, so the strip's float-equality
    check no longer matches the value the guard wrote, the record is
    silently dropped, and the boost then multiplies an already-boosted
    coefficient once per iteration, without bound.

    Here the controller's RISING branch fires every iter (`entropy_floor`
    sits above any achievable entropy), so the unboosted coefficient
    genuinely moves between the strip and the re-apply — this cannot
    pass by the controller happening to be a no-op.
    """
    def _go(states, tmp):
        return _run(_tiny_profile(states, {
            "floor": 10.0, "boost": BOOST, "trailing": 1, "min_samples": 1,
        }, entropy_floor=10.0), tmp, caplog)

    trainer, lines = _with_tape(_go)

    # Non-vacuity: the branch this test means to exercise is the one that
    # ran, and it really did move the unboosted coefficient.
    assert _observed_entropy(lines) < 10.0, _observed_entropy(lines)
    expected = _expected_unboosted_after(ITERS, rising=True)
    assert expected > BASE_COEF, expected

    assert trainer.entropy_coef == pytest.approx(expected * BOOST), (
        f"expected the floor controller to ratchet the base to {expected} "
        f"with the boost applied exactly once on top ({expected * BOOST}); "
        f"got {trainer.entropy_coef}. Stripping AFTER the controller "
        f"instead of before drops the record every iter and compounds "
        f"the boost (~{COEF_MAX * BOOST} by iter {ITERS} here, unbounded "
        f"on the shipped 1-2 decay branch)."
    )
    # While armed the documented ceiling is entropy_coef_max x boost.
    assert trainer.entropy_coef <= COEF_MAX * BOOST + 1e-12


def test_shipped_1_2_floor_value_leaves_the_boost_exactly_once(caplog) -> None:
    """The same ordering property at the literal value B5 will run.

    mario_1_2_backward.yaml ships `entropy_floor: 0.02` alongside the
    guard. With live entropy well above 1.5x that, the controller takes
    its DECAY branch, which is clamped below at the base coefficient — so
    the unboosted value is a fixed point at BASE_COEF and the armed
    coefficient must sit at exactly BASE_COEF x BOOST however long the
    guard holds. Inverted, that same decay branch multiplies the BOOSTED
    value by 0.9 and the guard re-boosts it: ~2.7x per iter, compounding
    for as long as the run continues.
    """
    def _go(states, tmp):
        return _run(_tiny_profile(states, {
            "floor": 10.0, "boost": BOOST, "trailing": 1, "min_samples": 1,
        }, entropy_floor=SHIPPED_1_2_ENTROPY_FLOOR), tmp, caplog)

    trainer, lines = _with_tape(_go)

    # Non-vacuity: entropy must be above the controller's upper band or
    # neither branch runs and the ordering is never exercised.
    seen = _observed_entropy(lines)
    assert seen > 1.5 * SHIPPED_1_2_ENTROPY_FLOOR, seen
    assert _expected_unboosted_after(ITERS, rising=False) == BASE_COEF

    assert trainer.entropy_coef == pytest.approx(BASE_COEF * BOOST), (
        f"expected {BASE_COEF * BOOST}; got {trainer.entropy_coef}. "
        f"Stripping after the 0.02 floor controller compounds ~2.7x per "
        f"iter ({BASE_COEF * BOOST * 2.7 ** (ITERS - 1)} by iter {ITERS})."
    )
    assert trainer.entropy_coef <= COEF_MAX * BOOST + 1e-12


def test_absent_guard_block_leaves_the_loop_untouched(caplog) -> None:
    """The opt-in contract: no block, no guard, no log line, no change to
    entropy_coef, and a `[backward]` telemetry line with no guard suffix
    (so every pre-guard run log stays comparable character-for-character).
    """
    def _go(states, tmp):
        return _run(_tiny_profile(states, None), tmp, caplog)

    trainer, lines = _with_tape(_go)

    assert not [m for m in lines if "[backward-guard]" in m], \
        [m for m in lines if "backward-guard" in m]
    assert trainer.entropy_coef == pytest.approx(BASE_COEF)
    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert len(iters) == ITERS, iters
    assert all("guard" not in m for m in iters), iters


def test_disabled_guard_block_is_inert(caplog) -> None:
    """`enabled: false` keeps the block documented in a profile without
    arming anything — same observable behavior as omitting it."""
    def _go(states, tmp):
        return _run(_tiny_profile(states, {
            "enabled": False, "floor": 10.0, "boost": BOOST,
        }), tmp, caplog)

    trainer, lines = _with_tape(_go)

    assert not [m for m in lines if "[backward-guard]" in m]
    assert trainer.entropy_coef == pytest.approx(BASE_COEF)


def test_malformed_guard_block_fails_loudly(caplog) -> None:
    """A pre-registered run must never spend hours on a knob that did
    nothing: a present-but-incomplete block raises at loop setup."""
    def _go(states, tmp):
        with pytest.raises(ValueError, match="boost"):
            _run(_tiny_profile(states, {"floor": 0.08}), tmp, caplog)
        return None, []

    _with_tape(_go)
