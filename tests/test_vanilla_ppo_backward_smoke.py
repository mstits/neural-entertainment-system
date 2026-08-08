"""End-to-end smoke for the backward start-state curriculum wired into the
vanilla_ppo loop.

The cursor, the window draw and the index format are unit-tested in
tests/test_backward_curriculum.py. These smokes cover the TRAINER WIRING
those pure tests cannot see: the states dir loads at setup, the inline
restart branch draws from the tape (instead of always reloading the
profile start state), the cursor's rung statistics accumulate, and the
`pin_entrance` guard reduces the branch to the pre-flag behavior.

The tape here is synthetic — a dozen states harvested by stepping the SMB
entrance forward with no-ops. It only has to be a valid state ladder; the
real tapes are minted by scripts/mint_backward_states.py.
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


def _tiny_profile(states_dir: Path, **block) -> dict:
    profile = yaml.safe_load(_PROFILE.read_text())
    rl = profile["reinforce"]
    rl["rollout_steps"] = 48
    rl["steps"] = 2
    rl["ppo_minibatch_size"] = 16
    rl["bc_epochs"] = 0
    rl["bc_replay_enabled"] = False
    rl["rnd_intrinsic_coef"] = 0.0
    rl["rnd_loss_coef"] = 0.0
    rl["num_envs"] = 4
    rl["backward_curriculum"] = dict(
        rl["backward_curriculum"],
        states_dir=str(states_dir),
        # Tiny tape: keep the rungs reachable inside a few iters. Boundary
        # truncations score so the rung statistics are deterministic
        # instead of hostage to whether a random policy happens to die.
        advance_actions=4, min_attempts=4, window_frames=16,
        truncation_is_failure=True,
        **block,
    )
    return profile


def _run(profile: dict, tmp: str, caplog, *, prefix="[backward]",
         seed=None, iters=3) -> list[str]:
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
        # Short episodes so the inline restart branch actually fires
        # several times inside each rollout.
        max_episode_steps=12,
        metrics_queue=metrics_q,
        seed=seed,
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        trainer.run(num_generations=iters, resume_from=None)
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith(prefix)]


def test_backward_curriculum_loads_and_scores_rungs(caplog) -> None:
    with tempfile.TemporaryDirectory(prefix="bwd_smoke_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        lines = _run(_tiny_profile(states), tmp, caplog)

    assert any(m.startswith("[backward] ENABLED") for m in lines), lines
    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert len(iters) == 3, iters
    # First iter runs off the cold reset — no attempt has resolved yet.
    assert "trailing 0/0" in iters[0], iters[0]
    # By the last iter the boundary hook has scored every env's attempt
    # against a rung or the entrance, so the counters have moved.
    assert "trailing 0/0" not in iters[-1] or "entrance 0/0" not in iters[-1], \
        iters
    assert "tau=" in iters[-1] and f"/{N_STATES - 1}" in iters[-1], iters[-1]
    # The shipped config trains sticky WITH the episode-boundary guard, so
    # no rung attempt starts by replaying the dead life's held action.
    assert not any("CONTAMINATED" in m for m in lines), lines


def test_sticky_without_boundary_guard_is_called_out(caplog) -> None:
    """The guard that would have caught the shipped-config defect.

    `_sticky_restart` is a no-op unless `sticky_episode_boundary_reset` is
    set, so sticky + backward without it silently replays the previous
    life's held action on each fresh rung attempt's first step. Silent is
    the whole problem — the trainer must say so.
    """
    with tempfile.TemporaryDirectory(prefix="bwd_contam_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        prof = _tiny_profile(states)
        assert prof["reinforce"]["sticky_action_prob"] > 0.0
        prof["reinforce"]["sticky_episode_boundary_reset"] = False
        lines = _run(prof, tmp, caplog, iters=1)

    assert any("[backward] CONTAMINATED" in m for m in lines), lines


def test_pin_entrance_reports_the_entrance_rung_only(caplog) -> None:
    """The zero-diff guard: pinned, every restart is the profile start
    state, so the cursor never leaves tau 0's accounting for the window."""
    with tempfile.TemporaryDirectory(prefix="bwd_pin_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        lines = _run(
            _tiny_profile(states, pin_entrance=True, tau_init=0), tmp, caplog
        )

    assert any("[PINNED AT ENTRANCE]" in m for m in lines), lines
    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert iters
    # Pinned draws are entrance draws, so no rung ever accumulates.
    assert all("trailing 0/0" in m for m in iters), iters
    assert all("tau=0/" in m for m in iters), iters


def test_pinned_run_is_identical_to_one_without_the_flag(caplog) -> None:
    """Zero-diff guard.

    With sticky off and the cursor pinned at the entrance, the backward
    branch loads no tape state and draws no extra RNG, so a run with the
    block must reproduce a run without it exactly. This is the guard that
    keeps every existing determinism lineage and honest baseline valid:
    if it ever fails, the flag is no longer opt-in.
    """
    ITER_PREFIX = "[vanilla_ppo] iter"

    def _iter_lines(profile, tag):
        with tempfile.TemporaryDirectory(prefix=tag) as tmp:
            return [m for m in _run(profile, tmp, caplog,
                                    prefix=ITER_PREFIX, seed=1234)
                    if "throughput" not in m]

    with tempfile.TemporaryDirectory(prefix="bwd_zd_tape_") as tape_tmp:
        states = Path(tape_tmp) / "tape"
        base = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / base["start_state_path"])

        off = _tiny_profile(states)
        off["reinforce"]["sticky_action_prob"] = 0.0
        off["reinforce"].pop("backward_curriculum")

        pinned = _tiny_profile(states, pin_entrance=True, tau_init=0)
        pinned["reinforce"]["sticky_action_prob"] = 0.0

        a = _iter_lines(off, "bwd_zd_off_")
        b = _iter_lines(pinned, "bwd_zd_pin_")
        # Reproducibility of the baseline itself, so a mismatch below can
        # only be the flag.
        c = _iter_lines(off, "bwd_zd_off2_")

    assert a and a == c, "baseline is not reproducible; guard is unusable"
    assert a == b, "\n".join(f"{x}\n{y}" for x, y in zip(a, b) if x != y)


def test_backward_curriculum_is_inert_without_tile_mode(caplog) -> None:
    """A pixel encoder never reaches the tile restart branch, so the
    curriculum must refuse to arm rather than half-arm."""
    with tempfile.TemporaryDirectory(prefix="bwd_inert_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        prof = _tiny_profile(states)
        prof["reinforce"]["encoder"] = "nature_dqn"
        lines = _run(prof, tmp, caplog)

    assert any("INERT" in m for m in lines), lines
    assert not [m for m in lines if m.startswith("[backward] iter")]
