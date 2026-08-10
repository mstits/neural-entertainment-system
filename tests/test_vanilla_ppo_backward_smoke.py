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

The checkpoint-resume wiring at the bottom needs neither: it drives the
save/load path with a duck-typed net and optimizer, so it runs everywhere.
"""

from __future__ import annotations

import logging
import queue as _queue
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_1_1_backward.yaml"

# Per-test rather than module-wide: the pure resume-wiring smokes below
# never touch the emulator and must not be skipped with the ROM.
needs_rom = pytest.mark.skipif(
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
    blk = dict(rl["backward_curriculum"])
    # Tiny tape: keep the rungs reachable inside a few iters. Boundary
    # truncations score so the rung statistics are deterministic instead
    # of hostage to whether a random policy happens to die.
    blk.update(states_dir=str(states_dir), advance_actions=4,
               min_attempts=4, window_frames=16, count_truncations=True)
    blk.update(block)
    # `count_truncations` is the registered name and
    # `truncation_is_failure` the legacy alias for the same knob. The
    # profile on disk still carries the alias and shipping both is a
    # schema warning, so exactly one survives — the one under test.
    blk.pop("count_truncations" if "truncation_is_failure" in block
            else "truncation_is_failure", None)
    rl["backward_curriculum"] = blk
    return profile


def _trailing(line: str) -> tuple[int, int]:
    """(successes, attempts) off a `[backward] iter` line's `trailing S/N`."""
    s, n = line.split("trailing ", 1)[1].split("=", 1)[0].split("/")
    return int(s), int(n)


def _truncated(line: str) -> tuple[int, int]:
    """(dropped, scored) off the same line's `truncated D (S scored)`."""
    tail = line.split("truncated ", 1)[1]
    dropped = int(tail.split()[0])
    scored = (int(tail.split("(", 1)[1].split(" scored)", 1)[0])
              if " scored)" in tail else 0)
    return dropped, scored


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


@needs_rom
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


@needs_rom
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


@needs_rom
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


@needs_rom
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


# ---- the two run-2 mechanism repairs ---------------------------------
#
# B5 run 2 stopped at iter 266 on the registered stall criterion: tau sat
# 156 iters at rung 893 with a trailing window of 0/0 while 15,914
# attempts were truncated and dropped. Nothing died, nothing cleared, so
# nothing was ever SCORED and the advance gate's attempt floor could not
# be met. Both repairs are opt-in; these smokes drive them through the
# real loop, which is the only place the two of them meet.


@needs_rom
def test_counted_truncations_are_the_evidence_dropping_censors(caplog) -> None:
    """Repair 1, as a paired run: same seed, same trajectories, one knob.

    Dropped (the default) a truncated attempt leaves nothing behind but a
    censoring count, and the advance gate's attempt floor has to be met
    without it — on 1-2 that meant 15,914 attempts thrown away and a
    window that never left 0/0. Counted, those same attempts land in the
    window as failures: attempts_on == attempts_off + truncations, exactly.
    """
    def _measure(lines):
        last = [m for m in lines if m.startswith("[backward] iter")][-1]
        succ, att = _trailing(last)
        ent = int(last.split("entrance ", 1)[1].split("/")[1].split("=")[0])
        return (succ, att + ent) + _truncated(last), last

    with tempfile.TemporaryDirectory(prefix="bwd_trunc_tape_") as tape_tmp:
        states = Path(tape_tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])

        def _go(count, tag):
            with tempfile.TemporaryDirectory(prefix=tag) as tmp:
                # A trailing window long enough that nothing falls out of
                # it: the identity below counts attempts, and a saturated
                # deque forgets the oldest ones.
                return _measure(_run(
                    _tiny_profile(states, count_truncations=count,
                                  min_attempts=20),
                    tmp, caplog, seed=1234,
                ))

        (s_off, att_off, drop_off, sc_off), line_off = _go(False, "bwd_drop_")
        (s_on, att_on, drop_on, sc_on), line_on = _go(True, "bwd_count_")

    # Censored: the drop count is the only trace a truncation leaves, and
    # the column that would report it as evidence is absent entirely.
    assert drop_off > 0 and sc_off == 0, line_off
    assert "scored" not in line_off, line_off
    # Counted: nothing censored, and every one of those attempts is now in
    # the rung window or the entrance column.
    assert drop_on == 0 and sc_on == drop_off, (line_off, line_on)
    assert att_on == att_off + sc_on, (line_off, line_on)
    # A timeout is a failure, never a win — on either setting.
    assert s_off == 0 and s_on == 0, (line_off, line_on)


@needs_rom
def test_the_legacy_truncation_alias_still_arms_the_same_knob(caplog) -> None:
    """B4's profiles say `truncation_is_failure`; their meaning must not
    change under them just because the registered name did."""
    with tempfile.TemporaryDirectory(prefix="bwd_alias_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        prof = _tiny_profile(states, truncation_is_failure=True)
        assert "count_truncations" not in \
            prof["reinforce"]["backward_curriculum"]
        lines = _run(prof, tmp, caplog)

    iters = [m for m in lines if m.startswith("[backward] iter")]
    dropped, scored = _truncated(iters[-1])
    assert scored > 0 and dropped == 0, iters[-1]


@needs_rom
def test_the_rung_budget_cuts_episodes_short_and_scores_them(caplog) -> None:
    """Repair 2: a per-rung step cap, so attempts resolve INSIDE the
    rollout instead of running out the clock.

    Without it each env contributes exactly one (boundary-truncated)
    attempt per iter; with a cap of ~5-9 steps on a 48-step rollout each
    env contributes several, and every one of them is scored.
    """
    ENVS, ITERS = 4, 3
    with tempfile.TemporaryDirectory(prefix="bwd_budget_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        lines = _run(
            _tiny_profile(states, count_truncations=True,
                          rung_step_budget={"base": 4, "per_entry": 1.0}),
            tmp, caplog, iters=ITERS,
        )

    budget = [m for m in lines if m.startswith("[backward] BUDGET")]
    assert budget, [m for m in lines if m.startswith("[backward]")]
    # min(48, 4 + 1.0 * (12 - r)): 5 at the ladder top, 16 at the entrance.
    assert "5 at the ladder top" in budget[0], budget[0]
    assert "16 at the entrance" in budget[0], budget[0]

    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert "budget " in iters[-1], iters[-1]
    _, scored = _truncated(iters[-1])
    assert scored > ENVS * ITERS, (
        f"{scored} scored attempts is no more than the {ENVS * ITERS} the "
        f"iter boundary alone produces — the budget never fired: "
        f"{iters[-1]}"
    )


@needs_rom
def test_no_budget_block_leaves_episodes_uncapped(caplog) -> None:
    """Absent, the knob is not merely off — it is not there. No budget
    line, no per-iter budget column, and no attempt is ever cut short: the
    only truncations are the iter boundary's, at most one per env per
    iter (and the boundary that ends the last rollout is not reported
    until the next iter's line, so it is strictly fewer)."""
    ENVS, ITERS = 4, 3
    with tempfile.TemporaryDirectory(prefix="bwd_nobudget_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        lines = _run(_tiny_profile(states, count_truncations=True),
                     tmp, caplog, iters=ITERS)

    assert not [m for m in lines if m.startswith("[backward] BUDGET")]
    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert "budget" not in iters[-1], iters[-1]
    _, scored = _truncated(iters[-1])
    assert 0 < scored <= ENVS * ITERS, iters[-1]


@needs_rom
def test_a_malformed_budget_block_raises_instead_of_no_opping(caplog) -> None:
    """The dead-knob class: a pre-registered run must fail at second one,
    not spend its window with the cap silently absent."""
    with tempfile.TemporaryDirectory(prefix="bwd_badbudget_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        prof = _tiny_profile(states, rung_step_budget={"base": 600})
        with pytest.raises(ValueError, match="per_entry"):
            _run(prof, tmp, caplog, iters=1)


@needs_rom
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


# ---- checkpoint / resume wiring (no ROM, no pool) --------------------
#
# The cursor used to be absent from the checkpoint payload, so `--resume`
# silently reset tau to the ladder top: B4 v3 came back at iter 80 with
# tau=757/757 and the at-entrance winner gate never fired for the rest of
# the run. These drive the real save/load path with duck-typed stand-ins
# for the net and the optimizer.


class _FakeModule:
    """Enough of nn.Module for the resume path: it only loads."""

    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, sd, strict=True) -> None:
        self.loaded = sd


def _shell_trainer(ckpt_dir: Path):
    """A Trainer with only the attributes the resume path reads.

    Constructing a real one wants a ROM, a worker pool and a profile;
    `_maybe_resume_vanilla_ppo` wants a checkpoint dir and a device.
    """
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.checkpoint_dir = Path(ckpt_dir)
    t.device = torch.device("cpu")
    t._pending_rnd_state = None
    t._pending_backward_curriculum = None
    t._gx_counts = {}
    t._vppo_resumed_from_iter = None
    return t


def _sched(n_entries=757, **kw):
    from src.training.backward_curriculum import TauScheduler

    kw.setdefault("advance_entries", 10)
    kw.setdefault("advance_threshold", 0.2)
    kw.setdefault("min_attempts", 5)
    return TauScheduler(n_entries, **kw)


def _write_ckpt(ckpt_dir: Path, it: int, **extra) -> Path:
    """A checkpoint payload shaped like the one the trainer saves."""
    path = Path(ckpt_dir) / f"vanilla_ppo_iter_{it:05d}.pt"
    torch.save({"iter": it, "net_state_dict": {}, "optimizer_state_dict": {},
                **extra}, str(path))
    return path


def test_tau_survives_a_checkpoint_save_load_cycle(tmp_path, caplog) -> None:
    saved = _sched()
    for _ in range(20):                    # walk a few rungs up the ladder
        saved.record(True)
        saved.maybe_advance()
    for _ in range(3):
        saved.record(True)
    saved.record_entrance(False)
    assert saved.tau < saved.n_entries - 1

    _write_ckpt(tmp_path, 80, backward_curriculum=saved.state_dict())
    trainer = _shell_trainer(tmp_path)
    assert trainer._maybe_resume_vanilla_ppo(_FakeModule(), _FakeModule()) == 80

    resumed = _sched()
    assert resumed.tau == 756            # a fresh cursor is at the top
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        trainer._apply_pending_backward_state(resumed)

    assert resumed.snapshot() == saved.snapshot()
    msgs = [r.getMessage() for r in caplog.records]
    assert any("RESUMED cursor" in m and f"tau={saved.tau}/756" in m
               for m in msgs), msgs
    # Consumed, so a later rebuild in the same process can't re-apply it.
    assert trainer._pending_backward_curriculum is None


def test_a_checkpoint_without_the_key_keeps_todays_behavior(
    tmp_path, caplog,
) -> None:
    """Old checkpoints predate the cursor and must still load — at the
    configured tau_init, exactly as they did before this existed."""
    _write_ckpt(tmp_path, 80)
    trainer = _shell_trainer(tmp_path)
    assert trainer._maybe_resume_vanilla_ppo(_FakeModule(), _FakeModule()) == 80
    assert trainer._pending_backward_curriculum is None

    sched = _sched(tau_init=100)
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        trainer._apply_pending_backward_state(sched)
    assert sched.tau == 100
    assert not [r for r in caplog.records if "RESUMED cursor" in r.getMessage()]


def test_a_cursor_from_a_different_tape_is_refused_not_fatal(
    tmp_path, caplog,
) -> None:
    """A re-minted tape makes tau meaningless. The run must say so and
    keep going from the configured rung, not die and not adopt it."""
    stale = _sched(n_entries=400)
    for _ in range(10):
        stale.record(True)
        stale.maybe_advance()
    _write_ckpt(tmp_path, 80, backward_curriculum=stale.state_dict())
    trainer = _shell_trainer(tmp_path)
    trainer._maybe_resume_vanilla_ppo(_FakeModule(), _FakeModule())

    sched = _sched(n_entries=757, tau_init=-1)
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        trainer._apply_pending_backward_state(sched)
    assert sched.tau == 756
    assert sched.attempts == 0
    msgs = [r.getMessage() for r in caplog.records]
    assert any("cursor NOT restored" in m for m in msgs), msgs


@needs_rom
def test_a_real_run_checkpoints_the_cursor_and_resumes_it(caplog) -> None:
    """End to end: the trainer's own save writes the cursor, and its own
    resume adopts it. The saved cursor is rewritten mid-ladder between the
    two runs so the second run's tau can only have come from the file."""
    with tempfile.TemporaryDirectory(prefix="bwd_resume_") as tmp:
        states = Path(tmp) / "tape"
        profile = yaml.safe_load(_PROFILE.read_text())
        _mint_synthetic_tape(states, ROOT / profile["start_state_path"])
        prof = _tiny_profile(states)
        _run(prof, tmp, caplog, iters=11)      # checkpoints at iter 10

        ckpts = sorted(Path(tmp).glob("vanilla_ppo_iter_*.pt"))
        assert ckpts, sorted(p.name for p in Path(tmp).iterdir())
        payload = torch.load(str(ckpts[-1]), map_location="cpu",
                             weights_only=False)
        assert "backward_curriculum" in payload, sorted(payload)
        cursor = payload["backward_curriculum"]
        assert cursor["n_entries"] == N_STATES
        cursor.update(tau=3, advances=2, window=[False, True],
                      entrance_attempts=9, entrance_successes=1)
        torch.save(payload, str(ckpts[-1]))

        lines = _run(prof, tmp, caplog, iters=1)

    restored = [m for m in lines if "RESUMED cursor" in m]
    assert restored, lines
    assert f"tau=3/{N_STATES - 1}" in restored[0], restored[0]
    iters = [m for m in lines if m.startswith("[backward] iter")]
    assert iters and f"tau=3/{N_STATES - 1}" in iters[0], iters
    # The trailing window came back too — not just the cursor.
    assert "trailing 1/2" in iters[0], iters[0]
