"""Tests for parallel episode evaluation in `scripts/eval_game.py`.

`eval_game.py` ran its episodes one at a time in a one-worker pool, so an
honest-protocol receipt (30-50 episodes, sticky + jitter) cost 30-50 serial
rollouts on a 16-core box. `--eval-workers K` runs K episodes concurrently
with per-lane bookkeeping.

The load-bearing property is that the ANSWER cannot depend on the schedule.
That is only true if each episode's randomness is a function of
(--eval-seed, episode index) — which the historical shared-stream RNG is NOT:
it threads one `RandomState`/`Generator` through the episodes in order, and
the number of draws an episode consumes depends on how long it ran. So the
parallel branch requires `--eval-rng per-episode`, and a stochastic run asked
to go wide under the default mode is REFUSED rather than silently answered
with a different set of draws.

The reference every invariance test here measures against is
`_run_episodes_serial` — the production serial loop, the same object
`eval_one_game` calls at `--eval-workers 1`. That matters: a suite that
baselines the parallel executor on itself at `lanes=1` cannot see an
off-by-one in the parallel branch's Machado jitter range or its sticky
first-step guard, because the bug is applied identically to both sides. Those
two numbers ARE the honest protocol, so a mutation there would silently make
`--eval-workers K` measure a different protocol than the receipt claims.
Nothing in this file compares the parallel executor to another copy of itself.

Layers:
  * pure RNG derivation — `episode_rng_seeds` is a function of (seed, index)
    alone, decorrelated across both; `run_consumes_randomness` is the exact
    predicate the step loop's guards imply (fast, no ROM);
  * argument contract — bad worker counts / RNG modes and the
    workers>1 + shared-stream + stochastic combination are rejected before any
    disk I/O, and on the single-JSON-line CLI contract (fast, no ROM);
  * stubbed differential — `_run_episodes_serial` vs `_run_episodes_parallel`
    over a fake pool + fake policy + fake reward function: per-episode records
    are byte-identical for every lane count and in every stochastic regime,
    waves refill correctly, warm-start blobs reach every lane, finished lanes
    are parked (fast, no ROM);
  * ROM-gated end-to-end — the serial loop at 1 worker vs 4 lanes under
    sticky + jitter must agree on clear_rate and max_gx_per_episode, and the
    default path still emits a serial receipt (slow).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
# `eval_game.py` lives under scripts/ and is imported by module name (it is
# not a package). Mirrors tests/test_eval_action_select.py.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import eval_game  # noqa: E402
from eval_game import (  # noqa: E402
    _run_episodes_parallel, _run_episodes_serial, episode_rng_seeds,
    run_consumes_randomness,
)


# --------------------------------------------------------------------------
# gitignored data lookup. Mirrors tests/test_eval_action_select.py::_find.
# --------------------------------------------------------------------------
def _find(rel_candidates: list[str], newest: bool = False) -> Path | None:
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            hits = sorted(root.glob(rel))
            if hits:
                return hits[-1] if newest else hits[0]
    return None


_ROM = _find(["roms/Super Mario Bros. (World).nes"])
_START = _find(["roms/Super Mario Bros. (World)_start.state.bin"])
# NEWEST iter, not the first: the differential below is only worth running on a
# policy that actually travels. The earliest checkpoint dies at the same pixel
# in every episode, which would make "the two executors agree" vacuously true.
_TILE_CKPT = _find([
    "checkpoints/super_mario_bros_one_shot_tiles/vanilla_ppo_iter_*.pt",
], newest=True)
_TILE_CFG = _ROOT / "configs" / "smb_oneshot_tiles.yaml"
_HAVE_EMU = _ROM is not None and _START is not None


# --- pure RNG derivation ---------------------------------------------------

def test_episode_rng_seeds_depend_only_on_seed_and_index():
    """A lane derives its streams without knowing what ran before it."""
    for ep in (0, 1, 7, 500):
        assert episode_rng_seeds(3, ep) == episode_rng_seeds(3, ep)
    # Computing episode 5's seeds in isolation must match computing them
    # after walking 0..4 — the property the shared stream cannot offer.
    walked = [episode_rng_seeds(3, i) for i in range(6)]
    assert walked[5] == episode_rng_seeds(3, 5)


def test_episode_rng_seeds_are_distinct_across_episodes_and_seeds():
    per_seed = {
        seed: [episode_rng_seeds(seed, ep) for ep in range(32)]
        for seed in (0, 1, 17)
    }
    for seeds in per_seed.values():
        flat = [s for pair in seeds for s in pair]
        # 64 draws (numpy + torch seed per episode), all distinct: no episode
        # shares a stream with another, and no episode's numpy stream is its
        # own torch stream.
        assert len(set(flat)) == len(flat)
    assert per_seed[0] != per_seed[1] != per_seed[17]


def test_episode_rng_seeds_are_plain_ints_torch_and_numpy_accept():
    np_seed, torch_seed = episode_rng_seeds(9, 4)
    assert type(np_seed) is int and type(torch_seed) is int
    assert 0 <= np_seed < 2 ** 32 and 0 <= torch_seed < 2 ** 32
    np.random.RandomState(np_seed).random()
    torch.Generator(device="cpu").manual_seed(torch_seed)


@pytest.mark.parametrize(
    "sticky,jitter,mode,expected",
    [
        (0.0, 0, "greedy", False),      # nothing draws: streams never touched
        (0.25, 0, "greedy", True),      # sticky repeat draw
        (0.0, 16, "greedy", True),      # Machado no-op start draw
        (0.0, 0, "sampled", True),      # multinomial draw
        (0.25, 16, "sampled", True),
    ],
)
def test_run_consumes_randomness_matches_the_step_loop_guards(
    sticky, jitter, mode, expected,
):
    assert run_consumes_randomness(sticky, jitter, mode) is expected


# --- argument contract (raises before any I/O) -----------------------------

@pytest.mark.parametrize("bad", [0, -1])
def test_eval_one_game_rejects_non_positive_worker_count(bad):
    with pytest.raises(ValueError, match="eval_workers must be >= 1"):
        eval_game.eval_one_game(
            "mario", Path("/nonexistent/profile.yaml"), "/nonexistent.nes",
            1, 1, eval_workers=bad,
        )


def test_eval_one_game_rejects_unknown_rng_mode():
    with pytest.raises(ValueError, match="eval_rng must be one of"):
        eval_game.eval_one_game(
            "mario", Path("/nonexistent/profile.yaml"), "/nonexistent.nes",
            1, 1, eval_rng="per_episode",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sticky_prob": 0.25},
        {"start_jitter": 16},
        {"action_select": "sampled"},
    ],
)
def test_eval_one_game_refuses_wide_stochastic_run_on_the_shared_stream(kwargs):
    """The shared stream cannot be re-derived per episode, so going wide with
    it would answer a different question under the same flags."""
    with pytest.raises(ValueError, match="needs eval_rng='per-episode'"):
        eval_game.eval_one_game(
            "mario", Path("/nonexistent/profile.yaml"), "/nonexistent.nes",
            4, 1, eval_workers=4, **kwargs,
        )


def test_eval_one_game_allows_wide_deterministic_run_on_the_shared_stream():
    """Greedy + no sticky + no jitter never touches either stream, so the two
    RNG modes are the same run and the worker count cannot matter. Reaching
    the missing-profile error proves validation passed."""
    with pytest.raises(FileNotFoundError):
        eval_game.eval_one_game(
            "mario", Path("/nonexistent/profile.yaml"), "/nonexistent.nes",
            4, 1, eval_workers=4,
        )


# --- CLI contract (no ROM needed) ------------------------------------------

def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "eval_game.py"), *args],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=180,
    )


def test_cli_rejects_zero_workers_on_the_json_contract():
    r = _cli("--game", "mario", "--eval-workers", "0")
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["status"] == "bad_args"
    assert "--eval-workers" in out["detail"]


def test_cli_rejects_wide_stochastic_run_on_the_shared_stream():
    r = _cli("--game", "mario", "--eval-workers", "4", "--sticky-prob", "0.25")
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["status"] == "bad_args"
    assert "--eval-rng per-episode" in out["detail"]


def test_cli_rejects_unknown_rng_mode():
    r = _cli("--game", "mario", "--eval-rng", "whatever")
    assert r.returncode == 2
    assert "--eval-rng" in r.stderr


# --- stubbed differential --------------------------------------------------

_BITMASKS = [0, 1, 2, 3]
_STUB_MAX_EMU_STEPS = 40   # stub worker raises `done` here
_STUB_SUCCESS_X = 660      # stub reward function latches success here
# Per-step advance is `_STUB_X_SCALE * (1 + action % 3)`, so x routinely runs
# past 255 and the HIGH byte of the gx read (`ram[0x006D] << 8`) carries real
# information. With a small scale every gx fits in the low byte and an executor
# that dropped the high byte would score identically — the test would be blind
# to it.
_STUB_X_SCALE = 11


class _StubPool:
    """A pool of independent counters with the real Pool's surface.

    Each worker's state is a pure function of the actions it was fed, which is
    exactly the property real NES workers have and the property the executor's
    schedule-invariance rests on.
    """

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.resets = 0
        self.loads: list[tuple[int, bytes]] = []
        self.done_calls: list[tuple[int, bool]] = []
        self.step_calls = 0
        # How many times each worker was actually ADVANCED (never reset), so a
        # test can prove a parked lane burned no emulation rather than merely
        # that `set_worker_done` was called.
        self.advanced = [0] * num_workers
        self._x = [0] * num_workers
        self._steps = [0] * num_workers
        self._done = [False] * num_workers

    def reset_all(self):
        self.resets += 1
        self._x = [0] * self.num_workers
        self._steps = [0] * self.num_workers
        self._done = [False] * self.num_workers
        return [self._out(i) for i in range(self.num_workers)]

    def load_worker_state(self, worker_id: int, blob: bytes) -> None:
        self.loads.append((worker_id, bytes(blob)))
        self._x[worker_id] = int(blob[0])

    def set_worker_done(self, worker_id: int, done: bool) -> None:
        self.done_calls.append((worker_id, bool(done)))
        self._done[worker_id] = bool(done)

    def step_all(self, actions):
        assert len(actions) == self.num_workers, "action vector must cover the pool"
        self.step_calls += 1
        for i, a in enumerate(actions):
            if self._done[i]:
                continue  # parked worker: the real pool skips its emulation
            self._x[i] += _STUB_X_SCALE * (1 + (int(a) % 3))
            self._steps[i] += 1
            self.advanced[i] += 1
        return [self._out(i) for i in range(self.num_workers)]

    def shutdown(self) -> None:
        pass

    def _out(self, i: int):
        ram = bytearray(2048)
        ram[0x006D] = (self._x[i] >> 8) & 0xFF
        ram[0x0086] = self._x[i] & 0xFF
        # The stage byte is `(0x075F << 4) | (0x0760 & 0x0F)`. 0x0760 carries
        # bits ABOVE the nibble on purpose: with a small value the mask is a
        # no-op and dropping it would be invisible.
        ram[0x075F] = 1
        ram[0x0760] = 0x25
        return (b"frame", b"pp", bytes(ram), self._steps[i] >= _STUB_MAX_EMU_STEPS)


def _stub_x(ram) -> int:
    return (int(ram[0x006D]) << 8) | int(ram[0x0086])


class _StubPolicy:
    """Stateful (like the real stackers) and a pure function of what it saw."""

    def __init__(self) -> None:
        self.last_x = 0

    def reset(self, out):
        self.last_x = _stub_x(out[2])
        return np.array([float(self.last_x)], dtype=np.float32)

    def push(self, out):
        self.last_x = _stub_x(out[2])
        return np.array([float(self.last_x)], dtype=np.float32)

    def initial_hidden(self, device):
        return None

    def logits(self, obs, hidden):
        # Near-uniform on purpose: sampled draws must actually spread, so a
        # crossed generator would show up as a different trajectory.
        base = float(obs[0]) * 0.001
        return torch.tensor(
            [[base, base + 0.05, base - 0.05, 0.0]], dtype=torch.float32,
        ), hidden


class _StubReward:
    def __init__(self) -> None:
        self.prev = 0
        self.success = False

    def reset(self) -> None:
        self.prev = 0
        self.success = False

    def compute(self, ram, action: int = 0):
        x = _stub_x(ram)
        r = float(x - self.prev)
        self.prev = x
        if x >= _STUB_SUCCESS_X:
            self.success = True
        return r, False, {}

    def episode_success(self) -> bool:
        return self.success


class _StubTracker:
    """Stand-in for SequentialTracker / LevelClearTracker: the four fields both
    executors read, plus `seq_clear` as an episode terminator.

    Every field is a function of the frames it was fed — INCLUDING the
    warm-start frame both loops must seed it with before the first action — so
    an executor that skips that seeding, or feeds the frames in a different
    order, produces different records.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.seq_clear = False
        self.warp_taken = False
        self.furthest_seq = None
        self.furthest_any = None

    def update(self, ram) -> None:
        x = _stub_x(ram)
        self.frames += 1
        self.furthest_any = (1, self.frames)
        self.furthest_seq = (1, x)
        self.warp_taken = x >= _STUB_SUCCESS_X // 2
        self.seq_clear = x >= _STUB_SUCCESS_X


def _stub_kwargs(kw: dict) -> dict:
    """The executor arguments both branches take, so a differential can be
    sure the ONLY difference between the two calls is the executor itself."""
    out = dict(
        n_episodes=kw.pop("n_episodes", 7),
        max_steps=kw.pop("max_steps", 30),
        bitmasks=_BITMASKS,
        make_policy=_StubPolicy,
        make_reward_fn=_StubReward,
        make_tracker=kw.pop("make_tracker", lambda: None),
        stage_blob=kw.pop("stage_blob", None),
        device=torch.device("cpu"),
        sticky_prob=kw.pop("sticky_prob", 0.25),
        start_jitter=kw.pop("start_jitter", 5),
        eval_seed=kw.pop("eval_seed", 11),
        action_select=kw.pop("action_select", "sampled"),
        temperature=kw.pop("temperature", 1.0),
    )
    assert not kw, f"unused kwargs {sorted(kw)}"
    return out


def _run_stub(lanes: int, **kw) -> tuple[list[dict], _StubPool]:
    """The PARALLEL executor over `lanes` stub workers."""
    args = _stub_kwargs(kw)
    pool = _StubPool(lanes)
    return _run_episodes_parallel(pool, lanes=lanes, **args), pool


def _run_stub_serial(**kw) -> tuple[list[dict], _StubPool]:
    """The PRODUCTION SERIAL loop over a 1-worker stub pool — the reference.

    Same fake pool, same fake policy, same fake reward function, same seed:
    the only thing that differs from `_run_stub` is which executor is driving,
    which is the whole point.
    """
    eval_rng = kw.pop("eval_rng", "per-episode")
    args = _stub_kwargs(kw)
    pool = _StubPool(1)
    return _run_episodes_serial(pool, eval_rng=eval_rng, **args), pool


@pytest.mark.parametrize("lanes", [1, 2, 3, 4, 7])
def test_stub_parallel_reproduces_the_serial_loop_for_every_lane_count(lanes):
    """THE gate, at unit speed: the parallel executor's per-episode records
    must equal the SERIAL loop's, for every lane count including the uneven
    final wave. Baselining on the parallel executor at lanes=1 instead would
    make this test blind to any bug the parallel branch applies uniformly."""
    reference, _ = _run_stub_serial()
    got, _ = _run_stub(lanes)
    assert got == reference


@pytest.mark.parametrize(
    "regime",
    [
        {"sticky_prob": 0.25, "start_jitter": 16, "action_select": "greedy"},
        {"sticky_prob": 0.25, "start_jitter": 0, "action_select": "greedy"},
        {"sticky_prob": 0.0, "start_jitter": 16, "action_select": "greedy"},
        {"sticky_prob": 0.0, "start_jitter": 0, "action_select": "sampled"},
        {"sticky_prob": 0.25, "start_jitter": 16, "action_select": "sampled"},
        {"sticky_prob": 0.0, "start_jitter": 0, "action_select": "greedy"},
    ],
)
def test_stub_parallel_reproduces_the_serial_loop_in_every_regime(regime):
    """Each stochastic knob isolated, then together, then off.

    sticky-alone and jitter-alone matter individually: they are the two
    parameters the honest protocol is DEFINED by (sticky 0.25 / jitter 16), and
    a run with both on can hide an error in one of them behind the other's
    noise. jitter-alone pins the Machado range (`randint(0, jitter + 1)`, i.e.
    0..16 inclusive); sticky-alone pins the first-step guard (`step > 0`, so
    the very first action can never be a repeat).
    """
    reference, _ = _run_stub_serial(**dict(regime))
    got, _ = _run_stub(4, **dict(regime))
    assert got == reference


@pytest.mark.parametrize("lanes", [2, 4])
def test_stub_parallel_reproduces_the_serial_loop_in_sequential_mode(lanes):
    """--sequential plumbing, differentially: seq_clear / warp_taken /
    furthest_seq / furthest_any, and the tracker-driven episode termination.

    The tracker counts the frames it was shown, so this also pins that both
    loops seed it with the warm-start frame BEFORE the first action — the
    seeding that makes a level-clear probe's start level lock to where the
    episode actually began.
    """
    reference, _ = _run_stub_serial(make_tracker=_StubTracker)
    got, _ = _run_stub(lanes, make_tracker=_StubTracker)
    assert got == reference
    # Anti-vacuity: the tracker fields must actually vary, and the
    # tracker-driven terminator must actually fire at least once.
    assert len({r["furthest_seq"] for r in reference}) > 1
    assert any(r["seq_clear"] for r in reference)
    assert any(r["warp_taken"] for r in reference)


def test_stub_idle_lane_in_a_short_wave_burns_no_emulation():
    """A wave shorter than the lane count must PARK its unused workers before
    stepping, not merely at the end: an unparked lane is emulated for the whole
    wave for a trajectory nobody reads."""
    _, pool = _run_stub(3, n_episodes=2)
    assert pool.advanced[2] == 0
    assert pool.advanced[0] > 0 and pool.advanced[1] > 0


def test_stub_serial_reference_actually_exercises_the_high_gx_byte():
    """Anti-vacuity: the gx read packs a high byte (`ram[0x006D] << 8`), so at
    least one episode must run past 255 or a high-byte bug is unobservable."""
    reference, _ = _run_stub_serial()
    assert max(r["max_gx"] for r in reference) > 255


def test_stub_records_are_ordered_by_episode_not_by_completion():
    records, _ = _run_stub(4)
    assert len(records) == 7
    # Episodes finish out of order (different jitter + different draws), so an
    # append-on-completion executor would scramble them. The serial loop's
    # order IS episode order by construction.
    reference, _ = _run_stub_serial()
    assert records == reference


def test_stub_episodes_actually_differ_from_one_another():
    """Guards against a vacuous identity test: if every episode were the same
    trajectory, lane-invariance would be trivially true."""
    records, _ = _run_stub(3)
    assert len({r["length"] for r in records}) > 1
    assert len({r["max_gx"] for r in records}) > 1


def test_stub_changing_the_eval_seed_changes_the_episodes():
    a, _ = _run_stub(3, eval_seed=11)
    b, _ = _run_stub(3, eval_seed=12)
    assert a != b


def test_stub_deterministic_run_needs_no_rng_and_repeats_one_trajectory():
    """Greedy + no sticky + no jitter: every episode is the same rollout, which
    is why that regime can go wide under the default shared-stream mode."""
    records, _ = _run_stub(
        4, sticky_prob=0.0, start_jitter=0, action_select="greedy",
    )
    assert all(r == records[0] for r in records)


def test_stub_serial_shared_stream_is_a_different_run_but_reproducible():
    """The default RNG mode is NOT the per-episode mode's draws — it is the
    historical single stream. It must still be deterministic, and it must not
    be silently interchangeable with per-episode."""
    a, _ = _run_stub_serial(eval_rng="shared-stream")
    b, _ = _run_stub_serial(eval_rng="shared-stream")
    assert a == b
    per_episode, _ = _run_stub_serial(eval_rng="per-episode")
    assert a != per_episode


def test_stub_serial_shared_stream_and_per_episode_agree_when_nothing_draws():
    """Greedy + no sticky + no jitter never touches either stream, so the two
    modes are provably the same run — the exact claim
    `run_consumes_randomness` makes and the reason a wide deterministic run is
    allowed on the default RNG."""
    deterministic = {"sticky_prob": 0.0, "start_jitter": 0,
                     "action_select": "greedy"}
    shared, _ = _run_stub_serial(eval_rng="shared-stream", **dict(deterministic))
    per_ep, _ = _run_stub_serial(eval_rng="per-episode", **dict(deterministic))
    wide, _ = _run_stub(4, **dict(deterministic))
    assert shared == per_ep == wide


def test_stub_lengths_and_returns_track_the_stub_emulator():
    records, _ = _run_stub(2)
    for rec in records:
        assert 1 <= rec["length"] <= 30
        # The stub advances a fixed multiple of 1..3 x per step and the reward
        # is the x delta, so the return is the total distance travelled after
        # the no-op prefix.
        assert rec["ep_return"] >= 0.0
        assert rec["max_gx"] >= 0
        assert rec["cleared"] is (rec["max_gx"] >= _STUB_SUCCESS_X)


def test_stub_max_steps_caps_the_episode_length():
    records, _ = _run_stub(
        3, max_steps=5, sticky_prob=0.0, start_jitter=0, action_select="greedy",
    )
    assert {r["length"] for r in records} == {5}
    reference, _ = _run_stub_serial(
        max_steps=5, sticky_prob=0.0, start_jitter=0, action_select="greedy",
    )
    assert records == reference


def test_stub_zero_max_steps_reports_length_one_like_the_serial_loop():
    """`for step in range(0)` never runs; the serial loop still appends
    `step + 1` off its pre-loop `step = 0`. Asserted against that loop, not
    against a transcription of what it is believed to do."""
    records, _ = _run_stub(2, max_steps=0)
    reference, _ = _run_stub_serial(max_steps=0)
    assert records == reference
    assert [r["length"] for r in reference] == [1] * 7
    assert [r["ep_return"] for r in reference] == [0.0] * 7


def test_stub_warm_start_blob_reaches_every_lane_every_wave():
    blob = bytes([7]) + b"\x00" * 8
    records, pool = _run_stub(3, stage_blob=blob)
    # 7 episodes over 3 lanes = 3 waves (3 + 3 + 1) => one load per episode.
    assert len(pool.loads) == 7
    assert all(b == blob for _, b in pool.loads)
    assert pool.resets == 3
    # And the warm start is visible in the trajectory: x starts at 7, not 0.
    assert all(r["max_gx"] >= 7 for r in records)
    # Same warm start, same episodes: the serial loop loads worker 0 once per
    # episode and must land on the identical trajectories.
    reference, ref_pool = _run_stub_serial(stage_blob=blob)
    assert records == reference
    assert [w for w, _ in ref_pool.loads] == [0] * 7


def test_stub_parks_finished_and_unused_lanes():
    _, pool = _run_stub(4)
    parked = {w for w, flag in pool.done_calls if flag}
    # Every lane finishes at least once across the waves, and the final wave
    # (7 % 4 == 3 episodes) parks its one idle worker.
    assert parked == {0, 1, 2, 3}
    assert (3, True) in pool.done_calls


def test_stub_wide_run_takes_fewer_pool_steps_than_serial():
    """The point of the feature: episodes overlap instead of queueing.
    Measured against the real serial loop, not against a 1-lane parallel run."""
    _, serial = _run_stub_serial()
    _, wide = _run_stub(4)
    assert wide.step_calls < serial.step_calls


# --- ROM-gated end-to-end --------------------------------------------------

def _run_eval(ckpt: Path, *, episodes: int, max_steps: int,
              extra: list[str] | None = None) -> dict:
    """Subprocess eval_game in an isolated temp CWD (like the cold probe) with
    an absolutized profile, so no campaign eval.jsonl is touched.
    Mirrors tests/test_eval_action_select.py::_run_eval.
    """
    prof = yaml.safe_load(_TILE_CFG.read_text())
    prof["start_state_path"] = str(_START)
    slug = eval_game.derive_checkpoint_dir("./checkpoints", prof.get("name")).name
    with tempfile.TemporaryDirectory(prefix="eval_parallel_test_") as td:
        td = Path(td)
        (td / "checkpoints" / slug).mkdir(parents=True)
        pf = td / "probe_profile.yaml"
        pf.write_text(yaml.safe_dump(prof))
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "eval_game.py"),
            "--game", "mario", "--profile", str(pf), "--rom", str(_ROM),
            "--checkpoint", str(ckpt),
            "--episodes", str(episodes), "--max-steps", str(max_steps),
            "--eval-seed", "0",
        ] + list(extra or [])
        r = subprocess.run(cmd, cwd=str(td), capture_output=True, text=True,
                           timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


_EMU_SKIP = pytest.mark.skipif(
    not _HAVE_EMU or _TILE_CKPT is None,
    reason="needs the SMB ROM/start state and a tile one-shot checkpoint",
)


@pytest.mark.slow
@_EMU_SKIP
def test_default_receipt_reports_a_serial_shared_stream_run():
    out = _run_eval(_TILE_CKPT, episodes=2, max_steps=300)
    assert out["status"] == "ok", out
    assert out["eval_workers"] == 1
    assert out["eval_rng"] == "shared-stream"


@pytest.mark.slow
@_EMU_SKIP
def test_parallel_matches_the_serial_loop_under_sticky_and_jitter():
    """The honest-protocol regime, on real emulation: same seed, same episodes,
    identical per-episode outcomes.

    `--eval-workers 1` runs `_run_episodes_serial` and `--eval-workers 4` runs
    `_run_episodes_parallel` — the RNG mode does not pick the executor, the
    worker count does — so this is a true cross-implementation differential and
    not the parallel branch checking itself. It is also the only test here that
    covers the real stacker/reward-function reset semantics: the serial loop
    reuses ONE policy wrapper and ONE reward object across episodes while a
    lane builds its own, and this asserts those are the same thing.
    """
    honest = ["--sticky-prob", "0.25", "--start-jitter", "16",
              "--eval-rng", "per-episode"]
    one = _run_eval(_TILE_CKPT, episodes=4, max_steps=400,
                    extra=honest + ["--eval-workers", "1"])
    four = _run_eval(_TILE_CKPT, episodes=4, max_steps=400,
                     extra=honest + ["--eval-workers", "4"])
    assert one["status"] == "ok" and four["status"] == "ok"
    assert one["eval_workers"] == 1 and four["eval_workers"] == 4
    assert one["eval_rng"] == four["eval_rng"] == "per-episode"
    assert one["clear_rate"] == four["clear_rate"]
    assert one["max_gx_per_episode"] == four["max_gx_per_episode"]
    assert one["mean_return"] == four["mean_return"]
    assert one["mean_length"] == four["mean_length"]
    # Anti-vacuity: under jitter + sticky the episodes must not all collapse to
    # one trajectory, or agreement would be free.
    assert len(set(one["max_gx_per_episode"])) > 1, (
        "every episode reached the same gx, so this differential proved "
        f"nothing: {_TILE_CKPT} does not travel far enough under sticky + "
        "jitter for the episodes to diverge"
    )
