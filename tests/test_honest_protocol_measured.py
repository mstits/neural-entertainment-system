"""MEASURE the honest protocol's two defining constants inside the eval loop.

Every headline number in this repo is produced by one path,
`scripts/eval_game.py`, under one protocol: cold entrance, greedy, **sticky
p=0.25**, **start-jitter 16**, 50 episodes x 2 eval seeds. The receipt the
script writes carries `sticky_prob: 0.25` and `start_jitter: 16` — but those
fields are the values that were *requested*, echoed straight back out of the
argument list. Nothing in the receipt, and until this file nothing in the
suite, was a function of whether the mechanisms actually ran.

That gap is measurable and it was measured: deleting the sticky repeat from
BOTH executors (`if False and sticky_prob > 0.0 ...`) leaves 534 tests green,
including every test in `tests/test_eval_parallel.py`; deleting the Machado
no-op prologue from both leaves 141 green. The existing differential suite
compares `_run_episodes_serial` against `_run_episodes_parallel`, so it catches
a mutation in *one* branch and is blind to a mutation applied to both — which
is exactly what a removed constant, a mis-scaled probability, or a collapsed
range looks like.

So these tests do not compare the executors to each other. They compare each
executor to the PROTOCOL, by counting the events the protocol is defined by:

  * **sticky rate.** The stub policy's argmax is always `(prev_action + 1) %
    n`, so the freshly chosen action can never equal the previous executed one.
    Under that policy `executed == previous` happens if and only if the sticky
    roll fired, which turns the unobservable roll into a directly countable
    event: the divergent-stick rate IS p.
  * **jitter range.** The stub policy never emits action index 0 (the bitmask
    table is bijective and starts at 1), so the leading run of literal
    zero-action steps in a lane's column is exactly the `1 + jitter` no-op
    prologue and nothing else.

Both are asserted against a 5-sigma binomial/uniform band around the requested
value, so a halved probability or a collapsed range fails, not just a deleted
one. Seeds are fixed, so the numbers are reproducible, not flaky.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from eval_game import (  # noqa: E402
    _run_episodes_parallel, _run_episodes_serial,
)

# Bijective and never zero: the executed action index is recoverable from the
# bitmask the pool was handed, and a control step can never be mistaken for a
# no-op prologue step.
_N_ACTIONS = 6
_BITMASKS = tuple(1 << i for i in range(_N_ACTIONS))


class _ActionRecorderPool:
    """Records every action vector `step_all` was handed. Never ends an
    episode on its own, so episode length is exactly `max_steps` and the
    per-lane columns line up."""

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.vectors: list[tuple[int, ...]] = []
        # Index into `vectors` at which each wave began. The parallel executor
        # calls reset_all() exactly once per wave, which is what separates one
        # lane's parked tail from the next wave's no-op prologue.
        self.wave_starts: list[int] = []
        self._done = [False] * num_workers

    def reset_all(self):
        self.wave_starts.append(len(self.vectors))
        self._done = [False] * self.num_workers
        return [self._out() for _ in range(self.num_workers)]

    def load_worker_state(self, worker_id: int, blob: bytes) -> None:
        pass

    def set_worker_done(self, worker_id: int, done: bool) -> None:
        self._done[worker_id] = bool(done)

    def step_all(self, actions):
        self.vectors.append(tuple(int(a) for a in actions))
        return [self._out() for _ in range(self.num_workers)]

    def shutdown(self) -> None:
        pass

    @staticmethod
    def _out():
        return (b"frame", b"pp", bytes(2048), False)


class _CyclePolicy:
    """argmax is ALWAYS `(prev_action + 1) % n`.

    The freshly chosen action therefore never equals the previous executed
    action, so `executed == previous` is a sticky repeat and nothing else.
    """

    def reset(self, out):
        return np.zeros(1, dtype=np.float32)

    def push(self, out):
        return np.zeros(1, dtype=np.float32)

    def initial_hidden(self, device):
        return None

    def logits(self, obs, hidden, prev_action: int = 0):
        v = torch.full((1, _N_ACTIONS), -10.0)
        v[0, (int(prev_action) + 1) % _N_ACTIONS] = 10.0
        return v, hidden


class _NullReward:
    def reset(self) -> None:
        pass

    def compute(self, ram, action: int = 0):
        return 0.0, False, {}

    def episode_success(self) -> bool:
        return False


def _executor_kwargs(**over):
    kw = dict(
        n_episodes=25,
        max_steps=120,
        bitmasks=_BITMASKS,
        make_policy=_CyclePolicy,
        make_reward_fn=_NullReward,
        make_tracker=lambda: None,
        stage_blob=None,
        device=torch.device("cpu"),
        sticky_prob=0.25,
        start_jitter=16,
        eval_seed=20260827,
        action_select="greedy",
        temperature=1.0,
    )
    kw.update(over)
    return kw


def _columns(pool: _ActionRecorderPool, worker: int) -> list[int]:
    return [v[worker] for v in pool.vectors]


def _split_episodes_serial(pool, n_episodes: int, max_steps: int):
    """-> [(jitter, [executed action indices])] for the 1-worker serial loop."""
    col = _columns(pool, 0)
    out, i = [], 0
    for _ in range(n_episodes):
        noops = 0
        while col[i] == 0:
            noops += 1
            i += 1
        acts = [_BITMASKS.index(col[i + k]) for k in range(max_steps)]
        i += max_steps
        out.append((noops - 1, acts))
    return out


def _split_episodes_parallel(pool, lanes: int, n_episodes: int, max_steps: int):
    """-> [(jitter, [executed action indices])] in episode order.

    Within one wave a lane's column is `[0] * (1 + jitter)`, then `max_steps`
    non-zero control actions, then trailing zeros while the lane is parked
    waiting for its slower wave-mates. Wave boundaries come from the pool's
    recorded `reset_all()` calls, so a parked tail is never mistaken for the
    next wave's no-op prologue.
    """
    bounds = list(pool.wave_starts) + [len(pool.vectors)]
    out = []
    for w in range(len(pool.wave_starts)):
        lo, hi = bounds[w], bounds[w + 1]
        for worker in range(lanes):
            col = _columns(pool, worker)[lo:hi]
            if not any(col):
                continue  # lane carried no episode this wave
            noops = 0
            while col[noops] == 0:
                noops += 1
            acts = [_BITMASKS.index(col[noops + k]) for k in range(max_steps)]
            out.append((noops - 1, acts))
    assert len(out) == n_episodes, f"reconstructed {len(out)} of {n_episodes}"
    return out


def _stick_stats(episodes) -> tuple[int, int]:
    """(sticky repeats, eligible steps). Step 0 of an episode is not eligible:
    the protocol's roll is gated on `step > 0`."""
    repeats = eligible = 0
    for _jitter, acts in episodes:
        for t in range(1, len(acts)):
            eligible += 1
            if acts[t] == acts[t - 1]:
                repeats += 1
    return repeats, eligible


# --------------------------------------------------------------------------
# sticky p = 0.25 — the first of the two numbers the protocol IS
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.0, 0.25, 1.0])
def test_serial_loop_applies_sticky_at_the_requested_probability(p):
    pool = _ActionRecorderPool(1)
    kw = _executor_kwargs(sticky_prob=p, start_jitter=0)
    _run_episodes_serial(pool, eval_rng="per-episode", **kw)
    eps = _split_episodes_serial(pool, kw["n_episodes"], kw["max_steps"])
    repeats, eligible = _stick_stats(eps)
    rate = repeats / eligible
    se = (0.25 * 0.75 / eligible) ** 0.5  # widest binomial sd on [0, 1]
    assert abs(rate - p) <= 5 * se + 1e-12, (
        f"sticky_prob={p} requested, {rate:.4f} measured over {eligible} "
        f"eligible steps ({repeats} repeats). The chosen action can never "
        f"equal the previous one under _CyclePolicy, so this rate IS the "
        f"sticky-repeat probability the eval loop applied."
    )


@pytest.mark.parametrize("p", [0.0, 0.25, 1.0])
def test_parallel_executor_applies_sticky_at_the_requested_probability(p):
    lanes = 5
    pool = _ActionRecorderPool(lanes)
    kw = _executor_kwargs(sticky_prob=p, start_jitter=0)
    _run_episodes_parallel(pool, lanes=lanes, **kw)
    eps = _split_episodes_parallel(pool, lanes, kw["n_episodes"], kw["max_steps"])
    repeats, eligible = _stick_stats(eps)
    rate = repeats / eligible
    se = (0.25 * 0.75 / eligible) ** 0.5
    assert abs(rate - p) <= 5 * se + 1e-12, (
        f"sticky_prob={p} requested, {rate:.4f} measured over {eligible} "
        f"eligible steps in the {lanes}-lane executor"
    )


def test_sticky_never_repeats_on_an_episodes_first_step():
    """`step > 0` guard: the first action of every episode is the policy's own
    choice, never a carried repeat from the previous episode."""
    pool = _ActionRecorderPool(1)
    kw = _executor_kwargs(sticky_prob=1.0, start_jitter=3, n_episodes=30)
    _run_episodes_serial(pool, eval_rng="per-episode", **kw)
    eps = _split_episodes_serial(pool, kw["n_episodes"], kw["max_steps"])
    # prev_action is 0 at the top of every episode, so the untouched choice is
    # index 1. At sticky 1.0 every LATER step repeats it.
    assert all(acts[0] == 1 for _j, acts in eps)
    assert all(set(acts[1:]) == {1} for _j, acts in eps)


# --------------------------------------------------------------------------
# start-jitter 16 — the second of the two numbers the protocol IS
# --------------------------------------------------------------------------

def _jitters(pool, n_episodes, max_steps):
    return [j for j, _ in _split_episodes_serial(pool, n_episodes, max_steps)]


@pytest.mark.parametrize("jitter", [0, 16])
def test_serial_loop_draws_the_full_machado_noop_range(jitter):
    pool = _ActionRecorderPool(1)
    kw = _executor_kwargs(start_jitter=jitter, n_episodes=400, max_steps=2,
                          sticky_prob=0.0)
    _run_episodes_serial(pool, eval_rng="per-episode", **kw)
    js = _jitters(pool, kw["n_episodes"], kw["max_steps"])
    assert min(js) == 0
    assert max(js) == jitter, (
        f"--start-jitter {jitter} must be able to draw {jitter} no-op steps; "
        f"the widest draw observed over {len(js)} episodes was {max(js)}"
    )
    assert len(set(js)) == jitter + 1, (
        f"the draw must cover all {jitter + 1} values of "
        f"randint(0, start_jitter + 1); saw {sorted(set(js))}"
    )
    if jitter:
        # uniform on [0, jitter]: mean jitter/2, sd sqrt(((j+1)^2-1)/12)
        sd = (((jitter + 1) ** 2 - 1) / 12) ** 0.5
        assert abs(float(np.mean(js)) - jitter / 2) <= 5 * sd / len(js) ** 0.5


def test_parallel_executor_draws_the_full_machado_noop_range():
    lanes = 4
    pool = _ActionRecorderPool(lanes)
    kw = _executor_kwargs(start_jitter=16, n_episodes=400, max_steps=2,
                          sticky_prob=0.0)
    _run_episodes_parallel(pool, lanes=lanes, **kw)
    js = [j for j, _ in
          _split_episodes_parallel(pool, lanes, kw["n_episodes"], kw["max_steps"])]
    assert min(js) == 0 and max(js) == 16
    assert len(set(js)) == 17
    sd = ((17 ** 2 - 1) / 12) ** 0.5
    assert abs(float(np.mean(js)) - 8.0) <= 5 * sd / len(js) ** 0.5


# --------------------------------------------------------------------------
# the two RNG modes must be the SAME protocol, not merely both reproducible
# --------------------------------------------------------------------------

def test_both_eval_rng_modes_deliver_the_same_protocol():
    """`shared-stream` and `per-episode` receipts are pooled and compared to
    each other across the ledger (the banked 1-2 number is shared-stream; the
    v27/v28 gate numbers are per-episode). Their DRAWS differ by design; their
    DISTRIBUTIONS must not."""
    stats = {}
    for mode in ("shared-stream", "per-episode"):
        pool = _ActionRecorderPool(1)
        kw = _executor_kwargs(n_episodes=200, max_steps=60)
        _run_episodes_serial(pool, eval_rng=mode, **kw)
        eps = _split_episodes_serial(pool, kw["n_episodes"], kw["max_steps"])
        repeats, eligible = _stick_stats(eps)
        stats[mode] = (repeats / eligible, float(np.mean([j for j, _ in eps])),
                       eligible)
    for mode, (rate, mean_j, eligible) in stats.items():
        se = (0.25 * 0.75 / eligible) ** 0.5
        assert abs(rate - 0.25) <= 5 * se, f"{mode}: sticky {rate:.4f}"
        assert abs(mean_j - 8.0) <= 5 * (((17 ** 2 - 1) / 12) ** 0.5) / 200 ** 0.5, (
            f"{mode}: mean jitter {mean_j:.3f}")
