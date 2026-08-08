"""Tests for scripts/segment_probe.py — the S3 composition probe.

The instrument the World 1-2 verdict rests on ("from gx X, how often does the
policy reach gx Y under sticky?") lived only in /tmp and was lost; this suite
pins the restored version down so it cannot be lost or silently drift again.

Layers:
  * pure bank parsing / nearest-rung selection (no emulator) — including the
    naming-scheme trap where an act-indexed rung's action count looks like an
    area index;
  * pure statistics (Wilson interval, episode summarization) — the numbers a
    published negative is quoted from;
  * `run_segment` — the loop that actually produces the verdict — driven off a
    scripted pool/policy/reward fake, so the success predicate, the outcome
    precedence and the sticky/jitter RNG protocol are pinned behaviourally;
  * guard paths through `probe()` with the checkpoint resolver stubbed, so an
    empty or far-away bank fails loud instead of probing from gx 0;
  * a real-bank selection check and a ROM-gated end-to-end probe (marked slow).
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
# segment_probe.py imports `eval_game` / `train_game` by bare module name (they
# live under scripts/ and are not a package), so both roots must be importable.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from scripts import segment_probe  # noqa: E402
from scripts.segment_probe import (  # noqa: E402
    alive,
    bank_area,
    bank_gx,
    gx_of,
    run_segment,
    scan_bank,
    select_bank_state,
    summarize_episodes,
    wilson_interval,
)


def _find(rel_candidates: list[str]) -> Path | None:
    """Locate gitignored data (ROM / bank / checkpoint) in this checkout or an
    ancestor working copy. Mirrors tests/test_eval_pixel.py::_find."""
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            for hit in sorted(root.glob(rel)):
                if hit.exists():
                    return hit
    return None


_ROM = _find(["roms/Super Mario Bros. (World).nes"])
_START = _find(["roms/Super Mario Bros. (World)_start.state.bin"])
_BANK = _find(["checkpoints/ge_1_2_rungs_gx"])
_CKPT_1_2 = _find([
    "checkpoints/mario_1_2_cgsa_s1/vanilla_ppo_iter_*.pt",
    "checkpoints/mario_1_2_cgsa_s2/vanilla_ppo_iter_*.pt",
])
_CFG_1_2 = _ROOT / "configs" / "mario_1_2_cgsa_s1.yaml"


def _touch_states(d: Path, names: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")


# --- bank filename parsing --------------------------------------------------

def test_bank_gx_parses_every_shipped_naming_scheme():
    assert bank_gx("checkpoints/ge_1_2_rungs_gx/r_p05490_a2_gx1490.state") == 1490
    assert bank_gx("checkpoints/ge_1_2_rungs_act/r_a438_ar2_gx1812.state") == 1812
    assert bank_gx("checkpoints/ge_1_2_rungs_dense/rung_a020_area1_gx0.state") == 0
    assert bank_gx("checkpoints/mario_1_1_ladder/rung_2_gx2818.state") == 2818
    assert bank_gx("checkpoints/go_explore_2_2/beam_root.state") is None


def test_bank_area_prefers_the_area_field_over_an_action_count():
    """`r_a438_ar2_gx1812`: a438 is the ACTION COUNT, ar2 is the area. A bare
    `_a(\\d+)` probe reads 438 as the area and the area filter then silently
    selects nothing (or the wrong rung)."""
    assert bank_area("r_a438_ar2_gx1812.state") == 2
    assert bank_area("r_p05490_a2_gx1490.state") == 2
    assert bank_area("rung_a020_area1_gx0.state") == 1
    assert bank_area("r_p00092_a1_gx92.state") == 1
    assert bank_area("rung_2_gx2818.state") is None


def test_scan_bank_sorts_by_gx_and_skips_unlabeled(tmp_path):
    _touch_states(tmp_path / "b", [
        "r_p05490_a2_gx1490.state", "r_p04000_a2_gx0.state",
        "r_p00092_a1_gx92.state", "beam_root.state", "notes.txt",
    ])
    entries = scan_bank([tmp_path / "b", tmp_path / "missing"])
    assert [e["gx"] for e in entries] == [0, 92, 1490]
    assert [e["area"] for e in entries] == [2, 1, 2]
    assert all(e["path"].endswith(".state") for e in entries)


def test_scan_bank_empty_for_nonexistent_dirs(tmp_path):
    assert scan_bank([tmp_path / "nope"]) == []
    assert scan_bank([]) == []


# --- nearest-rung selection -------------------------------------------------

def _entries(*pairs) -> list[dict]:
    return [{"path": f"/bank/s{gx}_a{area}.state", "gx": gx, "area": area}
            for gx, area in pairs]


def test_select_bank_state_picks_the_nearest_gx():
    e = _entries((1430, 2), (1490, 2), (1524, 2), (1584, 2))
    assert select_bank_state(e, 1500)["gx"] == 1490
    assert select_bank_state(e, 1520)["gx"] == 1524
    assert select_bank_state(e, 99999)["gx"] == 1584


def test_select_bank_state_area_filter_disambiguates_overlapping_gx():
    """The 1-2 bank holds an overworld stub (area 1) and the underground main
    area (area 2) at overlapping gx — unfiltered, gx 100 is ambiguous."""
    e = _entries((92, 1), (112, 2))
    assert select_bank_state(e, 100, area=1)["gx"] == 92
    assert select_bank_state(e, 100, area=2)["gx"] == 112
    assert select_bank_state(e, 100, area=3) is None
    assert select_bank_state([], 100) is None


def test_select_bank_state_tie_breaks_to_the_shallower_rung():
    """Equidistant rungs must not hand a repeat probe an easier start."""
    e = _entries((1400, 2), (1600, 2))
    assert select_bank_state(e, 1500)["gx"] == 1400


# --- statistics -------------------------------------------------------------

def test_wilson_interval_zero_of_thirty_is_not_zero():
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0
    assert 0.10 < hi < 0.12          # the published 0/30 upper bound ~0.114
    lo, hi = wilson_interval(15, 30)
    assert lo == pytest.approx(0.3320, abs=1e-3)
    assert hi == pytest.approx(0.6680, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(30, 30)
    assert hi == 1.0 and lo > 0.85


def _rec(outcome: str, max_gx: int, steps: int = 100) -> dict:
    return {"outcome": outcome, "max_gx": max_gx, "steps": steps,
            "end_gx": max_gx}


def test_summarize_episodes_reproduces_a_0_of_30_negative():
    recs = [_rec("died", 1800 + i) for i in range(30)]
    out = summarize_episodes(recs, to_gx=2200)
    assert out["episodes"] == 30
    assert out["reached"] == 0
    assert out["reach_rate"] == 0.0
    assert out["outcomes"] == {"died": 30}
    assert out["reach_ci95"][0] == 0.0 and out["reach_ci95"][1] < 0.12
    assert out["best_max_gx"] == 1829
    assert out["min_max_gx"] == 1800
    assert out["mean_steps_to_reach"] is None
    assert out["max_gx_per_episode"] == [1800 + i for i in range(30)]


def test_summarize_episodes_tallies_every_outcome_and_reach_stats():
    recs = [
        _rec("reached", 2200, steps=400), _rec("reached", 2250, steps=600),
        _rec("died", 1827), _rec("timeout", 2100), _rec("left_area", 900),
    ]
    out = summarize_episodes(recs, to_gx=2200)
    assert out["to_gx"] == 2200
    assert out["reached"] == 2
    assert out["reach_rate"] == pytest.approx(0.4)
    assert out["outcomes"] == {"reached": 2, "died": 1, "timeout": 1,
                               "left_area": 1}
    assert out["mean_steps_to_reach"] == pytest.approx(500.0)
    assert out["median_max_gx"] == pytest.approx(2100.0)
    assert out["mean_max_gx"] == pytest.approx(np.mean([2200, 2250, 1827, 2100, 900]))


def test_summarize_episodes_handles_no_episodes():
    out = summarize_episodes([], to_gx=2200)
    assert out["episodes"] == 0 and out["reach_rate"] == 0.0
    assert out["max_gx_per_episode"] == [] and out["best_max_gx"] == 0


# --- RAM taps ---------------------------------------------------------------

def test_gx_of_and_alive_read_the_harness_addresses():
    ram = np.zeros(0x0800, dtype=np.uint8)
    ram[0x006D], ram[0x0086] = 8, 0x30
    assert gx_of(ram) == (8 << 8) | 0x30
    assert alive(ram) is True
    for dying in (6, 11):
        ram[0x000E] = dying
        assert alive(ram) is False
    ram[0x000E] = 8
    assert alive(ram) is True


# --- run_segment: the loop that produces the verdict ------------------------
#
# `run_segment` is fully duck-typed (pool.reset_all / load_worker_state /
# step_all -> [(frame, prep, ram, done)]; pol.reset / initial_hidden / logits /
# push; reward_fn.reset / compute), so the whole loop runs off scripted RAM
# with no emulator. These tests exist because the pure helpers alone let seven
# independent mutations inside run_segment survive: scoring `left_area` as a
# win, dropping the `alive` clause from the success predicate, `>=` -> `>` on
# the target gx, the area check never firing, sticky firing on step 0, an
# exclusive jitter bound, and reporting a post-jitter start gx.

_BM = (0x80, 0x81, 0x82, 0x83)   # all NON-ZERO: a 0 bitmask is a no-op frame


def _ram(gx: int, *, area: int = 2, pstate: int = 8) -> np.ndarray:
    ram = np.zeros(0x0800, dtype=np.uint8)
    ram[0x006D] = (gx >> 8) & 0xFF
    ram[0x0086] = gx & 0xFF
    ram[0x0760] = area
    ram[0x000E] = pstate
    return ram


class _ScriptedPool:
    """Duck-typed `nes_core.Pool`. `plan(i)` gives the RAM for step index i,
    where i == 0 is the post-load no-op flush (so i counts jitter frames too,
    exactly as the real pool would advance the game during them)."""

    def __init__(self, plan):
        self._plan = plan
        self._i = -1
        self.resets = 0
        self.loads: list[tuple[int, bytes]] = []
        self.frames: list[list[int]] = []   # bitmask per step_all, per episode

    def reset_all(self):
        self.resets += 1

    def load_worker_state(self, worker, blob):
        self.loads.append((worker, blob))
        self._i = -1
        self.frames.append([])

    def step_all(self, actions):
        self._i += 1
        self.frames[-1].append(int(actions[0]))
        st = self._plan(self._i)
        ram = _ram(st["gx"], area=st.get("area", 2), pstate=st.get("pstate", 8))
        return [(None, None, ram, bool(st.get("done", False)))]

    def prologue(self, ep: int) -> int:
        """Leading no-op frames of an episode = 1 flush + the jitter draw."""
        n = 0
        while n < len(self.frames[ep]) and self.frames[ep][n] == 0:
            n += 1
        return n

    def jitter(self, ep: int) -> int:
        return self.prologue(ep) - 1

    def actions(self, ep: int) -> list[int]:
        return self.frames[ep][self.prologue(ep):]


def _plan(script: list[dict]):
    """Step-indexed RAM script; the final entry repeats forever."""
    return lambda i: script[min(i, len(script) - 1)]


def _ramp(start: int, step: int, n: int, **kw) -> list[dict]:
    return [{"gx": start + step * i, **kw} for i in range(n)]


class _ScriptedPolicy:
    """Duck-typed eval policy: `fn(step)` is the argmax action index, where
    `step` is the 0-based index within the episode's action loop."""

    def __init__(self, fn, n_actions: int = len(_BM), logit_gap: float = 20.0):
        self._fn = fn
        self._n = n_actions
        self._gap = float(logit_gap)
        self.step = 0
        self.resets = 0

    def reset(self, step_tuple):
        self.resets += 1
        self.step = 0
        return np.zeros(1, dtype=np.float32)

    def push(self, step_tuple):
        return np.zeros(1, dtype=np.float32)

    def initial_hidden(self, device):
        return None

    def logits(self, obs, hidden):
        out = torch.zeros((1, self._n))
        out[0, int(self._fn(self.step)) % self._n] = self._gap
        self.step += 1
        return out, hidden


class _ScriptedReward:
    """Duck-typed reward function; terminates on the `done_at`-th action step
    (1-based) so the reward-side death signal can be tested on its own."""

    def __init__(self, done_at=None):
        self.done_at = done_at
        self.resets = 0
        self.step = 0

    def reset(self):
        self.resets += 1
        self.step = 0

    def compute(self, ram, action=0):
        self.step += 1
        return 0.0, (self.done_at is not None and self.step >= self.done_at), {}


def _run(script, *, to_gx, episodes=1, sticky_prob=0.0, start_jitter=0,
         max_steps=10, policy_fn=lambda s: 1, reward_done_at=None,
         eval_seed=0, action_select="greedy", logit_gap=20.0):
    pool = _ScriptedPool(_plan(script))
    pol = _ScriptedPolicy(policy_fn, logit_gap=logit_gap)
    rf = _ScriptedReward(reward_done_at)
    records, start = run_segment(
        pool, pol, list(_BM), b"blob", rf, torch.device("cpu"), to_gx=to_gx,
        episodes=episodes, sticky_prob=sticky_prob, start_jitter=start_jitter,
        action_select=action_select, eval_seed=eval_seed, max_steps=max_steps,
    )
    return pool, pol, rf, records, start


def _reference_stream(*, eval_seed, episodes, start_jitter, sticky_prob,
                      max_steps, policy_fn):
    """Independent re-derivation of the perturbation protocol: ONE
    RandomState(eval_seed) shared by every episode, consumed jitter-draw first
    then one sticky draw per action step AFTER the first."""
    rng = np.random.RandomState(eval_seed)
    jitters, streams = [], []
    for _ in range(episodes):
        jitters.append(int(rng.randint(0, start_jitter + 1)) if start_jitter > 0 else 0)
        prev, stream = 0, []
        for step in range(max_steps):
            idx = int(policy_fn(step)) % len(_BM)
            if sticky_prob > 0.0 and step > 0 and rng.random() < sticky_prob:
                idx = prev
            prev = idx
            stream.append(_BM[idx])
        streams.append(stream)
    return jitters, streams


def test_run_segment_success_is_the_exact_gx_boundary():
    """`end_gx >= to_gx`, not `>`: a run that lands exactly ON the target is a
    reach. The script clamps at 1100 so `>` can never fire."""
    script = _ramp(1000, 20, 6)          # 1000,1020,...,1100 then held
    _, _, _, recs, _ = _run(script, to_gx=1100, max_steps=10)
    assert recs[0]["outcome"] == "reached"
    assert recs[0]["end_gx"] == 1100
    assert recs[0]["steps"] == 5         # loop index 4 -> 5 steps taken
    # One pixel further is genuinely out of reach for the same script.
    _, _, _, recs, _ = _run(script, to_gx=1101, max_steps=10)
    assert recs[0]["outcome"] == "timeout"
    assert recs[0]["max_gx"] == 1100


def test_run_segment_crossing_the_target_while_dying_is_not_a_win():
    """Dropping `and alive(ram)` turns a corpse sliding past the target into a
    fabricated segment win — the confluence clear-detector defect class."""
    dead = _ramp(1000, 50, 4) + [{"gx": 1200, "pstate": 6}]
    _, _, _, recs, _ = _run(dead, to_gx=1200, max_steps=8)
    assert recs[0]["outcome"] == "timeout"
    assert recs[0]["max_gx"] == 1200     # it DID cross; it just was not alive
    # pstate 11 (the other dying value) is equally disqualifying.
    _, _, _, recs, _ = _run(_ramp(1000, 50, 4) + [{"gx": 1200, "pstate": 11}],
                            to_gx=1200, max_steps=8)
    assert recs[0]["outcome"] == "timeout"
    # ...and the predicate is not simply dead: recover, and the same crossing
    # scores on the next step.
    revive = _ramp(1000, 50, 4) + [{"gx": 1200, "pstate": 6},
                                   {"gx": 1200, "pstate": 8}]
    _, _, _, recs, _ = _run(revive, to_gx=1200, max_steps=8)
    assert recs[0]["outcome"] == "reached"


def test_run_segment_leaving_the_area_is_never_a_win():
    """Area change ends the measurement (gx restarts in the new area). It must
    be reported as `left_area` and must never be scored as `reached`, even when
    the new area's gx happens to be past the target."""
    left_past_target = [{"gx": 1000}, {"gx": 1100}, {"gx": 9999, "area": 3}]
    _, _, _, recs, _ = _run(left_past_target, to_gx=1100 + 1, max_steps=8)
    assert recs[0]["outcome"] == "left_area"
    assert recs[0]["end_gx"] == 1100     # last in-area gx, not the new area's
    assert summarize_episodes(recs, to_gx=1101)["reach_rate"] == 0.0
    # The area check must also fire when the target was never in play.
    left_short = [{"gx": 1000}, {"gx": 1020}, {"gx": 40, "area": 1}]
    _, _, _, recs, _ = _run(left_short, to_gx=9999, max_steps=8)
    assert recs[0]["outcome"] == "left_area"


def test_run_segment_outcome_precedence_reached_beats_died():
    """Reaching the target alive on the very step the episode terminates is a
    reach, not a death — and either terminal signal alone is a death."""
    both = [{"gx": 1000}, {"gx": 2200, "done": True}]
    _, _, _, recs, _ = _run(both, to_gx=2200, max_steps=8)
    assert recs[0]["outcome"] == "reached"
    _, _, _, recs, _ = _run([{"gx": 1000}, {"gx": 2200}], to_gx=2200,
                            max_steps=8, reward_done_at=1)
    assert recs[0]["outcome"] == "reached"
    # Pool done flag alone.
    _, _, _, recs, _ = _run([{"gx": 1000}, {"gx": 1100, "done": True}],
                            to_gx=9999, max_steps=8)
    assert recs[0]["outcome"] == "died" and recs[0]["steps"] == 1
    # Reward-function terminal alone.
    _, _, _, recs, _ = _run(_ramp(1000, 10, 4), to_gx=9999, max_steps=8,
                            reward_done_at=2)
    assert recs[0]["outcome"] == "died" and recs[0]["steps"] == 2


def test_run_segment_reports_the_pre_jitter_start_gx():
    """The reported start is the rung's own gx (after the flush no-op, before
    jitter), so two probes of the same rung report the same segment."""
    script = _ramp(1000, 10, 40)         # every frame, no-ops included, moves gx
    pool, _, _, _, start = _run(script, to_gx=9999, start_jitter=8, max_steps=3,
                                eval_seed=0)
    assert pool.jitter(0) > 0, "seed 0 must actually jitter for this to bite"
    assert start["start_gx"] == 1000     # plan(0) — NOT 1000 + 10 * jitter
    assert start["start_area"] == 2


def test_run_segment_jitter_bound_is_inclusive_and_seed_faithful():
    """`randint(0, start_jitter + 1)`: the full 0..start_jitter range must be
    reachable, and the draw must come from the shared eval-seed stream."""
    observed = {}
    for seed in range(40):
        pool, _, _, _, _ = _run([{"gx": 1000}], to_gx=9999, start_jitter=4,
                                max_steps=1, eval_seed=seed)
        observed[seed] = pool.jitter(0)
        expected, _ = _reference_stream(
            eval_seed=seed, episodes=1, start_jitter=4, sticky_prob=0.0,
            max_steps=1, policy_fn=lambda s: 1)
        assert observed[seed] == expected[0], seed
    assert set(observed.values()) == {0, 1, 2, 3, 4}, observed
    # start_jitter 0 draws nothing at all: only the flush no-op is consumed.
    pool, _, _, _, _ = _run([{"gx": 1000}], to_gx=9999, start_jitter=0,
                            max_steps=1)
    assert pool.prologue(0) == 1


def test_run_segment_sticky_cannot_fire_on_the_first_step():
    """At step 0 there is no previous action; `step >= 0` would let sticky
    replay the initialiser (index 0) and silently pin the whole episode to it,
    because the sticky repeat then feeds itself."""
    pool, _, _, _, _ = _run(_ramp(1000, 5, 30), to_gx=9999, sticky_prob=1.0,
                            max_steps=6, policy_fn=lambda s: 3)
    assert pool.actions(0) == [_BM[3]] * 6
    # With sticky certain, every later step repeats the FIRST policy action
    # even though the policy keeps changing its mind.
    pool, _, _, _, _ = _run(_ramp(1000, 5, 30), to_gx=9999, sticky_prob=1.0,
                            max_steps=6, policy_fn=lambda s: s)
    assert pool.actions(0) == [_BM[0]] * 6


def test_run_segment_action_stream_matches_the_seeded_protocol():
    """End-to-end pin of the perturbation protocol across episodes: one shared
    RandomState, jitter drawn before the sticky draws, one sticky draw per
    action step after the first."""
    kw = dict(eval_seed=7, episodes=3, start_jitter=4, sticky_prob=0.25,
              max_steps=8, policy_fn=lambda s: s)
    pool, _, _, recs, _ = _run(_ramp(1000, 5, 60), to_gx=9999, **kw)
    jitters, streams = _reference_stream(**kw)
    assert [pool.jitter(e) for e in range(3)] == jitters
    assert [pool.actions(e) for e in range(3)] == streams
    assert [r["outcome"] for r in recs] == ["timeout"] * 3
    assert [r["steps"] for r in recs] == [8] * 3


def test_run_segment_zero_sticky_consumes_no_sticky_draws():
    """`sticky_prob > 0.0` short-circuits: with sticky off the RandomState is
    spent on jitter alone, so episode 2's jitter is the SECOND jitter draw."""
    kw = dict(eval_seed=3, episodes=3, start_jitter=6, sticky_prob=0.0,
              max_steps=5, policy_fn=lambda s: s)
    pool, _, _, _, _ = _run(_ramp(1000, 5, 60), to_gx=9999, **kw)
    jitters, streams = _reference_stream(**kw)
    assert [pool.jitter(e) for e in range(3)] == jitters
    assert [pool.actions(e) for e in range(3)] == streams


def test_run_segment_tracks_the_high_water_gx_not_the_final_one():
    script = [{"gx": 1000}, {"gx": 1200}, {"gx": 1100}, {"gx": 1050}]
    _, _, _, recs, _ = _run(script, to_gx=9999, max_steps=3)
    assert recs[0] == {"outcome": "timeout", "max_gx": 1200, "steps": 3,
                       "end_gx": 1050}


def test_run_segment_reloads_the_rung_and_resets_the_reward_each_episode():
    pool, pol, rf, recs, _ = _run(_ramp(1000, 5, 20), to_gx=9999, episodes=4,
                                  max_steps=3)
    assert len(recs) == 4
    assert pool.resets == 4
    assert pool.loads == [(0, b"blob")] * 4
    assert rf.resets == 4 and pol.resets == 4


def test_run_segment_zero_max_steps_reports_zero_steps():
    """Degenerate but must not claim a step that never happened."""
    _, _, _, recs, _ = _run([{"gx": 1000}], to_gx=9999, max_steps=0)
    assert recs[0] == {"outcome": "timeout", "max_gx": 1000, "steps": 0,
                       "end_gx": 1000}


def test_run_segment_sampled_mode_draws_from_the_policy_distribution():
    """`--action-select sampled` must actually draw from the softmax rather
    than fall through to argmax: a near-flat policy has to produce more than
    one action index, where greedy on the identical script produces exactly
    one. This is the Brute check the honest protocol turns on."""
    pool, _, _, _, _ = _run(_ramp(1000, 5, 60), to_gx=9999, max_steps=40,
                            action_select="sampled", policy_fn=lambda s: 1,
                            eval_seed=1, logit_gap=0.2)
    assert len(set(pool.actions(0))) > 1
    pool, _, _, _, _ = _run(_ramp(1000, 5, 60), to_gx=9999, max_steps=40,
                            action_select="greedy", policy_fn=lambda s: 1,
                            eval_seed=1, logit_gap=0.2)
    assert set(pool.actions(0)) == {_BM[1]}


# --- probe() guard paths (checkpoint resolver stubbed; no emulator) ---------

def _stub_ckpt(monkeypatch, tmp_path: Path) -> Path:
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setattr(segment_probe, "resolve_eval_checkpoint",
                        lambda *a, **k: ckpt)
    return ckpt


def _profile(tmp_path: Path) -> Path:
    p = tmp_path / "prof.yaml"
    p.write_text(yaml.safe_dump({"name": "Segment Probe Test"}))
    return p


def test_probe_rejects_an_empty_bank(monkeypatch, tmp_path):
    _stub_ckpt(monkeypatch, tmp_path)
    out = segment_probe.probe("mario", _profile(tmp_path), "rom.nes",
                              to_gx=2200, from_gx=1500,
                              banks=[str(tmp_path / "empty")])
    assert out["status"] == "empty_bank"


def test_probe_refuses_a_bank_with_nothing_near_the_requested_gx(monkeypatch, tmp_path):
    """Without this guard the probe silently starts from gx 0 and reports a
    segment nobody asked for."""
    _stub_ckpt(monkeypatch, tmp_path)
    _touch_states(tmp_path / "bank", ["r_p04000_a2_gx0.state"])
    out = segment_probe.probe("mario", _profile(tmp_path), "rom.nes",
                              to_gx=2200, from_gx=1500,
                              banks=[str(tmp_path / "bank")])
    assert out["status"] == "no_bank_state_near_gx"
    assert out["nearest"]["gx"] == 0
    out = segment_probe.probe("mario", _profile(tmp_path), "rom.nes",
                              to_gx=2200, from_gx=1500,
                              banks=[str(tmp_path / "bank")], max_gx_delta=5000)
    # Tolerance widened: the guard yields and the run proceeds past selection.
    assert out["status"] != "no_bank_state_near_gx"


def test_probe_requires_a_start_state_or_from_gx(monkeypatch, tmp_path):
    _stub_ckpt(monkeypatch, tmp_path)
    out = segment_probe.probe("mario", _profile(tmp_path), "rom.nes", to_gx=2200)
    assert out["status"] == "no_start_state"


def test_probe_reports_a_missing_explicit_start_state(monkeypatch, tmp_path):
    _stub_ckpt(monkeypatch, tmp_path)
    out = segment_probe.probe("mario", _profile(tmp_path), "rom.nes", to_gx=2200,
                              start_state=str(tmp_path / "nope.state"))
    assert out["status"] == "no_such_start_state"


def test_probe_rejects_a_bad_action_select_or_temperature(monkeypatch, tmp_path):
    """The action-select vocabulary is eval_game's ("greedy"/"sampled"); a
    near-miss like "sample" must not silently fall through to argmax and be
    reported as a sampled result."""
    _stub_ckpt(monkeypatch, tmp_path)
    assert "sample" not in segment_probe._ACTION_SELECT_MODES
    with pytest.raises(ValueError):
        segment_probe.probe("mario", _profile(tmp_path), "rom.nes", to_gx=2200,
                            start_state="x", action_select="sample")
    with pytest.raises(ValueError):
        segment_probe.probe("mario", _profile(tmp_path), "rom.nes", to_gx=2200,
                            start_state="x", temperature=0.0)


# --- real bank ---------------------------------------------------------------

@pytest.mark.skipif(_BANK is None, reason="needs the 1-2 rung bank")
def test_real_1_2_bank_resolves_the_published_s3_start():
    """`--from-gx 1500 --from-area 2` against the shipped bank must land on the
    rung the published S3 probe started from."""
    entries = scan_bank([_BANK])
    assert len(entries) > 50
    pick = select_bank_state(entries, 1500, area=2)
    assert pick is not None
    assert abs(pick["gx"] - 1500) <= 30, pick
    assert pick["area"] == 2
    # And the destination rung exists, so the segment is bank-representable.
    assert select_bank_state(entries, 2200, area=2)["gx"] == pytest.approx(2200, abs=30)


# --- ROM-gated end-to-end ----------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    _ROM is None or _START is None or _BANK is None or _CKPT_1_2 is None
    or not _CFG_1_2.exists(),
    reason="needs the SMB ROM/start state, the 1-2 rung bank and a 1-2 checkpoint",
)
@pytest.mark.parametrize("action_select", ["greedy", "sampled"])
def test_segment_probe_end_to_end_reports_a_verdict(action_select):
    """Real emulator, real checkpoint, 2 short episodes: the probe must land on
    the gx1500 rung, measure it from RAM, and emit a complete verdict block —
    in both action-selection modes the honest protocol requires reporting."""
    prof = yaml.safe_load(_CFG_1_2.read_text())
    prof["start_state_path"] = str(_START)
    with tempfile.TemporaryDirectory(prefix="segment_probe_test_") as td:
        td = Path(td)
        (td / "checkpoints").mkdir(parents=True)
        pf = td / "probe_profile.yaml"
        pf.write_text(yaml.safe_dump(prof))
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "segment_probe.py"),
            "--game", "mario", "--profile", str(pf), "--rom", str(_ROM),
            "--checkpoint", str(_CKPT_1_2), "--bank", str(_BANK),
            "--from-gx", "1500", "--from-area", "2", "--to-gx", "2200",
            "--episodes", "2", "--max-steps", "300", "--sticky-prob", "0.25",
            "--start-jitter", "16", "--action-select", action_select,
        ]
        r = subprocess.run(cmd, cwd=str(td), capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        out = json.loads(r.stdout)
        # The per-episode progress stream is on stderr, so stdout stays pure JSON.
        assert "[segment]" in r.stderr
        # The run is logged next to the checkpoint like eval.jsonl.
        logs = list(td.glob("checkpoints/*/segment_probe.jsonl"))
        assert len(logs) == 1 and logs[0].read_text().strip()
    assert out["status"] == "ok", out
    assert out["episodes"] == 2
    assert out["to_gx"] == 2200
    assert abs(out["from_gx_measured"] - 1500) <= 100, out
    assert out["from_area_measured"] == 2
    assert out["bank_entry"]["area"] == 2
    assert 0.0 <= out["reach_rate"] <= 1.0
    assert len(out["max_gx_per_episode"]) == 2
    assert set(out["outcomes"]) <= {"reached", "died", "timeout", "left_area"}
    assert out["reach_ci95"][0] <= out["reach_rate"] <= out["reach_ci95"][1]
    assert out["action_select"] == action_select and out["sticky_prob"] == 0.25


@pytest.mark.slow
@pytest.mark.skipif(
    _ROM is None or _START is None or _BANK is None or _CKPT_1_2 is None
    or not _CFG_1_2.exists(),
    reason="needs the SMB ROM/start state, the 1-2 rung bank and a 1-2 checkpoint",
)
def test_segment_probe_trivial_target_is_reached_from_the_bank_state():
    """Sanity floor for the success predicate: a target BEHIND the start gx is
    reached immediately by every episode. If this fails, a 0/30 from this
    instrument means nothing."""
    prof = yaml.safe_load(_CFG_1_2.read_text())
    prof["start_state_path"] = str(_START)
    with tempfile.TemporaryDirectory(prefix="segment_probe_test_") as td:
        td = Path(td)
        (td / "checkpoints").mkdir(parents=True)
        pf = td / "probe_profile.yaml"
        pf.write_text(yaml.safe_dump(prof))
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "segment_probe.py"),
            "--game", "mario", "--profile", str(pf), "--rom", str(_ROM),
            "--checkpoint", str(_CKPT_1_2), "--bank", str(_BANK),
            "--from-gx", "1500", "--from-area", "2", "--to-gx", "1400",
            "--episodes", "2", "--max-steps", "60", "--sticky-prob", "0.25",
            "--start-jitter", "0",
        ]
        r = subprocess.run(cmd, cwd=str(td), capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        out = json.loads(r.stdout)
    assert out["reach_rate"] == 1.0, out
    assert out["outcomes"] == {"reached": 2}
