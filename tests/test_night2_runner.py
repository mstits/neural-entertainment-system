"""Night-2 orchestrator (SMB 1-2) — SIL persistence determination, BC
lock-in cure, pre-registered gate, and consol2 handoff of
scripts/night2_runner.py.

Everything here runs on SYNTHETIC payloads and tensors — synthetic
checkpoint payloads for the SIL-extraction logic, tiny TilePolicyNetwork
state dicts for the BC cure, stubbed probe summaries for the gate, and a
fake pool for the collection loop. No emulator, no rollout, no training
subprocess. The one live-ish test is the --dry-run subprocess, which is
explicitly allowed to parse configs, load the source checkpoint, and
assemble eval commands (none of which is a rollout).

Pre-registration under test (mirrors of the in-file CONFIG):
* step 1 — SIL acquisition: extract persisted (obs, action) trajectories
  from the source checkpoint payload IF any are persisted; otherwise
  collect fresh trajectories by running the final checkpoint SAMPLED
  under the house noise (sticky 0.25, jitter 16) from the 1-2 entrance,
  keeping episodes whose max_gx >= 3267 or that scored episode_success
  (BOTH labels stored per trajectory);
* step 2 — BC cure: 15 epochs cross-entropy, ACTOR PARAMETERS ONLY
  (trunk + critic frozen bit-identical), lr 1e-4 linear decay to 0,
  batch 256, saved in the standard vanilla_ppo_iter net_state_dict
  format at the consol2 continuation iter + 1;
* step 3 — gate: 50-episode cold ARGMAX honest probe (sticky 0.25,
  jitter 16) median max-x must EXCEED 2059, AND a 10-episode zero-noise
  deterministic probe from rung s_000559 median must EXCEED 2979; a
  failed probe leg can never pass the gate;
* step 4 — on PASS only: seed the cured net where run_consol2.py's
  resume picks it up, then exec run_consol2.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.night2_runner import (  # noqa: E402
    CONFIG, CURED_SEED_ITER, bc_cure, build_collection_reference_command,
    build_det_probe_command, build_honest_probe_command, build_manifest,
    collect_trajectories, collection_plan, extract_sil_pairs, gate_verdict,
    keep_episode, linear_decay_factor, seed_cured_for_consol2,
    sil_persistence_verdict, steps_from, trainable_param_names,
    write_cured_checkpoint, write_sil_pairs,
)
from scripts.run_consol2 import CONFIG as CONSOL2_CONFIG  # noqa: E402


# ---- synthetic payload helpers -----------------------------------------


def _tile_state_dict(num_actions=4, feature_dim=12, hidden=8, trunk=6):
    from src.models.tile_policy import TilePolicyNetwork
    torch.manual_seed(7)
    net = TilePolicyNetwork(num_actions=num_actions, feature_dim=feature_dim,
                            hidden_dim=hidden, trunk_dim=trunk)
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _real_shaped_payload() -> dict:
    """The key set the checkpoint manager actually persists (verified by
    torch.load on online_v2_FINAL_consolidated.pt): NO SIL key exists."""
    sd = _tile_state_dict()
    return {
        "iter": 910,
        "net_state_dict": sd,
        "optimizer_state_dict": {"state": {0: {"exp_avg": torch.zeros(3)}},
                                 "param_groups": [{"lr": 1e-4}]},
        "anticollapse": {"best_net_snapshot": sd,
                         "best_snapshot_fitness": 1.0,
                         "collapse_strikes": 0},
        "backward_curriculum": {"tau": -1},
        "curriculum_resume": {"advance_history": []},
    }


def _traj(length, feature_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.integers(-4, 4, size=(length, feature_dim)).astype(np.int8)
    act = rng.integers(0, 4, size=(length,)).astype(np.int64)
    return obs, act


# ---- step 1: SIL persistence determination ------------------------------


def test_real_shaped_payload_reports_sil_absent():
    """The verified production payload key set (iter / net_state_dict /
    optimizer / anticollapse / backward_curriculum / curriculum_resume)
    carries no SIL trajectories — the verdict must say so and extraction
    must return None, which is what flips step 1 into collection mode."""
    payload = _real_shaped_payload()
    verdict = sil_persistence_verdict(payload)
    assert verdict["persisted"] is False
    assert verdict["sil_keys"] == []
    assert set(verdict["payload_keys"]) == set(payload.keys())
    assert extract_sil_pairs(payload) is None


def test_sil_buffer_trajs_extracted_in_order():
    o1, a1 = _traj(9, seed=1)
    o2, a2 = _traj(4, seed=2)
    payload = {"iter": 3, "net_state_dict": {},
               "sil_buffer": {"trajs": [(o1, a1), (o2, a2)],
                              "total_clears": 2}}
    verdict = sil_persistence_verdict(payload)
    assert verdict["persisted"] is True
    assert verdict["n_trajs"] == 2
    assert verdict["n_steps"] == 13
    obs, act, traj_len = extract_sil_pairs(payload)
    assert obs.shape == (13, 5)
    assert act.dtype == np.int64
    assert traj_len.tolist() == [9, 4]
    np.testing.assert_array_equal(obs[:9], o1)
    np.testing.assert_array_equal(act[9:], a2)


def test_sil_key_detected_when_nested():
    o, a = _traj(6)
    payload = {"iter": 1, "mechanisms": {"self_imitation": [(o, a)]}}
    verdict = sil_persistence_verdict(payload)
    assert verdict["persisted"] is True
    obs, act, traj_len = extract_sil_pairs(payload)
    assert traj_len.tolist() == [6]


def test_sil_flat_obs_actions_dict_detected():
    o, a = _traj(7)
    payload = {"sil_pairs": {"obs": o, "actions": a}}
    assert sil_persistence_verdict(payload)["persisted"] is True
    obs, act, traj_len = extract_sil_pairs(payload)
    assert obs.shape[0] == 7 and traj_len.tolist() == [7]


def test_sil_pair_length_mismatch_raises():
    o, a = _traj(5)
    payload = {"sil_buffer": {"trajs": [(o, a[:3])]}}
    with pytest.raises(ValueError):
        extract_sil_pairs(payload)


def test_sil_pairs_npz_roundtrip(tmp_path):
    o, a = _traj(11)
    out = tmp_path / "sil_pairs.npz"
    write_sil_pairs(
        out, obs=o, act=a, traj_len=np.array([11]),
        label_max_gx=np.array([3400]),
        label_episode_success=np.array([True]),
        provenance={"mode": "collection", "source": "synthetic"})
    d = np.load(out)
    # House demo convention (obs_0 / act_0 / n=1) so downstream distill
    # scripts can consume the pairs unchanged, plus the night-2 labels.
    assert int(d["n"]) == 1
    np.testing.assert_array_equal(d["obs_0"], o)
    np.testing.assert_array_equal(d["act_0"], a)
    assert d["traj_len"].tolist() == [11]
    assert d["label_max_gx"].tolist() == [3400]
    # Both success predicates, named, per trajectory (2026-08-15
    # forensics: a rate without its predicate is uninterpretable).
    assert d["label_reached_gx"].tolist() == [True]
    assert d["label_episode_success"].tolist() == [True]
    prov = json.loads(str(d["provenance"].item()))
    assert prov["mode"] == "collection"


def test_keep_episode_predicate():
    assert keep_episode({"max_gx": 3267, "episode_success": False},
                        CONFIG["keep_min_max_gx"]) is True
    assert keep_episode({"max_gx": 3266, "episode_success": False},
                        CONFIG["keep_min_max_gx"]) is False
    # episode_success keeps a trajectory even below the gx mark — the
    # two predicates disagreed 13:1 on real data, so neither is allowed
    # to veto the other; both labels are stored either way.
    assert keep_episode({"max_gx": 100, "episode_success": True},
                        CONFIG["keep_min_max_gx"]) is True


# ---- step 1 fallback: the collection loop (stubbed pool) ----------------


SUCCESS_GX = 4000  # the fake reward's episode_success threshold


class _FakePool:
    """Scripted pool: episode scripts are keyed by the round-major episode
    index the collector assigns (round * lanes + lane), each script a
    (peak_gx, die_step) pair. gx ramps 100/step toward the peak; `done`
    fires at die_step. step_all returns eval_game-shaped 4-tuples with a
    2 KiB RAM whose gx bytes (0x006D page, 0x0086 low) drive the labels."""

    def __init__(self, lanes, scripts):
        self.lanes = lanes
        self.scripts = scripts
        self.round = -1
        self.t = None
        self.masks_seen: list[list[int]] = []

    def reset_all(self):
        self.round += 1
        self.t = [0] * self.lanes

    def load_worker_state(self, i, blob):
        assert 0 <= i < self.lanes

    def step_all(self, acts):
        self.masks_seen.append([int(x) for x in acts])
        out = []
        for i in range(self.lanes):
            self.t[i] += 1
            ep = self.round * self.lanes + i
            peak, die = self.scripts[ep] if ep < len(self.scripts) else (0, 10)
            gx = min(self.t[i] * 100, peak)
            ram = np.zeros(2048, dtype=np.uint8)
            ram[0x006D] = (gx >> 8) & 0xFF
            ram[0x0086] = gx & 0xFF
            out.append((None, None, ram, self.t[i] >= die))
        return out


class _FakePolicy:
    """Tile-policy-shaped wrapper: obs is the RAM's gx bytes (so recorded
    obs are checkable), logits favor `first_action` on the first control
    step and `later_action` afterwards."""

    def __init__(self, num_actions=3, first_action=1, later_action=2):
        self.num_actions = num_actions
        self.first = first_action
        self.later = later_action
        self.calls = 0

    def reset(self, step):
        self.calls = 0
        return step[2][:4].astype(np.int8)

    def push(self, step):
        return step[2][:4].astype(np.int8)

    def initial_hidden(self, device):
        return None

    def logits(self, obs, hidden, prev_action):
        idx = self.first if self.calls == 0 else self.later
        self.calls += 1
        row = torch.full((1, self.num_actions), -10.0)
        row[0, idx] = 10.0
        return row, hidden


class _FakeReward:
    def __init__(self):
        self.last_gx = 0

    def reset(self):
        self.last_gx = 0

    def compute(self, ram, action=0):
        self.last_gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
        return 0.0, False, {}

    def episode_success(self):
        return self.last_gx >= SUCCESS_GX


def _collect(scripts, *, lanes, n_episodes, sticky=0.0, jitter=0,
             action_select="greedy", policy_kwargs=None):
    pool = _FakePool(lanes, scripts)
    return pool, collect_trajectories(
        pool,
        n_episodes=n_episodes,
        lanes=lanes,
        max_steps=60,
        bitmasks=[10, 11, 12],
        make_policy=lambda: _FakePolicy(**(policy_kwargs or {})),
        make_reward_fn=_FakeReward,
        start_blob=b"blob",
        sticky_prob=sticky,
        start_jitter=jitter,
        eval_seed=5,
        action_select=action_select,
        temperature=1.0,
    )


def test_collection_records_indices_and_labels():
    # ep0 clears (peak 4000 -> success), ep1 dies early at gx 1000,
    # ep2 reaches 3300 without success.
    scripts = [(4000, 60), (1000, 10), (3300, 60)]
    _, eps = _collect(scripts, lanes=1, n_episodes=3)
    assert [e["episode_index"] for e in eps] == [0, 1, 2]
    assert eps[0]["episode_success"] is True
    assert eps[0]["max_gx"] >= SUCCESS_GX
    assert eps[1]["episode_success"] is False and eps[1]["max_gx"] == 1000
    assert eps[2]["max_gx"] == 3300 and eps[2]["episode_success"] is False
    kept = [e for e in eps if keep_episode(e, CONFIG["keep_min_max_gx"])]
    assert [e["episode_index"] for e in kept] == [0, 2]
    for e in eps:
        assert e["obs"].shape[0] == e["actions"].shape[0] == e["length"]
        assert e["actions"].dtype == np.int64


def test_collection_feeds_masks_but_records_action_indices():
    # Greedy fake policy always picks index 1 first then 2: the pool must
    # see the BITMASKS (11 then 12), the trajectory the INDICES (1 then 2)
    # — the exact confusion a bitmask/index swap would produce.
    pool, eps = _collect([(1000, 4)], lanes=1, n_episodes=1)
    assert eps[0]["actions"].tolist()[:2] == [1, 2]
    control_masks = [m[0] for m in pool.masks_seen[1:]]  # [0] is the flush
    assert control_masks[:2] == [11, 12]


def test_collection_sticky_repeats_previous_executed_action():
    # sticky 1.0 with the eval_game guard (no draw on control step 0):
    # first action is the policy's (index 1), every later action repeats
    # it regardless of the policy preferring index 2.
    _, eps = _collect([(1000, 6)], lanes=1, n_episodes=1, sticky=1.0)
    acts = eps[0]["actions"].tolist()
    assert acts[0] == 1
    assert set(acts[1:]) == {1}


def test_collection_invariant_to_lane_count():
    # Per-episode RNG derivation (eval_game.episode_rng_seeds) makes the
    # collected episodes a pure function of (eval_seed, episode index),
    # so 1-lane and 2-lane collection must produce identical trajectories.
    scripts = [(4000, 60), (1000, 12), (3300, 60), (2000, 20)]
    _, serial = _collect(scripts, lanes=1, n_episodes=4, sticky=0.25,
                         jitter=5)
    _, wide = _collect(scripts, lanes=2, n_episodes=4, sticky=0.25,
                       jitter=5)
    assert len(serial) == len(wide) == 4
    for s, w in zip(serial, wide):
        assert s["episode_index"] == w["episode_index"]
        assert s["max_gx"] == w["max_gx"]
        np.testing.assert_array_equal(s["actions"], w["actions"])
        np.testing.assert_array_equal(s["obs"], w["obs"])


# ---- step 2: the BC lock-in cure ----------------------------------------


def _bc_fixture(n=300, feature_dim=12, num_actions=4, seed=3):
    sd = _tile_state_dict(num_actions=num_actions, feature_dim=feature_dim)
    payload = {"iter": 910, "net_state_dict": sd}
    rng = np.random.default_rng(seed)
    obs = rng.integers(-4, 4, size=(n, feature_dim)).astype(np.int8)
    act = rng.integers(0, num_actions, size=(n,)).astype(np.int64)
    return payload, sd, obs, act


def test_bc_cure_moves_actor_and_freezes_everything_else():
    payload, sd, obs, act = _bc_fixture()
    cured, stats = bc_cure(payload, obs, act, epochs=2, lr=1e-2,
                           batch_size=64, seed=0, scope="actor")
    assert not torch.equal(cured["actor.weight"], sd["actor.weight"])
    for k in sd:
        if k.startswith("actor."):
            continue
        # Frozen means BIT-IDENTICAL, not merely close.
        assert torch.equal(cured[k], sd[k]), k


def test_bc_cure_trunk_variant_still_freezes_critic():
    payload, sd, obs, act = _bc_fixture()
    cured, _ = bc_cure(payload, obs, act, epochs=2, lr=1e-2,
                       batch_size=64, seed=0, scope="actor+trunk")
    assert not torch.equal(cured["actor.weight"], sd["actor.weight"])
    assert not torch.equal(cured["fc1.weight"], sd["fc1.weight"])
    assert torch.equal(cured["critic.weight"], sd["critic.weight"])
    assert torch.equal(cured["critic.bias"], sd["critic.bias"])


def test_bc_cure_is_deterministic():
    payload, _, obs, act = _bc_fixture()
    a, _ = bc_cure(payload, obs, act, epochs=2, lr=1e-2, batch_size=64,
                   seed=11, scope="actor")
    b, _ = bc_cure(payload, obs, act, epochs=2, lr=1e-2, batch_size=64,
                   seed=11, scope="actor")
    assert torch.equal(a["actor.weight"], b["actor.weight"])
    assert torch.equal(a["actor.bias"], b["actor.bias"])


def test_bc_cure_reduces_loss_on_learnable_labels():
    payload, _, obs, _ = _bc_fixture()
    act = np.zeros(obs.shape[0], dtype=np.int64)  # constant label
    _, stats = bc_cure(payload, obs, act, epochs=5, lr=5e-2,
                       batch_size=64, seed=0, scope="actor")
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["n_pairs"] == obs.shape[0]


def test_linear_decay_reaches_zero():
    assert linear_decay_factor(0, 10) == 1.0
    assert linear_decay_factor(5, 10) == 0.5
    assert linear_decay_factor(10, 10) == 0.0
    assert linear_decay_factor(3, 0) == 0.0  # degenerate: no steps


def test_trainable_param_names_scopes():
    names = ["fc1.weight", "fc1.bias", "norm1.weight", "norm1.bias",
             "fc2.weight", "fc2.bias", "norm2.weight", "norm2.bias",
             "actor.weight", "actor.bias", "critic.weight", "critic.bias"]
    assert trainable_param_names(names, "actor") == {
        "actor.weight", "actor.bias"}
    trunk = trainable_param_names(names, "actor+trunk")
    assert "fc1.weight" in trunk and "norm2.bias" in trunk
    assert "critic.weight" not in trunk and "critic.bias" not in trunk
    with pytest.raises(ValueError):
        trainable_param_names(names, "everything")


def test_cured_checkpoint_standard_format(tmp_path):
    _, sd, _, _ = _bc_fixture()
    out = tmp_path / "cured.pt"
    write_cured_checkpoint(out, sd, stats={"final_loss": 0.1},
                           provenance={"scope": "actor"})
    payload = torch.load(str(out), map_location="cpu", weights_only=False)
    # Standard vanilla_ppo_iter payload: the trainer's tear-tolerant
    # resume needs `net_state_dict` + `iter`; the iter is the consol2
    # continuation iter + 1 so the cured net sorts NEWEST in the seeded
    # dir and the resume picks it over the pinned source.
    assert payload["iter"] == CURED_SEED_ITER
    assert set(payload["net_state_dict"]) == set(sd)
    for v in payload["net_state_dict"].values():
        assert v.device.type == "cpu"
    assert payload["provenance"] == "night2_bc_cure"


# ---- step 3: the pre-registered gate ------------------------------------


def _summ(*, status="ok", n=50, median=2200.0):
    return {"status": status, "n_episodes": n, "median_max_x": median,
            "clear_rate": 0.1}


def test_gate_passes_above_both_baselines():
    v = gate_verdict(_summ(median=2200.0), _summ(n=10, median=3100.0))
    assert v["passed"] is True
    assert v["reasons"] == []


def test_gate_fails_at_exact_honest_baseline():
    # Pre-registered as strictly GREATER THAN 2059 — equal is a fail.
    v = gate_verdict(_summ(median=2059.0), _summ(n=10, median=3100.0))
    assert v["passed"] is False
    assert any("2059" in r for r in v["reasons"])


def test_gate_fails_below_det_rung_baseline():
    v = gate_verdict(_summ(median=2500.0), _summ(n=10, median=2979.0))
    assert v["passed"] is False
    assert any("2979" in r for r in v["reasons"])


def test_gate_failed_leg_never_passes():
    # A broken probe harness must never read as a pass — median 0.0 from
    # a probe_failed leg is not evidence about the policy.
    v = gate_verdict(_summ(status="probe_failed", median=9999.0),
                     _summ(n=10, median=3100.0))
    assert v["passed"] is False
    v2 = gate_verdict(_summ(median=2200.0),
                      _summ(status="probe_timeout", n=10, median=9999.0))
    assert v2["passed"] is False


def test_gate_empty_probe_never_passes():
    v = gate_verdict(_summ(n=0, median=9999.0), _summ(n=10, median=3100.0))
    assert v["passed"] is False


# ---- eval command assembly ----------------------------------------------


def test_honest_probe_command_matches_registered_protocol():
    cmd = build_honest_probe_command(
        checkpoint=ROOT / CONFIG["source_checkpoint"])
    joined = " ".join(cmd)
    assert "--episodes 50" in joined
    assert "--sticky-prob 0.25" in joined
    assert "--start-jitter 16" in joined
    # Cold ARGMAX: greedy is eval_game's default; the command must not
    # opt into sampling.
    assert "sampled" not in joined
    assert "--eval-workers 5" in joined and "per-episode" in joined
    assert "--sequential" in cmd and "--level-clear" in cmd
    # Cold from the 1-2 entrance (the consol2 profile's start state).
    assert "stage_03.state" in joined
    for a in cmd:
        if "/" in a and not a.startswith("-"):
            assert Path(a).exists(), a


def test_det_probe_command_is_zero_noise_from_the_rung():
    cmd = build_det_probe_command(
        checkpoint=ROOT / CONFIG["source_checkpoint"])
    joined = " ".join(cmd)
    assert "--episodes 10" in joined
    assert "--sticky-prob 0.0" in joined
    assert "--start-jitter 0" in joined
    assert "sampled" not in joined
    assert "--eval-workers 1" in joined and "shared-stream" in joined
    assert "s_000559.state" in joined
    for a in cmd:
        if "/" in a and not a.startswith("-"):
            assert Path(a).exists(), a


def test_collection_reference_command_is_sampled_house_noise():
    cmd = build_collection_reference_command(
        checkpoint=ROOT / CONFIG["source_checkpoint"])
    joined = " ".join(cmd)
    assert "--action-select sampled" in joined
    assert "--episodes 600" in joined
    assert "--sticky-prob 0.25" in joined and "--start-jitter 16" in joined
    assert "per-episode" in joined
    for a in cmd:
        if "/" in a and not a.startswith("-"):
            assert Path(a).exists(), a


def test_collection_plan_carries_the_predicates():
    plan = collection_plan()
    assert plan["mode"] == "collection"
    assert plan["action_select"] == "sampled"
    assert plan["episodes"] == 600
    assert plan["sticky_prob"] == 0.25 and plan["start_jitter"] == 16
    assert plan["keep_min_max_gx"] == 3267
    assert set(plan["labels"]) == {"max_gx", "episode_success"}
    # eval_game was checked for a trace-saving flag and has none — the
    # plan must say where the collection loop actually lives.
    assert plan["trace_flag"] is None


# ---- consol2 seeding ----------------------------------------------------


def _write_cured(tmp_path):
    _, sd, _, _ = _bc_fixture()
    cured = tmp_path / "cured.pt"
    write_cured_checkpoint(cured, sd, stats={}, provenance={})
    return cured


def test_seed_writes_cured_at_source_iter_plus_one(tmp_path):
    cured = _write_cured(tmp_path)
    ckpt_dir = tmp_path / "consol2_ckpts"
    note = seed_cured_for_consol2(cured, ckpt_dir=ckpt_dir,
                                  source_iter=910)
    dst = ckpt_dir / "vanilla_ppo_iter_00911.pt"
    assert dst.exists()
    assert dst.read_bytes() == cured.read_bytes()
    assert "00911" in note


def test_seed_is_idempotent_on_identical_dst(tmp_path):
    cured = _write_cured(tmp_path)
    ckpt_dir = tmp_path / "consol2_ckpts"
    seed_cured_for_consol2(cured, ckpt_dir=ckpt_dir, source_iter=910)
    note = seed_cured_for_consol2(cured, ckpt_dir=ckpt_dir,
                                  source_iter=910)
    assert "already" in note


def test_seed_refuses_conflicting_existing_dst(tmp_path):
    cured = _write_cured(tmp_path)
    ckpt_dir = tmp_path / "consol2_ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "vanilla_ppo_iter_00911.pt").write_bytes(b"other lineage")
    with pytest.raises(RuntimeError):
        seed_cured_for_consol2(cured, ckpt_dir=ckpt_dir, source_iter=910)


def test_seed_refuses_when_dir_already_progressed_past(tmp_path):
    # A consol2 run that already produced iter 920 must not be undercut:
    # the resume would ignore the cured net silently.
    cured = _write_cured(tmp_path)
    ckpt_dir = tmp_path / "consol2_ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "vanilla_ppo_iter_00920.pt").write_bytes(b"newer")
    with pytest.raises(RuntimeError):
        seed_cured_for_consol2(cured, ckpt_dir=ckpt_dir, source_iter=910)


def test_seed_plan_only_writes_nothing(tmp_path):
    cured = _write_cured(tmp_path)
    ckpt_dir = tmp_path / "consol2_ckpts"
    note = seed_cured_for_consol2(cured, ckpt_dir=ckpt_dir,
                                  source_iter=910, plan_only=True)
    assert not ckpt_dir.exists() or not any(ckpt_dir.iterdir())
    assert "will seed" in note


# ---- config / manifest / resumability -----------------------------------


def test_config_pins_match_consol2():
    """Night-2 continues the consol2 lineage: same pinned source
    checkpoint, and the cured net seeds at the consol2 continuation
    iter + 1 so run_consol2's own seeding sees an already-seeded dir and
    the trainer's resume picks the cured (newest) checkpoint."""
    assert CONFIG["source_checkpoint"] == CONSOL2_CONFIG["source_checkpoint"]
    assert (CONFIG["source_checkpoint_sha256"]
            == CONSOL2_CONFIG["source_checkpoint_sha256"])
    assert CURED_SEED_ITER == CONSOL2_CONFIG["source_iter"] + 1


def test_preregistered_gate_and_bc_constants():
    assert CONFIG["gate_honest_episodes"] == 50
    assert CONFIG["gate_honest_sticky"] == 0.25
    assert CONFIG["gate_honest_jitter"] == 16
    assert CONFIG["gate_honest_median_baseline"] == 2059.0
    assert CONFIG["gate_det_episodes"] == 10
    assert CONFIG["gate_det_median_baseline"] == 2979.0
    assert "s_000559.state" in CONFIG["gate_det_rung_state"]
    assert CONFIG["bc_epochs"] == 15
    assert CONFIG["bc_lr"] == 1e-4
    assert CONFIG["bc_batch_size"] == 256
    assert CONFIG["bc_train_scope"] == "actor"
    assert CONFIG["collect_episodes"] == 600
    assert CONFIG["keep_min_max_gx"] == 3267


def test_manifest_mirrors_config():
    man = build_manifest()
    assert man["config"] == CONFIG
    assert man["campaign"] == "smb_1_2_night2"
    gate = man["gate"]
    assert gate["honest_median_must_exceed"] == 2059.0
    assert gate["det_rung_median_must_exceed"] == 2979.0
    assert len(man["steps"]) == 4


def test_steps_from_skip_to():
    assert steps_from(1) == [1, 2, 3, 4]
    assert steps_from(3) == [3, 4]
    assert steps_from(4) == [4]
    with pytest.raises(ValueError):
        steps_from(0)
    with pytest.raises(ValueError):
        steps_from(5)


# ---- dry-run (live; checkpoint load + command assembly, no rollouts) ----


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_dry_run_passes_live():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "night2_runner.py"),
         "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=540)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout
    assert "FAIL" not in proc.stdout
    # The key step-1 finding must be REPORTED, not silently branched on.
    assert "SIL" in proc.stdout
