"""Mechanism 4 — kernel-matched binary adversary (`reinforce.adversary`).

Refit of the PR-MDP machinery to the honest-eval noise kernel: the adversary
head outputs 2 actions {pass, repeat-previous-executed}; on 'repeat' the
previous EXECUTED action replaces the protagonist's. Adversary reward =
-protagonist_reward - budget_penalty * I(repeat), trained with negated GAE on
its own decision steps (every executed step — pass IS a decision), epochs
matched to the protagonist's (10:10, not 10:2), entropy coef 0.01. Adversary
entropy (vs ln 2) and the realized repeat-fraction are logged.

Default-off (no `adversary` block) keeps PR-MDP legacy behavior — pinned by
test_char_prmdp.py's golden.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.tile_policy import TilePolicyNetwork
from src.training.kernel_adversary import (
    KernelStickyAdversary,
    adversary_rewards,
)

ROOT = Path(__file__).resolve().parent.parent

_NA_PRO, _FDIM = 5, 8  # protagonist action count / tile feature dim
_T, _N = 4, 3


def _make_adv(*, epochs=10, budget=0.1, ent=0.01):
    torch.manual_seed(0)
    net = TilePolicyNetwork(
        num_actions=2, feature_dim=_FDIM, hidden_dim=8, trunk_dim=4
    )
    return KernelStickyAdversary(
        net=net,
        num_envs=_N,
        rollout_steps=_T,
        budget_penalty=budget,
        entropy_coef=ent,
        epochs=epochs,
        clip=0.1,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        value_loss_kind="huber",
        grad_clip=0.5,
        device=torch.device("cpu"),
        preprocess_f16=False,
        is_tile_mode=True,
    )


def _rig(net, always_repeat: bool) -> None:
    """Pin the adversary's actor head to a deterministic decision."""
    with torch.no_grad():
        net.actor.weight.zero_()
        bias = [-50.0, 50.0] if always_repeat else [50.0, -50.0]
        net.actor.bias.copy_(torch.tensor(bias))


def _step_inputs(seed=11):
    torch.manual_seed(seed)
    batch_t = torch.randn(_N, _FDIM)
    logits = torch.randn(_N, _NA_PRO)
    log_probs_all = F.log_softmax(logits, dim=-1)
    actions = torch.tensor([1, 2, 3], dtype=torch.long)
    log_probs_taken = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    prev_exec = np.array([4, 0, 2], dtype=np.int64)
    return batch_t, actions, log_probs_all, log_probs_taken, prev_exec


def test_repeat_overrides_with_previous_executed_action() -> None:
    adv = _make_adv()
    _rig(adv.net, always_repeat=True)
    adv.begin_iter()
    batch_t, actions, lp_all, lp_taken, prev_exec = _step_inputs()
    prev_before = prev_exec.copy()
    with torch.no_grad():
        adv.decide(0, batch_t, actions, lp_all, lp_taken, prev_exec)
    # The previous EXECUTED action replaced the protagonist's everywhere.
    np.testing.assert_array_equal(actions.numpy(), prev_before)
    # The recorded log-prob follows the EXECUTED action (clamped house rule).
    for i in range(_N):
        want = float(
            torch.clamp(lp_all[i, prev_before[i]], min=-13.0)
        )
        assert lp_taken[i].item() == pytest.approx(want)
    # Executed-action tracking: repeating leaves prev_exec unchanged.
    np.testing.assert_array_equal(prev_exec, prev_before)
    assert adv.repeat_buf[0].all()
    assert adv.action_buf[0].tolist() == [KernelStickyAdversary.REPEAT] * _N


def test_pass_leaves_protagonist_actions_untouched() -> None:
    adv = _make_adv()
    _rig(adv.net, always_repeat=False)
    adv.begin_iter()
    batch_t, actions, lp_all, lp_taken, prev_exec = _step_inputs()
    actions_before = actions.clone()
    lp_before = lp_taken.clone()
    with torch.no_grad():
        adv.decide(0, batch_t, actions, lp_all, lp_taken, prev_exec)
    assert torch.equal(actions, actions_before)
    assert torch.equal(lp_taken, lp_before)
    assert not adv.repeat_buf[0].any()
    # Executed-action tracking advances to the fresh protagonist actions.
    np.testing.assert_array_equal(prev_exec, actions_before.numpy())


def test_adversary_reward_sign_and_budget_penalty() -> None:
    reward = np.array([[2.0, -1.0]], dtype=np.float32)
    repeat = np.array([[True, False]])
    out = adversary_rewards(reward, repeat, 0.1)
    # Zero-sum core: negated protagonist reward; repeats pay the budget.
    assert out[0, 0] == pytest.approx(-2.0 - 0.1)
    assert out[0, 1] == pytest.approx(1.0)
    assert out.dtype == np.float32


def test_update_trains_ten_epochs_and_reports_entropy() -> None:
    adv = _make_adv(epochs=10)
    adv.begin_iter()
    rng = np.random.default_rng(5)
    prev_exec = np.zeros(_N, dtype=np.int64)
    with torch.no_grad():
        for t in range(_T):
            batch_t = torch.randn(_N, _FDIM)
            logits = torch.randn(_N, _NA_PRO)
            lp_all = F.log_softmax(logits, dim=-1)
            actions = torch.from_numpy(
                rng.integers(0, _NA_PRO, size=_N)
            ).long()
            lp_taken = lp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            adv.decide(t, batch_t, actions, lp_all, lp_taken, prev_exec)
        adv.drain_values()
        adv.compute_final_values(torch.randn(_N, _FDIM))

    reward_buf = rng.normal(size=(_T, _N)).astype(np.float32)
    done_buf = np.zeros((_T, _N), dtype=np.bool_)
    valid_buf = np.ones((_T, _N), dtype=np.bool_)
    obs_flat = rng.integers(-4, 4, size=(_T * _N, _FDIM)).astype(np.int8)
    obs_all = torch.from_numpy(obs_flat).float()

    before = {k: v.clone() for k, v in adv.net.state_dict().items()}
    stats = adv.update(
        reward_buf=reward_buf, done_buf=done_buf, valid_buf=valid_buf,
        obs_all=obs_all, obs_flat=obs_flat, mb_size=_T * _N,
    )
    after = adv.net.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), (
        "adversary net did not train"
    )
    # 10 epochs x 1 full-batch minibatch each -> exactly 10 Adam steps.
    steps = [int(s["step"]) for s in adv.opt.state.values() if "step" in s]
    assert steps and min(steps) == 10 and max(steps) == 10
    # Entropy is logged against the binary ceiling ln 2.
    assert 0.0 <= stats["adversary_entropy"] <= math.log(2.0) + 1e-4
    assert 0.0 <= stats["adversary_repeat_frac"] <= 1.0
    assert stats["adversary_policy_loss"] == stats["adversary_policy_loss"]


def test_realized_repeat_fraction_counts_valid_steps_only() -> None:
    adv = _make_adv()
    adv.begin_iter()
    adv.repeat_buf[:] = False
    adv.repeat_buf[0, :] = True  # 3 repeats out of 12 slots
    valid = np.ones((_T, _N), dtype=np.bool_)
    valid[1:, 0] = False  # 3 slots invalid, none of them repeats
    assert adv.repeat_fraction(valid) == pytest.approx(3.0 / 9.0)


# ---------------------------------------------------------------------------
# Trainer wiring + schema (legacy PR-MDP defaults pinned by test_char_prmdp)
# ---------------------------------------------------------------------------

def test_trainer_wires_kernel_adversary() -> None:
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    assert "KernelStickyAdversary" in src
    assert 'kernel_sticky' in src
    # Legacy prmdp and the kernel adversary are mutually exclusive.
    assert "mutually exclusive" in src.lower() or "both" in src.lower()


def test_config_schema_adversary_block() -> None:
    from src.training.config_schema import (
        KNOWN_REINFORCE_KEYS, validate_profile,
    )
    assert "adversary" in KNOWN_REINFORCE_KEYS
    clean = {"name": "x", "reinforce": {"adversary": {
        "mode": "kernel_sticky", "budget_penalty": 0.1,
        "entropy_coef": 0.01, "epochs": 10,
    }}}
    assert validate_profile(clean) == []
    typo = {"name": "x", "reinforce": {"adversary": {
        "mode": "kernel_sticky", "budget_pnealty": 0.1,
    }}}
    assert any("budget_pnealty" in w for w in validate_profile(typo))
    bad_mode = {"name": "x", "reinforce": {"adversary": {"mode": "prmdp"}}}
    assert any("mode" in w for w in validate_profile(bad_mode))


# ---------------------------------------------------------------------------
# Integration: the real tile+CPU loop with the kernel adversary on (C0
# harness geometry). A fresh 2-action head is near-uniform, so the realized
# repeat fraction sits near 0.5 and entropy near ln 2 on the first iters.
# ---------------------------------------------------------------------------

_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_kernel_adversary_runs_and_logs_in_real_loop() -> None:
    import queue as _queue
    import random
    import tempfile

    import yaml

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = 32
    profile["reinforce"]["ppo_minibatch_size"] = 16
    profile["reinforce"]["device"] = "cpu"
    profile["reinforce"]["adversary"] = {
        "mode": "kernel_sticky", "budget_penalty": 0.1,
        "entropy_coef": 0.01, "epochs": 3,
    }

    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="kernel_adv_it_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=2,
            population_size=2,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
            device_override="cpu",
            seed=1234,
        )
        trainer.run(num_generations=2, resume_from=None, fresh_start=True)

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    assert ppo_rows, "no PPO metric rows emitted"
    for r in ppo_rows:
        assert 0.0 <= r["adversary_entropy"] <= math.log(2.0) + 1e-4
        # Fresh near-uniform binary head: repeats land well inside (0, 1).
        assert 0.05 <= r["adversary_repeat_frac"] <= 0.95
        assert "adversary_policy_loss" in r


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_kernel_adversary_and_legacy_prmdp_are_mutually_exclusive() -> None:
    import queue as _queue
    import tempfile

    import yaml

    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = 32
    profile["reinforce"]["ppo_minibatch_size"] = 16
    profile["reinforce"]["device"] = "cpu"
    profile["reinforce"]["adversary"] = {"mode": "kernel_sticky"}
    profile["reinforce"]["prmdp"] = {"enabled": True}

    from src.training.trainer import Trainer

    with tempfile.TemporaryDirectory(prefix="kernel_adv_x_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=2,
            population_size=2,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=_queue.Queue(),
            device_override="cpu",
            seed=1234,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            trainer.run(num_generations=1, resume_from=None, fresh_start=True)
