"""Mechanism 2 — KL-anchored warm start + critic warmup.

`reinforce.kl_anchor_checkpoint` loads a solved-level policy's ACTOR weights
(trunk + actor head) into the live net at startup, keeps a frozen copy as the
prior, and leaves the critic at its fresh orthogonal init. While global
env-steps < `actor_freeze_steps` only the critic updates (actor grads are
dropped). After unfreeze, beta * KL(prior || pi) is subtracted per-step from
the reward stream, with beta linearly decayed `kl_beta_start` ->
`kl_beta_end` over `kl_beta_decay_steps`. D_KL is exposed in metrics every
iteration (the campaign kill criterion reads it).

Absent config = untouched behavior (pinned by the C0 master golden).
"""
from __future__ import annotations

import queue as _queue
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from src.models.tile_policy import TilePolicyNetwork
from src.training.kl_anchor import KLAnchor

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"

_NA, _FDIM, _HID, _TRUNK = 5, 12, 16, 8


def _make_anchor_ckpt(tmp: Path, seed: int = 3) -> tuple[Path, TilePolicyNetwork]:
    torch.manual_seed(seed)
    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID, trunk_dim=_TRUNK
    )
    path = tmp / "anchor.pt"
    torch.save({"net_state_dict": net.state_dict()}, str(path))
    return path, net


def _build_anchor(path: Path) -> KLAnchor:
    return KLAnchor(
        checkpoint_path=str(path),
        beta_start=0.05,
        beta_end=0.01,
        beta_decay_steps=50e6,
        actor_freeze_steps=5e6,
        num_actions=_NA,
        feature_dim=_FDIM,
        device=torch.device("cpu"),
    )


def test_load_actor_into_copies_actor_and_keeps_critic_fresh(tmp_path) -> None:
    path, src_net = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    torch.manual_seed(99)  # live net gets a DIFFERENT fresh init
    live = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID, trunk_dim=_TRUNK
    )
    critic_before = {
        k: v.clone() for k, v in live.state_dict().items()
        if k.startswith("critic")
    }
    anchor.load_actor_into(live)
    src_sd = src_net.state_dict()
    live_sd = live.state_dict()
    for k, v in src_sd.items():
        if k.startswith("critic"):
            # Critic stays at the live net's own fresh orthogonal init.
            assert torch.equal(live_sd[k], critic_before[k]), k
        else:
            assert torch.equal(live_sd[k], v), f"actor weight {k} not loaded"


def test_load_actor_shape_mismatch_raises(tmp_path) -> None:
    path, _ = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    wrong = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID * 2,
        trunk_dim=_TRUNK,
    )
    with pytest.raises(ValueError):
        anchor.load_actor_into(wrong)


def test_prior_is_shape_inferred_and_frozen(tmp_path) -> None:
    path, src_net = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)  # non-default widths: must shape-infer
    prior_sd = anchor.prior.state_dict()
    for k, v in src_net.state_dict().items():
        assert torch.equal(prior_sd[k], v), f"prior weight {k} differs"
    for p in anchor.prior.parameters():
        assert not p.requires_grad, "prior must be frozen"


def test_beta_schedule_and_freeze_flag(tmp_path) -> None:
    path, _ = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    anchor.set_env_steps(0)
    assert anchor.beta == pytest.approx(0.05)
    assert anchor.frozen is True
    anchor.set_env_steps(2.5e6)
    assert anchor.frozen is True  # still below 5e6
    anchor.set_env_steps(5e6)
    assert anchor.frozen is False
    anchor.set_env_steps(25e6)  # halfway through the decay
    assert anchor.beta == pytest.approx(0.03)
    anchor.set_env_steps(50e6)
    assert anchor.beta == pytest.approx(0.01)
    anchor.set_env_steps(120e6)  # clamped past the end
    assert anchor.beta == pytest.approx(0.01)


def test_kl_divergence_zero_against_self_positive_against_other(tmp_path) -> None:
    path, src_net = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    obs = torch.randn(6, _FDIM)
    with torch.no_grad():
        logits, _ = src_net.forward_ac(obs)
    logp_same = F.log_softmax(logits.float(), dim=-1)
    kl_same = anchor.kl_divergence(obs, logp_same)
    assert kl_same.shape == (6,)
    np.testing.assert_allclose(kl_same, 0.0, atol=1e-6)

    torch.manual_seed(1234)
    other = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID, trunk_dim=_TRUNK
    )
    with torch.no_grad():
        logits_o, _ = other.forward_ac(obs)
    kl_other = anchor.kl_divergence(obs, F.log_softmax(logits_o.float(), -1))
    assert (kl_other > 0.0).all(), "KL(prior || different policy) must be > 0"


def test_freeze_drops_actor_grads_critic_still_moves(tmp_path) -> None:
    path, _ = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    anchor.set_env_steps(0)  # frozen phase
    assert anchor.frozen
    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID, trunk_dim=_TRUNK
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    actor_before = {
        k: v.clone() for k, v in net.state_dict().items()
        if not k.startswith("critic")
    }
    critic_before = {
        k: v.clone() for k, v in net.state_dict().items()
        if k.startswith("critic")
    }
    logits, value = net.forward_ac(torch.randn(8, _FDIM))
    loss = logits.pow(2).mean() + value.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    anchor.zero_actor_grads(net)
    opt.step()
    sd = net.state_dict()
    for k, v in actor_before.items():
        assert torch.equal(sd[k], v), f"actor weight {k} moved during freeze"
    critic_moved = any(
        not torch.equal(sd[k], v) for k, v in critic_before.items()
    )
    assert critic_moved, "critic did not move — the freeze zeroed too much"


def test_config_schema_registers_kl_keys() -> None:
    from src.training.config_schema import KNOWN_REINFORCE_KEYS, validate_profile
    for key in ("kl_anchor_checkpoint", "kl_beta_start", "kl_beta_end",
                "kl_beta_decay_steps", "actor_freeze_steps"):
        assert key in KNOWN_REINFORCE_KEYS, key
    prof = {"name": "x", "reinforce": {
        "kl_anchor_checkpoint": "runs/x.pt", "kl_beta_start": 0.05,
        "kl_beta_end": 0.01, "kl_beta_decay_steps": 50e6,
        "actor_freeze_steps": 5e6,
    }}
    assert validate_profile(prof) == []


def test_trainer_applies_penalty_with_negative_sign() -> None:
    """The per-step reward stream SUBTRACTS beta*KL (source anchor; the
    end-to-end freeze path is exercised by the integration test below)."""
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    assert "kl_anchor_checkpoint" in src
    assert "reward -= float(_kl_pen_step[i])" in src


# ---------------------------------------------------------------------------
# Integration: the real 2-iteration tile+CPU loop with the anchor configured.
# Mirrors the C0/C-prmdp harness (same profile, same determinism basis).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_freeze_phase_keeps_actor_bit_identical_in_real_loop() -> None:
    from src.training.trainer import Trainer

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = 32
    profile["reinforce"]["ppo_minibatch_size"] = 16
    profile["reinforce"]["device"] = "cpu"

    with tempfile.TemporaryDirectory(prefix="kl_anchor_it_") as tmp:
        tmpd = Path(tmp)
        # Anchor checkpoint at the PROFILE's architecture (700-dim stacked
        # tile features, 6 actions, default 64/32 widths).
        torch.manual_seed(77)
        anchor_net = TilePolicyNetwork(num_actions=6, feature_dim=700)
        anchor_path = tmpd / "anchor.pt"
        torch.save({"net_state_dict": anchor_net.state_dict()}, str(anchor_path))

        profile["reinforce"]["kl_anchor_checkpoint"] = str(anchor_path)
        profile["reinforce"]["actor_freeze_steps"] = 1e9  # frozen for the whole run

        metrics_q: _queue.Queue = _queue.Queue()
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

        live_sd = trainer._ppo_net.state_dict()
        anchor_sd = anchor_net.state_dict()
        for k, v in anchor_sd.items():
            if k.startswith("critic"):
                continue
            assert torch.equal(live_sd[k].cpu(), v), (
                f"actor weight {k} drifted during the freeze phase — "
                f"the critic-only warmup leaked gradient into the actor"
            )
        # The prior itself must still be the checkpoint, bit-for-bit.
        prior_sd = trainer._kl_anchor.prior.state_dict()
        for k, v in anchor_sd.items():
            assert torch.equal(prior_sd[k].cpu(), v), f"prior {k} drifted"

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    assert ppo_rows, "no PPO metric rows emitted"
    for r in ppo_rows:
        assert "kl_anchor_div" in r, "D_KL not exposed in metrics"
        assert "kl_anchor_beta" in r
        assert r["kl_anchor_actor_frozen"] == 1
        # Actor pinned at the prior => KL(prior || pi) is identically 0.
        assert abs(r["kl_anchor_div"]) < 1e-5
