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
# Loss-level tether (reinforce.kl_anchor_loss_coef): each PPO minibatch adds
# coef * mean KL(prior(.|s) || pi_theta(.|s)) on its own states directly to
# the update loss — the per-update pull the reward-level beta penalty cannot
# provide once the ratio clips. Exercised against the REAL PPOUpdater on a
# synthetic rollout (no ROM, no pool; mirrors test_char_ppo_update's bare
# trainer shell).
# ---------------------------------------------------------------------------

_ROLL_T, _ROLL_E = 8, 4


class _NullTimer:
    def add(self, name: str, ns: int) -> None:
        pass


def _tether_trainer(anchor, loss_coef: float):
    """A Trainer shell carrying only the attributes PPOUpdater.update reads
    on the tile feedforward path. `anchor=None` leaves the KL attributes
    entirely unset — the key-absent shape of the config."""
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.device = torch.device("cpu")
    t._gen_timer = _NullTimer()
    t._demo_bank = None
    t._rnd = None
    t._is_tile_mode = True
    t._recurrent = False
    t.preprocess_f16 = False
    t.reinforce_gamma = 0.99
    t.gae_lambda = 0.95
    t.reinforce_steps = 1          # one epoch ...
    t.ppo_minibatch_size = 64      # ... one minibatch => ONE optimizer step
    t.ppo_clip_eps = 0.2
    t.value_coef = 0.5
    t.entropy_coef = 0.0
    t.value_loss_kind = "huber"
    t.reinforce_grad_clip = 10.0
    t.rnd_intrinsic_coef = 0.0
    t.rnd_loss_coef = 0.0
    t.rnd_predictor_update_fraction = 1.0
    t._gx_count_beta = 0.0
    if anchor is not None:
        t._kl_anchor = anchor
        t._kl_anchor_loss_coef = float(loss_coef)
    return t


def _synthetic_rollout(net) -> dict:
    """On-policy-shaped synthetic rollout buffers for the tile updater."""
    rng = np.random.default_rng(5)
    obs = rng.standard_normal((_ROLL_T, _ROLL_E, _FDIM)).astype(np.float32)
    actions = rng.integers(0, _NA, size=(_ROLL_T, _ROLL_E))
    with torch.no_grad():
        logits, values = net.forward_ac(
            torch.from_numpy(obs.reshape(-1, _FDIM))
        )
        logp = F.log_softmax(logits.float(), dim=-1)
    act_flat = torch.from_numpy(actions.reshape(-1).astype(np.int64))
    log_prob = (
        logp.gather(1, act_flat.unsqueeze(1)).squeeze(1)
        .numpy().reshape(_ROLL_T, _ROLL_E).astype(np.float32)
    )
    return {
        "obs_buf": obs,
        "action_buf": actions.astype(np.int64),
        "reward_buf": rng.standard_normal(
            (_ROLL_T, _ROLL_E)).astype(np.float32),
        "value_buf": values.float().numpy().reshape(
            _ROLL_T, _ROLL_E).astype(np.float32),
        "log_prob_buf": log_prob,
        "done_buf": np.zeros((_ROLL_T, _ROLL_E), dtype=bool),
        "valid_buf": np.ones((_ROLL_T, _ROLL_E), dtype=bool),
        "bonus_buf": np.zeros((_ROLL_T, _ROLL_E), dtype=np.float32),
        "final_values_np": np.zeros(_ROLL_E, dtype=np.float32),
    }


def _run_tether_update(anchor, loss_coef: float, *, seed: int = 11):
    """One seeded PPOUpdater.update (single optimizer step) on the same
    synthetic rollout; returns the post-update net + the updater's dict."""
    from src.training.ppo_updater import PPOUpdater

    torch.manual_seed(seed)
    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID, trunk_dim=_TRUNK
    )
    t = _tether_trainer(anchor, loss_coef)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    roll = _synthetic_rollout(net)
    np.random.seed(seed)  # identical minibatch permutation across variants
    out = PPOUpdater(t).update(
        net=net, optimizer=optimizer, rollout_steps=_ROLL_T,
        num_envs=_ROLL_E, obs_shape=(_FDIM,), global_it=0, sam_rho=0.0,
        **roll,
    )
    return net, out


def _make_sharp_anchor_ckpt(tmp: Path) -> Path:
    """An anchor whose actor head is decisively NON-uniform. Two fresh
    orthogonal inits are both near-uniform (KL ~1e-4 — pure noise), so the
    pull test spreads the prior's action logits to get a gradient worth
    measuring."""
    torch.manual_seed(3)
    net = TilePolicyNetwork(
        num_actions=_NA, feature_dim=_FDIM, hidden_dim=_HID, trunk_dim=_TRUNK
    )
    with torch.no_grad():
        net.actor.bias += torch.linspace(-2.0, 2.0, _NA)
    path = tmp / "anchor_sharp.pt"
    torch.save({"net_state_dict": net.state_dict()}, str(path))
    return path


def _mean_kl_to_prior(anchor, net, obs_t: torch.Tensor) -> float:
    with torch.no_grad():
        prior_logits, _ = anchor.prior.forward_ac(obs_t)
        prior_logp = F.log_softmax(prior_logits.float(), dim=-1)
        pi_logp = F.log_softmax(net.forward_ac(obs_t)[0].float(), dim=-1)
        kl = (prior_logp.exp() * (prior_logp - pi_logp)).sum(dim=-1)
    return float(kl.mean())


def test_loss_tether_term_enters_loss_and_pulls_toward_prior(tmp_path) -> None:
    """Direct pull: identical single optimizer steps, with and without the
    tether — the coef>0 step must land strictly CLOSER to the prior."""
    path = _make_sharp_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    anchor.set_env_steps(1e9)  # unfrozen: the actor is free to move
    assert not anchor.frozen

    net0, out0 = _run_tether_update(anchor, 0.0)
    netc, outc = _run_tether_update(anchor, 5.0)

    # The term is in the loss: the (single) minibatch carried it, and the
    # final minibatch loss is exactly the untethered loss + coef * KL.
    assert out0["kl_loss_n"] == 0 and out0["kl_loss_accum"] == 0.0
    assert outc["kl_loss_n"] == 1 and outc["kl_loss_accum"] > 0.0
    assert outc["last_loss"] == pytest.approx(
        out0["last_loss"] + 5.0 * outc["kl_loss_accum"], rel=1e-5
    )

    obs_t = torch.from_numpy(
        np.random.default_rng(7).standard_normal((64, _FDIM))
        .astype(np.float32)
    )
    kl_free = _mean_kl_to_prior(anchor, net0, obs_t)
    kl_tethered = _mean_kl_to_prior(anchor, netc, obs_t)
    assert kl_tethered < kl_free, (
        f"tethered step did not pull toward the prior: "
        f"KL {kl_tethered:.6f} vs untethered {kl_free:.6f}"
    )


def test_loss_coef_zero_is_bitwise_identical_to_no_tether(tmp_path) -> None:
    """coef 0.0 (and the key absent entirely) must not perturb the update
    by one bit — same rollout, same seeds, identical weights out."""
    path, _ = _make_anchor_ckpt(tmp_path)
    anchor = _build_anchor(path)
    anchor.set_env_steps(1e9)

    net_plain, out_plain = _run_tether_update(None, 0.0)
    net_zero, out_zero = _run_tether_update(anchor, 0.0)

    assert out_zero["kl_loss_n"] == 0
    assert out_zero["last_loss"] == out_plain["last_loss"]
    plain_sd = net_plain.state_dict()
    zero_sd = net_zero.state_dict()
    for k, v in plain_sd.items():
        assert torch.equal(zero_sd[k], v), (
            f"weight {k} changed with coef=0 — the tether must be inert"
        )


def test_config_schema_registers_kl_loss_coef() -> None:
    from src.training.config_schema import KNOWN_REINFORCE_KEYS, validate_profile
    assert "kl_anchor_loss_coef" in KNOWN_REINFORCE_KEYS
    prof = {"name": "x", "reinforce": {"kl_anchor_loss_coef": 0.3}}
    assert validate_profile(prof) == []


def test_trainer_reads_coef_and_logs_loss_term() -> None:
    """The trainer consumes reinforce.kl_anchor_loss_coef and emits the
    per-iter loss-term magnitude alongside kl_anchor_div (source anchor;
    the update math is pinned by the direct-pull test above)."""
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    assert "kl_anchor_loss_coef" in src
    assert '_mech_metrics["kl_anchor_loss"]' in src
    upd = (ROOT / "src" / "training" / "ppo_updater.py").read_text()
    assert "_kl_anchor_loss_coef" in upd


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
