"""Tests for the F1 instrument — `scripts/critic_explained_variance.py`.

F1 is V29's registered KILL SWITCH: a `Δ_fixed > -0.10` majority refutes
the whole hypothesis and cancels an 8-hour training spend. So the arithmetic
underneath it has to be pinned to reference points that cannot drift —
a perfect critic scores 1.0, a constant critic scores 0.0, an
anti-correlated one scores below 0 — and the return construction has to
prove it censors (never bootstraps) the states whose episode ran past the
end of the rollout.

Everything here is emulator-free. The rollout path is exercised by feeding
`build_batch` a hand-built `Rollout`, which is the same struct the real
collector emits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import critic_explained_variance as cev  # noqa: E402
from src.models.tile_policy import TilePolicyNetwork  # noqa: E402

FEATURE_DIM = 40
NUM_ACTIONS = 6


# ===== EV: the three reference points the decision rule rests on ======


def test_perfect_predictor_scores_one() -> None:
    r = np.array([1.0, 5.0, -2.0, 7.5, 0.25])
    assert cev.explained_variance(r, r) == pytest.approx(1.0)


def test_constant_predictor_scores_zero() -> None:
    """A critic that has learned nothing but the mean explains no variance —
    and, because EV is offset-invariant, ANY constant scores the same 0.0."""
    r = np.array([1.0, 5.0, -2.0, 7.5, 0.25])
    assert cev.explained_variance(r, np.full_like(r, r.mean())) == pytest.approx(0.0)
    assert cev.explained_variance(r, np.full_like(r, 1000.0)) == pytest.approx(0.0)


def test_anticorrelated_predictor_scores_below_zero() -> None:
    """V = -R is worse than predicting the mean. EV is unbounded below on
    purpose: `1 - Var(2R)/Var(R) = -3`."""
    r = np.array([1.0, 5.0, -2.0, 7.5, 0.25])
    assert cev.explained_variance(r, -r) == pytest.approx(-3.0)


def test_offset_invariance() -> None:
    """A miscalibrated MEAN cannot move EV — which is why the instrument
    tolerates the MC-vs-GAE-target offset it measures against a critic
    trained on bootstrapped targets."""
    rng = np.random.default_rng(0)
    r = rng.normal(size=256)
    v = 0.7 * r + rng.normal(scale=0.2, size=256)
    assert cev.explained_variance(r, v) == pytest.approx(
        cev.explained_variance(r, v + 137.0)
    )


def test_scale_sensitivity() -> None:
    """EV is NOT scale-invariant: `V = 2R` scores 0. This is the property
    that makes a wrongly-transformed target (e.g. symlogging a stream the
    vanilla_ppo critic was never fit on) show up as a bogus EV collapse."""
    r = np.array([1.0, 5.0, -2.0, 7.5, 0.25])
    assert cev.explained_variance(r, 2.0 * r) == pytest.approx(0.0)


def test_partial_fit_matches_closed_form() -> None:
    r = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([0.0, 1.0, 2.0, 5.0])          # one residual of 2.0
    expected = 1.0 - float(np.var(r - v)) / float(np.var(r))
    assert cev.explained_variance(r, v) == pytest.approx(expected)


def test_zero_variance_returns_is_undefined() -> None:
    assert cev.explained_variance(np.ones(8), np.zeros(8)) is None


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cev.explained_variance(np.ones(4), np.ones(5))


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError):
        cev.explained_variance(np.zeros(0), np.zeros(0))


# ===== Monte-Carlo returns + censoring ================================


def test_mc_returns_single_terminated_episode() -> None:
    rewards = np.array([1.0, 2.0, 3.0])
    terminal = np.array([False, False, True])
    ret, valid = cev.mc_returns(rewards, terminal, gamma=0.5)
    assert valid.all()
    np.testing.assert_allclose(ret, [1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3.0])


def test_mc_returns_restarts_at_every_terminal() -> None:
    """Two episodes in one buffer: the second episode's reward must not leak
    backwards into the first — that leak is exactly what `done` masking
    exists to prevent in the trainer's GAE."""
    rewards = np.array([1.0, 1.0, 100.0, 1.0])
    terminal = np.array([False, True, False, True])
    ret, valid = cev.mc_returns(rewards, terminal, gamma=0.9)
    assert valid.all()
    np.testing.assert_allclose(ret, [1 + 0.9 * 1, 1.0, 100 + 0.9 * 1, 1.0])


def test_mc_returns_censors_the_unterminated_tail() -> None:
    """No terminal after t => no realized return. The mask must say so; a
    bootstrap here would fold the critic's own opinion into its own target."""
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    terminal = np.array([False, True, False, False])
    _, valid = cev.mc_returns(rewards, terminal, gamma=0.9)
    np.testing.assert_array_equal(valid, [True, True, False, False])


def test_mc_returns_all_censored_when_nothing_terminates() -> None:
    _, valid = cev.mc_returns(np.ones(16), np.zeros(16, dtype=bool), gamma=0.99)
    assert not valid.any()


def test_mc_returns_2d_matches_column_wise_1d() -> None:
    rng = np.random.default_rng(3)
    rewards = rng.normal(size=(24, 3))
    terminal = rng.random((24, 3)) < 0.2
    ret2, valid2 = cev.mc_returns(rewards, terminal, gamma=0.97)
    for col in range(3):
        ret1, valid1 = cev.mc_returns(rewards[:, col], terminal[:, col], 0.97)
        np.testing.assert_allclose(ret2[:, col], ret1)
        np.testing.assert_array_equal(valid2[:, col], valid1)


def test_mc_returns_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cev.mc_returns(np.ones((4, 2)), np.zeros((4, 3), dtype=bool), 0.99)


def test_mc_returns_gamma_one_is_the_undiscounted_sum() -> None:
    rewards = np.array([3.0, -1.0, 4.0])
    terminal = np.array([False, False, True])
    ret, _ = cev.mc_returns(rewards, terminal, gamma=1.0)
    assert ret[0] == pytest.approx(6.0)


# ===== subsampling ====================================================


def test_subsample_is_uniform_without_replacement_and_sorted() -> None:
    idx = cev.subsample_indices(1000, 50, np.random.default_rng(0))
    assert idx.size == 50
    assert len(set(idx.tolist())) == 50
    assert (np.diff(idx) > 0).all()
    assert idx.max() < 1000


def test_subsample_returns_everything_when_batch_exceeds_supply() -> None:
    idx = cev.subsample_indices(7, 16384, np.random.default_rng(0))
    np.testing.assert_array_equal(idx, np.arange(7))


def test_subsample_is_seed_deterministic() -> None:
    a = cev.subsample_indices(500, 32, np.random.default_rng(11))
    b = cev.subsample_indices(500, 32, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


def test_subsample_of_nothing_is_empty() -> None:
    assert cev.subsample_indices(0, 10, np.random.default_rng(0)).size == 0


# ===== the pre-registered three-way kill rule =========================


def test_classify_delta_bands() -> None:
    assert cev.classify_delta(-0.15) == "live"      # boundary is inclusive
    assert cev.classify_delta(-0.40) == "live"
    assert cev.classify_delta(-0.10) == "band"      # > -0.10 is required for dead
    assert cev.classify_delta(-0.12) == "band"
    assert cev.classify_delta(-0.09) == "dead"
    assert cev.classify_delta(+0.30) == "dead"


def test_verdict_live_needs_five_of_eight() -> None:
    deltas = {f"r{i}": (-0.2 if i < 5 else 0.0) for i in range(8)}
    v = cev.f1_verdict(deltas)
    assert v["verdict"] == "LIVE"
    assert v["counts"] == {"live": 5, "band": 0, "dead": 3, "runs_measured": 8}


def test_verdict_dead_needs_five_of_eight() -> None:
    deltas = {f"r{i}": (0.0 if i < 5 else -0.9) for i in range(8)}
    assert cev.f1_verdict(deltas)["verdict"] == "DEAD"


def test_verdict_four_four_split_is_inconclusive() -> None:
    deltas = {f"r{i}": (-0.5 if i < 4 else 0.0) for i in range(8)}
    assert cev.f1_verdict(deltas)["verdict"] == "INCONCLUSIVE"


def test_verdict_majority_in_the_dead_band_is_inconclusive() -> None:
    """A field parked in the -0.15..-0.10 gap is ambiguous by construction
    and must not be read as a green light."""
    assert cev.f1_verdict({f"r{i}": -0.12 for i in range(8)})["verdict"] == "INCONCLUSIVE"


def test_verdict_partial_field_is_inconclusive_even_when_unanimous() -> None:
    """Three runs all screaming LIVE is still not the registered majority."""
    v = cev.f1_verdict({f"r{i}": -0.9 for i in range(3)})
    assert v["verdict"] == "INCONCLUSIVE"
    assert "3 of 8" in v["reason"]


# ===== critic scoring against a network with a known value head =======


def _net(seed: int = 0) -> TilePolicyNetwork:
    torch.manual_seed(seed)
    return TilePolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
        hidden_dim=16, trunk_dim=8,
    ).eval()


def _obs(n: int = 200, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-8, 8, size=(n, FEATURE_DIM), dtype=np.int8)


def test_critic_values_match_a_single_unchunked_forward() -> None:
    """The chunked scoring pass must be exact, not approximately right —
    a chunk-boundary bug would show up as a small EV wobble nobody could
    attribute."""
    net, obs = _net(), _obs()
    with torch.no_grad():
        _, direct = net.forward_ac(torch.from_numpy(obs).float())
    np.testing.assert_allclose(
        cev.critic_values(net, obs, chunk=37), direct.numpy(), rtol=0, atol=1e-6,
    )


def test_critic_that_predicts_the_returns_exactly_scores_one() -> None:
    """A synthetic critic with known EV: take the net's own V as the target
    and the instrument must return exactly 1.0."""
    net, obs = _net(), _obs()
    v = cev.critic_values(net, obs)
    assert cev.explained_variance(v, cev.critic_values(net, obs)) == pytest.approx(1.0)


def test_synthetic_critic_with_planted_noise_recovers_the_planted_ev() -> None:
    """Plant a residual of known variance on top of the critic's own output
    and check the recovered EV matches the closed form."""
    net, obs = _net(), _obs(n=4000)
    v = cev.critic_values(net, obs)
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=float(v.std()) * 0.5, size=v.size)
    returns = v + noise
    expected = 1.0 - float(np.var(returns - v)) / float(np.var(returns))
    assert cev.explained_variance(returns, v) == pytest.approx(expected)
    assert 0.6 < expected < 0.9        # a genuinely partial fit, not 0 or 1


def test_load_policy_refuses_a_recurrent_checkpoint() -> None:
    """A tile_gru critic scored through the stateless forward would zero its
    GRU state at every sample and report a fabricated collapse. Fail loud."""
    from src.models.tile_policy import TileRecurrentPolicyNetwork
    net = TileRecurrentPolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM, hidden_dim=16, gru_dim=8,
    )
    ckpt = {"kind": "tile_gru", "net_state_dict": net.state_dict()}
    with pytest.raises(ValueError, match="recurrent"):
        cev.load_policy(ckpt, _spec())


# ===== batch construction from a hand-built rollout ===================


def _spec(**overrides) -> cev.RunSpec:
    base = dict(
        profile_path=Path("configs/fake.yaml"), profile={}, run_name="fake_run",
        rom_path="fake.nes", ckpt_dir=Path("checkpoints/fake"),
        start_state_path=Path("fake.state"), start_bytes=b"", bitmasks=[0] * NUM_ACTIONS,
        extractor=None, feature_dim=FEATURE_DIM, stacked_dim=FEATURE_DIM, stack_size=1,
        gamma=0.9, symlog_rewards=True, sticky_prob=0.25, sticky_boundary_reset=True,
        rnd_intrinsic_coef=0.0, rollout_steps=8, num_envs=2, max_episode_steps=2400,
        bwd_states_dir=None, bwd_window_frames=160, bwd_entrance_weight=1.0,
        bwd_pin_entrance=False,
    )
    base.update(overrides)
    return cev.RunSpec(**base)


def _rollout(rewards, terminal, *, obs=None, age=None) -> cev.Rollout:
    rewards = np.asarray(rewards, dtype=np.float32)
    terminal = np.asarray(terminal, dtype=bool)
    n_steps, n_envs = rewards.shape
    if obs is None:
        obs = np.zeros((n_steps, n_envs, FEATURE_DIM), dtype=np.int8)
        obs[..., 0] = np.arange(n_steps * n_envs).reshape(n_steps, n_envs) % 100
    if age is None:
        age = np.full((n_steps, n_envs), 99, dtype=np.uint16)
    return cev.Rollout(
        obs=obs, rewards=rewards, terminal=terminal, restart_age=age,
        n_episodes=int(terminal.sum()), n_clears=0, tau=0, seconds=0.0,
    )


def test_build_batch_keeps_only_states_with_a_realized_return() -> None:
    rewards = np.array([[1.0], [1.0], [1.0], [1.0]])
    terminal = np.array([[False], [True], [False], [False]])
    batch = cev.build_batch(
        _rollout(rewards, terminal), _spec(), rnd=None, batch_size=100,
        rng=np.random.default_rng(0),
    )
    assert batch["n_sampled"] == 2          # steps 2 and 3 are censored
    np.testing.assert_allclose(batch["returns"], [1 + 0.9 * 1, 1.0])


def test_build_batch_does_not_symlog_the_vanilla_ppo_reward_stream() -> None:
    """`reinforce.symlog_rewards` is inert under trainer_mode: vanilla_ppo
    (symlog lives only in `trainer._reinforce_update`). Applying it here
    would put R on a scale the critic was never fit to — measured as a
    spurious EV of -0.21 against a genuine +0.38."""
    rewards = np.array([[100.0]])
    batch = cev.build_batch(
        _rollout(rewards, np.array([[True]])), _spec(symlog_rewards=True),
        rnd=None, batch_size=8, rng=np.random.default_rng(0),
    )
    assert batch["returns"][0] == pytest.approx(100.0)   # not log1p(100) ~ 4.6


def test_build_batch_refuses_a_fully_censored_rollout() -> None:
    """An EV over zero samples is not a measurement — say so instead of
    emitting a receipt built on nothing."""
    with pytest.raises(RuntimeError, match="realized return"):
        cev.build_batch(
            _rollout(np.ones((6, 2)), np.zeros((6, 2), dtype=bool)), _spec(),
            rnd=None, batch_size=8, rng=np.random.default_rng(0),
        )


def test_build_batch_min_restart_age_filters_post_restart_states() -> None:
    rewards = np.ones((4, 1), dtype=np.float32)
    terminal = np.array([[False], [False], [False], [True]])
    age = np.array([[0], [1], [2], [3]], dtype=np.uint16)
    kept = cev.build_batch(
        _rollout(rewards, terminal, age=age), _spec(), rnd=None, batch_size=100,
        rng=np.random.default_rng(0), min_restart_age=2,
    )
    assert kept["n_sampled"] == 2
    np.testing.assert_array_equal(kept["restart_age"], [2, 3])
    # ...and the default keeps them, because the critic was fit on them.
    allkept = cev.build_batch(
        _rollout(rewards, terminal, age=age), _spec(), rnd=None, batch_size=100,
        rng=np.random.default_rng(0),
    )
    assert allkept["n_sampled"] == 4


def test_build_batch_without_rnd_emits_no_intrinsic_returns() -> None:
    batch = cev.build_batch(
        _rollout(np.ones((2, 1)), np.array([[False], [True]])), _spec(),
        rnd=None, batch_size=8, rng=np.random.default_rng(0),
    )
    assert batch["returns_with_intrinsic"] is None


def test_build_batch_folds_intrinsic_and_zeroes_it_on_terminals() -> None:
    """Mirrors `ppo.fold_intrinsic_into_rewards`: the bonus is added to every
    live step and zeroed on the terminal one."""

    class _ConstRND:
        """A stand-in whose normalized bonus is exactly 1.0 per state."""

        def __call__(self, x):
            return torch.ones(x.shape[0])

        def normalize_bonus(self, err):
            return err

    rewards = np.array([[0.0], [0.0]], dtype=np.float32)
    terminal = np.array([[False], [True]])
    batch = cev.build_batch(
        _rollout(rewards, terminal), _spec(rnd_intrinsic_coef=0.5),
        rnd=_ConstRND(), batch_size=8, rng=np.random.default_rng(0),
    )
    # step 1 is terminal -> bonus 0; step 0 is live -> bonus 0.5.
    np.testing.assert_allclose(batch["returns_with_intrinsic"], [0.5, 0.0])
    np.testing.assert_allclose(batch["returns"], [0.0, 0.0])


# ===== receipts =======================================================


def test_reference_batch_round_trips(tmp_path: Path) -> None:
    rewards = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    terminal = np.array([[False, True], [True, True]])
    batch = cev.build_batch(
        _rollout(rewards, terminal), _spec(), rnd=None, batch_size=100,
        rng=np.random.default_rng(0),
    )
    meta = {"iter": 120, "run": "fake_run", "n_valid": batch["n_valid"]}
    path = tmp_path / "ref.npz"
    cev.save_reference(path, batch, meta)
    back, back_meta = cev.load_reference(path)
    assert back_meta == meta
    np.testing.assert_array_equal(back["obs"], batch["obs"])
    np.testing.assert_allclose(back["returns"], batch["returns"])
    np.testing.assert_array_equal(back["restart_age"], batch["restart_age"])


def test_reference_mismatches_accepts_an_identical_harness() -> None:
    harness = {"num_envs": 60, "rollout_steps": 1024, "batch_size": 16384,
               "gamma": 0.99, "seed": 0, "torch_threads": 8}
    meta = {"run": "r", "iter": 120, "harness": dict(harness)}
    assert cev.reference_mismatches(meta, harness, "r", 120) == []
    # torch_threads is deliberately NOT batch-determining.
    meta["harness"]["torch_threads"] = 1
    assert cev.reference_mismatches(meta, harness, "r", 120) == []


def test_reference_mismatches_flags_a_different_rollout_shape() -> None:
    """A cached batch collected at a different shape must not be reused
    silently: EV_fixed's whole premise is that every grid point sees the
    same states, and the emitted harness block would misdescribe it."""
    harness = {"num_envs": 60, "rollout_steps": 1024, "batch_size": 16384,
               "gamma": 0.99, "seed": 0}
    meta = {"run": "r", "iter": 120,
            "harness": dict(harness, num_envs=4, rollout_steps=256)}
    bad = cev.reference_mismatches(meta, harness, "r", 120)
    assert len(bad) == 2
    assert any("num_envs" in b for b in bad)
    assert any("rollout_steps" in b for b in bad)


def test_reference_mismatches_flags_a_different_run_or_iter() -> None:
    harness = {"num_envs": 60}
    meta = {"run": "other", "iter": 240, "harness": dict(harness)}
    bad = cev.reference_mismatches(meta, harness, "r", 120)
    assert any(b.startswith("run:") for b in bad)
    assert any(b.startswith("iter:") for b in bad)


def test_verdict_mode_writes_a_receipt(tmp_path: Path) -> None:
    for i in range(8):
        (tmp_path / f"f1_run{i}.json").write_text(json.dumps(
            {"run": f"run{i}", "delta_fixed": -0.3}
        ))
    args = cev.build_parser().parse_args(["--verdict", str(tmp_path)])
    assert cev.run_verdict(args) == 0
    out = json.loads((tmp_path / "f1_verdict.json").read_text())
    assert out["verdict"] == "LIVE"
    assert len(out["per_run"]) == 8


def test_verdict_mode_skips_runs_with_no_delta(tmp_path: Path) -> None:
    (tmp_path / "f1_a.json").write_text(json.dumps({"run": "a", "delta_fixed": None}))
    (tmp_path / "f1_b.json").write_text(json.dumps({"run": "b", "delta_fixed": -0.4}))
    args = cev.build_parser().parse_args(["--verdict", str(tmp_path)])
    cev.run_verdict(args)
    out = json.loads((tmp_path / "f1_verdict.json").read_text())
    assert set(out["per_run"]) == {"b"}
    assert out["verdict"] == "INCONCLUSIVE"


# ===== grid resolution ================================================


def _fake_ckpt_dir(tmp_path: Path, iters) -> Path:
    for it in iters:
        (tmp_path / f"vanilla_ppo_iter_{it:05d}.pt").write_bytes(b"")
    return tmp_path


def test_available_iters_reads_the_archive(tmp_path: Path) -> None:
    d = _fake_ckpt_dir(tmp_path, [10, 240, 120])
    (tmp_path / "winners").mkdir()
    (tmp_path / "vanilla_ppo_iter_bogus.pt").write_bytes(b"")
    assert cev.available_iters(d) == [10, 120, 240]


def test_resolve_grid_intersects_offsets_with_what_is_on_disk(tmp_path: Path) -> None:
    spec = _spec(ckpt_dir=_fake_ckpt_dir(tmp_path, range(10, 250, 10)))
    grid = cev.resolve_grid(spec, 200, cev.DEFAULT_GRID_OFFSETS, None)
    # 200+40 = 240 exists; 200+60 = 260 does not.
    assert grid == [180, 190, 200, 210, 220, 240]


def test_resolve_grid_requires_the_peak_checkpoint(tmp_path: Path) -> None:
    spec = _spec(ckpt_dir=_fake_ckpt_dir(tmp_path, [10, 20, 30]))
    with pytest.raises(FileNotFoundError, match="peak checkpoint"):
        cev.resolve_grid(spec, 200, cev.DEFAULT_GRID_OFFSETS, None)


def test_resolve_grid_explicit_iters_must_all_exist(tmp_path: Path) -> None:
    spec = _spec(ckpt_dir=_fake_ckpt_dir(tmp_path, [10, 20]))
    assert cev.resolve_grid(spec, None, (), [20, 10]) == [10, 20]
    with pytest.raises(FileNotFoundError):
        cev.resolve_grid(spec, None, (), [10, 999])


def test_resolve_grid_without_a_peak_or_iters_raises(tmp_path: Path) -> None:
    spec = _spec(ckpt_dir=_fake_ckpt_dir(tmp_path, [10]))
    with pytest.raises(ValueError):
        cev.resolve_grid(spec, None, cev.DEFAULT_GRID_OFFSETS, None)
