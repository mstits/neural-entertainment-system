"""Mechanism 1 — invariance-preserving terminal rule for the wavefront PBRS.

`reinforce.wave_terminal_rule: "monotone"` switches the wavefront shaping
stream to the peak-augmented (monotone) potential: shaping pays only on new
per-episode peaks of Phi, true deaths charge -peak (the Grzes zero-terminal
rule on the augmented potential, so any non-clearing episode telescopes to
exactly zero net shaping), and stall/timeout cuts (the WAVE_LOST_K
off-envelope cut) become TRUNCATIONS — no terminal charge, and GAE
bootstraps V instead of 0 (Pardo partial-episode bootstrapping).

Key absent (the default) = the legacy -peak-on-any-non-clear behavior,
byte-identical — pinned by the C0 master golden
(test_char_vanilla_ppo_golden.py) plus the trunc-None equivalence test here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.training.ppo import batched_gae
from src.training.wave_shaping import (
    monotone_wave_step,
    resolve_wave_terminal_rule,
    wave_terminal_charge,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The monotone shaping stream (pure helpers the trainer's inline block calls)
# ---------------------------------------------------------------------------

def _roll_episode(phis, gamma=1.0):
    """Run the monotone stream over an episode's per-step potentials.

    Returns (total_shaping, final_peak) with the per-step helper, exactly
    as the trainer's live branch applies it (peak starts at 0 per episode).
    """
    peak = 0.0
    total = 0.0
    for phi in phis:
        f, peak = monotone_wave_step(peak, float(phi), gamma)
        total += f
    return total, peak


def test_monotone_telescopes_to_zero_on_death() -> None:
    """Advance-retreat-advance-death nets EXACTLY zero total shaping."""
    phis = [10.0, 25.0, 5.0, 0.5, 30.0, 12.0]  # advance, retreat, advance...
    total, peak = _roll_episode(phis, gamma=1.0)
    assert peak == 30.0
    # True death charges -peak; the whole episode's shaping sums to 0.0.
    total += wave_terminal_charge(peak, is_clear=False, truncated=False)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_monotone_pays_only_on_new_peaks() -> None:
    """F == 0 on non-improving steps; F == (new peak - old peak) on peaks."""
    f1, peak = monotone_wave_step(0.0, 10.0, 1.0)
    assert f1 == pytest.approx(10.0)
    assert peak == 10.0
    # Retreat: no payment, peak holds.
    f2, peak = monotone_wave_step(peak, 4.0, 1.0)
    assert f2 == pytest.approx(0.0)
    assert peak == 10.0
    # Re-advance below the peak: still nothing.
    f3, peak = monotone_wave_step(peak, 9.9, 1.0)
    assert f3 == pytest.approx(0.0)
    assert peak == 10.0
    # New peak: pays exactly the increment.
    f4, peak = monotone_wave_step(peak, 17.5, 1.0)
    assert f4 == pytest.approx(7.5)
    assert peak == 17.5


def test_monotone_terminal_charges() -> None:
    """Death charges -peak; clear and truncation charge nothing."""
    assert wave_terminal_charge(12.0, is_clear=False, truncated=False) == -12.0
    assert wave_terminal_charge(12.0, is_clear=True, truncated=False) == 0.0
    assert wave_terminal_charge(12.0, is_clear=False, truncated=True) == 0.0
    # Zero peak: nothing to refund either way.
    assert wave_terminal_charge(0.0, is_clear=False, truncated=False) == 0.0


def test_monotone_gamma_form_matches_spec() -> None:
    """F_t = gamma*max(peak, phi') - peak, per the spec formula."""
    f, peak = monotone_wave_step(10.0, 30.0, 0.99)
    assert f == pytest.approx(0.99 * 30.0 - 10.0)
    assert peak == 30.0


def test_resolve_wave_terminal_rule() -> None:
    assert resolve_wave_terminal_rule(None) is False
    assert resolve_wave_terminal_rule("") is False
    assert resolve_wave_terminal_rule("monotone") is True
    with pytest.raises(ValueError):
        resolve_wave_terminal_rule("montone")  # typo must fail LOUD


# ---------------------------------------------------------------------------
# Truncation bootstrap in the batched GAE sweep (Pardo)
# ---------------------------------------------------------------------------

def test_batched_gae_truncation_bootstraps_value_not_zero() -> None:
    """A truncated done bootstraps the critic's own V; a real done uses 0."""
    T, N = 3, 1
    rewards = np.zeros((T, N), dtype=np.float32)
    rewards[1, 0] = 1.0
    values = np.full((T, N), 5.0, dtype=np.float32)
    dones = np.zeros((T, N), dtype=np.bool_)
    dones[1, 0] = True
    final_values = np.zeros(N, dtype=np.float32)
    gamma, lam = 0.9, 0.95

    # Natural termination: target at the done step is just r (V_next = 0).
    _, tgt_term = batched_gae(rewards, values, dones, final_values, gamma, lam)
    assert tgt_term[1, 0] == pytest.approx(1.0)

    # Truncation: target bootstraps the critic's own estimate,
    # r + gamma * V(s_t) (the house pattern from gae.gae's truncation branch).
    trunc = np.zeros((T, N), dtype=np.bool_)
    trunc[1, 0] = True
    _, tgt_trunc = batched_gae(
        rewards, values, dones, final_values, gamma, lam, trunc_buf=trunc
    )
    assert tgt_trunc[1, 0] == pytest.approx(1.0 + gamma * 5.0)


def test_batched_gae_truncation_still_breaks_the_accumulator() -> None:
    """Advantage from the next episode never leaks across a truncation."""
    T, N = 4, 1
    rng = np.random.default_rng(7)
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = np.zeros((T, N), dtype=np.bool_)
    dones[1, 0] = True
    trunc = np.zeros((T, N), dtype=np.bool_)
    trunc[1, 0] = True
    final_values = rng.normal(size=N).astype(np.float32)
    adv, _ = batched_gae(
        rewards, values, dones, final_values, 0.99, 0.95, trunc_buf=trunc
    )
    # Steps 0-1 must be unaffected by rewards after the boundary.
    rewards2 = rewards.copy()
    rewards2[2:] += 100.0
    adv2, _ = batched_gae(
        rewards2, values, dones, final_values, 0.99, 0.95, trunc_buf=trunc
    )
    np.testing.assert_allclose(adv[:2], adv2[:2], rtol=0, atol=0)


def test_batched_gae_trunc_none_and_all_false_are_byte_identical() -> None:
    """Legacy default unchanged: trunc absent == trunc all-False == old code."""
    T, N = 8, 3
    rng = np.random.default_rng(1234)
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = rng.random(size=(T, N)) < 0.2
    final_values = rng.normal(size=N).astype(np.float32)

    a0, t0 = batched_gae(rewards, values, dones, final_values, 0.99, 0.95)
    a1, t1 = batched_gae(
        rewards, values, dones, final_values, 0.99, 0.95, trunc_buf=None
    )
    a2, t2 = batched_gae(
        rewards, values, dones, final_values, 0.99, 0.95,
        trunc_buf=np.zeros((T, N), dtype=np.bool_),
    )
    assert a0.tobytes() == a1.tobytes() == a2.tobytes()
    assert t0.tobytes() == t1.tobytes() == t2.tobytes()


# ---------------------------------------------------------------------------
# Trainer wiring anchors (the inline block is exercised end-to-end by the C0
# golden for the OFF path; these anchor the ON path's plumbing in source).
# ---------------------------------------------------------------------------

def test_trainer_wires_the_monotone_rule() -> None:
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    assert 'wave_terminal_rule' in src, "trainer never reads the config key"
    assert "monotone_wave_step" in src, "live branch does not use the helper"
    assert "wave_terminal_charge" in src, (
        "terminal branch does not use the helper"
    )
    # The lost-cut marks a truncation under monotone, and the updater
    # receives the truncation mask only when the rule is on.
    assert "trunc_buf[t, i] = True" in src
    assert "trunc_buf=(trunc_buf if wave_monotone else None)" in src


def test_config_schema_registers_wave_terminal_rule() -> None:
    from src.training.config_schema import KNOWN_REINFORCE_KEYS, validate_profile
    assert "wave_terminal_rule" in KNOWN_REINFORCE_KEYS
    prof = {"name": "x", "reinforce": {"wave_terminal_rule": "monotone"}}
    assert validate_profile(prof) == []


# ---------------------------------------------------------------------------
# Integration: the real tile+CPU loop with an unreachable-area dmap, so every
# state reads Phi=0 and the WAVE_LOST_K=150 cut fires for any env surviving
# 150 consecutive steps. Deterministic on the tile+CPU basis (same as C0):
# with seed 1234 the cut lands at t=149 in both iterations.
# ---------------------------------------------------------------------------

_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_monotone_lost_cut_truncates_in_real_loop(monkeypatch) -> None:
    import pickle
    import queue as _queue
    import random
    import tempfile

    import torch
    import yaml

    from src.training.ppo_updater import PPOUpdater

    captured: list = []
    orig_update = PPOUpdater.update

    def _spy(self, **kw):
        tb = kw.get("trunc_buf")
        captured.append((
            None if tb is None else tb.copy(), kw["done_buf"].copy(),
        ))
        return orig_update(self, **kw)

    monkeypatch.setattr(PPOUpdater, "update", _spy)

    def _run(rule):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        with open(_PROFILE) as f:
            profile = yaml.safe_load(f)
        profile["reinforce"]["rollout_steps"] = 200
        profile["reinforce"]["ppo_minibatch_size"] = 64
        profile["reinforce"]["device"] = "cpu"
        with tempfile.TemporaryDirectory(prefix="wave_mono_") as tmp:
            dmap_path = Path(tmp) / "dmap.pkl"
            with open(dmap_path, "wb") as f:
                pickle.dump({(255, 0, 0): 100.0}, f)
            profile["reinforce"]["wavefront_reward"] = {
                "enabled": True, "dmap": str(dmap_path), "phi_target": 100.0,
            }
            if rule:
                profile["reinforce"]["wave_terminal_rule"] = rule
            from src.training.trainer import Trainer
            trainer = Trainer(
                rom_path=str(_SMB_ROM),
                game_profile=profile,
                num_instances=2,
                population_size=2,
                checkpoint_dir=tmp,
                start_state_path=profile.get("start_state_path"),
                env_spec="nes_core",
                max_episode_steps=400,
                metrics_queue=_queue.Queue(),
                device_override="cpu",
                seed=1234,
            )
            trainer.run(num_generations=2, resume_from=None, fresh_start=True)

    _run("monotone")
    assert len(captured) == 2
    for tb, db in captured:
        assert tb is not None, "monotone run must hand the mask to GAE"
        assert tb.any(), "the 150-step off-envelope cut never truncated"
        assert not (tb & ~db).any(), "truncation marked on a non-done step"
        # Phi==0 everywhere -> the cut is exactly 150 consecutive steps in.
        assert tb[149].any()

    captured.clear()
    _run(None)  # legacy default: no mask reaches the GAE path
    assert len(captured) == 2
    assert all(tb is None for tb, _ in captured)
