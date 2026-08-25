"""Regression guards for the 2026-08-24 trainer audit findings.

Each test pins one confirmed defect in `src/training/trainer.py`:

  1. hazard_mask x recurrent: HazardMaskedPolicy masks forward_ac only,
     and the recurrent path calls forward_ac_recurrent through
     __getattr__ delegation — the veto would print ARMED and never
     execute. The trainer must fail loud on the combination.
  2. adversary nets must be built RAW: a hazard-masked PR-MDP adversary
     is vetoed from the death-causing actions it exists to force, and
     the kernel adversary's 2-action head crashes against the 6-action
     risky mask.
  3. `reinforce.recurrent: true` on a pixel encoder is dropped — that
     drop must be loud, and the pixel branch must leave the same
     positive `[policy]` log evidence the tile branch does.
  4. a rung_step_budget cut is a truncation for the TauScheduler and
     must be one for GAE too (bootstrap V(s), not 0) — while the
     wavefront -peak terminal charge, documented deliberate, still
     treats it as a non-clear.
  5. a failed mid-rollout `load_worker_state` must drop the rung
     provenance assigned before the try block, or the iter boundary
     scores a phantom truncation for an attempt with zero steps.

Source-structure assertions follow the precedent set by
`test_vanilla_ppo_characterization.py` (NaN-backstop / clamp-floor
source pins): the rollout loop is a single function, so ordering
contracts inside it are pinned against the source text.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_PATH = ROOT / "src" / "training" / "trainer.py"

_NUM_ACTIONS = 6
_TILE_FEATURE_DIM = 175 * 4


def _trainer_source() -> str:
    return _TRAINER_PATH.read_text()


def _bare_trainer(**attrs):
    """A `Trainer` shell with just the attributes the net-build path
    touches — no pool / ROM / emulator booted (same pattern as
    `test_vanilla_ppo_characterization._bare_tile_trainer`)."""
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t._is_tile_mode = True
    t._recurrent = False
    t.num_actions = _NUM_ACTIONS
    t._tile_feature_dim = _TILE_FEATURE_DIM
    t._tile_hidden_dim = 64
    t._tile_trunk_dim = 32
    t.encoder_kind = "smb_tiles"
    t.use_layernorm = False
    t.device = torch.device("cpu")
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


def _hazard_profile(checkpoint: str) -> dict:
    return {"reinforce": {"hazard_mask": {
        "enabled": True, "checkpoint": checkpoint, "threshold": 0.9,
    }}}


# ---------------------------------------------------------------------------
# Finding 1 (P1, trainer.py hazard block): hazard_mask x recurrent.
# ---------------------------------------------------------------------------
def test_hazard_mask_rejects_recurrent_policy() -> None:
    """The recurrent rollout/update call forward_ac_recurrent, which
    bypasses HazardMaskedPolicy's veto entirely — arming the mask on a
    recurrent run must raise, not print ARMED over a dead mechanism."""
    t = _bare_trainer(
        _recurrent=True,
        game_profile=_hazard_profile("does_not_need_to_exist.pt"),
    )
    with pytest.raises(ValueError, match="hazard_mask"):
        t._make_network()


def test_hazard_mask_still_arms_on_feedforward_tile_policy() -> None:
    """The guard must not break the banked Phase-3 arms: non-recurrent
    tile mode still wraps in HazardMaskedPolicy."""
    from src.training.hazard_mask import HazardMaskedPolicy
    from src.training.hazard_model import HazardMLP

    with tempfile.TemporaryDirectory(prefix="hazard_guard_") as tmp:
        ckpt = Path(tmp) / "hazard.pt"
        torch.save({"state_dict": HazardMLP().state_dict(), "config": {}},
                   ckpt)
        t = _bare_trainer(game_profile=_hazard_profile(str(ckpt)))
        net = t._make_network()
        assert isinstance(net, HazardMaskedPolicy)


# ---------------------------------------------------------------------------
# Finding 2 (P2, trainer.py adversary construction): adversary nets are
# built raw — never hazard-vetoed, never commitment-wrapped.
# ---------------------------------------------------------------------------
def test_adversary_nets_are_built_raw_in_source() -> None:
    src = _trainer_source()
    assert "_adv_net = self._make_network_raw()" in src
    assert "_kadv_net = self._make_network_raw(num_actions=2)" in src
    # The wrapped builder must not reappear at either adversary site.
    assert "_adv_net = self._make_network()" not in src
    assert "_kadv_net = self._make_network(num_actions=2)" not in src


def test_make_network_raw_never_wraps_even_with_hazard_armed() -> None:
    """The raw builder is the adversary path's contract: with the hazard
    arm enabled in the profile, _make_network_raw still returns the bare
    policy (2-action kernel head included, which the 6-action risky mask
    would otherwise masked_fill into a shape crash)."""
    from src.models.tile_policy import TilePolicyNetwork

    t = _bare_trainer(
        game_profile=_hazard_profile("does_not_need_to_exist.pt"))
    net = t._make_network_raw(num_actions=2)
    assert isinstance(net, TilePolicyNetwork)
    logits, _ = net.forward_ac(torch.zeros(3, _TILE_FEATURE_DIM))
    assert logits.shape == (3, 2)


# ---------------------------------------------------------------------------
# Finding 3 (P2, trainer.py __init__ + _make_network_raw): the recurrent
# knob must never be dropped silently, and the pixel branch must leave
# [policy] armed-evidence like the tile branch does.
# ---------------------------------------------------------------------------
def test_recurrent_knob_on_pixel_encoder_warns_loudly(caplog) -> None:
    t = _bare_trainer(_is_tile_mode=False, encoder_kind="nature_dqn")
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        resolved = t._resolve_recurrent({"recurrent": True})
    assert resolved is False
    warnings = [r for r in caplog.records
                if "recurrent" in r.getMessage() and "IGNORED" in r.getMessage()]
    assert warnings, "dropping the recurrent knob must be loud"


def test_recurrent_knob_on_tile_encoder_resolves_silently(caplog) -> None:
    t = _bare_trainer()
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        assert t._resolve_recurrent({"recurrent": True}) is True
        assert t._resolve_recurrent({}) is False
    assert not caplog.records


def test_pixel_policy_logs_armed_evidence(caplog) -> None:
    """Policy-class A/Bs need positive log proof of the class that
    trained; the tile branch always logged it, the pixel branch must
    too."""
    t = _bare_trainer(_is_tile_mode=False, encoder_kind="nature_dqn")
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        net = t._make_network_raw()
    assert type(net).__name__ == "PolicyNetwork"
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("[policy]")]
    assert lines and "PolicyNetwork" in lines[0]


# ---------------------------------------------------------------------------
# Finding 4 (P2, trainer.py rollout loop): the per-rung budget cut must
# reach GAE as a truncation (bootstrap V(s)), written AFTER the wave
# terminal charge so the deliberate -peak(Phi) on a cut is preserved.
# ---------------------------------------------------------------------------
def test_budget_cut_flags_gae_truncation_after_wave_charge() -> None:
    src = _trainer_source()
    cut = src.index("_bwd_env_budget[i] > 0")          # the budget-cut guard
    charge = src.index("wave_terminal_charge(", cut)   # the -peak charge site
    flag = src.index("if done and _bwd_env_trunc[i]:", cut)
    assert src[flag:flag + 120].count("trunc_buf[t, i] = True") == 1
    done_write = src.index("done_buf[t, i] = done", cut)
    # Ordering contract: cut -> wave charge -> trunc flag -> buffers.
    assert cut < charge < flag < done_write, (
        "the budget-cut trunc_buf write must come after the wave terminal "
        "charge (which deliberately still sees the cut as a non-clear) "
        "and inside the same step as the buffer writes"
    )


def test_batched_gae_bootstraps_value_on_truncation() -> None:
    """The semantic the flag buys: a done marked truncated bootstraps
    the critic's V(s) instead of 0 at the cap (Pardo time-limit
    correction) — the budget cut now gets what the lost-cut always got."""
    import numpy as np
    from src.training.ppo import batched_gae

    rewards = np.zeros((3, 1), dtype=np.float32)
    values = np.full((3, 1), 5.0, dtype=np.float32)
    done = np.zeros((3, 1), dtype=bool)
    done[1, 0] = True
    finals = np.zeros(1, dtype=np.float32)
    trunc = np.zeros((3, 1), dtype=bool)

    _, targets_terminal = batched_gae(
        rewards, values, done, finals, 0.99, 0.95, trunc_buf=trunc)
    trunc[1, 0] = True
    _, targets_trunc = batched_gae(
        rewards, values, done, finals, 0.99, 0.95, trunc_buf=trunc)
    # Terminal at the cap: target 0 return. Truncated at the cap: the
    # bootstrap carries the state's value forward.
    assert targets_terminal[1, 0] == pytest.approx(0.0)
    assert targets_trunc[1, 0] > targets_terminal[1, 0]


# ---------------------------------------------------------------------------
# Finding 5 (P2, trainer.py mid-rollout restart): a failed
# load_worker_state must drop the rung provenance assigned above it.
# ---------------------------------------------------------------------------
def test_failed_restart_drops_rung_provenance() -> None:
    src = _trainer_source()
    load = src.index("self.pool.load_worker_state(i, _restart_bytes)")
    handler = src[load:load + 1200]
    assert "except Exception:" in handler
    assert "_bwd_env_src[i] = -1" in handler, (
        "the except path must reset the env's rung provenance, or the "
        "iter boundary scores a phantom truncation for an attempt that "
        "never executed a step"
    )
    # The reset belongs to the failure path, not the success path.
    assert handler.index("except Exception:") < handler.index(
        "_bwd_env_src[i] = -1")
