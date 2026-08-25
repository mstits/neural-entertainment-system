"""Regression tests for the eval_game.py hazard-veto / recurrent-policy guard.

`reinforce.hazard_mask` wraps the loaded net in `HazardMaskedPolicy`, which
overrides `forward_ac` only. A recurrent-stepped policy — a plain tile_gru
checkpoint, or a `commitment_options` checkpoint (which rides the same
hidden-state plumbing) — is driven through `forward_ac_recurrent` instead,
which `HazardMaskedPolicy.__getattr__` silently delegates straight to the
WRAPPED net, bypassing `HazardMask.apply` entirely. `src/training/trainer.py`
raises for exactly this combination (see
`tests/test_trainer_mechanism_guards.py::test_hazard_mask_rejects_recurrent_policy`);
`scripts/eval_game.py` had its own, separate wiring for the same wrap and
never got the guard, so it would print "[eval] hazard veto ARMED", tag
`net_kind` with `+hazard_veto`, and score a run that was, in fact, unmasked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
# eval_game.py lives under scripts/ and is imported by module name (it is
# not a package). Mirrors tests/test_eval_action_select.py.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import eval_game  # noqa: E402
from src.models.tile_policy import (  # noqa: E402
    TilePolicyNetwork,
    TileRecurrentPolicyNetwork,
)
from src.training.hazard_model import HazardMLP, NUM_ACTIONS, OBS_DIM  # noqa: E402

# 6 actions / 712-wide stacked obs to match hazard_model's NUM_ACTIONS/OBS_DIM
# exactly, so a real (non-test-only) profile shape is exercised.
_ACTION_SPACE = [[], ["right"], ["right", "A"], ["right", "B"], ["A"], ["left"]]
assert len(_ACTION_SPACE) == NUM_ACTIONS
_STACK = 4
_FEATURE_DIM = OBS_DIM // _STACK  # 178
_STACKED_DIM = _FEATURE_DIM * _STACK  # 712 == OBS_DIM


def _hazard_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "hazard.pt"
    torch.save({"state_dict": HazardMLP().state_dict(), "config": {}}, p)
    return p


def _profile(tmp_path: Path, hazard_ckpt: Path, *, commitment: bool = False) -> Path:
    data = {
        "name": "Fake Recurrent Guard Game",
        "action_space": _ACTION_SPACE,
        "reinforce": {
            "encoder": "smb_tiles_pos",
            "tile_frame_stack": _STACK,
            "hazard_mask": {
                "enabled": True,
                "checkpoint": str(hazard_ckpt),
                "threshold": 0.9,
            },
        },
    }
    if commitment:
        data["reinforce"]["commitment_options"] = {
            "enabled": True,
            "durations": [1, 2, 4],
        }
    p = tmp_path / "profile.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def _recurrent_checkpoint(tmp_path: Path) -> Path:
    """A plain tile_gru checkpoint: build_tile_policy_from_checkpoint
    detects the `gru.*` keys and returns is_recurrent=True."""
    net = TileRecurrentPolicyNetwork(
        num_actions=len(_ACTION_SPACE), feature_dim=_STACKED_DIM,
    )
    p = tmp_path / "ckpt.pt"
    torch.save({"net_state_dict": net.state_dict()}, p)
    return p


def _commitment_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint shaped for the commitment_options branch (eval_game.py
    strips the `trunk.flat.` prefix back off to rebuild the flat net)."""
    flat = TilePolicyNetwork(num_actions=len(_ACTION_SPACE), feature_dim=_STACKED_DIM)
    sd = {f"trunk.flat.{k}": v for k, v in flat.state_dict().items()}
    p = tmp_path / "ckpt.pt"
    torch.save({"net_state_dict": sd}, p)
    return p


def _plain_checkpoint(tmp_path: Path) -> Path:
    net = TilePolicyNetwork(num_actions=len(_ACTION_SPACE), feature_dim=_STACKED_DIM)
    p = tmp_path / "ckpt.pt"
    torch.save({"net_state_dict": net.state_dict()}, p)
    return p


def _run(profile: Path, ckpt: Path, tmp_path: Path) -> dict:
    return eval_game.eval_one_game(
        game="fakegame",
        profile_path=profile,
        rom_path=str(tmp_path / "nope.nes"),
        n_episodes=1,
        max_steps=10,
        checkpoint=str(ckpt),
    )


def test_hazard_mask_is_refused_for_a_recurrent_tile_gru_checkpoint(tmp_path, capsys):
    hz = _hazard_ckpt(tmp_path)
    profile = _profile(tmp_path, hz)
    ckpt = _recurrent_checkpoint(tmp_path)

    result = _run(profile, ckpt, tmp_path)

    assert result["status"] == "hazard_recurrent_conflict"
    assert result["recurrent"] is True
    assert "TileRecurrentPolicyNetwork" in result["net_kind"]
    assert "hazard_veto" not in result["net_kind"], (
        "net_kind must not claim the veto is armed when it was never wrapped"
    )
    out = capsys.readouterr().out
    assert "ARMED" not in out, "the veto must not be reported armed if it never wraps the net"


def test_hazard_mask_is_refused_for_a_commitment_options_checkpoint(tmp_path, capsys):
    """commitment_options forces is_recurrent=True the same way a plain
    tile_gru checkpoint does; the guard must cover both call sites."""
    hz = _hazard_ckpt(tmp_path)
    profile = _profile(tmp_path, hz, commitment=True)
    ckpt = _commitment_checkpoint(tmp_path)

    result = _run(profile, ckpt, tmp_path)

    assert result["status"] == "hazard_recurrent_conflict"
    assert result["recurrent"] is True
    assert result["net_kind"] == "CommitmentPolicy"
    out = capsys.readouterr().out
    assert "ARMED" not in out


def test_hazard_mask_still_wraps_a_genuine_non_recurrent_checkpoint(
    tmp_path, capsys, monkeypatch,
):
    """The guard must not over-fire: a non-recurrent checkpoint with
    hazard_mask enabled must still reach the wrap. Patches `Pool` to abort
    with a sentinel right after (no ROM/pool needed) so the test only has
    to prove execution got past the hazard block unblocked."""
    hz = _hazard_ckpt(tmp_path)
    profile = _profile(tmp_path, hz)
    ckpt = _plain_checkpoint(tmp_path)

    class _StopHere(Exception):
        pass

    def _boom(*a, **k):
        raise _StopHere("reached pool construction")

    monkeypatch.setattr(eval_game, "Pool", _boom)

    with pytest.raises(_StopHere):
        _run(profile, ckpt, tmp_path)

    out = capsys.readouterr().out
    assert "hazard veto ARMED" in out, (
        "a non-recurrent + hazard_mask run must still arm the veto"
    )
