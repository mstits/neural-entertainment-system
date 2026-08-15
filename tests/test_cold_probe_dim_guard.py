"""Tests for the cold_probe checkpoint/observation width guard.

`eval_game.py --prev-action-feature` trains tile nets on a 718-dim
observation (712-dim stacked tiles + a 6-wide one-hot of a_{t-1}).
`cold_probe.probe()` never passes that flag, so it always builds the
plain 712-dim observation. Loading a 718-wide checkpoint's `fc1.weight`
into a 712-wide net is a shape mismatch that `load_state_dict(...,
strict=False)` silently drops instead of raising, so without this guard
the probe would score such a checkpoint with a half-random first layer
and still report a clean number.
"""

from __future__ import annotations

import subprocess
import json
from unittest import mock

import pytest
import torch

import src.training.cold_probe as cold_probe
from src.training.cold_probe import COLD_METRIC_KEYS, probe


def _cfg(**over) -> dict:
    d = {
        "name": "Super Mario Bros.",
        "reinforce": {"encoder": "smb_tiles_pos", "tile_frame_stack": 4},
    }
    d.update(over)
    return d


def _sd_with_fc1_width(width: int) -> dict:
    return {
        "fc1.weight": torch.zeros(256, width),
        "fc1.bias": torch.zeros(256),
    }


# --- pure guard function -----------------------------------------------------

def test_checkpoint_input_width_reads_fc1_columns():
    sd = _sd_with_fc1_width(712)
    assert cold_probe._checkpoint_input_width(sd) == 712


def test_checkpoint_input_width_reads_recurrent_input_proj():
    sd = {"input_proj.0.weight": torch.zeros(128, 712)}
    assert cold_probe._checkpoint_input_width(sd) == 712


def test_checkpoint_input_width_none_when_no_known_key():
    assert cold_probe._checkpoint_input_width({"actor.weight": torch.zeros(6, 64)}) is None


def test_check_observation_width_raises_on_prev_action_feature_checkpoint():
    sd = _sd_with_fc1_width(718)
    with pytest.raises(ValueError) as exc:
        cold_probe._check_observation_width(sd, _cfg())
    msg = str(exc.value)
    assert "718" in msg
    assert "712" in msg
    assert "prev-action-feature" in msg or "prev_action_feature" in msg


def test_check_observation_width_passes_on_matching_checkpoint():
    sd = _sd_with_fc1_width(712)
    cold_probe._check_observation_width(sd, _cfg())  # must not raise


def test_check_observation_width_skips_non_tile_profile():
    sd = _sd_with_fc1_width(718)
    cold_probe._check_observation_width(sd, {"name": "Pixel Game"})  # no reinforce.encoder


# --- wired into probe() ------------------------------------------------------

@pytest.fixture
def rom_file(tmp_path) -> str:
    p = tmp_path / "fake.nes"
    p.write_bytes(b"\x00" * 16)
    return str(p)


def test_probe_rejects_prev_action_feature_checkpoint_without_subprocess(rom_file):
    net = {"fc1.weight": torch.zeros(256, 718)}
    with mock.patch.object(subprocess, "run") as m:
        out = probe(net, _cfg(), episodes=8, rom_path=rom_file)
    m.assert_not_called()
    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_status"] == "probe_failed"
    assert out["cold_error"] and "718" in out["cold_error"] and "712" in out["cold_error"]


def test_probe_accepts_matching_width_checkpoint(rom_file):
    net = {"fc1.weight": torch.zeros(256, 712)}
    ok_json = {
        "status": "ok", "checkpoint": "x", "n_episodes": 8,
        "mean_return": 1.0, "mean_length": 1.0, "max_byte_seen": 0, "clear_rate": 0.0,
    }
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(ok_json), stderr="")
    with mock.patch.object(subprocess, "run", return_value=completed) as m:
        out = probe(net, _cfg(), episodes=8, sequential=False, rom_path=rom_file)
    m.assert_called_once()
    assert out["cold_error"] is None
    assert out["cold_status"] == "ok"
