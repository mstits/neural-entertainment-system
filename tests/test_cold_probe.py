"""Tests for the cold-eval probe (Lane 3 of the SMB one-shot campaign).

Three layers:
  * pure `_map_metrics` / `_parse_eval_stdout` — the JSON → ``cold_*`` projection;
  * mocked-`subprocess.run` probe paths — sequential success and the graceful
    degrade when `eval_game.py --sequential` is not yet available (Lane 1);
  * one real end-to-end probe against a known super_mario_bros checkpoint,
    asserting the guaranteed keys and their ranges (marked slow).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

import src.training.cold_probe as cold_probe
from src.training.cold_probe import COLD_METRIC_KEYS, probe

_ROOT = Path(__file__).resolve().parent.parent


# --- synthetic eval_game.py result JSONs -----------------------------------

def _legacy_json(**over) -> dict:
    """An `eval_game.py` result as it looks WITHOUT --sequential (Lane 1 absent)."""
    d = {
        "game": "mario", "status": "ok",
        "checkpoint": "checkpoints/super_mario_bros/vanilla_ppo_iter_00000.pt",
        "recurrent": False, "stage": "start", "n_episodes": 8,
        "mean_return": 1200.0, "mean_length": 640.0,
        "max_byte_seen": 0, "mean_max_byte": 0.0, "clear_rate": 0.25,
        "timestamp": 1.0,
    }
    d.update(over)
    return d


def _sequential_json(**over) -> dict:
    """An `eval_game.py --sequential` result (Lane 1 present)."""
    d = _legacy_json(
        seq_clear_rate=0.125, furthest_seq_level=[1, 4],
        furthest_any_level=[2, 1], warp_rate=0.25, max_byte_seen=0x14,
    )
    d.update(over)
    return d


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# --- pure projection --------------------------------------------------------

def test_map_metrics_sequential_populates_seq_fields():
    out = cold_probe._map_metrics(_sequential_json(), sequential_ran=True)
    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_sequential"] is True
    assert out["cold_seq_clear_rate"] == pytest.approx(0.125)
    assert out["cold_furthest_seq"] == [1, 4]
    assert out["cold_furthest_any"] == [2, 1]
    assert out["cold_warp_rate"] == pytest.approx(0.25)
    assert out["cold_clear_1_1"] == pytest.approx(0.25)
    assert out["cold_error"] is None


def test_map_metrics_degraded_leaves_seq_fields_none():
    out = cold_probe._map_metrics(_legacy_json(), sequential_ran=False)
    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_sequential"] is False
    # Sequential predicate did not run — these are unknown, not fabricated.
    assert out["cold_seq_clear_rate"] is None
    assert out["cold_furthest_seq"] is None
    assert out["cold_furthest_any"] is None
    assert out["cold_warp_rate"] is None
    # Always-available legacy metrics still flow through.
    assert out["cold_clear_1_1"] == pytest.approx(0.25)
    assert out["cold_mean_return"] == pytest.approx(1200.0)
    assert out["cold_n_episodes"] == 8


def test_map_metrics_ran_sequential_but_json_lacks_keys_stays_none():
    # Flag was passed but the run errored before emitting seq metrics.
    out = cold_probe._map_metrics(_legacy_json(), sequential_ran=True)
    assert out["cold_sequential"] is False
    assert out["cold_seq_clear_rate"] is None


def test_map_metrics_nonok_status_sets_error():
    out = cold_probe._map_metrics(
        {"status": "unsupported_profile", "detail": "pixel encoder"}, sequential_ran=False
    )
    assert out["cold_status"] == "unsupported_profile"
    assert out["cold_error"] and "unsupported_profile" in out["cold_error"]


def test_parse_eval_stdout_clean_and_noisy():
    j = _sequential_json()
    assert cold_probe._parse_eval_stdout(json.dumps(j, indent=2)) == j
    noisy = "[nes_core] warming up\nrandom banner line\n" + json.dumps(j)
    assert cold_probe._parse_eval_stdout(noisy) == j


# --- mocked subprocess paths ------------------------------------------------

@pytest.fixture
def rom_file(tmp_path) -> str:
    """A stand-in ROM file so `probe`'s existence check passes under mocks."""
    p = tmp_path / "fake.nes"
    p.write_bytes(b"\x00" * 16)
    return str(p)


@pytest.fixture(autouse=True)
def _reset_seq_memo():
    cold_probe._SEQUENTIAL_SUPPORTED = None
    yield
    cold_probe._SEQUENTIAL_SUPPORTED = None


def test_probe_sequential_success(rom_file):
    cfg = {"name": "Super Mario Bros."}
    net = {"fc1.weight": torch.zeros(2, 2)}
    with mock.patch.object(subprocess, "run", return_value=_completed(json.dumps(_sequential_json()))):
        out = probe(net, cfg, episodes=8, sequential=True, rom_path=rom_file)
    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_sequential"] is True
    assert out["cold_seq_clear_rate"] == pytest.approx(0.125)
    assert out["cold_furthest_seq"] == [1, 4]
    assert out["cold_error"] is None


def test_probe_auto_degrades_when_flag_unrecognized(rom_file):
    """Auto mode: first call rejects --sequential, probe retries without it."""
    cfg = {"name": "Super Mario Bros."}
    net = {"fc1.weight": torch.zeros(2, 2)}
    unrec = _completed("", returncode=2,
                       stderr="eval_game.py: error: unrecognized arguments: --sequential")
    ok = _completed(json.dumps(_legacy_json()))
    with mock.patch.object(subprocess, "run", side_effect=[unrec, ok]) as m:
        out = probe(net, cfg, episodes=8, sequential=None, rom_path=rom_file)
    assert m.call_count == 2  # tried with the flag, then degraded
    assert out["cold_sequential"] is False
    assert out["cold_seq_clear_rate"] is None
    assert out["cold_clear_1_1"] == pytest.approx(0.25)
    assert cold_probe._SEQUENTIAL_SUPPORTED is False


def test_probe_nonzero_exit_is_nonblocking(rom_file):
    cfg = {"name": "Super Mario Bros."}
    net = {"fc1.weight": torch.zeros(2, 2)}
    with mock.patch.object(subprocess, "run",
                           return_value=_completed("", returncode=1, stderr="boom")):
        out = probe(net, cfg, sequential=False, rom_path=rom_file)
    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_status"] == "probe_failed"
    assert out["cold_error"] and "boom" in out["cold_error"]


def test_probe_timeout_is_nonblocking(rom_file):
    cfg = {"name": "Super Mario Bros."}
    net = {"fc1.weight": torch.zeros(2, 2)}
    with mock.patch.object(subprocess, "run",
                           side_effect=subprocess.TimeoutExpired(cmd="eval", timeout=1.0)):
        out = probe(net, cfg, sequential=False, rom_path=rom_file, timeout=1.0)
    assert out["cold_status"] == "timeout"
    assert out["cold_error"] is not None


def test_probe_missing_rom_is_nonblocking():
    out = probe({"w": torch.zeros(1)}, {"name": "X"}, rom_path="/no/such/rom.nes")
    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_status"] == "no_rom"


# --- real end-to-end probe against a known checkpoint -----------------------

def _find(rel_candidates: list[str]) -> Path | None:
    """Locate gitignored data (ROM / start state / checkpoint).

    Searches this checkout then each ancestor directory, so the test finds the
    data whether it runs in the primary working copy (data at ``_ROOT``) or in
    an isolated worktree whose gitignored ``roms``/``checkpoints`` live in a
    parent working copy — without hardcoding any tooling-specific directory.
    """
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            for hit in sorted(root.glob(rel)):
                if hit.exists():
                    return hit
    return None


@pytest.mark.slow
def test_probe_real_checkpoint_keys_and_ranges():
    """DoD-style: run the probe against a real super_mario_bros iter checkpoint."""
    rom = _find(["roms/Super Mario Bros. (World).nes"])
    start = _find(["roms/Super Mario Bros. (World)_start.state.bin"])
    ckpt = _find(["checkpoints/super_mario_bros/vanilla_ppo_iter_*.pt"])
    if rom is None or start is None:
        pytest.skip("SMB ROM / start state not present in this checkout")

    cfg = yaml.safe_load((_ROOT / "configs" / "mario_vanilla_ppo.yaml").read_text())
    cfg["start_state_path"] = str(start)

    if ckpt is not None:
        net = torch.load(str(ckpt), map_location="cpu")["net_state_dict"]
    else:
        # Fall back to a correctly-shaped fresh net so the path still runs.
        from src.models.tile_policy import TilePolicyNetwork
        from src.training.profile_utils import action_space_to_bitmasks, resolve_encoder
        _, _, stacked = resolve_encoder(cfg)
        n_actions = len(action_space_to_bitmasks(cfg["action_space"]))
        net = TilePolicyNetwork(num_actions=n_actions, feature_dim=stacked)

    out = probe(net, cfg, episodes=2, max_steps=200, rom_path=str(rom), timeout=600.0)

    assert set(out) == set(COLD_METRIC_KEYS)
    assert out["cold_error"] is None, out
    assert out["cold_status"] == "ok"
    assert out["cold_n_episodes"] == 2
    assert 0.0 <= out["cold_clear_1_1"] <= 1.0
    assert out["cold_max_byte"] >= 0
    assert isinstance(out["cold_sequential"], bool)
    if out["cold_sequential"]:
        assert 0.0 <= out["cold_seq_clear_rate"] <= 1.0
        assert out["cold_warp_rate"] is None or 0.0 <= out["cold_warp_rate"] <= 1.0
