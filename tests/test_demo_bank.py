"""DemoBank.from_npz per-file validation.

Two failure modes were reaching users late and unhelpfully: a file with
no obs_-prefixed keys (e.g. the state/action/next_state schema written
by scripts/gen_iq_transitions.py) silently contributed zero rows and
only tripped the aggregate "no demos found" assert once every file in
the list happened to be empty, and a feature-width mismatch across
files died inside torch.cat with no indication of which file was at
fault.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.demo_bank import DemoBank


def _write_obs_npz(path, n=8, feature_dim=4, num_actions=4):
    obs = np.random.randint(0, 127, size=(n, feature_dim), dtype=np.int8)
    act = np.random.randint(0, num_actions, size=(n,), dtype=np.int64)
    np.savez(str(path), obs_0=obs, act_0=act)


def _write_transition_npz(path, n=8, feature_dim=4):
    state = np.random.randint(0, 127, size=(n, feature_dim), dtype=np.int8)
    action = np.random.randint(0, 4, size=(n,), dtype=np.int64)
    next_state = np.random.randint(0, 127, size=(n, feature_dim), dtype=np.int8)
    np.savez(str(path), state=state, action=action, next_state=next_state)


def test_from_npz_raises_on_file_with_no_obs_keys(tmp_path):
    bad = tmp_path / "transitions.npz"
    _write_transition_npz(bad)

    with pytest.raises(ValueError) as exc:
        DemoBank.from_npz([str(bad)], feature_dim=4, num_actions=4,
                           device=torch.device("cpu"))

    msg = str(exc.value)
    assert str(bad) in msg
    assert "state" in msg or "action" in msg or "next_state" in msg


def test_from_npz_raises_on_mixed_good_and_empty_files(tmp_path):
    good = tmp_path / "good.npz"
    bad = tmp_path / "transitions.npz"
    _write_obs_npz(good)
    _write_transition_npz(bad)

    with pytest.raises(ValueError) as exc:
        DemoBank.from_npz([str(good), str(bad)], feature_dim=4, num_actions=4,
                           device=torch.device("cpu"))

    assert str(bad) in str(exc.value)


def test_from_npz_raises_on_mismatched_feature_dim_across_files(tmp_path):
    narrow = tmp_path / "narrow.npz"
    wide = tmp_path / "wide.npz"
    _write_obs_npz(narrow, feature_dim=4)
    _write_obs_npz(wide, feature_dim=8)

    with pytest.raises(ValueError) as exc:
        DemoBank.from_npz([str(narrow), str(wide)], feature_dim=4, num_actions=4,
                           device=torch.device("cpu"))

    msg = str(exc.value)
    assert str(narrow) in msg
    assert str(wide) in msg
    assert "4" in msg and "8" in msg


def test_from_npz_loads_valid_files(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    _write_obs_npz(a, n=5, feature_dim=4)
    _write_obs_npz(b, n=3, feature_dim=4)

    bank = DemoBank.from_npz([str(a), str(b)], feature_dim=4, num_actions=4,
                              device=torch.device("cpu"))

    assert bank.n == 8
    assert bank.obs.shape == (8, 4)
