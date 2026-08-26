"""Tests for scripts/mint_backward_states.py's clear-reached abort gate.

The gate exists so a hand-picked --start-state/--actions pair can never
mint a curriculum ladder that was never actually checked against a
recorded clear: scripts/run_online_campaign.py's preflight trusts
`reached_clear` in index.json to gate launching a multi-hour unattended
campaign. No real Pool/ROM is needed here — `machine_from_profile`,
`load_root` and `mint` are the only calls that would touch the
emulator, and a scripted fake stands in for each.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.mint_backward_states as mbs


def _stub_machine(*_a, **_k):
    return "fake.nes", 4, [0, 1], ()


def _stub_load_root(*_a, **_k):
    return b"\x00" * 8


def _stub_mint(**_k):
    # A tape that ends somewhere — the point is nothing here should be
    # trusted as a verified clear when there is no recorded clear_wd.
    return [], {"end_wd": [9, 9], "end_area": 0, "end_gx": 0}


def _run_main(tmp_path, monkeypatch, run_dir):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("rom_path: fake.nes\n")
    start_state = tmp_path / "root.state"
    start_state.write_bytes(b"\x00" * 8)
    actions_path = tmp_path / "tape.actions.npy"
    np.save(actions_path, np.zeros(4, dtype=np.int64))
    out_dir = tmp_path / "out"

    monkeypatch.setattr(mbs, "machine_from_profile", _stub_machine)
    monkeypatch.setattr(mbs, "load_root", _stub_load_root)
    monkeypatch.setattr(mbs, "mint", _stub_mint)
    monkeypatch.setattr(
        "sys.argv",
        ["mint_backward_states.py", "--level", "9-9",
         "--run", str(run_dir), "--start-state", str(start_state),
         "--actions", str(actions_path), "--profile", str(profile_path),
         "--out", str(out_dir)])
    return out_dir


def test_missing_sidecar_aborts_instead_of_silently_verifying(
    tmp_path, monkeypatch,
):
    """No sidecar under --run for this level -> no clear_wd to compare
    against -> the gate must fail closed, not default to `cleared=True`."""
    run_dir = tmp_path / "run_with_no_9-9_solution"
    out_dir = _run_main(tmp_path, monkeypatch, run_dir)

    with pytest.raises(SystemExit):
        mbs.main()

    assert not (out_dir / "index.json").exists()


def test_recorded_clear_wd_mismatch_still_aborts(tmp_path, monkeypatch):
    """A sidecar that DOES exist but disagrees with the tape's actual end
    must keep aborting (the pre-existing, correctly-gated case)."""
    run_dir = tmp_path / "run"
    sidecar = run_dir / "lvl_9-9" / "solutions" / "sol_000.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"clear_wd": [1, 1], "start_wd": [1, 0]}))
    out_dir = _run_main(tmp_path, monkeypatch, run_dir)

    with pytest.raises(SystemExit):
        mbs.main()

    assert not (out_dir / "index.json").exists()
