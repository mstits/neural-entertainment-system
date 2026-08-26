"""Regression guard: run_ground_truth_test must isolate a per-run replay
failure instead of taking the whole self-test down with it.

History: the per-run loop already wraps two other failure modes in their
own `{"run": base, "error": ...}` entry (missing solution files, a
root-replay mismatch), but the replay itself -- env creation, load_state,
run_episode -- had no exception handling. A single stale/incompatible trace
(an action id outside ACTION_SPACE, a savestate load_state rejects, a
malformed .npy) crashed the whole process with an uncaught exception,
discarding every already-computed result for the runs processed before it
and skipping the receipt write entirely -- instead of the per-run isolation
the surrounding code clearly intends.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ROM = REPO / "roms/Super Mario Bros. (World).nes"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.clear_detect import ACTION_SPACE, FS, run_ground_truth_test  # noqa: E402
from scripts.go_explore_solve import SmbGame  # noqa: E402


def _write_run(tmp_path: Path, name: str, root_path: Path, actions: list[int],
               start_wd: tuple[int, int]) -> str:
    base = tmp_path / name
    base.with_suffix(".json").write_text(json.dumps({
        "root_state": str(root_path),
        "start_wd": list(start_wd),
        "clear_wd": None,
    }))
    np.save(str(base) + ".actions", np.array(actions, dtype=np.int64))
    return str(base)


@pytest.mark.skipif(not ROM.exists(), reason="SMB ROM not present")
def test_a_bad_action_id_in_one_run_does_not_discard_the_others(tmp_path) -> None:
    import nes_core

    game = SmbGame()
    env = nes_core.NESEnvironment(game.rom, frame_skip=1)
    env.reset()
    root_bytes = env.save_state()
    for _ in range(FS):   # rooting convention: one NOOP = FS raw frames
        env.step(0)
    ram0 = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
    start_wd = tuple(game.level_key(ram0))
    env.close()

    root_path = tmp_path / "root.state"
    root_path.write_bytes(root_bytes)

    good = _write_run(tmp_path, "good", root_path, [0], start_wd)
    # An action id outside ACTION_SPACE -- e.g. a trace recorded against a
    # larger action space than the one this script replays with -- is
    # exactly what `bitmasks[a]` in run_episode raises IndexError on.
    bad = _write_run(tmp_path, "bad", root_path, [len(ACTION_SPACE) + 5], start_wd)

    # Before the fix this raised IndexError out of run_ground_truth_test
    # itself, so the test would error here rather than reach an assertion.
    summary = run_ground_truth_test([good, bad], verbose=False)

    assert [r["run"] for r in summary["per_run"]] == [good, bad]

    good_result = summary["per_run"][0]
    assert "error" not in good_result
    assert good_result["n_actions"] == 1

    bad_result = summary["per_run"][1]
    assert "error" in bad_result

    assert summary["n_valid"] == 1
