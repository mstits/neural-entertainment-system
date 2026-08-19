"""scripts/make_campaign_config.py — derivation, not cloning.

The defect this file exists to make impossible: a config pair where the
trainer's `states_dir` and the gates' `restart_states_dir` name different
levels. That shipped twice (2-1, and 1-4 on the highest banked rate) and
both times survived an entire campaign.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_campaign_config import (  # noqa: E402
    derived_paths, disarm_competence_floor, generate, missing_prerequisites,
    residual_references, substitute, tag, validate_level,
)


def test_trainer_and_gate_ladders_are_the_same_string_by_construction():
    """The 2-1 / 1-4 defect, made structurally impossible."""
    for level in ("3-1", "4-4", "8-2"):
        d = derived_paths(level)
        assert d["states_dir"] == d["restart_states_dir"]
        assert tag(level) in d["states_dir"]


def test_every_derived_path_names_this_level_and_no_other():
    d = derived_paths("5-3")
    for key, val in d.items():
        if key in ("name", "campaign_level"):
            continue
        assert "5_3" in val or "5-3" in val, (key, val)
        for foreign in ("2_1", "1_3", "1_4"):
            assert foreign not in val, (key, val)


def test_substitute_rewrites_both_spellings():
    text = "states_dir: checkpoints/online_2_1/x\ncampaign_level: \"2-1\"\n"
    out = substitute(text, "3-2")
    assert "online_3_2" in out and '"3-2"' in out
    assert "2_1" not in out and "2-1" not in out


def test_residual_references_catches_a_survivor():
    """A single missed line is the whole bug."""
    text = "states_dir: checkpoints/online_1_3/restart_states\nok: yes\n"
    hits = residual_references(text, "3-1", template="1-3")
    assert len(hits) == 1 and "online_1_3" in hits[0]


def test_residual_references_catches_a_stale_comment():
    """The 2-1 profile's comment told readers to use the 1-3 manifest."""
    text = "# read the deepest rung from checkpoints/online_1_3/manifest.json\n"
    assert residual_references(text, "3-1", template="1-3")


def test_generate_refuses_when_a_residual_would_survive(monkeypatch):
    monkeypatch.setattr("scripts.make_campaign_config.substitute",
                        lambda t, l, template=None: t)
    with pytest.raises(RuntimeError) as e:
        generate("3-1")
    assert "residual reference" in str(e.value)


def test_generated_pair_has_no_reference_to_the_template():
    files = generate("6-2")
    for path, text in files.items():
        assert not residual_references(text, "6-2"), path
        assert "6_2" in text


def test_competence_floor_ships_disarmed():
    """An inherited floor is tuned against another level's difficulty."""
    files = generate("6-2")
    campaign = files["configs/campaign_6_2.yaml"]
    assert "kill_probe_median_floor: 0.0" in campaign
    assert "DISARMED" in campaign


def test_disarm_is_idempotent():
    once = disarm_competence_floor("kill_probe_median_floor: 360.8  # x")
    assert disarm_competence_floor(once).count("kill_probe_median_floor") == 1


def test_bad_level_strings_are_rejected():
    for bad in ("9-1", "1-5", "one-one", "11", "", "2_1"):
        with pytest.raises(ValueError):
            validate_level(bad)


def test_generating_the_template_itself_is_refused():
    with pytest.raises(ValueError):
        generate("2-1")


def test_missing_prerequisites_are_reported_not_invented():
    owed = missing_prerequisites("7-4")
    assert any("solver tapes" in o for o in owed)
    assert any("restart ladder" in o for o in owed)
