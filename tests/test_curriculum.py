"""Tests for the curriculum manager."""

from __future__ import annotations

import pytest

from src.training.curriculum import CurriculumManager, CurriculumStage


def _stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(name="s1", levels=["1-1"], advance_threshold=0.8, min_episodes=10),
        CurriculumStage(
            name="s2", levels=["1-1", "1-2"], advance_threshold=0.8, min_episodes=10
        ),
        CurriculumStage(
            name="s3",
            levels=["1-1", "1-2", "1-3"],
            advance_threshold=0.8,
            min_episodes=10,
        ),
    ]


def test_requires_nonempty_stage_list() -> None:
    with pytest.raises(ValueError):
        CurriculumManager(stages=[])


def test_sample_level_returns_active_level() -> None:
    cm = CurriculumManager(stages=_stages())
    for _ in range(50):
        assert cm.sample_level() in ["1-1"]


def test_record_and_stage_success_rate() -> None:
    cm = CurriculumManager(stages=_stages())
    for _ in range(10):
        cm.record_episode("1-1", True)
    assert cm.stage_success_rate() == 1.0

    for _ in range(10):
        cm.record_episode("1-1", False)
    # half-and-half window
    assert 0.0 < cm.stage_success_rate() <= 0.5


def test_advance_when_threshold_met() -> None:
    cm = CurriculumManager(stages=_stages())
    for _ in range(20):
        cm.record_episode("1-1", True)
    assert cm.maybe_advance() is True
    assert cm.current_stage.name == "s2"


def test_does_not_advance_before_min_episodes() -> None:
    cm = CurriculumManager(stages=_stages())
    for _ in range(5):
        cm.record_episode("1-1", True)
    assert cm.maybe_advance() is False
    assert cm.current_stage.name == "s1"


def test_regression_when_success_drops() -> None:
    stages = _stages()
    cm = CurriculumManager(stages=stages, regression_threshold=0.3)
    # Get to stage 2
    for _ in range(20):
        cm.record_episode("1-1", True)
    assert cm.maybe_advance()
    assert cm.current_stage.name == "s2"

    # Bomb out in stage 2
    for _ in range(15):
        cm.record_episode("1-1", False)
        cm.record_episode("1-2", False)
    assert cm.maybe_regress() is True
    assert cm.current_stage.name == "s1"


def test_state_dict_round_trip() -> None:
    cm = CurriculumManager(stages=_stages())
    for _ in range(20):
        cm.record_episode("1-1", True)
    cm.maybe_advance()
    cm.record_episode("1-2", True)

    state = cm.state_dict()
    cm2 = CurriculumManager(stages=_stages())
    cm2.load_state_dict(state)

    assert cm2.current_stage.name == cm.current_stage.name
    assert cm2.episodes_in_stage == cm.episodes_in_stage
    assert cm2.stage_success_rate() == pytest.approx(cm.stage_success_rate())


def test_sample_biases_toward_harder_levels() -> None:
    cm = CurriculumManager(stages=_stages())
    for _ in range(20):
        cm.record_episode("1-1", True)
    cm.maybe_advance()
    # Now in s2 = ["1-1", "1-2"]. Make 1-1 easy (always success),
    # 1-2 hard (always fail). Sampler should pick 1-2 more often.
    for _ in range(20):
        cm.record_episode("1-1", True)
    for _ in range(20):
        cm.record_episode("1-2", False)

    counts = {"1-1": 0, "1-2": 0}
    for _ in range(500):
        counts[cm.sample_level()] += 1
    assert counts["1-2"] > counts["1-1"]
