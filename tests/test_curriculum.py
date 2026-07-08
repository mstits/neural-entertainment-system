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
    import pytest

    cm = CurriculumManager(stages=_stages())
    for _ in range(10):
        cm.record_episode("1-1", True)
    assert cm.stage_success_rate() == 1.0

    for _ in range(10):
        cm.record_episode("1-1", False)
    # 10 successes + 10 failures in the window → exactly 0.5. The old
    # `0.0 < x <= 0.5` was overly permissive — it passed even if the
    # sliding-window math degenerated to ~0.0.
    assert cm.stage_success_rate() == pytest.approx(0.5)


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


def test_under_sampled_concrete_level_blocks_advance() -> None:
    """A stage must not advance on a hard concrete level it has barely
    sampled — under-sampled required levels count as 0.0, not skipped.
    (Audit F19.)"""
    stages = [
        CurriculumStage(
            name="s1", levels=["easy", "hard"],
            advance_threshold=0.8, min_episodes=10,
        ),
        CurriculumStage(
            name="s2", levels=["done"],
            advance_threshold=0.8, min_episodes=10,
        ),
    ]
    cm = CurriculumManager(stages=stages)
    for _ in range(15):
        cm.record_episode("easy", True)   # well-sampled, 100%
    for _ in range(3):
        cm.record_episode("hard", True)   # under-sampled (<10), would-be 100%
    # 'hard' counts as 0.0 (not skipped) -> (1.0 + 0.0) / 2 = 0.5 < 0.8.
    assert cm.stage_success_rate() == pytest.approx(0.5)
    assert cm.maybe_advance() is False
    for _ in range(10):
        cm.record_episode("hard", True)   # now well-sampled
    assert cm.stage_success_rate() == pytest.approx(1.0)
    assert cm.maybe_advance() is True


def test_strong_under_sampled_agent_does_not_regress() -> None:
    """F19 regress fix: a strong agent whose concrete levels are merely
    UNDER-sampled must NOT be dragged to 0.0 and regressed spuriously (the
    0.0-injection is advance-only)."""
    stages = [
        CurriculumStage(name="s1", levels=["a"], advance_threshold=0.8, min_episodes=5),
        CurriculumStage(
            name="s2", levels=["hard1", "hard2"],
            advance_threshold=0.8, min_episodes=5,
        ),
    ]
    cm = CurriculumManager(stages=stages, regression_threshold=0.3)
    # Force to stage 2.
    for _ in range(10):
        cm.record_episode("a", True)
    assert cm.maybe_advance() is True
    # On stage 2: perfect but under-sampled (< 10 episodes each).
    for _ in range(4):
        cm.record_episode("hard1", True)
        cm.record_episode("hard2", True)
    # Advance gate treats under-sampled as 0.0 -> won't advance yet.
    assert cm.stage_success_rate() == pytest.approx(0.0)
    # Regress gate skips under-sampled -> sees 1.0 -> must NOT regress.
    assert cm.stage_success_rate(undersampled_as_zero=False) == pytest.approx(1.0)
    assert cm.maybe_regress() is False
