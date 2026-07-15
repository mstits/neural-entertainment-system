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


def test_top_k_gate_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        CurriculumManager(stages=_stages(), top_k_gate=0)
    with pytest.raises(ValueError):
        CurriculumManager(stages=_stages(), top_k_gate=-1)


def test_top_k_gate_measures_only_elite_episodes() -> None:
    """The structural fix (validation 2026-07-12): a mutated GA population
    can never push the whole-population success mean above the 0.8 advance
    threshold, but the top-k elite clears reliably. Gating on the top-k must
    surface that mastery signal while the legacy gate stays stuck near 0."""
    legacy = CurriculumManager(stages=_stages())          # whole-population
    gated = CurriculumManager(stages=_stages(), top_k_gate=5)

    # One generation of a 60-genome population: ranks 0..4 are the elite and
    # clear both of their episodes; the other 55 always fail. Interleaved the
    # way the trainer flushes them (episode-major, then genome).
    for _ep in range(2):
        for genome_rank in range(60):
            is_top_k = genome_rank < 5
            success = is_top_k  # only the elite clears 1-1
            legacy.record_episode("1-1", success)
            gated.record_episode("1-1", success, top_k_eligible=is_top_k)

    # Both managers saw all 120 episodes (min_episodes floor is total play).
    assert legacy.episodes_in_stage == 120
    assert gated.episodes_in_stage == 120

    # Legacy: 10 clears / 120 episodes ~= 0.083 — structurally below 0.8.
    assert legacy.stage_success_rate() < 0.2
    assert legacy.maybe_advance() is False
    assert legacy.current_stage.name == "s1"

    # Gated: only the 10 elite episodes entered the stage window, all clears.
    assert gated.stage_success_rate() == pytest.approx(1.0)
    assert gated.maybe_advance() is True
    assert gated.current_stage.name == "s2"


def test_top_k_gate_backward_compatible() -> None:
    """With top_k_gate unset, the top_k_eligible flag is ignored and the
    manager behaves byte-identically to a plain record_episode caller."""
    plain = CurriculumManager(stages=_stages())     # never passes the flag
    flagged = CurriculumManager(stages=_stages())   # passes an ignored flag

    import random
    rng = random.Random(1234)
    for _ in range(80):
        success = rng.random() < 0.4
        # A gate-off manager must ignore whatever eligibility is passed.
        eligible = rng.random() < 0.3
        plain.record_episode("1-1", success)
        flagged.record_episode("1-1", success, top_k_eligible=eligible)

    assert flagged.episodes_in_stage == plain.episodes_in_stage
    assert flagged.stage_success_rate() == pytest.approx(plain.stage_success_rate())
    assert list(flagged._stage_success_history["1-1"]) == list(
        plain._stage_success_history["1-1"]
    )


def test_top_k_gate_window_bounds_elite_history() -> None:
    """The rolling window bounds the number of elite episodes considered, and
    non-elite episodes never occupy a window slot when gating is active."""
    cm = CurriculumManager(
        stages=_stages(), history_window=30, top_k_gate=3
    )
    # 200 episodes: 3-in-10 are elite (60 elite episodes total), all clears;
    # the rest fail. Only the elite ones enter the stage window.
    for i in range(200):
        is_top_k = (i % 10) < 3
        cm.record_episode("1-1", is_top_k, top_k_eligible=is_top_k)

    stage_hist = cm._stage_success_history["1-1"]
    # Window is capped at history_window and holds ONLY elite (all-True) recs.
    assert len(stage_hist) == 30
    assert all(stage_hist)
    assert cm.stage_success_rate() == pytest.approx(1.0)
    # Every episode still counted toward the min_episodes floor.
    assert cm.episodes_in_stage == 200


def test_top_k_gate_undersampled_level_blocks_advance() -> None:
    """Under the gate, a required level the elite has barely played counts as
    under-sampled (0.0 for advance), same as the whole-population gate."""
    stages = [
        CurriculumStage(
            name="s1", levels=["easy", "hard"],
            advance_threshold=0.8, min_episodes=10,
        ),
        CurriculumStage(name="s2", levels=["done"], advance_threshold=0.8, min_episodes=10),
    ]
    cm = CurriculumManager(stages=stages, top_k_gate=3)
    # Elite clears 'easy' 15 times (well-sampled) but only 3 elite plays of
    # 'hard' so far. Many non-elite plays of both are ignored by the gate.
    for _ in range(15):
        cm.record_episode("easy", True, top_k_eligible=True)
        cm.record_episode("easy", False, top_k_eligible=False)  # ignored
    for _ in range(3):
        cm.record_episode("hard", True, top_k_eligible=True)
        cm.record_episode("hard", False, top_k_eligible=False)  # ignored
    # 'hard' has <10 elite episodes -> counts as 0.0 -> (1.0 + 0.0)/2 = 0.5.
    assert cm.stage_success_rate() == pytest.approx(0.5)
    assert cm.maybe_advance() is False
    # Once the elite has cleared 'hard' enough, the gate opens.
    for _ in range(10):
        cm.record_episode("hard", True, top_k_eligible=True)
    assert cm.stage_success_rate() == pytest.approx(1.0)
    assert cm.maybe_advance() is True


def test_top_k_gate_regress_uses_elite_rate() -> None:
    """Regression is also driven by the elite rate under the gate: when the
    top-k stops clearing, the stage regresses even though non-elite genomes
    were failing all along (their episodes never counted)."""
    stages = [
        CurriculumStage(name="s1", levels=["1-1"], advance_threshold=0.8, min_episodes=10),
        CurriculumStage(name="s2", levels=["1-1", "1-2"], advance_threshold=0.8, min_episodes=10),
    ]
    cm = CurriculumManager(stages=stages, regression_threshold=0.3, top_k_gate=3)
    # Advance on elite mastery of 1-1.
    for _ in range(12):
        cm.record_episode("1-1", True, top_k_eligible=True)
        cm.record_episode("1-1", False, top_k_eligible=False)
    assert cm.maybe_advance() is True
    assert cm.current_stage.name == "s2"
    # In s2 the elite bombs both levels; non-elite noise is ignored.
    for _ in range(12):
        cm.record_episode("1-1", False, top_k_eligible=True)
        cm.record_episode("1-2", False, top_k_eligible=True)
        cm.record_episode("1-1", True, top_k_eligible=False)   # ignored
    assert cm.maybe_regress() is True
    assert cm.current_stage.name == "s1"


def test_top_k_gate_state_dict_round_trip() -> None:
    """Serialization is unchanged by the gate: the stage history is still a
    plain per-level success deque, so checkpoints round-trip cleanly."""
    cm = CurriculumManager(stages=_stages(), top_k_gate=5)
    for _ in range(20):
        cm.record_episode("1-1", True, top_k_eligible=True)
        cm.record_episode("1-1", False, top_k_eligible=False)
    state = cm.state_dict()
    cm2 = CurriculumManager(stages=_stages(), top_k_gate=5)
    cm2.load_state_dict(state)
    assert cm2.episodes_in_stage == cm.episodes_in_stage
    assert cm2.stage_success_rate() == pytest.approx(cm.stage_success_rate())


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
