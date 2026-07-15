"""Trainer-side wiring for the top-k curriculum advance gate.

The CurriculumManager only stores/filters; the trainer is what ranks the
generation's genomes by fitness and tags each episode with its top-k
membership before flushing. These tests exercise the real
`Trainer._record_curriculum_episodes` ranking path (via `__new__` + stamp,
the same pattern as test_freeze_pre_ppo_elite) so the attribution — clears
credited to top-k genomes, a lucky non-elite clear ignored — is pinned
without spinning up the emulator/network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.training.curriculum import CurriculumManager, CurriculumStage
from src.training.trainer import Trainer


def _stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(name="s1", levels=["1-1"], advance_threshold=0.8, min_episodes=10),
        CurriculumStage(name="s2", levels=["1-1", "1-2"], advance_threshold=0.8, min_episodes=10),
    ]


def _stub(cm: CurriculumManager) -> Trainer:
    t = Trainer.__new__(Trainer)
    t.curriculum = cm
    return t


def test_flush_credits_only_top_k_genomes() -> None:
    """Ranking is by mean fitness; only the top-k genomes' episodes enter the
    gate history, and a lower-fitness genome's lucky clear is excluded."""
    cm = CurriculumManager(stages=_stages(), top_k_gate=2)
    t = _stub(cm)
    # 5 genomes; top-2 by fitness are indices 0 and 2.
    pop = [SimpleNamespace(fitness=f) for f in (100.0, 5.0, 90.0, 3.0, 2.0)]
    # (level, success, global_genome_index). Index 1 (fitness 5) "clears" but
    # is NOT top-k, so it must be ignored; indices 0 and 2 are the elite.
    records = [
        ("1-1", True, 0),   # elite clear -> counts
        ("1-1", True, 1),   # non-elite lucky clear -> ignored
        ("1-1", True, 2),   # elite clear -> counts
        ("1-1", False, 3),  # non-elite fail -> ignored
        ("1-1", False, 4),  # non-elite fail -> ignored
    ]
    t._record_curriculum_episodes(pop, records)

    # Exactly the two elite episodes landed in the gate history.
    assert list(cm._stage_success_history["1-1"]) == [True, True]
    # Every episode still counted toward the min_episodes floor.
    assert cm.episodes_in_stage == 5


def test_flush_gate_moves_once_elite_sample_is_enough() -> None:
    """Across enough episodes the elite clears push stage_success_rate above 0
    (and the threshold), where the whole-population mean would stay stuck."""
    cm = CurriculumManager(stages=_stages(), top_k_gate=3)
    t = _stub(cm)
    pop = [SimpleNamespace(fitness=float(100 - i)) for i in range(30)]  # 0..2 elite
    # Six "generations" worth of a single-level population: the top-3 clear
    # every episode, everyone else fails. 6 gens * 3 elite = 18 elite clears.
    for _gen in range(6):
        records = [("1-1", i < 3, i) for i in range(30)]
        t._record_curriculum_episodes(pop, records)

    assert cm.stage_success_rate() == pytest.approx(1.0)
    assert cm.maybe_advance() is True
    assert cm.current_stage.name == "s2"


def test_flush_legacy_records_every_episode_in_order() -> None:
    """With no gate the flush records every episode, in the collected order,
    ignoring fitness entirely — identical to the old inline recording."""
    cm = CurriculumManager(stages=_stages())  # top_k_gate is None
    t = _stub(cm)
    pop = [SimpleNamespace(fitness=float(i)) for i in range(4)]
    records = [
        ("1-1", True, 0),
        ("1-1", False, 1),
        ("1-1", True, 2),
        ("1-1", False, 3),
    ]
    t._record_curriculum_episodes(pop, records)
    assert list(cm._stage_success_history["1-1"]) == [True, False, True, False]
    assert cm.episodes_in_stage == 4
