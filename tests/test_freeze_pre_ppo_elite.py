"""Pin behavior of the pre-PPO elite snapshot.

Diagnostic from the 2026-05-06 overnight run: SMB tile mode hit
`best_fitness = 7716.5` at gen 151 (cleared 1-1 once), then
collapsed to 1567.5 in gen 152 — a 5x regression in one generation.
The cause was PPO mutating `best.state_dict` in place: the GA's
elitism preserved the now-degraded post-PPO weights as the new top
genome, with no fallback to the original winning policy.

The fix snapshots `best.state_dict` BEFORE PPO runs and injects a
clone into the weakest non-best slot, so GA elitism keeps both the
post-PPO `best` (might be better, might be worse) and the pre-PPO
weights (the genuine generation-N winner). These tests verify the
two helpers behave as advertised:

* `_snapshot_pre_ppo_elite` returns a deep clone, not a view.
* `_inject_pre_ppo_snapshot` lands on the lowest-fitness slot,
  preserves the snapshot's fitness, doesn't mutate `best`.
* The freeze knob defaults to ON in tile mode; can be disabled.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.training.genetic_algorithm import Genome
from src.training.trainer import Trainer


def _genome(gid: int, fit: float, value: float = 1.0) -> Genome:
    """Tiny genome with a single 4-element state_dict tensor."""
    return Genome(
        genome_id=gid,
        state_dict={"w": torch.full((4,), value, dtype=torch.float32)},
        fitness=fit,
    )


def _stub_trainer(freeze: bool = True) -> Trainer:
    """Build a Trainer just enough to call the snapshot helpers.

    The helpers only read `self.freeze_pre_ppo_elite` and use stdlib
    + torch — no need to construct the heavyweight emulator/network
    state. We instantiate via `__new__` and stamp the one attribute.
    """
    t = Trainer.__new__(Trainer)
    t.freeze_pre_ppo_elite = freeze
    return t


def test_snapshot_returns_none_when_disabled() -> None:
    t = _stub_trainer(freeze=False)
    best = _genome(0, 7716.5)
    assert t._snapshot_pre_ppo_elite(best, elite_idx=[0]) is None


def test_snapshot_returns_none_when_no_elites() -> None:
    """If no genome has a non-trivial trajectory, PPO is skipped — and so
    is the snapshot, since there's nothing to protect."""
    t = _stub_trainer(freeze=True)
    best = _genome(0, 7716.5)
    assert t._snapshot_pre_ppo_elite(best, elite_idx=[]) is None


def test_snapshot_is_a_deep_clone() -> None:
    """The captured tensors must NOT alias `best.state_dict`. After PPO
    mutates the live state_dict in place, the snapshot's values must
    still equal the pre-PPO values."""
    t = _stub_trainer(freeze=True)
    best = _genome(0, 7716.5, value=1.0)
    snap = t._snapshot_pre_ppo_elite(best, elite_idx=[0])
    assert snap is not None
    snap_fit, snap_state = snap
    assert snap_fit == pytest.approx(7716.5)
    # Simulate PPO modifying best in place.
    best.state_dict["w"].fill_(99.0)
    # Snapshot must be untouched.
    assert torch.all(snap_state["w"] == 1.0)


def test_inject_lands_on_weakest_slot() -> None:
    """Injection writes to the genome with the LOWEST fitness, not the
    middle, not the head."""
    t = _stub_trainer(freeze=True)
    best = _genome(0, 7716.5, value=1.0)
    pop = [
        best,
        _genome(1, 1500.0, value=2.0),
        _genome(2, 100.0, value=3.0),  # weakest
        _genome(3, 800.0, value=4.0),
    ]
    snap = t._snapshot_pre_ppo_elite(best, elite_idx=[0])
    # Simulate PPO regressing `best`.
    best.state_dict["w"].fill_(99.0)
    t._inject_pre_ppo_snapshot(snap, pop, best)
    # Slot 2 is the weakest — its weights should now match the pre-PPO best.
    assert torch.all(pop[2].state_dict["w"] == 1.0)
    assert pop[2].fitness == pytest.approx(7716.5)
    # Other slots untouched.
    assert torch.all(pop[1].state_dict["w"] == 2.0)
    assert torch.all(pop[3].state_dict["w"] == 4.0)
    # `best` itself remains the post-PPO weights — we don't undo PPO.
    assert torch.all(pop[0].state_dict["w"] == 99.0)


def test_inject_skips_when_snapshot_is_none() -> None:
    t = _stub_trainer(freeze=False)
    best = _genome(0, 7716.5, value=1.0)
    pop = [best, _genome(1, 100.0, value=2.0)]
    t._inject_pre_ppo_snapshot(None, pop, best)
    # No mutation — all original values intact.
    assert torch.all(pop[1].state_dict["w"] == 2.0)
    assert pop[1].fitness == pytest.approx(100.0)


def test_inject_safe_when_best_is_only_genome() -> None:
    """Population of size 1: the only slot IS `best` and we must not
    overwrite it with itself."""
    t = _stub_trainer(freeze=True)
    best = _genome(0, 7716.5, value=1.0)
    pop = [best]
    snap = t._snapshot_pre_ppo_elite(best, elite_idx=[0])
    best.state_dict["w"].fill_(99.0)
    t._inject_pre_ppo_snapshot(snap, pop, best)
    # `best` keeps its post-PPO weights. The snapshot was discarded.
    assert torch.all(pop[0].state_dict["w"] == 99.0)


def test_full_cycle_preserves_winning_policy() -> None:
    """End-to-end: the post-PPO `best` is regressed, but the population
    after injection still contains a genome with the winning weights —
    so GA elitism on the next generation has a fallback."""
    t = _stub_trainer(freeze=True)
    best = _genome(0, 7716.5, value=1.0)  # the winning policy
    pop = [
        best,
        _genome(1, 1500.0, value=2.0),
        _genome(2, 800.0, value=3.0),
        _genome(3, 50.0, value=4.0),  # weakest, will receive snapshot
    ]
    snap = t._snapshot_pre_ppo_elite(best, elite_idx=[0])
    # PPO destroys the winning policy.
    best.state_dict["w"].fill_(0.0)

    t._inject_pre_ppo_snapshot(snap, pop, best)

    # Count genomes whose weights match the pre-PPO winner.
    matches_pre_ppo = [
        i for i, g in enumerate(pop)
        if torch.all(g.state_dict["w"] == 1.0)
    ]
    assert matches_pre_ppo == [3], (
        f"expected only slot 3 (weakest, recipient of snapshot) to hold "
        f"pre-PPO weights, got slots {matches_pre_ppo}"
    )
    # And that slot is now ranked as a top-fitness genome — GA elitism
    # will keep it across the next evolve() call.
    fitnesses_sorted_desc = sorted(pop, key=lambda g: g.fitness, reverse=True)
    top_two_ids = {fitnesses_sorted_desc[0].genome_id, fitnesses_sorted_desc[1].genome_id}
    assert 3 in top_two_ids, (
        f"snapshot slot should be in top-2 by fitness; ranking={[g.fitness for g in fitnesses_sorted_desc]}"
    )
