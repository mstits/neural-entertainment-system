"""
Curriculum manager.

This is what solves the "every level has to be relearned" problem. Instead of
throwing the agent at the whole game from the start, we start on the first
level, and as the population masters it (success rate above a threshold), we
gradually introduce later levels.

The key insight from "Robust RL via Genetic Curriculum" (arXiv 2202.08393):
the curriculum is itself something that can be adapted. If the agent is
struggling on stage 2, we can go back to mixing in more stage 1 training.
If it's breezing through, we advance faster.

This class is game-agnostic. Level identifiers are opaque strings read from
the game's YAML profile. The reward function is responsible for detecting
"which level am I currently on" from RAM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CurriculumStage:
    """One stage in the curriculum."""
    name: str
    levels: list[str]           # levels active in this stage
    advance_threshold: float    # success rate required to move to next stage
    min_episodes: int = 50      # minimum episodes before we consider advancing
    # Optional per-stage start-state file. When set, the pool spawns workers
    # with this state instead of the globally-configured start_state_path.
    # Lets the curriculum actually *route* the agent into different parts
    # of the game, not just nominally name them.
    start_state_path: str | None = None


class CurriculumManager:
    """Manages staged introduction of game levels."""

    def __init__(
        self,
        stages: list[CurriculumStage],
        regression_threshold: float = 0.3,
        history_window: int = 200,
    ) -> None:
        """
        Args:
            stages: ordered list of curriculum stages
            regression_threshold: if success rate drops below this, go back a stage
            history_window: how many recent episodes to track per level
        """
        if not stages:
            raise ValueError("At least one curriculum stage is required")

        self.stages = stages
        self.regression_threshold = regression_threshold
        self.history_window = history_window

        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        # Cross-run success history, keyed by level. Preserved across stage
        # transitions so sample_level() can weight by long-term
        # difficulty.
        self._success_history: dict[str, deque[bool]] = {}
        # Within-stage success history. Cleared on every advance/regress so
        # stage_success_rate reflects ONLY episodes recorded during the
        # current stage. Without this, a "*" wildcard in the stage config
        # expanded to every level ever recorded (including prior stages),
        # which meant the rate could never fall below prior stages' rates
        # — effectively pinning the curriculum at the first stage that
        # ever earned a high success rate.
        self._stage_success_history: dict[str, deque[bool]] = {}
        self._init_history_for_stage()

    def _init_history_for_stage(self) -> None:
        """Ensure every level in the current stage has a history deque.

        Also resets the within-stage history so stage_success_rate()
        only considers episodes recorded under the current stage.
        """
        self._stage_success_history = {}
        for level in self.current_stage.levels:
            if level not in self._success_history:
                self._success_history[level] = deque(maxlen=self.history_window)
            # Seed the stage history too (wildcard stages use dynamic keys).
            if level != "*":
                self._stage_success_history[level] = deque(maxlen=self.history_window)

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    def sample_level(self) -> str:
        """
        Return a level to play next. Biases toward levels with lower success
        rates (we want to train on what we're bad at).
        """
        import random

        active_levels = self.current_stage.levels

        # Weight each level inversely by its recent success rate
        weights = []
        for level in active_levels:
            history = self._success_history.get(level, deque())
            if len(history) < 10:
                # Not enough data — weight equally
                weights.append(1.0)
            else:
                success_rate = sum(history) / len(history)
                # Harder levels (lower success) get more weight
                weights.append(max(0.1, 1.0 - success_rate))

        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for level, w in zip(active_levels, weights):
            cumulative += w
            if r <= cumulative:
                return level
        return active_levels[-1]  # floating point edge case

    def record_episode(self, level: str, success: bool) -> None:
        """Record whether an episode succeeded (completed the level).

        Appends to BOTH the cross-run history (used by sample_level) and
        the within-stage history (used by stage_success_rate). Tracking
        them separately means stage advancement is governed only by
        episodes that happened in the current stage, while difficulty-
        weighted level sampling can still consult cumulative experience.
        """
        if level not in self._success_history:
            self._success_history[level] = deque(maxlen=self.history_window)
        self._success_history[level].append(success)
        if level not in self._stage_success_history:
            self._stage_success_history[level] = deque(maxlen=self.history_window)
        self._stage_success_history[level].append(success)
        self.episodes_in_stage += 1

    def stage_success_rate(self) -> float:
        """Average success rate across all levels in the current stage.

        Reads from `_stage_success_history` (within-stage only) so a
        wildcard "*" stage expansion only counts episodes since this
        stage began. Otherwise prior stages' history would dominate
        and prevent further advancement.

        "*" matches every level the reward function has reported during
        the CURRENT stage — useful for games where the reward fn emits
        concrete RAM-derived level names that don't line up with the
        stage's abstract labels.
        """
        rates = []
        levels_to_check: list[str] = []
        for level in self.current_stage.levels:
            if level == "*":
                levels_to_check.extend(self._stage_success_history.keys())
            else:
                levels_to_check.append(level)
        for level in levels_to_check:
            history = self._stage_success_history.get(level, deque())
            if len(history) >= 10:
                rates.append(sum(history) / len(history))
        return sum(rates) / len(rates) if rates else 0.0

    def maybe_advance(self) -> bool:
        """
        Check if we should advance to the next stage. Call this periodically.

        Returns:
            True if we advanced, False otherwise.
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # already at final stage

        if self.episodes_in_stage < self.current_stage.min_episodes:
            return False

        if self.stage_success_rate() >= self.current_stage.advance_threshold:
            self.current_stage_idx += 1
            self.episodes_in_stage = 0
            self._init_history_for_stage()
            return True

        return False

    def maybe_regress(self) -> bool:
        """If performance craters, go back to an earlier stage to rebuild skills."""
        if self.current_stage_idx == 0:
            return False

        if self.episodes_in_stage < self.current_stage.min_episodes:
            return False

        if self.stage_success_rate() < self.regression_threshold:
            self.current_stage_idx -= 1
            self.episodes_in_stage = 0
            # Same invariant as maybe_advance: clear within-stage history
            # so the regressed stage's success rate is computed only from
            # episodes recorded after re-entering it. Without this, a
            # second-attempt regress threshold check would still see the
            # FIRST attempt's history mixed in.
            self._init_history_for_stage()
            return True

        return False

    def state_dict(self) -> dict:
        """For checkpointing."""
        return {
            "current_stage_idx": self.current_stage_idx,
            "episodes_in_stage": self.episodes_in_stage,
            "success_history": {
                lvl: list(h) for lvl, h in self._success_history.items()
            },
            "stage_success_history": {
                lvl: list(h) for lvl, h in self._stage_success_history.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_stage_idx = state["current_stage_idx"]
        self.episodes_in_stage = state["episodes_in_stage"]
        self._success_history = {
            lvl: deque(h, maxlen=self.history_window)
            for lvl, h in state["success_history"].items()
        }
        # Stage history was added later — fall back to empty if loading
        # an older checkpoint.
        self._stage_success_history = {
            lvl: deque(h, maxlen=self.history_window)
            for lvl, h in state.get("stage_success_history", {}).items()
        }
