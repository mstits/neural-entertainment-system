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

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


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
        top_k_gate: int | None = None,
    ) -> None:
        """
        Args:
            stages: ordered list of curriculum stages
            regression_threshold: if success rate drops below this, go back a stage
            history_window: how many recent episodes to track per level
            top_k_gate: when set to k, the advance/regress gate measures only
                episodes recorded by the k highest-fitness genomes of each
                generation (see record_episode's top_k_eligible). This exists
                for the GA path (ga_ppo), where the population is mostly junk
                mutations: a whole-population success mean can never reach the
                0.8 advance threshold even with a perfect elite (validation
                2026-07-12: a 60-genome population maxed 0.005 over 300 gens),
                so the gate is structurally unreachable. Gating on the top-k
                mirrors how the vanilla trainer gates on its single deployed
                policy's clear fraction rather than a population average.
                When None (default) EVERY episode counts — byte-identical to
                the pre-gate behavior, so non-GA callers are unaffected.
        """
        if not stages:
            raise ValueError("At least one curriculum stage is required")
        if top_k_gate is not None and top_k_gate < 1:
            raise ValueError(
                f"top_k_gate must be a positive integer or None, got {top_k_gate}"
            )

        self.stages = stages
        self.regression_threshold = regression_threshold
        self.history_window = history_window
        self.top_k_gate = top_k_gate

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

    def record_episode(
        self, level: str, success: bool, top_k_eligible: bool | None = None
    ) -> None:
        """Record whether an episode succeeded (completed the level).

        Appends to BOTH the cross-run history (used by sample_level) and
        the within-stage history (used by stage_success_rate). Tracking
        them separately means stage advancement is governed only by
        episodes that happened in the current stage, while difficulty-
        weighted level sampling can still consult cumulative experience.

        `top_k_eligible` is only consulted when this manager was built with
        `top_k_gate` set (the ga_ppo path). It marks whether the recording
        genome landed in this generation's top-k by fitness. When gating is
        active, ONLY top-k episodes enter the within-stage history, so the
        advance/regress gate measures "can the best policies this stage
        produces clear it reliably" instead of a whole-population mean the
        mutated GA population can never push above ~0. The cross-run history
        (difficulty-weighted sampling) always sees every episode regardless
        — the sampler should weight by the whole population's experience, not
        just the elite's. When gating is OFF (`top_k_gate is None`) the flag
        is ignored and every episode is recorded, byte-identical to the
        pre-gate behavior.
        """
        if level not in self._success_history:
            self._success_history[level] = deque(maxlen=self.history_window)
        self._success_history[level].append(success)
        # Within-stage (gate) history: all episodes when gating is off; only
        # top-k episodes when it's on. `episodes_in_stage` still counts every
        # episode below, so the min_episodes floor reflects total play, not
        # just elite play.
        record_to_stage = self.top_k_gate is None or bool(top_k_eligible)
        if record_to_stage:
            if level not in self._stage_success_history:
                self._stage_success_history[level] = deque(maxlen=self.history_window)
            self._stage_success_history[level].append(success)
        self.episodes_in_stage += 1

    def stage_success_rate(self, undersampled_as_zero: bool = True) -> float:
        """Average success rate across all levels in the current stage.

        Reads from `_stage_success_history` (within-stage only) so a
        wildcard "*" stage expansion only counts episodes since this
        stage began. Otherwise prior stages' history would dominate
        and prevent further advancement.

        When the manager was built with `top_k_gate`, that history was
        already filtered to top-k episodes at record time (see
        record_episode), so this method needs no top-k logic of its own —
        it just averages whatever the gate admitted.

        "*" matches every level the reward function has reported during
        the CURRENT stage — useful for games where the reward fn emits
        concrete RAM-derived level names that don't line up with the
        stage's abstract labels.

        `undersampled_as_zero` controls how an explicitly-required concrete
        level with < 10 within-stage episodes is scored. For the ADVANCE gate
        it is True (an under-sampled hard level counts as 0.0, so the stage
        can't advance on levels it hasn't really played). For the REGRESS gate
        it MUST be False (skip under-sampled levels): otherwise a strong agent
        whose concrete levels are merely under-sampled scores 0.0 and regresses
        spuriously — thrashing between stages and never reaching later worlds.
        """
        rates = []
        concrete_levels: list[str] = []
        wildcard_levels: list[str] = []
        for level in self.current_stage.levels:
            if level == "*":
                wildcard_levels.extend(self._stage_success_history.keys())
            else:
                concrete_levels.append(level)
        # Dedup, preserving order; a concretely-listed level takes precedence
        # over the same label surfaced via "*" so it's never counted twice.
        concrete_levels = list(dict.fromkeys(concrete_levels))
        wildcard_only = [
            lv for lv in dict.fromkeys(wildcard_levels)
            if lv not in concrete_levels
        ]
        for level in concrete_levels:
            history = self._stage_success_history.get(level, deque())
            if len(history) >= 10:
                rates.append(sum(history) / len(history))
            elif undersampled_as_zero:
                # ADVANCE gate: an under-sampled required level counts as 0.0
                # (don't advance on a hard level played only 0-for-5). For the
                # REGRESS gate this is skipped, so a strong-but-under-sampled
                # agent isn't dragged to 0.0 and regressed spuriously.
                rates.append(0.0)
        for level in wildcard_only:
            # Incidental RAM-derived labels surfaced by "*": count only once
            # sampled, so a level that just appeared doesn't spuriously block.
            history = self._stage_success_history.get(level, deque())
            if len(history) >= 10:
                rates.append(sum(history) / len(history))
        if rates:
            return sum(rates) / len(rates)
        # No level has enough data. Return the "don't transition" default for
        # each gate: 0.0 blocks ADVANCE (need evidence to advance), 1.0 blocks
        # REGRESS (no evidence of failure -> don't drop back). Without the
        # mode-aware default, an all-under-sampled stage returned 0.0 and
        # regressed a strong agent anyway.
        return 0.0 if undersampled_as_zero else 1.0

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

        # Regress only on genuinely poor performance on SAMPLED levels — do
        # NOT treat under-sampled levels as 0.0 here (that would regress a
        # strong agent whose levels are merely under-sampled; see F19).
        if self.stage_success_rate(undersampled_as_zero=False) < self.regression_threshold:
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


class Curriculum:
    """SMB save-state curriculum owner for the vanilla-PPO conductor.

    This is the fourth strangler step of
    `docs/proposals/trainer_decomposition_plan.md` (Task 4). It is the home
    for the curriculum LIFECYCLE that used to be inlined in
    `Trainer._run_vanilla_ppo` — beginning with the disk-load seam
    (`load_from_disk`), the one part of the curriculum that is a genuine leaf:
    it reads the `stage_NN.state`/`.meta.json` file-set and returns the
    `(states, anchors, stage)` triple with zero coupling to the rollout loop.

    The controller holds a reference back to the owning `Trainer` and reads
    the trainer-owned curriculum config through `self.trainer`
    (`checkpoint_dir`, `_smb_curriculum_active`, `_inherit_curriculum_on_fresh`,
    `pool`) so the loop sees exactly the state it expects. Every read, side
    effect, and log message is preserved verbatim, just relocated.

    NOT owned here yet (reported as the inseparable Task-5 seam — see the
    Task-4 handoff): the mid-rollout capture DECISION, the advance detection +
    stage-file write, the cold-eval/forgetting/consolidate triggers, the
    coefficient schedules, and the warm-start ASSIGNMENT. Those read/write the
    same loop locals the `RolloutCollector` (Task 5) mechanics produce and
    mutate (`env_stage`, `stage_seed_results`, the collector trackers, the
    per-env load-failure fallback, the still-in-conductor Go-Explore burst /
    CGSA state), so they cannot move without moving Task-5 collector code or
    changing behavior. They stay in the conductor until Task 5 extracts the
    surrounding seams.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def load_from_disk(
        self, *, fresh_start: bool = False,
    ) -> tuple[list, list, int]:
        """Scan `checkpoint_dir/smb_curriculum` for `stage_NN.state` files
        and return `(states, anchors, stage)` for the SMB save-state
        curriculum.

        `states`/`anchors` seed with the stage-0 cold-boot entry
        (`[None]`/`[0]`); each successfully loaded stage appends its blob +
        anchor byte and bumps `stage`. Anchor bytes come from the sidecar
        `stage_NN.meta.json`; a legacy state without one is peeked via
        worker 0 and the sidecar backfilled.

        `fresh_start=True` (GUI "Resume" unticked / headless `--no-resume`)
        IGNORES any saved stage states and returns stage 0 — a fresh run
        trains from 1-1 rather than silently inheriting a prior run's mid-
        game curriculum (a validated confusion where a "fresh" run resumed
        at stage 2). The `stage_NN.state` files are NEVER deleted or moved
        (a paused experiment must stay resumable); they are only skipped.
        Set `reinforce.inherit_curriculum_on_fresh: true` to opt back into
        the historical inherit-on-fresh behavior. Resume (non-fresh) runs
        always inherit the saved curriculum.
        """
        smb_curriculum_anchors: list[int] = [0]  # area-byte for each stage
        smb_curriculum_states: list[Optional[bytes]] = [None]  # state per stage
        smb_curriculum_stage = 0  # index into anchors

        smb_curriculum_dir = self.trainer.checkpoint_dir / "smb_curriculum"
        smb_curriculum_dir.mkdir(parents=True, exist_ok=True)

        if fresh_start and not getattr(
            self.trainer, "_inherit_curriculum_on_fresh", False
        ):
            # A fresh run ignores saved curriculum state. Surface exactly
            # what was found and how to opt in, but leave every file on disk
            # untouched so a paused run stays resumable.
            existing = sorted(smb_curriculum_dir.glob("stage_*.state"))
            if existing and self.trainer._smb_curriculum_active:
                log.info(
                    "[vanilla_ppo] fresh start requested — ignoring %d saved "
                    "curriculum stage state(s) in %s (files left untouched); "
                    "starting curriculum at stage 0. Set "
                    "reinforce.inherit_curriculum_on_fresh: true to resume "
                    "the saved curriculum on a fresh run.",
                    len(existing), smb_curriculum_dir,
                )
            return (
                smb_curriculum_states,
                smb_curriculum_anchors,
                smb_curriculum_stage,
            )

        # Non-Mario games skip the SMB area-byte curriculum entirely
        # (stage stays 0 -> no warm-start, no capture, no advance): plain
        # single-stage vanilla PPO from the profile start state.
        _curr_range = range(1, 32) if self.trainer._smb_curriculum_active else range(0)
        for n in _curr_range:
            stage_path = smb_curriculum_dir / f"stage_{n:02d}.state"
            meta_path = smb_curriculum_dir / f"stage_{n:02d}.meta.json"
            if not stage_path.exists():
                break
            try:
                blob = stage_path.read_bytes()
                # Read the anchor byte from the sidecar metadata. If
                # absent (state written before this fix), fall back to
                # reading ram[$0760] from the save state by briefly
                # loading it into worker 0. Pool exists at this point.
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    anchor = int(meta["anchor"])
                else:
                    # Legacy state without sidecar: derive anchor by
                    # peeking at the saved RAM. Load into worker 0,
                    # read byte, then re-reset so we don't pollute the
                    # cold-boot initial state for the actual run.
                    self.trainer.pool.load_worker_state(0, blob)
                    peek = self.trainer.pool.step_all(
                        np.zeros(self.trainer.pool.num_workers, dtype=np.uint8)
                    )
                    ram = peek[0][2]
                    anchor = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
                    log.info(
                        "[vanilla_ppo] derived anchor=%d for legacy "
                        "stage_%02d.state (no sidecar); writing it now.",
                        anchor, n,
                    )
                    meta_path.write_text(json.dumps({"anchor": anchor}))
                    # Reset before continuing setup.
                    self.trainer.pool.reset_all()
                smb_curriculum_states.append(blob)
                smb_curriculum_anchors.append(anchor)
                smb_curriculum_stage = n
            except Exception as e:
                log.warning(
                    "[vanilla_ppo] failed to load stage_%02d: %s", n, e,
                )
                break
        return (
            smb_curriculum_states,
            smb_curriculum_anchors,
            smb_curriculum_stage,
        )
