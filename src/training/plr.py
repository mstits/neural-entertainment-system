"""Prioritized Level Replay (PLR) context for the generalist vanilla-PPO path.

This is the go-forward generalist mechanism: instead of a per-level specialist
or a forward-welding ladder, ONE policy trains on a FIXED set of level
ENTRANCES sampled by inverse-recent-success weight (train hardest on what
we're worst at — Jiang et al. 2021's core idea, with a failure-rate proxy for
the learning-potential score). No stage advancement, no mid-rollout capture,
no save-state crutch: PLR only selects WHICH level an env plays, never skips a
level's hard first half.

All the DECISION logic (parse, index maps, sampling, outcome recording) lives
here so it is unit-testable in isolation; the trainer holds only thin glue
(see `_run_vanilla_ppo`), gated on `plr_enabled` so every existing mode is
byte-for-byte untouched.

The sampler reuses the tested `CurriculumManager.sample_level` /
`record_episode` inverse-success weighting; a single non-advancing stage over
the training levels turns it into a PLR distribution.

Index convention (matches the trainer's warm-start plumbing):
  * index 0 is the COLD-BOOT level (its state is ``None`` — the pool's
    ``reset_all`` already lands every worker on this level's entrance, so
    index-0 envs need no per-worker load).
  * index k>0 loads ``states[k]`` (the level's clean entry blob) into worker k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.training.curriculum import CurriculumManager, CurriculumStage


@dataclass
class PLRContext:
    """Everything the trainer needs to route envs across levels via PLR."""

    train_labels: list[str]                    # training level labels, index-aligned
    idx_to_level: list[str]                    # index -> label (index 0 == cold boot)
    level_to_idx: dict[str, int]               # label -> index
    states: list[Optional[bytes]]              # index-aligned state blobs (0 == None)
    level_state_path: dict[str, Optional[str]] # label -> entry-state path (None == cold)
    holdout: dict[str, str]                    # holdout label -> entry-state path
    sched: CurriculumManager                   # inverse-success level sampler
    env_level: list[str] = field(default_factory=list)  # per-env current level

    def sample(self) -> str:
        """Sample one training level by inverse-recent-success weight."""
        return self.sched.sample_level()

    def record(self, level: str, success: bool) -> None:
        """Record an episode outcome so the sampler up-weights hard levels."""
        self.sched.record_episode(level, bool(success))

    def index_of(self, level: str) -> int:
        return self.level_to_idx[level]

    def distribution(self) -> dict[str, int]:
        """Current per-env level counts (telemetry)."""
        d: dict[str, int] = {}
        for lvl in self.env_level:
            d[lvl] = d.get(lvl, 0) + 1
        return d

    def success_rates(self) -> dict[str, float]:
        """Recent success rate per training level (telemetry / weighting view)."""
        out: dict[str, float] = {}
        for lvl in self.train_labels:
            hist = self.sched._success_history.get(lvl)
            out[lvl] = (sum(hist) / len(hist)) if hist else 0.0
        return out


def build_plr_context(
    profile: dict,
    num_envs: int,
    *,
    base_dir: Optional[str] = None,
) -> Optional[PLRContext]:
    """Build a :class:`PLRContext` from a profile, or ``None`` if PLR is off.

    Enabled when ``reinforce.plr_enabled`` is truthy AND a non-empty
    ``plr_levels`` list is present. Each ``plr_levels`` entry is
    ``{level: str, state: path|null, holdout?: bool}``. Exactly one training
    level must have ``state: null`` (the cold-boot level) and it is pinned to
    index 0. ``holdout`` levels are never sampled for training.

    Raises ``ValueError`` on a malformed spec (surfaced at setup, never mid-run).
    """
    rl_cfg = profile.get("reinforce", {}) or {}
    if not rl_cfg.get("plr_enabled", False):
        return None
    levels_spec = profile.get("plr_levels") or []
    if not levels_spec:
        return None

    root = Path(base_dir) if base_dir else Path(".")

    def _resolve(p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else (root / pp))

    train: list[tuple[str, Optional[str]]] = []   # (label, path|None)
    holdout: dict[str, str] = {}
    for entry in levels_spec:
        label = str(entry["level"])
        state = entry.get("state")
        if entry.get("holdout", False):
            if state is None:
                raise ValueError(f"holdout level {label!r} needs an entry state")
            holdout[label] = _resolve(state)
        else:
            train.append((label, _resolve(state) if state is not None else None))

    if not train:
        raise ValueError("plr_levels has no training levels (all holdout?)")

    # The cold-boot level (state == None) must exist and go to index 0.
    cold = [lbl for lbl, st in train if st is None]
    if len(cold) != 1:
        raise ValueError(
            f"exactly one training level must have state: null (the cold-boot "
            f"level); got {len(cold)}: {cold}"
        )
    # Order: cold-boot level first, then the rest in declared order.
    ordered = [next(t for t in train if t[1] is None)] + [
        t for t in train if t[1] is not None
    ]
    idx_to_level = [lbl for lbl, _ in ordered]
    level_to_idx = {lbl: i for i, lbl in enumerate(idx_to_level)}
    level_state_path = {lbl: st for lbl, st in ordered}

    # Load entry blobs (index 0 == None; the pool reset_all covers it).
    states: list[Optional[bytes]] = []
    for lbl, st in ordered:
        if st is None:
            states.append(None)
        else:
            sp = Path(st)
            if not sp.exists():
                raise ValueError(f"level {lbl!r} entry state missing: {st}")
            states.append(sp.read_bytes())

    train_labels = idx_to_level[:]  # training set == the index set
    sched = CurriculumManager(
        stages=[CurriculumStage(
            name="plr",
            levels=train_labels,
            # Unreachable threshold: PLR never advances stages — a single
            # fixed distribution is the whole point.
            advance_threshold=2.0,
            min_episodes=10**9,
        )],
    )
    env_level = [idx_to_level[0]] * int(num_envs)
    return PLRContext(
        train_labels=train_labels,
        idx_to_level=idx_to_level,
        level_to_idx=level_to_idx,
        states=states,
        level_state_path=level_state_path,
        holdout=holdout,
        sched=sched,
        env_level=env_level,
    )
