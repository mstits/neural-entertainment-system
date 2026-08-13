"""Typed curriculum config, extracted from `Trainer._run_vanilla_ppo`'s
`__init__`-time parsing of the SMB one-shot campaign's curriculum knobs.

This module owns exactly the parsing — same `reinforce.*` key, same
default, same nesting — for the seven blocks:

  * `reinforce.substage_ladder`    -> `SubstageLadderConfig`
  * `reinforce.consolidate_level`  -> `ConsolidateLevelConfig`
  * `reinforce.advance`            -> `AdvanceConfig`
  * `reinforce.warm_start`         -> `WarmStartConfig`
  * `reinforce.cold_eval`          -> `ColdEvalConfig`
  * `reinforce.consolidate`        -> `ConsolidateConfig`
  * `reinforce.go_explore_fallback`-> `GoExploreFallbackConfig`

Two sub-blocks (`consolidate.entropy`/`consolidate.rnd` and
`consolidate_level.schedule.entropy`/`.rnd`) default their `from` value
to the trainer's already-resolved `entropy_coef` / `rnd_intrinsic_coef`
(`reinforce.entropy_coef` / `reinforce.rnd_intrinsic_coef`, themselves
defaulted) rather than a block-local literal — that cross-field default
is reproduced here by resolving those two scalars once and threading
them into both sub-blocks, exactly as `_run_vanilla_ppo` does via
`self.entropy_coef` / `self.rnd_intrinsic_coef`.

Two blocks are intentionally left as raw dicts rather than fully typed:
`consolidate_level.probe` (resolved through
`oneshot_curriculum.resolve_probe_settings`, a separate module's job,
not a `.get()` default) and `consolidate_level.protect` (a list of
str-or-dict entries resolved against the filesystem/ladder at
construction time, not at parse time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.training.oneshot_curriculum import ACCEPT_RULE_POINT


def _block(d: dict, key: str) -> dict:
    """Mirrors `dict(<cfg>.get(key, {}) or {})` as read throughout
    `_run_vanilla_ppo`'s curriculum setup."""
    return dict(d.get(key, {}) or {})


@dataclass(frozen=True)
class SubstageLadderConfig:
    """`reinforce.substage_ladder` (Lane 4 sub-stage ladder)."""

    enabled: bool = False
    seed_globs: Optional[Any] = None

    @classmethod
    def from_block(cls, block: dict) -> "SubstageLadderConfig":
        return cls(
            enabled=bool(block.get("enabled", False)),
            seed_globs=block.get("seed_globs"),
        )


@dataclass(frozen=True)
class AdvanceConfig:
    """`reinforce.advance` — overrides the scalar-curriculum
    SMB_ADVANCE_PCT / SMB_ADVANCE_WINDOW defaults inside `ladder_on`."""

    pct: float = 0.50
    window: int = 5

    @classmethod
    def from_block(cls, block: dict) -> "AdvanceConfig":
        return cls(
            pct=float(block.get("pct", 0.50)),
            window=int(block.get("window", 5)),
        )


@dataclass(frozen=True)
class WarmStartConfig:
    """`reinforce.warm_start` — Frontier / Retention / Spread partition."""

    frontier: float = 0.50
    retention: float = 0.25
    spread: float = 0.25

    @classmethod
    def from_block(cls, block: dict) -> "WarmStartConfig":
        return cls(
            frontier=float(block.get("frontier", 0.50)),
            retention=float(block.get("retention", 0.25)),
            spread=float(block.get("spread", 0.25)),
        )


@dataclass(frozen=True)
class ForgettingConfig:
    """`reinforce.cold_eval.forgetting` — the cold-probe regression alarm."""

    retention_bump: float = 0.40
    probes: int = 2
    bump_iters: int = 100

    @classmethod
    def from_block(cls, block: dict) -> "ForgettingConfig":
        return cls(
            retention_bump=float(block.get("retention_bump", 0.40)),
            probes=int(block.get("probes", 2)),
            bump_iters=int(block.get("bump_iters", 100)),
        )


@dataclass(frozen=True)
class ColdEvalConfig:
    """`reinforce.cold_eval` — periodic cold-boot probe."""

    every: int = 25
    curve_episodes: int = 8
    winner: str = "best_cold.pt"
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)

    @classmethod
    def from_block(cls, block: dict) -> "ColdEvalConfig":
        return cls(
            every=max(1, int(block.get("every", 25))),
            curve_episodes=max(1, int(block.get("curve_episodes", 8))),
            winner=str(block.get("winner", "best_cold.pt")),
            forgetting=ForgettingConfig.from_block(_block(block, "forgetting")),
        )


@dataclass(frozen=True)
class ConsolidateLevelScheduleConfig:
    """`reinforce.consolidate_level.schedule` — the entropy/RND lerp
    consolidation schedules used while a level-scoped weld is active."""

    ent_from: float
    ent_to: float
    ent_iters: int = 80
    rnd_from: float = 0.0
    rnd_to: float = 0.0

    @classmethod
    def from_block(
        cls, block: dict, *, entropy_coef: float, rnd_intrinsic_coef: float,
    ) -> "ConsolidateLevelScheduleConfig":
        ent = _block(block, "entropy")
        rnd = _block(block, "rnd")
        return cls(
            ent_from=float(ent.get("from", entropy_coef)),
            ent_to=float(ent.get("to", entropy_coef)),
            ent_iters=int(ent.get("iters", 80)),
            rnd_from=float(rnd.get("from", rnd_intrinsic_coef)),
            rnd_to=float(rnd.get("to", 0.0)),
        )


@dataclass(frozen=True)
class ConsolidateLevelConfig:
    """`reinforce.consolidate_level` — level-scoped consolidation
    ("weld ONE level"), mutually exclusive with the frontier ladder."""

    schedule: ConsolidateLevelScheduleConfig
    enabled: bool = False
    probe: dict = field(default_factory=dict)
    accept_bar: float = 0.75
    accept_probes: int = 3
    cooldown: int = 40
    tol: float = 1e-9
    use_wilson_bound: bool = False
    wilson_confidence: float = 0.95
    accept_rule: str = ACCEPT_RULE_POINT
    best_decay: float = 0.0
    seed_globs: Optional[Any] = None
    target: str = ""
    target_entry_state: Optional[str] = None
    protect: list = field(default_factory=list)
    winner: str = ""

    @classmethod
    def from_block(
        cls, block: dict, *, entropy_coef: float, rnd_intrinsic_coef: float,
    ) -> "ConsolidateLevelConfig":
        target = str(block.get("target", "")).strip()
        return cls(
            enabled=bool(block.get("enabled", False)),
            probe=_block(block, "probe"),
            accept_bar=float(block.get("accept_bar", 0.75)),
            accept_probes=max(1, int(block.get("accept_probes", 3))),
            cooldown=max(0, int(block.get("cooldown", 40))),
            tol=float(block.get("tol", 1e-9)),
            use_wilson_bound=bool(block.get("use_wilson_bound", False)),
            wilson_confidence=float(block.get("wilson_confidence", 0.95)),
            accept_rule=str(block.get("accept_rule", ACCEPT_RULE_POINT)),
            best_decay=float(block.get("best_decay", 0.0)),
            schedule=ConsolidateLevelScheduleConfig.from_block(
                _block(block, "schedule"),
                entropy_coef=entropy_coef,
                rnd_intrinsic_coef=rnd_intrinsic_coef,
            ),
            seed_globs=block.get("seed_globs"),
            target=target,
            target_entry_state=block.get("target_entry_state"),
            protect=list(block.get("protect", []) or []),
            winner=str(block.get("winner", f"best_{target}.pt")),
        )


@dataclass(frozen=True)
class ConsolidateConfig:
    """`reinforce.consolidate` — the (non-level-scoped) whole-pool
    consolidation entropy/RND schedule + no-gain abort + fallback mode."""

    ent_from: float
    ent_to: float
    ent_iters: int = 80
    rnd_from: float = 0.0
    rnd_to: float = 0.0
    no_gain_iters: int = 40
    fallback: str = "cyclic"

    @classmethod
    def from_block(
        cls, block: dict, *, entropy_coef: float, rnd_intrinsic_coef: float,
    ) -> "ConsolidateConfig":
        ent = _block(block, "entropy")
        rnd = _block(block, "rnd")
        abort_if = _block(block, "abort_if")
        return cls(
            ent_from=float(ent.get("from", entropy_coef)),
            ent_to=float(ent.get("to", entropy_coef)),
            ent_iters=int(ent.get("iters", 80)),
            rnd_from=float(rnd.get("from", rnd_intrinsic_coef)),
            rnd_to=float(rnd.get("to", 0.0)),
            no_gain_iters=int(abort_if.get("no_gain_iters", 40)),
            fallback=str(block.get("fallback", "cyclic")),
        )


@dataclass(frozen=True)
class GoExploreFallbackConfig:
    """`reinforce.go_explore_fallback` — the bounded, reversible
    Go-Explore unstick burst (Lane 5, deferred; off unless enabled)."""

    enabled: bool = False
    stall_patience: int = 60
    burst_iters: int = 30
    burst_env_frac: float = 0.25
    burst_env_cap: int = 8

    @classmethod
    def from_block(cls, block: dict) -> "GoExploreFallbackConfig":
        return cls(
            enabled=bool(block.get("enabled", False)),
            stall_patience=max(1, int(block.get("stall_patience", 60))),
            burst_iters=max(1, int(block.get("burst_iters", 30))),
            burst_env_frac=float(block.get("burst_env_frac", 0.25)),
            burst_env_cap=int(block.get("burst_env_cap", 8)),
        )


@dataclass(frozen=True)
class CurriculumConfig:
    """The full set of SMB one-shot-campaign curriculum blocks, parsed
    from a game profile's `reinforce` section exactly as
    `Trainer._run_vanilla_ppo`'s `__init__`-time setup reads them."""

    substage_ladder: SubstageLadderConfig
    consolidate_level: ConsolidateLevelConfig
    advance: AdvanceConfig
    warm_start: WarmStartConfig
    cold_eval: ColdEvalConfig
    consolidate: ConsolidateConfig
    go_explore_fallback: GoExploreFallbackConfig

    @classmethod
    def from_profile(cls, profile: dict) -> "CurriculumConfig":
        rl_cfg = dict(profile.get("reinforce", {}) or {})
        # Cross-field defaults: `consolidate.entropy.from` / `.to` and
        # `consolidate_level.schedule.entropy.from` / `.to` fall back to
        # the trainer's already-resolved entropy_coef, not a literal —
        # mirrored here from `reinforce.entropy_coef`'s own default.
        # Likewise the two blocks' `rnd.from` falls back to
        # rnd_intrinsic_coef, from `reinforce.rnd_intrinsic_coef`.
        entropy_coef = float(rl_cfg.get("entropy_coef", 0.01))
        rnd_intrinsic_coef = float(rl_cfg.get("rnd_intrinsic_coef", 0.0))

        return cls(
            substage_ladder=SubstageLadderConfig.from_block(
                _block(rl_cfg, "substage_ladder")
            ),
            consolidate_level=ConsolidateLevelConfig.from_block(
                _block(rl_cfg, "consolidate_level"),
                entropy_coef=entropy_coef,
                rnd_intrinsic_coef=rnd_intrinsic_coef,
            ),
            advance=AdvanceConfig.from_block(_block(rl_cfg, "advance")),
            warm_start=WarmStartConfig.from_block(_block(rl_cfg, "warm_start")),
            cold_eval=ColdEvalConfig.from_block(_block(rl_cfg, "cold_eval")),
            consolidate=ConsolidateConfig.from_block(
                _block(rl_cfg, "consolidate"),
                entropy_coef=entropy_coef,
                rnd_intrinsic_coef=rnd_intrinsic_coef,
            ),
            go_explore_fallback=GoExploreFallbackConfig.from_block(
                _block(rl_cfg, "go_explore_fallback")
            ),
        )
