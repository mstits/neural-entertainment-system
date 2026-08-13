"""Typed exploration config (RND + count-based bonus + Go-Explore),
extracted from the config parsing block of ``Trainer.__init__`` and the
neighboring ``_run_vanilla_ppo`` setup in ``src/training/trainer.py``.

This is a pure, additive extraction (trainer_decomposition_plan.md, Task
0.5 / §5): ``ExplorationConfig.from_profile`` reproduces exactly what the
trainer currently computes for these fields from a game profile, so it
can be unit-tested standalone before any wiring happens. It is not yet
consumed by the trainer.

Field-for-field mapping (all under ``rl_cfg = profile.get("reinforce", {})``
unless noted):

    rnd_intrinsic_coef            <- float(rl_cfg.get("rnd_intrinsic_coef", 0.0))
    rnd_loss_coef                 <- float(rl_cfg.get("rnd_loss_coef", 1.0))
    gx_count_beta                 <- float(rl_cfg.get("gx_count_bonus_coef", 0.0))
                                      (trainer.py's private `self._gx_count_beta`)
    rnd_predictor_update_fraction <- float(rl_cfg.get(
                                          "rnd_predictor_update_fraction", 1.0))
                                      raises ValueError if not in (0.0, 1.0]
                                      ("reinforce.rnd_predictor_update_fraction
                                      must be in (0, 1]; got {value!r}")

The Go-Explore archive block lives one level further: it is parsed lazily
inside ``_run_vanilla_ppo`` (not ``__init__`` proper), gated on
``reinforce.go_explore.enabled``. Mirrored here as ``GoExploreConfig``:

    _ge_cfg = dict(
        (profile.get("reinforce", {}) or {}).get("go_explore", {}) or {}
    )
    enabled            <- bool(_ge_cfg.get("enabled", False))
                           (the trainer additionally ANDs this with
                           `not self._smb_curriculum_active` to get the
                           runtime `go_explore_on` flag -- that composition
                           is cross-field trainer state, out of scope for a
                           pure profile parse, so this field mirrors only
                           the raw profile flag)
    save_every         <- max(1, int(_ge_cfg.get("save_every", 10)))
    inline_return_prob <- float(_ge_cfg.get("inline_return_prob", 0.0))
    seed               <- _ge_cfg.get("seed"), coerced to int if present
                           else None. The trainer's actual default is
                           `self.seed` (the constructor's own `seed`
                           parameter, not a profile field), so
                           `int(_ge_cfg.get("seed", self.seed) or 0)` is
                           not reproducible from a profile dict alone;
                           None here means "falls back to the trainer's
                           seed at the call site."
    score              <- str(_ge_cfg.get("score", "auto"))
    cell               <- dict(_ge_cfg.get("cell", {}) or {})
                           (left as a raw dict: the resolved per-key
                           defaults for stride/bucket/gx_bucket/y_bucket/
                           addresses are branch-dependent on `cell.type`
                           and are only ever consumed downstream when
                           building the concrete cell_fn -- that selection
                           is cell_fn-construction logic, not config
                           parsing, so it is out of scope here)

The bounded, reversible "unstick burst" fallback is a SEPARATE top-level
`reinforce.go_explore_fallback` block (its own registered key in
config_schema.py), read via a differently-guarded `_rl_cfg`:

    _rl_cfg = profile.get("reinforce", {}) or {}
    _geb_cfg = dict(_rl_cfg.get("go_explore_fallback", {}) or {})
    enabled         <- bool(_geb_cfg.get("enabled", False))
                        (the trainer additionally ANDs this with
                        `ladder_on`, cross-field trainer state -- same
                        scoping note as go_explore.enabled above)
    stall_patience  <- max(1, int(_geb_cfg.get("stall_patience", 60)))
    burst_iters     <- max(1, int(_geb_cfg.get("burst_iters", 30)))
    burst_env_frac  <- float(_geb_cfg.get("burst_env_frac", 0.25))
    burst_env_cap   <- int(_geb_cfg.get("burst_env_cap", 8))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoExploreConfig:
    """`reinforce.go_explore.*` -- the first-return-then-explore archive."""

    enabled: bool = False
    save_every: int = 10
    inline_return_prob: float = 0.0
    seed: int | None = None
    score: str = "auto"
    cell: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoExploreFallbackConfig:
    """`reinforce.go_explore_fallback.*` -- the bounded, reversible
    archive "unstick burst" armed when a substage-ladder rung stalls."""

    enabled: bool = False
    stall_patience: int = 60
    burst_iters: int = 30
    burst_env_frac: float = 0.25
    burst_env_cap: int = 8


@dataclass(frozen=True)
class ExplorationConfig:
    """RND intrinsic motivation, the count-based frontier bonus, and the
    Go-Explore archive/burst knobs (see trainer.py's `rl_cfg =
    profile.get("reinforce", {})` block plus the `_ge_cfg` / `_geb_cfg`
    reads in `_run_vanilla_ppo`)."""

    rnd_intrinsic_coef: float = 0.0
    rnd_loss_coef: float = 1.0
    rnd_predictor_update_fraction: float = 1.0
    gx_count_beta: float = 0.0
    go_explore: GoExploreConfig = field(default_factory=GoExploreConfig)
    go_explore_fallback: GoExploreFallbackConfig = field(
        default_factory=GoExploreFallbackConfig
    )

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "ExplorationConfig":
        rl_cfg = profile.get("reinforce", {})

        rnd_intrinsic_coef = float(rl_cfg.get("rnd_intrinsic_coef", 0.0))
        rnd_loss_coef = float(rl_cfg.get("rnd_loss_coef", 1.0))
        gx_count_beta = float(rl_cfg.get("gx_count_bonus_coef", 0.0))

        _rnd_pred_frac = float(
            rl_cfg.get("rnd_predictor_update_fraction", 1.0)
        )
        if not (0.0 < _rnd_pred_frac <= 1.0):
            raise ValueError(
                "reinforce.rnd_predictor_update_fraction must be in (0, 1]; "
                f"got {_rnd_pred_frac!r}"
            )
        rnd_predictor_update_fraction = _rnd_pred_frac

        _ge_cfg = dict(
            (profile.get("reinforce", {}) or {}).get("go_explore", {}) or {}
        )
        _ge_seed_raw = _ge_cfg.get("seed")
        go_explore = GoExploreConfig(
            enabled=bool(_ge_cfg.get("enabled", False)),
            save_every=max(1, int(_ge_cfg.get("save_every", 10))),
            inline_return_prob=float(_ge_cfg.get("inline_return_prob", 0.0)),
            seed=int(_ge_seed_raw) if _ge_seed_raw is not None else None,
            score=str(_ge_cfg.get("score", "auto")),
            cell=dict(_ge_cfg.get("cell", {}) or {}),
        )

        _rl_cfg = profile.get("reinforce", {}) or {}
        _geb_cfg = dict(_rl_cfg.get("go_explore_fallback", {}) or {})
        go_explore_fallback = GoExploreFallbackConfig(
            enabled=bool(_geb_cfg.get("enabled", False)),
            stall_patience=max(1, int(_geb_cfg.get("stall_patience", 60))),
            burst_iters=max(1, int(_geb_cfg.get("burst_iters", 30))),
            burst_env_frac=float(_geb_cfg.get("burst_env_frac", 0.25)),
            burst_env_cap=int(_geb_cfg.get("burst_env_cap", 8)),
        )

        return cls(
            rnd_intrinsic_coef=rnd_intrinsic_coef,
            rnd_loss_coef=rnd_loss_coef,
            rnd_predictor_update_fraction=rnd_predictor_update_fraction,
            gx_count_beta=gx_count_beta,
            go_explore=go_explore,
            go_explore_fallback=go_explore_fallback,
        )
