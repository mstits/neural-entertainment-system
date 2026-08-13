"""Parse tests for CurriculumConfig.from_profile — pinned against the
exact expressions in Trainer._run_vanilla_ppo's __init__-time curriculum
setup (src/training/trainer.py, the

    _rl_cfg = self.game_profile.get("reinforce", {}) or {}
    _substage_cfg = dict(_rl_cfg.get("substage_ladder", {}) or {})
    _clevel_cfg  = dict(_rl_cfg.get("consolidate_level", {}) or {})
    _adv_cfg     = dict(_rl_cfg.get("advance", {}) or {})
    _ws_cfg      = dict(_rl_cfg.get("warm_start", {}) or {})
    _cold_cfg    = dict(_rl_cfg.get("cold_eval", {}) or {})
    _cons_cfg    = dict(_rl_cfg.get("consolidate", {}) or {})
    _geb_cfg     = dict(_rl_cfg.get("go_explore_fallback", {}) or {})

block (trainer.py roughly lines 5132-5530).

Two mandatory cases: a real shipped profile (configs/mario_tiles_vanilla.yaml
— which sets none of these seven blocks, so it pins the defaults PLUS the
cross-field entropy_coef/rnd_intrinsic_coef default threading) and an empty
profile (every field falls back to __init__'s exact literal default).
Additional cases below exercise every block-local key with a distinct,
non-default value so a drifted key/default bites even though the shipped
profile happens not to set any of these blocks.
"""
from pathlib import Path

import yaml

from src.training.curriculum_config import CurriculumConfig

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text()) or {}


def test_mario_tiles_vanilla_profile_matches_trainer_init_expressions():
    profile = _load("mario_tiles_vanilla.yaml")
    rl_cfg = profile["reinforce"]

    # Sanity: this profile deliberately sets none of the seven curriculum
    # blocks, so every field below must come from __init__'s literal
    # defaults — EXCEPT the two cross-field defaults (consolidate*.entropy/
    # rnd "from") which thread through the profile's own entropy_coef /
    # rnd_intrinsic_coef.
    for key in (
        "substage_ladder", "consolidate_level", "advance", "warm_start",
        "cold_eval", "consolidate", "go_explore_fallback",
    ):
        assert key not in rl_cfg
    assert rl_cfg["entropy_coef"] == 0.01
    assert rl_cfg["rnd_intrinsic_coef"] == 0.1

    cfg = CurriculumConfig.from_profile(profile)

    assert cfg.substage_ladder.enabled is False
    assert cfg.substage_ladder.seed_globs is None

    assert cfg.advance.pct == 0.50
    assert cfg.advance.window == 5

    assert cfg.warm_start.frontier == 0.50
    assert cfg.warm_start.retention == 0.25
    assert cfg.warm_start.spread == 0.25

    assert cfg.cold_eval.every == 25
    assert cfg.cold_eval.curve_episodes == 8
    assert cfg.cold_eval.winner == "best_cold.pt"
    assert cfg.cold_eval.forgetting.retention_bump == 0.40
    assert cfg.cold_eval.forgetting.probes == 2
    assert cfg.cold_eval.forgetting.bump_iters == 100

    # Cross-field: profile's entropy_coef (0.01) == the literal default
    # (0.01) so it doesn't discriminate here, but rnd_intrinsic_coef
    # (0.1) != its literal default (0.0) — this DOES discriminate, and
    # must show up as the "from" floor on both consolidation schedules.
    assert cfg.consolidate.ent_from == 0.01
    assert cfg.consolidate.ent_to == 0.01
    assert cfg.consolidate.ent_iters == 80
    assert cfg.consolidate.rnd_from == 0.1
    assert cfg.consolidate.rnd_to == 0.0
    assert cfg.consolidate.no_gain_iters == 40
    assert cfg.consolidate.fallback == "cyclic"

    assert cfg.consolidate_level.enabled is False
    assert cfg.consolidate_level.probe == {}
    assert cfg.consolidate_level.accept_bar == 0.75
    assert cfg.consolidate_level.accept_probes == 3
    assert cfg.consolidate_level.cooldown == 40
    assert cfg.consolidate_level.tol == 1e-9
    assert cfg.consolidate_level.use_wilson_bound is False
    assert cfg.consolidate_level.wilson_confidence == 0.95
    assert cfg.consolidate_level.accept_rule == "point_gt_best"
    assert cfg.consolidate_level.best_decay == 0.0
    assert cfg.consolidate_level.seed_globs is None
    assert cfg.consolidate_level.target == ""
    assert cfg.consolidate_level.target_entry_state is None
    assert cfg.consolidate_level.protect == []
    assert cfg.consolidate_level.winner == "best_.pt"
    assert cfg.consolidate_level.schedule.ent_from == 0.01
    assert cfg.consolidate_level.schedule.ent_to == 0.01
    assert cfg.consolidate_level.schedule.ent_iters == 80
    assert cfg.consolidate_level.schedule.rnd_from == 0.1
    assert cfg.consolidate_level.schedule.rnd_to == 0.0

    assert cfg.go_explore_fallback.enabled is False
    assert cfg.go_explore_fallback.stall_patience == 60
    assert cfg.go_explore_fallback.burst_iters == 30
    assert cfg.go_explore_fallback.burst_env_frac == 0.25
    assert cfg.go_explore_fallback.burst_env_cap == 8


def test_empty_profile_falls_back_to_trainer_init_defaults():
    cfg = CurriculumConfig.from_profile({})

    assert cfg.substage_ladder.enabled is False
    assert cfg.substage_ladder.seed_globs is None

    assert cfg.advance.pct == 0.50
    assert cfg.advance.window == 5

    assert cfg.warm_start.frontier == 0.50
    assert cfg.warm_start.retention == 0.25
    assert cfg.warm_start.spread == 0.25

    assert cfg.cold_eval.every == 25
    assert cfg.cold_eval.curve_episodes == 8
    assert cfg.cold_eval.winner == "best_cold.pt"
    assert cfg.cold_eval.forgetting.retention_bump == 0.40
    assert cfg.cold_eval.forgetting.probes == 2
    assert cfg.cold_eval.forgetting.bump_iters == 100

    # Both entropy_coef (0.01) and rnd_intrinsic_coef (0.0) are the bare
    # trainer __init__ defaults now (no profile override).
    assert cfg.consolidate.ent_from == 0.01
    assert cfg.consolidate.ent_to == 0.01
    assert cfg.consolidate.ent_iters == 80
    assert cfg.consolidate.rnd_from == 0.0
    assert cfg.consolidate.rnd_to == 0.0
    assert cfg.consolidate.no_gain_iters == 40
    assert cfg.consolidate.fallback == "cyclic"

    assert cfg.consolidate_level.enabled is False
    assert cfg.consolidate_level.probe == {}
    assert cfg.consolidate_level.accept_bar == 0.75
    assert cfg.consolidate_level.accept_probes == 3
    assert cfg.consolidate_level.cooldown == 40
    assert cfg.consolidate_level.tol == 1e-9
    assert cfg.consolidate_level.use_wilson_bound is False
    assert cfg.consolidate_level.wilson_confidence == 0.95
    assert cfg.consolidate_level.accept_rule == "point_gt_best"
    assert cfg.consolidate_level.best_decay == 0.0
    assert cfg.consolidate_level.seed_globs is None
    assert cfg.consolidate_level.target == ""
    assert cfg.consolidate_level.target_entry_state is None
    assert cfg.consolidate_level.protect == []
    # target is "" -> f"best_{target}.pt" == "best_.pt"
    assert cfg.consolidate_level.winner == "best_.pt"
    assert cfg.consolidate_level.schedule.ent_from == 0.01
    assert cfg.consolidate_level.schedule.ent_to == 0.01
    assert cfg.consolidate_level.schedule.ent_iters == 80
    assert cfg.consolidate_level.schedule.rnd_from == 0.0
    assert cfg.consolidate_level.schedule.rnd_to == 0.0

    assert cfg.go_explore_fallback.enabled is False
    assert cfg.go_explore_fallback.stall_patience == 60
    assert cfg.go_explore_fallback.burst_iters == 30
    assert cfg.go_explore_fallback.burst_env_frac == 0.25
    assert cfg.go_explore_fallback.burst_env_cap == 8


def test_missing_reinforce_block_behaves_like_empty_dict():
    """`self.game_profile.get("reinforce", {}) or {}` — a profile with no
    `reinforce` key at all must parse identically to an explicit empty
    one."""
    assert CurriculumConfig.from_profile({"name": "no reinforce here"}) == (
        CurriculumConfig.from_profile({})
    )


def test_reinforce_null_behaves_like_empty_dict():
    """`... or {}` also absorbs an explicit `reinforce: null` in yaml."""
    assert CurriculumConfig.from_profile({"reinforce": None}) == (
        CurriculumConfig.from_profile({})
    )


def test_substage_ladder_block_every_key():
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "substage_ladder": {
                "enabled": True,
                "seed_globs": ["runs/seeds/*.state"],
            },
        },
    })
    assert cfg.substage_ladder.enabled is True
    assert cfg.substage_ladder.seed_globs == ["runs/seeds/*.state"]


def test_advance_block_every_key():
    cfg = CurriculumConfig.from_profile(
        {"reinforce": {"advance": {"pct": 0.65, "window": 9}}}
    )
    assert cfg.advance.pct == 0.65
    assert cfg.advance.window == 9


def test_warm_start_block_every_key():
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "warm_start": {"frontier": 0.40, "retention": 0.35, "spread": 0.20},
        },
    })
    assert cfg.warm_start.frontier == 0.40
    assert cfg.warm_start.retention == 0.35
    assert cfg.warm_start.spread == 0.20


def test_cold_eval_block_every_key_including_nested_forgetting():
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "cold_eval": {
                "every": 10,
                "curve_episodes": 4,
                "winner": "custom_cold.pt",
                "forgetting": {
                    "retention_bump": 0.55,
                    "probes": 5,
                    "bump_iters": 33,
                },
            },
        },
    })
    assert cfg.cold_eval.every == 10
    assert cfg.cold_eval.curve_episodes == 4
    assert cfg.cold_eval.winner == "custom_cold.pt"
    assert cfg.cold_eval.forgetting.retention_bump == 0.55
    assert cfg.cold_eval.forgetting.probes == 5
    assert cfg.cold_eval.forgetting.bump_iters == 33


def test_cold_eval_every_and_curve_episodes_are_floored_at_1():
    """__init__ wraps both in max(1, int(...))."""
    cfg = CurriculumConfig.from_profile({
        "reinforce": {"cold_eval": {"every": 0, "curve_episodes": -3}},
    })
    assert cfg.cold_eval.every == 1
    assert cfg.cold_eval.curve_episodes == 1


def test_consolidate_block_every_key():
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "consolidate": {
                "entropy": {"from": 0.02, "to": 0.005, "iters": 120},
                "rnd": {"from": 0.3, "to": 0.05},
                "abort_if": {"no_gain_iters": 15},
                "fallback": "hold",
            },
        },
    })
    assert cfg.consolidate.ent_from == 0.02
    assert cfg.consolidate.ent_to == 0.005
    assert cfg.consolidate.ent_iters == 120
    assert cfg.consolidate.rnd_from == 0.3
    assert cfg.consolidate.rnd_to == 0.05
    assert cfg.consolidate.no_gain_iters == 15
    assert cfg.consolidate.fallback == "hold"


def test_consolidate_entropy_and_rnd_from_default_to_top_level_coefs():
    """`_cons_ent.get("from", self.entropy_coef)` / `_cons_rnd.get("from",
    self.rnd_intrinsic_coef)` — absent 'from' falls back to the
    profile's OWN reinforce.entropy_coef / reinforce.rnd_intrinsic_coef,
    not the block-local hardcoded literal."""
    cfg = CurriculumConfig.from_profile({
        "reinforce": {"entropy_coef": 0.42, "rnd_intrinsic_coef": 0.77},
    })
    assert cfg.consolidate.ent_from == 0.42
    assert cfg.consolidate.ent_to == 0.42
    assert cfg.consolidate.rnd_from == 0.77
    # rnd "to" has its own flat 0.0 default, independent of rnd_intrinsic_coef.
    assert cfg.consolidate.rnd_to == 0.0
    assert cfg.consolidate_level.schedule.ent_from == 0.42
    assert cfg.consolidate_level.schedule.ent_to == 0.42
    assert cfg.consolidate_level.schedule.rnd_from == 0.77
    assert cfg.consolidate_level.schedule.rnd_to == 0.0


def test_consolidate_level_block_every_key():
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "consolidate_level": {
                "enabled": True,
                "probe": {"episodes": 5, "sticky_prob": 0.25},
                "accept_bar": 0.9,
                "accept_probes": 7,
                "cooldown": 12,
                "tol": 1e-6,
                "use_wilson_bound": True,
                "wilson_confidence": 0.99,
                "accept_rule": "wilson_lb_gt_best_point",
                "best_decay": 0.5,
                "schedule": {
                    "entropy": {"from": 0.03, "to": 0.001, "iters": 50},
                    "rnd": {"from": 0.4, "to": 0.1},
                },
                "seed_globs": ["runs/1-2/*.state"],
                "target": "1-2",
                "target_entry_state": "runs/1-2/entry.state",
                "protect": ["1-1", {"level": "1-2b", "entry_state": "x.state"}],
                "winner": "best_custom.pt",
            },
        },
    })
    cl = cfg.consolidate_level
    assert cl.enabled is True
    assert cl.probe == {"episodes": 5, "sticky_prob": 0.25}
    assert cl.accept_bar == 0.9
    assert cl.accept_probes == 7
    assert cl.cooldown == 12
    assert cl.tol == 1e-6
    assert cl.use_wilson_bound is True
    assert cl.wilson_confidence == 0.99
    assert cl.accept_rule == "wilson_lb_gt_best_point"
    assert cl.best_decay == 0.5
    assert cl.schedule.ent_from == 0.03
    assert cl.schedule.ent_to == 0.001
    assert cl.schedule.ent_iters == 50
    assert cl.schedule.rnd_from == 0.4
    assert cl.schedule.rnd_to == 0.1
    assert cl.seed_globs == ["runs/1-2/*.state"]
    assert cl.target == "1-2"
    assert cl.target_entry_state == "runs/1-2/entry.state"
    assert cl.protect == ["1-1", {"level": "1-2b", "entry_state": "x.state"}]
    assert cl.winner == "best_custom.pt"


def test_consolidate_level_accept_probes_and_cooldown_are_clamped():
    """__init__ wraps accept_probes in max(1, int(...)) and cooldown in
    max(0, int(...))."""
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "consolidate_level": {"accept_probes": 0, "cooldown": -5},
        },
    })
    assert cfg.consolidate_level.accept_probes == 1
    assert cfg.consolidate_level.cooldown == 0


def test_consolidate_level_winner_default_depends_on_target():
    """`_clevel_cfg.get("winner", f"best_{clevel_target}.pt")` — the
    default is derived from the SAME block's own (stripped) target, not a
    flat literal."""
    cfg = CurriculumConfig.from_profile({
        "reinforce": {"consolidate_level": {"target": "  1-3  "}},
    })
    assert cfg.consolidate_level.target == "1-3"
    assert cfg.consolidate_level.winner == "best_1-3.pt"


def test_go_explore_fallback_block_every_key():
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "go_explore_fallback": {
                "enabled": True,
                "stall_patience": 20,
                "burst_iters": 15,
                "burst_env_frac": 0.5,
                "burst_env_cap": 4,
            },
        },
    })
    ge = cfg.go_explore_fallback
    assert ge.enabled is True
    assert ge.stall_patience == 20
    assert ge.burst_iters == 15
    assert ge.burst_env_frac == 0.5
    assert ge.burst_env_cap == 4


def test_go_explore_fallback_stall_patience_and_burst_iters_floored_at_1():
    """__init__ wraps both in max(1, int(...))."""
    cfg = CurriculumConfig.from_profile({
        "reinforce": {
            "go_explore_fallback": {"stall_patience": 0, "burst_iters": -2},
        },
    })
    assert cfg.go_explore_fallback.stall_patience == 1
    assert cfg.go_explore_fallback.burst_iters == 1


def test_config_is_frozen():
    cfg = CurriculumConfig.from_profile({})
    try:
        cfg.advance = None  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("CurriculumConfig must be frozen (immutable)")
