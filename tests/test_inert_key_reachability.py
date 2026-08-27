"""The OTHER direction of the config registry.

`test_config_schema.py` enforces "every key the trainer consumes is
registered". That check is ONE-WAY, which is exactly why all 34 schema
tests pass green while the flagship 1-1 recipe declares eight lines of
machinery that never ran. `configs/mario_1_1_backward.yaml` and all eight
v27/v28 seed configs carry a "Phase-A recipe, verbatim" header over
`bc_replay_enabled: true`, `warmup_gens_ga_only: 10`,
`preserve_elite_diversity: true`, `freeze_pre_ppo_elite: true` and
friends — every one of which is consumed only inside
`_run_one_generation` / `_run_bc_replay` / `_snapshot_pre_ppo_elite`,
none of which `trainer_mode: vanilla_ppo` ever enters.

The numbers are untouched by this (a knob inert in the baseline is inert
in every comparison arm too), but the recipe describes an experiment that
was not run.

This is derived from the AST rather than listed, because a hand-written
list of dead knobs rots the first time somebody moves a method — and
because the trainer already contains two hand-written "this knob is a
no-op, warn and disable" guards, one of which checks the WRONG mode
(`trainer_mode == 'pure_ppo'`) and therefore stays silent in the only
mode any banked run used.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.training.config_schema import (
    ReachabilityDerivationError,
    inert_reinforce_keys_under_vanilla_ppo,
    vanilla_reachable_methods,
)

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "src" / "training" / "trainer.py"
SIBLINGS = tuple(
    p for p in (REPO / "src" / "training").glob("*.py") if p.name != "trainer.py"
)


@pytest.fixture(scope="module")
def inert() -> dict[str, list[str]]:
    return inert_reinforce_keys_under_vanilla_ppo(TRAINER, SIBLINGS)


# ---------------------------------------------------------------------------
# The reachability walk itself
# ---------------------------------------------------------------------------

def test_vanilla_ppo_reaches_its_own_trainer_but_not_the_ga_loop() -> None:
    reachable = vanilla_reachable_methods(TRAINER)
    assert "_run_vanilla_ppo" in reachable
    for ga_only in ("_run_one_generation", "_run_bc_replay",
                    "_snapshot_pre_ppo_elite", "_reinforce_update"):
        assert ga_only not in reachable, (
            f"{ga_only} is reachable under vanilla_ppo — either run() was "
            "refactored or this derivation is now wrong")


def test_derivation_raises_rather_than_silently_reporting_nothing(
    tmp_path: Path,
) -> None:
    """ANTI-VACUITY. A checker that returns "all clear" because it could not
    find the code it reads is the defect it exists to catch."""
    stub = tmp_path / "trainer.py"
    stub.write_text("class Trainer:\n    def run(self):\n        return 1\n")
    with pytest.raises(ReachabilityDerivationError):
        vanilla_reachable_methods(stub)

    missing = tmp_path / "no_class.py"
    missing.write_text("def run():\n    return 1\n")
    with pytest.raises(ReachabilityDerivationError):
        vanilla_reachable_methods(missing)


# ---------------------------------------------------------------------------
# The keys themselves — both directions
# ---------------------------------------------------------------------------

# Consumed ONLY inside the GA call graph. Every one of these is set in at
# least one vanilla_ppo profile in configs/.
KNOWN_INERT = {
    "bc_replay_enabled", "bc_replay_epochs", "bc_replay_every_gens",
    "bc_replay_max_buffer", "bc_replay_train_window",
    "episodes_per_genome", "warmup_gens_ga_only",
    "preserve_elite_diversity", "freeze_pre_ppo_elite",
    "symlog_rewards",
}

# Genuinely live on the vanilla_ppo path. The negative control: an analysis
# that flagged everything would satisfy the assertion above and be useless.
KNOWN_LIVE = {
    "lr", "gamma", "gae_lambda", "value_coef", "entropy_coef", "grad_clip",
    "rollout_steps", "num_envs", "ppo_clip_eps", "ppo_minibatch_size",
    "rnd_loss_coef", "redo_enabled", "redo_tau", "backward_curriculum",
    "sil", "kl_anchor_loss_coef", "entropy_floor", "trainer_mode",
}


def test_the_known_dead_ga_knobs_are_all_detected(inert) -> None:
    missing = KNOWN_INERT - set(inert)
    assert not missing, (
        f"these keys are consumed only in the GA call graph but were not "
        f"flagged: {sorted(missing)}")


def test_live_vanilla_ppo_knobs_are_not_flagged(inert) -> None:
    """The negative control.

    `ppo_clip_eps` and `rnd_loss_coef` are the interesting ones: inside
    trainer.py they appear only in `_reinforce_update` /
    `_recurrent_ppo_update`, so a trainer-only scan calls them dead. They
    are read by `ppo_updater.py` off a trainer handle (`t.ppo_clip_eps`),
    which is why the derivation takes sibling modules. A false "this knob
    is inert" is the costlier error here, so the analysis errs toward
    silence and this test pins that.
    """
    wrong = KNOWN_LIVE & set(inert)
    assert not wrong, f"live knobs flagged as inert: {sorted(wrong)}"


def test_symlog_rewards_is_detected_as_the_calibration_case(inert) -> None:
    """`symlog_rewards` was found inert by hand on 2026-08-26 and annotated
    in the v28 config. Deriving it from source reproduces that finding, which
    is what makes the other nineteen credible."""
    assert "symlog_rewards" in inert
    assert inert["symlog_rewards"] == ["_reinforce_update"]


def test_every_flagged_key_names_a_real_consumer(inert) -> None:
    """A flagged key must point at the method that reads it — an empty
    consumer list would mean the key is simply unused, which is a different
    (and less interesting) defect."""
    tree = ast.parse(TRAINER.read_text())
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for key, sites in inert.items():
        assert sites, f"{key} flagged with no consumer"
        for s in sites:
            assert s in defined, f"{key} names a non-existent method {s}"


def test_the_flagged_set_is_neither_empty_nor_everything(inert) -> None:
    """Bounds the analysis from both sides so a degenerate implementation
    (flag nothing / flag all) fails rather than passing quietly."""
    from src.training.config_schema import KNOWN_REINFORCE_KEYS
    assert 5 <= len(inert) <= len(KNOWN_REINFORCE_KEYS) // 2


# ---------------------------------------------------------------------------
# What the banked profiles actually declare
# ---------------------------------------------------------------------------

BANKED_CONFIGS = [
    "mario_1_1_backward.yaml",          # the 1-1 LEARNABLE 0.76 recipe
    "mario_1_1_v27_seed0.yaml",         # v27 FAIL 0.530
    "mario_1_1_v28_seed0.yaml",         # v28 FAIL 0.670
]


@pytest.mark.parametrize("name", BANKED_CONFIGS)
def test_banked_configs_declare_machinery_that_cannot_run(name, inert) -> None:
    """Documents the defect rather than forbidding it.

    These profiles are historical artifacts of banked results and must not
    be edited to make a test pass — the point is that the recipe text and
    the executed experiment disagree, and that this is now detectable
    mechanically instead of by a 70-agent audit.
    """
    import yaml
    p = REPO / "configs" / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    rl = (yaml.safe_load(p.read_text()) or {}).get("reinforce", {}) or {}
    if str(rl.get("trainer_mode", "vanilla_ppo")).lower() != "vanilla_ppo":
        pytest.skip("not a vanilla_ppo profile")
    declared_dead = sorted(set(rl) & set(inert))
    assert declared_dead, (
        f"{name} was expected to carry the inherited GA knobs; if it no "
        "longer does, this documentation test should be retired")
