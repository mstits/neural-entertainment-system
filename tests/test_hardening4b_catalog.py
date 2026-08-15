"""Guard configs/overrides/online_1_2_hardening4b.yaml — the inert,
pre-registered ANCHORED hardening package for the 1-2 online campaign
(the attempt-7 redesign: adversary behind an anchor + gated budget ramp).

Four check families, none of which launch an emulator or touch the live
campaign's files:

  - the YAML parses, has the documented block set, and every block with
    an `override` deep-merges onto the campaign base profile
    (configs/mario_1_2_online_v2.yaml, read-only here) into a profile
    that passes config_schema's STRICT validation;
  - the anchor block's shape matches the registration (self-referential
    prior = the preserved FINAL consolidated checkpoint, loss tether
    0.3, reward-level beta OFF, actor unfrozen) and the prior checkpoint
    actually exists on disk;
  - the budget ramp's decision logic (scripts/hardening4b_ramp.py) is a
    pure function exercised here on stub probe-median streams — advance
    on 2 consecutive holds >= 800, counter reset on a sub-800 probe,
    kill-rollback on any median < 500, absorbing after kill;
  - the trainer's re-anchor-on-resume contract holds at source level:
    the KL prior is rebuilt from the CONFIG's kl_anchor_checkpoint on
    every construction, while load_actor_into (actor seeding) stays
    guarded behind iter_offset == 0 — the mechanism the H1 block's
    documentation depends on.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from scripts.hardening4b_ramp import (
    BUDGET_RAMP,
    HOLD_MEDIAN,
    HOLDS_TO_ADVANCE,
    KILL_MEDIAN,
    RampState,
    budget_for,
    check_armed_profile,
    run_stream,
    step_ramp,
)
from src.training.config_schema import ConfigSchemaError, check_profile

REPO = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO / "configs" / "overrides" / "online_1_2_hardening4b.yaml"
BASE_PROFILE_PATH = REPO / "configs" / "mario_1_2_online_v2.yaml"
TRAINER_SRC = REPO / "src" / "training" / "trainer.py"

ANCHOR_CKPT = "checkpoints/_preserved/online_v2_FINAL_consolidated.pt"

OVERRIDE_BLOCK_IDS = {
    "H1_anchor_prior",
    "H2_ramp_r1_budget_030",
    "H2_ramp_r2_budget_020",
    "H2_ramp_r3_budget_010",
}
DOC_BLOCK_IDS = {"K1_kill_rollback"}
RAMP_BLOCK_IDS_IN_ORDER = [
    "H2_ramp_r1_budget_030",
    "H2_ramp_r2_budget_020",
    "H2_ramp_r3_budget_010",
]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@pytest.fixture(scope="module")
def catalog() -> dict:
    assert CATALOG_PATH.exists(), f"missing catalog: {CATALOG_PATH}"
    with CATALOG_PATH.open() as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict), "catalog must parse to a mapping"
    return doc


@pytest.fixture(scope="module")
def blocks(catalog: dict) -> dict:
    assert "hardening4b" in catalog, (
        "catalog must have a top-level 'hardening4b' map"
    )
    assert isinstance(catalog["hardening4b"], dict)
    return catalog["hardening4b"]


@pytest.fixture(scope="module")
def base_profile() -> dict:
    assert BASE_PROFILE_PATH.exists(), (
        f"missing base profile: {BASE_PROFILE_PATH}"
    )
    with BASE_PROFILE_PATH.open() as f:
        profile = yaml.safe_load(f)
    assert isinstance(profile, dict)
    return profile


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------

def test_catalog_has_all_registered_blocks(blocks: dict):
    ids = set(blocks.keys())
    missing = (OVERRIDE_BLOCK_IDS | DOC_BLOCK_IDS) - ids
    assert not missing, f"catalog is missing expected blocks: {missing}"


@pytest.mark.parametrize("block_id", sorted(OVERRIDE_BLOCK_IDS))
def test_override_blocks_carry_trigger_override_receipt(
    blocks: dict, block_id: str,
):
    block = blocks[block_id]
    assert isinstance(block, dict), f"{block_id} must be a mapping"
    for key in ("trigger", "receipt"):
        assert isinstance(block.get(key), str) and block[key].strip(), (
            f"{block_id}.{key} must be non-empty prose"
        )
    assert isinstance(block.get("override"), dict) and block["override"], (
        f"{block_id}.override must be a non-empty partial-profile mapping"
    )


def test_kill_block_is_a_procedure_not_an_override(blocks: dict):
    """K1 is a rollback PROCEDURE (checkpoint surgery), not a config
    knob — it must not carry an `override` that could be hand-merged as
    if aborting were a hyperparameter."""
    block = blocks["K1_kill_rollback"]
    assert "override" not in block, (
        "K1_kill_rollback must not have an override: rollback is "
        "checkpoint surgery, not a profile merge"
    )
    proc = block.get("procedure")
    assert isinstance(proc, str) and proc.strip()
    # The registered kill line lives in the trigger (median < 500 on
    # any honest probe); the response in the procedure must be a
    # ROLLBACK to the pre-hardening checkpoint with the degraded
    # post-hardening iters quarantined — not merely stopping training.
    assert "500" in block["trigger"]
    assert "quarantine" in proc.lower()
    assert ANCHOR_CKPT in proc and "pre-hardening" in proc.lower()


@pytest.mark.parametrize("block_id", sorted(OVERRIDE_BLOCK_IDS))
def test_override_merged_onto_base_profile_passes_strict_schema(
    blocks: dict, base_profile: dict, block_id: str,
):
    merged = _deep_merge(base_profile, blocks[block_id]["override"])
    warnings = check_profile(merged, strict=True)
    assert warnings == []


def test_a_bad_override_is_actually_caught_by_this_harness(
    base_profile: dict,
):
    bad_override = {"reinforce": {"adversary": {"budget_penality": 0.3}}}
    merged = _deep_merge(base_profile, bad_override)
    with pytest.raises(ConfigSchemaError):
        check_profile(merged, strict=True)


# ---------------------------------------------------------------------------
# H1 anchor block — self-referential prior + loss tether, betas off
# ---------------------------------------------------------------------------

def test_anchor_prior_checkpoint_exists_on_disk():
    assert (REPO / ANCHOR_CKPT).exists(), (
        f"anchor prior {ANCHOR_CKPT} is not on disk — the H1 block "
        f"would fail at KLAnchor construction"
    )


def test_h1_anchor_block_shape(blocks: dict):
    rl = blocks["H1_anchor_prior"]["override"]["reinforce"]
    assert rl["kl_anchor_checkpoint"] == ANCHOR_CKPT
    assert rl["kl_anchor_loss_coef"] == 0.3
    # Reward-level beta penalty OFF: the tether is loss-level only.
    assert rl["kl_beta_start"] == 0.0
    assert rl["kl_beta_end"] == 0.0
    # The base profile freezes the actor for 1e12 steps (phase-0
    # critic warmup); the hardening resume must explicitly unfreeze or
    # the adversary would be fighting a statue.
    assert rl["actor_freeze_steps"] == 0


# ---------------------------------------------------------------------------
# H2 ramp blocks — 0.3 -> 0.2 -> 0.1, each self-contained
# ---------------------------------------------------------------------------

def test_ramp_budget_sequence_matches_registration(blocks: dict):
    budgets = [
        blocks[b]["override"]["reinforce"]["adversary"]["budget_penalty"]
        for b in RAMP_BLOCK_IDS_IN_ORDER
    ]
    assert budgets == [0.3, 0.2, 0.1]


def test_ramp_budget_sequence_matches_pure_function_constants(blocks: dict):
    """The yaml's registered ramp and the decision module's constants
    must agree — the pre-registration is one package, not two."""
    budgets = [
        blocks[b]["override"]["reinforce"]["adversary"]["budget_penalty"]
        for b in RAMP_BLOCK_IDS_IN_ORDER
    ]
    assert tuple(budgets) == BUDGET_RAMP
    assert HOLD_MEDIAN == 800.0
    assert KILL_MEDIAN == 500.0
    assert HOLDS_TO_ADVANCE == 2


@pytest.mark.parametrize("block_id", RAMP_BLOCK_IDS_IN_ORDER)
def test_each_ramp_rung_is_self_contained(blocks: dict, block_id: str):
    """Every rung's override carries the FULL hardening stack (anchor +
    adversary + epochs + entropy), so hand-merging exactly one block
    onto the base profile yields the complete registered configuration —
    an operator cannot drop the anchor by merging only the rung."""
    rl = blocks[block_id]["override"]["reinforce"]
    anchor_rl = blocks["H1_anchor_prior"]["override"]["reinforce"]
    for k, v in anchor_rl.items():
        assert rl.get(k) == v, (
            f"{block_id} must restate anchor key {k!r}={v!r} "
            f"(got {rl.get(k)!r})"
        )
    # 10:10 epochs (protagonist reinforce.steps : adversary.epochs).
    assert rl["steps"] == 10
    assert rl["adversary"]["epochs"] == 10
    # Entropy 0.01 on both sides.
    assert rl["entropy_coef"] == 0.01
    assert rl["adversary"]["entropy_coef"] == 0.01
    # The adversary is ARMED in every rung.
    assert rl["adversary"]["mode"] == "kernel_sticky"


@pytest.mark.parametrize(
    "block_id", ["H2_ramp_r2_budget_020", "H2_ramp_r3_budget_010"],
)
def test_ramp_advance_triggers_state_the_gate(blocks: dict, block_id: str):
    """Rungs 2 and 3 are reachable only through the hold gate: the
    trigger prose must state the >=800 median and the 2-consecutive
    requirement."""
    trigger = blocks[block_id]["trigger"]
    assert "800" in trigger
    assert "consecutive" in trigger.lower()


# ---------------------------------------------------------------------------
# Ramp/gate decision logic — pure functions on stub probe streams
# ---------------------------------------------------------------------------

def test_two_consecutive_holds_advance_one_rung():
    rows = run_stream([850.0, 900.0])
    assert [r["action"] for r in rows] == ["hold", "advance"]
    assert rows[-1]["rung"] == 1
    assert rows[-1]["budget"] == 0.2


def test_sub_800_probe_resets_the_consecutive_counter():
    rows = run_stream([850.0, 700.0, 850.0, 900.0])
    assert [r["action"] for r in rows] == [
        "hold", "reset", "hold", "advance",
    ]
    assert rows[-1]["rung"] == 1


def test_full_walk_to_budget_010_then_complete():
    rows = run_stream([810.0, 900.0, 820.0, 830.0, 805.0, 900.0])
    assert [r["action"] for r in rows] == [
        "hold", "advance", "hold", "advance", "hold", "complete",
    ]
    final = rows[-1]
    assert final["rung"] == 2
    assert final["budget"] == 0.1
    assert final["complete"] is True


def test_kill_fires_on_any_probe_below_500():
    rows = run_stream([850.0, 499.0])
    assert [r["action"] for r in rows] == ["hold", "kill_rollback"]
    assert rows[-1]["killed"] is True


def test_kill_can_fire_on_the_first_probe():
    rows = run_stream([400.0])
    assert rows[0]["action"] == "kill_rollback"
    assert rows[0]["killed"] is True


def test_killed_state_is_absorbing():
    state, action = step_ramp(RampState(), 400.0)
    assert action == "kill_rollback" and state.killed
    after, action2 = step_ramp(state, 900.0)
    assert action2 == "noop"
    assert after == state


def test_gate_boundaries_are_exact():
    # >= 800 is a hold (inclusive).
    _, action = step_ramp(RampState(), 800.0)
    assert action == "hold"
    # 500.0 is NOT a kill (< 500 strict) but IS a counter reset.
    _, action = step_ramp(RampState(), 500.0)
    assert action == "reset"
    # 499.999... kills.
    _, action = step_ramp(RampState(), 499.99)
    assert action == "kill_rollback"


def test_no_advance_past_the_final_rung():
    # Reach completion, then keep probing healthy medians: the rung must
    # stay pinned at the last index, never step out of BUDGET_RAMP.
    rows = run_stream([900.0] * 10)
    assert all(r["rung"] <= len(BUDGET_RAMP) - 1 for r in rows)
    assert rows[-1]["rung"] == len(BUDGET_RAMP) - 1
    assert rows[-1]["budget"] == 0.1
    assert rows[-1]["complete"] is True
    # Post-completion healthy probes are plain holds.
    assert rows[-1]["action"] == "hold"


def test_budget_for_maps_rungs_to_the_ramp():
    assert budget_for(RampState(rung=0)) == 0.3
    assert budget_for(RampState(rung=1)) == 0.2
    assert budget_for(RampState(rung=2)) == 0.1


def test_kill_still_armed_after_completion():
    stream = [900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 300.0]
    rows = run_stream(stream)
    assert rows[-1]["action"] == "kill_rollback"
    assert rows[-1]["killed"] is True


# ---------------------------------------------------------------------------
# Launch-path guard — the controller's phase-4/5 rows pin
# kl_anchor_loss_coef 0.0 ONTO whatever base they are given, so launching
# a hand-merged hardening-4b base through run_online_campaign.py
# --start-phase 4/5 silently zeroes the tether (the attempt-7 failure).
# check_armed_profile is the registered pre-launch gate that catches it.
# ---------------------------------------------------------------------------

def _phase(idx: int) -> dict:
    from scripts.run_online_campaign import PHASES
    return next(p for p in PHASES if p["idx"] == idx)


@pytest.mark.parametrize("block_id", RAMP_BLOCK_IDS_IN_ORDER)
def test_hand_merged_rungs_pass_the_armed_guard(
    blocks: dict, base_profile: dict, block_id: str,
):
    """The registered launch path — hand-merge exactly one rung block
    onto the base profile, hand the merged file straight to
    train_game.py — must produce a profile the guard calls armed."""
    merged = _deep_merge(base_profile, blocks[block_id]["override"])
    budget = blocks[block_id]["override"]["reinforce"]["adversary"][
        "budget_penalty"]
    assert check_armed_profile(merged, expected_budget=budget) == []


def test_hand_merged_h1_alone_passes_the_anchor_checks(
    blocks: dict, base_profile: dict,
):
    """H1 without a rung (anchor on, adversary still inert) is a valid
    registered state — the guard must not demand an armed adversary."""
    merged = _deep_merge(base_profile, blocks["H1_anchor_prior"]["override"])
    assert check_armed_profile(merged) == []


@pytest.mark.parametrize("phase_idx", [4, 5])
def test_controller_phase_overrides_zero_the_tether_and_guard_catches_it(
    blocks: dict, base_profile: dict, phase_idx: int,
):
    """THE DEFECT SCENARIO, pinned: operator hand-merges rung 1 onto the
    base, then (wrongly) relaunches via the controller at phase 4 or 5.
    build_phase_profile deep-merges the phase row ONTO the merged base —
    and both rows pin kl_anchor_loss_coef 0.0, zeroing the 0.3 tether.
    The guard must flag exactly that key so the launch dies preflight."""
    from scripts.run_online_campaign import build_phase_profile
    merged_base = _deep_merge(
        base_profile, blocks["H2_ramp_r1_budget_030"]["override"])
    prof = build_phase_profile(merged_base, _phase(phase_idx), 100_000_000)
    # The clobber is real: the controller row zeroed the tether.
    assert prof["reinforce"]["kl_anchor_loss_coef"] == 0.0
    violations = check_armed_profile(prof, expected_budget=0.3)
    assert violations, "guard must reject a controller-clobbered profile"
    assert any("kl_anchor_loss_coef" in v for v in violations)


def test_guard_flags_a_stale_prior_path(blocks: dict, base_profile: dict):
    """Anchoring to the stale A7 prior (the base profile's default)
    instead of the consolidated head is a distinct disarm mode: the
    guard must name kl_anchor_checkpoint."""
    merged = _deep_merge(
        base_profile, blocks["H2_ramp_r1_budget_030"]["override"])
    merged["reinforce"]["kl_anchor_checkpoint"] = (
        "checkpoints/smodice_1_2/abl_A7/vanilla_ppo_iter_00000.pt")
    violations = check_armed_profile(merged, expected_budget=0.3)
    assert any("kl_anchor_checkpoint" in v for v in violations)


def test_guard_flags_sam_rho_disabling_the_tether(
    blocks: dict, base_profile: dict,
):
    """ppo_updater applies the loss-level tether only when sam_rho <= 0;
    a profile that turns SAM on has silently disarmed the anchor even
    with coef 0.3 present."""
    merged = _deep_merge(
        base_profile, blocks["H2_ramp_r1_budget_030"]["override"])
    merged["reinforce"]["sam_rho"] = 0.05
    violations = check_armed_profile(merged, expected_budget=0.3)
    assert any("sam_rho" in v for v in violations)


def test_catalog_registers_the_controller_clobber_and_preflight(
    blocks: dict,
):
    """The yaml itself must carry the launch-path registration: name the
    phase-4/5 kl_anchor_loss_coef clobber, prohibit --start-phase
    launches, register the direct train_game.py path, and make the
    check_armed_profile preflight part of every override receipt."""
    text = CATALOG_PATH.read_text()
    assert "--start-phase" in text, (
        "catalog must explicitly prohibit the run_online_campaign.py "
        "--start-phase launch path"
    )
    assert "kl_anchor_loss_coef: 0.0" in text, (
        "catalog must name the phase-4/5 kl_anchor_loss_coef: 0.0 "
        "clobber, not just the entropy interaction"
    )
    assert "train_game.py" in text, (
        "catalog must register the direct trainer launch path"
    )
    assert "check_armed_profile" in text, (
        "catalog must register the pre-launch guard"
    )
    for block_id in sorted(OVERRIDE_BLOCK_IDS):
        assert "check_armed_profile" in blocks[block_id]["receipt"], (
            f"{block_id}.receipt must require the pre-launch "
            f"check_armed_profile verdict"
        )


# ---------------------------------------------------------------------------
# Re-anchor-on-resume source contract (trainer.py + kl_anchor.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trainer_src() -> str:
    return TRAINER_SRC.read_text()


def test_prior_is_rebuilt_from_config_path_every_construction(
    trainer_src: str,
):
    """The KLAnchor prior always loads from the CONFIG's
    kl_anchor_checkpoint — never from the resumed checkpoint blob — so
    pointing the resumed profile at a new checkpoint re-anchors the
    lineage. H1's documentation depends on exactly this."""
    assert '_rl_cfg.get("kl_anchor_checkpoint")' in trainer_src
    assert "checkpoint_path=str(_kl_ckpt)" in trainer_src


def test_actor_seeding_is_guarded_behind_fresh_start(trainer_src: str):
    """load_actor_into (seeding the live net's actor from the prior)
    must run only when iter_offset == 0: a RESUME keeps its trained
    weights and rebuilds only the prior. If this guard moves or
    disappears, the H1 block's semantics silently change from
    'tether the resumed policy to a snapshot of itself' to
    'overwrite the resumed policy with the snapshot'."""
    calls = [
        m.start() for m in re.finditer(
            r"\.load_actor_into\(net\)", trainer_src,
        )
    ]
    assert len(calls) == 1, (
        f"expected exactly one load_actor_into(net) call site in "
        f"trainer.py, found {len(calls)}"
    )
    # The guard must sit within the few lines directly above the call.
    window = trainer_src[:calls[0]].splitlines()[-4:]
    assert any("if iter_offset == 0:" in line for line in window), (
        "load_actor_into(net) is no longer guarded by "
        "'if iter_offset == 0:' within 3 lines above the call — the "
        "resume-rebuilds-prior-only contract is broken"
    )
