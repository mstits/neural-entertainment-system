"""Reward dispatch is driven by an explicit `reward_id`, never by the
profile's display name.

Why this file exists
--------------------
`build_reward` used to run sixteen `name.contains(...)` tests over
`profile["name"]`. Two consequences, both live and both verified before
this suite was written:

* configs/legend_of_zelda.yaml — 31 lines, no `reward_weights`, no
  `ram_mapping`, no address of any kind, self-described as "Not a
  training profile" — silently inherited `ZeldaReward` and with it the
  quarantined win predicate at RAM 0x0672. Flipping that one byte on an
  otherwise-zeroed 2KB buffer took it from `(-0.001, False)` to
  `(19999.999, True)` with `episode_success() == True`. The quarantine
  did not cost us Zelda; the substring dispatch did.
* configs/smb_4_4_micro.yaml — display name "SMB 4-4 micro (full
  controller)", no "mario" substring — got `GenericReward` while
  declaring nine MarioReward-only weights, on the flagship game, on a
  profile with banked live-show receipts.

The same mechanism was handing out a reward nobody asked for and
withholding one that was asked for. A display name is not a selector in
either direction.

Every test here fails on the pre-change tree. `test_no_display_name_token_selects_an_arm`
is the one that cannot be made to pass by a substring branch: re-adding
any `name.contains(...)` fails it, and deleting the id table fails its
partner assertion in the same parametrised case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import nes_core
from src.utils.reward_functions import build_reward_function

REPO = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO / "tests" / "reward_dispatch_baseline.json"


def test_extension_is_the_freshly_built_one():
    """Guard against the stale-.so trap.

    `make build` writes a wheel; the venv keeps an older
    `nes_core.abi3.so` unless it is copied over. Every assertion below
    measures the loaded binary, so a stale one would either fail the
    anti-substring tests against correct code or pass the roster lint
    against dispatch that never changed. Error loudly instead.
    """
    assert hasattr(nes_core, "reward_ids"), (
        "nes_core has no reward_ids() — the loaded extension predates this "
        "change. Run `make build`, then copy nes_core/target/release/"
        "libnes_core.dylib over the venv's nes_core.abi3.so."
    )


def valid_ids() -> tuple[str, ...]:
    """The id table, read from the loaded extension rather than kept as
    a second copy here. Deliberately a function, not a module constant:
    a stale extension must fail the individual tests with a readable
    message instead of blowing up collection for the whole file."""
    table = getattr(nes_core, "reward_ids", None)
    assert table is not None, (
        "nes_core has no reward_ids() — the loaded extension predates this "
        "change; see test_extension_is_the_freshly_built_one"
    )
    return tuple(table())


# The sixteen display-name tokens the deleted substring table matched
# on. This is a FROZEN HISTORICAL CONSTANT, not a live selector — it is
# the alarm, and it is documented never to grow. The valid-id half of
# the lint below comes from `nes_core.reward_ids()` instead, so adding a
# reward arm cannot leave that half silently behind.
LEGACY_NAME_TOKENS: tuple[tuple[str, str], ...] = (
    ("mario", "mario"),
    ("zelda", "zelda"),
    ("contra", "contra"),
    ("mega man", "mega_man"),
    ("megaman", "mega_man"),
    ("castlevania", "castlevania"),
    ("metroid", "metroid"),
    ("tetris", "tetris"),
    ("bubble bobble", "bubble_bobble"),
    ("bubblebobble", "bubble_bobble"),
    ("punch-out", "punch_out"),
    ("punch out", "punch_out"),
    ("punchout", "punch_out"),
    ("kung fu", "kung_fu"),
    ("gradius", "gradius"),
    ("excitebike", "excitebike"),
    ("excite bike", "excitebike"),
    ("ghosts", "ghosts"),
    ("ducktales", "ducktales"),
    ("kid icarus", "kid_icarus"),
    ("double dragon", "double_dragon"),
)

# The quarantined Zelda win-predicate byte (RAM_GANON_DEFEATED,
# provenance "aldonunez disassembly + Data Crystal", status
# UNVERIFIED_EXTERNAL). Named here only as the thing a generic profile
# must NOT react to.
QUARANTINED_GANON_BYTE = 0x0672


def _one_byte_jackpot(rf) -> tuple[float, bool, bool]:
    """Flip the quarantined byte on a zeroed 2KB RAM and report the
    reward delta, the done flag, and episode_success()."""
    ram = bytearray(2048)
    base, _, _ = rf.compute(bytes(ram))
    ram[QUARANTINED_GANON_BYTE] = 1
    after, done, _ = rf.compute(bytes(ram))
    return after - base, done, rf.episode_success()


def test_profile_declaring_no_reward_gets_the_generic_reward():
    """The one asked for: no declaration means the arm that cannot
    witness a clear, not the arm that can."""
    rf = build_reward_function({"name": "The Legend of Zelda", "reward_weights": {}})
    assert rf.kind == "generic"
    delta, done, success = _one_byte_jackpot(rf)
    assert not done, "one flipped RAM byte must not terminate a generic episode"
    assert not success, "a profile that declared no reward cannot witness a clear"
    assert abs(delta) < 1.0, f"no 20000 jackpot from one flipped byte (got {delta})"


def test_the_shipped_breach_profile_cannot_claim_a_zelda_win():
    """Same assertions, but against the file on disk rather than a
    synthetic dict — this is the profile that actually shipped."""
    profile = yaml.safe_load((REPO / "configs" / "legend_of_zelda.yaml").read_text())
    assert "reward_weights" not in profile or not profile["reward_weights"], (
        "fixture assumption: legend_of_zelda.yaml declares no reward weights"
    )
    rf = build_reward_function(profile)
    assert rf.kind == "generic"
    delta, done, success = _one_byte_jackpot(rf)
    assert not done and not success
    assert abs(delta) < 1.0


@pytest.mark.parametrize("token,arm", LEGACY_NAME_TOKENS)
def test_no_display_name_token_selects_an_arm(token, arm):
    """Name and id are independent in BOTH directions.

    Line one fails the moment anyone re-adds a `name.contains(...)`
    branch. Line two fails the moment the id table stops working. No
    implementation can pass both before and after this change.
    """
    named = build_reward_function({"name": f"Some {token} Game", "reward_weights": {}})
    assert named.kind == "generic", (
        f"display name containing {token!r} must select nothing; got {named.kind!r}"
    )
    identified = build_reward_function({"name": "totally unrelated", "reward_id": arm})
    assert identified.kind == arm


def test_unknown_reward_id_is_loud_not_a_silent_downgrade():
    with pytest.raises(ValueError) as exc:
        build_reward_function({"name": "x", "reward_id": "mrio"})
    message = str(exc.value)
    assert "mrio" in message
    for valid in valid_ids():
        assert valid in message, f"error must name the valid set; {valid!r} missing"


@pytest.mark.parametrize("declared", [None, "", "   "])
def test_absent_or_empty_reward_id_resolves_to_generic(declared):
    profile = {"name": "The Legend of Zelda"}
    if declared is not None:
        profile["reward_id"] = declared
    assert build_reward_function(profile).kind == "generic"


@pytest.mark.parametrize("spelling,arm", [
    ("Kung Fu", "kung_fu"),
    ("kung-fu", "kung_fu"),
    ("Punch-Out", "punch_out"),
    ("MARIO", "mario"),
])
def test_reward_id_is_normalised(spelling, arm):
    assert build_reward_function({"name": "n", "reward_id": spelling}).kind == arm


def test_every_advertised_reward_id_constructs():
    for rid in valid_ids():
        assert build_reward_function({"name": "n", "reward_id": rid}).kind == rid


# --------------------------------------------------------------------
# Roster lint / migration completeness.
# --------------------------------------------------------------------

def _roster():
    for path in sorted((REPO / "configs").rglob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("name"), str):
            yield path, doc


def _legacy_arm(name: str) -> str:
    """What the deleted substring table would have resolved `name` to.

    Frozen history, kept only so the lint can spot a config whose name
    would once have inherited an arm silently.
    """
    lowered = name.lower()
    for token, arm in LEGACY_NAME_TOKENS:
        if token not in lowered:
            continue
        if arm == "double_dragon" and ("battletoads" in lowered or " ii" in lowered):
            continue
        return arm
    return "generic"


def test_every_name_that_used_to_inherit_an_arm_now_declares_one():
    """A config whose display name matches a legacy token must say what
    it wants — including `generic`. Fails today on 126 rows; fails again
    the first time someone adds configs/mario_5_1.yaml without the key.
    """
    missing = [
        str(path.relative_to(REPO))
        for path, doc in _roster()
        if _legacy_arm(doc["name"]) != "generic" and "reward_id" not in doc
    ]
    assert not missing, (
        "these profiles would have inherited a reward from their display name "
        "and now declare nothing, so they silently get generic instead: "
        + ", ".join(missing)
    )


def test_declared_reward_ids_are_all_valid():
    declared = {
        str(path.relative_to(REPO)): doc["reward_id"]
        for path, doc in _roster()
        if "reward_id" in doc
    }
    # Anti-vacuity. On a roster where nothing declares the key this
    # check has nothing to reject and passes for free — the exact shape
    # of gate that has shipped green twice this week. Assert it has
    # something to work on before asserting it found nothing wrong.
    # Derived, not a magic constant: the frozen baseline is the list of
    # every profile that resolved to a specialized arm on migration day,
    # and each of those had to declare the key. A hardcoded count instead
    # couples this test to roster size and reds on any unrelated profile
    # being added or removed, with a misleading message.
    required = set(json.loads(BASELINE_PATH.read_text()))
    missing = sorted(rel for rel in required if rel not in declared)
    assert not missing, (
        f"{len(missing)} profile(s) from the frozen dispatch baseline no "
        f"longer declare reward_id: {missing[:10]}. This check would then "
        f"be running on a partly-unmigrated roster and passing vacuously"
    )
    bad = {rel: rid for rel, rid in declared.items() if rid not in valid_ids()}
    assert not bad, f"invalid reward_id values: {bad}"


def test_roster_dispatch_matches_the_frozen_pre_change_baseline():
    """The migration is behaviour-preserving, asserted row by row.

    tests/reward_dispatch_baseline.json was computed by replaying the
    old substring predicates over the roster BEFORE the Rust dispatch
    changed (scripts/migrate_reward_id.py --freeze). It is never
    regenerated against the new code — that would make this test
    tautological. Three rows are deliberate exceptions, listed below
    with what changed and why.
    """
    baseline = json.loads(BASELINE_PATH.read_text())

    # The intended behaviour changes in this migration: three profiles
    # that declare no reward_weights at all and were inheriting a
    # hand-authored win predicate purely from their display name. Each
    # was reproduced on the pre-change binary before being listed here.
    intended_changes = {
        # 31 lines, no reward_weights, no ram_mapping, self-described as
        # "Not a training profile". Pre-change: ram[0x0672] = 1 took it
        # from (-0.001, False) to (19999.999, True) with
        # episode_success() True.
        "configs/legend_of_zelda.yaml": ("zelda", "generic"),
        # Room-graph gate / solver profile, no reward_weights. Pre-change
        # it reproduced the identical jackpot: (-0.001, False) ->
        # (19999.999, True), episode_success() True on one flipped byte.
        "configs/zelda_roomfp.yaml": ("zelda", "generic"),
        # Same class: a room-graph gate profile with no reward_weights
        # inheriting MetroidReward — and its win predicate — off the
        # word "Metroid" in its title.
        "configs/metroid_roomfp.yaml": ("metroid", "generic"),
    }

    resolved = {}
    for path, doc in _roster():
        rel = str(path.relative_to(REPO))
        resolved[rel] = build_reward_function(doc).kind

    mismatches = []
    for rel, was in baseline.items():
        now = resolved.get(rel)
        if rel in intended_changes:
            expected_was, expected_now = intended_changes[rel]
            assert was == expected_was and now == expected_now, (
                f"{rel}: intended change is {expected_was} -> {expected_now}, "
                f"saw {was} -> {now}"
            )
            continue
        if now != was:
            mismatches.append(f"{rel}: was {was}, now {now}")
    assert not mismatches, "migration changed a reward arm:\n" + "\n".join(mismatches)

    # And nothing outside the baseline picked up a specialized arm.
    gained = [
        f"{rel}: now {arm}"
        for rel, arm in resolved.items()
        if arm != "generic" and rel not in baseline
    ]
    assert not gained, "profile gained a specialized reward it never had:\n" + "\n".join(gained)


def test_baseline_covers_the_whole_roster_and_is_not_empty():
    baseline = json.loads(BASELINE_PATH.read_text())
    # 126 rows were frozen at the migration commit (c89a816) — that count
    # is history and must never be REGENERATED to paper over a real
    # drift. It is legitimate to APPEND rows for profiles authored AFTER
    # the freeze that deliberately declare a specialized reward_id: they
    # have no "pre-change" resolution to preserve (they didn't exist
    # pre-change), so test_roster_dispatch_matches_the_frozen_pre_change_
    # baseline's "gained" check would otherwise reject every legitimate
    # new specialized profile forever. Each append below is one
    # documented batch, not a silent bump.
    # +4: configs/mario_1_1_v31_redo_seed{0..3}.yaml
    #     (V31_REDO_SURGICAL_2026-08-27.md) — explicit reward_id: mario,
    #     the v27/v27 sibling recipe's own arm, single-variable redo_tau
    #     diff from the already-baselined v27 seed configs.
    # +1: configs/mario_1_1_v32_redo_bk_seed0.yaml
    #     (V32_REDO_BOTTOM_K_2026-08-28.md) — explicit reward_id: mario,
    #     the same v27 recipe with the ReDo selection rule as its single
    #     functional diff (threshold -> rank-based bottom-k). Seeds 1-3
    #     are owed and will be one further documented batch.
    # +5: configs/mario_1_1_v32_redo_bk_seed{1,2,3}.yaml plus the two
    #     Phase R preflight profiles configs/mario_1_1_v32_redo_bk_
    #     phase_r.yaml (rung 1, k=2/C=5) and ..._phase_r2.yaml (rung 2,
    #     k=4/C=10, the V32 §8 escalation). This is the "seeds 1-3 are
    #     owed" batch promised directly above, settled together with the
    #     two preflight profiles that were authored alongside them. All
    #     five are byte-identical to the already-baselined seed0 profile
    #     except `name` and the registered redo numerals, and all five
    #     declare `reward_id: mario` explicitly.
    expected = 126 + 4 + 1 + 5
    assert len(baseline) == expected, (
        f"expected {expected} baseline rows (126 frozen at migration + "
        f"documented later appends); found {len(baseline)}. A frozen row "
        "changing value is history and must never be regenerated; a "
        "brand-new specialized profile is fine to append but must be "
        "documented here, in one batch per addition, with the "
        "registration that authorized it."
    )
    for rel, arm in baseline.items():
        assert (REPO / rel).exists(), f"baseline names a config that no longer exists: {rel}"
        assert arm in valid_ids(), f"baseline names an arm that is not a valid id: {arm}"


def test_a_gate_profile_with_no_weights_cannot_inherit_a_win_predicate():
    """The general form of the breach, not just the one instance.

    A profile that declares no `reward_weights` and no `reward_id` must
    not be able to witness a clear off a single RAM byte. Both room-graph
    gate profiles did, pre-change, purely because of their titles.
    """
    for rel in ("configs/zelda_roomfp.yaml", "configs/metroid_roomfp.yaml"):
        doc = yaml.safe_load((REPO / rel).read_text())
        assert not doc.get("reward_weights"), f"fixture assumption: {rel} has no weights"
        rf = build_reward_function(doc)
        assert rf.kind == "generic", rel
        delta, done, success = _one_byte_jackpot(rf)
        assert not done and not success, rel
        assert abs(delta) < 1.0, rel


def test_smb_profiles_keep_the_arm_their_banked_runs_trained_on():
    """SMB training output is banked; its reward must not move.

    Spelled out per-file rather than left to the roster lint because
    these are the profiles a regression would cost the most.
    """
    expected = {
        "configs/mario.yaml": "mario",
        "configs/mario_canonical.yaml": "mario",
        "configs/mario_tiles.yaml": "mario",
        "configs/mario_1_2_javi.yaml": "mario",
        "configs/mario_1_2_consol2.yaml": "mario",
        "configs/lost_levels.yaml": "mario",
        # Preserved, NOT blessed: these two run GenericReward today
        # because their display name has no "mario" in it, and their
        # nine declared MarioReward-only weights are inert. Pinned here
        # so the defect cannot be quietly "fixed" without a measured
        # before/after on a profile with live-show receipts.
        "configs/smb_4_4_micro.yaml": "generic",
        "configs/smb_blank_slate.yaml": "generic",
    }
    for rel, arm in expected.items():
        path = REPO / rel
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text())
        assert build_reward_function(doc).kind == arm, f"{rel} must resolve to {arm}"
