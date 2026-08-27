"""Purity quarantine guard for every `configs/*zelda*.yaml` profile.

Surfaced 2026-08-25 during an external research audit's review (not by a
Learned-ledger run hitting it): `configs/zelda.yaml`'s `ram_mapping:`
block was a disassembly-sourced parallel copy of the same addresses
`ZeldaReward` (nes_core/src/rewards.rs) declares as its own hardcoded
constants — the YAML block was never the reward's source, just an
undisclosed second copy with no `[VERIFIED: ...]` receipts. It now lives
in a `quarantined_external_knowledge:` block that nothing reads.

**Re-parametrised 2026-08-26** (quarantine-hardening pass). The original
version of this file hardcoded `PROFILE = configs/zelda.yaml` and tested
only that one file, even though five `configs/*zelda*.yaml` profiles
exist. That gap is exactly how `configs/zelda_gui_tuned.yaml` shipped a
live, int-valued 13-entry `ram_mapping:` block — six of its values the
exact quarantined win-chain addresses — that this test file would have
caught immediately had it ever looked at that file. It is fixed by the
same conversion `zelda.yaml` got (H4) and is now covered by the glob
below, not a second hardcoded constant.

Two more leaks the re-parametrisation surfaced, both reproduced live
against the running `nes_core` build: `configs/legend_of_zelda.yaml` and
`configs/zelda_roomfp.yaml` each declare zero `reward_weights` yet — pre-
fix — resolved to `ZeldaReward` and its quarantined win predicate purely
because their display name contains "Zelda" (`ram[0x0672] = 1` flipped
`(-0.001, False)` to `(19999.999, True)` with `episode_success() ==
True` on both). Both are now pinned to `reward_id: generic` and covered
by `test_reward_id_is_never_inferred_from_a_bare_name` below.

Full history: CLAIMS.md's Quarantine section.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"

QUARANTINE_KEY = "quarantined_external_knowledge"

QUARANTINED_ADDRESSES = (
    "0x0070", "0x0084", "0x00EB", "0x00EC", "0x10",
    "0x066F", "0x066D", "0x0658", "0x066E", "0x0671",
    "0x0672", "0x0609",
)

# The profiles this file is pinned to know about today, so a rename or a
# deletion out from under the glob fails loudly instead of just quietly
# parametrizing over fewer rows. A NEW `*zelda*.yaml` file appearing is
# fine and gets picked up automatically — that is the entire point of a
# glob instead of a hardcoded constant.
_EXPECTED_ZELDA_PROFILE_NAMES = {
    "legend_of_zelda.yaml",
    "zelda.yaml",
    "zelda_gui_tuned.yaml",
    "zelda_multidemo_overrides.yaml",
    "zelda_roomfp.yaml",
}


def _zelda_profile_paths() -> list[Path]:
    """Every top-level `configs/*zelda*.yaml`."""
    return sorted(CONFIG_DIR.glob("*zelda*.yaml"))


def _load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text())
    return doc if isinstance(doc, dict) else {}


ZELDA_PROFILE_PATHS = _zelda_profile_paths()


def test_the_glob_still_finds_every_known_zelda_profile() -> None:
    found = {p.name for p in ZELDA_PROFILE_PATHS}
    missing = _EXPECTED_ZELDA_PROFILE_NAMES - found
    assert not missing, (
        f"expected Zelda profile(s) missing from configs/: {missing} — "
        f"a rename must update this file's expectation, not silently "
        f"shrink what gets tested."
    )


# ---------------------------------------------------------------------------
# 1. no live ram_mapping anywhere in the family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ZELDA_PROFILE_PATHS, ids=lambda p: p.name)
def test_no_ram_mapping_block_remains(path: Path) -> None:
    """The one non-negotiable invariant across the WHOLE Zelda family: no
    profile whose filename says Zelda may carry a live, int-valued
    `ram_mapping:` block. `zelda_gui_tuned.yaml` was the sole violator in
    the entire `configs/` tree (13 entries, 6 the exact quarantined
    win-chain addresses) until the H4 fix converted it to the same
    string-valued `quarantined_external_knowledge` form `zelda.yaml` uses."""
    doc = _load(path)
    assert "ram_mapping" not in doc, (
        f"{path.name}: top-level `ram_mapping:` must not exist on a Zelda "
        f"profile — convert it to `{QUARANTINE_KEY}` (string-valued) or "
        f"drop it if it was never quarantined material."
    )


# ---------------------------------------------------------------------------
# 2. no display-name reward inheritance for a profile that asked for nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ZELDA_PROFILE_PATHS, ids=lambda p: p.name)
def test_reward_id_is_never_inferred_from_a_bare_name(path: Path) -> None:
    """BREACH PATH 1, extended to the whole family. A profile that
    declares no `reward_weights` of its own must not resolve to Zelda's
    reward arm — `reward_id` must be absent or explicitly `generic`.

    Reproduced live pre-fix on two files that both declare zero reward
    weights: `legend_of_zelda.yaml` and `zelda_roomfp.yaml` each went
    `(-0.001, False)` -> `(19999.999, True)` with `episode_success() ==
    True` on `ram[0x0672] = 1`, purely because their display name
    contains "Zelda". Only `legend_of_zelda.yaml` was named in the
    original adjudication; `zelda_roomfp.yaml` is a second instance this
    parametrisation found.
    """
    doc = _load(path)
    if doc.get("reward_weights"):
        pytest.skip(
            f"{path.name} declares its own reward_weights — "
            f"reward_id: zelda is a deliberate authored choice here, "
            f"not an inferred default."
        )
    reward_id = doc.get("reward_id")
    assert reward_id in (None, "generic"), (
        f"{path.name}: no reward_weights declared but reward_id="
        f"{reward_id!r} — a profile with no reward weights of its own "
        f"must not silently inherit a hand-authored win predicate."
    )


# ---------------------------------------------------------------------------
# 3. the quarantine block itself, wherever one is declared
# ---------------------------------------------------------------------------

def _profiles_with_quarantine_block() -> list[Path]:
    return [p for p in ZELDA_PROFILE_PATHS if QUARANTINE_KEY in _load(p)]


QUARANTINED_PROFILE_PATHS = _profiles_with_quarantine_block()


def test_at_least_one_zelda_profile_declares_the_quarantine_block() -> None:
    # A checker that quietly parametrizes over zero rows passes vacuously.
    assert QUARANTINED_PROFILE_PATHS, (
        "no configs/*zelda*.yaml profile declares "
        f"`{QUARANTINE_KEY}:` — the quarantine record itself is gone."
    )


@pytest.mark.parametrize("path", QUARANTINED_PROFILE_PATHS, ids=lambda p: p.name)
def test_profile_parses_and_declares_the_quarantine_block(path: Path) -> None:
    q = _load(path)[QUARANTINE_KEY]
    assert q["status"] == "UNVERIFIED_EXTERNAL"
    assert "aldonunez" in q["provenance"].lower() or "disassembly" in q["provenance"].lower()


@pytest.mark.parametrize("path", QUARANTINED_PROFILE_PATHS, ids=lambda p: p.name)
def test_rediscovery_rule_states_the_re_derivation_requirement(path: Path) -> None:
    rule = _load(path)[QUARANTINE_KEY]["rediscovery_rule"]
    assert "re-derived" in rule or "re-derive" in rule
    assert "receipt" in rule


@pytest.mark.parametrize("path", QUARANTINED_PROFILE_PATHS, ids=lambda p: p.name)
def test_quarantined_values_cannot_be_folded_into_an_address_map(path: Path) -> None:
    q = _load(path)[QUARANTINE_KEY]
    addr_keys = [k for k in q if k.startswith("q_")]
    assert len(addr_keys) >= len(QUARANTINED_ADDRESSES)
    for k in addr_keys:
        assert isinstance(q[k], str), (
            f"{path.name}: {k} is not a string — a tool folding this "
            f"block the way ram_mapping folds (int(v) over .values()) "
            f"would silently consume a quarantined address instead of "
            f"raising"
        )
        with pytest.raises((ValueError, TypeError)):
            int(q[k])


def test_the_two_quarantine_blocks_do_not_drift() -> None:
    """Every profile that declares a quarantine block must agree on the
    same set of `q_*` addresses. A second copy of a contamination record
    is a second thing that can drift; if it ever does, that is worth
    knowing about rather than silently tolerating."""
    by_profile = {}
    for path in QUARANTINED_PROFILE_PATHS:
        q = _load(path)[QUARANTINE_KEY]
        by_profile[path.name] = {
            k: v for k, v in q.items() if k.startswith("q_")
        }
    if len(by_profile) < 2:
        pytest.skip("fewer than two quarantine blocks exist — nothing to compare")
    names = list(by_profile)
    reference = by_profile[names[0]]
    for other in names[1:]:
        assert by_profile[other] == reference, (
            f"quarantine block in {other} disagrees with {names[0]}: "
            f"{by_profile[other]} != {reference}"
        )


# ---------------------------------------------------------------------------
# 4. boot contract survives the quarantine edits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path", [p for p in ZELDA_PROFILE_PATHS if _load(p).get("action_space")],
    ids=lambda p: p.name,
)
def test_standalone_profile_action_space_still_folds_to_bitmasks(path: Path) -> None:
    """Every Zelda profile that declares an action_space (i.e. every
    standalone profile, as opposed to an overlay fragment like
    zelda_multidemo_overrides.yaml) must still fold to NES bitmasks after
    the quarantine edits — the exact contract test_profile_configs.py
    pins for the GUI dropdown, scoped to this family so a quarantine edit
    can't silently break boot the way the pre-fix zelda_gui_tuned.yaml
    once did (it originally shipped with no action_space at all)."""
    from src.training.profile_utils import action_space_to_bitmasks

    action_space_to_bitmasks(_load(path)["action_space"])


@pytest.mark.parametrize(
    "path", [p for p in ZELDA_PROFILE_PATHS if _load(p).get("start_state_path")],
    ids=lambda p: p.name,
)
def test_declared_start_state_path_exists(path: Path) -> None:
    ssp = _load(path)["start_state_path"]
    assert (REPO / ssp).exists(), (
        f"{path.name}: start_state_path {ssp!r} does not exist — training "
        f"would silently fall back to the title-screen demo."
    )


# ---------------------------------------------------------------------------
# 5. nothing consumes the quarantine key; the reward stays independent
# ---------------------------------------------------------------------------

def test_no_source_file_reads_the_quarantine_key() -> None:
    # Excludes this test and its sibling quarantine test files, all of
    # which legitimately name the shared "quarantined_external_knowledge"
    # convention in their own source — not a read of any profile's block.
    excluded = {
        Path(__file__),
        REPO / "tests" / "test_metroid_purity_quarantine.py",
        REPO / "tests" / "test_purity_quarantine_sweep.py",
        REPO / "tests" / "test_onboard_game.py",
        # The engine sweep and its scanner. They read every quarantine
        # block in order to ENFORCE it against Rust/Python/tests — the
        # opposite of consuming it as an input — and live under tests/
        # precisely so no production path reads the key.
        REPO / "tests" / "test_purity_engine_sweep.py",
        REPO / "tests" / "purity_engine_scan.py",
    }
    hits = []
    for path in REPO.rglob("*.py"):
        if "/.venv/" in str(path) or "/worktrees/" in str(path):
            continue
        if path in excluded:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if QUARANTINE_KEY in text:
            hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"quarantine key read outside the profile: {hits}"


def test_reward_constants_are_independent_of_the_yaml_block() -> None:
    # ZeldaReward's addresses are compiled Rust constants, not read from
    # any profile YAML at all — confirm the claim the quarantine note
    # rests on by checking the ZeldaReward impl block specifically (the
    # file as a whole legitimately mentions "ram_mapping" in unrelated
    # comments about other games' configs, e.g. Contra's).
    rewards_rs = (REPO / "nes_core" / "src" / "rewards.rs").read_text()
    start = rewards_rs.index("impl ZeldaReward")
    end = rewards_rs.index("\nimpl ", start + 1)
    zelda_impl = rewards_rs[start:end]
    assert "ram_mapping" not in zelda_impl
