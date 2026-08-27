"""THE YAML QUARANTINE IS CLEAN. THE ENGINE IS NOT. This closes the gap.

The 994-entry config sweep (0557896) quarantined 7 addresses and named its
own scope limit in the same commit message:

    "Quarantining the YAML retracts the DOCUMENTATION claim, NOT the
     Rust constant."

`tests/test_purity_quarantine_sweep.py` guards `configs/`. It cannot see
`nes_core/src/rewards.rs`, where `RAM_MATCH_ID = 0x0001` still drove
Punch-Out's `won` / `done` / `episode_success()` and
`RAM_BOSS_HEALTH = 0x06C1` still drove Mega Man's boss term, both
quarantined in YAML and both untouched in code. It also could not see that
`impl ZeldaReward` hardcodes ALL TWELVE of `configs/zelda.yaml`'s
quarantined addresses — the largest single finding of the engine sweep, and
one nobody had written down anywhere.

This module is the check that runs without anyone deciding to look. Three
purity sweeps landed in two days (Zelda, the 994-entry config sweep, this
one); each found real breaches, and each happened because a person chose to
look. Seven vacuous gates have shipped here, the most recent written by
work that was holding the previous six in its brief. A text rule
demonstrably does not work. So:

  * `tests/purity_engine_scan.py` derives the quarantined address set
    from the `quarantined_external_knowledge:` blocks themselves, derives
    which reward arm owns each ROM from the source's own dispatch table,
    and finds every place a quarantined address is still reachable by
    running code — Rust, Python, tests.
  * Every ENFORCED site (one in the executing layer) must appear in
    `docs/purity/engine_quarantine_disclosures.yaml` with a non-vacuous
    justification. An undeclared site FAILS. A stale declaration FAILS.
    The site count may be LOWERED, never raised.

FALSE POSITIVES ARE THE FAILURE MODE THAT MATTERS. A check that cries wolf
gets disabled inside a week and is worse than nothing. Two scoping rules do
the work, and both are load-bearing on the current tree:

  TYPE-SCOPED. Only `const NAME: usize = 0x..` counts as an address in
  Rust. `SONG_ENDING: u8 = 0x10` sits four lines from
  `RAM_DUNGEON_LEVEL: usize = 0x10` in the same impl, carries the same
  value, and is a VALUE not an address — `test_a_u8_value_constant_is_
  never_mistaken_for_an_address` pins that it is not flagged.

  OWNER-SCOPED. An address is only checked against code that owns its ROM.
  `apu.rs` masks with `& 0x0001` and `nes.rs` writes `prg[0x0001]`;
  `tests/test_button_bit_alignment.py` discusses "0x0000..0x0001" in a
  comment about controller bits. None are Punch-Out and none are flagged —
  `test_unowned_uses_of_a_quarantined_value_are_not_flagged` pins it.

AND TESTS ARE GUARDS, NOT CLAIMS. `tests/test_zelda_purity_quarantine.py`
writes `ram[0x0672] = 1` to PROVE the reward cannot be steered by it. That
is the mechanism holding the quarantine shut. Failing on it would make the
check fight its own guards, so test-kind sites are REPORTED and not
enforced. That is a deliberate scope decision, not an oversight, and
`test_reported_kinds_are_still_visible` keeps them from vanishing.

ANTI-VACUITY. Every predicate here is mutation-tested against a synthetic
tree, and `test_actual_revert_*` reconstructs the pre-fix state and counts
the assertions that fail — including the half-fix case the predecessor
sweep caught, where restoring only ONE of a sibling pair still failed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from purity_engine_scan import (  # noqa: E402
    DISCLOSURES,
    ENFORCED_KINDS,
    REQUIRED_FIELDS,
    VACUOUS,
    VALID_DISPOSITIONS,
    Quarantine,
    check_disclosures,
    load_disclosures,
    quarantines,
    scan_python,
    scan_quarantined_uses,
    scan_rust,
    witness_ledger,
)


# =========================================================================
# Fixtures — a synthetic tree the mutation tests can contaminate
# =========================================================================

MINIMAL_RUST = '''\
pub enum Reward { Fake(FakeReward) }
impl Reward {
    pub fn reward_id(&self) -> &'static str {
        match self { Reward::Fake(_) => "fake", }
    }
}
pub struct FakeReward { x: u8 }
impl FakeReward {
    const RAM_SAFE: usize = 0x0123;
    const MASK: u8 = 0x0077;
    pub fn compute(&mut self, ram: &[u8]) -> u8 { ram[Self::RAM_SAFE] }
}
'''

MINIMAL_CONFIG = """\
reward_id: fake
quarantined_external_knowledge:
  applies_to_rom: roms/Fake Game (USA).nes
  provenance: an external RAM map
  status: UNVERIFIED_EXTERNAL
  rediscovery_rule: re-derive it differentially
  q_thing: "0x0077"
"""


@pytest.fixture()
def synth(tmp_path: Path) -> Path:
    """A tiny self-contained tree with one quarantine and one reward arm.

    `q_thing` is 0x0077, which the Rust carries as `MASK: u8` — a VALUE.
    So a correct checker reports ZERO sites here, and each mutation below
    introduces exactly one thing it must catch.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "fake.yaml").write_text(MINIMAL_CONFIG)
    src = tmp_path / "nes_core" / "src"
    src.mkdir(parents=True)
    (src / "rewards.rs").write_text(MINIMAL_RUST)
    return tmp_path


# =========================================================================
# 1. The quarantine record is derived, and it is not empty
# =========================================================================

def test_the_quarantine_set_is_derived_and_non_empty() -> None:
    """If this ever returns nothing, every check below passes vacuously.

    That is failure mode #1 for a sweep like this: delete the blocks and
    the guard evaporates. This test is the tripwire.
    """
    qs = quarantines()
    assert qs, (
        "no quarantined addresses found. Either every "
        "`quarantined_external_knowledge:` block was deleted, or the "
        "block's value shape changed and this checker went blind.")
    addrs = {q.addr for q in qs}
    for named in (0x0001, 0x06C1, 0x0672, 0x0030, 0x003B, 0x04A0):
        assert named in addrs, (
            f"0x{named:04X} is no longer in any quarantine block. It was "
            f"retracted for asserting semantics tied to an event with no "
            f"witness here; a quarantine is not lifted by deleting it.")


def test_every_quarantine_block_resolves_to_a_reward_arm() -> None:
    """Ownership is what keeps this check from crying wolf, so a block
    whose config declares no `reward_id` would silently scope to nothing.
    Fail instead — over-reporting beats going quiet."""
    unscoped = [f"{q.config}:{q.key}" for q in quarantines() if not q.reward_id]
    assert not unscoped, (
        f"quarantine entries with no owning `reward_id`: {unscoped}. Without "
        f"one, the Rust scan cannot tell which impl block owns the address "
        f"and the entry is checked against nothing.")


# =========================================================================
# 2. THE CHECK — no quarantined address is live and undeclared
# =========================================================================

def _declared() -> dict[str, dict]:
    doc = load_disclosures()
    return {str(r["site"]): r for r in (doc.get("sites") or [])
            if isinstance(r, dict) and r.get("site")}


def test_every_live_quarantined_address_is_declared() -> None:
    """THE HEADLINE. A quarantined address reachable by running code must
    be recorded, with its justification, in the disclosure inventory.

    This is what fires when someone adds a new live use, or quarantines an
    address that is already live somewhere in the tree. Nobody has to
    decide to look.
    """
    declared = _declared()
    missing = [s for s in scan_quarantined_uses()
               if s.enforced and s.site_id not in declared]
    assert not missing, (
        "quarantined address(es) live in the executing layer with NO "
        "disclosure:\n" + "\n".join(
            f"  {s.file}:{','.join(map(str, s.lines))}  {s.symbol} = "
            f"0x{s.addr:04X}  (quarantined by {'; '.join(s.quarantines)})"
            for s in missing) +
        f"\n\nEither remove the use, or add a row to "
        f"{DISCLOSURES.relative_to(REPO)} stating what it ASSERTS, what "
        f"event has NO WITNESS here, and what would EARN it back. "
        f"Quarantining the YAML retracts the documentation claim, not the "
        f"constant — this is the constant.")


def test_no_disclosure_is_stale() -> None:
    """A declaration for a site that no longer exists is how a disclosure
    record rots into a blanket exemption: the rows outlive the code, and
    the next real breach lands on a site some ghost row already 'covers'.
    """
    live = {s.site_id for s in scan_quarantined_uses() if s.enforced}
    stale = sorted(set(_declared()) - live)
    assert not stale, (
        f"disclosure rows with no matching live site: {stale}. The use was "
        f"removed (good) or renamed. Delete the row and lower "
        f"`max_enforced_sites`.")


def test_the_site_count_ratchets_downward_only() -> None:
    """`max_enforced_sites` may be lowered when a site is removed. Raising
    it is the exact move this file exists to make visible, so it has to
    cost a deliberate edit with a number in it."""
    doc = load_disclosures()
    cap = doc.get("max_enforced_sites")
    assert isinstance(cap, int), (
        "`max_enforced_sites:` must be an int — it is the ratchet.")
    live = [s for s in scan_quarantined_uses() if s.enforced]
    assert len(live) <= cap, (
        f"{len(live)} enforced sites against a cap of {cap}. A quarantined "
        f"address became live in code. Remove the use; do not raise the cap.")


def test_every_disclosure_is_non_vacuous() -> None:
    """Seven vacuous gates have shipped here. A disclosure whose
    justification reads "TODO" is the eighth."""
    problems: list[str] = []
    for site, row in sorted(_declared().items()):
        for f in REQUIRED_FIELDS:
            v = str(row.get(f, "")).strip()
            if not v or v.lower() in VACUOUS:
                problems.append(f"{site}: `{f}` is empty or a placeholder")
        d = str(row.get("disposition", "")).strip()
        if d and d not in VALID_DISPOSITIONS:
            problems.append(f"{site}: unknown disposition {d!r}")
        # A justification shorter than a sentence is not a justification.
        for f in ("asserts", "no_witness"):
            if 0 < len(str(row.get(f, "")).strip()) < 25:
                problems.append(f"{site}: `{f}` is too short to justify anything")
    assert not problems, "vacuous disclosure rows:\n  " + "\n  ".join(problems)


def test_live_unretracted_rows_state_what_would_earn_the_claim_back() -> None:
    """The rediscovery rule is the whole exit path. A LIVE_UNRETRACTED row
    that does not name the observation which would lift it is a permanent
    exemption wearing a disclosure's clothes."""
    bad = [s for s, r in _declared().items()
           if r.get("disposition") == "LIVE_UNRETRACTED"
           and len(str(r.get("earns_it", "")).strip()) < 25]
    assert not bad, (
        f"LIVE_UNRETRACTED rows with no usable `earns_it`: {bad}")


# =========================================================================
# 3. FALSE-POSITIVE CONTROL — pinned, because this is what kills a check
# =========================================================================

def test_a_u8_value_constant_is_never_mistaken_for_an_address() -> None:
    """`SONG_ENDING: u8 = 0x10` sits in the same impl as
    `RAM_DUNGEON_LEVEL: usize = 0x10`, carries the same value, and is a
    VALUE. Flagging it would be a false positive inside the very file this
    check exists to audit."""
    flagged = {s.symbol for s in scan_quarantined_uses()}
    assert "SONG_ENDING" not in flagged
    assert "DIRECTIONAL_MASK" not in flagged
    assert "RAM_DUNGEON_LEVEL" in flagged, (
        "the usize sibling at the same value must still be caught — "
        "otherwise this test is passing for the wrong reason")


def test_unowned_uses_of_a_quarantined_value_are_not_flagged() -> None:
    """0x0001 is quarantined for Punch-Out. It is also an APU shift-register
    mask, a PRG byte index, and a controller-bit comment. Owner-scoping is
    what keeps those three out."""
    files = {s.file for s in scan_quarantined_uses()}
    for innocent in ("nes_core/src/apu.rs", "nes_core/src/nes.rs",
                     "tests/test_button_bit_alignment.py"):
        assert innocent not in files, (
            f"{innocent} was flagged. It does not own any quarantined ROM; "
            f"scoping has broken and this check will now cry wolf.")


def test_the_scan_is_clean_on_a_tree_with_nothing_to_find(synth: Path) -> None:
    """The synthetic tree quarantines 0x0077, which the Rust carries only
    as a `u8` mask. A correct checker finds nothing. If this ever fails,
    every mutation test below is passing on noise."""
    assert scan_quarantined_uses(repo=synth) == []


def test_reported_kinds_are_still_visible() -> None:
    """Test-kind sites are deliberately not enforced. They must still be
    REPORTED, or the decision quietly becomes 'tests are invisible' and the
    next breach hides in a test helper."""
    sites = scan_quarantined_uses()
    reported = [s for s in sites if not s.enforced]
    assert reported, (
        "no reported-only sites at all. Either the tests that guard the "
        "quarantine were deleted, or the kind split stopped working.")
    assert any(s.file == "tests/test_zelda_purity_quarantine.py"
               for s in reported), (
        "the Zelda quarantine guard no longer shows up in the scan")


# =========================================================================
# 4. MUTATION TESTS — the predicates must actually bite
# =========================================================================

def test_a_new_rust_const_on_a_quarantined_address_is_caught(synth: Path) -> None:
    rs = synth / "nes_core" / "src" / "rewards.rs"
    rs.write_text(rs.read_text().replace(
        "    const RAM_SAFE: usize = 0x0123;",
        "    const RAM_SAFE: usize = 0x0123;\n"
        "    const RAM_THING: usize = 0x0077;"))
    hits = scan_quarantined_uses(repo=synth)
    assert [h.symbol for h in hits] == ["RAM_THING"]
    assert hits[0].enforced and hits[0].kind == "rust-const"


def test_a_direct_ram_subscript_is_caught(synth: Path) -> None:
    """Deleting the named constant and inlining the literal is the obvious
    way to slip past a name-based check."""
    rs = synth / "nes_core" / "src" / "rewards.rs"
    rs.write_text(rs.read_text().replace(
        "ram[Self::RAM_SAFE]", "ram[0x0077]"))
    hits = scan_quarantined_uses(repo=synth)
    assert len(hits) == 1 and hits[0].kind == "rust-index"


def test_an_address_array_entry_is_caught(synth: Path) -> None:
    """`const RAM_SCORE: [usize; 5] = [...]` is a real shape in this file;
    a quarantined address could hide inside one."""
    rs = synth / "nes_core" / "src" / "rewards.rs"
    rs.write_text(rs.read_text().replace(
        "    const RAM_SAFE: usize = 0x0123;",
        "    const RAM_SAFE: usize = 0x0123;\n"
        "    const RAM_BANK: [usize; 2] = [0x0123, 0x0077];"))
    assert [h.symbol for h in scan_quarantined_uses(repo=synth)] == ["RAM_BANK"]


def test_moving_the_constant_to_a_helper_module_does_not_evade_the_check(
        synth: Path) -> None:
    """The obvious evasion of an owner-scoped check: move the constant out
    of the owned `impl` into a module with no reward dispatch, where
    ownership cannot be derived. `RAM_*: usize` is this codebase's own
    naming convention for a RAM address, and it is flagged there whichever
    game it belongs to."""
    helper = synth / "nes_core" / "src" / "helper.rs"
    helper.write_text("const RAM_SNEAKY: usize = 0x0077;\n")
    hits = scan_quarantined_uses(repo=synth)
    assert [(h.file, h.symbol, h.kind) for h in hits] == [
        ("nes_core/src/helper.rs", "RAM_SNEAKY", "rust-unowned")]
    assert hits[0].enforced


def test_the_unowned_branch_does_not_fire_on_a_non_ram_constant(
        synth: Path) -> None:
    """Scoping in the unowned branch rests entirely on the `RAM_` prefix.
    A size or a bank constant carrying the same value must not flag —
    `nes_core/src/memory.rs` has `RAM_SIZE: usize = 0x0800` and the
    mappers are full of `PRG_BANK: usize = 0x2000`."""
    helper = synth / "nes_core" / "src" / "helper.rs"
    helper.write_text("const SCRATCH_LEN: usize = 0x0077;\n"
                      "const PRG_BANK: usize = 0x0077;\n")
    assert scan_quarantined_uses(repo=synth) == []


def test_a_python_site_is_caught(synth: Path) -> None:
    (synth / "fake_runner.py").write_text(
        "# reward_id fake\n"
        "def go(ram):\n"
        "    return ram[0x0077]\n")
    hits = scan_quarantined_uses(repo=synth)
    assert len(hits) == 1 and hits[0].kind == "python" and hits[0].enforced


def test_a_python_site_in_an_unowned_module_is_not_caught(synth: Path) -> None:
    """Same literal, same shape, no mention of the owning game. This is the
    `test_button_bit_alignment.py` case in miniature."""
    (synth / "unrelated.py").write_text("def go(ram):\n    return ram[0x0077]\n")
    assert scan_quarantined_uses(repo=synth) == []


def test_a_renamed_reward_arm_does_not_escape_the_scan(synth: Path) -> None:
    """Ownership is read from the source's own `Reward::X(_) => "id"`
    dispatch, so renaming the struct cannot orphan the check — as long as
    the rename is consistent, which the compiler enforces."""
    rs = synth / "nes_core" / "src" / "rewards.rs"
    rs.write_text(rs.read_text()
                  .replace("FakeReward", "RenamedReward")
                  .replace("Reward::Fake(", "Reward::Renamed(")
                  .replace("Fake(FakeReward)", "Renamed(RenamedReward)")
                  .replace("const RAM_SAFE: usize = 0x0123;",
                           "const RAM_SAFE: usize = 0x0077;"))
    assert [h.symbol for h in scan_quarantined_uses(repo=synth)] == ["RAM_SAFE"]


def test_deleting_a_quarantine_block_is_not_a_way_to_pass(synth: Path) -> None:
    """The scan derives its address set from the blocks, so deleting one
    empties the set and everything passes. `test_the_quarantine_set_is_
    derived_and_non_empty` is the tripwire for that on the real tree; here
    we pin that the emptiness is at least visible."""
    (synth / "configs" / "fake.yaml").write_text("reward_id: fake\n")
    assert quarantines(synth / "configs", synth) == []


# =========================================================================
# 5. VERIFY BY ACTUAL REVERT
#
# The predecessor sweep verified its guard by restoring all five
# quarantined configs (14/14 new assertions failed) and then by restoring
# only ONE of a sibling pair (still failed 7, catching a half-fix). Same
# bar here: reconstruct the pre-fix world and count what fails.
# =========================================================================

def _assertions_failing_against(repo: Path, disclosure_doc: dict) -> int:
    """Run every enforcement predicate against a reconstructed tree and
    count the failures, instead of asserting one and stopping."""
    declared = {str(r["site"]): r for r in (disclosure_doc.get("sites") or [])
                if isinstance(r, dict) and r.get("site")}
    live = [s for s in scan_quarantined_uses(repo=repo) if s.enforced]
    failures = 0
    # (a) undeclared live sites
    failures += sum(1 for s in live if s.site_id not in declared)
    # (b) stale declarations
    failures += len(set(declared) - {s.site_id for s in live})
    # (c) the ratchet
    cap = disclosure_doc.get("max_enforced_sites")
    if not isinstance(cap, int) or len(live) > cap:
        failures += 1
    # (d) vacuous rows
    for row in declared.values():
        if any(not str(row.get(f, "")).strip()
               or str(row.get(f, "")).strip().lower() in VACUOUS
               for f in REQUIRED_FIELDS):
            failures += 1
    return failures


@pytest.fixture()
def real_tree(tmp_path: Path) -> Path:
    """A copy of the parts of the real tree this checker reads. Copying
    means a revert experiment cannot touch the working tree."""
    dst = tmp_path / "tree"
    (dst / "nes_core").mkdir(parents=True)
    shutil.copytree(REPO / "configs", dst / "configs")
    shutil.copytree(REPO / "nes_core" / "src", dst / "nes_core" / "src")
    for rel in ("docs/receipts/games/zelda_hp_ladder_probe.py",
                "scripts/tracing/nes_core_nmi_trace.py"):
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, p)
    return dst


def test_the_real_tree_passes_every_enforcement_predicate(real_tree: Path) -> None:
    """Baseline. If this is not 0, the revert counts below mean nothing."""
    assert _assertions_failing_against(real_tree, load_disclosures()) == 0


def test_actual_revert_deleting_the_disclosure_record_fails_loudly(
        real_tree: Path) -> None:
    """Revert #1: the whole inventory goes. Every live site becomes
    undeclared and the ratchet has no number."""
    n = _assertions_failing_against(real_tree, {})
    assert n >= 17, (
        f"deleting the entire disclosure record produced only {n} failures. "
        f"The check is not holding the inventory up.")


def test_actual_revert_dropping_one_row_catches_a_half_fix(
        real_tree: Path) -> None:
    """Revert #2, the predecessor's half-fix case. `RAM_LINK_X` and
    `RAM_LINK_Y` are siblings introduced by the same external map. Drop the
    disclosure for exactly one and the other must NOT cover for it."""
    doc = load_disclosures()
    assert doc.get("sites"), (
        f"{DISCLOSURES.relative_to(REPO)} has no `sites:` — the disclosure "
        f"record is missing or empty, so there is no half-fix to catch. "
        f"Restore it before reading this result.")
    for victim in ("nes_core/src/rewards.rs::RAM_LINK_X::0x0070",
                   "nes_core/src/rewards.rs::RAM_MATCH_ID::0x0001",
                   "nes_core/src/rewards.rs::RAM_BOSS_HEALTH::0x06C1"):
        half = dict(doc)
        half["sites"] = [r for r in doc["sites"] if r.get("site") != victim]
        assert _assertions_failing_against(real_tree, half) == 1, (
            f"dropping the disclosure for {victim} alone did not fail "
            f"exactly one assertion — a sibling row is covering for it.")


def test_actual_revert_reintroducing_the_punchout_latch_fails(
        real_tree: Path) -> None:
    """Revert #3: the concrete breach the workflow was called for. Add a
    SECOND live use of the quarantined Punch-Out latch — the shape a future
    refactor would take — and it must be caught undeclared."""
    rs = real_tree / "nes_core" / "src" / "rewards.rs"
    text = rs.read_text()
    assert "const RAM_MATCH_ID: usize = 0x0001;" in text
    rs.write_text(text.replace(
        "    const RAM_MATCH_ID: usize = 0x0001;",
        "    const RAM_MATCH_ID: usize = 0x0001;\n"
        "    const RAM_WIN_LATCH_MIRROR: usize = 0x0001;"))
    assert _assertions_failing_against(real_tree, load_disclosures()) >= 2, (
        "a new live use of the quarantined win latch, plus the ratchet "
        "breach it causes, went undetected")


def test_actual_revert_a_newly_quarantined_live_address_fails(
        real_tree: Path) -> None:
    """Revert #4, the forward-looking one. Quarantine an address that is
    ALREADY live in code — Castlevania's $01A9, sourced in-comment to "Data
    Crystal + tasvideos ... not yet Dracula-fight-verified" — and the check
    must fire without anyone touching the Rust.

    This is the case that makes the check useful rather than historical:
    the next quarantine will land on live code, and it will fail on the
    same day it lands.
    """
    cfg = real_tree / "configs" / "castlevania.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["quarantined_external_knowledge"]["q_boss_health_live"] = "0x01A9"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
    sites = [s for s in scan_quarantined_uses(repo=real_tree)
             if s.addr == 0x01A9 and s.enforced]
    assert sites, (
        "quarantining Castlevania's live $01A9 did not surface the Rust "
        "constant that reads it. The check would not notice the NEXT "
        "quarantine landing on live code.")
    assert _assertions_failing_against(real_tree, load_disclosures()) >= 1


# =========================================================================
# 6. THE WITNESS LEDGER, derived from evidence
#
# A semantic claim tied to an event this project has never witnessed is
# injected knowledge, because it could not have been measured here. That
# discriminant needs a denominator: which games HAVE a witnessed clear.
# `witness_ledger()` derives one from the solver's own tapes rather than
# from anybody's memory.
#
# The tapes are not tracked by git (`runs/` is ignored, with 840 files
# force-added and zero of them solution tapes), so the real-tree assertions
# skip when the data is absent. The DISCRIMINANT is tested unconditionally
# on a synthetic fixture, so the mechanism can never rot behind a skip.
# =========================================================================

def _fake_run(root: Path, tree: str, rom_state: str, tapes: list[dict]) -> None:
    d = root / tree
    (d / "solutions").mkdir(parents=True)
    (d / "roots.json").write_text(json.dumps(
        {"entrance": {"path": rom_state, "start_wd": [0]}}))
    for i, t in enumerate(tapes):
        (d / "solutions" / f"sol_{i:03d}.json").write_text(json.dumps(t))


def test_the_witness_discriminant_rejects_a_witnessless_tape(tmp_path: Path) -> None:
    """THE load-bearing rule. Kirby's and Double Dragon's banked `sol_000`
    tapes are WITHDRAWN FALSE POSITIVES: both carry `start_wd []` and
    `clear_wd []`, fired by the confluence detector with no independent
    witness. A ledger that counted those would launder exactly the two
    fake clears this project already had to retract."""
    runs = tmp_path / "runs"
    _fake_run(runs, "good", "roms/Real Game (USA)_start.state.bin",
              [{"start_wd": [0], "clear_wd": [1]}])
    _fake_run(runs, "empty_wd", "roms/Fake Clear (USA)_start.state.bin",
              [{"start_wd": [], "clear_wd": []}])
    _fake_run(runs, "unchanged_wd", "roms/No Delta (USA)_start.state.bin",
              [{"start_wd": [3], "clear_wd": [3]}])

    led = {r["rom"]: r for r in witness_ledger(runs)["roms"]}
    assert led["Real Game (USA)"]["witnessed"] is True
    assert led["Fake Clear (USA)"]["witnessed"] is False, (
        "an empty-witness tape counted as a clear — this is the Kirby / "
        "Double Dragon false positive, reintroduced")
    assert led["No Delta (USA)"]["witnessed"] is False, (
        "a tape whose witness observable never moved counted as a clear")
    assert led["Real Game (USA)"]["evidence"], "a witness must cite its tape"


def test_the_ledger_attributes_a_chained_run_to_its_rom(tmp_path: Path) -> None:
    """Chained levels mint their entrance from the previous clear, so
    `roots.json` points at a minted state rather than at `roms/`. Without
    inheritance the whole chain — which is most of the evidence — falls out
    of the ledger and every chained game reads as unwitnessed."""
    runs = tmp_path / "runs"
    _fake_run(runs, "camp/boot", "roms/Chained (USA)_start.state.bin",
              [{"start_wd": [0], "clear_wd": [1]}])
    _fake_run(runs, "camp/lvl_01", "runs/camp/boot/entrances/after_1.state",
              [{"start_wd": [1], "clear_wd": [2]}])

    led = {r["rom"]: r for r in witness_ledger(runs)["roms"]}
    assert led["Chained (USA)"]["witness_tapes"] == 2, (
        "the chained level did not inherit its ROM from the tree that "
        "minted its entrance")
    assert witness_ledger(runs)["unattributed_trees"] == 0


def test_the_ledger_reports_what_it_could_not_attribute(tmp_path: Path) -> None:
    """Under-attribution must be VISIBLE. A ledger that silently drops
    trees it cannot place understates the witness set, and understating the
    witness set is what makes a purity check flag honest work."""
    runs = tmp_path / "runs"
    _fake_run(runs, "orphan", "somewhere/else/mystery.state",
              [{"start_wd": [0], "clear_wd": [1]}])
    out = witness_ledger(runs)
    assert out["unattributed_trees"] == 1
    assert out["roms"] == []
    assert "NO WITNESS FROM THIS SOURCE" in out["caveat"], (
        "the ledger must say out loud that witnessed:false means "
        "'no tape from this source', not 'never cleared'")


@pytest.mark.skipif(not (REPO / "runs").is_dir(),
                    reason="runs/ is git-ignored; no tapes on this machine")
def test_the_real_ledger_agrees_with_the_withdrawn_false_positives() -> None:
    """On a machine that has the tapes, the two known-bad clears must come
    out as non-witnesses and the known-good ones as witnesses. This is the
    ledger checked against the record it is supposed to reproduce."""
    led = {r["rom"]: r for r in witness_ledger()["roms"]}
    if not led:
        pytest.skip("no attributable solver tapes present")
    for bad in ("Double Dragon (USA)", "Kirby's Adventure (USA) (Rev A)"):
        if bad in led:
            assert led[bad]["witnessed"] is False, (
                f"{bad}'s banked sol_000 is a WITHDRAWN FALSE POSITIVE "
                f"(start_wd [] / clear_wd []) and must not read as a clear")
    for good in ("Castlevania (USA)", "Bubble Bobble (USA)"):
        if good in led:
            assert led[good]["witnessed"] is True, (
                f"{good} has a receipted clear chain and must read as one")


# =========================================================================
# 7. Cross-check against the config-layer sweep
# =========================================================================

def test_this_module_and_the_config_sweep_see_the_same_quarantines() -> None:
    """Two independent extractors over the same blocks. If they disagree,
    one of them has gone blind and both suites would keep passing."""
    import test_purity_quarantine_sweep as cfg_sweep

    mine = {q.addr for q in quarantines()}
    theirs = set(cfg_sweep.QUARANTINED_ADDRESSES_INT)
    assert mine <= theirs, (
        f"this module sees quarantined addresses the config sweep does "
        f"not: {sorted(hex(a) for a in mine - theirs)}")


def test_the_make_gate_and_the_pytest_gate_cannot_drift_apart() -> None:
    """`make purity-check` and this module must reach the same verdict.

    Two gates that reimplement the same rule drift, and the one that
    drifts quiet is the one nobody notices. `check_disclosures()` is the
    single implementation; this pins that the suite agrees with it and
    that the Makefile actually runs it.
    """
    assert check_disclosures() == [], (
        "the shared verdict function reports problems the individual "
        "assertions above did not:\n  " + "\n  ".join(check_disclosures()))
    makefile = (REPO / "Makefile").read_text()
    assert "purity-check" in makefile, (
        "the `purity-check` target is gone from the Makefile — the check "
        "no longer runs in the local dev loop")
    assert "clear-lint purity-check" in makefile, (
        "`purity-check` is no longer a prerequisite of `make test`; it "
        "would only run when somebody decides to look, which is exactly "
        "the failure mode it exists to remove")


def test_the_shared_verdict_reports_a_real_breach(real_tree: Path) -> None:
    """The verdict function must BITE, not just return []. Introduce the
    breach and confirm it is named, with the site in the message."""
    rs = real_tree / "nes_core" / "src" / "rewards.rs"
    rs.write_text(rs.read_text().replace(
        "    const RAM_MATCH_ID: usize = 0x0001;",
        "    const RAM_MATCH_ID: usize = 0x0001;\n"
        "    const RAM_WIN_MIRROR: usize = 0x0001;"))
    problems = check_disclosures(repo=real_tree,
                                 disclosures=load_disclosures())
    assert any("UNDECLARED" in p and "RAM_WIN_MIRROR" in p for p in problems), \
        problems
    assert any(p.startswith("RATCHET") for p in problems), problems


def test_the_scanners_are_pure_functions_of_the_tree(synth: Path) -> None:
    """Two runs, same answer. A scan that depends on iteration order would
    make the disclosure inventory unstable and the check unusable."""
    qs = quarantines(synth / "configs", synth)
    a = scan_rust(qs, synth / "nes_core" / "src", synth)
    b = scan_rust(qs, synth / "nes_core" / "src", synth)
    assert a == b
    assert (scan_python(qs, synth) == scan_python(qs, synth))


def test_the_scanner_is_not_reachable_from_any_production_path() -> None:
    """The scanner READS every quarantine block. That is the one thing
    `test_no_source_file_reads_the_quarantine_key` forbids of production
    code, and for a good reason: `scripts/observatory.py` once folded
    `{int(a) for a in ram_mapping.values()}` into its pre-probe exclusion
    set, so a quarantine block that leaked into a pipeline steered the
    discovery instrument AWAY from the very bytes it exists to force a
    rediscovery of.

    This checker enforces the quarantine rather than consuming it, but
    "it's fine because of what it's for" is not a mechanism. The
    mechanism is containment: it lives under tests/, and nothing in
    src/, scripts/ or nes_core/ may import it.
    """
    offenders = []
    for root in ("src", "scripts", "nes_core"):
        d = REPO / root
        if not d.is_dir():
            continue
        for path in d.rglob("*.py"):
            if any(x in path.parts for x in (".venv", "__pycache__")):
                continue
            if "purity_engine_scan" in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        f"the quarantine scanner is reachable from production code: "
        f"{offenders}. A module that reads every quarantine block must not "
        f"be importable by anything that trains or discovers.")


def test_generated_directories_are_never_scanned() -> None:
    """`.claude/worktrees/` holds full copies of this tree. Scanning them
    would multiply every site by the number of parallel lanes and make the
    inventory impossible to keep accurate."""
    files = {s.file for s in scan_quarantined_uses()}
    assert not any(f.startswith((".claude", ".venv", "target"))
                   for f in files), sorted(files)


def test_the_scan_covers_the_layers_the_config_sweep_cannot_see() -> None:
    """The point of this module, stated as an assertion: it reaches Rust
    and Python, which `tests/test_purity_quarantine_sweep.py` does not."""
    kinds = {s.kind for s in scan_quarantined_uses()}
    assert "rust-const" in kinds, "the Rust layer is no longer being scanned"
    assert kinds & {"python", "python-test"}, (
        "the Python layer is no longer being scanned")
