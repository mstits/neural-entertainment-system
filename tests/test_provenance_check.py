"""demo_anchor_paths sweep in scripts/provenance_check.py.

demo_anchor_paths entries live in tracked configs/*.yaml but usually
point at runs/ — git-ignored, so the SEEDS rglob (checkpoints/harvested_
seeds/) never sees them. A config referencing a runs/ npz that was never
committed (or got cleaned up) must fail the gate, not pass silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts import provenance_check
from scripts.provenance_check import check_demo_anchor_paths, collect_demo_anchor_refs


def _write_config(configs_dir, name, demo_anchor_paths):
    configs_dir.mkdir(parents=True, exist_ok=True)
    lines = ["demo_anchor_paths:"] + [f'- "{p}"' for p in demo_anchor_paths]
    (configs_dir / name).write_text("\n".join(lines) + "\n")


def test_collect_demo_anchor_refs_reads_paths_from_tracked_config(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    _write_config(configs, "mario_1_2.yaml", ["runs/ge_1_2_solve/demos/demo_000.npz"])

    refs = collect_demo_anchor_refs(configs, repo)

    assert refs == {
        "runs/ge_1_2_solve/demos/demo_000.npz": ["configs/mario_1_2.yaml"],
    }


def test_check_demo_anchor_paths_fails_on_missing_runs_file(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    _write_config(configs, "mario_1_2.yaml", ["runs/ge_1_2_solve/demos/demo_000.npz"])
    seeds = repo / "checkpoints/harvested_seeds"

    errors, hashes = check_demo_anchor_paths(repo, seeds)

    assert hashes == {}
    assert len(errors) == 1
    assert "runs/ge_1_2_solve/demos/demo_000.npz" in errors[0]
    assert "configs/mario_1_2.yaml" in errors[0]


def test_check_demo_anchor_paths_hashes_existing_runs_file(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    demo_rel = "runs/ge_1_2_solve/demos/demo_000.npz"
    _write_config(configs, "mario_1_2.yaml", [demo_rel])
    demo_path = repo / demo_rel
    demo_path.parent.mkdir(parents=True)
    np.savez(str(demo_path), obs_0=np.zeros((3, 4), dtype=np.int8),
              act_0=np.zeros((3,), dtype=np.int64))
    seeds = repo / "checkpoints/harvested_seeds"

    errors, hashes = check_demo_anchor_paths(repo, seeds)

    assert errors == []
    assert demo_rel in hashes
    assert len(hashes[demo_rel]) == 64


def test_check_demo_anchor_paths_skips_files_already_under_seeds(tmp_path):
    repo = tmp_path
    configs = repo / "configs"
    seeds = repo / "checkpoints/harvested_seeds"
    demo_rel = "checkpoints/harvested_seeds/demos_x.npz"
    _write_config(configs, "mario_1_2.yaml", [demo_rel])
    demo_path = repo / demo_rel
    demo_path.parent.mkdir(parents=True)
    np.savez(str(demo_path), obs_0=np.zeros((3, 4), dtype=np.int8),
              act_0=np.zeros((3,), dtype=np.int64))

    errors, hashes = check_demo_anchor_paths(repo, seeds)

    assert errors == []
    assert hashes == {}


# ---- soak receipt trails (gate extension approved 2026-08-15) ----------

FAKE_HARNESS_OK = """
def verify_receipt_trail(soak_dir):
    from pathlib import Path
    if (Path(soak_dir) / "TAMPERED").exists():
        return [f"receipt chain broken at seg_0001"]
    return []
"""


def _soak_repo(tmp_path, *, harness=True, dirs=()):
    repo = tmp_path
    (repo / "scripts").mkdir(exist_ok=True)
    if harness:
        (repo / "scripts" / "soak_harness.py").write_text(FAKE_HARNESS_OK)
    for name, final in dirs:
        d = repo / "runs" / "soak" / name
        d.mkdir(parents=True)
        if final is not None:
            import json as _json
            (d / "final_receipt.json").write_text(_json.dumps(final))
    return repo


def test_soak_absent_runs_dir_is_clean(tmp_path):
    errors, verified, scoreable = provenance_check.check_soak_trails(tmp_path)
    assert errors == [] and verified == 0 and scoreable == 0


def test_soak_dirs_without_harness_fail_loud(tmp_path):
    repo = _soak_repo(tmp_path, harness=False,
                      dirs=[("soak_a", None)])
    errors, verified, scoreable = provenance_check.check_soak_trails(repo)
    assert len(errors) == 1 and "verifier" in errors[0].replace(
        "chain verifier", "verifier")
    assert verified == 0


def test_soak_valid_chain_counts_scoreable_and_selfcheck(tmp_path):
    repo = _soak_repo(tmp_path, dirs=[
        ("soak_real", {"backend_scoreable": True, "selfcheck": False}),
        ("soak_stub", {"backend_scoreable": True, "selfcheck": True}),
        ("soak_unscoreable", {"backend_scoreable": False,
                              "selfcheck": False}),
    ])
    errors, verified, scoreable = provenance_check.check_soak_trails(repo)
    assert errors == []
    assert verified == 3
    assert scoreable == 1  # stub + unscoreable chains verify but never score


def test_soak_tampered_chain_fails(tmp_path):
    repo = _soak_repo(tmp_path, dirs=[
        ("soak_bad", {"backend_scoreable": True, "selfcheck": False})])
    (repo / "runs" / "soak" / "soak_bad" / "TAMPERED").touch()
    errors, verified, scoreable = provenance_check.check_soak_trails(repo)
    assert any("chain broken" in e for e in errors)
    assert verified == 0 and scoreable == 0


# ---- FORGE ledger entries (CLAIMS.md "### FORGE entries") -------------
#
# Only two of the FORGE definition's four criteria have real, checkable
# structure (see check_forge_entries' docstring for why detection and
# authorship are not mechanically checkable at all): a named flag/config
# key that actually defaults off in tracked source, a cited tests/*.py
# file that actually exists and passes, and an explicit PASS/FAIL/VOID/
# PENDING-VALIDATION status word somewhere in the entry. These fixtures
# exercise all three against real source and a real pytest subprocess —
# no mocking — so a change that silently breaks the flag/test lookups
# cannot pass this file by accident.

_FAKE_FLAG_SOURCE = '''
import argparse

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frobnicate", choices=("off", "on"), default="off",
                     help="test-only flag")
    return ap
'''

_FAKE_TEST_OK = '''
def test_trivial():
    assert True
'''


def _write_forge_repo(tmp_path, entry_body: str, *, with_test_file=True,
                       readme: str | None = None):
    repo = tmp_path
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "fake_tool.py").write_text(_FAKE_FLAG_SOURCE)
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    if with_test_file:
        (repo / "tests" / "test_fake_forge_ok.py").write_text(_FAKE_TEST_OK)
    if readme is not None:
        (repo / "README.md").write_text(readme)
    claims = f"""# Claims Policy

### FORGE entries

{entry_body}

## Quarantine

nothing here.
"""
    (repo / "CLAIMS.md").write_text(claims)
    return repo


_WELL_FORMED_ENTRY = (
    "**FORGE-SHIPPED — the frobnicate arm (`--frobnicate`), commit "
    "`abc1234`, 2026-01-01.** Self-measured detection: found in the "
    "run's own telemetry. Agentic authorship: designed and reviewed by "
    "the pipeline with no human algorithmic contribution. Standard "
    "gates: default off (`--frobnicate`), covered by "
    "`tests/test_fake_forge_ok.py`. Honest status: PASS.")


def test_forge_entry_well_formed_passes(tmp_path):
    repo = _write_forge_repo(tmp_path, _WELL_FORMED_ENTRY)

    errors, report = provenance_check.check_forge_entries(repo)

    assert errors == []
    assert report["total"] == 1
    assert report["passed"] == 1
    entry = report["entries"][0]
    assert entry["overall"] == "pass"
    assert entry["flag_default_off"][0] == "pass"
    assert entry["cited_tests"][0] == "pass"
    assert entry["explicit_status"][0] == "pass"


def test_forge_entry_nonexistent_test_file_fails(tmp_path):
    entry = (
        "**FORGE-SHIPPED — the frobnicate arm (`--frobnicate`), commit "
        "`abc1234`, 2026-01-01.** Standard gates: default off "
        "(`--frobnicate`), covered by `tests/test_does_not_exist_xyzzy.py`. "
        "Honest status: PASS.")
    repo = _write_forge_repo(tmp_path, entry, with_test_file=False)

    errors, report = provenance_check.check_forge_entries(repo)

    assert report["passed"] == 0
    assert any(
        "test_does_not_exist_xyzzy.py" in e and "do not exist" in e
        for e in errors), errors
    entry_report = report["entries"][0]
    assert entry_report["cited_tests"][0] == "fail"
    assert "test_does_not_exist_xyzzy.py" in entry_report["cited_tests"][1]


def test_forge_entry_no_status_word_fails(tmp_path):
    entry = (
        "**FORGE-SHIPPED — the frobnicate arm (`--frobnicate`), commit "
        "`abc1234`, 2026-01-01.** Standard gates: default off "
        "(`--frobnicate`), covered by `tests/test_fake_forge_ok.py`. "
        "Honest status: the mechanism is confirmed working end to end.")
    repo = _write_forge_repo(tmp_path, entry)

    errors, report = provenance_check.check_forge_entries(repo)

    assert report["passed"] == 0
    entry_report = report["entries"][0]
    assert entry_report["explicit_status"][0] == "fail"
    assert any("no explicit PASS/FAIL/VOID/PENDING-VALIDATION word" in e
               for e in errors), errors
    # the other two criteria are unaffected by the missing status word
    assert entry_report["flag_default_off"][0] == "pass"
    assert entry_report["cited_tests"][0] == "pass"


def test_parse_forge_entries_splits_on_bold_forge_headers_only(tmp_path):
    entry = (
        _WELL_FORMED_ENTRY + "\n\n"
        "*Status, updated later — not a new entry, just an addendum "
        "to the one above.* Nothing new to check here.\n\n"
        "**FORGE — a second, untagged entry, commit `def5678`.** Cites "
        "`tests/test_fake_forge_ok.py` too. Honest status: VOID.")
    repo = _write_forge_repo(tmp_path, entry)

    entries = provenance_check.parse_forge_entries(repo / "CLAIMS.md")

    assert len(entries) == 2
    assert entries[0]["tag"] == "SHIPPED"
    assert "addendum" in entries[0]["text"]
    assert entries[1]["tag"] is None
    assert "second, untagged entry" in entries[1]["text"]


# --- currency check: a validated arm's README summary must not still say
# --- the validation never ran (DECIDE-10, README's ortho paragraph).

_VALIDATED_ENTRY = (
    _WELL_FORMED_ENTRY + "\n\n"
    "*Status, updated 2026-01-08 — the pre-registered A/B validation "
    "ran (seed 303, 90 min each).* **Split verdict.** MECHANISM "
    "VALIDATED; PREMISE STALE. No clear may be attributed to it.")

_STALE_README = """# Fake project

The `--frobnicate` arm was diagnosed from the run's own telemetry, then
designed, implemented and gated by agents.

**And it has not been shown to work.** No validation run has been performed;
the hall is still unsolved. `CLAIMS.md` files that arm as
**FORGE-PENDING-VALIDATION**.
"""

_CURRENT_README = """# Fake project

The `--frobnicate` arm was diagnosed from the run's own telemetry, then
designed, implemented and gated by agents.

**And it has not cracked the wall.** The pre-registered A/B ran on
2026-01-08 and returned a split verdict: validated as a selection-pressure
mechanism, not validated as a wall-cracking mechanism.
"""


def test_forge_status_updated_with_stale_readme_fails(tmp_path):
    repo = _write_forge_repo(tmp_path, _VALIDATED_ENTRY,
                             readme=_STALE_README)

    errors, report = provenance_check.check_forge_entries(repo)

    entry = report["entries"][0]
    assert entry["readme_current"][0] == "fail", entry["readme_current"]
    assert entry["overall"] == "fail"
    assert report["passed"] == 0
    assert any("No validation run has been performed" in e
               and "*Status, updated" in e for e in errors), errors
    # the drift is reported with a line number a human can go fix.
    assert any("README.md line(s) 6" in e for e in errors), errors


def test_forge_status_updated_with_current_readme_passes(tmp_path):
    repo = _write_forge_repo(tmp_path, _VALIDATED_ENTRY,
                             readme=_CURRENT_README)

    errors, report = provenance_check.check_forge_entries(repo)

    entry = report["entries"][0]
    assert entry["readme_current"][0] == "pass", entry["readme_current"]
    assert entry["overall"] == "pass"
    assert errors == []


def test_forge_without_status_update_tolerates_pending_readme(tmp_path):
    """An arm whose validation genuinely has NOT run keeps the sentence.
    The check fires on the ledger updating, not on the sentence existing."""
    repo = _write_forge_repo(tmp_path, _WELL_FORMED_ENTRY,
                             readme=_STALE_README)

    errors, report = provenance_check.check_forge_entries(repo)

    entry = report["entries"][0]
    assert entry["readme_current"][0] == "n/a", entry["readme_current"]
    assert entry["overall"] == "pass"
    assert errors == []


def test_forge_stale_sentence_about_a_different_arm_is_not_blamed(tmp_path):
    """Two arms, one validated: the stale sentence sits beside the OTHER
    arm's flag, so the validated entry must not be charged with it."""
    readme = """# Fake project

The `--other-arm` was forged last week and is still pending.

**And it has not been shown to work.** No validation run has been performed;
the hall is still unsolved.
""" + "filler\n" * 40 + (
        "Far below, in a different section entirely, the `--frobnicate`\n"
        "arm's A/B ran and returned a split verdict.\n")
    repo = _write_forge_repo(tmp_path, _VALIDATED_ENTRY, readme=readme)

    errors, report = provenance_check.check_forge_entries(repo)

    entry = report["entries"][0]
    assert entry["readme_current"][0] == "pass", entry["readme_current"]
    assert errors == []
