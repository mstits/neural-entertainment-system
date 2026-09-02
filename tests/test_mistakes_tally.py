"""check_enforcement (scripts/mistakes_tally.py, DO-21): a SHIPPED/PROMOTED
watch-table cell names only Makefile targets, paths, and symbols that exist
(rule A), and never quotes a stale value for an integer constant it names
(rule B). Each case builds a synthetic table against a synthetic REPO in
tmp_path -- no dependency on MISTAKES.md's real contents, so this file does
not rot when the ledger changes.

Amended by ruling 9 / review correction 3: a cell citing "(project
instruction file)" is checked against CLAUDE.md only when that file exists
in REPO; on a checkout without it (CLAUDE.md is gitignored here), the rule
is skipped with a printed note rather than failing, so `make test` stays
green on a clean clone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mistakes_tally as mt  # noqa: E402


def build_repo(tmp_path: Path) -> Path:
    """A minimal REPO: one real Makefile target, one real module-level int
    constant in a real .py file under scripts/ -- enough for _resolve and
    _defines to have something true to find."""
    (tmp_path / "Makefile").write_text("real-target:\n\techo hi\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "real_thing.py").write_text("THRESHOLD = 5\n")
    for root in ("src", "tests", "nes_core/src"):
        (tmp_path / root).mkdir(parents=True)
    return tmp_path


def test_check_enforcement_fires_on_missing_path_and_stale_numeral(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    monkeypatch.setattr(mt, "REPO", repo)
    text = (
        "| `[stale-artifact]` | **6** | **SHIPPED**: `scripts/does_not_exist.py` |\n"
        "| `[inert-treatment]` | **7** | **PROMOTED**: `THRESHOLD` set to 999 |\n"
    )
    problems = mt.check_enforcement(text)
    assert len(problems) == 2
    n1, cat1, msg1 = problems[0]
    assert (n1, cat1) == (1, "stale-artifact")
    assert "scripts/does_not_exist.py" in msg1 and "does not exist" in msg1
    n2, cat2, msg2 = problems[1]
    assert (n2, cat2) == (2, "inert-treatment")
    assert "THRESHOLD" in msg2 and "= 5" in msg2 and "[999]" in msg2


def test_check_enforcement_negative_control_returns_empty(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    monkeypatch.setattr(mt, "REPO", repo)
    # Real Makefile target, real path, real symbol quoted with its actual
    # value (5) -- nothing here is false, so this must VOID.
    text = (
        "| `[process]` | **5** | **SHIPPED**: `make real-target` + "
        "`scripts/real_thing.py` + `THRESHOLD` |\n"
    )
    assert mt.check_enforcement(text) == []


def test_check_enforcement_skips_absent_instruction_file(tmp_path, monkeypatch, capsys):
    repo = build_repo(tmp_path)
    assert not (repo / "CLAUDE.md").exists()
    monkeypatch.setattr(mt, "REPO", repo)
    text = "| `[unverified-claim]` | **8** | **PROMOTED 2026-08-28** (project instruction file) |\n"
    problems = mt.check_enforcement(text)
    assert problems == []
    out = capsys.readouterr().out
    assert "CLAUDE.md absent" in out and "MISTAKES.md:1" in out


def test_check_enforcement_fires_when_instruction_file_lacks_heading(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    (repo / "CLAUDE.md").write_text("# Some other project file\nNo such heading here.\n")
    monkeypatch.setattr(mt, "REPO", repo)
    text = "| `[unverified-claim]` | **8** | **PROMOTED 2026-08-28** (project instruction file) |\n"
    problems = mt.check_enforcement(text)
    assert len(problems) == 1
    n, cat, msg = problems[0]
    assert (n, cat) == (1, "unverified-claim")
    assert "Enforced invariants" in msg


def test_check_enforcement_rejects_regression_to_a_value_the_cell_records_as_history(
    tmp_path, monkeypatch
):
    """Rule B is directional. The number a cell attributes to a symbol is the
    first integer after the symbol, not any integer in the cell, so a cell
    that keeps the superseded value as history ("kills the run at 5 (raised
    from 3 ...)") still fires when the constant regresses to 3.

    This is the hole the 2026-09-02 audit measured on MISTAKES.md's
    [inert-treatment] cell: with rule B as set membership, the cell quoting
    both 40 and 25 accepted _REDO_ARM_DEADLINE_ITERS = 25, so the guard
    caught ledger drift but not the code regression the cell describes.
    """
    repo = build_repo(tmp_path)  # scripts/real_thing.py: THRESHOLD = 5
    monkeypatch.setattr(mt, "REPO", repo)
    cell = (
        "| `[inert-treatment]` | **7** | **SHIPPED**: `THRESHOLD` kills the "
        "run at 5 (raised from 3 on 2026-08-27) |\n"
    )
    # Current value matches the number the cell attributes to it: VOID.
    assert mt.check_enforcement(cell) == []

    # Regress the constant to the superseded value the cell still records.
    (repo / "scripts" / "real_thing.py").write_text("THRESHOLD = 3\n")
    problems = mt.check_enforcement(cell)
    assert len(problems) == 1
    n, cat, msg = problems[0]
    assert (n, cat) == (1, "inert-treatment")
    assert "`THRESHOLD` = 3" in msg and "[5]" in msg
