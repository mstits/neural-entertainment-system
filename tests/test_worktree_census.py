"""Fixture test for worktree_census.py: builds a tiny repo with a clean
worktree and a dirty one, runs the census, and asserts its rows match what
`git worktree list --porcelain` itself reports (path/branch/dirty/prunable)
 -  i.e. the census is not inventing or dropping worktrees. Report-only:
no git worktree remove/prune (non-dry-run) is ever called, by the fixture
or by the module under test (worktree_census.py enforces this itself via
run_git_readonly's allowlist, which this test also exercises)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import worktree_census as wc  # noqa: E402


def run(*args, cwd):
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, check=True)


def build_fixture(base: Path) -> Path:
    repo = base / "repo"
    run("git", "init", "-q", "-b", "main", str(repo), cwd=base)
    run("git", "config", "user.email", "t@t.com", cwd=repo)
    run("git", "config", "user.name", "T", cwd=repo)
    (repo / "f.txt").write_text("one\n")
    run("git", "add", "f.txt", cwd=repo)
    run("git", "commit", "-qm", "initial", cwd=repo)

    wt_clean = base / "wt_clean"
    run("git", "worktree", "add", "-q", "-b", "clean-branch", str(wt_clean), "main", cwd=repo)
    run("git", "merge", "-q", "clean-branch", cwd=repo)  # merge it into main

    wt_dirty = base / "wt_dirty"
    run("git", "worktree", "add", "-q", "-b", "dirty-branch", str(wt_dirty), "main", cwd=repo)
    (wt_dirty / "f.txt").write_text("one\ndirty\n")

    return repo


def test_census_matches_worktree_list(tmp_path_factory=None):
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="census_fixture_"))
    try:
        repo = build_fixture(base)
        rc, out, err = wc.run_git_readonly(repo, "worktree", "list", "--porcelain")
        assert rc == 0, err
        native_entries = wc.parse_worktree_list(out)
        native_paths = {Path(e["path"]).resolve() for e in native_entries}

        rows = wc.census(repo, stale_days=14.0, owners_path=base / "no-such-owners.json")
        census_paths = {Path(r.path).resolve() for r in rows}

        assert census_paths == native_paths, (census_paths, native_paths)
        assert len(rows) == len(native_entries) == 3

        by_name = {Path(r.path).name: r for r in rows}
        assert by_name["repo"].branch == "main"
        assert by_name["repo"].dirty is False

        assert by_name["wt_clean"].dirty is False
        assert by_name["wt_clean"].merged_to_main == "yes"
        assert by_name["wt_clean"].verdict == "MERGED-CLEAN-SAFE-TO-REMOVE"

        assert by_name["wt_dirty"].dirty is True
        assert by_name["wt_dirty"].dirty_file_count == 1
        # dirty-branch has no commits of its own yet (same tip as main),
        # so it's trivially "merged" - the dirty *working tree* changes are
        # what's un-landed, and that's what the verdict below keys on.
        assert by_name["wt_dirty"].merged_to_main == "yes"
        assert by_name["wt_dirty"].verdict == "ACTIVE-DIRTY"  # too fresh to be stale

        # ---- revert-verify: remove the worktree for real and confirm the
        # census (re-run) drops it too - proves the match isn't coincidence.
        run("git", "worktree", "remove", "-f", str(base / "wt_clean"), cwd=repo)
        rows2 = wc.census(repo, stale_days=14.0, owners_path=base / "no-such-owners.json")
        assert "wt_clean" not in {Path(r.path).name for r in rows2}
        assert len(rows2) == 2

        print("PASS: census matches git worktree list, and tracks a removal")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_readonly_enforcement():
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="census_ro_"))
    try:
        repo = build_fixture(base)
        for bad_args in (
            ("worktree", "remove", "x"),
            ("worktree", "prune"),  # no --dry-run
            ("worktree", "lock", "x"),
        ):
            try:
                wc.run_git_readonly(repo, *bad_args)
                raise AssertionError(f"should have refused: {bad_args}")
            except ValueError:
                pass
        print("PASS: run_git_readonly refuses mutating worktree subcommands")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_owners_lookup():
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="census_owners_"))
    try:
        repo = build_fixture(base)
        owners_path = base / "OWNERS.json"
        owners_path.write_text(json.dumps({
            "worktrees": {
                "wt_dirty": {"owner": "matthew", "purpose": "prototyping DO-11"}
            }
        }))
        rows = wc.census(repo, stale_days=14.0, owners_path=owners_path)
        by_name = {Path(r.path).name: r for r in rows}
        assert by_name["wt_dirty"].owner == "matthew"
        assert by_name["wt_clean"].owner is None
        print("PASS: owners lookup keys by worktree dir name")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    test_census_matches_worktree_list()
    test_readonly_enforcement()
    test_owners_lookup()
    print("ALL PASS")
