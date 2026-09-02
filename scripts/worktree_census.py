#!/usr/bin/env python3
"""Report-only worktree census: every worktree, its branch, dirty state,
last-commit age, whether `git worktree prune --dry-run` flags it, and an
ownership lookup against `.claude/worktrees/OWNERS.json`.

Never writes to the target repo - no `git worktree remove`, no `prune`
without `--dry-run`, no file writes inside the repo. Every git call this
script makes is read-only; the only writes are the report file the caller
names on the command line (default: stdout).

Usage:
    python worktree_census.py --repo /path/to/repo [--json] [--stale-days N]

Exit codes: 0 on a completed census (even one full of stale worktrees  - 
that's data, not failure); 2 if the repo can't be read at all.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def run_git(repo: Path, *args: str, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a read-only git command. Never pass a mutating subcommand here."""
    proc = subprocess.run(
        ["git", "-C", str(cwd or repo), *args],
        capture_output=True, text=True, timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


READONLY_ALLOW = {
    "worktree", "status", "log", "rev-parse", "symbolic-ref", "merge-base",
    "for-each-ref", "show",
}


def run_git_readonly(repo: Path, *args: str, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    if args and args[0] not in READONLY_ALLOW:
        raise ValueError(f"refusing non-read-only git subcommand: {args[0]!r}")
    if "worktree" in args and any(a in ("remove", "prune", "lock", "unlock", "add", "move") for a in args):
        # 'prune' is allowed ONLY as --dry-run; enforced by the one call
        # site below, never here generically.
        if "prune" in args and "--dry-run" not in args:
            raise ValueError("refusing 'git worktree prune' without --dry-run")
        if any(a in ("remove", "lock", "unlock", "add", "move") for a in args):
            raise ValueError("refusing a mutating worktree subcommand")
    return run_git(repo, *args, cwd=cwd)


@dataclass
class WorktreeInfo:
    path: str
    head: str
    branch: Optional[str]
    detached: bool
    porcelain_prunable: bool
    porcelain_prunable_reason: str
    prune_dry_run_flagged: bool
    exists_on_disk: bool
    status_readable: bool
    dirty: Optional[bool]
    dirty_file_count: Optional[int]
    last_commit_iso: Optional[str]
    age_days: Optional[float]
    merged_to_main: Optional[str]  # "yes" / "no" / "n/a"
    owner: Optional[str]
    owner_purpose: Optional[str]
    verdict: str
    verdict_reason: str


def parse_worktree_list(porcelain: str) -> list[dict]:
    entries: list[dict] = []
    cur: dict = {}
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            if cur:
                entries.append(cur)
            cur = {"path": line[len("worktree "):], "prunable": False,
                   "prunable_reason": "", "detached": False, "branch": None,
                   "head": ""}
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):]
        elif line == "detached":
            cur["detached"] = True
        elif line.startswith("prunable "):
            cur["prunable"] = True
            cur["prunable_reason"] = line[len("prunable "):]
        elif line.startswith("locked"):
            cur["locked"] = True
        elif line == "":
            continue
    if cur:
        entries.append(cur)
    return entries


def parse_prune_dry_run(text: str) -> set[str]:
    """Parse 'Removing worktrees/<name>: <reason>' lines into a set of
    trailing path components git used to name them (best-effort match  - 
    git names them by the worktree's directory basename under
    .git/worktrees/, which is usually but not always the leaf dir name)."""
    flagged = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Removing worktrees/"):
            rest = line[len("Removing worktrees/"):]
            name = rest.split(":", 1)[0].strip()
            flagged.add(name)
    return flagged


def load_owners(owners_path: Path) -> dict:
    if not owners_path.exists():
        return {}
    try:
        data = json.loads(owners_path.read_text())
        return data.get("worktrees", {})
    except (OSError, json.JSONDecodeError):
        return {}


def owner_lookup(owners: dict, wt_path: Path, branch: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    key_candidates = [wt_path.name]
    if branch:
        key_candidates.append(branch)
    for key in key_candidates:
        if key in owners:
            rec = owners[key]
            return rec.get("owner"), rec.get("purpose")
    return None, None


def census(repo: Path, stale_days: float, owners_path: Path) -> list[WorktreeInfo]:
    rc, out, err = run_git_readonly(repo, "worktree", "list", "--porcelain")
    if rc != 0:
        raise RuntimeError(f"git worktree list failed: {err.strip()}")
    entries = parse_worktree_list(out)

    # `git worktree prune -v` writes its "Removing worktrees/..." lines to
    # stderr, not stdout - confirmed against a live run, not assumed.
    rc2, prune_out, prune_err = run_git_readonly(repo, "worktree", "prune", "--dry-run", "-v")
    dry_run_flagged = parse_prune_dry_run(prune_out + prune_err) if rc2 == 0 else set()

    owners = load_owners(owners_path)
    now = datetime.now(timezone.utc)
    results: list[WorktreeInfo] = []

    for e in entries:
        wt_path = Path(e["path"])
        branch = e["branch"]
        branch_short = branch[len("refs/heads/"):] if branch and branch.startswith("refs/heads/") else branch
        exists = wt_path.exists()

        dry_flag = wt_path.name in dry_run_flagged

        dirty: Optional[bool] = None
        dirty_count: Optional[int] = None
        status_ok = False
        if exists:
            src, sout, _serr = run_git_readonly(repo, "status", "--porcelain", cwd=wt_path)
            if src == 0:
                status_ok = True
                lines = [l for l in sout.splitlines() if l]
                dirty = len(lines) > 0
                dirty_count = len(lines)

        last_commit_iso: Optional[str] = None
        age_days: Optional[float] = None
        if exists:
            lrc, lout, _lerr = run_git_readonly(repo, "log", "-1", "--format=%cI", cwd=wt_path)
            if lrc == 0 and lout.strip():
                last_commit_iso = lout.strip()
                try:
                    commit_dt = datetime.fromisoformat(last_commit_iso)
                    age_days = round((now - commit_dt).total_seconds() / 86400, 1)
                except ValueError:
                    pass

        merged: Optional[str] = None
        if branch_short and not e["detached"]:
            mrc, _mo, _me = run_git_readonly(repo, "merge-base", "--is-ancestor", branch_short, "main")
            merged = "yes" if mrc == 0 else "no"
        elif e["detached"]:
            merged = "n/a (detached)"

        owner, purpose = owner_lookup(owners, wt_path, branch_short)

        # ---- verdict: report-only classification, never acted on here ---
        if e["prunable"]:
            verdict = "PRUNABLE-GIT-NATIVE"
            reason = f"git worktree list flags it prunable: {e['prunable_reason']}"
        elif not exists:
            verdict = "MISSING-ON-DISK"
            reason = "registered but the directory is gone"
        elif not status_ok:
            verdict = "UNREADABLE"
            reason = "git status failed in this worktree - investigate before touching"
        elif branch_short == "main" and not e["detached"] and wt_path == Path(repo):
            verdict = "PRIMARY"
            reason = "the main worktree"
        elif dirty and age_days is not None and age_days >= stale_days:
            verdict = "STALE-DIRTY-NEEDS-REVIEW"
            reason = (f"{dirty_count} uncommitted change(s), last commit "
                      f"{age_days:.0f}d ago - has work nobody has looked at")
        elif merged == "yes" and not dirty:
            verdict = "MERGED-CLEAN-SAFE-TO-REMOVE"
            reason = "branch fully merged to main, no uncommitted changes"
        elif age_days is not None and age_days >= stale_days and not dirty:
            verdict = "STALE-CLEAN-CANDIDATE"
            reason = f"clean, but {age_days:.0f}d since last commit and not merged to main"
        elif dirty:
            verdict = "ACTIVE-DIRTY"
            reason = f"{dirty_count} uncommitted change(s), within the {stale_days:.0f}d window"
        else:
            verdict = "ACTIVE-CLEAN"
            reason = "clean and recent"

        results.append(WorktreeInfo(
            path=str(wt_path), head=e["head"][:12], branch=branch_short,
            detached=e["detached"], porcelain_prunable=e["prunable"],
            porcelain_prunable_reason=e["prunable_reason"],
            prune_dry_run_flagged=dry_flag, exists_on_disk=exists,
            status_readable=status_ok, dirty=dirty, dirty_file_count=dirty_count,
            last_commit_iso=last_commit_iso, age_days=age_days,
            merged_to_main=merged, owner=owner, owner_purpose=purpose,
            verdict=verdict, verdict_reason=reason,
        ))
    return results


def render_table(rows: list[WorktreeInfo]) -> str:
    headers = ["worktree", "branch", "dirty", "age(d)", "merged", "owner", "verdict"]
    body = []
    for r in rows:
        body.append([
            Path(r.path).name,
            (r.branch or ("DETACHED" if r.detached else "?"))[:34],
            ("?" if r.dirty is None else (f"{r.dirty_file_count}" if r.dirty else "clean")),
            "?" if r.age_days is None else f"{r.age_days:.0f}",
            r.merged_to_main or "?",
            r.owner or "-",
            r.verdict,
        ])
    widths = [max(len(h), *(len(row[i]) for row in body)) if body else len(h)
              for i, h in enumerate(headers)]
    lines = []
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for row in body:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def summarize(rows: list[WorktreeInfo]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {
        "total": len(rows),
        "by_verdict": counts,
        "git_native_prunable": sum(1 for r in rows if r.porcelain_prunable),
        "prune_dry_run_flagged": sum(1 for r in rows if r.prune_dry_run_flagged),
        "dirty": sum(1 for r in rows if r.dirty),
        "owned": sum(1 for r in rows if r.owner),
        "unowned": sum(1 for r in rows if not r.owner),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--stale-days", default=14.0, type=float)
    ap.add_argument("--owners", default=None, type=Path,
                     help="default: <repo>/.claude/worktrees/OWNERS.json")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    repo = args.repo.resolve()
    owners_path = args.owners or (repo / ".claude" / "worktrees" / "OWNERS.json")

    try:
        rows = census(repo, args.stale_days, owners_path)
    except RuntimeError as exc:
        print(f"worktree_census: {exc}", file=sys.stderr)
        return 2

    summary = summarize(rows)
    if args.json:
        print(json.dumps({"summary": summary, "worktrees": [asdict(r) for r in rows]}, indent=2))
    else:
        print(f"# Worktree census - {repo}")
        print(f"# {summary['total']} worktrees, stale threshold {args.stale_days:.0f}d, "
              f"owners file: {owners_path} ({'found' if owners_path.exists() else 'not found - all unowned'})")
        print()
        print(render_table(rows))
        print()
        print("## Summary")
        for k, v in summary["by_verdict"].items():
            print(f"  {v:3d}  {k}")
        print(f"  git-native prunable (porcelain flag): {summary['git_native_prunable']}")
        print(f"  git worktree prune --dry-run flagged: {summary['prune_dry_run_flagged']}")
        print(f"  dirty (uncommitted work present):     {summary['dirty']}")
        print(f"  owned / unowned:                       {summary['owned']} / {summary['unowned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
