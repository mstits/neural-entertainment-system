"""Skip gates for tests whose inputs are gitignored or machine-local.

A test that reads `runs/`, `checkpoints/` or `roms/` usually depends on
an input a clean clone does not carry (`.gitignore` lines 10, 71, 93).
Gating such a test on the repo root existing is not a gate at all: the
root is always there, so the test runs anyway and fails on a missing
file. Gate on the exact path the test reads instead, and name that path
in the skip reason, so a clean clone reports a reason someone can act on
rather than a failure someone has to diagnose.

Some paths under those three trees are tracked anyway, either force-added
or allowlisted, and a clean clone does carry them. Gating on one of those
gates exactly as much as gating on the repo root, so check `git ls-files`
on a path before it goes in a gate and keep only the ones a clean clone
really lacks.

Use `requires()` as a decorator on the test (or class) that needs the
path. Prefer the narrowest path that is still a real gate: gating a test
on the one file it asserts about turns the assertion into a tautology, so
gate on the containing directory when the assertion IS existence.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def missing(*rel_paths: str) -> list[str]:
    """The repo-relative entries of `rel_paths` that are not on disk."""
    return [rel for rel in rel_paths if not (REPO / rel).exists()]


def requires(*rel_paths: str):
    """`skipif` that fires when any of `rel_paths` is absent from the repo.

    The reason names the absent paths, never the repo root.
    """
    absent = missing(*rel_paths)
    return pytest.mark.skipif(
        bool(absent),
        reason="gitignored or machine-local inputs absent: " + ", ".join(
            absent or rel_paths),
    )


def jsonl_record_absent(rel_path: str, **match) -> bool:
    """True when `rel_path` holds no JSON line matching every kwarg.

    Existence is not a gate on a path the suite itself writes.
    `runs/interference/interference.jsonl` is appended to by the
    interference falsifier's dry run, so a file-exists gate stops firing
    the moment a dry run has happened and the gated test then fails on
    the record it wanted rather than skipping. Gate on the record.
    """
    path = REPO / rel_path
    if not path.exists():
        return True
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return True
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and all(
                rec.get(k) == v for k, v in match.items()):
            return False
    return True


def requires_jsonl_record(rel_path: str, **match):
    """`skipif` that fires unless `rel_path` holds a matching record.

    For receipt logs a real run produces and the suite can also append
    to: the gate asserts the record the test reads, so a file that
    exists for some other reason cannot let the test through.
    """
    absent = jsonl_record_absent(rel_path, **match)
    return pytest.mark.skipif(
        absent,
        reason=f"no {match} record in {rel_path} (real run receipt absent)",
    )


def missing_modules(*names: str) -> list[str]:
    """The entries of `names` that cannot be imported in this interpreter."""
    out = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def requires_module(*names: str):
    """`skipif` that fires when any of `names` is not installed.

    For optional dependencies that are deliberately NOT in
    `requirements.txt` (nes-py lives in
    `requirements-legacy-bakeoff.txt`), so a default install reports a
    named skip instead of a ModuleNotFoundError the reader has to trace
    back to a quarantine decision.
    """
    absent = missing_modules(*names)
    return pytest.mark.skipif(
        bool(absent),
        reason="optional modules not installed: " + ", ".join(
            absent or names),
    )
