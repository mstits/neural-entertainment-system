"""Skip gates for tests whose inputs are gitignored or machine-local.

A test that reads `runs/`, `checkpoints/` or `roms/` usually depends on
an input a clean clone does not carry (`.gitignore` lines 10, 65, 86).
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
