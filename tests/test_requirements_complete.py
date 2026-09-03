"""Guard against the works-on-my-machine defect that broke a clean clone
on 2026-09-02: `tests/test_properties.py` imports `hypothesis`, nothing in
requirements.txt ever declared it, and the author's .venv had it anyway.
A stranger's `pytest tests/` exited 2 at collection with zero tests run,
while the same command was green here. The dependency was invisible
precisely because every machine that could have noticed already had it.

This file checks the requirements file against the imports instead of
against the environment. It walks every import statement in tests/, drops
the standard library and the in-tree modules, maps each remaining import
name to the distribution that provides it, and asserts requirements.txt
names that distribution.

Four distributions are deliberately absent from requirements.txt and are
exempt here, each paired with the file that does install it and with the
declaration in that file that does the installing. The second test
re-reads those declarations, so an exemption whose install site moved or
was deleted fails instead of silently widening the hole.
"""
from __future__ import annotations

import ast
import functools
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
REQUIREMENTS = ROOT / "requirements.txt"

# The only third-party imports in tests/ that requirements.txt may omit,
# each with the file that installs it instead and the exact declaration
# that does the installing. The pattern has to match the declaration and
# not a passing mention: requirements-legacy-bakeoff.txt names nes-py in
# three comments, so a substring search there stays satisfied even after
# the requirement line itself is deleted.
EXEMPT = {
    "nes-core": (
        "nes_core/Cargo.toml",
        r'^\s*name\s*=\s*"nes_core"',
        "in-tree Rust crate, built by maturin",
    ),
    "nes-py": (
        "requirements-legacy-bakeoff.txt",
        r"^\s*nes[-_]py\s*[<>=!~]",
        "quarantined to the legacy bake-off",
    ),
    "torch": (
        "scripts/install_macos.sh",
        r"^\s*pip install\b.*\btorch==",
        "pinned by the installer for the MPS wheel",
    ),
    "torchvision": (
        "scripts/install_macos.sh",
        r"^\s*pip install\b.*\btorchvision==",
        "pinned by the installer for the MPS wheel",
    ),
}

# Directories whose .py files are importable in-tree, not from PyPI.
_LOCAL_DIRS = (ROOT, ROOT / "src", ROOT / "scripts", TESTS, TESTS / "parity")


def _norm(name: str) -> str:
    """PEP 503 normalization, so PyYAML, pyyaml and py_yaml compare equal."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _is_local(name: str) -> bool:
    for d in _LOCAL_DIRS:
        if (d / f"{name}.py").is_file():
            return True
        sub = d / name
        if sub.is_dir() and any(sub.glob("*.py")):
            return True
    return False


def _imports_in_tests() -> dict[str, set[str]]:
    """Top-level import name -> the test files that import it."""
    found: dict[str, set[str]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], set()).add(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.setdefault(node.module.split(".")[0], set()).add(rel)
    return found


def _declared_distributions() -> set[str]:
    """Normalized distribution names requirements.txt asks pip to install."""
    declared = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"[A-Za-z0-9._-]+", line)
        if match:
            declared.add(_norm(match.group(0)))
    return declared


@functools.lru_cache(maxsize=1)
def _installed_modules() -> dict[str, list[str]]:
    """Cached: scanning site-packages metadata once is enough for one run."""
    return packages_distributions()


def _providers(import_name: str) -> frozenset[str]:
    """Normalized distributions that could provide this import name.

    `packages_distributions()` resolves the cases where the two differ
    (PIL comes from pillow, yaml from PyYAML). When the module is not
    installed there is nothing to resolve, so fall back to the import
    name itself: a requirements file that declares it under that name
    still passes, and one that declares it nowhere still fails.
    """
    mapped = _installed_modules().get(import_name)
    return frozenset(_norm(d) for d in mapped) if mapped else frozenset({_norm(import_name)})


def test_every_third_party_import_in_tests_is_declared():
    declared = _declared_distributions()
    exempt = set(EXEMPT)
    stdlib = set(sys.stdlib_module_names)

    undeclared = {}
    for name, users in sorted(_imports_in_tests().items()):
        if name in stdlib or _is_local(name):
            continue
        providers = _providers(name)
        if providers & (declared | exempt):
            continue
        undeclared[name] = (sorted(providers), sorted(users)[:3])

    assert not undeclared, (
        "tests/ imports these, and requirements.txt declares none of them, so a "
        "fresh clone dies at collection while this machine stays green:\n"
        + "\n".join(
            f"  import {name}  (from {'/'.join(provs)})  used by: {', '.join(users)}"
            for name, (provs, users) in sorted(undeclared.items())
        )
        + "\nAdd each to requirements.txt with a bounded pin, or add it to EXEMPT "
        "here alongside the file that really installs it."
    )


def test_each_exemption_still_names_a_real_install_site():
    broken = []
    for dist, (site, pattern, why) in sorted(EXEMPT.items()):
        path = ROOT / site
        if not path.is_file():
            broken.append(f"  {dist}: {site} does not exist ({why})")
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(pattern, text, re.MULTILINE):
            broken.append(f"  {dist}: {site} has no line matching {pattern!r} ({why})")

    assert not broken, (
        "these distributions are exempt from requirements.txt on the grounds that "
        "another file installs them, and that file no longer does:\n"
        + "\n".join(broken)
        + "\nEither restore the install site or move the dependency into "
        "requirements.txt."
    )
