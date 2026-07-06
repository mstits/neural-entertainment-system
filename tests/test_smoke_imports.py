"""Fast import-surface smoke test.

A broken import — a symbol renamed out from under a top-level `from x
import y`, a bad relative import after a refactor, a stray syntax error
— should fail LOUDLY and IMMEDIATELY at collection, not minutes into a
training run or the first time a user opens a GUI window. This test
imports every key module the product depends on so a single `pytest`
catches the break before it reaches a customer's one-command run.

Coverage:
  * CORE_MODULES — the non-GUI spine (Rust core, trainer, policy nets,
    pool adapter). A failure here is always a real regression; never
    skipped.
  * src.gui.* — discovered from the filesystem (no import at collection
    time) and imported under Qt's headless `offscreen` platform. On a
    training-only install without PyQt6 the GUI is optional, so those
    imports skip gracefully instead of failing.

Note this only imports modules; it never constructs a QWidget/
QApplication (which SIGABRTs under pytest on macOS + PyQt6 — see
tests/test_gui_imports.py). Widget construction is covered separately
by tests/gui_selftest.py.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

# Must be set BEFORE any PyQt6 import so a widget-class module body that
# touches the platform plugin resolves to the headless backend rather
# than trying to reach a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parent.parent

# The non-GUI modules the product cannot run without. A failure to
# import any of these is a real regression — never skipped.
CORE_MODULES = [
    "nes_core",
    "src.training.trainer",
    "src.models.policy_network",
    "src.models.tile_policy",
    "src.emulation.rust_pool_adapter",
]


def _discover_gui_modules() -> list[str]:
    """List `src.gui.*` submodules by scanning the filesystem, so test
    COLLECTION never has to import the package (which would pull in
    PyQt6 before importorskip can guard it)."""
    gui_dir = _ROOT / "src" / "gui"
    return sorted(
        f"src.gui.{p.stem}"
        for p in gui_dir.glob("*.py")
        if p.stem != "__init__"
    )


GUI_MODULES = _discover_gui_modules()


@pytest.mark.parametrize("modname", CORE_MODULES)
def test_core_module_imports(modname: str) -> None:
    importlib.import_module(modname)


@pytest.mark.parametrize("modname", GUI_MODULES)
def test_gui_module_imports(modname: str) -> None:
    # A headless / training-only install may not ship PyQt6; the GUI is
    # optional there, so skip rather than fail. When PyQt6 IS present, a
    # broken GUI import still fails loudly.
    pytest.importorskip("PyQt6", reason="PyQt6 not installed — GUI optional")
    importlib.import_module(modname)


def test_gui_modules_were_discovered() -> None:
    """Guard the guard: if the glob ever finds nothing (package moved,
    path typo), the parametrized GUI test would vacuously pass and the
    GUI import surface would go uncovered without anyone noticing."""
    assert GUI_MODULES, (
        "no src.gui.* modules discovered — the GUI import smoke test is "
        "covering nothing; check the package path in _discover_gui_modules"
    )
