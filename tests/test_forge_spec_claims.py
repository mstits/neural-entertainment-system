"""docs/proposals/FORGE_SPEC_2026-09-01.md -- its scripts/*.py claims match the tree.

FORGE-FIX-30. Section 2 names ``scripts/forge.py {stall,bundle,select,cycle,
block} [--dry-run]`` as "one CLI" for the six Forge pieces. The file was never
written -- there is no operator entry point to any Forge piece outside the
engine's ``--forge`` hook -- and the sentence gave no hint that the file does
not exist. This test does not build the CLI; it holds the document to the
same rule its own vocabulary gates hold code to: a shipped-sounding claim
about a script must be true, or the sentence naming it must say plainly that
it is not built.

Revert-verify: restore the pre-fix sentence (the one this test's own
docstring quotes above, with no UNBUILT_MARKERS phrase near the forge.py
citation) and this test fails, because scripts/forge.py still does not
exist on disk and nothing admits it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPEC_PATH = ROOT / "docs" / "proposals" / "FORGE_SPEC_2026-09-01.md"

# Present in the same sentence as a scripts/*.py reference, any of these
# reads as the spec admitting the file is not shipped yet.
UNBUILT_MARKERS = (
    "unwritten",
    "not yet built",
    "no cli ships",
    "does not exist",
    "library-only",
    "is not written",
)

SCRIPT_PATH_RE = re.compile(r"scripts/[\w./-]+\.py")


def _sentences_naming_scripts(text: str):
    # Markdown prose in this doc is sentence-per-clause enough that a plain
    # split on sentence-ending punctuation followed by whitespace is enough
    # to keep each scripts/*.py citation with the words around it.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        paths = SCRIPT_PATH_RE.findall(sentence)
        if paths:
            yield sentence, paths


def test_forge_spec_script_claims_are_true_or_admitted():
    text = SPEC_PATH.read_text()
    checked = 0
    for sentence, paths in _sentences_naming_scripts(text):
        lowered = sentence.lower()
        admitted = any(marker in lowered for marker in UNBUILT_MARKERS)
        for rel in paths:
            checked += 1
            on_disk = (ROOT / rel).exists()
            assert on_disk or admitted, (
                f"FORGE_SPEC cites {rel!r} as though it ships, but the file "
                f"is not on disk and the sentence admits nothing:\n{sentence!r}"
            )
    assert checked > 0, "no scripts/*.py reference found in the spec; test is vacuous"


def test_forge_spec_forge_cli_claim_is_specifically_covered():
    # Narrower pin for the exact defect: forge.py must either exist, or the
    # spec's own words must say the six pieces have no CLI yet. Kept
    # separate from the broad scan above so a future edit that removes the
    # forge.py sentence entirely does not silently stop testing this case.
    text = SPEC_PATH.read_text()
    assert "scripts/forge.py" in text, "spec no longer names the CLI at all; update this test"
    forge_py_exists = (ROOT / "scripts" / "forge.py").exists()
    sentence = next(
        s for s, paths in _sentences_naming_scripts(text) if "scripts/forge.py" in s
    )
    admitted = any(marker in sentence.lower() for marker in UNBUILT_MARKERS)
    assert forge_py_exists or admitted, (
        "spec's forge.py sentence claims a CLI that is not on disk, with no "
        f"admission it is unbuilt:\n{sentence!r}"
    )
