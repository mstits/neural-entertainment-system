"""README must carry the number CLAIMS.md marks as live.

CLAIMS.md is the ledger; README is the summary a reader meets first. When a
ledger addendum supersedes a headline figure, README is the surface that keeps
quoting the old one, because nothing mechanical connects the two documents.

The case pinned here is 1-2 on Super Mario Bros. README's learned-results
section described 1-2 only as a pre-registered three-seed negative for the
CGSA-PPO policy class. That sentence is still true about *that campaign*, and
about power-on, but the ledger has since moved: the consolidation-round-2
checkpoint clears 1-2 at 31/100 under the canonical protocol
(`--eval-rng per-episode`, 50 episodes x seeds {7,101}), and CLAIMS.md names
that the live number. Both facts belong in README — the at-entrance positive
and the power-on negative are different measurements, and dropping either one
misleads in a different direction — so this test asserts the positive is
present rather than that the negative is gone.

The trigger is CLAIMS.md's own supersession marker, not a hardcoded belief that
1-2 always has a live number: if the ledger stops naming one, this test says so
and stops, rather than pinning README to a figure the ledger has retired.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
CLAIMS = REPO / "CLAIMS.md"

# The phrase CLAIMS.md uses to mark a figure as the one currently in force.
LIVE_1_2_MARKER = "is the ledger's live 1-2 number"
# The canonical-protocol figure that marker currently governs, as
# CLAIMS.md's 2026-08-27 addendum states it.
LIVE_1_2_NUMBER = "31/100"


def test_readme_states_the_live_1_2_number():
    claims = CLAIMS.read_text()
    if LIVE_1_2_MARKER not in claims:
        pytest.skip(
            f"CLAIMS.md no longer marks a live 1-2 number ({LIVE_1_2_MARKER!r} "
            f"is absent); there is nothing for README to be behind on. If the "
            f"ledger moved on purpose, retire this test with it.")

    assert LIVE_1_2_NUMBER in README.read_text(), (
        f"CLAIMS.md marks a live 1-2 number ({LIVE_1_2_MARKER!r}) but README.md "
        f"never states {LIVE_1_2_NUMBER!r} — the canonical-protocol rate for "
        f"the consolidation-round-2 checkpoint. README is quoting a superseded "
        f"picture of 1-2.")


def test_the_live_1_2_number_is_the_one_claims_records():
    """The literal above is not a free-floating constant: CLAIMS.md must
    state it too, so a ledger correction cannot leave this test pinning
    README to a number the ledger has itself abandoned."""
    claims = CLAIMS.read_text()
    if LIVE_1_2_MARKER not in claims:
        pytest.skip("no live 1-2 number in CLAIMS.md; nothing to cross-check")

    assert LIVE_1_2_NUMBER in claims, (
        f"{LIVE_1_2_NUMBER!r} is not in CLAIMS.md. This test's constant has "
        f"drifted from the ledger it is supposed to mirror; fix the constant "
        f"(and README) against the ledger, not the other way round.")
