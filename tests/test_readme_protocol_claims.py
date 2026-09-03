"""README claims that must stay pinned to the artifacts that make them true.

Each test names one fact somewhere else in the tree, and asserts the README
sentence that fact falsified is gone. The anchor comes first: if the fact
itself moves, the test says so instead of silently passing on a README that
drifted back.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CLAIMS = ROOT / "CLAIMS.md"
EVAL_GAME = ROOT / "scripts" / "eval_game.py"


def _flat(path: Path) -> str:
    """File text with every run of whitespace collapsed to one space.

    The README is hand-wrapped, so a banned phrase can straddle a newline and
    a plain substring search would miss it.
    """
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def test_headline_protocol_does_not_claim_zero_test_time_state_loads():
    """CLAIMS.md calls the old wording false; the README must not carry it.

    `CLAIMS.md` correction 1 records that every honest per-level episode
    calls `pool.load_worker_state` on an entrance state, and that only 1-1's
    is power-on equivalent. The README said "cold power-on, zero test-time
    state loads" at four sites.
    """
    claims = _flat(CLAIMS)
    assert (
        '"Cold power-on start, zero state loads at test time" is false for'
        in claims
    ), (
        "anchor gone: CLAIMS.md no longer records that the zero-state-loads "
        "wording is false. Re-derive this guard before deleting it."
    )

    readme = _flat(README)
    for banned in (
        "zero test-time state loads",
        "zero state loads at test time",
    ):
        assert banned not in readme, (
            f"README still says {banned!r}, which CLAIMS.md records as false "
            "for every per-level number. Use the HP-1 wording: cold power-on "
            "for 1-1 (whose entrance state is power-on equivalent), a "
            "declared entrance state for every other level, zero mid-episode "
            "state loads."
        )


def test_receipt_field_gap_is_recorded_as_closed():
    """The receipt fields the README called missing are written today.

    `scripts/eval_game.py` puts the requested protocol (`sticky_prob`,
    `start_jitter`, `eval_seed`) and the measured protocol (`sticky_applied`
    and the rest) into the same `eval.jsonl` row it writes. The README
    described both as absent and the fix as an open item.
    """
    src = EVAL_GAME.read_text(encoding="utf-8")
    for field in ("sticky_applied", "sticky_measured", "jitter_hist",
                  "n_episodes_delivered", "rom_sha256", "eval_seed"):
        assert f'"{field}"' in src, (
            f"anchor gone: {EVAL_GAME.name} no longer emits {field!r}. If the "
            "receipt really lost a field, the README paragraph goes back to "
            "naming the gap, and this guard is rewritten, not deleted."
        )

    readme = _flat(README)
    assert "folding them into the record is an open item" not in readme, (
        "README still calls the receipt-field gap an open item; it closed "
        "2026-08-27 (scripts/eval_game.py:331,374)."
    )
    assert "but **not** `sticky_prob`" not in readme, (
        "README still says eval.jsonl omits sticky_prob, start_jitter and "
        "eval_seed; scripts/eval_game.py writes all three into the row."
    )


def test_1_2_literature_claim_is_not_compressed():
    """CLAIMS.md bans the compression; the README must not print it.

    The approved quotable form is "we are aware of no published per-level 1-2
    clear rate under Machado sticky-0.25, in either direction". CLAIMS.md
    says it must never be compressed to "1-2 is unsolved by the field", and
    the README had appended "the level is an open problem at the field's
    frontier", which is that compression in other words.
    """
    claims = _flat(CLAIMS)
    assert "It must never be compressed to" in claims, (
        "anchor gone: CLAIMS.md no longer bans compressing the 1-2 "
        "literature claim. Re-derive this guard before deleting it."
    )

    readme = _flat(README)
    for banned in (
        "unsolved by the field",
        "open problem at the field's frontier",
    ):
        assert banned not in readme, (
            f"README compresses the 1-2 literature claim to {banned!r}. "
            "CLAIMS.md fixes the quotable form and bans the compression."
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
