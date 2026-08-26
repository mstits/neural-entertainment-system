"""Regression test for verify_receipt_trail's segment ordering.

Segment directories are named seg_{count:04d}_{name} — :04d is a
*minimum* width, not a cap. Past 9999 segments "seg_10000_x" has five
digits and sits next to four-digit "seg_9999_x", so a plain lexical
sort of the directory names visits 10000 (and 10001) before 9999,
walking a perfectly good hash chain out of chronological order and
reporting a broken chain where none exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from soak_harness import (  # noqa: E402
    OUTCOME_CRASH, Roster, RosterEntry, SegmentBackend, SegmentResult,
    SoakHarness, _canonical_json, _segment_sort_key, _sha256_bytes,
    _sha256_file, verify_receipt_trail,
)


def _write_chain_receipt(seg_dir: Path, prev_hash: str, info: dict) -> str:
    """Mirror SoakRunner._write_receipt's chaining exactly, standalone."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    body = dict(info)
    body["prev_sha256"] = prev_hash
    body["receipt_sha256"] = _sha256_bytes(_canonical_json(body))
    rp = seg_dir / "receipt.json"
    rp.write_text(json.dumps(body, sort_keys=True, indent=1))
    return _sha256_file(rp)


def test_verify_receipt_trail_crosses_9999_segment_boundary(tmp_path):
    soak_dir = tmp_path / "soak_x"
    soak_dir.mkdir()
    manifest_p = soak_dir / "soak_manifest.json"
    manifest_p.write_text(json.dumps({"kind": "soak_manifest"}))
    prev_hash = _sha256_file(manifest_p)

    seg_root = soak_dir / "segments"
    # True chronological order straddles the 4-digit width: 9998, 9999,
    # 10000, 10001. Nothing here is tampered, deleted, or reordered on
    # disk — the receipts are chained in the exact order they were
    # written.
    for idx in (9998, 9999, 10000, 10001):
        seg_dir = seg_root / f"seg_{idx:04d}_gamex"
        receipt = {
            "kind": "segment_receipt",
            "segment_index": idx,
            "name": "gamex",
            "outcome": "BUDGET",
        }
        prev_hash = _write_chain_receipt(seg_dir, prev_hash, receipt)

    problems = verify_receipt_trail(soak_dir)
    assert problems == []


def test_segment_sort_key_orders_numerically_past_four_digits():
    names = ["seg_10001_gamex", "seg_9999_gamex", "seg_10000_gamex",
             "seg_9998_gamex"]
    ordered = sorted((Path(n) for n in names), key=_segment_sort_key)
    assert [d.name for d in ordered] == [
        "seg_9998_gamex", "seg_9999_gamex",
        "seg_10000_gamex", "seg_10001_gamex",
    ]


class _CrashByReturnBackend(SegmentBackend):
    """Mimics SolverBackend on a hang: scores CRASH without raising.

    _run_one_segment must route a CRASH scored this way through
    log_intervention exactly like a raised exception would — a segment
    a human would need to look at can't be a silent, unbundled CRASH
    just because the backend returned instead of raised.
    """

    scoreable = True

    def preflight(self) -> None:
        pass

    def run_segment(self, entry: RosterEntry, seg_dir: Path) -> SegmentResult:
        return SegmentResult(
            OUTCOME_CRASH,
            {"diagnosis": "no progress record for >210s; killed"},
        )


def _one_entry_roster() -> Roster:
    entry = RosterEntry(name="alpha", config=str(Path(__file__)), budget_s=60.0)
    return Roster(path=Path("roster.yaml"), sha256="0" * 64,
                  audio_sink="null", entries=(entry,))


def test_crash_by_return_is_logged_as_intervention(tmp_path):
    harness = SoakHarness(
        roster=_one_entry_roster(),
        out_dir=tmp_path,
        backend=_CrashByReturnBackend(),
        duration_s=60.0,
        max_segments=1,
    )

    summary = harness.run()

    assert summary["outcomes"]["CRASH"] == 1
    assert summary["interventions"] == 1
    assert summary["passed_zero_interventions"] is False

    assert harness.interventions_path.exists()
    assert len(harness.interventions_path.read_text().splitlines()) == 1

    bundles = list((tmp_path / "diagnosis").glob("bundle_*"))
    assert len(bundles) == 1

    final_receipt = json.loads((tmp_path / "final_receipt.json").read_text())
    assert final_receipt["interventions"] == 1
    assert final_receipt["passed_zero_interventions"] is False
