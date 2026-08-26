"""Unit tests for scripts/adjudicate_soak.py — trails faked on disk.

Builds miniature soak receipt trails (manifest, final receipt, segment
receipts, solution sidecars, action tapes) and checks that the static
adjudication checker passes a clean trail and flags each defect class:
tampered tape, duplicated tape, missed clear in a BUDGET tail, weak
attempt, non-scoreable receipts, selfcheck stubs, count mismatches, and
broken rotation order.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from adjudicate_soak import check_trail, compare_baseline, main  # noqa: E402

ROSTER = ["alpha", "beta"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def make_trail(root: Path, outcomes, *, selfcheck=False, scoreable=True,
               final_overrides=None, name="soak_x") -> Path:
    """outcomes: list of (game, outcome, detail_overrides)."""
    soak = root / name
    _write_json(soak / "soak_manifest.json", {
        "selfcheck": selfcheck, "backend_scoreable": scoreable,
        "roster": [{"name": g} for g in ROSTER],
        "roster_sha256": "0" * 64,
    })
    tally: dict = {}
    for idx, (game, outcome, over) in enumerate(outcomes):
        seg = soak / "segments" / f"seg_{idx + 1:04d}_{game}"
        seg.mkdir(parents=True)
        detail = {"argv": ["--seed", str(1000 + idx)], "solver_exit": 0}
        if outcome == "CLEAR":
            tape_bytes = over.pop("tape_bytes", f"tape-{name}-{idx}".encode())
            steps = over.pop("steps", 100 + idx)
            side = seg / "solve" / "solutions" / "sol_000.json"
            _write_json(side, {"replay_verified": True, "steps": steps})
            tape = side.parent / "sol_000.actions.npy"
            tape.write_bytes(tape_bytes)
            detail.update({
                "solution_sidecar": str(side),
                "sidecar_sha256": _sha(side.read_bytes()),
                "tape_sha256": _sha(tape_bytes),
                "solution_steps": steps,
            })
        elif outcome == "BUDGET":
            detail["progress_last"] = {"solutions": 0, "steps": 500_000}
        detail.update(over)
        _write_json(seg / "receipt.json", {
            "name": game, "outcome": outcome, "backend_scoreable": scoreable,
            "detail": detail,
        })
        tally[outcome] = tally.get(outcome, 0) + 1
    final = {
        "status": "completed", "segments": len(outcomes),
        "outcomes": {k: tally.get(k, 0) for k in
                     ("CLEAR", "BUDGET", "STALL", "CRASH", "UNSCORABLE")},
        # The harness writes passed_zero_interventions unconditionally and
        # it is inside the hash chain, so a fixture without it was not
        # modelling a real trail. Added when the checker started enforcing
        # it (adversary audit #3).
        "interventions": 0, "passed_zero_interventions": True,
        "backend_scoreable": scoreable, "selfcheck": selfcheck,
    }
    final.update(final_overrides or {})
    _write_json(soak / "final_receipt.json", final)
    return soak


CLEAN = [("alpha", "CLEAR", {}), ("beta", "BUDGET", {}),
         ("alpha", "CLEAR", {}), ("beta", "BUDGET", {})]


def test_clean_trail_passes(tmp_path):
    soak = make_trail(tmp_path, CLEAN)
    report = check_trail(soak)
    assert report["problems"] == []
    assert report["outcomes"] == {"CLEAR": 2, "BUDGET": 2}
    assert report["per_game"] == {"alpha": {"CLEAR": 2},
                                  "beta": {"BUDGET": 2}}
    assert main([str(soak)]) == 0


def test_tampered_tape_flagged(tmp_path):
    soak = make_trail(tmp_path, CLEAN)
    tape = next(soak.glob("segments/seg_0001_*/solve/solutions/*.npy"))
    tape.write_bytes(b"doctored")
    problems = check_trail(soak)["problems"]
    assert any("tape sha256 mismatch" in p for p in problems)
    assert main([str(soak)]) == 1


def test_tampered_sidecar_flagged(tmp_path):
    soak = make_trail(tmp_path, CLEAN)
    side = next(soak.glob("segments/seg_0001_*/solve/solutions/sol_000.json"))
    side.write_text(json.dumps({"replay_verified": True, "steps": 999}))
    problems = check_trail(soak)["problems"]
    assert any("sidecar sha256 mismatch" in p for p in problems)


def test_duplicate_tape_flagged(tmp_path):
    outcomes = [("alpha", "CLEAR", {"tape_bytes": b"same"}),
                ("beta", "BUDGET", {}),
                ("alpha", "CLEAR", {"tape_bytes": b"same"}),
                ("beta", "BUDGET", {})]
    problems = check_trail(make_trail(tmp_path, outcomes))["problems"]
    assert any("duplicates" in p for p in problems)


def test_unverified_sidecar_flagged(tmp_path):
    soak = make_trail(tmp_path, CLEAN)
    side = next(soak.glob("segments/seg_0001_*/solve/solutions/sol_000.json"))
    sc = json.loads(side.read_text())
    sc["replay_verified"] = False
    side.write_text(json.dumps(sc))
    rec_path = soak / "segments" / side.parts[-4] / "receipt.json"
    rec = json.loads(rec_path.read_text())
    rec["detail"]["sidecar_sha256"] = _sha(side.read_bytes())
    rec_path.write_text(json.dumps(rec))
    problems = check_trail(soak)["problems"]
    assert any("not replay_verified" in p for p in problems)


def test_missed_clear_in_budget_tail_flagged(tmp_path):
    outcomes = list(CLEAN)
    outcomes[1] = ("beta", "BUDGET",
                   {"progress_last": {"solutions": 1, "steps": 500_000}})
    problems = check_trail(make_trail(tmp_path, outcomes))["problems"]
    assert any("missed clear" in p for p in problems)


def test_weak_attempt_flagged(tmp_path):
    outcomes = list(CLEAN)
    outcomes[1] = ("beta", "BUDGET",
                   {"progress_last": {"solutions": 0, "steps": 12}})
    problems = check_trail(make_trail(tmp_path, outcomes))["problems"]
    assert any("below floor" in p for p in problems)


def test_selfcheck_trail_refused(tmp_path):
    soak = make_trail(tmp_path, CLEAN, selfcheck=True)
    problems = check_trail(soak)["problems"]
    assert any("selfcheck" in p for p in problems)


def test_non_scoreable_refused(tmp_path):
    soak = make_trail(tmp_path, CLEAN, scoreable=False)
    problems = check_trail(soak)["problems"]
    assert any("not scoreable" in p for p in problems)


def test_final_count_mismatch_flagged(tmp_path):
    soak = make_trail(tmp_path, CLEAN,
                      final_overrides={"outcomes": {
                          "CLEAR": 3, "BUDGET": 1, "STALL": 0,
                          "CRASH": 0, "UNSCORABLE": 0}})
    problems = check_trail(soak)["problems"]
    assert any("count mismatch" in p for p in problems)


def test_missing_final_receipt_flagged(tmp_path):
    soak = make_trail(tmp_path, CLEAN)
    (soak / "final_receipt.json").unlink()
    problems = check_trail(soak)["problems"]
    assert any("did not complete" in p for p in problems)


def test_in_flight_scores_closed_segments_without_a_final_receipt(tmp_path):
    # A running soak has no final receipt and its newest segment has no
    # receipt yet. In-flight mode still scores everything that closed,
    # so a content defect surfaces hours before the verdict run.
    soak = make_trail(tmp_path, CLEAN)
    (soak / "final_receipt.json").unlink()
    last = sorted((soak / "segments").glob("seg_*"))[-1]
    (last / "receipt.json").unlink()

    report = check_trail(soak, in_flight=True)
    assert report["problems"] == []
    assert report["mode"] == "in_flight"
    assert report["scored_segments"] == 3 and report["segments"] == 4
    assert any("NOT a verdict" in i for i in report["info"])
    assert any("in progress" in i for i in report["info"])
    assert main([str(soak), "--in-flight"]) == 0


def test_in_flight_still_flags_content_defects(tmp_path):
    # The tolerance is only for absence at the tail; everything that
    # closed is judged exactly as strictly as in a verdict run.
    soak = make_trail(tmp_path, CLEAN)
    (soak / "final_receipt.json").unlink()
    tape = next(soak.glob("segments/seg_0001_*/solve/solutions/*.npy"))
    tape.write_bytes(b"doctored")
    problems = check_trail(soak, in_flight=True)["problems"]
    assert any("tape sha256 mismatch" in p for p in problems)
    assert main([str(soak), "--in-flight"]) == 1


def test_in_flight_does_not_excuse_a_gap_before_the_last_segment(tmp_path):
    # Only the newest segment may legitimately lack a receipt. An
    # earlier gap is a real hole in the trail in either mode.
    soak = make_trail(tmp_path, CLEAN)
    (soak / "final_receipt.json").unlink()
    first = sorted((soak / "segments").glob("seg_*"))[0]
    (first / "receipt.json").unlink()
    problems = check_trail(soak, in_flight=True)["problems"]
    assert any("receipt.json missing" in p for p in problems)


def test_verdict_mode_still_requires_a_completed_soak(tmp_path):
    # The in-flight escape hatch must not leak into the verdict path.
    soak = make_trail(tmp_path, CLEAN)
    (soak / "final_receipt.json").unlink()
    report = check_trail(soak)
    assert report["mode"] == "verdict"
    assert any("did not complete" in p for p in report["problems"])
    assert main([str(soak)]) == 1


def test_rotation_break_flagged(tmp_path):
    outcomes = [("alpha", "CLEAR", {}), ("alpha", "CLEAR", {})]
    soak = make_trail(tmp_path, outcomes,
                      final_overrides={"outcomes": {
                          "CLEAR": 2, "BUDGET": 0, "STALL": 0,
                          "CRASH": 0, "UNSCORABLE": 0},
                          "segments": 2})
    problems = check_trail(soak)["problems"]
    assert any("rotation order" in p for p in problems)


def test_unscorable_routed_to_info_not_problems(tmp_path):
    outcomes = list(CLEAN)
    outcomes[1] = ("beta", "UNSCORABLE", {})
    soak = make_trail(tmp_path, outcomes)
    report = check_trail(soak)
    assert report["problems"] == []
    assert any("adjudicate by hand" in i for i in report["info"])


def test_baseline_same_seed_overlap_reported(tmp_path):
    a = make_trail(tmp_path, CLEAN, name="soak_a")
    b = make_trail(tmp_path, CLEAN, name="soak_b")
    info: list = []
    compare_baseline(b, a, info)
    assert any("4 same-index segment(s)" in i for i in info)


# --- adversary audit #3: the pass criterion the checker used to drop ------
# A fully self-consistent trail can carry human interventions. Until this
# was fixed, such a trail passed BOTH --verify and this checker with
# "0 problem(s)", exit 0 — the count surfaced only as an info line and
# passed_zero_interventions was never read at all.


def test_interventions_are_a_problem_not_a_note(tmp_path):
    soak = make_trail(tmp_path, CLEAN, final_overrides={
        "interventions": 7, "passed_zero_interventions": False})
    report = check_trail(soak)
    assert any("intervention" in p for p in report["problems"]), report
    assert main([str(soak)]) == 1


def test_interventions_caught_even_if_the_pass_bit_says_otherwise(tmp_path):
    """A trail claiming it passed while logging interventions is worse than
    one that admits it: both must fail."""
    soak = make_trail(tmp_path, CLEAN, final_overrides={
        "interventions": 3, "passed_zero_interventions": True})
    report = check_trail(soak)
    assert any("3 human intervention" in p for p in report["problems"])
    assert main([str(soak)]) == 1


def test_missing_pass_bit_is_a_problem(tmp_path):
    soak = make_trail(tmp_path, CLEAN)
    final = json.loads((soak / "final_receipt.json").read_text())
    del final["passed_zero_interventions"]
    _write_json(soak / "final_receipt.json", final)
    report = check_trail(soak)
    assert any("passed_zero_interventions" in p for p in report["problems"])
    assert main([str(soak)]) == 1


def test_all_crash_trail_is_not_reported_as_uneventful(tmp_path, capsys):
    """Every segment CRASHed. That is not automatically a failed soak, but
    it must never print as a clean, unremarkable run."""
    crashes = [(g, "CRASH", {}) for g in ("alpha", "beta") * 2]
    soak = make_trail(tmp_path, crashes)
    report = check_trail(soak)
    assert report["outcomes"] == {"CRASH": 4}
    assert any("hand adjudication" in i and "CRASH=4" in i
               for i in report["info"]), report["info"]
    main([str(soak)])
    assert "CRASH=4" in capsys.readouterr().err


def test_clean_trail_still_passes_after_the_new_checks(tmp_path):
    """The new checks must not fire on an honest unattended trail."""
    soak = make_trail(tmp_path, CLEAN)
    assert check_trail(soak)["problems"] == []
    assert main([str(soak)]) == 0
