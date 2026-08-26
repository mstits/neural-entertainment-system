"""Provenance gate for Learned-ledger training inputs (see CLAIMS.md).

Checks, failing loud on any violation:
  1. Every path in configs/demo_allowlist.txt exists.
  2. Every demo .npz under checkpoints/harvested_seeds/ is either on
     the allowlist or explicitly quarantined — no unaccounted demos.
  3. The quarantine directory still holds the Tier-3 artifacts named in
     CLAIMS.md (nothing quietly restored).
  4. No profile/manifest yaml under configs/ or checkpoints/ references
     a quarantined artifact.
  5. Every demo_anchor_paths entry in every tracked configs/**/*.yaml
     resolves to a file that actually exists — including the ones under
     the git-ignored runs/ tree, which the SEEDS sweep above never sees.
  6. Every entry in CLAIMS.md's FORGE ledger (### FORGE entries) is
     checked against the two of its four defining criteria that are
     actually mechanical — see check_forge_entries.

The allowlist is authoritative; provenance sidecars are advisory (a
sidecar mislabel has already happened once).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ALLOWLIST = REPO / "configs/demo_allowlist.txt"
SEEDS = REPO / "checkpoints/harvested_seeds"
QUARANTINE = REPO / "checkpoints/QUARANTINE_tier3"
CLAIMS = REPO / "CLAIMS.md"
QUARANTINED_NAMES = [
    "demos_4_2_full.npz",
    "demos_4_2_pilot.npz",
    "full_4_2_solution.npy",
    "full_4_2_trimmed.npy",
    "pilot_4_2.pt",
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _find_demo_anchor_paths(node) -> list:
    """demo_anchor_paths lives under whatever nested block a profile puts
    its PPO knobs in (`reinforce:` in most configs, top-level in others),
    so walk the whole parsed tree rather than assuming a fixed depth.
    """
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "demo_anchor_paths" and isinstance(v, list):
                found.extend(v)
            else:
                found.extend(_find_demo_anchor_paths(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_demo_anchor_paths(item))
    return found


def collect_demo_anchor_refs(configs_root: Path, repo: Path) -> dict[str, list[str]]:
    """Map each demo_anchor_paths entry (as written in the yaml) to the
    tracked configs that reference it, scanning every configs/**/*.yaml
    — not just top-level configs/*.yaml — so a profile filed under
    configs/overrides/ or configs/onboard/ is not invisible to the gate.
    """
    refs: dict[str, list[str]] = {}
    if not configs_root.exists():
        return refs
    for y in sorted(configs_root.rglob("*.yaml")):
        try:
            data = yaml.safe_load(y.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        paths = _find_demo_anchor_paths(data)
        if not paths:
            continue
        cfg_rel = str(y.relative_to(repo))
        for rel in paths:
            refs.setdefault(str(rel), []).append(cfg_rel)
    return refs


def check_demo_anchor_paths(
    repo: Path, seeds: Path
) -> tuple[list[str], dict[str, str]]:
    """Sweep demo_anchor_paths across every tracked config.

    checkpoints/harvested_seeds/ already gets a full allowlist sweep
    above; demo_anchor_paths pointing anywhere else (runs/ chief among
    them — git-ignored, so otherwise invisible to this gate entirely)
    are hashed and recorded here instead, and a reference to a file
    that does not exist on disk fails the check outright.
    """
    errors: list[str] = []
    hashes: dict[str, str] = {}
    refs = collect_demo_anchor_refs(repo / "configs", repo)
    for rel, configs in sorted(refs.items()):
        p = repo / rel
        if not p.exists():
            cfg_list = ", ".join(configs)
            errors.append(
                f"demo_anchor_paths references missing file: {rel} "
                f"(referenced by {cfg_list})")
            continue
        if seeds in p.parents:
            continue
        try:
            hashes[rel] = _sha256(p)
        except OSError:
            errors.append(f"demo_anchor_paths file unreadable for hashing: {rel}")
    return errors, hashes


def check_soak_trails(repo: Path) -> tuple[list[str], int, int]:
    """Verify every soak receipt trail under runs/soak/ (approved
    2026-08-15: the gate reads runs/soak/).

    The canonical verifier lives in scripts/soak_harness.py — the same
    code that writes the chains checks them; this gate never
    reimplements it. Semantics: no runs/soak/ -> nothing to verify;
    runs/soak/ present WITHOUT the harness -> unverifiable receipts are
    a failure, not a skip; harness present -> every soak dir's chain
    must verify. Selfcheck/non-scoreable trails still must
    chain-verify; they are counted separately so a stub run can never
    inflate the scoreable count. Returns (errors, verified, scoreable).
    """
    errors: list[str] = []
    verified = scoreable = 0
    soak_root = repo / "runs" / "soak"
    if not soak_root.exists():
        return errors, verified, scoreable
    soak_dirs = sorted(d for d in soak_root.iterdir() if d.is_dir())
    if not soak_dirs:
        return errors, verified, scoreable
    harness_p = repo / "scripts" / "soak_harness.py"
    if not harness_p.exists():
        errors.append(
            f"runs/soak/ holds {len(soak_dirs)} receipt trail(s) but "
            f"scripts/soak_harness.py (the chain verifier) is absent "
            f"in this checkout — unverifiable receipts fail the gate")
        return errors, verified, scoreable
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("soak_harness", harness_p)
    _sh = _ilu.module_from_spec(spec)
    spec.loader.exec_module(_sh)
    for d in soak_dirs:
        problems = _sh.verify_receipt_trail(d)
        if problems:
            for pr in problems[:5]:
                errors.append(f"soak {d.name}: {pr}")
            continue
        verified += 1
        fp = d / "final_receipt.json"
        if fp.exists():
            try:
                final = json.loads(fp.read_text())
                if (final.get("backend_scoreable")
                        and not final.get("selfcheck")):
                    scoreable += 1
            except (OSError, json.JSONDecodeError):
                errors.append(
                    f"soak {d.name}: final receipt unreadable after "
                    f"chain verify")
    return errors, verified, scoreable


FORGE_ENTRY_RE = re.compile(r'^\*\*FORGE(?:-([A-Z][A-Z-]*))?\s')
FORGE_SECTION_HEADER = "### FORGE entries"
_TOKEN_RE = re.compile(r'`([^`]+)`', re.DOTALL)
_FLAG_TOKEN_RE = re.compile(r'--[\w][\w-]*')
_DOTTED_TOKEN_RE = re.compile(r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+')
_TESTFILE_TOKEN_RE = re.compile(r'tests?/[\w./-]+\.py')
_FILELIKE_EXTS = (".py", ".md", ".json", ".yaml", ".yml", ".rs", ".txt",
                   ".npz", ".npy", ".pt", ".state", ".log", ".toml")
_DEFAULT_OFF_RE = re.compile(r'default[\s_-]+off', re.IGNORECASE)
_STATUS_WORD_RE = re.compile(r'\bPASS\b|\bFAIL\b|\bVOID\b|PENDING-VALIDATION')

# Criteria 1 and 2 of the FORGE definition (CLAIMS.md, "The FORGE
# ledger" -> "Definition") are judgments about how a mechanism was
# *found* and *written* — self-measured detection and agentic
# authorship. Nothing in the repo distinguishes "an agent noticed this
# in its own telemetry" from "a human noticed it and wrote it up as if
# an agent had" after the fact; that is a process/trust question for
# the human reviewing the commit, not a grep. They are named here, not
# silently skipped, so this function's silence is never mistaken for a
# pass.
FORGE_UNCHECKABLE_CRITERIA = [
    "criterion 1 (self-measured detection: the need was found in the "
    "system's own telemetry, not by a human or a walkthrough) is a "
    "provenance/process judgment — not mechanically checkable from repo "
    "contents",
    "criterion 2 (agentic authorship: design/implementation/review with "
    "no human algorithmic contribution) is a provenance/process "
    "judgment — not mechanically checkable from repo contents",
]


def parse_forge_entries(claims_path: Path) -> list[dict]:
    """Split CLAIMS.md's '### FORGE entries' section into individual
    entries. Entries are delimited by lines opening with a bold
    '**FORGE' header (optionally '**FORGE-TAG'); an "*Addendum*" or
    "*Status, updated*" sub-block that follows one (several entries have
    both) is part of that entry, not a new one, because it does not
    itself open with '**FORGE'. The section runs until the next '## '
    (top-level) heading.
    """
    if not claims_path.exists():
        return []
    lines = claims_path.read_text().split("\n")
    try:
        section_start = next(
            i for i, l in enumerate(lines) if l.strip() == FORGE_SECTION_HEADER)
    except StopIteration:
        return []
    section_end = len(lines)
    for i in range(section_start + 1, len(lines)):
        if lines[i].startswith("## "):
            section_end = i
            break
    starts = [i for i in range(section_start, section_end)
              if FORGE_ENTRY_RE.match(lines[i])]
    entries = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else section_end
        tag_m = FORGE_ENTRY_RE.match(lines[s])
        entries.append({
            "line": s + 1,
            "tag": tag_m.group(1),  # None for a bare "**FORGE —" header
            "header": lines[s].lstrip("*").strip(),
            "text": "\n".join(lines[s:e]),
        })
    return entries


def _forge_entry_tokens(entry_text: str) -> tuple[str | None, list[str]]:
    """Pull the first flag-like/dotted-config-key token and every cited
    tests/*.py path out of an entry's backtick spans, in document order.

    A markdown hard-wrap sometimes splits a single token across a line
    break inside its own backticks (e.g. "`tests/\\ntest_room_fp.py`",
    "`default\\noff`" outside backticks too — see the caller) — every
    backtick span's contents has its whitespace stripped before matching
    so a wrapped token still resolves to the identifier it names, rather
    than silently failing to match at all.
    """
    flag_token = None
    test_files: list[str] = []
    for m in _TOKEN_RE.finditer(entry_text):
        token = re.sub(r'\s+', '', m.group(1))
        if not token:
            continue
        if flag_token is None and _FLAG_TOKEN_RE.fullmatch(token):
            flag_token = token
        elif (flag_token is None and _DOTTED_TOKEN_RE.fullmatch(token)
                and not token.lower().endswith(_FILELIKE_EXTS)):
            flag_token = token
        if _TESTFILE_TOKEN_RE.fullmatch(token):
            test_files.append(token)
    return flag_token, test_files


def _source_py_files(repo: Path):
    for p in sorted(repo.rglob("*.py")):
        parts = p.relative_to(repo).parts
        if parts[0] in ("tests", ".venv", "node_modules"):
            continue
        yield p


def _flag_default_is_off(repo: Path, flag: str) -> tuple[bool | None, str]:
    """Best-effort lookup of a named flag/config-key's shipped default.

    CLI flags (`--foo`) are looked up as an argparse `add_argument`
    call; dotted config keys (`block.key`, e.g. `reinforce.redo_enabled`)
    are looked up by their last component only, since the dotted prefix
    is the yaml block name, not part of the Python identifier — the code
    reads it via `.get("key", default)` or `getattr(_, "key", default)`.
    Deliberately searches tracked *source* (scripts/, src/, …), never
    configs/*.yaml: a specific experiment config legitimately overriding
    a flag to True (see configs/mario_1_1_v27_seed0.yaml for
    redo_enabled) is not evidence against the *shipped* default being
    off, so grepping yaml would produce a false failure.

    Returns (is_off, detail); is_off is None when the flag/key could not
    be located in source at all — reported as a failure by the caller,
    since an unverifiable default-off claim is not a verified one.
    """
    if flag.startswith("--"):
        pat = re.compile(
            r'add_argument\(\s*[\'"]' + re.escape(flag) + r'[\'"].{0,400}?'
            r'default\s*=\s*([^\s,)]+)', re.DOTALL)
        not_found_detail = (
            f"flag {flag} not found via add_argument(...) in tracked "
            f"source")
    else:
        key = flag.rsplit(".", 1)[-1]
        pat = re.compile(
            r'\.get\(\s*[\'"]' + re.escape(key) + r'[\'"]\s*,\s*([^)]+?)\)'
            r'|getattr\([^,]+,\s*[\'"]' + re.escape(key) + r'[\'"]\s*,\s*'
            r'([^)]+?)\)')
        not_found_detail = (
            f"config key '{key}' (from {flag}) not found via "
            f".get(...)/getattr(...) in tracked source")
    for p in _source_py_files(repo):
        try:
            text = p.read_text()
        except OSError:
            continue
        m = pat.search(text)
        if not m:
            continue
        default_repr = next(g for g in m.groups() if g is not None).strip()
        is_off = default_repr.strip('\'"').lower() in ("off", "false", "none", "0")
        lineno = text[:m.start()].count("\n") + 1
        rel = p.relative_to(repo)
        return is_off, f"{rel}:{lineno} default={default_repr}"
    return None, not_found_detail


def _run_pytest(repo: Path, rel_path: str, timeout_s: float = 240.0) -> tuple[bool, str]:
    """Actually execute a cited test file with pytest — 'the tests that
    prove it' is a claim about tests that pass, not tests that merely
    exist. Bounded by a wall-clock timeout so a hung test cannot wedge
    this gate forever; a timeout is reported as inconclusive, not
    credited as a pass.
    """
    cmd = [sys.executable, "-m", "pytest", "-q", "--timeout=120", rel_path]
    try:
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                               timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"pytest {rel_path} did not finish within {timeout_s:.0f}s"
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    tail_line = tail[0] if tail else "(no output)"
    if proc.returncode == 0:
        return True, f"pytest {rel_path}: {tail_line}"
    return False, f"pytest {rel_path} FAILED (exit {proc.returncode}): {tail_line}"


def check_forge_entries(repo: Path, run_tests: bool = True) -> tuple[list[str], dict]:
    """Mechanically enforce as much of CLAIMS.md's FORGE ledger
    definition ("The FORGE ledger" -> "Definition", "### FORGE entries")
    as a script can actually verify from the repo alone.

    The definition names four criteria for FORGE-class status; only two
    have real, checkable structure:

      (1) self-measured detection and (2) agentic authorship are
          judgments about how a mechanism was found and written — see
          FORGE_UNCHECKABLE_CRITERIA, returned in `report["uncheckable"]`
          rather than silently skipped.
      (3) 'the standard gates' is checked two ways per entry: the entry
          must pair a literal "default off" claim with a specific named
          flag/config-key, and that flag's *current* shipped default
          must actually read as off (see _flag_default_is_off); and the
          entry must cite at least one real tests/*.py file, which must
          exist and (bounded by a wall-clock timeout) actually pass
          under pytest right now, not merely be named.
      (4) 'honest status' is checked as a literal search for one of the
          ledger's own explicit-verdict words — PASS, FAIL, VOID,
          PENDING-VALIDATION — matched case-sensitively and whole-word so
          ordinary prose ("an audit pass", "fail-any-quarantine",
          "passed 5/5") cannot satisfy it by accident. This is a narrow,
          surface-lexical check by design (see the module docstring's
          stance on sidecars): entries that state their status in other
          words — a CERTIFIED/VALIDATED/SHIPPED header tag, or prose like
          "the instrument is certified, the games remain unsolved" —
          read as an honest status paragraph to a human but will not
          satisfy this specific check, and are reported as failing it,
          not quietly credited.

    Returns (errors, report). `errors` follows this module's existing
    convention — one string per concrete, checkable violation, meant to
    be extended onto main()'s error list. `report` carries a per-entry
    breakdown (`report["entries"]`) plus `report["uncheckable"]`.
    """
    errors: list[str] = []
    entries = parse_forge_entries(repo / "CLAIMS.md")
    entry_reports = []
    test_result_cache: dict[str, tuple[bool, str]] = {}

    for n, entry in enumerate(entries, start=1):
        label = f"FORGE entry #{n} (CLAIMS.md:{entry['line']}, {entry['header'][:70]!r})"
        flag_token, test_files = _forge_entry_tokens(entry["text"])
        entry_errors = []

        # (3a) default-off flag.
        claims_default_off = bool(_DEFAULT_OFF_RE.search(
            re.sub(r'\s+', ' ', entry["text"])))
        if not claims_default_off:
            flag_result = "fail"
            flag_detail = "entry states no 'default off' claim anywhere"
            entry_errors.append(f"{label}: {flag_detail}")
        elif flag_token is None:
            flag_result = "fail"
            flag_detail = "claims default-off but names no specific, greppable flag/config-key"
            entry_errors.append(f"{label}: {flag_detail}")
        else:
            is_off, detail = _flag_default_is_off(repo, flag_token)
            if is_off is None:
                flag_result = "fail"
                flag_detail = f"names {flag_token!r} but {detail}"
                entry_errors.append(f"{label}: {flag_detail}")
            elif not is_off:
                flag_result = "fail"
                flag_detail = f"{flag_token!r} does NOT default off ({detail})"
                entry_errors.append(f"{label}: {flag_detail}")
            else:
                flag_result = "pass"
                flag_detail = f"{flag_token!r} confirmed default-off ({detail})"

        # (3b) cited tests exist and pass.
        if not test_files:
            tests_result = "fail"
            tests_detail = "cites no tests/*.py file at all"
            entry_errors.append(f"{label}: {tests_detail}")
        else:
            missing = [t for t in test_files if not (repo / t).exists()]
            if missing:
                tests_result = "fail"
                tests_detail = f"cited test file(s) do not exist: {', '.join(missing)}"
                entry_errors.append(f"{label}: {tests_detail}")
            elif not run_tests:
                tests_result = "pass"
                tests_detail = f"cited and present (not executed): {', '.join(test_files)}"
            else:
                fails = []
                oks = []
                for t in test_files:
                    if t not in test_result_cache:
                        test_result_cache[t] = _run_pytest(repo, t)
                    ok, detail = test_result_cache[t]
                    (oks if ok else fails).append(detail)
                if fails:
                    tests_result = "fail"
                    tests_detail = "; ".join(fails)
                    entry_errors.append(f"{label}: {tests_detail}")
                else:
                    tests_result = "pass"
                    tests_detail = "; ".join(oks)

        # (4) explicit status word.
        if _STATUS_WORD_RE.search(entry["text"]):
            status_result = "pass"
            status_detail = "found an explicit PASS/FAIL/VOID/PENDING-VALIDATION word"
        else:
            status_result = "fail"
            status_detail = "no explicit PASS/FAIL/VOID/PENDING-VALIDATION word anywhere in the entry"
            entry_errors.append(f"{label}: {status_detail}")

        overall = "pass" if not entry_errors else "fail"
        entry_reports.append({
            "n": n, "line": entry["line"], "tag": entry["tag"],
            "header": entry["header"], "overall": overall,
            "flag_default_off": (flag_result, flag_detail),
            "cited_tests": (tests_result, tests_detail),
            "explicit_status": (status_result, status_detail),
        })
        errors.extend(entry_errors)

    report = {
        "total": len(entries),
        "passed": sum(1 for r in entry_reports if r["overall"] == "pass"),
        "entries": entry_reports,
        "uncheckable": FORGE_UNCHECKABLE_CRITERIA,
    }
    return errors, report


def main() -> int:
    errors: list[str] = []
    allow = [ln.strip() for ln in ALLOWLIST.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    allowed = set(allow)

    for rel in allow:
        if not (REPO / rel).exists():
            errors.append(f"allowlisted but missing: {rel}")

    # Every demo bank (.npz — what the trainer's demo loader consumes)
    # under SEEDS must be allowlisted, RECURSIVE so a bank dropped in a
    # subdir is not invisible (the original top-level glob missed it).
    # .npy/.state artifacts are harvester/replay inputs, not demo banks,
    # so they are covered by the quarantine hash check below rather than
    # the allowlist.
    if SEEDS.exists():
        for f in sorted(SEEDS.rglob("*.npz")):
            rel = str(f.relative_to(REPO))
            if rel not in allowed:
                errors.append(
                    f"demo bank not on allowlist (add or quarantine): {rel}")

    # demo_anchor_paths referenced from tracked configs, resolved and
    # hashed for anything outside the SEEDS sweep above (chiefly the
    # git-ignored runs/ tree) — see check_demo_anchor_paths.
    demo_anchor_errors, demo_anchor_hashes = check_demo_anchor_paths(REPO, SEEDS)
    errors.extend(demo_anchor_errors)

    # Content-hash every quarantined file, then confirm NO copy of it
    # exists anywhere under checkpoints/ outside the quarantine — a
    # restore-by-copy (not just moving the original back) must be caught.
    quarantined_hashes = {}
    for name in QUARANTINED_NAMES:
        qp = QUARANTINE / name
        if not qp.exists():
            errors.append(f"quarantined artifact missing from quarantine: {name}")
        else:
            try:
                quarantined_hashes[_sha256(qp)] = name
            except OSError:
                pass
    ck = REPO / "checkpoints"
    if ck.exists() and quarantined_hashes:
        for f in ck.rglob("*"):
            if not f.is_file() or QUARANTINE in f.parents:
                continue
            if f.suffix not in (".npz", ".npy", ".pt", ".state"):
                continue
            try:
                if (h := _sha256(f)) in quarantined_hashes:
                    errors.append(
                        f"quarantined artifact {quarantined_hashes[h]} "
                        f"copied back into the tree at "
                        f"{f.relative_to(REPO)}")
            except OSError:
                continue

    # Reference scan across every manifest format (.yaml/.yml/.json), not
    # just .yaml.
    ref_roots = [REPO / "configs", REPO / "checkpoints"]
    for root in ref_roots:
        if not root.exists():
            continue
        for y in root.rglob("*"):
            if y.suffix not in (".yaml", ".yml", ".json"):
                continue
            if QUARANTINE in y.parents:
                continue
            try:
                text = y.read_text()
            except OSError:
                continue
            for name in QUARANTINED_NAMES:
                if name in text:
                    errors.append(f"{y.relative_to(REPO)} references quarantined {name}")

    soak_errors, soak_verified, soak_scoreable = check_soak_trails(REPO)
    errors.extend(soak_errors)

    # FORGE ledger (CLAIMS.md, "### FORGE entries") — see check_forge_entries
    # for exactly which of the definition's four criteria this can and
    # cannot verify from the repo alone.
    forge_errors, forge_report = check_forge_entries(REPO)
    errors.extend(forge_errors)

    if errors:
        print("PROVENANCE CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        print("FORGE ledger — criteria not mechanically checkable "
              "(reported, not silently skipped):")
        for c in forge_report["uncheckable"]:
            print(f"  - {c}")
        return 1
    print(f"provenance check OK: {len(allowed)} allowlisted demos, "
          f"{len(QUARANTINED_NAMES)} artifacts confirmed quarantined, "
          f"{len(demo_anchor_hashes)} demo_anchor_paths hashed, "
          f"{soak_verified} soak trail(s) chain-verified "
          f"({soak_scoreable} scoreable), "
          f"{forge_report['passed']}/{forge_report['total']} FORGE "
          f"entries mechanically clean")
    print("FORGE ledger — criteria not mechanically checkable "
          "(reported, not silently skipped):")
    for c in forge_report["uncheckable"]:
        print(f"  - {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
