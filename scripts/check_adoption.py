#!/usr/bin/env python3
"""Adoption census: catches config knobs and safety wiring nobody uses.

Four independent sub-checks, each importable and independently testable:

  (a) census_inert_keys()     -- registry keys that ZERO configs set
                                  (INERT-BY-NONADOPTION)
  (b) census_lockless_writers() -- scripts writing under runs/ or
                                  checkpoints/ that never import
                                  src.utils.run_lock
  (c) census_solve_keys()     -- solve: key adoption across the 45
                                  solve-shaped configs, checked against
                                  the proposed KNOWN_SOLVE_KEYS registry
  (d) census_unquarantined_globs() -- glob-based checkpoint/run readers
                                  that do not filter through
                                  is_quarantined()

Exit code is nonzero iff any sub-check finds a violation, so this can be
wired into `make test` / `make check` as a fast, no-pytest gate. Run
standalone for a human-readable report; `--json` for machine output.

Read-only: never writes into --repo. All four sub-checks are pure
functions over the filesystem plus (a) an import of
src.training.config_schema -- the registry lives there so this stays a
census of that module's own truth rather than a second copy of it that
can drift.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# (d) is_quarantined -- generalizes provenance_check.py:35's single
# QUARANTINE_tier3 constant to the naming conventions actually in use
# under checkpoints/ today (census: see census_unquarantined_globs).
# ---------------------------------------------------------------------------

_QUARANTINE_DIR_NAMES = {"quarantine", "quarantine_mock_test_pollution"}
_QUARANTINE_DIR_PREFIXES = ("quarantine_", "_archived_", "archived_")
# Directory-name suffix: catches "demos_quarantine" and any other
# "<label>_quarantine" directory alongside the exact-name and prefix
# forms above -- a plain "quarantine_" prefix check does not match a
# quarantine label that comes BEFORE the word, which is exactly the
# shape checkpoints/bc_1_3/demos_quarantine, bc_1_4/demos_quarantine,
# and bc_2_1/demos_quarantine use.
_QUARANTINE_DIR_SUFFIXES = ("_quarantine",)
_QUARANTINE_SUFFIXES = (".stale-pixel", ".archived")
_QUARANTINE_INFIX = ".poisoned"  # matches .POISONED_dmap and .POISONED.pkl


def is_quarantined(path: Path | str) -> bool:
    """True if any path component or the filename marks this artifact
    quarantined, under the conventions provenance_check.py (QUARANTINE_tier3)
    and checkpoints/ (`_archived_*`, `*_quarantine` dirs, `*.stale-pixel`,
    `*.POISONED*`, `*.archived`, `*quarantine*` dirs) actually use.

    Conservative in the direction of "flag it": a path is quarantined if
    ANY component matches, not just the leaf -- an artifact nested under
    checkpoints/QUARANTINE_tier3/whatever.pt is quarantined even though
    its own filename is clean.
    """
    p = Path(path)
    for part in p.parts:
        lower = part.lower()
        if part == "QUARANTINE_tier3":
            return True
        if lower in _QUARANTINE_DIR_NAMES:
            return True
        if lower.startswith(_QUARANTINE_DIR_PREFIXES):
            return True
        if lower.endswith(_QUARANTINE_DIR_SUFFIXES):
            return True
        if lower.endswith(_QUARANTINE_SUFFIXES):
            return True
        if _QUARANTINE_INFIX in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# (a) INERT-BY-NONADOPTION -- registry keys zero configs set
# ---------------------------------------------------------------------------

# (registry attr name, path from a parsed profile dict to the dict whose
# keys should be checked against that registry). All our blocks are plain
# nesting, no lists-of-dicts, so a tuple of literal dict keys is enough.
_REGISTRY_BLOCKS = [
    ("KNOWN_TOP_KEYS", ()),
    ("KNOWN_REINFORCE_KEYS", ("reinforce",)),
    ("KNOWN_BACKWARD_CURRICULUM_KEYS", ("reinforce", "backward_curriculum")),
    ("KNOWN_CONSOLIDATE_LEVEL_KEYS", ("reinforce", "consolidate_level")),
    ("KNOWN_CONSOLIDATE_PROBE_KEYS", ("reinforce", "consolidate_level", "probe")),
    ("KNOWN_SIL_KEYS", ("reinforce", "sil")),
    ("KNOWN_ADVERSARY_KEYS", ("reinforce", "adversary")),
]


def _descend(d: dict, path: tuple) -> dict | None:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, dict) else None


def census_inert_keys(repo: Path, config_schema) -> dict[str, list[str]]:
    """Every registered key, in every registry block in _REGISTRY_BLOCKS,
    that zero parsed config files set. Returns {registry_name: [keys]},
    empty lists omitted.
    """
    import yaml

    observed: dict[str, set[str]] = {name: set() for name, _ in _REGISTRY_BLOCKS}
    for yml in sorted(repo.glob("configs/**/*.yaml")):
        try:
            doc = yaml.safe_load(yml.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        for name, path in _REGISTRY_BLOCKS:
            block = _descend(doc, path)
            if block:
                observed[name].update(block.keys())

    inert: dict[str, list[str]] = {}
    for name, _ in _REGISTRY_BLOCKS:
        registry: frozenset[str] = getattr(config_schema, name)
        missing = sorted(registry - observed[name])
        if missing:
            inert[name] = missing
    return inert


# ---------------------------------------------------------------------------
# (b) lockless writers under runs/ or checkpoints/
#
# AST-based, not a flat text/regex scan: a script that writes into some
# OTHER directory (docs/receipts/..., say) while merely reading replay
# data from runs/ elsewhere in the file is not a lockless writer, and a
# naive "write-call regex co-occurs anywhere with a runs/ path-literal
# regex" scan flags it anyway (verified false positive: scripts/show_fx.py
# writes only to docs/receipts/show_lane and its --out-dir arg, but reads
# replay sources from runs/live_show/ two functions away in the same
# file -- the co-occurrence scan can't tell "reads from A, writes to B"
# from "writes to A"). Instead: track which local variables are actually
# derived from a runs/ or checkpoints/ path literal (by name, transitively
# through `var = other_tracked_var / "x"`-shaped assignments), then only
# count a write-shaped call whose target expression mentions one of
# those variables (or a runs/checkpoints/ literal directly).
#
# Path-tracking sources, fixed-pointed together (each can feed the next):
#   1. `var = <expr containing a runs/checkpoints literal, or an
#      already-tracked name>` -- plain Assign/AnnAssign.
#   2. a function parameter whose DEFAULT value contains a literal or an
#      already-tracked name (`def f(out_dir=REPO / "runs" / "x"): ...` or
#      `def f(runs_dir=RUNS_DIR): ...`) -- these never appear as Assign
#      nodes, so a purely-Assign walk misses the module-constant-into-
#      keyword-argument-default shape entirely (confirmed miss:
#      scripts/onboard_game.py's `runs_dir: str | Path = RUNS_DIR`).
#   3. an `argparse` `add_argument(..., default=<expr containing a
#      literal>)` call -- the resulting `args.<dest>` attribute access is
#      treated as a tracked name (confirmed misses: scripts/soak_harness.py's
#      `--out-root` default `REPO / "runs" / "soak"`, and
#      scripts/critic_explained_variance.py's `--out-dir` default
#      `"runs/v29_stability/f1_explained_variance"` -- both flow straight
#      into a write with no intervening Assign of a literal).
# ---------------------------------------------------------------------------

_PATH_LITERAL_RE = re.compile(r"""['"](?:\./)?(runs|checkpoints)(?:/|['"])""")
_RUN_LOCK_IMPORT_RE = re.compile(
    r"from\s+src\.utils\.run_lock\s+import|import\s+src\.utils\.run_lock\b"
)
_WRITE_METHODS = {"write_text", "write_bytes", "mkdir"}


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _add_argument_dest(call: ast.Call) -> str | None:
    """Best-effort argparse dest inference for a `.add_argument(...)`
    call: explicit `dest=` wins; otherwise the first long-option-shaped
    positional string arg (`--out-root` -> `out_root`), matching
    argparse's own default-dest rule closely enough for this census.
    """
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                return kw.value.value
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            if a.value.startswith("--"):
                return a.value[2:].replace("-", "_")
    return None


def _tracked_path_candidates(
    tree: ast.AST,
) -> list[tuple[str, str, ast.AST | None]]:
    """(name, source-text, value-node) triples from all three sources
    described above -- fed into the fixed-point closure in
    _tracked_path_vars. value-node is the actual RHS/default AST node
    (None only for the synthetic argparse-default case where we already
    have the keyword value node directly), kept alongside the unparsed
    text so the fixed point can tell a narrow wrapper call
    (`Path(x)`, `_default_out_dir(x, selfcheck=True)`) apart from an
    arbitrary multi-argument call that merely happens to take a tracked
    value as ONE of several unrelated arguments -- see
    _is_unrestricted_value's docstring for why that distinction matters.
    """
    candidates: list[tuple[str, str, ast.AST | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            src = _unparse(node.value)
            for t in node.targets:
                if isinstance(t, ast.Name):
                    candidates.append((t.id, src, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                candidates.append((node.target.id, _unparse(node.value), node.value))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            positional = [*a.posonlyargs, *a.args]
            if a.defaults:
                for arg, default in zip(positional[-len(a.defaults):], a.defaults):
                    candidates.append((arg.arg, _unparse(default), default))
            for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                if default is not None:
                    candidates.append((arg.arg, _unparse(default), default))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "add_argument":
                dest = _add_argument_dest(node)
                if dest:
                    for kw in node.keywords:
                        if kw.arg == "default":
                            candidates.append(
                                (f"args.{dest}", _unparse(kw.value), kw.value)
                            )
    return candidates


def _is_unrestricted_value(node: ast.AST | None) -> bool:
    """False for a Call node with more than one positional argument to
    an arbitrary function -- e.g. `workload_fn(Path(args.rom),
    Path(args.tape), args.frames)`, where a tracked value (`args.tape`,
    genuinely a runs/-rooted read path) is just one of several unrelated
    arguments and the call's return value (a benchmark-result dict, not
    a path) should NOT become tracked merely because one argument among
    several was.

    True for everything else, including a Call with 0-1 positional args
    (`Path(x)`, `str(x)`, `_default_out_dir(x, selfcheck=True)` --
    keyword args don't count against this), since those are the
    single-value wrapper/join idioms the real missed writers
    (soak_harness.py, critic_explained_variance.py) actually use.
    Non-Call nodes (BinOp path joins, bare Name aliases, Attribute
    access) are always unrestricted -- that's the existing, already
    spot-checked `show_dir / "y"`-shaped propagation.
    """
    if isinstance(node, ast.Call) and len(node.args) > 1:
        return False
    return True


def _tracked_path_vars(tree: ast.AST) -> set[str]:
    """Names (bare local variables, function parameters, and `args.<dest>`
    attribute accesses) whose value is, or is built from, a runs/ or
    checkpoints/ path literal. Whole-file, scope-blind (a script rebinding
    the same name in an unrelated function, or two different argparse
    parsers both using `--out-dir`, could produce a false positive) --
    acceptable for a candidate-flagging census, not a hard-gate
    correctness proof.
    """
    candidates = _tracked_path_candidates(tree)
    tracked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, src, node in candidates:
            if name in tracked:
                continue
            if _PATH_LITERAL_RE.search(src):
                tracked.add(name)
                changed = True
                continue
            if not _is_unrestricted_value(node):
                continue
            if any(re.search(rf"\b{re.escape(tv)}\b", src) for tv in tracked):
                tracked.add(name)
                changed = True
    return tracked


def _mentions_target(src: str, tracked: set[str]) -> bool:
    if _PATH_LITERAL_RE.search(src):
        return True
    return any(re.search(rf"\b{re.escape(tv)}\b", src) for tv in tracked)


def _write_hit_lines(tree: ast.AST, tracked: set[str]) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _WRITE_METHODS:
            if _mentions_target(_unparse(f.value), tracked):
                lines.append(node.lineno)
        elif isinstance(f, ast.Name) and f.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and any(c in mode for c in "wax") and node.args:
                if _mentions_target(_unparse(node.args[0]), tracked):
                    lines.append(node.lineno)
        elif (isinstance(f, ast.Attribute) and f.attr == "save"
              and isinstance(f.value, ast.Name) and f.value.id in ("torch", "np")):
            idx = 1 if f.value.id == "torch" else 0
            if len(node.args) > idx and _mentions_target(_unparse(node.args[idx]), tracked):
                lines.append(node.lineno)
        elif (isinstance(f, ast.Attribute) and f.attr == "makedirs"
              and isinstance(f.value, ast.Name) and f.value.id == "os"):
            if node.args and _mentions_target(_unparse(node.args[0]), tracked):
                lines.append(node.lineno)
    return lines


def census_lockless_writers(repo: Path) -> dict[str, list[int]]:
    """{relpath: [line numbers]} for scripts under scripts/ that write
    (write_text/write_bytes/mkdir/open(...'w'.../torch.save/np.save/
    os.makedirs) into a path derived from a runs/ or checkpoints/
    literal, and never import src.utils.run_lock.
    """
    hits: dict[str, list[int]] = {}
    for f in sorted((repo / "scripts").glob("*.py")):
        text = f.read_text(errors="replace")
        if _RUN_LOCK_IMPORT_RE.search(text):
            continue
        try:
            tree = ast.parse(text, filename=str(f))
        except SyntaxError:
            continue
        tracked = _tracked_path_vars(tree)
        lines = _write_hit_lines(tree, tracked)
        if lines:
            hits[str(f.relative_to(repo))] = lines
    return hits


# ---------------------------------------------------------------------------
# (c) solve: key census against the proposed KNOWN_SOLVE_KEYS registry
# ---------------------------------------------------------------------------

# Mirrors config-and-fidelity.md's Ready-to-apply #1 (DO-30, not yet
# landed in config_schema.py as of this census). Kept here as a literal
# so this sub-check runs before DO-30 ships and regression-checks DO-30's
# own list once it does.
PROPOSED_KNOWN_SOLVE_KEYS: frozenset[str] = frozenset({
    "rom", "progress", "y", "lives", "level_key", "no_clear_predicate",
    "clear", "area", "state_sig", "player_state", "death_states",
    "progress_cap", "hold_macros", "room_advance", "entity_slots",
    "kill_key_local", "boss_typed", "room_sig", "room_fp", "boss",
    "finale", "transit_source", "area_key", "min_blank_frames",
    "constructible", "reason", "hw_flags", "stasis",
})

# Consumed by go_explore_solve.py's GenericGame but never set in any
# configs/*.yaml today (config-and-fidelity.md's "consumed-but-not-in-
# any-YAML-yet" finding) -- these are expected to show up as
# "registered, 0 files" and are not a regression.
_SOLVE_KEYS_EXPECTED_UNSET = frozenset({"hw_flags", "stasis"})


def census_solve_keys(repo: Path) -> dict:
    """Per-key file counts for every `solve:` block in configs/*.yaml,
    plus the diff against PROPOSED_KNOWN_SOLVE_KEYS: keys observed in a
    YAML file but not in the registry (registry gap -- should be empty),
    and registry keys with zero YAML adoption beyond the two expected
    consumed-but-unset keys (registry drift, not necessarily a problem
    but worth a human look if the expected-unset set changes).
    """
    import yaml

    counts: dict[str, int] = {}
    files_with_solve = 0
    for yml in sorted(repo.glob("configs/*.yaml")):
        try:
            doc = yaml.safe_load(yml.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        solve = doc.get("solve")
        if isinstance(solve, dict):
            files_with_solve += 1
            for k in solve:
                counts[k] = counts.get(k, 0) + 1

    observed_keys = set(counts)
    unregistered = sorted(observed_keys - PROPOSED_KNOWN_SOLVE_KEYS)
    unused = sorted(
        PROPOSED_KNOWN_SOLVE_KEYS - observed_keys - _SOLVE_KEYS_EXPECTED_UNSET
    )
    return {
        "files_with_solve": files_with_solve,
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "unregistered_in_yaml_not_in_registry": unregistered,
        "registered_but_unused_beyond_expected": unused,
    }


def check_profile_call_site_diff() -> str:
    """The proposed wiring for scripts/go_explore_solve.py (not applied
    -- repo is read-only; shown for review). Depends on DO-30 landing
    config_schema.py's `solve` top-key + KNOWN_SOLVE_KEYS registration
    first, or every one of the 45 solve configs will warn on 'solve'
    itself before ever reaching a solve-key typo.
    """
    return '''\
--- a/scripts/go_explore_solve.py
+++ b/scripts/go_explore_solve.py
@@ -3678,3 +3678,5 @@
         atexit.register(lambda: _lock.exists() and _lock.unlink())
         profile = yaml.safe_load(Path(args.profile).read_text())
+        from src.training.config_schema import check_profile
+        check_profile(profile, strict=False)
         # CLEAR-REACHABILITY PRE-FLIGHT (2026-08-26). Runs before the pool
'''


# ---------------------------------------------------------------------------
# (d) glob-based checkpoint/run readers that should call is_quarantined()
# ---------------------------------------------------------------------------

_QUARANTINE_AWARE_RE = re.compile(
    r"is_quarantined\(|QUARANTINE\b|quarantine", re.IGNORECASE
)
_GLOB_METHODS = {"glob", "rglob", "iglob"}


def _glob_hit_lines(tree: ast.AST, tracked: set[str]) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _GLOB_METHODS:
            # Path(...).glob(pattern) -- either the receiver or the
            # pattern argument can carry the runs/checkpoints literal
            # (e.g. `REPO.glob("checkpoints/**/*.pt")` vs
            # `(REPO / "checkpoints").glob("**/*.pt")`).
            recv = _unparse(f.value)
            pat = _unparse(node.args[0]) if node.args else ""
            if _mentions_target(recv, tracked) or _mentions_target(pat, tracked):
                lines.append(node.lineno)
        elif (isinstance(f, ast.Attribute) and f.attr == "glob"
              and isinstance(f.value, ast.Name) and f.value.id == "glob"):
            if node.args and _mentions_target(_unparse(node.args[0]), tracked):
                lines.append(node.lineno)
    return lines


def census_unquarantined_globs(repo: Path) -> dict[str, list[int]]:
    """{relpath: [line numbers]} for scripts/*.py that glob over a path
    derived from runs/ or checkpoints/ and contain no quarantine-aware
    filtering anywhere in the file (no is_quarantined call, no textual
    mention of QUARANTINE/quarantine at all -- the loosest possible bar,
    so a hit is a script that has never once considered the question,
    not one whose filter this heuristic merely failed to recognize).
    AST-scoped to the actual glob call's receiver/pattern for the same
    precision reason as census_lockless_writers (b): a script that globs
    some unrelated directory while separately mentioning "checkpoints/"
    in a docstring should not be flagged.
    """
    hits: dict[str, list[int]] = {}
    for f in sorted((repo / "scripts").glob("*.py")):
        text = f.read_text(errors="replace")
        if _QUARANTINE_AWARE_RE.search(text):
            continue
        try:
            tree = ast.parse(text, filename=str(f))
        except SyntaxError:
            continue
        tracked = _tracked_path_vars(tree)
        lines = _glob_hit_lines(tree, tracked)
        if lines:
            hits[str(f.relative_to(repo))] = lines
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config_schema(repo: Path):
    sys.path.insert(0, str(repo))
    import importlib
    return importlib.import_module("src.training.config_schema")


def run_all(repo: Path) -> dict:
    config_schema = _load_config_schema(repo)
    return {
        "inert_by_nonadoption": census_inert_keys(repo, config_schema),
        "lockless_writers": census_lockless_writers(repo),
        "solve_keys": census_solve_keys(repo),
        "unquarantined_globs": census_unquarantined_globs(repo),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = run_all(args.repo)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("== (a) INERT-BY-NONADOPTION ==")
        if not result["inert_by_nonadoption"]:
            print("  none")
        for registry, keys in result["inert_by_nonadoption"].items():
            print(f"  {registry}: {len(keys)} inert")
            for k in keys:
                print(f"    - {k}")

        print("\n== (b) writers under runs/ or checkpoints/ with no run_lock import ==")
        if not result["lockless_writers"]:
            print("  none")
        for f, lines in result["lockless_writers"].items():
            print(f"  - {f}:{','.join(str(l) for l in lines)}")

        sk = result["solve_keys"]
        print(f"\n== (c) solve: key census ({sk['files_with_solve']} files with a solve: block) ==")
        for k, n in sk["counts"].items():
            flag = "" if k in PROPOSED_KNOWN_SOLVE_KEYS else "  <-- NOT IN PROPOSED REGISTRY"
            print(f"  {k}: {n}{flag}")
        if sk["unregistered_in_yaml_not_in_registry"]:
            print("  registry gap:", sk["unregistered_in_yaml_not_in_registry"])
        if sk["registered_but_unused_beyond_expected"]:
            print("  unexpected zero-adoption:", sk["registered_but_unused_beyond_expected"])

        print("\n== (d) checkpoint/run globbers with no quarantine awareness ==")
        if not result["unquarantined_globs"]:
            print("  none")
        for f, lines in result["unquarantined_globs"].items():
            print(f"  - {f}:{','.join(str(l) for l in lines)}")

    violation = bool(
        result["inert_by_nonadoption"]
        or result["lockless_writers"]
        or result["solve_keys"]["unregistered_in_yaml_not_in_registry"]
        or result["unquarantined_globs"]
    )
    return 1 if violation else 0


if __name__ == "__main__":
    raise SystemExit(main())
