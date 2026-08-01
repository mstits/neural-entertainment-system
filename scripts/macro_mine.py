"""Automated macro-action mining from our own successful solver trajectories.

Mines frequent variable-length action n-grams out of two corpora built
purely from this repo's own solver archives:

  (a) SMB-family chain solutions — every `sol_*.actions.npy` under any run
      tree's `solutions/` directory. Go-Explore chain solves for SMB and
      Castlevania both route through the same solver (`go_explore_solve.py`)
      and directory layout, so both land in this one corpus; each file's
      raw action indices are decoded through whichever profile actually
      produced it (see `resolve_smb_profile` below), not a single assumed
      profile, since SMB and Castlevania assign different buttons to the
      same integer.
  (b) Contra deep-exploration traces — the highest-progress cells (by
      `key[-1]`) out of a Go-Explore `traces.pkl` archive.

For both corpora, raw action indices are decoded to canonical button-name
tokens (e.g. "right+A", "noop") *before* n-grams are mined, so frequency
counts are comparable across files that happen to use different profiles —
mining on raw integers would silently conflate "index 2" meaning `right+A`
in one profile and `left` in another.

Frequent n-grams (n = 4..24) are counted with non-overlapping occurrence
counting per trace: for a specific candidate window, a trace is scanned
left to right and a match consumes (skips past) its own span, so a long
held button isn't inflated by counting every overlapping sub-window of
itself. Score = frequency * (n - 1), so a longer recurring sequence
outranks a shorter equally-common one. Single-action repeats (a plain
button held for the n-gram's length) are dropped unless that run is longer
than 8 steps — those are exactly the hold-macro class the hand-written
`hold_macros` config entries already exploit (see configs/contra.yaml,
configs/castlevania.yaml), so they're kept on purpose instead of being
treated as boring noise.

This script only reads existing run archives and writes one receipt JSON.
It never edits any config and never launches a Pool.

Usage:
    python scripts/macro_mine.py
    python scripts/macro_mine.py --smb-cap 100 --top-k 10
    python scripts/macro_mine.py --contra-traces runs/other_run/traces.pkl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# A solver chain writes here live (10 pool workers) — never read or touch
# this directory's contents from a side script.
HARD_EXCLUDE_SUBSTRINGS = ("cv_chain_hw2",)


# ---------------------------------------------------------------------------
# Action-space decoding
# ---------------------------------------------------------------------------

def canon_token(buttons: list) -> str:
    """A profile action_space entry (list of button names) -> one string."""
    if not buttons:
        return "noop"
    return "+".join(sorted(buttons))


_profile_token_cache: dict = {}


def load_action_tokens(profile_path: Path) -> list:
    key = str(profile_path)
    if key in _profile_token_cache:
        return _profile_token_cache[key]
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    action_space = profile["action_space"]
    tokens = [canon_token(list(b) if b else []) for b in action_space]
    _profile_token_cache[key] = tokens
    return tokens


def decode(indices: Iterable, tokens: list, fallback_tokens: Optional[list],
           warnings: list, context: str) -> list:
    out = []
    for raw in indices:
        idx = int(raw)
        if 0 <= idx < len(tokens):
            out.append(tokens[idx])
        elif fallback_tokens is not None and 0 <= idx < len(fallback_tokens):
            out.append(fallback_tokens[idx])
            warnings.append(
                f"{context}: action idx {idx} outside resolved profile "
                f"({len(tokens)} actions) — used fallback superset")
        else:
            out.append(f"UNMAPPED_{idx}")
            warnings.append(
                f"{context}: action idx {idx} unmapped by any known profile")
    return out


# ---------------------------------------------------------------------------
# SMB-family corpus
# ---------------------------------------------------------------------------

def resolve_smb_files(repo_root: Path, glob_pattern: str, cap: int,
                       prefer_substrings: tuple) -> list:
    raw = glob.glob(str(repo_root / glob_pattern), recursive=True)
    matches = [Path(p) for p in raw
               if not any(s in p for s in HARD_EXCLUDE_SUBSTRINGS)]
    matches.sort(key=lambda p: (
        0 if any(s.lower() in str(p).lower() for s in prefer_substrings) else 1,
        str(p),
    ))
    n_excluded_running = len(raw) - len(matches)
    return matches[:cap], len(matches), n_excluded_running


def resolve_smb_profile(npy_path: Path, repo_root: Path, default_tokens: list,
                         cv_tokens: list, warnings: list) -> tuple:
    """Return (tokens, label) for one sol_*.actions.npy file.

    Priority: (1) the sidecar sol_*.json's solver_args.profile, the most
    authoritative source since it's literally the CLI arg the solve was run
    with; (2) Castlevania inference from root_state / the profile string /
    the file's own path (cv_chain_a's early runs predate solver_args being
    recorded, so path substrings are the only signal left for those); (3)
    the SMB default (empirically the superset of every other SMB profile's
    action_space prefix — see the module docstring in the receipt).
    """
    json_path = Path(str(npy_path).replace(".actions.npy", ".json"))
    meta = {}
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
        except Exception as e:
            warnings.append(f"{npy_path}: sidecar json unreadable ({e})")
    else:
        warnings.append(f"{npy_path}: no sidecar sol_*.json found")

    solver_args = meta.get("solver_args") or {}
    prof = solver_args.get("profile")
    if prof:
        candidate = Path(prof) if os.path.isabs(prof) else (repo_root / prof)
        if candidate.exists():
            return load_action_tokens(candidate), str(candidate)
        warnings.append(f"{npy_path}: solver_args.profile {prof!r} not found on disk")

    root_state = str(meta.get("root_state", "")).lower()
    prof_str = str(prof or "").lower()
    is_castlevania = (
        "castlevania" in root_state
        or "castlevania" in prof_str
        or "cv_chain" in str(npy_path).lower()
        or "cv_smoke" in str(npy_path).lower()
    )
    if is_castlevania:
        return cv_tokens, "castlevania (inferred, no usable solver_args.profile)"

    return default_tokens, "smb-default (superset)"


def load_smb_corpus(repo_root: Path, glob_pattern: str, cap: int,
                     prefer_substrings: tuple, default_profile: Path,
                     cv_profile: Path, warnings: list) -> tuple:
    default_tokens = load_action_tokens(default_profile)
    cv_tokens = load_action_tokens(cv_profile)

    files, n_after_glob_filter, n_hard_excluded = resolve_smb_files(
        repo_root, glob_pattern, cap, prefer_substrings)

    traces = []
    profile_usage = Counter()
    for f in files:
        arr = np.load(f)
        tokens, label = resolve_smb_profile(f, repo_root, default_tokens,
                                             cv_tokens, warnings)
        profile_usage[label] += 1
        decoded = decode(arr.tolist(), tokens, None, warnings, str(f))
        trace_id = str(f.relative_to(repo_root))
        traces.append((trace_id, decoded))

    meta = {
        "glob_pattern": glob_pattern,
        "hard_excluded_running_dir": n_hard_excluded,
        "n_files_after_filter": n_after_glob_filter,
        "cap": cap,
        "n_files_used": len(traces),
        "profile_usage": dict(profile_usage),
        "total_tokens": sum(len(t) for _, t in traces),
    }
    return traces, meta


# ---------------------------------------------------------------------------
# Contra corpus
# ---------------------------------------------------------------------------

def load_contra_corpus(traces_path: Path, top_n: int, profile_path: Path,
                        warnings: list) -> tuple:
    t0 = time.time()
    with open(traces_path, "rb") as f:
        raw_traces = pickle.load(f)
    load_s = time.time() - t0

    items = []
    n_no_actions = 0
    for key, value in raw_traces.items():
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            continue
        actions = value[1]
        if not actions:
            n_no_actions += 1
            continue
        items.append((key, list(actions)))

    items.sort(key=lambda kv: (kv[0][-1], len(kv[1])), reverse=True)
    top = items[:top_n]

    max_progress = top[0][0][-1] if top else None
    n_tied_at_max = sum(1 for k, _ in items if k[-1] == max_progress) if top else 0

    tokens = load_action_tokens(profile_path)
    traces = []
    for i, (key, actions) in enumerate(top):
        decoded = decode(actions, tokens, None, warnings,
                          f"contra cell#{i} key={key}")
        traces.append((f"cell_{i:04d}", decoded))

    meta = {
        "traces_path": str(traces_path),
        "profile": str(profile_path),
        "load_seconds": round(load_s, 3),
        "n_total_cells_in_archive": len(raw_traces),
        "n_cells_with_actions": len(items),
        "n_cells_empty_actions": n_no_actions,
        "top_n_requested": top_n,
        "top_n_selected": len(top),
        "selection_key": "key[-1] (progress), ties broken by len(actions) desc",
        "max_progress_key_last": max_progress,
        "min_progress_in_selection": top[-1][0][-1] if top else None,
        "n_cells_tied_at_max_progress": n_tied_at_max,
        "total_tokens": sum(len(t) for _, t in traces),
    }
    return traces, meta


# ---------------------------------------------------------------------------
# N-gram mining (shared)
# ---------------------------------------------------------------------------

def mine_ngrams(traces: list, n_min: int, n_max: int) -> tuple:
    """Non-overlapping-counted variable-length n-grams over decoded traces.

    traces: list of (trace_id, token_list).
    Returns (freq, coverage): freq[(n, window)] = total non-overlapping
    occurrence count across the whole corpus; coverage[(n, window)] = set
    of trace_ids containing at least one occurrence.
    """
    freq = Counter()
    coverage = defaultdict(set)
    for trace_id, tokens in traces:
        L = len(tokens)
        if L < n_min:
            continue
        hi = min(n_max, L)
        for n in range(n_min, hi + 1):
            local = defaultdict(list)
            for pos in range(0, L - n + 1):
                local[tuple(tokens[pos:pos + n])].append(pos)
            for window, positions in local.items():
                count = 0
                last_end = -1
                for pos in positions:
                    if pos >= last_end:
                        count += 1
                        last_end = pos + n
                if count:
                    freq[(n, window)] += count
                    coverage[(n, window)].add(trace_id)
    return freq, coverage


def build_candidates(freq: Counter, coverage: dict, n_traces_total: int,
                      hold_min_run: int) -> list:
    candidates = []
    for (n, window), f in freq.items():
        is_repeat = len(set(window)) == 1
        if is_repeat and n <= hold_min_run:
            continue
        score = f * (n - 1)
        cov = coverage[(n, window)]
        candidates.append({
            "n": n,
            "tokens": list(window),
            "freq": f,
            "score": score,
            "is_hold_macro": is_repeat,
            "coverage_traces": len(cov),
            "coverage_fraction": round(len(cov) / n_traces_total, 4) if n_traces_total else 0.0,
        })
    candidates.sort(key=lambda c: (c["score"], c["freq"], c["n"]), reverse=True)
    return candidates


def human_repr(c: dict) -> str:
    if c["is_hold_macro"]:
        return f"HOLD {c['tokens'][0]} x{c['n']}"
    return ", ".join(c["tokens"])


def find_hold_candidates(all_candidates: list, max_entries: int = 10) -> list:
    """Best-scoring genuine (non-noop) uniform-hold candidate per button combo.

    Pulled from the *full* candidate pool, not just the overall top-K —
    a real hold-macro (e.g. a 9-frame run-jump held for a long carry, or a
    24-frame stair-climb) routinely scores below the flood of short, very
    frequent movement 4-grams on raw score alone, but it is exactly the
    "known-valuable class" the mining brief calls out to keep, so it gets
    its own ranking pass here instead of being crowded out.
    """
    best_by_combo: dict = {}
    for c in all_candidates:
        if not c["is_hold_macro"] or c["tokens"][0] == "noop":
            continue
        combo = c["tokens"][0]
        if combo not in best_by_combo or c["score"] > best_by_combo[combo]["score"]:
            best_by_combo[combo] = c
    ordered = sorted(best_by_combo.values(), key=lambda c: c["score"], reverse=True)[:max_entries]
    for i, c in enumerate(ordered):
        c["rank"] = i + 1
        c["human"] = human_repr(c)
    return ordered


def hold_macro_yaml_snippet(hold_candidates: list) -> str:
    """Ready-to-paste `hold_macros:` list for a config's `solve:` stanza."""
    if not hold_candidates:
        return "# no uniform-hold macros found in this corpus\n"
    lines = ["solve:", "  hold_macros:"]
    for c in hold_candidates:
        buttons = c["tokens"][0].split("+")
        p = round(min(0.05, 0.01 + 0.02 * c["coverage_fraction"]), 3)
        buttons_yaml = ", ".join(f'"{b}"' for b in buttons)
        lines.append(
            f'    - {{buttons: [{buttons_yaml}], steps: {c["n"]}, p: {p}}}'
            f'  # freq={c["freq"]} coverage={c["coverage_fraction"]:.0%}'
        )
    return "\n".join(lines) + "\n"


def sequence_suggestion_text(top_macros: list, max_entries: int = 10) -> str:
    """Informational (not schema-backed) list of the best non-hold macros.

    hold_macros only encodes a single button combo held N steps; a macro
    that *varies* buttons over time (a jump-run-jump sequence, say) has no
    matching config field today, so these are reported for a human to look
    at rather than emitted as paste-ready YAML.
    """
    ordered = [c for c in top_macros if not c["is_hold_macro"]][:max_entries]
    if not ordered:
        return "# no non-hold sequence macros survived the top-K cut\n"
    lines = []
    for c in ordered:
        lines.append(
            f'# n={c["n"]} freq={c["freq"]} score={c["score"]} '
            f'coverage={c["coverage_fraction"]:.0%} :: ' + " -> ".join(c["tokens"])
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def report_corpus(name: str, traces: list, meta: dict, n_min: int, n_max: int,
                   top_k: int, hold_min_run: int) -> dict:
    t0 = time.time()
    freq, coverage = mine_ngrams(traces, n_min, n_max)
    candidates = build_candidates(freq, coverage, len(traces), hold_min_run)
    top = candidates[:top_k]
    for i, c in enumerate(top):
        c["rank"] = i + 1
        c["human"] = human_repr(c)
    hold_candidates = find_hold_candidates(candidates, max_entries=10)
    mining_seconds = round(time.time() - t0, 3)

    return {
        "meta": {**meta, "n_traces": len(traces), "n_candidates_total": len(candidates),
                 "mining_seconds": mining_seconds, "ngram_range": [n_min, n_max]},
        "top_macros": top,
        "hold_macro_candidates": hold_candidates,
        "hold_macros_yaml": hold_macro_yaml_snippet(hold_candidates),
        "sequence_suggestions": sequence_suggestion_text(top),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p.add_argument("--smb-glob", type=str,
                   default="runs/**/solutions/sol_*.actions.npy")
    p.add_argument("--smb-cap", type=int, default=200)
    p.add_argument("--smb-prefer", type=str, default="smb,mario,cv_chain",
                   help="Comma-separated substrings preferred when the cap bites.")
    p.add_argument("--smb-default-profile", type=str,
                   default="configs/smb_4_4_micro.yaml",
                   help="Superset SMB action_space used unless a file's "
                        "sidecar json names a different profile.")
    p.add_argument("--cv-profile", type=str, default="configs/castlevania.yaml")
    p.add_argument("--contra-traces", type=str,
                   default="runs/breadth_contra/stage1_v6_resume/traces.pkl")
    p.add_argument("--contra-profile", type=str, default="configs/contra.yaml")
    p.add_argument("--contra-top-n", type=int, default=200)
    p.add_argument("--ngram-min", type=int, default=4)
    p.add_argument("--ngram-max", type=int, default=24)
    p.add_argument("--hold-min-run", type=int, default=8,
                   help="Single-action repeats of length <= this are dropped "
                        "as noise; longer ones are kept as hold-macros.")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", type=str, default="runs/macro_mine_receipt.json")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    warnings: list = []
    t_start = time.time()

    smb_traces, smb_meta = load_smb_corpus(
        repo_root, args.smb_glob, args.smb_cap,
        tuple(s.strip() for s in args.smb_prefer.split(",") if s.strip()),
        repo_root / args.smb_default_profile, repo_root / args.cv_profile,
        warnings,
    )
    contra_traces, contra_meta = load_contra_corpus(
        repo_root / args.contra_traces, args.contra_top_n,
        repo_root / args.contra_profile, warnings,
    )

    smb_report = report_corpus("smb_family", smb_traces, smb_meta,
                                args.ngram_min, args.ngram_max, args.top_k,
                                args.hold_min_run)
    contra_report = report_corpus("contra", contra_traces, contra_meta,
                                   args.ngram_min, args.ngram_max, args.top_k,
                                   args.hold_min_run)

    receipt = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(repo_root),
        "total_wall_seconds": round(time.time() - t_start, 3),
        "params": vars(args),
        "corpora": {
            "smb_family": smb_report,
            "contra": contra_report,
        },
        "warnings": warnings,
    }

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2))

    print(f"[macro_mine] wrote {out_path}")
    for name, rep in (("smb_family", smb_report), ("contra", contra_report)):
        m = rep["meta"]
        print(f"[macro_mine] {name}: {m['n_traces']} traces, "
              f"{m.get('total_tokens', '?')} tokens, "
              f"{m['n_candidates_total']} candidates, top macro: "
              f"{rep['top_macros'][0]['human'] if rep['top_macros'] else 'none'}")
    if warnings:
        print(f"[macro_mine] {len(warnings)} warnings (see receipt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
