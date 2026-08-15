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

The allowlist is authoritative; provenance sidecars are advisory (a
sidecar mislabel has already happened once).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ALLOWLIST = REPO / "configs/demo_allowlist.txt"
SEEDS = REPO / "checkpoints/harvested_seeds"
QUARANTINE = REPO / "checkpoints/QUARANTINE_tier3"
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

    if errors:
        print("PROVENANCE CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"provenance check OK: {len(allowed)} allowlisted demos, "
          f"{len(QUARANTINED_NAMES)} artifacts confirmed quarantined, "
          f"{len(demo_anchor_hashes)} demo_anchor_paths hashed, "
          f"{soak_verified} soak trail(s) chain-verified "
          f"({soak_scoreable} scoreable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
