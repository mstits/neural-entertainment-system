"""Provenance gate for Learned-ledger training inputs (see CLAIMS.md).

Checks, failing loud on any violation:
  1. Every path in configs/demo_allowlist.txt exists.
  2. Every demo .npz under checkpoints/harvested_seeds/ is either on
     the allowlist or explicitly quarantined — no unaccounted demos.
  3. The quarantine directory still holds the Tier-3 artifacts named in
     CLAIMS.md (nothing quietly restored).
  4. No profile/manifest yaml under configs/ or checkpoints/ references
     a quarantined artifact.

The allowlist is authoritative; provenance sidecars are advisory (a
sidecar mislabel has already happened once).
"""
from __future__ import annotations

import sys
from pathlib import Path

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

    # Content-hash every quarantined file, then confirm NO copy of it
    # exists anywhere under checkpoints/ outside the quarantine — a
    # restore-by-copy (not just moving the original back) must be caught.
    import hashlib

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    quarantined_hashes = {}
    for name in QUARANTINED_NAMES:
        qp = QUARANTINE / name
        if not qp.exists():
            errors.append(f"quarantined artifact missing from quarantine: {name}")
        else:
            try:
                quarantined_hashes[_sha(qp)] = name
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
                if (h := _sha(f)) in quarantined_hashes:
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

    if errors:
        print("PROVENANCE CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"provenance check OK: {len(allowed)} allowlisted demos, "
          f"{len(QUARANTINED_NAMES)} artifacts confirmed quarantined")
    return 0


if __name__ == "__main__":
    sys.exit(main())
