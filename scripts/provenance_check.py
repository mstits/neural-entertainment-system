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

    if SEEDS.exists():
        for f in sorted(SEEDS.glob("*.npz")):
            rel = str(f.relative_to(REPO))
            if rel not in allowed:
                errors.append(f"demo not on allowlist (add or quarantine): {rel}")

    for name in QUARANTINED_NAMES:
        if not (QUARANTINE / name).exists():
            errors.append(f"quarantined artifact missing from quarantine: {name}")

    yaml_roots = [REPO / "configs", REPO / "checkpoints"]
    for root in yaml_roots:
        if not root.exists():
            continue
        for y in root.rglob("*.yaml"):
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
