"""Generate `configs/auto/<rom>.yaml` stubs for every ROM in the
`advances` bucket of `playability_sweep_v3.json`.

Each stub carries the metadata that's mechanical to extract — MD5
hash, mapper number, iNES vs NES 2.0 flag, battery flag — plus a
default `generic_exploration` reward profile. Hand-tuned reward
profiles for specific games (mario, zelda, contra, megaman,
castlevania, metroid) live in `configs/` directly and take precedence
when the ROM filename matches one of those titles.

Output is one YAML per ROM in `configs/auto/`. Designed as starting
points for hand-tuning, not as production training profiles.

Usage:
    python scripts/gen_auto_configs.py [--out-dir configs/auto]
                                       [--sweep playability_sweep_v3.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROMS_DIR = REPO / "roms"

# Names that already have hand-crafted reward profiles in configs/.
# We skip generating auto-stubs for these so they don't shadow the
# real config.
HAND_CRAFTED = {"mario", "zelda", "contra", "megaman", "castlevania", "metroid"}


def parse_header(rom_path: Path) -> dict:
    """Return iNES/NES 2.0 metadata for a single ROM."""
    data = rom_path.read_bytes()
    if len(data) < 16:
        return {"error": "header too short"}
    if data[:4] != b"NES\x1a":
        return {"error": "bad magic"}
    flags6 = data[6]
    flags7 = data[7]
    flags8 = data[8]
    flags10 = data[10]
    is_nes20 = (flags7 & 0x0C) == 0x08
    if is_nes20:
        mapper = ((flags8 & 0x0F) << 8) | (flags7 & 0xF0) | (flags6 >> 4)
    else:
        mapper = ((flags7 & 0xF0) | (flags6 >> 4))
    battery = bool(flags6 & 0x02)
    md5 = hashlib.md5(data).hexdigest()
    return {
        "md5": md5,
        "mapper": mapper,
        "is_nes20": is_nes20,
        "battery": battery,
        "prg_size_kb": data[4] * 16,
        "chr_size_kb": data[5] * 8,
    }


def slugify(name: str) -> str:
    """Filename-safe slug derived from ROM name."""
    keep = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    s = name.lower().replace(".nes", "").replace(" ", "_")
    return "".join(c if c in keep else "_" for c in s)


def is_hand_crafted(rom_name: str) -> bool:
    low = rom_name.lower()
    return any(tag in low for tag in HAND_CRAFTED)


def gen_yaml_text(rom_name: str, header: dict) -> str:
    """Return YAML text for one ROM's auto-stub."""
    title_clean = rom_name.replace('"', "'").rstrip(".nes").rstrip(".NES")
    nes20_str = "true" if header["is_nes20"] else "false"
    battery_str = "true" if header["battery"] else "false"
    return f"""# AUTO-GENERATED — playability-sweep-derived stub.
# Hand-tune the reward_weights + ram_mapping for productive training.
# Hand-crafted profiles in configs/ (mario, zelda, contra, megaman,
# castlevania, metroid) take precedence over auto-stubs of the same
# game.
name: "{title_clean}"
rom_filename: "{rom_name}"
rom_hashes:
  - "{header['md5']}"

cartridge:
  mapper: {header['mapper']}
  nes20: {nes20_str}
  battery: {battery_str}
  prg_kb: {header['prg_size_kb']}
  chr_kb: {header['chr_size_kb']}

description: >
  Auto-generated training profile from the playability sweep. Uses the
  GenericReward signals (motion proxy via RAM-byte churn, survival
  bonus, time penalty, and auto-detected score-candidate addresses
  during the warmup window). For best results, replace `reward_weights`
  with game-specific signals derived from the ROM's RAM map.

reward_weights:
  # Generic exploration profile. The Rust GenericReward auto-detects
  # which RAM bytes look like score counters (monotonically increasing
  # during a warmup window) and rewards positive deltas on those.
  motion: 0.05
  survival: 0.01
  time_penalty: -0.001
  score: 1.0
  stuck_steps: 240
  warmup_steps: 600
  activity_timeout_steps: 600

# Default action space — covers all 8 buttons + common combos.
# Trim per-game once you know what controls matter.
action_space:
  - []
  - ["right"]
  - ["left"]
  - ["up"]
  - ["down"]
  - ["A"]
  - ["B"]
  - ["right", "A"]
  - ["right", "B"]
  - ["left", "A"]
  - ["start"]
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep", type=Path, default=REPO / "playability_sweep_v3.json",
        help="playability sweep JSON to derive the ROM list from",
    )
    p.add_argument(
        "--out-dir", type=Path, default=REPO / "configs" / "auto",
        help="destination directory for stub YAML files",
    )
    p.add_argument(
        "--bucket", default="advances",
        help="only generate stubs for ROMs in this playability bucket",
    )
    args = p.parse_args()

    if not args.sweep.exists():
        print(f"sweep file missing: {args.sweep}", file=sys.stderr)
        return 1

    sweep = json.loads(args.sweep.read_text())
    target_roms = [r for r in sweep if r.get("bucket") == args.bucket]
    print(f"found {len(target_roms)} ROMs in bucket={args.bucket}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_handcrafted = 0
    skipped_missing = 0
    errored = 0
    for entry in target_roms:
        rom_name = entry["rom"]
        rom_path = ROMS_DIR / rom_name
        if not rom_path.exists():
            skipped_missing += 1
            continue
        if is_hand_crafted(rom_name):
            skipped_handcrafted += 1
            continue
        try:
            header = parse_header(rom_path)
            if "error" in header:
                errored += 1
                print(f"  skip {rom_name}: header {header['error']}", flush=True)
                continue
        except Exception as exc:
            errored += 1
            print(f"  err  {rom_name}: {exc!s}", flush=True)
            continue

        out = args.out_dir / f"{slugify(rom_name)}.yaml"
        out.write_text(gen_yaml_text(rom_name, header))
        written += 1

    print(
        f"\n  wrote   {written:>4d} stubs to {args.out_dir.relative_to(REPO)}/\n"
        f"  skipped {skipped_handcrafted:>4d} (hand-crafted profile in configs/)\n"
        f"  skipped {skipped_missing:>4d} (ROM file missing)\n"
        f"  errored {errored:>4d} (header parse)\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
