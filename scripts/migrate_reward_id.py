#!/usr/bin/env python3
"""Freeze and migrate reward dispatch from display-name substrings to an
explicit ``reward_id`` profile key.

Two jobs, both idempotent:

``--freeze``
    Replay the *historical* substring predicate list (the exact set of
    ``name.contains(...)`` tests that lived in ``nes_core/src/rewards.rs``
    before this change) over every ``configs/**/*.yaml`` and write the
    resulting ``{path: arm}`` map to ``tests/reward_dispatch_baseline.json``.
    That file is the migration's ground truth: it is computed once,
    checked in, and never regenerated after the Rust dispatch changes —
    regenerating it against the new code would make the roster lint
    tautological.

``--apply``
    Write ``reward_id: <arm>`` into every config the frozen baseline
    names, inserting the line immediately after the ``name:`` key so
    YAML comments survive (a safe_load/safe_dump round-trip would strip
    every comment in files like configs/zelda_gui_tuned.yaml).

Three profiles are hand-audited overrides rather than mechanical rows —
see ``HAND_AUDITED`` below.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
BASELINE = REPO / "tests" / "reward_dispatch_baseline.json"


def legacy_arm(name: str) -> str:
    """The arm ``build_reward`` resolved to on the pre-change tree.

    A verbatim transcription of the 16 ``name.contains(...)`` branches,
    in source order, including the Double Dragon exclusion guard. This
    function is frozen history — it must never be "fixed" to match new
    behaviour.
    """
    n = name.lower()
    if "mario" in n:
        return "mario"
    if "zelda" in n:
        return "zelda"
    if "contra" in n:
        return "contra"
    if "mega man" in n or "megaman" in n:
        return "mega_man"
    if "castlevania" in n:
        return "castlevania"
    if "metroid" in n:
        return "metroid"
    if "tetris" in n:
        return "tetris"
    if "bubble bobble" in n or "bubblebobble" in n:
        return "bubble_bobble"
    if "punch-out" in n or "punch out" in n or "punchout" in n:
        return "punch_out"
    if "kung fu" in n:
        return "kung_fu"
    if "gradius" in n:
        return "gradius"
    if "excitebike" in n or "excite bike" in n:
        return "excitebike"
    if "ghosts" in n:
        return "ghosts"
    if "ducktales" in n:
        return "ducktales"
    if "kid icarus" in n:
        return "kid_icarus"
    if "double dragon" in n and "battletoads" not in n and " ii" not in n:
        return "double_dragon"
    return "generic"


# Profiles whose declaration is a judgement call, not a transcription.
# Each maps to (arm, inline comment written above the key).
HAND_AUDITED: dict[str, tuple[str, str]] = {
    # BREACH FIX. Declares no reward weights and describes itself as
    # "Not a training profile", yet its display name handed it the
    # quarantined Zelda win predicate. Generic is what it always meant.
    "configs/legend_of_zelda.yaml": (
        "generic",
        "reward_id: explicit dispatch. This profile declares no reward weights;\n"
        "# generic is the axis-free arm with no win predicate. It previously\n"
        "# inherited ZeldaReward (and its quarantined 0x0672 predicate) purely\n"
        "# because its display name contains \"Zelda\".",
    ),
    # Behaviour-preserving: these two run GenericReward today because
    # their display name has no "mario" substring. Pinning that, not
    # blessing it.
    "configs/smb_4_4_micro.yaml": (
        "generic",
        "reward_id: explicit dispatch — preserves today's behaviour.\n"
        "# TODO: this profile declares nine MarioReward-only weights\n"
        "# (forward_progress, air_bonus, jump_clear_bonus, checkpoint_scale, ...)\n"
        "# which GenericReward ignores, and its level_id comes back \"stage_1\"\n"
        "# not \"1-1\". Switching to reward_id: mario is a real behaviour change\n"
        "# on a profile with banked live-show receipts — its own change.",
    ),
    "configs/smb_blank_slate.yaml": (
        "generic",
        "reward_id: explicit dispatch — preserves today's behaviour.\n"
        "# TODO: same defect as configs/smb_4_4_micro.yaml — declares\n"
        "# MarioReward-only weights that GenericReward ignores. Flipping to\n"
        "# reward_id: mario is a separate, measured change.",
    ),
}

# Arms inherited from a reward authored for a DIFFERENT game. Migrating
# mechanically would launder an unverified inheritance into a
# deliberate-looking config line, so each gets a comment saying so.
UNAUDITED_INHERITANCE: dict[str, str] = {
    "configs/castlevania_iii.yaml": "UNAUDITED INHERITANCE: this is CastlevaniaReward, authored against "
    "CV1's $0044/$002A/$0028 and its Dracula stage-0x12 win predicate. Never "
    "verified on CV3. Declared explicitly to preserve today's behaviour, not "
    "because it is known correct.",
    "configs/onboard/castlevania_iii_-_dracula_s_curse__usa_.yaml": "UNAUDITED INHERITANCE: CV1-authored reward applied to CV3. Preserves "
    "today's behaviour; not verified on this ROM.",
    "configs/mega_man_3.yaml": "UNAUDITED INHERITANCE: MegaManReward was authored against Mega Man 1. "
    "Never verified on MM3. Preserves today's behaviour.",
    "configs/ducktales_2.yaml": "UNAUDITED INHERITANCE: DuckTalesReward was authored against DuckTales 1. "
    "Never verified on DuckTales 2. Preserves today's behaviour.",
    "configs/auto/bubble_bobble_part_2__usa_.yaml": "UNAUDITED INHERITANCE: BubbleBobbleReward was authored against Bubble "
    "Bobble 1, a different game. Preserves today's behaviour.",
}


def iter_configs():
    for path in sorted(CONFIGS.rglob("*.yaml")):
        yield path


def load_name(path: Path) -> str | None:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    name = doc.get("name")
    return name if isinstance(name, str) else None


def freeze() -> dict:
    rows = {}
    for path in iter_configs():
        name = load_name(path)
        if name is None:
            continue
        arm = legacy_arm(name)
        if arm == "generic":
            continue
        rows[str(path.relative_to(REPO))] = arm
    return rows


def insert_key(path: Path, arm: str, comment: str | None) -> bool:
    """Insert ``reward_id: <arm>`` after the top-level ``name:`` line.

    Returns True if the file changed. Idempotent: a file that already
    declares reward_id at top level is left alone (but its value is
    checked).
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if re.match(r"^reward_id\s*:", line):
            have = line.split(":", 1)[1].strip().strip("\"'")
            if have != arm:
                raise SystemExit(f"{path}: declares reward_id: {have}, baseline says {arm}")
            return False
    anchor = None
    for i, line in enumerate(lines):
        if re.match(r"^name\s*:", line):
            anchor = i
            break
    if anchor is None:
        raise SystemExit(f"{path}: no top-level 'name:' line to anchor on")
    block = ""
    if comment:
        import textwrap

        for para in comment.split("\n"):
            if para.startswith("#"):
                block += para + "\n"
                continue
            for cline in textwrap.wrap(para, width=74) or [""]:
                block += "# " + cline + "\n"
    block += f"reward_id: {arm}\n"
    lines.insert(anchor + 1, block)
    path.write_text("".join(lines))
    return True


def apply() -> None:
    baseline = json.loads(BASELINE.read_text())
    plan: dict[str, tuple[str, str | None]] = {}
    for rel, arm in baseline.items():
        plan[rel] = (arm, UNAUDITED_INHERITANCE.get(rel))
    for rel, (arm, comment) in HAND_AUDITED.items():
        plan[rel] = (arm, comment)
    changed = 0
    for rel, (arm, comment) in sorted(plan.items()):
        if insert_key(REPO / rel, arm, comment):
            changed += 1
    print(f"migrated {changed} of {len(plan)} profiles")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        rows = freeze()
        BASELINE.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"froze {len(rows)} rows -> {BASELINE.relative_to(REPO)}")
    if args.apply:
        apply()
    if not args.freeze and not args.apply:
        ap.error("pass --freeze and/or --apply")


if __name__ == "__main__":
    main()
