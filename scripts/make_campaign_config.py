"""Generate a level's campaign config pair by DERIVATION, not by cloning.

Every level so far was onboarded by copying the previous level's configs
and hand-editing the paths. That has now failed twice, both times
silently and both times expensively:

  * `configs/mario_2_1_online_v1.yaml` kept 1-3's `states_dir`, so 20.1M
    steps of backward-curriculum training restarted into the wrong level
    while every instrument read plausibly;
  * `configs/mario_1_4_online_v1.yaml` had the same defect, on the level
    with the highest banked honest rate, and it survived that level's
    entire campaign undetected.

Hand-editing a clone is the hazard. So here no path is ever carried
over: every level-specific value is COMPUTED from the level tag, and
`residual_references` then proves the output contains no mention of the
template's level at all. A generator that derives cannot reproduce the
bug that copying kept reproducing.

The competence floor ships DISARMED (0.0) exactly as the proven runbook
requires, because it must be calibrated from this level's own BC anchor
(runbook step W6, floor = 0.8x the anchor's honest median). Shipping an
inherited floor would be a threshold tuned against another level's
difficulty — the class of mistake that cost the 1-2 campaign attempt 3.

Usage:
    .venv/bin/python scripts/make_campaign_config.py --level 3-1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE_LEVEL = "2-1"          # the most recently corrected pair


def tag(level: str) -> str:
    return level.replace("-", "_")


def validate_level(level: str) -> None:
    if not re.fullmatch(r"[1-8]-[1-4]", level):
        raise ValueError(f"level must look like W-L, got {level!r}")


def derived_paths(level: str) -> dict[str, str]:
    """Every level-specific value, computed from the tag alone.

    This mapping is the whole point: `states_dir` cannot disagree with
    `restart_states_dir` because both are generated from `t`, and neither
    can retain a template value because neither is copied.
    """
    t = tag(level)
    return {
        "name": f"Mario {level} online v1",
        "start_state_path": f"checkpoints/ge_entrances/smb_{t}_entrance.state",
        "dmap": f"checkpoints/wavefront/mario_{t}_dmap.pkl",
        "kl_anchor_checkpoint":
            f"checkpoints/bc_{t}/anchor_h256/vanilla_ppo_iter_00000.pt",
        "states_dir": f"checkpoints/online_{t}/restart_states",
        "base_profile": f"configs/mario_{t}_online_v1.yaml",
        "run_dir": f"runs/online_{t}",
        "campaign_log": f"runs/online_{t}/campaign.jsonl",
        "restart_states_dir": f"checkpoints/online_{t}/restart_states",
        "campaign_name": f"smb_{t}_online_v1",
        "probe_game": f"mario_{t}_online_v1",
        "campaign_level": level,
    }


def substitute(text: str, level: str, template: str = TEMPLATE_LEVEL) -> str:
    """Rewrite every occurrence of the template level into `level`.

    Both spellings are rewritten — `2-1` as it appears in prose and
    `campaign_level`, and `2_1` as it appears in every path — because
    missing either is precisely how a residual reference survives.
    """
    out = text.replace(tag(template), tag(level))
    return out.replace(template, level)


def residual_references(text: str, level: str,
                        template: str = TEMPLATE_LEVEL) -> list[str]:
    """Lines still mentioning the template level. Non-empty means STOP.

    This is the mechanical form of the check that was missing when 2-1
    and 1-4 shipped with 1-3's ladder. It is deliberately dumb: any line
    naming the template level at all is reported, comments included,
    because a stale comment is how the 2-1 profile instructed a reader to
    take its rung budget from the 1-3 manifest.
    """
    if level == template:
        return []
    pats = (template, tag(template))
    return [f"{i}: {ln.strip()}"
            for i, ln in enumerate(text.splitlines(), 1)
            if any(p in ln for p in pats)]


def disarm_competence_floor(text: str) -> str:
    """Force the floor to 0.0 with the reason recorded inline."""
    return re.sub(
        r"kill_probe_median_floor:\s*[0-9.]+.*",
        "kill_probe_median_floor: 0.0   # DISARMED until runbook W6 "
        "calibrates it from THIS level's BC anchor (0.8x its honest "
        "median). An inherited floor is a threshold tuned against another "
        "level's difficulty.",
        text)


def generate(level: str, repo: Path = REPO,
             template: str = TEMPLATE_LEVEL) -> dict[str, str]:
    """Return {path: content} for the pair. Writes nothing."""
    validate_level(level)
    if level == template:
        raise ValueError(f"{level} is the template; nothing to generate")
    t, tt = tag(level), tag(template)
    out: dict[str, str] = {}
    for src, dst in ((f"configs/mario_{tt}_online_v1.yaml",
                      f"configs/mario_{t}_online_v1.yaml"),
                     (f"configs/campaign_{tt}.yaml",
                      f"configs/campaign_{t}.yaml")):
        text = (repo / src).read_text()
        text = substitute(text, level, template)
        if dst.startswith("configs/campaign_"):
            text = disarm_competence_floor(text)
        leftovers = residual_references(text, level, template)
        if leftovers:
            raise RuntimeError(
                f"{dst}: {len(leftovers)} residual reference(s) to "
                f"{template} survived derivation — refusing to write:\n  "
                + "\n  ".join(leftovers[:8]))
        out[dst] = text
    return out


def missing_prerequisites(level: str, repo: Path = REPO) -> list[str]:
    """Artifacts the runbook's W1-W6 must produce before a launch.

    Reported rather than created: generating a config is cheap and safe,
    while minting a ladder or training an anchor is neither, so this
    script writes the config and names what is still owed.
    """
    d = derived_paths(level)
    owed = []
    for key in ("start_state_path", "dmap", "kl_anchor_checkpoint"):
        if not (repo / d[key]).exists():
            owed.append(f"{key}: {d[key]}")
    if not (repo / d["states_dir"] / "index.json").exists():
        owed.append(f"restart ladder: {d['states_dir']}/index.json")
    if not any((repo / "runs").glob(f"ge_{tag(level)}_solve/solutions/sol_*.json")):
        owed.append(f"solver tapes: runs/ge_{tag(level)}_solve/solutions/")
    return owed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", required=True)
    ap.add_argument("--template", default=TEMPLATE_LEVEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        files = generate(args.level, REPO, args.template)
    except (ValueError, RuntimeError) as e:
        sys.stderr.write(f"{e}\n")
        return 2

    for path, text in files.items():
        p = REPO / path
        if args.dry_run:
            print(f"[dry-run] would write {path} ({len(text)} bytes)")
            continue
        if p.exists():
            print(f"refusing to overwrite existing {path}")
            return 3
        p.write_text(text)
        print(f"wrote {path}")

    owed = missing_prerequisites(args.level)
    if owed:
        print(f"\nstill owed before this config can launch "
              f"({len(owed)}), per runbook W1-W6:")
        for o in owed:
            print(f"  - {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
