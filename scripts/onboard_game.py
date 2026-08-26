"""Point-at-a-game onboarding: ROM in, draft solve profile out.

One command turns a bare `.nes` file into everything the solver fleet
needs to take a first honest swing at it:

  1. a live-gameplay START STATE (reused if one is already banked beside
     the ROM, otherwise captured with the house boot + START-tap masher);
  2. a DRAFT `solve:` block, drafted by running
     `scripts/discover_observables.py`'s probe protocol — find_progress /
     find_room_counter / find_y / find_hp_lives and the two gates — over
     our own scripted rollouts;
  3. `configs/onboard/<slug>.yaml`, a provenance-stamped, lint-clean
     draft profile carrying that block;
  4. the EXACT bounded calibration-solve command the orchestrator should
     run next (this tool never runs a solve itself), and
  5. `classify_fn` — a post-run hook that reads the resulting run
     directory and returns the wall verdict plus the next-arm
     recommendation.

THE CLASSIFIER IS INJECTED, NOT IMPORTED. The calibrated wall
discriminator lives in `src/training/` behind an explicit hold: nothing
outside tests and docs may import it until the self-arming decision
lands. So this module never reaches for it. `classify_run` takes a
`verdict_fn`, `load_verdict_fn("<module path>")` builds one from a path
the operator names, and the arm table is duck-typed on the verdict's
`.wall_class` / `.reasons` / `.remedy` / `.descriptor`. `tests/test_onboard_game.py` —
which the hold carves out — pins the mirrored constants and the arm
table against the real discriminator, so drift fails in the suite
rather than in a campaign.

Later, once that game has a solved-tape chain, `refine_from_tapes`
re-ranks the progress byte with the learnfun path
(`discover_observables.find_progress_from_tapes`) and writes a refined
sibling profile — never editing the draft in place.

PURITY. Every address that reaches a generated profile comes from our
own differential probing of this ROM. No third-party address lists, no
guides, no reverse-engineered sources. `lint_profile_text` enforces
that mechanically: it rejects a profile carrying an outside-provenance
token, a quarantine block, or an address outside the 2 KB of system RAM
`get_ram()` can actually see.

Usage
-----
    # draft a profile (probes only — seconds of emulator time)
    python scripts/onboard_game.py --rom "roms/Kirby's Adventure (USA) (Rev A).nes"

    # ... the orchestrator then runs the printed calibration command,
    #     and afterwards asks for the verdict + next arm:
    python scripts/onboard_game.py --classify runs/onboard/<slug>/calib \\
        --profile configs/onboard/<slug>.yaml \\
        --verdict-module <the calibrated discriminator's import path>

    # ... and once a solved chain exists:
    python scripts/onboard_game.py --refine-chain runs/<solve dir> \\
        --profile configs/onboard/<slug>.yaml

Programmatic entry point:

    from scripts.onboard_game import onboard
    report = onboard("roms/Foo.nes")
    report.profile_path      # configs/onboard/foo.yaml
    report.solve_command     # argv tuple, or None when blocked
    report.classify_fn()     # wall verdict + next-arm recommendation
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------
# House constants.
# --------------------------------------------------------------------------

#: Progress records the calibrated wall discriminator needs before it
#: will classify a run at all. MIRRORED, not imported: the discriminator
#: is under an explicit "nothing outside tests and docs may import this
#: module" hold until the self-arming decision lands, so this module
#: stays free of it and `tests/test_onboard_game.py` — which the hold
#: carves out — pins this constant to the real one. Drift fails there.
WALL_RECORD_FLOOR = 12

#: Where drafts land. Kept out of `configs/` proper so a draft can never
#: be mistaken for a profile a human has signed off on.
ONBOARD_DIR = REPO / "configs" / "onboard"
#: Where the receipt + the calibration run directory go.
RUNS_DIR = REPO / "runs" / "onboard"
#: `scripts/go_explore_solve.py` appends one `progress.jsonl` line per
#: minute of wall clock. Everything the wall taxonomy needs is counted in
#: those records, so the calibration budget is derived from this, not
#: guessed.
PROGRESS_LINE_SECS = 60
#: Slack on top of the taxonomy's record floor: startup, the first
#: partial minute, and a worker restart must not push a healthy run back
#: under the floor and get it misread as INSUFFICIENT.
CALIBRATION_MARGIN_MINUTES = 3
#: An archive snapshot is the only thing that lets the taxonomy test the
#: cell key's spatial projection at all (its KEY_BLIND test reads
#: `spatial_span` off the snapshot), and without one a stalled run
#: abstains with the frontier evidence still on its missing list. The two
#: banked show runs in the taxonomy's corpus had `flush_secs` set to
#: ~forever and so persisted no `archive.pkl` at all. Calibration runs
#: flush often enough that the snapshot always exists.
CALIBRATION_FLUSH_SECS = 60
DEFAULT_WORKERS = 8
DEFAULT_MAX_STEPS = 4000
DEFAULT_WANT_SOLUTIONS = 1
DEFAULT_FRAME_SKIP = 4
DEFAULT_MAX_EPISODE_STEPS = 2500
DEFAULT_PYTHON = ".venv/bin/python"
SOLVER = "scripts/go_explore_solve.py"

#: `nes_core.get_ram()` masks to $07FF. An address above this is
#: cartridge PRG-RAM: unreachable by every consuming path, and in
#: practice the signature of an address copied in from outside rather
#: than measured here.
SYSTEM_RAM_TOP = 0x07FF

#: Keys `GenericGame.__init__` dereferences unconditionally. A draft
#: missing any of them would raise at solver startup, so the command is
#: withheld instead of handed over.
REQUIRED_SOLVE_KEYS = ("rom", "progress", "y", "level_key", "lives")
REQUIRED_PROFILE_KEYS = ("name", "start_state_path", "frame_skip",
                         "action_space", "solve")

#: Substrings that mark a claim as sourced from outside our own probing.
#: The generated text is written to avoid every one of them, so a hit is
#: always contamination someone added by hand.
EXTERNAL_PROVENANCE_TOKENS = (
    "datacrystal", "data crystal", "gamefaqs", "romhacking", "tcrf",
    "nesmaps", "walkthrough", "disassembl", "speedrun route",
    "community ram map", "third-party ram map",
)
#: Every top-level key this tool writes. A generated draft is a fully
#: known artifact, so ANY other block means the file was hand-edited and
#: is no longer what it claims to be — which is a stronger rule than
#: naming individual forbidden blocks, and catches the one that matters
#: (quarantined material smuggled back into a live profile) without
#: this module having to name banned knowledge to reject it.
KNOWN_PROFILE_KEYS = frozenset({
    "name", "start_state_path", "rom_hashes", "frame_skip",
    "max_episode_steps", "description", "cartridge", "provenance",
    "action_space", "solve",
})

#: Generic controller coverage. This is the PAD, not the game — every
#: button and the combos a search needs to express jumping, firing and
#: door-entry, with START deliberately absent (it pauses). Trim per game
#: once the calibration run says which axes matter.
DEFAULT_ACTION_SPACE: tuple[tuple[str, ...], ...] = (
    (), ("right",), ("left",), ("up",), ("down",),
    ("A",), ("B",),
    ("right", "A"), ("left", "A"), ("right", "B"), ("left", "B"),
    ("up", "A"), ("right", "up"), ("down", "B"),
)

#: Directions tried, in order, by `--forward auto`.
AUTO_FORWARD_ORDER = ("right", "up", "left", "down")


# --------------------------------------------------------------------------
# Lazy imports — the pure half of this module (YAML rendering, linting,
# command building, wall classification) must be importable, and
# testable, without an emulator in the process.
# --------------------------------------------------------------------------

def _discover():
    return importlib.import_module("scripts.discover_observables")


def _gen_auto_configs():
    return importlib.import_module("scripts.gen_auto_configs")


def _capture_start_state():
    return importlib.import_module("scripts.capture_start_state")


def _yaml():
    return importlib.import_module("yaml")


# --------------------------------------------------------------------------
# Small helpers.
# --------------------------------------------------------------------------

def _rel(path: str | Path) -> str:
    """Repo-relative when possible; absolute otherwise."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except (ValueError, OSError):
        return str(p)


def slug_for(rom: str | Path) -> str:
    """Filename slug for a ROM, using the same rule as `configs/auto/`."""
    return _gen_auto_configs().slugify(Path(rom).name)


def default_start_state_path(rom: str | Path) -> Path:
    """`roms/<stem>_start.state.bin` — the sidecar every profile names."""
    p = Path(rom)
    return p.with_name(p.stem + "_start.state.bin")


def min_calibration_minutes() -> int:
    """Shortest calibration budget that can yield a real wall verdict.

    The discriminator refuses to classify below `WALL_RECORD_FLOOR`
    progress records and the solver writes one per `PROGRESS_LINE_SECS`,
    so a shorter budget cannot return anything but INSUFFICIENT however
    healthy the search was.
    """
    return (math.ceil(WALL_RECORD_FLOOR * PROGRESS_LINE_SECS / 60)
            + CALIBRATION_MARGIN_MINUTES)


def _json_default(obj: Any) -> Any:
    for cast in (int, float):
        with contextlib.suppress(TypeError, ValueError):
            return cast(obj)
    return str(obj)


# --------------------------------------------------------------------------
# Start state.
# --------------------------------------------------------------------------

def ensure_start_state(rom: str | Path, *, out: Optional[str | Path] = None,
                       force: bool = False, max_frames: int = 2000
                       ) -> tuple[Path, str]:
    """Return `(path, source)` for a live-gameplay start state.

    `source` is one of `existing` (a sidecar was already banked),
    `generated` (captured now) — a caller-supplied path is resolved by
    `onboard` before this is reached and reported as `supplied`.

    Capture is delegated wholesale to `scripts/capture_start_state.py`:
    boot, pulse START/A through the title and menu flow, and snapshot the
    first moment RIGHT-vs-LEFT divergence proves the player is
    controllable. Cold-booting without this lands in the attract-mode
    demo, which ignores input.
    """
    rom = Path(rom)
    dest = Path(out) if out is not None else default_start_state_path(rom)
    if dest.exists() and not force:
        return dest, "existing"

    cap = _capture_start_state()
    argv = ["capture_start_state.py", "--game", slug_for(rom),
            "--rom", str(rom), "--out", str(dest),
            "--max-frames", str(int(max_frames))]
    saved, sys.argv = sys.argv, argv
    try:
        rc = cap.main()
    finally:
        sys.argv = saved
    if rc != 0 or not dest.exists():
        raise RuntimeError(
            f"could not capture a live-gameplay start state for {rom.name} "
            f"(capture_start_state rc={rc}). The game likely needs manual "
            f"menu navigation (a save-file registration screen, say); "
            f"capture one by hand and pass it with --start-state.")
    return dest, "generated"


# --------------------------------------------------------------------------
# Profile rendering.
# --------------------------------------------------------------------------

def _action_space_block(space: Iterable[Sequence[str]]) -> str:
    lines = ["action_space:"]
    for combo in space:
        body = ", ".join(f'"{b}"' for b in combo)
        lines.append(f"  - [{body}]")
    return "\n".join(lines) + "\n"


def render_profile_yaml(rom: str | Path, *, slug: str, solve_block: str,
                        start_state_path: str | Path,
                        start_state_source: str, forward: str,
                        frame_skip: int = DEFAULT_FRAME_SKIP,
                        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
                        header: Optional[Mapping[str, Any]] = None,
                        generated: Optional[str] = None,
                        receipt_path: Optional[str | Path] = None,
                        action_space: Iterable[Sequence[str]] = DEFAULT_ACTION_SPACE,
                        hw_flags: Sequence[str] = (),
                        probe_budget: Optional[Mapping[str, int]] = None,
                        ) -> str:
    """Render the draft profile.

    The structured head is dumped by PyYAML (no hand-rolled quoting), and
    the `solve:` block is appended VERBATIM from
    `discover_observables.emit_solve_yaml` — its inline comments carry
    the saturation warnings and the "confirm before trusting" markers,
    which are the most valuable thing discovery produces and must not be
    paraphrased away.
    """
    yaml = _yaml()
    rom = Path(rom)
    header = dict(header or {})
    generated = generated or _dt.date.today().isoformat()

    head: dict[str, Any] = {
        "name": rom.stem,
        "start_state_path": _rel(start_state_path),
        "rom_hashes": [header["md5"]] if header.get("md5") else [],
        "frame_skip": int(frame_skip),
        "max_episode_steps": int(max_episode_steps),
        "description": (
            f"DRAFT onboarding profile for {rom.stem}. Every address in the "
            f"solve: block below was drafted by the probe protocol in "
            f"scripts/discover_observables.py — clean directional holds, an "
            f"idle probe, a maneuver/death-recovery drive and a reload jump "
            f"probe, all run on this core against this ROM. Nothing here is "
            f"signed off: the calibration solve named under provenance."
            f"calibration has not been run yet."
        ),
    }
    if header.get("mapper") is not None:
        head["cartridge"] = {
            "mapper": int(header["mapper"]),
            "nes20": bool(header.get("is_nes20", False)),
            "battery": bool(header.get("battery", False)),
            "prg_kb": int(header.get("prg_size_kb", 0)),
            "chr_kb": int(header.get("chr_size_kb", 0)),
        }
    head["provenance"] = {
        "tool": "scripts/onboard_game.py",
        "generated": generated,
        "status": "DRAFT — calibration solve not yet run; no address here "
                  "has survived a search.",
        "protocol": "scripts/discover_observables.py (find_progress / "
                    "find_room_counter / find_y / find_hp_lives, Gate 1 "
                    "flat-under-idle + Gate 2 saturation)",
        "evidence": "our own scripted rollouts on this ROM from the start "
                    "state named above",
        "forward": forward,
        "start_state": {"path": _rel(start_state_path),
                        "source": start_state_source},
        "outside_sources_consulted": "none",
        "calibration": {
            "run_by": "the orchestrator, not this tool",
            "min_minutes_for_a_verdict": min_calibration_minutes(),
            "why": (f"the wall discriminator needs {WALL_RECORD_FLOOR} "
                    f"progress records and the solver writes one every "
                    f"{PROGRESS_LINE_SECS}s"),
        },
    }
    if probe_budget:
        head["provenance"]["probe_budget"] = dict(probe_budget)
    if receipt_path is not None:
        head["provenance"]["receipt"] = _rel(receipt_path)

    banner = (
        f"# DRAFT PROFILE — generated by scripts/onboard_game.py "
        f"({generated}).\n"
        f"# Not hand-verified. Read the solve: block's own comments before\n"
        f"# trusting any address, and re-generate rather than editing in\n"
        f"# place while the draft is still being calibrated.\n"
    )
    body = yaml.safe_dump(head, sort_keys=False, allow_unicode=True,
                          width=88, default_flow_style=False)
    text = banner + body + "\n" + _action_space_block(action_space) + "\n"
    text += solve_block if solve_block.endswith("\n") else solve_block + "\n"
    if hw_flags:
        flags = ", ".join(f'"{f}"' for f in hw_flags)
        text += (f"  # Timing flags this draft was probed under; a lineage "
                 f"built with them must be re-solved with them.\n"
                 f"  hw_flags: [{flags}]\n")
    return text


def _try_probe_budget() -> dict[str, int]:
    """The probe budget, when discovery is importable; `{}` otherwise."""
    try:
        d = _discover()
    except Exception:
        return {}
    return {"clean_forward": int(d.CLEAN_N), "clean_reverse": int(d.CLEAN_N),
            "noop": int(d.NOOP_N), "advance": int(d.ADVANCE_N),
            "settle_scan": int(d.SETTLE_SCAN),
            "death_drives": int(d.DEATH_REPS) * int(d.DEATH_MAX_N)}


# --------------------------------------------------------------------------
# Lint. The mechanical half of the purity line.
# --------------------------------------------------------------------------

#: Keys under `solve:` whose integers are VALUES, not addresses.
#: `progress_cap` is the one that matters — Metroid pins it at 16384 and
#: Kirby at 4000, both legitimately far above the top of system RAM.
#: Everything not listed here is treated as an address, so a newly
#: invented address key is checked by default and only a new *value* key
#: needs this list extended.
NON_ADDRESS_SOLVE_KEYS = frozenset({
    "progress_cap", "threshold", "scale", "blank", "steps", "near", "p",
    "value", "mod", "start", "types", "match", "death_states",
})


def _walk_ints(obj: Any, path: str = "solve", key: str = "solve"
               ) -> Iterable[tuple[str, str, int]]:
    """Yield `(path, nearest enclosing key, value)` for every int found."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _walk_ints(v, f"{path}.{k}", str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_ints(v, f"{path}[{i}]", key)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, int):
        yield path, key, obj


def lint_profile_text(text: str) -> tuple[str, ...]:
    """Every reason this profile is not fit to hand to the solver.

    Empty tuple = clean. The checks are deliberately mechanical so a
    contaminated copy fails loudly in a test rather than being caught by
    a reader's attention:

      * it parses, and carries the keys the loader and `GenericGame`
        dereference;
      * every address under `solve:` is inside the 2 KB of system RAM
        `get_ram()` can reach (a $6000-range byte is unreadable AND is
        the classic outside-provenance tell);
      * no outside-provenance token, and no top-level block this tool
        did not write;
      * the provenance stamp names the probe protocol, so a
        hand-assembled profile cannot pass by omitting its own history.
    """
    yaml = _yaml()
    problems: list[str] = []
    try:
        prof = yaml.safe_load(text)
    except Exception as e:  # pragma: no cover - exercised via mutation test
        return (f"profile does not parse as YAML: {e}",)
    if not isinstance(prof, dict):
        return ("profile is not a YAML mapping",)

    for key in REQUIRED_PROFILE_KEYS:
        if key not in prof:
            problems.append(f"missing top-level key: {key}")
    for key in sorted(set(prof) - KNOWN_PROFILE_KEYS):
        problems.append(
            f"unrecognized top-level key: {key} — a generated draft carries "
            f"only the keys this tool writes, so an extra block (quarantined "
            f"material pasted back in, a hand-added address table) means the "
            f"file is no longer the artifact its provenance stamp claims")

    low = text.lower()
    for token in EXTERNAL_PROVENANCE_TOKENS:
        if token in low:
            problems.append(f"outside-provenance token in profile text: {token!r}")

    solve = prof.get("solve")
    if not isinstance(solve, dict):
        problems.append("solve: block missing or not a mapping")
    else:
        for key in REQUIRED_SOLVE_KEYS:
            if key not in solve:
                problems.append(
                    f"solve.{key} missing — GenericGame dereferences it "
                    f"unconditionally, so the solver would fail at startup")
        for where, key, value in _walk_ints(solve):
            if key in NON_ADDRESS_SOLVE_KEYS:
                continue
            if value > SYSTEM_RAM_TOP:
                problems.append(
                    f"{where} = 0x{value:04X} is above 0x{SYSTEM_RAM_TOP:04X}: "
                    f"outside the system RAM get_ram() can read")
            elif value < 0:
                problems.append(f"{where} = {value} is negative")

    prov = prof.get("provenance")
    if not isinstance(prov, dict):
        problems.append("missing provenance: block")
    else:
        if "discover_observables" not in str(prov.get("protocol", "")):
            problems.append("provenance.protocol does not name the probe "
                            "protocol (scripts/discover_observables.py)")
        if not prov.get("evidence"):
            problems.append("provenance.evidence is empty")
    return tuple(problems)


# --------------------------------------------------------------------------
# The bounded calibration command.
# --------------------------------------------------------------------------

def _merge_argv(base: Sequence[str],
                overrides: Sequence[tuple[str, Optional[str]]]) -> tuple[str, ...]:
    """Apply `--opt value` overrides to an argv, replacing in place.

    Appending would also work (argparse takes the last occurrence) but
    produces commands that read as contradictory. `(opt, None)` is a
    store-true flag.
    """
    out = list(base)
    for opt, value in overrides:
        try:
            i = out.index(opt)
        except ValueError:
            out.append(opt)
            if value is not None:
                out.append(value)
            continue
        if value is None:
            continue
        if i + 1 < len(out) and not out[i + 1].startswith("--"):
            out[i + 1] = value
        else:
            out.insert(i + 1, value)
    return tuple(out)


def _argv_value(argv: Sequence[str], opt: str) -> Optional[str]:
    try:
        i = list(argv).index(opt)
    except ValueError:
        return None
    return argv[i + 1] if i + 1 < len(argv) else None


def _drop_argv(argv: Sequence[str], opt: str) -> tuple[str, ...]:
    """Remove `--opt` and its value from an argv, every occurrence.

    `_merge_argv` cannot express this: there `(opt, None)` means a
    store-true flag. Dropping is needed because an inherited
    `--resume-archive` is not inert — it points the solver at a
    directory it will unconditionally open.
    """
    out = list(argv)
    i = 0
    while i < len(out):
        if out[i] != opt:
            i += 1
            continue
        end = i + 1
        if end < len(out) and not out[end].startswith("--"):
            end += 1
        del out[i:end]
    return tuple(out)


#: What `go_explore_solve.py --resume-archive DIR` opens, in order, with
#: no existence check of its own (go_explore_solve.py:2273-2276 ->
#: src/training/go_explore.py:310). A dir missing ANY of these is not a
#: resume point: the solver raises FileNotFoundError at startup, before
#: a single step, and the whole budgeted run is lost.
RESUME_ARTIFACTS: tuple[str, ...] = ("archive.pkl", "traces.pkl", "roots.json")


def missing_resume_artifacts(run_dir: Optional[str | Path]) -> tuple[str, ...]:
    """Which of `RESUME_ARTIFACTS` a candidate resume dir does not have.

    The commands this module emits carry REPO-RELATIVE paths, so a
    relative dir is checked against the repo root as well as the cwd —
    otherwise the check would silently fail from anywhere else and every
    resume would be dropped.
    """
    if run_dir is None:
        return RESUME_ARTIFACTS
    d = Path(run_dir)
    if not d.is_absolute() and not d.is_dir():
        d = REPO / d
    return tuple(n for n in RESUME_ARTIFACTS if not (d / n).exists())


def resume_ready(run_dir: Optional[str | Path]) -> bool:
    """True only when a dir holds a COMPLETE flushed snapshot.

    A short run, a crashed run, or any run whose `--flush-secs` never
    came around writes `progress.jsonl` and nothing else. Resuming from
    it is not degraded — it is a hard crash at solver startup.
    """
    return not missing_resume_artifacts(run_dir)


def build_solve_command(profile_path: str | Path, root_state: str | Path,
                        out_dir: str | Path, *,
                        minutes: Optional[float] = None,
                        workers: int = DEFAULT_WORKERS,
                        want_solutions: int = DEFAULT_WANT_SOLUTIONS,
                        max_steps: int = DEFAULT_MAX_STEPS,
                        flush_secs: float = CALIBRATION_FLUSH_SECS,
                        seed: int = 0,
                        hw_flags: Sequence[str] = (),
                        python: str = DEFAULT_PYTHON) -> tuple[str, ...]:
    """The exact bounded calibration-solve argv for the orchestrator.

    Two of the defaults are load-bearing rather than taste:
    `--minutes` is at least `min_calibration_minutes()` so the run
    produces enough progress records for a verdict, and `--flush-secs`
    is short enough that an `archive.pkl` actually lands — without one
    the taxonomy cannot test the cell key's spatial projection at all, so
    a key-blind search is indistinguishable from one that merely stalled,
    and the abstention that comes back is missing the frontier evidence a
    human would need to act on it.
    """
    minutes = min_calibration_minutes() if minutes is None else float(minutes)
    argv = [python, SOLVER,
            "--out", _rel(out_dir),
            "--root-state", _rel(root_state),
            "--profile", _rel(profile_path),
            "--minutes", f"{minutes:g}",
            "--workers", str(int(workers)),
            "--want-solutions", str(int(want_solutions)),
            "--max-steps", str(int(max_steps)),
            "--flush-secs", f"{float(flush_secs):g}",
            "--seed", str(int(seed))]
    if hw_flags:
        argv += ["--hw-flags", ",".join(hw_flags)]
    return tuple(argv)


# --------------------------------------------------------------------------
# Post-run hook: wall verdict + next arm.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NextArm:
    """What to do about a verdict, and the command that does it."""

    arm: str
    #: run = launch `command`; stop = the campaign is done here;
    #: fix = do not spend more compute until the listed repair lands.
    action: str
    remedy: str
    rationale: str
    command: Optional[tuple[str, ...]] = None
    profile_edits: tuple[str, ...] = ()
    calibration: str = ""
    #: Facts about the emitted command itself (as opposed to the search):
    #: today, why it does or does not resume from the previous run dir.
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "action": self.action, "remedy": self.remedy,
                "rationale": self.rationale,
                "command": list(self.command) if self.command else None,
                "profile_edits": list(self.profile_edits),
                "calibration": self.calibration,
                "notes": list(self.notes)}


# Verdict labels, as the WIRE STRINGS the discriminator emits. Mirrored
# rather than imported for the reason given on WALL_RECORD_FLOOR; the
# test file pins this set against the real enumeration.
RESOLVED = "resolved"
PROGRESSING = "progressing"
COVERAGE_LIMITED = "coverage_limited"
BARREN = "barren"
KEY_BLIND = "key_blind"
INDETERMINATE = "indeterminate"
INSUFFICIENT = "insufficient"

#: One arm per verdict. Exhaustive over the discriminator's labels (a
#: test pins that) so a new label cannot silently fall through to "run it
#: longer", which is the wrong answer for half of them. An unrecognized
#: label stops rather than guesses — see `UNKNOWN_ARM`.
#:
#: There is deliberately NO arm for a certified wall, because the
#: discriminator no longer certifies one: the classification was struck
#: upstream (its tag reads CLASSIFICATION-STRUCK) after 22 candidate
#: statistics scored over 103 archives all failed to separate a search
#: that is stuck from one that is about to finish. A stalled run that is
#: neither barren nor key-blind therefore ABSTAINS, and the only honest
#: arm for an abstention is the instrumentation checklist — never a
#: campaign bought on the strength of a verdict nobody made.
ARM_FOR_CLASS: dict[str, str] = {
    RESOLVED: "harvest",
    PROGRESSING: "extend",
    COVERAGE_LIMITED: "extend",
    BARREN: "repair_search",
    KEY_BLIND: "enrich_key",
    INDETERMINATE: "instrument",
    INSUFFICIENT: "extend",
}
UNKNOWN_ARM = "hold"


def _next_out_dir(base_out: Optional[str], arm: str) -> Optional[str]:
    if not base_out:
        return None
    p = Path(base_out)
    return str(p.with_name(f"{p.name}_{arm}"))


def _verdict_field(verdict: Any, name: str, default: Any) -> Any:
    value = getattr(verdict, name, None)
    if value is None:
        return default
    return value


def next_arm(verdict: Any, *, base_command: Optional[Sequence[str]] = None,
             run_dir: Optional[str | Path] = None,
             profile: Optional[Mapping[str, Any]] = None) -> NextArm:
    """Turn a wall verdict into the next thing to run (or not run).

    Pure, and DUCK-TYPED on the verdict: it reads `.wall_class`,
    `.reasons`, `.remedy`, `.missing` and `.calibration`, so the
    calibrated discriminator can be handed in without this module
    importing it. Everything else it needs is the command that produced
    the run and — optionally — the profile, which only ever makes a
    suggestion more concrete (naming the `area` byte a transition counter
    could be instrumented off, say) and never makes it more certain. No
    game knowledge enters here; the arms are properties of the SEARCH.

    An emitted command names `--resume-archive` only when the dir it
    points at holds a complete flushed snapshot (`RESUME_ARTIFACTS`),
    and never on the INDETERMINATE or KEY_BLIND arms — see `resume_ready`
    and the comment on the resume-eligibility block below. `.notes`
    records the decision either way.

    A verdict may also carry a DESCRIPTOR — a non-verdict annotation
    saying what the archive looks like. It is read here only to be
    repeated back with its caveat attached: no arm is selected from it,
    because the statistic that used to turn that observation into a
    classification was struck.
    """
    raw = _verdict_field(verdict, "wall_class", "")
    cls = str(getattr(raw, "value", raw))
    arm = ARM_FOR_CLASS.get(cls, UNKNOWN_ARM)
    remedy = str(_verdict_field(verdict, "remedy", ""))
    calibration = str(_verdict_field(verdict, "calibration", ""))
    reasons = "; ".join(_verdict_field(verdict, "reasons", ())) or "no reasons recorded"
    base = tuple(base_command or ())
    prev_minutes = _argv_value(base, "--minutes")
    solve = dict((profile or {}).get("solve") or {})

    # --- resume eligibility -------------------------------------------
    # `--resume-archive DIR` makes the solver open DIR/archive.pkl,
    # DIR/traces.pkl and DIR/roots.json with no existence check, so a dir
    # that never flushed is a startup FileNotFoundError, not a degraded
    # resume. Every arm therefore names a resume dir only after checking
    # it, and an INDETERMINATE run is never resumed at all: that arm
    # re-instruments from the root, so carrying this run's archive into
    # the next attempt would confound the very telemetry it was launched
    # to collect — and when no snapshot landed there is nothing there to
    # resume from in the first place.
    lacks = missing_resume_artifacts(run_dir)
    resume = str(run_dir) if run_dir is not None and not lacks else None
    notes: list[str] = []
    if run_dir is not None and lacks and base:
        notes.append(
            f"not resuming from {run_dir}: no flushed snapshot there "
            f"(missing {', '.join(lacks)}) — --resume-archive would "
            f"FileNotFoundError at solver startup, before a single step")

    def rebuild(overrides: Sequence[tuple[str, Optional[str]]], *,
                may_resume: bool = True) -> Optional[tuple[str, ...]]:
        if not base:
            return None
        ov = list(overrides)
        out = _next_out_dir(_argv_value(base, "--out"), arm)
        if out:
            ov.append(("--out", out))
        keep = resume if may_resume else None
        if keep is None:
            if not may_resume and resume:
                notes.append(f"deliberately NOT resuming from {resume} even "
                             f"though it has a snapshot: this arm re-"
                             f"instruments from the root so the next verdict "
                             f"is not confounded by the last one's archive")
            # An inherited --resume-archive from the command that
            # produced this run is not inert — it names a directory the
            # solver will open. Keep it only if it is still a complete
            # snapshot, and never when this arm has ruled resuming out.
            inherited = _argv_value(base, "--resume-archive")
            if may_resume and inherited and resume_ready(inherited):
                keep = inherited
            elif inherited:
                notes.append(
                    f"dropped the inherited --resume-archive {inherited}: "
                    + ("this arm must re-instrument from the root"
                       if not may_resume else
                       "that dir no longer holds a complete snapshot"))
        merged = _merge_argv(base, ov)
        if keep:
            return _merge_argv(merged, [("--resume-archive", keep)])
        return _drop_argv(merged, "--resume-archive")

    def longer(factor: float = 2.0, floor: Optional[int] = None) -> str:
        try:
            m = float(prev_minutes) if prev_minutes else float(min_calibration_minutes())
        except ValueError:
            m = float(min_calibration_minutes())
        m *= factor
        if floor is not None:
            m = max(m, float(floor))
        return f"{m:g}"

    def made(action: str, rationale: str, *,
             command: Optional[tuple[str, ...]] = None,
             edits: Sequence[str] = ()) -> NextArm:
        # `notes` describe the emitted command, so they only mean
        # something when there is one. `rebuild` has already run by the
        # time this is called (it is an argument), so the list is final.
        return NextArm(arm, action, remedy, rationale, command=command,
                       profile_edits=tuple(edits), calibration=calibration,
                       notes=tuple(notes) if command else ())

    if cls == RESOLVED:
        return made("stop", f"{reasons}. Harvest the banked tape and move the "
                            f"draft profile toward a signed-off one.")

    if cls in (PROGRESSING, COVERAGE_LIMITED):
        return made("run",
                    f"{reasons}. The frontier is still moving, so the cheapest "
                    f"correct action is more wall clock on the SAME arm — "
                    f"changing the search now would confound the next verdict.",
                    command=rebuild([("--minutes", longer())]))

    if cls == INSUFFICIENT:
        return made("run",
                    f"{reasons}. Below the discriminator's evidence floor "
                    f"({WALL_RECORD_FLOOR} progress records at one per "
                    f"{PROGRESS_LINE_SECS}s); re-run long enough to be "
                    f"classified at all.",
                    command=rebuild([("--minutes",
                                      longer(1.5, floor=min_calibration_minutes()))]))

    if cls == INDETERMINATE:
        missing = ", ".join(_verdict_field(verdict, "missing", ())) \
            or "corroborating telemetry"
        descriptor = str(_verdict_field(verdict, "descriptor", ""))
        # NOTHING WAS CLASSIFIED, and that is the whole content of this
        # arm. Since the wall classification was struck upstream this is
        # the terminal class for every stalled search that is not barren
        # and not key-blind, so the recommendation is the discovery /
        # telemetry checklist — collect what the discriminator said it
        # lacked, then decide on grounds it does not supply. NEVER
        # resumes: the run is re-instrumented from the root, with a flush
        # short enough to bank a snapshot, so the next verdict is read
        # off fresh telemetry rather than this run's archive.
        edits = [
            f"FIRST: nothing classified this wall — the discriminator "
            f"ABSTAINED. Collect what it named missing ({missing}) and read "
            f"the verdict again before spending compute on any orthogonal "
            f"arm: no statistic in this telemetry tells a search that is "
            f"stuck from one that is about to finish, so an arm chosen here "
            f"is a guess wearing a verdict's clothes",
            "re-run discovery (scripts/discover_observables.py) against this "
            "profile: an abstention taken over a cell key the probes never "
            "enriched is a measurement gap before it is a hard game",
        ]
        if "area" in solve:
            edits.append(
                f"solve.area is already 0x{int(solve['area']):04X} — that is "
                f"the byte a monotone room/area transition counter can be "
                f"instrumented off, and a real map-progress counter is one "
                f"of the terms the discriminator is currently substituting "
                f"a stand-in for")
        else:
            edits.append("no solve.area in this profile: re-run discovery, or "
                         "there is no observable to count discrete "
                         "transitions with")
        if descriptor:
            edits.append(
                f"the verdict carries the descriptor {descriptor}: a "
                f"DESCRIPTION of what this archive looks like, not a "
                f"diagnosis and not a licence to act. The statistic that "
                f"used to turn that observation into a classification was "
                f"struck upstream, so read it as an annotation on the "
                f"abstention above and not as a wall")
        return made("run",
                    f"{reasons}. Missing: {missing}. NOT CLASSIFIED — the "
                    f"discriminator abstained, so the next spend is "
                    f"instrumentation, not a campaign: re-run FROM THE ROOT "
                    f"with a short flush so an archive.pkl actually lands and "
                    f"the checklist above has something to read.",
                    command=rebuild([("--flush-secs", str(CALIBRATION_FLUSH_SECS)),
                                     ("--minutes", longer(1.5))],
                                    may_resume=False),
                    edits=edits)

    if cls == KEY_BLIND:
        edits = []
        if "y" not in solve:
            edits.append("solve.y is absent — the vertical axis is invisible "
                         "to the cell key")
        if "area" not in solve:
            edits.append("solve.area is absent — room/screen changes do not "
                         "partition the archive")
        edits.append("finer cell buckets are the cheap first move; a missing "
                     "state axis is the real fix")
        return made("run",
                    f"{reasons}. No coverage statistic over this key means "
                    f"anything until the key can see where progress happens.",
                    command=rebuild([("--gx-bucket", "8"), ("--y-band", "16"),
                                     ("--minutes", longer())],
                                    may_resume=False),
                    edits=edits)

    if cls == BARREN:
        # The one verdict where more compute cannot help.
        return made("fix",
                    f"{reasons}. The archive never accumulated, so this is a "
                    f"broken search rather than a hard game: do not buy more "
                    f"wall clock.",
                    edits=(
                        "verify the start state is LIVE GAMEPLAY, not a title/"
                        "attract screen (re-run onboarding with --force-state)",
                        "verify solve.progress actually moves under the action "
                        "space (re-read the draft's own Gate 1 / Gate 2 comments)",
                        "verify determinism: a lineage built under hw timing "
                        "flags must be re-solved under the same flags",
                    ))

    # An unrecognized label. Stopping is the only safe default: every
    # wrong guess here spends orchestrator hours, and half the known
    # labels want something other than "run it longer".
    return made("fix",
                f"unrecognized wall class {cls!r} (known: "
                f"{', '.join(sorted(ARM_FOR_CLASS))}). {reasons}. Refusing to "
                f"guess an arm — the recommendation table needs the new label "
                f"added before this verdict can be acted on.")


def load_verdict_fn(module_path: str, *, window: Optional[int] = None
                    ) -> Callable[..., Any]:
    """Bind a classifier from a module the OPERATOR names.

    The module must expose the two-function contract the discriminator
    already has:

        telemetry_from_paths(progress_path, archive_path=, segment=,
                             label=) -> telemetry
        gated_wall_verdict(telemetry, window=) -> verdict

    and the verdict must carry `.wall_class`, `.reasons`, `.remedy`,
    `.missing` and (optionally) `.descriptor` / `.calibration` /
    `.as_dict()`.

    Kept as a runtime lookup on a caller-supplied path rather than an
    import so this module has no compile-time dependency on the
    classifier — see `classify_run` for why that matters.
    """
    mod = importlib.import_module(module_path)
    for fn in ("telemetry_from_paths", "gated_wall_verdict"):
        if not hasattr(mod, fn):
            raise TypeError(
                f"{module_path} does not expose {fn}(): it cannot be used as a "
                f"wall classifier. Expected the two-function contract "
                f"documented on load_verdict_fn.")

    def verdict_fn(progress_path, *, archive_path=None, segment=-1, label=""):
        telemetry = mod.telemetry_from_paths(
            progress_path, archive_path=archive_path, segment=segment,
            label=label)
        kw = {} if window is None else {"window": window}
        return mod.gated_wall_verdict(telemetry, **kw)

    verdict_fn.source = module_path  # type: ignore[attr-defined]
    return verdict_fn


@dataclass(frozen=True)
class NoTelemetryVerdict:
    """Stand-in verdict for a run that produced no telemetry at all.

    Same duck-typed surface the calibrated discriminator presents, so
    `next_arm` and the result shape do not special-case it.
    """

    wall_class: str = INSUFFICIENT
    label: str = ""
    degraded: bool = True
    missing: tuple[str, ...] = ("progress.jsonl",)
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=lambda: {"records": 0})
    remedy: str = "keep running; too little evidence to classify"
    #: Same non-verdict annotation slot the real discriminator carries;
    #: always empty here, because a run with no telemetry has nothing to
    #: describe.
    descriptor: str = ""
    calibration: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"wall_class": self.wall_class, "label": self.label,
                "degraded": self.degraded, "missing": list(self.missing),
                "reasons": list(self.reasons), "evidence": dict(self.evidence),
                "remedy": self.remedy, "descriptor": self.descriptor,
                "calibration": self.calibration}


def _verdict_dict(verdict: Any) -> dict[str, Any]:
    as_dict = getattr(verdict, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    raw = _verdict_field(verdict, "wall_class", "")
    return {"wall_class": str(getattr(raw, "value", raw)),
            "label": str(_verdict_field(verdict, "label", "")),
            "degraded": bool(_verdict_field(verdict, "degraded", True)),
            "missing": list(_verdict_field(verdict, "missing", ())),
            "reasons": list(_verdict_field(verdict, "reasons", ())),
            "evidence": dict(_verdict_field(verdict, "evidence", {})),
            "remedy": str(_verdict_field(verdict, "remedy", "")),
            "descriptor": str(_verdict_field(verdict, "descriptor", "")),
            "calibration": str(_verdict_field(verdict, "calibration", ""))}


def classify_run(run_dir: str | Path, *,
                 verdict_fn: Optional[Callable[..., Any]] = None,
                 base_command: Optional[Sequence[str]] = None,
                 profile_path: Optional[str | Path] = None,
                 label: str = "", segment: int = -1) -> dict[str, Any]:
    """Wall verdict + next-arm recommendation for a finished run dir.

    This is the hook the orchestrator calls after the bounded solve:
    point it at the run directory and it locates `progress.jsonl` and
    `archive.pkl`, scores them, and returns the verdict alongside the
    concrete next command.

    `verdict_fn(progress_path, archive_path=..., segment=..., label=...)`
    supplies the CLASSIFIER, and is deliberately not defaulted. The
    calibrated discriminator lives in `src/training/` under an explicit
    hold — nothing outside tests and docs may import it until the
    self-arming decision lands — so this module never reaches for it;
    the caller binds it. `load_verdict_fn` builds the binding from a
    module path the operator names, which is how the CLI does it.

    Without a `verdict_fn` the run is still inspected and reported, but
    the verdict is left unclassified rather than guessed at.
    """
    run_dir = Path(run_dir)
    progress = run_dir / "progress.jsonl"
    archive = run_dir / "archive.pkl"
    profile: Optional[dict] = None
    if profile_path is not None and Path(profile_path).exists():
        profile = _yaml().safe_load(Path(profile_path).read_text())

    # The programmatic hook (`report.classify_fn`) always carries the
    # exact command that produced the run. A bare CLI call does not, so
    # rebuild the same command the draft would have produced rather than
    # returning advice with no way to act on it. Flagged as reconstructed
    # so nobody reads a differing --workers as the run's real setting.
    base_source = "supplied" if base_command else "none"
    if not base_command and profile and profile.get("start_state_path"):
        base_command = build_solve_command(
            profile_path, REPO / str(profile["start_state_path"]), run_dir)
        base_source = "reconstructed from the profile's defaults"

    if not progress.exists():
        verdict = NoTelemetryVerdict(
            label=label or str(run_dir),
            reasons=(f"no progress.jsonl under {run_dir}: the run produced no "
                     f"telemetry at all",))
        arm = NextArm(
            "repair_launch", "fix", verdict.remedy,
            f"{verdict.reasons[0]}. A run that wrote no progress line did not "
            f"start; check the launch (profile path, root state, hw flags) "
            f"before spending more compute.")
        return {
            "run_dir": str(run_dir), "progress_path": None,
            "archive_path": None, "records": 0,
            "wall_class": verdict.wall_class,
            "verdict": verdict.as_dict(), "next_arm": arm.as_dict(),
            "base_command": list(base_command) if base_command else None,
            "base_command_source": base_source,
            "classifier": "n/a — no telemetry to classify",
        }

    if verdict_fn is None:
        return {
            "run_dir": str(run_dir), "progress_path": str(progress),
            "archive_path": str(archive) if archive.exists() else None,
            "records": None, "wall_class": None, "verdict": None,
            "next_arm": None,
            "base_command": list(base_command) if base_command else None,
            "base_command_source": base_source,
            "classifier": "none supplied — pass verdict_fn= (or the CLI's "
                          "--verdict-module) to have this run classified",
        }

    verdict = verdict_fn(progress,
                         archive_path=archive if archive.exists() else None,
                         segment=segment, label=label or str(run_dir))
    arm = next_arm(verdict, base_command=base_command, run_dir=run_dir,
                   profile=profile)
    vd = _verdict_dict(verdict)
    return {
        "run_dir": str(run_dir),
        "progress_path": str(progress),
        "archive_path": str(archive) if archive.exists() else None,
        "records": _verdict_field(verdict, "evidence", {}).get("records"),
        "wall_class": vd["wall_class"],
        "verdict": vd,
        "next_arm": arm.as_dict(),
        "base_command": list(base_command) if base_command else None,
        "base_command_source": base_source,
        "classifier": getattr(verdict_fn, "source", "supplied by the caller"),
    }


# --------------------------------------------------------------------------
# Refinement: the learnfun path, once solved tapes exist.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementResult:
    profile_path: Path
    refined_path: Optional[Path]
    findings: dict
    gates: dict
    quality: dict
    changed: bool
    notes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"profile_path": str(self.profile_path),
                "refined_path": str(self.refined_path) if self.refined_path else None,
                "gates": self.gates, "quality": self.quality,
                "changed": self.changed, "notes": list(self.notes),
                "blockers": list(self.blockers)}


def refine_from_tapes(profile_path: str | Path, chain: Any, *,
                      rom: Optional[str] = None,
                      hw_flags: Optional[str] = None,
                      state: Any = None,
                      forward: Optional[str] = None,
                      reference: Any = "shuffled",
                      out_path: Optional[str | Path] = None,
                      write: bool = True,
                      find_from_tapes: Optional[Callable[..., dict]] = None,
                      emit: Optional[Callable[..., str]] = None,
                      ) -> RefinementResult:
    """Re-draft the progress byte from an ALREADY-SOLVED tape chain.

    Wires `discover_observables.find_progress_from_tapes`: the
    lexicographic objective learner ranks RAM locations over the banked
    chain, and the same two gates the live probe uses adjudicate the
    ranking. Chains, not single tapes — a per-run counter can be
    invisible on one tape and obvious across thirty.

    The result is written as a SIBLING (`<slug>.refined.yaml`), never
    over the draft. The learnfun path emits its progress line marked
    "confirm before trusting" for a reason: the ranking is a shortlist,
    it cannot tell a timer from progress on its own, and nothing it
    produces may become a reward term.
    """
    profile_path = Path(profile_path)
    yaml = _yaml()
    prof = yaml.safe_load(profile_path.read_text())
    solve = dict(prof.get("solve") or {})
    d = _discover()
    find_from_tapes = find_from_tapes or d.find_progress_from_tapes
    emit = emit or d.emit_solve_yaml
    forward = forward or (prof.get("provenance") or {}).get("forward") or "right"
    rom_rel = rom or solve.get("rom")

    tape = find_from_tapes(chain, profile=prof, rom=rom, hw_flags=hw_flags,
                           state=state, forward=forward, reference=reference)

    room = tape.get("room_counter")
    findings = {
        "rom": rom_rel,
        "state": prof.get("start_state_path"),
        "forward": tape.get("forward", forward),
        "progress": tape,
        "room_counters": ([{"addr": room}] if room is not None else []),
        # The tape path re-ranks PROGRESS only; y and the death proxy are
        # carried over from the draft rather than silently dropped, which
        # is what would happen if the emitted block were rebuilt from the
        # tape findings alone.
        "y": ([{"addr": int(solve["y"])}] if "y" in solve else []),
        "hp_lives": ({"addr": int(solve["lives"]), "kind": "lives",
                      "value": None,
                      "note": "carried over from the draft profile"}
                     if "lives" in solve else
                     {"addr": None, "kind": "none", "note": "not in the draft"}),
    }
    block = emit(rom_rel, findings)

    rec = tape.get("recommended") or {}
    old_lo = (solve.get("progress") or {}).get("lo")
    changed = bool(rec) and rec.get("lo") != old_lo
    notes: list[str] = []
    blockers: list[str] = []
    if not rec:
        blockers.append("no tape candidate cleared both gates — the draft's "
                        "progress byte stands unchanged")
    elif changed:
        notes.append(f"tape ranking proposes 0x{int(rec['lo']):04X} where the "
                     f"draft has "
                     f"{('0x%04X' % int(old_lo)) if old_lo is not None else 'none'}"
                     f" — SHORTLIST ONLY, confirm before wiring")
    else:
        notes.append("tape ranking agrees with the draft's progress byte")

    rendered = _render_refinement(prof, block, tape, source=profile_path,
                                  notes=notes)
    blockers.extend(lint_profile_text(rendered))

    refined_path: Optional[Path] = None
    if write:
        refined_path = (
            Path(out_path) if out_path is not None
            else profile_path.with_name(profile_path.stem + ".refined.yaml"))
        refined_path.parent.mkdir(parents=True, exist_ok=True)
        refined_path.write_text(rendered)
    return RefinementResult(profile_path=profile_path, refined_path=refined_path,
                            findings=findings, gates=dict(tape.get("gates") or {}),
                            quality=dict(tape.get("quality") or {}),
                            changed=changed, notes=tuple(notes),
                            blockers=tuple(blockers))


def _render_refinement(prof: Mapping[str, Any], solve_block: str,
                       tape: Mapping[str, Any], *, source: Path,
                       notes: Sequence[str]) -> str:
    yaml = _yaml()
    head = {k: v for k, v in prof.items() if k not in ("solve", "provenance")}
    prov = dict(prof.get("provenance") or {})
    learnfun = dict(tape.get("learnfun") or {})
    prov.update({
        "refined_from": _rel(source),
        "refined": _dt.date.today().isoformat(),
        "refinement_protocol":
            "scripts/discover_observables.py find_progress_from_tapes "
            "(lexicographic ranking over banked solver tapes, re-adjudicated "
            "by Gate 1 flat-under-idle + Gate 2 saturation)",
        "refinement_status":
            "SHORTLIST — a ranked candidate is not a confirmed observable, "
            "and nothing ranked here may become a reward term, archive score "
            "or cell key",
        "refinement_evidence": {
            "n_segments": learnfun.get("n_segments"),
            "n_memories": learnfun.get("n_memories"),
            "gates": dict(tape.get("gates") or {}),
            "quality_verdict": (tape.get("quality") or {}).get("verdict"),
        },
    })
    head["provenance"] = prov
    banner = ("# REFINED DRAFT — regenerated by scripts/onboard_game.py from a\n"
              "# solved-tape chain. The draft it refines is left untouched.\n")
    for n in notes:
        banner += f"# note: {n}\n"
    body = yaml.safe_dump(head, sort_keys=False, allow_unicode=True, width=88,
                          default_flow_style=False)
    space = prof.get("action_space")
    text = banner + body + "\n"
    if space:
        text += _action_space_block(tuple(tuple(c or ()) for c in space)) + "\n"
    return text + (solve_block if solve_block.endswith("\n") else solve_block + "\n")


# --------------------------------------------------------------------------
# The deliverable.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OnboardReport:
    rom: str
    slug: str
    profile_path: Path
    start_state_path: Path
    start_state_source: str
    forward: str
    frame_skip: int
    run_dir: Path
    solve_minutes: float
    findings: dict = field(repr=False, default_factory=dict)
    #: The `solve:` mapping as it was actually drafted into the profile.
    #: Carried on the report so a reader can check what the solver will be
    #: handed without re-parsing the YAML the tool just wrote.
    solve: dict = field(repr=False, default_factory=dict)
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    solve_command: Optional[tuple[str, ...]] = None
    receipt_path: Optional[Path] = None
    classify_fn: Callable[..., dict] = field(repr=False, default=None)  # type: ignore[assignment]
    refine_fn: Callable[..., RefinementResult] = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def ready(self) -> bool:
        """True when the draft is fit to hand to the solver as-is."""
        return not self.blockers and self.solve_command is not None

    @property
    def solve_command_str(self) -> str:
        import shlex
        return " ".join(shlex.quote(a) for a in (self.solve_command or ()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rom": self.rom, "slug": self.slug,
            "profile_path": _rel(self.profile_path),
            "start_state": {"path": _rel(self.start_state_path),
                            "source": self.start_state_source},
            "forward": self.forward, "frame_skip": self.frame_skip,
            "run_dir": _rel(self.run_dir),
            "ready": self.ready,
            "blockers": list(self.blockers), "notes": list(self.notes),
            "solve_command": list(self.solve_command) if self.solve_command else None,
            "solve_minutes": self.solve_minutes,
            "solve": self.solve,
            "findings": self.findings,
        }


def _forward_score(findings: Mapping[str, Any]) -> tuple[int, int, int]:
    """Rank one direction's discovery result. Higher is better.

    A true spatial spine beats a room-counter fallback beats nothing;
    ties break on how many candidates the gates actually adjudicated,
    then on how far the recommended byte travelled.
    """
    prog = findings.get("progress") or {}
    rec = prog.get("recommended") or {}
    cands = prog.get("candidates") or []
    if not rec:
        tier = 0
    elif rec.get("kind") == "room_counter_fallback":
        tier = 1
    else:
        tier = 2
    adjudicated = sum(1 for c in cands if not c.get("saturates"))
    travel = 0
    for c in cands:
        if c.get("lo") == rec.get("lo"):
            travel = int(c.get("net_forward", 0) or 0)
            break
    return tier, adjudicated, travel


def onboard(rom: str | Path, *,
            start_state: Optional[str | Path] = None,
            forward: str = "right",
            frame_skip: int = DEFAULT_FRAME_SKIP,
            seed: int = 1,
            out_dir: str | Path = ONBOARD_DIR,
            runs_dir: str | Path = RUNS_DIR,
            minutes: Optional[float] = None,
            workers: int = DEFAULT_WORKERS,
            hw_flags: Sequence[str] = (),
            python: str = DEFAULT_PYTHON,
            force_state: bool = False,
            write: bool = True,
            write_receipt: bool = True,
            discover: Optional[Callable[..., dict]] = None,
            emit: Optional[Callable[..., str]] = None,
            ensure_state: Optional[Callable[..., tuple[Path, str]]] = None,
            header_fn: Optional[Callable[[Path], dict]] = None,
            ) -> OnboardReport:
    """Onboard one ROM: start state -> probes -> draft profile -> command.

    Every step that touches an emulator is injectable (`discover`,
    `emit`, `ensure_state`, `header_fn`) so the wiring can be tested
    without a ROM. Defaults resolve lazily to the real implementations in
    `scripts/discover_observables.py`, `scripts/capture_start_state.py`
    and `scripts/gen_auto_configs.py` — this module reuses that
    machinery and reimplements none of it.

    Deliberately does NOT run a solve: the calibration attempt is the
    orchestrator's job, and `solve_command` / `classify_fn` are how it is
    handed off and read back.
    """
    rom = Path(rom)
    if not rom.exists():
        raise FileNotFoundError(f"ROM not found: {rom}")
    slug = slug_for(rom)
    out_dir = Path(out_dir)
    run_dir = Path(runs_dir) / slug
    calib_dir = run_dir / "calib"
    notes: list[str] = []
    blockers: list[str] = []

    ensure_state = ensure_state or ensure_start_state
    if start_state is not None:
        state_path, state_source = Path(start_state), "supplied"
        if not state_path.exists():
            raise FileNotFoundError(f"start state not found: {state_path}")
    else:
        state_path, state_source = ensure_state(rom, force=force_state)
    if state_source == "generated":
        notes.append(f"captured a live-gameplay start state -> {_rel(state_path)}")
    elif state_source == "existing":
        notes.append(f"reused the banked start state {_rel(state_path)} "
                     f"(--force-state re-captures)")

    # When the probe machinery is injected (tests, or a caller with its
    # own rollout engine) the real module is never imported, so the
    # budget stamp is simply omitted rather than dragging the emulator in.
    stamp_budget = discover is None
    discover = discover or (lambda *a, **k: _discover().discover_all(*a, **k))
    emit = emit or (lambda *a, **k: _discover().emit_solve_yaml(*a, **k))

    if forward == "auto":
        best, best_score, tried = None, None, []
        for direction in AUTO_FORWARD_ORDER:
            f = discover(str(rom), str(state_path), frame_skip=frame_skip,
                         forward=direction, seed=seed)
            score = _forward_score(f)
            tried.append({"forward": direction, "score": list(score)})
            if best_score is None or score > best_score:
                best, best_score, forward = f, score, direction
            if score[0] == 2:
                break
        findings = best or {}
        notes.append(f"--forward auto picked {forward!r} "
                     f"(tried {', '.join(t['forward'] for t in tried)})")
    else:
        findings = discover(str(rom), str(state_path), frame_skip=frame_skip,
                            forward=forward, seed=seed)

    prog = (findings.get("progress") or {})
    if not prog.get("recommended"):
        blockers.append(
            f"no progress signal isolated under forward={forward!r}: the "
            f"probes found nothing that both rises with travel and stays flat "
            f"while idling. Re-run with another --forward, or --forward auto.")
        # Two shapes account for most of these, and neither is visible
        # from the candidate list alone, so name them rather than leaving
        # the next reader to rediscover them.
        notes.append(
            "the clean forward/reverse drives hold a DIRECTION only, so a "
            "game whose motion needs a held action button (a throttle, a "
            "swim stroke) shows no travel on them however well it plays")
        notes.append(
            "check the start state is at a place travel is possible at all: "
            "a menu, a cutscene, or a room whose only exit is a discrete "
            "transition all read as 'nothing rises'")
    if not (findings.get("y") or []):
        notes.append("no vertical axis isolated — the cell key will be "
                     "horizontally blind (expect KEY_BLIND on the "
                     "calibration run)")
    if (findings.get("hp_lives") or {}).get("addr") is None:
        notes.append("no health/lives byte isolated — no death signal, so "
                     "lineages will never be terminated on a death")

    header = (header_fn or (lambda p: _gen_auto_configs().parse_header(p)))(rom)
    if header.get("error"):
        notes.append(f"ROM header unreadable ({header['error']}); "
                     f"rom_hashes left empty")
        header = {}

    solve_block = emit(str(rom), findings)
    profile_path = out_dir / f"{slug}.yaml"
    receipt_path = run_dir / "onboard_receipt.json"
    text = render_profile_yaml(
        rom, slug=slug, solve_block=solve_block, start_state_path=state_path,
        start_state_source=state_source, forward=forward,
        frame_skip=frame_skip, header=header,
        receipt_path=receipt_path if write_receipt else None,
        hw_flags=hw_flags,
        probe_budget=_try_probe_budget() if stamp_budget else None)

    blockers.extend(lint_profile_text(text))

    try:
        drafted_solve = dict((_yaml().safe_load(text) or {}).get("solve") or {})
    except Exception:
        drafted_solve = {}

    if not drafted_solve.get("level_key"):
        # `level_key` advancing is what the solver reads as "the stage is
        # cleared", so it can only be drafted from a stage advance the
        # probes actually WATCHED. Scripted holds do not finish levels,
        # so on most games there is nothing to see and the empty list is
        # the honest answer rather than an oversight — say so, because
        # an empty key silently means "coverage baseline, no win
        # condition" and a reader who assumes otherwise waits forever
        # for a clear that cannot fire.
        notes.append(
            "level_key empty: the probe drives observed no stage advance "
            "(scripted holds do not finish a level), so no byte has "
            "evidence behind it as a clear signal — the calibration run "
            "is what can supply one, via --refine-chain")

    if write:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(text)

    solve_minutes = float(min_calibration_minutes() if minutes is None else minutes)
    command: Optional[tuple[str, ...]] = None
    if not blockers:
        command = build_solve_command(
            profile_path, state_path, calib_dir, minutes=solve_minutes,
            workers=workers, hw_flags=hw_flags, python=python)
    else:
        notes.append("solve command withheld: the draft has blockers, and a "
                     "profile the solver would choke on is worse than none")

    def _classify(run: Optional[str | Path] = None, **kw: Any) -> dict:
        kw.setdefault("base_command", command)
        kw.setdefault("profile_path", profile_path)
        return classify_run(run or calib_dir, **kw)

    def _refine(chain: Any, **kw: Any) -> RefinementResult:
        return refine_from_tapes(profile_path, chain, **kw)

    report = OnboardReport(
        rom=str(rom), slug=slug, profile_path=profile_path,
        start_state_path=Path(state_path), start_state_source=state_source,
        forward=forward, frame_skip=frame_skip, run_dir=calib_dir,
        solve_minutes=solve_minutes, findings=findings, solve=drafted_solve,
        blockers=tuple(blockers), notes=tuple(notes), solve_command=command,
        receipt_path=receipt_path if write_receipt else None,
        classify_fn=_classify, refine_fn=_refine)

    if write and write_receipt:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({
            "tool": "scripts/onboard_game.py",
            "purpose": "draft a solve profile for one ROM from our own probes",
            "provenance": "scripts/discover_observables.py probe protocol on "
                          "this ROM; no outside sources consulted",
            "generated": _dt.date.today().isoformat(),
            **report.as_dict(),
        }, indent=2, default=_json_default) + "\n")
    return report


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _print_report(report: OnboardReport) -> None:
    print(f"\n=== onboard: {Path(report.rom).name} ===")
    print(f"[slug]          {report.slug}")
    print(f"[start state]   {_rel(report.start_state_path)} "
          f"({report.start_state_source})")
    print(f"[profile]       {_rel(report.profile_path)}")
    if report.receipt_path:
        print(f"[receipt]       {_rel(report.receipt_path)}")
    for n in report.notes:
        print(f"[note]          {n}")
    for b in report.blockers:
        print(f"[BLOCKER]       {b}")
    print(f"[ready]         {report.ready}")
    if report.solve_command:
        print(f"\n--- bounded calibration solve (~{report.solve_minutes:g} min, "
              f"run by the orchestrator) ---")
        print(report.solve_command_str)
        print(f"\n--- then, for the wall verdict + next arm "
              f"(--verdict-module names the classifier) ---")
        print(f"{DEFAULT_PYTHON} {_rel(Path(__file__))} "
              f"--classify {_rel(report.run_dir)} "
              f"--profile {_rel(report.profile_path)} --verdict-module MODULE")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rom", help="Path to the .nes ROM to onboard.")
    ap.add_argument("--start-state", default=None,
                    help="Existing start state; captured beside the ROM if omitted.")
    ap.add_argument("--force-state", action="store_true",
                    help="Re-capture the start state even if one is banked.")
    ap.add_argument("--forward", default="right",
                    help="Forward-travel direction: right|left|up|down|auto.")
    ap.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=str(ONBOARD_DIR))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--minutes", type=float, default=None,
                    help=f"Calibration budget (default "
                         f"{min_calibration_minutes()} = the taxonomy's "
                         f"evidence floor plus margin).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--hw-flags", default=None, metavar="a,b,c")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--dry-run", action="store_true",
                    help="Probe and report without writing any file.")
    ap.add_argument("--classify", metavar="RUN_DIR", default=None,
                    help="Post-run hook: wall verdict + next arm for a run dir.")
    ap.add_argument("--refine-chain", default=None, metavar="CHAIN",
                    help="Solver run dir / chain JSON of SOLVED tapes; "
                         "re-drafts the progress byte via learnfun.")
    ap.add_argument("--profile", default=None,
                    help="Draft profile, for --classify and --refine-chain.")
    ap.add_argument("--verdict-module", default=None, metavar="MODULE",
                    help="Import path of the wall classifier to use with "
                         "--classify (it must expose telemetry_from_paths + "
                         "gated_wall_verdict). Named by the operator rather "
                         "than hardcoded: the calibrated discriminator is "
                         "under a no-runtime-import hold until the "
                         "self-arming decision lands. Without it the run is "
                         "reported but not classified.")
    ap.add_argument("--base-command", default=None,
                    help="The solve command that produced --classify's run "
                         "dir. Without it the next-arm command is rebuilt "
                         "from the profile's defaults, which may not match "
                         "the flags the run actually used.")
    ap.add_argument("--json", action="store_true", help="Emit JSON on stdout.")
    args = ap.parse_args(argv)

    hw_flags = tuple(f for f in (args.hw_flags or "").split(",") if f)

    if args.classify:
        import shlex
        base = shlex.split(args.base_command) if args.base_command else None
        vf = load_verdict_fn(args.verdict_module) if args.verdict_module else None
        res = classify_run(args.classify, verdict_fn=vf,
                           profile_path=args.profile, base_command=base)
        if args.json:
            print(json.dumps(res, indent=2, default=_json_default))
        elif res["verdict"] is None:
            print(f"\n=== {res['run_dir']} ===")
            print(f"[progress]  {res['progress_path']}")
            print(f"[archive]   {res['archive_path'] or 'none flushed'}")
            print(f"[verdict]   {res['classifier']}")
            return 2
        else:
            v, arm = res["verdict"], res["next_arm"]
            print(f"\n=== wall verdict: {v['wall_class'].upper()} "
                  f"({'degraded' if v['degraded'] else 'full evidence'}) ===")
            print(f"[classifier] {res['classifier']}")
            print(f"[records]   {res['records']}")
            print(f"[archive]   {res['archive_path'] or 'none flushed'}")
            for r in v["reasons"]:
                print(f"[reason]    {r}")
            if v.get("descriptor"):
                print(f"[descriptor] {v['descriptor']} — DESCRIPTIVE only: it "
                      f"says what the archive looks like, classifies nothing, "
                      f"and selects no arm")
            print(f"[remedy]    {v['remedy']}")
            print(f"\n[next arm]  {arm['arm']} ({arm['action']})")
            print(f"[why]       {arm['rationale']}")
            for e in arm["profile_edits"]:
                print(f"[edit]      {e}")
            for n in arm.get("notes", ()):
                print(f"[note]      {n}")
            if arm["command"]:
                print(f"[base cmd]  {res['base_command_source']}")
                print("\n" + " ".join(shlex.quote(a) for a in arm["command"]))
        return 0

    if args.refine_chain:
        if not args.profile:
            ap.error("--refine-chain needs --profile <draft profile>")
        res = refine_from_tapes(args.profile, args.refine_chain,
                                hw_flags=args.hw_flags,
                                write=not args.dry_run)
        print(json.dumps(res.as_dict(), indent=2, default=_json_default))
        return 0 if not res.blockers else 1

    if not args.rom:
        ap.error("--rom is required (or use --classify / --refine-chain)")

    report = onboard(
        args.rom, start_state=args.start_state, forward=args.forward,
        frame_skip=args.frame_skip, seed=args.seed, out_dir=args.out_dir,
        runs_dir=args.runs_dir, minutes=args.minutes, workers=args.workers,
        hw_flags=hw_flags, python=args.python, force_state=args.force_state,
        write=not args.dry_run, write_receipt=not args.dry_run)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=_json_default))
    else:
        _print_report(report)
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
