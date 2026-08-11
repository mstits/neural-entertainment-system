"""Tests for `scripts/onboard_game.py` — the point-at-a-game pipeline.

The emulator half (start-state capture, the four `find_*` probe passes)
belongs to `scripts/capture_start_state.py` and
`scripts/discover_observables.py`, which have their own ground-truth
self-tests. What is covered here is everything this module actually
owns, and it is all duck-typed: the probe machinery is stubbed, so the
whole file runs without a ROM, a Pool, or a start state.

Three contracts get most of the attention, because each one is a way the
pipeline could quietly hand the fleet something broken:

  * the LINT is the mechanical half of the purity line, so every rule in
    it is mutation-checked — a contaminated copy must fail, or the rule
    is decorative;
  * the SOLVE COMMAND is withheld, not emitted-with-a-warning, when the
    draft is missing a key `GenericGame` dereferences, and its budget is
    derived from the wall taxonomy's own evidence floor rather than
    guessed;
  * the NEXT ARM is exhaustive over `WallClass` — a verdict that fell
    through to "run it longer" would burn hours on a search that is not
    searching.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

from scripts.onboard_game import (
    ARM_FOR_CLASS,
    CALIBRATION_FLUSH_SECS,
    DEFAULT_ACTION_SPACE,
    EXTERNAL_PROVENANCE_TOKENS,
    NON_ADDRESS_SOLVE_KEYS,
    KNOWN_PROFILE_KEYS,
    REQUIRED_SOLVE_KEYS,
    SYSTEM_RAM_TOP,
    UNKNOWN_ARM,
    WALL_RECORD_FLOOR,
    RESUME_ARTIFACTS,
    NextArm,
    NoTelemetryVerdict,
    OnboardReport,
    _argv_value,
    _drop_argv,
    _forward_score,
    _merge_argv,
    build_solve_command,
    classify_run,
    lint_profile_text,
    load_verdict_fn,
    min_calibration_minutes,
    missing_resume_artifacts,
    next_arm,
    resume_ready,
    onboard,
    refine_from_tapes,
    render_profile_yaml,
    slug_for,
)

# The calibrated discriminator is under a "nothing outside tests and
# docs may import this module" hold, so `scripts/onboard_game.py`
# deliberately does not touch it — it takes an injected `verdict_fn` and
# mirrors two constants. `tests/` is the carve-out, which makes THIS
# file the place the mirror is pinned to the real thing. Every
# assertion below that reads `wt` exists so drift fails here instead of
# in an orchestrated campaign.
from src.training import wall_taxonomy as wt  # noqa: E402
from src.training.wall_taxonomy import (  # noqa: E402
    COVERAGE_FLOOR_CELLS, EFFORT_MIN_STEPS, MIN_RECORDS, WINDOW_RECORDS,
    WallClass,
)

#: The binding the orchestrator supplies. Built through the module's own
#: public helper from a path named here rather than in the shipped code.
TAXONOMY_MODULE = wt.__name__


def _verdict_fn(**kw):
    return load_verdict_fn(TAXONOMY_MODULE, **kw)

PROGRESS_LO, PROGRESS_HI, Y, LIVES, AREA = 0x0083, 0x0095, 0x001E, 0x0025, 0x004F


# ---------------------------------------------------------------------
# Stubs. Everything that would touch an emulator.
# ---------------------------------------------------------------------

def _fake_findings(*, recommended=True, y=True, hp=True, room=AREA,
                   lives=LIVES) -> dict:
    rec = ({"lo": PROGRESS_LO, "hi": PROGRESS_HI, "kind": "pair",
            "label": "$0083|$0095", "saturates": False,
            "as_progress": "$0083|$0095<<8"} if recommended else None)
    return {
        "rom": "roms/Fake Game (USA).nes",
        "state": "roms/Fake Game (USA)_start.state.bin",
        "forward": "right",
        "progress": {
            "forward": "right", "room_counter": room,
            "candidates": [{"lo": PROGRESS_LO, "hi": PROGRESS_HI,
                            "kind": "pair", "label": "$0083|$0095",
                            "saturates": False, "net_forward": 900,
                            "mono_forward": 0.99, "wrap_coupled": 1,
                            "reversible_bonus": True}],
            "recommended": rec,
        },
        "room_counters": ([{"addr": room}] if room is not None else []),
        "y": ([{"addr": Y}] if y else []),
        "hp_lives": ({"addr": lives, "kind": "lives", "value": 4,
                      "note": "decrements on death"} if hp
                     else {"addr": None, "kind": "none", "note": "none"}),
    }


def _fake_solve_block(rom, findings) -> str:
    """Stand-in for `discover_observables.emit_solve_yaml`.

    Mirrors the real emitter's SHAPE, including the property that
    matters most here: a key is simply absent when discovery isolated
    nothing for it.
    """
    prog = findings["progress"]
    rec = prog["recommended"]
    out = ["solve:", '  rom: "roms/Fake Game (USA).nes"']
    if rec:
        out.append(f"  progress: {{lo: 0x{rec['lo']:04X}, hi: 0x{rec['hi']:04X}}}")
    if prog.get("room_counter") is not None:
        out.append(f"  area: 0x{prog['room_counter']:04X}")
    if findings["y"]:
        out.append(f"  y: 0x{findings['y'][0]['addr']:04X}")
    out.append("  level_key: []")
    if findings["hp_lives"].get("addr") is not None:
        out.append(f"  lives: 0x{findings['hp_lives']['addr']:04X}")
    return "\n".join(out) + "\n"


def _fake_header(_rom) -> dict:
    return {"md5": "0" * 32, "mapper": 4, "is_nes20": False, "battery": False,
            "prg_size_kb": 512, "chr_size_kb": 256}


@pytest.fixture()
def rom(tmp_path):
    p = tmp_path / "roms" / "Fake Game (USA).nes"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"NES\x1a" + bytes(12) + bytes(64))
    return p


@pytest.fixture()
def state(rom):
    p = rom.with_name(rom.stem + "_start.state.bin")
    p.write_bytes(b"\x00" * 32)
    return p


def _run(rom, tmp_path, *, findings=None, emit=_fake_solve_block, **kw):
    calls: dict = {}

    def discover(rom_arg, state_arg, **dk):
        calls["discover"] = (rom_arg, state_arg, dk)
        f = findings or _fake_findings()
        f = dict(f)
        f["forward"] = dk.get("forward", "right")
        f["progress"] = dict(f["progress"], forward=f["forward"])
        return f

    def ensure_state(rom_arg, *, force=False, **_):
        calls["ensure_state"] = (rom_arg, force)
        p = rom.with_name(rom.stem + "_start.state.bin")
        if not p.exists():
            p.write_bytes(b"\x00" * 32)
            return p, "generated"
        return p, "existing"

    report = onboard(rom, out_dir=tmp_path / "configs" / "onboard",
                     runs_dir=tmp_path / "runs" / "onboard",
                     discover=discover, emit=emit, ensure_state=ensure_state,
                     header_fn=_fake_header, **kw)
    return report, calls


# ---------------------------------------------------------------------
# Slug + paths.
# ---------------------------------------------------------------------

def test_slug_matches_the_configs_auto_convention():
    """The slug rule is imported from `gen_auto_configs`, not reinvented,
    so an onboarded game files itself the way the 793-ROM sweep already
    does."""
    from scripts.gen_auto_configs import slugify
    name = "Kirby's Adventure (USA) (Rev A).nes"
    assert slug_for(f"roms/{name}") == slugify(name)
    assert slug_for("roms/Fake Game (USA).nes") == "fake_game__usa_"


# ---------------------------------------------------------------------
# The happy path.
# ---------------------------------------------------------------------

def test_onboard_writes_a_lint_clean_draft_and_a_receipt(rom, state, tmp_path):
    report, calls = _run(rom, tmp_path)

    assert report.ready
    assert report.blockers == ()
    assert report.profile_path.exists()
    assert report.profile_path.name == "fake_game__usa_.yaml"
    assert lint_profile_text(report.profile_path.read_text()) == ()

    prof = yaml.safe_load(report.profile_path.read_text())
    assert prof["solve"]["progress"] == {"lo": PROGRESS_LO, "hi": PROGRESS_HI}
    assert prof["solve"]["y"] == Y
    assert prof["solve"]["lives"] == LIVES
    assert prof["solve"]["level_key"] == []
    assert prof["start_state_path"].endswith("_start.state.bin")
    assert prof["frame_skip"] == 4
    assert prof["rom_hashes"] == ["0" * 32]
    assert prof["cartridge"]["mapper"] == 4

    receipt = json.loads(report.receipt_path.read_text())
    assert receipt["tool"] == "scripts/onboard_game.py"
    assert receipt["ready"] is True
    assert receipt["solve_command"] == list(report.solve_command)

    # The probes were driven from the start state, not from a cold boot.
    assert calls["discover"][1] == str(state)


def test_a_banked_start_state_is_reused_and_force_recaptures(rom, state, tmp_path):
    report, calls = _run(rom, tmp_path)
    assert report.start_state_source == "existing"
    assert calls["ensure_state"][1] is False
    assert any("reused" in n for n in report.notes)

    report2, calls2 = _run(rom, tmp_path, force_state=True)
    assert calls2["ensure_state"][1] is True


def test_a_supplied_start_state_skips_capture_entirely(rom, tmp_path):
    supplied = tmp_path / "elsewhere.state.bin"
    supplied.write_bytes(b"\x01" * 16)
    report, calls = _run(rom, tmp_path, start_state=supplied)
    assert report.start_state_source == "supplied"
    assert "ensure_state" not in calls
    assert calls["discover"][1] == str(supplied)


def test_a_missing_start_state_is_captured_and_banked_beside_the_rom(rom, tmp_path):
    report, calls = _run(rom, tmp_path)
    assert report.start_state_source == "generated"
    assert report.start_state_path == rom.with_name(rom.stem + "_start.state.bin")
    assert report.start_state_path.exists()
    assert any("captured" in n for n in report.notes)


def test_dry_run_writes_nothing(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path, write=False)
    assert not report.profile_path.exists()
    assert not (tmp_path / "runs").exists()
    # ...but the command is still produced, so a caller can preview it.
    assert report.solve_command is not None


def test_the_draft_action_space_is_the_pad_not_a_route(rom, state, tmp_path):
    """A generic controller space is not game knowledge; a hand-authored
    button sequence would be. START is deliberately absent — it pauses."""
    report, _ = _run(rom, tmp_path)
    space = yaml.safe_load(report.profile_path.read_text())["action_space"]
    assert space[0] == []
    assert ["right"] in space and ["left"] in space and ["up"] in space
    assert not any("start" in [b.lower() for b in combo or []] for combo in space)
    assert len(space) == len(DEFAULT_ACTION_SPACE)


# ---------------------------------------------------------------------
# Blockers: a draft the solver would choke on is worse than none.
# ---------------------------------------------------------------------

def test_no_progress_signal_blocks_the_command_and_names_the_retry(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path, findings=_fake_findings(recommended=False))
    assert not report.ready
    assert report.solve_command is None
    assert any("no progress signal isolated" in b for b in report.blockers)
    assert any("--forward" in b for b in report.blockers)
    # The two shapes that account for most of these are named, because
    # neither is visible from the candidate list alone: a game whose
    # motion needs a held action button (Excitebike's throttle) and a
    # start state parked somewhere travel is impossible.
    assert any("held action button" in n for n in report.notes)
    assert any("start state is at a place travel is possible" in n
               for n in report.notes)


@pytest.mark.parametrize("missing", ["y", "lives"])
def test_a_missing_generic_game_key_withholds_the_command(rom, state, tmp_path, missing):
    """`GenericGame.__init__` dereferences `s["y"]` and `s["lives"]`
    unconditionally, so shipping a draft without them would fail at
    solver startup — minutes into an orchestrated run, not here."""
    kw = {"y": False} if missing == "y" else {"hp": False}
    report, _ = _run(rom, tmp_path, findings=_fake_findings(**kw))
    assert report.solve_command is None
    assert any(f"solve.{missing} missing" in b for b in report.blockers)
    assert any("withheld" in n for n in report.notes)


def test_missing_observables_are_noted_even_when_they_do_not_block(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path, findings=_fake_findings(y=False, hp=False))
    assert any("vertical axis" in n for n in report.notes)
    assert any("death signal" in n for n in report.notes)


# ---------------------------------------------------------------------
# The lint. Every rule mutation-checked.
# ---------------------------------------------------------------------

def _clean_profile(rom, state, tmp_path) -> str:
    report, _ = _run(rom, tmp_path)
    return report.profile_path.read_text()


def test_lint_passes_the_generated_draft(rom, state, tmp_path):
    assert lint_profile_text(_clean_profile(rom, state, tmp_path)) == ()


def test_lint_rejects_an_unreadable_prg_ram_address(rom, state, tmp_path):
    """$6879-class addresses are both unreadable (get_ram masks to $07FF)
    and the classic tell of an address copied in from outside rather than
    measured here — the exact material the Metroid quarantine isolates."""
    text = _clean_profile(rom, state, tmp_path).replace(
        "level_key: []", "level_key: [0x6879]")
    problems = lint_profile_text(text)
    assert any("0x6879" in p and "system RAM" in p for p in problems)


def test_lint_leaves_value_keys_alone(rom, state, tmp_path):
    """`progress_cap` is a VALUE ceiling, not an address: Metroid pins it
    at 16384 and Kirby at 4000. Reading it as an address would flag every
    real profile in the tree."""
    assert "progress_cap" in NON_ADDRESS_SOLVE_KEYS
    text = _clean_profile(rom, state, tmp_path) + "  progress_cap: 16384\n"
    assert lint_profile_text(text) == ()


# The real key name may appear HERE: the quarantine guard scans src /
# scripts / nes_core only, and a mutation check is worthless if it
# cannot name the thing it is checking for.
QUARANTINE_KEY = "quarantined_external_knowledge"


def test_lint_rejects_quarantined_material_pasted_into_a_draft(rom, state, tmp_path):
    """`scripts/onboard_game.py` may not name the quarantine key — the
    Metroid guard bars any source file from mentioning it. So the lint
    rejects it structurally, as an unrecognized top-level block, which
    also catches every other smuggled-in table."""
    assert QUARANTINE_KEY not in KNOWN_PROFILE_KEYS
    text = (_clean_profile(rom, state, tmp_path)
            + f"\n{QUARANTINE_KEY}:\n  q_item_flags: \"0x6878 UNVERIFIED\"\n")
    problems = lint_profile_text(text)
    assert any(QUARANTINE_KEY in p and "unrecognized top-level key" in p
               for p in problems)


def test_lint_rejects_any_unrecognized_top_level_block(rom, state, tmp_path):
    text = _clean_profile(rom, state, tmp_path) + "\nram_mapping:\n  foo: 0x0010\n"
    assert any("ram_mapping" in p for p in lint_profile_text(text))


def test_known_profile_keys_covers_what_the_renderer_actually_writes(rom, state, tmp_path):
    """If the renderer grows a key and the allowlist does not, every
    generated draft starts failing its own lint."""
    written = set(yaml.safe_load(_clean_profile(rom, state, tmp_path)))
    assert written <= KNOWN_PROFILE_KEYS


@pytest.mark.parametrize("token", EXTERNAL_PROVENANCE_TOKENS)
def test_lint_rejects_every_outside_provenance_token(rom, state, tmp_path, token):
    text = _clean_profile(rom, state, tmp_path).replace(
        "level_key: []", f"level_key: []   # per a {token} source")
    assert any(token in p for p in lint_profile_text(text))


def test_the_generated_draft_trips_none_of_its_own_tokens(rom, state, tmp_path):
    """The lint is only useful if the tool's own prose stays clear of it;
    otherwise the first thing anyone does is widen the rule."""
    low = _clean_profile(rom, state, tmp_path).lower()
    assert [t for t in EXTERNAL_PROVENANCE_TOKENS if t in low] == []


def test_lint_requires_the_provenance_stamp(rom, state, tmp_path):
    prof = yaml.safe_load(_clean_profile(rom, state, tmp_path))
    prof.pop("provenance")
    assert any("provenance" in p for p in lint_profile_text(yaml.safe_dump(prof)))

    prof2 = yaml.safe_load(_clean_profile(rom, state, tmp_path))
    prof2["provenance"]["protocol"] = "somebody wrote it down"
    assert any("probe protocol" in p for p in lint_profile_text(yaml.safe_dump(prof2)))


def test_lint_rejects_unparseable_and_non_mapping_input():
    assert lint_profile_text("solve: [") [0].startswith("profile does not parse")
    assert lint_profile_text("- a\n- b") == ("profile is not a YAML mapping",)


def test_lint_names_every_required_solve_key():
    """The rule set must actually cover what `GenericGame` reads; an
    empty solve: block should produce one complaint per required key."""
    text = render_profile_yaml(
        "roms/X.nes", slug="x", solve_block="solve:\n  level_key: []\n",
        start_state_path="roms/X_start.state.bin", start_state_source="supplied",
        forward="right")
    problems = " | ".join(lint_profile_text(text))
    for key in REQUIRED_SOLVE_KEYS:
        if key == "level_key":
            continue
        assert f"solve.{key} missing" in problems


# ---------------------------------------------------------------------
# The bounded calibration command.
# ---------------------------------------------------------------------

def test_the_default_budget_is_derived_from_the_taxonomys_evidence_floor():
    """`gated_wall_verdict` refuses to classify below its record floor,
    and the solver writes one progress line a minute. A budget shorter
    than that cannot return anything but INSUFFICIENT, however healthy
    the search was."""
    floor = max(MIN_RECORDS, WINDOW_RECORDS + 1)
    assert min_calibration_minutes() > floor
    cmd = build_solve_command("configs/onboard/x.yaml", "roms/x.state.bin",
                              "runs/onboard/x/calib")
    assert float(_argv_value(cmd, "--minutes")) >= floor


def test_the_command_flushes_often_enough_that_an_archive_lands():
    """Without an `archive.pkl` the taxonomy cannot separate GATED from
    COVERAGE_LIMITED and abstains — which is exactly what happened to the
    two banked show runs, whose flush interval was set to ~forever."""
    cmd = build_solve_command("p.yaml", "s.bin", "out")
    flush = float(_argv_value(cmd, "--flush-secs"))
    minutes = float(_argv_value(cmd, "--minutes"))
    assert flush == CALIBRATION_FLUSH_SECS
    assert flush <= minutes * 60 / 2


def test_the_command_targets_the_real_solver_with_repo_relative_paths(rom, state, tmp_path):
    cmd = build_solve_command("configs/onboard/x.yaml", "roms/x.state.bin",
                              "runs/onboard/x/calib", workers=4, seed=7,
                              hw_flags=("mmio_read_timing",))
    assert cmd[1] == "scripts/go_explore_solve.py"
    assert _argv_value(cmd, "--profile") == "configs/onboard/x.yaml"
    assert _argv_value(cmd, "--root-state") == "roms/x.state.bin"
    assert _argv_value(cmd, "--out") == "runs/onboard/x/calib"
    assert _argv_value(cmd, "--workers") == "4"
    assert _argv_value(cmd, "--seed") == "7"
    assert _argv_value(cmd, "--hw-flags") == "mmio_read_timing"


def test_the_command_roots_the_solve_at_the_start_state(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    assert _argv_value(report.solve_command, "--root-state") == str(state)
    assert _argv_value(report.solve_command, "--profile") == str(report.profile_path)
    assert "go_explore_solve.py" in report.solve_command_str


def test_every_flag_this_module_emits_exists_on_the_real_solver():
    """The commands here are handed straight to an orchestrator, so a
    flag that the solver does not define fails minutes into a run rather
    than here. Scanned from the solver's source (never imported — that
    module owns a large import graph) so a renamed or dropped option
    breaks this test instead of a campaign."""
    src = (REPO / "scripts" / "go_explore_solve.py").read_text()
    defined = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    assert "--out" in defined, "flag scan found nothing — the regex has rotted"

    emitted = set()
    emitted.update(f for f in build_solve_command(
        "p.yaml", "s.bin", "out", hw_flags=("x",)) if f.startswith("--"))
    for cls in WallClass:
        arm = next_arm(_verdict(cls), base_command=BASE, run_dir="runs/x")
        emitted.update(f for f in (arm.command or ()) if f.startswith("--"))
    assert emitted <= defined, f"unknown solver flags: {sorted(emitted - defined)}"


def test_merge_argv_replaces_rather_than_duplicating():
    base = ("py", "s.py", "--minutes", "15", "--out", "a")
    assert _merge_argv(base, [("--minutes", "30")]) == (
        "py", "s.py", "--minutes", "30", "--out", "a")
    assert _merge_argv(base, [("--ortho", "up")])[-2:] == ("--ortho", "up")
    # A store-true flag carries no value and must not eat the next token.
    assert _merge_argv(base, [("--time-bins", None)])[-1] == "--time-bins"


# ---------------------------------------------------------------------
# Next arm.
# ---------------------------------------------------------------------

def _verdict(cls: WallClass, **kw):
    """A REAL verdict object from the calibrated discriminator, so the
    duck-typed arm table is exercised against the actual shape rather
    than a hand-rolled look-alike."""
    return wt.WallVerdict(
        wall_class=cls, label="t", degraded=kw.pop("degraded", True),
        missing=kw.pop("missing", ()), reasons=kw.pop("reasons", ("because",)),
        evidence=kw.pop("evidence", {}), remedy=wt.REMEDY[cls])


BASE = ("py", "scripts/go_explore_solve.py", "--out", "runs/onboard/x/calib",
        "--root-state", "s.bin", "--profile", "p.yaml", "--minutes", "15",
        "--flush-secs", "60")


def _snapshot(run_dir: Path, *, omit: str = "") -> Path:
    """A run dir the solver's `--resume-archive` can actually open.

    The three files are exactly what `go_explore_solve.py` opens on the
    resume path with no existence check, so `omit=` models a run killed
    mid-flush (they are written sequentially, not atomically as a set).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RESUME_ARTIFACTS:
        if name == omit:
            continue
        if name == "archive.pkl":
            _write_archive(run_dir / name)
        elif name.endswith(".pkl"):
            (run_dir / name).write_bytes(pickle.dumps({}))
        else:
            (run_dir / name).write_text("{}")
    return run_dir


@pytest.mark.parametrize("cls", list(WallClass))
def test_every_wall_class_has_an_arm_and_a_coherent_action(cls):
    """Exhaustive on purpose: a new verdict label falling through to
    'run it longer' would spend hours on, say, a BARREN search that is
    not searching at all. This is also the drift pin — a label added to
    the discriminator fails here until the arm table names it."""
    assert cls.value in ARM_FOR_CLASS
    arm = next_arm(_verdict(cls), base_command=BASE, run_dir="runs/onboard/x/calib")
    assert arm.arm == ARM_FOR_CLASS[cls.value]
    assert arm.action in ("run", "stop", "fix")
    assert arm.rationale
    assert arm.calibration == wt.CALIBRATION_TAG
    assert (arm.command is not None) == (arm.action == "run")


def test_the_arm_table_matches_the_discriminators_labels_exactly():
    assert set(ARM_FOR_CLASS) == {c.value for c in WallClass}


def test_the_mirrored_record_floor_matches_the_real_one():
    """`scripts/onboard_game.py` may not import the discriminator, so it
    mirrors this number. The mirror is only safe if something pins it."""
    assert WALL_RECORD_FLOOR == max(MIN_RECORDS, WINDOW_RECORDS + 1)


def test_an_unknown_verdict_label_stops_instead_of_guessing():
    """Every wrong guess here spends orchestrator hours, and half the
    known labels want something other than 'run it longer'."""
    arm = next_arm(NoTelemetryVerdict(wall_class="brand_new_label",
                                      reasons=("something novel",)),
                   base_command=BASE)
    assert arm.arm == UNKNOWN_ARM
    assert arm.action == "fix"
    assert arm.command is None
    assert "brand_new_label" in arm.rationale


def test_next_arm_accepts_a_plain_duck_typed_verdict():
    """The whole point of not importing the discriminator: anything
    exposing the documented attributes classifies."""
    duck = types.SimpleNamespace(wall_class="gated", reasons=("r",),
                                 remedy="switch", missing=(), calibration="tag")
    arm = next_arm(duck, base_command=BASE)
    assert arm.arm == "interaction" and arm.calibration == "tag"


def test_a_resolved_run_is_harvested_not_extended():
    arm = next_arm(_verdict(WallClass.RESOLVED), base_command=BASE)
    assert (arm.arm, arm.action, arm.command) == ("harvest", "stop", None)


def test_a_barren_run_buys_no_more_compute():
    """The one verdict where more wall clock cannot help: the archive
    never accumulated, so the search — start state, cell key, determinism
    — is what is broken."""
    arm = next_arm(_verdict(WallClass.BARREN), base_command=BASE)
    assert arm.action == "fix" and arm.command is None
    assert any("start state" in e for e in arm.profile_edits)
    assert any("determinism" in e for e in arm.profile_edits)


def test_a_progressing_run_is_extended_on_the_same_arm_and_resumes(tmp_path):
    """Resuming is only correct when the dir it names is openable, so
    the run this extends has a real flushed snapshot."""
    run = _snapshot(tmp_path / "calib")
    arm = next_arm(_verdict(WallClass.PROGRESSING), base_command=BASE,
                   run_dir=run)
    assert float(_argv_value(arm.command, "--minutes")) == 30
    assert _argv_value(arm.command, "--resume-archive") == str(run)
    assert _argv_value(arm.command, "--out") == "runs/onboard/x/calib_extend"
    assert "--ortho" not in arm.command


def test_a_gated_run_leads_with_interaction_and_caveats_the_position_arm():
    """A gated wall is opened by causing a STATE CHANGE, not by finding
    another position. The position-orthogonal arm has been A/B'd against
    its own control at a gated wall and did not move the frontier, so it
    ships as a caveated fallback while the interaction edits lead."""
    arm = next_arm(_verdict(WallClass.GATED), base_command=BASE,
                   run_dir="runs/onboard/x/calib",
                   profile={"solve": {"area": AREA}})
    assert arm.arm == "interaction"
    assert arm.profile_edits[0].startswith("FIRST:")
    assert "hold_macros" in arm.profile_edits[0]
    assert any("room_advance" in e for e in arm.profile_edits)
    assert any(f"0x{AREA:04X}" in e for e in arm.profile_edits)
    # The fallback flag is still emitted, and still labelled a fallback.
    assert _argv_value(arm.command, "--ortho") == "up"
    assert _argv_value(arm.command, "--sel-mode") == "count"
    assert "fallback" in arm.rationale


def test_a_gated_run_without_an_area_byte_says_so():
    arm = next_arm(_verdict(WallClass.GATED), base_command=BASE, profile={"solve": {}})
    assert any("no solve.area" in e for e in arm.profile_edits)


def test_a_key_blind_run_shrinks_the_buckets_and_names_the_missing_axis():
    arm = next_arm(_verdict(WallClass.KEY_BLIND), base_command=BASE,
                   profile={"solve": {"area": AREA}})
    assert _argv_value(arm.command, "--gx-bucket") == "8"
    assert _argv_value(arm.command, "--y-band") == "16"
    assert any("solve.y is absent" in e for e in arm.profile_edits)


def test_an_indeterminate_run_is_re_instrumented_not_just_extended():
    arm = next_arm(_verdict(WallClass.INDETERMINATE,
                            missing=("frontier_bucket_cells",)),
                   base_command=BASE)
    assert _argv_value(arm.command, "--flush-secs") == str(CALIBRATION_FLUSH_SECS)
    assert "frontier_bucket_cells" in arm.rationale


def test_an_insufficient_run_is_lifted_to_at_least_the_floor():
    arm = next_arm(_verdict(WallClass.INSUFFICIENT), base_command=("py", "s.py",
                                                                   "--minutes", "2"))
    assert float(_argv_value(arm.command, "--minutes")) >= min_calibration_minutes()


def test_next_arm_without_a_base_command_still_advises():
    arm = next_arm(_verdict(WallClass.GATED))
    assert arm.command is None and arm.arm == "interaction" and arm.rationale


# ---------------------------------------------------------------------
# --resume-archive. The solver opens DIR/archive.pkl, DIR/traces.pkl and
# DIR/roots.json with no existence check, so naming a dir that never
# flushed is not a degraded resume — it is FileNotFoundError at startup,
# before a single step, and the whole budgeted run is lost.
# ---------------------------------------------------------------------

def test_resume_artifacts_are_what_the_solver_actually_opens():
    """The drift pin, and it is an EQUALITY on purpose.

    A superset would make the guard reject resumable dirs; a subset
    would let a dir through that the solver still cannot open, which is
    the bug this whole section exists for. So the set is read off the
    solver's resume block and compared both ways.
    """
    src = (REPO / "scripts" / "go_explore_solve.py").read_text()
    start = src.index('prev = getattr(self.args, "resume_archive"')
    block = src[start:src.index("[seed] RESUMED", start)]
    opened = set(re.findall(r'prev\s*/\s*"([^"]+)"', block))
    assert opened, "resume block no longer reads files off `prev` — regex rotted"
    assert opened == set(RESUME_ARTIFACTS)
    # And it still opens them unguarded, which is the reason for all of
    # this: an `exists()` check landing upstream would relax the rule.
    assert ".exists()" not in block


@pytest.mark.parametrize("omit", RESUME_ARTIFACTS)
def test_a_partial_snapshot_is_not_a_resume_point(tmp_path, omit):
    """The three files are flushed sequentially, not atomically, so a
    run killed mid-flush leaves exactly this shape."""
    run = _snapshot(tmp_path / "calib", omit=omit)
    assert missing_resume_artifacts(run) == (omit,)
    assert not resume_ready(run)
    arm = next_arm(_verdict(WallClass.PROGRESSING), base_command=BASE,
                   run_dir=run)
    assert "--resume-archive" not in arm.command
    assert any(omit in n for n in arm.notes)


def test_a_run_that_never_flushed_is_never_resumed(tmp_path):
    """The reported blocker, at the arm level: a run dir holding only
    progress.jsonl must not be handed to --resume-archive."""
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    assert missing_resume_artifacts(run) == RESUME_ARTIFACTS
    for cls in WallClass:
        arm = next_arm(_verdict(cls), base_command=BASE, run_dir=run)
        if arm.command:
            assert "--resume-archive" not in arm.command, cls


def test_an_indeterminate_run_never_resumes_even_with_a_snapshot(tmp_path):
    """INDETERMINATE can also be reached WITH an archive (a
    concentration that contradicts the C_local series). Resuming would
    still be wrong: the arm re-instruments from the root so the next
    verdict is not confounded by the last run's archive."""
    run = _snapshot(tmp_path / "calib")
    assert resume_ready(run)
    arm = next_arm(_verdict(WallClass.INDETERMINATE), base_command=BASE,
                   run_dir=run)
    assert "--resume-archive" not in arm.command
    assert any("NOT resuming" in n for n in arm.notes)
    assert "FROM THE ROOT" in arm.rationale


def test_an_inherited_resume_archive_is_dropped_when_it_cannot_be_opened(tmp_path):
    """A chained arm carries the previous generation's
    --resume-archive in its base command. That flag is not inert."""
    dead = tmp_path / "gen0"
    dead.mkdir()
    base = BASE + ("--resume-archive", str(dead))
    arm = next_arm(_verdict(WallClass.PROGRESSING), base_command=base,
                   run_dir=tmp_path / "gen1")
    assert "--resume-archive" not in arm.command
    assert any("dropped the inherited" in n for n in arm.notes)


def test_an_inherited_resume_archive_survives_when_it_is_still_complete(tmp_path):
    """Not over-broad: a crashed generation should fall back to the last
    dir that DID flush rather than throwing the frontier away."""
    good = _snapshot(tmp_path / "gen0")
    base = BASE + ("--resume-archive", str(good))
    arm = next_arm(_verdict(WallClass.PROGRESSING), base_command=base,
                   run_dir=tmp_path / "gen1_crashed")
    assert _argv_value(arm.command, "--resume-archive") == str(good)


def test_an_inherited_resume_archive_is_dropped_by_the_indeterminate_arm(tmp_path):
    good = _snapshot(tmp_path / "gen0")
    base = BASE + ("--resume-archive", str(good))
    arm = next_arm(_verdict(WallClass.INDETERMINATE), base_command=base,
                   run_dir=tmp_path / "gen1")
    assert "--resume-archive" not in arm.command


def test_a_resume_dir_is_resolved_against_the_repo_when_relative(tmp_path):
    """The emitted commands carry repo-relative paths, so the existence
    check has to agree with them from any cwd."""
    assert missing_resume_artifacts("runs/onboard/definitely-not-a-run") \
        == RESUME_ARTIFACTS
    assert not resume_ready(None)
    assert resume_ready(_snapshot(tmp_path / "abs"))


def test_drop_argv_removes_the_option_and_its_value():
    assert _drop_argv(("py", "--a", "1", "--b", "2"), "--b") == ("py", "--a", "1")
    assert _drop_argv(("py", "--b", "2", "--a", "1"), "--b") == ("py", "--a", "1")
    # A valueless trailing occurrence, and every occurrence.
    assert _drop_argv(("py", "--b"), "--b") == ("py",)
    assert _drop_argv(("py", "--b", "1", "--a", "--b", "2"), "--b") == ("py", "--a")
    assert _drop_argv(("py", "--a", "1"), "--b") == ("py", "--a", "1")


def test_classify_does_not_emit_a_resume_the_solver_cannot_open(tmp_path):
    """End to end through the hook, on the reported repro: 13 plateau
    records, no archive.pkl -> indeterminate -> a command that must not
    resume from the dir that has no snapshot."""
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["wall_class"] == "indeterminate"
    assert res["archive_path"] is None
    cmd = res["next_arm"]["command"]
    assert "--resume-archive" not in cmd
    assert _argv_value(cmd, "--flush-secs") == str(CALIBRATION_FLUSH_SECS)


def test_classify_resumes_when_the_snapshot_is_there(tmp_path):
    run = _snapshot(tmp_path / "calib")
    _write_progress(run / "progress.jsonl", _flat_records())
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["wall_class"] == "gated"
    assert _argv_value(res["next_arm"]["command"], "--resume-archive") == str(run)


# ---------------------------------------------------------------------
# classify_fn — the post-run hook.
# ---------------------------------------------------------------------

def _write_progress(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _flat_records(n=MIN_RECORDS + 1, *, cells=300, solutions=0,
                  step=EFFORT_MIN_STEPS // 5):
    return [{"elapsed_s": 60 * (i + 1), "steps": step * (i + 1),
             "cells": cells, "solutions": solutions, "max_area": 0,
             "max_sect": 0, "max_gx_in_max_area": 100}
            for i in range(n)]


def _write_archive(path, *, cells=300, buckets=12):
    """A real `archive.pkl` the taxonomy's own reader can consume.

    Cell keys are 5-tuples so `_spatial_key` reads `(area, y_band, gx)`
    off `key[-5] / key[-2] / key[-1]`; the values only need `.visits`.
    """
    archive = {(0, i, 0, 0, i % buckets): types.SimpleNamespace(
        visits=1, explored=False) for i in range(cells)}
    path.write_bytes(pickle.dumps(archive))


def test_classify_reads_a_run_dir_and_recommends(tmp_path):
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records(solutions=2))
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["wall_class"] == "resolved"
    assert res["records"] == MIN_RECORDS + 1
    assert res["next_arm"]["arm"] == "harvest"
    assert res["archive_path"] is None
    # The calibration tag rides in from the INJECTED discriminator, which
    # is the only place it may come from — this module holds no copy.
    assert res["verdict"]["calibration"] == wt.CALIBRATION_TAG
    assert res["next_arm"]["calibration"] == wt.CALIBRATION_TAG
    assert res["classifier"] == TAXONOMY_MODULE


def test_classify_is_insufficient_below_the_record_floor(tmp_path):
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records(n=4))
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["wall_class"] == "insufficient"
    assert res["next_arm"]["action"] == "run"


def test_classify_calls_a_frozen_archive_barren(tmp_path):
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl",
                    _flat_records(cells=COVERAGE_FLOOR_CELLS - 1))
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["wall_class"] == "barren"
    assert res["next_arm"]["command"] is None


def test_classify_uses_the_archive_snapshot_when_one_was_flushed(tmp_path):
    """300 cells over 12 spatial buckets = concentration 25, the
    calibrated GATED threshold: coverage plateaued while the key keeps
    manufacturing novelty."""
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    _write_archive(run / "archive.pkl")
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["archive_path"] is not None
    assert res["verdict"]["evidence"]["concentration"] == 25.0
    assert res["wall_class"] == "gated"
    assert "--ortho" in res["next_arm"]["command"]


def test_classify_reads_the_profile_to_sharpen_its_advice(tmp_path):
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    _write_archive(run / "archive.pkl")
    prof = tmp_path / "p.yaml"
    prof.write_text(yaml.safe_dump({"solve": {"area": AREA}}))
    res = classify_run(run, verdict_fn=_verdict_fn(), base_command=BASE, profile_path=prof)
    assert any(f"0x{AREA:04X}" in e for e in res["next_arm"]["profile_edits"])


def test_classify_rebuilds_a_base_command_from_the_profile_when_given_none(tmp_path):
    """A bare CLI call carries no command, and advice with no way to act
    on it is half a hook — so the same command the draft would have
    produced is rebuilt, and labelled as rebuilt."""
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    _write_archive(run / "archive.pkl")
    prof = tmp_path / "p.yaml"
    prof.write_text(yaml.safe_dump(
        {"solve": {"area": AREA}, "start_state_path": "roms/x.state.bin"}))

    res = classify_run(run, verdict_fn=_verdict_fn(), profile_path=prof)
    assert res["base_command_source"].startswith("reconstructed")
    assert "--ortho" in res["next_arm"]["command"]
    assert _argv_value(res["next_arm"]["command"], "--profile") == str(prof)

    res2 = classify_run(run, verdict_fn=_verdict_fn(), profile_path=prof, base_command=BASE)
    assert res2["base_command_source"] == "supplied"
    assert list(BASE) == res2["base_command"]


def test_classify_without_a_profile_advises_but_cannot_rebuild(tmp_path):
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    _write_archive(run / "archive.pkl")
    res = classify_run(run, verdict_fn=_verdict_fn())
    assert res["base_command_source"] == "none"
    assert res["wall_class"] == "gated"
    assert res["next_arm"]["command"] is None
    assert res["next_arm"]["profile_edits"]


def test_classify_without_a_classifier_reports_but_does_not_guess(tmp_path):
    """The discriminator is injected, so a caller that supplies none gets
    an honest 'not classified' rather than a made-up verdict."""
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    res = classify_run(run, base_command=BASE)
    assert res["wall_class"] is None
    assert res["verdict"] is None and res["next_arm"] is None
    assert "verdict_fn" in res["classifier"]
    assert res["progress_path"].endswith("progress.jsonl")


def test_load_verdict_fn_rejects_a_module_that_is_not_a_classifier():
    with pytest.raises(TypeError, match="telemetry_from_paths"):
        load_verdict_fn("json")


def test_load_verdict_fn_binds_the_two_function_contract(tmp_path):
    """Anything exposing telemetry_from_paths + gated_wall_verdict works,
    which is what keeps the shipped module free of the import."""
    run = tmp_path / "calib"
    _write_progress(run / "progress.jsonl", _flat_records())
    seen: dict = {}
    fake = types.ModuleType("fake_classifier")
    fake.telemetry_from_paths = lambda p, **kw: seen.setdefault("kw", kw) or "TEL"
    fake.gated_wall_verdict = lambda tel, **kw: types.SimpleNamespace(
        wall_class="progressing", reasons=("moving",), remedy="wait",
        missing=(), calibration="fake-tag", evidence={"records": 13})
    sys.modules["fake_classifier"] = fake
    try:
        res = classify_run(run, verdict_fn=load_verdict_fn("fake_classifier"),
                           base_command=BASE)
    finally:
        del sys.modules["fake_classifier"]
    assert res["wall_class"] == "progressing"
    assert res["records"] == 13
    assert res["classifier"] == "fake_classifier"
    assert res["next_arm"]["arm"] == "extend"
    # The archive path is handed through whenever one was flushed.
    assert seen["kw"]["archive_path"] is None


def test_classify_on_a_run_that_never_started(tmp_path):
    """A launch that wrote no progress line is a launch failure, not a
    slow search — recommending 'run it longer' would be nonsense."""
    res = classify_run(tmp_path / "nothing-here", verdict_fn=_verdict_fn(), base_command=BASE)
    assert res["wall_class"] == "insufficient"
    assert res["records"] == 0
    assert res["next_arm"]["arm"] == "repair_launch"
    assert res["next_arm"]["action"] == "fix"
    assert res["next_arm"]["command"] is None


def test_report_classify_fn_is_bound_to_the_calibration_dir(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    _write_progress(report.run_dir / "progress.jsonl", _flat_records(solutions=1))
    res = report.classify_fn(verdict_fn=_verdict_fn())
    assert res["run_dir"] == str(report.run_dir)
    assert res["wall_class"] == "resolved"
    # ...and still accepts an explicit directory, for a re-run elsewhere.
    other = tmp_path / "elsewhere"
    _write_progress(other / "progress.jsonl", _flat_records(n=3))
    assert classify_run(other, verdict_fn=_verdict_fn())["run_dir"] == str(other)
    assert report.classify_fn(other, verdict_fn=_verdict_fn())["run_dir"] == str(other)


# ---------------------------------------------------------------------
# Refinement from solved tapes.
# ---------------------------------------------------------------------

REFINED_LO = 0x0401


def _fake_tape_findings(*, lo=REFINED_LO, passing=True) -> dict:
    rec = ({"lo": lo, "hi": None, "label": f"${lo:04X}",
            "kind": "learnfun_shortlist", "learnfun_rank": 28,
            "shortlist_only": True, "saturates": False,
            "as_progress": f"${lo:04X} (learnfun shortlist)"} if passing else None)
    return {
        "forward": "right", "room_counter": AREA,
        "candidates": [], "recommended": rec,
        "gates": {"n_gated": 32, "passed": [f"${lo:04X}"] if passing else []},
        "quality": {"verdict": "concentrated"},
        "learnfun": {"n_segments": 30, "n_memories": 41000},
    }


def _refine(profile_path, chain, **kw):
    calls: dict = {}
    tape = kw.pop("tape", None) or _fake_tape_findings()

    def find_from_tapes(chain_arg, **fk):
        calls["chain"] = chain_arg
        calls["kwargs"] = fk
        return tape

    def emit(rom_arg, findings):
        prog = findings["progress"]["recommended"]
        out = ["solve:", f'  rom: "{rom_arg}"']
        if prog:
            out.append(f"  progress: {{lo: 0x{prog['lo']:04X}}}   "
                       f"# CONFIRM before trusting: shortlisted from tapes")
        out.append(f"  area: 0x{findings['progress']['room_counter']:04X}")
        if findings["y"]:
            out.append(f"  y: 0x{findings['y'][0]['addr']:04X}")
        out.append("  level_key: []")
        if findings["hp_lives"].get("addr") is not None:
            out.append(f"  lives: 0x{findings['hp_lives']['addr']:04X}")
        return "\n".join(out) + "\n"

    res = refine_from_tapes(profile_path, chain, find_from_tapes=find_from_tapes,
                            emit=emit, **kw)
    return res, calls


def test_refinement_writes_a_sibling_and_never_touches_the_draft(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    before = report.profile_path.read_text()

    res, calls = _refine(report.profile_path, "runs/solved_chain")

    assert report.profile_path.read_text() == before
    assert res.refined_path.name == "fake_game__usa_.refined.yaml"
    assert res.refined_path.exists()
    assert calls["chain"] == "runs/solved_chain"
    # The profile the tapes were produced under indexes them, so it is
    # what gets handed to the learnfun path.
    assert calls["kwargs"]["profile"]["solve"]["progress"]["lo"] == PROGRESS_LO


def test_refinement_carries_the_shortlist_marking_into_the_file(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    res, _ = _refine(report.profile_path, "chain")
    text = res.refined_path.read_text()
    assert "CONFIRM before trusting" in text
    prof = yaml.safe_load(text)
    assert prof["solve"]["progress"] == {"lo": REFINED_LO}
    assert "SHORTLIST" in prof["provenance"]["refinement_status"]
    assert "reward term" in prof["provenance"]["refinement_status"]
    assert prof["provenance"]["refined_from"].endswith("fake_game__usa_.yaml")
    assert prof["provenance"]["refinement_evidence"]["n_segments"] == 30
    assert res.changed is True
    assert lint_profile_text(text) == ()


def test_refinement_carries_y_and_the_death_proxy_forward(rom, state, tmp_path):
    """The tape path re-ranks PROGRESS only. Rebuilding the block from
    its findings alone would silently drop `y` and `lives`, and the
    resulting profile would fail at solver startup."""
    report, _ = _run(rom, tmp_path)
    res, _ = _refine(report.profile_path, "chain")
    prof = yaml.safe_load(res.refined_path.read_text())
    assert prof["solve"]["y"] == Y
    assert prof["solve"]["lives"] == LIVES
    assert res.findings["hp_lives"]["note"].startswith("carried over")


def test_refinement_that_agrees_with_the_draft_is_not_a_change(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    res, _ = _refine(report.profile_path, "chain",
                     tape=_fake_tape_findings(lo=PROGRESS_LO))
    assert res.changed is False
    assert any("agrees" in n for n in res.notes)


def test_refinement_with_nothing_clearing_the_gates_blocks(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    res, _ = _refine(report.profile_path, "chain",
                     tape=_fake_tape_findings(passing=False))
    assert res.blockers and "cleared both gates" in res.blockers[0]
    assert res.changed is False


def test_refinement_can_be_previewed_without_writing(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    res, _ = _refine(report.profile_path, "chain", write=False)
    assert res.refined_path is None
    assert not (report.profile_path.parent / "fake_game__usa_.refined.yaml").exists()


def test_report_refine_fn_is_bound_to_the_draft(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    seen: dict = {}

    def find_from_tapes(chain, **kw):
        seen["profile"] = kw["profile"]
        return _fake_tape_findings()

    res = report.refine_fn("chain", find_from_tapes=find_from_tapes, write=False)
    assert seen["profile"]["name"] == "Fake Game (USA)"
    assert res.profile_path == report.profile_path


# ---------------------------------------------------------------------
# --forward auto.
# ---------------------------------------------------------------------

def test_forward_auto_prefers_a_spatial_spine_over_a_room_fallback():
    spine = _fake_findings()
    fallback = _fake_findings()
    fallback["progress"]["recommended"] = {
        "lo": AREA, "hi": None, "kind": "room_counter_fallback",
        "spatial_saturator": "$0083"}
    nothing = _fake_findings(recommended=False)
    assert _forward_score(spine) > _forward_score(fallback) > _forward_score(nothing)


def test_forward_auto_stops_at_the_first_true_spine(rom, tmp_path):
    tried: list[str] = []

    def discover(rom_arg, state_arg, **dk):
        tried.append(dk["forward"])
        return _fake_findings()

    def ensure_state(rom_arg, *, force=False, **_):
        p = rom.with_name(rom.stem + "_start.state.bin")
        p.write_bytes(b"\x00")
        return p, "generated"

    report = onboard(rom, forward="auto", out_dir=tmp_path / "c",
                     runs_dir=tmp_path / "r", discover=discover,
                     emit=_fake_solve_block, ensure_state=ensure_state,
                     header_fn=_fake_header)
    assert tried == ["right"]
    assert report.forward == "right"
    assert any("auto picked" in n for n in report.notes)


def test_forward_auto_keeps_looking_when_a_direction_finds_nothing(rom, tmp_path):
    def discover(rom_arg, state_arg, **dk):
        return (_fake_findings() if dk["forward"] == "up"
                else _fake_findings(recommended=False))

    def ensure_state(rom_arg, *, force=False, **_):
        p = rom.with_name(rom.stem + "_start.state.bin")
        p.write_bytes(b"\x00")
        return p, "generated"

    report = onboard(rom, forward="auto", out_dir=tmp_path / "c",
                     runs_dir=tmp_path / "r", discover=discover,
                     emit=_fake_solve_block, ensure_state=ensure_state,
                     header_fn=_fake_header)
    assert report.forward == "up"
    assert report.ready
    assert yaml.safe_load(report.profile_path.read_text())["provenance"]["forward"] == "up"


# ---------------------------------------------------------------------
# Report shape — what the orchestrator consumes.
# ---------------------------------------------------------------------

def test_report_exposes_the_three_spec_fields(rom, state, tmp_path):
    report, _ = _run(rom, tmp_path)
    assert isinstance(report, OnboardReport)
    assert report.profile_path.exists()
    assert isinstance(report.solve_command, tuple)
    assert callable(report.classify_fn)
    assert callable(report.refine_fn)
    d = report.as_dict()
    assert set(d) >= {"profile_path", "solve_command", "blockers", "ready",
                      "start_state", "run_dir", "solve_minutes"}
    assert json.loads(json.dumps(d, default=str))


def test_onboard_refuses_a_missing_rom(tmp_path):
    with pytest.raises(FileNotFoundError):
        onboard(tmp_path / "nope.nes", discover=lambda *a, **k: {},
                emit=_fake_solve_block)


def test_onboard_refuses_a_missing_supplied_state(rom, tmp_path):
    with pytest.raises(FileNotFoundError):
        onboard(rom, start_state=tmp_path / "nope.bin",
                discover=lambda *a, **k: _fake_findings(),
                emit=_fake_solve_block, out_dir=tmp_path / "c",
                runs_dir=tmp_path / "r", header_fn=_fake_header)


def test_next_arm_dataclass_round_trips_to_json():
    arm = NextArm("interaction", "run", "switch axis", "because",
                  command=("py", "s.py"), profile_edits=("do a thing",))
    assert json.loads(json.dumps(arm.as_dict()))["command"] == ["py", "s.py"]


# ---------------------------------------------------------------------
# Probe recall reaching the draft.
#
# The control failure this covers: the SMB draft came back `ready:
# false` on "solve.lives missing", because the probe protocol could not
# nominate a counter kept above the zero page. Nothing downstream was
# wrong — the profile, the lint and the withheld command all did their
# job on the findings they were given. What is asserted here is the
# other half of that contract: when discovery DOES recall a high-RAM
# counter and a paged position pair, both survive to the profile the
# solver is handed, and the draft is ready.
# ---------------------------------------------------------------------

#: The two shapes the zero-page-only scan could not express: a life
#: counter in high work RAM, and a position split across a fine byte
#: and the page byte that steps when it wraps.
HIGH_LIVES = 0x075A
PAGED_LO, PAGED_HI = 0x0086, 0x006D


def _paged_findings(**kw) -> dict:
    f = _fake_findings(lives=HIGH_LIVES, **kw)
    rec = f["progress"]["recommended"]
    if rec:
        rec.update(lo=PAGED_LO, hi=PAGED_HI, label="$0086|$006D",
                   as_progress="$0086|$006D<<8")
    f["progress"]["candidates"][0].update(lo=PAGED_LO, hi=PAGED_HI,
                                          label="$0086|$006D")
    return f


def test_a_counter_above_the_zero_page_reaches_the_draft_and_is_ready(
        rom, state, tmp_path):
    """`lives` is a REQUIRED_SOLVE_KEY, so a counter the probes cannot
    reach is not a missing nicety — it withholds the whole calibration
    command."""
    report, _ = _run(rom, tmp_path, findings=_paged_findings())
    assert report.ready
    assert report.blockers == ()
    assert report.solve_command is not None
    assert not any("death signal" in n for n in report.notes)
    assert f"lives: 0x{HIGH_LIVES:04X}" in report.profile_path.read_text()


def test_a_high_ram_address_is_inside_what_get_ram_can_read(rom, state, tmp_path):
    """The lint bounds every solve address by the top of system RAM, and
    a counter at $075A is well inside it — so widening the scan past the
    zero page cannot trip the purity check that exists to catch a
    $6000-range address pasted in from outside."""
    report, _ = _run(rom, tmp_path, findings=_paged_findings())
    assert HIGH_LIVES <= SYSTEM_RAM_TOP
    assert lint_profile_text(report.profile_path.read_text()) == ()


def test_a_paged_position_pair_survives_to_the_profile(rom, state, tmp_path):
    """A fine byte alone wraps every 256 pixels of travel; the pair is
    what makes progress monotone across a screen boundary. The draft has
    to carry both halves or the solver keys on a sawtooth."""
    report, _ = _run(rom, tmp_path, findings=_paged_findings())
    solve = yaml.safe_load(report.profile_path.read_text())["solve"]
    assert solve["progress"] == {"lo": PAGED_LO, "hi": PAGED_HI}


def test_the_report_carries_the_solve_block_it_drafted(rom, state, tmp_path):
    """The orchestrator reads the report, not the YAML. Anything it has
    to re-parse the profile to learn is something it can get wrong."""
    report, _ = _run(rom, tmp_path, findings=_paged_findings())
    assert report.solve["progress"] == {"lo": PAGED_LO, "hi": PAGED_HI}
    assert report.solve["lives"] == HIGH_LIVES
    assert report.solve == yaml.safe_load(
        report.profile_path.read_text())["solve"]
    d = report.as_dict()
    assert d["solve"] == report.solve
    assert json.loads(json.dumps(d, default=str))


def test_the_reported_solve_block_is_empty_when_the_draft_is_withheld(
        rom, state, tmp_path):
    """A blocked draft still writes a profile, and the report still says
    what is in it — reporting a `solve` block the tool never stood
    behind would be worse than reporting none."""
    report, _ = _run(rom, tmp_path, findings=_fake_findings(hp=False))
    assert not report.ready
    assert "lives" not in report.solve
    assert report.as_dict()["solve"] == report.solve
