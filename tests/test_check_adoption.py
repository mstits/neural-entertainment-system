"""Tests for scripts/check_adoption.py.

One test per sub-check, each with a fixture engineered to make that
sub-check fail (report a hit) so the test proves the detector actually
detects, not just that it runs. A companion "clean" case per sub-check
proves it doesn't over-fire on the innocent shape.

Two real-repo regression blocks were added in the 2026-09-01 fix round,
after a verify pass found is_quarantined() returning False for the three
real demos_quarantine/ directories the report claimed it covered, and the
(b) census missing three real lockless writers whose write path runs
through an argparse default or a function-parameter default rather than
a plain `var = <literal>` assignment:
  - TestIsQuarantinedAgainstRealRepo
  - test_census_lockless_writers_finds_real_missed_writers
Both are read-only against the real repo. Each skips on the input it
actually needs: the class needs the gitignored
checkpoints/bc_*/demos_quarantine directories, the function needs only
committed files under scripts/.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_adoption as ca  # noqa: E402

REAL_REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# (a) census_inert_keys
# ---------------------------------------------------------------------------

def _fake_config_schema(**registries):
    mod = types.SimpleNamespace()
    for name, keys in registries.items():
        setattr(mod, name, frozenset(keys))
    return mod


def test_census_inert_keys_flags_unset_registry_key(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text(
        "name: a\nreinforce:\n  lr: 0.1\n"
    )
    schema = _fake_config_schema(
        KNOWN_TOP_KEYS=frozenset(),
        KNOWN_REINFORCE_KEYS={"lr", "gamma"},  # gamma never set below
        KNOWN_BACKWARD_CURRICULUM_KEYS=frozenset(),
        KNOWN_CONSOLIDATE_LEVEL_KEYS=frozenset(),
        KNOWN_CONSOLIDATE_PROBE_KEYS=frozenset(),
        KNOWN_SIL_KEYS=frozenset(),
        KNOWN_ADVERSARY_KEYS=frozenset(),
    )
    result = ca.census_inert_keys(tmp_path, schema)
    assert result["KNOWN_REINFORCE_KEYS"] == ["gamma"]


def test_census_inert_keys_clean_when_every_key_is_set(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text(
        "name: a\nreinforce:\n  lr: 0.1\n  gamma: 0.99\n"
    )
    schema = _fake_config_schema(
        KNOWN_TOP_KEYS=frozenset(),
        KNOWN_REINFORCE_KEYS={"lr", "gamma"},
        KNOWN_BACKWARD_CURRICULUM_KEYS=frozenset(),
        KNOWN_CONSOLIDATE_LEVEL_KEYS=frozenset(),
        KNOWN_CONSOLIDATE_PROBE_KEYS=frozenset(),
        KNOWN_SIL_KEYS=frozenset(),
        KNOWN_ADVERSARY_KEYS=frozenset(),
    )
    assert ca.census_inert_keys(tmp_path, schema) == {}


# ---------------------------------------------------------------------------
# (b) census_lockless_writers
# ---------------------------------------------------------------------------

def test_census_lockless_writers_flags_write_with_no_lock_import(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "writer.py").write_text(
        "from pathlib import Path\n"
        "REPO = Path('.')\n"
        "out = REPO / 'checkpoints' / 'x.pt'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_bytes(b'data')\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/writer.py" in hits
    assert hits["scripts/writer.py"]  # at least one line number


def test_census_lockless_writers_ignores_writer_with_lock_import(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "writer.py").write_text(
        "from pathlib import Path\n"
        "from src.utils.run_lock import acquire\n"
        "REPO = Path('.')\n"
        "out = REPO / 'checkpoints' / 'x.pt'\n"
        "out.write_bytes(b'data')\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/writer.py" not in hits


def test_census_lockless_writers_ignores_write_to_unrelated_dir(tmp_path):
    """Regression case for the false positive this detector was
    rewritten to avoid: reading from runs/ while writing elsewhere in
    the same file must not be flagged (see check_adoption.py's comment
    citing the real scripts/show_fx.py false positive)."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reader_writer.py").write_text(
        "from pathlib import Path\n"
        "REPO = Path('.')\n"
        "src = sorted((REPO / 'runs/live_show').glob('*/sol_*.npy'))\n"
        "out_dir = REPO / 'docs/receipts/show_lane'\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        "(out_dir / 'r.json').write_text('{}')\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/reader_writer.py" not in hits


def test_census_lockless_writers_flags_write_via_argparse_default(tmp_path):
    """Regression case for the 2026-09-01 verify miss: a write whose
    target path flows from an argparse `add_argument(..., default=...)`
    keyword (never an Assign node) through `args.<dest>` and an
    intermediate function call, not a direct `var = <literal>` chain.
    Shape mirrors the real scripts/soak_harness.py miss.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "writer.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "REPO = Path('.')\n"
        "\n"
        "def _default_out_dir(root, selfcheck=False):\n"
        "    return Path(root) / ('a' if selfcheck else 'b')\n"
        "\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--out-root', type=Path, default=REPO / 'runs' / 'soak')\n"
        "    args = ap.parse_args()\n"
        "    out_dir = _default_out_dir(args.out_root, selfcheck=True)\n"
        "    out_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (out_dir / 'log.json').write_text('{}')\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/writer.py" in hits
    assert hits["scripts/writer.py"]


def test_census_lockless_writers_flags_write_via_argparse_default_direct(tmp_path):
    """Same miss, simpler shape: `out_dir = Path(args.out_dir)` with no
    intervening function call. Mirrors the real
    scripts/critic_explained_variance.py miss.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "writer.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--out-dir', default='runs/v29_stability/f1_ev')\n"
        "    args = ap.parse_args()\n"
        "    out_dir = Path(args.out_dir)\n"
        "    out_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (out_dir / 'summary.json').write_text('{}')\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/writer.py" in hits
    assert hits["scripts/writer.py"]


def test_census_lockless_writers_ignores_multi_arg_call_carrying_tracked_value(tmp_path):
    """Regression case for a false positive the broadened tracking (added
    in the 2026-09-01 fix round) introduced and then had to be closed:
    a tracked value passed as one of several arguments to an unrelated
    function must not make that function's return value "tracked," or
    an unrelated write elsewhere in the file that shares a variable name
    with something in the return value gets misclassified. Mirrors the
    real scripts/perf_counters.py false positive: `args.tape` genuinely
    reads from runs/, but `workload_fn(Path(args.rom), Path(args.tape),
    args.frames)`'s return value is a stats dict, not a path, and must
    not taint the unrelated `Path(args.out).write_text(...)` write
    later in the file just because a same-named local elsewhere happens
    to be built from that dict.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bench.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "REPO = Path('.')\n"
        "\n"
        "def run_workload(rom, tape, frames):\n"
        "    return {'ipc': 1.0}\n"
        "\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--tape', default=str(REPO / 'runs' / 'x' / 'tape.npy'))\n"
        "    ap.add_argument('--out', default=None)\n"
        "    args = ap.parse_args()\n"
        "    workload_result = run_workload(Path('rom.nes'), Path(args.tape), 10)\n"
        "    result = {'workload': workload_result}\n"
        "    text = str(result)\n"
        "    if args.out:\n"
        "        Path(args.out).write_text(text)\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/bench.py" not in hits


def test_census_lockless_writers_flags_write_via_function_param_default(tmp_path):
    """Regression case: a module-level constant flowing into a function
    PARAMETER default (`def f(runs_dir=RUNS_DIR): ...`) rather than into
    a same-scope `var = ...` assignment. Mirrors the real
    scripts/onboard_game.py miss (`runs_dir: str | Path = RUNS_DIR`).
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "writer.py").write_text(
        "from pathlib import Path\n"
        "REPO = Path('.')\n"
        "RUNS_DIR = REPO / 'runs' / 'onboard'\n"
        "\n"
        "def build(slug, runs_dir=RUNS_DIR):\n"
        "    run_dir = Path(runs_dir) / slug\n"
        "    receipt_path = run_dir / 'onboard_receipt.json'\n"
        "    receipt_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    receipt_path.write_text('{}')\n"
    )
    hits = ca.census_lockless_writers(tmp_path)
    assert "scripts/writer.py" in hits
    assert hits["scripts/writer.py"]


# ---------------------------------------------------------------------------
# (c) census_solve_keys
# ---------------------------------------------------------------------------

def test_census_solve_keys_flags_unregistered_key(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "game.yaml").write_text(
        "name: game\nsolve:\n  rom: game.nes\n  totally_new_key: 1\n"
    )
    monkeypatch.setattr(
        ca, "PROPOSED_KNOWN_SOLVE_KEYS", frozenset({"rom", "progress"})
    )
    result = ca.census_solve_keys(tmp_path)
    assert result["unregistered_in_yaml_not_in_registry"] == ["totally_new_key"]


def test_census_solve_keys_clean_when_registry_matches(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "game.yaml").write_text(
        "name: game\nsolve:\n  rom: game.nes\n"
    )
    monkeypatch.setattr(
        ca, "PROPOSED_KNOWN_SOLVE_KEYS", frozenset({"rom"})
    )
    result = ca.census_solve_keys(tmp_path)
    assert result["unregistered_in_yaml_not_in_registry"] == []


# ---------------------------------------------------------------------------
# (d) is_quarantined + census_unquarantined_globs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "checkpoints/QUARANTINE_tier3/demo.npz",
    "checkpoints/bc_seed_d525075e1a91.pt.stale-pixel",
    "checkpoints/mario_1_2_dmap.POISONED.pkl",
    "checkpoints/_archived_pre_peakx_2226/iter_100.pt",
    "checkpoints/bc_seed_595e7cf4519c.pt.archived",
    # Regression case: the naming convention the report itself claimed
    # was covered ("demos_quarantine/ subdirectories under
    # checkpoints/bc_1_3, bc_1_4, bc_2_1") but the shipped
    # is_quarantined() returned False for -- a "<label>_quarantine"
    # directory name, which is neither the exact name "quarantine" nor
    # prefixed with "quarantine_".
    "checkpoints/bc_1_3/demos_quarantine/ep_001.demo",
])
def test_is_quarantined_true_for_known_patterns(path):
    assert ca.is_quarantined(path) is True


def test_is_quarantined_false_for_clean_path():
    assert ca.is_quarantined("checkpoints/vanilla_ppo_iter_100.pt") is False


def test_census_unquarantined_globs_flags_naive_reader(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reader.py").write_text(
        "from pathlib import Path\n"
        "REPO = Path('.')\n"
        "ckpt_dir = REPO / 'checkpoints'\n"
        "latest = sorted(ckpt_dir.glob('iter_*.pt'))[-1]\n"
    )
    hits = ca.census_unquarantined_globs(tmp_path)
    assert "scripts/reader.py" in hits


def test_census_unquarantined_globs_ignores_quarantine_aware_reader(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reader.py").write_text(
        "from pathlib import Path\n"
        "from check_adoption import is_quarantined\n"
        "REPO = Path('.')\n"
        "ckpt_dir = REPO / 'checkpoints'\n"
        "cands = [p for p in ckpt_dir.glob('iter_*.pt') if not is_quarantined(p)]\n"
    )
    hits = ca.census_unquarantined_globs(tmp_path)
    assert "scripts/reader.py" not in hits


# ---------------------------------------------------------------------------
# Real-repo regression checks (read-only). These run against the actual paths
# named in the DO-7 report, not synthetic fixtures, so a fix here can't
# silently regress again while every synthetic test still passes.
#
# Two different gates, because the two blocks need two different things:
#
#   pytestmark_real_repo         needs only committed files (scripts/), so the
#                                repo root existing is the right condition.
#   pytestmark_quarantine_fixtures  needs checkpoints/bc_*/demos_quarantine,
#                                which is gitignored (.gitignore:10) and so is
#                                absent on a clean clone even though the repo
#                                root is present. Gating that block on the repo
#                                root made it fail rather than skip there.
# ---------------------------------------------------------------------------

pytestmark_real_repo = pytest.mark.skipif(
    not REAL_REPO.is_dir(), reason="real repo not present at expected path"
)

QUARANTINE_FIXTURE_DIRS = [
    REAL_REPO / "checkpoints" / sub / "demos_quarantine"
    for sub in ("bc_1_3", "bc_1_4", "bc_2_1")
]

pytestmark_quarantine_fixtures = pytest.mark.skipif(
    not all(p.is_dir() for p in QUARANTINE_FIXTURE_DIRS),
    reason="gitignored fixtures absent: checkpoints/bc_{1_3,1_4,2_1}/demos_quarantine",
)


@pytestmark_quarantine_fixtures
class TestIsQuarantinedAgainstRealRepo:
    @pytest.mark.parametrize("subdir", ["bc_1_3", "bc_1_4", "bc_2_1"])
    def test_real_demos_quarantine_dirs_are_quarantined(self, subdir):
        p = REAL_REPO / "checkpoints" / subdir / "demos_quarantine"
        assert p.is_dir(), f"fixture path missing or moved: {p}"
        assert ca.is_quarantined(p.relative_to(REAL_REPO)) is True
        assert ca.is_quarantined(p) is True

    def test_real_clean_checkpoint_paths_stay_unquarantined(self):
        """Sanity check the fix didn't turn is_quarantined() into an
        always-True detector: a handful of ordinary checkpoint files
        (no quarantine marker in any path component) must still read
        as False.
        """
        ckpt_dir = REAL_REPO / "checkpoints"
        candidates = sorted(ckpt_dir.glob("*.pt"))[:20]
        # The class gate above keys on checkpoints/bc_*/demos_quarantine.
        # This body reads checkpoints/*.pt, a different path, and nothing
        # under checkpoints/ is tracked, so a tree can carry the three
        # quarantine directories and no top-level .pt at all. Ordinary
        # housekeeping drains this glob on its own: the repo quarantines a
        # top-level checkpoint by renaming it to *.pt.stale-pixel or
        # *.pt.archived, which stops matching *.pt. With an empty glob the
        # loop below never runs and this reports a pass having asserted
        # nothing, even against an is_quarantined() that answers True for
        # every path, which is the one regression this test exists to catch.
        if not candidates:
            pytest.skip(
                "gitignored or machine-local inputs absent: checkpoints/*.pt")
        for p in candidates:
            rel = p.relative_to(REAL_REPO)
            expect_true = any(
                marker in str(rel).lower()
                for marker in ("quarantine", "_archived_", "archived_",
                               ".stale-pixel", ".archived", ".poisoned")
            )
            assert ca.is_quarantined(rel) == expect_true, rel


@pytestmark_real_repo
def test_census_lockless_writers_finds_real_missed_writers():
    """The three scripts the verify pass found the shipped census
    missing: soak_harness.py (argparse default -> function call ->
    attribute write), critic_explained_variance.py (argparse default ->
    Path(args.x) directly), onboard_game.py (module constant -> function
    parameter default -> local var). All three write under runs/ with
    no run_lock import.
    """
    hits = ca.census_lockless_writers(REAL_REPO)
    for expected in (
        "scripts/soak_harness.py",
        "scripts/critic_explained_variance.py",
        "scripts/onboard_game.py",
    ):
        assert expected in hits, f"{expected} not flagged: {sorted(hits)}"
        assert hits[expected]
