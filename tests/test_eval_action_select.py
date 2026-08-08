"""Tests for sampled-action evaluation in `scripts/eval_game.py` (L1).

The honest protocol mandates reporting a GREEDY and a SAMPLED number side by
side: an argmax-only harness cannot tell a learned policy from a Brute whose
own action distribution never reproduces the trajectory its argmax walks.
`eval_game.py` was argmax-only, so half the mandated pair was unmeasurable.

Layers:
  * pure `select_action` — argmax equivalence, generator-seeded reproducibility,
    global-RNG isolation, temperature limits, bad-mode rejection (fast, no ROM);
  * CLI contract — default is greedy, bad mode / non-positive temperature are
    rejected on the single-JSON-line contract (fast, no ROM);
  * `cold_probe.probe` plumbing under a mocked subprocess — the greedy default
    emits a byte-identical command line, and the sampled flags land on `cmd`
    (never `extra_args`, which the non-sequential degrade path drops);
  * ROM-gated end-to-end — the default receipt is reproducible and labelled
    greedy, a near-zero temperature reproduces greedy exactly, and T=1.0
    returns populated stats.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
# `eval_game.py` lives under scripts/ and is imported by module name (it is
# not a package). Mirrors tests/test_eval_pixel.py.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import eval_game  # noqa: E402
from eval_game import select_action  # noqa: E402
from src.training.cold_probe import probe  # noqa: E402


# --------------------------------------------------------------------------
# gitignored data lookup (ROM / start state / checkpoint) — searches this
# checkout then each ancestor. Mirrors tests/test_eval_pixel.py::_find.
# --------------------------------------------------------------------------
def _find(rel_candidates: list[str]) -> Path | None:
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            for hit in sorted(root.glob(rel)):
                if hit.exists():
                    return hit
    return None


_ROM = _find(["roms/Super Mario Bros. (World).nes"])
_START = _find(["roms/Super Mario Bros. (World)_start.state.bin"])
_TILE_CKPT = _find([
    "checkpoints/super_mario_bros_one_shot_tiles/vanilla_ppo_iter_*.pt",
])
_TILE_CFG = _ROOT / "configs" / "smb_oneshot_tiles.yaml"
_HAVE_EMU = _ROM is not None and _START is not None


def _logits(*values: float) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.float32)


# --- pure select_action ----------------------------------------------------

def test_select_action_greedy_is_argmax_and_ignores_generator():
    lg = _logits(0.1, 3.5, -2.0, 3.4)
    want = int(torch.argmax(lg[0]).item())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    # Greedy must not consume the generator: same answer with, without, and
    # after any number of calls.
    assert select_action(lg, "greedy") == want
    assert select_action(lg, "greedy", 1.0, gen) == want
    assert select_action(lg, "greedy", 0.5, gen) == want
    assert gen.initial_seed() == 7


def test_select_action_greedy_is_the_default_mode():
    lg = _logits(-1.0, 0.0, 5.0)
    assert select_action(lg) == 2


def test_select_action_sampled_is_reproducible_from_the_generator_seed():
    lg = _logits(0.5, 0.4, 0.6, 0.45)

    def draw(seed: int, n: int = 40) -> list[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return [select_action(lg, "sampled", 1.0, g) for _ in range(n)]

    assert draw(0) == draw(0)
    assert draw(1) == draw(1)
    # Near-uniform logits over 40 draws: two different seeds diverging is
    # certain to any practical tolerance.
    assert draw(0) != draw(1)


def test_select_action_sampled_does_not_disturb_global_torch_rng():
    """The eval loop must not perturb any surrounding torch RNG consumer."""
    lg = _logits(0.2, 0.3, 0.1, 0.4)
    torch.manual_seed(1234)
    baseline = torch.rand(4)

    torch.manual_seed(1234)
    g = torch.Generator(device="cpu")
    g.manual_seed(99)
    for _ in range(25):
        select_action(lg, "sampled", 1.0, g)
    after = torch.rand(4)
    assert torch.equal(baseline, after)


def test_select_action_low_temperature_collapses_onto_argmax():
    lg = _logits(0.0, 1.0, 2.5, 2.0)
    g = torch.Generator(device="cpu")
    g.manual_seed(3)
    want = int(torch.argmax(lg[0]).item())
    assert all(select_action(lg, "sampled", 1e-3, g) == want for _ in range(30))


def test_select_action_high_temperature_spreads_across_actions():
    lg = _logits(0.0, 1.0, 2.5, 2.0)
    g = torch.Generator(device="cpu")
    g.manual_seed(5)
    seen = {select_action(lg, "sampled", 50.0, g) for _ in range(200)}
    assert seen == {0, 1, 2, 3}


def test_select_action_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown action_select mode"):
        select_action(_logits(1.0, 2.0), "epsilon_greedy")


# --- eval_one_game argument validation (raises before any I/O) -------------

def test_eval_one_game_rejects_bad_action_select_before_touching_disk():
    with pytest.raises(ValueError, match="action_select must be one of"):
        eval_game.eval_one_game(
            "mario", Path("/nonexistent/profile.yaml"), "/nonexistent.nes",
            1, 1, action_select="stochastic",
        )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_eval_one_game_rejects_non_positive_temperature(bad):
    with pytest.raises(ValueError, match="temperature must be > 0"):
        eval_game.eval_one_game(
            "mario", Path("/nonexistent/profile.yaml"), "/nonexistent.nes",
            1, 1, action_select="sampled", temperature=bad,
        )


# --- CLI contract (no ROM needed) ------------------------------------------

def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "eval_game.py"), *args],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=180,
    )


def test_cli_rejects_unknown_action_select_mode():
    r = _cli("--game", "mario", "--action-select", "maybe")
    assert r.returncode == 2
    assert "--action-select" in r.stderr


def test_cli_rejects_non_positive_temperature_on_the_json_contract():
    """Bad --temperature must surface as one JSON line, not a traceback — the
    cold probe parses stdout, not stderr."""
    r = _cli("--game", "mario", "--temperature", "0")
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["status"] == "bad_args"
    assert "temperature" in out["detail"]


# --- cold_probe plumbing (mocked subprocess) -------------------------------

def _probe_cmds(**kw) -> list[list[str]]:
    """Run `probe` with subprocess.run mocked; return every argv it launched."""
    calls: list[list[str]] = []

    def fake_run(cmd, **_kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout=json.dumps({
                "game": "mario", "status": "ok", "checkpoint": "c.pt",
                "n_episodes": 2, "mean_return": 1.0, "mean_length": 2.0,
                "max_byte_seen": 0, "clear_rate": 0.0,
                "action_select": "greedy", "temperature": 1.0,
            }),
            stderr="",
        )

    # `probe` only requires the ROM path to EXIST (the mocked subprocess never
    # opens it), so this file stands in and the plumbing tests stay hermetic.
    fake_rom = str(Path(__file__))
    net = {"w": torch.zeros(2)}
    cfg = {"name": "mario", "rom_path": fake_rom}
    with mock.patch("subprocess.run", side_effect=fake_run):
        probe(net, cfg, episodes=2, rom_path=fake_rom, **kw)
    assert calls, "probe never launched a subprocess"
    return calls


def test_probe_default_command_line_carries_no_sampling_flags():
    """The greedy default must leave the historical invocation untouched."""
    cmd = _probe_cmds(sequential=True)[0]
    assert "--action-select" not in cmd
    assert "--temperature" not in cmd
    assert "--eval-seed" not in cmd


def test_probe_sampled_passes_mode_and_temperature():
    cmd = _probe_cmds(sequential=True, action_select="sampled", temperature=0.7)[0]
    assert cmd[cmd.index("--action-select") + 1] == "sampled"
    assert float(cmd[cmd.index("--temperature") + 1]) == pytest.approx(0.7)


def test_probe_sampled_flags_survive_the_non_sequential_path():
    """`extra_args` ride only with --sequential and are dropped on the degrade
    path; a dropped --action-select would report a greedy number labelled
    sampled, so these flags must live on the base command."""
    cmd = _probe_cmds(sequential=False, action_select="sampled")[0]
    assert "--sequential" not in cmd
    assert "--action-select" in cmd


def test_probe_forwards_eval_seed_only_when_given():
    cmd = _probe_cmds(sequential=True, eval_seed=17)[0]
    assert cmd[cmd.index("--eval-seed") + 1] == "17"


# --- ROM-gated end-to-end --------------------------------------------------

def _run_eval(ckpt: Path, *, episodes: int, max_steps: int,
              extra: list[str] | None = None) -> dict:
    """Subprocess eval_game in an isolated temp CWD (like the cold probe) with
    an absolutized profile, so the live campaign's eval.jsonl is never touched.
    Mirrors tests/test_eval_pixel.py::_run_eval.
    """
    prof = yaml.safe_load(_TILE_CFG.read_text())
    prof["start_state_path"] = str(_START)
    slug = eval_game.derive_checkpoint_dir("./checkpoints", prof.get("name")).name
    with tempfile.TemporaryDirectory(prefix="eval_action_select_test_") as td:
        td = Path(td)
        (td / "checkpoints" / slug).mkdir(parents=True)
        pf = td / "probe_profile.yaml"
        pf.write_text(yaml.safe_dump(prof))
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "eval_game.py"),
            "--game", "mario", "--profile", str(pf), "--rom", str(_ROM),
            "--checkpoint", str(ckpt),
            "--episodes", str(episodes), "--max-steps", str(max_steps),
            "--eval-seed", "0",
        ] + list(extra or [])
        r = subprocess.run(cmd, cwd=str(td), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def _stats(out: dict) -> dict:
    """The result minus wall-clock / path fields, for run-to-run comparison."""
    return {k: v for k, v in out.items()
            if k not in ("timestamp", "checkpoint", "action_select", "temperature")}


_EMU_SKIP = pytest.mark.skipif(
    not _HAVE_EMU or _TILE_CKPT is None,
    reason="needs the SMB ROM/start state and a tile one-shot checkpoint",
)


@pytest.mark.slow
@_EMU_SKIP
def test_default_path_is_greedy_labelled_and_reproducible():
    a = _run_eval(_TILE_CKPT, episodes=2, max_steps=300)
    b = _run_eval(_TILE_CKPT, episodes=2, max_steps=300)
    assert a["status"] == "ok", a
    assert a["action_select"] == "greedy"
    assert a["temperature"] == 1.0
    assert _stats(a) == _stats(b)


@pytest.mark.slow
@_EMU_SKIP
def test_sampled_at_near_zero_temperature_reproduces_greedy_exactly():
    """softmax(logits/T) -> one-hot as T -> 0, so the sampled path must collapse
    onto the greedy path. Pins the sampling maths against the argmax baseline
    on a real rollout, not just synthetic logits."""
    greedy = _run_eval(_TILE_CKPT, episodes=2, max_steps=300)
    cold = _run_eval(_TILE_CKPT, episodes=2, max_steps=300,
                     extra=["--action-select", "sampled", "--temperature", "1e-3"])
    assert cold["action_select"] == "sampled"
    assert _stats(cold) == _stats(greedy)


@pytest.mark.slow
@_EMU_SKIP
def test_sampled_mode_returns_populated_stats_and_is_seed_reproducible():
    a = _run_eval(_TILE_CKPT, episodes=3, max_steps=300,
                  extra=["--action-select", "sampled"])
    assert a["status"] == "ok", a
    assert a["action_select"] == "sampled"
    assert a["temperature"] == 1.0
    assert a["n_episodes"] == 3
    assert a["mean_length"] > 0.0
    assert len(a["max_gx_per_episode"]) == 3
    assert 0.0 <= a["clear_rate"] <= 1.0
    # Same --eval-seed => byte-identical stats: a sampled receipt is a
    # reproducible artifact, not a one-off draw.
    b = _run_eval(_TILE_CKPT, episodes=3, max_steps=300,
                  extra=["--action-select", "sampled"])
    assert _stats(a) == _stats(b)
