"""Tests for the pixel-encoder evaluation path in `scripts/eval_game.py`.

Historical gap: `eval_game.py` routed every profile through
`profile_utils.resolve_encoder`, which is tile-only, so any pixel profile
(`reinforce.encoder: nature_dqn`, e.g. `configs/smb_oneshot.yaml`) returned
`status=unsupported_profile` and every cold probe of the pixel campaign failed
inert. These tests pin the pixel path down and — crucially — prove its
observation assembly matches the trainer's collection path bit-for-bit, the
exact class of check the original tile-only tests never exercised.

Layers:
  * pure `resolve_pixel_encoder` dispatch (fast, no ROM);
  * a ROM-gated pixel smoke run (status ok, furthest_seq populated, full keys);
  * a ROM-gated tile-path key-set regression (the pixel dispatch must not
    perturb the tile output);
  * a ROM-gated PAIRING test: 20 steps through the eval stack vs a minimal
    replica of the trainer's collection forward, asserting identical obs and
    identical greedy actions on identical frames.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
# `eval_game.py` and `train_game.py` live under scripts/ and are imported by
# module name (they are not a package). Put both the repo root and scripts/ on
# the path so `import eval_game` — which does `from train_game import ...` and
# `from src...` — resolves against the code under test in THIS worktree.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import eval_game  # noqa: E402
from eval_game import _PixelPolicy, resolve_pixel_encoder  # noqa: E402


# --------------------------------------------------------------------------
# gitignored data lookup (ROM / start state / checkpoint) — searches this
# checkout then each ancestor, so the data is found whether the test runs in
# the primary working copy or an isolated worktree whose roms/checkpoints live
# in a parent working copy. Mirrors tests/test_cold_probe.py::_find.
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
_PIXEL_CKPT = _find([
    "checkpoints/super_mario_bros_one_shot/vanilla_ppo_iter_00400.pt",
    "checkpoints/super_mario_bros_one_shot/vanilla_ppo_iter_*.pt",
])
_TILE_CKPT = _find([
    "checkpoints/super_mario_bros_one_shot_tiles/vanilla_ppo_iter_*.pt",
])
_PIXEL_CFG = _ROOT / "configs" / "smb_oneshot.yaml"
_TILE_CFG = _ROOT / "configs" / "smb_oneshot_tiles.yaml"

_HAVE_EMU = _ROM is not None and _START is not None


# All the JSON fields eval_game emits for a `--sequential` run. `level_clear`
# is always set once `sequential=True` (scripts/eval_game.py); `start_state`
# is NOT — it's only emitted under the separate `--start-state` CLI probe
# (gated on `if start_state is not None`, independent of `--sequential`),
# which `_run_eval` never passes.
_SEQ_KEYS = {
    "game", "status", "checkpoint", "recurrent", "stage", "n_episodes",
    "mean_return", "mean_length", "max_byte_seen", "mean_max_byte",
    "clear_rate", "timestamp", "sequential", "seq_clear_rate", "warp_rate",
    "furthest_seq_level", "furthest_any_level", "furthest_seq", "furthest_any",
    "max_gx_per_episode", "level_clear",
    # Action-draw provenance — always emitted so a clear rate can never be
    # read without knowing whether it came from argmax or the policy's own
    # distribution (see tests/test_eval_action_select.py).
    "action_select", "temperature",
    # Schedule + randomness provenance — the effective worker count and which
    # RNG derivation produced the episodes (see tests/test_eval_parallel.py).
    "eval_workers", "eval_rng",
}


def _run_eval(cfg_path: Path, ckpt: Path, *, episodes: int, max_steps: int) -> dict:
    """Subprocess eval_game in an isolated temp CWD (like the cold probe) with
    an absolutized profile, so the live campaign's eval.jsonl is never touched.
    """
    prof = yaml.safe_load(cfg_path.read_text())
    prof["start_state_path"] = str(_START)
    slug = eval_game.derive_checkpoint_dir("./checkpoints", prof.get("name")).name
    with tempfile.TemporaryDirectory(prefix="eval_pixel_test_") as td:
        td = Path(td)
        (td / "checkpoints" / slug).mkdir(parents=True)
        pf = td / "probe_profile.yaml"
        pf.write_text(yaml.safe_dump(prof))
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "eval_game.py"),
            "--game", "mario", "--profile", str(pf), "--rom", str(_ROM),
            "--checkpoint", str(ckpt),
            "--episodes", str(episodes), "--max-steps", str(max_steps),
            "--sequential",
        ]
        r = subprocess.run(cmd, cwd=str(td), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


# --- pure dispatch ---------------------------------------------------------

def test_resolve_pixel_encoder_accepts_cnn_rejects_tile_and_unknown():
    assert resolve_pixel_encoder({"reinforce": {"encoder": "nature_dqn"}}) == "nature_dqn"
    assert resolve_pixel_encoder({"reinforce": {"encoder": "impala"}}) == "impala"
    # Tile encoders are NOT pixel encoders — they route through resolve_encoder.
    assert resolve_pixel_encoder({"reinforce": {"encoder": "smb_tiles"}}) is None
    # Unknown / missing encoder stays unsupported.
    assert resolve_pixel_encoder({"reinforce": {"encoder": "made_up"}}) is None
    assert resolve_pixel_encoder({"reinforce": {}}) is None
    assert resolve_pixel_encoder({}) is None


def test_smb_oneshot_profile_is_a_pixel_profile():
    """The live campaign profile is exactly the one the old code rejected."""
    prof = yaml.safe_load(_PIXEL_CFG.read_text())
    assert resolve_pixel_encoder(prof) == "nature_dqn"
    assert prof["reinforce"].get("preprocess_f16") is True


# --- ROM-gated pixel smoke -------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_EMU or _PIXEL_CKPT is None,
    reason="needs the SMB ROM/start state and a pixel one-shot checkpoint",
)
def test_eval_pixel_smoke_status_ok_and_keys_complete():
    out = _run_eval(_PIXEL_CFG, _PIXEL_CKPT, episodes=2, max_steps=500)
    assert out["status"] == "ok", out
    assert out["recurrent"] is False
    assert out["n_episodes"] == 2
    assert out["sequential"] is True
    # The DoD metric surface is fully populated (not the unsupported_profile stub).
    assert set(out) == _SEQ_KEYS, set(out) ^ _SEQ_KEYS
    # --sequential ran: furthest_seq is a real (world, level), not None.
    assert out["furthest_seq"] is not None
    assert out["furthest_seq_level"] is not None
    assert 0.0 <= out["seq_clear_rate"] <= 1.0


# --- ROM-gated tile-path regression (dispatch must not perturb tile) -------

@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_EMU or _TILE_CKPT is None,
    reason="needs the SMB ROM/start state and a tile one-shot checkpoint",
)
def test_eval_tile_path_unchanged_key_surface():
    out = _run_eval(_TILE_CFG, _TILE_CKPT, episodes=2, max_steps=400)
    assert out["status"] == "ok", out
    assert set(out) == _SEQ_KEYS, set(out) ^ _SEQ_KEYS


# --- ROM-gated pairing: eval obs assembly == trainer collection forward ----

def _trainer_pp_f16(pp: np.ndarray) -> np.ndarray:
    """Reinterpret the raw pool's f16-mode preprocessed obs the way the trainer
    does via RustPool._materialize: (84, 168) uint8 half-bytes -> (84, 84) f16.
    Independent of _PixelPolicy._pp so the pairing test is a genuine cross-check.
    """
    assert pp.dtype == np.uint8 and pp.shape == (84, 168), (pp.dtype, pp.shape)
    return pp.view(np.float16).reshape((84, 84))


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_EMU or _PIXEL_CKPT is None,
    reason="needs the SMB ROM/start state and a pixel one-shot checkpoint",
)
def test_pixel_obs_matches_trainer_collection_forward():
    """20 steps through (a) the eval stack (_PixelPolicy) and (b) a minimal
    replica of the trainer's collection forward. Both consume the SAME frames
    from one pool; assert identical stacked obs and identical greedy actions.

    This is the check the tile-only tests never had: a dtype/scale mismatch in
    the pixel obs assembly (e.g. an erroneous /255 on already-normalized f16)
    would diverge the two argmax streams here.
    """
    from nes_core import Pool
    from src.emulation.frame_utils import FrameStacker
    from src.models.policy_network import PolicyNetwork
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load(_PIXEL_CFG.read_text())
    rl = prof["reinforce"]
    preprocess_f16 = bool(rl.get("preprocess_f16", False))
    assert preprocess_f16, "smb_oneshot is expected to exercise the f16 path"
    frame_stack = int(rl.get("frame_stack", 4))
    bitmasks = action_space_to_bitmasks(prof["action_space"])

    state = torch.load(str(_PIXEL_CKPT), map_location="cpu")
    net = PolicyNetwork(
        num_actions=len(bitmasks), frame_stack=frame_stack, frame_size=84,
        encoder="nature_dqn", use_layernorm=bool(rl.get("layernorm", True)),
    )
    miss = net.load_state_dict(state["net_state_dict"], strict=False).missing_keys
    assert not miss, miss
    net.eval()

    pool = Pool(
        rom_path=str(_ROM), num_workers=1, frame_skip=4,
        start_state_path=str(_START),
    )
    pool.set_preprocess_f16(True)  # match the trainer / smb_oneshot profile
    try:
        # (a) eval stack under test.
        pol = _PixelPolicy(net, FrameStacker(stack_size=frame_stack, dtype=np.float16),
                           preprocess_f16=True)
        # (b) trainer-collection replica: its own FrameStacker + its own
        # reinterpret + the trainer's exact tensor build (no /255 for f16).
        tr_stacker = FrameStacker(stack_size=frame_stack, dtype=np.float16)

        pool.reset_all()
        init = pool.step_all(np.zeros(1, dtype=np.uint8))
        obs_eval = pol.reset(init[0])
        obs_tr = tr_stacker.reset(init[0][0], _trainer_pp_f16(init[0][1]))

        for step in range(20):
            # Identical obs into both forwards.
            assert np.array_equal(obs_eval, obs_tr), f"obs diverged at step {step}"

            with torch.no_grad():
                logits_e, _ = pol.logits(obs_eval, None)
                a_eval = int(torch.argmax(logits_e[0]).item())

                # Trainer collection forward (trainer.py rollout, 1 env):
                #   np.stack(stacked_obs) -> from_numpy -> .to(dev).float()
                #   -> /255 iff (not tile and not preprocess_f16)  [skipped: f16]
                #   -> net.forward_ac -> (greedy) argmax.
                batch = torch.from_numpy(np.stack([obs_tr], axis=0)).to("cpu").float()
                logits_t, _ = net.forward_ac(batch)
                a_tr = int(torch.argmax(logits_t[0]).item())

            assert a_eval == a_tr, (step, a_eval, a_tr)

            # Advance the single pool with the (shared) action; feed the same
            # post-step frame into both stacks.
            r = pool.step_all(np.array([bitmasks[a_eval]], dtype=np.uint8))
            obs_eval = pol.push(r[0])
            obs_tr = tr_stacker.push(r[0][0], _trainer_pp_f16(r[0][1]))
            if bool(r[0][3]):  # episode ended within 20 steps — enough coverage
                break
    finally:
        pool.shutdown()
