"""Unit + smoke tests for the level-keyed hierarchical (composite) evaluator.

The unit cases prove the switch logic — manifest resolution, hysteresis, and the
on-switch stack/hidden reset — plus that MIXED encoders (a tile net and a pixel
net) compose in one controller, all with stubbed/synthetic nets and no emulator.
The final (slow, ROM-gated) case is the acceptance test: a champion-everywhere
manifest run cold from 1-1 through `scripts/eval_composite.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.policy_network import PolicyNetwork
from src.models.tile_policy import TilePolicyNetwork
from src.training.composite_policy import (
    CompositeController,
    HysteresisSwitch,
    PixelObsAdapter,
    TileObsAdapter,
    build_level_net,
    label_from_ram,
    resolve_level_key,
)
from src.training.smb_sequential import RAM_DISPLAY, RAM_WORLD

_ROOT = Path(__file__).resolve().parent.parent
_MAIN_ROOT = Path("/Users/stits/Documents/macos-emulation-and-training")

# Seven-action SMB tile scout action space (matches smb_oneshot_tiles.yaml).
_ACTIONS_7 = [[], ["right"], ["right", "A"], ["right", "B"],
              ["right", "A", "B"], ["A"], ["left"]]


def mk_ram(world: int, display: int) -> bytearray:
    """Minimal RAM blob carrying just the router bytes ($075F world, $075C level)."""
    ram = bytearray(0x800)
    ram[RAM_WORLD] = world & 0xFF
    ram[RAM_DISPLAY] = display & 0xFF
    return ram


# --- label + manifest resolution ------------------------------------------

def test_label_from_ram_is_displayed_world_level() -> None:
    assert label_from_ram(mk_ram(0, 0)) == "1-1"   # world byte 0 -> World 1
    assert label_from_ram(mk_ram(0, 1)) == "1-2"
    assert label_from_ram(mk_ram(0, 3)) == "1-4"
    assert label_from_ram(mk_ram(1, 0)) == "2-1"


def test_resolve_level_key_exact_then_default_then_none() -> None:
    labels = {"1-1", "1-4", "default"}
    assert resolve_level_key(labels, "1-1") == "1-1"      # exact wins
    assert resolve_level_key(labels, "1-3") == "default"  # falls back
    assert resolve_level_key({"1-1"}, "1-3") is None      # no default -> None


# --- hysteresis ------------------------------------------------------------

def test_hysteresis_single_frame_flicker_does_not_switch() -> None:
    h = HysteresisSwitch(k=2)
    h.reset("1-1")
    assert h.observe("1-2") is None   # pending (1 frame)
    assert h.observe("1-1") is None   # back to active -> cancels pending
    assert h.observe("1-2") is None   # pending again (only 1 frame)
    assert h.active == "1-1"          # never switched on a flicker


def test_hysteresis_switches_after_k_consecutive() -> None:
    h = HysteresisSwitch(k=2)
    h.reset("1-1")
    assert h.observe("1-2") is None   # 1st agreeing frame
    assert h.observe("1-2") == "1-2"  # 2nd -> commit
    assert h.active == "1-2"
    # A new candidate needs its own K frames from the new active label.
    assert h.observe("1-3") is None
    assert h.observe("1-3") == "1-3"


def test_hysteresis_k1_switches_immediately() -> None:
    h = HysteresisSwitch(k=1)
    h.reset("1-1")
    assert h.observe("1-2") == "1-2"


# --- obs adapters ----------------------------------------------------------

class _StubExtractor:
    """Tile extractor whose feature vector is the displayed level byte, repeated."""

    feature_dim = 8

    def extract(self, ram) -> np.ndarray:
        return np.full(self.feature_dim, int(ram[RAM_DISPLAY]), dtype=np.int8)


def test_tile_adapter_reset_fills_stack_with_current_features() -> None:
    ad = TileObsAdapter(_StubExtractor(), stack_size=4, feature_dim=8)
    obs = ad.reset(mk_ram(0, 1))          # display byte 1
    # reset fills all 4 slots with the current features -> flat all-ones, len 32.
    assert obs.shape == (32,)
    assert np.all(obs == 1)


def test_pixel_adapter_normalizes_uint8_at_forward() -> None:
    ad = PixelObsAdapter(stack_size=4, frame_size=84)
    pp = np.full((84, 84), 128, dtype=np.uint8)
    obs = ad.reset(None, pp)
    assert obs.shape == (4, 84, 84)
    t = ad.to_tensor(obs, torch.device("cpu"))
    assert t.shape == (1, 4, 84, 84)
    assert abs(float(t.max()) - 128.0 / 255.0) < 1e-4   # uint8 divided by 255


# --- controller: switch resets stack + hidden -----------------------------

class _StubNet:
    """Stateless net that always argmaxes to a fixed action index."""

    def __init__(self, action_idx: int) -> None:
        self._idx = action_idx

    def forward_ac(self, x):
        logits = torch.full((x.shape[0], 7), -1.0)
        logits[:, self._idx] = 10.0
        return logits, torch.zeros(x.shape[0])


def _tile_levelnet(label, action_idx):
    from src.training.composite_policy import LevelNet
    return LevelNet(
        label=label, net=_StubNet(action_idx), is_recurrent=False, mode="tile",
        bitmasks=(0, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40), device=torch.device("cpu"),
        extractor=_StubExtractor(), stack_size=4, feature_dim=8,
    )


def test_controller_switch_resets_stack_and_records_segments() -> None:
    net_a = _tile_levelnet("1-1", action_idx=1)
    net_b = _tile_levelnet("1-2", action_idx=2)
    nets = {"1-1": net_a, "1-2": net_b, "default": net_a}
    ctrl = CompositeController(nets, nets.keys(), torch.device("cpu"), k=2)

    ctrl.begin(mk_ram(0, 0), None)        # 1-1
    assert ctrl.active_net is net_a
    # Fill the 1-1 stack with a few steps of live pushes.
    for step in range(1, 4):
        assert ctrl.observe(mk_ram(0, 0), None, step) is None
    assert ctrl.switches == 0

    # Two consecutive 1-2 frames -> committed switch.
    assert ctrl.observe(mk_ram(0, 1), None, 4) is None      # pending
    switched = ctrl.observe(mk_ram(0, 1), None, 5)          # commit
    assert switched == "1-2"
    assert ctrl.active_net is net_b
    assert ctrl.switches == 1
    # The incoming net's stack was rebuilt + reset-seeded from the CURRENT frame
    # (display byte 1) — all slots equal, no stale 1-1 features, no zero pad.
    assert np.all(ctrl.obs == 1)
    assert ctrl.hidden is None            # fresh hidden (stateless -> None)
    # Segments: 1-1 exited "advanced" at step 5, new 1-2 segment opened.
    assert ctrl.segments[0].level == "1-1"
    assert ctrl.segments[0].exit_reason == "advanced"
    assert ctrl.segments[0].exited_step == 5
    assert ctrl.segments[1].level == "1-2"
    assert ctrl.segments[1].entered_step == 5


def test_controller_flicker_never_switches_net() -> None:
    net_a = _tile_levelnet("1-1", 1)
    net_b = _tile_levelnet("1-2", 2)
    ctrl = CompositeController({"1-1": net_a, "1-2": net_b}, ["1-1", "1-2"],
                               torch.device("cpu"), k=2)
    ctrl.begin(mk_ram(0, 0), None)
    ctrl.observe(mk_ram(0, 1), None, 1)   # pending 1-2
    ctrl.observe(mk_ram(0, 0), None, 2)   # flicker back to 1-1 -> cancel
    ctrl.observe(mk_ram(0, 1), None, 3)   # pending 1-2 (only 1 frame)
    ctrl.observe(mk_ram(0, 0), None, 4)   # back to 1-1
    assert ctrl.switches == 0
    assert ctrl.active_net is net_a


# --- build_level_net: tile + pixel, and MIXED in one controller -----------

def test_build_pixel_net_from_raw_checkpoint(tmp_path: Path) -> None:
    net = PolicyNetwork(num_actions=7, frame_stack=4, frame_size=84,
                        encoder="nature_dqn", use_layernorm=True)
    ckpt = tmp_path / "pixel.pt"
    torch.save({"net_state_dict": net.state_dict()}, ckpt)
    profile = {"action_space": _ACTIONS_7, "reinforce": {"encoder": "nature_dqn"}}
    ln = build_level_net(profile, str(ckpt), torch.device("cpu"), label="1-4")
    assert ln.mode == "pixel" and ln.stack_size == 4 and not ln.is_recurrent
    # A synthetic frame forwards to a valid action bitmask.
    ad = ln.new_adapter()
    obs = ad.reset(None, np.zeros((84, 84), dtype=np.uint8))
    bitmask, _ = ln.forward_greedy(ad.to_tensor(obs, ln.device), None)
    assert bitmask in ln.bitmasks


def test_build_tile_net_from_raw_checkpoint(tmp_path: Path) -> None:
    net = TilePolicyNetwork(num_actions=7, feature_dim=700)   # 175 * 4 stack
    ckpt = tmp_path / "tile.pt"
    torch.save({"net_state_dict": net.state_dict()}, ckpt)
    profile = {"action_space": _ACTIONS_7, "reinforce": {"encoder": "smb_tiles"}}
    ln = build_level_net(profile, str(ckpt), torch.device("cpu"), label="1-2")
    assert ln.mode == "tile" and ln.stack_size == 4 and ln.feature_dim == 175


def test_mixed_encoders_compose_in_one_controller(tmp_path: Path) -> None:
    # Tile specialist for 1-1, pixel specialist for 1-4 — one session, one router.
    tnet = TilePolicyNetwork(num_actions=7, feature_dim=700)
    tck = tmp_path / "t.pt"
    torch.save({"net_state_dict": tnet.state_dict()}, tck)
    pnet = PolicyNetwork(num_actions=7, frame_stack=4, encoder="nature_dqn")
    pck = tmp_path / "p.pt"
    torch.save({"net_state_dict": pnet.state_dict()}, pck)
    tile_ln = build_level_net(
        {"action_space": _ACTIONS_7, "reinforce": {"encoder": "smb_tiles"}},
        str(tck), torch.device("cpu"), "1-1")
    pix_ln = build_level_net(
        {"action_space": _ACTIONS_7, "reinforce": {"encoder": "nature_dqn"}},
        str(pck), torch.device("cpu"), "1-4")

    ctrl = CompositeController({"1-1": tile_ln, "1-4": pix_ln, "default": tile_ln},
                               ["1-1", "1-4", "default"], torch.device("cpu"), k=2)
    pp = np.zeros((84, 84), dtype=np.uint8)
    # The real smb_tiles extractor requires bytes (production ram_snapshot is
    # always bytes); mk_ram returns a bytearray, so convert for the tile path.
    ctrl.begin(bytes(mk_ram(0, 0)), pp)        # tile 1-1
    assert ctrl.active_net.mode == "tile"
    b1 = ctrl.act()
    assert b1 in tile_ln.bitmasks
    # Route into 1-4 (pixel) via two agreeing frames.
    ctrl.observe(bytes(mk_ram(0, 3)), pp, 1)
    ctrl.observe(bytes(mk_ram(0, 3)), pp, 2)
    assert ctrl.active_net.mode == "pixel"
    b2 = ctrl.act()                            # pixel forward, same session
    assert b2 in pix_ln.bitmasks


# --- ROM-gated acceptance smoke -------------------------------------------

_SMB_ROM = _ROOT / "roms" / "Super Mario Bros. (World).nes"
_MAIN_ROM = _MAIN_ROOT / "roms" / "Super Mario Bros. (World).nes"
_MAIN_START = _MAIN_ROOT / "roms" / "Super Mario Bros. (World)_start.state.bin"
_CHAMP = (_MAIN_ROOT / "checkpoints" / "super_mario_bros_one_shot_tiles"
          / "vanilla_ppo_iter_01580.pt")
_TILES_PROFILE = _ROOT / "configs" / "smb_oneshot_tiles.yaml"


@pytest.mark.slow
@pytest.mark.skipif(
    not ((_SMB_ROM.exists() or _MAIN_ROM.exists()) and _CHAMP.exists()),
    reason="needs the SMB ROM + the tile scout champion checkpoint",
)
def test_composite_smoke_champion_everywhere(tmp_path: Path) -> None:
    rom = _SMB_ROM if _SMB_ROM.exists() else _MAIN_ROM
    entry = {"ckpt": str(_CHAMP), "profile": str(_TILES_PROFILE)}
    manifest = {
        "name": "smoke_champion_everywhere",
        "game": "mario",
        "levels": {lvl: dict(entry) for lvl in
                   ("1-1", "1-2", "1-3", "1-4", "default")},
    }
    mpath = tmp_path / "composite.yaml"
    import yaml
    mpath.write_text(yaml.safe_dump(manifest))

    out = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "eval_composite.py"),
         "--manifest", str(mpath), "--episodes", "2", "--max-steps", "1600",
         "--rom", str(rom), "--start-state", str(_MAIN_START),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=900,
    )
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["status"] == "ok", result
    assert result["hierarchical"] is True
    for key in ("seq_clear_rate", "furthest_seq_level", "warp_rate",
                "per_level", "mean_switches"):
        assert key in result, (key, result)
    # Cold run from 1-1 must clear 1-1 and enter 1-2 (the champion does), so the
    # session's furthest sequential reach is at least 1-2 and a 1-2 segment exists.
    assert result["furthest_seq"] is not None
    assert tuple(result["furthest_seq"]) >= (1, 2), result["furthest_seq"]
    assert "1-2" in result["per_level"], result["per_level"]
    # Plays THROUGH the 1-1 flag (does not stop on the latch at ~step 415).
    assert result["mean_length"] > 450, result
