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
    _per_level_breakdown,
    build_level_net,
    label_from_ram,
    resolve_level_key,
)
from src.training.smb_sequential import RAM_AREA, RAM_DISPLAY, RAM_WORLD
from tests.skip_gates import requires

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


class _AreaKeyedExtractor:
    """Tile extractor whose feature is the internal area byte ($0760),
    repeated. Lets a test tell a `push` (history preserved across an
    in-level scene cut) apart from a `reset` (history wiped) even though
    the DISPLAYED level (world/level bytes) never changes."""

    feature_dim = 8

    def extract(self, ram) -> np.ndarray:
        return np.full(self.feature_dim, int(ram[RAM_AREA]), dtype=np.int8)


def _area_levelnet(label, action_idx):
    from src.training.composite_policy import LevelNet
    return LevelNet(
        label=label, net=_StubNet(action_idx), is_recurrent=False, mode="tile",
        bitmasks=(0, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40), device=torch.device("cpu"),
        extractor=_AreaKeyedExtractor(), stack_size=4, feature_dim=8,
    )


def test_begin_honors_continuous_stack_entry_opt_like_observe_does() -> None:
    """An isolated per-link probe (begin() straight into a BC-pilot level from
    a captured handoff state, e.g. runs/chain_handoffs/handoff_2-2.state) must
    ride an in-level scene cut the same way a mid-chain arrival at the
    identical manifest key does. Before the fix, begin() hardcoded
    `_gx_continuous = False` regardless of entry_opts, so the very next
    area-byte change rebuilt the adapter from scratch and wiped the stack
    that had just been reset-seeded — the exact silent divergence between
    the two entry paths into the same level (begin() vs observe()'s
    mid-episode switch branch, which already threads entry_opts)."""
    net = _area_levelnet("2-2", action_idx=1)
    ctrl = CompositeController(
        {"2-2": net}, ["2-2"], torch.device("cpu"), k=2,
        entry_opts={"2-2": {"continuous_stack": True}},
    )

    ram_arrival = mk_ram(1, 1)          # world=2, level=2 -> label "2-2"
    ram_arrival[RAM_AREA] = 0
    ctrl.begin(ram_arrival, None)
    seeded_adapter = ctrl.adapter
    assert ctrl._gx_continuous is True   # consulted entry_opts, not hardcoded False
    assert np.all(ctrl.obs == 0)         # reset-seeded from area 0

    # In-level scene cut (e.g. the water entrance): area byte changes, the
    # displayed level does not, so this hits observe()'s area-change branch.
    ram_scene_cut = mk_ram(1, 1)
    ram_scene_cut[RAM_AREA] = 1
    assert ctrl.observe(ram_scene_cut, None, 1) is None

    # Fixed: the SAME adapter rides through via push() -> 3 old (0) slots +
    # 1 new (1) slot. Pre-fix, a fresh adapter().reset() would replace it
    # and fill all 4 slots with 1, wiping the just-seeded history.
    assert ctrl.adapter is seeded_adapter
    assert np.all(ctrl.obs[:24] == 0) and np.all(ctrl.obs[24:] == 1)


def test_warp_pipe_exit_is_not_counted_as_a_level_clear() -> None:
    """A level exited via a warp pipe (a World rise NOT out of an x-4 castle)
    must never inflate that level's `_per_level_breakdown["cleared"]` count —
    the exact warp-vs-clear distinction the headline seq_clear/warp_taken
    fields exist to guard against."""
    net_a = _tile_levelnet("1-1", 1)
    net_b = _tile_levelnet("1-2", 2)
    nets = {"1-1": net_a, "1-2": net_b, "default": net_b}
    ctrl = CompositeController(nets, nets.keys(), torch.device("cpu"), k=2)

    ctrl.begin(mk_ram(0, 0), None)                    # 1-1
    ctrl.observe(mk_ram(0, 1), None, 1)               # pending 1-2
    ctrl.observe(mk_ram(0, 1), None, 2)               # commit: 1-1 -> 1-2
    assert ctrl.segments[0].exit_reason == "advanced"  # a real clear

    # 1-2's underground main leads to a warp pipe straight to World 4 — a
    # World rise NOT out of the x-4 castle (display level stays 2, not 4).
    ctrl.observe(mk_ram(3, 0), None, 3)               # pending 4-1
    ctrl.observe(mk_ram(3, 0), None, 4)               # commit: 1-2 -> 4-1
    assert ctrl.segments[1].level == "1-2"
    assert ctrl.segments[1].exit_reason == "warped"    # NOT "advanced"
    ctrl.close(5, "timeout")

    record = {
        "seq_clear": False, "warp_taken": True, "furthest_seq": (1, 2),
        "furthest_any": (4, 1), "worlds_cleared": 0, "furthest_nowarp": (1, 2),
        "cleared": False, "return": 0.0, "length": 5, "max_byte": 0,
        "switches": ctrl.switches, "gx_switches": [], "level_max_gx": {},
        "end_reason": "timeout",
        "segments": [s.as_dict() for s in ctrl.segments],
    }

    per_level = _per_level_breakdown([record])
    assert per_level["1-1"]["cleared"] == 1
    assert per_level["1-2"]["cleared"] == 0            # the bug: was 1
    assert per_level["1-2"]["warped"] == 1


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


# --- provenance: per-level checkpoint content hash -------------------------

# roms/ is gitignored (.gitignore:65). eval_composite returns
# status="no_start_state" rather than raising when the start state is
# absent, so without this gate a clean clone reads the miss as an
# assertion failure on "ok" instead of a missing input.
@requires("roms/Super Mario Bros. (World)_start.state.bin")
def test_result_records_content_hash_of_loaded_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """checkpoints/ is gitignored and can drift in place (same path, new
    weights) without the manifest or git_commit changing. A banked eval row
    must carry the actually-loaded checkpoint's own content hash so two runs
    of the identical manifest at the identical commit are distinguishable."""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import eval_composite  # noqa: E402
    import yaml

    net = TilePolicyNetwork(num_actions=7, feature_dim=700)
    ckpt = tmp_path / "specialist.pt"
    torch.save({"net_state_dict": net.state_dict()}, ckpt)
    profile_path = _ROOT / "configs" / "smb_oneshot_tiles.yaml"

    manifest_path = tmp_path / "composite.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "name": "hash_probe",
        "game": "mario",
        "levels": {"1-1": {"ckpt": str(ckpt), "profile": str(profile_path)}},
    }))
    rom_path = tmp_path / "fake.rom"
    rom_path.write_bytes(b"stand-in bytes; the stubbed pool never reads this")

    class _StubPool:
        def __init__(self, *a, **kw) -> None: ...
        def start(self) -> None: ...
        def shutdown(self) -> None: ...

    stub_record = {
        "seq_clear": False, "warp_taken": False, "furthest_seq": None,
        "furthest_any": None, "worlds_cleared": 0, "furthest_nowarp": None,
        "cleared": False, "return": 0.0, "length": 1, "max_byte": 0,
        "switches": 0, "gx_switches": [], "level_max_gx": {}, "segments": [],
        "end_reason": "test_stub",
    }
    monkeypatch.setattr(eval_composite, "RustPool", _StubPool)
    monkeypatch.setattr(eval_composite, "run_episode",
                         lambda *a, **kw: dict(stub_record))

    def _run():
        return eval_composite.eval_composite(
            manifest_path, episodes=1, max_steps=10, seed=0,
            rom=str(rom_path),
            start_state=str(_ROOT / "roms" / "Super Mario Bros. (World)_start.state.bin"),
            out_dir=str(tmp_path / "out"),
        )

    result = _run()
    assert result["status"] == "ok", result
    import hashlib
    expected = hashlib.md5(ckpt.read_bytes()).hexdigest()
    assert result["level_checkpoints"]["1-1"]["ckpt"] == str(ckpt)
    assert result["level_checkpoints"]["1-1"]["profile"] == str(profile_path)
    assert result["level_checkpoints"]["1-1"]["ckpt_md5"] == expected

    # The exact drift this campaign was already bitten by: the SAME path is
    # overwritten in place with different weights, while the manifest (and
    # therefore git_commit, rom_md5, levels) is untouched. The recorded hash
    # is the only field that may change, and it MUST change.
    torch.save({"net_state_dict": TilePolicyNetwork(
        num_actions=7, feature_dim=700).state_dict()}, ckpt)
    result2 = _run()
    assert result2["level_checkpoints"]["1-1"]["ckpt_md5"] != expected


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
