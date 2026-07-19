"""Level-keyed hierarchical (composite) policy for the SMB one-shot cold run.

The campaign's definition of done is a single COLD emulator session that starts
at 1-1 and beats World 1 in order. Per-level specialists exist (or are being
consolidated); this module composes them into that one cold session by switching
which policy checkpoint produces the action based on the CURRENT displayed level
read from RAM. It is the push's banker: reported honestly as a *hierarchical*
policy (a router over per-level nets), never as a single monolithic net.

Design (the pieces are pure/testable; only the run loop touches the pool):

  * `label_from_ram(ram)` -> ``"1-1"`` .. ``"8-4"`` — the DISPLAYED (world, level)
    using `smb_sequential`'s RAM constants (World byte ``$075F`` + displayed
    level byte ``$075C``, both +1). This is the router key and the manifest key.
    It intentionally uses the displayed level (not the ``$0760`` area byte), so a
    1-2 entrance and its underground are one "1-2" specialist bucket.

  * `resolve_level_key(labels, label)` — manifest resolution: the exact level,
    else the ``"default"`` fallback, else ``None`` (fail loud).

  * `HysteresisSwitch(k)` — a transient one-frame byte flicker (RAM updates
    mid-transition) must not thrash nets, so a candidate label only becomes
    active after it agrees for ``k`` CONSECUTIVE frames (default 2). Returning to
    the active label cancels a pending switch.

  * `TileObsAdapter` / `PixelObsAdapter` — thin per-level observation builders so
    MIXED encoders compose in one session (a tile specialist for 1-2 + a pixel
    specialist for 1-4). Each mirrors how the trainer/eval builds that encoder's
    obs: tile = `TileFeatureStacker` over `extractor.extract(ram)`; pixel =
    `FrameStacker` over the pool's Rust-preprocessed 84x84 frame (the same NEON
    output training saw — CPU re-preprocessing would drift from the trained
    distribution). Both are seeded fresh (reset) on level entry.

  * `CompositeController` — ties it together: preload every net up front, per
    step read RAM, route via hysteresis, and on a committed level change build a
    FRESH adapter for the incoming net and reset it with the CURRENT observation
    (see the stack-reset note on `observe`). Records per-level segment stats.

Transition / stack-reset choice (documented): on a committed switch the incoming
net's frame/feature stack is REBUILT and reset-seeded from the present frame —
NOT carried across from the outgoing net. Two reasons: (1) the stacker is
encoder-specific (a tile feature vector and an 84x84 pixel stack are not
interchangeable buffers), and (2) a level transition is a scene cut — carrying
the previous level's frames would inject out-of-distribution temporal context
into the incoming specialist. Reset-seed (fill all slots with the current obs)
matches `TileFeatureStacker.reset` / `FrameStacker.reset` and avoids a
zero-padding leak on the incoming net's first forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.emulation.frame_utils import FrameStacker, TileFeatureStacker
from src.models.policy_network import PolicyNetwork
from src.models.tile_policy import build_tile_policy_from_checkpoint
from src.training.profile_utils import action_space_to_bitmasks, resolve_encoder
# Reuse smb_sequential's RAM constants + label helper — never copy them, so the
# composite router reads the world/level exactly where the sequential predicate
# and the trainer's curriculum do.
from src.training.smb_sequential import (
    RAM_AREA,
    RAM_DISPLAY,
    RAM_WORLD,
    SequentialTracker,
    level_label,
)

# Pixel encoders route to the CNN `PolicyNetwork`; anything else is a tile
# encoder resolved via `resolve_encoder` into the small MLP tile policy.
PIXEL_ENCODERS = ("nature_dqn", "impala")

DEFAULT_LEVEL_KEY = "default"


# --------------------------------------------------------------------------
# Level-label helpers (the router key)
# --------------------------------------------------------------------------

def label_from_ram(ram) -> str:
    """DISPLAYED ``"world-level"`` from RAM (e.g. ``"1-2"``).

    World byte ``$075F`` and displayed level byte ``$075C`` are both 0-indexed
    (World 1 == 0), so we +1 each — the same displayed semantics
    `smb_sequential._displayed` encodes. This is the router / manifest key.
    """
    world = int(ram[RAM_WORLD]) + 1
    level = int(ram[RAM_DISPLAY]) + 1
    return f"{world}-{level}"


def _label_rank(label: Optional[str]) -> tuple[int, int]:
    """(world, level) ints for depth comparison; unparseable -> ``(0, 0)``."""
    if not label:
        return (0, 0)
    try:
        w, l = label.split("-", 1)
        return (int(w), int(l))
    except (ValueError, AttributeError):
        return (0, 0)


def resolve_level_key(labels, label: str, area: Optional[int] = None) -> Optional[str]:
    """Resolve a router label to a manifest key.

    Area-granular key ``"<label>@<area>"`` wins when present (a level's
    sub-scenes can need different specialists: 1-2's above-ground vestibule
    is champion terrain while its underground belongs to the June
    specialist); then the exact level; then the ``"default"`` catch-all;
    otherwise ``None`` (the caller fails loud rather than silently reusing
    the wrong specialist).
    """
    labels = set(labels)
    if area is not None and f"{label}@{area}" in labels:
        return f"{label}@{area}"
    if label in labels:
        return label
    if DEFAULT_LEVEL_KEY in labels:
        return DEFAULT_LEVEL_KEY
    return None


# --------------------------------------------------------------------------
# Hysteresis (anti-thrash switch)
# --------------------------------------------------------------------------

class HysteresisSwitch:
    """Commit a level switch only after ``k`` consecutive agreeing frames.

    A single-frame candidate (a byte flicker during a level transition) never
    switches nets. Returning to the current active label mid-pending cancels the
    pending switch. `observe` returns the new active label on a committed switch,
    else ``None``.
    """

    def __init__(self, k: int = 2) -> None:
        self.k = max(1, int(k))
        self.active: Optional[str] = None
        self.pending: Optional[str] = None
        self.count = 0

    def reset(self, active: Optional[str]) -> None:
        self.active = active
        self.pending = None
        self.count = 0

    def observe(self, candidate: str) -> Optional[str]:
        if candidate == self.active:
            # Back on the active label — drop any half-formed pending switch.
            self.pending = None
            self.count = 0
            return None
        if candidate == self.pending:
            self.count += 1
        else:
            self.pending = candidate
            self.count = 1
        if self.count >= self.k:
            self.active = candidate
            self.pending = None
            self.count = 0
            return candidate
        return None


# --------------------------------------------------------------------------
# Per-level observation adapters (mixed encoders in one session)
# --------------------------------------------------------------------------

def _prep_preprocessed(pp) -> np.ndarray:
    """Normalize a pool `preprocessed` slot to an (84, 84) array.

    Mirrors `rust_pool_adapter`'s materialize: a ``(84, 168)`` uint8 fallback
    is a reinterpreted float16 buffer; everything else is already ``(84, 84)``.
    """
    a = np.asarray(pp)
    if a.dtype == np.uint8 and a.shape == (84, 168):
        a = a.view(np.float16).reshape(84, 84)
    return a


class TileObsAdapter:
    """Stacked tile-feature observation for one tile specialist.

    Mirrors the trainer's tile path: `extractor.extract(ram)` -> int8 feature
    vector, stacked oldest->newest into a flat ``(stack*feature_dim,)`` vector.
    `pp` is ignored (tile policies read RAM, never the 84x84 frame).
    """

    mode = "tile"

    def __init__(self, extractor, stack_size: int, feature_dim: int) -> None:
        self._extractor = extractor
        self._stacker = TileFeatureStacker(
            stack_size=stack_size, feature_dim=feature_dim
        )

    def reset(self, ram, pp=None) -> np.ndarray:
        return self._stacker.reset(self._extractor.extract(ram))

    def push(self, ram, pp=None) -> np.ndarray:
        return self._stacker.push(self._extractor.extract(ram))

    def to_tensor(self, obs: np.ndarray, device) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(obs[None, :])).float().to(device)


class PixelObsAdapter:
    """Stacked pixel observation for one pixel specialist.

    Mirrors the trainer's pixel path: consume the pool's Rust-preprocessed
    ``(84, 84)`` frame (the same NEON gray+resize the CNN trained on) directly
    into a `FrameStacker`. Normalization matches the trainer — uint8 obs are
    divided by 255 at forward; already-normalized float16 obs are not. `ram` is
    ignored (pixel policies read frames, never RAM).
    """

    mode = "pixel"

    def __init__(self, stack_size: int = 4, frame_size: int = 84) -> None:
        self.stack_size = stack_size
        self.frame_size = frame_size
        self._stacker: Optional[FrameStacker] = None
        self._is_uint8 = True

    def _ensure(self, pp: np.ndarray) -> None:
        if self._stacker is None:
            self._is_uint8 = pp.dtype == np.uint8
            self._stacker = FrameStacker(
                stack_size=self.stack_size,
                target_size=self.frame_size,
                dtype=pp.dtype,
            )

    def reset(self, ram, pp) -> np.ndarray:
        pp = _prep_preprocessed(pp)
        self._ensure(pp)
        return self._stacker.reset(None, preprocessed=pp)

    def push(self, ram, pp) -> np.ndarray:
        pp = _prep_preprocessed(pp)
        self._ensure(pp)
        return self._stacker.push(None, preprocessed=pp)

    def to_tensor(self, obs: np.ndarray, device) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(obs[None])).float().to(device)
        if self._is_uint8:
            t = t.div_(255.0)
        return t


# --------------------------------------------------------------------------
# Per-level net container + loader
# --------------------------------------------------------------------------

@dataclass
class LevelNet:
    """One loaded specialist: net + its encoder's action table and adapter kind."""

    label: str
    net: Any
    is_recurrent: bool
    mode: str  # "tile" | "pixel"
    bitmasks: tuple
    device: Any
    extractor: Any = None
    stack_size: int = 4
    feature_dim: int = 175

    def new_adapter(self):
        if self.mode == "tile":
            return TileObsAdapter(self.extractor, self.stack_size, self.feature_dim)
        return PixelObsAdapter(self.stack_size, 84)

    def initial_hidden(self):
        return self.net.initial_hidden(1, self.device) if self.is_recurrent else None

    def forward_greedy(self, obs_tensor: torch.Tensor, hidden):
        with torch.no_grad():
            if self.is_recurrent:
                logits, _, hidden = self.net.forward_ac_recurrent(obs_tensor, hidden)
            else:
                logits, _ = self.net.forward_ac(obs_tensor)
            idx = int(torch.argmax(logits[0]).item())
        return self.bitmasks[idx], hidden


def _pixel_frame_stack(migrated_sd: dict) -> int:
    """Frame-stack (conv in-channels) recovered structurally from a pixel sd."""
    key = "encoder.conv1.weight"
    if key in migrated_sd:
        return int(migrated_sd[key].shape[1])
    return 4


def _build_pixel_net(sd: dict, encoder: str, num_actions: int) -> PolicyNetwork:
    """Reconstruct a pixel `PolicyNetwork` from a raw policy state_dict.

    Trainer/winner pixel checkpoints store only the raw weights (no self-
    describing hparams), so frame-stack / action-count / layernorm are recovered
    structurally from the sd — no profile guessing that could mis-size the net.
    """
    migrated = PolicyNetwork._migrate_legacy_state_dict(sd)
    frame_stack = _pixel_frame_stack(migrated)
    ckpt_actions = (
        int(migrated["fc2.weight"].shape[0])
        if "fc2.weight" in migrated else num_actions
    )
    if ckpt_actions != num_actions:
        raise ValueError(
            f"pixel checkpoint has {ckpt_actions} actions but the profile's "
            f"action_space has {num_actions}; they must match."
        )
    use_layernorm = "trunk_norm.weight" in migrated
    net = PolicyNetwork(
        num_actions=num_actions,
        frame_stack=frame_stack,
        frame_size=84,
        encoder=encoder,
        use_layernorm=use_layernorm,
    )
    res = net.load_state_dict(migrated, strict=False)
    if res.missing_keys:
        raise ValueError(
            f"pixel checkpoint is missing weights for {encoder}: "
            f"{list(res.missing_keys)}; refusing a partially-initialized policy."
        )
    return net


def build_level_net(
    profile: dict, ckpt_path: str, device, label: str = "?",
) -> LevelNet:
    """Load one specialist for `label` from `ckpt_path` under `profile`.

    Dispatches on the profile's ``reinforce.encoder``: a pixel encoder builds the
    CNN `PolicyNetwork`; any tile encoder resolves via `resolve_encoder` into the
    MLP tile policy (with structural recurrent detection). Fails loud on a
    partial/mismatched load rather than scoring a half-random policy.
    """
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    rl = profile.get("reinforce", {}) or {}
    encoder = rl.get("encoder")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("net_state_dict")
    if sd is None:
        sd = state.get("state_dict", state)

    if encoder in PIXEL_ENCODERS:
        net = _build_pixel_net(sd, encoder, len(bitmasks))
        net.to(device).eval()
        return LevelNet(
            label=label, net=net, is_recurrent=False, mode="pixel",
            bitmasks=bitmasks, device=device, extractor=None,
            stack_size=net.frame_stack, feature_dim=84 * 84,
        )

    # Tile encoder path.
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim
    net, is_recurrent = build_tile_policy_from_checkpoint(
        state if "net_state_dict" in state else {"net_state_dict": sd},
        num_actions=len(bitmasks), feature_dim=stacked_dim,
    )
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise ValueError(
            f"tile checkpoint {ckpt_path} is missing weights: "
            f"{list(res.missing_keys)}; refusing a partially-initialized policy."
        )
    net.to(device).eval()
    return LevelNet(
        label=label, net=net, is_recurrent=is_recurrent, mode="tile",
        bitmasks=bitmasks, device=device, extractor=extractor,
        stack_size=stack_size, feature_dim=feature_dim,
    )


# --------------------------------------------------------------------------
# Segment bookkeeping
# --------------------------------------------------------------------------

@dataclass
class Segment:
    """One contiguous stretch the session spent on a single level."""

    level: str
    entered_step: int
    exited_step: Optional[int] = None
    exit_reason: Optional[str] = None  # advanced|regressed|died|timeout|seq_clear

    def as_dict(self) -> dict:
        steps = (
            None if self.exited_step is None
            else max(0, self.exited_step - self.entered_step)
        )
        return {
            "level": self.level,
            "entered_step": self.entered_step,
            "exited_step": self.exited_step,
            "exit_reason": self.exit_reason,
            "steps_in_level": steps,
        }


# --------------------------------------------------------------------------
# The composite controller (the router)
# --------------------------------------------------------------------------

class CompositeController:
    """Routes actions through per-level specialists over one episode.

    `nets` is keyed by MANIFEST key (``"1-1"`` .. plus ``"default"``); `labels`
    is that key set (used for resolution). Lifecycle per episode: `begin(ram, pp)`
    seeds the initial specialist; `act()` forwards it greedily; `observe(ram, pp,
    step)` advances the stack and may commit a hysteresis-gated switch; `close`
    seals the final segment.
    """

    def __init__(self, nets: dict, labels, device, k: int = 2,
                 gx_routes: Optional[dict] = None,
                 entry_opts: Optional[dict] = None) -> None:
        self.nets = nets
        self.labels = set(labels)
        self.device = device
        self.hyst = HysteresisSwitch(k=k)
        self.active_label: Optional[str] = None
        self.active_net: Optional[LevelNet] = None
        self.adapter = None
        self.hidden = None
        self.obs: Optional[np.ndarray] = None
        self.switches = 0
        self.segments: list[Segment] = []
        # Intra-level gx-keyed routing: manifest key -> sorted
        # [(at_gx, net_key), ...]. A level can hand off mid-level from one
        # specialist to the next at a disclosed x threshold (e.g. a welded
        # front-half net to a back-half net at the seam). One-way per level
        # visit; 2-frame confirm so a mid-transition gx glitch can't fire it.
        self.gx_routes = gx_routes or {}
        # Per-level ENTRY options (manifest `entry:` block): noop_pad
        # reproduces the load -> one-noop -> act replay convention for a
        # BC-pilot level net; continuous_stack rides the stack through
        # in-level scene cuts (replay_to_demos never resets its stack).
        self.entry_opts = entry_opts or {}
        self._active_key: Optional[str] = None
        self._gx_stage = 0
        self._gx_confirm = 0
        self._gx_pad = 0
        self._gx_continuous = False
        self.gx_switch_events: list[dict] = []

    def _select(self, label: str, area: Optional[int] = None) -> LevelNet:
        key = resolve_level_key(self.labels, label, area)
        if key is None:
            raise KeyError(
                f"level {label!r} has no manifest entry and no "
                f"{DEFAULT_LEVEL_KEY!r} fallback"
            )
        self._active_key = key
        return self.nets[key]

    def _gx_stage_active(self) -> bool:
        return bool(self.gx_routes.get(self._active_key)) and self._gx_stage > 0

    def begin(self, ram, pp) -> None:
        label = label_from_ram(ram)
        self.active_label = label
        self._area = int(ram[RAM_AREA])
        self.active_net = self._select(label, self._area)
        self.adapter = self.active_net.new_adapter()
        self.obs = self.adapter.reset(ram, pp)
        self.hidden = self.active_net.initial_hidden()
        self.hyst.reset(label)
        self.switches = 0
        self.segments = [Segment(level=label, entered_step=0)]
        self._gx_stage = 0
        self._gx_confirm = 0
        self._gx_pad = 0
        self._gx_continuous = False
        self.gx_switch_events = []

    def act(self) -> int:
        if self._gx_pad > 0:
            return 0  # noop through the switch pad
        t = self.adapter.to_tensor(self.obs, self.device)
        bitmask, self.hidden = self.active_net.forward_greedy(t, self.hidden)
        return bitmask

    def observe(self, ram, pp, step: int) -> Optional[str]:
        """Advance the obs for the next `act`; maybe commit a level switch.

        Returns the new active label on a committed switch, else ``None``. On a
        switch, the incoming net gets a FRESH adapter reset-seeded from the
        current (ram, pp) and a fresh hidden state (the documented scene-cut
        reset — see the module docstring).
        """
        candidate = label_from_ram(ram)
        switched = self.hyst.observe(candidate)
        if switched is not None and switched != self.active_label:
            prev = self.active_label
            reason = (
                "advanced" if _label_rank(switched) > _label_rank(prev)
                else "regressed"
            )
            self.segments[-1].exited_step = step
            self.segments[-1].exit_reason = reason
            self.active_label = switched
            self._area = int(ram[RAM_AREA])
            self._gx_stage = 0
            self._gx_confirm = 0
            self.active_net = self._select(switched, self._area)
            _opts = self.entry_opts.get(self._active_key) or {}
            self._gx_continuous = bool(_opts.get("continuous_stack", False))
            self.switches += 1
            self.segments.append(Segment(level=switched, entered_step=step))
            _pad = int(_opts.get("noop_pad", 0))
            if _pad > 0:
                # act() emits noops through the pad; the incoming net's
                # stack seeds at pad end (the replay convention).
                self._gx_pad = _pad
            else:
                self.adapter = self.active_net.new_adapter()
                self.obs = self.adapter.reset(ram, pp)
                self.hidden = self.active_net.initial_hidden()
            return switched
        # Scene cut WITHIN a level: an area-byte change (pipe entry, bonus
        # room, vestibule -> underground) is a hard visual/positional
        # discontinuity, and area-granular manifest keys may hand the new
        # scene to a different specialist. Reselect + fresh adapter/hidden
        # either way — measured: the 1-2 specialist wanders 2100+ steps
        # against a stale stack vs clearing in 573 with a fresh one from
        # the identical state, and it was never trained on the vestibule
        # at all (that scene belongs to the champion's rungs).
        area = int(ram[RAM_AREA])
        if area != self._area:
            self._area = area
            if self._gx_continuous:
                # A continuous-stack net (BC pilot, whether the level's
                # base net or a committed gx stage) replays demos whose
                # feature stack was built CONTINUOUSLY through in-level
                # area changes (replay_to_demos never resets); a scene-
                # cut reset here would desync it from its memorized
                # observation sequence at the boundary.
                self.obs = self.adapter.push(ram, pp)
                return None
            if self._gx_stage_active():
                # A committed gx-stage net rides through in-level scene
                # cuts — reselecting the base net would silently undo
                # the routed handoff. Fresh adapter either way.
                pass
            else:
                new_net = self._select(self.active_label, area)
                if new_net is not self.active_net:
                    self.active_net = new_net
                    self.switches += 1
            self.adapter = self.active_net.new_adapter()
            self.obs = self.adapter.reset(ram, pp)
            self.hidden = self.active_net.initial_hidden()
            return None
        if self._gx_pad > 0:
            # Mid-pad: the incoming stage net's stack seeds AFTER the pad
            # so its first observation matches the load -> one-noop ->
            # act replay convention its demos were minted under.
            self._gx_pad -= 1
            if self._gx_pad == 0:
                self.adapter = self.active_net.new_adapter()
                self.obs = self.adapter.reset(ram, pp)
                self.hidden = self.active_net.initial_hidden()
            return None
        routes = self.gx_routes.get(self._active_key)
        if routes and self._gx_stage < len(routes):
            at_gx, net_key, noop_pad, continuous = routes[self._gx_stage]
            gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
            if gx >= at_gx:
                self._gx_confirm += 1
            else:
                self._gx_confirm = 0
            if self._gx_confirm >= 2:
                self._gx_confirm = 0
                self._gx_stage += 1
                self.active_net = self.nets[net_key]
                self.switches += 1
                self._gx_continuous = continuous
                self.gx_switch_events.append({
                    "level": self.active_label, "at_gx": at_gx,
                    "gx": gx, "step": step, "net": net_key,
                    "noop_pad": noop_pad, "continuous_stack": continuous,
                })
                if noop_pad > 0:
                    # act() emits noops for the pad; stack seeds at pad end.
                    self._gx_pad = noop_pad
                else:
                    self.adapter = self.active_net.new_adapter()
                    self.obs = self.adapter.reset(ram, pp)
                    self.hidden = self.active_net.initial_hidden()
                return None
        self.obs = self.adapter.push(ram, pp)
        return None

    def close(self, step: int, reason: str) -> None:
        if self.segments and self.segments[-1].exited_step is None:
            self.segments[-1].exited_step = step
            self.segments[-1].exit_reason = reason


# --------------------------------------------------------------------------
# Episode runner + aggregation (the run loop is the only pool-touching part)
# --------------------------------------------------------------------------

def run_episode(pool, controller: CompositeController, reward_fn, max_steps: int,
                capture_dir=None, stop_after_worlds: int = 1,
                sticky_prob: float = 0.0, start_jitter: int = 0,
                seed: int = 0, record_dir=None, episode_idx: int = 0) -> dict:
    """Play one cold episode; return its per-episode record.

    Terminal semantics mirror `eval_game.py --sequential`: play THROUGH the 1-1
    flag (never stop on the `episode_success` latch), ending only on enough
    real castle clears (`stop_after_worlds`; default 1 = the World-1 DoD, so
    existing callers are unchanged; 0 = never stop on clears, play until death
    or `max_steps` — the "how far can it go" mode), a death (`rew_done` / pool
    done), or `max_steps`.

    `capture_dir`: when set, the emulator state is saved the first time the
    controller commits a switch into each level, as
    `<capture_dir>/handoff_<label>.state` — the exact frame the incoming
    specialist takes over. These blobs are the entry states the per-link
    robustifier / consolidation weld trains and probes from.

    `sticky_prob` / `start_jitter`: the HONEST evaluation protocol (Machado
    et al. 2018). With probability `sticky_prob`, the previous action is
    repeated instead of the controller's choice; `start_jitter` holds up to
    that many noop steps before control begins, desynchronizing every enemy
    and hazard phase from the trained trajectory. Deterministic greedy eval
    structurally overstates trajectory-replay systems — a chain that only
    clears at sticky 0.0 has memorized, not learned. Both default off so
    existing callers/tests are byte-identical.
    """
    _rng = np.random.RandomState((1_000_003 * (seed + 1)) % (2**32))
    _recorded: list[int] = []
    pool.reset_all()
    reward_fn.reset()
    _captured: set = set()
    # Noop step after reset flushes the post-restore frame (matches eval_game).
    init = pool.step_all(np.zeros(1, dtype=np.uint8))
    if start_jitter > 0:
        for _ in range(int(_rng.randint(0, start_jitter + 1))):
            init = pool.step_all(np.zeros(1, dtype=np.uint8))
    sr = init[0]
    tracker = SequentialTracker()
    tracker.update(sr.ram_snapshot)
    controller.begin(sr.ram_snapshot, sr.preprocessed)

    ep_return = 0.0
    ep_max_byte = 0
    ep_level_max_gx: dict = {}
    ep_cleared = False
    steps = 0
    end_reason = "timeout"
    _last_bitmask = 0
    for step in range(max_steps):
        steps = step + 1
        bitmask = controller.act()
        if sticky_prob > 0.0 and step > 0 and _rng.rand() < sticky_prob:
            bitmask = _last_bitmask
        _last_bitmask = bitmask
        if record_dir is not None:
            # The receipt records the post-sticky STEPPED mask at the
            # single choke point, so even sticky episodes replay exactly.
            _recorded.append(int(bitmask))
        r = pool.step_all(np.array([bitmask], dtype=np.uint8))
        sr = r[0]
        ram = sr.ram_snapshot
        reward, rew_done, _ = reward_fn.compute(ram, action=int(bitmask))
        ep_return += float(reward)
        byte = (int(ram[RAM_WORLD]) << 4) | (int(ram[RAM_AREA]) & 0x0F)
        if byte > ep_max_byte:
            ep_max_byte = byte
        # Per-level death-gx telemetry: the level's horizontal frontier
        # this episode, keyed by the controller's active label — WHERE a
        # segment stalled, not just that it died.
        _gx_now = (int(ram[0x006D]) << 8) | int(ram[0x0086])
        if _gx_now > ep_level_max_gx.get(controller.active_label, 0):
            ep_level_max_gx[controller.active_label] = _gx_now
        if reward_fn.episode_success():
            ep_cleared = True
        tracker.update(ram)
        prev_label = controller.active_label
        controller.observe(ram, sr.preprocessed, step + 1)
        if (capture_dir is not None
                and controller.active_label != prev_label
                and controller.active_label not in _captured):
            _captured.add(controller.active_label)
            safe = str(controller.active_label).replace("@", "_a")
            blob = pool.save_worker_state(0)
            Path(capture_dir).mkdir(parents=True, exist_ok=True)
            (Path(capture_dir) / f"handoff_{safe}.state").write_bytes(blob)
        if stop_after_worlds and tracker.worlds_cleared >= stop_after_worlds:
            end_reason = "seq_clear"
            break
        if rew_done or sr.done:
            end_reason = "died"
            break
    controller.close(steps, end_reason)

    record = {
        "seq_clear": tracker.seq_clear,
        "warp_taken": tracker.warp_taken,
        "furthest_seq": tracker.furthest_seq,
        "furthest_any": tracker.furthest_any,
        "worlds_cleared": tracker.worlds_cleared,
        "furthest_nowarp": tracker.furthest_nowarp,
        "cleared": ep_cleared,
        "return": ep_return,
        "length": steps,
        "max_byte": ep_max_byte,
        "switches": controller.switches,
        "gx_switches": controller.gx_switch_events,
        "level_max_gx": ep_level_max_gx,
        "end_reason": end_reason,
        "segments": [s.as_dict() for s in controller.segments],
    }
    if record_dir is not None:
        import json as _json
        rd = Path(record_dir)
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / f"ep{episode_idx:03d}_actions.npy",
                np.array(_recorded, dtype=np.uint8))
        (rd / f"ep{episode_idx:03d}_meta.json").write_text(_json.dumps({
            "seed": seed, "sticky_prob": sticky_prob,
            "start_jitter": start_jitter, "steps": len(_recorded),
            "end_reason": end_reason,
            "worlds_cleared": record.get("worlds_cleared", 0),
            "gx_switches": record["gx_switches"],
        }, indent=1))
        record["receipt"] = str(rd / f"ep{episode_idx:03d}_actions.npy")
    return record


def _per_level_breakdown(records: list[dict]) -> dict:
    """Aggregate segment stats per level across episodes."""
    out: dict[str, dict] = {}
    for rec in records:
        for seg in rec["segments"]:
            lvl = seg["level"]
            d = out.setdefault(lvl, {
                "entered": 0, "cleared": 0, "died": 0, "timeout": 0,
                "regressed": 0, "first_entered_step": None,
                "_steps_sum": 0, "_steps_n": 0,
            })
            d["entered"] += 1
            reason = seg["exit_reason"]
            if reason in ("advanced", "seq_clear"):
                d["cleared"] += 1
            elif reason == "died":
                d["died"] += 1
            elif reason == "timeout":
                d["timeout"] += 1
            elif reason == "regressed":
                d["regressed"] += 1
            es = seg["entered_step"]
            if d["first_entered_step"] is None or es < d["first_entered_step"]:
                d["first_entered_step"] = es
            if seg["steps_in_level"] is not None:
                d["_steps_sum"] += seg["steps_in_level"]
                d["_steps_n"] += 1
    for d in out.values():
        n = d.pop("_steps_n")
        s = d.pop("_steps_sum")
        d["mean_steps_in_level"] = (s / n) if n else 0.0
    return out


def summarize(records: list[dict], *, manifest_path: str, game: str,
              n_episodes: int) -> dict:
    """Build the composite result JSON (eval_game-shaped + per-level breakdown)."""
    def _fmax(vals):
        best = None
        for v in vals:
            if v is not None and (best is None or tuple(v) > tuple(best)):
                best = v
        return best

    returns = [r["return"] for r in records]
    lengths = [r["length"] for r in records]
    max_bytes = [r["max_byte"] for r in records]
    best_seq = _fmax(r["furthest_seq"] for r in records)
    best_any = _fmax(r["furthest_any"] for r in records)

    return {
        "game": game,
        "status": "ok",
        "hierarchical": True,
        "manifest": manifest_path,
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_byte_seen": int(max(max_bytes)) if max_bytes else 0,
        "mean_max_byte": float(np.mean(max_bytes)) if max_bytes else 0.0,
        # 1-1-latch rate, kept for back-compat with eval_game's `clear_rate`.
        "clear_rate": sum(r["cleared"] for r in records) / max(1, n_episodes),
        # THE campaign metric: sequential World-1 clear rate + furthest reach.
        "seq_clear_rate": sum(r["seq_clear"] for r in records) / max(1, n_episodes),
        "warp_rate": sum(r["warp_taken"] for r in records) / max(1, n_episodes),
        "furthest_seq_level": level_label(best_seq),
        "furthest_any_level": level_label(best_any),
        "furthest_seq": list(best_seq) if best_seq is not None else None,
        "furthest_any": list(best_any) if best_any is not None else None,
        # Beyond World 1: max castle-clear count and the deepest no-warp reach.
        "worlds_cleared": max(
            (r.get("worlds_cleared", 0) for r in records), default=0
        ),
        "furthest_nowarp_level": level_label(
            _fmax(r.get("furthest_nowarp") for r in records)
        ),
        "mean_switches": (
            float(np.mean([r["switches"] for r in records])) if records else 0.0
        ),
        "per_level": _per_level_breakdown(records),
        "episodes": [
            {
                "seq_clear": r["seq_clear"],
                "warp_taken": r["warp_taken"],
                "worlds_cleared": r.get("worlds_cleared", 0),
                "furthest_nowarp_level": level_label(r.get("furthest_nowarp")),
                "end_reason": r["end_reason"],
                "length": r["length"],
                "furthest_seq_level": level_label(r["furthest_seq"]),
                "switches": r["switches"],
                "gx_switches": r.get("gx_switches", []),
                "level_max_gx": r.get("level_max_gx", {}),
                "segments": r["segments"],
            }
            for r in records
        ],
    }
