"""Latent-cell archive abstraction: a VQ-VAE codebook over RAM windows,
used as an alternative Go-Explore cell key that carries no spatial
position at all.

WHY THIS EXISTS
----------------
`go_explore.py`'s existing cell functions (`ram_bytes_cell`,
`ram_downsample_cell`) key archive cells on coarse spatial position
(bucketed X/Y bytes, or a downsampled RAM signature that is still
dominated by position). The research synthesis on the combat class
(docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md, v19 round, on
Castlevania's block-3 hall) names the failure mode precisely: for
committed-action combat, spatial position aliases exactly the
distinctions that decide survival — attack animation frame, knockback
trajectory, invulnerability window, enemy phase. Two states standing at
the identical (x, y) can be about to die or perfectly safe, and a
position-keyed archive cannot tell them apart.

The recommendation this module implements: remove spatial position from
the cell key entirely and replace it with a discrete latent code over a
short window of RAM (and optionally APU channel state), learned by a
small VQ-VAE (van den Oord, Vinyals & Kavukcuoglu 2017, "Neural Discrete
Representation Learning"). The codebook id IS the cell key.

WHAT THIS MODULE IS NOT
------------------------
This is the abstraction only:

  * It does not touch `scripts/go_explore_solve.py`. Swapping a
    position-keyed `cell_fn` for `LatentCellCodebook.encode_to_cell` in
    `GoExploreArchive` is a separate, explicitly gated integration step
    — see "INTEGRATION (not wired here)" below for the exact shape of
    that swap.
  * It does not train on real gameplay. The encoder trains on recorded
    RAM windows, which requires either the live emulator or a bank of
    already-recorded traces — this lane owns neither. Training here is
    exercised only against synthetic tensors in tests. The deferred,
    real training command is documented below.
  * It authors no game-specific semantics. The encoder sees only raw
    RAM bytes (and, optionally, raw APU channel values) already
    normalized to [0, 1] — no address is privileged, no byte is named,
    no per-game meaning is assumed anywhere in this file. This is a
    point where an LLM's NES priors are tempting to lean on (e.g. "byte
    X is usually the invulnerability timer on this mapper") — that
    temptation is exactly what the purity line in CLAIMS.md forbids;
    nothing in this module or its tests reaches for it.

CODEBOOK COLLAPSE AND THE KILL TABLE
--------------------------------------
A VQ-VAE codebook can collapse: a handful of codewords absorb all
traffic and the rest sit dead forever, which would silently turn this
back into a low-resolution version of the position-keyed archive it was
built to replace. Two defenses:

  1. k-means++-style reinitialization of dead codewords (see
     `LatentCellCodebook.reinit_dead_codes`), run automatically every
     `config.dead_code_reinit_interval` training steps.
  2. The diagnostics the research round's kill table needs, computed as
     pure functions over a (timestamp, cell_id) event stream so they are
     testable without a live encoder or the emulator: `occupancy`,
     `dead_code_count`, `discovery_rate_per_hour`, and
     `check_discovery_kill_criterion`, which implements the v19 round's
     pre-registered kill exactly: "fewer than 2 new latent cells per
     hour for 3 consecutive hours means the encoder has converged on a
     subspace that cannot solve the problem."

INTEGRATION (not wired here)
------------------------------
`GoExploreArchive.__init__` (go_explore.py) takes a
`cell_fn: Callable[[bytes], CellKey]` — a RAM snapshot in, a hashable
cell key out. `LatentCellCodebook.encode_to_cell` almost has that shape
already; the only adapter a caller needs is something that keeps the
trailing N-frame window and calls it each time a RAM snapshot arrives:

    import collections
    import numpy as np
    from src.training.latent_cells import LatentCellCodebook, LatentCellConfig
    from src.training.go_explore import GoExploreArchive

    codebook = LatentCellCodebook(LatentCellConfig())
    window_buf = collections.deque(maxlen=codebook.config.window_frames)

    def latent_cell_fn(ram: bytes):
        window_buf.append(np.frombuffer(ram, dtype=np.uint8))
        if len(window_buf) < codebook.config.window_frames:
            return None  # not enough history yet; caller decides how to
                          # treat a not-yet-warmed-up cell (e.g. skip
                          # recording until the buffer fills)
        return codebook.encode_to_cell(np.stack(window_buf))

    archive = GoExploreArchive(cell_fn=latent_cell_fn)

That is a drop-in, config-gated alternative to `ram_bytes_cell` /
`ram_downsample_cell` for archives where position aliases survival
(the combat classes v19 targets). It is deliberately not made here.

DEFERRED — the encoder needs recorded RAM windows (from the live
emulator or a banked trace file), which this lane does not run:

    # (deferred; requires emulator or banked traces — NOT run in this lane)
    python scripts/train_latent_cells.py \\
        --traces runs/<game>/ram_traces/*.npz \\
        --window-frames 16 --latent-dim 64 --codebook-size 512 \\
        --epochs 50 --out checkpoints/latent_cells/<game>_vqvae.pt

`scripts/train_latent_cells.py` does not exist yet. Its job would be
narrow: load recorded (T, 2048) RAM arrays (plus optional APU arrays),
slice sliding windows, and call `LatentCellCodebook.train_step` in a
loop, checkpointing the model and periodically dumping a discovery
event log for `check_discovery_kill_criterion`. That script is
scaffolding around this module, not new modeling, and building it is
out of scope for this lane too — it is named here only so the deferred
step has an exact command, per the campaign's documentation rule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# van den Oord et al. 2017 do not hard-cap the codebook; this repo does,
# per the v19 research round's design ("codebook hard-capped at
# K=512") so a runaway config can't quietly blow up the archive's
# memory footprint or the k-NN cost of every `encode_to_cell` call.
LATENT_CODEBOOK_HARD_CAP = 512

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]]


def _as_float_array(x: ArrayLike) -> np.ndarray:
    """Accept a numpy array, a torch tensor, or nested sequences and
    return a float32 numpy array. Centralizes the one bit of input
    flexibility this module offers so shape/dtype checks downstream can
    assume a plain ndarray."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


@dataclass
class LatentCellConfig:
    """Everything the encoder, the codebook, and the training loop need.

    Defaults match the research round's spec: a 16-frame window, latent
    dim 64, codebook capped at 512. `ram_bytes` defaults to the NES's 2
    KB CPU RAM (the same convention `go_explore.py`'s cell functions
    use). `apu_channels` is 0 by default (RAM-only); set it > 0 to also
    fold per-frame APU channel state into the window, one row of
    `apu_channels` values per frame, concatenated onto that frame's RAM
    row before flattening.
    """

    window_frames: int = 16
    ram_bytes: int = 2048
    apu_channels: int = 0
    latent_dim: int = 64
    codebook_size: int = 512
    hidden_dim: int = 256
    commitment_cost: float = 0.25
    # 0 disables automatic reinit inside train_step (manual-only via
    # reinit_dead_codes). A code is "dead" for a reinit pass if it was
    # selected fewer than dead_code_usage_threshold times since the
    # previous pass (or since construction, for the first pass).
    dead_code_reinit_interval: int = 200
    dead_code_usage_threshold: int = 1
    lr: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.codebook_size < 1:
            raise ValueError("codebook_size must be >= 1")
        if self.codebook_size > LATENT_CODEBOOK_HARD_CAP:
            raise ValueError(
                f"codebook_size={self.codebook_size} exceeds the hard cap "
                f"K={LATENT_CODEBOOK_HARD_CAP} (v19 research round design "
                "constraint — raise LATENT_CODEBOOK_HARD_CAP deliberately "
                "if this ever needs to change, don't just pass a bigger "
                "number)."
            )
        if self.window_frames < 1:
            raise ValueError("window_frames must be >= 1")
        if self.ram_bytes < 1:
            raise ValueError("ram_bytes must be >= 1")
        if self.apu_channels < 0:
            raise ValueError("apu_channels must be >= 0")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.dead_code_reinit_interval < 0:
            raise ValueError("dead_code_reinit_interval must be >= 0")
        if self.dead_code_usage_threshold < 0:
            raise ValueError("dead_code_usage_threshold must be >= 0")

    @property
    def frame_dim(self) -> int:
        return self.ram_bytes + self.apu_channels

    @property
    def input_dim(self) -> int:
        return self.window_frames * self.frame_dim


class _VQVAE(nn.Module):
    """3-layer MLP encoder, 3-layer MLP decoder, and a learned codebook.

    Vector quantization is nearest-neighbor lookup in latent space with
    the straight-through gradient estimator (van den Oord et al. 2017):
    the encoder gets a gradient as if the quantized vector were its own
    output, so the reconstruction loss can still shape the encoder even
    though `argmin` itself is non-differentiable.
    """

    def __init__(self, config: LatentCellConfig) -> None:
        super().__init__()
        d_in, d_h, d_z = config.input_dim, config.hidden_dim, config.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_in),
        )
        self.codebook = nn.Embedding(config.codebook_size, d_z)
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / config.codebook_size,
            1.0 / config.codebook_size,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest codeword (squared Euclidean) for each row of z_e.

        Returns (z_q, indices): z_q has the same shape as z_e; indices
        is a 1D LongTensor of codebook ids, always in
        [0, config.codebook_size) — the codebook never grows past its
        configured size, so distinct inputs beyond that many clusters
        necessarily collide onto shared ids (the intended, documented
        behavior of a hard-capped codebook, not a bug).
        """
        codebook = self.codebook.weight
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices)
        return z_q, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)


@dataclass
class CellDiscoveryEvent:
    """One row of an annotated discovery log: a cell key was produced at
    a point in time, plus whether it was the first time that exact key
    had ever appeared in the stream. Built by `build_discovery_log`, not
    meant to be constructed by hand."""

    timestamp_s: float
    cell_id: int
    is_new: bool


def build_discovery_log(
    events: Sequence[tuple[float, int]],
) -> list[CellDiscoveryEvent]:
    """Turn a raw (timestamp_s, cell_id) stream — e.g. one row per
    `encode_to_cell` call during a run, or a synthetic stub in a test —
    into a discovery log annotated with first-sighting flags. Pure
    function of the stream: this is exactly the decoupling that makes
    the kill-table diagnostics testable without a live encoder or the
    emulator. Assumes `events` is already sorted by timestamp (the way
    a real run would emit it); out-of-order input is not resorted."""
    seen: set[int] = set()
    log: list[CellDiscoveryEvent] = []
    for ts, cell_id in events:
        is_new = cell_id not in seen
        if is_new:
            seen.add(cell_id)
        log.append(CellDiscoveryEvent(timestamp_s=float(ts), cell_id=int(cell_id), is_new=is_new))
    return log


def occupancy(log: Sequence[CellDiscoveryEvent], codebook_size: int) -> float:
    """Fraction of the codebook that has EVER been used, in [0, 1]."""
    if codebook_size <= 0:
        return 0.0
    distinct = len({e.cell_id for e in log})
    return min(distinct / codebook_size, 1.0)


def dead_code_count(log: Sequence[CellDiscoveryEvent], codebook_size: int) -> int:
    """Codebook entries never seen anywhere in the log, i.e. entries
    with zero lifetime usage. (This is the whole-log notion of "dead";
    `LatentCellCodebook.dead_code_count` on the live model instead
    tracks usage *since the last reinit pass*, which is the quantity
    the reinit trigger itself needs.)"""
    distinct = len({e.cell_id for e in log})
    return max(codebook_size - distinct, 0)


def discovery_rate_per_hour(
    log: Sequence[CellDiscoveryEvent],
    window_hours: float = 1.0,
    now_s: Optional[float] = None,
) -> float:
    """New distinct cells discovered per hour, over the trailing
    `window_hours` ending at `now_s` (defaults to the last event's
    timestamp, i.e. "the rate right now")."""
    if not log:
        return 0.0
    if now_s is None:
        now_s = log[-1].timestamp_s
    window_start = now_s - window_hours * 3600.0
    n_new = sum(1 for e in log if e.is_new and window_start < e.timestamp_s <= now_s)
    return n_new / window_hours if window_hours > 0 else 0.0


@dataclass
class KillCheckResult:
    triggered: bool
    bucket_starts_s: list[float] = field(default_factory=list)
    rates_per_hour: list[float] = field(default_factory=list)
    reason: str = ""


def check_discovery_kill_criterion(
    log: Sequence[CellDiscoveryEvent],
    *,
    rate_threshold_per_hour: float = 2.0,
    consecutive_hours: int = 3,
    bucket_hours: float = 1.0,
) -> KillCheckResult:
    """The v19 research round's pre-registered kill, computed exactly:
    "fewer than 2 new latent cells per hour for 3 consecutive hours
    means the encoder has converged on a subspace that cannot solve the
    problem." Buckets the discovery log into non-overlapping
    `bucket_hours`-wide windows spanning [first event, last event] and
    triggers when `consecutive_hours` worth of consecutive buckets all
    fall below `rate_threshold_per_hour`.

    This is analysis over an already-recorded log, same shape as the
    v20 round's envelope estimator (pure log analysis, no rollouts) — it
    is meant to be run against a real run's discovery log after the
    fact, not live inside a training step.
    """
    if not log:
        return KillCheckResult(False, [], [], "no events")
    t0 = log[0].timestamp_s
    t_last = log[-1].timestamp_s
    bucket_s = bucket_hours * 3600.0
    n_buckets = max(int((t_last - t0) // bucket_s) + 1, 1)
    bucket_starts = [t0 + b * bucket_s for b in range(n_buckets)]
    rates: list[float] = []
    for start in bucket_starts:
        end = start + bucket_s
        n_new = sum(1 for e in log if start <= e.timestamp_s < end and e.is_new)
        rates.append(n_new / bucket_hours if bucket_hours > 0 else 0.0)

    buckets_needed = max(int(round(consecutive_hours / bucket_hours)), 1) if bucket_hours > 0 else 1
    triggered = False
    trigger_at: Optional[int] = None
    for i in range(len(rates) - buckets_needed + 1):
        window = rates[i : i + buckets_needed]
        if all(r < rate_threshold_per_hour for r in window):
            triggered = True
            trigger_at = i
            break

    if triggered:
        reason = (
            f"{buckets_needed} consecutive bucket(s) starting at bucket "
            f"{trigger_at} (t={bucket_starts[trigger_at]:.0f}s) all below "
            f"{rate_threshold_per_hour}/hr discovery rate"
        )
    else:
        reason = "not triggered"
    return KillCheckResult(triggered, bucket_starts, rates, reason)


def _seeded_vqvae(config: LatentCellConfig) -> _VQVAE:
    """Construct the model under a locally seeded RNG without perturbing
    the caller's global torch RNG state (this module is a library used
    inside a much larger training process; it has no business moving
    anyone else's random stream)."""
    saved_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(config.seed)
        model = _VQVAE(config)
    finally:
        torch.random.set_rng_state(saved_state)
    return model


class LatentCellCodebook:
    """The archive-facing wrapper: owns the VQ-VAE, the optimizer, usage
    bookkeeping for dead-code detection, and the public
    `encode_to_cell` cell-key function.

    Not a torch.nn.Module itself (the model lives at `self.model`) so
    that construction, training, and encoding can all be plain method
    calls without the caller needing to know anything about
    `nn.Module` — matching the shape of `GoExploreArchive`, whose only
    game-specific input is a plain `cell_fn` closure.
    """

    def __init__(self, config: Optional[LatentCellConfig] = None) -> None:
        self.config = config or LatentCellConfig()
        self.model = _seeded_vqvae(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        k = self.config.codebook_size
        # Usage since the last reinit pass (drives dead-code detection);
        # reset to zero at the start of every pass, including the first.
        self._usage_since_reinit = torch.zeros(k, dtype=torch.long)
        # Usage since construction, ever — this is what "occupancy"
        # means for a live codebook (an entry that was used once and
        # has been dead ever since still counts as having been
        # occupied; it is not the same question as "usage since reinit
        # is this entry currently dead").
        self._lifetime_usage = torch.zeros(k, dtype=torch.long)

        # Independent RNG stream for k-means++ reinit sampling, so
        # reinit outcomes are reproducible from config.seed without
        # touching the global torch RNG (same reasoning as
        # _seeded_vqvae).
        self._reinit_gen = torch.Generator().manual_seed(self.config.seed + 1)

        self.step_count = 0
        self.reinit_pass_count = 0
        self.total_codes_reinitialized = 0

        # Raw (timestamp_s, cell_id) rows, populated only when a caller
        # passes timestamp_s to encode_to_cell. Empty by default so a
        # caller who never passes timestamps pays no memory cost.
        self.discovery_events: list[tuple[float, int]] = []

    # ---- input shaping -----------------------------------------------

    def _build_input(
        self,
        ram_window: ArrayLike,
        apu_window: Optional[ArrayLike] = None,
    ) -> torch.Tensor:
        """One window -> a flat float32 tensor of length config.input_dim.
        RAM bytes are normalized to [0, 1]; APU values are used exactly
        as given (the caller owns their normalization/units)."""
        ram_arr = _as_float_array(ram_window)
        expected_ram_shape = (self.config.window_frames, self.config.ram_bytes)
        if ram_arr.shape != expected_ram_shape:
            raise ValueError(
                f"ram_window shape {tuple(ram_arr.shape)} != expected "
                f"{expected_ram_shape} (window_frames, ram_bytes)"
            )
        ram_arr = ram_arr / 255.0

        if self.config.apu_channels > 0:
            if apu_window is None:
                raise ValueError(
                    f"config.apu_channels={self.config.apu_channels} but no "
                    "apu_window was given"
                )
            apu_arr = _as_float_array(apu_window)
            expected_apu_shape = (self.config.window_frames, self.config.apu_channels)
            if apu_arr.shape != expected_apu_shape:
                raise ValueError(
                    f"apu_window shape {tuple(apu_arr.shape)} != expected "
                    f"{expected_apu_shape} (window_frames, apu_channels)"
                )
            combined = np.concatenate([ram_arr, apu_arr], axis=1)
        else:
            if apu_window is not None:
                raise ValueError(
                    "apu_window was given but config.apu_channels == 0"
                )
            combined = ram_arr

        return torch.from_numpy(combined.reshape(-1).astype(np.float32))

    def _build_input_batch(
        self,
        ram_windows: Sequence[ArrayLike],
        apu_windows: Optional[Sequence[Optional[ArrayLike]]] = None,
    ) -> torch.Tensor:
        if apu_windows is None:
            apu_windows = [None] * len(ram_windows)
        if len(apu_windows) != len(ram_windows):
            raise ValueError("apu_windows must have the same length as ram_windows")
        rows = [
            self._build_input(rw, aw) for rw, aw in zip(ram_windows, apu_windows)
        ]
        return torch.stack(rows, dim=0)

    # ---- usage bookkeeping --------------------------------------------

    def _record_usage(self, indices: Sequence[int]) -> None:
        for idx in indices:
            self._usage_since_reinit[idx] += 1
            self._lifetime_usage[idx] += 1

    # ---- the cell-key entry point ---------------------------------------

    @torch.no_grad()
    def encode_to_cell(
        self,
        ram_window: ArrayLike,
        apu_window: Optional[ArrayLike] = None,
        timestamp_s: Optional[float] = None,
    ) -> int:
        """The archive cell key: one RAM (+ optional APU) window in, one
        codebook id out. Deterministic given fixed model weights — no
        dropout, no batchnorm, no sampling anywhere on this path, so the
        same window always returns the same id until the model is
        trained or a reinit pass moves a codeword.

        Pass `timestamp_s` to also append this call to
        `self.discovery_events`, feeding the live diagnostics
        (`discovery_log`, `occupancy`, etc.). Omit it (the default) to
        use this purely as a stateless cell-key function with no log
        growth.
        """
        x = self._build_input(ram_window, apu_window).unsqueeze(0)
        self.model.eval()
        z_e = self.model.encode(x)
        _, indices = self.model.quantize(z_e)
        cell_id = int(indices.item())
        self._record_usage([cell_id])
        if timestamp_s is not None:
            self.discovery_events.append((float(timestamp_s), cell_id))
        return cell_id

    # ---- training ------------------------------------------------------

    def train_step(
        self,
        ram_windows: Sequence[ArrayLike],
        apu_windows: Optional[Sequence[Optional[ArrayLike]]] = None,
    ) -> dict[str, float]:
        """One gradient step of the standard VQ-VAE objective (van den
        Oord et al. 2017): reconstruction loss + codebook loss (moves
        codewords toward the encoder output) + a commitment loss (moves
        the encoder output toward its codeword, weighted by
        config.commitment_cost), through the straight-through estimator.

        Runs entirely on the given synthetic/recorded tensors — it does
        not read any live game state itself; the caller supplies
        whatever windows it has (recorded traces in the deferred real
        training script, synthetic tensors in tests).

        Every `config.dead_code_reinit_interval` steps (if > 0), a
        reinit pass runs automatically using this same batch's encoder
        outputs as k-means++ candidates; the count reinitialized is
        included in the returned dict.
        """
        x = self._build_input_batch(ram_windows, apu_windows)
        self.model.train()
        z_e = self.model.encode(x)
        z_q, indices = self.model.quantize(z_e)

        # Straight-through estimator: forward value is the quantized
        # vector, but gradient flows to z_e as if quantization were the
        # identity.
        z_q_straight_through = z_e + (z_q - z_e).detach()
        recon = self.model.decode(z_q_straight_through)

        recon_loss = F.mse_loss(recon, x)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        loss = recon_loss + codebook_loss + self.config.commitment_cost * commitment_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._record_usage(indices.detach().tolist())
        self.step_count += 1

        reinit_count = 0
        interval = self.config.dead_code_reinit_interval
        if interval > 0 and self.step_count % interval == 0:
            reinit_count = self._reinit_dead_codes(z_e.detach())

        return {
            "loss": float(loss.item()),
            "recon_loss": float(recon_loss.item()),
            "codebook_loss": float(codebook_loss.item()),
            "commitment_loss": float(commitment_loss.item()),
            "reinit_count": float(reinit_count),
        }

    def reinit_dead_codes(
        self,
        ram_windows: Sequence[ArrayLike],
        apu_windows: Optional[Sequence[Optional[ArrayLike]]] = None,
    ) -> int:
        """Public, standalone entry to force a reinit pass using
        `ram_windows` (any batch — a validation slice, a stub, whatever
        the caller has) as k-means++ candidate positions, without
        stepping the optimizer. `train_step` calls the same underlying
        logic automatically every `config.dead_code_reinit_interval`
        steps; this is exposed separately so it can be exercised (and
        tested) on its own."""
        x = self._build_input_batch(ram_windows, apu_windows)
        with torch.no_grad():
            self.model.eval()
            z_e = self.model.encode(x)
        return self._reinit_dead_codes(z_e)

    def _reinit_dead_codes(self, z_e_batch: torch.Tensor) -> int:
        """k-means++-style reinit (Arthur & Vassilvitskii 2007 seeding
        rule) of codewords that were selected fewer than
        `config.dead_code_usage_threshold` times since the previous
        pass. Each dead codeword is replaced by a candidate drawn from
        `z_e_batch`, sampled with probability proportional to its
        squared distance to the nearest currently-alive codeword — so
        replacements land in under-represented regions of latent space
        instead of duplicating a codeword that is already live. A small
        amount of jitter is added so multiple dead codewords reinitialized
        in the same pass don't collapse onto the exact same point.

        Resets the since-last-reinit usage counters to zero for
        everything (both the codewords that stayed and the ones that
        were just replaced) at the end of the pass, so the next
        interval's dead-code judgment is made fresh.
        """
        usage = self._usage_since_reinit
        dead_mask = usage < self.config.dead_code_usage_threshold
        dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()
        n_dead = len(dead_indices)

        if n_dead == 0 or z_e_batch.shape[0] == 0:
            self._usage_since_reinit.zero_()
            self.reinit_pass_count += 1
            return 0

        alive_mask = ~dead_mask
        with torch.no_grad():
            codebook = self.model.codebook.weight
            for dead_idx in dead_indices:
                if alive_mask.any():
                    alive_codes = codebook[alive_mask]
                    d2 = torch.cdist(z_e_batch, alive_codes).pow(2).min(dim=1).values
                else:
                    # Everything is dead (e.g. right after construction) —
                    # fall back to uniform candidate weighting.
                    d2 = torch.ones(z_e_batch.shape[0])
                total = d2.sum()
                probs = d2 / total if total > 0 else torch.full_like(d2, 1.0 / d2.shape[0])
                choice = int(torch.multinomial(probs, 1, generator=self._reinit_gen).item())
                jitter = torch.randn(
                    codebook.shape[1], generator=self._reinit_gen
                ) * 1e-3
                codebook[dead_idx] = z_e_batch[choice] + jitter
                alive_mask[dead_idx] = True

        self._usage_since_reinit.zero_()
        self.reinit_pass_count += 1
        self.total_codes_reinitialized += n_dead
        return n_dead

    # ---- diagnostics -----------------------------------------------------

    def occupancy(self) -> float:
        """Fraction of the codebook that has EVER been selected by
        `encode_to_cell` / `train_step`, in [0, 1]."""
        used = int((self._lifetime_usage > 0).sum().item())
        return used / self.config.codebook_size

    def dead_code_count(self) -> int:
        """Codewords selected fewer than `config.dead_code_usage_threshold`
        times since the last reinit pass (or since construction, if none
        has run yet) — the live quantity the reinit trigger itself acts
        on."""
        return int((self._usage_since_reinit < self.config.dead_code_usage_threshold).sum().item())

    def discovery_log(self) -> list[CellDiscoveryEvent]:
        """The annotated discovery log built from every `encode_to_cell`
        call made with a `timestamp_s`. Feed this to `occupancy`,
        `dead_code_count`, `discovery_rate_per_hour`, or
        `check_discovery_kill_criterion` at module scope for the offline
        (pure-log) versions of the same diagnostics."""
        return build_discovery_log(self.discovery_events)
