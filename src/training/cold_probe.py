"""Cold-eval probe: score a live training net without touching its device state.

The one-shot campaign gates every phase (and selects the deliverable
`best_cold.pt`) on a *cold* sequential World-1 clear rate — greedy play from
the 1-1 start, run *through* the 1-1 flagpole, with the sequential predicate
reconstructed from RAM. That predicate lives in exactly one place,
`scripts/eval_game.py --sequential` (Lane 1). This module is the in-loop
wrapper the trainer calls: it snapshots the current policy weights to an
isolated temporary checkpoint, subprocesses `eval_game.py` to score them, and
maps the resulting JSON into a flat ``cold_*`` metrics dict.

Design constraints this satisfies:

* **Non-blocking / MPS-safe.** The score runs in a *subprocess*, so the
  trainer's CUDA/MPS context and RNG are never perturbed. Only a CPU copy of
  `net.state_dict()` crosses the boundary; the on-device training tensors are
  left untouched.
* **Isolation.** The snapshot is written into a throwaway temp directory laid
  out as ``checkpoints/<slug>/vanilla_ppo_iter_00000.pt`` and the subprocess is
  run with that temp dir as its CWD, so `eval_game.py`'s relative
  ``./checkpoints/<slug>`` resolution loads *our* snapshot and appends its
  ``eval.jsonl`` row *inside the temp dir* — the live checkpoint tree is never
  read, written, or polluted.
* **Forward-compatible & standalone-testable.** ``--sequential`` is a
  parameter. When Lane 1 has landed the flag, the probe passes it and returns
  the true sequential metrics. When it has *not* (or the flag is force-off),
  the probe degrades gracefully to the existing (1-1-latched) metrics and marks
  ``cold_sequential=False`` — so this lane is testable today against any
  existing tile checkpoint with zero edits to `eval_game.py`.
* **Never raises into the trainer.** Any failure (missing ROM, subprocess
  error, timeout, unparseable output) returns a fully-shaped dict with
  ``cold_error`` set and metric fields defaulted, so the caller can log and
  carry on rather than crash a multi-day run.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.training.profile_utils import profile_slug

_ROOT = Path(__file__).resolve().parents[2]

# The flat metric surface this probe guarantees. Every key is always present
# in the returned dict; sequential-only fields are ``None`` when the predicate
# did not run (Lane 1 absent or ``sequential=False``).
COLD_METRIC_KEYS: tuple[str, ...] = (
    "cold_seq_clear_rate",   # frac of episodes that cleared World-1 sequentially
    "cold_furthest_seq",     # deepest sequential (world, level) — passthrough
    "cold_furthest_any",     # deepest incl. warps — passthrough
    "cold_warp_rate",        # frac of episodes that took a warp
    "cold_clear_1_1",        # legacy 1-1-flagpole clear rate (always available)
    "cold_max_byte",         # max packed world<<4|area seen (zero-indexed)
    "cold_mean_return",
    "cold_mean_length",
    "cold_n_episodes",
    "cold_status",           # eval_game status, or a probe-side failure label
    "cold_sequential",       # True iff the --sequential predicate produced metrics
    "cold_checkpoint",       # checkpoint string eval_game reported (debug breadcrumb)
    "cold_error",            # None on success; failure detail otherwise
)

# Process-lifetime memo of whether `eval_game.py` accepts ``--sequential`` so
# the auto path pays the "unrecognized argument" probe at most once.
_SEQUENTIAL_SUPPORTED: Optional[bool] = None


def _base_result(status: str, error: Optional[str], checkpoint: Optional[str]) -> dict:
    """A fully-shaped cold-metrics dict with every field at a safe default."""
    return {
        "cold_seq_clear_rate": None,
        "cold_furthest_seq": None,
        "cold_furthest_any": None,
        "cold_warp_rate": None,
        "cold_clear_1_1": 0.0,
        "cold_max_byte": 0,
        "cold_mean_return": 0.0,
        "cold_mean_length": 0.0,
        "cold_n_episodes": 0,
        "cold_status": status,
        "cold_sequential": False,
        "cold_checkpoint": checkpoint,
        "cold_error": error,
    }


def _map_metrics(eval_json: dict, sequential_ran: bool) -> dict:
    """Project an `eval_game.py` result JSON onto the flat ``cold_*`` surface.

    Pure function (no I/O) so the parse path is unit-testable with a synthetic
    dict. Sequential-only fields are populated only when the predicate actually
    ran *and* produced a value; otherwise they stay ``None``.
    """
    status = str(eval_json.get("status", "unknown"))
    have_seq = sequential_ran and "seq_clear_rate" in eval_json
    out = _base_result(status, None, eval_json.get("checkpoint"))
    out["cold_sequential"] = have_seq
    out["cold_clear_1_1"] = float(eval_json.get("clear_rate", 0.0) or 0.0)
    out["cold_max_byte"] = int(eval_json.get("max_byte_seen", 0) or 0)
    out["cold_mean_return"] = float(eval_json.get("mean_return", 0.0) or 0.0)
    out["cold_mean_length"] = float(eval_json.get("mean_length", 0.0) or 0.0)
    out["cold_n_episodes"] = int(eval_json.get("n_episodes", 0) or 0)
    if have_seq:
        out["cold_seq_clear_rate"] = float(eval_json.get("seq_clear_rate", 0.0) or 0.0)
        out["cold_furthest_seq"] = eval_json.get("furthest_seq_level")
        out["cold_furthest_any"] = eval_json.get("furthest_any_level")
        wr = eval_json.get("warp_rate")
        out["cold_warp_rate"] = float(wr) if wr is not None else None
    if status != "ok":
        out["cold_error"] = f"eval_game status={status}: {eval_json.get('detail', '')}".strip()
    return out


def _parse_eval_stdout(stdout: str) -> dict:
    """Parse the JSON `eval_game.py` prints on stdout, tolerating stray lines."""
    text = stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(text[start:end + 1])


def _run_eval(cmd: list[str], want_sequential: Optional[bool], timeout: float,
              cwd: str, extra_args: Optional[list[str]] = None,
              ) -> tuple[subprocess.CompletedProcess, bool]:
    """Run `eval_game.py`, negotiating ``--sequential`` support.

    ``want_sequential``: ``None`` = auto (try unless known-unsupported, degrade
    on "unrecognized argument"); ``True`` = force on (unsupported surfaces as a
    nonzero return, no silent degrade); ``False`` = never pass the flag.

    ``extra_args`` are always-passed flags that go WITH the sequential predicate
    (e.g. ``--start-state PATH`` / ``--level-clear`` for the level-scoped gate).
    They ride along only when the sequential predicate itself is used — the
    degrade path (Lane 1 absent) drops them too, since they are meaningless
    without it.

    Returns ``(completed_process, sequential_was_passed)``.
    """
    global _SEQUENTIAL_SUPPORTED
    extra = list(extra_args or [])
    if want_sequential is False:
        use_seq = False
    elif want_sequential is True:
        use_seq = True
    else:  # auto
        use_seq = _SEQUENTIAL_SUPPORTED is not False

    proc = subprocess.run(
        cmd + (extra + ["--sequential"] if use_seq else []),
        cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )
    unrecognized = (
        use_seq and proc.returncode == 2
        and "unrecognized arguments" in proc.stderr
        and "--sequential" in proc.stderr
    )
    if unrecognized and want_sequential is None:
        # Lane 1 not landed yet — memoize and degrade to the legacy metrics.
        _SEQUENTIAL_SUPPORTED = False
        return (
            subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout),
            False,
        )
    if use_seq and proc.returncode == 0:
        _SEQUENTIAL_SUPPORTED = True
    return proc, use_seq


def _resolve_path(value: Any) -> Optional[str]:
    """Absolutize a path against the caller's CWD (relative → cwd-anchored)."""
    if not value:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p.resolve())


def probe(
    net: Any,
    cfg: dict,
    episodes: int = 8,
    device: Any = None,
    *,
    max_steps: int = 3000,
    sequential: Optional[bool] = None,
    start_state: Optional[str] = None,
    level_clear: bool = False,
    timeout: float = 1800.0,
    rom_path: Optional[str] = None,
    game: Optional[str] = None,
    eval_script: Optional[str] = None,
    python_exe: Optional[str] = None,
) -> dict:
    """Cold-eval ``net`` in an isolated subprocess and return ``cold_*`` metrics.

    Snapshots ``net``'s weights (CPU copy — ``device`` is where they currently
    live and is never disturbed) into a throwaway temp checkpoint, subprocesses
    ``eval_game.py`` to greedily score ``episodes`` episodes from ``cfg``'s
    start state, and maps the JSON onto :data:`COLD_METRIC_KEYS`.

    Args:
        net: the policy module (or a raw ``state_dict``) to score.
        cfg: the game profile dict (needs ``name``, encoder, action space, and
            a start state; a relative ``start_state_path`` resolves against the
            caller's CWD).
        episodes: eval episodes (8 for the cheap curve, 24 at a GO gate).
        device: the net's current device — informational; weights are copied to
            CPU for serialization so training tensors are untouched.
        max_steps: per-episode step cap handed to ``eval_game.py``.
        sequential: ``None`` = auto-detect the ``--sequential`` flag and degrade
            if absent; ``True`` = require it; ``False`` = legacy metrics only.
        start_state: warm-start every eval episode from this save-state file
            (a level's entry state) instead of the profile cold boot — the
            level-scoped consolidation gate probes each level from its entry.
            A relative path resolves against the caller's CWD. Requires the
            ``--start-state`` flag on ``eval_game.py``.
        level_clear: with ``sequential``, score "cleared the level it STARTED
            in" (``cold_seq_clear_rate`` becomes the per-level clear rate)
            instead of the full World-1 chain — the number the gate accepts /
            rolls back on. Requires the ``--level-clear`` flag on ``eval_game.py``.
        timeout: seconds before the subprocess is killed (failure, not crash).
        rom_path: ROM path; falls back to ``cfg['rom_path']`` / ``cfg['rom']``.
        game: cosmetic ``--game`` label for the subprocess (defaults to the
            profile slug); the explicit ``--profile`` is authoritative.
        eval_script / python_exe: overrides for the eval script and interpreter
            (default: this repo's ``scripts/eval_game.py`` under ``sys.executable``).

    Returns:
        A dict with exactly :data:`COLD_METRIC_KEYS`. Never raises: on any
        failure it returns a shaped dict with ``cold_error`` set.
    """
    try:
        state_dict = net.state_dict() if hasattr(net, "state_dict") else net
        if not isinstance(state_dict, dict):
            return _base_result("bad_net", f"net is neither a module nor a state_dict: {type(net)!r}", None)
        cpu_sd = {k: (v.detach().to("cpu") if hasattr(v, "detach") else v) for k, v in state_dict.items()}

        rom = _resolve_path(rom_path or cfg.get("rom_path") or cfg.get("rom"))
        if rom is None or not Path(rom).exists():
            return _base_result("no_rom", f"no existing ROM (rom_path={rom_path!r})", None)

        script = eval_script or str(_ROOT / "scripts" / "eval_game.py")
        py = python_exe or sys.executable
        slug = profile_slug(cfg.get("name"))
        game_label = game or slug

        with tempfile.TemporaryDirectory(prefix="cold_probe_") as td:
            td_path = Path(td)
            ckpt_dir = td_path / "checkpoints" / slug
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"net_state_dict": cpu_sd}, ckpt_dir / "vanilla_ppo_iter_00000.pt")

            # Temp profile = the real config with paths absolutized, so the
            # subprocess (run from the temp CWD) still finds the start state.
            tp = copy.deepcopy(cfg)
            ss = _resolve_path(cfg.get("start_state_path"))
            if ss is not None:
                tp["start_state_path"] = ss
            prof_file = td_path / "probe_profile.yaml"
            prof_file.write_text(yaml.safe_dump(tp))

            cmd = [
                py, script,
                "--game", str(game_label),
                "--profile", str(prof_file),
                "--rom", rom,
                "--iter", "0",
                "--episodes", str(int(episodes)),
                "--max-steps", str(int(max_steps)),
            ]
            # Level-scoped gate extras (ride WITH --sequential). The start
            # state is absolutized against the caller's CWD because the
            # subprocess runs from the throwaway temp dir.
            extra_args: list[str] = []
            ss_probe = _resolve_path(start_state)
            if ss_probe is not None:
                extra_args += ["--start-state", ss_probe]
            if level_clear:
                extra_args += ["--level-clear"]
            try:
                proc, seq_ran = _run_eval(
                    cmd, sequential, timeout, str(td_path), extra_args=extra_args,
                )
            except subprocess.TimeoutExpired:
                return _base_result("timeout", f"eval subprocess exceeded {timeout}s", None)

            if proc.returncode != 0:
                return _base_result(
                    "probe_failed",
                    f"eval_game exited {proc.returncode}: {proc.stderr.strip()[-500:]}",
                    None,
                )
            try:
                eval_json = _parse_eval_stdout(proc.stdout)
            except json.JSONDecodeError as e:
                return _base_result(
                    "probe_failed",
                    f"could not parse eval JSON ({e}); stdout tail: {proc.stdout.strip()[-300:]}",
                    None,
                )
            return _map_metrics(eval_json, seq_ran)
    except Exception as e:  # never propagate into the training loop
        return _base_result("probe_failed", f"{type(e).__name__}: {e}", None)
