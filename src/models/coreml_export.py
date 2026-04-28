"""
Core ML export of a trained PolicyNetwork.

Converts the PyTorch model to a Core ML .mlpackage that runs on macOS
with ANE (Apple Neural Engine) as the preferred compute unit. ANE is
the most power-efficient matrix engine on Apple Silicon — inference on
our Nature-DQN CNN runs ~3-5x faster + draws a fraction of the power
compared to MPS when the model fits the supported op set.

Use cases:
  * Ship a compiled `.mlpackage` alongside the app for the Genome Replay
    viewer so watching a trained agent doesn't saturate MPS.
  * Potentially swap the runtime inference path to Core ML in future
    (currently optional — training stays on PyTorch MPS).

Core ML conversion is opportunistic: if any op fails to convert we warn
and fall back. Call `maybe_export` with a path; on failure you get None.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch


log = logging.getLogger(__name__)


def _convert_in_process(
    network,
    save_path: Path,
    frame_stack: int,
    frame_size: int,
) -> Optional[Path]:
    """In-process Core ML conversion. Used by the subprocess worker
    (see `maybe_export`) so the multi-GB MLIR context that coremltools
    creates lives and dies with that subprocess instead of accumulating
    in the trainer."""
    try:
        import coremltools as ct
    except ImportError:
        log.warning("coremltools not installed; skipping Core ML export")
        return None

    network.eval()
    network.to("cpu")

    example_input = torch.zeros(1, frame_stack, frame_size, frame_size)
    try:
        traced = torch.jit.trace(network, example_input)
    except Exception as exc:
        log.warning("torch.jit.trace failed: %s", exc)
        return None

    # Use numpy.float32 explicitly instead of Python's builtin `float`
    # (which coremltools historically resolved to float32 but the
    # behaviour is fragile across versions). Pinning the dtype removes
    # an ambiguity that could otherwise produce float64 buffers and an
    # unnecessarily large `.mlpackage`.
    import numpy as _np
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="stacked_frames",
                    shape=(1, frame_stack, frame_size, frame_size),
                    dtype=_np.float32,
                )
            ],
            outputs=[ct.TensorType(name="action_logits", dtype=_np.float32)],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,  # prefer ANE, fall back to GPU/CPU
            minimum_deployment_target=ct.target.macOS14,
        )
    except Exception as exc:
        log.warning("coremltools.convert failed: %s", exc)
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix != ".mlpackage":
        save_path = save_path.with_suffix(".mlpackage")

    try:
        mlmodel.save(str(save_path))
    except Exception as exc:
        log.warning("mlmodel.save failed: %s", exc)
        return None

    log.info("Core ML export written to %s", save_path)
    return save_path


def maybe_export(
    network,
    save_path: str | Path,
    num_actions: int,
    frame_stack: int = 4,
    frame_size: int = 84,
) -> Optional[Path]:
    """Convert `network` to Core ML and save to `save_path`, isolated in
    a subprocess so coremltools's MLIR context is freed when the worker
    exits. Returns the Path on success, None on failure. Failure is
    logged at warning level — never raises.

    Why subprocess: `coremltools.convert` instantiates an MLIRContext
    that loads ~100 dialects/types and allocates ~2 GB the first time it
    runs in a process. The MLIRContext is NOT released when the
    `mlmodel` object goes out of scope; it lives in a process-global
    cache. Calling `maybe_export` once per checkpoint inside the
    long-lived trainer process therefore leaks ~2 GB per call (verified
    via /usr/bin/sample 2026-04-24: main thread spending 95% of cycles
    in MPSGraph init / fill_scalar_mps after ~3 generations). Running
    the conversion in a fresh `python -c` worker confines the leak to
    that worker's lifetime — the OS reclaims it on exit.
    """
    save_path = Path(save_path)
    if save_path.suffix != ".mlpackage":
        save_path = save_path.with_suffix(".mlpackage")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Stash the network state to a temp file the subprocess can pick up.
    # State-dict round-trip is cheap (<10 MB for our Nature-DQN) vs the
    # ~2 GB MLIR context we're avoiding.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        weights_path = Path(f.name)
    try:
        torch.save(network.state_dict(), weights_path)
        # Spawn a worker that imports coremltools + converts + exits.
        import subprocess, sys
        import os
        repo_root = Path(__file__).resolve().parents[2]
        worker = (
            "import sys, os\n"
            f"sys.path.insert(0, {str(repo_root)!r})\n"
            "import torch\n"
            "from src.models.policy_network import PolicyNetwork\n"
            "from src.models.coreml_export import _convert_in_process\n"
            "from pathlib import Path\n"
            f"net = PolicyNetwork(num_actions={num_actions}, frame_stack={frame_stack}, frame_size={frame_size})\n"
            f"net.load_state_dict(torch.load({str(weights_path)!r}, map_location='cpu'))\n"
            f"_convert_in_process(net, Path({str(save_path)!r}), {frame_stack}, {frame_size})\n"
        )
        try:
            subprocess.run(
                [sys.executable, "-c", worker],
                check=False,
                timeout=120.0,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
            )
        except subprocess.TimeoutExpired:
            log.warning("CoreML export subprocess timed out (120s)")
            return None
        except Exception as exc:
            log.warning("CoreML export subprocess failed: %s", exc)
            return None
        if save_path.exists():
            return save_path
        return None
    finally:
        weights_path.unlink(missing_ok=True)


class CoreMLPolicy:
    """Thin wrapper around a compiled Core ML policy, usable as a drop-in
    inference target for the replay viewer. Preserves the same interface
    as PolicyNetwork.act() so callers don't care which backend runs."""

    def __init__(self, mlpackage_path: str | Path) -> None:
        import coremltools as ct
        self._model = ct.models.MLModel(str(mlpackage_path))
        self._input_name = next(
            iter(self._model.get_spec().description.input)
        ).name
        self._output_name = next(
            iter(self._model.get_spec().description.output)
        ).name

    def act(self, frame_stack: torch.Tensor, deterministic: bool = False) -> int:
        import numpy as np
        arr = frame_stack.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        arr = arr.astype(np.float32)
        out = self._model.predict({self._input_name: arr})
        logits = out[self._output_name].squeeze()
        if deterministic:
            return int(np.argmax(logits))
        # Numerically stable softmax + sample
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))
