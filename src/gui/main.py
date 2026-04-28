"""
Neural Entertainment System (NES) GUI entry point.

Launches the PyQt6 main window. When the user clicks Start, launches a
training Thread in the same process and opens the emulator grid window.

All coupling between trainer and GUI is in-process:
  * The trainer owns a `nes_core.Pool` (Rust, rayon-parallel) which runs
    N NES instances inside the GUI process.
  * Frames flow trainer → GUI via a thread-safe callback (`frame_sink`).
  * Metrics, reward updates, audio mixer, and narrator captions go
    through plain `queue.Queue` instances — no multiprocessing.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src.gui.audio_mixer_window import AudioMixerWindow
from src.gui.emulator_grid import EmulatorGrid
from src.gui.main_window import MainWindow
from src.gui.metrics_window import MetricsWindow
from src.gui.play_window import PlayWindow
from src.gui.replay_window import ReplayWindow
from src.gui.reward_tuning_window import RewardTuningWindow


def _run_trainer(
    config: dict,
    frame_sink,
    metrics_queue,
    control_queue,
    reward_queue,
    audio_queue,
    narrator_queue,
    stop_event: threading.Event,
    error_holder: Optional[list] = None,
) -> None:
    """Thread entry point. Builds a Trainer and runs it until stop.

    `error_holder`, if provided, is a 1-element list the caller can
    inspect after the thread dies to tell whether it exited cleanly
    (error_holder[0] stays None) or crashed with an exception
    (error_holder[0] == exception instance). Without this, the GUI
    status message can't distinguish "training finished normally"
    from "training crashed in gen 0" — both surface as a dead thread.
    """
    from src.training.trainer import Trainer, find_latest_checkpoint, load_game_profile

    profile = load_game_profile(config["profile_path"])
    rw_override_path = config.get("reward_overrides_path")
    if rw_override_path:
        try:
            with open(rw_override_path) as _fh:
                override = yaml.safe_load(_fh) or {}
        except Exception:
            override = {}
        for section in ("reward_weights", "reinforce", "ga_params"):
            if section in override:
                merged = dict(profile.get(section, {}))
                merged.update(override[section])
                profile[section] = merged
    if config.get("max_episode_steps"):
        profile["max_episode_steps"] = int(config["max_episode_steps"])
    if config.get("activity_timeout_steps") is not None:
        weights = profile.setdefault("reward_weights", {})
        weights["activity_timeout_steps"] = int(config["activity_timeout_steps"])

    env_spec = config.get("env_spec", "nes_core:NESEnvironment")
    trainer = Trainer(
        rom_path=config["rom_path"],
        game_profile=profile,
        num_instances=config["num_instances"],
        population_size=config["population_size"],
        frame_sink=frame_sink,
        metrics_queue=metrics_queue,
        reward_queue=reward_queue,
        audio_queue=audio_queue,
        narrator_queue=narrator_queue,
        start_state_path=config.get("start_state_path"),
        bc_demo_path=config.get("bc_demo_path"),
        env_spec=env_spec,
        max_episode_steps=int(config.get("max_episode_steps", 1000)),
    )

    resume_from = None
    if config.get("resume"):
        latest = find_latest_checkpoint(trainer.checkpoint_dir)
        if latest:
            resume_from = str(latest)

    def _control_listener() -> None:
        while not stop_event.is_set():
            try:
                msg = control_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg == "stop":
                trainer.stop()
                return

    t = threading.Thread(target=_control_listener, daemon=True)
    t.start()

    try:
        trainer.run(num_generations=10_000, resume_from=resume_from)
    except Exception as exc:
        # Capture the exception so _check_trainer_alive can surface it
        # to the user as a real error instead of the misleading
        # "Training finished (generation budget reached)" message.
        import logging as _logging
        _logging.getLogger(__name__).exception(
            "training thread crashed with uncaught exception"
        )
        if error_holder is not None:
            error_holder.append(exc)
        raise
    finally:
        stop_event.set()


class AppController:
    """Glue between the main window, the emulator grid, and the trainer thread."""

    def __init__(self) -> None:
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("NES")
        self.app.aboutToQuit.connect(self._on_quit_sync)

        self.main_window = MainWindow()
        self.main_window.start_requested.connect(self._on_start)
        self.main_window.stop_requested.connect(self._on_stop)
        self.main_window.play_requested.connect(self._on_play)
        self.main_window.replay_requested.connect(self._on_replay)
        self.main_window.tune_rewards_requested.connect(self._on_tune_rewards)
        self.main_window.audio_mixer_requested.connect(self._on_audio_mixer)
        self.main_window.start_state_change_requested.connect(
            self._on_start_state_changed
        )

        self.grid_window: EmulatorGrid | None = None
        self.metrics_window: MetricsWindow | None = None
        self.play_window: PlayWindow | None = None
        self.replay_window: ReplayWindow | None = None
        self.tuning_window: RewardTuningWindow | None = None
        self.audio_window: AudioMixerWindow | None = None
        self.training_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self.watchdog = None
        self.metrics_queue: Optional[queue.Queue] = None
        self.control_queue: Optional[queue.Queue] = None
        self.reward_queue: Optional[queue.Queue] = None
        self.audio_queue: Optional[queue.Queue] = None
        self.narrator_queue: Optional[queue.Queue] = None
        self.metrics_timer: QTimer | None = None
        self.narrator_timer: QTimer | None = None
        self.highlight_recorder = None
        self._shutdown_timer: QTimer | None = None
        self._shutdown_deadline: float = 0.0
        self._active_profile_weights: dict = {}
        self._num_instances: int = 0

        # Latest frame per worker — written by the trainer thread via
        # `_frame_sink`, read by the grid's QTimer on the main thread.
        # Access is protected by `_frames_lock`; we swap references
        # rather than copy the arrays, so the lock is held only long
        # enough to replace a pointer.
        self._latest_frames: list[np.ndarray] = []
        self._latest_seq: list[int] = []
        self._frames_lock = threading.Lock()

    def _frame_sink(self, results) -> None:
        """Called by the trainer thread after every step_all / reset_all.
        Swap per-worker frames into the shared slot; the GUI polling
        callback picks them up on the next tick."""
        with self._frames_lock:
            for r in results:
                wid = r.worker_id
                if 0 <= wid < len(self._latest_frames):
                    self._latest_frames[wid] = r.frame
                    self._latest_seq[wid] += 1

    def _on_start(self, config: dict) -> None:
        import logging as _logging
        _log = _logging.getLogger(__name__)
        if self.training_thread is not None and not self.training_thread.is_alive():
            _log.info("[start] reaping dead trainer thread before fresh start")
            self._finalize_stop()
        if self.training_thread is not None and self.training_thread.is_alive():
            _log.warning(
                "[start] IGNORED — prior training thread still alive. "
                "Wait for Stop to complete before restarting."
            )
            if hasattr(self.main_window, "_show_status"):
                try:
                    self.main_window._show_status(
                        "Start ignored: previous training still shutting down."
                    )
                except Exception:
                    pass
            return
        _log.info(
            "[start] prior state: thread=%s grid_window=%s",
            self.training_thread, self.grid_window,
        )
        env_spec = getattr(self, "env_spec", "nes_core:NESEnvironment")
        config = {**config, "env_spec": env_spec}
        num_instances = config["num_instances"]

        # In-process frame exchange. Trainer thread writes; GUI timer reads.
        self._latest_frames = [
            np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(num_instances)
        ]
        self._latest_seq = [0] * num_instances

        # Plain in-process queues — no multiprocessing, no shm.
        self.metrics_queue = queue.Queue(maxsize=128)
        self.control_queue = queue.Queue(maxsize=8)
        self.reward_queue = queue.Queue(maxsize=32)
        self.audio_queue = queue.Queue(maxsize=32)
        self.narrator_queue = queue.Queue(maxsize=256)
        self._num_instances = num_instances

        try:
            profile_data = yaml.safe_load(Path(config["profile_path"]).read_text())
            merged = dict(profile_data.get("reward_weights", {}))
        except Exception:
            merged = {}
        overrides_path = config.get("reward_overrides_path")
        if overrides_path:
            try:
                overrides_data = yaml.safe_load(Path(overrides_path).read_text()) or {}
                for k, v in (overrides_data.get("reward_weights") or {}).items():
                    merged[k] = v
            except Exception:
                pass
        self._active_profile_weights = merged

        controller_ref = self

        def _grid_provider() -> list:
            out: list = []
            with self._frames_lock:
                for i in range(num_instances):
                    out.append((self._latest_frames[i], self._latest_seq[i]))
            for i, (frame, _seq) in enumerate(out):
                rec = getattr(controller_ref, "highlight_recorder", None)
                if rec is not None:
                    rec.ingest(i, frame)
            return out

        self.grid_window = EmulatorGrid(num_instances=num_instances)
        self.grid_window.set_frame_provider(_grid_provider)
        self.grid_window.show()
        _log.info(
            "[start] grid_window created + shown (num_instances=%d)", num_instances,
        )

        from src.gui.highlight_recorder import HighlightRecorder
        self.highlight_recorder = HighlightRecorder(
            num_workers=num_instances,
            out_dir="highlights",
        )

        metrics_path = Path("./checkpoints") / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_window = MetricsWindow(metrics_path=str(metrics_path))
        self.metrics_window.show()

        self._stop_event = threading.Event()
        # 1-element mutable slot the trainer thread fills with its
        # exception (if any) so _check_trainer_alive can distinguish
        # clean completion from a mid-run crash.
        self._trainer_error: list = []
        self.training_thread = threading.Thread(
            target=_run_trainer,
            args=(
                config,
                self._frame_sink,
                self.metrics_queue,
                self.control_queue,
                self.reward_queue,
                self.audio_queue,
                self.narrator_queue,
                self._stop_event,
                self._trainer_error,
            ),
            daemon=True,
        )
        self.training_thread.start()

        try:
            from src.gui.watchdog import Watchdog
            self.watchdog = Watchdog()
            wd_path = self.watchdog.start()
            _log.info("[start] watchdog logging to %s", wd_path)
        except Exception as exc:
            _log.warning("[start] watchdog failed to start: %s", exc)
            self.watchdog = None

        self.metrics_timer = QTimer(self.app)
        self.metrics_timer.timeout.connect(self._on_tick)
        self.metrics_timer.start(500)

        self.narrator_timer = QTimer(self.app)
        self.narrator_timer.timeout.connect(self._drain_narrator)
        self.narrator_timer.start(50)

    def _on_tick(self) -> None:
        self._drain_metrics()
        self._check_trainer_alive()

    def _drain_narrator(self) -> None:
        q = self.narrator_queue
        if q is None or self.grid_window is None:
            return
        try:
            while True:
                ev = q.get_nowait()
                try:
                    wid = int(ev.get("worker_id", -1))
                    text = str(ev.get("caption") or "")
                    banner = bool(ev.get("first_ever", False))
                    name = str(ev.get("genome_name") or "")
                except Exception:
                    continue
                if not text:
                    continue
                if name:
                    self.grid_window.set_tile_name(wid, name)
                self.grid_window.show_caption(wid, text, banner=banner)
                kind = str(ev.get("kind") or "")
                should_clip = banner or kind == "triforce"
                if should_clip and self.highlight_recorder is not None:
                    try:
                        self.highlight_recorder.capture(
                            worker_id=wid,
                            kind=kind or "event",
                            genome_name=name or "unknown",
                            label=text,
                        )
                    except Exception as exc:
                        import logging as _logging
                        _logging.getLogger(__name__).debug(
                            "highlight capture failed: %s", exc
                        )
        except queue.Empty:
            pass
        except Exception:
            pass

    def _drain_metrics(self) -> None:
        if self.metrics_queue is None:
            return
        latest = None
        try:
            while True:
                latest = self.metrics_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass
        if latest is not None:
            self.main_window.update_metrics(latest)

    def _check_trainer_alive(self) -> None:
        t = self.training_thread
        if t is None or t.is_alive():
            return
        # Distinguish a clean completion/stop from an exception-died
        # thread. _run_trainer's except branch appends the exception
        # to self._trainer_error; if it's non-empty the thread
        # crashed, surface the underlying error class + message so
        # the user sees the real problem instead of the misleading
        # "generation budget reached" (which was tonight's 2026-04-22
        # UX complaint — every exception looked like a normal finish).
        errs = getattr(self, "_trainer_error", None)
        if errs:
            exc = errs[0]
            self.main_window._show_status(
                f"Training stopped — {type(exc).__name__}: {str(exc)[:160]}"
            )
        else:
            self.main_window._show_status(
                "Training finished (generation budget reached or stopped)."
            )
        self._on_stop()
        self.main_window.start_btn.setEnabled(True)
        self.main_window.stop_btn.setEnabled(False)

    def _show_window(self, attr: str, factory) -> None:
        widget = getattr(self, attr, None)
        if widget is not None:
            try:
                widget.show()
                widget.raise_()
                widget.activateWindow()
                return
            except RuntimeError:
                setattr(self, attr, None)
        widget = factory()
        setattr(self, attr, widget)
        widget.show()
        widget.raise_()
        widget.activateWindow()

    def _on_play(self, rom_path: str) -> None:
        existing = self.play_window
        if existing is not None:
            try:
                existing.close()
                existing.deleteLater()
            except Exception:
                pass
            self.play_window = None
        self.play_window = PlayWindow(rom_path=rom_path)
        self.play_window.show()
        self.play_window.raise_()
        self.play_window.activateWindow()

    def _on_audio_mixer(self) -> None:
        if self.audio_queue is None:
            return

        def _push(upd: dict) -> None:
            try:
                if self.audio_queue is not None:
                    self.audio_queue.put_nowait(upd)
            except Exception:
                pass

        existing = self.audio_window
        instance_match = (
            existing is not None
            and getattr(existing, "num", None) == (self._num_instances or 1)
        )
        if instance_match:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self.audio_window = None

        if self.audio_window is not None:
            try:
                self.audio_window.close()
                self.audio_window.deleteLater()
            except RuntimeError:
                pass
            self.audio_window = None

        self.audio_window = AudioMixerWindow(
            num_instances=self._num_instances or 1,
            on_change=_push,
        )
        self.audio_window.show()
        self.audio_window.raise_()
        self.audio_window.activateWindow()

    def _on_tune_rewards(self) -> None:
        if self.reward_queue is None:
            return

        def _push(updates: dict) -> None:
            for k, v in updates.items():
                if not k.startswith("__"):
                    self._active_profile_weights[k] = v
            try:
                if self.reward_queue is not None:
                    self.reward_queue.put_nowait(updates)
            except Exception:
                pass

        if self.tuning_window is not None:
            try:
                self.tuning_window.show()
                self.tuning_window.raise_()
                self.tuning_window.activateWindow()
                return
            except RuntimeError:
                self.tuning_window = None

        self.tuning_window = RewardTuningWindow(
            initial_weights=self._active_profile_weights,
            on_change=_push,
        )
        self.tuning_window.show()
        self.tuning_window.raise_()
        self.tuning_window.activateWindow()

    def _on_start_state_changed(self, path: str) -> None:
        if self.training_thread is None or not self.training_thread.is_alive():
            return
        if self.reward_queue is None:
            return
        try:
            payload = {"__start_state_path__": path if path else None}
            self.reward_queue.put_nowait(payload)
            label = Path(path).name if path else "(none — cold reset)"
            self.main_window._show_status(
                f"Start-state swap queued: {label}. "
                "Pool will rebuild before the next generation."
            )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Could not queue start-state swap (%s)", exc,
            )
            self.main_window._show_status(
                "Start-state swap could not be queued — Stop and Start to apply."
            )

    def _on_replay(self, config: dict) -> None:
        if self.replay_window is not None:
            try:
                self.replay_window.close()
                self.replay_window.deleteLater()
            except RuntimeError:
                pass
            self.replay_window = None
        self.replay_window = ReplayWindow(
            rom_path=config["rom_path"],
            profile_path=config["profile_path"],
            checkpoint_path=config["checkpoint_path"],
            start_state_path=config.get("start_state_path"),
        )
        self.replay_window.show()
        self.replay_window.raise_()
        self.replay_window.activateWindow()

    def _on_stop(self) -> None:
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info(
            "[stop] received — thread=%s alive=%s grid_window=%s",
            self.training_thread,
            self.training_thread.is_alive() if self.training_thread else False,
            self.grid_window,
        )
        if self.control_queue is not None:
            try:
                self.control_queue.put_nowait("stop")
            except Exception as exc:
                _log.warning("control_queue full on stop (%s)", exc)
        if self._stop_event is not None:
            self._stop_event.set()

        if self.metrics_timer:
            self.metrics_timer.stop()
            self.metrics_timer = None
        if self.narrator_timer:
            self.narrator_timer.stop()
            self.narrator_timer = None

        if self.grid_window is not None:
            self.grid_window.close()
            self.grid_window = None
        if self.metrics_window is not None:
            self.metrics_window.close()
            self.metrics_window = None
        if self.tuning_window is not None:
            self.tuning_window.close()
            self.tuning_window = None
        if self.audio_window is not None:
            self.audio_window.close()
            self.audio_window = None

        if self.training_thread is None or not self.training_thread.is_alive():
            self._finalize_stop()
            return

        import time as _time
        self._shutdown_deadline = _time.monotonic() + 5.0
        if self._shutdown_timer is None:
            self._shutdown_timer = QTimer(self.app)
            self._shutdown_timer.timeout.connect(self._poll_shutdown)
        self._shutdown_timer.start(100)

    def _poll_shutdown(self) -> None:
        t = self.training_thread
        if t is None:
            self._stop_shutdown_timer()
            self._finalize_stop()
            return
        if not t.is_alive():
            t.join(timeout=0.0)
            self._stop_shutdown_timer()
            self._finalize_stop()
            return
        import time as _time
        if _time.monotonic() >= self._shutdown_deadline:
            # Thread won't join. Clear the reference and let it die
            # with the interpreter. (We can't forcibly kill a Python
            # thread; the Rust pool drops its workers when trainer.run
            # returns, which should happen on the next step_all.)
            self._stop_shutdown_timer()
            self._finalize_stop()

    def _stop_shutdown_timer(self) -> None:
        if self._shutdown_timer is not None:
            self._shutdown_timer.stop()
            self._shutdown_timer = None

    def _finalize_stop(self) -> None:
        import gc as _gc
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info("[finalize_stop] entering")
        if self.watchdog is not None:
            try:
                self.watchdog.stop()
            except Exception:
                pass
            self.watchdog = None
        self.training_thread = None
        self._stop_event = None
        # Force cycle collection so the prior Trainer's pre-allocated
        # trajectory buffers (up to ~8 GB each) release their pages
        # before the next Start. Without this, Stop→Start quickly
        # doubles RSS and can OOM the host.
        _freed = _gc.collect()
        _log.info("[finalize_stop] gc.collect freed %d cycles", _freed)
        if self.grid_window is not None:
            self.grid_window = None
        self.highlight_recorder = None
        self.metrics_queue = None
        self.control_queue = None
        self.reward_queue = None
        self.audio_queue = None
        self.narrator_queue = None
        self._latest_frames = []
        self._latest_seq = []
        _log.info("[finalize_stop] complete — ready for next Start")

    def _on_quit_sync(self) -> None:
        if self._shutdown_timer is not None:
            try:
                self._shutdown_timer.stop()
            except Exception:
                pass
            self._shutdown_timer = None
        if self.metrics_timer is not None:
            try:
                self.metrics_timer.stop()
            except Exception:
                pass
            self.metrics_timer = None
        if self._stop_event is not None:
            self._stop_event.set()
        if self.control_queue is not None:
            try:
                self.control_queue.put_nowait("stop")
            except Exception:
                pass
        t = self.training_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        if self.watchdog is not None:
            try:
                self.watchdog.stop()
            except Exception:
                pass
            self.watchdog = None
        self.training_thread = None
        self._stop_event = None
        self.grid_window = None
        self.metrics_window = None
        self.tuning_window = None
        self.audio_window = None
        self.replay_window = None
        self.play_window = None
        self.metrics_queue = None
        self.control_queue = None
        self.reward_queue = None
        self.audio_queue = None
        self.narrator_queue = None
        self._latest_frames = []
        self._latest_seq = []

    def run(self) -> int:
        self.main_window.show()
        return self.app.exec()


def main() -> int:
    import argparse
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Neural Entertainment System GUI")
    parser.add_argument(
        "--env-backend",
        choices=("nes_core",),
        default="nes_core",
        help="Emulator core. `nes_core` is the in-tree Rust NES core.",
    )
    parser.parse_known_args()

    controller = AppController()
    controller.env_spec = "nes_core:NESEnvironment"
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
