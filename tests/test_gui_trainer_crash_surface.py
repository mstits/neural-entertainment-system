"""
Guard tests for the crash-safety contract in `src/gui/main._run_trainer`.

`_run_trainer` builds a Trainer (or DreamerTrainer) inside the training
thread. A construction-time failure (bad profile, bad ROM, a config the
Trainer rejects) has to reach the GUI via the shared `error_holder` list
so `AppController._check_trainer_alive` can surface the real error instead
of the misleading "generation budget reached" message. These tests drive
`_run_trainer` directly with plain queues (no Qt widgets, so they run
under vanilla pytest) and assert the exception lands in error_holder.

Also covers ISSUE-6: `_on_audio_mixer` must not be a silent no-op when no
live trainer is running.
"""

from __future__ import annotations

import os
import queue as _queue
import threading
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


def _base_config() -> dict:
    return {
        "profile_path": "dummy.yaml",
        "rom_path": "dummy.nes",
        "num_instances": 1,
        "population_size": 1,
    }


def _call_run_trainer(config: dict, error_holder: list, stop_event: threading.Event):
    from src.gui import main as gui_main

    gui_main._run_trainer(
        config,
        lambda results: None,   # frame_sink
        _queue.Queue(),         # metrics_queue
        _queue.Queue(),         # control_queue
        _queue.Queue(),         # reward_queue
        _queue.Queue(),         # audio_queue
        _queue.Queue(),         # narrator_queue
        stop_event,
        error_holder,
    )


def test_run_trainer_captures_trainer_construction_failure(monkeypatch):
    from src.training import trainer as trainer_mod

    # load_game_profile must succeed so we reach the Trainer(...) call.
    monkeypatch.setattr(
        trainer_mod, "load_game_profile",
        lambda path: {"name": "test", "training_mode": "ga_ppo"},
    )

    class _BoomTrainer:
        def __init__(self, *a, **kw):
            raise ValueError("Game profile must define a non-empty action_space")

    monkeypatch.setattr(trainer_mod, "Trainer", _BoomTrainer)

    stop_event = threading.Event()
    error_holder: list = []

    with pytest.raises(ValueError):
        _call_run_trainer(_base_config(), error_holder, stop_event)

    assert len(error_holder) == 1
    assert isinstance(error_holder[0], ValueError)
    assert "action_space" in str(error_holder[0])
    # The thread's stop_event must be set so the GUI shutdown path unwinds.
    assert stop_event.is_set()


def test_run_trainer_captures_dreamer_construction_failure(monkeypatch):
    from src.training import dreamer as dreamer_mod
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod, "load_game_profile",
        lambda path: {"name": "test", "training_mode": "dreamer"},
    )

    class _BoomDreamer:
        def __init__(self, *a, **kw):
            raise RuntimeError("bad ROM for dreamer world model")

    monkeypatch.setattr(dreamer_mod, "DreamerTrainer", _BoomDreamer)

    stop_event = threading.Event()
    error_holder: list = []

    with pytest.raises(RuntimeError):
        _call_run_trainer(_base_config(), error_holder, stop_event)

    assert len(error_holder) == 1
    assert isinstance(error_holder[0], RuntimeError)
    assert stop_event.is_set()


def test_run_trainer_records_bad_reward_overrides_path(monkeypatch, tmp_path):
    # A missing/unreadable/malformed reward_overrides_path must not be
    # swallowed silently: the run still proceeds on the profile's baseline
    # weights (that fallback is fine), but the failure has to land in
    # error_holder so the GUI can surface it -- otherwise a checkpoint
    # later claimed to be "trained with reward config X" would be
    # silently false.
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod, "load_game_profile",
        lambda path: {
            "name": "test",
            "training_mode": "ga_ppo",
            "reward_weights": {"progress": 1.0},
        },
    )

    captured_profiles: list = []

    class _RecordingTrainer:
        def __init__(self, *a, **kw):
            captured_profiles.append(kw["game_profile"])
            self.checkpoint_dir = str(tmp_path)

        def run(self, **kw):
            return

        def stop(self):
            pass

    monkeypatch.setattr(trainer_mod, "Trainer", _RecordingTrainer)

    config = _base_config()
    config["reward_overrides_path"] = str(tmp_path / "does_not_exist.yaml")

    stop_event = threading.Event()
    error_holder: list = []

    # A bad overrides file must NOT abort the run.
    _call_run_trainer(config, error_holder, stop_event)

    assert len(error_holder) == 1
    assert isinstance(error_holder[0], OSError)
    assert len(captured_profiles) == 1
    # Baseline weights must be untouched -- the override never applied.
    assert captured_profiles[0]["reward_weights"] == {"progress": 1.0}


def test_run_trainer_no_error_holder_still_raises(monkeypatch):
    # error_holder is optional; a None holder must not mask the failure.
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod, "load_game_profile",
        lambda path: {"name": "test", "training_mode": "ga_ppo"},
    )

    class _BoomTrainer:
        def __init__(self, *a, **kw):
            raise ValueError("boom")

    monkeypatch.setattr(trainer_mod, "Trainer", _BoomTrainer)

    from src.gui import main as gui_main

    stop_event = threading.Event()
    with pytest.raises(ValueError):
        gui_main._run_trainer(
            _base_config(),
            lambda results: None,
            _queue.Queue(), _queue.Queue(), _queue.Queue(),
            _queue.Queue(), _queue.Queue(),
            stop_event,
            None,  # no error_holder
        )
    assert stop_event.is_set()


def test_audio_mixer_reports_status_when_no_trainer():
    # ISSUE-6: with no live trainer the mixer button must not be a silent
    # no-op — it should surface a status message. Drive the method on a
    # duck-typed stub so no Qt widgets are constructed under pytest.
    from src.gui.main import AppController

    shown: list[str] = []
    fake = types.SimpleNamespace(
        training_thread=None,
        audio_queue=None,
        main_window=types.SimpleNamespace(_show_status=shown.append),
    )
    AppController._on_audio_mixer(fake)
    assert len(shown) == 1
    assert "trainer" in shown[0].lower()


def test_audio_mixer_reports_status_when_trainer_dead():
    # audio_queue may still be set after the thread died but before
    # finalize; a dead thread must still gate the button.
    from src.gui.main import AppController

    shown: list[str] = []
    dead_thread = types.SimpleNamespace(is_alive=lambda: False)
    fake = types.SimpleNamespace(
        training_thread=dead_thread,
        audio_queue=_queue.Queue(),
        main_window=types.SimpleNamespace(_show_status=shown.append),
    )
    AppController._on_audio_mixer(fake)
    assert len(shown) == 1
    assert "trainer" in shown[0].lower()


class _FakeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, value):
        self.enabled = value


def test_ignored_start_does_not_leave_buttons_in_running_state(monkeypatch):
    # Reproduces the reported defect: MainWindow._on_start() emits
    # start_requested (a same-thread signal, delivered synchronously into
    # AppController._on_start) and then, once that call returns,
    # unconditionally re-enables its buttons and overwrites the status
    # line with "Starting training…". If a prior training thread is
    # still mid-shutdown, _on_start must ignore the request WITHOUT that
    # correction being clobbered by MainWindow's own post-emit code —
    # otherwise the GUI is left showing a fully "running" state (Stop /
    # Tune / Audio / Dashboard enabled) for a run that was never started.
    from src.gui import main as gui_main
    from src.gui.main import AppController

    deferred: list = []

    class _FakeQTimer:
        @staticmethod
        def singleShot(_delay, fn):
            deferred.append(fn)

    monkeypatch.setattr(gui_main, "QTimer", _FakeQTimer)

    shown: list[str] = []
    fake_mw = types.SimpleNamespace(
        start_btn=_FakeButton(),
        stop_btn=_FakeButton(),
        tune_btn=_FakeButton(),
        audio_btn=_FakeButton(),
        dashboard_btn=_FakeButton(),
        _show_status=shown.append,
    )
    fake = types.SimpleNamespace(
        training_thread=types.SimpleNamespace(is_alive=lambda: True),
        main_window=fake_mw,
    )
    fake._revert_ignored_start = lambda: AppController._revert_ignored_start(fake)

    AppController._on_start(fake, {})

    # Ignoring the request must not touch main_window synchronously — it
    # only schedules a correction for the next event-loop iteration.
    assert fake_mw.start_btn.enabled is None
    assert shown == []
    assert len(deferred) == 1

    # Simulate MainWindow._on_start()'s own post-emit code running to
    # completion right after the (synchronous) signal emit returns.
    fake_mw.start_btn.setEnabled(False)
    fake_mw.stop_btn.setEnabled(True)
    fake_mw.tune_btn.setEnabled(True)
    fake_mw.audio_btn.setEnabled(True)
    fake_mw.dashboard_btn.setEnabled(True)
    fake_mw._show_status("Starting training…")

    # Now let the deferred correction run, as the real Qt event loop
    # would once MainWindow's handler has returned.
    deferred[0]()

    assert fake_mw.start_btn.enabled is True
    assert fake_mw.stop_btn.enabled is False
    assert fake_mw.tune_btn.enabled is False
    assert fake_mw.audio_btn.enabled is False
    assert fake_mw.dashboard_btn.enabled is False
    assert shown[-1] == "Start ignored: previous training still shutting down."


def test_on_start_dashboard_metrics_path_matches_trainer_checkpoint_dir(
    monkeypatch, tmp_path,
):
    # Reproduces the reported defect: AppController._on_start used to
    # hardcode metrics_path to the flat "./checkpoints/metrics.jsonl",
    # but Trainer.__init__ always derives a per-profile subdirectory via
    # derive_checkpoint_dir("./checkpoints", game_profile["name"]) and
    # appends metrics.jsonl under THAT (trainer.py). The dashboard has to
    # watch the same file the trainer thread actually appends to, or it
    # silently shows stale/empty data for the whole run.
    from src.gui import main as gui_main
    from src.gui.main import AppController
    from src.training.profile_utils import derive_checkpoint_dir

    monkeypatch.chdir(tmp_path)

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Super Mario Bros.\naction_space: [[]]\n")

    # Neutralize every heavy/Qt/thread side effect on the way to
    # self._dashboard_config -- none of it is what this test checks.
    class _Stub:
        def __init__(self, *a, **kw):
            pass

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    class _FakeTimer:
        def __init__(self, *a, **kw):
            self.timeout = types.SimpleNamespace(connect=lambda *a, **kw: None)

        def start(self, *a, **kw):
            pass

    class _FakeThread:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

    monkeypatch.setattr(gui_main, "EmulatorGrid", _Stub)
    monkeypatch.setattr(gui_main, "TrainingDashboardWindow", _Stub)
    monkeypatch.setattr(gui_main, "QTimer", _FakeTimer)
    monkeypatch.setattr(gui_main.threading, "Thread", _FakeThread)
    monkeypatch.setattr("src.gui.highlight_recorder.HighlightRecorder", _Stub)
    monkeypatch.setattr("src.gui.watchdog.Watchdog", _Stub)

    fake = types.SimpleNamespace(
        training_thread=None, app=None, grid_window=None,
        _frame_sink=lambda results: None,
        _on_tick=lambda: None,
        _drain_narrator=lambda: None,
    )

    config = {
        "profile_path": str(profile_path),
        "rom_path": "dummy.nes",
        "num_instances": 1,
        "population_size": 1,
    }

    AppController._on_start(fake, config)

    expected = derive_checkpoint_dir("./checkpoints", "Super Mario Bros.") / "metrics.jsonl"
    assert fake._dashboard_config["metrics_path"] == str(expected)
    assert fake._dashboard_config["metrics_path"] != str(Path("./checkpoints/metrics.jsonl"))


def test_poll_shutdown_deadline_leaves_orphan_thread_alive(caplog):
    # Reproduces the reported defect: if the training thread fails to
    # join within the Stop deadline, _poll_shutdown must NOT clear
    # training_thread or finalize the stop -- the thread is still alive
    # and running for real. Clearing it would make AppController._on_start's
    # alive-thread guard fall through (it only fires when training_thread
    # is not None), letting a fresh Trainer spin up on top of the
    # still-running orphan, both writing into the same checkpoint_dir.
    import logging
    import time

    from src.gui.main import AppController

    class _NeverJoins:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    finalize_calls: list = []
    stop_timer_calls: list = []
    orphan = _NeverJoins()
    fake = types.SimpleNamespace(
        training_thread=orphan,
        _shutdown_deadline=time.monotonic() - 1.0,
        _stop_shutdown_timer=lambda: stop_timer_calls.append(True),
        _finalize_stop=lambda: finalize_calls.append(True),
    )

    with caplog.at_level(logging.WARNING):
        AppController._poll_shutdown(fake)

    # The join was abandoned, not confirmed -- training_thread must
    # still point at the live thread so a subsequent Start is rejected
    # instead of racing a second Trainer against this orphan.
    assert fake.training_thread is orphan
    assert finalize_calls == []
    assert stop_timer_calls == [True]
    assert any(
        "abandon" in rec.message.lower() and "alive" in rec.message.lower()
        for rec in caplog.records
    )
