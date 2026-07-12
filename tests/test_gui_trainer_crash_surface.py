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
