"""
Audio mixer UI: mute / solo one instance / all instances mixed, plus
a master volume slider.

The actual audio output + song-change detection lives in the trainer
subprocess (see `src/audio/ram_music.py`) because the RAM snapshots
flow through there. This window just pushes mixer-mode changes over
a queue that the trainer drains each generation.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


_SETTINGS_ORG = "NeuralEntertainmentSystem"
_SETTINGS_APP = "gui"


class AudioMixerWindow(QMainWindow):
    def __init__(
        self,
        num_instances: int,
        on_change: Callable[[dict], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NES — Audio Mixer")
        self.resize(520, 360)
        self._on_change = on_change
        self.num = num_instances
        self._build_ui()
        self._restore_state()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        help_label = QLabel(
            "Audio is driven by each instance's RAM 'current-song' byte.\n"
            "Drop pre-rendered .ogg/.wav in audio/<game>/<song_id>.ogg for\n"
            "real music. Missing songs play a deterministic tone."
        )
        help_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        # Track each radio's mode-string so _restore_state can find the
        # right button to re-check when persisted state is loaded.
        self._mode_buttons: dict[str, QRadioButton] = {}

        mute = QRadioButton("Mute")
        mute.setChecked(True)
        mute.toggled.connect(lambda ok: ok and self._emit("mute"))
        self._group.addButton(mute)
        root.addWidget(mute)
        self._mode_buttons["mute"] = mute

        all_btn = QRadioButton("All instances (mixed)")
        all_btn.toggled.connect(lambda ok: ok and self._emit("all"))
        self._group.addButton(all_btn)
        root.addWidget(all_btn)
        self._mode_buttons["all"] = all_btn

        root.addWidget(QLabel("Solo one instance:"))
        # Per-instance radio + (future) per-instance gain. Kept minimal —
        # most users want mute/all/solo; the gain knob is the master only.
        for i in range(self.num):
            rb = QRadioButton(f"Instance {i}")
            rb.toggled.connect(
                lambda ok, idx=i: ok and self._emit(f"solo-{idx}")
            )
            self._group.addButton(rb)
            root.addWidget(rb)
            self._mode_buttons[f"solo-{i}"] = rb

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Master volume:"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(40)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.vol_slider)
        self.vol_value_label = QLabel("40%")
        self.vol_value_label.setStyleSheet("font-family: Menlo")
        vol_row.addWidget(self.vol_value_label)
        root.addLayout(vol_row)

    def _on_volume_changed(self, v: int) -> None:
        self.vol_value_label.setText(f"{v}%")
        self._on_change({"volume": v / 100.0})
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue("audio_mixer/volume", v)

    def _emit(self, mode: str) -> None:
        self._on_change({"mode": mode})
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue("audio_mixer/mode", mode)

    def _restore_state(self) -> None:
        """Pull last-session mode + volume from QSettings. Falls back to
        defaults if nothing is stored or the stored mode references an
        instance index past the current num_instances (e.g. last run
        was 16 workers, this run is 8). Suppresses the toggle/change
        signals during restore so we don't ping the trainer with stale
        events before the user has actually touched anything."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        stored_vol = s.value("audio_mixer/volume", 40, type=int)
        stored_vol = max(0, min(100, int(stored_vol)))
        # blockSignals so the slider's restore doesn't fire _on_volume_changed.
        self.vol_slider.blockSignals(True)
        self.vol_slider.setValue(stored_vol)
        self.vol_slider.blockSignals(False)
        self.vol_value_label.setText(f"{stored_vol}%")

        stored_mode = s.value("audio_mixer/mode", "mute", type=str)
        button = self._mode_buttons.get(stored_mode)
        if button is None:
            # Stored "solo-N" but N is now out of range — fall back to mute.
            button = self._mode_buttons["mute"]
            stored_mode = "mute"
        # blockSignals on the whole group so re-checking doesn't fire
        # _emit (which would push to the trainer + re-write settings).
        for b in self._group.buttons():
            b.blockSignals(True)
        button.setChecked(True)
        for b in self._group.buttons():
            b.blockSignals(False)

        # Push the restored state to the trainer in a single explicit
        # call so the audio backend reflects the visible UI on open.
        self._on_change({"mode": stored_mode, "volume": stored_vol / 100.0})
