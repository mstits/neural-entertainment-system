# Manual GUI Test Checklist

The PyQt6 GUI can't be automated on this macOS + Qt 6.11 combination —
pytest's stdio redirection triggers a Qt fatal assertion on widget
construction. Exercise these paths by hand on a real desktop session.

## Launch
- [ ] `python -m src.gui.main` starts with no tracebacks on stderr
- [ ] Main window titled "NES" appears at roughly 720×480
- [ ] Game profile dropdown is populated from `configs/*.yaml`

## ROM picker
- [ ] Clicking "Choose ROM…" opens a file dialog filtered to `*.nes`
- [ ] Selecting a ROM updates the label next to "ROM:"

## Start/Stop
- [ ] Clicking Start with no ROM shows the "Choose a ROM first." message
- [ ] Clicking Start with a valid ROM:
    - opens the emulator grid window
    - opens the metrics window
    - start button disables, stop enables
- [ ] Metrics label updates ≤1s after start (shows "Generation: 0…")
- [ ] Emulator grid updates at roughly 30 fps
- [ ] Clicking Stop closes the grid + metrics windows and re-enables Start

## Shutdown
- [ ] Closing the main window cleanly exits the process
- [ ] No stray `.venv/bin/python` processes remain (`pgrep -f src.gui.main`)
- [ ] No `/dev/shm` leaks remain (`ls /tmp/psm_*` should be empty after quit)

## Standalone headless self-check
- [ ] `QT_QPA_PLATFORM=offscreen python tests/gui_selftest.py` prints
      "GUI selftest OK"
