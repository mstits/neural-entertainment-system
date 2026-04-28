#!/usr/bin/env bash
# Build an NES.app macOS bundle with py2app.
# Run this from the repo root with the venv activated.

set -e

if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: macOS only." >&2
    exit 1
fi

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "ERROR: activate the venv first: source .venv/bin/activate" >&2
    exit 1
fi

pip install --quiet py2app

# Minimal setup.py for py2app; written inline so there's no clutter at the
# repo root when we're not bundling.
cat > setup_app.py <<'PY'
from setuptools import setup

APP = ["src/gui/main.py"]
OPTIONS = {
    "argv_emulation": False,
    "packages": [
        "src",
        "torch",
        "numpy",
        "PyQt6",
        "pyqtgraph",
        "nes_core",
        "yaml",
    ],
    "includes": ["src.emulation", "src.gui", "src.models", "src.training", "src.utils"],
    "iconfile": None,
    "plist": {
        "CFBundleName": "NES",
        "CFBundleDisplayName": "NES",
        "CFBundleIdentifier": "org.nes-evolve.app",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "NSHighResolutionCapable": True,
    },
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
PY

rm -rf build dist
python setup_app.py py2app

echo ""
echo "Built: dist/main.app"
echo "Rename and/or notarize as needed."
