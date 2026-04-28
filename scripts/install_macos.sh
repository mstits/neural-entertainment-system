#!/usr/bin/env bash
# Neural Entertainment System macOS installer
# Tested on macOS 13+ with Apple Silicon (M1/M2/M3/M4)

set -e

echo "==> Neural Entertainment System macOS Installer"
echo ""

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This installer is for macOS only."
    exit 1
fi

# Check for Xcode CLI tools (needed to build the Rust nes_core crate
# via maturin — clang + lld come from the CLT package).
if ! xcode-select -p &>/dev/null; then
    echo "==> Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    echo "Please complete the installer dialog, then re-run this script."
    exit 1
fi

# Check for Rust toolchain (needed to build nes_core).
if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo (Rust toolchain) not found."
    echo "Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check for Python 3.11
if ! command -v python3.11 &>/dev/null; then
    echo "ERROR: Python 3.11 is required."
    echo "Install via Homebrew: brew install python@3.11"
    exit 1
fi

# Create virtual environment
if [[ ! -d ".venv" ]]; then
    echo "==> Creating Python virtual environment..."
    python3.11 -m venv .venv
fi

# Activate and install
echo "==> Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# PyTorch with MPS support (comes with standard macOS wheel)
pip install torch torchvision

# Core dependencies
pip install -r requirements.txt

# Build + install the Rust nes_core crate as an editable wheel.
pip install maturin
(cd nes_core && maturin develop --release)

# Verify MPS is available
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS backend not available'; print('==> MPS backend OK')"

# Verify nes_core (the Rust NES core).
python -c "import nes_core; print('==> nes_core OK:', nes_core.NESEnvironment)"

# Optional: PGO build for +81% emulator throughput. Skippable for
# faster installs; users can run `scripts/pgo_build.sh full` later.
if [[ "${NES_EVOLVE_PGO:-1}" == "1" && -f "roms/zelda.nes" ]]; then
    echo ""
    echo "==> Applying Profile-Guided Optimization (PGO)..."
    echo "    (+81% throughput, ~3 minutes. Set NES_EVOLVE_PGO=0 to skip.)"
    bash scripts/pgo_build.sh full bench || {
        echo "==> PGO build failed; continuing with the non-PGO wheel."
        echo "    Non-fatal — training will still work, just slower."
    }
elif [[ "${NES_EVOLVE_PGO:-1}" != "1" ]]; then
    echo "==> PGO skipped (NES_EVOLVE_PGO=${NES_EVOLVE_PGO})."
else
    echo "==> PGO skipped (roms/zelda.nes missing)."
    echo "    Run 'bash scripts/pgo_build.sh full' after adding ROMs."
fi

echo ""
echo "==> Installation complete!"
echo ""
echo "To activate the environment:  source .venv/bin/activate"
echo "To launch the GUI:            python -m src.gui.main"
echo "To run training headless:     python -m src.training.trainer --help"
echo "To re-apply PGO later:        bash scripts/pgo_build.sh apply"
echo ""
