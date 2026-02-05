#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime-intel"
REQ_FILE="$SCRIPT_DIR/requirements-inference-intel.txt"
CLEAN=0
DO_FREEZE=0

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    freeze) DO_FREEZE=1 ;;
    *) ;;
  esac
done

echo "============================================================================="
echo "DSU-Audiosep Runtime Builder (Mac Intel x86_64 / CPU)"
echo "============================================================================="
echo "Python: 3.10 (recommended for llvmlite/numba wheel compatibility)"
echo "============================================================================="

ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
  echo "WARNING: Non-x86_64 detected: $ARCH (use build_runtime_mac_mps.sh on ARM)"
fi

if [ -d "$RUNTIME_DIR" ] && [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning existing runtime: $RUNTIME_DIR"
  rm -rf "$RUNTIME_DIR"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

if [ ! -d "$RUNTIME_DIR" ]; then
  echo "Creating runtime with uv (Python 3.10)..."
  uv venv "$RUNTIME_DIR" --python 3.10
else
  echo "Reusing runtime: $RUNTIME_DIR"
fi

source "$RUNTIME_DIR/bin/activate"

# CRITICAL: Install llvmlite/numba with pre-built wheels first to avoid C++ compilation
# (llvmlite 0.46+ tries to build from source and fails on Intel Mac with LLVM 20.x)
echo ""
echo "Installing llvmlite/numba with pre-built wheels (avoids compilation on Intel Mac)..."
uv pip install --only-binary=:all: llvmlite==0.41.1 numba==0.58.1

echo ""
echo "Installing dependencies from requirements..."
uv pip install -r "$REQ_FILE"

echo ""
echo "Runtime created at: $RUNTIME_DIR"
echo "Python: $RUNTIME_DIR/bin/python"
echo "============================================================================="

if [ "$DO_FREEZE" -eq 1 ]; then
  echo ""
  echo "Building frozen executable (cx_Freeze)..."
  python build_dsu_audiosep.py
  echo ""
  echo "Build complete. Output: dist/dsu-audiosep/"
fi

deactivate 2>/dev/null || true
