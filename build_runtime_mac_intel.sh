#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime-intel"
REQ_FILE="$SCRIPT_DIR/requirements-inference-intel.txt"
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    *) ;;
  esac
done

echo "============================================================================="
echo "DSU-Audiosep Runtime Builder (Mac Intel x86_64 / CPU)"
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
  echo "Creating runtime with uv..."
  uv venv "$RUNTIME_DIR"
else
  echo "Reusing runtime: $RUNTIME_DIR"
fi

source "$RUNTIME_DIR/bin/activate"
uv pip install -r "$REQ_FILE"

echo ""
echo "Runtime created at: $RUNTIME_DIR"
echo "Python: $RUNTIME_DIR/bin/python"
echo "============================================================================="

deactivate 2>/dev/null || true
