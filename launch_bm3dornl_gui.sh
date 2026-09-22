#!/usr/bin/env bash
# Launch the bm3dornl egui GUI.
#
# Runs the pre-built binary if present; otherwise builds and runs it
# through the pixi "gui" task.
#
# Usage: ./launch_bm3dornl_gui.sh [bm3dornl-gui arguments...]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$REPO_DIR/src/rust_core/target/release/bm3dornl-gui"

# GUI apps need a display (e.g. a ThinLinc session).
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "Error: no display found (DISPLAY/WAYLAND_DISPLAY unset)." >&2
    echo "Run this from a graphical session such as ThinLinc." >&2
    exit 1
fi

if [[ -x "$BINARY" ]]; then
    exec "$BINARY" "$@"
fi

PIXI="$(command -v pixi || true)"
if [[ -z "$PIXI" ]]; then
    echo "Error: bm3dornl-gui is not built and pixi was not found to build it." >&2
    exit 1
fi

echo "Building bm3dornl-gui (release) via pixi..."
cd "$REPO_DIR"
exec "$PIXI" run gui
