#!/usr/bin/env bash
# Launch the bm3dornl egui GUI. Run from anywhere:
#   ./scripts/launch_gui.sh          # release build (default)
#   ./scripts/launch_gui.sh --debug  # debug build with backtraces
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v pixi >/dev/null 2>&1; then
    echo "Error: pixi not found on PATH. Install it from https://pixi.sh" >&2
    exit 1
fi

TASK="gui"
if [[ "${1:-}" == "--debug" ]]; then
    TASK="gui-debug"
fi

cd "$REPO_ROOT"
exec pixi run "$TASK"
