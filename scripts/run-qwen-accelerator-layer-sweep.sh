#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
locked=0
if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock
    locked=1
fi
unlock() {
    if (( locked )) && command -v termux-wake-unlock >/dev/null 2>&1; then
        termux-wake-unlock || true
    fi
}
trap unlock EXIT HUP INT TERM
export TERMUX_WAKE_LOCK_ACTIVE="$locked"
cd -- "$ROOT"
python scripts/run-qwen-accelerator-layer-sweep.py "$@"
