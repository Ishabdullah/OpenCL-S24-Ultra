#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ $# != 5 ]]; then
    echo 'Usage: run-unified-qwen-pilot.sh BACKEND SLOTS TOKENS NGL OUT_DIR' >&2
    exit 2
fi
backend=$1
slots=$2
tokens=$3
ngl=$4
out_dir=$5
case $backend in cpu|gpu|npu|gpu+npu) ;; *) echo "Invalid backend: $backend" >&2; exit 2 ;; esac
case $slots in 48|64|80) ;; *) echo "Invalid slot count: $slots" >&2; exit 2 ;; esac
case $tokens in 2|8|16|32|64) ;; *) echo "Invalid token count: $tokens" >&2; exit 2 ;; esac
case $ngl in 2|4|6) ;; *) echo "Invalid accelerator layer count: $ngl" >&2; exit 2 ;; esac
[[ $out_dir = /* ]] || out_dir="$ROOT/$out_dir"
mkdir -p -- "$out_dir"
out_dir=$(cd -- "$out_dir" && pwd -P)
if [[ -e "$out_dir/run.json" ]]; then
    echo "Refusing to overwrite completed run: $out_dir" >&2
    exit 2
fi

locked=0
lock_json=false
if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock
    locked=1
    lock_json=true
fi
unlock() {
    if (( locked )) && command -v termux-wake-unlock >/dev/null 2>&1; then
        termux-wake-unlock || true
    fi
}
trap unlock EXIT HUP INT TERM
printf '%s\n' "$$" > "$out_dir/launcher.pid"
printf '{"backend":"%s","slots":%s,"tokens":%s,"ngl":%s,"wake_lock":%s}\n' \
    "$backend" "$slots" "$tokens" "$ngl" "$lock_json" \
    > "$out_dir/launcher.json"
cd -- "$ROOT"
python reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py \
    --run --pilot --backend "$backend" --pilot-slots "$slots" \
    --pilot-tokens "$tokens" --ngl "$ngl" --out-dir "$out_dir"
