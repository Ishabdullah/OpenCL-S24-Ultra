#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: examples/run-qwen.sh /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec "$root/scripts/run-model.sh" "$1" -n 32 --temp 0 --seed 1234 -no-cnv -p 'def add(a, b):'
