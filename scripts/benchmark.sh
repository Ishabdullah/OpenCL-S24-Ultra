#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ $# == 0 || ${1:-} == --help ]]; then
    echo 'Usage: scripts/benchmark.sh MODEL [--threads N] [--prompt-tokens N] [--generation-tokens N] [--repetitions N]'
    echo 'Defaults: threads=4, prompt=128, generation=64, repetitions=3, b/ub=128, Flash Attention off, f16 KV.'
    [[ $# != 0 ]] || exit 2
    exit 0
fi
model=$1; shift
threads=4; prompt=128; generation=64; repetitions=3
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || usage_error "Option requires a value: $1"
    case $1 in
        --threads) threads=$2 ;;
        --prompt-tokens) prompt=$2 ;;
        --generation-tokens) generation=$2 ;;
        --repetitions) repetitions=$2 ;;
        *) usage_error "Unknown benchmark option: $1" ;;
    esac
    shift 2
done
for n in "$threads" "$prompt" "$generation" "$repetitions"; do
    [[ $n =~ ^[1-9][0-9]*$ ]] || usage_error 'Benchmark values must be single positive integers.'
done
check_environment
require_build
check_model "$model"
python "$PROJECT_ROOT/scripts/validate-elf.py" "$PROJECT_ROOT"
model=$(realpath -- "$model")
mkdir -p "$LOG_DIR"
out=$(mktemp -d "$LOG_DIR/benchmark-XXXXXXXX")
python "$PROJECT_ROOT/scripts/benchmark.py" "$BIN_DIR/llama-bench" "$model" "$out" "$threads" "$prompt" "$generation" "$repetitions"
