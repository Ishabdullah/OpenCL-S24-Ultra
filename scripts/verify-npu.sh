#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/npu-common.sh"
model=''; ngl=99; matrix=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --help) echo 'Usage: scripts/verify-npu.sh [--model FILE] [--ngl N] [--device-only]'; exit 0 ;;
        --model) [[ $# -ge 2 ]] || usage_error '--model requires FILE'; model=$2; shift 2 ;;
        --ngl) [[ $# -ge 2 ]] || usage_error '--ngl requires N'; ngl=$2; shift 2 ;;
        --device-only) matrix=0; shift ;;
        *) usage_error "Unknown NPU verification option: $1" ;;
    esac
done
[[ $ngl =~ ^[1-9][0-9]*$ ]] || usage_error 'Verification requires positive --ngl.'
"$PROJECT_ROOT/scripts/check-npu.sh"
npu_require
npu_prepare
python "$PROJECT_ROOT/scripts/npu/validate.py" source "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/npu/validate.py" elf "$PROJECT_ROOT"
run_logged 'Enumerate native Hexagon backend (no model)' "$LOG_DIR/device.log" "$BIN_DIR/llama-bench" --list-devices
python "$PROJECT_ROOT/scripts/npu/validate.py" device "$PROJECT_ROOT" "$LOG_DIR/device.log"
if [[ $matrix == 1 ]]; then
    run_logged 'Execute eight source-built Q4_K/Q6_K DSP matrices against CPU reference' "$LOG_DIR/matrix.log" \
        "$BIN_DIR/test-backend-ops" test -b HTP0 -o MUL_MAT --test-file "$PROJECT_ROOT/examples/npu-matrix-cases.txt" -j 1
    python "$PROJECT_ROOT/scripts/npu/validate.py" matrix "$PROJECT_ROOT" "$LOG_DIR/matrix.log"
fi
if [[ -n $model ]]; then
    check_model "$model"
    model=$(realpath -- "$model")
    info "Generate 32 deterministic tokens on HTP0 (logs: $LOG_DIR/inference.*)"
    if "$BIN_DIR/llama-completion" -m "$model" -ngl "$ngl" -dev HTP0 \
        -c 512 -b 128 -ub 128 -t 4 -fa off -n 32 --temp 0 --seed 1234 -no-cnv \
        --log-verbosity 4 -p 'def add(a, b):' > "$LOG_DIR/inference.stdout" 2> "$LOG_DIR/inference.stderr"; then
        python "$PROJECT_ROOT/scripts/npu/validate.py" inference "$PROJECT_ROOT" "$LOG_DIR/inference.stderr"
        cat "$LOG_DIR/inference.stdout"
    else
        rc=$?; tail -n 40 "$LOG_DIR/inference.stderr" >&2
        die "NPU inference failed (exit $rc). Device detection alone is not a model inference pass."
    fi
else
    info 'No model supplied: model generation/offload was NOT tested in this invocation.'
fi
