#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
model=''; matrix=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --help) echo 'Usage: scripts/verify-opencl.sh [--model /path/to/model.gguf] [--matrix-tests]'; exit 0 ;;
        --model) [[ $# -ge 2 ]] || usage_error '--model requires a file'; model=$2; shift 2 ;;
        --matrix-tests) matrix=1; shift ;;
        *) usage_error "Unknown verification option: $1" ;;
    esac
done
"$PROJECT_ROOT/scripts/check-device.sh"
require_build
prepare_work
python "$PROJECT_ROOT/scripts/validate-source.py" "$PROJECT_ROOT" "$SOURCE_DIR"
python "$PROJECT_ROOT/scripts/validate-elf.py" "$PROJECT_ROOT"
run_logged 'Initialize sphal OpenCL and enumerate device (no model)' "$LOG_DIR/device.log" "$BIN_DIR/llama-bench" --list-devices
python "$PROJECT_ROOT/scripts/validate-output.py" device "$LOG_DIR/device.log" "$LOG_DIR/device.json"
cat "$LOG_DIR/device.log"
if [[ -n $model ]]; then
    check_model "$model"
    model=$(realpath -- "$model")
    info "Run deterministic inference (log: $LOG_DIR/inference.stderr)"
    if "$BIN_DIR/llama-completion" -m "$model" -ngl 99 -dev GPUOpenCL \
        -c 512 -b 128 -ub 128 -t 4 -fa off -n 32 --temp 0 --seed 1234 -no-cnv \
        --log-verbosity 4 -p 'def add(a, b):' \
        > "$LOG_DIR/inference.stdout" 2> "$LOG_DIR/inference.stderr"; then
        python "$PROJECT_ROOT/scripts/validate-output.py" inference "$LOG_DIR/inference.stderr" "$LOG_DIR/inference.json"
        cat "$LOG_DIR/inference.stdout"
    else
        rc=$?
        tail -n 40 "$LOG_DIR/inference.stderr" >&2
        die "Inference failed (exit $rc); device detection alone is not an inference pass."
    fi
else
    info 'Device detected. Actual model offload has NOT been tested in this invocation.'
fi
if [[ $matrix == 1 ]]; then
    cd "$LOG_DIR"
    run_logged 'Numerical Q4_K/Q6_K matrix tests against CPU' "$LOG_DIR/matrix-tests.log" \
        "$BIN_DIR/test-backend-ops" test -b GPUOpenCL -o MUL_MAT -p 'type_a=q[46]_K,type_b=f32,m=16,'
    tail -n 8 "$LOG_DIR/matrix-tests.log"
fi
