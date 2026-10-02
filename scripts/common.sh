#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
WORK_DIR="$PROJECT_ROOT/.work"
SOURCE_DIR="$WORK_DIR/llama.cpp"
BUILD_DIR="$WORK_DIR/build"
BIN_DIR="$BUILD_DIR/bin"
LOG_DIR="$WORK_DIR/logs"
OWNER_TEXT='OpenCL-S24-Ultra generated data v1'
info() { printf '[OpenCL-S24-Ultra] %s\n' "$*"; }
die() { printf '[OpenCL-S24-Ultra] ERROR: %s\n' "$*" >&2; exit 1; }
usage_error() { printf '%s\n' "$*" >&2; exit 2; }
check_environment() {
    [[ $EUID -ne 0 ]] || die 'Run as the ordinary Termux user, not root.'
    [[ -n ${PREFIX:-} && -x "$PREFIX/bin/pkg" && -x "$PREFIX/bin/bash" ]] || die 'Android Termux from F-Droid is required.'
    [[ -x /system/bin/getprop ]] || die 'Android getprop is unavailable.'
    [[ $(uname -m) == aarch64 ]] || die "Unsupported architecture: $(uname -m); aarch64 is required."
}
prepare_work() {
    [[ ! -L "$WORK_DIR" ]] || die '.work must not be a symlink.'
    mkdir -p "$WORK_DIR"
    if [[ -f "$WORK_DIR/.owner" ]]; then
        [[ $(cat "$WORK_DIR/.owner") == "$OWNER_TEXT" ]] || die 'Unknown .work owner; refusing to modify it.'
    else
        [[ ! -e "$SOURCE_DIR" && ! -e "$BUILD_DIR" ]] || die 'Unmarked source/build directories exist; move them aside.'
        printf '%s\n' "$OWNER_TEXT" > "$WORK_DIR/.owner"
    fi
    [[ ! -L "$SOURCE_DIR" && ! -L "$BUILD_DIR" && ! -L "$LOG_DIR" ]] || die 'Managed source/build/log directories must not be symlinks.'
    mkdir -p "$LOG_DIR"
}
isolate_runtime() {
    # Never append the inherited path or $PREFIX/lib: vendor binder loading breaks.
    export LD_LIBRARY_PATH="$BIN_DIR"
    export GGML_OPENCL_KERNEL_CACHE_DIR="$WORK_DIR/kernel-cache"
    unset GGML_BACKEND_PATH
}
require_build() {
    [[ -f "$WORK_DIR/.owner" && $(cat "$WORK_DIR/.owner") == "$OWNER_TEXT" ]] || die 'No installation; run ./install.sh.'
    [[ ! -L "$WORK_DIR" && ! -L "$BUILD_DIR" && ! -L "$SOURCE_DIR" ]] || die 'Managed directories must not be symlinks.'
    for binary in llama-completion llama-bench test-backend-ops; do
        [[ -x "$BIN_DIR/$binary" ]] || die "Missing project binary $binary; run ./install.sh --rebuild."
        [[ $(dirname -- "$(realpath -- "$BIN_DIR/$binary")") == "$BIN_DIR" ]] || die "Project binary $binary points outside this build."
    done
    [[ -f "$BUILD_DIR/CMakeCache.txt" ]] || die 'Build configuration is missing.'
    isolate_runtime
}
check_model() {
    [[ -f "$1" && -r "$1" ]] || die "GGUF is not a readable file: $1"
    [[ $(head -c 4 -- "$1") == GGUF ]] || die "File lacks a GGUF header: $1"
}
run_logged() {
    local stage=$1 log=$2
    shift 2
    info "$stage (log: $log)"
    if "$@" > "$log" 2>&1; then return 0; else
        local rc=$?
        tail -n 40 "$log" >&2
        die "$stage failed (exit $rc). See $log; installation has not been marked complete."
    fi
}
