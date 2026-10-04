#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
WORK_DIR="$PROJECT_ROOT/.work-npu"
SOURCE_DIR="$WORK_DIR/llama.cpp"
BUILD_DIR="$WORK_DIR/build"
BIN_DIR="$BUILD_DIR/bin"
LOG_DIR="$WORK_DIR/logs"
OWNER_TEXT='OpenCL-S24-Ultra NPU generated data v1'
DSP_DIR="$BUILD_DIR/ggml/src/ggml-hexagon"
npu_prepare() {
    prepare_work
    [[ ! -L "$WORK_DIR/toolchain" && ! -L "$WORK_DIR/downloads" ]] || die 'NPU toolchain/download directories must not be symlinks.'
}
npu_runtime() {
    export LD_LIBRARY_PATH="$BIN_DIR"
    export ADSP_LIBRARY_PATH="$DSP_DIR"
    unset LD_PRELOAD GGML_BACKEND_PATH
    local name
    while IFS= read -r name; do unset "$name"; done < <(compgen -e | sed -n '/^GGML_HEXAGON_/p')
    export GGML_HEXAGON_PROFILE=0
}
npu_require() {
    [[ -f "$WORK_DIR/.owner" && $(cat "$WORK_DIR/.owner") == "$OWNER_TEXT" ]] || die 'No NPU installation; run ./install-npu.sh.'
    [[ ! -L "$WORK_DIR" && ! -L "$BUILD_DIR" && ! -L "$SOURCE_DIR" ]] || die 'Managed directories must not be symlinks.'
    for binary in llama-completion llama-bench test-backend-ops; do
        [[ -x "$BIN_DIR/$binary" && $(dirname -- "$(realpath -- "$BIN_DIR/$binary")") == "$BIN_DIR" ]] || die "Missing/outside-build NPU binary $binary; run ./install-npu.sh --rebuild."
    done
    [[ -f "$DSP_DIR/libggml-htp-v75.so" && $(dirname -- "$(realpath -- "$DSP_DIR/libggml-htp-v75.so")") == "$DSP_DIR" ]] || die 'Source-built v75 DSP library is missing/outside this build.'
    npu_runtime
}
