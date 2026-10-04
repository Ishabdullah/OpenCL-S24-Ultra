#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/npu-common.sh"
jobs=2; clean=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --help) echo 'Usage: scripts/build-npu.sh [--jobs 1|2] [--clean-build] (run install-npu.sh first)'; exit 0 ;;
        --jobs) [[ $# -ge 2 ]] || usage_error '--jobs requires N'; jobs=$2; shift 2 ;;
        --clean-build) clean=1; shift ;;
        *) usage_error "Unknown build option: $1" ;;
    esac
done
[[ $jobs == 1 || $jobs == 2 ]] || usage_error 'Use --jobs 1 or 2 to bound native/nested DSP build load.'
check_environment
npu_prepare
mapfile -t pin < <(python "$PROJECT_ROOT/scripts/npu/validate.py" metadata "$PROJECT_ROOT")
[[ ${#pin[@]} == 2 ]] || die 'Invalid NPU source metadata.'
url=${pin[0]}; commit=${pin[1]}
[[ ! -L "$SOURCE_DIR/.git" ]] || die "Source Git directory must not be a symlink."
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    [[ ! -e "$SOURCE_DIR" ]] || die 'NPU source directory exists without a separate Git checkout.'
    git init -q "$SOURCE_DIR"
    git -C "$SOURCE_DIR" remote add origin "$url"
fi
[[ $(git -C "$SOURCE_DIR" rev-parse --show-toplevel) == "$SOURCE_DIR" && $(git -C "$SOURCE_DIR" remote get-url origin) == "$url" ]] || die 'NPU source checkout/origin differs from the managed pin.'
if ! current=$(git -C "$SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null); then
    if ! git -C "$SOURCE_DIR" cat-file -e "$commit^{commit}" 2>/dev/null; then
        fetched=0
        for attempt in 1 2 3; do
            info "Fetch pinned llama.cpp, attempt $attempt/3 (log: $LOG_DIR/fetch-$attempt.log)"
            if git -C "$SOURCE_DIR" -c http.version=HTTP/1.1 fetch --depth 1 origin "$commit" > "$LOG_DIR/fetch-$attempt.log" 2>&1; then
                fetched=1; break
            fi
            tail -n 8 "$LOG_DIR/fetch-$attempt.log" >&2
        done
        [[ $fetched == 1 ]] || die 'Pinned source download failed after three attempts; check network and rerun. Partial source is not built.'
        [[ $(git -C "$SOURCE_DIR" rev-parse FETCH_HEAD) == "$commit" ]] || die 'Fetched source differs from pin.'
    fi
    run_logged 'Check out pinned source' "$LOG_DIR/checkout.log" git -C "$SOURCE_DIR" checkout --detach "$commit"
else
    [[ $current == "$commit" ]] || die 'NPU source HEAD differs from pin; refusing to reset user work.'
fi
patch="$PROJECT_ROOT/patches/llama.cpp-hexagon-sphal.patch"
if git -C "$SOURCE_DIR" apply --check "$patch" > "$LOG_DIR/patch.log" 2>&1; then
    run_logged 'Apply native Hexagon sphal patch' "$LOG_DIR/patch.log" git -C "$SOURCE_DIR" apply "$patch"
elif git -C "$SOURCE_DIR" apply --reverse --check "$patch" >> "$LOG_DIR/patch.log" 2>&1; then
    info 'NPU patch already applied.'
else
    tail -n 25 "$LOG_DIR/patch.log" >&2
    die 'NPU patch cannot apply/reverse at the pinned source. No reset performed.'
fi
python "$PROJECT_ROOT/scripts/npu/validate.py" source "$PROJECT_ROOT"
[[ $clean == 0 ]] || { info "Remove only managed NPU build: $BUILD_DIR"; rm -rf -- "$BUILD_DIR"; }
sdk="$WORK_DIR/toolchain/sdk/6.6.0.0"
emulation="$WORK_DIR/toolchain/emulation"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_LIBRARY_PATH LD_PRELOAD
run_logged 'Configure native ARM64 host and Hexagon v75 DSP build' "$LOG_DIR/configure.log" \
    cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$PREFIX/bin/cc" -DCMAKE_CXX_COMPILER="$PREFIX/bin/c++" \
    -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS= \
    -DGGML_HEXAGON=ON -DGGML_HEXAGON_USE_SPHAL=ON -DGGML_HEXAGON_ARCHITECTURES=v75 \
    -DGGML_HEXAGON_DSP_BUILD_JOBS="$jobs" -DGGML_HEXAGON_HTP_BUILD_TYPE=Release \
    -DHEXAGON_SDK_ROOT="$sdk" -DHEXAGON_TOOLS_ROOT="$sdk/tools/HEXAGON_Tools/19.0.07" \
    -DGGML_HEXAGON_QAIC_DIR="$emulation" -DGGML_HEXAGON_EMULATION_DIR="$emulation" \
    -DPREBUILT_LIB_DIR=android_aarch64 -DGGML_OPENCL=OFF -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON
run_logged 'Build source-based v75 DSP kernels (emulated compiler tools, native runtime)' "$LOG_DIR/dsp-build.log" \
    python "$PROJECT_ROOT/scripts/npu/guarded-build.py" "$LOG_DIR/dsp-build.log" \
    nice -n 10 cmake --build "$BUILD_DIR" --target htp-v75 -j 1
run_logged 'Build native llama inference, benchmark and numerical tests' "$LOG_DIR/host-build.log" \
    python "$PROJECT_ROOT/scripts/npu/guarded-build.py" "$LOG_DIR/host-build.log" \
    nice -n 10 cmake --build "$BUILD_DIR" --target llama-completion llama-bench test-backend-ops -j "$jobs"
info "NPU build completed: $BIN_DIR"
