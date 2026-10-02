#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
clean=0; jobs=4
while [[ $# -gt 0 ]]; do
    case $1 in
        --help) echo 'Usage: scripts/build.sh [--clean-build] [--jobs N]'; exit 0 ;;
        --clean-build) clean=1; shift ;;
        --jobs) [[ $# -ge 2 ]] || usage_error '--jobs requires N'; jobs=$2; shift 2 ;;
        *) usage_error "Unknown build option: $1" ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || usage_error 'Jobs must be a positive integer.'
check_environment
prepare_work
for tool in git cmake ninja python cc c++ llvm-readelf; do
    command -v "$tool" >/dev/null || die "Missing $tool; run ./install.sh."
done
mapfile -t upstream < <(python "$PROJECT_ROOT/scripts/validate-source.py" --metadata "$PROJECT_ROOT")
[[ ${#upstream[@]} == 2 ]] || die 'Invalid upstream metadata.'
url=${upstream[0]}; commit=${upstream[1]}
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    [[ ! -e "$SOURCE_DIR" ]] || die 'Source path exists without its own Git repository.'
    git init -q "$SOURCE_DIR"
    git -C "$SOURCE_DIR" remote add origin "$url"
fi
[[ $(git -C "$SOURCE_DIR" rev-parse --show-toplevel) == "$SOURCE_DIR" ]] || die 'Source is not a separate checkout.'
[[ $(git -C "$SOURCE_DIR" remote get-url origin) == "$url" ]] || die 'Source origin differs from pinned upstream.'
if ! head_commit=$(git -C "$SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null); then
    run_logged 'Fetch pinned upstream source' "$LOG_DIR/fetch.log" git -C "$SOURCE_DIR" fetch --depth 1 origin "$commit"
    [[ $(git -C "$SOURCE_DIR" rev-parse FETCH_HEAD) == "$commit" ]] || die 'Fetched commit differs from pin.'
    run_logged 'Check out pinned upstream source' "$LOG_DIR/checkout.log" git -C "$SOURCE_DIR" checkout --detach "$commit"
else
    [[ $head_commit == "$commit" ]] || die "Source HEAD is $head_commit, expected $commit. Refusing to reset user work."
fi
patch="$PROJECT_ROOT/patches/llama.cpp-qualcomm-sphal.patch"
if git -C "$SOURCE_DIR" apply --check "$patch" > "$LOG_DIR/patch.log" 2>&1; then
    run_logged 'Apply Qualcomm sphal patch' "$LOG_DIR/patch.log" git -C "$SOURCE_DIR" apply "$patch"
elif git -C "$SOURCE_DIR" apply --reverse --check "$patch" >> "$LOG_DIR/patch.log" 2>&1; then
    info 'Patch already applied; validating source hashes.'
else
    tail -n 25 "$LOG_DIR/patch.log" >&2
    die 'Patch neither applies nor reverses cleanly. Check pinned commit/local changes; no reset performed.'
fi
python "$PROJECT_ROOT/scripts/validate-source.py" "$PROJECT_ROOT" "$SOURCE_DIR"
if [[ $clean == 1 ]]; then
    info "Removing only managed build directory: $BUILD_DIR"
    rm -rf -- "$BUILD_DIR"
fi
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
run_logged 'Configure generic OpenCL build' "$LOG_DIR/configure.log" \
    cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$PREFIX/bin/cc" -DCMAKE_CXX_COMPILER="$PREFIX/bin/c++" \
    -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS= \
    -DGGML_OPENCL=ON -DGGML_OPENCL_USE_SPHAL=ON -DGGML_OPENCL_TARGET_VERSION=300 \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF \
    -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_PROFILING=OFF -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON
run_logged "Build inference, benchmark and backend validation ($jobs jobs)" "$LOG_DIR/build.log" \
    cmake --build "$BUILD_DIR" --target llama-completion llama-bench test-backend-ops -j "$jobs"
info "Built project binaries: $BIN_DIR"
