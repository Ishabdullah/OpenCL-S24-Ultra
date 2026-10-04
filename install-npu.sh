#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/npu-common.sh"
jobs=2; rebuild=0; clean=0; verify=0; model=''; download_model=0; sdk_archive=''
while [[ $# -gt 0 ]]; do
    case $1 in
        --help) cat <<'HELP'
Usage: ./install-npu.sh [options]
Install Termux packages, prepare a checksum-pinned Hexagon SDK/compiler runtime,
fetch/patch pinned llama.cpp, source-build native ARM64 + Hexagon v75 binaries,
verify ELF state/device access and execute numerical DSP matrix tests.
--download-test-model  Fetch/reuse pinned Qwen2.5-Coder-1.5B Q4_K_M, then generate.
--model FILE           Use your existing GGUF and verify 32-token NPU generation.
--jobs 1|2             Build parallelism (default 2; includes bounded DSP workers).
--rebuild              Reconfigure/incrementally rebuild the managed source.
--clean-build          Replace only .work-npu/build and build from scratch.
--verify               Verify an existing installation without building/packages.
--sdk-archive FILE     Reuse the exact pinned SDK archive; SHA-256 still checked.
No root, system/vendor edits, model redistribution or emulated inference.
SDK download ~662 MiB, unpacked ~3.2 GiB. Allow at least 8 GiB free plus models.
Model-free success verifies DSP computation, not LLM model generation.
HELP
            exit 0 ;;
        --jobs) [[ $# -ge 2 ]] || usage_error '--jobs requires N'; jobs=$2; shift 2 ;;
        --model) [[ $# -ge 2 ]] || usage_error '--model requires FILE'; model=$2; shift 2 ;;
        --sdk-archive) [[ $# -ge 2 ]] || usage_error '--sdk-archive requires FILE'; sdk_archive=$2; shift 2 ;;
        --download-test-model) download_model=1; shift ;;
        --rebuild) rebuild=1; shift ;;
        --clean-build) clean=1; rebuild=1; shift ;;
        --verify) verify=1; shift ;;
        *) usage_error "Unknown NPU installer option: $1" ;;
    esac
done
[[ $jobs == 1 || $jobs == 2 ]] || usage_error '--jobs must be 1 or 2.'
[[ -z $model || $download_model == 0 ]] || usage_error 'Choose --model or --download-test-model.'
[[ $verify == 0 || $rebuild == 0 ]] || usage_error '--verify cannot be combined with build options.'
"$PROJECT_ROOT/scripts/check-npu.sh"
[[ -z $model ]] || check_model "$model"
[[ -z $sdk_archive || -r $sdk_archive ]] || die '--sdk-archive must be a readable file.'
npu_prepare
[[ $verify == 0 ]] || npu_require
if [[ $verify == 0 ]]; then
    packages=(git clang cmake ninja python curl openssl qemu-user-x86-64)
    missing=()
    for package in "${packages[@]}"; do
        [[ $(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true) == installed ]] || missing+=("$package")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        info "Install missing Termux packages: ${missing[*]}"
        pkg install -y "${missing[@]}" || die 'Package installation failed; run pkg update/check mirror and rerun.'
    else
        info "Termux dependencies already installed: ${packages[*]}"
    fi
    mkdir -p "$WORK_DIR/tools"
    run_logged 'Compile native C CDSP capability probe' "$LOG_DIR/probe-build.log" \
        "$PREFIX/bin/cc" "$PROJECT_ROOT/examples/npu-capabilities.c" -o "$WORK_DIR/tools/npu-capabilities" -landroid -ldl
    run_logged 'Check actual sphal/vendor symbols and Hexagon v75 capability' "$LOG_DIR/capabilities.log" \
        env -u LD_LIBRARY_PATH -u LD_PRELOAD "$WORK_DIR/tools/npu-capabilities" sphal
    if [[ $rebuild == 1 || ! -f "$WORK_DIR/installed.json" ]]; then
        tool_args=("$PROJECT_ROOT")
        [[ -z $sdk_archive ]] || tool_args+=(--sdk-archive "$sdk_archive")
        info 'SDK/compiler and Debian runtime remain in project storage; their own license notices are retained. See docs/NPU_INSTALL.md.'
        run_logged 'Prepare checksum-pinned SDK and isolated compiler-tool emulation' "$LOG_DIR/toolchain.log" \
            python "$PROJECT_ROOT/scripts/npu/prepare-toolchain.py" "${tool_args[@]}"
        args=(--jobs "$jobs")
        [[ $clean == 0 ]] || args+=(--clean-build)
        "$PROJECT_ROOT/scripts/build-npu.sh" "${args[@]}"
    fi
fi
if [[ $download_model == 1 ]]; then
    command -v curl >/dev/null || die 'curl is required for the explicit model download.'
    python "$PROJECT_ROOT/scripts/npu/download-test-model.py" "$PROJECT_ROOT"
    model=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["path"])' "$WORK_DIR/test-model.json")
fi
args=()
[[ -z $model ]] || args+=(--model "$model")
"$PROJECT_ROOT/scripts/verify-npu.sh" "${args[@]}"
python - "$PROJECT_ROOT" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);meta=json.loads((root/'npu-upstream.json').read_text())
meta['installed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/'.work-npu/installed.json').write_text(json.dumps(meta,indent=2)+'\n')
PY
info 'NPU installation, native DSP matrix execution and selected verification succeeded.'
[[ -n $model ]] || info 'Supply --model or --download-test-model to verify actual LLM generation.'
printf '\nNPU binaries: %s\nRun: ./scripts/run-npu.sh /path/to/model.gguf\nVerify: ./scripts/verify-npu.sh --model /path/to/model.gguf\nDocumentation: docs/NPU_INSTALL.md\n' "$BIN_DIR"
