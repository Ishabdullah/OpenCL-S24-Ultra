#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
verify_only=0; rebuild=0; clean=0; jobs=4
while [[ $# -gt 0 ]]; do
    case $1 in
        --help) cat <<'HELP'
Usage: ./install.sh [--rebuild | --clean-build | --verify] [--jobs N]
Default: install missing Termux packages, fetch/apply pinned source, build once,
then validate ELF dependencies and Qualcomm GPU detection. No model download.
--rebuild      Reconfigure and incrementally rebuild the managed source.
--clean-build  Remove only the managed build directory and rebuild from scratch.
--verify       Check an existing build without installing packages/building.
--jobs N       Build concurrency (default: 4).
HELP
            exit 0 ;;
        --verify) verify_only=1; shift ;;
        --rebuild) rebuild=1; shift ;;
        --clean-build) clean=1; rebuild=1; shift ;;
        --jobs) [[ $# -ge 2 ]] || usage_error '--jobs requires N'; jobs=$2; shift 2 ;;
        *) usage_error "Unknown installer option: $1" ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || usage_error 'Jobs must be a positive integer.'
[[ $verify_only == 0 || $rebuild == 0 ]] || usage_error '--verify cannot be combined with build options.'
"$PROJECT_ROOT/scripts/check-device.sh"
if [[ $verify_only == 1 ]]; then exec "$PROJECT_ROOT/scripts/verify-opencl.sh"; fi
prepare_work
packages=(git clang cmake ninja python opencl-headers openssl)
missing=()
for package in "${packages[@]}"; do
    if [[ $(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true) != installed ]]; then missing+=("$package"); fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
    info "Installing missing Termux packages: ${missing[*]}"
    if ! pkg install -y "${missing[@]}"; then
        die 'Termux package installation failed. Check network/mirror, run pkg update, then rerun ./install.sh.'
    fi
else
    info "Required Termux packages already installed: ${packages[*]}"
fi
if [[ $rebuild == 1 || ! -f "$WORK_DIR/installed.json" ]]; then
    build_args=(--jobs "$jobs")
    [[ $clean == 0 ]] || build_args+=(--clean-build)
    "$PROJECT_ROOT/scripts/build.sh" "${build_args[@]}"
else
    info 'Existing installation found; validating pinned source and build.'
    python "$PROJECT_ROOT/scripts/validate-source.py" "$PROJECT_ROOT" "$SOURCE_DIR"
fi
"$PROJECT_ROOT/scripts/verify-opencl.sh"
python - "$PROJECT_ROOT" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); meta=json.loads((root/'upstream.json').read_text())
meta['installed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/'.work/installed.json').write_text(json.dumps(meta,indent=2)+'\n')
PY
info 'Installation and GPU detection succeeded. Actual inference requires your GGUF.'
printf '\nBinaries: %s\nRun: %s/scripts/run-model.sh /path/to/model.gguf\nVerify offload: %s/scripts/verify-opencl.sh --model /path/to/model.gguf\nBenchmark: %s/scripts/benchmark.sh /path/to/model.gguf\n' "$BIN_DIR" "$PROJECT_ROOT" "$PROJECT_ROOT" "$PROJECT_ROOT"
