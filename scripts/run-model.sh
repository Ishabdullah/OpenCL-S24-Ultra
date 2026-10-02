#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ $# == 0 || ${1:-} == --help ]]; then
    echo 'Usage: scripts/run-model.sh /path/to/model.gguf [llama-completion arguments...]'
    echo 'Defaults: -ngl 99 -c 512 -b 128 -ub 128 -t 4 -fa off -n 128.'
    echo 'Later arguments override defaults. For raw generation add -no-cnv -p PROMPT.'
    [[ $# != 0 ]] || exit 2
    exit 0
fi
model=$1; shift
check_environment
require_build
prepare_work
python "$PROJECT_ROOT/scripts/validate-elf.py" "$PROJECT_ROOT"
check_model "$model"
model=$(realpath -- "$model")
info "Running project llama-completion with model $model"
info "Library path: $BIN_DIR; look for device, offloaded-layer and OpenCL-buffer messages."
cd "$PROJECT_ROOT"
exec "$BIN_DIR/llama-completion" -m "$model" -ngl 99 -c 512 -b 128 -ub 128 -t 4 \
    -fa off -n 128 --log-verbosity 4 "$@"
