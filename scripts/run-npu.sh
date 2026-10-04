#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/npu-common.sh"
if [[ $# == 0 || ${1:-} == --help ]]; then
    echo 'Usage: scripts/run-npu.sh /path/to/model.gguf [llama-completion arguments]'
    echo 'Defaults: HTP0, -ngl 99, ctx512, b/ub128, t4, FAoff. Extra arguments override defaults.'
    echo 'Example: scripts/run-npu.sh model.gguf -no-cnv -p "def add(a, b):" -n 32'
    exit 0
fi
check_environment
npu_require
model=$1; shift
check_model "$model"
model=$(realpath -- "$model")
info 'Using the project-native CPU/Hexagon build. Requested default offload: HTP0, 99 layers.'
info 'Check HTP0 model/KV buffers and positive offloaded layers in llama logs; not every operation is NPU-executed.'
exec "$BIN_DIR/llama-completion" -m "$model" -dev HTP0 -ngl 99 -c 512 -b 128 -ub 128 -t 4 -fa off -n 128 --log-verbosity 4 "$@"
