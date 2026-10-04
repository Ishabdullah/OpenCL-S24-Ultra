#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/npu-common.sh"
[[ ${1:-} != --help ]] || { echo 'Usage: scripts/check-npu.sh (no model/build required)'; exit 0; }
[[ $# == 0 ]] || usage_error 'Usage: scripts/check-npu.sh'
check_environment
info "Device: $(/system/bin/getprop ro.product.manufacturer) $(/system/bin/getprop ro.product.model)"
info "SoC: $(/system/bin/getprop ro.soc.model); architecture: $(uname -m); Android: $(/system/bin/getprop ro.build.version.release)"
[[ -r /vendor/lib64/libcdsprpc.so ]] || die 'Qualcomm /vendor/lib64/libcdsprpc.so is missing/unreadable. Device-provided CDSP/FastRPC is required.'
info 'Vendor CDSP runtime present. File presence is not proof of usable DSP access.'
info 'Confirmed: S24 Ultra / Snapdragon 8 Gen 3 / Hexagon v75 / Android 16. Other devices are unverified.'
