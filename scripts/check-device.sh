#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ ${1:-} != --help ]] || { echo 'Usage: scripts/check-device.sh (no model or build required)'; exit 0; }
[[ $# == 0 ]] || usage_error 'Usage: scripts/check-device.sh'
check_environment
info "Architecture: $(uname -m)"
info "Device: $(/system/bin/getprop ro.product.manufacturer) $(/system/bin/getprop ro.product.model)"
info "SoC: $(/system/bin/getprop ro.soc.manufacturer) $(/system/bin/getprop ro.soc.model)"
info "Android: $(/system/bin/getprop ro.build.version.release); Termux: ${TERMUX_VERSION:-unknown}"
missing=0
for lib in libOpenCL.so libOpenCL_adreno.so libCB.so libgsl.so; do
    if [[ -r /vendor/lib64/$lib ]]; then info "Present: /vendor/lib64/$lib"; else
        printf 'Missing/unreadable: /vendor/lib64/%s\n' "$lib" >&2
        missing=1
    fi
done
[[ $missing == 0 ]] || die 'Required Qualcomm vendor stack is unavailable. Do not install substitute vendor libraries.'
info 'Environment prerequisites detected. GPU initialization requires verify-opencl.sh.'
info 'Confirmed hardware: S24 Ultra / Adreno 750 / Android 16. Other environments remain unverified.'
