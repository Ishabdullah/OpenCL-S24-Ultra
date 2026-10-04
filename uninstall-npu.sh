#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/npu-common.sh"
case ${1:---dry-run} in
    --help) echo 'Usage: ./uninstall-npu.sh [--dry-run | --yes]'; echo 'Remove only generated .work-npu; preserve models outside it and shared packages.'; exit 0 ;;
    --yes|--dry-run) mode=${1:---dry-run} ;;
    *) usage_error 'Usage: ./uninstall-npu.sh [--dry-run | --yes]' ;;
esac
[[ $# -le 1 ]] || usage_error 'Too many arguments.'
[[ -e "$WORK_DIR" ]] || { info 'No NPU installation in this repository.'; exit 0; }
[[ ! -L "$WORK_DIR" && -f "$WORK_DIR/.owner" && $(cat "$WORK_DIR/.owner") == "$OWNER_TEXT" ]] || die 'Unknown/symlinked NPU storage; refusing removal.'
python - "$WORK_DIR" "$PROJECT_ROOT/npu-upstream.json" <<'PY'
import json,pathlib,subprocess,sys
work=pathlib.Path(sys.argv[1]);source=work/'llama.cpp'
pin=json.loads(pathlib.Path(sys.argv[2]).read_text())['commit']
protected=[]
for p in work.rglob('*'):
    if p.suffix.lower()!='.gguf':continue
    try:
        relative=p.relative_to(source).as_posix()
        if p.is_symlink():raise ValueError('symlink')
        expected=subprocess.check_output(['git','-C',str(source),'rev-parse',pin+':'+relative],stderr=subprocess.DEVNULL,text=True).strip()
        actual=subprocess.check_output(['git','-C',str(source),'hash-object','--no-filters','--',str(p)],stderr=subprocess.DEVNULL,text=True).strip()
        if expected==actual:continue
    except (OSError,ValueError,subprocess.CalledProcessError):pass
    protected.append(str(p))
if protected:sys.exit('Move user-added/modified GGUFs out of .work-npu first: '+', '.join(protected))
PY
info "Generated NPU directory: $WORK_DIR"
[[ $mode == --yes ]] || { info 'Dry run; run ./uninstall-npu.sh --yes to remove only this directory.'; exit 0; }
rm -rf -- "$WORK_DIR"
info 'NPU generated source/SDK/compiler runtime/build/logs removed. Other builds, shared packages and external models retained.'
