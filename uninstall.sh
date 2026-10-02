#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
case ${1:---dry-run} in
    --help) echo 'Usage: ./uninstall.sh [--dry-run | --yes]'; echo 'Removes only generated .work data. Keeps project scripts and shared Termux packages.'; exit 0 ;;
    --yes|--dry-run) mode=${1:---dry-run} ;;
    *) usage_error 'Usage: ./uninstall.sh [--dry-run | --yes]' ;;
esac
[[ $# -le 1 ]] || usage_error 'Too many arguments.'
[[ -e "$WORK_DIR" ]] || { info 'Nothing installed in this repository.'; exit 0; }
[[ ! -L "$WORK_DIR" && -f "$WORK_DIR/.owner" && $(cat "$WORK_DIR/.owner") == "$OWNER_TEXT" ]] || die 'Unknown or symlinked .work; refusing removal.'
python - "$WORK_DIR" "$PROJECT_ROOT/upstream.json" <<'PYTHON'
import json,pathlib,subprocess,sys
work=pathlib.Path(sys.argv[1]); source=work/'llama.cpp'
pin=json.loads(pathlib.Path(sys.argv[2]).read_text())['commit']
models=[]
for p in work.rglob('*'):
    if p.suffix.lower()!='.gguf': continue
    # Upstream includes small vocabulary fixtures. Only exact pinned Git blobs
    # are generated source data; new/modified GGUFs remain protected.
    try:
        relative=p.relative_to(source).as_posix()
        if p.is_symlink(): raise ValueError('symlink')
        expected=subprocess.check_output(['git','-C',str(source),'rev-parse',pin+':'+relative],stderr=subprocess.DEVNULL,text=True).strip()
        actual=subprocess.check_output(['git','-C',str(source),'hash-object','--no-filters','--',str(p)],stderr=subprocess.DEVNULL,text=True).strip()
        if actual==expected: continue
    except (ValueError,OSError,subprocess.CalledProcessError):
        pass
    models.append(str(p))
if models: sys.exit('Move user GGUF models out of .work before uninstalling: '+', '.join(models))
PYTHON
info "Generated directory to remove: $WORK_DIR"
[[ $mode == --yes ]] || { info 'Dry run. Run ./uninstall.sh --yes to remove it.'; exit 0; }
rm -rf -- "$WORK_DIR"
info 'Generated source, binaries, kernel cache and logs removed. Models outside .work and Termux packages retained.'
