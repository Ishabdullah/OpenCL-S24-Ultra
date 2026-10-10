#!/usr/bin/env python3
"""Apply the opt-in CPU expert-cache patch to this project's pinned checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else root / '.work-npu/llama.cpp'
if len(sys.argv) > 2:
    raise SystemExit('usage: scripts/apply-moe-cache.py [LLAMA_CPP_CHECKOUT]')
manifest = json.loads((root / 'patches/experimental-cpu-moe-cache.json').read_text())
patch = root / 'patches/experimental-cpu-moe-cache.patch'
target = source / 'ggml/src/ggml-cpu/ggml-cpu.cpp'
header = source / 'ggml/src/ggml-cpu/moe-slot-cache.h'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
if sha(patch) != manifest['patch_sha256']:
    raise SystemExit('patch checksum differs from manifest')
head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if head != manifest['upstream_commit']:
    raise SystemExit('checkout is not at the tested pinned revision')
if sha(target) == manifest['after_sha256'] and sha(header) == manifest['header_sha256']:
    print('experimental CPU MoE cache patch already applied')
    raise SystemExit(0)
if sha(target) != manifest['before_sha256'] or header.exists():
    raise SystemExit('CPU source differs from expected base; refusing to overwrite local changes')
subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True)
subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
if sha(target) != manifest['after_sha256'] or sha(header) != manifest['header_sha256']:
    raise SystemExit('post-apply checksum mismatch')
print('experimental CPU MoE cache patch applied; build ggml-cpu, then opt in with GGML_CPU_MOE_CACHE_SLOTS=32')
