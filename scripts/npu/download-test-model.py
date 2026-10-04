"""Explicit opt-in download of one pinned, openly licensed test GGUF."""
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
meta = json.loads((root / 'npu-upstream.json').read_text())['test_model']
spec = importlib.util.spec_from_file_location('prepare', root / 'scripts/npu/prepare-toolchain.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
# Prefer the already-used model location; never download a duplicate of it.
existing = Path.home() / 'models/qwen2.5-coder-1.5b' / meta['filename']
dest = existing if existing.exists() else Path.home() / 'models/OpenCL-S24-Ultra' / meta['filename']
record = {**meta, 'url': 'https://huggingface.co/' + meta['repository'] + '/resolve/' + meta['revision'] + '/' + meta['filename']}
prepare.fetch(record, dest)
(root / '.work-npu/test-model.json').write_text(json.dumps({'path': str(dest), **meta}, indent=2) + '\n')
print('Test GGUF available: ' + str(dest))
