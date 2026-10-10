"""Bounded live experiment; terminate only our child on timeout or RAM pressure."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent
model = Path('/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf')
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GGML_HEXAGON_', 'GGML_CPU_MOE_CACHE_')) or key in ('LD_PRELOAD', 'GGML_BACKEND_PATH'):
        env.pop(key)
env['LD_LIBRARY_PATH'] = str(root / '.work-npu/build/bin')
env['ADSP_LIBRARY_PATH'] = str(root / '.work-npu/build/ggml/src/ggml-hexagon')
env['GGML_HEXAGON_PROFILE'] = '0'
command = [str(root / '.work-npu/build/bin/moe-cache-validate'), str(model), '16']
start = time.monotonic()
minimum_available = None
peak_rss = 0
reason = None
with (output / 'validation.json').open('w') as out, (output / 'validation.log').open('w') as log:
    child = subprocess.Popen(command, stdout=out, stderr=log, env=env)
    while child.poll() is None:
        memory = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines())
        available = memory['MemAvailable']
        minimum_available = min(minimum_available or available, available)
        try:
            status = Path(f'/proc/{child.pid}/status').read_text().splitlines()
            rss = next(int(line.split()[1]) for line in status if line.startswith('VmRSS:'))
            peak_rss = max(peak_rss, rss)
        except (FileNotFoundError, StopIteration):
            pass
        if available < 400 * 1024:
            reason = 'MemAvailable below 400 MiB'
        elif time.monotonic() - start > 600:
            reason = '600-second limit'
        if reason:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            break
        time.sleep(1)
result = dict(command=command, returncode=child.returncode, stop_reason=reason,
              elapsed_s=time.monotonic() - start, minimum_available_kib=minimum_available,
              peak_rss_kib=peak_rss)
(output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if child.returncode == 0 and reason is None else 1)
