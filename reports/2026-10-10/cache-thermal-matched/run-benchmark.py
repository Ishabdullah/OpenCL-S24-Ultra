"""ABBA trial with per-context thermal admission and bounded child lifetime."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, root / 'scripts' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

thermal = module('thermal', 'moe-thermal-monitor.py')
policy = module('admission', 'moe-thermal-admission.py')
gate = policy.Admission()
sensors = thermal.zones()
if any(thermal.snapshot(sensors).get(key) is None for key in policy.KEYS):
    raise SystemExit('required battery/CPU/HTP sensors unavailable')
# Refuse to overwrite either an earlier result or stale approval files.
admission_dir = output / 'admission'
admission_dir.mkdir(exist_ok=False)
model = Path('/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf')
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GGML_HEXAGON_', 'GGML_CPU_MOE_CACHE_')) or key in ('LD_PRELOAD', 'GGML_BACKEND_PATH'):
        env.pop(key)
env.update(LD_LIBRARY_PATH=str(root / '.work-npu/build/bin'),
           ADSP_LIBRARY_PATH=str(root / '.work-npu/build/ggml/src/ggml-hexagon'),
           GGML_HEXAGON_PROFILE='0', MOE_BENCH_ADMISSION_DIR=str(admission_dir))
command = [str(root / '.work-npu/build/bin/moe-cache-benchmark'), str(model), '64']
start = time.monotonic()
minimum_available = None
peak_rss = 0
reason = None
waiting_pass = None
waiting_since = None
admitted = set()
admissions = []
next_sample = start
with (output / 'benchmark.jsonl').open('w') as out, (output / 'benchmark.log').open('w') as log, (output / 'thermal.jsonl').open('w') as trace:
    child = subprocess.Popen(command, stdout=out, stderr=log, env=env)
    try:
        while child.poll() is None:
            now = time.monotonic()
            memory = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines())
            available = memory['MemAvailable']
            minimum_available = min(minimum_available or available, available)
            try:
                status = Path(f'/proc/{child.pid}/status').read_text().splitlines()
                rss = next(int(line.split()[1]) for line in status if line.startswith('VmRSS:'))
                peak_rss = max(peak_rss, rss)
            except (FileNotFoundError, StopIteration):
                pass
            request = None
            try:
                request = json.loads((admission_dir / 'request.json').read_text())
            except (OSError, ValueError):
                pass
            if request and request['pass'] not in admitted and request['pass'] != waiting_pass:
                waiting_pass = request['pass']
                waiting_since = now
                gate.begin_pass()
                print(f'Waiting for thermal admission: pass {waiting_pass}; target {gate.target}', flush=True)
            if now >= next_sample:
                row = thermal.snapshot(sensors)
                row.update(utc=datetime.now(timezone.utc).isoformat(), monotonic_s=now)
                markers = re.findall(r'CACHE_BENCH (waiting |admitted |start |complete )?pass=(\d+)(?:/4)? slots=(\d+)', (output / 'benchmark.log').read_text())
                if markers:
                    kind, position, slots = markers[-1]
                    row.update(pass_number=int(position), slots=int(slots),
                               stage={'waiting ': 'admission', 'complete ': 'rest_or_setup'}.get(kind, 'evaluation'))
                else:
                    row['stage'] = 'model_load'
                trace.write(json.dumps(row) + '\n')
                trace.flush()
                if waiting_pass is not None and row.get("stage") == "admission" and row.get("pass_number") == waiting_pass:
                    decision = gate.observe(row)
                    if decision is not None:
                        decision.update(pass_number=waiting_pass, slots=request['slots'],
                                        wait_s=now-waiting_since, utc=row['utc'])
                        admissions.append(decision)
                        (output / 'admissions.json').write_text(json.dumps(admissions, indent=2) + '\n')
                        temporary = admission_dir / f'approve-{waiting_pass}.tmp'
                        temporary.write_text(str(waiting_pass) + '\n')
                        temporary.rename(admission_dir / f'approve-{waiting_pass}')
                        admitted.add(waiting_pass)
                        print(f'Admitted pass {waiting_pass}: {decision["admitted_c"]}; waited {decision["wait_s"]:.1f}s', flush=True)
                        waiting_pass = None
                next_sample = now + 2
            if available < 400 * 1024:
                reason = 'MemAvailable below 400 MiB'
            elif waiting_pass is not None and now - waiting_since > 900:
                reason = f'pass {waiting_pass} thermal admission exceeded 900 seconds'
            elif now - start > 4800:
                reason = '4800-second limit'
            if reason:
                break
            time.sleep(1)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
result = dict(command=command, returncode=child.returncode, stop_reason=reason,
              elapsed_s=time.monotonic()-start, minimum_available_kib=minimum_available,
              peak_rss_kib=peak_rss, admissions=admissions)
(output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if child.returncode == 0 and reason is None and len(admissions) == 4 else 1)
