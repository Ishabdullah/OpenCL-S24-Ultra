"""Confirm 64/128-slot throughput with cooling between every pass."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent
model = Path('/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf')
env = os.environ.copy()
for key in ('MOE_BENCH_ADMISSION_DIR', 'MOE_BENCH_PLAN'):
    env.pop(key, None)
for key in list(env):
    if key.startswith(('GGML_HEXAGON_', 'GGML_CPU_MOE_CACHE_')) or key in ('LD_PRELOAD', 'GGML_BACKEND_PATH'):
        env.pop(key)
env['LD_LIBRARY_PATH'] = str(root / '.work-npu/build/bin')
env['ADSP_LIBRARY_PATH'] = str(root / '.work-npu/build/ggml/src/ggml-hexagon')
env['GGML_HEXAGON_PROFILE'] = '0'
env['MOE_BENCH_PLAN'] = '64:4:2560,128:4:5120,128:6:5120,128:4:5120,64:4:2560'
command = [str(root / '.work-npu/build/bin/moe-cache-benchmark'), str(model), '64']
# Cool before the first run as well, after the preceding sweep.
spec = importlib.util.spec_from_file_location('thermal', root / 'scripts/moe-thermal-monitor.py')
thermal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thermal_module)
sensors = thermal_module.zones()
cooling_start = time.monotonic()
first = thermal_module.snapshot(sensors)
print('Initial cooling: at least 180 seconds before the first pass', flush=True)
while True:
    snapshot = thermal_module.snapshot(sensors)
    elapsed = time.monotonic() - cooling_start
    met = all(snapshot.get(key) is not None and snapshot[key] <= ceiling
              for key, ceiling in dict(battery_c=40, cpu_max_c=55, htp_max_c=50).items())
    if elapsed >= 180 and (met or elapsed >= 300):
        break
    time.sleep(2)
initial_cooling = dict(elapsed_s=elapsed, target_met=met,
                       first_c={key:first[key] for key in ('battery_c','cpu_max_c','htp_max_c')},
                       last_c={key:snapshot[key] for key in ('battery_c','cpu_max_c','htp_max_c')})
(output / 'initial-cooling.json').write_text(json.dumps(initial_cooling, indent=2) + '\n')
print(f'Initial cooling completed: {initial_cooling}', flush=True)
start = time.monotonic()
(output / 'cooldown.jsonl').write_text('')
minimum_available = None
peak_rss = 0
reason = None
with (output / 'benchmark.jsonl').open('w') as out, (output / 'benchmark.log').open('w') as log:
    child = subprocess.Popen(command, stdout=out, stderr=log, env=env)
    thermal_log = (output / 'thermal-monitor.log').open('w')
    thermal = subprocess.Popen(['python', str(root / 'scripts/moe-thermal-monitor.py'),
                                str(child.pid), str(output / 'thermal.jsonl'),
                                '--log', str(output / 'benchmark.log')],
                               stdout=subprocess.DEVNULL, stderr=thermal_log)
    cooldown_log = (output / 'cooldown-monitor.log').open('w')
    cooldown = subprocess.Popen(['python', str(root / 'scripts/moe-between-run-cooldown.py'),
                                 str(child.pid), str(output), '--after-pass', '1'],
                                stdout=cooldown_log, stderr=cooldown_log)
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
        if available < 768 * 1024:
            reason = 'MemAvailable below 768 MiB'
        elif time.monotonic() - start > 2100:
            reason = '2100-second limit'
        if reason:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            break
        time.sleep(1)
try:
    thermal.wait(timeout=7)
except subprocess.TimeoutExpired:
    thermal.terminate()
    thermal.wait(timeout=5)
thermal_log.close()
try:
    cooldown.wait(timeout=7)
except subprocess.TimeoutExpired:
    cooldown.terminate()
    cooldown.wait(timeout=5)
cooldown_log.close()
result = dict(plan=env['MOE_BENCH_PLAN'], cooldown_monitor_returncode=cooldown.returncode, thermal_monitor_returncode=thermal.returncode, command=command, returncode=child.returncode, stop_reason=reason,
              elapsed_s=time.monotonic() - start, minimum_available_kib=minimum_available,
              peak_rss_kib=peak_rss)
(output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if child.returncode == 0 and reason is None else 1)
