#!/usr/bin/env python3
"""Six-pass controlled four-thread MoE cache experiment (explicit --run required)."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PLAN = [(64, 4, 2560), (80, 4, 3200), (96, 4, 3840),
        (96, 4, 3840), (80, 4, 3200), (64, 4, 2560)]
MODEL = Path('/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf')
BIN = ROOT / '.work-npu/build/bin/moe-cache-benchmark'
SOURCES = ['scripts/moe-cache-benchmark.cpp', 'scripts/moe-slot-cache.h',
           'scripts/moe-between-run-cooldown.py', 'scripts/moe-thermal-monitor.py',
           'scripts/analyze-moe-cache-sweep.py', 'patches/experimental-cpu-moe-cache.patch',
           '.work-npu/build/bin/moe-cache-benchmark', '.work-npu/build/bin/libggml-cpu.so',
           'reports/2026-10-13/cache-64-80-96-four-retry-01/run-benchmark.py']
EVIDENCE = ['provenance.json', 'initial-cooling.json', 'cooldown.jsonl',
            'cooldown-status.json', 'benchmark.jsonl', 'benchmark.log',
            'memory.jsonl', 'thermal.jsonl', 'thermal-monitor.log',
            'cooldown-monitor.log', 'run.json']

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def field(path, key):
    try:
        return next(int(line.split()[1]) for line in path.read_text().splitlines()
                    if line.startswith(key + ':'))
    except (OSError, StopIteration, ValueError):
        return None

def validate_replay(passes, complete):
    return (len(passes) == len(PLAN) and complete is not None
            and complete['passes'] == len(PLAN)
            and all((row['slots'], row['threads'], row['cache_mib']) == config
                    and row['bit_identical_positions'] == (0 if index == 0 else row['evaluated_positions'])
                    for index, (row, config) in enumerate(zip(passes, PLAN)))
            and complete['reference_positions'] == passes[0]['evaluated_positions']
            and complete['verified_positions'] == complete['reference_positions'] * (len(PLAN) - 1))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='permit physical six-pass experiment')
    args = parser.parse_args()
    plan = ','.join(f'{s}:{t}:{m}' for s, t, m in PLAN)
    command = [str(BIN), str(MODEL), '64']
    if not args.run:
        print(json.dumps(dict(dry_run=True, plan=plan, command=command,
                              cooling_s=[180, 300], watchdog_s=3600,
                              expected_evidence_files=EVIDENCE), indent=2))
        return
    if any((OUT / name).exists() for name in ('benchmark.jsonl', 'run.json', 'provenance.json')):
        raise SystemExit('Evidence exists: use a new directory, never overwrite a run')
    if not MODEL.is_file() or any(not (ROOT / name).is_file() for name in SOURCES):
        raise SystemExit('Missing model/source/binary; run not started')
    spec = importlib.util.spec_from_file_location('thermal', ROOT / 'scripts/moe-thermal-monitor.py')
    thermal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(thermal_module)
    sensors = thermal_module.zones()
    write('provenance.json', dict(utc=datetime.now(timezone.utc).isoformat(), plan=plan,
                                  model=str(MODEL), model_bytes=MODEL.stat().st_size,
                                  sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                          for name in SOURCES}))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('GGML_HEXAGON_', 'GGML_CPU_MOE_CACHE_')) or key in (
                'MOE_BENCH_ADMISSION_DIR', 'MOE_BENCH_PLAN', 'LD_PRELOAD', 'GGML_BACKEND_PATH'):
            env.pop(key)
    env.update(LD_LIBRARY_PATH=str(ROOT / '.work-npu/build/bin'),
               ADSP_LIBRARY_PATH=str(ROOT / '.work-npu/build/ggml/src/ggml-hexagon'),
               GGML_HEXAGON_PROFILE='0', MOE_BENCH_PLAN=plan)
    targets = dict(battery_c=40, cpu_max_c=55, htp_max_c=50)
    beginning = time.monotonic()
    first = thermal_module.snapshot(sensors)
    while True:
        last = thermal_module.snapshot(sensors)
        elapsed = time.monotonic() - beginning
        met = all(last.get(k) is not None and last[k] <= limit for k, limit in targets.items())
        if elapsed >= 180 and (met or elapsed >= 300):
            break
        time.sleep(2)
    write('initial-cooling.json', dict(elapsed_s=elapsed, target_met=met,
                                       first_c={k: first.get(k) for k in targets},
                                       last_c={k: last.get(k) for k in targets}))
    (OUT / 'cooldown.jsonl').write_text('')
    reason = None
    minimum_available = None
    peak_rss = 0
    with (OUT / 'benchmark.jsonl').open('w') as output, (OUT / 'benchmark.log').open('w') as log, (OUT / 'memory.jsonl').open('w') as memory, (OUT / 'thermal-monitor.log').open('w') as tlog, (OUT / 'cooldown-monitor.log').open('w') as clog:
        child = subprocess.Popen(command, stdout=output, stderr=log, env=env)
        thermal = subprocess.Popen(['python', str(ROOT / 'scripts/moe-thermal-monitor.py'), str(child.pid), str(OUT / 'thermal.jsonl'), '--log', str(OUT / 'benchmark.log')], stdout=subprocess.DEVNULL, stderr=tlog)
        cooldown = subprocess.Popen(['python', str(ROOT / 'scripts/moe-between-run-cooldown.py'), str(child.pid), str(OUT), '--after-pass', '1'], stdout=clog, stderr=clog)
        started = time.monotonic()
        try:
            while child.poll() is None:
                proc = Path(f'/proc/{child.pid}')
                row = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic(),
                           available_kib=field(Path('/proc/meminfo'), 'MemAvailable'),
                           rss_kib=field(proc / 'status', 'VmRSS'),
                           vm_swap_kib=field(proc / 'status', 'VmSwap'),
                           pss_kib=field(proc / 'smaps_rollup', 'Pss'),
                           swap_kib=field(proc / 'smaps_rollup', 'Swap'))
                memory.write(json.dumps(row) + '\n')
                memory.flush()
                if row['available_kib'] is not None:
                    minimum_available = min(minimum_available or row['available_kib'], row['available_kib'])
                peak_rss = max(peak_rss, row['rss_kib'] or 0)
                if cooldown.poll() is not None or thermal.poll() is not None:
                    reason = 'monitor exited early'
                elif minimum_available is not None and minimum_available < 768 * 1024:
                    reason = 'MemAvailable below 768 MiB'
                elif time.monotonic() - started > 3600:
                    reason = '3600-second watchdog'
                if reason:
                    os.kill(child.pid, signal.SIGCONT)
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                time.sleep(1)
        finally:
            for monitor in (thermal, cooldown):
                try:
                    monitor.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    monitor.terminate()
                    monitor.wait(timeout=5)
    records = [json.loads(line) for line in (OUT / 'benchmark.jsonl').read_text().splitlines()]
    passes = [r for r in records if r.get('type') == 'pass']
    cooling = [json.loads(line) for line in (OUT / 'cooldown.jsonl').read_text().splitlines()]
    complete = next((r for r in records if r.get('type') == 'complete'), None)
    verified = (validate_replay(passes, complete)
                and [r['after_pass'] for r in cooling] == list(range(1, 6))
                and all(180 <= r['paused_s'] <= 305 for r in cooling))
    result = dict(command=command, plan=plan, returncode=child.returncode,
                  thermal_returncode=thermal.returncode, cooldown_returncode=cooldown.returncode,
                  stop_reason=reason, minimum_available_kib=minimum_available, peak_rss_kib=peak_rss,
                  passes=len(passes), cooling_events=len(cooling), replay_verified=verified)
    write('run.json', result)
    print(json.dumps(result))
    if child.returncode or thermal.returncode or cooldown.returncode or reason or not verified:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
