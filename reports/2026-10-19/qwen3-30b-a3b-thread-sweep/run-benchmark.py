#!/usr/bin/env python3
"""Cooldown-supervised CPU/GPU/HTP benchmark (explicit --run required)."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PLAN = [(80, 4, 3200), (80, 3, 3200), (80, 2, 3200),
        (80, 2, 3200), (80, 3, 3200), (80, 4, 3200)]
MODEL = Path('/data/data/com.termux/files/home/models/Qwen3-30B-A3B-Q4_K_M.gguf')
BIN_UNIFIED_MATRIX = ROOT / '.work-unified-build/bin/qwen3-30b-a3b-matrix'
SOURCES = ['reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.cpp',
           'reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py',
           'reports/2026-10-19/qwen3-30b-a3b-thread-sweep/cooldown.py',
           'scripts/moe-slot-cache.h', 'scripts/moe-thermal-monitor.py',
           'patches/experimental-cpu-moe-cache.patch',
           'patches/experimental-hexagon-moe-cache-trace.patch',
           'patches/experimental-explicit-layer-map.patch',
           'patches/experimental-htp-bounded-expert-cache.patch',
           'patches/llama.cpp-qualcomm-sphal.patch',
           '.work-unified-build/CMakeCache.txt',
           '.work-unified-source/src/llama-model.cpp']
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

def battery_status():
    try:
        result = subprocess.run(['termux-battery-status'], text=True, capture_output=True,
                                timeout=5, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return None

def validate_replay(passes, complete, expected_plan):
    return (len(passes) == len(expected_plan) and complete is not None
            and complete['passes'] == len(expected_plan)
            and all((row['slots'], row['threads'], row['cache_mib']) == config
                    and row['bit_identical_positions'] == (0 if index == 0 else row['evaluated_positions'])
                    for index, (row, config) in enumerate(zip(passes, expected_plan)))
            and complete['reference_positions'] == passes[0]['evaluated_positions']
            and complete['verified_positions'] == complete['reference_positions'] * (len(expected_plan) - 1))

def evidence(out):
    try:
        records = [json.loads(line) for line in (out / 'benchmark.jsonl').read_text().splitlines()]
        cooling = [json.loads(line) for line in (out / 'cooldown.jsonl').read_text().splitlines()]
        config = next((row for row in records if row.get('type') == 'config'), None)
        if config is None:
            return 0, len(cooling), False, False
        expected_plan = [(row['slots'], row['threads'], row['cache_mib'])
                         for row in config['plan']]
        passes = [row for row in records if row.get('type') == 'pass']
        footers = [row for row in records if row.get('type') == 'complete']
        cooled = (len(cooling) == len(expected_plan) - 1 and
                  [row['after_pass'] for row in cooling] == list(range(1, len(expected_plan))) and
                  all(180 <= row['paused_s'] <= 305 for row in cooling))
        return len(passes), len(cooling), cooled, (cooled and len(footers) == 1 and
                                                  validate_replay(passes, footers[0], expected_plan))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return 0, 0, False, False

def monitor_reason(child_rc, thermal_rc, cooldown_rc, out, now, deadline):
    if child_rc is not None:
        return None, deadline
    if thermal_rc is not None:
        return 'thermal monitor exited early', deadline
    if cooldown_rc is not None:
        passes, _, cooled, _ = evidence(out)
        if cooldown_rc != 0 or passes != len(PLAN) or not cooled:
            return 'cooldown monitor exited early or evidence incomplete', deadline
        deadline = now + 120 if deadline is None else deadline
        if now >= deadline:
            return 'benchmark footer/exit teardown timeout', deadline
    return None, deadline

def count_layer_assignments(log_text):
    counts = {'GPUOpenCL': 0, 'HTP0': 0, 'CPU': 0}
    for device in layer_assignments(log_text).values():
        if device in counts:
            counts[device] += 1
    return counts

def layer_assignments(log_text):
    return {int(match.group(1)): match.group(2)
            for match in re.finditer(r'layer\s+(\d+) assigned to device ([A-Za-z0-9_]+)', log_text)}

def htp_cache_staging_evidence(log_text):
    staged = re.findall(
        r'bounded HTP cache staged tensor=([^ ]+) routed_ids=(\d+) loaded_slots=(\d+) bytes=(\d+)',
        log_text)
    allocated = re.findall(r'HTP expert cache allocated (blk\.[^ ]+) \d+ slots, \d+ bytes', log_text)
    graph_nodes = re.findall(r'bounded HTP graph node opcode=\d+ weight=([^ ]+) type=', log_text)
    return dict(operations=len(staged), routed_ids=sum(int(row[1]) for row in staged),
                loaded_slots=sum(int(row[2]) for row in staged),
                staged_bytes=sum(int(row[3]) for row in staged),
                tensors=sorted({row[0] for row in staged}),
                allocated_tensors=sorted(set(allocated)),
                graph_tensors=sorted(set(graph_nodes)))

def htp_resident_evidence(log_text):
    weights = re.findall(r'resident HTP expert weight (blk\.[^ ]+) bytes=(\d+) cumulative=(\d+) cap_mib=(\d+)', log_text)
    operations = re.findall(r'resident HTP MoE op tensor=(blk\.[^ ]+) buffer=([^ ]+) bytes=(\d+)', log_text)
    return dict(weights=sorted({row[0] for row in weights}),
                weight_bytes=sum(int(row[1]) for row in weights),
                operations=len(operations),
                operation_tensors=sorted({row[0] for row in operations}),
                operation_buffers=sorted({row[1] for row in operations}))

def summarize_layer_profile(profile_path, assignments, total_layers=49):
    samples = {layer: [] for layer in range(total_layers)}
    for line in profile_path.read_text().splitlines():
        row = json.loads(line)
        if row['step'] > 0 and row['layer'] in samples:
            samples[row['layer']].append(row['elapsed_ns'])
    per_layer = {}
    for layer, values in samples.items():
        if not values:
            continue
        if layer not in assignments:
            raise ValueError(f'missing placement for profiled layer {layer}')
        per_layer[str(layer)] = dict(device=assignments[layer], samples=len(values),
                                     median_ns=statistics.median(values))
    by_device = {}
    for layer, result in per_layer.items():
        by_device.setdefault(result['device'], 0.0)
        by_device[result['device']] += result['median_ns']
    total = sum(by_device.values())
    missing = sorted(set(range(total_layers)) - {int(layer) for layer in per_layer})
    return dict(per_layer=per_layer, complete=not missing, missing_layers=missing,
                device_median_sum_ns=by_device,
                device_fraction={device: value / total for device, value in by_device.items()},
                timing_note='diagnostic layer boundaries synchronize backend execution; do not use these rates as normal decode throughput')

def main():
    global OUT, PLAN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='permit physical device benchmark')
    parser.add_argument('--out-dir', type=Path, help='fresh evidence directory for this attempt')
    parser.add_argument('--pilot', action='store_true', help='two-pass deterministic pilot')
    parser.add_argument('--pilot-threads', type=int, choices=tuple(range(1, 9)), default=1,
                        help='thread count for a two-pass pilot or bounded-cache run (1–8)')
    parser.add_argument('--thread-sweep', action='store_true',
                        help='run CPU-only at 1,2,3,4,5,6,7,8 threads with cooldown between passes')
    parser.add_argument('--pilot-efficient-cores', action='store_true',
                        help='pin the pilot process to CPUs 0-1, including during graph setup and decode')
    parser.add_argument('--pilot-slots', type=int, choices=(48, 64, 80), default=80,
                        help='expert cache capacity for the two-pass pilot; each slot budgets 40 MiB')
    parser.add_argument('--pilot-tokens', type=int, choices=(2, 8, 16, 32, 64), default=2,
                        help='maximum generated tokens per pass (2 is a load/replay smoke, use 16+ for speed comparisons)')
    parser.add_argument('--prompt', default='Hi',
                        help='UTF-8 prompt for controlled inference and logit comparison (max 128 model tokens)')
    parser.add_argument('--ngl', type=int, choices=tuple(range(1, 50)), default=2,
                        help='number of model layers to offload to the selected accelerator (1–49)')
    parser.add_argument('--gpu-layers', type=int,
                        help='exact GPU layer share for a GPU+NPU run; pair with --npu-layers')
    parser.add_argument('--npu-layers', type=int,
                        help='exact HTP layer share for a GPU+NPU run; pair with --gpu-layers')
    parser.add_argument('--cpu-layers', type=int,
                        help='expected CPU layer share for a 49-layer GPU+NPU run')
    parser.add_argument('--triad-output', choices=('cpu', 'gpu', 'npu'),
                        help='experimental 16/16/16 repeating-layer map; assign the output layer to this backend')
    parser.add_argument('--bounded-htp-layers', type=int, choices=(1, 2, 4, 8, 16),
                        help='experimental streamed Q4_K HTP expert cache over this many blocks (CPU/GPU/HTP map)')
    parser.add_argument('--htp-resident-layers', type=int, choices=(1,),
                        help='resident HTP control for one routed-expert block at model layer 32')
    parser.add_argument('--htp-resident-budget-mib', type=int, choices=(1024,), default=1024,
                        help='hard pre-allocation cap for the one-layer resident HTP control')
    parser.add_argument('--htp-cache-slots', type=int, choices=(8, 16), default=8,
                        help='selected-expert slots per HTP weight tensor in bounded-cache mode')
    parser.add_argument('--htp-cache-budget-mib', type=int, choices=(256, 512, 1024), default=512,
                        help='maximum HTP allocation for all bounded expert tensors')
    parser.add_argument('--layer-profile', action='store_true',
                        help='record synchronized per-layer decode timings (diagnostic, not throughput-comparable)')
    parser.add_argument('--memory-floor-gib', type=float, default=2.0,
                        help='stop the child if MemAvailable falls below this floor (default: 2 GiB)')
    parser.add_argument('--backend', choices=('cpu', 'gpu', 'npu', 'gpu+npu'), default='npu',
                        help='one unified binary; mixed mode schedules layers over both accelerators, with CPU fallback')
    args = parser.parse_args()
    if args.thread_sweep and args.backend != 'cpu':
        parser.error('--thread-sweep requires --backend cpu')
    if args.thread_sweep and args.pilot:
        parser.error('--thread-sweep and --pilot are separate run modes')
    if args.bounded_htp_layers is not None and (args.pilot or args.thread_sweep):
        parser.error('--bounded-htp-layers uses its own two-pass replay plan')
    device = {'cpu': 'CPU', 'gpu': 'GPUOpenCL', 'npu': 'HTP0',
              'gpu+npu': 'GPUOpenCL,HTP0'}[args.backend]
    ngl = 0 if args.backend == 'cpu' else args.ngl
    if (args.gpu_layers is None) != (args.npu_layers is None):
        parser.error('--gpu-layers and --npu-layers must be used together')
    tensor_split = None
    layer_split = None
    layer_map = None
    if args.bounded_htp_layers is not None and args.triad_output and args.bounded_htp_layers != 16:
        parser.error('--triad-output with bounded HTP caching requires --bounded-htp-layers 16')
    if args.htp_resident_layers is not None and args.triad_output:
        parser.error('--triad-output can be combined with bounded HTP caching, not resident HTP mode')
    if args.bounded_htp_layers is not None and args.htp_resident_layers is not None:
        parser.error('streamed and resident HTP modes are separate controls')
    if args.bounded_htp_layers is not None or args.htp_resident_layers is not None:
        if args.backend != 'gpu+npu':
            parser.error('HTP expert controls require --backend gpu+npu')
        n_htp = args.bounded_htp_layers or args.htp_resident_layers
        # Keep the earlier balanced CPU/GPU 16-block regions. Stream only a
        # small initial portion of the HTP region; all remaining blocks stay CPU.
        layer_map = (['CPU'] * 16 + ['GPUOpenCL'] * 16 + ['HTP0'] * n_htp +
                     ['CPU'] * (16 - n_htp) + ['CPU'])
        ngl = 49
        layer_split = dict(cpu=33 - n_htp, gpu=16, npu=n_htp)
    if args.triad_output:
        if args.backend != 'gpu+npu':
            parser.error('--triad-output requires --backend gpu+npu')
        if args.gpu_layers is not None or args.cpu_layers is not None:
            parser.error('--triad-output cannot be combined with accelerator layer-share arguments')
        # Qwen3-30B-A3B: 48 repeating blocks in three contiguous 16-layer
        # segments; layer 48 is the output layer and is varied independently.
        layer_map = ['CPU'] * 16 + ['GPUOpenCL'] * 16 + ['HTP0'] * 16
        layer_map.append({'cpu': 'CPU', 'gpu': 'GPUOpenCL', 'npu': 'HTP0'}[args.triad_output])
        ngl = 49  # enable both accelerator backends; explicit map routes every layer
        layer_split = dict(cpu=16 + (args.triad_output == 'cpu'),
                           gpu=16 + (args.triad_output == 'gpu'),
                           npu=16 + (args.triad_output == 'npu'))
    elif args.gpu_layers is not None:
        if args.backend != 'gpu+npu':
            parser.error('explicit accelerator layer shares require --backend gpu+npu')
        if args.gpu_layers < 0 or args.npu_layers < 0 or args.gpu_layers + args.npu_layers != ngl:
            parser.error('GPU and NPU layer shares must be nonnegative and sum to --ngl')
        if args.gpu_layers + args.npu_layers == 0:
            parser.error('at least one accelerator layer is required')
        cpu_layers = 49 - ngl
        if args.cpu_layers is not None and args.cpu_layers != cpu_layers:
            parser.error('CPU, GPU, and NPU layer counts must sum to 49')
        layer_split = dict(cpu=cpu_layers, gpu=args.gpu_layers, npu=args.npu_layers)
        tensor_split = f'{args.gpu_layers},{args.npu_layers}'
    elif args.cpu_layers is not None:
        parser.error('--cpu-layers requires explicit --gpu-layers and --npu-layers')
    benchmark_bin = BIN_UNIFIED_MATRIX
    if args.pilot:
        cache_mib = args.pilot_slots * 40
        PLAN = [(args.pilot_slots, args.pilot_threads, cache_mib),
                (args.pilot_slots, args.pilot_threads, cache_mib)]
    elif args.thread_sweep:
        cache_mib = args.pilot_slots * 40
        PLAN = [(args.pilot_slots, threads, cache_mib) for threads in range(1, 9)]
    elif args.bounded_htp_layers is not None or args.htp_resident_layers is not None:
        cache_mib = args.htp_cache_slots * 40
        PLAN = [(args.htp_cache_slots, args.pilot_threads, cache_mib),
                (args.htp_cache_slots, args.pilot_threads, cache_mib)]
    if args.out_dir:
        OUT = args.out_dir.resolve()
        OUT.mkdir(parents=True, exist_ok=True)
    plan = ','.join(f'{s}:{t}:{m}' for s, t, m in PLAN)
    token_limit = args.pilot_tokens if args.pilot or args.thread_sweep or args.bounded_htp_layers is not None or args.htp_resident_layers is not None else 64
    command = [str(benchmark_bin), str(MODEL), str(token_limit)]
    if not args.run:
        print(json.dumps(dict(dry_run=True, plan=plan, command=command,
                              model=str(MODEL), backend=args.backend, device=device,
                              n_gpu_layers=ngl,
                              accelerator_layer_split=(dict(gpu=args.gpu_layers, npu=args.npu_layers)
                                                       if tensor_split else None),
                              model_layer_split=layer_split, layer_profile=args.layer_profile,
                              output_layer=args.triad_output,
                              bounded_htp_layers=args.bounded_htp_layers,
                              htp_resident_layers=args.htp_resident_layers,
                              htp_resident_budget_mib=args.htp_resident_budget_mib,
                              htp_cache_slots=args.htp_cache_slots,
                              htp_cache_budget_mib=args.htp_cache_budget_mib,
                              thermal_policy='cooldown-only', mmap=True,
                              pilot=args.pilot, pilot_threads=args.pilot_threads,
                              thread_sweep=args.thread_sweep,
                              pilot_slots=args.pilot_slots,
                              pilot_tokens=args.pilot_tokens,
                              prompt=args.prompt,
                              pilot_affinity=('0,1' if args.pilot and args.pilot_efficient_cores
                                              else 'all allowed CPUs after load'),
                              token_limit=token_limit,
                              mmap_prefetch=False, weight_repack=False,
                              context_min=256, batch=1,
                              cache_mib=PLAN[0][2],
                              cooling_s=[180, 300], watchdog_s=3600,
                              memory_floor_gib=args.memory_floor_gib,
                              admission={'plugged': 'AC', 'battery_min_pct': 40,
                                         'memavailable_min_gib': 5,
                                         'initial_cool_hold_s': 180,
                                         'initial_cool_targets_c': {'cpu': 55, 'battery': 40, 'htp': 50}},
                              expected_evidence_files=EVIDENCE), indent=2))
        return
    if any((OUT / name).exists() for name in ('benchmark.jsonl', 'run.json', 'provenance.json')):
        raise SystemExit('Evidence exists: use a new directory, never overwrite a run')
    sources = SOURCES + [str(benchmark_bin.relative_to(ROOT))]
    if not MODEL.is_file() or not benchmark_bin.is_file() or any(not (ROOT / name).is_file() for name in sources):
        raise SystemExit('Missing model/source/binary; run not started')
    spec = importlib.util.spec_from_file_location('thermal', ROOT / 'scripts/moe-thermal-monitor.py')
    thermal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(thermal_module)
    sensors = thermal_module.zones()
    history = OUT / 'preflight.jsonl'
    if not history.exists() and (OUT / 'preflight.json').exists():
        history.write_text(json.dumps(json.loads((OUT / 'preflight.json').read_text())) + '\n')
    wait_started = time.monotonic()
    last_reported = -60
    while True:
        battery = battery_status()
        thermal_preflight = thermal_module.snapshot(sensors)
        available = field(Path('/proc/meminfo'), 'MemAvailable')
        preflight = dict(utc=datetime.now(timezone.utc).isoformat(),
                         waited_s=round(time.monotonic() - wait_started, 1),
                         battery=battery, memavailable_kib=available, thermal=thermal_preflight,
                         gates={'plugged_ac': bool(battery and battery.get('plugged') == 'PLUGGED_AC'),
                                'battery_at_least_40_pct': bool(battery and battery.get('percentage', 0) >= 40),
                                'memavailable_at_least_5_gib': bool(available and available >= 5 * 1024 * 1024)})
        with history.open('a') as records:
            records.write(json.dumps(preflight) + '\n')
        write('preflight.json', preflight)
        if all(preflight['gates'].values()):
            break
        waited = time.monotonic() - wait_started
        if waited >= 900:
            raise SystemExit('Preflight did not pass within 15 minutes; no benchmark process was started.')
        if waited - last_reported >= 60:
            print(json.dumps({'waiting_for_preflight': preflight['gates'],
                              'cpu_c': thermal_preflight.get('cpu_max_c'),
                              'htp_c': thermal_preflight.get('htp_max_c'),
                              'memavailable_kib': available}), flush=True)
            last_reported = waited
        time.sleep(20)
    write('provenance.json', dict(utc=datetime.now(timezone.utc).isoformat(), plan=plan,
                                  n_gpu_layers=ngl, backend=args.backend, device=device,
                                  accelerator_layer_split=(dict(gpu=args.gpu_layers, npu=args.npu_layers)
                                                           if tensor_split else None),
                                  model_layer_split=layer_split, layer_profile=args.layer_profile,
                                  output_layer=args.triad_output,
                                  bounded_htp_layers=args.bounded_htp_layers,
                                  htp_resident_layers=args.htp_resident_layers,
                                  htp_resident_budget_mib=args.htp_resident_budget_mib,
                                  htp_cache_slots=args.htp_cache_slots,
                                  htp_cache_budget_mib=args.htp_cache_budget_mib,
                                  thermal_policy='cooldown-only', pilot=args.pilot,
                                  pilot_threads=args.pilot_threads,
                                  thread_sweep=args.thread_sweep,
                                  prompt=args.prompt,
                                  model=str(MODEL), model_bytes=MODEL.stat().st_size,
                                  sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                          for name in sources}))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('GGML_HEXAGON_', 'GGML_CPU_MOE_CACHE_')) or key in (
                'MOE_BENCH_ADMISSION_DIR', 'MOE_BENCH_PLAN', 'MOE_BENCH_TENSOR_SPLIT',
                'MOE_BENCH_LAYER_PROFILE_PATH', 'MOE_BENCH_LAYER_MAP', 'MOE_BENCH_PROMPT',
                'LD_PRELOAD', 'GGML_BACKEND_PATH'):
            env.pop(key)
    libdir = ROOT / '.work-unified-build/bin'
    env.update(LD_LIBRARY_PATH=str(libdir),
               ADSP_LIBRARY_PATH=str(ROOT / '.work-unified-build/ggml/src/ggml-hexagon'),
               GGML_HEXAGON_PROFILE='0', LLAMA_MMAP_PREFETCH='0', MOE_BENCH_PLAN=plan,
               MOE_BENCH_DEVICE=device, MOE_BENCH_N_GPU_LAYERS=str(ngl))
    env['MOE_BENCH_PROMPT'] = args.prompt
    if tensor_split:
        env['MOE_BENCH_TENSOR_SPLIT'] = tensor_split
    if layer_map:
        env['MOE_BENCH_LAYER_MAP'] = ','.join(layer_map)
    if args.bounded_htp_layers is not None:
        env['GGML_HEXAGON_MOE_EXPERT_CACHE_SLOTS'] = str(args.htp_cache_slots)
        env['GGML_HEXAGON_MOE_EXPERT_CACHE_MAX_MIB'] = str(args.htp_cache_budget_mib)
        env['GGML_HEXAGON_MOE_EXPERT_CACHE_LAYER_FIRST'] = '32'
        env['GGML_HEXAGON_MOE_EXPERT_CACHE_LAYER_COUNT'] = str(args.bounded_htp_layers)
        # The first bounded-cache prototype stages one routed weight tensor per
        # HTP op; keep Hexagon from combining gate/up into MUL_MAT_ID_NX.
        env['GGML_HEXAGON_OPFUSION'] = '0'
    if args.htp_resident_layers is not None:
        env['GGML_HEXAGON_MOE_EXPERT_RESIDENT_LAYER_FIRST'] = '32'
        env['GGML_HEXAGON_MOE_EXPERT_RESIDENT_LAYER_COUNT'] = str(args.htp_resident_layers)
        env['GGML_HEXAGON_MOE_EXPERT_RESIDENT_MAX_MIB'] = str(args.htp_resident_budget_mib)
        env['GGML_HEXAGON_OPFUSION'] = '0'
    if args.layer_profile:
        env['MOE_BENCH_LAYER_PROFILE_PATH'] = str(OUT / 'layer-timing.jsonl')
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
    if not met:
        raise SystemExit('Initial cooling refused after 300 seconds; no benchmark process was started.')
    (OUT / 'cooldown.jsonl').write_text('')
    reason = None
    minimum_available = None
    peak_rss = 0
    peak_vm_swap = 0
    with (OUT / 'benchmark.jsonl').open('w') as output, (OUT / 'benchmark.log').open('w') as log, (OUT / 'memory.jsonl').open('w') as memory, (OUT / 'thermal-monitor.log').open('w') as tlog, (OUT / 'cooldown-monitor.log').open('w') as clog:
        def prepare_child():
            os.nice(10)
            if args.pilot and args.pilot_efficient_cores:
                os.sched_setaffinity(0, {0, 1})
        child = subprocess.Popen(command, stdout=output, stderr=log, env=env,
                                 preexec_fn=prepare_child)
        thermal = subprocess.Popen(['python', str(ROOT / 'scripts/moe-thermal-monitor.py'), str(child.pid), str(OUT / 'thermal.jsonl'), '--log', str(OUT / 'benchmark.log')], stdout=subprocess.DEVNULL, stderr=tlog)
        cooldown_script = Path(__file__).resolve().with_name('cooldown.py')
        cooldown = subprocess.Popen(['python', str(cooldown_script), str(child.pid), str(OUT), '--after-pass', '1', '--exe', benchmark_bin], stdout=clog, stderr=clog)
        started = time.monotonic()
        teardown_deadline = None
        last_battery_check = 0.0
        latest_battery = battery
        try:
            while True:
                if child.poll() is not None:
                    break
                proc = Path(f'/proc/{child.pid}')
                row = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic(),
                           available_kib=field(Path('/proc/meminfo'), 'MemAvailable'),
                           rss_kib=field(proc / 'status', 'VmRSS'),
                           vm_swap_kib=field(proc / 'status', 'VmSwap'),
                           pss_kib=field(proc / 'smaps_rollup', 'Pss'),
                           swap_kib=field(proc / 'smaps_rollup', 'Swap'))
                memory.write(json.dumps(row) + '\n')
                memory.flush()
                live_thermal = thermal_module.snapshot(sensors)
                live_battery = time.monotonic()
                if live_battery - last_battery_check >= 30:
                    latest_battery = battery_status()
                    last_battery_check = live_battery
                if row['available_kib'] is not None:
                    minimum_available = min(minimum_available or row['available_kib'], row['available_kib'])
                peak_rss = max(peak_rss, row['rss_kib'] or 0)
                peak_vm_swap = max(peak_vm_swap, row['vm_swap_kib'] or 0)
                reason, teardown_deadline = monitor_reason(
                    child.poll(), thermal.poll(), cooldown.poll(), OUT,
                    time.monotonic(), teardown_deadline)
                if reason is None and minimum_available is not None and minimum_available < args.memory_floor_gib * 1024 * 1024:
                    reason = f'MemAvailable below {args.memory_floor_gib:g} GiB'
                if reason is None and latest_battery and latest_battery.get('percentage', 100) < 45:
                    reason = 'battery below 45 percent'
                if reason is None and time.monotonic() - started > 3600:
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
    passes_count, cooling_count, _, verified = evidence(OUT)
    result = dict(command=command, plan=plan, returncode=child.returncode,
                  n_gpu_layers=ngl, backend=args.backend, device=device,
                  accelerator_layer_split=(dict(gpu=args.gpu_layers, npu=args.npu_layers)
                                           if tensor_split else None),
                  model_layer_split=layer_split, layer_profile=args.layer_profile,
                  output_layer=args.triad_output,
                  bounded_htp_layers=args.bounded_htp_layers,
                  htp_cache_slots=args.htp_cache_slots,
                  htp_cache_budget_mib=args.htp_cache_budget_mib,
                  thermal_policy='cooldown-only', pilot=args.pilot,
                  pilot_threads=args.pilot_threads,
                  thread_sweep=args.thread_sweep,
                  thermal_returncode=thermal.returncode, cooldown_returncode=cooldown.returncode,
                  stop_reason=reason, minimum_available_kib=minimum_available, peak_rss_kib=peak_rss,
                  peak_vm_swap_kib=peak_vm_swap,
                  passes=passes_count, cooling_events=cooling_count, replay_verified=verified)
    if (tensor_split or layer_map) and child.returncode == 0:
        assignments = layer_assignments((OUT / 'benchmark.log').read_text(errors='replace'))
        counts = count_layer_assignments((OUT / 'benchmark.log').read_text(errors='replace'))
        result['layer_assignment_counts'] = counts
        if layer_map:
            expected_counts = dict(CPU=layer_split['cpu'], GPUOpenCL=layer_split['gpu'], HTP0=layer_split['npu'])
        else:
            expected_counts = dict(CPU=layer_split['cpu'], GPUOpenCL=args.gpu_layers, HTP0=args.npu_layers)
        result['layer_assignment_verified'] = (
            counts['GPUOpenCL'] == expected_counts['GPUOpenCL'] and
            counts['HTP0'] == expected_counts['HTP0'] and
            counts['CPU'] == expected_counts['CPU'] and
            counts['GPUOpenCL'] + counts['HTP0'] + counts['CPU'] == 49)
        write('placement.json', dict(requested=(dict(cpu=layer_split['cpu'], gpu=layer_split['gpu'], npu=layer_split['npu'], output=args.triad_output)
                                                if layer_map else dict(gpu=args.gpu_layers, npu=args.npu_layers,
                                                                       cpu='remaining model layers')),
                                     actual=counts,
                                     verified=result['layer_assignment_verified']))
        if not result['layer_assignment_verified']:
            reason = 'actual accelerator layer assignment did not match requested split'
            result['stop_reason'] = reason
        if args.layer_profile and result['layer_assignment_verified']:
            try:
                summary = summarize_layer_profile(OUT / 'layer-timing.jsonl', assignments)
                result['layer_profile_verified'] = summary['complete']
                result['layer_profile_missing_layers'] = summary['missing_layers']
                write('layer-profile-summary.json', summary)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result['layer_profile_verified'] = False
                result['layer_profile_error'] = str(error)
                reason = 'incomplete per-layer timing evidence'
                result['stop_reason'] = reason
    if args.bounded_htp_layers is not None:
        log_text = (OUT / 'benchmark.log').read_text(errors='replace')
        staging = htp_cache_staging_evidence(log_text)
        result['htp_cache_staging'] = staging
        stage_log_verified = staging['operations'] > 0 and staging['staged_bytes'] > 0
        expected_cache_tensors = {
            f'blk.{layer}.ffn_{projection}_exps.weight'
            for layer in range(32, 32 + args.bounded_htp_layers)
            for projection in ('gate', 'up', 'down')
        }
        legacy_trace_verified = (child.returncode == 0 and verified and
                                 expected_cache_tensors.issubset(set(staging['allocated_tensors'])) and
                                 expected_cache_tensors.issubset(set(staging['graph_tensors'])))
        result['htp_cache_verified'] = stage_log_verified or legacy_trace_verified
        result['htp_cache_requests'] = staging['operations']
        result['htp_cache_staged_host_bytes'] = staging['staged_bytes']
        result['htp_cache_verification_basis'] = (
            'per-operation staged-byte log' if stage_log_verified else
            'cache-buffer allocation + HTP graph node + successful replay' if legacy_trace_verified else
            'no HTP expert-cache operation was proven')
        if child.returncode == 0 and not result['htp_cache_verified']:
            reason = 'no bounded HTP expert-cache operation was observed'
            result['stop_reason'] = reason
    if args.htp_resident_layers is not None:
        log_text = (OUT / 'benchmark.log').read_text(errors='replace')
        resident = htp_resident_evidence(log_text)
        expected = {f'blk.{layer}.ffn_{projection}_exps.weight'
                    for layer in range(32, 32 + args.htp_resident_layers)
                    for projection in ('gate', 'up', 'down')}
        result['htp_resident_evidence'] = resident
        result['htp_resident_verified'] = (
            child.returncode == 0 and verified and expected.issubset(set(resident['weights'])) and
            expected.issubset(set(resident['operation_tensors'])) and
            'HTP0' in resident['operation_buffers'])
        result['htp_resident_verification_basis'] = 'loader HTP allocations + HTP-buffer operation logs + successful replay'
        if child.returncode == 0 and not result['htp_resident_verified']:
            reason = 'resident HTP expert control was not proven on all selected tensors'
            result['stop_reason'] = reason
    write('run.json', result)
    print(json.dumps(result))
    if child.returncode or thermal.returncode or cooldown.returncode or reason or not verified:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
