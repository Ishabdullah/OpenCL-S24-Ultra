#!/usr/bin/env python3
"""Export existing local evidence; never start or rerun an experiment.

Usage: python export_snapshot.py /path/to/performance-investigation
Requires that investigation's validation.py and original run directories.
The public snapshot retains individual timings and explicit exclusions, while
large diagnostic traces, models, SDK files and binaries stay on the device.
"""
import collections
import csv
import datetime
import hashlib
import json
import pathlib
import re
import sys

base = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(base))
from validation import issues

out = pathlib.Path(__file__).resolve().parent / 'data'
out.mkdir(parents=True, exist_ok=True)
home = str(pathlib.Path.home())
prefix = '/data/data/com.termux/files/usr'

def public(value):
    if isinstance(value, str):
        return value.replace(home, '~').replace(prefix, '$PREFIX')
    if isinstance(value, dict):
        return {public(k): public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value

def write_json(name, value):
    if name == 'runs.json':
        text = '[\n' + ',\n'.join(json.dumps(public(row), separators=(',', ':')) for row in value) + '\n]\n'
    else:
        text = json.dumps(public(value), indent=2) + '\n'
    (out / name).write_text(text)

def write_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(public(v), separators=(',', ':'))
                         if isinstance(v, (dict, list)) else public(v)
                         for k, v in row.items()} for row in rows)

runs, benches, latencies, allocations, sustained, telemetry = [], [], [], [], [], []
for path in sorted((base / 'runs').glob('*/result.json')):
    result = json.loads(path.read_text())
    config = result['config']
    errors = issues(result, path.parent)
    common = {'id': path.parent.name, 'model': config.get('model'),
              'group': config.get('group'), 'mode': config.get('mode', 'bench'),
              'power_epoch': result.get('power_conditions', {}).get('power_epoch'),
              'validation_issues': errors, 'passes_current_run_checks': not errors,
              'diagnostic_timing': config.get('validation_timing_not_comparable', False),
              'exit_code': result.get('exit_code'), 'abort_reason': result.get('abort_reason')}
    keep = {k: v for k, v in result.items() if k not in [
        'samples', 'power_conditions', 'before', 'after',
        'controlled_comparison_issues', 'controlled_comparison_valid']}
    keep.update(common)
    keep['source_result_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    keep['power_conditions'] = {k: v for k, v in result.get('power_conditions', {}).items()
                                if k != 'history'}
    for when in ['before', 'after']:
        keep[when] = {k: v for k, v in result.get(when, {}).items()
                      if k not in ['cpusets', 'loadavg']}
    samples = result.get('samples', [])
    names = sorted({name for s in samples for name in s.get('temperatures_C', {})})
    keep['temperature_ranges_C'] = {name: [min(vals), max(vals)] for name in names
                                    if (vals := [s['temperatures_C'][name] for s in samples
                                                 if name in s.get('temperatures_C', {})])}
    keep['CPU_governor_ceiling_kHz_values'] = {
        str(i): sorted({s['cpu_max_kHz'][str(i)] for s in samples
                        if s.get('cpu_max_kHz', {}).get(str(i))}) for i in range(8)}
    keep['sample_count'] = len(samples)
    keep['harness_cgroups'] = sorted({s['harness_cgroup'] for s in samples
                                    if s.get('harness_cgroup')})
    command = path.parent / 'command.json'
    if command.exists():
        keep['command_record'] = json.loads(command.read_text())
    for name in ['stdout.txt', 'stderr.txt']:
        p = path.parent / name
        if p.exists():
            keep[name + '_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    runs.append(keep)
    for row in result.get('benchmark_rows') or []:
        benches.append({**common, 'threads': row.get('n_threads'),
                        'batch': row.get('n_batch'), 'ubatch': row.get('n_ubatch'),
                        'ngl': row.get('n_gpu_layers'), 'device': row.get('devices'),
                        'attention': row.get('flash_attn'), 'prompt': row.get('n_prompt'),
                        'generation': row.get('n_gen'), 'depth': row.get('n_depth'),
                        'mean_tok_s': row.get('avg_ts'), 'stddev_tok_s': row.get('stddev_ts'),
                        'samples_ns': row.get('samples_ns'), 'samples_tok_s': row.get('samples_ts'),
                        'max_RSS_MiB': result.get('max_process_RSS_kB', 0) / 1024,
                        'min_MemAvailable_MiB': result.get('min_system_MemAvailable_kB', 0) / 1024})
    if result.get('latency_metrics'):
        latencies.append({**common, **result['latency_metrics']})
    for index, row in enumerate((result.get('allocations') or {}).get('buffers', [])):
        allocations.append({**common, 'requested_context': config.get('context'),
                            'record_index': index, **row})
    for row in result.get('request_rows') or []:
        sustained.append({**common, **row})
    if config.get('mode') == 'sustained':
        for s in samples:
            temps = s.get('temperatures_C', {})
            def maximum(starts):
                return max((v for k, v in temps.items() if k.startswith(starts)), default=None)
            status = s.get('process', {}).get('status') or ''
            rss = re.search(r'VmRSS:\s+(\d+)', status)
            telemetry.append({**common, 'UTC': s.get('UTC'), 'monotonic': s.get('monotonic'),
                              'battery_C': temps.get('battery'), 'hottest_CPU_C': maximum('cpu-'),
                              'hottest_GPU_C': maximum('gpuss-'),
                              'hottest_NPU_C': maximum(('nsphmx-', 'nsphvx-')),
                              'CPU7_ceiling_kHz': s.get('cpu_max_kHz', {}).get('7'),
                              'GPU': s.get('gpu'),
                              'MemAvailable_MiB': s.get('meminfo_kB', {}).get('MemAvailable', 0) / 1024,
                              'RSS_MiB': int(rss.group(1)) / 1024 if rss else None,
                              'process_cgroup': s.get('process', {}).get('cgroup')})

write_json('runs.json', runs)
write_csv('llama-bench.csv', benches)
write_csv('whole-response.csv', latencies)
write_csv('allocations.csv', allocations)
write_csv('sustained-requests.csv', sustained)
write_csv('sustained-telemetry.csv', telemetry)
with (base / 'model-metadata.csv').open() as f:
    write_csv('model-metadata.csv', list(csv.DictReader(f)))
write_json('model-metadata.json', [{k: v for k, v in model.items() if k != 'tensors'}
                                  for model in json.loads((base / 'models.json').read_text())])
selected = [
    'profiling-results.json', 'specialized-roundtrip-results.json',
    'npu-investigation/first-matched-results.json', 'npu-investigation/fourb-matched-results.json',
    'npu-investigation/fa-quality-results.json', 'npu-investigation/hybrid-quality-results.json',
    'npu-investigation/hmx-execution-proof.json', 'npu-investigation/decode-weight-bandwidth-proxy.json',
    'npu-investigation/known-good-npu-checkpoint.json', 'npu-investigation/npu-configure-command.json',
    'npu-investigation/npu-elf-validation.json', 'npu-investigation/known-good-preservation-check.json',
    'npu-investigation/original-preservation-audit.json',
    'npu-investigation/sustained-fixed-context-15min-r0-results.json',
    'npu-investigation/sustained-matched-cold-r1-results.json',
    'npu-investigation/sustained-start-bias-correction.json', 'USER_PAUSE.json']
for name in selected:
    p = base / name
    if p.exists():
        write_json(p.name, json.loads(p.read_text()))
write_json('inventory.json', {
    'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'completed_result_records': len(runs),
    'mode_counts': dict(collections.Counter(r['mode'] for r in runs)),
    'passes_current_run_checks_by_mode': dict(collections.Counter(
        r['mode'] for r in runs if r['passes_current_run_checks'])),
    'benchmark_rows': len(benches), 'whole_response_rows': len(latencies),
    'meaning_of_passes_current_run_checks': 'No detected exclusion; not proof of a matched pair, final confirmation, thermal equality or non-diagnostic timing.',
    'raw_storage': '~/llama-opencl-build/performance-investigation/runs',
    'omissions': ['large raw per-thread traces', 'stdout/stderr text (hashes retained)',
                  'binary/model/SDK/vendor artifacts'],
    'paused_by_user': True})
print(json.dumps({'runs': len(runs), 'benchmark_rows': len(benches),
                  'whole_response_rows': len(latencies), 'output': str(out)}))
