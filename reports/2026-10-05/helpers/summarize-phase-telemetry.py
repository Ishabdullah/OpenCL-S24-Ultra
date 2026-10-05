"""Summarize sampled phase clocks, temperatures and worker placement."""
import collections
import datetime
import json
import pathlib
import statistics
import sys

B = pathlib.Path(__file__).resolve().parent
plan = pathlib.Path(sys.argv[1])
records = []

def span(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else None

for config in json.loads(plan.read_text()):
    path = B / 'runs' / config['id'] / 'result.json'
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    metrics = result.get('latency_metrics') or {}
    if not metrics.get('request_start_monotonic_s'):
        continue
    start = metrics['request_start_monotonic_s']
    split = metrics['switch_start_monotonic_s']
    end = start + metrics['total_response_s']
    phases = {}
    for name, lo, hi in [('prefill', start, split), ('decode-and-handoff', split, end)]:
        samples = [s for s in result.get('samples', []) if lo <= s['monotonic'] < hi]
        placements = collections.Counter()
        for sample in samples:
            placements.update(str(w['processor']) for w in sample.get('process', {}).get('workers', {}).values())
        phases[name] = {
            'sample_count': len(samples),
            'temperature_C': {
                label: span([max([v for k, v in s['temperatures_C'].items() if k.startswith(prefix)], default=0) for s in samples])
                for label, prefix in [('CPU-max', ('cpu-',)), ('NPU-max', ('nsphmx-', 'nsphvx-')), ('GPU-max', ('gpuss-',))]
            },
            'battery_C': span([s['temperatures_C']['battery'] for s in samples]),
            'cpu_current_kHz': {str(i): span([int(s['cpu_kHz'][str(i)]) for s in samples if s['cpu_kHz'].get(str(i))]) for i in range(8)},
            'cpu_max_kHz': {str(i): sorted({int(s['cpu_max_kHz'][str(i)]) for s in samples if s['cpu_max_kHz'].get(str(i))}) for i in range(8)},
            'public_thermal_statuses': sorted({s['public_thermal']['thermal_status'] for s in samples if s.get('public_thermal')}),
            'cooling_device_states': {key: sorted({s.get('cooling_device_states', {}).get(key) for s in samples if s.get('cooling_device_states', {}).get(key) is not None}) for key in result.get('before', {}).get('cooling_device_states', {})},
            'worker_processor_snapshot_counts': dict(placements),
            'RSS_MiB': span([s['process']['status_kB']['VmRSS'] / 1024 for s in samples if s.get('process', {}).get('status_kB', {}).get('VmRSS')]),
            'PSS_MiB': span([s['process']['smaps_kB']['Pss'] / 1024 for s in samples if s.get('process', {}).get('smaps_kB', {}).get('Pss')]),
            'MemAvailable_MiB': span([s['meminfo_kB']['MemAvailable'] / 1024 for s in samples]),
        }
    records.append({'id': config['id'], 'metrics': metrics, 'phases': phases})

output = plan.with_name(plan.stem + '.telemetry-summary.json')
output.write_text(json.dumps({
    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Sampled clocks/caps, not cycle counters. Worker processor snapshots are not per-core utilization. Decode interval includes handoff. Zero public cooling state does not prove absence of other clock regulation. Process RSS/PSS omit some shared driver allocations; not total physical device use.',
    'records': records,
}, indent=2) + '\n')
for record in records:
    print(record['id'])
    for name, phase in record['phases'].items():
        print(name, phase['sample_count'], 'CPU-C', phase['temperature_C']['CPU-max'],
              'CPU7-max-kHz', phase['cpu_max_kHz']['7'], 'CPU7-current-kHz', phase['cpu_current_kHz']['7'])
print(output)
