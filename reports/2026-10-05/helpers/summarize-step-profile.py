"""Decompose diagnostic evaluation wall time, CPU time and demand faults."""
import datetime
import json
import pathlib
import statistics
import sys

from validation import issues

B = pathlib.Path(__file__).resolve().parent
plan = pathlib.Path(sys.argv[1])
records = []
for config in json.loads(plan.read_text()):
    directory = B / 'runs' / config['id']
    result_path = directory / 'result.json'
    if not result_path.exists():
        records.append({'id': config['id'], 'state': 'no completed result'})
        continue
    result = json.loads(result_path.read_text())
    samples = []
    for line in directory.joinpath('stderr.txt').read_text().splitlines():
        if not line.startswith('PHASE_STEP: '):
            continue
        row = {}
        for field in line.split()[1:]:
            name, value = field.split('=', 1)
            row[name] = float(value) if name.endswith('_s') else int(value)
        samples.append(row)
    groups = {}
    for name, values in [('prefill', [r for r in samples if r['tokens'] > 1]),
                         ('decode', [r for r in samples if r['tokens'] == 1])]:
        if not values:
            groups[name] = None
            continue
        wall = sum(r['wall_s'] for r in values)
        cpu = sum(r['user_s'] + r['system_s'] for r in values)
        groups[name] = {
            'evaluations': len(values), 'wall_s': wall,
            'process_CPU_s': cpu, 'process_CPU_percent': 100 * cpu / wall,
            'median_step_s': statistics.median(r['wall_s'] for r in values),
            'min_step_s': min(r['wall_s'] for r in values),
            'max_step_s': max(r['wall_s'] for r in values),
            'major_faults': sum(r['majflt'] for r in values),
            'minor_faults': sum(r['minflt'] for r in values),
            'input_blocks': sum(r['inblock'] for r in values),
            'voluntary_switches': sum(r['nvcsw'] for r in values),
            'involuntary_switches': sum(r['nivcsw'] for r in values),
            'first_step': values[0], 'last_step': values[-1],
        }
    errors = issues(result, directory)
    if not errors:
        assert len(samples) == (config['prompt'] + config['batch'] - 1) // config['batch'] + config['generation'] - 1
    records.append({
        'id': config['id'], 'state': 'completed diagnostic',
        'controlled_valid': not errors, 'issues': errors,
        'power_epoch': result['power_conditions']['power_epoch'],
        'config': config, 'metrics': result.get('latency_metrics'),
        'resource_groups': groups, 'steps': samples,
        'max_RSS_MiB': result.get('max_process_RSS_kB', 0) / 1024,
        'min_MemAvailable_MiB': result.get('min_system_MemAvailable_kB', 0) / 1024,
        'allocations': result.get('allocations'),
        'llama_library_paths': result.get('llama_library_paths'),
    })
output = plan.with_name(plan.stem + '.summary.json')
output.write_text(json.dumps({
    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Per-step diagnostic helper; wall/CPU/fault correlations do not establish pure causal overhead. RUSAGE_SELF includes all process threads. Not an uninstrumented benchmark or electrical-energy measurement.',
    'records': records,
}, indent=2) + '\n')
for record in records:
    metrics = record.get('metrics') or {}
    print(record['id'], record['state'], record.get('controlled_valid'),
          'PP/TG', metrics.get('prefill_tok_s'), metrics.get('decode_tok_s'),
          'decode', record.get('resource_groups', {}).get('decode'))
print(output)
