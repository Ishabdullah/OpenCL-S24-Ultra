"""Add explicit OpenMP placement checks without changing the base harness."""
import hashlib
import json
import pathlib
import re
import measure
import validation

_original_process = measure.process
_original_issues = validation.issues

def cpus(value):
    result = set()
    for part in (value or '').split(','):
        if part:
            ends = part.split('-')
            result.update(range(int(ends[0]), int(ends[-1]) + 1))
    return result

def process_with_affinity(pid):
    result = _original_process(pid)
    for tid, worker in result['workers'].items():
        status = measure.read(f'/proc/{pid}/task/{tid}/status') or ''
        found = re.search(r'^Cpus_allowed_list:\s*(.+)$', status, re.M)
        worker['allowed_CPUs'] = found[1] if found else None
    return result

def placement_issues(result, directory):
    found = _original_issues(result, directory)
    config = result['config']
    if not config.get('explicit_openmp_placement'):
        return found
    expected = {i for i in range(8) if int(config['process_mask'], 16) & (1 << i)}
    metrics = result.get('latency_metrics') or {}
    start = metrics.get('request_start_monotonic_s')
    end = start + metrics.get('total_response_s', 0) if start else None
    samples = [s for s in result.get('samples', []) if start and start <= s['monotonic'] <= end and validation.live_process_sample(s)]
    if not samples:
        return found + ['Explicit placement lacks measured request samples']
    failures = []
    strategy = config['strategy']
    cpu_start = start if strategy == 'cpu' else metrics['switch_start_monotonic_s'] + metrics['handoff_ms'] / 1000 + metrics['first_decode_s']
    cpu_samples = [s for s in samples if strategy != 'gpu' and s['monotonic'] >= cpu_start]
    for sample in samples:
        main = cpus(sample.get('process', {}).get('allowed_CPUs'))
        if sample in cpu_samples:
            if main != {7}:
                failures.append('Explicit OpenMP CPU-phase main thread did not retain core7')
        elif main not in [expected, {7}]:
            failures.append('NPU-phase host affinity differs from declared team or master place')
        for worker in sample.get('process', {}).get('workers', {}).values():
            allowed = cpus(worker.get('allowed_CPUs'))
            if not allowed or not allowed <= expected or worker['processor'] not in expected:
                failures.append('Explicit OpenMP worker affinity/processor escaped requested core set')
    if len(cpu_samples) >= 2:
        first = cpu_samples[0]['process']['workers']
        last = cpu_samples[-1]['process']['workers']
        active = [tid for tid in first if tid in last and last[tid]['utime'] + last[tid]['stime'] - first[tid]['utime'] - first[tid]['stime'] >= 10]
        places = set()
        for tid in active:
            masks = {tuple(sorted(cpus(s['process']['workers'][tid].get('allowed_CPUs')))) for s in cpu_samples if tid in s['process']['workers']}
            if len(masks) != 1 or len(next(iter(masks), ())) != 1:
                failures.append('Active OpenMP CPU worker did not retain one singleton place')
            else:
                places.add(next(iter(masks))[0])
        if places != expected or len(active) != config['threads']:
            failures.append('Active OpenMP CPU team does not match declared unique places')
    elif strategy != 'gpu':
        failures.append('Too few CPU-phase placement samples')
    if not failures:
        # A pinned main thread intentionally has a subset of the team mask.
        found = [s for s in found if s != 'Benchmark process lost required cores despite process-level affinity']
    return found + sorted(set(failures))

measure.process = process_with_affinity
validation.issues = placement_issues

if __name__ == '__main__':
    import datetime
    import sys
    plan = pathlib.Path(sys.argv[1])
    epoch = measure.conditions()['power_epoch']
    records = []
    for config in json.loads(plan.read_text()):
        if measure.conditions()['power_epoch'] != epoch:
            raise SystemExit('Power epoch changed; preserve and create fresh matched plans')
        config['placement_extension_sha256'] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        result = measure.run(config)
        errors = placement_issues(result, measure.BASE / 'runs' / config['id'])
        records.append({'id': config['id'], 'errors': errors, 'metrics': result.get('latency_metrics'), 'status': result.get('status')})
        plan.with_suffix('.state.json').write_text(json.dumps({'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'records': records}, indent=2) + '\n')
        if result.get('status') == 'guarded_skip':
            print('Memory admission skip', config['id'], flush=True)
            continue
        if errors:
            raise SystemExit('Placement measurement excluded: ' + '; '.join(errors))
        metrics = result.get('latency_metrics') or {}
        if not (metrics.get('all_logits_finite') and metrics.get('positions_preserved') and metrics.get('same_model_and_cache')):
            raise SystemExit('Phase finite/state gate failed')
        print(config['id'], {k: metrics.get(k) for k in ['prefill_tok_s', 'decode_tok_s', 'total_response_s']}, flush=True)
