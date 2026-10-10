#!/usr/bin/env python3
"""Validate and summarize a four-pass ABBA experiment; no significance claim."""
import argparse
import json
from pathlib import Path
import re
import statistics


def validate_admissions(admissions, thermal, require_complete=True):
    assert 1 <= len(admissions) <= 4, 'no valid thermal admission'
    if require_complete:
        assert len(admissions) == 4, 'incomplete thermal admission'
    assert [row['pass_number'] for row in admissions] == list(range(1, len(admissions)+1))
    assert [row['slots'] for row in admissions] == [0, 32, 32, 0][:len(admissions)]
    target = admissions[0]['target_c']
    policy_id = admissions[0].get('policy_id', 'strict-v1')
    if policy_id == 'relaxed-plus-3c-v1':
        tolerance = dict(battery_c=3.4, cpu_max_c=7.0, htp_max_c=7.0)
        ceiling = dict(battery_c=37.0, cpu_max_c=58.0, htp_max_c=48.0)
    else:
        assert policy_id == 'strict-v1', 'unknown admission policy'
        tolerance = dict(battery_c=0.4, cpu_max_c=4.0, htp_max_c=4.0)
        ceiling = dict(battery_c=34.0, cpu_max_c=55.0, htp_max_c=45.0)
    for decision in admissions:
        assert decision.get('policy_id', 'strict-v1') == policy_id, 'admission policy changed'
        assert decision['target_c'] == target, 'thermal target changed'
        assert decision['tolerance_c'] == tolerance and decision['hold_s'] == 30
        assert decision['initial_ceiling_c'] == ceiling
        start, end = decision['window_start_monotonic_s'], decision['window_end_monotonic_s']
        assert end - start >= 30, 'thermal hold too short'
        samples = [row for row in thermal if row.get('pass_number') == decision['pass_number']
                   and row.get('stage') == 'admission' and start <= row['monotonic_s'] <= end]
        assert len(samples) == decision['samples'] and len(samples) >= 2, 'missing admission samples'
        assert samples[0]['monotonic_s'] == start and samples[-1]['monotonic_s'] == end
        assert all(0 < after['monotonic_s'] - before['monotonic_s'] <= 3.5
                   for before, after in zip(samples, samples[1:])), 'thermal sampling gap'
        for key in tolerance:
            assert all(row.get(key) is not None and abs(row[key]-target[key]) <= tolerance[key]+1e-9
                       for row in samples), 'temperature outside admission band'
            assert samples[-1][key] == decision['admitted_c'][key]
            assert decision['window_c'][key] == dict(min=min(row[key] for row in samples), max=max(row[key] for row in samples))
            if decision['pass_number'] == 1:
                assert all(row[key] <= ceiling[key] for row in samples)
                assert statistics.median(row[key] for row in samples) == target[key]
    return admissions


def summarize(records, cache_stats=None, thermal=None, partial_first_thermal=False, admissions=None):
    config = next(row for row in records if row['type'] == 'config')
    reference = next(row for row in records if row['type'] == 'reference')
    passes = [row for row in records if row['type'] == 'pass']
    complete = [row for row in records if row['type'] == 'complete']
    assert len(passes) == 4 and len(complete) == 1, 'incomplete benchmark'
    assert [row['slots'] for row in passes] == [0, 32, 32, 0], 'order is not ABBA'
    assert [row['pass'] for row in passes] == [1, 2, 3, 4], 'invalid pass numbering'
    positions = reference['steps']
    assert positions > 1 and len(reference['tokens']) == positions, 'invalid reference'
    assert complete[0]['verified_positions'] == 3 * positions
    for row in passes:
        assert row['evaluated_positions'] == positions
        assert row['decode_evaluations'] == positions - 1
        assert row['bit_identical_positions'] == (0 if row['pass'] == 1 else positions), 'logit verification failed'
        assert row['prefill_s'] > 0 and row['decode_s'] > 0
        assert abs(row['evaluation_s'] - row['prefill_s'] - row['decode_s']) < 0.00001
        assert row['storage_bytes'] == row['prefill_storage_bytes'] + row['decode_storage_bytes']
    modes = {}
    for slots in (0, 32):
        group = [row for row in passes if row['slots'] == slots]
        modes[str(slots)] = dict(repetitions=len(group),
            mean_evaluation_s=statistics.mean(row['evaluation_s'] for row in group),
            mean_prefill_s=statistics.mean(row['prefill_s'] for row in group),
            mean_decode_s=statistics.mean(row['decode_s'] for row in group),
            pooled_decode_tok_s=sum(row['decode_evaluations'] for row in group) / sum(row['decode_s'] for row in group),
            mean_storage_bytes=statistics.mean(row['storage_bytes'] for row in group),
            mean_decode_storage_bytes=statistics.mean(row['decode_storage_bytes'] for row in group))
    baseline, cached = modes['0'], modes['32']
    pairs = []
    for before, after in ((passes[0], passes[1]), (passes[2], passes[3])):
        base = before if before['slots'] == 0 else after
        cache = after if after['slots'] == 32 else before
        pairs.append(dict(order=[before['slots'], after['slots']],
                          evaluation_speedup=base['evaluation_s'] / cache['evaluation_s'],
                          decode_speedup=base['decode_s'] / cache['decode_s'],
                          storage_reduction_fraction=1 - cache['storage_bytes'] / base['storage_bytes']))
    temperature = {}
    for row in passes:
        samples = [sample for sample in thermal or []
                   if sample.get('pass_number', 1) == row['pass']
                   and sample.get('stage', 'evaluation') == 'evaluation']
        if samples:
            def values(key):
                return [sample[key] for sample in samples if sample.get(key) is not None]
            battery, cpu, htp = values('battery_c'), values('cpu_max_c'), values('htp_max_c')
            temperature[str(row['pass'])] = dict(samples=len(samples),
                battery_first_c=battery[0] if battery else None,
                battery_last_c=battery[-1] if battery else None,
                battery_max_c=max(battery) if battery else None,
                cpu_max_c=max(cpu) if cpu else None,
                htp_max_c=max(htp) if htp else None)
    if admissions is not None:
        validate_admissions(admissions, thermal or [])
    return dict(config=config, reference=reference, passes=passes, modes=modes, pairs=pairs,
                evaluation_speedup=baseline['mean_evaluation_s'] / cached['mean_evaluation_s'],
                decode_speedup=baseline['mean_decode_s'] / cached['mean_decode_s'],
                storage_reduction_fraction=1 - cached['mean_storage_bytes'] / baseline['mean_storage_bytes'],
                cache_stats=cache_stats or [], temperature_by_pass=temperature,
                admissions=admissions,
                thermal_capture_partial_first_pass=partial_first_thermal,
                limitations=['Two repetitions per mode, one prompt/trajectory/device/session.',
                             ('Balanced order; thermal waits vary and page cache/scheduling are not reset.'
                              if admissions is not None else 'Balanced order and equal rests; page cache and scheduling are not reset.'),
                             ('Starting temperatures held within common admission bands; within-pass heating remains uncontrolled.'
                              if admissions is not None else 'Temperature is observed, without matched-temperature admission.'),
                             'No energy or sustained-thermal superiority claim.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--partial-first-thermal', action='store_true')
    args = parser.parse_args()
    folder = args.directory
    run = json.loads((folder / 'run.json').read_text())
    assert run['returncode'] == 0 and run['stop_reason'] is None, 'run failed or was stopped'
    records = [json.loads(line) for line in (folder / 'benchmark.jsonl').read_text().splitlines()]
    thermal = [json.loads(line) for line in (folder / 'thermal.jsonl').read_text().splitlines()]
    stats = []
    if (folder / 'benchmark.log').exists():
        completed = None
        for line in (folder / 'benchmark.log').read_text().splitlines():
            found = re.match(r'CACHE_BENCH complete pass=(\d+)', line)
            if found:
                completed = int(found.group(1))
            if line.startswith('MOE_CACHE slots='):
                assert completed in (2, 3), 'unexpected cache lifetime'
                row = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)}
                row['pass'] = completed
                stats.append(row)
        (folder / 'cache-stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    else:
        stats = json.loads((folder / 'cache-stats.json').read_text())
    assert len(stats) == 2 and [row['pass'] for row in stats] == [2, 3]
    admissions = json.loads((folder / 'admissions.json').read_text()) if (folder / 'admissions.json').exists() else None
    result = summarize(records, stats, thermal, args.partial_first_thermal, admissions)
    result['run'] = run
    (folder / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'evaluation_speedup': result['evaluation_speedup'],
                      'decode_speedup': result['decode_speedup'],
                      'storage_reduction_fraction': result['storage_reduction_fraction']}))


if __name__ == '__main__':
    main()
