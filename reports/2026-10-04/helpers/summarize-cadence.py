"""Summarize equal offered-work blocks; never pool them with saturation tests."""
import argparse
import csv
import fcntl
import json
import math
import pathlib
import statistics

from validation import issues

B = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('schedule')
parser.add_argument('--output', required=True)
args = parser.parse_args()

def metrics(rows):
    if not rows:
        return None
    latency = sorted(r['service_s'] for r in rows)
    return {
        'requests': len(rows),
        'prefill_tok_s': sum(r['prompt'] for r in rows) / sum(r['prefill_s'] for r in rows),
        'decode_tok_s': sum(r['decode_steps'] for r in rows) / sum(r['decode_s'] for r in rows),
        'service_mean_s': statistics.mean(latency),
        'service_stdev_s': statistics.stdev(latency) if len(latency) > 1 else None,
        'service_p95_s': latency[math.ceil(.95 * len(latency)) - 1],
        'arrival_to_completion_mean_s': statistics.mean(r['arrival_to_completion_s'] for r in rows),
        'queue_max_s': max(r['queue_delay_s'] for r in rows),
        'deadline_misses': sum(r['missed_deadline'] for r in rows),
    }

with (B / 'resource.lock').open('a') as gate:
    fcntl.flock(gate, fcntl.LOCK_EX)
    summaries, requests = [], []
    for config in json.loads(pathlib.Path(args.schedule).read_text()):
        directory = B / 'runs' / config['id']
        if not (directory / 'result.json').exists():
            summaries.append({'id': config['id'], 'status': 'no final result; not included'})
            continue
        result = json.loads((directory / 'result.json').read_text())
        rows = result.get('request_rows') or []
        if not rows:
            for line in (directory / 'stdout.txt').read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if 'iteration' in row:
                    rows.append(row)
        rows = [r for r in rows if r['iteration'] >= 0]
        errors = issues(result, directory)
        if not rows:
            summaries.append({'id': config['id'], 'completed_requests': 0, 'issues': errors})
            continue
        duration = config['duration_seconds']
        period = rows[0]['request_period_s']
        offered = math.ceil(duration / period)
        start = rows[0]['scheduled_arrival_monotonic_s']
        final_completion = rows[-1]['service_start_monotonic_s'] + rows[-1]['service_s']
        # Define windows by offered arrival time, independent of execution speed.
        first = [r for r in rows if r['scheduled_arrival_monotonic_s'] < start + duration / 3]
        last = [r for r in rows if r['scheduled_arrival_monotonic_s'] >= start + 2 * duration / 3]
        first_metrics, last_metrics = metrics(first), metrics(last)
        samples = result.get('samples', [])
        measured = [s for s in samples if start <= s['monotonic'] <= max(start + duration, final_completion)]
        thermal = [s for s in result.get('public_thermal_samples', []) if start <= s['monotonic_s'] <= max(start + duration, final_completion)]
        complete = len(rows) == offered and result.get('abort_reason') is None and not errors
        points = []
        for sample in measured:
            stat = sample.get('process', {}).get('stat') or ''
            fields = stat[stat.rfind(')') + 2:].split()
            if len(fields) > 12:
                points.append((sample['monotonic'], int(fields[11]) + int(fields[12])))
        import os
        cpu_percent = 100 * (points[-1][1] - points[0][1]) / os.sysconf('SC_CLK_TCK') / (points[-1][0] - points[0][0]) if len(points) > 1 else None
        summary = {
            'id': config['id'], 'model': config['model'], 'device': rows[0]['device'],
            'power_epoch': result['power_conditions']['power_epoch'],
            'controlled_complete': complete, 'issues': errors,
            'abort_reason': result.get('abort_reason'),
            'offered_requests': offered, 'completed_requests': len(rows),
            'offered_duration_s': duration, 'request_period_s': period,
            'last_completion_since_first_arrival_s': final_completion - start,
            'backlog_drain_beyond_offered_window_s': max(0, final_completion - start - duration),
            'all_logits_finite': all(r['all_logits_finite'] for r in rows),
            'prompt_hashes': sorted({r['prompt_hash'] for r in rows}),
            'output_hashes': sorted({r['output_hash'] for r in rows}),
            'all': metrics(rows), 'first_third': first_metrics, 'last_third': last_metrics,
            'last_first_decode_ratio': last_metrics['decode_tok_s'] / first_metrics['decode_tok_s'] if first_metrics and last_metrics else None,
            'last_first_service_ratio': last_metrics['service_mean_s'] / first_metrics['service_mean_s'] if first_metrics and last_metrics else None,
            'postload_initial_conditions': {k: v for k, v in rows[0].items() if k.startswith('initial_') or k == 'postload_admission_s'},
            'process_CPU_percent_during_offered_window': cpu_percent,
            'max_process_RSS_MiB_whole_run': result.get('max_process_RSS_kB', 0) / 1024,
            'min_MemAvailable_MiB_whole_run': result.get('min_system_MemAvailable_kB', 0) / 1024,
            'allocations': result.get('allocations'),
            'thermal_statuses_measured_window': sorted({s['thermal_status'] for s in thermal}),
            'temperatures_C_ranges_measured_window': {
                name: [min(values), max(values)]
                for name in sorted({name for s in measured for name in s['temperatures_C'] if name == 'battery' or name.startswith(('cpu-', 'gpuss-', 'nsphmx-', 'nsphvx-'))})
                if (values := [s['temperatures_C'][name] for s in measured if name in s['temperatures_C']])
            },
            'CPU_max_kHz_values_measured_window': {str(i): sorted({s['cpu_max_kHz'][str(i)] for s in measured if s['cpu_max_kHz'].get(str(i))}) for i in range(2, 8)},
            'GPU_max_clock_values_measured_window': sorted({s['gpu']['max_gpuclk'] for s in measured if s['gpu'].get('max_gpuclk')}),
            'scope': 'One resident-model block per backend; fixed identical offered work, not saturation, energy measurement, cold launch, or replicated thermal superiority. RSS excludes some RPC/driver storage. Public thermal NONE does not establish absence of frequency throttling.',
        }
        summaries.append(summary)
        requests.extend({'run_id': config['id'], 'controlled_complete': complete, **r} for r in rows)
    output = pathlib.Path(args.output)
    output.with_suffix('.json').write_text(json.dumps({'summaries': summaries}, indent=2) + '\n')
    if requests:
        with output.with_suffix('.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(requests[0]))
            writer.writeheader()
            writer.writerows(requests)
    for s in summaries:
        print(json.dumps({k: s.get(k) for k in ['id', 'controlled_complete', 'completed_requests', 'all', 'last_first_decode_ratio']}), flush=True)
