"""Compact current-epoch records, including skips and numerical exclusions."""
import argparse
import csv
import datetime
import fcntl
import json
import pathlib

from validation import issues

B = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--epoch', help='Export a recorded power epoch without changing current guards')
args = parser.parse_args()
with (B / 'resource.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    epoch = args.epoch or json.loads((B / 'conditions.json').read_text())['power_epoch']
    out = B / 'npu-investigation' / ('results-' + epoch)
    out.mkdir(exist_ok=True)
    records, rows, numerical = [], [], []
    for path in sorted((B / 'runs').glob('*/result.json')):
        result = json.loads(path.read_text())
        if result.get('power_conditions', {}).get('power_epoch') != epoch:
            continue
        config = result['config']
        if config.get('mode') not in ['sustained', 'logits']:
            continue
        errors = issues(result, path.parent)
        command_path = path.parent / 'command.json'
        command = json.loads(command_path.read_text()) if command_path.exists() else {}
        record = {
            'id': result['id'], 'power_epoch': epoch,
            'config': config, 'command': command.get('command'),
            'artifact_sha256': command.get('artifact_sha256'),
            'model_metadata': command.get('model_metadata'),
            'harness_source_sha256': command.get('harness_source_sha256'),
            'controlled_valid': not errors and result.get('status') != 'guarded_skip',
            'issues': errors, 'status': result.get('status'),
            'exit_code': result.get('exit_code'),
            'abort_reason': result.get('abort_reason'),
            'wall_seconds': result.get('wall_seconds'),
            'cooldown_seconds': result.get('cooldown_seconds'),
            'llama_library_paths': result.get('llama_library_paths'),
            'allocations': result.get('allocations'),
            'numerical_validation': result.get('numerical_validation'),
            'max_RSS_MiB': result.get('max_process_RSS_kB', 0) / 1024,
            'max_PSS_MiB': result.get('max_process_PSS_kB', 0) / 1024,
            'min_MemAvailable_MiB': result.get('min_system_MemAvailable_kB', 0) / 1024,
            'sample_count': len(result.get('samples', [])),
            'raw_result_path': str(path),
        }
        records.append(record)
        for row in result.get('request_rows') or []:
            if row.get('iteration', -1) < 0:
                continue
            rows.append({'run_id': result['id'], 'controlled_valid': record['controlled_valid'],
                         'source_variant': config.get('source_variant'),
                         'model': config['model'], 'power_epoch': epoch, **row})
        if record['numerical_validation']:
            numerical.append({'run_id': result['id'], 'controlled_valid': record['controlled_valid'],
                              'model': config['model'], 'ngl': config['ngl'],
                              'repack': config.get('repack'), 'power_epoch': epoch,
                              **record['numerical_validation']})
    (out / 'cadence-quality-records.json').write_text(json.dumps({
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'power_epoch': epoch, 'records': records,
        'scope': 'Immutable raw records retain full telemetry. Compact export preserves configuration, skips, exclusions, allocations, numerical checks, and artifact provenance. Warm response screens and sustained blocks must remain separate.',
    }, indent=2) + '\n')
    for name, values in [('request-responses', rows), ('numerical-gates', numerical)]:
        (out / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        if values:
            fields = list(dict.fromkeys(k for row in values for k in row))
            with (out / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
    print(out, 'records', len(records), 'requests', len(rows), 'numerical', len(numerical))
