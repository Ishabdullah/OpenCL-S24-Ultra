"""Summarize guarded phase experiments without pooling different settings."""
import csv
import datetime
import hashlib
import importlib
import json
import pathlib
import sys

B = pathlib.Path(__file__).resolve().parent
plan = pathlib.Path(sys.argv[1])
configs = json.loads(plan.read_text())
checker = importlib.import_module('placement_measure_dual' if any(c.get('explicit_openmp_placement') for c in configs) else 'validation')
check = checker.placement_issues if hasattr(checker, 'placement_issues') else checker.issues
rows = []
tokens = []
for c in configs:
    d = B / 'runs' / c['id']
    if not (d / 'result.json').exists():
        rows.append({'id': c['id'], 'valid': False, 'issues': ['No completed result']})
        continue
    r = json.loads((d / 'result.json').read_text())
    m = r.get('latency_metrics') or {}
    bad = check(r, d)
    if not all(m.get(k) for k in ['all_logits_finite', 'same_model_and_cache', 'positions_preserved']):
        bad.append('Finite/model/cache/position gate failed')
    if 'PHASE cleanup complete' not in (d / 'stderr.txt').read_text():
        bad.append('Cleanup marker missing')
    rows.append({
        'id': c['id'], 'valid': not bad, 'issues': bad,
        'backend': 'CPU' if c['strategy'] == 'cpu' else (('OpenCL' if c.get('device', c.get('environment', {}).get('GGML_PERF_PHASE_DEVICE')) == 'GPUOpenCL' else 'NPU') + ('-prefill-CPU-decode' if c['strategy'] == 'hybrid' else '')),
        'model': c['model'], 'round': c.get('confirmation_round'),
        'threads': c['threads'], 'threads_batch': m.get('threads_batch', c.get('threads_batch', c['threads'])),
        'prompt': c['prompt'], 'generation': c['generation'], 'context': m.get('actual_context'),
        'batch': c['batch'], 'ubatch': c['ubatch'], 'attention': c['attention'],
        'requested_ngl': c['ngl'], 'repack': c.get('repack'),
        'OMP_PLACES': c.get('environment', {}).get('OMP_PLACES'),
        'registration_window_MiB': c.get('environment', {}).get('GGML_HEXAGON_REGISTRATION_WINDOW'),
        'power_epoch': r.get('power_conditions', {}).get('power_epoch'),
        **{k: m.get(k) for k in ['prefill_tok_s', 'decode_tok_s', 'prefill_s', 'decode_s', 'total_response_s', 'ttft_s', 'handoff_ms', 'first_decode_s', 'initialization_s', 'postload_admission_s', 'kv_copy_bytes', 'weight_copy_bytes']},
        'max_RSS_MiB': r.get('max_process_RSS_kB', 0) / 1024,
        'max_PSS_MiB': r.get('max_process_PSS_kB', 0) / 1024,
        'min_MemAvailable_MiB': r.get('min_system_MemAvailable_kB', 0) / 1024,
        'allocations': r.get('allocations'), 'library_paths': r.get('llama_library_paths'),
        'prompt_token_hash': m.get('prompt_token_hash'),
        'output_sha256': hashlib.sha256(json.dumps(m.get('output_tokens')).encode()).hexdigest(),
    })
    if not bad:
        tokens.append(m['output_tokens'])
summary = {
    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'plan': str(plan), 'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
    'complete_and_valid': len(tokens) == len(configs),
    'valid_outputs_all_agree': bool(tokens) and all(t == tokens[0] for t in tokens),
    'scope': 'Warm response excludes model load and admission, both separately recorded. Each row retains settings and power epoch. Single screens do not establish a repeated winner; profiled runs are diagnostics.',
    'records': rows,
}
output = plan.with_name(plan.stem + '.phase-summary.json')
output.write_text(json.dumps(summary, indent=2) + '\n')
columns = [k for row in rows for k in row if k not in ['allocations', 'library_paths', 'issues']]
columns = list(dict.fromkeys(columns))
with output.with_suffix('.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'summary': str(output), 'complete_and_valid': summary['complete_and_valid'], 'valid_outputs_all_agree': summary['valid_outputs_all_agree'], 'rows': [{k: r.get(k) for k in ['id', 'valid', 'issues', 'prefill_tok_s', 'decode_tok_s', 'total_response_s']} for r in rows]}, indent=2))
