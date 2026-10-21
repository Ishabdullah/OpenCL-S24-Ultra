#!/usr/bin/env python3
"""Run a sequential, replay-verified accelerator layer-count sweep."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py'
DEFAULT_OUT = ROOT / 'reports/2026-10-19/qwen3-30b-a3b-thread-sweep/accelerator-layer-sweeps'


def pooled_rate(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'benchmark.jsonl').read_text().splitlines()]
    passes = [row for row in rows if row.get('type') == 'pass']
    complete = next((row for row in rows if row.get('type') == 'complete'), None)
    if complete is None or len(passes) != 2 or not all(row['bit_identical_positions'] == (0 if i == 0 else row['evaluated_positions'])
                                                           for i, row in enumerate(passes)):
        raise ValueError(f'{run_dir}: missing verified two-pass result')
    evaluations = sum(row['decode_evaluations'] for row in passes)
    seconds = sum(row['decode_s'] for row in passes)
    return evaluations / seconds, passes


def declining_at_midpoint(history, layer, midpoint, consecutive=3):
    """Stop only after midpoint and three consecutive layer-by-layer declines."""
    if layer < midpoint or len(history) < consecutive + 1:
        return False
    tail = history[-(consecutive + 1):]
    return all(tail[i]['pooled_tok_s'] > tail[i + 1]['pooled_tok_s']
               for i in range(consecutive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='permit physical device runs')
    parser.add_argument('--backend', choices=('gpu', 'npu', 'both'), default='both')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT / datetime.now(timezone.utc).strftime('attempt-%Y%m%dT%H%M%SZ'))
    parser.add_argument('--first-layer', type=int, default=1, choices=range(1, 50))
    parser.add_argument('--last-layer', type=int, default=49, choices=range(1, 50))
    parser.add_argument('--slots', type=int, default=64, choices=(48, 64, 80))
    parser.add_argument('--tokens', type=int, default=16, choices=(8, 16, 32, 64))
    parser.add_argument('--memory-floor-gib', type=float, default=3.0)
    parser.add_argument('--decline-streak', type=int, default=3)
    args = parser.parse_args()
    if args.first_layer > args.last_layer:
        parser.error('--first-layer must not exceed --last-layer')
    layers = list(range(args.first_layer, args.last_layer + 1))
    midpoint = (args.first_layer + args.last_layer + 1) // 2
    backends = ('gpu', 'npu') if args.backend == 'both' else (args.backend,)
    print(json.dumps(dict(dry_run=not args.run, backends=backends, layers=layers,
                          midpoint_layer=midpoint, decline_streak=args.decline_streak,
                          threads=1, slots=args.slots, cache_mib=args.slots * 40,
                          tokens=args.tokens, memory_floor_gib=args.memory_floor_gib,
                          replay='two passes per layer count; bit-identical within setting',
                          thermal_policy='cooldown-only; 180-second inter-pass and admission holds'), indent=2))
    if not args.run:
        return
    out = args.out_dir.resolve()
    if out.exists():
        unexpected = [item.name for item in out.iterdir() if item.name not in ('launcher.log', 'launcher.pid')]
        if unexpected:
            raise SystemExit(f'refusing to reuse evidence directory containing {unexpected}: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps(dict(
        started_utc=datetime.now(timezone.utc).isoformat(), backends=backends,
        layers=layers, midpoint_layer=midpoint, decline_streak=args.decline_streak,
        threads=1, slots=args.slots, cache_mib=args.slots * 40, tokens=args.tokens,
        memory_floor_gib=args.memory_floor_gib, thermal_policy='cooldown-only'), indent=2) + '\n')
    (out / 'launcher.json').write_text(json.dumps(dict(
        pid=os.getpid(), wake_lock=os.environ.get('TERMUX_WAKE_LOCK_ACTIVE') == '1',
        backends=backends, layer_counts=layers), indent=2) + '\n')
    (out / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    (out / 'status.json').write_text(json.dumps(dict(state='running', completed=[]), indent=2) + '\n')
    csv_path = out / 'summary.csv'
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=('backend', 'n_gpu_layers', 'pooled_tok_s',
                                                       'pass1_tok_s', 'pass2_tok_s',
                                                       'replay_verified', 'minimum_available_kib',
                                                       'peak_rss_kib', 'peak_vm_swap_kib'))
        writer.writeheader()
        all_results = []
        early_stops = []
        for backend in backends:
            history = []
            for layer in layers:
                run_dir = out / backend / f'ngl-{layer:02d}'
                command = [sys.executable, str(RUNNER), '--run', '--pilot',
                           '--backend', backend, '--ngl', str(layer),
                           '--pilot-threads', '1', '--pilot-slots', str(args.slots),
                           '--pilot-tokens', str(args.tokens), '--memory-floor-gib', str(args.memory_floor_gib),
                           '--out-dir', str(run_dir)]
                print(json.dumps(dict(event='starting', backend=backend, n_gpu_layers=layer,
                                      command=command)), flush=True)
                proc = subprocess.run(command, cwd=ROOT)
                if proc.returncode != 0:
                    status = dict(state='stopped_on_run_failure', backend=backend,
                                  n_gpu_layers=layer, returncode=proc.returncode,
                                  completed=all_results, utc=datetime.now(timezone.utc).isoformat())
                    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                    raise SystemExit(proc.returncode)
                rate, passes = pooled_rate(run_dir)
                run_meta = json.loads((run_dir / 'run.json').read_text())
                row = dict(backend=backend, n_gpu_layers=layer, pooled_tok_s=rate,
                           pass1_tok_s=passes[0]['decode_tok_s'], pass2_tok_s=passes[1]['decode_tok_s'],
                           replay_verified=run_meta['replay_verified'],
                           minimum_available_kib=run_meta['minimum_available_kib'],
                           peak_rss_kib=run_meta['peak_rss_kib'], peak_vm_swap_kib=run_meta['peak_vm_swap_kib'])
                writer.writerow(row)
                csv_file.flush()
                history.append(row)
                all_results.append(row)
                with (out / 'summary.jsonl').open('a') as records:
                    records.write(json.dumps(row) + '\n')
                status = dict(state='running', completed=all_results,
                              current_backend=backend, last_layer=layer,
                              utc=datetime.now(timezone.utc).isoformat())
                if declining_at_midpoint(history, layer, midpoint, args.decline_streak):
                    early_stops.append(dict(backend=backend, stop_layer=layer,
                                            decline_streak=args.decline_streak))
                    status.update(state='running', early_stops=early_stops)
                    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                    print(json.dumps(dict(event='backend_stopped_after_sustained_decline', backend=backend,
                                          n_gpu_layers=layer, results=history)), flush=True)
                    break
                (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                print(json.dumps(dict(event='result', **row)), flush=True)
            if (out / 'status.json').exists():
                status = json.loads((out / 'status.json').read_text())
                if status.get('state') == 'stopped_on_run_failure':
                    break
        status = json.loads((out / 'status.json').read_text())
        if status.get('state') == 'running':
            status.update(state='complete_with_early_stops' if early_stops else 'complete',
                          early_stops=early_stops, finished_utc=datetime.now(timezone.utc).isoformat())
            (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
