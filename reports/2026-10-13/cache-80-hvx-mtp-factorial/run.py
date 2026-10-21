#!/usr/bin/env python3
"""Guarded, resumable-by-new-directory factorial experiment; default is dry-run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = Path('/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf')
BIN = ROOT / '.work-npu/build/bin/llama-cli'
PROMPT = 'Explain why a cached expert can reduce decode latency in one paragraph.'


def plan(stage):
    if stage == 'qualification':
        cells = [(4, 4, m) for m in ('off', 'on')] + [(1, 6, m) for m in ('off', 'on')]
    elif stage == 'screen':
        # Thread-major, descending HVX, paired off/on to minimize temporal drift.
        cells = [(h, t, m) for t in (4, 5, 6) for h in (4, 3, 2, 1) for m in ('off', 'on')]
    else:
        raise ValueError(stage)
    return [dict(h=h, threads=t, mtp=m, tokens=16 if stage == 'qualification' else 64)
            for h, t, m in cells]


def command(cell):
    spec = (['--spec-type', 'draft-mtp', '--spec-draft-n-max', '2',
             '--spec-draft-n-min', '0', '--spec-draft-p-min', '0']
            if cell['mtp'] == 'on' else ['--spec-type', 'none'])
    return [str(BIN), '-m', str(MODEL), '-p', PROMPT, '-n', str(cell['tokens']),
            '-t', str(cell['threads']), '--no-display-prompt', *spec]


def environment(cell, source=None):
    source = os.environ if source is None else source
    env = {k: v for k, v in source.items() if not k.startswith(('GGML_', 'MOE_'))
           and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'ADSP_LIBRARY_PATH')}
    env.update(GGML_HEXAGON_NHVX=str(cell['h']), GGML_HEXAGON_NHMX='1',
               GGML_CPU_MOE_CACHE_SLOTS='80', GGML_CPU_MOE_CACHE_MIB='3200',
               LD_LIBRARY_PATH=str(BIN.parent),
               ADSP_LIBRARY_PATH=str(ROOT / '.work-npu/build/ggml/src/ggml-hexagon'))
    return env


def mem_kib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise RuntimeError('MemAvailable unavailable')


def battery():
    try:
        result = subprocess.run(['termux-battery-status'], capture_output=True, text=True,
                                timeout=5, check=True)
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ValueError('not an object')
        level = data['percentage']
        status = data['status']
        if type(level) is not int or not 0 <= level <= 100 or not isinstance(status, str):
            raise ValueError('invalid battery fields')
        status = status.strip().lower()
        if status not in ('charging', 'full', 'discharging', 'not charging'):
            raise ValueError('unknown battery status')
        return level, status in ('charging', 'full')
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('battery status unavailable') from exc


def temperatures():
    values = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            name = (zone / 'type').read_text().strip()
            temp = int((zone / 'temp').read_text()) / 1000
            if 0 <= temp <= 150:
                values[name] = temp
        except (OSError, ValueError):
            pass
    def maximum(prefixes):
        readings = [v for k, v in values.items() if k.startswith(prefixes)]
        return max(readings) if readings else None
    return dict(b=values.get('battery'), c=maximum(('cpu-', 'cpuss-')),
                htp=maximum(('nsphmx-', 'nsphvx-')), zones=values)


def sample():
    capacity, charging = battery()
    return dict(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
                capacity=capacity, charging=charging, available_kib=mem_kib(),
                thermal=temperatures())


def admission(row):
    if not row['charging'] or row['capacity'] < 40:
        return 'charging and battery >=40% required'
    if row['available_kib'] < 1536 * 1024:
        return 'MemAvailable below 1.5 GiB reserve'
    if any(row['thermal'][k] is None for k in ('b', 'c', 'htp')):
        return 'thermal sensors unavailable'
    return None


def hard_stop(row, low_since=None):
    t = row['thermal']
    if any(t[k] is None for k in ('b', 'c', 'htp')):
        return 'thermal sensor unavailable', low_since
    if t['b'] >= 45 or t['c'] >= 100 or t['htp'] >= 85:
        return 'hard thermal threshold', low_since
    if row['available_kib'] < 768 * 1024:
        return 'MemAvailable below 768 MiB', low_since
    low_since = (low_since or row['monotonic']) if row['available_kib'] < 1024 * 1024 else None
    if low_since is not None and row['monotonic'] - low_since >= 5:
        return 'MemAvailable below 1 GiB for 5 seconds', low_since
    return None, low_since


def append(path, record):
    with path.open('a') as f:
        f.write(json.dumps(record) + '\n')

def cool(telemetry, lifecycle):
    start = time.monotonic()
    while True:
        row = sample()
        append(telemetry, dict(phase='cooling', **row))
        reason, _ = hard_stop(row)
        if reason or admission(row):
            raise RuntimeError(reason or admission(row))
        elapsed = time.monotonic() - start
        soft = all(row['thermal'][k] <= limit for k, limit in [('b', 40), ('c', 55), ('htp', 50)])
        if elapsed >= 180 and soft:
            append(lifecycle, dict(event='cool_complete', elapsed_s=elapsed, soft=True))
            return
        if elapsed >= 300:
            raise RuntimeError('cooling maximum 300 seconds exceeded without soft targets')
        time.sleep(min(1, max(0.01, 300 - elapsed)))


def run(stage, evidence=None, preflight_only=False):
    out = HERE / (evidence or stage)
    if out.parent != HERE or out.name in ('', '.', '..'):
        raise RuntimeError('evidence must be a directory name')
    reserved = out.is_dir() and sorted(p.name for p in out.iterdir()) == ['RESERVED']
    if out.exists() and not reserved:
        raise RuntimeError(f'{out} exists; never overwrite evidence')
    if not BIN.is_file() or not MODEL.is_file():
        raise RuntimeError('llama-cli or model missing')
    initial = sample()
    if admission(initial):
        raise RuntimeError(admission(initial))
    if reserved:
        (out / 'RESERVED').unlink()
    else:
        out.mkdir()
    lifecycle, telemetry = out / 'lifecycle.jsonl', out / 'telemetry.jsonl'
    append(lifecycle, dict(event='preflight', **initial))
    if preflight_only:
        append(lifecycle, dict(event='preflight_only_complete'))
        return
    (out / 'plan.json').write_text(json.dumps(plan(stage), indent=2) + '\n')
    results = []
    try:
        for index, cell in enumerate(plan(stage)):
            cool(telemetry, lifecycle)
            row = sample()
            if admission(row):
                raise RuntimeError(admission(row))
            cmd = command(cell)
            append(lifecycle, dict(event='start', index=index, cell=cell, command=cmd, utc=row['utc']))
            stdout, stderr = out / f'{index:02d}.stdout.log', out / f'{index:02d}.stderr.log'
            reason = None
            with stdout.open('wb') as so, stderr.open('wb') as se:
                child = subprocess.Popen(cmd, stdout=so, stderr=se, env=environment(cell), start_new_session=True)
                started = time.monotonic()
                low_since = None
                try:
                    while child.poll() is None:
                        row = sample()
                        append(telemetry, dict(phase='generation', index=index, pid=child.pid, **row))
                        reason, low_since = hard_stop(row, low_since)
                        if not reason and not row['charging']:
                            reason = 'charging lost'
                        if not reason and row['monotonic'] - started >= 720:
                            reason = '12 minute generation watchdog'
                        if reason:
                            break
                        time.sleep(1)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
            result = dict(index=index, cell=cell, command=cmd, returncode=child.returncode,
                          duration_s=time.monotonic()-started, stop_reason=reason,
                          stdout=stdout.name, stderr=stderr.name)
            results.append(result)
            append(out / 'results.jsonl', result)
            append(lifecycle, dict(event='finish', **result))
            if reason or child.returncode != 0:
                raise RuntimeError(reason or f'cell {index} exited {child.returncode}')
    except Exception as exc:
        append(lifecycle, dict(event='abort', error=str(exc)))
        raise
    finally:
        (out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('qualification', 'screen'), required=True)
    parser.add_argument('--run', action='store_true', help='execute inference; otherwise print dry-run plan')
    parser.add_argument('--evidence', help='new evidence directory name (never overwrite)')
    parser.add_argument('--preflight-only', action='store_true', help='record admission without inference')
    args = parser.parse_args()
    if args.preflight_only and not args.run:
        parser.error('--preflight-only requires --run')
    if args.run:
        run(args.stage, args.evidence, args.preflight_only)
    else:
        print(json.dumps(dict(dry_run=True, stage=args.stage, cells=plan(args.stage),
                              commands=[command(c) for c in plan(args.stage)],
                              cooling_s=[180, 300]), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
