#!/usr/bin/env python3
"""Launch a fresh evidence run outside the caller's process group."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def launch(evidence, preflight_only=False):
    if Path(evidence).name != evidence or evidence in ('', '.', '..'):
        raise ValueError('evidence must be a single directory name')
    target = HERE / evidence
    if target.exists() and not (target.is_dir() and sorted(p.name for p in target.iterdir()) == ['RESERVED']):
        raise FileExistsError(target)
    log = HERE / (evidence + '.driver.log')
    pidfile = HERE / (evidence + '.pid')
    if log.exists() or pidfile.exists():
        raise FileExistsError('launcher evidence already exists')
    cmd = [sys.executable, '-u', str(HERE / 'run.py'), '--stage', 'qualification',
           '--run', '--evidence', evidence]
    if preflight_only:
        cmd.append('--preflight-only')
    with log.open('x') as stream, open('/dev/null', 'rb') as null:
        child = subprocess.Popen(cmd, stdin=null, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=HERE)
    pidfile.write_text(str(child.pid) + chr(10))
    return child.pid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    print(launch(args.evidence, args.preflight_only))
