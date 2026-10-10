#!/usr/bin/env python3
"""Pause our live benchmark only during its untimed between-pass rest."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import time

spec=importlib.util.spec_from_file_location('thermal',Path(__file__).with_name('moe-thermal-monitor.py'))
thermal=importlib.util.module_from_spec(spec)
spec.loader.exec_module(thermal)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid',type=int)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--after-pass',type=int,required=True)
    args=parser.parse_args()
    expected=str(Path(__file__).resolve().parents[1]/'.work-npu/build/bin/moe-cache-benchmark')
    proc=Path(f'/proc/{args.pid}')
    def alive():
        try: return (proc/'cmdline').read_bytes().split(b'\0')[0].decode()==expected
        except (OSError,UnicodeDecodeError): return False
    sensors=thermal.zones()
    handled=args.after_pass-1
    stopped=False
    log=args.directory/'benchmark.log'
    events=args.directory/'cooldown.jsonl'
    latest=args.directory/'cooldown-status.json'
    targets=dict(battery_c=40.0,cpu_max_c=55.0,htp_max_c=50.0)
    def status(value):
        value.update(utc=datetime.now(timezone.utc).isoformat())
        temporary=latest.with_suffix('.tmp')
        temporary.write_text(json.dumps(value)+'\n')
        temporary.replace(latest)
    try:
        while alive():
            content=log.read_text()
            completes=[int(value) for value in re.findall(r'CACHE_BENCH complete pass=(\d+)',content)]
            if completes and max(completes)>=args.after_pass and max(completes)>handled:
                position=max(completes)
                config_lines=re.findall(r'CACHE_BENCH start pass=(\d+)',content)
                started=max(map(int,config_lines)) if config_lines else 0
                records=[json.loads(line) for line in (args.directory/'benchmark.jsonl').read_text().splitlines()]
                total=len(next(row for row in records if row['type']=='config')['plan'])
                if position>=total: break
                if started>position:
                    # Never stop an evaluation: try the next untimed boundary.
                    handled=position
                    continue
                os.kill(args.pid,signal.SIGSTOP)
                stopped=True
                start=time.monotonic()
                first=thermal.snapshot(sensors)
                print(f'Cooling after pass {position}: at least 180 seconds; {first["battery_c"]} C battery',flush=True)
                status(dict(stage='cooling',after_pass=position,started_monotonic_s=start,minimum_s=180,maximum_s=300,targets_c=targets))
                while alive():
                    now=time.monotonic()
                    row=thermal.snapshot(sensors)
                    met=all(row.get(key) is not None and row[key]<=limit for key,limit in targets.items())
                    elapsed=now-start
                    if elapsed>=180 and (met or elapsed>=300): break
                    time.sleep(2)
                elapsed=time.monotonic()-start
                if alive(): os.kill(args.pid,signal.SIGCONT)
                stopped=False
                event=dict(after_pass=position,paused_s=elapsed,targets_c=targets,target_met=met,
                           first_c={key:first[key] for key in targets},last_c={key:row[key] for key in targets},
                           ended_monotonic_s=time.monotonic(),utc=datetime.now(timezone.utc).isoformat())
                with events.open('a') as output: output.write(json.dumps(event)+'\n')
                status(dict(stage='running',after_pass=position,last_c=event['last_c'],paused_s=elapsed))
                print(f'Resumed after pass {position}: {elapsed:.1f} seconds; {event["last_c"]}',flush=True)
                handled=position
            time.sleep(0.5)
    finally:
        if stopped and alive(): os.kill(args.pid,signal.SIGCONT)
        status(dict(stage='finished',after_pass=handled))

if __name__=='__main__': main()
