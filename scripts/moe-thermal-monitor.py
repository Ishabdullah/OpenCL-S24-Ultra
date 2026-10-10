#!/usr/bin/env python3
"""Sample readable thermal-zone temperatures for an existing benchmark PID."""
import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import time


def zones():
    result = []
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        try:
            name = (zone / 'type').read_text().strip()
            if name == 'battery' or name.startswith(('cpu-', 'cpuss-', 'nsphmx-', 'nsphvx-')):
                result.append((name, zone / 'temp'))
        except OSError:
            pass
    return result


def snapshot(sensors):
    readings = {}
    for name, path in sensors:
        try:
            degrees = int(path.read_text()) / 1000
            if 0 <= degrees <= 150:
                readings[name] = degrees
        except (OSError, ValueError):
            pass
    cpu = [v for k, v in readings.items() if k.startswith(('cpu-', 'cpuss-'))]
    htp = [v for k, v in readings.items() if k.startswith(('nsphmx-', 'nsphvx-'))]
    return dict(battery_c=readings.get('battery'), cpu_max_c=max(cpu) if cpu else None,
                htp_max_c=max(htp) if htp else None, sensors_c=readings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('--log', type=Path, help='benchmark log for pass/stage tags')
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    sensors = zones()
    if not sensors:
        raise SystemExit('no readable target thermal zones')
    with args.output.open('a' if args.append else 'w') as output:
        while Path(f'/proc/{args.pid}').exists():
            row = snapshot(sensors)
            if args.log:
                try:
                    markers = re.findall(r'CACHE_BENCH (waiting |admitted |start |complete )?pass=(\d+)(?:/\d+)? slots=(\d+)', args.log.read_text())
                    if markers:
                        complete, position, slots = markers[-1]
                        row.update(pass_number=int(position), slots=int(slots), stage={'complete ': 'rest_or_setup', 'waiting ': 'admission'}.get(complete, 'evaluation'))
                except OSError:
                    pass
            row['utc'] = datetime.now(timezone.utc).isoformat()
            row['monotonic_s'] = time.monotonic()
            output.write(json.dumps(row) + '\n')
            output.flush()
            time.sleep(5)


if __name__ == '__main__':
    main()
