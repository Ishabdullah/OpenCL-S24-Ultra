"""Stop this build's own process group if system memory becomes unsafe."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

command = sys.argv[2:]
if not command:
    raise SystemExit('Missing build command')
child = subprocess.Popen(command, start_new_session=True)
try:
    while child.poll() is None:
        fields = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                fields[parts[0].rstrip(':')] = int(parts[1])
        if fields.get('MemAvailable', 0) < 1536 * 1024:
            raise RuntimeError('MemAvailable dropped below the 1536 MiB build reserve; close other apps and rerun.')
        for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
            try:
                if (zone / 'type').read_text().strip() == 'battery':
                    degrees = int((zone / 'temp').read_text()) / 1000
                    if degrees > 43:
                        raise RuntimeError('Battery thermal sensor exceeded 43C; let the phone cool and rerun.')
            except (OSError, ValueError):
                continue
        time.sleep(5)
    raise SystemExit(child.wait())
except (KeyboardInterrupt, RuntimeError) as error:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    raise SystemExit('NPU build stopped: ' + str(error))
