#!/usr/bin/env python3
"""Safe revision of the cache-80 HVX/MTP factorial harness.

This is a NEW, SEPARATE module.  It does NOT modify or import the original
``run.py``/``launch.py``; those and the original evidence directories are left
untouched as failure evidence (bundled under ``evidence/``).

ROOT CAUSE (per execution_history.log of the parent project, lines 48-49):
  Under llama.cpp ``LLAMA_LOAD_MODE_AUTO`` the loader calls
  ``ggml_backend_dev_get_props`` on EVERY registered device.  The Hexagon NPU
  backend hard-codes ``mmap_support = false`` (ggml-hexagon.cpp:7517).  When ANY
  device reports ``mmap_support=false`` the loader DISABLES mmap for the ENTIRE
  model load, forcing an eager full-file read of the 8.4 GiB GGUF into RSS.  On
  this device that overflows RAM in ~8-10 s and reproduces exactly the collapse
  captured in evidence/retry-01/telemetry.jsonl (4.46 -> 1.85 GiB available in
  ~8.35 s; minimum 1.846 GiB at t+44.11 s).

FIX (concretely pinned in :func:`command`, never left to defaults):
  ``-lm mmap``  -> forces mmap ON, overriding the hexagon mmap_support=false veto.
  ``-dev HTP0`` -> pinned NPU offload device (the factorial's NPU axis).
  ``-ngl 99``   -> 99 offloaded layers (all transformer layers).
  ``-c 256``    -> context window pinned to 256.
  ``-b 128``    -> batch (n_batch) pinned to 128.
  ``-ub 128``   -> microbatch (n_batch_sent) pinned to 128.
  ``-np 1``     -> one sequence (n_parallel = 1).
  ``-fa off``   -> flash attention OFF (earlier MTP race used auto/on).
  No warmup runs; no ``--split-mode``/``--main-gpu`` auto-placement/fitting.

SAFETY POLICY (all hard-gated, see README.md):
  * Admission requires MemAvailable >= 5 GiB ELIGIBILITY floor AND charging AND
    battery >= 40% AND thermal sensors present AND the screen prerequisite.
    The byte-accurate admission threshold factors in free swap: cache + working
    allowances (3200 + 87 + 2560 MiB) are backstopped by free swap at 50%
    effectiveness; if swap is insufficient, the shortfall is required from
    MemAvailable. Without swap, the threshold is 7895 MiB (MTP-on worst case).
    Final thresholds await revised architect policy (pending revision).
  * Retained-headroom floor = 2 GiB.  MemAvailable is never allowed to fall below
    2 GiB.  SwapFree is never allowed to fall below 512 MiB (emergency floor).
  * Declining-memory AND declining-swap projections: least-squares slopes over
    the most recent samples project both forward; if either crosses its floor,
    the cell is aborted EARLY -- before the floor is actually reached -- and the
    child process group is torn down promptly.
  * Fast memory sampling is independent of the (subprocess-backed) battery query:
    /proc/meminfo is read on every tick; battery is refreshed on a background
    thread and cached, so a hung ``termux-battery-status`` can never stall the
    memory watchdog.
  * Child RSS/PSS/swap telemetry is recorded each generation tick.
  * Strong lifecycle recording: every event (preflight/cool_complete/start/finish/
    abort/signal_abort) is fsync'd; summary.json is written in a ``finally``;
    SIGTERM/SIGINT are caught and recorded.  (SIGKILL is un-catchable -- a
    documented limitation.)
  * Single-run lock via fcntl.flock prevents two drivers running inference at once.
  * Inference (``--run``) is refused unless a screen prerequisite marker exists.
  * Default behaviour is a dry-run that only prints the pinned plan/commands.
    The implementer did NOT execute ``--run``.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent                                   # .../safe-factorial-harness
ROOT = HERE.parents[2]                                                   # .../OpenCL-S24-Ultra
BIN = ROOT / '.work-npu/build/bin/llama-cli'
MODEL = Path('/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/'
             'Qwen3.6-35B-A3B-UD-IQ1_M.gguf')
PROMPT = 'Explain why a cached expert can reduce decode latency in one paragraph.'

# ---- Root-cause / architect pins (never left to defaults) -----------------
MMAP_MODE = 'mmap'          # pin mmap ON (defeats hexagon mmap_support=false veto)
DEVICE = 'HTP0'             # pinned NPU offload device
OFFLOAD_LAYERS = 99         # 99 offloaded layers
CTX = 256                   # context window
BATCH = 128                 # n_batch
MICROBATCH = 128            # n_batch_sent (microbatch)
N_SEQ = 1                   # one sequence (n_parallel)
FLASH = 'off'               # flash attention off

# ---- Admission / retention policy -----------------------------------------
ADMISSION_MIN_AVAIL_KIB = 5 * 1024 * 1024      # 5 GiB minimum at admission
RETAINED_HEADROOM_KIB = 2 * 1024 * 1024        # 2 GiB retained-headroom floor
BATTERY_MIN = 40
ADMISSION_MIN_AVAIL_GIB = 5
# ---- Swap-aware admission arithmetic (pending revision) -------------------
# architect-6 specified fixed 7808/7895 MiB thresholds (never trust swap).
# User feedback: those are too high for a device with 15 GiB zram swap;
# swap and history are being factored in. Final thresholds await revised
# architect policy.
#
# Current working policy:
#   * 5 GiB MemAvailable is a minimum ELIGIBILITY floor (not sufficient alone).
#   * 2 GiB retained reserve is a hard floor on MemAvailable (never crossed).
#   * Cache + working allowances (3200 + 87 + 2560 MiB) are backstopped by
#     free swap at SWAP_BACKSTOP_RATIO effectiveness. If swap is insufficient
#     to backstop, the shortfall is required from MemAvailable.
#   * SwapFree is never allowed to fall below SWAP_HARD_FLOOR_KIB (512 MiB).
SWAP_BACKSTOP_RATIO = 0.5       # count 50% of free swap toward cache+working
CACHE_ALLOWANCE_KIB = 3200 * 1024       # 3200 MiB target cache
MTP_CACHE_KIB = 87 * 1024               # ~86.875 MiB MTP cache
WORKING_ALLOWANCE_KIB = 2560 * 1024     # 2560 MiB non-cache working
SWAP_HARD_FLOOR_KIB = 512 * 1024        # 512 MiB swap free is a hard stop
PROJECTION_HORIZON_S = 5.0   # project availability forward this many seconds
PROJECTION_MIN_SAMPLES = 3   # need >=3 recent samples to fit a slope
GEN_WATCHDOG_S = 720         # 12 minute hard generation watchdog
TERM_SIGTERM_WAIT_S = 3.0    # short process-group termination: SIGTERM then SIGKILL
BATTERY_REFRESH_S = 5.0      # refresh cached battery on this cadence (off the fast path)

LOCK_FILE = HERE / '.run.lock'
SCREEN_PREREQ = 'screen.passed.json'           # prerequisite evidence marker

# Thermal hard / soft limits (unchanged from original harness contract).
HARD_B = 45
HARD_C = 100
HARD_HTP = 85
SOFT_B, SOFT_C, SOFT_HTP = 40, 55, 50
COOL_MIN_S, COOL_MAX_S = 180, 300


# --------------------------------------------------------------------------- #
# Telemetry sources (fast + decoupled).                                       #
# --------------------------------------------------------------------------- #
def mem_kib() -> int:
    """Fast MemAvailable (KiB) read from /proc/meminfo.  No subprocess."""
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise RuntimeError('MemAvailable unavailable')


def swap_kib() -> int:
    """SwapFree (KiB) read from /proc/meminfo.  No subprocess."""
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('SwapFree:'):
            return int(line.split()[1])
    return 0


def _parse_status(text: str) -> dict:
    """Parse the VmRSS / VmSwap / Threads fields from a /proc/<pid>/status blob."""
    rss_kib = pss_kib = swp_kib = threads = None
    for line in text.splitlines():
        if line.startswith('VmRSS:'):
            rss_kib = int(line.split()[1])
        elif line.startswith('VmSwap:'):
            swp_kib = int(line.split()[1])
        elif line.startswith('Threads:'):
            threads = int(line.split()[1])
    return dict(rss_kib=rss_kib, pss_kib=pss_kib, swap_kib=swp_kib, threads=threads)


def _parse_smaps_rollup(text: str) -> int | None:
    pss_kib = None
    for line in text.splitlines():
        if line.startswith('Pss:'):
            pss_kib = int(line.split()[1])
    return pss_kib


def child_telemetry(pid: int) -> dict | None:
    """Read child RSS / PSS / swap / threads from /proc/<pid>.

    Returns None if the process is gone; PSS may be absent on kernels without
    smaps_rollup.  Never raises -- telemetry is best-effort.
    """
    if pid <= 0:
        return None
    try:
        status = (Path('/proc') / str(pid) / 'status').read_text()
    except OSError:
        return None
    info = _parse_status(status)
    try:
        info['pss_kib'] = _parse_smaps_rollup(
            (Path('/proc') / str(pid) / 'smaps_rollup').read_text())
    except OSError:
        info['pss_kib'] = None
    return info


class BatteryCache:
    """Battery is sampled via termux-battery-status (a 5s-timeout subprocess).

    It is intentionally taken OFF the fast memory-sampling path: ``sample_memory``
    only ever reads the cached value.  The cache is populated on a background
    thread at most every BATTERY_REFRESH_S so a slow/hung battery query can never
    stall the 1 s memory watchdog tick.
    """
    def __init__(self, refresh_s: float = BATTERY_REFRESH_S):
        self.refresh_s = refresh_s
        self._level: int | None = None
        self._charging: bool | None = None
        self._last: float = 0.0
        self._inflight = False
        self._lock = threading.Lock()

    def _fetch(self):
        result = subprocess.run(['termux-battery-status'], capture_output=True,
                                text=True, timeout=5, check=True)
        data = json.loads(result.stdout)
        level = data['percentage']; status = data['status']
        if type(level) is not int or not 0 <= level <= 100 or not isinstance(status, str):
            raise ValueError('invalid battery fields')
        status = status.strip().lower()
        if status not in ('charging', 'full', 'discharging', 'not charging'):
            raise ValueError('unknown battery status')
        with self._lock:
            self._level = level
            self._charging = status in ('charging', 'full')
            self._last = time.monotonic()

    def _fetch_wrap(self):
        try:
            self._fetch()
        except Exception:
            with self._lock:
                self._last = time.monotonic()   # back off, do not spin
        finally:
            self._inflight = False

    def refresh(self):
        """Non-blocking: spawns a background thread to refresh if due."""
        if self._level is None or (time.monotonic() - self._last) >= self.refresh_s:
            if not self._inflight:
                self._inflight = True
                threading.Thread(target=self._fetch_wrap, daemon=True).start()

    def value(self):
        """Return (level, charging) -- never blocks, never subprocess."""
        with self._lock:
            return self._level, self._charging


def temperatures() -> dict:
    values: dict[str, float] = {}
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


def sample_memory(battery: BatteryCache | None = None) -> dict:
    """FAST, battery-independent telemetry tick.

    Reads /proc/meminfo + thermal only.  Battery (if a cache is supplied) is read
    from the non-blocking cache; no subprocess runs here.
    """
    level, charging = (battery.value() if battery else (None, None))
    return dict(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
                available_kib=mem_kib(), swap_kib=swap_kib(),
                capacity=level, charging=charging, thermal=temperatures())


# --------------------------------------------------------------------------- #
# Admission / stop decisions.                                                 #
# --------------------------------------------------------------------------- #
def required_available_kib(swap_kib: int = 0, mtp_on: bool = True) -> int:
    """Byte-accurate admission requirement, factoring in free swap.

    5 GiB is a minimum ELIGIBILITY floor (not sufficient alone).  The 2 GiB
    retained reserve is a hard floor on MemAvailable.  Cache + working
    allowances (3200 + 87 + 2560 MiB) are backstopped by free swap at
    ``SWAP_BACKSTOP_RATIO`` effectiveness; if swap is insufficient to backstop,
    the shortfall is required from MemAvailable.

    Without swap this reproduces the architect-6 conservative values
    (7808 MiB off / 7895 MiB on).  With sufficient swap it relaxes to the
    5 GiB floor.  Final thresholds await revised architect policy.
    """
    cache_working = CACHE_ALLOWANCE_KIB + (MTP_CACHE_KIB if mtp_on else 0) + WORKING_ALLOWANCE_KIB
    swap_credit = int(swap_kib * SWAP_BACKSTOP_RATIO)
    mem_for_cache = max(0, cache_working - swap_credit)
    return max(ADMISSION_MIN_AVAIL_KIB, RETAINED_HEADROOM_KIB + mem_for_cache)


def admission(row: dict, screen_passed: bool = True) -> str | None:
    """Return an abort reason string, or None if admitted.

    Enforces: 5 GiB MemAvailable eligibility floor, swap-aware byte-accurate
    admission threshold, charging + battery>=40%, thermal sensors present,
    and the screen prerequisite.
    """
    if not screen_passed:
        return 'screen prerequisite not satisfied'
    if row['capacity'] is None or row['charging'] is None:
        return 'battery status unavailable'
    if not row['charging'] or row['capacity'] < BATTERY_MIN:
        return 'charging and battery >=40% required'
    swap_kib = row.get('swap_kib', 0)
    required = required_available_kib(swap_kib, mtp_on=True)
    if row['available_kib'] < required:
        return (f'MemAvailable {row["available_kib"]/1024/1024:.2f} GiB below '
                f'{required/1024/1024:.2f} GiB swap-aware admission minimum '
                f'(5 GiB floor + 2 GiB reserve + cache/working backstopped by '
                f'{swap_kib/1024/1024:.2f} GiB free swap)')
    t = row['thermal']
    if any(t.get(k) is None for k in ('b', 'c', 'htp')):
        return 'thermal sensors unavailable'
    return None


def declining_memory_projection(history, floor_kib=RETAINED_HEADROOM_KIB,
                                horizon_s=PROJECTION_HORIZON_S,
                                min_samples=PROJECTION_MIN_SAMPLES):
    """Least-squares projection of MemAvailable onto the near future.

    ``history`` is a list of ``(monotonic, available_kib)`` tuples.  If the most
    recent ``min_samples`` points show a negative slope and the projection
    ``latest + slope*horizon`` falls below ``floor_kib``, return a reason dict
    (the run should be aborted NOW, before the floor is reached).  Otherwise
    return None.
    """
    if len(history) < min_samples:
        return None
    recent = history[-min_samples:]
    n = len(recent)
    xs = [h[0] for h in recent]
    ys = [h[1] for h in recent]
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx                            # KiB/s
    if slope >= 0:
        return None
    projected = ys[-1] + slope * horizon_s
    if projected < floor_kib:
        return dict(reason='declining-memory projection below 2 GiB retained-headroom',
                    slope_kib_s=slope, projected_kib=projected, horizon_s=horizon_s,
                    floor_kib=floor_kib,
                    floor_gib=round(floor_kib / 1024 / 1024, 3),
                    current_kib=ys[-1], window_samples=n)
    return None


def declining_swap_projection(history, floor_kib=SWAP_HARD_FLOOR_KIB,
                              horizon_s=PROJECTION_HORIZON_S,
                              min_samples=PROJECTION_MIN_SAMPLES):
    """Least-squares projection of SwapFree onto the near future.

    Same algorithm as :func:`declining_memory_projection` but for swap.  If the
    most recent ``min_samples`` points show a negative slope and the projection
    ``latest + slope*horizon`` falls below ``floor_kib`` (512 MiB), return a
    reason dict (the run should be aborted NOW, before the floor is reached).
    Otherwise return None.
    """
    if len(history) < min_samples:
        return None
    recent = history[-min_samples:]
    n = len(recent)
    xs = [h[0] for h in recent]
    ys = [h[1] for h in recent]
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx                            # KiB/s
    if slope >= 0:
        return None
    projected = ys[-1] + slope * horizon_s
    if projected < floor_kib:
        return dict(reason='declining-swap projection below 512 MiB emergency floor',
                    slope_kib_s=slope, projected_kib=projected, horizon_s=horizon_s,
                    floor_kib=floor_kib,
                    floor_mib=round(floor_kib / 1024, 3),
                    current_kib=ys[-1], window_samples=n)
    return None


def hard_stop(row: dict) -> str | None:
    """Immediate, non-projectable abort conditions (thermal + 2 GiB floor + swap)."""
    t = row['thermal']
    if any(t.get(k) is None for k in ('b', 'c', 'htp')):
        return 'thermal sensor unavailable'
    if t['b'] >= HARD_B or t['c'] >= HARD_C or t['htp'] >= HARD_HTP:
        return 'hard thermal threshold'
    if row['available_kib'] < RETAINED_HEADROOM_KIB:
        return 'MemAvailable below 2 GiB retained-headroom floor'
    if row.get('swap_kib', 0) < SWAP_HARD_FLOOR_KIB:
        return 'SwapFree below 512 MiB emergency floor'
    return None


# --------------------------------------------------------------------------- #
# Plan / command / environment (the root-cause fix).                           #
# --------------------------------------------------------------------------- #
def plan(stage: str) -> list[dict]:
    if stage == 'qualification':
        cells = [(4, 4, m) for m in ('off', 'on')] + [(1, 6, m) for m in ('off', 'on')]
    elif stage == 'screen':
        cells = [(h, t, m) for t in (4, 5, 6) for h in (4, 3, 2, 1) for m in ('off', 'on')]
    else:
        raise ValueError(stage)
    return [dict(h=h, threads=t, mtp=m,
                 tokens=16 if stage == 'qualification' else 64) for h, t, m in cells]


def command(cell: dict) -> list[str]:
    """Build the llama-cli invocation with EVERY inference flag explicitly pinned.

    This is the concrete root-cause fix: the original harness left -lm / -dev /
    -ngl / -c / -b / -ub / -np / -fa to defaults, so LLAMA_LOAD_MODE_AUTO ran and
    hexagon's mmap_support=false disabled mmap globally -> eager 8.4 GiB load.
    Here mmap is pinned ON, the device, layer count, context, batch and flash
    attention are all explicit.
    """
    spec = (['--spec-type', 'draft-mtp', '--spec-draft-n-max', '2',
             '--spec-draft-n-min', '0', '--spec-draft-p-min', '0']
            if cell['mtp'] == 'on' else ['--spec-type', 'none'])
    return [str(BIN), '-m', str(MODEL), '-p', PROMPT, '-n', str(cell['tokens']),
            '-t', str(cell['threads']), '-b', str(BATCH), '-ub', str(MICROBATCH),
            '-c', str(CTX), '-ngl', str(OFFLOAD_LAYERS),
            '-dev', DEVICE, '-lm', MMAP_MODE, '-fa', FLASH, '-np', str(N_SEQ),
            '--no-display-prompt', *spec]


# Staleness markers scrubbed from the inherited environment.
_SCRUB_PREFIXES = ('LLAMA_ARG_', 'GGML_', 'MOE_')
_SCRUB_EXACT = ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'ADSP_LIBRARY_PATH',
                'LD_PRELOAD_PATH', 'GGS_CUBLAS', 'GGML_CUDA_', 'CUDA_VISIBLE_DEVICES')
_KEEP = ('PATH', 'HOME', 'TERM', 'ANDROID_ROOT', 'ANDROID_DATA',
         'EXTERNAL_STORAGE', 'SECONDARY_STORAGE', 'ANDROID_STORAGE_ROOT')


def environment(cell: dict, source=None) -> dict:
    """Construct a CLEAN child environment.

    Starts from a minimal, explicit base (NOT a copy of the ambient Termux
    environment) so no stale ``LLAMA_ARG_*`` / ``GGML_*`` / ``MOE_*`` / loader
    vars from a prior run can leak in.  Only the pinned vars below are injected.
    """
    source = os.environ if source is None else source
    env = {k: v for k, v in source.items() if k in _KEEP}
    # Defensive double-scrub in case _KEEP logic changes: strip any staleness.
    for k in list(env):
        if k.startswith(_SCRUB_PREFIXES) or k in _SCRUB_EXACT:
            env.pop(k, None)
    env.update({
        'GGML_HEXAGON_NHVX': str(cell['h']),
        'GGML_HEXAGON_NHMX': '1',
        'GGML_CPU_MOE_CACHE_SLOTS': '80',     # intentional, part of this factorial
        'GGML_CPU_MOE_CACHE_MIB': '3200',
        'LD_LIBRARY_PATH': str(BIN.parent),
        'ADSP_LIBRARY_PATH': str(ROOT / '.work-npu/build/ggml/src/ggml-hexagon'),
        'MOI_DISABLE_AUTO_PLACEMENT': '1',     # explicit: no auto placement / fitting
    })
    return env


def scrubbed_env_is_clean(source: dict) -> list[str]:
    """Return list of stale inference env vars present in ``source`` (screen check)."""
    return [k for k in source if k.startswith(_SCRUB_PREFIXES) or k in _SCRUB_EXACT]


# --------------------------------------------------------------------------- #
# I/O helpers.                                                                  #
# --------------------------------------------------------------------------- #
def append(path: Path, record: dict) -> None:
    with path.open('a') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()
        os.fsync(f.fileno())


def cool(telemetry: Path, lifecycle: Path, battery: BatteryCache | None = None) -> None:
    start = time.monotonic()
    while True:
        row = sample_memory(battery)
        if battery is not None:
            battery.refresh()
        append(telemetry, dict(phase='cooling', **row))
        reason = hard_stop(row)
        if reason:
            raise RuntimeError(reason)
        if admission(row):
            raise RuntimeError(admission(row))
        elapsed = time.monotonic() - start
        soft = all(row['thermal'][k] <= lim for k, lim in
                   [('b', SOFT_B), ('c', SOFT_C), ('htp', SOFT_HTP)])
        if elapsed >= COOL_MIN_S and soft:
            append(lifecycle, dict(event='cool_complete', elapsed_s=elapsed, soft=True))
            return
        if elapsed >= COOL_MAX_S:
            raise RuntimeError('cooling maximum 300 seconds exceeded without soft targets')
        time.sleep(min(1, max(0.01, COOL_MAX_S - elapsed)))


def terminate_child(child, timeout: float = TERM_SIGTERM_WAIT_S) -> bool:
    """Short process-group termination: SIGTERM the group, wait ``timeout`` s,
    then SIGKILL the whole group.  Returns True if the child had to be signalled.
    """
    if child.poll() is not None:
        return False
    signalled = False
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        signalled = True
    except (ProcessLookupError, PermissionError):
        try:
            child.terminate()
            signalled = True
        except ProcessLookupError:
            return False
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            child.kill()
        child.wait()
    return signalled


# --------------------------------------------------------------------------- #
# Single-cell generation with projection abort + child telemetry.             #
# --------------------------------------------------------------------------- #
def run_cell(cell: dict, index: int, evidence: Path, spawn, battery: BatteryCache,
             telemetry: Path, lifecycle: Path,
             watchdog_s: float = GEN_WATCHDOG_S,
             sampler=sample_memory, child_telem=child_telemetry) -> dict:
    """Execute one cell.  ``spawn(cell, cmd)`` returns a child handle with
    .pid/.poll/.wait/.returncode/.terminate/.kill.

    ``sampler(battery)`` returns a telemetry row (fast, battery-independent).
    ``child_telem(pid)`` returns child RSS/PSS/swap (injectable for tests).

    A declining-memory OR declining-swap projection can abort the cell EARLY,
    before the 2 GiB retained-headroom floor or 512 MiB swap floor is reached,
    and the child process group is torn down promptly (short process-group
    termination).
    """
    cmd = command(cell)
    append(lifecycle, dict(event='start', index=index, cell=cell, command=cmd,
                           utc=datetime.now(timezone.utc).isoformat()))
    child = spawn(cell, cmd)
    pid = child.pid
    started = time.monotonic()
    history: list[tuple[float, int]] = []
    swap_history: list[tuple[float, int]] = []
    reason: str | None = None
    proj_detail: dict | None = None
    try:
        while child.poll() is None:
            row = sampler(battery)                 # fast, never blocks on battery
            row['child'] = child_telem(pid)        # child RSS/PSS/swap telemetry
            append(telemetry, dict(phase='generation', index=index, pid=pid, **row))
            history.append((row['monotonic'], row['available_kib']))
            swap_history.append((row['monotonic'], row.get('swap_kib', 0)))
            if battery is not None:
                battery.refresh()

            proj = declining_memory_projection(history)
            if proj:
                reason = proj['reason']
                proj_detail = proj
                break
            swap_proj = declining_swap_projection(swap_history)
            if swap_proj:
                reason = swap_proj['reason']
                proj_detail = swap_proj
                break
            hs = hard_stop(row)
            if hs:
                reason = hs
                break
            if row['charging'] is False:
                reason = 'charging lost'
                break
            if row['capacity'] is None and row['charging'] is None:
                reason = 'battery status unavailable (lost)'
                break
            if time.monotonic() - started >= watchdog_s:
                reason = 'generation watchdog'
                break
            time.sleep(1)
    finally:
        signalled = terminate_child(child)        # short process-group termination
        if signalled and not reason:
            reason = 'signal'
    # A child that died on its own with a nonzero exit counts as an abort.
    rc = child.returncode
    if not reason and rc is not None and rc != 0:
        reason = f'cell exited {rc}'
    result = dict(index=index, cell=cell, command=cmd,
                  returncode=child.returncode, duration_s=time.monotonic() - started,
                  stop_reason=reason, pid=pid,
                  aborted_by_memory_projection=(reason and 'memory' in reason and 'projection' in reason),
                  aborted_by_swap_projection=(reason and 'swap' in reason and 'projection' in reason))
    if proj_detail:
        result['projection'] = proj_detail
    event = 'finish' if not reason else 'abort'
    append(lifecycle, dict(event=event, **result))
    append(lifecycle, dict(event='cell_' + event, index=index, stop_reason=reason,
                           projected=proj_detail is not None))
    return result


# --------------------------------------------------------------------------- #
# Screen prerequisite (non-inference validation).                              #
# --------------------------------------------------------------------------- #
def screen_prereq(evidence_dir: Path) -> dict:
    """Run the non-inference prerequisite gate WITHOUT launching any model.

    Validates: binary/model presence, /proc + thermal readability, the 5 GiB
    eligibility floor + swap-aware admission gate, thermal sensors, that the
    ambient environment carries no stale ``LLAMA_ARG_*`` / ``GGML_*`` /
    ``MOE_*`` vars, and that the projection unit behaves (declining slope is
    detected, flat slope is not).  Writes a ``screen.passed.json`` record to
    ``evidence_dir``; ``main()`` additionally writes the canonical
    ``HERE/SCREEN_OK`` marker ONLY when passed.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    record = dict(utc=datetime.now(timezone.utc).isoformat(), checks={})
    checks = {'bin': BIN.is_file(), 'model': MODEL.is_file(),
              'meminfo': os.path.exists('/proc/meminfo'),
              'thermal': os.path.exists('/sys/class/thermal')}
    battery = BatteryCache()
    battery._fetch_wrap()
    row = sample_memory(battery)
    checks['available_gib'] = round(row['available_kib'] / 1024 / 1024, 2)
    checks['swap_free_gib'] = round(row['swap_kib'] / 1024 / 1024, 2)
    checks['sensors_present'] = all(row['thermal'].get(k) is not None for k in ('b', 'c', 'htp'))
    checks['stale_env'] = scrubbed_env_is_clean(os.environ)
    checks['pinned_command'] = command(dict(h=4, threads=4, mtp='off', tokens=16))
    checks['required_available_gib'] = round(
        required_available_kib(row['swap_kib'], mtp_on=True) / 1024 / 1024, 2)
    checks['projection_unit'] = (
        declining_memory_projection([(0, 4500000), (1, 4400000), (2, 4300000), (3, 4200000)]) is None
        and declining_memory_projection([(0, 4500000), (2, 4400000), (4, 3000000)]) is not None)
    checks['swap_projection_unit'] = (
        declining_swap_projection([(0, 2000000), (1, 1900000), (2, 1800000), (3, 1700000)]) is None
        and declining_swap_projection([(0, 2000000), (2, 1900000), (4, 500000)]) is not None)
    record['checks'] = checks
    record['admission'] = admission(row)
    record['passed'] = (checks['bin'] and checks['model'] and checks['meminfo']
                        and checks['thermal'] and checks['sensors_present']
                        and not checks['stale_env'] and record['admission'] is None)
    (evidence_dir / SCREEN_PREREQ).write_text(json.dumps(record, indent=2) + '\n')
    return record


# --------------------------------------------------------------------------- #
# Single-run lock + signal-driven lifecycle.                                  #
# --------------------------------------------------------------------------- #
class RunLock:
    def __init__(self, path: Path = LOCK_FILE):
        self.path = path
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, 'w')
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another safe-factorial-harness run is in progress '
                               '(single-run lock held)') from exc

    def release(self) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def install_signal_handlers(lifecycle: Path) -> None:
    """On SIGTERM/SIGINT, record a 'signal_abort' lifecycle event before exit.

    Mitigates the retry-01 gap (driver died with NO lifecycle record at all).
    Note: SIGKILL cannot be caught -- any run killed by SIGKILL/OOM-killer will
    leave no in-memory record beyond what fsync'd telemetry already holds.
    """
    def handler(signum, _frame):
        try:
            append(lifecycle, dict(event='signal_abort', signal=signum,
                                   utc=datetime.now(timezone.utc).isoformat()))
        except Exception:
            pass
        raise SystemExit(f'received signal {signum}')
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #
def spawn_real(cell: dict, cmd: list[str]):
    return subprocess.Popen(cmd, env=environment(cell),
                            start_new_session=True)   # own process group


def run(stage: str, evidence: str | None = None, run_flag: bool = False,
        preflight_only: bool = False, screen_passed: bool = False,
        spawn=spawn_real, battery: BatteryCache | None = None) -> None:
    """Orchestrate a stage.  Default (run_flag=False) is a dry-run: it writes the
    pinned plan/commands to the evidence dir and NEVER launches inference.

    ``--run`` (inference) is only permitted when ``screen_passed`` is True.
    ``preflight_only`` records the admission snapshot without inference.
    """
    if stage not in ('qualification', 'screen'):
        raise ValueError("stage must be 'qualification' (4 cells) or 'screen' (24 cells); "
                         "use screen_prereq() for the non-inference prerequisite gate")
    out = HERE / (evidence or ('safe-' + stage))
    if out.parent != HERE or out.name in ('', '.', '..'):
        raise RuntimeError('evidence must be a directory name')
    if out.exists() and not (out.is_dir() and
            sorted(p.name for p in out.iterdir()) == ['RESERVED']):
        raise RuntimeError(f'{out} exists; never overwrite evidence')
    if run_flag and not screen_passed:
        raise RuntimeError('--run (inference) requires the screen prerequisite: '
                           'run `--stage screen` first')
    lock = RunLock()
    lock.acquire()
    try:
        reserved = out.is_dir() and sorted(p.name for p in out.iterdir()) == ['RESERVED']
        if reserved:
            (out / 'RESERVED').unlink()
        else:
            out.mkdir()
        lifecycle, telemetry = out / 'lifecycle.jsonl', out / 'telemetry.jsonl'
        install_signal_handlers(lifecycle)
        if battery is None:
            battery = BatteryCache()
        initial = sample_memory(battery)
        append(lifecycle, dict(event='preflight', **initial,
                               pins=dict(mmap=MMAP_MODE, device=DEVICE, ngl=OFFLOAD_LAYERS,
                                         ctx=CTX, batch=BATCH, microbatch=MICROBATCH,
                                         n_seq=N_SEQ, flash=FLASH)))
        adm = admission(initial)
        append(lifecycle, dict(event='preflight_admission', reason=adm,
                               available_gib=round(initial['available_kib']/1024/1024, 2)))
        # Dry-run only prints/validates; it must not hard-fail on a momentary
        # memory reading.  --preflight-only and --run enforce the 5 GiB gate.
        if adm and (preflight_only or run_flag):
            raise RuntimeError(adm)
        if preflight_only:
            append(lifecycle, dict(event='preflight_only_complete', admission=adm))
            (out / 'summary.json').write_text('{"preflight_only": true}\n')
            return
        if not run_flag:
            # DRY RUN: never launch inference.
            plan_data = dict(dry_run=True, stage=stage, cells=plan(stage),
                             commands=[command(c) for c in plan(stage)],
                             pins=dict(mmap=MMAP_MODE, device=DEVICE, ngl=OFFLOAD_LAYERS,
                                       ctx=CTX, batch=BATCH, microbatch=MICROBATCH,
                                       n_seq=N_SEQ, flash=FLASH),
                             cooling_s=[COOL_MIN_S, COOL_MAX_S],
                             projection=dict(horizon_s=PROJECTION_HORIZON_S,
                                             min_samples=PROJECTION_MIN_SAMPLES,
                                             floor_gib=RETAINED_HEADROOM_KIB / 1024 / 1024))
            (out / 'plan.json').write_text(json.dumps(plan_data, indent=2) + '\n')
            append(lifecycle, dict(event='dry_run_complete'))
            return
        # ---- INFERENCE PATH (only reached with --run + screen_passed) ----
        (out / 'plan.json').write_text(json.dumps(plan(stage), indent=2) + '\n')
        results = []
        try:
            for index, cell in enumerate(plan(stage)):
                cool(telemetry, lifecycle, battery)
                row = sample_memory(battery)
                battery.refresh()
                append(telemetry, dict(phase='pre_cell', index=index, **row))
                if admission(row):
                    raise RuntimeError(admission(row))
                res = run_cell(cell, index, out, spawn, battery,
                               telemetry, lifecycle)
                results.append(res)
                append(out / 'results.jsonl', res)
                if res['stop_reason'] or res['returncode'] != 0:
                    raise RuntimeError(res['stop_reason'] or
                                       f'cell {index} exited {res["returncode"]}')
        except Exception as exc:
            append(lifecycle, dict(event='abort', error=str(exc),
                                   results_so_far=len(results)))
            raise
        finally:
            (out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            append(lifecycle, dict(event='summary_written', results=len(results)))
    finally:
        lock.release()


def screen_marker() -> bool:
    return (HERE / 'SCREEN_OK').exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('qualification', 'screen'),
                        default='qualification',
                        help='qualification = 4-cell HVX/MTP factorial (16 tokens); '
                             'screen = 24-cell thread-major sweep (64 tokens)')
    parser.add_argument('--evidence', help='new evidence directory name (never overwrite)')
    parser.add_argument('--preflight-only', action='store_true',
                        help='record the admission snapshot only; no inference, no --run needed')
    parser.add_argument('--run', action='store_true',
                        help='LAUNCH inference for the stage (REQUIRES --screen-prereq to have '
                             'passed; dry-run is the default and prints only)')
    parser.add_argument('--screen-prereq', action='store_true',
                        help='run the non-inference prerequisite gate (writes SCREEN_OK iff passed)')
    parser.add_argument('--screen-only', action='store_true',
                        help='print the raw screen record without writing any marker')
    args = parser.parse_args()

    # ---- Non-inference prerequisite gate -------------------------------- #
    if args.screen_prereq or args.screen_only:
        if args.run or args.preflight_only:
            parser.error('--screen-prereq/--screen-only takes no other inference flags')
        out = HERE / (args.evidence or 'screen-prereq')
        if out.exists() and not (out.is_dir() and
                sorted(p.name for p in out.iterdir()) == ['RESERVED']):
            raise RuntimeError(f'{out} exists; never overwrite evidence')
        out.mkdir(parents=True, exist_ok=True)
        rec = screen_prereq(out)
        if not args.screen_only:
            if rec['passed']:
                (HERE / 'SCREEN_OK').write_text('screen passed ' + rec['utc'] + '\n')
            else:
                # A failed screen must NOT leave a stale authorization marker.
                try:
                    (HERE / 'SCREEN_OK').unlink()
                except FileNotFoundError:
                    pass
        print(json.dumps(rec, indent=2))
        return

    if args.preflight_only and args.run:
        parser.error('--preflight-only and --run are mutually exclusive')
    if args.run and not screen_marker():
        parser.error('--run requires the screen prerequisite: run `python safe_run.py --screen-prereq` first')

    run(args.stage, args.evidence, run_flag=args.run,
        preflight_only=args.preflight_only, screen_passed=screen_marker())


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
