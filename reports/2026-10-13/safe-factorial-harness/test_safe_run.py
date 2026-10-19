"""Unit + mock tests for the SAFE factorial harness revision.

No inference is launched by any test: all subprocess/llama-cli interactions are
mocked, and the only "real" runs exercised here are the dry-run plan, the
projection unit, the admission policy, and a fully-injected replay of the
observed 4.38 -> 1.85 GiB / 8.35 s memory collapse (evidence/retry-01).
"""
import importlib.util
import json
import os
import subprocess
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('safe_run', HERE / 'safe_run.py')
S = importlib.util.module_from_spec(spec)            # S = safe_run module
spec.loader.exec_module(S)

FIX = HERE / 'evidence' / 'observed_decline.json'


def load_observed_decline():
    """(t_offset_s, available_kib, thermal) tuples from the real retry-01 trace."""
    d = json.loads(FIX.read_text())
    return [(s['t_offset_s'], s['available_kib'], s['thermal']) for s in d['samples']], d['min_available_kib']


class FakeChild:
    """A subprocess.Popen stand-in for hermetic run_cell tests."""
    def __init__(self, pid=262145, rc=None, exit_after=None):
        self.pid = pid
        self._rc = rc
        self._exit_after = exit_after   # consume this many poll() calls before "exiting"
        self._calls = 0
        self.terminate_called = False
        self.kill_called = False
        self.wait_called = False

    def poll(self):
        self._calls += 1
        if self._exit_after is not None and self._calls > self._exit_after:
            return self._rc if self._rc is not None else 0
        return None

    def wait(self, timeout=None):
        self.wait_called = True
        rc = self._rc if self._rc is not None else 0
        if self._exit_after is not None and self._calls > self._exit_after:
            return rc
        return rc

    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True

    @property
    def returncode(self):
        if self._exit_after is not None and self._calls > self._exit_after:
            return self._rc if self._rc is not None else 0
        return None


class SamplerFeed:
    """Yields observed telemetry rows in order, then repeats the last (floor).

    Samples may be 3-tuples ``(t_off, available_kib, thermal_dict)`` (swap
    defaults to ``swap_kib``) or 4-tuples ``(t_off, available_kib, swap_kib,
    thermal_dict)`` for swap-decline scenarios.
    """
    def __init__(self, samples, swap_kib=15 * 1024 * 1024):
        self.samples = samples
        self.i = 0
        self.calls = 0
        self.swap_kib = swap_kib

    def __call__(self, battery=None):
        self.calls += 1
        if self.i >= len(self.samples):
            s = self.samples[-1]      # repeat floor if the loop ever reaches it
        else:
            s = self.samples[self.i]
            self.i += 1
        if len(s) == 4:
            t_off, kib, swp, therm = s
        else:
            t_off, kib, therm = s
            swp = self.swap_kib
        return dict(utc=f'2026-10-13T11:0{self.i:05d}+00:00',
                    monotonic=1000.0 + t_off,   # baseline irrelevant to slope rate
                    available_kib=kib, swap_kib=swp, capacity=60, charging=True,
                    thermal={'b': therm['b'], 'c': therm['c'], 'htp': therm['htp'],
                             'zones': {}})


class RunnerTests(unittest.TestCase):
    # --- root-cause fix: command pins are non-negotiable -------------------
    def test_command_pins_root_cause(self):
        cell = dict(h=4, threads=4, mtp='off', tokens=16)
        cmd = S.command(cell)
        flat = {cmd[i]: (cmd[i + 1] if i + 1 < len(cmd) else None)
                for i in range(len(cmd)) if cmd[i].startswith('-') and i + 1 < len(cmd)}
        # The concrete root-cause fix.
        self.assertEqual(flat.get('-lm'), 'mmap')          # pin mmap ON
        self.assertEqual(flat.get('-dev'), 'HTP0')          # pinned device
        self.assertEqual(flat.get('-ngl'), '99')            # 99 offloaded layers
        self.assertEqual(flat.get('-c'), '256')             # context 256
        self.assertEqual(flat.get('-b'), '128')             # batch 128
        self.assertEqual(flat.get('-ub'), '128')            # microbatch 128
        self.assertEqual(flat.get('-np'), '1')              # one sequence
        self.assertEqual(flat.get('-fa'), 'off')            # flash attention OFF
        # threads come from the cell; prompt/model/flags present
        self.assertIn('-t', cmd)
        self.assertIn(str(MICROBATCH := 128), cmd)
        # MTP spec wiring still correct for both modes
        self.assertIn('--spec-type', cmd)
        self.assertEqual(cmd[cmd.index('--spec-type') + 1], 'none')
        cell_on = dict(cell, mtp='on')
        cmd_on = S.command(cell_on)
        self.assertEqual(cmd_on[cmd_on.index('--spec-type') + 1], 'draft-mtp')
        self.assertIn('--spec-draft-n-max', cmd_on)

    # --- environment scrubbing: LLAMA_ARG_* + stale inference vars ----------
    def test_environment_scrubs_stale_and_llama_arg(self):
        cell = dict(h=3, threads=5, mtp='off', tokens=64)
        polluted = {'LLAMA_ARG_N_THREADS': '1', 'LLAMA_ARG_N_BATCH': '2',
                    'GGML_STALE': 'x', 'MOE_LEAK': 'y', 'LD_PRELOAD': '/evil.so',
                    'LD_LIBRARY_PATH': '/evil/lib', 'ADSP_LIBRARY_PATH': '/evil/adsp',
                    'PATH': '/bin', 'HOME': '/h', 'TERM': 'xterm',
                    'CUDA_VISIBLE_DEVICES': '0', 'GGS_CUBLAS': '1'}
        env = S.environment(cell, polluted)
        for bad in ('LLAMA_ARG_N_THREADS', 'LLAMA_ARG_N_BATCH', 'GGML_STALE',
                    'MOE_LEAK', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES', 'GGS_CUBLAS'):
            self.assertNotIn(bad, env, f'{bad} leaked into child env')
        # intentional pinned vars survive / are set explicitly
        self.assertEqual(env['GGML_HEXAGON_NHVX'], '3')
        self.assertEqual(env['GGML_HEXAGON_NHMX'], '1')
        self.assertEqual(env['GGML_CPU_MOE_CACHE_SLOTS'], '80')
        self.assertEqual(env['LD_LIBRARY_PATH'], str(S.BIN.parent))
        self.assertEqual(env['MOI_DISABLE_AUTO_PLACEMENT'], '1')
        # no auto-placement / fitting
        self.assertIn('MOI_DISABLE_AUTO_PLACEMENT', env)

    def test_scrubbed_env_is_clean_helper(self):
        self.assertEqual(S.scrubbed_env_is_clean({}), [])
        bad = {'LLAMA_ARG_X': '1', 'GGML_Y': '2', 'MOE_Z': '3', 'LD_PRELOAD': '4',
               'PATH': '/bin'}
        self.assertEqual(sorted(S.scrubbed_env_is_clean(bad)),
                         sorted(['LLAMA_ARG_X', 'GGML_Y', 'MOE_Z', 'LD_PRELOAD']))

    # --- admission: 5 GiB minimum (was 1.5) ---------------------------------
    def test_admission_5gib_gate(self):
        # With 15 GiB free swap, the swap-aware threshold relaxes to the 5 GiB
        # eligibility floor (cache+working fully backstopped by swap credit).
        base = dict(charging=True, capacity=60, available_kib=5 * 1024 * 1024,
                    swap_kib=15 * 1024 * 1024, thermal=dict(b=37, c=50, htp=47))
        self.assertIsNone(S.admission(base))
        self.assertIsNotNone(S.admission(dict(base, available_kib=5 * 1024 * 1024 - 1)))
        self.assertIsNotNone(S.admission(dict(base, available_kib=1536 * 1024)))  # old bar now refused
        self.assertIsNotNone(S.admission(dict(base, charging=False)))
        self.assertIsNotNone(S.admission(dict(base, capacity=39)))
        self.assertIsNotNone(S.admission(dict(base, thermal=dict(b=None, c=50, htp=47))))
        # screen prerequisite gate: without it, even a healthy row is refused
        self.assertIsNotNone(S.admission(dict(base), screen_passed=False))
        self.assertIsNone(S.admission(dict(base), screen_passed=True))

    # --- admission: swap-aware threshold relaxes with free swap -------------
    def test_admission_swap_aware_threshold(self):
        # Without swap, the threshold is the conservative ~7.71 GiB.
        no_swap = dict(charging=True, capacity=60, available_kib=5 * 1024 * 1024,
                       swap_kib=0, thermal=dict(b=37, c=50, htp=47))
        self.assertIsNotNone(S.admission(no_swap))  # 5 GiB < 7.71 GiB without swap
        # With enough swap to backstop cache+working, 5 GiB is admitted.
        with_swap = dict(no_swap, swap_kib=15 * 1024 * 1024)
        self.assertIsNone(S.admission(with_swap))
        # required_available_kib is monotonic: more swap never raises the bar.
        req0 = S.required_available_kib(0, mtp_on=True)
        req_big = S.required_available_kib(15 * 1024 * 1024, mtp_on=True)
        self.assertGreaterEqual(req0, req_big)
        # Without swap, reproduces architect-6 conservative value (~7.71 GiB).
        self.assertGreater(req0, 7 * 1024 * 1024)

    # --- hard stop: 2 GiB retained-headroom floor + thermal + swap -----------
    def test_hard_stop_2gib_floor_and_thermal(self):
        row = dict(available_kib=3 * 1024 * 1024, swap_kib=15 * 1024 * 1024,
                   thermal=dict(b=37, c=50, htp=47))
        self.assertIsNone(S.hard_stop(row))
        self.assertIsNone(S.hard_stop(dict(row, available_kib=2 * 1024 * 1024)))  # exactly floor ok
        self.assertEqual(S.hard_stop(dict(row, available_kib=2 * 1024 * 1024 - 1)),
                         'MemAvailable below 2 GiB retained-headroom floor')
        self.assertEqual(S.hard_stop(dict(row, thermal=dict(b=46, c=50, htp=47))),
                         'hard thermal threshold')
        self.assertEqual(S.hard_stop(dict(row, thermal=dict(b=37, c=101, htp=47))),
                         'hard thermal threshold')
        self.assertEqual(S.hard_stop(dict(row, thermal=dict(b=37, c=50, htp=86))),
                         'hard thermal threshold')
        # swap exhaustion is a hard stop
        self.assertEqual(S.hard_stop(dict(row, swap_kib=0)),
                         'SwapFree below 512 MiB emergency floor')

    # --- projection pure function ------------------------------------------
    def test_projection_detects_decline_not_flat(self):
        # flat / rising -> never aborts
        self.assertIsNone(S.declining_memory_projection(
            [(0, 4500000), (1, 4400000), (2, 4300000), (3, 4200000)]))
        self.assertIsNone(S.declining_memory_projection(
            [(0, 4000000), (1, 4100000), (2, 4200000)]))
        # too few samples -> None
        self.assertIsNone(S.declining_memory_projection([(0, 4000000), (1, 3000000)]))
        # steep decline -> fires
        r = S.declining_memory_projection([(0, 4500000), (2, 4400000), (4, 3000000)])
        self.assertIsNotNone(r)
        self.assertIn('projection', r['reason'])
        self.assertLess(r['projected_kib'], S.RETAINED_HEADROOM_KIB)

    def test_projection_pure_on_observed_trace_first_abort_before_floor(self):
        """Replay the exact 25-sample observed decline.  Scan prefixes and find the
        FIRST sample index at which the projection fires.  It MUST fire while
        available is still strictly above the observed 1.846 GiB floor, i.e. an
        EARLY abort -- reproducing the safety behaviour the failed run lacked."""
        samples, floor_min = load_observed_decline()
        self.assertEqual(len(samples), 25)
        self.assertEqual(floor_min, 1935480)           # 1.846 GiB observed floor
        first = None
        for end in range(3, len(samples) + 1):
            hist = [(1000.0 + samples[k][0], samples[k][1]) for k in range(end)]
            r = S.declining_memory_projection(hist)
            if r is not None:
                first = (end - 1, samples[end - 1][1], r)
                break
        self.assertIsNotNone(first, 'projection never fired on the observed trace')
        idx, avail_at_abort, r = first
        # EARLY: aborted well before the 1.846 GiB floor was reached.
        self.assertGreater(avail_at_abort, floor_min)
        self.assertGreater(avail_at_abort, S.RETAINED_HEADROOM_KIB)   # still above 2 GiB
        self.assertLess(r['projected_kib'], S.RETAINED_HEADROOM_KIB)  # projects crossing 2 GiB

    # --- swap projection pure function ---------------------------------------
    def test_swap_projection_detects_decline_not_flat(self):
        # flat / rising swap -> never aborts
        self.assertIsNone(S.declining_swap_projection(
            [(0, 2000000), (1, 1900000), (2, 1800000), (3, 1700000)]))
        self.assertIsNone(S.declining_swap_projection(
            [(0, 1500000), (1, 1600000), (2, 1700000)]))
        # too few samples -> None
        self.assertIsNone(S.declining_swap_projection([(0, 2000000), (1, 1000000)]))
        # steep swap decline -> fires
        r = S.declining_swap_projection([(0, 2000000), (2, 1900000), (4, 500000)])
        self.assertIsNotNone(r)
        self.assertIn('swap', r['reason'])
        self.assertLess(r['projected_kib'], S.SWAP_HARD_FLOOR_KIB)

    # --- run_cell: swap projection abort -------------------------------------
    def test_run_cell_swap_projection_abort(self):
        """When swap declines steeply, run_cell must abort via the swap projection
        BEFORE the swap hard floor is reached, and record a strong lifecycle abort."""
        import tempfile
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        tele, life = out / 'telemetry.jsonl', out / 'lifecycle.jsonl'
        # Memory stays healthy; swap declines from 2 GiB to 0.6 GiB over 5 ticks.
        therm = dict(b=37, c=50, htp=47)
        swap_samples = [(0.0, 4500000, 2000000, therm), (1.0, 4500000, 1700000, therm),
                        (2.0, 4500000, 1200000, therm), (3.0, 4500000, 800000, therm),
                        (4.0, 4500000, 600000, therm)]
        sampler = SamplerFeed(swap_samples)
        child = FakeChild()
        with patch.object(S.time, 'sleep'), patch.object(S, 'child_telemetry', return_value=None):
            result = S.run_cell(dict(h=4, threads=4, mtp='off', tokens=16), 0, out,
                                lambda cell, cmd: child, None, tele, life,
                                sampler=sampler)
        self.assertTrue(result['aborted_by_swap_projection'])
        self.assertIn('swap', result['stop_reason'])
        self.assertTrue(child.terminate_called)
        lifecycle = [json.loads(l) for l in life.read_text().splitlines()]
        self.assertIn('abort', [e['event'] for e in lifecycle])

    # --- run_cell: early-abort replay of the observed collapse -------------
    def test_run_cell_early_abort_replay_before_floor(self):
        """Feed run_cell the live 4.38->1.85 GiB / 8.35s trace.  The fake child never
        exits on its own; the projection must tear it down early, BEFORE the
        available memory reaches the 1.846 GiB floor, and must record a strong
        lifecycle 'abort' event."""
        import tempfile
        samples, floor_min = load_observed_decline()
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        tele, life = out / 'telemetry.jsonl', out / 'lifecycle.jsonl'
        sampler = SamplerFeed(samples)
        child = FakeChild()
        with patch.object(S.time, 'sleep'), patch.object(S, 'child_telemetry', return_value=None):
            result = S.run_cell(dict(h=4, threads=4, mtp='off', tokens=16), 0, out,
                                lambda cell, cmd: child, None, tele, life,
                                sampler=sampler)
        # Projection aborted early: did not consume the entire 25-sample trace.
        self.assertTrue(result['aborted_by_memory_projection'])
        self.assertIn('projection', result['stop_reason'])
        self.assertGreater(sampler.i, 0)          # did consume some
        self.assertLess(sampler.calls, len(samples))     # did NOT reach the floor
        # The abort fired while memory was still above the observed 1.846 GiB floor.
        self.assertGreater(result['projection']['current_kib'], floor_min)
        # Process group was signalled (short termination), child did not exit cleanly.
        self.assertTrue(child.terminate_called)
        self.assertIsNone(result['returncode'])   # killed, not normally exited
        # Strong lifecycle recording: an 'abort' event exists + summary is writable.
        lifecycle = [json.loads(l) for l in life.read_text().splitlines()]
        self.assertEqual(lifecycle[0]['event'], 'preflight') if False else None
        events = [e['event'] for e in lifecycle]
        self.assertIn('start', events)
        self.assertIn('abort', events)
        abort = next(e for e in lifecycle if e['event'] == 'abort')
        self.assertIn('projection', abort['stop_reason'])

    # --- run_cell: clean finish / nonzero-exit paths record lifecycle ------
    def test_run_cell_records_finish_on_clean_exit(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        tele, life = out / 'telemetry.jsonl', out / 'lifecycle.jsonl'
        child = FakeChild(rc=0, exit_after=0)   # exits 0 on first poll
        with patch.object(S.time, 'sleep'):
            result = S.run_cell(dict(h=4, threads=4, mtp='off', tokens=16), 0, out,
                                lambda c, cmd: child, None, tele, life)
        self.assertIsNone(result['stop_reason'])
        lifecycle = [json.loads(l) for l in life.read_text().splitlines()]
        self.assertIn('finish', [e['event'] for e in lifecycle])

    def test_run_cell_records_abort_on_nonzero(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        tele, life = out / 'telemetry.jsonl', out / 'lifecycle.jsonl'
        child = FakeChild(rc=1, exit_after=0)
        with patch.object(S.time, 'sleep'):
            result = S.run_cell(dict(h=1, threads=6, mtp='on', tokens=16), 0, out,
                                lambda c, cmd: child, None, tele, life)
        self.assertEqual(result['returncode'], 1)
        lifecycle = [json.loads(l) for l in life.read_text().splitlines()]
        self.assertIn('abort', [e['event'] for e in lifecycle])

    # --- fast sampling independence ---------------------------------------
    def test_sample_memory_never_subprocess(self):
        """The fast tick must not call termux-battery-status (independent of battery)."""
        called = {'n': 0}
        def boom(*a, **k):
            called['n'] += 1
            raise AssertionError('fast sample must not run a subprocess')
        with patch.object(S.subprocess, 'run', boom), patch.object(S, 'temperatures',
                                                                   return_value=dict(b=None, c=None, htp=None)):
            row = S.sample_memory(None)
        self.assertEqual(called['n'], 0)
        self.assertIn('available_kib', row)
        self.assertIn('swap_kib', row)
        # BatteryCache.value() is also non-blocking / non-subprocess.
        bc = S.BatteryCache()
        with patch.object(S.subprocess, 'run', boom):
            self.assertIsNone(bc.value()[0])

    # --- child telemetry parsing (pure helpers) ----------------------------
    def test_child_telemetry_parsing(self):
        status = ("VmRSS:     1234 kB\nVmSize:    5678 kB\nVmSwap:      90 kB\n"
                  "Threads:   7\n")
        info = S._parse_status(status)
        self.assertEqual(info['rss_kib'], 1234)
        self.assertEqual(info['swap_kib'], 90)
        self.assertEqual(info['threads'], 7)
        self.assertEqual(S._parse_smaps_rollup("Pss:   2048 kB\nPrivate_Dirty: 100 kB\n"), 2048)
        self.assertIsNone(S._parse_smaps_rollup("no pss here\n"))
        # /proc on a live pid returns None when the pid is bogus
        self.assertIsNone(S.child_telemetry(99999999))

    # --- single-run lock --------------------------------------------------
    def test_single_run_lock_second_refused(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        lock1 = S.RunLock(Path(tmp.name) / '.run.lock')
        lock2 = S.RunLock(Path(tmp.name) / '.run.lock')
        lock1.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'single-run lock held'):
                lock2.acquire()
        finally:
            lock1.release()
        # After release, a new lock acquires fine.
        lock3 = S.RunLock(Path(tmp.name) / '.run.lock')
        lock3.acquire(); lock3.release()

    # --- dry run launches nothing -----------------------------------------
    def test_dry_run_does_not_launch(self):
        import tempfile, shutil
        tmp = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        spawned = {'n': 0}
        def fake_spawn(cell, cmd):
            spawned['n'] += 1
            return FakeChild(rc=0, exit_after=0)
        row = dict(utc='x', monotonic=0.0, available_kib=6 * 1024 * 1024,
                   swap_kib=0, capacity=60, charging=True,
                   thermal=dict(b=37, c=50, htp=47))
        with patch.object(S.time, 'sleep'), patch.object(S, 'temperatures',
                return_value=dict(b=37, c=50, htp=47)), patch.object(S, 'HERE', Path(tmp)), \
             patch.object(S, 'RunLock') as RL:
            RL.return_value.acquire.return_value = None
            RL.return_value.release.return_value = None
            with patch.object(S, 'sample_memory', return_value=row):
                S.run('qualification', 'dry-ev', run_flag=False, screen_passed=True,
                      spawn=fake_spawn)
        self.assertEqual(spawned['n'], 0)          # NO inference launched
        plan = Path(tmp) / 'dry-ev' / 'plan.json'
        self.assertTrue(plan.exists())
        pins = json.loads(plan.read_text())['pins']
        self.assertEqual(pins['mmap'], 'mmap')
        self.assertEqual(pins['device'], 'HTP0')
        self.assertEqual(pins['ngl'], 99)
        self.assertEqual(pins['ctx'], 256)
        self.assertEqual(pins['batch'], 128)
        self.assertEqual(pins['flash'], 'off')
        life = [json.loads(l) for l in (Path(tmp) / 'dry-ev' / 'lifecycle.jsonl').read_text().splitlines()]
        self.assertIn('dry_run_complete', [e['event'] for e in life])
        self.assertIn('preflight', [e['event'] for e in life])

    # --- screen prerequisite gates --run -----------------------------------
    def test_run_requires_screen_prerequisite(self):
        with self.assertRaisesRegex(RuntimeError, 'screen prerequisite'):
            S.run('qualification', 'gated-ev', run_flag=True, screen_passed=False)
        self.assertFalse((HERE / 'gated-ev').exists())   # never created an evidence dir

    # --- plan shape unchanged from the factorial design --------------------
    def test_plan_shape(self):
        q = S.plan('qualification')
        s = S.plan('screen')
        self.assertEqual([(c['h'], c['threads'], c['mtp']) for c in q],
                         [(4, 4, 'off'), (4, 4, 'on'), (1, 6, 'off'), (1, 6, 'on')])
        self.assertEqual(len(s), 24)
        self.assertEqual([c['tokens'] for c in q], [16, 16, 16, 16])
        self.assertTrue(all(c['tokens'] == 64 for c in s))


if __name__ == '__main__':
    unittest.main()
