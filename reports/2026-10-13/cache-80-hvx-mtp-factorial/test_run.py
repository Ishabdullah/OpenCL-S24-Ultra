"""Offline checks; never launch inference."""
import importlib.util
from pathlib import Path
import unittest
import json
import subprocess
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('factorial', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
launch_spec = importlib.util.spec_from_file_location('factorial_launch', Path(__file__).with_name('launch.py'))
launcher = importlib.util.module_from_spec(launch_spec)
launch_spec.loader.exec_module(launcher)


class RunnerTests(unittest.TestCase):
    def test_detached_launcher_contract(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock
        with TemporaryDirectory() as directory, patch.object(launcher, 'HERE', Path(directory)), patch.object(
                launcher.subprocess, 'Popen', return_value=Mock(pid=1234)) as popen:
            launcher.launch('qualification-retry-check', preflight_only=True)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertIn('--preflight-only', popen.call_args.args[0])
            self.assertEqual((Path(directory) / 'qualification-retry-check.pid').read_text(), '1234' + chr(10))
            with self.assertRaises(FileExistsError):
                launcher.launch('qualification-retry-check')

    def test_battery_api(self):
        with patch.object(runner.subprocess, "run") as call:
            call.return_value.stdout = json.dumps({"status": "CHARGING", "percentage": 41})
            self.assertEqual(runner.battery(), (41, True))
            call.assert_called_with(["termux-battery-status"], capture_output=True,
                                    text=True, timeout=5, check=True)
            for status, level, expected in (("FULL", 100, True), ("DISCHARGING", 39, False),
                                            ("NOT CHARGING", 52, False)):
                call.return_value.stdout = json.dumps({"status": status, "percentage": level})
                self.assertEqual(runner.battery(), (level, expected))
            for output in ("", "{}", "not json", json.dumps({"status": "charging", "percentage": True}),
                           json.dumps({"status": "charging", "percentage": 101}),
                           json.dumps({"status": "unknown", "percentage": 41})):
                call.return_value.stdout = output
                with self.assertRaisesRegex(RuntimeError, "battery status unavailable"):
                    runner.battery()
            for error in (subprocess.TimeoutExpired("termux-battery-status", 5),
                          FileNotFoundError(), subprocess.CalledProcessError(1, "termux-battery-status")):
                call.side_effect = error
                with self.assertRaisesRegex(RuntimeError, "battery status unavailable"):
                    runner.battery()

    def test_battery_failure_blocks_sampling(self):
        with patch.object(runner, 'battery', side_effect=RuntimeError('battery status unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'battery status unavailable'):
                runner.sample()

    def test_plan(self):
        q = runner.plan('qualification')
        s = runner.plan('screen')
        self.assertEqual([(c['h'], c['threads'], c['mtp']) for c in q],
                         [(4, 4, 'off'), (4, 4, 'on'), (1, 6, 'off'), (1, 6, 'on')])
        self.assertEqual(len(s), 24)
        self.assertEqual(len({(c['h'], c['threads'], c['mtp']) for c in s}), 24)
        self.assertEqual([c['tokens'] for c in q], [16]*4)
        self.assertTrue(all(c['tokens'] == 64 for c in s))

    def test_config(self):
        for mtp in ('off', 'on'):
            cell = dict(h=3, threads=5, mtp=mtp, tokens=64)
            env = runner.environment(cell, {'GGML_OLD': 'bad', 'LD_PRELOAD': 'bad', 'PATH': '/bin'})
            self.assertNotIn('GGML_OLD', env)
            self.assertNotIn('LD_PRELOAD', env)
            self.assertEqual(env['GGML_HEXAGON_NHVX'], '3')
            self.assertEqual(env['GGML_HEXAGON_NHMX'], '1')
            self.assertEqual(env['GGML_CPU_MOE_CACHE_SLOTS'], '80')
            self.assertEqual(env['GGML_CPU_MOE_CACHE_MIB'], '3200')
            cmd = runner.command(cell)
            self.assertEqual(cmd[cmd.index('--spec-type') + 1], 'draft-mtp' if mtp == 'on' else 'none')
            self.assertEqual('--spec-draft-n-max' in cmd, mtp == 'on')

    def test_admission_and_abort(self):
        row = dict(charging=True, capacity=40, available_kib=5*1024*1024,
                   monotonic=10, thermal=dict(b=40, c=55, htp=50))
        self.assertIsNone(runner.admission(row))
        self.assertEqual(runner.hard_stop(row)[0], None)
        self.assertIsNotNone(runner.admission(dict(row, charging=False)))
        self.assertIsNotNone(runner.admission(dict(row, available_kib=1536*1024-1)))
        self.assertIsNotNone(runner.hard_stop(dict(row, thermal=dict(b=45, c=55, htp=50)))[0])
        self.assertIsNotNone(runner.hard_stop(dict(row, available_kib=767*1024))[0])
        low = dict(row, available_kib=900*1024)
        self.assertIsNone(runner.hard_stop(low)[0])
        self.assertIsNotNone(runner.hard_stop(dict(low, monotonic=15), 10)[0])


if __name__ == '__main__':
    unittest.main()
