"""Lifecycle checks without model execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / 'reports/2026-10-13/cache-64-80-96-four-retry-02/run-benchmark.py'
spec = importlib.util.spec_from_file_location('retry02', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def records(self, count=6, cool=5, footer=False, mismatch=False):
        passes = [dict(type='pass', slots=s, threads=t, cache_mib=m,
                       evaluated_positions=64, bit_identical_positions=(0 if i == 0 else 64))
                  for i, (s, t, m) in enumerate(runner.PLAN[:count])]
        if mismatch:
            passes[-1]['bit_identical_positions'] = 63
        if footer:
            passes.append(dict(type='complete', passes=6, reference_positions=64,
                               verified_positions=320))
        (self.out / 'benchmark.jsonl').write_text(''.join(json.dumps(p) + chr(10) for p in passes))
        (self.out / 'cooldown.jsonl').write_text(''.join(
            json.dumps(dict(after_pass=i, paused_s=180)) + chr(10) for i in range(1, cool + 1)))

    def test_final_cooldown_waits_for_footer_and_zero_exit(self):
        self.records()
        reason, deadline = runner.monitor_reason(None, None, 0, self.out, 10, None)
        self.assertIsNone(reason)
        self.assertEqual(deadline, 130)
        self.assertEqual(runner.monitor_reason(None, None, 0, self.out, 130, None),
                         (None, 250))
        self.assertEqual(runner.monitor_reason(None, None, 0, self.out, 129, 130),
                         (None, 130))
        self.assertEqual(runner.monitor_reason(None, None, 0, self.out, 130, 130),
                         ('benchmark footer/exit teardown timeout', 130))
        self.records(footer=True)
        self.assertEqual(runner.monitor_reason(None, None, 0, self.out, 10, None), (None, 130))
        self.assertTrue(runner.evidence(self.out)[-1])
        self.assertEqual(runner.monitor_reason(0, 0, 0, self.out, 200, 130)[0], None)

    def test_normal_completion_at_deadline(self):
        self.records(footer=True)
        self.assertEqual(runner.monitor_reason(None, None, 0, self.out, 129, 130),
                         (None, 130))
        self.assertEqual(runner.monitor_reason(0, 0, 0, self.out, 130, 130),
                         (None, 130))
        self.assertTrue(runner.evidence(self.out)[-1])

    def test_early_monitor_and_incomplete(self):
        self.records(count=3, cool=2)
        self.assertIsNotNone(runner.monitor_reason(None, None, 0, self.out, 1, None)[0])
        self.assertIsNotNone(runner.monitor_reason(None, 1, None, self.out, 1, None)[0])
        self.records(cool=4, footer=True)
        self.assertFalse(runner.evidence(self.out)[-1])

    def test_replay_mismatch(self):
        self.records(footer=True, mismatch=True)
        self.assertFalse(runner.evidence(self.out)[-1])

    def test_child_exit_race_and_timeout(self):
        self.records(count=2)
        self.assertIsNone(runner.monitor_reason(1, 0, 0, self.out, 10, None)[0])
        self.records(footer=True)
        self.assertIsNone(runner.monitor_reason(None, None, 0, self.out, 129, 130)[0])
        self.assertIn('timeout', runner.monitor_reason(None, None, 0, self.out, 130, 130)[0])


if __name__ == '__main__':
    unittest.main()
