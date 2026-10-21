"""CPU sustained-stop policy for the guarded Qwen3-30B HTP runner."""
import importlib.util
from pathlib import Path
import unittest

RUNNER = (Path(__file__).resolve().parents[1] /
          'reports/2026-10-19/qwen3-30b-a3b-htp/run-guarded-smoke.py')
spec = importlib.util.spec_from_file_location('qwen3htp', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SustainedCpuStop(unittest.TestCase):
    def test_limit_is_strict_and_requires_two_consecutive_samples(self):
        self.assertEqual(runner.update_cpu_over_limit_streak(70, 1), 0)
        self.assertEqual(runner.update_cpu_over_limit_streak(70.1, 0), 1)
        self.assertEqual(runner.update_cpu_over_limit_streak(70.2, 1), 2)

    def test_safe_or_missing_sample_resets_streak(self):
        self.assertEqual(runner.update_cpu_over_limit_streak(69.9, 1), 0)
        self.assertEqual(runner.update_cpu_over_limit_streak(None, 1), 0)

    def test_htp_limit_is_strictly_above_60(self):
        self.assertIsNone(runner.htp_stop_reason(60))
        self.assertIsNone(runner.htp_stop_reason(None))
        self.assertEqual(runner.htp_stop_reason(60.1), 'HTP above 60 C')


if __name__ == '__main__':
    unittest.main()
