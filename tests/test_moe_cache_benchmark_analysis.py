import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze-moe-cache-benchmark.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class BenchmarkAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.records = [dict(type='config'), dict(type='reference', steps=3, tokens=[1, 2, 3])]
        for position, slots, seconds, storage in [(1, 0, 10, 100), (2, 32, 5, 40), (3, 32, 6, 60), (4, 0, 12, 120)]:
            self.records.append(dict(type='pass', **{'pass': position}, slots=slots,
                evaluated_positions=3, decode_evaluations=2,
                bit_identical_positions=0 if position == 1 else 3,
                prefill_s=1, decode_s=seconds-1, evaluation_s=seconds,
                prefill_storage_bytes=10, decode_storage_bytes=storage-10, storage_bytes=storage))
        self.records.append(dict(type='complete', verified_positions=9))

    def test_balanced_pairs_and_pooled_rates(self):
        result = analysis.summarize(self.records)
        self.assertEqual(result['evaluation_speedup'], 2)
        self.assertEqual([pair['order'] for pair in result['pairs']], [[0, 32], [32, 0]])
        self.assertAlmostEqual(result['modes']['0']['pooled_decode_tok_s'], 4/20)
        self.assertAlmostEqual(result['storage_reduction_fraction'], 1-50/110)

    def test_missing_and_unverified_passes_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'incomplete'):
            analysis.summarize(self.records[:-1])
        broken = copy.deepcopy(self.records)
        broken[3]['bit_identical_positions'] = 2
        with self.assertRaisesRegex(AssertionError, 'verification failed'):
            analysis.summarize(broken)

    def test_admission_trace_is_checked_and_target_cannot_change(self):
        thermal, decisions = [], []
        target = dict(battery_c=32, cpu_max_c=50, htp_max_c=40)
        for position, slots in enumerate([0, 32, 32, 0], 1):
            start = position * 100
            for second in range(start, start+31, 2):
                thermal.append(dict(pass_number=position, stage='admission', monotonic_s=second, **target))
            decisions.append(dict(pass_number=position, slots=slots, target_c=target.copy(),
                tolerance_c=dict(battery_c=0.4, cpu_max_c=4.0, htp_max_c=4.0),
                initial_ceiling_c=dict(battery_c=34, cpu_max_c=55, htp_max_c=45), hold_s=30,
                window_start_monotonic_s=start, window_end_monotonic_s=start+30,
                samples=16, admitted_c=target.copy(),
                window_c={key: dict(min=value, max=value) for key, value in target.items()}))
        result = analysis.summarize(self.records, thermal=thermal, admissions=decisions)
        self.assertEqual(len(result['admissions']), 4)
        self.assertIn('common admission bands', result['limitations'][2])
        with self.assertRaisesRegex(AssertionError, 'incomplete thermal'):
            analysis.validate_admissions(decisions[:1], thermal)
        self.assertEqual(len(analysis.validate_admissions(decisions[:1], thermal, require_complete=False)), 1)
        bad = copy.deepcopy(thermal)
        bad[20]['battery_c'] = 34
        with self.assertRaisesRegex(AssertionError, 'outside admission'):
            analysis.summarize(self.records, thermal=bad, admissions=decisions)
        bad = copy.deepcopy(decisions)
        bad[2]['target_c']['battery_c'] = 33
        with self.assertRaisesRegex(AssertionError, 'target changed'):
            analysis.summarize(self.records, thermal=thermal, admissions=bad)

    def test_rest_samples_are_excluded_from_pass_temperatures(self):
        thermal = [dict(pass_number=2, stage='evaluation', battery_c=35, cpu_max_c=66, htp_max_c=55),
                   dict(pass_number=2, stage='rest_or_setup', battery_c=34, cpu_max_c=45, htp_max_c=44)]
        result = analysis.summarize(self.records, thermal=thermal)
        self.assertEqual(result['temperature_by_pass']['2']['samples'], 1)
        self.assertEqual(result['temperature_by_pass']['2']['battery_first_c'], 35)


if __name__ == '__main__':
    unittest.main()
