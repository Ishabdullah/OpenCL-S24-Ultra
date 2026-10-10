"""Check held-out coverage and byte accounting independently of live hardware."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mapping', Path(__file__).resolve().parents[1] / 'scripts/analyze-moe-mapping.py')
mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping)


class MappingAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.records = [dict(type='config', trials=3, experts=4, experts_used=1, prompt_tokens=1),
                        dict(type='trial', trial=1, prefill={'0': {'0': 1}}, decode={'0': {'1': 1}}, generated_tokens=2, text='a', new_experts_total=2, storage_read_bytes=10, elapsed_s=1),
                        dict(type='trial', trial=2, prefill={'0': {'0': 1}}, decode={'0': {'2': 1}}, generated_tokens=2, text='b', new_experts_total=1, storage_read_bytes=20, elapsed_s=2),
                        dict(type='trial', trial=3, prefill={'0': {'0': 1}}, decode={'0': {'2': 1}}, generated_tokens=2, text='b', new_experts_total=0, storage_read_bytes=30, elapsed_s=3)]
        self.sizes = [('blk.0.ffn_gate_exps.weight', [2, 2, 4], 40),
                      ('blk.0.ffn_up_exps.weight', [2, 2, 4], 80),
                      ('blk.0.ffn_down_exps.weight', [2, 2, 4], 120),
                      ('token_embd.weight', [2, 2], 100)]

    def test_heldout_map_stays_frozen_and_sizes_are_exact(self):
        result = mapping.analyze(self.records, 1, self.sizes)
        # Expert 2 stays a miss in both held-out trials, even after first observation.
        self.assertEqual(result['heldout_coverage']['decode']['fraction'], 0)
        self.assertEqual(result['heldout_coverage']['prefill']['fraction'], 1)
        self.assertEqual(result['heldout_new_distinct_total'], 1)
        self.assertEqual(result['weights']['training_expert_bytes'], 120)
        self.assertEqual(result['weights']['training_plus_fixed_bytes'], 220)
        self.assertEqual(result['weights']['final_expert_bytes'], 180)
        self.assertEqual(result['storage_read_bytes'], 60)

    def test_unused_mtp_experts_do_not_enter_target_map(self):
        sizes = self.sizes + [('blk.1.ffn_gate_exps.weight', [2, 2, 4], 40)]
        result = mapping.analyze(self.records, 1, sizes)
        self.assertEqual(result['weights']['full_routed_bytes'], 240)
        self.assertEqual(result['weights']['fixed_tensor_bytes'], 140)
        self.assertEqual(result['weights']['training_plus_fixed_bytes'], 260)

    def test_missing_capture_is_rejected(self):
        broken = copy.deepcopy(self.records)
        broken[1]['decode']['0']['1'] = 2
        with self.assertRaisesRegex(AssertionError, 'count mismatch'):
            mapping.analyze(broken, 1, self.sizes)

    def test_incomplete_run_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'incomplete run'):
            mapping.analyze(self.records[:-1], 1, self.sizes)


if __name__ == '__main__':
    unittest.main()
