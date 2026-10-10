import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('admission', Path(__file__).resolve().parents[1] / 'scripts/moe-thermal-admission.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

class AdmissionTest(unittest.TestCase):
    def sample(self, seconds, battery=32.0, cpu=50.0, htp=40.0):
        return dict(monotonic_s=seconds, battery_c=battery, cpu_max_c=cpu, htp_max_c=htp)

    def established(self):
        gate = a.Admission()
        for second in range(0, 31, 2):
            result = gate.observe(self.sample(second))
            if second < 30:
                self.assertIsNone(result)
        self.assertIsNotNone(result)
        self.assertEqual(result['target_c']['battery_c'], 32)
        return gate

    def test_same_target_retained_across_passes(self):
        gate = self.established()
        gate.begin_pass()
        for second in range(100, 131, 2):
            result = gate.observe(self.sample(second, battery=32.3, cpu=53, htp=43))
        self.assertEqual(result['target_c']['battery_c'], 32)
        self.assertEqual(result['admitted_c']['battery_c'], 32.3)

    def test_hot_or_missing_reading_resets_continuous_hold(self):
        for bad in (self.sample(20, battery=36.0), self.sample(20, cpu=None), self.sample(20, htp=float('nan'))):
            gate = self.established()
            gate.begin_pass()
            for second in range(0, 20, 2):
                gate.observe(self.sample(second))
            self.assertIsNone(gate.observe(bad))
            for second in range(22, 52, 2):
                self.assertIsNone(gate.observe(self.sample(second)))
            self.assertIsNotNone(gate.observe(self.sample(52)))

    def test_sampling_gap_and_initial_ceiling_prevent_admission(self):
        gate = a.Admission()
        for second in range(0, 40, 2):
            self.assertIsNone(gate.observe(self.sample(second, battery=40)))
        for second in range(100, 130, 2):
            self.assertIsNone(gate.observe(self.sample(second)))
        self.assertIsNone(gate.observe(self.sample(140)))
        for second in range(142, 170, 2):
            self.assertIsNone(gate.observe(self.sample(second)))
        self.assertIsNotNone(gate.observe(self.sample(170)))

if __name__ == '__main__':
    unittest.main()
