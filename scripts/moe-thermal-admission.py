"""Thermal admission policy shared by the runner and synthetic trace tests."""
import math
import statistics

KEYS = ('battery_c', 'cpu_max_c', 'htp_max_c')
POLICY_ID = 'relaxed-plus-3c-v1'
TOLERANCE = dict(battery_c=3.4, cpu_max_c=7.0, htp_max_c=7.0)
INITIAL_CEILING = dict(battery_c=37.0, cpu_max_c=58.0, htp_max_c=48.0)
HOLD_SECONDS = 30.0
MAX_SAMPLE_GAP = 3.5


class Admission:
    def __init__(self):
        self.target = None
        self.window = []

    def begin_pass(self):
        self.window = []

    def observe(self, row):
        if any(not isinstance(row.get(key), (int, float)) or
               not math.isfinite(row[key]) or not 0 <= row[key] <= 150 for key in KEYS):
            self.window = []
            return None
        now = row['monotonic_s']
        if self.window and (now <= self.window[-1]['monotonic_s'] or
                            now - self.window[-1]['monotonic_s'] > MAX_SAMPLE_GAP):
            self.window = []
        valid = (all(row[key] <= INITIAL_CEILING[key] for key in KEYS) if self.target is None
                 else all(abs(row[key] - self.target[key]) <= TOLERANCE[key] + 1e-9 for key in KEYS))
        if not valid:
            self.window = []
            return None
        self.window.append(row)
        # Keep one boundary sample so the retained interval spans at least 30 s.
        while len(self.window) > 1 and now - self.window[1]['monotonic_s'] >= HOLD_SECONDS:
            self.window.pop(0)
        if now - self.window[0]['monotonic_s'] < HOLD_SECONDS:
            return None
        target = self.target or {key: statistics.median(sample[key] for sample in self.window) for key in KEYS}
        if any(abs(sample[key] - target[key]) > TOLERANCE[key] + 1e-9
               for sample in self.window for key in KEYS):
            return None
        self.target = target
        return dict(policy_id=POLICY_ID, target_c=target.copy(), tolerance_c=TOLERANCE.copy(),
                    initial_ceiling_c=INITIAL_CEILING.copy(), hold_s=HOLD_SECONDS,
                    window_start_monotonic_s=self.window[0]['monotonic_s'],
                    window_end_monotonic_s=now, samples=len(self.window),
                    window_c={key: dict(min=min(sample[key] for sample in self.window),
                                        max=max(sample[key] for sample in self.window)) for key in KEYS},
                    admitted_c={key: row[key] for key in KEYS})
