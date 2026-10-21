"""Ensure the physical pilot only uses temperature readings for cooldown."""
import importlib.util
import json
from pathlib import Path
import unittest

RUNNER = (Path(__file__).resolve().parents[1] /
          'reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py')
spec = importlib.util.spec_from_file_location('qwen3sweep', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CooldownOnlyPolicy(unittest.TestCase):
    def test_no_inference_temperature_stop_path_exists(self):
        self.assertFalse(hasattr(runner, 'thermal_stop_reason'))
        source = RUNNER.read_text()
        self.assertNotIn("choices=('cooldown-only', 'guarded')", source)
        self.assertNotIn('stop_limits', source)
        self.assertNotIn('cpu_over_limit_streak', source)

    def test_dry_run_reports_cooldown_only(self):
        import contextlib
        import io
        from unittest.mock import patch

        stdout = io.StringIO()
        with patch('sys.argv', ['run-benchmark.py', '--pilot']), contextlib.redirect_stdout(stdout):
            runner.main()
        import json
        self.assertEqual(json.loads(stdout.getvalue())['thermal_policy'], 'cooldown-only')

    def test_thread_sweep_covers_one_through_eight_with_fixed_cache(self):
        import contextlib
        import io
        from unittest.mock import patch
        stdout = io.StringIO()
        argv = ['run-benchmark.py', '--thread-sweep', '--backend', 'cpu',
                '--pilot-slots', '64', '--pilot-tokens', '16']
        with patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
            runner.main()
        result = json.loads(stdout.getvalue())
        self.assertTrue(result['thread_sweep'])
        self.assertEqual(result['token_limit'], 16)
        self.assertEqual(result['plan'], ','.join(f'64:{t}:2560' for t in range(1, 9)))
        self.assertEqual(result['pilot_affinity'], 'all allowed CPUs after load')

    def test_eight_pass_replay_validation_checks_every_thread_count(self):
        plan = [(64, threads, 2560) for threads in range(1, 9)]
        passes = [dict(slots=slots, threads=threads, cache_mib=mib,
                       bit_identical_positions=0 if index == 0 else 16,
                       evaluated_positions=16)
                  for index, (slots, threads, mib) in enumerate(plan)]
        footer = dict(passes=8, reference_positions=16, verified_positions=112)
        self.assertTrue(runner.validate_replay(passes, footer, plan))
        passes[6]['threads'] = 5
        self.assertFalse(runner.validate_replay(passes, footer, plan))

    def test_accelerator_layer_range_dry_run_and_decline_gate(self):
        import contextlib
        import importlib.util
        import io
        from unittest.mock import patch
        script = Path(__file__).resolve().parents[1] / 'scripts/run-qwen-accelerator-layer-sweep.py'
        spec = importlib.util.spec_from_file_location('accelerator_layer_sweep', script)
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        stdout = io.StringIO()
        with patch('sys.argv', [str(script)]), contextlib.redirect_stdout(stdout):
            sweep.main()
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan['layers'], list(range(1, 50)))
        self.assertEqual(plan['midpoint_layer'], 25)
        history = [dict(pooled_tok_s=rate) for rate in (0.30, 0.29, 0.28, 0.27)]
        self.assertTrue(sweep.declining_at_midpoint(history, 25, 25, consecutive=3))
        self.assertFalse(sweep.declining_at_midpoint(history, 24, 25, consecutive=3))
        self.assertFalse(sweep.declining_at_midpoint(
            [dict(pooled_tok_s=rate) for rate in (0.30, 0.29, 0.31, 0.27)], 25, 25, consecutive=3))

    def test_explicit_gpu_npu_layer_split_dry_run(self):
        import contextlib
        import io
        from unittest.mock import patch
        stdout = io.StringIO()
        argv = ['run-benchmark.py', '--pilot', '--backend', 'gpu+npu', '--ngl', '3',
                '--gpu-layers', '2', '--npu-layers', '1', '--cpu-layers', '46', '--pilot-slots', '64',
                '--pilot-tokens', '16']
        with patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
            runner.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result['accelerator_layer_split'], {'gpu': 2, 'npu': 1})
        self.assertEqual(result['model_layer_split'], {'cpu': 46, 'gpu': 2, 'npu': 1})
        self.assertEqual(result['device'], 'GPUOpenCL,HTP0')

    def test_balanced_triad_output_placement_dry_runs(self):
        import contextlib
        import io
        from unittest.mock import patch
        expected = {
            'cpu': {'cpu': 17, 'gpu': 16, 'npu': 16},
            'gpu': {'cpu': 16, 'gpu': 17, 'npu': 16},
            'npu': {'cpu': 16, 'gpu': 16, 'npu': 17},
        }
        for output_layer, split in expected.items():
            stdout = io.StringIO()
            argv = ['run-benchmark.py', '--pilot', '--backend', 'gpu+npu',
                    '--triad-output', output_layer, '--pilot-slots', '64',
                    '--pilot-tokens', '2']
            with self.subTest(output_layer=output_layer), \
                    patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
                runner.main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result['model_layer_split'], split)
            self.assertEqual(result['n_gpu_layers'], 49)
            self.assertEqual(result['output_layer'], output_layer)

    def test_bounded_htp_cache_starts_with_one_block_and_two_pass_replay(self):
        import contextlib
        import io
        from unittest.mock import patch
        stdout = io.StringIO()
        argv = ['run-benchmark.py', '--backend', 'gpu+npu', '--bounded-htp-layers', '1',
                '--htp-cache-slots', '8', '--htp-cache-budget-mib', '512', '--pilot-tokens', '2']
        with patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
            runner.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result['plan'], '8:1:320,8:1:320')
        self.assertEqual(result['model_layer_split'], {'cpu': 32, 'gpu': 16, 'npu': 1})
        self.assertEqual(result['bounded_htp_layers'], 1)
        self.assertEqual(result['htp_cache_slots'], 8)
        self.assertEqual(result['htp_cache_budget_mib'], 512)
        self.assertEqual(result['token_limit'], 2)

    def test_bounded_htp_map_scales_only_the_htp_region(self):
        import contextlib
        import io
        from unittest.mock import patch
        for n_htp in (1, 2, 4, 8, 16):
            stdout = io.StringIO()
            argv = ['run-benchmark.py', '--backend', 'gpu+npu', '--bounded-htp-layers', str(n_htp)]
            with patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
                runner.main()
            result = json.loads(stdout.getvalue())
            split = result['model_layer_split']
            self.assertEqual(split, {'cpu': 33 - n_htp, 'gpu': 16, 'npu': n_htp})
            self.assertEqual(sum(split.values()), 49)

    def test_bounded_sixteen_layer_cache_supports_triad_output_placement(self):
        import contextlib
        import io
        from unittest.mock import patch
        expected = {
            'cpu': {'cpu': 17, 'gpu': 16, 'npu': 16},
            'gpu': {'cpu': 16, 'gpu': 17, 'npu': 16},
            'npu': {'cpu': 16, 'gpu': 16, 'npu': 17},
        }
        for output_layer, split in expected.items():
            stdout = io.StringIO()
            argv = ['run-benchmark.py', '--backend', 'gpu+npu',
                    '--bounded-htp-layers', '16', '--htp-cache-slots', '8',
                    '--htp-cache-budget-mib', '512', '--triad-output', output_layer]
            with self.subTest(output_layer=output_layer), \
                    patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
                runner.main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result['model_layer_split'], split)
            self.assertEqual(result['bounded_htp_layers'], 16)
            self.assertEqual(result['output_layer'], output_layer)
            self.assertEqual(result['n_gpu_layers'], 49)

    def test_bounded_triad_run_uses_requested_cpu_thread_count(self):
        import contextlib
        import io
        from unittest.mock import patch
        for threads in (2, 3, 4, 5):
            stdout = io.StringIO()
            argv = ['run-benchmark.py', '--backend', 'gpu+npu',
                    '--bounded-htp-layers', '16', '--htp-cache-slots', '8',
                    '--htp-cache-budget-mib', '512', '--triad-output', 'cpu',
                    '--pilot-threads', str(threads), '--pilot-tokens', '32']
            with self.subTest(threads=threads), \
                    patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
                runner.main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result['plan'], f'8:{threads}:320,8:{threads}:320')
            self.assertEqual(result['output_layer'], 'cpu')
            self.assertEqual(result['bounded_htp_layers'], 16)

    def test_htp_cache_verification_requires_staged_bytes(self):
        evidence = runner.htp_cache_staging_evidence(
            'ggml-hex: bounded HTP cache staged tensor=blk.32.ffn_gate_exps.weight '
            'routed_ids=8 loaded_slots=3 bytes=24576\n'
            'ggml-hex: bounded HTP cache staged tensor=blk.32.ffn_up_exps.weight '
            'routed_ids=8 loaded_slots=2 bytes=16384\n')
        self.assertEqual(evidence['operations'], 2)
        self.assertEqual(evidence['loaded_slots'], 5)
        self.assertEqual(evidence['staged_bytes'], 40960)
        self.assertEqual(evidence['tensors'], [
            'blk.32.ffn_gate_exps.weight', 'blk.32.ffn_up_exps.weight'])
        self.assertEqual(runner.htp_cache_staging_evidence('')['operations'], 0)

    def test_htp_resident_verification_parses_device_buffer_operations(self):
        evidence = runner.htp_resident_evidence(
            'resident HTP expert weight blk.32.ffn_gate_exps.weight bytes=12 cumulative=12 cap_mib=1024\n'
            'ggml-hex: resident HTP MoE op tensor=blk.32.ffn_gate_exps.weight buffer=HTP0 bytes=12\n')
        self.assertEqual(evidence['weights'], ['blk.32.ffn_gate_exps.weight'])
        self.assertEqual(evidence['weight_bytes'], 12)
        self.assertEqual(evidence['operations'], 1)
        self.assertEqual(evidence['operation_buffers'], ['HTP0'])

    def test_dry_run_records_prompt_for_replay_provenance(self):
        import contextlib
        import io
        from unittest.mock import patch
        prompt = 'A longer prompt for a cross-backend replay check.'
        stdout = io.StringIO()
        argv = ['run-benchmark.py', '--backend', 'gpu+npu', '--htp-resident-layers', '1',
                '--pilot-tokens', '16', '--prompt', prompt]
        with patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
            runner.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result['prompt'], prompt)

    def test_evidence_handles_load_abort_before_config_record(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'benchmark.jsonl').write_text('')
            (out / 'cooldown.jsonl').write_text('')
            self.assertEqual(runner.evidence(out), (0, 0, False, False))

    def test_layer_profile_summary_uses_decode_medians_and_device_placement(self):
        import tempfile
        assignments = {layer: ('CPU' if layer < 46 else 'GPUOpenCL' if layer < 48 else 'HTP0')
                       for layer in range(49)}
        with tempfile.TemporaryDirectory() as temp:
            profile = Path(temp) / 'layer-timing.jsonl'
            profile.write_text(''.join(
                json.dumps(dict(pass_id=pass_id, step=step, layer=layer,
                                elapsed_ns=(layer + 1) * (100 if pass_id == 1 else 200))) + '\n'
                for pass_id in (1, 2) for step in (0, 1)
                for layer in range(49)))
            summary = runner.summarize_layer_profile(profile, assignments)
        self.assertEqual(summary['per_layer']['0']['median_ns'], 150)
        self.assertEqual(summary['per_layer']['48']['device'], 'HTP0')
        self.assertIn('synchronize', summary['timing_note'])

    def test_cpu_layer_count_must_match_remaining_model_layers(self):
        import contextlib
        import io
        from unittest.mock import patch
        stdout = io.StringIO()
        argv = ['run-benchmark.py', '--pilot', '--backend', 'gpu+npu', '--ngl', '3',
                '--gpu-layers', '2', '--npu-layers', '1', '--cpu-layers', '45']
        with patch('sys.argv', argv), contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_layer_assignment_verifier_counts_each_backend(self):
        log = '\n'.join([
            'load_tensors: layer   0 assigned to device CPU, is_swa = 0',
            'load_tensors: layer  47 assigned to device GPUOpenCL, is_swa = 0',
            'load_tensors: layer  48 assigned to device GPUOpenCL, is_swa = 0',
            'load_tensors: layer  49 assigned to device HTP0, is_swa = 0',
        ])
        self.assertEqual(runner.count_layer_assignments(log),
                         {'GPUOpenCL': 2, 'HTP0': 1, 'CPU': 1})


if __name__ == '__main__':
    unittest.main()
