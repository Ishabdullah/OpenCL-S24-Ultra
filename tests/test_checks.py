"""Regression checks for false success, unsafe linkage and installer/launcher behavior."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

OUTPUT = module('validate-output')
ELF = module('validate-elf')

class EvidenceChecks(unittest.TestCase):
    def test_cpu_only_and_failed_driver_are_rejected(self):
        for log in ['Available devices:\n CPU: CPU',
                    'sphal InitOpenCLDriver() = -30\n GPUOpenCL: QUALCOMM Adreno(TM) 750']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                OUTPUT.parse('device', log)

    def test_device_detection_does_not_claim_inference(self):
        result = OUTPUT.parse('device', 'sphal InitOpenCLDriver() = 0\n GPUOpenCL: QUALCOMM Adreno(TM) 750')
        self.assertFalse(result['model_offload_tested'])

    def test_prompt_processing_alone_is_not_generation(self):
        base = 'offloaded 29/29 layers to GPU\nOpenCL model buffer size = 739.03 MiB\n'
        with self.assertRaises(ValueError):
            OUTPUT.parse('inference', base + 'prompt eval time = 100.0 ms / 8 runs')
        result = OUTPUT.parse('inference', base + 'eval time = 100.0 ms / 31 runs')
        self.assertEqual(result['offloaded_layers'], 29)
        with self.assertRaises(ValueError):
            OUTPUT.parse('inference', base.replace('29/29', '0/29') + 'eval time = 100.0 ms / 31 runs')

class ElfChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='opencl-check-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bins = self.root / '.work/build/bin'
        bins.mkdir(parents=True)
        (self.root / '.work/logs').mkdir()
        (bins / 'libggml-opencl.so.0').write_bytes(b'\x7fELF fixture')
        options = dict(CMAKE_BUILD_TYPE='Release', GGML_OPENCL='ON', GGML_OPENCL_USE_SPHAL='ON',
            GGML_OPENCL_TARGET_VERSION='300', GGML_OPENCL_USE_ADRENO_KERNELS='OFF',
            GGML_OPENCL_USE_ADRENO_BIN_KERNELS='OFF', GGML_OPENCL_EMBED_KERNELS='ON',
            GGML_OPENCL_PROFILING='OFF', GGML_BACKEND_DL='OFF',
            CMAKE_HOME_DIRECTORY=str(self.root / '.work/llama.cpp'))
        (self.root / '.work/build/CMakeCache.txt').write_text(''.join(f'{k}:STRING={v}\n' for k,v in options.items()))

    def check_failure(self, dynamic='', versions='', symbols=''):
        def read(command, **kwargs):
            return {'--dynamic': dynamic, '--version-info': versions, '--dyn-syms': symbols}[command[1]]
        with patch.object(ELF.shutil, 'which', return_value='readelf'), patch.object(ELF.subprocess, 'check_output', side_effect=read):
            with self.assertRaises(ValueError):
                ELF.validate(self.root)

    def test_startup_opencl_link_is_rejected(self):
        self.check_failure(dynamic='0x1 (NEEDED) Shared library: [libOpenCL.so]')

    def test_symbol_versions_are_rejected(self):
        self.check_failure(versions='Name: OPENCL_1.2')

    def test_missing_project_dependency_is_rejected(self):
        self.check_failure(dynamic='0x1 (NEEDED) Shared library: [libllama.so.0]')

    def test_unresolved_opencl_api_is_rejected(self):
        self.check_failure(symbols='1: 0 0 FUNC GLOBAL DEFAULT UND clGetPlatformIDs')

class ShellChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='opencl scripts with spaces ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        common = (ROOT / 'scripts/common.sh').read_text() + '\ncheck_environment() { :; }\n'
        (self.root / 'scripts/common.sh').write_text(common)
        (self.root / 'upstream.json').write_text((ROOT / 'upstream.json').read_text())
        self.env = os.environ.copy()
        self.env['TEST_OUTPUT'] = str(self.root / 'trace.json')

    def script(self, relative, body):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        p.chmod(0o755)
        return p

    def run_script(self, relative, *args):
        return subprocess.run([BASH, str(self.root / relative), *args], env=self.env,
                              text=True, capture_output=True)

    def test_launcher_preserves_arguments_and_isolates_libraries(self):
        shutil.copy(ROOT / 'scripts/run-model.sh', self.root / 'scripts/run-model.sh')
        self.script('scripts/validate-elf.py', '# Transport fixture; ELF failures are tested separately.\n')
        (self.root / '.work').mkdir()
        (self.root / '.work/.owner').write_text('OpenCL-S24-Ultra generated data v1\n')
        self.script('.work/build/bin/llama-completion', f'#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["TEST_OUTPUT"]).write_text(json.dumps(dict(args=sys.argv[1:],ld=os.environ["LD_LIBRARY_PATH"],backend=os.environ.get("GGML_BACKEND_PATH"))))\n')
        for name in ['llama-bench', 'test-backend-ops']:
            self.script('.work/build/bin/' + name, '#!/bin/sh\nexit 0\n')
        (self.root / '.work/build/CMakeCache.txt').touch()
        model = self.root / 'model with spaces.gguf'
        model.write_bytes(b'GGUF test fixture')
        self.env['LD_LIBRARY_PATH'] = os.environ.get('PREFIX', '') + '/lib'
        self.env['GGML_BACKEND_PATH'] = '/unrelated/backend'
        result = self.run_script('scripts/run-model.sh', str(model), '-p', 'literal $HOME; spaces', '-ngl', '7')
        self.assertEqual(result.returncode, 0, result.stderr)
        trace = json.loads(Path(self.env['TEST_OUTPUT']).read_text())
        self.assertEqual(trace['args'][-4:], ['-p', 'literal $HOME; spaces', '-ngl', '7'])
        self.assertIn(str(model), trace['args'])
        self.assertEqual(trace['ld'], str(self.root / '.work/build/bin'))
        self.assertIsNone(trace['backend'])

    def test_missing_packages_are_installed_and_rerun_skips_build(self):
        shutil.copy(ROOT / 'install.sh', self.root / 'install.sh')
        self.script('scripts/check-device.sh', f'#!{BASH}\nexit 0\n')
        self.script('scripts/build.sh', f'#!{BASH}\nprintf "build\\n" >> "$TEST_OUTPUT"\n')
        self.script('scripts/verify-opencl.sh', f'#!{BASH}\nprintf "verify\\n" >> "$TEST_OUTPUT"\n')
        self.script('scripts/validate-source.py', '')
        self.script('tools/dpkg-query', f'#!{BASH}\nexit 1\n')
        self.script('tools/pkg', f'#!{BASH}\nprintf "%s\\n" "$*" >> "$TEST_OUTPUT"\n')
        self.env['PATH'] = str(self.root / 'tools') + os.pathsep + self.env['PATH']
        first = self.run_script('install.sh')
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.run_script('install.sh')
        self.assertEqual(second.returncode, 0, second.stderr)
        trace = Path(self.env['TEST_OUTPUT']).read_text()
        self.assertIn('install -y git clang cmake ninja python opencl-headers openssl', trace)
        self.assertEqual(trace.count('build\n'), 1)
        self.assertEqual(trace.count('verify\n'), 2)

    def test_uninstall_protects_models_and_unowned_data(self):
        shutil.copy(ROOT / 'uninstall.sh', self.root / 'uninstall.sh')
        work = self.root / '.work'
        work.mkdir()
        (work / 'keep.txt').write_text('unowned')
        self.assertNotEqual(self.run_script('uninstall.sh', '--yes').returncode, 0)
        self.assertTrue(work.is_dir())
        (work / '.owner').write_text('OpenCL-S24-Ultra generated data v1\n')
        model = work / 'keep.GGUF'
        model.touch()
        self.assertNotEqual(self.run_script('uninstall.sh', '--yes').returncode, 0)
        self.assertTrue(model.exists())
        model.unlink()
        source = work / 'llama.cpp'
        fixture = source / 'models/vocabulary.gguf'
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b'GGUF upstream vocabulary fixture')
        subprocess.run(['git', 'init', '-q', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        (self.root / 'upstream.json').write_text(json.dumps({'commit': commit}))
        self.assertEqual(self.run_script('uninstall.sh').returncode, 0)
        fixture.write_bytes(b'GGUF user modified fixture')
        self.assertNotEqual(self.run_script('uninstall.sh', '--yes').returncode, 0)
        self.assertTrue(fixture.exists())
        fixture.write_bytes(b'GGUF upstream vocabulary fixture')
        self.assertEqual(self.run_script('uninstall.sh').returncode, 0)
        self.assertTrue(work.exists())
        self.assertEqual(self.run_script('uninstall.sh', '--yes').returncode, 0)
        self.assertFalse(work.exists())

if __name__ == '__main__':
    unittest.main()
