"""Validate pinned NPU source, ELF state and explicit execution evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


def metadata(root):
    meta = json.loads((root / 'npu-upstream.json').read_text())
    patch = root / 'patches/llama.cpp-hexagon-sphal.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != meta['patch_sha256']:
        raise ValueError('NPU patch checksum differs from npu-upstream.json')
    return meta


def source(root):
    meta = metadata(root)
    tree = root / '.work-npu/llama.cpp'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tree), *args], text=True).strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != tree.resolve():
        raise ValueError('NPU source is not a separate checkout')
    if Path(git('rev-parse', '--absolute-git-dir')).resolve() != (tree / '.git').resolve() or (tree / '.git').is_symlink():
        raise ValueError('Source Git directory is outside this checkout or symlinked')
    if git('rev-parse', 'HEAD') != meta['commit'] or git('remote', 'get-url', 'origin') != meta['repository']:
        raise ValueError('Source commit/origin differs from pin')
    changes = subprocess.check_output(['git', '-C', str(tree), 'ls-files', '-m', '-o', '--exclude-standard', '-z']).decode().split('\0')
    unexpected = set(changes) - set(meta['patched_files_sha256']) - {''}
    if unexpected:
        raise ValueError('Unexpected NPU source changes: ' + ', '.join(sorted(unexpected)))
    for name, sha in meta['patched_files_sha256'].items():
        path = tree / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Patched source differs: ' + name)
    return {'pinned_source_verified': True, 'commit': meta['commit'], 'patched_files': list(meta['patched_files_sha256'])}


def elf(root):
    build = root / '.work-npu/build'
    bins = build / 'bin'
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if '=' in line and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            cache[key.split(':')[0]] = value
    for name, value in {'CMAKE_BUILD_TYPE': 'Release', 'GGML_HEXAGON': 'ON',
                        'GGML_HEXAGON_USE_SPHAL': 'ON', 'GGML_HEXAGON_ARCHITECTURES': 'v75',
                        'GGML_HEXAGON_HTP_BUILD_TYPE': 'Release', 'GGML_OPENCL': 'OFF',
                        'GGML_BACKEND_DL': 'OFF',
                        'CMAKE_HOME_DIRECTORY': str(root / '.work-npu/llama.cpp')}.items():
        if cache.get(name) != value:
            raise ValueError(name + ' differs from the proven NPU configuration')
    reader = shutil.which('llvm-readelf')
    if not reader:
        raise ValueError('llvm-readelf missing; Termux clang/llvm is required')
    backend = bins / 'libggml-hexagon.so.0'
    dsp = build / 'ggml/src/ggml-hexagon/libggml-htp-v75.so'
    records = {}
    for path in [p for p in sorted(bins.iterdir()) if p.is_file() and not p.is_symlink()] + [dsp]:
        with path.open('rb') as f:
            header = f.read(64)
        if header[:4] != b'\x7fELF':
            continue
        expected_machine, expected_class = (164, 1) if path == dsp else (183, 2)
        if header[4] != expected_class or struct.unpack_from('<H', header, 18)[0] != expected_machine:
            raise ValueError('Unexpected ELF architecture: ' + path.name)
        if path == dsp and struct.unpack_from('<I', header, 36)[0] & 0xff != 0x75:
            raise ValueError('DSP ELF does not declare v75')
        dynamic = subprocess.check_output([reader, '--dynamic', str(path)], text=True)
        needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
        if any(n.startswith(('libOpenCL', 'libcdsprpc', 'libadsprpc')) for n in needed):
            raise ValueError('Startup vendor/OpenCL dependency: ' + path.name)
        for name in needed:
            if name.startswith(('libllama', 'libggml')):
                if not (bins / name).is_file() or (bins / name).resolve().parent != bins.resolve():
                    raise ValueError('Project llama dependency missing/outside build: ' + name)
        records[path.name] = {'DT_NEEDED': needed, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    versions = subprocess.check_output([reader, '--version-info', str(backend)], text=True)
    if re.search(r'OPENCL_\w+', versions):
        raise ValueError('Unexpected OpenCL symbol-version requirement')
    if not backend.is_file() or not dsp.is_file():
        raise ValueError('Host backend or native DSP kernel library missing')
    return {'ELF_verified': True, 'native_ARM64_host': True, 'native_Hexagon_v75_DSP': True,
            'no_startup_vendor_or_OpenCL_dependency': True, 'artifacts': records}


def evidence(mode, text):
    if mode == 'device':
        match = re.search(r'^\s*HTP0:\s*(.*Hexagon.*)$', text, re.M)
        if not match:
            raise ValueError('HTP0 Hexagon device not detected; CPU-only enumeration is not a pass')
        if 'Hexagon Arch version v75' not in text:
            raise ValueError('Expected detected Hexagon architecture v75')
        return {'NPU_detected': True, 'device': match[1], 'model_offload_tested': False}
    if mode == 'matrix':
        if 'Backend HTP0:' not in text or 'HTP0 new session' not in text:
            raise ValueError('Numerical pass must name HTP0 and a successfully opened native DSP session')
        matches = re.findall(r'(\d+)/(\d+) tests passed', text)
        if not matches or not any(int(a) == int(b) == 8 for a, b in matches):
            raise ValueError('Expected 8/8 supported Q4_K/Q6_K numerical matrix tests')
        return {'native_NPU_matrix_execution_confirmed': True, 'passed': 8, 'supported': 8,
                'model_offload_tested': False}
    if mode == 'inference':
        layers = re.search(r'offloaded (\d+)/(\d+) layers to GPU', text)
        weights = re.search(r'HTP0 model buffer size\s*=\s*([\d.]+) MiB', text)
        generation = re.search(r'(?<!prompt )eval time\s*=\s*[\d.]+ ms /\s*(\d+) runs', text)
        if not layers or int(layers[1]) <= 0 or not weights or float(weights[1]) <= 0:
            raise ValueError('Positive NPU layer offload and HTP weights not confirmed')
        if not generation or int(generation[1]) <= 0:
            raise ValueError('No completed token-generation evaluations')
        return {'model_offload_tested': True, 'backend': 'HTP0',
                'offloaded_layers': int(layers[1]), 'total_layers': int(layers[2]),
                'HTP_weights_MiB': float(weights[1]), 'generation_evaluations': int(generation[1]),
                'process_exit_code': 0}
    raise ValueError('Unknown verification mode')


if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        root = Path(sys.argv[2]).resolve()
        if mode == 'metadata':
            meta = metadata(root)
            print(meta['repository']); print(meta['commit'])
            raise SystemExit(0)
        result = source(root) if mode == 'source' else elf(root) if mode == 'elf' else evidence(mode, Path(sys.argv[3]).read_text())
        logs = root / '.work-npu/logs'
        logs.mkdir(exist_ok=True)
        (logs / (mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        summary = {key: value for key, value in result.items() if key != 'artifacts'}
        if 'artifacts' in result:
            summary['artifacts_checked'] = len(result['artifacts'])
            summary['details'] = str(logs / (mode + '.json'))
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit('NPU verification failed: ' + str(error))
