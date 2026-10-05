"""Combine validated host prototypes in an isolated build; reuse unchanged DSP code."""
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

import measure

B = pathlib.Path(__file__).resolve().parent
R = B / 'npu-investigation'
S = B / 'source-npu-registration-hybrid'
D = B / 'build-npu-registration-hybrid'
old_source = B / 'source-npu-hybrid'
old_build = B / 'build-npu-hybrid'
baseline = '53d539ccc327858498c055909d19468954b90531'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with (B / 'resource.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    admission, _ = measure.cooldown(30, max_battery=36, max_cpu=50,
        max_gpu=43, max_npu=43, public_thermal=True, public_status_limit=0)
    epoch = measure.conditions()['power_epoch']
    if S.exists() or D.exists():
        raise RuntimeError('Isolated destination already exists; inspect it instead of overwriting')
    subprocess.run(['git', 'clone', '--shared', str(B / 'source-npu'), str(S)], check=True)
    subprocess.run(['git', '-C', str(S), 'switch', '-c', 'experiment/registration-hybrid', baseline], check=True)
    copied = ['include/llama.h', 'src/llama-context.cpp', 'src/llama-context.h',
        'src/llama-kv-cache.cpp', 'src/llama-kv-cache.h', 'src/llama-model-loader.cpp',
        'src/llama-model.cpp', 'src/llama-model.h']
    inputs = {name: digest(old_source / name) for name in copied}
    for name in copied:
        shutil.copy2(old_source / name, S / name)
    name = 'ggml/src/ggml-hexagon/ggml-hexagon.cpp'
    inputs[name] = digest(B / 'source-npu-registration' / name)
    shutil.copy2(B / 'source-npu-registration' / name, S / name)
    context = S / 'src/llama-context.cpp'
    text = context.read_text()
    before = '''    if (!kv || (model.arch != LLM_ARCH_QWEN2 && model.arch != LLM_ARCH_QWEN3)) {
        throw std::runtime_error("hybrid prototype supports plain Qwen2/Qwen3 KV only");
    }'''
    after = '''    const bool supported_arch = model.arch == LLM_ARCH_QWEN2 || model.arch == LLM_ARCH_QWEN3 ||
            (model.arch == LLM_ARCH_LLAMA && !model.hparams.is_swa_any());
    if (!kv || !supported_arch) {
        throw std::runtime_error("hybrid prototype requires supported plain KV without Llama SWA");
    }'''
    assert text.count(before) == 1
    context.write_text(text.replace(before, after))
    (D / 'bin').mkdir(parents=True)
    for path in (old_build / 'bin').glob('*.so*'):
        target = D / 'bin' / path.name
        if path.is_symlink():
            target.symlink_to(os.readlink(path))
        else:
            shutil.copy2(path, target)
    runtime = pathlib.Path(os.environ['PREFIX']) / 'lib/libc++_shared.so'
    (D / 'bin/libc++_shared.so').symlink_to(runtime)
    originals = json.loads((R / 'registration-hybrid-original-host-commands.json').read_text())
    compile_context = [x.replace(str(old_source), str(S)) for x in originals['context']]
    for flag in ['-MT', '-MF', '-o']:
        index = compile_context.index(flag) + 1
        compile_context[index] = str(D / ('llama-context.cpp.o.d' if flag == '-MF' else 'llama-context.cpp.o'))
    link_context = originals['link'].copy()
    for i, arg in enumerate(link_context):
        if arg == 'src/CMakeFiles/llama.dir/llama-context.cpp.o':
            link_context[i] = str(D / 'llama-context.cpp.o')
        elif arg == 'bin/libllama.so.0.5.0':
            link_context[i] = str(D / 'bin/libllama.so.0.5.0')
        elif arg.startswith('--dependency-file='):
            link_context[i] = '--dependency-file=' + str(D / 'libllama.link.d')
        elif arg.startswith('-Wl,-rpath,'):
            link_context[i] = '-Wl,-rpath,$ORIGIN'
    window = json.loads((R / 'registration-window-build.json').read_text())
    old_window_source = str(B / 'source-npu-registration')
    old_window_build = str(B / 'build-npu-registration')
    compile_window = [x.replace(old_window_source, str(S)).replace(old_window_build, str(D)) for x in window['compile']]
    link_window = [x.replace(old_window_build, str(D)) for x in window['link']]
    source = B / 'hybrid-latency-phase-v4.cpp'
    helper = [os.environ['PREFIX'] + '/bin/c++', '-O2', '-std=c++17', str(source),
        '-I' + str(S / 'include'), '-I' + str(S / 'ggml/include'), '-L' + str(D / 'bin'),
        '-Wl,-rpath,$ORIGIN', '-lllama', '-lggml', '-lggml-cpu', '-lggml-base',
        '-o', str(D / 'bin/phase-npu-registration')]
    commands = [compile_context, link_context, compile_window, link_window, helper]
    reused = {arg: digest(old_build / arg) for arg in originals['link']
        if arg.endswith('.o') and arg != 'src/CMakeFiles/llama.dir/llama-context.cpp.o'}
    dsp = B / 'build-npu/ggml/src/ggml-hexagon/libggml-htp-v75.so'
    manifest = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline': baseline, 'copied_source_sha256': inputs, 'commands': commands,
        'reused_objects_sha256': reused, 'unchanged_DSP_sha256': digest(dsp),
        'Termux_CXX_runtime_path': str(runtime), 'Termux_CXX_runtime_sha256': digest(runtime),
        'scope': 'Diagnostic host-only combination of existing validated registration window and one-way handoff. Adds only plain non-SWA LLAMA to architecture guard. Existing builds unchanged. Not clean-room installer evidence.'}
    (R / 'registration-hybrid-build-inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (R / 'registration-hybrid-build.log').open('w') as log:
        for index, command in enumerate(commands):
            print('Host-only build stage', index + 1, 'of', len(commands), flush=True)
            with measure.managed_process(['nice', '-n', '10', 'taskset', '03', *command],
                    cwd=old_build, stdout=log, stderr=subprocess.STDOUT) as proc:
                while proc.poll() is None:
                    policy = measure.conditions()
                    system = measure.system()
                    if any(policy.get(k) for k in ['phone_use_hold', 'battery_low_hold', 'investigation_paused_by_user']) or policy['power_epoch'] != epoch:
                        raise RuntimeError('Build stopped by user, battery or power epoch guard')
                    if system['meminfo_kB']['MemAvailable'] < 1536 * 1024 or system['temperatures_C'].get('battery', 0) >= 43:
                        raise RuntimeError('Build stopped by memory or battery-temperature guard')
                    time.sleep(1)
                if proc.returncode:
                    raise RuntimeError('Host build failed; inspect registration-hybrid-build.log')
    for path in (D / 'bin').glob('*.so*'):
        if not path.is_symlink():
            subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(path)], check=True)
    destination = D / 'ggml/src/ggml-hexagon/libggml-htp-v75.so'
    destination.parent.mkdir(parents=True)
    shutil.copy2(dsp, destination)
    assert digest(destination) == manifest['unchanged_DSP_sha256']
    patch = subprocess.check_output(['git', '-C', str(S), 'diff', '--binary', 'HEAD'])
    (R / 'registration-hybrid.patch').write_bytes(patch)
    result = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': 0,
        'patch_sha256': hashlib.sha256(patch).hexdigest(),
        'artifacts_sha256': {path.name: digest(path) for path in (D / 'bin').iterdir() if path.is_file() and not path.is_symlink()}}
    (R / 'registration-hybrid-build-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Isolated host-only registration/handoff build complete', flush=True)
