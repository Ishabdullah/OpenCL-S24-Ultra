"""Curate source and compact measurements; no model/runtime/toolchain redistribution."""
import csv
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

B = pathlib.Path(__file__).resolve().parent
R = B / 'npu-investigation'
P = B.parents[1] / 'OpenCL-S24-Ultra'
O = P / 'reports/2026-10-04'
E = 'api-20261004T141923Z-UNPLUGGED'
os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))
os.sched_setaffinity(0, {0, 1})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, str):
        return value.replace(str(pathlib.Path.home()), '~').replace(os.environ.get('PREFIX', '/data/data/com.termux/files/usr'), '$PREFIX')
    if isinstance(value, list):
        return [normalized(v) for v in value]
    if isinstance(value, dict):
        return {normalized(k): normalized(v) for k, v in value.items()}
    return value


manifest = []


def publish(source, target):
    assert source.is_file(), source
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == '.json':
        target.write_text(json.dumps(normalized(json.loads(source.read_text())), indent=2) + '\n')
    elif source.suffix == '.csv':
        target.write_text(normalized(source.read_text()))
    else:
        shutil.copyfile(source, target)
    manifest.append({'source': normalized(str(source)), 'source_sha256': sha(source), 'export': str(target.relative_to(O)), 'export_sha256': sha(target)})


names = [
    'fresh-bestcpu-matched-resume-20261004.json',
    'phase-v4-temperature-ready-20261004T062756Z-plan.summary.json',
    'cadence600-two-order-summary.json',
    'cadence600-foreground-20261004T151529Z-results.json',
    'cadence600-foreground-20261004T151529Z-results.csv',
    'cadence600-reverse-20261004T154432Z-results.json',
    'cadence600-reverse-20261004T154432Z-results.csv',
    'architecture-memory-canonical-summary.json',
    'architecture-memory-canonical-summary.csv',
    'mistral7-phase2048-screen-summary.json',
    'mistral7-handoff-qualified-summary.json',
    'mistral7-canonical-partial-response-screen.json',
    'mistral7-canonical-partial-quality-summary.json',
    'mmap-registration-controls-resume-20261004.json',
    'registration-retained-storage-control-resume-20261004.json',
    'automatic-registration-window-control-resume-20261004.json',
    'registration-window-first-llm-quality.json',
    'registration-window-qwen4-quality.json',
    'registration-window-mistral7-quality.json',
    'full7-first-response-screen.json',
    'full7-registration-profile-background-excluded.json',
    'registration-wall-time-bottleneck-lead.json',
    'registration-nomap-negative-controls.json',
    'specialized-workaround-config-audit.json',
    'specialized-corrected-faon-quality.json',
    'registration-hybrid-build-inputs.json',
    'registration-hybrid-build-result.json',
    'registration-hybrid-runtime-fix.json',
    'registration-hybrid-qualification-helper.json',
    'phase-canonical-helper-build.json',
    'mistral7-CPU2048-charging-tuning-summary.json',
    'final-preservation-check-oct04.json',
    'USER_PAUSE_OCT04.json',
]
for name in names:
    publish(R / name, O / 'data' / name)
for name in ['cadence-quality-records.json', 'request-responses.json', 'request-responses.csv', 'numerical-gates.json', 'numerical-gates.csv', 'phase-responses.csv', 'phase-responses.json', 'phase-records.json']:
    publish(R / ('results-' + E) / name, O / 'data' / name)
publish(B / 'models.json', O / 'data/model-metadata.json')
charging = 'api-recovered-20261004T213622Z-PLUGGED_AC'
for name in ['phase-responses.csv', 'phase-responses.json', 'phase-records.json']:
    publish(R / ('results-' + charging) / name, O / 'data/charging' / name)
for pointer in ['active-mistral-CPU2048-charging-plan.txt', 'active-mistral-charging-confirm-plan.txt']:
    plan = pathlib.Path((R / pointer).read_text().strip())
    publish(plan, O / 'data/plans' / plan.name)
    publish(plan.with_suffix('.state.json'), O / 'data/plans' / plan.with_suffix('.state.json').name)
for source in sorted((B / 'runs').glob('architecture-memory-canonical-20261004T184708Z-*/result.json')):
    record = json.loads(source.read_text())
    record.pop('samples', None)
    command = json.loads(source.with_name('command.json').read_text())
    target = O / 'data/memory-records' / (source.parent.name + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized({'result_without_full_telemetry': record, 'command_provenance': command, 'raw_result_sha256': sha(source)}), indent=2) + '\n')
    manifest.append({'source': normalized(str(source)), 'source_sha256': sha(source), 'export': str(target.relative_to(O)), 'export_sha256': sha(target)})
for name in ['registration-hybrid.patch', 'registration-window.patch']:
    publish(R / name, O / 'patches' / name)
for name in ['hybrid-latency-phase-v3.cpp', 'hybrid-latency-phase-v4.cpp', 'hybrid-latency-phase-v4-canonical.cpp', 'cadence-requests-v3.cpp', 'android-thermal-probe.c', 'mmap-registration-probe.cpp', 'measure.py', 'run-controlled-cases.py', 'build-registration-hybrid.py', 'validation.py', 'export-phase-results.py', 'export-cadence-quality.py', 'summarize-cadence.py', 'export-oct04-checkpoint.py']:
    publish(B / name, O / 'helpers' / name)

base = pathlib.Path.home() / 'npu-installer-cleanroom-20261004/.work-npu/llama.cpp'
check = subprocess.run(['git', '-C', str(base), 'apply', '--check', str(R / 'registration-hybrid.patch')], capture_output=True, text=True)
assert check.returncode == 0, check.stderr
patch_meta = {
    'upstream': 'https://github.com/ggml-org/llama.cpp.git',
    'upstream_commit': 'e358d59178377be4c58ba567925e05faadbccb57',
    'local_base_checkpoint': '53d539ccc327858498c055909d19468954b90531',
    'published_base_patch': 'patches/llama.cpp-hexagon-sphal.patch',
    'apply_check': {'returncode': check.returncode, 'scope': 'Read-only check against fresh installer-patched source; no new clean-room compile/inference'},
    'combined_patch_sha256': sha(R / 'registration-hybrid.patch'),
    'standalone_window_patch': 'Alternative single-file experiment; do not apply both window and combined patches',
    'build_scope': 'Isolated native host rebuild reused 41 unchanged preserved objects and unchanged v75 DSP. Not clean-room validation.',
}
(O / 'patches/manifest.json').write_text(json.dumps(patch_meta, indent=2) + '\n')
(O / 'data/export-provenance.json').write_text(json.dumps({'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'records': manifest, 'scope': 'Compact evidence/source only; local absolute paths normalized. Raw large logs and full telemetry remain on phone.'}, indent=2) + '\n')
print('Curated', len(manifest), 'source/data files; patch applies to fresh installer-patched source')
