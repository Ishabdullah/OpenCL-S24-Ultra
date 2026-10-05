"""Export compact performance evidence without runtime binaries or SDK files."""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

B = pathlib.Path(__file__).resolve().parent
R = B / 'npu-investigation'
P = pathlib.Path.home() / 'OpenCL-S24-Ultra'
O = P / 'reports/2026-10-05'
os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))
os.sched_setaffinity(0, {0, 1})
manifest = []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def normalize(v):
    if isinstance(v, str):
        return v.replace(str(pathlib.Path.home()), '~').replace(os.environ.get('PREFIX', '/data/data/com.termux/files/usr'), '$PREFIX')
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {normalize(k): normalize(x) for k, x in v.items()}
    return v

def publish(p, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == '.json':
        target.write_text(json.dumps(normalize(json.loads(p.read_text())), indent=2) + '\n')
    elif p.suffix == '.csv':
        target.write_text(normalize(p.read_text()))
    else:
        shutil.copyfile(p, target)
    manifest.append({'source': normalize(str(p)), 'source_sha256': sha(p), 'export': str(target.relative_to(O)), 'export_sha256': sha(target)})

names = ['mistral7-2K-confirmed-oct05.json', 'mistral7-2K-confirmed-oct05.csv',
         'mistral7-long-step-conclusion-oct05.json', 'mistral7-long-step-screen-oct05.csv',
         'mistral7-short-step-conclusion.json', 'mistral7-partial-screen-oct05.json',
         'mistral7-partial-screen-oct05.csv', 'mistral7-placement-screen-oct05.json',
         'mistral7-placement-screen-oct05.csv', 'registration-eviction-policy-inspection-oct05.json',
         'kv-memory-backing-inspection-oct05.json', 'openmp-placement-probe-oct05.json',
         'placement-protocol-correction-oct05.json', 'phase-v7-design-oct05.json',
         'phase-v7-opencl-build.json', 'cpu-idle-NPU-session-falsification-oct05.json',
         'qwen4-CPU-thread-screen-incomplete-oct05.json',
         'phase-v7-build.json', 'resume-oct05-preservation-check.json',
         'step-profile-scheduling-censor-correction.json', 'qwen4-power-transition-oct05.json',
         'phase-v8-build.json', 'virtual-session-design-oct05.json',
         'virtual-session-order-conclusion-oct05.json', 'generic-8K-initial-scheduling-exclusion-oct05.json',
         'cpu-opencl-registry-inspection-oct05.json', 'preservation-checkpoint-oct05.json',
         'one-way-scheduler-limits-oct05.json', 'generic-8K-retry-scheduling-exclusion-oct05.json',
         'three-test-memory-selection-oct05.json', 'three-test-outcome-oct05.json',
         'opencl-flash-attention-fallback-inspection-oct05.json']
for name in names:
    p = R / name
    if p.exists():
        publish(p, O / 'data' / name)
plans = sorted(p for p in R.glob('*20261005*-schedule.json') if 'T00' not in p.name)
for plan in plans:
    publish(plan, O / 'data/plans' / plan.name)
    for p in R.glob(plan.stem + '.*'):
        if p != plan and p.suffix in ['.json', '.csv']:
            publish(p, O / 'data/plans' / p.name)
    for c in json.loads(plan.read_text()):
        source = B / 'runs' / c['id'] / 'result.json'
        if not source.exists():
            continue
        result = json.loads(source.read_text())
        result.pop('samples', None)
        command = source.with_name('command.json')
        target = O / 'data/records' / (c['id'] + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(normalize({'result_without_full_telemetry': result,
            'command_provenance': json.loads(command.read_text()) if command.exists() else None,
            'raw_result_sha256': sha(source)}), indent=2) + '\n')
        manifest.append({'source': normalize(str(source)), 'source_sha256': sha(source), 'export': str(target.relative_to(O)), 'export_sha256': sha(target)})
for name in ['hybrid-latency-phase-v7-batch-threads.cpp', 'hybrid-latency-phase-v6-step-checkpoint.cpp',
             'hybrid-latency-phase-v5-step-profile.cpp', 'hybrid-latency-phase-v4-canonical.cpp',
             'openmp-placement-probe.cpp', 'build-phase-v7-batch-threads.py',
             'placement_measure.py', 'placement_measure_dual.py', 'summarize-phase-plan-oct05.py',
             'summarize-step-profile.py', 'summarize-phase-telemetry.py',
             'summarize-registration-profile.py', 'confirm-mistral-long-response-oct05.py',
             'build-phase-v7-opencl.py', 'hybrid-latency-phase-v8-virtual-sessions.cpp',
             'build-phase-v8-virtual-sessions.py', 'npu-virtual-session-order.cpp',
             'run-virtual-session-order-oct05.py', 'run-virtual-session-strace-oct05.py',
             'run-background-phases-oct05.py', 'run-numerical-phase-gates-oct05.py', 'export-oct05-performance.py']:
    p = B / name
    if p.exists():
        publish(p, O / 'helpers' / name)
current = (B / 'OPENCL_PERFORMANCE_ANALYSIS.md').read_text().split('\n---\n\n<!-- OCT04_CURRENT_REPORT -->')[0]
history = O / 'HISTORICAL_REPORT_THROUGH_OCT04.md'
if not history.exists():
    prior = subprocess.run(['git', '-C', str(P), 'show', 'f07f85d2b348289288125258312defa3a7744703:OPENCL_PERFORMANCE_ANALYSIS.md'], capture_output=True, text=True, check=True)
    history.write_text(prior.stdout)
intro = '\n\nThe investigation remains incomplete. New compact results, exact plans, command/artifact provenance and source helpers are in [October5 data](reports/2026-10-05/data/) and [helpers](reports/2026-10-05/helpers/). [Remaining work](reports/2026-10-05/NEXT_WORK.md) lists the pending controlled measurements. Experimental source patches remain separate from installer defaults.\n'
(P / 'OPENCL_PERFORMANCE_ANALYSIS.md').write_text(current + intro + '\n---\n\n# Historical report through October4\n\nThe following snapshot is retained unchanged for provenance. Its progress/installer/fastest-configuration statements are dated; the October4 archive and current findings above supersede them.\n\n' + history.read_text())
(O / 'README.md').write_text(current + '\n\n[Compact data and exact plans](data/) | [Helper source](helpers/) | [Remaining work](NEXT_WORK.md) | [Historical report](HISTORICAL_REPORT_THROUGH_OCT04.md) | [October4 experimental source patches](../2026-10-04/patches/README.md)\n\nPaths under npu-investigation in the prose identify original local files; corresponding CSV/JSON and command provenance are curated in data. This checkpoint is research evidence, not a new installer default or complete characterization.\n')
(O / 'data/export-provenance.json').write_text(json.dumps({
    'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'records': manifest,
    'scope': 'Compact evidence and original helper source only. Raw stderr and full telemetry remain local. No SDK, model, vendor library or compiled executable. Earlier experimental source patches remain in reports/2026-10-04/patches; helpers require that experimental workspace. Not an installer default or new clean-room test.'}, indent=2) + '\n')
print('Exported', len(manifest), 'compact evidence/source files; frozen October4 archive untouched')
