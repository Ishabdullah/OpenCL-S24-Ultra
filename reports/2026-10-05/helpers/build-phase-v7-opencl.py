"""Build the same phase helper against isolated generic OpenCL libraries."""
import datetime,fcntl,hashlib,json,os,pathlib,subprocess
B=pathlib.Path(__file__).resolve().parent;R=B/'npu-investigation';S=B/'source-hybrid';D=B/'build-hybrid/bin';source=B/'hybrid-latency-phase-v7-batch-threads.cpp';O=B/'phase-tools-v7';target=O/'phase-opencl-batch-threads'
print('Waiting for serialized helper build',flush=True)
with (B/'resource.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);O.mkdir(exist_ok=True)
 if target.exists():raise SystemExit('Preserve existing helper; choose a new version for rebuild')
 cmd=[os.environ['PREFIX']+'/bin/c++','-O2','-std=c++17',str(source),'-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-L'+str(D),'-Wl,-rpath,'+str(D),'-lllama','-lggml','-lggml-cpu','-lggml-base','-o',str(target)]
 with (R/'phase-v7-opencl-build.log').open('w') as f:r=subprocess.run(['nice','-n','10','taskset','03',*cmd],stdout=f,stderr=subprocess.STDOUT)
 d={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'exit_code':r.returncode,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Standalone helper only; generic backend, source and existing binaries unchanged. Same phase/KV/handoff ABI inspected in both headers.'}
 if not r.returncode:d['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
 (R/'phase-v7-opencl-build.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
 raise SystemExit(r.returncode)
