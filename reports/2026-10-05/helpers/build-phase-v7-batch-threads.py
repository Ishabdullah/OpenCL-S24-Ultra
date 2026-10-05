"""Build only the optional batch/decode thread helper under the resource lock."""
import datetime,fcntl,hashlib,json,os,pathlib,subprocess
B=pathlib.Path(__file__).resolve().parent;R=B/'npu-investigation';S=B/'source-npu-registration-hybrid';D=B/'build-npu-registration-hybrid/bin';source=B/'hybrid-latency-phase-v7-batch-threads.cpp';target=D/'phase-npu-batch-threads'
with (B/'resource.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not target.exists()
 command=[os.environ['PREFIX']+'/bin/c++','-O2','-std=c++17',str(source),'-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-L'+str(D),'-Wl,-rpath,$ORIGIN','-lllama','-lggml','-lggml-cpu','-lggml-base','-o',str(target)]
 with (R/'phase-v7-build.log').open('w') as log:p=subprocess.run(['nice','-n','10','taskset','03',*command],stdout=log,stderr=subprocess.STDOUT)
 result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'exit_code':p.returncode,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 if p.returncode==0:result['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
 (R/'phase-v7-build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);raise SystemExit(p.returncode)
