"""Resumable sequential measurements with raw commands, thermal/memory samples and exit status."""
import argparse,contextlib,datetime,fcntl,hashlib,json,os,pathlib,re,signal,statistics,subprocess,time
BASE=pathlib.Path(__file__).resolve().parent
HARNESS_SOURCE_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
HARNESS_TELEMETRY_REVISION='public-thermal-admission-light-v1'
SOURCE=BASE.parent
BIN=SOURCE/'build-opencl-generic/bin'
MODELS={m['id']:m for m in json.loads((BASE/'models.json').read_text())}
THERMALS={p.parent.name:p.read_text().strip() for p in pathlib.Path('/sys/class/thermal').glob('thermal_zone*/type')}
COOLING_FILE=BASE/'npu-investigation/cooling-monitors.json'
COOLING=json.loads(COOLING_FILE.read_text()) if COOLING_FILE.exists() else {}
class AdmissionDeferred(RuntimeError):pass
def read(p):
 try:return pathlib.Path(p).read_text().strip()
 except OSError:return None
def fields(text):return {k:int(v) for k,v in re.findall(r'^(\w+):\s+(\d+)',text or '',re.M)}
def terminate_group(proc):
 if proc.poll() is None:
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:proc.wait(timeout=8)
  except subprocess.TimeoutExpired:
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   proc.wait()
@contextlib.contextmanager
def managed_process(command,**kwargs):
 proc=subprocess.Popen(command,start_new_session=True,**kwargs)
 try:yield proc
 finally:terminate_group(proc)
class ThermalMonitor:
 def __init__(self,config,out):self.enabled=config.get('public_thermal',False);self.out=out;self.proc=None;self.files=[]
 def __enter__(self):
  if self.enabled:
   env=os.environ.copy()
   for name in ['LD_LIBRARY_PATH','LD_PRELOAD','GGML_BACKEND_PATH']:env.pop(name,None)
   probe=BASE/'npu-investigation/android-thermal-probe'
   self.files=[(self.out/'public-thermal.jsonl').open('w'),(self.out/'public-thermal-stderr.txt').open('w')]
   self.proc=subprocess.Popen(['nice','-n','10','taskset','03',str(probe),'1000'],env=env,stdout=self.files[0],stderr=self.files[1],start_new_session=True)
  return self
 def rows(self):
  path=self.out/'public-thermal.jsonl'
  rows=[]
  if self.enabled and path.exists():
   for line in path.read_text().splitlines():
    try:rows.append(json.loads(line))
    except ValueError:pass
  return rows
 def latest(self):
  rows=self.rows()
  return rows[-1] if rows else None
 def __exit__(self,*args):
  if self.proc:terminate_group(self.proc)
  for file in self.files:file.close()
def system():
 temps={name:int(v)/1000 for zone,name in THERMALS.items() if (v:=read('/sys/class/thermal/'+zone+'/temp')) is not None and -100000<int(v)<200000}
 gpu='/sys/class/kgsl/kgsl-3d0/'
 return {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic':time.monotonic(),'temperatures_C':temps,'cooling_device_states':{name:read('/sys/class/thermal/'+device+'/cur_state') for device,name in COOLING.items()},'meminfo_kB':fields(read('/proc/meminfo')),'cpu_kHz':{str(i):read(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq') for i in range(8)},'cpu_max_kHz':{str(i):read(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq') for i in range(8)},'cpu_online':read('/sys/devices/system/cpu/online'),'harness_allowed_CPUs':sorted(os.sched_getaffinity(0)),'harness_cgroup':read('/proc/self/cgroup'),'cpusets':{name:read('/dev/cpuset/'+name+'/cpus') for name in ['top-app','foreground','background']},'gpu':{name:read(gpu+name) for name in ['clock_mhz','max_gpuclk','thermal_pwrlevel','gpu_busy_percentage','gpubusy','temp','reset_count']},'loadavg':read('/proc/loadavg')}
def conditions():
 state=json.loads((BASE/'conditions.json').read_text())
 deadline=state.get('planned_pause_UTC')
 if deadline and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(deadline):state['investigation_paused_by_user']=True
 return state
def process(pid):
 workers={}
 for p in pathlib.Path(f'/proc/{pid}/task').glob('*/stat'):
  value=read(p)
  if value:
   parts=value[value.rfind(')')+2:].split()
   if len(parts)>36:workers[p.parent.name]={'processor':int(parts[36]),'utime':int(parts[11]),'stime':int(parts[12]),'priority':int(parts[15]),'nice':int(parts[16])}
 status=read(f'/proc/{pid}/status') or ''
 allowed=re.search(r'^Cpus_allowed_list:\s*(.+)$',status,re.M)
 return {'status_kB':fields(status),'allowed_CPUs':allowed[1] if allowed else None,'cgroup':read(f'/proc/{pid}/cgroup'),'smaps_kB':fields(read(f'/proc/{pid}/smaps_rollup')),'stat':read(f'/proc/{pid}/stat'),'workers':workers}
def cooldown(seconds=15, max_battery=38, max_cpu=65, max_gpu=47, timeout=600, trace_path=None, required_CPUs=None, defer_on_CPUs=False, max_npu=None, public_thermal=False, public_status_limit=1, public_headroom_limit=None):
 start=time.monotonic();deadline=start+timeout;last_notice=start;last_trace=0;last_public_check=0;public_status=None
 while True:
  state=system();temps=state['temperatures_C'];cpu=max([v for k,v in temps.items() if k.startswith('cpu-')]+[0]);gpu=max([v for k,v in temps.items() if k.startswith('gpuss-')]+[0]);npu=max([v for k,v in temps.items() if k.startswith(('nsphmx-','nsphvx-'))]+[0])
  policy=conditions()
  if public_thermal and time.monotonic()-last_public_check>=10:
   probe_env=os.environ.copy()
   for key in ['LD_LIBRARY_PATH','LD_PRELOAD','GGML_BACKEND_PATH']:probe_env.pop(key,None)
   checked=subprocess.run(['nice','-n','10','taskset','03',str(BASE/'npu-investigation/android-thermal-probe')],env=probe_env,capture_output=True,text=True,timeout=8)
   try:public_status=json.loads(checked.stdout) if checked.returncode==0 else None
   except ValueError:public_status=None
   if public_status is None:raise RuntimeError('Requested public thermal telemetry unavailable during admission')
   last_public_check=time.monotonic()
  public_ready=not public_thermal or (public_status is not None and public_status['thermal_status']<=public_status_limit and (public_headroom_limit is None or (public_status['thermal_headroom'] is not None and public_status['thermal_headroom']<=public_headroom_limit)))
  if policy.get('battery_low_hold'):
   raise AdmissionDeferred('Low battery hold: release admission resources until power conditions recover.')
  if policy.get('investigation_paused_by_user'):
   raise AdmissionDeferred('Investigation paused by user: release admission resources.')
  if policy.get('phone_use_hold'):
   raise AdmissionDeferred('User phone-use hold: demanding measurements remain paused until explicitly resumed.')
  earliest=policy.get('measurement_not_before_UTC')
  power_ready=not earliest or datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(earliest)
  if trace_path and time.monotonic()-last_trace>=30:
   with pathlib.Path(trace_path).open('a') as trace:
    trace.write(json.dumps({'UTC':state['UTC'],'battery_C':temps.get('battery'),'CPU_max_C':cpu,'GPU_max_C':gpu,'NPU_max_C':npu,'gpu_max_clock':state['gpu']['max_gpuclk'],'power_ready':power_ready,'public_thermal':public_status})+'\n')
   last_trace=time.monotonic()
  cpus_ready=required_CPUs is None or set(required_CPUs)<=set(state['harness_allowed_CPUs'])
  if defer_on_CPUs and not cpus_ready:raise AdmissionDeferred('Required CPU cores unavailable; release resource lock while waiting')
  if power_ready and cpus_ready and public_ready and time.monotonic()-start>=seconds and temps.get('battery',0)<=max_battery and cpu<=max_cpu and gpu<=max_gpu and (max_npu is None or npu<=max_npu):
   state['public_thermal_admission']=public_status
   return state,time.monotonic()-start
  if time.monotonic()>deadline:raise RuntimeError(f'Cooling/power stabilization criterion not met in {timeout} seconds; preserve results and retry after device cools.')
  if time.monotonic()-last_notice>=60:
   print(f'Admission: battery={temps.get("battery")} C CPU={cpu} C GPU={gpu} C power_ready={power_ready} available_CPUs={state["harness_allowed_CPUs"]} CPUs_ready={cpus_ready}',flush=True)
   last_notice=time.monotonic()
  time.sleep(2)
def allocations(text):
 result=[]
 pattern=r'([^\n]*?)(CPU_REPACK|CPU_Mapped|CPU|OpenCL|HTP[0-9]+(?::[0-9]+)?)\s+(model|KV|compute|RS|recurrent state|output) buffer size\s*=\s*([\d.]+) MiB'
 for m in re.finditer(pattern,text):result.append({'backend':m[2],'kind':m[3],'MiB':float(m[4]),'line':m[0].strip()})
 offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',text)
 kv=re.findall(r'[^\n]*(?:KV buffer|recurrent|state buffer|memory breakdown|n_ctx\s*=|model params\s*=)[^\n]*',text)
 return {'buffers':result,'layers_offloaded':[[int(a),int(b)] for a,b in offload],'context_and_memory_lines':kv}
def run_impl(config):
 config=dict(config)
 policy=conditions()
 config.setdefault('execution_policy',policy.get('execution_policy','legacy-nice0'))
 config.setdefault('cpu_nice_increment',policy.get('cpu_nice_increment',0))
 config['harness_nice']=os.getpriority(os.PRIO_PROCESS,0)
 config['expected_worker_nice']=min(19,config['harness_nice']+config['cpu_nice_increment'])
 if not config.get('validation_timing_not_comparable') and config['harness_nice']!=0:
  raise RuntimeError('Start the measurement controller at nice0; it applies the recorded nice adjustment exactly once to the benchmark child.')
 config.setdefault('minimum_MemAvailable_MiB',policy.get('minimum_MemAvailable_MiB',384))
 config['cooldown']=max(config.get('cooldown',15),policy.get('minimum_cooldown_seconds',15))
 if config.get('mode') not in ['memory','logits','ops','ops-perf']:config.setdefault('delay',10)
 label=config['id'];out=BASE/'runs'/label;out.mkdir(exist_ok=True)
 if (out/'result.json').exists():print('Already recorded:',label,flush=True);return json.loads((out/'result.json').read_text())
 model=MODELS[config['model']];ngl=config.get('ngl',0);threads=config.get('threads',4);batch=config.get('batch',128);ub=config.get('ubatch',batch)
 binary=pathlib.Path(config.get('binary',str(BIN/('llama-completion' if config.get('mode')=='memory' else 'llama-bench'))))
 library_directory=pathlib.Path(config.get('library_directory',str(binary.parent)))
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(library_directory);env['GGML_OPENCL_KERNEL_CACHE_DIR']=str(BASE/'kernel-cache');env.pop('GGML_BACKEND_PATH',None)
 env.pop('LD_PRELOAD',None)
 for k in list(env):
  if (k.startswith('GGML_OPENCL_') and k!='GGML_OPENCL_KERNEL_CACHE_DIR') or k.startswith('GGML_HEXAGON_'):env.pop(k)
 env.update(config.get('environment',{}))
 common=['-m',model['path'],'-ngl',str(ngl),'-dev',config.get('device','GPUOpenCL') if ngl else 'none','-t',str(threads),'-b',str(batch),'-ub',str(ub),'-fa',config.get('attention','off'),'-ctk',config.get('cache_k','f16'),'-ctv',config.get('cache_v','f16')]
 if config.get('mode') in ['ops','ops-perf']:
  command=[str(binary),'perf' if config.get('mode')=='ops-perf' else 'test','-b',config.get('backend',config.get('device','GPUOpenCL')),'-o','MUL_MAT','--test-file',config['test_file'],'-j','1']
 elif config.get('mode')=='logits':
  command=[str(binary),model['path'],str(config.get('steps',32))]
  if config.get('prefill_validation_tokens'):command.append(str(config['prefill_validation_tokens']))
  command+=config.get('validation_arguments',[])
 elif config.get('mode')=='hybrid':
  command=[str(binary),model['path'],config['strategy'],str(config['prompt']),str(config['generation']),str(config['context']),str(threads),str(batch),str(ub),str(ngl),config.get('shadow','mmap'),str(int(config.get('drop_gpu',True)))]
  if config.get('reference_in') or config.get('reference_out'):
   command += [config.get('reference_in','-'),config.get('reference_out','-')]
 elif config.get('mode')=='sustained':
  command=[str(binary),model['path'],config.get('device','GPUOpenCL') if ngl else 'none',str(ngl),str(threads),str(config['prompt']),str(config['generation']),str(config['context']),str(batch),str(ub),str(config['duration_seconds']),config.get('attention','off')]
 elif config.get('mode')=='memory':
  if config.get('repack')==0:common+=['--no-repack']
  command=[str(binary),*common,'-c',str(config['context']),'-n',str(config.get('generation',2)),'--temp','0','--seed','1234','-no-cnv','--log-verbosity','4','-p','def add(a, b):']
 else:
  command=[str(binary),*common,'-p',str(config.get('prompt',128)),'-n',str(config.get('generation',32)),'-d',str(config.get('depth',0)),'-r',str(config.get('repetitions',3)),'--progress','-v','-o','json']
  command+=['--delay',str(config.get('delay',10))]
  if not config.get('verbose',True):command.remove('-v')
  if 'poll' in config:command+=['--poll',str(config['poll'])]
  if 'mask' in config:command+=['-C',config['mask'],'--cpu-strict','1']
  if config.get('repack') is not None:command+=['--repack',str(config['repack'])]
 if config.get('process_mask'):command=['taskset',config['process_mask'],*command]
 if config['cpu_nice_increment']:command=['nice','-n',str(config['cpu_nice_increment']),*command]
 artifact_paths=[binary,*[p for name in ['libllama.so','libggml.so','libggml-base.so','libggml-cpu.so','libggml-opencl.so','libggml-hexagon.so'] if (p:=library_directory/name).exists()]]
 artifact_paths += [pathlib.Path(p) for p in config.get('additional_artifacts',[])]
 artifacts={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths}
 (out/'command.json').write_text(json.dumps({'harness_source_sha256':HARNESS_SOURCE_SHA256,'harness_telemetry_revision':HARNESS_TELEMETRY_REVISION,'config':config,'command':command,'artifact_sha256':artifacts,'environment':{k:env.get(k) for k in ['LD_LIBRARY_PATH','GGML_OPENCL_KERNEL_CACHE_DIR','KMP_BLOCKTIME','OMP_WAIT_POLICY','OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','ADSP_LIBRARY_PATH','GGML_HEXAGON_PROFILE','GGML_HEXAGON_NHVX','GGML_HEXAGON_NHMX','GGML_HEXAGON_MM_SELECT','GGML_HEXAGON_OPBATCH','GGML_HEXAGON_OPQUEUE']},'model_metadata':{k:v for k,v in model.items() if k not in ['tensors','metadata']}},indent=2)+'\n')
 required=None
 if config.get('mode') not in ['memory','logits','ops'] and not config.get('validation_timing_not_comparable'):
  mask=config.get('process_mask',config.get('mask'))
  required=[i for i in range(8) if int(mask,16)&(1<<i)] if mask else list(range(8))
 before,cooling_seconds=cooldown(config.get('cooldown',15),config.get('max_battery',38),config.get('max_cpu',65),config.get('max_gpu',47),config.get('cooling_timeout',600),out/'cooldown.jsonl',required,True,config.get('max_npu'),config.get('public_thermal',False),config.get('max_public_thermal_status',1),config.get('max_public_thermal_headroom'))
 power_conditions=conditions()
 if config.get('device','').startswith('HTP') and ngl and config.get('mode') not in ['ops','ops-perf']:
  # RPC buffers consume the same system DDR as CPU/Android. The OpenCL
  # advertised global-memory limit is unrelated to FastRPC allocation limits.
  # Conservative preflight estimates are admission decisions, not measurements.
  context=config.get('context',config.get('depth',0)+max(config.get('prompt',128),config.get('generation',32))+32)
  fraction=min(ngl/(model['blocks']+1),1)
  weights=model['weight_data_bytes']/2**20
  packed=weights*fraction*(160/144)
  kv=model.get('theoretical_f16_KV_MiB_per_token',0)*context
  workspace=max(128,context*batch*model['attention_heads']*8/2**20) if config.get('attention','off')=='off' else 128
  # CPU repacking may create a private copy for unoffloaded matrices.
  cpu_repack=weights*(1-fraction) if config.get('repack',1) else 0
  projected=packed+kv+workspace+cpu_repack+config.get('additional_memory_reserve_MiB',0)
  budget=before['meminfo_kB']['MemAvailable']/1024-config['minimum_MemAvailable_MiB']
  if projected>budget:
   why=f'HTP shared-DDR admission: estimated buffers {projected:.0f} MiB exceed available budget {budget:.0f} MiB. Not an observed allocation failure or extra NPU RAM.'
   result={'id':label,'config':config,'power_conditions':power_conditions,'exit_code':None,'status':'guarded_skip','abort_reason':why,'parse_error':None,'before':before,'benchmark_rows':None,'samples':[]}
   (out/'result.json.tmp').write_text(json.dumps(result,indent=2)+'\n');(out/'result.json.tmp').replace(out/'result.json');print('SKIP',label,why,flush=True);return result
 elif config.get('mode')=='memory':
  prior=BASE/'runs'/f'baseline-{model["id"]}-r0-ngl{99 if ngl else 0}'/'result.json'
  if not prior.exists() and config.get('group')=='capacity':
   projected=model['weight_data_bytes']/2**20+model.get('theoretical_f16_KV_MiB_per_token',0)*config['context']+256
   budget=before['meminfo_kB']['MemAvailable']/1024-max(512,config['minimum_MemAvailable_MiB'])
   if projected>budget:
    why=f'Conservative capacity admission guard: weights + KV + workspace {projected:.0f} MiB exceed currently available budget {budget:.0f} MiB. Not an observed allocation failure.'
    result={'id':label,'config':config,'power_conditions':power_conditions,'exit_code':None,'status':'guarded_skip','abort_reason':why,'parse_error':None,'before':before,'benchmark_rows':None,'samples':[]}
    (out/'result.json.tmp').write_text(json.dumps(result,indent=2)+'\n');(out/'result.json.tmp').replace(out/'result.json');print('SKIP',label,why,flush=True);return result
  if prior.exists():
   previous=json.loads(prior.read_text());buffers=previous['allocations']['buffers']
   by=lambda backend,kind:max([b['MiB'] for b in buffers if b['backend']==backend and b['kind']==kind]+[0])
   cache_ratio={'f16':1,'q8_0':34/64,'q4_0':18/64}
   factor=(cache_ratio.get(config.get('cache_k','f16'),1)+cache_ratio.get(config.get('cache_v','f16'),1))/2
   kv=model.get('theoretical_f16_KV_MiB_per_token',0)*config['context']*factor
   fraction=min(ngl/model['blocks'],1) if ngl else 0
   gpu_weight=by('OpenCL','model')*fraction if ngl else 0
   cpu_prior=BASE/'runs'/f'baseline-{model["id"]}-r0-ngl0'/'result.json'
   cpu_buffers=json.loads(cpu_prior.read_text())['allocations']['buffers'] if cpu_prior.exists() else []
   full_cpu_repack=max([b['MiB'] for b in cpu_buffers if b['backend']=='CPU_REPACK' and b['kind']=='model']+[0])
   repack=(by('CPU_REPACK','model')+(1-fraction)*max(0,full_cpu_repack-by('CPU_REPACK','model'))) if config.get('repack',1) else 0
   # Budget a bounded ordinary-attention workspace and recurrent state; this is an admission guard, not a reported allocation.
   workspace=128 if config.get('attention')=='on' else max(128,config['context']*config.get('batch',128)*model['attention_heads']*4/2**20*2)
   state=max([b['MiB'] for b in buffers if b['kind'] in ['RS','recurrent state']]+[0])
   projected_private=gpu_weight+repack+kv+workspace+state
   cap=before['meminfo_kB']['MemAvailable']/1024-max(512,config['minimum_MemAvailable_MiB'])
   gpu_projected=gpu_weight+fraction*kv+workspace+state
   if projected_private>cap or (ngl and gpu_projected>5542-256):
    why=f'Admission guard: projected private buffers {projected_private:.0f} MiB vs available budget {cap:.0f} MiB; projected device buffers {gpu_projected:.0f} MiB'
    result={'id':label,'config':config,'power_conditions':power_conditions,'exit_code':None,'status':'guarded_skip','abort_reason':why,'parse_error':None,'before':before,'benchmark_rows':None,'samples':[]}
    (out/'result.json.tmp').write_text(json.dumps(result,indent=2)+'\n');(out/'result.json.tmp').replace(out/'result.json');print('SKIP',label,why,flush=True);return result
 print('Run',label,'battery',before['temperatures_C'].get('battery'),'ngl',ngl,flush=True)
 samples=[];reason=None;low_memory=0;start=time.monotonic();library_paths=set();initial_child_cgroup=None
 with ThermalMonitor(config,out) as thermal,(out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
  with managed_process(command,stdout=stdout,stderr=stderr,env=env,cwd=out) as proc:
   while proc.poll() is None:
    state=system();state['process']=process(proc.pid);state['public_thermal']=thermal.latest();samples.append(state)
    from validation import live_process_sample
    if not live_process_sample(state):break
    child_cgroup=state['process'].get('cgroup')
    if initial_child_cgroup is None and child_cgroup:initial_child_cgroup=child_cgroup
    mapped=read(f'/proc/{proc.pid}/maps') or ''
    library_paths.update(line.split()[-1] for line in mapped.splitlines() if '/libggml' in line or '/libllama' in line)
    if any(pathlib.Path(path).parent.resolve()!=library_directory.resolve() for path in library_paths):reason='Mixed llama/ggml library paths detected: '+str(sorted(library_paths))
    if required and not set(required)<=set(state['harness_allowed_CPUs']):reason='Android removed required CPU cores during measurement; invalid controlled comparison'
    if required and state.get('harness_cgroup')!=before.get('harness_cgroup'):reason='Android harness cgroup changed during measurement; invalid controlled comparison'
    if required and initial_child_cgroup and child_cgroup and child_cgroup!=initial_child_cgroup:reason='Android benchmark cgroup changed during measurement; invalid controlled comparison'
    policy=conditions()
    if policy.get('power_epoch')!=power_conditions.get('power_epoch'):reason='Power condition changed during measurement; exclude from matched comparisons'
    if policy.get('phone_use_hold'):reason='User phone-use hold requested during measurement'
    if policy.get('investigation_paused_by_user'):reason='User-requested investigation pause reached during measurement'
    if policy.get('battery_low_hold'):reason='Battery guard requested a hold during measurement'
    if policy.get('execution_policy','legacy-nice0')!=config['execution_policy']:reason='Execution priority/headroom policy changed during measurement; exclude from matched comparisons'
    available=state['meminfo_kB'].get('MemAvailable',0);low_memory=low_memory+1 if available<config['minimum_MemAvailable_MiB']*1024 else 0
    if low_memory>=3:reason=f'MemAvailable below {config["minimum_MemAvailable_MiB"]} MiB for three samples'
    if time.monotonic()-start>config.get('timeout',900):reason='measurement timeout'
    if state['temperatures_C'].get('battery',0)>43:reason='battery above 43 C safety limit'
    if state['public_thermal'] and state['public_thermal']['thermal_status']>=3:reason='Android public thermal status reached SEVERE; stop sustained load'
    if config.get('test_graceful_cancel_after_requests'):
     completed_requests=[]
     for line in (out/'stdout.txt').read_text().splitlines():
      try:
       row=json.loads(line)
       if row.get('iteration',-1)>=0:completed_requests.append(row)
      except ValueError:pass
     if len(completed_requests)>=config['test_graceful_cancel_after_requests']:reason='Intentional graceful-shutdown validation'
    if reason:
     terminate_group(proc)
     break
    time.sleep(1)
   rc=proc.wait()
 elapsed=time.monotonic()-start;after=system();text=(out/'stderr.txt').read_text(errors='replace');rows=None;error=None
 if config.get('mode') not in ['memory','logits','ops','ops-perf','hybrid','sustained'] and rc==0:
  try:rows=json.loads((out/'stdout.txt').read_text())
  except ValueError as e:error=str(e)
 result={'id':label,'config':config,'power_conditions':power_conditions,'exit_code':rc,'abort_reason':reason,'parse_error':error,'wall_seconds':elapsed,'cooldown_seconds':cooling_seconds,'llama_library_paths':sorted(library_paths),'before':before,'after':after,'max_process_RSS_kB':max([max(s['process']['status_kB'].get('VmRSS',0),s['process']['status_kB'].get('VmHWM',0)) for s in samples]+[0]),'max_process_PSS_kB':max([s['process']['smaps_kB'].get('Pss',0) for s in samples]+[0]),'max_process_swap_kB':max([s['process']['status_kB'].get('VmSwap',0) for s in samples]+[0]),'min_system_MemAvailable_kB':min([s['meminfo_kB'].get('MemAvailable',0) for s in samples]+[before['meminfo_kB']['MemAvailable']]),'allocations':allocations(text),'benchmark_rows':rows,'samples':samples}
 if config.get('public_thermal'):
  result['public_thermal_samples']=thermal.rows()
  result['public_thermal_monitor_available']=bool(result['public_thermal_samples'])
 if config.get('test_graceful_cancel_after_requests'):
  result['graceful_shutdown_validation']={'passed':reason=='Intentional graceful-shutdown validation' and rc==128+signal.SIGTERM and 'SUSTAINED cleanup complete' in text,'cleanup_marker':'SUSTAINED cleanup complete' in text,'exit_code':rc}
 if config.get('mode')=='hybrid' and rc==0:
  try:
   result['latency_metrics']=json.loads((out/'stdout.txt').read_text())
   if not result['latency_metrics'].get('all_logits_finite'):result['parse_error']='Nonfinite hybrid output'
  except ValueError as e:result['parse_error']=str(e)
 if config.get('mode')=='sustained' and rc==0:
  try:
   result['request_rows']=[json.loads(line) for line in (out/'stdout.txt').read_text().splitlines() if line.strip()]
   measured=[r for r in result['request_rows'] if r['iteration']>=0]
   if not measured or any(not r.get('all_logits_finite') for r in measured):result['parse_error']='Missing sustained requests or nonfinite logits'
   result['identical_request_output_hashes']=len({r['output_hash'] for r in measured})==1
   if not result['identical_request_output_hashes']:result['parse_error']='Fixed deterministic requests produced different output hashes'
  except (ValueError,KeyError) as e:result['parse_error']=str(e)
 if config.get('mode')=='ops' and rc==0:
  output=(out/'stdout.txt').read_text(errors='replace');matches=re.findall(r'(\d+)/(\d+) tests passed',output)
  result['matrix_validation']=[{'passed':int(a),'supported':int(b)} for a,b in matches]
  if not matches or not any(int(b)>0 for a,b in matches):result['parse_error']='No supported numerical tests executed; inspect device/backend filter'
 if config.get('mode')=='ops-perf' and rc==0:
  output=re.sub(r'\x1b\[[0-9;]*m','',(out/'stdout.txt').read_text(errors='replace'))
  matches=re.findall(r'MUL_MAT\(([^\n]+?)\):\s+(\d+) runs -\s*([\d.]+) us/run -([^\n]+)',output)
  result['operation_performance']=[{'parameters':a,'total_runs':int(n),'mean_us_per_operation':float(us),'printed_rate':rate.strip()} for a,n,us,rate in matches]
  if not matches:result['parse_error']='No supported operation performance cases executed'
 if config.get('mode')=='logits' and rc==0:
  output=(out/'stdout.txt').read_text(errors='replace')
  match=re.search(r'steps=(\d+) top1_agreements=(\d+) logit_RMSE=([\d.eE+-]+) max_abs_logit_error=([\d.eE+-]+) mean_KL=([\d.eE+-]+) all_logits_finite=true',output)
  result['numerical_validation']={'steps':int(match[1]),'top1_agreements':int(match[2]),'logit_RMSE':float(match[3]),'max_abs_logit_error':float(match[4]),'mean_KL':float(match[5]),'all_logits_finite':True} if match else None
  if not match:result['parse_error']='Missing numerical validation summary'
 result['affinity_error_count']=text.count('failed to set affinity')
 from validation import issues
 result['controlled_comparison_issues']=issues(result,out)
 result['controlled_comparison_valid']=not result['controlled_comparison_issues']
 if result['affinity_error_count']:print('INVALID controlled comparison: requested affinity rejected',result['affinity_error_count'],'times; raw result preserved',flush=True)
 (out/'result.json.tmp').write_text(json.dumps(result,indent=2)+'\n');(out/'result.json.tmp').replace(out/'result.json');print('Done',label,'exit',rc,'wall',round(elapsed,1),'RSS MiB',round(result['max_process_RSS_kB']/1024),flush=True)
 if rows:
  for row in rows:print(' pp',row['n_prompt'],'tg',row['n_gen'],'depth',row['n_depth'],'tok/s',round(row['avg_ts'],2),'+/-',round(row['stddev_ts'],2),flush=True)
 return result
def run(config):
 # Completed records are immutable. Read them without occupying the compute lock.
 # A concurrent legacy writer may expose partial JSON; retry under the lock then.
 completed=BASE/'runs'/config['id']/'result.json'
 if completed.exists():
  try:
   result=json.loads(completed.read_text());print('Already recorded:',config['id'],flush=True);return result
  except (ValueError,OSError):pass
 while True:
  completed=BASE/'runs'/config['id']/'result.json'
  if not completed.exists() and config.get('mode') not in ['memory','logits','ops'] and not config.get('validation_timing_not_comparable'):
   mask=config.get('process_mask',config.get('mask'))
   required=[i for i in range(8) if int(mask,16)&(1<<i)] if mask else list(range(8))
   if not set(required)<=set(os.sched_getaffinity(0)):
    print('Waiting outside resource lock for required CPU availability:',config['id'],flush=True)
    cooldown(0,200,200,200,config.get('cooling_timeout',600),required_CPUs=required)
  try:
   with (BASE/'resource.lock').open('a') as gate:
    fcntl.flock(gate,fcntl.LOCK_EX)
    result=run_impl(config)
   break
  except AdmissionDeferred as e:
   if conditions().get('phone_use_hold'):raise
   print(str(e),flush=True)
 # Yield after unlocking so a waiting build/diagnostic cannot be starved by this loop.
 time.sleep(0.1)
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('schedule');parser.add_argument('--group');args=parser.parse_args()
 plan=json.loads(pathlib.Path(args.schedule).read_text())
 for config in plan:
  if not args.group or config['group']==args.group:
   result=run(config)
   if config.get('stop_schedule_if_failure') and (result.get('exit_code')!=0 or result.get('parse_error') or any(v['passed']!=v['supported'] for v in result.get('matrix_validation',[]))):
    if config.get('continue_on_thermal_censoring') and result.get('abort_reason')=='Android public thermal status reached SEVERE; stop sustained load' and result.get('exit_code') in [-signal.SIGTERM,128+signal.SIGTERM]:
     print('Thermal safety event retained; next independent block requires fresh cooling admission.',flush=True)
     continue
    raise SystemExit('Numerical prerequisite failed; later cases in this schedule withheld')
