"""Trace failed session opening without changing driver/backend code."""
import datetime,fcntl,hashlib,json,os,pathlib,subprocess,time
import measure
B=pathlib.Path(__file__).resolve().parent
R=B/'npu-investigation'
S=B/'source-npu-registration-hybrid'
D=B/'build-npu-registration-hybrid/bin'
source=B/'npu-virtual-session-order.cpp'
target=D/'npu-virtual-session-order'
records=[]
with (B/'resource.lock').open('a') as lock:
 print('Waiting for compute lock: no-model session syscall diagnostic',flush=True)
 fcntl.flock(lock,fcntl.LOCK_EX)
 utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=R/('virtual-session-strace-'+utc)
 out.mkdir()
 before,wait=measure.cooldown(30,36,50,43,1200,out/'cooldown.jsonl',[0,1],True,43,True,0,.9)
 assert target.exists(), 'Build and qualify the no-model order helper first'
 command=None
 env=os.environ.copy()
 for key in list(env):
  if key.startswith(('GGML_HEXAGON_','GGML_OPENCL_','GGML_PERF_')) or key in ['LD_PRELOAD','GGML_BACKEND_PATH','LD_LIBRARY_PATH']:env.pop(key)
 env.update({'LD_LIBRARY_PATH':str(D),'ADSP_LIBRARY_PATH':str(B/'build-npu-registration-hybrid/ggml/src/ggml-hexagon'),'GGML_HEXAGON_DEVICES':'HTP0:0,HTP0:1','GGML_HEXAGON_PROFILE':'1','GGML_HEXAGON_MBUF':'512','GGML_HEXAGON_VMEM':'3200','GGML_HEXAGON_REGISTRATION_WINDOW':'0'})
 initial=measure.conditions()
 for order in ['0','1','0,1']:
  current=measure.conditions()
  if current['power_epoch']!=initial['power_epoch'] or any(current.get(k) for k in ['phone_use_hold','battery_low_hold','investigation_paused_by_user']):raise SystemExit('Power/user/battery changed before session probe')
  cmd=['nice','-n','10','taskset','03','strace','-f','-qq','-tt','-s','1024','-e','trace=openat,ioctl,connect,sendto,sendmsg,write,writev','-o',str(out/('order-'+order.replace(',','-')+'.syscalls.txt')),str(target),order]
  stem='order-'+order.replace(',','-')
  reason=None;samples=[]
  with (out/(stem+'.stdout.jsonl')).open('w') as stdout,(out/(stem+'.stderr.txt')).open('w') as stderr,measure.ThermalMonitor({'public_thermal':True},out) as thermal:
   with measure.managed_process(cmd,env=env,stdout=stdout,stderr=stderr) as proc:
    start=time.monotonic()
    while proc.poll() is None:
     s=measure.system();s['public_thermal']=thermal.latest();samples.append(s)
     c=measure.conditions()
     if c['power_epoch']!=initial['power_epoch']:reason='Power transition'
     elif any(c.get(k) for k in ['phone_use_hold','battery_low_hold','investigation_paused_by_user']):reason='User/battery hold'
     elif s['meminfo_kB']['MemAvailable']<1536*1024:reason='Memory reserve'
     elif s['temperatures_C'].get('battery',0)>=43:reason='Battery temperature limit'
     elif s['public_thermal'] and s['public_thermal']['thermal_status']>=3:reason='Public SEVERE'
     elif time.monotonic()-start>120:reason='Diagnostic timeout'
     if reason:measure.terminate_group(proc);break
     time.sleep(.25)
    rc=proc.wait()
   text=(out/(stem+'.stderr.txt')).read_text()
  records.append({'order':order,'command':cmd,'exit_code':rc,'abort_reason':reason,'wall_s':time.monotonic()-start,'opened':[n for n in ['HTP0:0','HTP0:1'] if 'SESSION_ORDER opened '+n in text],'session_error_lines':[line for line in text.splitlines() if any(x in line for x in ['failed','error 0x','allocating new session','new session:','SESSION_ORDER'])],'samples':samples})
  print(order,'exit',rc,'opened',records[-1]['opened'],flush=True)
  time.sleep(2)
 result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'0x200 does not identify the failing syscall/service. Trace the same no-model single-session control and session1 failures to capture vendor diagnostics without changing libraries or privileged settings.','no_model':True,'no_graph_operations':True,'backend_modified':False,'power_conditions':initial,'build_command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'environment':{k:env[k] for k in env if k.startswith(('LD_LIBRARY_PATH','ADSP_LIBRARY_PATH','GGML_HEXAGON_'))},'records':records,'interpretation':'0x200 is AEE_ERPC in local FastRPC headers, a generic RPC implementation error. It does not alone identify permission, firmware or session-capability cause. No speed or capacity claim.'}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 (R/'virtual-session-strace-latest-oct05.txt').write_text(str(out)+'\n')
 print(out,flush=True)
