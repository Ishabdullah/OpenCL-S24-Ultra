"""Sequential immutable measurements; explicit skips, cancellations and stop budget."""
import pathlib,json,sys,datetime,time
import measure
from validation import issues
B=pathlib.Path(__file__).resolve().parent;p=pathlib.Path(sys.argv[1]);cases=json.loads(p.read_text());state=p.with_suffix('.state.json');epoch=measure.conditions()['power_epoch'];records=[]
def save(stage,**kw):
 d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,'power_epoch':epoch,'records':records,**kw};state.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'stage':stage,**kw}),flush=True)
try:
 for c in cases:
  policy=measure.conditions()
  if policy['power_epoch']!=epoch:raise RuntimeError('Power epoch changed; new independent IDs required')
  if any(policy.get(k) for k in ['phone_use_hold','battery_low_hold','investigation_paused_by_user']):raise RuntimeError('User or battery hold; retain checkpoint')
  deadline=policy.get('planned_pause_UTC')
  if deadline:
   remaining=(datetime.datetime.fromisoformat(deadline)-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
   if remaining<c.get('minimum_seconds_to_start',60):raise RuntimeError('Insufficient time before planned pause; do not start another case')
  save('running',current=c['id']);d=measure.run(c);bad=issues(d,B/'runs'/c['id']);graceful=d.get('graceful_shutdown_validation',{}).get('passed',False)
  if graceful:bad=[x for x in bad if x!='Run failed, was aborted, or did not parse']
  skipped=d.get('status')=='guarded_skip';rows=[r for r in d.get('request_rows',[]) if r.get('iteration',-1)>=0]
  if c.get('mode')=='sustained' and not skipped:
   if not graceful and 'SUSTAINED cleanup complete' not in (B/'runs'/c['id']/'stderr.txt').read_text():bad.append('Owned model/context cleanup marker missing')
   if not graceful and c.get('environment',{}).get('GGML_PERF_REQUEST_PERIOD_S','0')!='0':
    period=float(c['environment']['GGML_PERF_REQUEST_PERIOD_S']);expected=__import__('math').ceil(c['duration_seconds']/period)
    if len(rows)!=expected:bad.append('Offered request count not completed; retain censoring')
   if rows and any(r.get('initial_CPU_max_C',999)>45 or r.get('initial_NPU_max_C',999)>43 or r.get('initial_GPU_max_C',999)>43 or r.get('initial_battery_C',999)>36 for r in rows):bad.append('Initial post-load readiness failed')
  records.append({'id':c['id'],'valid':not bad and not skipped,'guarded_skip':skipped,'issues':bad,'graceful_cancel_passed':graceful,'benchmark_rows':d.get('benchmark_rows'),'request_rows':rows,'numerical_validation':d.get('numerical_validation'),'allocations':d.get('allocations'),'minimum_MemAvailable_MiB':d.get('min_system_MemAvailable_kB',0)/1024})
  save('recorded',current=c['id'],valid=not bad and not skipped,skipped=skipped,issues=bad,requests=len(rows),first_response=rows[0] if rows else None)
  if bad and not skipped and c.get('stop_on_failure',True):raise RuntimeError('Failed/excluded: '+'; '.join(bad))
 save('complete')
except (RuntimeError,KeyboardInterrupt) as e:
 save('stopped-with-evidence',reason=str(e) or 'Controller interrupted');raise SystemExit(str(e) or 'Controller interrupted')
